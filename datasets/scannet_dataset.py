import sys
sys.path.append('.')

from datasets.base.base_dataset import BaseDataset
import os
import numpy as np
import os.path as osp
from pathlib import Path
from PIL import Image
from datasets.base.transforms import *
import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer

class ScannetDataset(BaseDataset):
    '''
    PyTorch Dataset implementation for the ScanNet dataset.
    This class loads ScanNet sequences and provides a set of views for each sample
    (RGB images, depth maps, and camera intrinsics/extrinsics).
    '''
    def __init__(
        self,
        data_root=None,
        verbose=False,
        max_distance=240,
        text_model_name='./ckpts/roberta-base',
        text_max_len=64,
        dataset_source='scanrefer',
        **kwargs
    ):
        super().__init__(**kwargs)

        assert data_root is not None

        self.verbose = verbose
        self.dataset_label = 'ScanNet'
        mode = self.mode
        self.data_root = data_root
        self.dataset_source = dataset_source
        self.max_distance = max_distance
        self.neg_frame_ratio = 0.5

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name, use_fast=True, local_files_only=True)
        self.text_max_len = text_max_len

        self.val_scene_pixels = None
        if self.mode != 'train':
            val_pixels_path = 'data/mvrefer_val.json'
            if os.path.exists(val_pixels_path):
                with open(val_pixels_path, 'r') as f:
                    self.val_scene_pixels = json.load(f)
                print(f"[{self.dataset_label}] Loaded custom validation frames from {val_pixels_path}", flush=True)

        if not os.path.exists('data/dataset_cache'):
            os.makedirs('data/dataset_cache')

        scanrefer_cache_path = f'data/dataset_cache/{self.dataset_source}_{self.mode}_cache.npy'
        if not os.path.exists(scanrefer_cache_path):
            scanrefer_raw_data = []
            
            if self.dataset_source == 'scanrefer':
                if mode == 'train':
                    anno_file = 'data/ScanRefer/ScanRefer_filtered_train.json'
                else:
                    anno_file = 'data/ScanRefer/ScanRefer_filtered_val.json'
            elif self.dataset_source == 'nr3d':
                if mode == 'train':
                    anno_file = 'data/ReferIt3D/nr3d_train.json'
                else:
                    anno_file = 'data/ReferIt3D/nr3d_val.json'
            elif self.dataset_source == 'sr3d':
                if mode == 'train':
                    anno_file = 'data/ReferIt3D/sr3d_train.json'
                else:
                    anno_file = 'data/ReferIt3D/sr3d_val.json'
            else:
                raise ValueError(f"Unknown dataset source: {self.dataset_source}")
            
            if self.verbose:
                print(f"[{self.dataset_label}] Loading annotations from {anno_file}", flush=True)
                
            with open(anno_file, 'r') as f:
                scanrefer_raw_data = json.load(f)
            
            self.scanrefer_data = {}
            for ann in tqdm(scanrefer_raw_data, desc=f"Processing {self.dataset_source} {self.mode} data"):
                scene_id = ann['scene_id']
                if scene_id not in self.scanrefer_data:
                    self.scanrefer_data[scene_id] = []
                self.scanrefer_data[scene_id].append(ann)
            
            np.save(scanrefer_cache_path, self.scanrefer_data)
        else:
            self.scanrefer_data = np.load(scanrefer_cache_path, allow_pickle=True).item()

        print(f'[{self.dataset_label}] Loaded {self.dataset_source} data for {len(self.scanrefer_data)} scenes.', flush=True)

        # 2. Determine sequence list
        if self.dataset_source == 'scanrefer':
            # Load scenes based on the ScanRefer train/val split
            if mode == 'train':
                split_file = 'data/ScanRefer/ScanRefer_filtered_train.txt'
            else:  # 'test' or 'val'
                split_file = 'data/ScanRefer/ScanRefer_filtered_val.txt'
            
            with open(split_file, 'r') as f:
                sequences_from_file = [line.strip() for line in f.readlines()]
                print(f'[{self.dataset_label}] Found {len(sequences_from_file)} sequences in {split_file}', flush=True)
        else:
            sequences_from_file = sorted(list(self.scanrefer_data.keys()))
            print(f'[{self.dataset_label}] Found {len(sequences_from_file)} unique scenes in {self.dataset_source} annotations', flush=True)
        # Ensure scenes exist in data_root
        all_scenes_in_dir = os.listdir(data_root)
        self.sequences = [seq for seq in sequences_from_file if seq in all_scenes_in_dir]
        self.scene_to_idx = {scene: i for i, scene in enumerate(self.sequences)}

        if self.verbose:
            print(f'[{self.dataset_label}] Sequences of {self.dataset_label} dataset:', self.sequences)

        print(f'[{self.dataset_label}] Found %d unique videos in %s' % (len(self.sequences), data_root), flush=True)

        # Create a sample list based on text descriptions
        self.all_samples = []
        for scene_id, annotations in self.scanrefer_data.items():
            if scene_id in self.scene_to_idx:
                scene_idx = self.scene_to_idx[scene_id]
                for ann in annotations:
                    sample = {
                        'scene_id': scene_id,
                        'scene_idx': scene_idx,
                        'object_id': ann['object_id'],
                        'ann_id': ann['ann_id'],
                        'description': ann['description']
                    }
                    
                    # Preserve metadata for Nr3D/Sr3D
                    if self.dataset_source in ['nr3d', 'sr3d']:
                        sample['meta_data'] = ann.get('meta_data')
                        sample['view_dependent'] = ann.get('view_dependent')
                        
                    self.all_samples.append(sample)

        with open('data/scannet_invalid_list.json') as f:
            self.invalid_list = json.load(f)

        # Cache the number of images per scene for faster initialization
        num_imgs_cache_path = f'data/dataset_cache/scannetmv_{self.dataset_source}_{self.mode}_cache.npy'
        if not os.path.exists(num_imgs_cache_path):
            self.num_imgs = {}
            for seq in tqdm(self.sequences, desc="Generating num_imgs cache"):
                rgb_path = os.path.join(data_root, seq, 'color')
                self.num_imgs[seq] = len(os.listdir(rgb_path))

            np.save(num_imgs_cache_path, self.num_imgs)
        else:
            self.num_imgs = np.load(num_imgs_cache_path, allow_pickle=True).item()
        
        # Cache 2D instance annotation paths similarly to num_imgs
        instance_cache_path = f'data/dataset_cache/scannetmv_{self.dataset_source}_{self.mode}_instance_path_cache.npy'
        if not os.path.exists(instance_cache_path):
            self.instance_2d_paths = {}
            # Infer the 'scans' path from the data_root
            scans_path = os.path.join(os.path.dirname(self.data_root), 'scans')
            for seq in tqdm(self.sequences, desc="Generating instance path cache"):
                # Build the 2D instance-filt folder path for each scene
                instance_folder_path = os.path.join(scans_path, seq, f'{seq}_2d-instance-filt', 'instance-filt')
                self.instance_2d_paths[seq] = instance_folder_path

            np.save(instance_cache_path, self.instance_2d_paths)
        else:
            self.instance_2d_paths = np.load(instance_cache_path, allow_pickle=True).item()
        self.scene_frame_indices_dir = 'data/scene_frame_indices'
        self.scene_frame_indices_cache = {}
        self._missing_scene_frame_indices = set()
        self._current_target_id = None

    def __len__(self):
        '''Return the total number of text descriptions in the dataset.'''
        return len(self.all_samples)

    def _load_scene_frame_indices(self, scene):
        if scene in self.scene_frame_indices_cache:
            return self.scene_frame_indices_cache[scene]

        if scene in self._missing_scene_frame_indices:
            return None

        json_path = os.path.join(self.scene_frame_indices_dir, f'{scene}.json')

        if not os.path.exists(json_path):
            if self.verbose:
                print(f'[{self.dataset_label}] Scene frame indices not found for {scene} at {json_path}', flush=True)
            self._missing_scene_frame_indices.add(scene)
            self.scene_frame_indices_cache[scene] = None
            return None

        try:
            with open(json_path, 'r') as f:
                raw_data = json.load(f)
        except Exception as e:
            if self.verbose:
                print(f'[{self.dataset_label}] Failed to load scene frame indices for {scene}: {e}', flush=True)
            self.scene_frame_indices_cache[scene] = None
            return None

        frame_indices = {}
        for frame_key, instance_ids in raw_data.items():
            try:
                frame_idx = int(frame_key)
                frame_indices[frame_idx] = set(int(i) for i in instance_ids)
            except ValueError:
                continue

        self.scene_frame_indices_cache[scene] = frame_indices
        return frame_indices

    @staticmethod
    def _random_choice_with_replacement(rng, pool, count):
        if count <= 0:
            return []
        if not pool:
            return []
        if len(pool) == 1:
            return [int(pool[0])] * count

        samples = rng.choice(pool, size=count, replace=len(pool) < count)
        if np.isscalar(samples):
            return [int(samples)]
        return [int(x) for x in np.atleast_1d(samples)]

    def __getitem__(self, index):
        # Get processed views from parent class __getitem__
        if isinstance(index, tuple):
            # The index is specifying the aspect-ratio
            if len(index) == 3:
                idx, ar_idx, frame_num = index
            else:
                idx, ar_idx = index
        else:
            assert len(self._resolutions) == 1
            ar_idx = 0
            idx = index
        
        sample_info = self.all_samples[idx]
        scene_idx = sample_info['scene_idx']
        object_id = sample_info['object_id']
        description = sample_info['description']
        ann_id = sample_info['ann_id']
        target_id = int(object_id)
        target_instance_id = target_id + 1
        self._current_target_id = target_instance_id

        try:
            if 'frame_num' in locals():
                views = super().__getitem__((scene_idx, ar_idx, frame_num))
            else:
                views = super().__getitem__((scene_idx, ar_idx))
        finally:
            self._current_target_id = None

        tokenized_text = self.tokenizer(
            description,
            padding='max_length',
            truncation=True,
            max_length=self.text_max_len,
            return_tensors='pt'
        )
        input_ids = tokenized_text['input_ids'].squeeze(0)
        attention_mask = tokenized_text['attention_mask'].squeeze(0)

        text_info = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'description': description,
            'target_id': target_id,
            'scene_id': sample_info['scene_id'],
            'object_id': object_id,
            'ann_id': ann_id,
        }


        if self.dataset_source in ['nr3d', 'sr3d']:
            text_info['meta_data'] = sample_info['meta_data']
            text_info['view_dependent'] = sample_info['view_dependent']
    
        for view in views:
            instance_map = view['instance_map']
            referring_mask = (instance_map == target_instance_id).long()
            view['referring_mask'] = referring_mask

        return views, text_info
                    
    def _get_views(self, index, resolution, rng):
        '''
        Get a set of views for a given scene index.
        This is the core data-loading logic and contains the frame sampling strategy.

        Args:
            index (int): Index of the scene in self.sequences.
            resolution (tuple): Target image resolution.
            rng (numpy.random.Generator): Random number generator used for all sampling.

        Returns:
            list[dict]: A list of view dictionaries, one per sampled frame.
        '''
        scene = self.sequences[index]
        num_imgs = self.num_imgs[scene]
        # Exclude invalid frames from the 0..num_imgs-1 range based on invalid_list
        valid_idxs = [i for i in range(num_imgs) if i not in self.invalid_list.get(scene, [])]
        num_imgs = len(valid_idxs)
        if num_imgs == 0:
            raise ValueError(f"No valid frames for scene {scene}")

        idxs = None
        target_instance_id = getattr(self, '_current_target_id', None)

        if self.mode != 'train':
            if self.val_scene_pixels is not None and target_instance_id is not None:
                object_id = str(target_instance_id - 1)
                if scene in self.val_scene_pixels and object_id in self.val_scene_pixels[scene]:
                    frame_data = self.val_scene_pixels[scene][object_id]
                    if isinstance(frame_data, dict) and len(frame_data) > 0:
                        idxs = sorted(int(k) for k in frame_data.keys())[:self.frame_num]

        # For training, we want to focus on frames that contain the target object.
        if self.mode == 'train':
            if target_instance_id is not None and self.frame_num > 0:
                frame_instances = self._load_scene_frame_indices(scene)

                if frame_instances:
                    all_candidate_indices = {idx for idx in valid_idxs if idx in frame_instances}
                    pos_candidates = [idx for idx in all_candidate_indices if target_instance_id in frame_instances.get(idx, set())]
                    if pos_candidates:
                        neg_candidates = [idx for idx in all_candidate_indices if idx not in pos_candidates]
                        num_neg = int(round(self.frame_num * self.neg_frame_ratio))
                        num_pos = self.frame_num - num_neg
                        pos_samples = self._random_choice_with_replacement(rng, pos_candidates, num_pos)
                        neg_pool = neg_candidates if neg_candidates else pos_candidates
                        neg_samples = self._random_choice_with_replacement(rng, neg_pool, num_neg)
                        idxs = pos_samples + neg_samples
                        permutation = rng.permutation(len(idxs))
                        idxs = [idxs[i] for i in permutation]
                        if len(idxs) > self.frame_num:
                            idxs = idxs[:self.frame_num]
            
        if idxs is None:
            sampled_relative_indices = np.linspace(
                0, num_imgs - 1, self.frame_num
            ).round().astype(int)
            idxs = [valid_idxs[i] for i in sampled_relative_indices]
        
        self.this_views_info = dict(
            scene=scene,
            idxs=idxs,
        )

        base_path = os.path.join(self.data_root, scene)
        # Load camera intrinsics
        intrinsic_path = osp.join(base_path, 'intrinsic/intrinsic_depth.txt')
        with open(intrinsic_path, 'r') as f:
            intrinsic_text = f.read()
        intrinsic = np.array([float(x) for x in intrinsic_text.split()]).astype(np.float32).reshape(4, 4)[:3, :3]
        
        views = []
        # Iterate over sampled frame indices and load corresponding data
        for idx in idxs:
            impath = os.path.join(base_path, 'color', f'{idx}.jpg')
            disppath = os.path.join(base_path, 'depth', f'{idx}.png')
            posepath = os.path.join(base_path, 'pose', f'{idx}.txt')

            # Load 2D instance label
            instance2d = None
            instance_path = os.path.join(self.instance_2d_paths.get(scene, ''), f'{idx}.png')

            assert os.path.exists(instance_path), f'Instance path not found: {instance_path}'
            instance2d = Image.open(instance_path)

            # load camera params
            with open(posepath, 'r') as f:
                camera_pose_text = f.read()
            camera_pose = np.array([float(x) for x in camera_pose_text.split()]).astype(np.float32).reshape(4, 4)

            if not np.isfinite(camera_pose).all():
                print(f"Infinite in camera pose for view: {posepath}")
            assert np.isfinite(camera_pose).all(), 'Infinite in camera pose for view'
            if np.isnan(camera_pose).any():
                print(f"NaN in camera pose for view: {posepath}")
            assert ~np.isnan(camera_pose).any(), 'NaN in camera pose for view'

            rgb_image = np.array(Image.open(impath).resize((640, 480), resample=Image.LANCZOS))
            instance2d = np.array(instance2d.resize((640, 480), resample=Image.NEAREST))
            depthmap = np.array(Image.open(disppath)).astype(np.float32) / 1000.

            rgb_image, depthmap, intrinsic_, instance2d, *_ = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsic.copy(), resolution, rng=rng, info=impath, instance_map=instance2d)

            # Pack all information into a single dictionary
            views.append(dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose.astype(np.float32),
                camera_intrinsics=intrinsic_.astype(np.float32),
                dataset=self.dataset_label,
                label=scene, # scene id
                instance=str(idx), # view id
                instance_map=instance2d,
            ))
        return views


