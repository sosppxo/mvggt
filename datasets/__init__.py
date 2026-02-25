from .base.transforms import *
from .scannet_dataset import ScannetDataset

from utils.misc import get_world_size, get_rank
from torch.utils.data import DataLoader
import hydra
from datasets.base.base_dataset import sample_resolutions, unified_collate_fn
from datasets.base.batched_sampler import DynamicBatchSampler, DynamicDistributedSampler

__HIGH_QUALITY_DATASETS__ = ['BlinkVision', 'Game', 'GameNew', 'DynamicStereo', 'FlyingThings3D', 'GTA-sfm', 'Hypersim', 'MatrixCity', 'MidAir', 'Monkaa', 'PointOdyssey', 'Sintel', 'Spring', 'TarTanAir', 'Unreal4k', 'VirtualKitti', 'Habitat']
__MIDDLE_QUALITY_DATASETS__ = ['BlendedMVG', 'BlendedMVS', 'DTU', 'ETH3D', 'ScanNet', 'Scannetpp', 'Taskonomy']
__INDOOR_DATASETS__ = ['Hypersim', 'ScanNet', 'Scannetpp', 'Taskonomy', 'ARKitScenes', 'Habitat']

def create_dataloader(cfg, mode):
    """
    Creates a DataLoader for a given mode ('train' or 'test').

    This function is the main entry point for creating data loaders. It handles:
    - Single or multiple dataset configurations.
    - Distributed training sampling.
    - Dynamic batching based on resolution and image count.
    - Instantiation of datasets using Hydra.

    Args:
        cfg (OmegaConf): The global Hydra configuration object.
        mode (str): The mode, either 'train' or 'test'.

    Returns:
        DataLoader: A PyTorch DataLoader instance.
    """
    data_loader = DataLoader

    # --- 1. Select Configuration based on Mode ---
    # Determine which part of the configuration to use based on whether we are in 'train' or 'test' mode.
    if mode == 'train':
        cfg_dataset = cfg.train_dataset
        cfg_dataloader = cfg.train_dataloader
        batch_size = cfg.train.batch_size
        num_workers = cfg.train.num_workers
    else:
        cfg_dataset = cfg.test_dataset
        cfg_dataloader = cfg.test_dataloader
        batch_size = cfg.test.batch_size
        num_workers = cfg.test.num_workers

    # --- 2. Instantiate the Dataset ---
    # The dataset can be configured in multiple ways.
    
    # Legacy mode: if the dataset config is just a string, evaluate it.
    if isinstance(cfg_dataset, str):
        dataset = eval(cfg_dataset) 
    # Multi-dataset mode: if 'weights' are specified, combine multiple datasets.
    elif 'weights' in cfg_dataset:
        weights = cfg_dataset.weights
        # Optional: Adjust weights to match a total dataset length.
        if 'length' in cfg_dataset:
            dataset_length = cfg_dataset.length
            weight_sum = sum([v for k, v in weights.items()])
            new_weights = {}
            for dataset_name, weight in weights.items():
                new_weights[dataset_name] = max(int(weight / weight_sum * dataset_length), 1)
            weights = new_weights
            print(f'New weights for dataset (adjusting to dataset length {dataset_length}): {new_weights}')

        datasets_all = []

        num_resolution = cfg.train.num_resolution if 'num_resolution' in cfg.train and mode == 'train' else 1
        
        # Handle different resolution settings for training.
        if mode == 'train' and 'random_reslution' in cfg.train and cfg.train.random_reslution:
            # Sample random resolutions.
            seed = 777 + 0
            resolutions = sample_resolutions(aspect_ratio_range=cfg.train.aspect_ratio_range, pixel_count_range=cfg.train.pixel_count_range, patch_size=cfg.train.patch_size, num_resolutions=num_resolution, seed=seed)
            print('Initialized resolution', resolutions)
            num_resolution = len(resolutions)
            for dataset_name, weight in weights.items():
                # Instantiate each dataset with the sampled resolutions.
                dataset_i = hydra.utils.instantiate(cfg_dataset[dataset_name], resolution=resolutions)
                dataset_i.convert_attributes()
                # The '@' operator likely creates a weighted wrapper around the dataset.
                datasets_all.append(weight @ dataset_i)
        elif 'resolution' in cfg.train:
            # Use a fixed resolution from the config.
            resolutions = cfg.train.resolution
            print('Setting dataset resolution', resolutions)
            for dataset_name, weight in weights.items():
                # Instantiate each dataset with the fixed resolution.
                dataset_i = hydra.utils.instantiate(cfg_dataset[dataset_name], resolution=resolutions)
                dataset_i.convert_attributes()
                datasets_all.append(weight @ dataset_i)
        else:
            # No specific resolution in train config, instantiate without it.
            for dataset_name, weight in weights.items():
                dataset_i = hydra.utils.instantiate(cfg_dataset[dataset_name])
                dataset_i.convert_attributes()
                datasets_all.append(weight @ dataset_i)
        
        # Combine all weighted datasets into a single concatenated dataset.
        dataset = datasets_all[0]
        for dataset_ in datasets_all[1:]:
            dataset += dataset_
    else:
        # ScanNet
        dataset_name = 'ScanNet'
        # # Single-dataset mode: instantiate directly using Hydra.
        # dataset = hydra.utils.instantiate(cfg_dataset)
        # # Convert list/dict attributes to numpy/pandas for memory efficiency.
        # dataset.convert_attributes()
        num_resolution = cfg.train.num_resolution if 'num_resolution' in cfg.train and mode == 'train' else 1
        
        # Handle different resolution settings for training.
        if mode == 'train' and 'random_reslution' in cfg.train and cfg.train.random_reslution:
            # Sample random resolutions.
            seed = 777 + 0
            resolutions = sample_resolutions(aspect_ratio_range=cfg.train.aspect_ratio_range, pixel_count_range=cfg.train.pixel_count_range, patch_size=cfg.train.patch_size, num_resolutions=num_resolution, seed=seed)
            print('Initialized resolution', resolutions)
            num_resolution = len(resolutions)
            dataset = hydra.utils.instantiate(cfg_dataset[dataset_name], resolution=resolutions)
            dataset.convert_attributes()
        elif mode == 'test' and 'resolution' in cfg.test:
            # Use a fixed resolution from the config.
            resolutions = cfg.test.resolution
            print('Setting dataset resolution', resolutions)
            dataset = hydra.utils.instantiate(cfg_dataset[dataset_name], resolution=resolutions)
            dataset.convert_attributes()
        elif mode == 'train' and 'resolution' in cfg.train:
            # Use a fixed resolution from the config.
            resolutions = cfg.train.resolution
            print('Setting dataset resolution', resolutions)
            dataset = hydra.utils.instantiate(cfg_dataset[dataset_name], resolution=resolutions)
            dataset.convert_attributes()
        else:
            # No specific resolution in train config, instantiate without it.
            dataset = hydra.utils.instantiate(cfg_dataset[dataset_name])
            dataset.convert_attributes()
    
    # --- 3. Setup Samplers for Distributed Training and Dynamic Batching ---
    world_size = get_world_size()
    rank = get_rank()

    image_num_range = cfg.train.image_num_range if mode == 'train' else [8, 8]
    print(f'Sampling frame number range from {image_num_range}')
    
    # Determine the maximum number of images per GPU to manage memory.
    max_img_per_gpu = cfg.train.max_img_per_gpu if 'max_img_per_gpu' in cfg.train else image_num_range[0]
    print(f'Max frame number per rank {max_img_per_gpu}')
    if mode == 'train' and cfg.train.iters_per_epoch > 0:
        # Sanity check to ensure the dataset is large enough for the planned iterations.
        print('Needed batch number per epoch (per rank):', (max_img_per_gpu // image_num_range[0]) * cfg.train.iters_per_epoch)
        print('Dataset length per rank:', len(dataset) // world_size)
        assert (max_img_per_gpu // image_num_range[0]) * cfg.train.iters_per_epoch < len(dataset) // world_size

    # Sampler for distributed training.
    sampler = DynamicDistributedSampler(dataset, seed=cfg.train.base_seed, shuffle=cfg_dataloader.shuffle, rank=rank, drop_last=cfg_dataloader.drop_last)
    
    # Batch sampler that dynamically adjusts batch composition based on resolution and image count.
    batch_sampler = DynamicBatchSampler(
        sampler, 
        num_resolution, 
        image_num_range, 
        seed=cfg.train.base_seed,
        max_img_per_gpu=max_img_per_gpu,
        rank=rank
    )

    # --- 4. Create and Return the DataLoader ---
    return data_loader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=unified_collate_fn
    )