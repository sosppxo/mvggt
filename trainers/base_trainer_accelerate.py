import datetime
import os
import random
from accelerate import Accelerator
import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict
from easydict import EasyDict
import time
import json
import math
import sys
from PIL import Image
import numpy as np
from collections import defaultdict
import torch.nn.functional as F
from datasets import create_dataloader
# from model.network import Network
from utils.misc import get_logger, is_logging_process, pretty_print_hydra_config, move_to_device, get_rank
from utils.basic import seed_anything, count_parameters
from utils.optimizer import build_optimizer
from utils.scheduler import build_scheduler
from utils.dist import (
    MetricLogger,
    SmoothedValue,
    init_distributed_mode,
    setup_for_distributed,
)
from accelerate import DistributedDataParallelKwargs
from transformers.trainer_pt_utils import get_model_param_count
from accelerate import (
    DistributedType,
)
from accelerate.utils import (
    DataLoaderConfiguration,
    DynamoBackend,
    GradientAccumulationPlugin,
    ProjectConfiguration,
    TorchDynamoPlugin,
    set_seed,
    gather_object,
)


class BaseTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        with open_dict(cfg):
            cfg.job_logging_cfg = HydraConfig.get().job_logging
        # random seed
        if cfg.random_seed is None:
            cfg.random_seed = random.randint(1, 10000)
        seed_anything(cfg.random_seed, deterministic=False)
        
        ## Load lookup table for unique/multiple evaluation
        try:
            lookup_path = 'data/lookup.json'
            with open(lookup_path, 'r') as f:
                self.lookup = json.load(f)
        except FileNotFoundError:
            if is_logging_process():
                print("Warning: Lookup table not found at 'data/lookup.json'. Unique/multiple metrics will not be available.")
            self.lookup = None

        ## Load pixel percentage data for validation samples
        try:
            dataset_source = self.cfg.train_dataset.ScanNet.get('dataset_source', 'scanrefer')
            if dataset_source == 'sr3d':
                pixel_data_path = 'data/sr3d_val_scene_pixels.json'
            elif dataset_source == 'nr3d':
                pixel_data_path = 'data/nr3d_val_scene_pixels.json'
            else:
                pixel_data_path = 'data/mvrefer_val.json'
            
            if is_logging_process():
                print(f"Loading pixel data from: {pixel_data_path}")
            with open(pixel_data_path, 'r') as f:
                self.pixel_data = json.load(f)
        except FileNotFoundError:
            if is_logging_process():
                print(f"Warning: Pixel data not found at '{pixel_data_path}'. Per-view pixel percentages will not be available for logging.")
            self.pixel_data = None
                    
        ## Variables to store best epoch's detailed IoU results
        self.best_epoch_details = {}
        self.best_val_score_for_details = -float('inf')
  
        ## 1. Build accelerator
        self.build_accelerator()

        if is_logging_process():
            pretty_print_hydra_config(cfg)

        ## 2. Prepare model
        self.log_info("Preparing model...")
        self.model = self.prepare_model()
        self.n_learnable_parameters = get_model_param_count(
            self.model, trainable_only=True
        )
        self.n_fix_parameters = get_model_param_count(
            self.model, trainable_only=False
        )
        self.accelerator.wait_for_everyone()

        ## 3. Prepare dataloader
        self.log_info("Making train dataloader...")
        self.train_loader = create_dataloader(cfg, 'train')
        self.log_info("Making test dataloader...")
        self.test_loader = create_dataloader(cfg, 'test')
        self.accelerator.wait_for_everyone()

        ## 5. Prepare optimizer and scheduler (fsdp should after preparing the model using accelerate)
        if self.cfg.get("fsdp_plugin"):
            self.model = self.accelerator.prepare(self.model)
            self.accelerator.wait_for_everyone()

            self.optimizer = self.build_optimizer(self.cfg.train.optimizer, self.model)
            self.log_info(f"optimizer: {self.optimizer}")
        else:
            self.optimizer = self.build_optimizer(self.cfg.train.optimizer, self.model)
            self.log_info(f"optimizer: {self.optimizer}")

            self.model = self.accelerator.prepare(self.model)
            self.accelerator.wait_for_everyone()

        # Create the LR scheduler
        self.iters_per_epoch = self.cfg.train.iters_per_epoch if self.cfg.train.iters_per_epoch > 0 else len(self.train_loader)
        self.iters_per_test = self.cfg.test.iters_per_test if self.cfg.test.iters_per_test > 0 else len(self.test_loader)
        self.cfg.train.lr_scheduler.total_steps = self.cfg.train.num_epoch * self.iters_per_epoch
        self.log_info(f"Total step for lr scheduler: {self.cfg.train.lr_scheduler.total_steps} ({self.cfg.train.num_epoch} * {self.iters_per_epoch})")
        self.lr_scheduler = build_scheduler(
            self.cfg.train.lr_scheduler, optimizer=self.optimizer
        )
        self.log_info(f"LRScheduler: {self.lr_scheduler}")

        ## 6. Prepare accelerate training
        self.prepare_training()

    def build_optimizer(self, cfg_optimizer, model, param_group_fn=None):
        return build_optimizer(cfg_optimizer, model, param_group_fn=param_group_fn)

    def prepare_training(self):
        # report model details
        self.log_info(
            f"total number of learnable params: {self.n_learnable_parameters / 1e6} M"
        )
        self.log_info(
            f"total number of fixed params: {self.n_fix_parameters / 1e6} M"
        )

        # Wrap the model, optmizer, and scheduler with accelerate
        self.log_info("before accelerator.prepare")

        # don't wrap dataloader
        (
            self.optimizer,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.optimizer, self.lr_scheduler
        )

        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(os.path.basename(self.cfg.log.output_dir))

        # Report the training info
        self.total_batch_size = (
            self.cfg.train.batch_size
            * self.accelerator.num_processes
            * self.cfg.train.gradient_accumulation_steps
        )
        self.log_info("***** Running training *****")
        self.log_info(f"LR = {self.cfg.train.optimizer.lr:.8f}")
        self.log_info(f"Weigth Decay = {self.cfg.train.optimizer.weight_decay:.8f}")
        self.log_info(f"Instantaneous batch size per device = {self.cfg.train.batch_size}")
        self.log_info(f"Total Batch size = {self.total_batch_size}")
        self.log_info(
            f"Gradient Accumulation steps = {self.accelerator.gradient_accumulation_steps}"
        )
        self.log_info(f"Number of epochs = {self.cfg.train.num_epoch}")
        self.log_info(
            f"Number of training steps per epoch = {self.iters_per_epoch}"
        )
        self.log_info(
            f"Number of total training steps = {self.iters_per_epoch * self.cfg.train.num_epoch}"
        )
        # self.log_info(f"Number of training examples per epoch = {len(self.dataloader.dataset)}")
        self.log_info(
            f"Number of model parameters = {self.n_fix_parameters / 1e6:.2f}M"
        )
        self.log_info(
            f"Number of model trainable parameters = {self.n_learnable_parameters / 1e6:.2f}M"
        )

        # Auto resume the checkpoint
        latest_epoch = self.auto_resume()
        self.initial_global_step = self.iters_per_epoch * latest_epoch
        self.first_epoch = latest_epoch

        os.makedirs(self.cfg.log.ckpt_dir, exist_ok=True)

    def prepare_model(self):
        model = hydra.utils.instantiate(self.cfg.model)
        count_parameters(model)
        return model
    
    def before_epoch(self, epoch):
        pass

    def train(self):
        # eval only
        if self.cfg.train.get('eval_only', False):
            self.log_info("--- Running in evaluation-only mode ---")
            # The checkpoint is already loaded by auto_resume() during initialization.
            if not self.cfg.train.resume:
                self.log_info("Warning: eval_only is True, but no checkpoint was provided in train.resume.")
            
            self.validate(epoch=0)
            self.log_info("--- Evaluation complete, exiting ---")
            self.accelerator.end_training()
            return

        # Start Train!
        start_time = time.time()
        self.accelerator.wait_for_everyone()

        # Initialize variable to track the best validation metric
        best_val_metric = -float('inf')  # For metrics like IoU, higher is better
        best_model_path = None

        for epoch in range(self.first_epoch, self.cfg.train.num_epoch):
            torch.cuda.reset_peak_memory_stats()

            self.before_epoch(epoch)

            train_stats = self.train_one_epoch(epoch)

            # Perform validation at the end of each epoch
            val_stats, current_epoch_details = self.validate(epoch)

            # Logic to save best model based on IoU score
            current_val_metric = val_stats.get("refer_iou_score", -1.0)
            if current_val_metric > best_val_metric:
                best_val_metric = current_val_metric
                best_model_path = os.path.join(
                    self.cfg.log.ckpt_dir,
                    "best_model",
                )
                self.accelerator.save_state(best_model_path, safe_serialization=False)
                self.log_info(f"Saved best model at epoch {epoch} with val_metric: {best_val_metric:.4f}")

            self.accelerator.wait_for_everyone()

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"val_{k}": v for k, v in val_stats.items()},
                "epoch": epoch,
                "n_parameters": self.n_learnable_parameters,
            }

            if self.accelerator.is_main_process:
                # Convert numpy types to standard Python types for JSON serialization
                log_stats_serializable = {k: float(v) if hasattr(v, 'item') else v for k, v in log_stats.items()}
                with open(
                    os.path.join(self.cfg.log.ckpt_dir, "log.txt"),
                    mode="a",
                    encoding="utf-8",
                ) as f:
                    f.write(json.dumps(log_stats_serializable) + "\n")

                self.log_all(log_stats, step=self.global_step)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.log_info("Training time {}".format(total_time_str))

        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()

    def validate(self, epoch):
        self.model.eval()
        metric_logger = MetricLogger(delimiter="  ")
        header = f"Validation Epoch: [{epoch}]"

        total_samples = 0
        
        all_sample_avg_view_ious = []
        all_global_ious = []
        unique_sample_avg_view_ious = []
        unique_global_ious = []
        multiple_sample_avg_view_ious = []
        multiple_global_ious = []

        # New lists for visible/invisible IoUs per category
        all_sample_avg_visible_ious = []
        all_sample_avg_invisible_ious = []
        unique_sample_avg_visible_ious = []
        unique_sample_avg_invisible_ious = []
        multiple_sample_avg_visible_ious = []
        multiple_sample_avg_invisible_ious = []

        hard_sample_avg_view_ious = []
        hard_global_ious = []
        easy_sample_avg_view_ious = []
        easy_global_ious = []
        hard_sample_avg_visible_ious = []
        hard_sample_avg_invisible_ious = []
        easy_sample_avg_visible_ious = []
        easy_sample_avg_invisible_ious = []

        nr3d_hard_ious = []
        nr3d_easy_ious = []
        nr3d_vd_ious = []
        nr3d_vi_ious = []

        nr3d_hard_sample_avg_view_ious = []
        nr3d_hard_sample_avg_visible_ious = []
        nr3d_hard_sample_avg_invisible_ious = []
        nr3d_easy_sample_avg_view_ious = []
        nr3d_easy_sample_avg_visible_ious = []
        nr3d_easy_sample_avg_invisible_ious = []
        nr3d_vd_sample_avg_view_ious = []
        nr3d_vd_sample_avg_visible_ious = []
        nr3d_vd_sample_avg_invisible_ious = []
        nr3d_vi_sample_avg_view_ious = []
        nr3d_vi_sample_avg_visible_ious = []
        nr3d_vi_sample_avg_invisible_ious = []
        
        current_epoch_details_list = []
        
        total_processed_in_val_loop = 0
        samples_with_iou_metric = 0
        samples_in_lookup = 0
        samples_not_in_lookup = 0
        samples_without_ann_id = 0

        self.log_info(f"Start validation for epoch {epoch}")
        with torch.no_grad():
            for i, batch in enumerate(metric_logger.log_every(
                self.test_loader, self.cfg.train.print_freq, header
            )):

                # Stop validation early if iters_per_test is set
                if self.iters_per_test > 0 and i >= self.iters_per_test:
                    self.log_info(f"Stopping validation early after {self.iters_per_test} iterations.")
                    break
                
                batch = move_to_device(batch, self.accelerator.device)
                # Forward pass and loss calculation
                predictions = self.forward_batch(batch, mode='test')
                metrics = self.calculate_loss(predictions, batch, mode='test')
                
                # --- Collect detailed IoU information for saving ---
                if self.pixel_data and 'refer_iou_score_global_per_sample' in metrics and 'refer_iou_per_view' in metrics:
                    batched_text = batch[1]
                    scene_ids = batched_text['scene_id']
                    object_ids = batched_text['object_id']
                    ann_ids = batched_text.get('ann_id', [None] * len(scene_ids))
                    
                    sample_ious = metrics['refer_iou_score_global_per_sample'].cpu().numpy()
                    per_view_ious = metrics['refer_iou_per_view'].cpu().numpy()

                    for b in range(len(scene_ids)):
                        # Ensure all IDs are converted to native Python strings for robust serialization.
                        scene_id_val = scene_ids[b]
                        if isinstance(scene_id_val, torch.Tensor):
                            scene_id_val = scene_id_val.item()
                        
                        object_id_val = object_ids[b]
                        if isinstance(object_id_val, torch.Tensor):
                            object_id_val = object_id_val.item()

                        ann_id_val = ann_ids[b]
                        if isinstance(ann_id_val, torch.Tensor):
                            ann_id_val = ann_id_val.item()
                        ann_id_str = str(ann_id_val) if ann_id_val is not None else None

                        # The batch is collated view-wise. To get all frame IDs for a single sample `b`,
                        # we need to iterate through all view-dictionaries in `batch[0]` and pick the b-th element from the 'instance' list.
                        frame_ids = [view_dict['instance'][b] for view_dict in batch[0]]

                        current_epoch_details_list.append({
                            "scene_id": str(scene_id_val),
                            "object_id": str(object_id_val),
                            "ann_id": ann_id_str,
                            "frame_ids": frame_ids, # Pass the frame IDs along
                            "sample_iou": sample_ious[b],
                            "per_view_ious": per_view_ious[b]
                        })

                # Store IoU and uniqueness for final metric calculation
                current_batch_size_for_stats = len(batch[1]['scene_id'])
                total_processed_in_val_loop += current_batch_size_for_stats
   
                if 'refer_iou_per_view' in metrics and 'refer_iou_score_global_per_sample' in metrics:
                    samples_with_iou_metric += current_batch_size_for_stats
                    
                    # This function will now always return 'all_*' stats, and conditionally 'unique_*'/'multiple_*' stats
                    batch_stats = self._collect_uniqueness_stats(metrics, batch)
                    
                    all_global_ious.extend(batch_stats['all_global_ious'])
                    all_sample_avg_view_ious.extend(batch_stats['all_sample_avg_view_ious'])
                    all_sample_avg_visible_ious.extend(batch_stats['all_sample_avg_visible_ious'])
                    all_sample_avg_invisible_ious.extend(batch_stats['all_sample_avg_invisible_ious'])
                    
                    # Conditionally extend lists for ScanRefer-specific metrics
                    if self.lookup:
                        unique_global_ious.extend(batch_stats['unique_global_ious'])
                        unique_sample_avg_view_ious.extend(batch_stats['unique_sample_avg_view_ious'])
                        multiple_global_ious.extend(batch_stats['multiple_global_ious'])
                        multiple_sample_avg_view_ious.extend(batch_stats['multiple_sample_avg_view_ious'])
                        
                        unique_sample_avg_visible_ious.extend(batch_stats['unique_sample_avg_visible_ious'])
                        unique_sample_avg_invisible_ious.extend(batch_stats['unique_sample_avg_invisible_ious'])
                        multiple_sample_avg_visible_ious.extend(batch_stats['multiple_sample_avg_visible_ious'])
                        multiple_sample_avg_invisible_ious.extend(batch_stats['multiple_sample_avg_invisible_ious'])

                        # Update counters
                        samples_without_ann_id += batch_stats['samples_without_ann_id']
                        samples_in_lookup += batch_stats['samples_in_lookup']
                        samples_not_in_lookup += batch_stats['samples_not_in_lookup']
                        
                if self.pixel_data and 'refer_iou_per_view' in metrics and 'refer_iou_score_global_per_sample' in metrics:
                    difficulty_stats = self._collect_difficulty_stats(metrics, batch)
                    hard_global_ious.extend(difficulty_stats['hard_global_ious'])
                    hard_sample_avg_view_ious.extend(difficulty_stats['hard_sample_avg_view_ious'])
                    easy_global_ious.extend(difficulty_stats['easy_global_ious'])
                    easy_sample_avg_view_ious.extend(difficulty_stats['easy_sample_avg_view_ious'])
                    hard_sample_avg_visible_ious.extend(difficulty_stats['hard_sample_avg_visible_ious'])
                    hard_sample_avg_invisible_ious.extend(difficulty_stats['hard_sample_avg_invisible_ious'])
                    easy_sample_avg_visible_ious.extend(difficulty_stats['easy_sample_avg_visible_ious'])
                    easy_sample_avg_invisible_ious.extend(difficulty_stats['easy_sample_avg_invisible_ious'])

                if self.cfg.train_dataset.ScanNet.get('dataset_source') in ['nr3d', 'sr3d'] and 'refer_iou_score_global_per_sample' in metrics:
                     nr3d_stats = self._collect_nr3d_stats(metrics, batch)
                     nr3d_hard_ious.extend(nr3d_stats['nr3d_hard_ious'])
                     nr3d_easy_ious.extend(nr3d_stats['nr3d_easy_ious'])
                     nr3d_vd_ious.extend(nr3d_stats['nr3d_vd_ious'])
                     nr3d_vi_ious.extend(nr3d_stats['nr3d_vi_ious'])
                     # Also extend the per-view stats
                     nr3d_hard_sample_avg_view_ious.extend(nr3d_stats['nr3d_hard_sample_avg_view_ious'])
                     nr3d_hard_sample_avg_visible_ious.extend(nr3d_stats['nr3d_hard_sample_avg_visible_ious'])
                     nr3d_hard_sample_avg_invisible_ious.extend(nr3d_stats['nr3d_hard_sample_avg_invisible_ious'])
                     nr3d_easy_sample_avg_view_ious.extend(nr3d_stats['nr3d_easy_sample_avg_view_ious'])
                     nr3d_easy_sample_avg_visible_ious.extend(nr3d_stats['nr3d_easy_sample_avg_visible_ious'])
                     nr3d_easy_sample_avg_invisible_ious.extend(nr3d_stats['nr3d_easy_sample_avg_invisible_ious'])
                     nr3d_vd_sample_avg_view_ious.extend(nr3d_stats['nr3d_vd_sample_avg_view_ious'])
                     nr3d_vd_sample_avg_visible_ious.extend(nr3d_stats['nr3d_vd_sample_avg_visible_ious'])
                     nr3d_vd_sample_avg_invisible_ious.extend(nr3d_stats['nr3d_vd_sample_avg_invisible_ious'])
                     nr3d_vi_sample_avg_view_ious.extend(nr3d_stats['nr3d_vi_sample_avg_view_ious'])
                     nr3d_vi_sample_avg_visible_ious.extend(nr3d_stats['nr3d_vi_sample_avg_visible_ious'])
                     nr3d_vi_sample_avg_invisible_ious.extend(nr3d_stats['nr3d_vi_sample_avg_invisible_ious'])

                # Update metrics, excluding per-sample/per-view tensors which the logger cannot average.
                logger_metrics = {k: v for k, v in metrics.items() if 'per_sample' not in k and 'per_view' not in k}
                metric_logger.update(**logger_metrics)

                # Correctly accumulate total samples
                current_batch_size = len(batch[1]['scene_id'])
                total_samples += current_batch_size


        # Gather the stats from all processes
        metric_logger.synchronize_between_processes()
        
        # Gather and structure the detailed IoU information
        structured_details = {}
        if self.pixel_data:
            # Gather the list of dicts from all processes
            gathered_details_lists = gather_object(current_epoch_details_list)            
            
            if self.accelerator.is_main_process:
                # In single-GPU or CPU mode, gather_object might return a flat list directly.
                # In multi-GPU, it returns a list of lists. We handle both by checking the first element.
                if gathered_details_lists and isinstance(gathered_details_lists[0], list):
                    all_details = [item for sublist in gathered_details_lists for item in sublist]
                else:
                    all_details = gathered_details_lists
                    
                for item in all_details:
                    scene_id = item['scene_id']
                    object_id = item['object_id']
                    ann_id = item['ann_id']
                    frame_ids = item['frame_ids'] # Retrieve the frame IDs

                    # Add a safeguard: Due to batching/gathering complexities, frame_ids might be nested.
                    # We ensure it's a flat list of strings before proceeding.
                    if frame_ids and isinstance(frame_ids[0], list):
                        frame_ids = frame_ids[0]
                        
                    if not ann_id:
                        continue
                        
                    # Lookup pixel percentages using frame_ids
                    try:
                        # Access the dictionary of frames for the current object
                        frame_pixel_data = self.pixel_data[scene_id][object_id]
                        # Create the list of percentages using a clear loop to ensure one-to-one mapping.
                        pixel_percentages = []
                        for fid in frame_ids:
                            pixel_percentages.append(frame_pixel_data.get(fid, 0.0))
                    except KeyError:
                        continue # Skip if scene_id or object_id is not in our pixel data file

                    # Ensure we have the same number of views
                    if len(pixel_percentages) != len(item['per_view_ious']):
                        continue
                    
                    views_data = [
                        {"frame_id": fid, "pixel_percentage": pp, "iou": float(iou)} 
                        for fid, pp, iou in zip(frame_ids, pixel_percentages, item['per_view_ious'])
                    ]

                    # Create nested dictionary structure
                    structured_details.setdefault(scene_id, {}).setdefault(object_id, {})[ann_id] = {
                        "sample_iou": float(item['sample_iou']),
                        "views": views_data
                    }
                    
        # Aggregate and log sample statistics
        stats_tensor = torch.tensor([
            total_processed_in_val_loop, 
            samples_with_iou_metric, 
            samples_in_lookup, 
            samples_not_in_lookup,
            samples_without_ann_id
        ], device=self.accelerator.device, dtype=torch.long)
        
        dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)

        # Calculate and log unique/multiple accuracy metrics
        final_metrics = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        
        acc_metrics = {}
        dataset_source = self.cfg.train_dataset.ScanNet.get('dataset_source', 'scanrefer')

        # Overall
        all_sample_avg_view_ious_tensor = torch.tensor(all_sample_avg_view_ious, device=self.accelerator.device, dtype=torch.float32)
        all_global_ious_tensor = torch.tensor(all_global_ious, device=self.accelerator.device, dtype=torch.float32)
        all_sample_avg_visible_ious_tensor = torch.tensor(all_sample_avg_visible_ious, device=self.accelerator.device, dtype=torch.float32)
        all_sample_avg_invisible_ious_tensor = torch.tensor(all_sample_avg_invisible_ious, device=self.accelerator.device, dtype=torch.float32)
        gathered_sample_avg_view_overall = self.accelerator.gather_for_metrics(all_sample_avg_view_ious_tensor)
        gathered_global_overall = self.accelerator.gather_for_metrics(all_global_ious_tensor)
        gathered_sample_avg_visible_overall = self.accelerator.gather_for_metrics(all_sample_avg_visible_ious_tensor)
        gathered_sample_avg_invisible_overall = self.accelerator.gather_for_metrics(all_sample_avg_invisible_ious_tensor)

        # Hard/Easy (pixel-based)
        hard_sample_avg_view_ious_tensor = torch.tensor(hard_sample_avg_view_ious, device=self.accelerator.device, dtype=torch.float32)
        hard_global_ious_tensor = torch.tensor(hard_global_ious, device=self.accelerator.device, dtype=torch.float32)
        easy_sample_avg_view_ious_tensor = torch.tensor(easy_sample_avg_view_ious, device=self.accelerator.device, dtype=torch.float32)
        easy_global_ious_tensor = torch.tensor(easy_global_ious, device=self.accelerator.device, dtype=torch.float32)
        hard_sample_avg_visible_ious_tensor = torch.tensor(hard_sample_avg_visible_ious, device=self.accelerator.device, dtype=torch.float32)
        hard_sample_avg_invisible_ious_tensor = torch.tensor(hard_sample_avg_invisible_ious, device=self.accelerator.device, dtype=torch.float32)
        easy_sample_avg_visible_ious_tensor = torch.tensor(easy_sample_avg_visible_ious, device=self.accelerator.device, dtype=torch.float32)
        easy_sample_avg_invisible_ious_tensor = torch.tensor(easy_sample_avg_invisible_ious, device=self.accelerator.device, dtype=torch.float32)
        gathered_sample_avg_view_hard = self.accelerator.gather_for_metrics(hard_sample_avg_view_ious_tensor)
        gathered_global_hard = self.accelerator.gather_for_metrics(hard_global_ious_tensor)
        gathered_sample_avg_view_easy = self.accelerator.gather_for_metrics(easy_sample_avg_view_ious_tensor)
        gathered_global_easy = self.accelerator.gather_for_metrics(easy_global_ious_tensor)
        gathered_sample_avg_visible_hard = self.accelerator.gather_for_metrics(hard_sample_avg_visible_ious_tensor)
        gathered_sample_avg_invisible_hard = self.accelerator.gather_for_metrics(hard_sample_avg_invisible_ious_tensor)
        gathered_sample_avg_visible_easy = self.accelerator.gather_for_metrics(easy_sample_avg_visible_ious_tensor)
        gathered_sample_avg_invisible_easy = self.accelerator.gather_for_metrics(easy_sample_avg_invisible_ious_tensor)

        # Unique/Multiple (for ScanRefer)
        if self.lookup and dataset_source == 'scanrefer':
            unique_sample_avg_view_ious_tensor = torch.tensor(unique_sample_avg_view_ious, device=self.accelerator.device, dtype=torch.float32)
            unique_global_ious_tensor = torch.tensor(unique_global_ious, device=self.accelerator.device, dtype=torch.float32)
            multiple_sample_avg_view_ious_tensor = torch.tensor(multiple_sample_avg_view_ious, device=self.accelerator.device, dtype=torch.float32)
            multiple_global_ious_tensor = torch.tensor(multiple_global_ious, device=self.accelerator.device, dtype=torch.float32)
            unique_sample_avg_visible_ious_tensor = torch.tensor(unique_sample_avg_visible_ious, device=self.accelerator.device, dtype=torch.float32)
            unique_sample_avg_invisible_ious_tensor = torch.tensor(unique_sample_avg_invisible_ious, device=self.accelerator.device, dtype=torch.float32)
            multiple_sample_avg_visible_ious_tensor = torch.tensor(multiple_sample_avg_visible_ious, device=self.accelerator.device, dtype=torch.float32)
            multiple_sample_avg_invisible_ious_tensor = torch.tensor(multiple_sample_avg_invisible_ious, device=self.accelerator.device, dtype=torch.float32)
            gathered_sample_avg_view_unique = self.accelerator.gather_for_metrics(unique_sample_avg_view_ious_tensor)
            gathered_global_unique = self.accelerator.gather_for_metrics(unique_global_ious_tensor)
            gathered_sample_avg_view_multiple = self.accelerator.gather_for_metrics(multiple_sample_avg_view_ious_tensor)
            gathered_global_multiple = self.accelerator.gather_for_metrics(multiple_global_ious_tensor)
            gathered_sample_avg_visible_unique = self.accelerator.gather_for_metrics(unique_sample_avg_visible_ious_tensor)
            gathered_sample_avg_invisible_unique = self.accelerator.gather_for_metrics(unique_sample_avg_invisible_ious_tensor)
            gathered_sample_avg_visible_multiple = self.accelerator.gather_for_metrics(multiple_sample_avg_visible_ious_tensor)
            gathered_sample_avg_invisible_multiple = self.accelerator.gather_for_metrics(multiple_sample_avg_invisible_ious_tensor)

        # Nr3D/Sr3D specific
        if dataset_source in ['nr3d', 'sr3d']:
            nr3d_hard_ious_tensor = torch.tensor(nr3d_hard_ious, device=self.accelerator.device, dtype=torch.float32)
            nr3d_easy_ious_tensor = torch.tensor(nr3d_easy_ious, device=self.accelerator.device, dtype=torch.float32)
            nr3d_vd_ious_tensor = torch.tensor(nr3d_vd_ious, device=self.accelerator.device, dtype=torch.float32)
            nr3d_vi_ious_tensor = torch.tensor(nr3d_vi_ious, device=self.accelerator.device, dtype=torch.float32)
            nr3d_hard_sample_avg_view_ious_tensor = torch.tensor(nr3d_hard_sample_avg_view_ious, device=self.accelerator.device, dtype=torch.float32)
            nr3d_hard_sample_avg_visible_ious_tensor = torch.tensor(nr3d_hard_sample_avg_visible_ious, device=self.accelerator.device, dtype=torch.float32)
            nr3d_hard_sample_avg_invisible_ious_tensor = torch.tensor(nr3d_hard_sample_avg_invisible_ious, device=self.accelerator.device, dtype=torch.float32)
            nr3d_easy_sample_avg_view_ious_tensor = torch.tensor(nr3d_easy_sample_avg_view_ious, device=self.accelerator.device, dtype=torch.float32)
            nr3d_easy_sample_avg_visible_ious_tensor = torch.tensor(nr3d_easy_sample_avg_visible_ious, device=self.accelerator.device, dtype=torch.float32)
            nr3d_easy_sample_avg_invisible_ious_tensor = torch.tensor(nr3d_easy_sample_avg_invisible_ious, device=self.accelerator.device, dtype=torch.float32)
            nr3d_vd_sample_avg_view_ious_tensor = torch.tensor(nr3d_vd_sample_avg_view_ious, device=self.accelerator.device, dtype=torch.float32)
            nr3d_vd_sample_avg_visible_ious_tensor = torch.tensor(nr3d_vd_sample_avg_visible_ious, device=self.accelerator.device, dtype=torch.float32)
            nr3d_vd_sample_avg_invisible_ious_tensor = torch.tensor(nr3d_vd_sample_avg_invisible_ious, device=self.accelerator.device, dtype=torch.float32)
            nr3d_vi_sample_avg_view_ious_tensor = torch.tensor(nr3d_vi_sample_avg_view_ious, device=self.accelerator.device, dtype=torch.float32)
            nr3d_vi_sample_avg_visible_ious_tensor = torch.tensor(nr3d_vi_sample_avg_visible_ious, device=self.accelerator.device, dtype=torch.float32)
            nr3d_vi_sample_avg_invisible_ious_tensor = torch.tensor(nr3d_vi_sample_avg_invisible_ious, device=self.accelerator.device, dtype=torch.float32)
            gathered_nr3d_hard = self.accelerator.gather_for_metrics(nr3d_hard_ious_tensor)
            gathered_nr3d_easy = self.accelerator.gather_for_metrics(nr3d_easy_ious_tensor)
            gathered_nr3d_vd = self.accelerator.gather_for_metrics(nr3d_vd_ious_tensor)
            gathered_nr3d_vi = self.accelerator.gather_for_metrics(nr3d_vi_ious_tensor)
            gathered_nr3d_hard_sample_avg_view = self.accelerator.gather_for_metrics(nr3d_hard_sample_avg_view_ious_tensor)
            gathered_nr3d_hard_sample_avg_visible = self.accelerator.gather_for_metrics(nr3d_hard_sample_avg_visible_ious_tensor)
            gathered_nr3d_hard_sample_avg_invisible = self.accelerator.gather_for_metrics(nr3d_hard_sample_avg_invisible_ious_tensor)
            gathered_nr3d_easy_sample_avg_view = self.accelerator.gather_for_metrics(nr3d_easy_sample_avg_view_ious_tensor)
            gathered_nr3d_easy_sample_avg_visible = self.accelerator.gather_for_metrics(nr3d_easy_sample_avg_visible_ious_tensor)
            gathered_nr3d_easy_sample_avg_invisible = self.accelerator.gather_for_metrics(nr3d_easy_sample_avg_invisible_ious_tensor)
            gathered_nr3d_vd_sample_avg_view = self.accelerator.gather_for_metrics(nr3d_vd_sample_avg_view_ious_tensor)
            gathered_nr3d_vd_sample_avg_visible = self.accelerator.gather_for_metrics(nr3d_vd_sample_avg_visible_ious_tensor)
            gathered_nr3d_vd_sample_avg_invisible = self.accelerator.gather_for_metrics(nr3d_vd_sample_avg_invisible_ious_tensor)
            gathered_nr3d_vi_sample_avg_view = self.accelerator.gather_for_metrics(nr3d_vi_sample_avg_view_ious_tensor)
            gathered_nr3d_vi_sample_avg_visible = self.accelerator.gather_for_metrics(nr3d_vi_sample_avg_visible_ious_tensor)
            gathered_nr3d_vi_sample_avg_invisible = self.accelerator.gather_for_metrics(nr3d_vi_sample_avg_invisible_ious_tensor)
        
        if self.accelerator.is_main_process:
            # Overall
            global_ious_overall = gathered_global_overall.cpu().numpy()
            sample_avg_view_ious_overall = gathered_sample_avg_view_overall.cpu().numpy()
            sample_avg_visible_ious_overall = gathered_sample_avg_visible_overall.cpu().numpy()
            sample_avg_invisible_ious_overall = gathered_sample_avg_invisible_overall.cpu().numpy()
            overall_metrics = self._calculate_metrics_for_category(global_ious_overall, sample_avg_view_ious_overall, sample_avg_visible_ious_overall, sample_avg_invisible_ious_overall)
            acc_metrics.update({
                "Acc@0.25_per_view_overall": overall_metrics['acc25_per_view_all'], "Acc@0.50_per_view_overall": overall_metrics['acc50_per_view_all'], "iou_per_view_overall": overall_metrics['iou_per_view_all_mean'],
                "Acc@0.25_global_overall": overall_metrics['acc25_global'], "Acc@0.50_global_overall": overall_metrics['acc50_global'], "iou_global_overall": overall_metrics['iou_global_mean'],
            })

            # Hard/Easy (pixel-based)
            global_ious_hard = gathered_global_hard.cpu().numpy()
            sample_avg_view_ious_hard = gathered_sample_avg_view_hard.cpu().numpy()
            sample_avg_visible_ious_hard = gathered_sample_avg_visible_hard.cpu().numpy()
            sample_avg_invisible_ious_hard = gathered_sample_avg_invisible_hard.cpu().numpy()
            hard_metrics = self._calculate_metrics_for_category(global_ious_hard, sample_avg_view_ious_hard, sample_avg_visible_ious_hard, sample_avg_invisible_ious_hard)

            global_ious_easy = gathered_global_easy.cpu().numpy()
            sample_avg_view_ious_easy = gathered_sample_avg_view_easy.cpu().numpy()
            sample_avg_visible_ious_easy = gathered_sample_avg_visible_easy.cpu().numpy()
            sample_avg_invisible_ious_easy = gathered_sample_avg_invisible_easy.cpu().numpy()
            easy_metrics = self._calculate_metrics_for_category(global_ious_easy, sample_avg_view_ious_easy, sample_avg_visible_ious_easy, sample_avg_invisible_ious_easy)

            acc_metrics.update({
                "Acc@0.25_per_view_hard": hard_metrics['acc25_per_view_all'], "Acc@0.50_per_view_hard": hard_metrics['acc50_per_view_all'], "iou_per_view_hard": hard_metrics['iou_per_view_all_mean'],
                "Acc@0.25_global_hard": hard_metrics['acc25_global'], "Acc@0.50_global_hard": hard_metrics['acc50_global'], "iou_global_hard": hard_metrics['iou_global_mean'],
                "Acc@0.25_per_view_easy": easy_metrics['acc25_per_view_all'], "Acc@0.50_per_view_easy": easy_metrics['acc50_per_view_all'], "iou_per_view_easy": easy_metrics['iou_per_view_all_mean'],
                "Acc@0.25_global_easy": easy_metrics['acc25_global'], "Acc@0.50_global_easy": easy_metrics['acc50_global'], "iou_global_easy": easy_metrics['iou_global_mean'],
            })
            
            # --- LOGGING ---
            self.log_info("--- Accuracy Metrics ---")
            self.log_info(f"Overall ({len(global_ious_overall)} samples):")
            self.log_info(f"  - Per-View IoU (all):       Acc@0.25: {overall_metrics['acc25_per_view_all']:.4f}, Acc@0.50: {overall_metrics['acc50_per_view_all']:.4f}, Mean IoU: {overall_metrics['iou_per_view_all_mean']:.4f}")
            self.log_info(f"  - Per-View IoU (Visible):   Acc@0.25: {overall_metrics['acc25_per_view_visible']:.4f}, Acc@0.50: {overall_metrics['acc50_per_view_visible']:.4f}, Mean IoU: {overall_metrics['iou_per_view_visible_mean']:.4f}")
            self.log_info(f"  - Per-View IoU (Invisible): Acc@0.25: {overall_metrics['acc25_per_view_invisible']:.4f}, Acc@0.50: {overall_metrics['acc50_per_view_invisible']:.4f}, Mean IoU: {overall_metrics['iou_per_view_invisible_mean']:.4f}")
            self.log_info(f"  - Global IoU based:         Acc@0.25: {overall_metrics['acc25_global']:.4f}, Acc@0.50: {overall_metrics['acc50_global']:.4f}, Mean IoU: {overall_metrics['iou_global_mean']:.4f}")
            
            # Always log pixel-based metrics
            self.log_info(f"--- Pixel-based Difficulty Metrics ---")
            self.log_info(f"  - Hard ({len(global_ious_hard)} samples):")
            self.log_info(f"    - Per-View IoU (all):       Acc@0.25: {hard_metrics['acc25_per_view_all']:.4f}, Acc@0.50: {hard_metrics['acc50_per_view_all']:.4f}, Mean IoU: {hard_metrics['iou_per_view_all_mean']:.4f}")
            self.log_info(f"    - Per-View IoU (Visible):   Acc@0.25: {hard_metrics['acc25_per_view_visible']:.4f}, Acc@0.50: {hard_metrics['acc50_per_view_visible']:.4f}, Mean IoU: {hard_metrics['iou_per_view_visible_mean']:.4f}")
            self.log_info(f"    - Per-View IoU (Invisible): Acc@0.25: {hard_metrics['acc25_per_view_invisible']:.4f}, Acc@0.50: {hard_metrics['acc50_per_view_invisible']:.4f}, Mean IoU: {hard_metrics['iou_per_view_invisible_mean']:.4f}")
            self.log_info(f"    - Global IoU based:         Acc@0.25: {hard_metrics['acc25_global']:.4f}, Acc@0.50: {hard_metrics['acc50_global']:.4f}, Mean IoU: {hard_metrics['iou_global_mean']:.4f}")
            self.log_info(f"  - Easy ({len(global_ious_easy)} samples):")
            self.log_info(f"    - Per-View IoU (all):       Acc@0.25: {easy_metrics['acc25_per_view_all']:.4f}, Acc@0.50: {easy_metrics['acc50_per_view_all']:.4f}, Mean IoU: {easy_metrics['iou_per_view_all_mean']:.4f}")
            self.log_info(f"    - Per-View IoU (Visible):   Acc@0.25: {easy_metrics['acc25_per_view_visible']:.4f}, Acc@0.50: {easy_metrics['acc50_per_view_visible']:.4f}, Mean IoU: {easy_metrics['iou_per_view_visible_mean']:.4f}")
            self.log_info(f"    - Per-View IoU (Invisible): Acc@0.25: {easy_metrics['acc25_per_view_invisible']:.4f}, Acc@0.50: {easy_metrics['acc50_per_view_invisible']:.4f}, Mean IoU: {easy_metrics['iou_per_view_invisible_mean']:.4f}")
            self.log_info(f"    - Global IoU based:         Acc@0.25: {easy_metrics['acc25_global']:.4f}, Acc@0.50: {easy_metrics['acc50_global']:.4f}, Mean IoU: {easy_metrics['iou_global_mean']:.4f}")

            # --- CONDITIONAL LOGIC & CALCULATION ---
            if self.lookup and dataset_source == 'scanrefer':
                sample_avg_view_ious_unique = gathered_sample_avg_view_unique.cpu().numpy()
                global_ious_unique = gathered_global_unique.cpu().numpy()
                sample_avg_view_ious_multiple = gathered_sample_avg_view_multiple.cpu().numpy()
                global_ious_multiple = gathered_global_multiple.cpu().numpy()
                sample_avg_visible_ious_unique = gathered_sample_avg_visible_unique.cpu().numpy()
                sample_avg_invisible_ious_unique = gathered_sample_avg_invisible_unique.cpu().numpy()
                sample_avg_visible_ious_multiple = gathered_sample_avg_visible_multiple.cpu().numpy()
                sample_avg_invisible_ious_multiple = gathered_sample_avg_invisible_multiple.cpu().numpy()
                unique_metrics = self._calculate_metrics_for_category(global_ious_unique, sample_avg_view_ious_unique, sample_avg_visible_ious_unique, sample_avg_invisible_ious_unique)
                multiple_metrics = self._calculate_metrics_for_category(global_ious_multiple, sample_avg_view_ious_multiple, sample_avg_visible_ious_multiple, sample_avg_invisible_ious_multiple)
                acc_metrics.update({
                    "Acc@0.25_per_view_unique": unique_metrics['acc25_per_view_all'], "Acc@0.50_per_view_unique": unique_metrics['acc50_per_view_all'], "iou_per_view_unique": unique_metrics['iou_per_view_all_mean'],
                    "Acc@0.25_global_unique": unique_metrics['acc25_global'], "Acc@0.50_global_unique": unique_metrics['acc50_global'], "iou_global_unique": unique_metrics['iou_global_mean'],
                    "Acc@0.25_per_view_multiple": multiple_metrics['acc25_per_view_all'], "Acc@0.50_per_view_multiple": multiple_metrics['acc50_per_view_all'], "iou_per_view_multiple": multiple_metrics['iou_per_view_all_mean'],
                    "Acc@0.25_global_multiple": multiple_metrics['acc25_global'], "Acc@0.50_global_multiple": multiple_metrics['acc50_global'], "iou_global_multiple": multiple_metrics['iou_global_mean'],
                })
                self.log_info(f"Unique ({len(global_ious_unique)} samples):")
                self.log_info(f"  - Per-View IoU (all):       Acc@0.25: {unique_metrics['acc25_per_view_all']:.4f}, Acc@0.50: {unique_metrics['acc50_per_view_all']:.4f}, Mean IoU: {unique_metrics['iou_per_view_all_mean']:.4f}")
                self.log_info(f"  - Per-View IoU (Visible):   Acc@0.25: {unique_metrics['acc25_per_view_visible']:.4f}, Acc@0.50: {unique_metrics['acc50_per_view_visible']:.4f}, Mean IoU: {unique_metrics['iou_per_view_visible_mean']:.4f}")
                self.log_info(f"  - Per-View IoU (Invisible): Acc@0.25: {unique_metrics['acc25_per_view_invisible']:.4f}, Acc@0.50: {unique_metrics['acc50_per_view_invisible']:.4f}, Mean IoU: {unique_metrics['iou_per_view_invisible_mean']:.4f}")
                self.log_info(f"  - Global IoU based:         Acc@0.25: {unique_metrics['acc25_global']:.4f}, Acc@0.50: {unique_metrics['acc50_global']:.4f}, Mean IoU: {unique_metrics['iou_global_mean']:.4f}")
                self.log_info(f"Multiple ({len(global_ious_multiple)} samples):")
                self.log_info(f"  - Per-View IoU (all):       Acc@0.25: {multiple_metrics['acc25_per_view_all']:.4f}, Acc@0.50: {multiple_metrics['acc50_per_view_all']:.4f}, Mean IoU: {multiple_metrics['iou_per_view_all_mean']:.4f}")
                self.log_info(f"  - Per-View IoU (Visible):   Acc@0.25: {multiple_metrics['acc25_per_view_visible']:.4f}, Acc@0.50: {multiple_metrics['acc50_per_view_visible']:.4f}, Mean IoU: {multiple_metrics['iou_per_view_visible_mean']:.4f}")
                self.log_info(f"  - Per-View IoU (Invisible): Acc@0.25: {multiple_metrics['acc25_per_view_invisible']:.4f}, Acc@0.50: {multiple_metrics['acc50_per_view_invisible']:.4f}, Mean IoU: {multiple_metrics['iou_per_view_invisible_mean']:.4f}")
                self.log_info(f"  - Global IoU based:         Acc@0.25: {multiple_metrics['acc25_global']:.4f}, Acc@0.50: {multiple_metrics['acc50_global']:.4f}, Mean IoU: {multiple_metrics['iou_global_mean']:.4f}")
            
            if dataset_source in ['nr3d', 'sr3d']:
                nr3d_hard_np = gathered_nr3d_hard.cpu().numpy()
                nr3d_easy_np = gathered_nr3d_easy.cpu().numpy()
                nr3d_vd_np = gathered_nr3d_vd.cpu().numpy()
                nr3d_vi_np = gathered_nr3d_vi.cpu().numpy()
                nr3d_hard_sample_avg_view_np = gathered_nr3d_hard_sample_avg_view.cpu().numpy()
                nr3d_hard_sample_avg_visible_np = gathered_nr3d_hard_sample_avg_visible.cpu().numpy()
                nr3d_hard_sample_avg_invisible_np = gathered_nr3d_hard_sample_avg_invisible.cpu().numpy()
                nr3d_easy_sample_avg_view_np = gathered_nr3d_easy_sample_avg_view.cpu().numpy()
                nr3d_easy_sample_avg_visible_np = gathered_nr3d_easy_sample_avg_visible.cpu().numpy()
                nr3d_easy_sample_avg_invisible_np = gathered_nr3d_easy_sample_avg_invisible.cpu().numpy()
                nr3d_vd_sample_avg_view_np = gathered_nr3d_vd_sample_avg_view.cpu().numpy()
                nr3d_vd_sample_avg_visible_np = gathered_nr3d_vd_sample_avg_visible.cpu().numpy()
                nr3d_vd_sample_avg_invisible_np = gathered_nr3d_vd_sample_avg_invisible.cpu().numpy()
                nr3d_vi_sample_avg_view_np = gathered_nr3d_vi_sample_avg_view.cpu().numpy()
                nr3d_vi_sample_avg_visible_np = gathered_nr3d_vi_sample_avg_visible.cpu().numpy()
                nr3d_vi_sample_avg_invisible_np = gathered_nr3d_vi_sample_avg_invisible.cpu().numpy()
                nr3d_hard_metrics = self._calculate_metrics_for_category(nr3d_hard_np, nr3d_hard_sample_avg_view_np, nr3d_hard_sample_avg_visible_np, nr3d_hard_sample_avg_invisible_np)
                nr3d_easy_metrics = self._calculate_metrics_for_category(nr3d_easy_np, nr3d_easy_sample_avg_view_np, nr3d_easy_sample_avg_visible_np, nr3d_easy_sample_avg_invisible_np)
                nr3d_vd_metrics = self._calculate_metrics_for_category(nr3d_vd_np, nr3d_vd_sample_avg_view_np, nr3d_vd_sample_avg_visible_np, nr3d_vd_sample_avg_invisible_np)
                nr3d_vi_metrics = self._calculate_metrics_for_category(nr3d_vi_np, nr3d_vi_sample_avg_view_np, nr3d_vi_sample_avg_visible_np, nr3d_vi_sample_avg_invisible_np)
                acc_metrics.update({
                    "ReferIt3D_Acc@0.25_Hard": nr3d_hard_metrics['acc25_global'], "ReferIt3D_Acc@0.50_Hard": nr3d_hard_metrics['acc50_global'], "ReferIt3D_mIoU_Hard": nr3d_hard_metrics['iou_global_mean'],
                    "ReferIt3D_Acc@0.25_Easy": nr3d_easy_metrics['acc25_global'], "ReferIt3D_Acc@0.50_Easy": nr3d_easy_metrics['acc50_global'], "ReferIt3D_mIoU_Easy": nr3d_easy_metrics['iou_global_mean'],
                    "ReferIt3D_Acc@0.25_ViewDep": nr3d_vd_metrics['acc25_global'], "ReferIt3D_Acc@0.50_ViewDep": nr3d_vd_metrics['acc50_global'], "ReferIt3D_mIoU_ViewDep": nr3d_vd_metrics['iou_global_mean'],
                    "ReferIt3D_Acc@0.25_ViewIndep": nr3d_vi_metrics['acc25_global'], "ReferIt3D_Acc@0.50_ViewIndep": nr3d_vi_metrics['acc50_global'], "ReferIt3D_mIoU_ViewIndep": nr3d_vi_metrics['iou_global_mean'],
                })

                display_name = 'Nr3D' if dataset_source == 'nr3d' else 'Sr3D'
                self.log_info(f"--- {display_name} Metrics (meta-data based) ---")
                self.log_info(f"  - Hard ({len(nr3d_hard_np)}):")
                self.log_info(f"    - Per-View (all):       Acc@0.25: {nr3d_hard_metrics['acc25_per_view_all']:.4f}, Acc@0.50: {nr3d_hard_metrics['acc50_per_view_all']:.4f}, mIoU: {nr3d_hard_metrics['iou_per_view_all_mean']:.4f}")
                self.log_info(f"    - Per-View (Visible):   Acc@0.25: {nr3d_hard_metrics['acc25_per_view_visible']:.4f}, Acc@0.50: {nr3d_hard_metrics['acc50_per_view_visible']:.4f}, mIoU: {nr3d_hard_metrics['iou_per_view_visible_mean']:.4f}")
                self.log_info(f"    - Per-View (Invisible): Acc@0.25: {nr3d_hard_metrics['acc25_per_view_invisible']:.4f}, Acc@0.50: {nr3d_hard_metrics['acc50_per_view_invisible']:.4f}, mIoU: {nr3d_hard_metrics['iou_per_view_invisible_mean']:.4f}")
                self.log_info(f"    - Global IoU:           Acc@0.25: {nr3d_hard_metrics['acc25_global']:.4f}, Acc@0.50: {nr3d_hard_metrics['acc50_global']:.4f}, mIoU: {nr3d_hard_metrics['iou_global_mean']:.4f}")
                self.log_info(f"  - Easy ({len(nr3d_easy_np)}):")
                self.log_info(f"    - Per-View (all):       Acc@0.25: {nr3d_easy_metrics['acc25_per_view_all']:.4f}, Acc@0.50: {nr3d_easy_metrics['acc50_per_view_all']:.4f}, mIoU: {nr3d_easy_metrics['iou_per_view_all_mean']:.4f}")
                self.log_info(f"    - Per-View (Visible):   Acc@0.25: {nr3d_easy_metrics['acc25_per_view_visible']:.4f}, Acc@0.50: {nr3d_easy_metrics['acc50_per_view_visible']:.4f}, mIoU: {nr3d_easy_metrics['iou_per_view_visible_mean']:.4f}")
                self.log_info(f"    - Per-View (Invisible): Acc@0.25: {nr3d_easy_metrics['acc25_per_view_invisible']:.4f}, Acc@0.50: {nr3d_easy_metrics['acc50_per_view_invisible']:.4f}, mIoU: {nr3d_easy_metrics['iou_per_view_invisible_mean']:.4f}")
                self.log_info(f"    - Global IoU:           Acc@0.25: {nr3d_easy_metrics['acc25_global']:.4f}, Acc@0.50: {nr3d_easy_metrics['acc50_global']:.4f}, mIoU: {nr3d_easy_metrics['iou_global_mean']:.4f}")
                self.log_info(f"  - ViewDep ({len(nr3d_vd_np)}):")
                self.log_info(f"    - Per-View (all):       Acc@0.25: {nr3d_vd_metrics['acc25_per_view_all']:.4f}, Acc@0.50: {nr3d_vd_metrics['acc50_per_view_all']:.4f}, mIoU: {nr3d_vd_metrics['iou_per_view_all_mean']:.4f}")
                self.log_info(f"    - Per-View (Visible):   Acc@0.25: {nr3d_vd_metrics['acc25_per_view_visible']:.4f}, Acc@0.50: {nr3d_vd_metrics['acc50_per_view_visible']:.4f}, mIoU: {nr3d_vd_metrics['iou_per_view_visible_mean']:.4f}")
                self.log_info(f"    - Per-View (Invisible): Acc@0.25: {nr3d_vd_metrics['acc25_per_view_invisible']:.4f}, Acc@0.50: {nr3d_vd_metrics['acc50_per_view_invisible']:.4f}, mIoU: {nr3d_vd_metrics['iou_per_view_invisible_mean']:.4f}")
                self.log_info(f"    - Global IoU:           Acc@0.25: {nr3d_vd_metrics['acc25_global']:.4f}, Acc@0.50: {nr3d_vd_metrics['acc50_global']:.4f}, mIoU: {nr3d_vd_metrics['iou_global_mean']:.4f}")
                self.log_info(f"  - ViewIndep ({len(nr3d_vi_np)}):")
                self.log_info(f"    - Per-View (all):       Acc@0.25: {nr3d_vi_metrics['acc25_per_view_all']:.4f}, Acc@0.50: {nr3d_vi_metrics['acc50_per_view_all']:.4f}, mIoU: {nr3d_vi_metrics['iou_per_view_all_mean']:.4f}")
                self.log_info(f"    - Per-View (Visible):   Acc@0.25: {nr3d_vi_metrics['acc25_per_view_visible']:.4f}, Acc@0.50: {nr3d_vi_metrics['acc50_per_view_visible']:.4f}, mIoU: {nr3d_vi_metrics['iou_per_view_visible_mean']:.4f}")
                self.log_info(f"    - Per-View (Invisible): Acc@0.25: {nr3d_vi_metrics['acc25_per_view_invisible']:.4f}, Acc@0.50: {nr3d_vi_metrics['acc50_per_view_invisible']:.4f}, mIoU: {nr3d_vi_metrics['iou_per_view_invisible_mean']:.4f}")
                self.log_info(f"    - Global IoU:           Acc@0.25: {nr3d_vi_metrics['acc25_global']:.4f}, Acc@0.50: {nr3d_vi_metrics['acc50_global']:.4f}, mIoU: {nr3d_vi_metrics['iou_global_mean']:.4f}")

            final_metrics.update(acc_metrics)
        
        # Log sample count summary
        total_val_samples = len(self.test_loader.dataset)
        # Gather total samples from all processes for accurate counting
        total_samples_tensor = torch.tensor(total_samples, device=self.accelerator.device)
        dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
        total_samples_all_gpus = total_samples_tensor.item()

        self.log_info(f"Validation finished. Processed {total_samples_all_gpus}/{total_val_samples} samples.")
        self.log_info(f"Validation results: {metric_logger}")

        return final_metrics, structured_details

    def _is_hard_sample(self, pixel_percentages):
        """
        Determines if a sample is 'hard' based on its view pixel percentages.
        A sample is hard if:
        1. At least one view has a pixel percentage > 0.
        2. All views with pixel percentage > 0 have a percentage < 0.05.
        """
        positive_pixel_percentages = [p for p in pixel_percentages if p > 0]
        
        if not positive_pixel_percentages:
            return False  # Not hard if no positive pixels
            
        is_hard = all(p < 0.05 for p in positive_pixel_percentages)
        return is_hard

    def _collect_difficulty_stats(self, metrics, batch):
        stats = {
            'hard_global_ious': [], 'hard_sample_avg_view_ious': [], 'hard_sample_avg_visible_ious': [], 'hard_sample_avg_invisible_ious': [],
            'easy_global_ious': [], 'easy_sample_avg_view_ious': [], 'easy_sample_avg_visible_ious': [], 'easy_sample_avg_invisible_ious': []
        }
        
        per_view_ious_raw = metrics['refer_iou_per_view'].cpu().numpy()
        per_sample_global_ious = metrics['refer_iou_score_global_per_sample'].cpu().numpy()
        batched_text = batch[1]
        scene_ids = batched_text['scene_id']
        object_ids = batched_text['object_id']

        for b in range(len(scene_ids)):
            scene_id_val = scene_ids[b]
            if isinstance(scene_id_val, torch.Tensor):
                scene_id_val = scene_id_val.item()
            scene_id_str = str(scene_id_val)

            object_id_val = object_ids[b]
            if isinstance(object_id_val, torch.Tensor):
                object_id_val = object_id_val.item()
            object_id_str = str(object_id_val)

            frame_ids = [view_dict['instance'][b] for view_dict in batch[0]]
            
            try:
                frame_pixel_data = self.pixel_data[scene_id_str][object_id_str]
                pixel_percentages = [frame_pixel_data.get(fid, 0.0) for fid in frame_ids]
                
                is_hard = self._is_hard_sample(pixel_percentages)
                
                sample_view_ious = per_view_ious_raw[b]
                sample_global_iou = per_sample_global_ious[b]
                sample_avg_view_iou = np.mean(sample_view_ious) if len(sample_view_ious) > 0 else 0.0

                # Calculate visible/invisible IoUs
                pixel_percentages_np = np.array(pixel_percentages)
                visible_views_ious = sample_view_ious[pixel_percentages_np > 0]
                invisible_views_ious = sample_view_ious[pixel_percentages_np == 0]
                sample_avg_visible_iou = np.mean(visible_views_ious) if len(visible_views_ious) > 0 else 0.0
                sample_avg_invisible_iou = np.mean(invisible_views_ious) if len(invisible_views_ious) > 0 else 0.0

                if is_hard:
                    stats['hard_global_ious'].append(sample_global_iou)
                    stats['hard_sample_avg_view_ious'].append(sample_avg_view_iou)
                    stats['hard_sample_avg_visible_ious'].append(sample_avg_visible_iou)
                    stats['hard_sample_avg_invisible_ious'].append(sample_avg_invisible_iou)
                else:
                    stats['easy_global_ious'].append(sample_global_iou)
                    stats['easy_sample_avg_view_ious'].append(sample_avg_view_iou)
                    stats['easy_sample_avg_visible_ious'].append(sample_avg_visible_iou)
                    stats['easy_sample_avg_invisible_ious'].append(sample_avg_invisible_iou)

            except (KeyError, IndexError):
                # Skip if scene_id/object_id is not in pixel data or if frame_ids are mismatched
                continue
        
        return stats

    def _collect_uniqueness_stats(self, metrics, batch):
        stats = {
            'all_global_ious': [], 'all_sample_avg_view_ious': [], 'all_sample_avg_visible_ious': [], 'all_sample_avg_invisible_ious': [],
            'unique_global_ious': [], 'unique_sample_avg_view_ious': [], 'unique_sample_avg_visible_ious': [], 'unique_sample_avg_invisible_ious': [],
            'multiple_global_ious': [], 'multiple_sample_avg_view_ious': [], 'multiple_sample_avg_visible_ious': [], 'multiple_sample_avg_invisible_ious': [],
            'samples_without_ann_id': 0, 'samples_in_lookup': 0, 'samples_not_in_lookup': 0
        }
        
        per_view_ious_raw = metrics['refer_iou_per_view'].cpu().numpy()
        per_sample_global_ious = metrics['refer_iou_score_global_per_sample'].cpu().numpy()
        batched_text = batch[1]
        scene_ids = batched_text['scene_id']
        object_ids = batched_text['object_id']
        ann_ids = batched_text.get('ann_id', [None] * len(scene_ids))

        for b in range(len(scene_ids)):
            sample_view_ious = per_view_ious_raw[b]
            sample_global_iou = per_sample_global_ious[b]
            sample_avg_view_iou = np.mean(sample_view_ious) if len(sample_view_ious) > 0 else 0.0

            # Get pixel percentages for visible/invisible calculation
            pixel_percentages = []
            if self.pixel_data:
                try:
                    scene_id_val, object_id_val = scene_ids[b], object_ids[b]
                    if isinstance(scene_id_val, torch.Tensor): scene_id_val = scene_id_val.item()
                    if isinstance(object_id_val, torch.Tensor): object_id_val = object_id_val.item()
                    scene_id_str, object_id_str = str(scene_id_val), str(object_id_val)
                    frame_ids = [view_dict['instance'][b] for view_dict in batch[0]]
                    frame_pixel_data = self.pixel_data[scene_id_str][object_id_str]
                    pixel_percentages = [frame_pixel_data.get(fid, 0.0) for fid in frame_ids]
                except (KeyError, IndexError):
                    pixel_percentages = []

            # Calculate visible/invisible IoUs
            pixel_percentages_np = np.array(pixel_percentages)
            visible_views_ious = sample_view_ious[pixel_percentages_np > 0] if len(pixel_percentages) > 0 else np.array([])
            invisible_views_ious = sample_view_ious[pixel_percentages_np == 0] if len(pixel_percentages) > 0 else np.array([])
            sample_avg_visible_iou = np.mean(visible_views_ious) if len(visible_views_ious) > 0 else 0.0
            sample_avg_invisible_iou = np.mean(invisible_views_ious) if len(invisible_views_ious) > 0 else 0.0

            # Always populate 'all' stats
            stats['all_global_ious'].append(sample_global_iou)
            stats['all_sample_avg_view_ious'].append(sample_avg_view_iou)
            stats['all_sample_avg_visible_ious'].append(sample_avg_visible_iou)
            stats['all_sample_avg_invisible_ious'].append(sample_avg_invisible_iou)

            # Conditionally categorize into unique/multiple
            if self.lookup:
                scene_id, object_id, ann_id = scene_ids[b], object_ids[b], ann_ids[b]
                found_in_lookup = False
                if ann_id is not None:
                    ann_id_str = str(ann_id)
                    if scene_id in self.lookup and \
                        object_id in self.lookup[scene_id] and \
                        ann_id_str in self.lookup[scene_id][object_id]:
                        
                        uniqueness_flag = self.lookup[scene_id][object_id][ann_id_str]
                        
                        if uniqueness_flag == 0: # Unique sample
                            stats['unique_global_ious'].append(sample_global_iou)
                            stats['unique_sample_avg_view_ious'].append(sample_avg_view_iou)
                            stats['unique_sample_avg_visible_ious'].append(sample_avg_visible_iou)
                            stats['unique_sample_avg_invisible_ious'].append(sample_avg_invisible_iou)
                        else: # Multiple sample
                            stats['multiple_global_ious'].append(sample_global_iou)
                            stats['multiple_sample_avg_view_ious'].append(sample_avg_view_iou)
                            stats['multiple_sample_avg_visible_ious'].append(sample_avg_visible_iou)
                            stats['multiple_sample_avg_invisible_ious'].append(sample_avg_invisible_iou)
                        
                        found_in_lookup = True
                else:
                    stats['samples_without_ann_id'] += 1

                if found_in_lookup:
                    stats['samples_in_lookup'] += 1
                elif ann_id is not None:
                    stats['samples_not_in_lookup'] += 1
        
        return stats
    
    def _collect_nr3d_stats(self, metrics, batch):
        stats = {
            'nr3d_hard_ious': [], 'nr3d_hard_sample_avg_view_ious': [], 'nr3d_hard_sample_avg_visible_ious': [], 'nr3d_hard_sample_avg_invisible_ious': [],
            'nr3d_easy_ious': [], 'nr3d_easy_sample_avg_view_ious': [], 'nr3d_easy_sample_avg_visible_ious': [], 'nr3d_easy_sample_avg_invisible_ious': [],
            'nr3d_vd_ious': [], 'nr3d_vd_sample_avg_view_ious': [], 'nr3d_vd_sample_avg_visible_ious': [], 'nr3d_vd_sample_avg_invisible_ious': [],
            'nr3d_vi_ious': [], 'nr3d_vi_sample_avg_view_ious': [], 'nr3d_vi_sample_avg_visible_ious': [], 'nr3d_vi_sample_avg_invisible_ious': []
        }
        
        batched_text = batch[1]
        meta_datas = batched_text.get('meta_data')
        view_dependents = batched_text.get('view_dependent')
        scene_ids = batched_text['scene_id']
        object_ids = batched_text['object_id']
        
        if meta_datas is not None and 'refer_iou_score_global_per_sample' in metrics and 'refer_iou_per_view' in metrics:
            sample_global_ious = metrics['refer_iou_score_global_per_sample'].cpu().numpy()
            per_view_ious_raw = metrics['refer_iou_per_view'].cpu().numpy()
            
            for b in range(len(sample_global_ious)):
                global_iou = sample_global_ious[b]
                sample_view_ious = per_view_ious_raw[b]
                sample_avg_view_iou = np.mean(sample_view_ious) if len(sample_view_ious) > 0 else 0.0
                
                # Get pixel percentages for visible/invisible calculation
                pixel_percentages = []
                if self.pixel_data:
                    try:
                        scene_id_val = scene_ids[b]
                        object_id_val = object_ids[b]
                        if isinstance(scene_id_val, torch.Tensor): scene_id_val = scene_id_val.item()
                        if isinstance(object_id_val, torch.Tensor): object_id_val = object_id_val.item()
                        scene_id_str, object_id_str = str(scene_id_val), str(object_id_val)
                        frame_ids = [view_dict['instance'][b] for view_dict in batch[0]]
                        frame_pixel_data = self.pixel_data[scene_id_str][object_id_str]
                        pixel_percentages = [frame_pixel_data.get(fid, 0.0) for fid in frame_ids]
                    except (KeyError, IndexError):
                        pixel_percentages = []
                
                # Calculate visible/invisible IoUs
                pixel_percentages_np = np.array(pixel_percentages)
                visible_views_ious = sample_view_ious[pixel_percentages_np > 0] if len(pixel_percentages) > 0 else np.array([])
                invisible_views_ious = sample_view_ious[pixel_percentages_np == 0] if len(pixel_percentages) > 0 else np.array([])
                sample_avg_visible_iou = np.mean(visible_views_ious) if len(visible_views_ious) > 0 else 0.0
                sample_avg_invisible_iou = np.mean(invisible_views_ious) if len(invisible_views_ious) > 0 else 0.0
                
                # 1. Hard/Easy based on distractor count in meta_data
                try:
                    if b < len(meta_datas):
                        meta_str = meta_datas[b]
                        if isinstance(meta_str, str):
                            parts = meta_str.split('-')
                            if len(parts) > 2:
                                hardness = int(parts[2])
                                if hardness > 2:
                                    stats['nr3d_hard_ious'].append(global_iou)
                                    stats['nr3d_hard_sample_avg_view_ious'].append(sample_avg_view_iou)
                                    stats['nr3d_hard_sample_avg_visible_ious'].append(sample_avg_visible_iou)
                                    stats['nr3d_hard_sample_avg_invisible_ious'].append(sample_avg_invisible_iou)
                                else:
                                    stats['nr3d_easy_ious'].append(global_iou)
                                    stats['nr3d_easy_sample_avg_view_ious'].append(sample_avg_view_iou)
                                    stats['nr3d_easy_sample_avg_visible_ious'].append(sample_avg_visible_iou)
                                    stats['nr3d_easy_sample_avg_invisible_ious'].append(sample_avg_invisible_iou)
                except (ValueError, IndexError):
                    pass

                # 2. View Dependent / Independent
                try:
                    if view_dependents is not None and b < len(view_dependents):
                        is_view_dep = view_dependents[b]
                        if isinstance(is_view_dep, torch.Tensor):
                            is_view_dep = is_view_dep.item()
                        
                        if is_view_dep:
                            stats['nr3d_vd_ious'].append(global_iou)
                            stats['nr3d_vd_sample_avg_view_ious'].append(sample_avg_view_iou)
                            stats['nr3d_vd_sample_avg_visible_ious'].append(sample_avg_visible_iou)
                            stats['nr3d_vd_sample_avg_invisible_ious'].append(sample_avg_invisible_iou)
                        else:
                            stats['nr3d_vi_ious'].append(global_iou)
                            stats['nr3d_vi_sample_avg_view_ious'].append(sample_avg_view_iou)
                            stats['nr3d_vi_sample_avg_visible_ious'].append(sample_avg_visible_iou)
                            stats['nr3d_vi_sample_avg_invisible_ious'].append(sample_avg_invisible_iou)
                except (ValueError, IndexError):
                    pass
                    
        return stats

    def _calculate_metrics_for_category(self, global_data, all_view_data, visible_view_data, invisible_view_data):
        """Helper function to calculate metrics for a given category."""
        metrics = {}
        
        def calc_metrics(data):
            if data.size == 0: return 0.0, 0.0, 0.0
            return (data >= 0.25).mean(), (data >= 0.50).mean(), data.mean()

        metrics['acc25_global'], metrics['acc50_global'], metrics['iou_global_mean'] = calc_metrics(global_data)
        metrics['acc25_per_view_all'], metrics['acc50_per_view_all'], metrics['iou_per_view_all_mean'] = calc_metrics(all_view_data)
        metrics['acc25_per_view_visible'], metrics['acc50_per_view_visible'], metrics['iou_per_view_visible_mean'] = calc_metrics(visible_view_data)
        metrics['acc25_per_view_invisible'], metrics['acc50_per_view_invisible'], metrics['iou_per_view_invisible_mean'] = calc_metrics(invisible_view_data)
        
        return metrics
    
    def train_one_epoch(self, epoch):
        self.model.train()
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter(
            "min_lr", SmoothedValue(window_size=1, fmt="{value:.6f}")
        )
        # metric_logger.add_meter(
        #     "dataloader", SmoothedValue(window_size=1, fmt="{value:.6f}")
        # )
        header = "Epoch: [{}]".format(epoch)
        loss_details_dict = {}
        start_steps = epoch * self.iters_per_epoch
        self.global_step = start_steps

        self.log_info(
            "Start training epoch {}, {} iters per inner epoch. Training dtype {}".format(
                epoch, self.iters_per_epoch, self.cfg.train.model_dtype
            )
        )

        for it, batch in enumerate(metric_logger.log_every(
            self.train_loader, self.cfg.train.print_freq, header
        )):
            
            if it >= self.iters_per_epoch:
                break

            for _ in range(self.cfg.train.gradient_accumulation_steps):
                with self.accelerator.accumulate(self.model):
                    # Perform the forward using the accerlate
                    batch = move_to_device(batch, device=self.accelerator.device)
                    with self.accelerator.autocast():
                        forward_output = self.forward_batch(batch, mode='train')
                    batch_output = self.calculate_loss(
                        forward_output, batch, mode='train', 
                        current_epoch=epoch, total_epochs=self.cfg.train.num_epoch
                    )
                    loss = batch_output.loss
                    if loss > self.cfg.train.clip_loss:
                        loss = loss * 0.0

                    # Check if the loss is nan
                    loss_value = loss.item()
                    if not math.isfinite(loss_value):
                        rank = get_rank()
                        print(
                            f"Rank {rank}: Loss is {loss_value}, stopping training at iter {it} (epoch {epoch}, global step {self.global_step}).",
                            force=True,
                        )
                        sys.exit(1)

                    self.accelerator.backward(loss)

                    for item in batch_output:
                        if 'loss' in item:
                            batch_output[item] = self.accelerator.gather(batch_output[item]).mean().item()
                            if item in loss_details_dict:
                                loss_details_dict[item] += batch_output[item] / self.cfg.train.gradient_accumulation_steps if loss_value != 0 else 0.0
                            else:
                                loss_details_dict[item] = batch_output[item] / self.cfg.train.gradient_accumulation_steps if loss_value != 0 else 0.0

                    # clip the gradient
                    if self.accelerator.sync_gradients:
                        params_to_clip = self.model.parameters()
                        self.accelerator.clip_grad_norm_(
                            params_to_clip, self.cfg.train.clip_grad
                        )

                        def get_gradient_norm(parameters):
                            norm = 0
                            for param in parameters:
                                if param.grad is None:
                                    continue
                                local_norm = param.grad.detach().data.norm(2)
                                norm += local_norm.item() ** 2
                            norm = norm**0.5
                            return norm

                        grad_norm = get_gradient_norm(self.model.parameters())

                    if self.accelerator.state.deepspeed_plugin is None:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    self.lr_scheduler.step()

                if self.accelerator.sync_gradients:
                    start_steps += 1

                    # Report to tensorboard
                    batch_output.update(loss_details_dict)
                    loss_details_dict = {}

                    if start_steps % 10 == 0 :
                        self.log_all(batch_output, start_steps, prefix='train')
                    
                    # Create a dictionary for the metric logger, excluding per-sample metrics.
                    # Per-sample metrics (like 'refer_iou_score_per_sample') are tensors containing a value for each item in the batch.
                    # The metric logger expects a single scalar value for averaging, so passing these tensors would cause a crash.
                    logger_output = {k: v for k, v in batch_output.items() if 'per_sample' not in k and 'per_view' not in k}

                    metric_logger.update(**logger_output)

                    min_lr = 10.0
                    max_lr = 0.0
                    for group in self.optimizer.param_groups:
                        min_lr = min(min_lr, group["lr"])
                        max_lr = max(max_lr, group["lr"])

                    metric_logger.update(lr=max_lr)
                    metric_logger.update(min_lr=min_lr)
                    self.accelerator.log({"lr": max_lr}, step=start_steps)
                    self.accelerator.log({"min_lr": min_lr}, step=start_steps)

                    weight_decay_value = None
                    for group in self.optimizer.param_groups:
                        if group["weight_decay"] > 0:
                            weight_decay_value = group["weight_decay"]
                    metric_logger.update(weight_decay=weight_decay_value)
                    metric_logger.update(grad_norm=grad_norm)
                    self.accelerator.log(
                        {"weight_decay": weight_decay_value}, step=start_steps
                    )
                    self.accelerator.log({"grad_norm": grad_norm}, step=start_steps)

                    self.global_step = start_steps

        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


    def log_all(self, output, step, prefix=""):        
        if 'log_keys' in output:
            log_keys = output.log_keys
        else:
            log_keys = list(output.keys())

        log_scaler = {}
        log_img = {}
        for k in log_keys:
            v = output[k]
            if np.isscalar(v):
                log_scaler[prefix+'/'+k] = v
                continue
            if Image.isImageType(v):
                log_img[prefix+'/'+k] = v

        self.accelerator.log(log_scaler, step)
        for tracker in self.accelerator.trackers:
            tracker.log_images(log_img, step)

    def forward_batch(self, batch, mode='train'):
        output = self.model(batch)
        assert isinstance(output, EasyDict)
        return output

    def calculate_loss(self, output, batch, mode='train'):
        pass

    def build_accelerator(self):
        accelerator_project_config = ProjectConfiguration(
            project_dir=self.cfg.log.output_dir,
            logging_dir=self.cfg.log.output_dir,
            total_limit=4,      # self.cfg.save_total_limit = 4
            # automatic_checkpoint_naming=True,
        )

        # Initialize the Environment variables throught MPI run
        init_distributed_mode(
            self.cfg.train, init_pytorch_ddp=False
        )  # set `init_pytorch_ddp` to False, since the accelerate will do later

        if self.cfg.log.use_wandb:
            log_with = 'wandb'
        elif self.cfg.log.use_tensorboard:
            log_with = 'tensorboard'
        else:
            log_with = 'all'

        mixed_precision = 'no' if self.cfg.train.model_dtype not in ['fp8', 'fp16', 'bf16'] else self.cfg.train.model_dtype

        # For mixed precision training we cast all non-trainable weights to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        self.weight_dtype = torch.float32
        if mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        # dynamic complie
        if self.cfg.train.get("dynamo_backend"):
            if isinstance(self.cfg.train.dynamo_backend, str) and hasattr(
                DynamoBackend, self.cfg.train.dynamo_backend.upper()
            ):
                dynamo_backend = getattr(DynamoBackend, self.cfg.train.dynamo_backend.upper())
            elif isinstance(self.cfg.train.dynamo_backend, DynamoBackend):
                dynamo_backend = self.cfg.train.dynamo_backend
            else:
                print(
                    f"Invalid dynamo_backend {self.cfg.train.dynamo_backend}, using default. Please refer to "
                    "https://huggingface.co/docs/accelerate/v1.2.1/en/package_reference/utilities#accelerate.utils.DynamoBackend for available names."
                )
        else:
            dynamo_backend = DynamoBackend.NO

        print(f"Using dynamo backend: {dynamo_backend}")

        torch._inductor.config.reorder_for_compute_comm_overlap = True

        dynamo_plugin = TorchDynamoPlugin(
            backend=dynamo_backend,
            mode="max-autotune-no-cudagraphs",
            dynamic=self.cfg.train.get("dynamic_compile", True),
        )
        
        accelerate_config = dict(
            gradient_accumulation_steps=self.cfg.train.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with=log_with,
            project_config=accelerator_project_config,
            dataloader_config=DataLoaderConfiguration(
                non_blocking=True,
                split_batches=False,
                dispatch_batches=None,
                even_batches=True,
                use_seedable_sampler=False,
            ),
            step_scheduler_with_optimizer=False,             # not to step n_gpus times per step.
            dynamo_plugin=dynamo_plugin,
        )

        # fsdp
        if self.cfg.get("fsdp_plugin"):
            fsdp_plugin_kwargs = {}
            fsdp_plugin_kwargs[
                "mixed_precision_policy"
            ] = torch.distributed.fsdp.MixedPrecision(
                param_dtype=self.weight_dtype,
                reduce_dtype=self.weight_dtype,
                buffer_dtype=self.weight_dtype,
                cast_forward_inputs=True,
                cast_root_forward_inputs=True,
            )

            fsdp_plugin = hydra.utils.instantiate(self.cfg.fsdp_plugin)(**fsdp_plugin_kwargs)
            accelerate_config["fsdp_plugin"] = fsdp_plugin
        else:
            ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=self.cfg.train.find_unused_parameters)
            accelerate_config['kwargs_handlers'] = [ddp_kwargs]

        accelerator = Accelerator(**accelerate_config)

        self.logger = get_logger(self.cfg, os.path.basename(__file__))

        # To block the print on non main process
        setup_for_distributed(accelerator.is_main_process)

        # self.logger.rank_zero_only = False
        self.log_info(accelerator.state)
        # self.logger.rank_zero_only = True
        
        if self.cfg.random_seed is not None:
            set_seed(self.cfg.random_seed, device_specific=True)

        self.device = accelerator.device

        self.accelerator = accelerator

    def auto_resume(self):
        path = None
        if self.cfg.train.resume:
            path = self.cfg.train.resume
        elif os.path.exists(self.cfg.log.ckpt_dir):
            # Get the most recent checkpoint
            dirs = os.listdir(self.cfg.log.ckpt_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint_")]
            if dirs:
                dirs = sorted(dirs, key=lambda x: int(x.split("_")[1]))
                path = dirs[-1] if len(dirs) > 0 else None
                if path is not None:
                    path = os.path.join(self.cfg.log.ckpt_dir, path)

        if path is None or not os.path.exists(path):
            self.log_info("Checkpoint does not exist. Starting a new training run.")
            start_epoch = 0
        else:
            self.log_info(f"Resuming from checkpoint {path}")
            self.accelerator.load_state(path)
            
            # Try to parse epoch from path, otherwise default to 0 for fine-tuning or evaluation.
            try:
                basename = os.path.basename(path)
                if "checkpoint_" in basename:
                    # A checkpoint is saved AFTER an epoch is completed.
                    # So if we load checkpoint_9, we should start the next epoch, which is 10.
                    last_completed_epoch = int(basename.split("checkpoint_")[-1])
                    start_epoch = last_completed_epoch + 1
                    self.log_info(f"Resuming training from epoch {start_epoch}")
                else:
                    # For paths like 'best_model', we are not resuming a specific epoch.
                    # In eval_only mode this is fine. For training, it will start from epoch 0.
                    self.log_info("Could not parse epoch from checkpoint name. Starting from epoch 0.")
                    start_epoch = 0
            except (ValueError, IndexError):
                self.log_info(f"Could not parse epoch from {path}. Assuming starting at epoch 0.")
                start_epoch = 0
                
        return start_epoch

    def log_info(self, info):
        if is_logging_process():
            self.logger.info(info)
