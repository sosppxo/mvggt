import os
import torch
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
print(f"[Rank {local_rank}] Using device: {torch.cuda.current_device()}")
import sys
sys.path.append('.')

import hydra
import trainers


@hydra.main(version_base="1.2", config_path="../configs", config_name="default")
def main(hydra_cfg):
    trainer = eval(hydra_cfg.trainer)(hydra_cfg)
    trainer.train()

if __name__ == '__main__':
    main()