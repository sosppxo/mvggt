HYDRA_FULL_ERROR=1
accelerate launch --config_file configs/accelerate/ddp.yaml \
    --num_processes 1 --num_machines 1 \
    scripts/train_mvggt.py train=train_mvggt_refer_lowres name=mvggt_refer_low_res\
    train.eval_only=True \
    train.resume=your_path/ckpts/best