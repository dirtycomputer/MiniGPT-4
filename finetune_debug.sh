srun -p AI4Good_X --gres=gpu:1 --cpus-per-task=32 --ntasks-per-node=1 --kill-on-bad-exit \
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
