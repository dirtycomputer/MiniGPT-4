srun -p AI4Good_X --gres=gpu:8 --cpus-per-task=128 --ntasks-per-node=1 --kill-on-bad-exit \
torchrun --nproc-per-node 8 train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
