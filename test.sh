srun -p AI4Good_X --gres=gpu:1 --cpus-per-task=32 --ntasks-per-node=1 --kill-on-bad-exit \
python test.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
