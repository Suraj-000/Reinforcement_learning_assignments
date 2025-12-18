#!/bin/bash

start_time=$(date +%s)

echo "=== Running InvertedPendulum A2C Train ==="
python3 scripts/train_sb.py --env_name InvertedPendulum-v4 --algo A2C --timesteps 1000000

echo "=== Running InvertedPendulum A2C Test ==="
python3 scripts/train_sb.py --env_name InvertedPendulum-v4 --algo A2C --test --load_checkpoint logs/InvertedPendulum-v4/PPO/seed0/best_model.zip

echo "=== Running Hopper A2C Train ==="
python3 scripts/train_sb.py --env_name Hopper-v4 --algo A2C --timesteps 1000000

echo "=== Running Hopper A2C Test ==="
python3 scripts/train_sb.py --env_name Hopper-v4 --algo A2C --test --load_checkpoint logs/Hopper-v4/PPO/seed0/best_model.zip


echo "=== Running HalfCheetah A2C Train ==="
python3 scripts/train_sb.py --env_name HalfCheetah-v4 --algo A2C --timesteps 1500000

echo "=== Running HalfCheetah A2C Test ==="
python3 scripts/train_sb.py --env_name HalfCheetah-v4 --algo A2C --test --load_checkpoint logs/HalfCheetah-v4/PPO/seed0/best_model.zip


end_time=$(date +%s)
total_time=$((end_time - start_time))

echo "=== ALL RUNS COMPLETED ==="
echo "Total Execution Time: $((total_time / 3600))h $(((total_time / 60) % 60))m $((total_time % 60))s"