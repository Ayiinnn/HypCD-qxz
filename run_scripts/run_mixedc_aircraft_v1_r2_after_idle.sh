#!/usr/bin/env bash
set -e

cd /data/projects/HypCD
mkdir -p logs
TAG=mixedc_r2_aircraft_v1_$(date +%m%d_%H%M)

echo "[INFO] Waiting until GPUs 0-3 are idle..."
while true; do
  BUSY=$(for g in 0 1 2 3; do
    nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null
  done | awk 'NF' || true)

  if [ -z "$BUSY" ]; then
    echo "[INFO] GPUs idle. Launching new mixedc round."
    break
  fi

  date
  nvidia-smi --query-compute-apps=gpu_name,pid,process_name,used_memory --format=csv || true
  sleep 300
done

# GPU0: clean 1E+1P，半径目标贴近初始有效半径，最稳基线
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 nohup conda run --no-capture-output -n hypcd \
python -m train.train_HypCD_mixedc_det \
  --dataset_name aircraft \
  --batch_size 128 --grad_from_block 11 --epochs 200 --num_workers 8 \
  --use_ssb_splits --sup_weight 0.35 --weight_decay 5e-5 \
  --transform imagenet --lr 0.1 \
  --warmup_teacher_temp 0.07 --teacher_temp 0.04 --warmup_teacher_temp_epochs 30 \
  --memax_weight 1.0 --model_name v1 \
  --c 0.1 --cr 2.3 --train_c \
  --hyper_max_weight 1.0 --hyper_temp_scale 0.4 \
  --hyper_lr 0.01 --logit_temp 0.1 \
  --factors "E:384,P:384:c0.1:cr2.3" \
  --align_weight 1.0 --radius_weight 0.05 --target_radius 0.75 \
  --degeneracy_weight 0.0 \
  --eval_role cls --seed 0 \
  --exp_name aircraft_v1_mixedc_r2_E384_P384_r075_rw005 \
  > logs/${TAG}_gpu0_E384_P384_r075_rw005.log 2>&1 &

# GPU1: clean 1E+1P，稍强半径约束，测试是否稳定推 c 到更有效区域
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 nohup conda run --no-capture-output -n hypcd \
python -m train.train_HypCD_mixedc_det \
  --dataset_name aircraft \
  --batch_size 128 --grad_from_block 11 --epochs 200 --num_workers 8 \
  --use_ssb_splits --sup_weight 0.35 --weight_decay 5e-5 \
  --transform imagenet --lr 0.1 \
  --warmup_teacher_temp 0.07 --teacher_temp 0.04 --warmup_teacher_temp_epochs 30 \
  --memax_weight 1.0 --model_name v1 \
  --c 0.1 --cr 2.3 --train_c \
  --hyper_max_weight 1.0 --hyper_temp_scale 0.4 \
  --hyper_lr 0.01 --logit_temp 0.1 \
  --factors "E:384,P:384:c0.1:cr2.3" \
  --align_weight 1.0 --radius_weight 0.1 --target_radius 0.85 \
  --degeneracy_weight 0.0 \
  --eval_role cls --seed 0 \
  --exp_name aircraft_v1_mixedc_r2_E384_P384_r085_rw01 \
  > logs/${TAG}_gpu1_E384_P384_r085_rw01.log 2>&1 &

# GPU2: clean 1E+1P，分类温度放松，排查 product head/logit 尺度问题
CUDA_VISIBLE_DEVICES=2 CUBLAS_WORKSPACE_CONFIG=:4096:8 nohup conda run --no-capture-output -n hypcd \
python -m train.train_HypCD_mixedc_det \
  --dataset_name aircraft \
  --batch_size 128 --grad_from_block 11 --epochs 200 --num_workers 8 \
  --use_ssb_splits --sup_weight 0.35 --weight_decay 5e-5 \
  --transform imagenet --lr 0.1 \
  --warmup_teacher_temp 0.07 --teacher_temp 0.04 --warmup_teacher_temp_epochs 30 \
  --memax_weight 1.0 --model_name v1 \
  --c 0.1 --cr 2.3 --train_c \
  --hyper_max_weight 1.0 --hyper_temp_scale 0.4 \
  --hyper_lr 0.01 --logit_temp 0.2 \
  --factors "E:384,P:384:c0.1:cr2.3" \
  --align_weight 1.0 --radius_weight 0.05 --target_radius 0.75 \
  --degeneracy_weight 0.0 \
  --eval_role cls --seed 0 \
  --exp_name aircraft_v1_mixedc_r2_E384_P384_r075_logtemp02 \
  > logs/${TAG}_gpu2_E384_P384_r075_logtemp02.log 2>&1 &

# GPU3: clean 1E+2P，总维度仍 768，测试第二个 P 因子是否自然分工
CUDA_VISIBLE_DEVICES=3 CUBLAS_WORKSPACE_CONFIG=:4096:8 nohup conda run --no-capture-output -n hypcd \
python -m train.train_HypCD_mixedc_det \
  --dataset_name aircraft \
  --batch_size 128 --grad_from_block 11 --epochs 200 --num_workers 8 \
  --use_ssb_splits --sup_weight 0.35 --weight_decay 5e-5 \
  --transform imagenet --lr 0.1 \
  --warmup_teacher_temp 0.07 --teacher_temp 0.04 --warmup_teacher_temp_epochs 30 \
  --memax_weight 1.0 --model_name v1 \
  --c 0.1 --cr 2.3 --train_c \
  --hyper_max_weight 1.0 --hyper_temp_scale 0.4 \
  --hyper_lr 0.01 --logit_temp 0.1 \
  --factors "E:256,P:384:c0.1:cr2.3,P:128:c0.05:cr2.3" \
  --align_weight 1.0 --radius_weight 0.05 --target_radius 0.75 \
  --degeneracy_weight 0.0 \
  --eval_role cls --seed 0 \
  --exp_name aircraft_v1_mixedc_r2_E256_P384_P128_clean \
  > logs/${TAG}_gpu3_E256_P384_P128_clean.log 2>&1 &

echo "[INFO] Launched:"
jobs -l
echo "[INFO] Logs prefix: logs/${TAG}_gpu*.log"
