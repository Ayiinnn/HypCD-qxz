#!/usr/bin/env bash
set -euo pipefail

cd /data/projects/HypCD
mkdir -p logs

# 防止服务器上的代码并非支持 deterministic 的最新版本
if ! grep -q "add_argument('--deterministic'" \
  train/train_HypCD_mixedc_det.py; then
    echo "[ERROR] 当前 train/train_HypCD_mixedc_det.py 不支持 --deterministic"
    echo "[ERROR] 请先把最新版本同步到 /data/projects/HypCD"
    exit 1
fi

echo "[INFO] Waiting for GPUs 0,1,2,3 to become idle..."

IDLE_COUNT=0

while true; do
    BUSY_PIDS="$(
        for GPU in 0 1 2 3; do
            nvidia-smi -i "${GPU}" \
              --query-compute-apps=pid \
              --format=csv,noheader,nounits 2>/dev/null || true
        done | awk 'NF'
    )"

    if [ -z "${BUSY_PIDS}" ]; then
        IDLE_COUNT=$((IDLE_COUNT + 1))
        echo "[INFO] $(date '+%F %T') GPUs idle check ${IDLE_COUNT}/2"

        # 连续两次检测为空，避免进程切换瞬间误启动
        if [ "${IDLE_COUNT}" -ge 2 ]; then
            break
        fi
    else
        IDLE_COUNT=0
        echo "[INFO] $(date '+%F %T') Current jobs still running:"
        nvidia-smi \
          --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
          --format=csv,noheader || true
    fi

    sleep 60
done

TAG="mixedc_air2_scars2_det_$(date +%m%d_%H%M%S)"
echo "[INFO] All four GPUs are idle."
echo "[INFO] Launching four experiments simultaneously."
echo "[INFO] TAG=${TAG}"


# ============================================================
# GPU0: Aircraft, 50/50 E/P split, uniform fixed role weights
#
# role_init_strength=0:
#   cls weights = softmax([0,0]) = [0.5,0.5]
#   rep weights = softmax([0,0]) = [0.5,0.5]
#
# 不加 --learn_role_weights
# ============================================================
CUDA_VISIBLE_DEVICES=0 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
nohup conda run --no-capture-output -n hypcd \
python -m train.train_HypCD_mixedc_det \
  --dataset_name aircraft \
  --model_name v1 \
  --batch_size 128 \
  --grad_from_block 11 \
  --epochs 200 \
  --num_workers 8 \
  --use_ssb_splits \
  --sup_weight 0.35 \
  --weight_decay 5e-5 \
  --transform imagenet \
  --lr 0.1 \
  --warmup_teacher_temp 0.07 \
  --teacher_temp 0.04 \
  --warmup_teacher_temp_epochs 30 \
  --memax_weight 1.0 \
  --c 0.1 \
  --cr 2.3 \
  --train_c \
  --hyper_max_weight 1.0 \
  --hyper_temp_scale 0.4 \
  --hyper_lr 0.01 \
  --logit_temp 0.1 \
  --factors "E:384,P:384:c0.1:cr2.3" \
  --role_init_strength 0.0 \
  --align_weight 0.0 \
  --radius_weight 0.0 \
  --degeneracy_weight 0.0 \
  --eval_role cls \
  --seed 0 \
  --deterministic \
  --exp_name aircraft_mixedc_E384_P384_uniform_det \
  > "logs/${TAG}_gpu0_air_E384_P384_uniform.log" 2>&1 &

PID0=$!


# ============================================================
# GPU1: Aircraft, 50/50 E/P split + learnable role weights
#
# 初始：
#   cls 偏向欧式因子
#   rep 偏向双曲因子
# 随训练继续学习 cls/rep 的因子权重
# ============================================================
CUDA_VISIBLE_DEVICES=1 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
nohup conda run --no-capture-output -n hypcd \
python -m train.train_HypCD_mixedc_det \
  --dataset_name aircraft \
  --model_name v1 \
  --batch_size 128 \
  --grad_from_block 11 \
  --epochs 200 \
  --num_workers 8 \
  --use_ssb_splits \
  --sup_weight 0.35 \
  --weight_decay 5e-5 \
  --transform imagenet \
  --lr 0.1 \
  --warmup_teacher_temp 0.07 \
  --teacher_temp 0.04 \
  --warmup_teacher_temp_epochs 30 \
  --memax_weight 1.0 \
  --c 0.1 \
  --cr 2.3 \
  --train_c \
  --hyper_max_weight 1.0 \
  --hyper_temp_scale 0.4 \
  --hyper_lr 0.01 \
  --logit_temp 0.1 \
  --factors "E:384,P:384:c0.1:cr2.3" \
  --role_init_strength 0.7 \
  --learn_role_weights \
  --align_weight 0.0 \
  --radius_weight 0.0 \
  --degeneracy_weight 0.0 \
  --eval_role cls \
  --seed 0 \
  --deterministic \
  --exp_name aircraft_mixedc_E384_P384_learnrole_det \
  > "logs/${TAG}_gpu1_air_E384_P384_learnrole.log" 2>&1 &

PID1=$!


# ============================================================
# GPU2: SCars, full-hyperbolic P768, no factor split baseline
# ============================================================
CUDA_VISIBLE_DEVICES=2 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
nohup conda run --no-capture-output -n hypcd \
python -m train.train_HypCD_mixedc_det \
  --dataset_name scars \
  --model_name v1 \
  --batch_size 128 \
  --grad_from_block 11 \
  --epochs 200 \
  --num_workers 8 \
  --use_ssb_splits \
  --sup_weight 0.35 \
  --weight_decay 5e-5 \
  --transform imagenet \
  --lr 0.1 \
  --warmup_teacher_temp 0.07 \
  --teacher_temp 0.04 \
  --warmup_teacher_temp_epochs 30 \
  --memax_weight 1.0 \
  --c 0.1 \
  --cr 2.3 \
  --train_c \
  --hyper_max_weight 1.0 \
  --hyper_temp_scale 0.4 \
  --hyper_lr 0.01 \
  --logit_temp 0.1 \
  --factors "P:768:c0.1:cr2.3" \
  --role_init_strength 0.0 \
  --align_weight 0.0 \
  --radius_weight 0.0 \
  --degeneracy_weight 0.0 \
  --eval_role cls \
  --seed 0 \
  --deterministic \
  --exp_name scars_mixedc_P768_baseline_det \
  > "logs/${TAG}_gpu2_scars_P768_baseline.log" 2>&1 &

PID2=$!


# ============================================================
# GPU3: SCars, E128 + P640 split
#
# 固定均匀角色权重，隔离“切分比例”本身的影响
# ============================================================
CUDA_VISIBLE_DEVICES=3 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
nohup conda run --no-capture-output -n hypcd \
python -m train.train_HypCD_mixedc_det \
  --dataset_name scars \
  --model_name v1 \
  --batch_size 128 \
  --grad_from_block 11 \
  --epochs 200 \
  --num_workers 8 \
  --use_ssb_splits \
  --sup_weight 0.35 \
  --weight_decay 5e-5 \
  --transform imagenet \
  --lr 0.1 \
  --warmup_teacher_temp 0.07 \
  --teacher_temp 0.04 \
  --warmup_teacher_temp_epochs 30 \
  --memax_weight 1.0 \
  --c 0.1 \
  --cr 2.3 \
  --train_c \
  --hyper_max_weight 1.0 \
  --hyper_temp_scale 0.4 \
  --hyper_lr 0.01 \
  --logit_temp 0.1 \
  --factors "E:128,P:640:c0.1:cr2.3" \
  --role_init_strength 0.0 \
  --align_weight 0.0 \
  --radius_weight 0.0 \
  --degeneracy_weight 0.0 \
  --eval_role cls \
  --seed 0 \
  --deterministic \
  --exp_name scars_mixedc_E128_P640_uniform_det \
  > "logs/${TAG}_gpu3_scars_E128_P640_uniform.log" 2>&1 &

PID3=$!

echo "[INFO] Four jobs launched:"
echo "GPU0 PID=${PID0}"
echo "GPU1 PID=${PID1}"
echo "GPU2 PID=${PID2}"
echo "GPU3 PID=${PID3}"
echo "[INFO] Logs:"
echo "logs/${TAG}_gpu0_air_E384_P384_uniform.log"
echo "logs/${TAG}_gpu1_air_E384_P384_learnrole.log"
echo "logs/${TAG}_gpu2_scars_P768_baseline.log"
echo "logs/${TAG}_gpu3_scars_E128_P640_uniform.log"
