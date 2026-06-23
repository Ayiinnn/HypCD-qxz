#!/usr/bin/env bash
set -euo pipefail

# ===== conda env =====
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
  source /root/miniconda3/etc/profile.d/conda.sh
elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  source /opt/conda/etc/profile.d/conda.sh
else
  echo "Cannot find conda. Run: which conda"
  exit 1
fi

conda activate hypcd

cd /data/projects/HypCD

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

LOG_DIR=run_logs/obj_ab_det_cub_3rounds
mkdir -p "${LOG_DIR}"

COMMON=(
  python train/train_HypSimGCD_org_det_ab_obj.py
  --dataset_name cub
  --model_name v1
  --batch_size 128
  --num_workers 8
  --epochs 200
  --grad_from_block 11
  --lr 0.1
  --gamma 0.1
  --momentum 0.9
  --weight_decay 5e-5
  --transform imagenet
  --sup_weight 0.35
  --n_views 2
  --memax_weight 1.0
  --warmup_teacher_temp 0.07
  --teacher_temp 0.04
  --warmup_teacher_temp_epochs 30
  --c 0.1
  --cr 2.3
  --hyper_max_weight 1.0
  --hyper_temp_scale 0.4
  --obj_fg_source attention
  --obj_fg_keep 0.6
  --obj_fg_pad 0.1
  --obj_aperture_scale 1.2
  --obj_min_radius 0.1
  --obj_dist_temp 0.1
  --eval_funcs v2 v2b
  --deterministic
  --seed 0
)

run_job () {
  local gpu="$1"
  local name="$2"
  shift 2
  echo "[$(date '+%F %T')] launch GPU${gpu}: ${name}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${COMMON[@]}" \
    --exp_name "${name}" \
    "$@" \
    > "${LOG_DIR}/${name}.log" 2>&1 &
}

wait_round () {
  local round_name="$1"
  shift
  wait "$@"
  echo "[$(date '+%F %T')] ${round_name} finished."
}

# =========================
# Round 1: CUB basic ablation
# =========================

run_job 0 objab-r1_cub_noobj \
  --no_object_branch
p0=$!

run_job 1 objab-r1_cub_ent005 \
  --use_object_branch \
  --obj_entail_weight 0.05 \
  --obj_dist_weight 0.0 \
  --obj_cls_weight 0.0 \
  --obj_entail_parent image
p1=$!

run_job 2 objab-r1_cub_dist005 \
  --use_object_branch \
  --obj_entail_weight 0.0 \
  --obj_dist_weight 0.05 \
  --obj_cls_weight 0.0 \
  --obj_entail_parent image
p2=$!

run_job 3 objab-r1_cub_cls005 \
  --use_object_branch \
  --obj_entail_weight 0.0 \
  --obj_dist_weight 0.0 \
  --obj_cls_weight 0.05 \
  --obj_entail_parent image
p3=$!

wait_round "Round 1" "$p0" "$p1" "$p2" "$p3"


# =========================
# Round 2: combo tuning
# =========================

run_job 0 objab-r2_cub_combo_e002_d005_c005 \
  --use_object_branch \
  --obj_entail_weight 0.02 \
  --obj_dist_weight 0.05 \
  --obj_cls_weight 0.05 \
  --obj_entail_parent image
p0=$!

run_job 1 objab-r2_cub_combo_e001_d005_c005 \
  --use_object_branch \
  --obj_entail_weight 0.01 \
  --obj_dist_weight 0.05 \
  --obj_cls_weight 0.05 \
  --obj_entail_parent image
p1=$!

run_job 2 objab-r2_cub_combo_e005_d005_c005 \
  --use_object_branch \
  --obj_entail_weight 0.05 \
  --obj_dist_weight 0.05 \
  --obj_cls_weight 0.05 \
  --obj_entail_parent image
p2=$!

run_job 3 objab-r2_cub_combo_e002_d005_c000 \
  --use_object_branch \
  --obj_entail_weight 0.02 \
  --obj_dist_weight 0.05 \
  --obj_cls_weight 0.0 \
  --obj_entail_parent image
p3=$!

wait_round "Round 2" "$p0" "$p1" "$p2" "$p3"


# =========================
# Round 3: combo tuning + child/parent swap
# =========================

run_job 0 objab-r3_cub_combo_e002_d000_c005 \
  --use_object_branch \
  --obj_entail_weight 0.02 \
  --obj_dist_weight 0.0 \
  --obj_cls_weight 0.05 \
  --obj_entail_parent image
p0=$!

run_job 1 objab-r3_cub_combo_e002_d005_cneg002 \
  --use_object_branch \
  --obj_entail_weight 0.02 \
  --obj_dist_weight 0.05 \
  --obj_cls_weight -0.02 \
  --obj_entail_parent image
p1=$!

run_job 2 objab-r3_cub_combo_e001_d000_c005 \
  --use_object_branch \
  --obj_entail_weight 0.01 \
  --obj_dist_weight 0.0 \
  --obj_cls_weight 0.05 \
  --obj_entail_parent image
p2=$!

run_job 3 objab-r3_cub_combo_e002_d005_c005_parent_object \
  --use_object_branch \
  --obj_entail_weight 0.02 \
  --obj_dist_weight 0.05 \
  --obj_cls_weight 0.05 \
  --obj_entail_parent object
p3=$!

wait_round "Round 3" "$p0" "$p1" "$p2" "$p3"

echo "All 3 rounds finished. Logs: ${LOG_DIR}"
