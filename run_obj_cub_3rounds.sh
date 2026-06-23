#!/usr/bin/env bash
set -euo pipefail

cd /data/projects/HypCD
mkdir -p logs

export CUBLAS_WORKSPACE_CONFIG=:4096:8
ENTRY=train.train_HypSimGCD_org_det_ab_obj

BASE_ARGS="\
  --dataset_name cub \
  --batch_size 128 \
  --grad_from_block 11 \
  --epochs 200 \
  --num_workers 8 \
  --eval_funcs v2 v2b \
  --use_ssb_splits \
  --sup_weight 0.35 \
  --weight_decay 5e-5 \
  --transform imagenet \
  --lr 0.1 \
  --warmup_teacher_temp 0.07 \
  --teacher_temp 0.04 \
  --warmup_teacher_temp_epochs 30 \
  --memax_weight 1.0 \
  --model_name v1 \
  --c 0.1 \
  --cr 2.0 \
  --hyper_max_weight 1.0 \
  --hyper_temp_scale 0.3 \
  --hyper_start_epoch 0 \
  --hyper_end_epoch 200 \
  --seed 0 \
  --deterministic \
  --print_freq 10"

run_job () {
  local gpu="$1"
  local name="$2"
  shift 2

  echo "[START] GPU${gpu} ${name}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  conda run --no-capture-output -n hypcd \
  python -m ${ENTRY} ${BASE_ARGS} \
    --exp_name "${name}" \
    "$@" \
    > "logs/${name}.log" 2>&1 &
}

echo "========== ROUND 1: CUB 4/5 basic ablations =========="
run_job 0 obj-abdet-r1_cub_noobj \
  --no_object_branch

run_job 1 obj-abdet-r1_cub_entail005 \
  --use_object_branch \
  --obj_fg_source attention \
  --obj_entail_parent image \
  --obj_entail_weight 0.05 \
  --obj_dist_weight 0 \
  --obj_cls_weight 0

run_job 2 obj-abdet-r1_cub_dist005 \
  --use_object_branch \
  --obj_fg_source attention \
  --obj_entail_parent image \
  --obj_entail_weight 0 \
  --obj_dist_weight 0.05 \
  --obj_cls_weight 0

run_job 3 obj-abdet-r1_cub_cls005 \
  --use_object_branch \
  --obj_fg_source attention \
  --obj_entail_parent image \
  --obj_entail_weight 0 \
  --obj_dist_weight 0 \
  --obj_cls_weight 0.05

wait
echo "========== ROUND 1 DONE =========="


echo "========== ROUND 2: CUB combo base + zero-component tuning =========="
run_job 0 obj-abdet-r2_cub_combo_e002_d005_c005 \
  --use_object_branch \
  --obj_fg_source attention \
  --obj_entail_parent image \
  --obj_entail_weight 0.02 \
  --obj_dist_weight 0.05 \
  --obj_cls_weight 0.05

run_job 1 obj-abdet-r2_cub_combo_e000_d005_c005 \
  --use_object_branch \
  --obj_fg_source attention \
  --obj_entail_parent image \
  --obj_entail_weight 0 \
  --obj_dist_weight 0.05 \
  --obj_cls_weight 0.05

run_job 2 obj-abdet-r2_cub_combo_e002_d000_c005 \
  --use_object_branch \
  --obj_fg_source attention \
  --obj_entail_parent image \
  --obj_entail_weight 0.02 \
  --obj_dist_weight 0 \
  --obj_cls_weight 0.05

run_job 3 obj-abdet-r2_cub_combo_e002_d005_c000 \
  --use_object_branch \
  --obj_fg_source attention \
  --obj_entail_parent image \
  --obj_entail_weight 0.02 \
  --obj_dist_weight 0.05 \
  --obj_cls_weight 0

wait
echo "========== ROUND 2 DONE =========="


echo "========== ROUND 3: CUB combo strength / negative / parent-swap tuning =========="
run_job 0 obj-abdet-r3_cub_combo_e005_d005_c005 \
  --use_object_branch \
  --obj_fg_source attention \
  --obj_entail_parent image \
  --obj_entail_weight 0.05 \
  --obj_dist_weight 0.05 \
  --obj_cls_weight 0.05

run_job 1 obj-abdet-r3_cub_combo_e002_d010_c005 \
  --use_object_branch \
  --obj_fg_source attention \
  --obj_entail_parent image \
  --obj_entail_weight 0.02 \
  --obj_dist_weight 0.10 \
  --obj_cls_weight 0.05

run_job 2 obj-abdet-r3_cub_combo_e002_d005_cneg001 \
  --use_object_branch \
  --obj_fg_source attention \
  --obj_entail_parent image \
  --obj_entail_weight 0.02 \
  --obj_dist_weight 0.05 \
  --obj_cls_weight -0.01

run_job 3 obj-abdet-r3_cub_combo_swap_parent_e002_d005_c005 \
  --use_object_branch \
  --obj_fg_source attention \
  --obj_entail_parent object \
  --obj_entail_weight 0.02 \
  --obj_dist_weight 0.05 \
  --obj_cls_weight 0.05

wait
echo "========== ALL 3 ROUNDS DONE =========="
