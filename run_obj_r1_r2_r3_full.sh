#!/usr/bin/env bash
set -uo pipefail

cd /data/projects/HypCD
mkdir -p logs
export CUBLAS_WORKSPACE_CONFIG=:4096:8

ENTRY=train.train_HypSimGCD_org_det_ab_obj

launch_job () {
  local gpu="$1"
  local dataset="$2"
  local round="$3"
  local name="$4"
  shift 4

  local cr hts
  case "${dataset}" in
    scars)
      cr=1.2
      hts=0.3
      ;;
    aircraft)
      cr=2.3
      hts=0.4
      ;;
    *)
      echo "unknown dataset: ${dataset}"
      exit 1
      ;;
  esac

  local log_file="logs/${round}_gpu${gpu}_${dataset}_${name}.log"
  local exp_name="${round}_${dataset}_${name}"

  echo "[START] ${round} GPU${gpu} ${dataset} ${name}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  conda run --no-capture-output -n hypcd \
  python -m ${ENTRY} \
    --dataset_name "${dataset}" \
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
    --cr "${cr}" \
    --hyper_max_weight 1.0 \
    --hyper_temp_scale "${hts}" \
    --hyper_start_epoch 0 \
    --hyper_end_epoch 200 \
    --seed 0 \
    --deterministic \
    --print_freq 10 \
    --exp_name "${exp_name}" \
    "$@" \
    > "${log_file}" 2>&1 &

  PIDS+=("$!")
  LOGS+=("${log_file}")
}

run_round () {
  local round="$1"
  shift

  PIDS=()
  LOGS=()

  echo
  echo "============================================================"
  echo "[$(date '+%F %T')] START ${round}"
  echo "============================================================"

  "$@"

  failed=0
  for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
      echo "[ERROR] failed: ${LOGS[$i]}"
      failed=1
    else
      echo "[DONE] ${LOGS[$i]}"
    fi
  done

  if (( failed != 0 )); then
    echo "[ABORT] ${round} has failed job(s). Stop before next round."
    exit 1
  fi

  echo "[$(date '+%F %T')] FINISH ${round}"
}

round1_jobs () {
  launch_job 0 scars obj-r1 noobj \
    --no_object_branch

  launch_job 1 aircraft obj-r1 noobj \
    --no_object_branch

  launch_job 2 scars obj-r1 objzero \
    --use_object_branch \
    --obj_fg_source attention \
    --obj_entail_weight 0 \
    --obj_dist_weight 0 \
    --obj_cls_weight 0

  launch_job 3 aircraft obj-r1 objzero \
    --use_object_branch \
    --obj_fg_source attention \
    --obj_entail_weight 0 \
    --obj_dist_weight 0 \
    --obj_cls_weight 0
}

round2_jobs () {
  launch_job 0 scars obj-r2 entail005 \
    --use_object_branch \
    --obj_fg_source attention \
    --obj_entail_weight 0.05 \
    --obj_dist_weight 0 \
    --obj_cls_weight 0 \
    --obj_entail_parent image \
    --obj_aperture_scale 1.2 \
    --obj_min_radius 0.1

  launch_job 1 aircraft obj-r2 entail005 \
    --use_object_branch \
    --obj_fg_source attention \
    --obj_entail_weight 0.05 \
    --obj_dist_weight 0 \
    --obj_cls_weight 0 \
    --obj_entail_parent image \
    --obj_aperture_scale 1.2 \
    --obj_min_radius 0.1

  launch_job 2 scars obj-r2 dist005 \
    --use_object_branch \
    --obj_fg_source attention \
    --obj_entail_weight 0 \
    --obj_dist_weight 0.05 \
    --obj_cls_weight 0 \
    --obj_dist_temp 0.1

  launch_job 3 aircraft obj-r2 dist005 \
    --use_object_branch \
    --obj_fg_source attention \
    --obj_entail_weight 0 \
    --obj_dist_weight 0.05 \
    --obj_cls_weight 0 \
    --obj_dist_temp 0.1
}

round3_jobs () {
  launch_job 0 scars obj-r3 cls005 \
    --use_object_branch \
    --obj_fg_source attention \
    --obj_entail_weight 0 \
    --obj_dist_weight 0 \
    --obj_cls_weight 0.05

  launch_job 1 aircraft obj-r3 cls005 \
    --use_object_branch \
    --obj_fg_source attention \
    --obj_entail_weight 0 \
    --obj_dist_weight 0 \
    --obj_cls_weight 0.05

  launch_job 2 scars obj-r3 combo_e002_d005_c005 \
    --use_object_branch \
    --obj_fg_source attention \
    --obj_entail_weight 0.02 \
    --obj_dist_weight 0.05 \
    --obj_cls_weight 0.05 \
    --obj_entail_parent image \
    --obj_aperture_scale 1.2 \
    --obj_min_radius 0.1 \
    --obj_dist_temp 0.1

  launch_job 3 aircraft obj-r3 combo_e002_d005_c005 \
    --use_object_branch \
    --obj_fg_source attention \
    --obj_entail_weight 0.02 \
    --obj_dist_weight 0.05 \
    --obj_cls_weight 0.05 \
    --obj_entail_parent image \
    --obj_aperture_scale 1.2 \
    --obj_min_radius 0.1 \
    --obj_dist_temp 0.1
}

run_round obj-r1 round1_jobs
run_round obj-r2 round2_jobs
run_round obj-r3 round3_jobs

echo
echo "[$(date '+%F %T')] ALL obj-r1/r2/r3 FINISHED."
