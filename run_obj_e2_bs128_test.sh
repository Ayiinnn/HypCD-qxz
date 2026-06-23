#!/usr/bin/env bash
set -uo pipefail

cd /data/projects/HypCD
mkdir -p logs
export CUBLAS_WORKSPACE_CONFIG=:4096:8

ENTRY=train.train_HypSimGCD_org_det_ab_obj

run_job () {
  local gpu="$1"
  local dataset="$2"
  local tag="$3"
  shift 3

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

  local log_file="logs/obj-e2-bs128_gpu${gpu}_${dataset}_${tag}.log"

  echo "[START] GPU${gpu} ${dataset} ${tag}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  conda run --no-capture-output -n hypcd \
  python -m ${ENTRY} \
    --dataset_name "${dataset}" \
    --batch_size 128 \
    --grad_from_block 11 \
    --epochs 2 \
    --num_workers 8 \
    --eval_funcs v2 v2b \
    --use_ssb_splits \
    --sup_weight 0.35 \
    --weight_decay 5e-5 \
    --transform imagenet \
    --lr 0.1 \
    --warmup_teacher_temp 0.07 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 1 \
    --memax_weight 1.0 \
    --model_name v1 \
    --c 0.1 \
    --cr "${cr}" \
    --hyper_max_weight 1.0 \
    --hyper_temp_scale "${hts}" \
    --hyper_start_epoch 0 \
    --hyper_end_epoch 2 \
    --seed 0 \
    --deterministic \
    --print_freq 1 \
    --exp_name "obj-e2-bs128_${dataset}_${tag}" \
    "$@" \
    > "${log_file}" 2>&1 &

  PIDS+=("$!")
  LOGS+=("${log_file}")
}

PIDS=()
LOGS=()

run_job 0 scars noobj \
  --no_object_branch

run_job 1 aircraft noobj \
  --no_object_branch

run_job 2 scars objzero \
  --use_object_branch \
  --obj_fg_source attention \
  --obj_entail_weight 0 \
  --obj_dist_weight 0 \
  --obj_cls_weight 0

run_job 3 aircraft objzero \
  --use_object_branch \
  --obj_fg_source attention \
  --obj_entail_weight 0 \
  --obj_dist_weight 0 \
  --obj_cls_weight 0

failed=0
for i in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$i]}"; then
    echo "[ERROR] failed: ${LOGS[$i]}"
    failed=1
  else
    echo "[DONE] ${LOGS[$i]}"
  fi
done

echo
echo "[CHECK]"
grep -R "object-branch\|obj_ent\|obj_dist\|obj_cls\|best_acc_all\|Traceback\|nan" logs/obj-e2-bs128_*.log || true

exit ${failed}
