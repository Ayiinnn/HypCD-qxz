#!/usr/bin/env bash
set -euo pipefail

cd /data/projects/HypCD
mkdir -p logs
export CUBLAS_WORKSPACE_CONFIG=:4096:8

ENTRY=train.train_HypSimGCD_org_det_ab_obj

echo "[1/4] py_compile"
conda run --no-capture-output -n hypcd \
python -m py_compile \
  train/train_HypSimGCD_org_det_ab_obj.py \
  models/object_branch.py \
  models/foreground.py \
  hyptorch/entailment.py

echo "[2/4] argparse check"
conda run --no-capture-output -n hypcd \
python -m ${ENTRY} --help > logs/obj-pre_help.log 2>&1

grep -E -- '--no_object_branch|--obj_entail_weight|--obj_dist_weight|--obj_cls_weight|--deterministic' logs/obj-pre_help.log

run_smoke () {
  local gpu="$1"
  local dataset="$2"
  local name="$3"
  local epochs="$4"
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

  local log_file="logs/obj-pre_gpu${gpu}_${dataset}_${name}.log"

  echo "[SMOKE] GPU${gpu} ${dataset} ${name}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  conda run --no-capture-output -n hypcd \
  python -m ${ENTRY} \
    --dataset_name "${dataset}" \
    --batch_size 32 \
    --grad_from_block 11 \
    --epochs "${epochs}" \
    --num_workers 2 \
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
    --hyper_end_epoch "${epochs}" \
    --seed 0 \
    --deterministic \
    --print_freq 1 \
    --exp_name "obj-pre_${dataset}_${name}" \
    "$@" \
    > "${log_file}" 2>&1

  tail -n 20 "${log_file}"
}

echo "[3/4] smoke: original-degenerate path"
run_smoke 0 scars noobj_e2 2 \
  --no_object_branch

echo "[4/4] smoke: object branch forward + best_all save path"
run_smoke 0 scars objzero_e2 2 \
  --use_object_branch \
  --obj_fg_source attention \
  --obj_entail_weight 0 \
  --obj_dist_weight 0 \
  --obj_cls_weight 0

run_smoke 0 aircraft objcls001_e2 2 \
  --use_object_branch \
  --obj_fg_source attention \
  --obj_entail_weight 0 \
  --obj_dist_weight 0 \
  --obj_cls_weight 0.01

echo
echo "[CHECK] key lines"
grep -R "object-branch\|obj_ent\|best_acc_all\|Traceback\|nan" logs/obj-pre_*.log || true
