#!/usr/bin/env bash
set -uo pipefail

cd /data/projects/HypCD
mkdir -p logs

# --deterministic 所需的 cuBLAS 配置
export CUBLAS_WORKSPACE_CONFIG=:4096:8


launch_job() {
  local gpu="$1"
  local dataset="$2"
  local model="$3"
  local cr="$4"
  local hts="$5"
  local mode="$6"
  local tag="$7"

  local extra_args=()

  # epoch 0–32 冻结曲率，从 epoch 33 开始学习曲率
  if [[ "${mode}" == "unfreeze33" ]]; then
    extra_args=(
      --train_c
      --c_unfreeze_rep_cls_epoch 33
      --c_unfreeze_sup_unsup_epoch 33
    )
  fi

  local log_file="logs/${tag}_gpu${gpu}_${dataset}_${mode}.log"
  local exp_name="${tag}_${dataset}_${mode}"

  CUDA_VISIBLE_DEVICES="${gpu}" \
  conda run --no-capture-output -n hypcd \
  python -m train.train_HypCD_mc_det_ab \
    --dataset_name "${dataset}" \
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
    --model_name "${model}" \
    --c 0.1 \
    --cr "${cr}" \
    --hyper_max_weight 1.0 \
    --hyper_temp_scale "${hts}" \
    --c_tie_mode hard \
    --c_tie_rep_cls 1.0 \
    --c_tie_sup_unsup 1.0 \
    --c_unbind_rep_cls_epoch -1 \
    --c_unbind_sup_unsup_epoch -1 \
    --c_eval_role c_cls_unsup \
    --seed 0 \
    --deterministic \
    --exp_name "${exp_name}" \
    "${extra_args[@]}" \
    > "${log_file}" 2>&1 &

  PIDS+=("$!")
  LOGS+=("${log_file}")

  echo "[START] GPU${gpu}: ${dataset}, ${model}, ${mode}"
  echo "        log: ${log_file}"
}


run_batch() {
  local tag="$1"
  local model="$2"

  local scars_cr="$3"
  local scars_hts="$4"
  local cub_cr="$5"
  local cub_hts="$6"
  local aircraft_cr="$7"
  local aircraft_hts="$8"

  PIDS=()
  LOGS=()

  echo
  echo "============================================================"
  echo "[$(date '+%F %T')] Starting ${tag}, backbone=${model}"
  echo "============================================================"

  # GPU0：SCars，全绑定、全冻结
  launch_job \
    0 scars "${model}" \
    "${scars_cr}" "${scars_hts}" \
    freeze "${tag}"

  # GPU1：CUB，全绑定、全冻结
  launch_job \
    1 cub "${model}" \
    "${cub_cr}" "${cub_hts}" \
    freeze "${tag}"

  # GPU2：Aircraft，全绑定、全冻结
  launch_job \
    2 aircraft "${model}" \
    "${aircraft_cr}" "${aircraft_hts}" \
    freeze "${tag}"

  # GPU3：SCars，全绑定，epoch 33 解冻曲率
  launch_job \
    3 scars "${model}" \
    "${scars_cr}" "${scars_hts}" \
    unfreeze33 "${tag}"

  failed=0

  for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
      echo "[ERROR] Job failed: ${LOGS[$i]}"
      failed=1
    else
      echo "[DONE] ${LOGS[$i]}"
    fi
  done

  if (( failed != 0 )); then
    echo "[ABORT] ${tag} 中存在失败任务，不启动下一批。"
    exit 1
  fi

  echo "[$(date '+%F %T')] Finished ${tag}"
}


# ============================================================
# 第一批：vis_r2，DINO v1
#
# 论文参数：
#   SCars:    c=0.1, cr=1.2, hyper_temp_scale=0.3
#   CUB:      c=0.1, cr=2.0, hyper_temp_scale=0.3
#   Aircraft: c=0.1, cr=2.3, hyper_temp_scale=0.4
# ============================================================
run_batch \
  vis_r2 v1 \
  1.2 0.3 \
  2.0 0.3 \
  2.3 0.4


# ============================================================
# 第二批：vis_r3，DINOv2
# 仅在 vis_r2 四个任务全部正常完成后启动
#
# 论文参数：
#   SCars:    c=0.1, cr=1.2, hyper_temp_scale=0.35
#   CUB:      c=0.1, cr=1.2, hyper_temp_scale=0.4
#   Aircraft: c=0.1, cr=2.0, hyper_temp_scale=0.4
# ============================================================
run_batch \
  vis_r3 v2 \
  1.2 0.35 \
  1.2 0.4 \
  2.0 0.4


echo
echo "[$(date '+%F %T')] vis_r2 和 vis_r3 全部完成。"
