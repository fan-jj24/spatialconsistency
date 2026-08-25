#!/usr/bin/env bash
# ============================================================
# RL1 正式训练脚本 — verl 0.8.0 DAPO recipe
#
# 权重: /home/deepspeed/model_output/Qwen_SFT2_hf (SFT2 输出)
# 数据: /home/deepspeed/model_output/RL1 (11 个数据集, 39680 条)
# 训练: 2 epoch, LR=5e-7, warmup 10 步, ppo_epochs=1
#
# 关键配置:
#   - ppo_epochs=1: 每个 rollout batch 只做一次策略更新，
#     一个样本在一个 epoch 内只被迭代一次
#   - gpu_memory_utilization=0.7: 给 checkpoint 保存留出显存
#     (0.75 时初始保存 OOM: 需要 7.74 GiB, 只剩 7.37 GiB;
#      0.7 可多腾出 ~4.8 GiB, 预计够用; 若仍 OOM 降到 0.65)
#   - filter_overlong_prompts=False: max_prompt_length=2048 已覆盖,
#     不再做耗时的逐条长度过滤
#   - FREEZE_VISION=1 可选: 冻结 vision tower
#     (依赖已打过的 patch_freeze_vision_sft.py 补丁)
#
# 步数估算:
#   39680 条 / train_batch_size 128 = 310 个 gen batch
#   DAPO dynamic sampling 每步约消耗 3 个 gen batch → 约 100 个优化步
#   warmup = 0.03 × total_training_steps(310) ≈ 9 步
#
# 用法:
#   bash run_rl1.sh                        # 全参数训练
#   FREEZE_VISION=1 bash run_rl1.sh        # 冻结 vision（推荐）
# ============================================================

set -euo pipefail

# ------------------------------------------------------------
# 路径
# ------------------------------------------------------------
MODEL_PATH=${MODEL_PATH:-"/home/deepspeed/model_output/Qwen_SFT2_hf"}
VERL_ROOT=${VERL_ROOT:-"/home/deepspeed/qwen/verl080"}
DATA_ROOT=${DATA_ROOT:-"/home/deepspeed/model_output/RL1"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REWARD_FILE=${REWARD_FILE:-"${SCRIPT_DIR}/json_answer_reward.py"}
R4_SERVER_FILE=${R4_SERVER_FILE:-"${SCRIPT_DIR}/reward_model_server.py"}
R4_SERVER_PYTHON=${R4_SERVER_PYTHON:-python3}
R4_SERVER_HOST=${R4_SERVER_HOST:-127.0.0.1}
R4_SERVER_PORT=${R4_SERVER_PORT:-8765}
# 一对 summary 对应三条二分类模型输入；10 对即一次 forward 最多 30 条。
R4_SERVER_MAX_BATCH_SIZE=${R4_SERVER_MAX_BATCH_SIZE:-10}
R4_SERVER_MAX_WAIT_MS=${R4_SERVER_MAX_WAIT_MS:-20}
R4_SERVER_STARTUP_TIMEOUT=${R4_SERVER_STARTUP_TIMEOUT:-600}
R4_SERVER_LOG=${R4_SERVER_LOG:-"/tmp/spatialconsistency_r4_reward_server.log"}
export R4_REWARD_URL=${R4_REWARD_URL:-"http://${R4_SERVER_HOST}:${R4_SERVER_PORT}"}
export R4_REWARD_TIMEOUT_SECONDS=${R4_REWARD_TIMEOUT_SECONDS:-300}

# 11 个数据集目录
DATASET_NAMES=(
    "consistent_cot_verl"
    "inconsistent_cot_verl"
    "inconsistent_detection_verl_cot"
    "consistent_all"
    "inconsistent_all"
)

# 拼接所有 parquet 文件路径
# train_files: 只读 train_*.parquet（已移除验证样本）
# val_files: 只读 val_*.parquet（独立验证集，不参与训练）
TRAIN_FILES=""
VAL_FILES=""
for name in "${DATASET_NAMES[@]}"; do
    dir="${DATA_ROOT}/${name}"
    if [ ! -d "${dir}" ]; then
        echo "  [WARN] 目录不存在: ${dir}，跳过"
        continue
    fi
    # 训练文件
    for pf in "${dir}"/train_*.parquet; do
        [ -f "${pf}" ] || continue
        TRAIN_FILES="${TRAIN_FILES:+${TRAIN_FILES},}${pf}"
    done
    # 验证文件（独立 val_*.parquet，不参与训练）
    for vf in "${dir}"/val_*.parquet; do
        [ -f "${vf}" ] || continue
        VAL_FILES="${VAL_FILES:+${VAL_FILES},}${vf}"
    done
done

# 如果没有独立的 val 文件，回退到用 train 文件做验证
if [ -z "${VAL_FILES}" ]; then
    echo "  [WARN] 未找到 val_*.parquet，回退到用 train 文件做验证"
    VAL_FILES="${TRAIN_FILES}"
fi

# ---- 训练规模 ----
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
ROLLOUT_N=${ROLLOUT_N:-8}
# 2 epoch; ppo_epochs=1 保证每个样本每 epoch 只迭代一次
TOTAL_EPOCHS=${TOTAL_EPOCHS:-5}

# ---- 动态 batching ----
USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-false}

# ---- 序列长度 ----
# max_prompt_length=2048 已覆盖全部样本（audit 过），不做长度过滤
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
ROLLOUT_PROMPT_LENGTH=${ROLLOUT_PROMPT_LENGTH:-2048}

MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-20480}
LOG_PROB_MAX_TOKENS_PER_GPU=${LOG_PROB_MAX_TOKENS_PER_GPU:-40960}

PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-4}
LOG_PROB_MICRO_BATCH_SIZE=${LOG_PROB_MICRO_BATCH_SIZE:-8}

# ---- 验证 ----
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
TEST_FREQ=${TEST_FREQ:-10}
# val_max_samples=0 表示用全部验证样本（独立验证集只有 ~1000 条，不需要再抽样）
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-0}

# ---- 学习率 & KL ----
# LR=5e-7, warmup 固定 10 步
ACTOR_LR=${ACTOR_LR:-5e-7}
ACTOR_LR_WARMUP_STEPS=${ACTOR_LR_WARMUP_STEPS:-10}
KL_COEF=${KL_COEF:-0.0}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.0}

# ---- DAPO 专属参数 ----
CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.2}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.28}
CLIP_RATIO_C=${CLIP_RATIO_C:-3.0}
OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN:-820}
OVERLONG_PENALTY_FACTOR=${OVERLONG_PENALTY_FACTOR:-1.0}
MAX_NUM_GEN_BATCHES=${MAX_NUM_GEN_BATCHES:-30}
ENTROPY_COEFF=${ENTROPY_COEFF:-0.0}

# ---- Rollout ----
INFER_BACKEND=${INFER_BACKEND:-vllm}
ROLLOUT_TP=${ROLLOUT_TP:-2}
# 0.75 时 checkpoint 保存 OOM，降到 0.7 腾出 ~4.8 GiB
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.75}
ULYSSES_SP=${ULYSSES_SP:-1}

# ---- 日志 & checkpoint ----
PROJECT_NAME=${PROJECT_NAME:-"spatialscore_dapo_080"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"qwen3_5_9b_rl1"}
# 约 100 个优化步，每 25 步保存一次 → 约 4 个 ckpt，只保留最新 2 个
SAVE_FREQ=${SAVE_FREQ:-30}
CKPT_DIR=${CKPT_DIR:-"/home/deepspeed/model_output/rl1_ckpt"}
RESUME_MODE=${RESUME_MODE:-auto}

export WANDB_MODE=${WANDB_MODE:-"online"}
export WANDB_PROJECT=${WANDB_PROJECT:-"${PROJECT_NAME}"}

# ------------------------------------------------------------
# 环境变量
# ------------------------------------------------------------
# 关键: unset expandable_segments，否则与 vLLM CuMemAllocator 内存池冲突
unset PYTORCH_CUDA_ALLOC_CONF

# FREEZE_VISION 透传给 Ray worker（patch_freeze_vision_sft.py 的补丁读取）
export FREEZE_VISION=${FREEZE_VISION:-0}

export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-"FLASH_ATTN"}
export VLLM_USE_V1=${VLLM_USE_V1:-"1"}
export VERL_PPO_LOGGING_LEVEL=${VERL_PPO_LOGGING_LEVEL:-"INFO"}
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export RAY_memory_monitor_refresh_ms=0
export RAY_BACKEND=0
export RAY_gcs_server_request_timeout_seconds=300
export RAY_DASHBOARD_GRPC_PORT=0
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-4}

mkdir -p "${CKPT_DIR}"

# ------------------------------------------------------------
# 验证
# ------------------------------------------------------------
echo "============================================================"
echo "RL1 正式训练 (verl 0.8.0 DAPO, ${NNODES}×${NGPUS_PER_NODE} GPU)"
echo "============================================================"
echo "  模型:           ${MODEL_PATH}"
echo "  数据根目录:     ${DATA_ROOT}"
echo "  Reward:         ${REWARD_FILE}"
echo "  R4 service:     ${R4_REWARD_URL} (batch=${R4_SERVER_MAX_BATCH_SIZE}, wait=${R4_SERVER_MAX_WAIT_MS}ms)"
echo "  Epochs:         ${TOTAL_EPOCHS} (ppo_epochs=1, 样本不重复迭代)"
echo "  LR:             ${ACTOR_LR} (warmup steps=${ACTOR_LR_WARMUP_STEPS})"
echo "  Batch:          train=${TRAIN_BATCH_SIZE}, mini=${PPO_MINI_BATCH_SIZE}, rollout_n=${ROLLOUT_N}"
echo "  Prompt/Resp:    ${MAX_PROMPT_LENGTH} / ${MAX_RESPONSE_LENGTH}"
echo "  GPU mem util:   ${ROLLOUT_GPU_MEM_UTIL}"
echo "  Freeze vision:  ${FREEZE_VISION}"
echo "  Save/Test freq: ${SAVE_FREQ} / ${TEST_FREQ}"
echo "  Checkpoint:     ${CKPT_DIR}"
echo "============================================================"

# 校验文件
if [ ! -f "${REWARD_FILE}" ]; then
    echo "ERROR: Reward function 不存在: ${REWARD_FILE}"
    exit 1
fi

if [ ! -f "${R4_SERVER_FILE}" ]; then
    echo "ERROR: R4 reward service 不存在: ${R4_SERVER_FILE}"
    exit 1
fi

if [ ! -d "${VERL_ROOT}/recipe/dapo" ]; then
    echo "ERROR: verl 0.8.0 或 DAPO recipe 未安装"
    exit 1
fi

if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: 模型目录不存在: ${MODEL_PATH}"
    exit 1
fi

DATA_MISSING=0
for name in "${DATASET_NAMES[@]}"; do
    dir="${DATA_ROOT}/${name}"
    if [ ! -d "${dir}" ]; then
        echo "ERROR: 数据目录不存在: ${dir}"
        DATA_MISSING=1
    else
        parquet_count=$(ls "${dir}"/train_*.parquet 2>/dev/null | wc -l)
        if [ "${parquet_count}" -eq 0 ]; then
            echo "ERROR: ${dir} 下无 train_*.parquet 文件"
            DATA_MISSING=1
        fi
    fi
done
if [ ${DATA_MISSING} -eq 1 ]; then
    exit 1
fi

# ------------------------------------------------------------
# 配置参数
# ------------------------------------------------------------
DATA=(
    data.train_files="[${TRAIN_FILES}]"
    data.val_files="[${VAL_FILES}]"
    data.val_max_samples=${VAL_MAX_SAMPLES}
    data.prompt_key=prompt
    data.image_key=images
    data.truncation=left
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    # 不做逐条长度过滤（耗时；2048 已覆盖全部样本，超长走 left 截断）
    data.filter_overlong_prompts=False
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.trust_remote_code=True
)

# --- DAPO Actor ---
ACTOR_BASE=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.optim.lr_warmup_steps=${ACTOR_LR_WARMUP_STEPS}
    actor_rollout_ref.actor.optim.weight_decay=0.0
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    # ppo_epochs=1: 每个 rollout batch 只更新一次，样本不重复迭代
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW}
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH}
    actor_rollout_ref.actor.clip_ratio_c=${CLIP_RATIO_C}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    actor_rollout_ref.actor.grad_clip=1.0
    actor_rollout_ref.actor.strategy=fsdp
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    # use_orig_params=True: 支持混合 requires_grad（FREEZE_VISION 冻结 vision 后
    # FSDP 需要此模式；默认 False 会报 "Must flatten tensors with uniform
    # requires_grad"）。同时 optimizer 只为可训练参数分配 state，省显存。
    actor_rollout_ref.actor.fsdp_config.use_orig_params=True
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${ULYSSES_SP}
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1
)

# --- 动态 vs 固定 batching ---
if [ "${USE_DYNAMIC_BSZ}" = "true" ]; then
    ACTOR_BATCHING=(
        actor_rollout_ref.actor.use_dynamic_bsz=True
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOKENS_PER_GPU}
    )
    REF_BATCHING=(
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${LOG_PROB_MAX_TOKENS_PER_GPU}
    )
    ROLLOUT_BATCHING=(
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${LOG_PROB_MAX_TOKENS_PER_GPU}
    )
    echo "  Batching:       dynamic (token-based)"
else
    ACTOR_BATCHING=(
        actor_rollout_ref.actor.use_dynamic_bsz=False
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE}
    )
    REF_BATCHING=(
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE}
    )
    ROLLOUT_BATCHING=(
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE}
    )
    echo "  Batching:       fixed"
fi

# --- Rollout ---
ROLLOUT_BASE=(
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.prompt_length=${ROLLOUT_PROMPT_LENGTH}
    actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LENGTH}
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.multi_stage_wake_up=True
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=8192
)

# --- Reference Model ---
REF_BASE=(
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${ULYSSES_SP}
)

# --- 算法 ---
ALGO=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.kl_ctrl.kl_coef=${KL_COEF}
    algorithm.filter_groups.enable=True
    algorithm.filter_groups.metric=acc
    algorithm.filter_groups.max_num_gen_batches=${MAX_NUM_GEN_BATCHES}
)

# --- DAPO Reward ---
REWARD=(
    reward.reward_kwargs.overlong_buffer_cfg.enable=True
    reward.reward_kwargs.overlong_buffer_cfg.len=${OVERLONG_BUFFER_LEN}
    reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${OVERLONG_PENALTY_FACTOR}
    reward.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH}
    custom_reward_function.path="${REWARD_FILE}"
    custom_reward_function.name=compute_score
)

# --- Trainer ---
TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console","wandb"]'
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.val_before_train=${VAL_BEFORE_TRAIN}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.default_local_dir="${CKPT_DIR}"
    trainer.resume_mode=${RESUME_MODE}
    trainer.max_actor_ckpt_to_keep=3
    trainer.validation_data_dir="${CKPT_DIR}/val_generations"
    trainer.log_val_generations=10
)

# ------------------------------------------------------------
# 启动中央 R4 动态批处理服务
# ------------------------------------------------------------
echo "启动 R4 reward service，日志: ${R4_SERVER_LOG}"
"${R4_SERVER_PYTHON}" "${R4_SERVER_FILE}" \
    --host "${R4_SERVER_HOST}" \
    --port "${R4_SERVER_PORT}" \
    --max-batch-size "${R4_SERVER_MAX_BATCH_SIZE}" \
    --max-wait-ms "${R4_SERVER_MAX_WAIT_MS}" \
    >"${R4_SERVER_LOG}" 2>&1 &
R4_SERVER_PID=$!

cleanup_r4_server() {
    if kill -0 "${R4_SERVER_PID}" 2>/dev/null; then
        kill "${R4_SERVER_PID}" 2>/dev/null || true
        wait "${R4_SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup_r4_server EXIT INT TERM

R4_START_DEADLINE=$((SECONDS + R4_SERVER_STARTUP_TIMEOUT))
while true; do
    if ! kill -0 "${R4_SERVER_PID}" 2>/dev/null; then
        wait "${R4_SERVER_PID}" || R4_EXIT_CODE=$?
        echo "ERROR: R4 reward service 启动失败，exit=${R4_EXIT_CODE:-0}"
        tail -n 100 "${R4_SERVER_LOG}" || true
        exit 1
    fi
    if "${R4_SERVER_PYTHON}" - "${R4_REWARD_URL}/health" <<'PY' >/dev/null 2>&1
import sys
from urllib.request import urlopen

with urlopen(sys.argv[1], timeout=2) as response:
    if response.status != 200:
        raise RuntimeError(f"health check returned HTTP {response.status}")
PY
    then
        break
    fi
    if [ "${SECONDS}" -ge "${R4_START_DEADLINE}" ]; then
        echo "ERROR: R4 reward service 在 ${R4_SERVER_STARTUP_TIMEOUT}s 内未就绪"
        tail -n 100 "${R4_SERVER_LOG}" || true
        exit 1
    fi
    sleep 1
done
echo "R4 reward service 已就绪 (pid=${R4_SERVER_PID})"

# ------------------------------------------------------------
# 启动训练
# ------------------------------------------------------------
echo ""
echo "启动 RL1 训练 (verl 0.8.0 DAPO)..."
echo ""

cd "${VERL_ROOT}"
PYTHONPATH="${SCRIPT_DIR}:${VERL_ROOT}:${PYTHONPATH:-}" \
python3 -m recipe.dapo.main_dapo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR_BASE[@]}" \
    "${ACTOR_BATCHING[@]}" \
    "${ROLLOUT_BASE[@]}" \
    "${ROLLOUT_BATCHING[@]}" \
    "${REF_BASE[@]}" \
    "${REF_BATCHING[@]}" \
    "${ALGO[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "$@"
