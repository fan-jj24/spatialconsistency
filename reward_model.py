#!/usr/bin/env python3
"""
R4 奖励模型: Qwen3.5-0.8B summary 一致性校验。

在 verl RLVR 奖励框架中，R4 负责校验模型输出的 summary 与 GT summary
是否语义一致。使用 Qwen/Qwen3.5-0.8B（0.8B VLM）作为判官模型，
运行在 CPU 上（bfloat16）。

设计:
  - 模型单例: 全局只加载一次，进程内复用（verl reward worker 常驻）。
  - 下载: ModelScope snapshot_download（sandbox 内 HF 不可达）。
  - CPU 推理: dtype=bfloat16, device_map="cpu"。
  - 批处理: 支持单条和批量调用，批量时统一 left-padding。
  - 降级: 模型不可用时返回 -1.0，调用方自行降级处理。
  - 线程安全: 加载和推理各用一把锁，避免 verl 多线程 reward 并发问题。

调用方式:
    from reward_model import get_reward_model
    rm = get_reward_model()          # 获取单例
    score = rm.score_summary(pred_summary, gt_summary)  # → float ∈ [0,1]

    # 批量
    scores = rm.score_summaries([(pred1, gt1), (pred2, gt2)])

打分方案: 3 档分类 + forward logits 加权
  让模型从 A/B/C 三个选项中选一个，只做一次 forward（不 generate），
  取 last position 的 logits，对 A/B/C 三个 token 做 3-way softmax，
  最终分数 = Σ(档位分数 × P(档位))。

  forward 比 generate 更好: 确定性，无采样噪声，不会两次结果不一致，
  且只需一次 forward，速度更快。

  不用 argmax（离散值，单条和 batch 的 padding 差异会导致 argmax
  翻转，结果不一致）。用概率加权得到连续值，和旧方案 P(yes) 一样
  的思路，只是从 2 元变 3 元。

  token id 获取: 不用 encode() 取最后一个 token（BPE 会混入噪声），
  而是用 convert_tokens_to_ids() 直接获取单字母 token id，并反向
  convert_ids_to_tokens() 验证确实解码回原字母，过滤掉噪声 token。

  3 档而非 5 档: 0.8B 模型对 5 档区分能力不足，倾向于给中间偏高
  的安全档。3 档（一致/部分/不一致）区分度足够 RL 使用，
  且 0.8B 能可靠区分。

  prompt 用英文（与 summary 语言一致），避免跨语言理解力下降。

  档位    选项     分数     含义
    1      A       1.0      Consistent (same change)
    2      B       0.5      Partially consistent
    3      C       0.0      Inconsistent (different change)

  最终分数 = 1.0×P(A) + 0.5×P(B) + 0.0×P(C) ∈ [0, 1]

Prompt 设计:
    system: You are a spatial consistency judge. Given a reference
            summary and a model summary, determine if they describe
            the same spatial change (same change type, same direction,
            same object). Choose one:
            A - Consistent (same change)
            B - Partially consistent (same type but different detail)
            C - Inconsistent (different change)
            Different change types (e.g. movement vs background change)
            are always C.
            Reply with only one letter.
    user:   Reference summary: {gt_summary}
            Model summary: {pred_summary}
            Answer:
"""

import threading
import logging
from typing import Optional, List, Tuple

import torch

logger = logging.getLogger(__name__)

# ── 模型配置 ──
MODEL_NAME = "Qwen/Qwen3.5-0.8B"
# 本地模型路径: 设为 None 则从 ModelScope 自动下载到默认缓存目录；
# 设为本地路径（如 "E:/models/Qwen3.5-0.8B"）则直接从本地加载，跳过下载。
MODEL_LOCAL_PATH = "/home/deepspeed/model_output/Qwen3.5-0.8B"
# CPU 推理: bfloat16（0.8B 模型 ~1.6GB）
TORCH_DTYPE = "bfloat16"
DEVICE = "cpu"

# ── 3 档分类配置 ──
# 档位字母 → 对应分数
CHOICE_SCORES = {
    "A": 1.0,
    "B": 0.5,
    "C": 0.0,
}
# 所有选项字母
CHOICE_LETTERS = list(CHOICE_SCORES.keys())

# ── Prompt 模板（英文）──
SYSTEM_PROMPT = (
    "You are a spatial consistency judge. Given a reference summary and a model summary, "
    "determine if they describe the same spatial change "
    "(same change type, same direction, same object).\n"
    "Choose one:\n"
    "A - Consistent (same change)\n"
    "B - Partially consistent (same type but different detail)\n"
    "C - Inconsistent (different change)\n"
    "Different change types (e.g. movement vs background change) are always C.\n"
    "Reply with only one letter."
)
USER_TEMPLATE = (
    "Reference summary: {gt}\n"
    "Model summary: {pred}\n"
    "Answer:"
)


class RewardModel:
    """Qwen3.5-0.8B 奖励模型封装。

    线程安全: _load_lock 保护加载，_infer_lock 保护推理。
    """

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._loaded = False
        self._choice_token_ids = None  # {"A": [id, ...], "B": [...], "C": [...]}

    def _ensure_loaded(self):
        """延迟加载模型（线程安全，只加载一次）。

        使用 ModelScope 下载模型（sandbox 内 HF 不可达），
        下载到本地后用 transformers 从本地路径加载。
        """
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:  # double-check
                return
            logger.info("Loading Qwen3.5-0.8B reward model on CPU...")
            try:
                # ── 禁用 fla / causal_conv1d，强制纯 PyTorch ──
                # Qwen3.5 的 Gated DeltaNet 架构依赖 fla 和 causal_conv1d
                # 两个加速库，它们都用 Triton/CUDA kernel，在 CPU 上会崩。
                # transformers 用 is_causal_conv1d_available() 和
                # is_flash_linear_attention_available() 判断是否安装，
                # 直接 patch 这两个函数返回 False，让 transformers 走
                # 纯 PyTorch fallback。不卸载库，不影响主干模型。
                import transformers.utils.import_utils as iu
                iu.is_causal_conv1d_available = lambda: False
                iu.is_flash_linear_attention_available = lambda: False
                # 如果 qwen3_5 模块已经 import 了，也要 patch 模块级引用
                import sys
                if "transformers.models.qwen3_5.modeling_qwen3_5" in sys.modules:
                    qwen_mod = sys.modules["transformers.models.qwen3_5.modeling_qwen3_5"]
                    qwen_mod.causal_conv1d_fn = None
                    qwen_mod.causal_conv1d_update = None
                    qwen_mod.chunk_gated_delta_rule = None
                    qwen_mod.fused_recurrent_gated_delta_rule = None
                    qwen_mod.FusedRMSNormGated = None
                logger.info("Disabled fla/causal_conv1d for CPU-only inference.")

                from transformers import AutoModelForCausalLM, AutoTokenizer

                # 优先使用本地路径，否则从 ModelScope 下载
                if MODEL_LOCAL_PATH:
                    model_path = MODEL_LOCAL_PATH
                    logger.info("Using local model path: %s", model_path)
                    # 本地路径: 强制不联网
                    tok_kwargs = {"trust_remote_code": True, "local_files_only": True}
                    model_kwargs = {"dtype": TORCH_DTYPE, "device_map": DEVICE,
                                     "trust_remote_code": True, "local_files_only": True}
                else:
                    from modelscope import snapshot_download
                    model_path = snapshot_download(MODEL_NAME)
                    logger.info("Model downloaded to: %s", model_path)
                    tok_kwargs = {"trust_remote_code": True}
                    model_kwargs = {"dtype": TORCH_DTYPE, "device_map": DEVICE,
                                     "trust_remote_code": True}

                self._tokenizer = AutoTokenizer.from_pretrained(
                    model_path, **tok_kwargs,
                )
                # right-padding: padding 在序列末尾，不污染 DeltaNet
                # 的递归状态（left-padding 会把 padding token 放在前面，
                # 线性注意力的递归状态从头被污染，导致 logits 偏移）
                if self._tokenizer.pad_token is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token
                self._tokenizer.padding_side = "right"

                self._model = AutoModelForCausalLM.from_pretrained(
                    model_path, **model_kwargs,
                )
                self._model.eval()

                # 预计算 A/B/C 的 token ids（严格验证）
                self._choice_token_ids = self._get_choice_token_ids()
                if not self._choice_token_ids:
                    raise RuntimeError(
                        "Cannot find valid choice token ids for A/B/C"
                    )

                self._loaded = True
                logger.info(
                    "Reward model loaded. choice_token_ids=%s",
                    self._choice_token_ids,
                )
            except Exception as e:
                logger.error("Failed to load reward model: %s", e)
                raise

    def _get_choice_token_ids(self) -> dict:
        """获取 A/B/C 各字母的 token id（严格验证）。

        不用 encode() 取最后一个 token（BPE 会混入噪声 token），
        而是用 convert_tokens_to_ids() 直接获取单字母 token id，
        并用 convert_ids_to_tokens() 反向验证确实解码回原字母。

        尝试的变体: "A"/"a"/" A"/" a"（带前缀空格，因为 prompt
        末尾是 "Answer:" 后面模型可能输出 " A" 或 "A"）。
        只保留反向验证通过的 token id。
        """
        result = {}
        for letter in CHOICE_LETTERS:
            ids = set()
            for variant in [letter, letter.lower(), letter.upper(),
                            " " + letter, " " + letter.lower()]:
                # 直接用 convert_tokens_to_ids 获取 token id
                tid = self._tokenizer.convert_tokens_to_ids(variant)
                if tid is None or tid == self._tokenizer.unk_token_id:
                    continue
                # 反向验证: 这个 token id 解码回来应该包含原字母
                decoded = self._tokenizer.convert_ids_to_tokens(tid)
                # 检查解码结果是否就是该字母（允许大小写差异）
                if decoded.strip().lower() == letter.lower():
                    ids.add(tid)
            result[letter] = list(ids)
        return result

    def _build_prompt(self, pred_summary: str, gt_summary: str) -> str:
        """构建 chat 格式的 prompt。"""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                gt=gt_summary, pred=pred_summary)},
        ]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return text

    def _logits_to_score(self, last_logits: torch.Tensor) -> float:
        """从 last position 的 logits 计算 3 档加权分数。

        取 A/B/C 各自所有 token id 的 logits 最大值，
        转 float32 后做 3-way softmax（bfloat16 精度不足，
        会导致 P(A)≈1 时出现 1.0003 > 1 的溢出），
        最终分数 = Σ(档位分数 × P(档位))。
        """
        # 收集每个选项的 logit（取该选项所有 token id 中的最大值）
        choice_logits = []
        for letter in CHOICE_LETTERS:
            ids = self._choice_token_ids[letter]
            if not ids:
                choice_logits.append(torch.tensor(-1e9, dtype=torch.float32))
            else:
                choice_logits.append(last_logits[ids].max().to(torch.float32))

        logits_tensor = torch.stack(choice_logits)  # (3,) float32
        probs = torch.softmax(logits_tensor, dim=0)  # (3,) float32

        # 加权分数 = Σ(档位分数 × P(档位))
        scores_tensor = torch.tensor(
            [CHOICE_SCORES[letter] for letter in CHOICE_LETTERS],
            dtype=torch.float32,
        )
        expected_score = float((probs * scores_tensor).sum().item())
        return expected_score

    def score_summary(self, pred_summary: str, gt_summary: str) -> float:
        """校验单条 summary 一致性。

        3 档分类 forward 方案: 一次 forward，取 A/B/C 的 logits
        做 softmax，加权计算分数。

        Returns:
            float ∈ [0, 1]: 加权分数 = 1.0×P(A) + 0.5×P(B) + 0.0×P(C)。
            -1.0: 模型不可用（调用方降级）。
        """
        if not pred_summary or not gt_summary:
            return 0.0
        try:
            self._ensure_loaded()
        except Exception:
            return -1.0

        prompt = self._build_prompt(pred_summary, gt_summary)
        inputs = self._tokenizer(prompt, return_tensors="pt")

        with self._infer_lock:
            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits  # (1, seq_len, vocab_size)
                last_logits = logits[0, -1, :]  # (vocab_size,)

        return self._logits_to_score(last_logits)

    def score_summaries(
        self, pairs: List[Tuple[str, str]]
    ) -> List[float]:
        """批量校验 summary 一致性。

        使用 right-padding + batch forward。right-padding 把 padding
        token 放在序列末尾，不污染 DeltaNet 的递归状态（left-padding
        会把 padding 放在前面，线性注意力的递归状态从头被污染）。
        用 attention_mask 找每个序列的最后一个有效 token 位置。

        Args:
            pairs: [(pred_summary, gt_summary), ...]
        Returns:
            [float, ...]: 每对的加权分数，模型不可用时全 -1.0。
        """
        if not pairs:
            return []
        try:
            self._ensure_loaded()
        except Exception:
            return [-1.0] * len(pairs)

        # 构建所有 prompt
        prompts = [self._build_prompt(p, g) for p, g in pairs]
        # right-padding（padding 在末尾）
        encoded = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )

        with self._infer_lock:
            with torch.no_grad():
                outputs = self._model(**encoded)
                logits = outputs.logits  # (batch, seq_len, vocab_size)
                # right-padding: 最后一个有效 token 是 attention_mask
                # 中最后一个 1 的位置（不是 seq_len-1，因为后面是 padding）
                attention_mask = encoded["attention_mask"]
                # 找每个序列最后一个有效位置
                seq_lengths = attention_mask.sum(dim=1) - 1  # (batch,)

                scores = []
                for i in range(logits.size(0)):
                    last_idx = int(seq_lengths[i].item())
                    last_logits = logits[i, last_idx, :]
                    scores.append(self._logits_to_score(last_logits))

        return scores


# ── 全局单例 ──
_global_model: Optional[RewardModel] = None
_singleton_lock = threading.Lock()


def get_reward_model() -> RewardModel:
    """获取全局 RewardModel 单例。"""
    global _global_model
    if _global_model is None:
        with _singleton_lock:
            if _global_model is None:
                _global_model = RewardModel()
    return _global_model


# ── 便捷函数（供 reward function 直接调用）──


def score_summary(pred_summary: str, gt_summary: str) -> float:
    """便捷接口: 校验单条 summary 一致性。

    Returns:
        float ∈ [0, 1]: 加权分数 = 1.0×P(A) + 0.5×P(B) + 0.0×P(C)。
        -1.0: 模型不可用。
    """
    return get_reward_model().score_summary(pred_summary, gt_summary)


def score_summaries(pairs: List[Tuple[str, str]]) -> List[float]:
    """便捷接口: 批量校验。"""
    return get_reward_model().score_summaries(pairs)


# ── 本地测试 ──
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== R4 奖励模型测试 (3档 forward) ===\n")

    # 单条测试
    print("--- 单条 ---")
    import time

    t0 = time.time()
    s1 = score_summary(
        "The person moved from left to right.",
        "The person moved from left to right."
    )
    print(f"  完全一致: {s1:.4f}")

    s2 = score_summary(
        "The background changed significantly.",
        "The person moved from left to right."
    )
    print(f"  完全不同: {s2:.4f}")

    s3 = score_summary(
        "A person shifted towards the right side.",
        "The person moved from left to right."
    )
    print(f"  语义近似: {s3:.4f}")

    s4 = score_summary(
        "The cat sat on the mat.",
        "The person moved from left to right."
    )
    print(f"  完全不同领域: {s4:.4f}")

    s5 = score_summary(
        "The person moved from left to right.",
        "The person moved from left to right. The background also changed."
    )
    print(f"  GT是pred的超集: {s5:.4f}")
    t1 = time.time()
    print(f"  单条总耗时: {t1-t0:.2f}s ({(t1-t0)/5:.2f}s/条)")

    # 批量测试
    print("\n--- 批量 ---")
    pairs = [
        ("The person moved from left to right.",
         "The person moved from left to right."),
        ("The background changed significantly.",
         "The person moved from left to right."),
        ("A person shifted towards the right side.",
         "The person moved from left to right."),
        ("The cat sat on the mat.",
         "The person moved from left to right."),
        ("The person moved from left to right.",
         "The person moved from left to right. The background also changed."),
    ]
    t0 = time.time()
    scores = score_summaries(pairs)
    t1 = time.time()
    for i, s in enumerate(scores):
        print(f"  pair {i+1}: {s:.4f}")
    print(f"  批量总耗时: {t1-t0:.2f}s ({(t1-t0)/len(pairs):.2f}s/条)")

    # 128 样本批量测试
    print("\n--- 128 样本批量 ---")
    import random
    random.seed(42)
    templates_pred = [
        "The person moved from left to right.",
        "The background changed significantly.",
        "A person shifted towards the right side.",
        "The cat sat on the mat.",
        "The object rotated 90 degrees clockwise.",
        "A new object was added to the scene.",
        "The person moved forward.",
        "The lighting changed from bright to dark.",
    ]
    templates_gt = [
        "The person moved from left to right.",
        "The person moved from left to right. The background also changed.",
        "The object rotated 90 degrees clockwise.",
        "The background changed significantly.",
        "A completely different scene with no relation.",
    ]
    big_pairs = [
        (random.choice(templates_pred), random.choice(templates_gt))
        for _ in range(128)
    ]
    t0 = time.time()
    big_scores = score_summaries(big_pairs)
    t1 = time.time()
    print(f"  128 样本批量耗时: {t1-t0:.2f}s ({(t1-t0)/128:.3f}s/条)")
    print(f"  分数范围: [{min(big_scores):.4f}, {max(big_scores):.4f}]")
    print(f"  平均分数: {sum(big_scores)/len(big_scores):.4f}")

    print("\n=== 测试完毕 ===")
