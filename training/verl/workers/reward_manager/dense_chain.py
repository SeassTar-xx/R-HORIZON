import math
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import requests
import torch

from verl import DataProto


class DenseChainRewardManager:
    """
    Dense reward manager for multi-query chained math reasoning.

    This manager is designed to be robust against heterogeneous datasets:
    - If fields such as dependencies/difficulty are missing, it falls back to
      safe defaults instead of crashing.
    - If composed_query_num < 2, reward is 0.0 (sample is skipped logically).
    """

    DEFAULT_REWARD_WEIGHTS = {
        "prog": 0.4,
        "comp": 0.2,
        "neg": 0.15,
        "cons": 0.15,
        "form": 0.1,
    }

    DEFAULT_AVG_TOKENS_PER_PROBLEM = {
        "easy": 2500,
        "medium": 4000,
        "hard": 16000,
    }

    def __init__(self, tokenizer, num_examine, dense_reward_config: Dict[str, Any] = None, **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine

        cfg = dense_reward_config or {}
        self.reward_weights = cfg.get("reward_weights", self.DEFAULT_REWARD_WEIGHTS)
        self.eff_lambda = float(cfg.get("efficiency_penalty_lambda", 0.05))
        self.avg_tokens_per_problem = cfg.get("avg_tokens_per_problem", self.DEFAULT_AVG_TOKENS_PER_PROBLEM)
        self.neg_reward_min = float(cfg.get("neg_reward_min", -0.2))
        self.neg_reward_max = float(cfg.get("neg_reward_max", 0.2))
        # Looser matching fixes log pattern "prog_reward always 0" when \\boxed content differs cosmetically.
        self.progress_partial_credit = bool(cfg.get("progress_partial_credit", True))

        # LLM judge：OpenAI 兼容 Chat Completions，可通过 YAML / 环境变量覆盖。
        self.enable_llm_judge = bool(cfg.get("enable_llm_judge", True))
        self.max_workers = int(cfg.get("max_workers", 3))
        self.llm_base_url = cfg.get("base_url", "https://api.openai.com/v1/chat/completions")
        self.llm_model_name = cfg.get("model_name", "gpt-4o-mini")
        _key = (cfg.get("api_key") or "").strip()
        self.llm_api_key = _key or os.environ.get("LLM_JUDGE_API_KEY", "").strip() or os.environ.get(
            "DEEPSEEK_API_KEY", ""
        ).strip()
        self.llm_temperature = float(cfg.get("temperature", 0.0))
        self.llm_max_tokens = int(cfg.get("max_tokens", 16000))
        self.llm_timeout = int(cfg.get("timeout", 30))

        self._llm_stats_lock = threading.Lock()
        self._llm_calls_this_step = 0
        self._llm_success_this_step = 0
        self._llm_errors_this_step = 0

    # ---------------------------
    # Public entry
    # ---------------------------
    def __call__(self, data: DataProto):
        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]

        with self._llm_stats_lock:
            self._llm_calls_this_step = 0
            self._llm_success_this_step = 0
            self._llm_errors_this_step = 0

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        totals_prog: List[float] = []
        totals_comp: List[float] = []
        totals_neg: List[float] = []
        totals_cons: List[float] = []
        totals_form: List[float] = []
        totals_weighted: List[float] = []

        # Sample-level parallelism; each sample performs its own dense scoring.
        jobs = []
        with ThreadPoolExecutor(max_workers=max(1, self.max_workers)) as ex:
            for i in range(len(data)):
                data_item = data[i]
                jobs.append((i, ex.submit(self._score_single_item, data_item)))

            for i, fut in jobs:
                score, valid_response_length, comps = fut.result()
                if valid_response_length > 0:
                    reward_tensor[i, valid_response_length - 1] = float(score)
                totals_prog.append(comps["prog_reward"])
                totals_comp.append(comps["comp_reward"])
                totals_neg.append(comps["neg_reward"])
                totals_cons.append(comps["cons_reward"])
                totals_form.append(comps["form_reward"])
                totals_weighted.append(comps["total_reward"])

        self._last_step_dense_means = {
            "total_reward": float(mean(totals_weighted)) if totals_weighted else 0.0,
            "prog_reward": float(mean(totals_prog)) if totals_prog else 0.0,
            "comp_reward": float(mean(totals_comp)) if totals_comp else 0.0,
            "neg_reward": float(mean(totals_neg)) if totals_neg else 0.0,
            "cons_reward": float(mean(totals_cons)) if totals_cons else 0.0,
            "form_reward": float(mean(totals_form)) if totals_form else 0.0,
        }

        with self._llm_stats_lock:
            self._last_step_llm_stats = {
                "dense_llm_calls": int(self._llm_calls_this_step),
                "dense_llm_success": int(self._llm_success_this_step),
                "dense_llm_errors": int(self._llm_errors_this_step),
            }

        return reward_tensor

    # ---------------------------
    # Dense reward core
    # ---------------------------
    def _score_single_item(self, data_item) -> Tuple[float, int, Dict[str, float]]:
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        response_ids = data_item.batch["responses"]
        valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum().item())
        valid_response_ids = response_ids[:valid_response_length]
        response_text = self.tokenizer.decode(valid_response_ids)

        sample = self._build_sample_dict(data_item)
        n = int(sample["composed_query_num"])
        if n < 2:
            z = {
                "total_reward": 0.0,
                "prog_reward": 0.0,
                "comp_reward": 0.0,
                "neg_reward": 0.0,
                "cons_reward": 0.0,
                "form_reward": 0.0,
            }
            return 0.0, valid_response_length, z

        model_answers = self.extract_math_answers(response_text)
        reasonings = self.split_into_subproblems(response_text, n)

        r_prog = self.compute_progress_reward(
            model_answers=model_answers,
            true_answers=sample["answers"],
            reasonings=reasonings,
            difficulty=sample["difficulty"],
        )
        r_comp = self.compute_complementarity_reward(
            reasonings=reasonings,
            dependencies=sample["dependencies"],
            model_answers=model_answers,
        )
        r_neg = self.compute_negativity_reward(
            full_reasoning=response_text,
            question=sample["question"],
            true_answers=sample["answers"],
        )
        r_cons = self.compute_consistency_reward(
            reasonings=reasonings,
            model_answers=model_answers,
            true_answers=sample["answers"],
        )
        r_form = self.compute_format_reward(response=response_text, n=n)

        total_reward = (
            self.reward_weights["prog"] * r_prog
            + self.reward_weights["comp"] * r_comp
            + self.reward_weights["neg"] * r_neg
            + self.reward_weights["cons"] * r_cons
            + self.reward_weights["form"] * r_form
        )
        total_reward = max(0.0, min(1.0, float(total_reward)))
        comps = {
            "total_reward": total_reward,
            "prog_reward": float(r_prog),
            "comp_reward": float(r_comp),
            "neg_reward": float(r_neg),
            "cons_reward": float(r_cons),
            "form_reward": float(r_form),
        }
        return total_reward, valid_response_length, comps

    # ---------------------------
    # Sample parsing
    # ---------------------------
    def _build_sample_dict(self, data_item) -> Dict[str, Any]:
        non_tensor = data_item.non_tensor_batch
        reward_model = non_tensor.get("reward_model", {}) or {}
        extra_info = non_tensor.get("extra_info", {}) or {}
        prompt_raw = non_tensor.get("prompt", "")

        question = self._prompt_to_question(prompt_raw)

        gt = reward_model.get("ground_truth", None)
        answers = self._normalize_answers(gt)
        n = self._infer_composed_query_num(non_tensor, reward_model, extra_info, answers)

        dependencies = extra_info.get("dependencies", [])
        if not isinstance(dependencies, list):
            dependencies = []

        difficulty = extra_info.get("difficulty", None)
        if difficulty not in ("easy", "medium", "hard"):
            data_source = str(non_tensor.get("data_source", "")).lower()
            if "amc" in data_source:
                difficulty = "easy"
            elif "math500" in data_source or "math-500" in data_source:
                difficulty = "medium"
            else:
                difficulty = "hard"

        return {
            "question": question,
            "answers": answers,
            "dependencies": dependencies,
            "composed_query_num": n,
            "difficulty": difficulty,
        }

    @staticmethod
    def _normalize_answers(gt: Any) -> List[str]:
        if isinstance(gt, list):
            return [str(x) for x in gt]
        if isinstance(gt, tuple):
            return [str(x) for x in gt]
        return [str(gt)] if gt is not None else []

    @staticmethod
    def _infer_composed_query_num(non_tensor: Dict[str, Any], reward_model: Dict[str, Any], extra_info: Dict[str, Any],
                                  answers: List[str]) -> int:
        for src in (non_tensor, reward_model, extra_info):
            if not isinstance(src, dict):
                continue
            val = src.get("composed_query_num", None)
            if isinstance(val, int):
                return val
            if isinstance(val, str) and val.isdigit():
                return int(val)
        if isinstance(extra_info, dict) and "num_problems" in extra_info:
            v = extra_info.get("num_problems")
            if isinstance(v, int):
                return v
            if isinstance(v, str) and v.isdigit():
                return int(v)
        if len(answers) >= 2:
            return len(answers)
        return 1

    @staticmethod
    def _prompt_to_question(prompt_raw: Any) -> str:
        if isinstance(prompt_raw, str):
            return prompt_raw
        if isinstance(prompt_raw, list):
            # chat-format list: [{"role": "...", "content": "..."}]
            chunks = []
            for x in prompt_raw:
                if isinstance(x, dict):
                    chunks.append(str(x.get("content", "")))
                else:
                    chunks.append(str(x))
            return "\n".join(chunks).strip()
        return str(prompt_raw)

    # ---------------------------
    # Helpers
    # ---------------------------
    @staticmethod
    def extract_math_answers(response: str) -> List[str]:
        boxed = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", response)
        if boxed:
            return [x.strip() for x in boxed]
        # fallback: lines like "Problem i: xxx" or "Answer i: xxx"
        matches = re.findall(r"(?:Problem|Answer)\s*\d+\s*:\s*([^\n]+)", response, flags=re.IGNORECASE)
        if matches:
            return [m.strip() for m in matches]
        # Chinese / English final-answer lines (common when models omit \\boxed)
        for pat in (
            r"(?:最终答案|答案)\s*\d*\s*[：:]\s*([^\n]+)",
            r"Final\s*Answer\s*\d*\s*[：:]\s*([^\n]+)",
        ):
            matches = re.findall(pat, response, flags=re.IGNORECASE)
            if matches:
                return [m.strip() for m in matches]
        return []

    @staticmethod
    def split_into_subproblems(response: str, n: int) -> List[str]:
        # Primary split by explicit "Problem i" / Chinese "问题 i" markers.
        parts = re.split(r"(?=Problem\s*\d+\s*:)|(?=(?:问题)\s*\d+\s*[：:])", response, flags=re.IGNORECASE)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) >= n:
            return parts[:n]

        # Fallback: split by boxed answers as rough boundaries.
        spans = list(re.finditer(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", response))
        if spans and len(spans) >= n:
            chunks = []
            st = 0
            for i in range(n):
                ed = spans[i].end()
                chunks.append(response[st:ed].strip())
                st = ed
            return chunks

        # Last fallback: equal segmentation by length.
        if n <= 1:
            return [response]
        seg = max(1, len(response) // n)
        out = [response[i * seg:(i + 1) * seg].strip() for i in range(n - 1)]
        out.append(response[(n - 1) * seg:].strip())
        return out

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    @staticmethod
    def _norm(s: str) -> str:
        return re.sub(r"\s+", "", str(s)).lower()

    @staticmethod
    def _latex_light_clean(s: str) -> str:
        s = str(s).strip()
        s = re.sub(r"\$+", "", s)
        s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
        s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
        s = re.sub(r"\\left|\\right|\\\[|\\\]", "", s)
        return s.strip()

    @staticmethod
    def _parse_numeric(val: str) -> Optional[float]:
        if not val:
            return None
        v = val.strip().replace(",", "").replace(" ", "")
        if not v:
            return None
        frac_m = re.fullmatch(r"(-?\d+)/(\d+)", v)
        if frac_m:
            try:
                return float(Fraction(int(frac_m.group(1)), int(frac_m.group(2))))
            except (ValueError, ZeroDivisionError):
                return None
        try:
            return float(v)
        except ValueError:
            return None

    def _parse_numeric_from_math(self, s: str) -> Optional[float]:
        m = re.search(r"\\frac\{([^}]+)\}\{([^}]+)\}", s)
        if m:
            inner = f"{m.group(1).strip()}/{m.group(2).strip()}"
            return self._parse_numeric(inner)
        return self._parse_numeric(s)

    def _normalize_math_answer(self, raw: str) -> str:
        return self._norm(self._latex_light_clean(raw))

    def _answer_match_score(self, pred: str, gold: str) -> float:
        """1.0 exact (after cleanup); numeric equivalence; optional partial credit."""
        if not gold:
            return 0.0
        if not pred:
            return 0.0
        g = self._normalize_math_answer(gold)
        p = self._normalize_math_answer(pred)
        if not g:
            return 0.0
        if p == g:
            return 1.0
        if not self.progress_partial_credit:
            return 0.0
        gn = self._parse_numeric_from_math(g)
        pn = self._parse_numeric_from_math(p)
        if gn is not None and pn is not None:
            tol = 1e-5 * max(1.0, abs(gn))
            if abs(pn - gn) <= tol:
                return 1.0
            rel = abs(pn - gn) / max(abs(gn), 1e-8)
            if rel < 0.03:
                return 0.75
            if rel < 0.10:
                return 0.45
            if rel < 0.25:
                return 0.20
        if len(g) <= 16 and (g in p or p in g):
            return 0.35
        return 0.0

    def _call_llm_api(self, prompt: str, default_score: float = 0.0) -> float:
        if not self.enable_llm_judge:
            return float(default_score)
        if not self.llm_api_key:
            return float(default_score)

        with self._llm_stats_lock:
            self._llm_calls_this_step += 1

        payload = {
            "model": self.llm_model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.llm_temperature,
            "max_tokens": self.llm_max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.llm_api_key}",
            "Content-Type": "application/json",
        }
        try:
            r = requests.post(self.llm_base_url, json=payload, headers=headers, timeout=self.llm_timeout)
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            m = re.search(r"-?\d+(?:\.\d+)?", text)
            with self._llm_stats_lock:
                if m:
                    self._llm_success_this_step += 1
                else:
                    self._llm_errors_this_step += 1
            if m:
                return float(m.group(0))
            return float(default_score)
        except Exception:
            with self._llm_stats_lock:
                self._llm_errors_this_step += 1
            return float(default_score)

    # ---------------------------
    # Reward terms
    # ---------------------------
    def compute_progress_reward(self, model_answers: List[str], true_answers: List[str], reasonings: List[str],
                                difficulty: str) -> float:
        n = max(1, len(true_answers))
        avg_tokens = float(self.avg_tokens_per_problem.get(difficulty, self.avg_tokens_per_problem["medium"]))
        total = 0.0
        for i in range(n):
            pred = model_answers[i] if i < len(model_answers) else ""
            gold = true_answers[i]
            prog = self._answer_match_score(pred, gold)
            cur_reason = reasonings[i] if i < len(reasonings) else ""
            len_i = self._count_tokens(cur_reason)
            if len_i <= avg_tokens:
                eff = 1.0
            else:
                eff = math.exp(-self.eff_lambda * (len_i / avg_tokens - 1.0))
            total += prog * eff
        return total / n

    def compute_complementarity_reward(self, reasonings: List[str], dependencies: List[Any], model_answers: List[str]) -> float:
        n = len(reasonings)
        if n < 2:
            return 1.0
        total = 0.0
        for i in range(1, n):
            prev_answer = model_answers[i - 1] if i - 1 < len(model_answers) else ""
            dep_var, dep_func = "", ""
            if i - 1 < len(dependencies):
                dep = dependencies[i - 1]
                if isinstance(dep, (list, tuple)) and len(dep) >= 2:
                    dep_var, dep_func = str(dep[0]), str(dep[1])
            dep_desc = f"{dep_var} = {dep_func}" if dep_var or dep_func else "顺序依赖前一问答案"
            prompt = f"""
请检查以下推理是否正确使用了依赖关系。
依赖要求：{dep_desc}
上一问模型答案：{prev_answer}
当前推理：
{reasonings[i]}

请仅输出分数：1 / 0.5 / 0
"""
            score = self._call_llm_api(prompt, default_score=0.5)
            total += max(0.0, min(1.0, score))
        return total / (n - 1)

    def compute_negativity_reward(self, full_reasoning: str, question: str, true_answers: List[str]) -> float:
        prompt = f"""
请分析以下数学推理过程中的错误和自我修复行为。
问题：{question}
标准答案：{true_answers}
推理：{full_reasoning}

按照以下范围给出总分：[-0.2, 0.2]
仅输出一个数字。
"""
        score = self._call_llm_api(prompt, default_score=0.0)
        return max(self.neg_reward_min, min(self.neg_reward_max, score))

    def compute_consistency_reward(self, reasonings: List[str], model_answers: List[str], true_answers: List[str]) -> float:
        n = max(1, len(true_answers))
        total = 0.0
        for i in range(n):
            if i >= len(model_answers):
                continue
            prompt = f"""
评估以下推理与答案一致性。
子问题{i+1}推理：
{reasonings[i] if i < len(reasonings) else ''}
模型答案：{model_answers[i]}
标准答案：{true_answers[i]}

仅输出分数：1 / 0.5 / 0 / -0.5
"""
            score = self._call_llm_api(prompt, default_score=0.0)
            total += max(-0.5, min(1.0, score))
        return total / n

    def compute_format_reward(self, response: str, n: int) -> float:
        format_score = 0.0
        if "### Final Answers" in response and all(f"Problem {i+1}:" in response for i in range(n)):
            format_score = 0.4
        elif len(re.findall(r"\\boxed\{.*?\}", response)) >= n:
            format_score = 0.2
        elif n >= 2 and all(
            (f"问题{i + 1}" in response) or (f"问题 {i + 1}" in response) for i in range(n)
        ):
            format_score = 0.18

        answer_count = len(re.findall(r"\\boxed\{.*?\}", response))
        complete_score = 0.3 * (min(answer_count, n) / max(1, n))

        prompt = f"""
请评估推理过程清晰度：
{response}
仅输出一个数字：0 / 0.15 / 0.3
"""
        clear_score = self._call_llm_api(prompt, default_score=0.15)
        clear_score = max(0.0, min(0.3, clear_score))
        return 0.4 * format_score + 0.3 * complete_score + 0.3 * clear_score

