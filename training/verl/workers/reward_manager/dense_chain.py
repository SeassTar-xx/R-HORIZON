import math
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import requests
import torch

from verl import DataProto


class DenseChainRewardManager:
    """
    Dense reward manager for chained math reasoning.

    The manager keeps the public verl reward-manager interface unchanged while
    making the reward terms robust to the R-HORIZON data variants found in this
    repo: ground_truth, target, group_targets, num_problems, composed_query_num,
    comma-joined targets, and nested boxed LaTeX answers.
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
        self.reward_weights = {**self.DEFAULT_REWARD_WEIGHTS, **dict(cfg.get("reward_weights", {}))}
        self.eff_lambda = float(cfg.get("efficiency_penalty_lambda", 0.05))
        self.avg_tokens_per_problem = {**self.DEFAULT_AVG_TOKENS_PER_PROBLEM,
                                       **dict(cfg.get("avg_tokens_per_problem", {}))}
        self.neg_reward_min = float(cfg.get("neg_reward_min", -0.2))
        self.neg_reward_max = float(cfg.get("neg_reward_max", 0.2))
        self.progress_partial_credit = bool(cfg.get("progress_partial_credit", True))

        self.enable_llm_judge = bool(cfg.get("enable_llm_judge", True))
        self.max_workers = int(cfg.get("max_workers", 3))
        self.llm_base_url = cfg.get("base_url", "https://api.openai.com/v1/chat/completions")
        self.llm_model_name = cfg.get("model_name", "gpt-4o-mini")
        key = str(cfg.get("api_key") or "").strip()
        self.llm_api_key = (
            key
            or os.environ.get("LLM_JUDGE_API_KEY", "").strip()
            or os.environ.get("DEEPSEEK_API_KEY", "").strip()
        )
        self.llm_temperature = float(cfg.get("temperature", 0.0))
        self.llm_max_tokens = int(cfg.get("max_tokens", 16000))
        self.llm_timeout = int(cfg.get("timeout", 30))

        self._llm_stats_lock = threading.Lock()
        self._llm_calls_this_step = 0
        self._llm_success_this_step = 0
        self._llm_errors_this_step = 0
        self._last_step_dense_means: Dict[str, float] = {}
        self._last_step_llm_stats: Dict[str, int] = {}

    def __call__(self, data: DataProto):
        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]

        with self._llm_stats_lock:
            self._llm_calls_this_step = 0
            self._llm_success_this_step = 0
            self._llm_errors_this_step = 0

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        totals = {
            "total_reward": [],
            "prog_reward": [],
            "comp_reward": [],
            "neg_reward": [],
            "cons_reward": [],
            "form_reward": [],
        }

        jobs = []
        with ThreadPoolExecutor(max_workers=max(1, self.max_workers)) as ex:
            for i in range(len(data)):
                jobs.append((i, ex.submit(self._score_single_item, data[i])))

            for i, fut in jobs:
                score, valid_response_length, comps = fut.result()
                if valid_response_length > 0:
                    reward_tensor[i, valid_response_length - 1] = float(score)
                for key in totals:
                    totals[key].append(float(comps.get(key, 0.0)))

        self._last_step_dense_means = {
            key: float(mean(values)) if values else 0.0 for key, values in totals.items()
        }
        with self._llm_stats_lock:
            self._last_step_llm_stats = {
                "dense_llm_calls": int(self._llm_calls_this_step),
                "dense_llm_success": int(self._llm_success_this_step),
                "dense_llm_errors": int(self._llm_errors_this_step),
            }
        return reward_tensor

    def _score_single_item(self, data_item) -> Tuple[float, int, Dict[str, float]]:
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        response_ids = data_item.batch["responses"]
        valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum().item())
        valid_response_ids = response_ids[:valid_response_length]
        response_text = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

        sample = self._build_sample_dict(data_item)
        n = int(sample["composed_query_num"])
        if n < 2:
            return 0.0, valid_response_length, self._zero_components()

        model_answers = self.extract_math_answers(response_text)
        reasonings = self.split_into_subproblems(response_text, n)

        r_prog = self.compute_progress_reward(model_answers, sample["answers"], reasonings, sample["difficulty"])
        r_comp = self.compute_complementarity_reward(reasonings, sample["dependencies"], model_answers)
        r_neg = self.compute_negativity_reward(response_text, sample["question"], sample["answers"])
        r_cons = self.compute_consistency_reward(reasonings, model_answers, sample["answers"])
        r_form = self.compute_format_reward(response_text, n)

        total_reward = (
            self.reward_weights["prog"] * r_prog
            + self.reward_weights["comp"] * r_comp
            + self.reward_weights["neg"] * r_neg
            + self.reward_weights["cons"] * r_cons
            + self.reward_weights["form"] * r_form
        )
        total_reward = max(0.0, min(1.0, float(total_reward)))
        return total_reward, valid_response_length, {
            "total_reward": total_reward,
            "prog_reward": float(r_prog),
            "comp_reward": float(r_comp),
            "neg_reward": float(r_neg),
            "cons_reward": float(r_cons),
            "form_reward": float(r_form),
        }

    @staticmethod
    def _zero_components() -> Dict[str, float]:
        return {
            "total_reward": 0.0,
            "prog_reward": 0.0,
            "comp_reward": 0.0,
            "neg_reward": 0.0,
            "cons_reward": 0.0,
            "form_reward": 0.0,
        }

    def _build_sample_dict(self, data_item) -> Dict[str, Any]:
        non_tensor = {k: self._to_python(v) for k, v in data_item.non_tensor_batch.items()}
        reward_model = self._as_dict(non_tensor.get("reward_model", {}))
        extra_info = self._as_dict(non_tensor.get("extra_info", {}))

        prompt_raw = (
            non_tensor.get("prompt")
            or non_tensor.get("raw_prompt")
            or non_tensor.get("input")
            or non_tensor.get("question")
            or ""
        )
        question = self._prompt_to_question(prompt_raw)

        gt = (
            reward_model.get("ground_truth")
            if "ground_truth" in reward_model
            else reward_model.get("target")
        )
        if gt is None:
            gt = non_tensor.get("group_targets")
        if gt is None:
            gt = extra_info.get("group_targets")
        if gt is None:
            gt = non_tensor.get("target")
        answers = self._normalize_answers(gt)

        n = self._infer_composed_query_num(non_tensor, reward_model, extra_info, answers)

        dependencies = extra_info.get("dependencies", non_tensor.get("dependencies", []))
        dependencies = self._to_python(dependencies)
        if not isinstance(dependencies, list):
            dependencies = []

        difficulty = extra_info.get("difficulty") or extra_info.get("model_difficulty")
        if isinstance(difficulty, dict):
            difficulty = next(iter(difficulty.values()), None)
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
    def _to_python(value: Any) -> Any:
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        if hasattr(value, "tolist"):
            try:
                return value.tolist()
            except Exception:
                pass
        return value

    @classmethod
    def _as_dict(cls, value: Any) -> Dict[str, Any]:
        value = cls._to_python(value)
        return value if isinstance(value, dict) else {}

    @classmethod
    def _normalize_answers(cls, gt: Any) -> List[str]:
        gt = cls._to_python(gt)
        if gt is None:
            return []
        if isinstance(gt, dict):
            for key in ("answers", "answer", "target", "ground_truth", "group_targets"):
                if key in gt:
                    return cls._normalize_answers(gt[key])
            return [str(v) for v in gt.values()]
        if isinstance(gt, (list, tuple)):
            if len(gt) == 1 and isinstance(gt[0], str) and "," in gt[0]:
                return [x.strip() for x in gt[0].split(",") if x.strip()]
            answers = []
            for x in gt:
                x = cls._to_python(x)
                if isinstance(x, (list, tuple)):
                    if len(x) == 1:
                        answers.extend(cls._normalize_answers(x[0]))
                    else:
                        answers.extend(cls._normalize_answers(x))
                elif x is not None:
                    answers.append(str(x))
            return answers
        if isinstance(gt, str) and "," in gt:
            return [x.strip() for x in gt.split(",") if x.strip()]
        return [str(gt)]

    @staticmethod
    def _infer_composed_query_num(non_tensor: Dict[str, Any], reward_model: Dict[str, Any], extra_info: Dict[str, Any],
                                  answers: List[str]) -> int:
        def maybe_int(value: Any) -> Optional[int]:
            try:
                if isinstance(value, str):
                    value = value.strip()
                    if not value.isdigit():
                        return None
                out = int(value)
                return out if out > 0 else None
            except (TypeError, ValueError):
                return None

        for src in (non_tensor, reward_model, extra_info):
            if not isinstance(src, dict):
                continue
            for key in ("composed_query_num", "num_problems", "n_problems"):
                val = maybe_int(src.get(key))
                if val is not None:
                    return val
        return len(answers) if len(answers) >= 2 else 1

    @staticmethod
    def _prompt_to_question(prompt_raw: Any) -> str:
        if isinstance(prompt_raw, str):
            return prompt_raw
        if isinstance(prompt_raw, list):
            chunks = []
            for item in prompt_raw:
                if isinstance(item, dict):
                    chunks.append(str(item.get("content", "")))
                else:
                    chunks.append(str(item))
            return "\n".join(chunks).strip()
        return str(prompt_raw)

    @staticmethod
    def extract_math_answers(response: str) -> List[str]:
        answer_region = response
        final_match = re.search(r"###\s*Final\s*Answers?", response, flags=re.IGNORECASE)
        if final_match:
            answer_region = response[final_match.end():]

        boxed = DenseChainRewardManager._extract_boxed_contents(answer_region)
        if boxed:
            return [x.strip() for x in boxed]

        matches = re.findall(r"(?:Problem|Answer)\s*\d+\s*[:：]\s*([^\n]+)", answer_region, flags=re.IGNORECASE)
        if matches:
            return [m.strip() for m in matches]

        for pat in (
            r"(?:最终答案|答案|答)\s*\d*\s*[:：]\s*([^\n]+)",
            r"Final\s*Answers?\s*\d*\s*[:：]?\s*([^\n]+)",
        ):
            matches = re.findall(pat, answer_region, flags=re.IGNORECASE)
            if matches:
                return [m.strip() for m in matches]
        return []

    @staticmethod
    def _extract_boxed_contents(text: str) -> List[str]:
        answers = []
        marker = r"\boxed{"
        start = 0
        while True:
            pos = text.find(marker, start)
            if pos < 0:
                break
            i = pos + len(marker)
            depth = 1
            while i < len(text) and depth:
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                i += 1
            if depth == 0:
                answers.append(text[pos + len(marker):i - 1])
                start = i
            else:
                start = pos + len(marker)
        return answers

    @staticmethod
    def split_into_subproblems(response: str, n: int) -> List[str]:
        parts = re.split(r"(?=Problem\s*\d+\s*[:：])|(?=问题\s*\d+\s*[:：])", response, flags=re.IGNORECASE)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) >= n:
            return parts[:n]

        spans = DenseChainRewardManager._iter_boxed_spans(response)
        if spans and len(spans) >= n:
            chunks = []
            st = 0
            for i in range(n):
                ed = spans[i][1]
                chunks.append(response[st:ed].strip())
                st = ed
            return chunks

        if n <= 1:
            return [response]
        seg = max(1, len(response) // n)
        out = [response[i * seg:(i + 1) * seg].strip() for i in range(n - 1)]
        out.append(response[(n - 1) * seg:].strip())
        return out

    @staticmethod
    def _iter_boxed_spans(text: str) -> List[Tuple[int, int]]:
        spans = []
        marker = r"\boxed{"
        start = 0
        while True:
            pos = text.find(marker, start)
            if pos < 0:
                break
            i = pos + len(marker)
            depth = 1
            while i < len(text) and depth:
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                i += 1
            if depth == 0:
                spans.append((pos, i))
                start = i
            else:
                start = pos + len(marker)
        return spans

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
        boxed = DenseChainRewardManager._extract_boxed_contents(s)
        if len(boxed) == 1:
            s = boxed[0]
        s = re.sub(r"^\s*\[?answer\d+\]?\s*=\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"^['\"\[]+|['\"\]]+$", "", s)
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
            matches = re.findall(r"[-+]?\d+(?:\.\d+)?", v)
            return float(matches[-1]) if matches else None

    def _parse_numeric_from_math(self, s: str) -> Optional[float]:
        m = re.search(r"\\frac\{([^}]+)\}\{([^}]+)\}", s)
        if m:
            return self._parse_numeric(f"{m.group(1).strip()}/{m.group(2).strip()}")
        return self._parse_numeric(s)

    def _normalize_math_answer(self, raw: str) -> str:
        return self._norm(self._latex_light_clean(raw))

    def _answer_match_score(self, pred: str, gold: str) -> float:
        if not gold or not pred:
            return 0.0
        g = self._normalize_math_answer(gold)
        p = self._normalize_math_answer(pred)
        if not g:
            return 0.0
        if p == g:
            return 1.0

        gn = self._parse_numeric_from_math(g)
        pn = self._parse_numeric_from_math(p)
        if gn is not None and pn is not None:
            tol = 1e-5 * max(1.0, abs(gn))
            if abs(pn - gn) <= tol:
                return 1.0
            if self.progress_partial_credit:
                rel = abs(pn - gn) / max(abs(gn), 1e-8)
                if rel < 0.03:
                    return 0.75
                if rel < 0.10:
                    return 0.45
                if rel < 0.25:
                    return 0.20

        if self.progress_partial_credit and len(g) <= 16 and (g in p or p in g):
            return 0.35
        return 0.0

    def _call_llm_api(self, prompt: str, default_score: float = 0.0) -> float:
        if not self.enable_llm_judge or not self.llm_api_key:
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
            resp = requests.post(self.llm_base_url, json=payload, headers=headers, timeout=self.llm_timeout)
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            match = re.search(r"-?\d+(?:\.\d+)?", text)
            with self._llm_stats_lock:
                if match:
                    self._llm_success_this_step += 1
                else:
                    self._llm_errors_this_step += 1
            return float(match.group(0)) if match else float(default_score)
        except Exception:
            with self._llm_stats_lock:
                self._llm_errors_this_step += 1
            return float(default_score)

    def compute_progress_reward(self, model_answers: List[str], true_answers: List[str], reasonings: List[str],
                                difficulty: str) -> float:
        if not true_answers:
            return 0.0
        avg_tokens = float(self.avg_tokens_per_problem.get(difficulty, self.avg_tokens_per_problem["medium"]))
        total = 0.0
        for i, gold in enumerate(true_answers):
            pred = model_answers[i] if i < len(model_answers) else ""
            prog = self._answer_match_score(pred, gold)
            cur_reason = reasonings[i] if i < len(reasonings) else ""
            len_i = self._count_tokens(cur_reason)
            eff = 1.0 if len_i <= avg_tokens else math.exp(-self.eff_lambda * (len_i / avg_tokens - 1.0))
            total += prog * eff
        return total / len(true_answers)

    def compute_complementarity_reward(self, reasonings: List[str], dependencies: List[Any],
                                       model_answers: List[str]) -> float:
        n = len(reasonings)
        if n < 2:
            return 1.0
        total = 0.0
        for i in range(1, n):
            prev_answer = model_answers[i - 1] if i - 1 < len(model_answers) else ""
            dep_var, dep_func = "", ""
            if i - 1 < len(dependencies):
                dep = dependencies[i - 1]
                if isinstance(dep, dict):
                    dep_var = str(dep.get("var", dep.get("variable", "")))
                    dep_func = str(dep.get("func", dep.get("expression", "")))
                elif isinstance(dep, (list, tuple)) and len(dep) >= 2:
                    dep_var, dep_func = str(dep[0]), str(dep[1])

            current_norm = self._norm(reasonings[i])
            heuristic = 0.5
            if prev_answer and self._norm(prev_answer) in current_norm:
                heuristic = 1.0
            elif dep_var and self._norm(dep_var) in current_norm:
                heuristic = 0.75

            dep_desc = f"{dep_var} = {dep_func}" if dep_var or dep_func else "No explicit dependency metadata."
            prompt = f"""Score whether this sub-problem correctly uses the previous answer/dependency.
Dependency: {dep_desc}
Previous answer: {prev_answer}
Current reasoning:
{reasonings[i]}

Return only one number: 1 for clearly uses it, 0.5 for unclear/partial, 0 for not using it.
"""
            score = self._call_llm_api(prompt, default_score=heuristic)
            total += max(0.0, min(1.0, score))
        return total / (n - 1)

    def compute_negativity_reward(self, full_reasoning: str, question: str, true_answers: List[str]) -> float:
        lower = full_reasoning.lower()
        negative_markers = (
            "impossible", "cannot", "contradiction", "inconsistent", "no solution",
            "无法", "不能", "矛盾", "无解", "不可能",
        )
        heuristic = -0.05 if any(marker in lower for marker in negative_markers) else 0.0
        prompt = f"""Score harmful or self-contradictory reasoning.
Question: {question}
True answers: {true_answers}
Model reasoning: {full_reasoning}

Return only one number in [-0.2, 0.2]. Penalize contradictions and reward explicit correction of mistakes.
"""
        score = self._call_llm_api(prompt, default_score=heuristic)
        return max(self.neg_reward_min, min(self.neg_reward_max, score))

    def compute_consistency_reward(self, reasonings: List[str], model_answers: List[str],
                                   true_answers: List[str]) -> float:
        if not true_answers:
            return 0.0
        total = 0.0
        for i, gold in enumerate(true_answers):
            if i >= len(model_answers):
                continue
            heuristic = self._answer_match_score(model_answers[i], gold)
            prompt = f"""Score whether the reasoning supports the stated answer for problem {i + 1}.
Reasoning:
{reasonings[i] if i < len(reasonings) else ''}
Model answer: {model_answers[i]}
True answer: {gold}

Return only one number: 1 / 0.5 / 0 / -0.5.
"""
            score = self._call_llm_api(prompt, default_score=heuristic)
            total += max(-0.5, min(1.0, score))
        return total / len(true_answers)

    def compute_format_reward(self, response: str, n: int) -> float:
        format_score = 0.0
        if "### Final Answers" in response and all(f"Problem {i + 1}:" in response for i in range(n)):
            format_score = 0.4
        elif len(self._extract_boxed_contents(response)) >= n:
            format_score = 0.2
        elif n >= 2 and all((f"问题{i + 1}" in response) or (f"问题 {i + 1}" in response) for i in range(n)):
            format_score = 0.18

        answer_count = len(self._extract_boxed_contents(response))
        complete_score = 0.3 * (min(answer_count, n) / max(1, n))
        heuristic_clear = 0.3 if response.strip() else 0.0

        prompt = f"""Score the answer formatting clarity.
{response}

Return only one number: 0 / 0.15 / 0.3.
"""
        clear_score = self._call_llm_api(prompt, default_score=heuristic_clear)
        clear_score = max(0.0, min(0.3, clear_score))
        return 0.4 * format_score + 0.3 * complete_score + 0.3 * clear_score
