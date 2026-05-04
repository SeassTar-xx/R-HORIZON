# AIME-style accuracy for training-time benchmarks (independent of dense_chain n>=2).
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.reward_score.math import compute_score
from verl.utils.reward_score.deepscaler_math_multi_verify.utils.utils import (
    extract_k_boxed_answers,
    grade_answer_mathd,
    grade_answer_sympy,
)


def repo_root_r_horizon() -> str:
    # training/verl/trainer/aime_eval_metrics.py -> repo root is parents x3
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def default_aime_paths() -> Dict[str, str]:
    repo_root = repo_root_r_horizon()
    d = os.path.join(repo_root, "evaluation", "data")
    return {
        "aime24_n1": os.path.join(d, "R-HORIZON-AIME24", "AIME24-origin.jsonl"),
        "aime24_n2": os.path.join(d, "R-HORIZON-AIME24", "AIME24-combined-n2.jsonl"),
        "aime25_n1": os.path.join(d, "R-HORIZON-AIME25", "AIME25-origin.jsonl"),
        "aime25_n2": os.path.join(d, "R-HORIZON-AIME25", "AIME25-combined-n2.jsonl"),
    }


def jsonl_to_eval_row(obj: dict, line_index: int) -> dict:
    n = int(obj.get("num_problems", 1) or 1)
    return {
        "prompt": [{"role": "user", "content": obj["input"]}],
        "data_source": "aime-bench",
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": obj["target"]},
        "extra_info": {"index": line_index, "composed_query_num": n},
    }


def score_response_vs_target(response: str, target: Any, num_problems: int) -> float:
    if isinstance(target, (list, tuple, np.ndarray)):
        gt_list = [str(x) for x in list(target)]
    else:
        gt_list = [str(target)]

    if num_problems >= 2:
        if len(gt_list) == 1 and "," in str(gt_list[0]):
            expected = [p.strip() for p in str(gt_list[0]).split(",")]
        else:
            expected = [str(x).strip() for x in gt_list]
        k = len(expected)
        model_answers = extract_k_boxed_answers(response, k, strict_order=False)
        if len(model_answers) != k:
            return 0.0
        for ma, ex in zip(model_answers, expected):
            ok = grade_answer_mathd(ma, ex) or grade_answer_sympy(ma, ex)
            if not ok:
                return 0.0
        return 1.0

    return float(compute_score(response, gt_list))


def tokenize_rows(tokenizer, rows: List[dict], max_prompt_length: int, truncation: str = "error") -> dict:
    data_list = []
    for row in rows:
        row_dict = dict(row)
        chat = row_dict.pop("prompt")
        prompt_with_chat_template = tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, tokenize=False
        )
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_with_chat_template,
            tokenizer=tokenizer,
            max_length=max_prompt_length,
            pad_token_id=tokenizer.pad_token_id,
            left_pad=True,
            truncation=truncation,
        )
        position_ids = compute_position_id_with_mask(attention_mask)
        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index
        data_list.append(row_dict)
    return collate_fn(data_list)


def load_jsonl(path: str, limit: Optional[int] = None) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            out.append(json.loads(line))
            if limit is not None and len(out) >= limit:
                break
    return out
