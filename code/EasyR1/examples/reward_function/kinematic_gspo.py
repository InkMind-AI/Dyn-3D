import json
import math
import re
from typing import Any


REWARD_NAME = "kinematic_gspo"
REWARD_TYPE = "batch"

NUMBER_RE = r"[-+]?\d+(?:\.\d+)?"
OPTION_RE = re.compile(r"<\s*answer\s*>\s*([A-D])\s*</\s*answer\s*>", re.IGNORECASE | re.DOTALL)
REASONING_RE = re.compile(r"<\s*reasoning\s*>(.*?)</\s*reasoning\s*>", re.IGNORECASE | re.DOTALL)
ANSWER_LINE_RE = re.compile(r"(?:^|\n)\s*answer\s*:\s*([A-D])\b", re.IGNORECASE)
TRUTH_LINE_RE = re.compile(r"(?:^|\n)\s*truth\s*:(.*?)(?:\n|$)", re.IGNORECASE | re.DOTALL)
COT_LINE_RE = re.compile(r"(?:^|\n)\s*cot\s*:\s*(\S.*?)(?:\n|$)", re.IGNORECASE)
STRICT_TRUTH_RE = re.compile(
    rf"^\s*total_distance\s*:\s*({NUMBER_RE})\s*m\s*;\s*"
    rf"displacement_distance\s*:\s*({NUMBER_RE})\s*m\s*;\s*"
    rf"displacement_vector\s*:\s*\[\s*({NUMBER_RE})\s*,\s*({NUMBER_RE})\s*,\s*({NUMBER_RE})\s*\]\s*m\s*;\s*"
    rf"rotation_angle\s*:\s*({NUMBER_RE})\s*deg\s*;\s*"
    rf"average_speed\s*:\s*({NUMBER_RE})\s*m/s\s*$",
    re.IGNORECASE,
)
STRICT_RESPONSE_RE = re.compile(
    rf"^\s*<reasoning>\s*\n"
    rf"truth\s*:\s*total_distance\s*:\s*{NUMBER_RE}\s*m\s*;\s*"
    rf"displacement_distance\s*:\s*{NUMBER_RE}\s*m\s*;\s*"
    rf"displacement_vector\s*:\s*\[\s*{NUMBER_RE}\s*,\s*{NUMBER_RE}\s*,\s*{NUMBER_RE}\s*\]\s*m\s*;\s*"
    rf"rotation_angle\s*:\s*{NUMBER_RE}\s*deg\s*;\s*"
    rf"average_speed\s*:\s*{NUMBER_RE}\s*m/s\s*\n"
    rf"cot\s*:\s*\S[^\n]*\n"
    rf"answer\s*:\s*([A-D])\s*\n"
    rf"</reasoning>\s*\n<answer>\s*\1\s*</answer>\s*$",
    re.IGNORECASE,
)

SCALAR_FIELDS = ("total_distance", "displacement_distance", "rotation_angle", "average_speed")
FIELD_TYPE = {
    "total_distance": "total_distance",
    "displacement_distance": "displacement_range",
    "rotation_angle": "rotation_range",
    "average_speed": "average_speed",
}


def _load_gt(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        return json.loads(str(raw))
    except Exception:
        return {"ground_truth_answer": str(raw)}


def _extract_reasoning(text: str):
    match = REASONING_RE.search(text or "")
    return (match.group(1), True) if match else (None, False)


def _extract_answer(text: str):
    match = OPTION_RE.search(text or "")
    if match:
        return match.group(1).upper(), True
    match = ANSWER_LINE_RE.search(text or "")
    if match:
        return match.group(1).upper(), False
    return None, False


def _format_reward(has_reasoning: bool, has_answer_tag: bool, pred_answer):
    if pred_answer is None:
        return -1.0
    if has_reasoning and has_answer_tag:
        return 1.0
    if has_answer_tag:
        return 0.0
    return -0.5


def _strict_format_reward(text: str, reasoning: str | None, has_reasoning: bool):
    """Dense format score with a large bonus for exact fixed-five compliance."""
    outer_match = OPTION_RE.search(text or "")
    inner_match = ANSWER_LINE_RE.search(reasoning or "")
    truth_match = TRUTH_LINE_RE.search(reasoning or "")
    truth = truth_match.group(1).strip() if truth_match else ""
    strict_truth = STRICT_TRUTH_RE.fullmatch(truth)

    ordered_fields = bool(
        re.search(
            r"total_distance\s*:.*displacement_distance\s*:.*displacement_vector\s*:.*"
            r"rotation_angle\s*:.*average_speed\s*:",
            truth,
            re.IGNORECASE | re.DOTALL,
        )
    )
    units = {
        "total_m": bool(re.search(rf"total_distance\s*:\s*{NUMBER_RE}\s*m(?:\s*;|\s*$)", truth, re.IGNORECASE)),
        "displacement_m": bool(re.search(rf"displacement_distance\s*:\s*{NUMBER_RE}\s*m(?:\s*;|\s*$)", truth, re.IGNORECASE)),
        "vector_m": bool(re.search(r"displacement_vector\s*:\s*\[[^\]\n]+\]\s*m(?:\s*;|\s*$)", truth, re.IGNORECASE)),
        "rotation_deg": bool(re.search(rf"rotation_angle\s*:\s*{NUMBER_RE}\s*deg(?:\s*;|\s*$)", truth, re.IGNORECASE)),
        "speed_mps": bool(re.search(rf"average_speed\s*:\s*{NUMBER_RE}\s*m/s(?:\s*;|\s*$)", truth, re.IGNORECASE)),
    }
    outer_answer = outer_match.group(1).upper() if outer_match else None
    inner_answer = inner_match.group(1).upper() if inner_match else None
    components = [
        has_reasoning,
        outer_match is not None,
        ordered_fields,
        *units.values(),
        strict_truth is not None,
        COT_LINE_RE.search(reasoning or "") is not None,
        inner_match is not None,
        outer_answer is not None and outer_answer == inner_answer,
    ]
    dense_score = sum(float(value) for value in components) / len(components)
    exact = STRICT_RESPONSE_RE.fullmatch(text or "") is not None
    return 0.5 * dense_score + 0.5 * float(exact)


def _rank(value: float, kind: str):
    value = abs(float(value))
    if kind == "total_distance":
        if value < 0.5:
            return 0
        if value < 4.0:
            return 1
        if value < 9.0:
            return 2
        return 3
    if kind == "displacement_range":
        if value < 0.5:
            return 0
        if value < 1.5:
            return 1
        if value < 3.0:
            return 2
        return 3
    if kind == "rotation_range":
        if value < 15:
            return 0
        if value < 360:
            return 1
        if value < 600:
            return 2
        return 3
    if kind == "average_speed":
        if value < 0.15:
            return 0
        if value < 0.4:
            return 1
        if value < 0.8:
            return 2
        return 3
    return None


def _rank_reward(gt_rank, pred_rank):
    if gt_rank is None or pred_rank is None:
        return -1.0
    gap = abs(int(gt_rank) - int(pred_rank))
    if gap == 0:
        return 1.0
    if gap == 1:
        return 0.0
    return -1.0


def _field_value(text: str, field: str):
    pattern = re.compile(rf"{re.escape(field)}\s*:\s*({NUMBER_RE})", re.IGNORECASE)
    match = pattern.search(text or "")
    if not match:
        return None
    try:
        return float(match.group(1))
    except Exception:
        return None


def _vector_value(text: str):
    pattern = re.compile(
        rf"displacement_vector\s*:\s*\[\s*({NUMBER_RE})\s*,\s*({NUMBER_RE})\s*,\s*({NUMBER_RE})\s*\]",
        re.IGNORECASE,
    )
    match = pattern.search(text or "")
    if not match:
        return None
    try:
        return [float(match.group(i)) for i in range(1, 4)]
    except Exception:
        return None


def _norm(vec):
    return math.sqrt(sum(float(x) * float(x) for x in vec))


def _vector_reward(gt_vec, pred_vec):
    if not isinstance(gt_vec, list) or len(gt_vec) != 3:
        return None
    if not isinstance(pred_vec, list) or len(pred_vec) != 3:
        return -1.0
    gt_norm = _norm(gt_vec)
    pred_norm = _norm(pred_vec)
    mag_score = _rank_reward(_rank(gt_norm, "displacement_range"), _rank(pred_norm, "displacement_range"))
    if gt_norm < 1e-6 or pred_norm < 1e-6:
        direction_score = 1.0 if gt_norm < 0.5 and pred_norm < 0.5 else -1.0
    else:
        cos = sum(g * p for g, p in zip(gt_vec, pred_vec)) / (gt_norm * pred_norm)
        if cos >= 0.866:
            direction_score = 1.0
        elif cos >= 0.5:
            direction_score = 0.0
        else:
            direction_score = -1.0
    return 0.5 * mag_score + 0.5 * direction_score


def _kinematic_reward(reasoning: str | None, motion_truth: dict[str, Any], field_weights: dict[str, float]):
    if reasoning is None:
        return -1.0, {}
    rewards = {}
    for field in SCALAR_FIELDS:
        if field not in motion_truth:
            continue
        pred = _field_value(reasoning, field)
        gt_rank = _rank(float(motion_truth[field]), FIELD_TYPE[field])
        pred_rank = _rank(float(pred), FIELD_TYPE[field]) if pred is not None else None
        rewards[field] = _rank_reward(gt_rank, pred_rank)

    vec_reward = _vector_reward(motion_truth.get("displacement_vector"), _vector_value(reasoning))
    if vec_reward is not None:
        rewards["displacement_vector"] = vec_reward

    weighted = [(v, float(field_weights.get(k, 1.0))) for k, v in rewards.items() if float(field_weights.get(k, 1.0)) > 0]
    if not weighted:
        return 0.0, rewards
    return sum(v * w for v, w in weighted) / sum(w for _, w in weighted), rewards


def compute_score(
    reward_inputs: list[dict[str, Any]],
    lambda_kinematic: float = 0.1,
    format_weight: float = 0.1,
    answer_weight: float = 1.0,
    field_weights: dict[str, float] | None = None,
    strict_format: bool = False,
) -> list[dict[str, float]]:
    field_weights = field_weights or {}
    scores = []
    for item in reward_inputs:
        response = str(item.get("response") or "")
        gt = _load_gt(item.get("ground_truth"))
        reasoning, has_reasoning = _extract_reasoning(response)
        pred_answer, has_answer_tag = _extract_answer(response)
        gt_answer = str(gt.get("ground_truth_answer") or "").upper()
        answer_score = answer_weight if pred_answer == gt_answer and gt_answer else 0.0
        if strict_format:
            format_score_raw = _strict_format_reward(response, reasoning, has_reasoning)
        else:
            format_score_raw = _format_reward(has_reasoning, has_answer_tag, pred_answer)
        kin_raw, kin_fields = _kinematic_reward(reasoning, gt.get("motion_truth") or {}, field_weights)
        overall = answer_score + format_weight * format_score_raw + lambda_kinematic * kin_raw
        row = {
            "overall": overall,
            "answer": answer_score,
            "format": format_score_raw,
            "kinematic": kin_raw,
        }
        for key, value in kin_fields.items():
            row[f"kinematic_{key}"] = value
        scores.append(row)
    return scores
