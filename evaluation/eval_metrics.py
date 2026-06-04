"""Soft span-matching evaluation metrics for rationale extraction."""
from typing import List, Tuple, Dict, Set

Span = Tuple[str, str]  # (text, label)
TokenSet = Set[str]


def tokenize(text: str) -> List[str]:
    return text.lower().split()


def jaccard(tokens1: TokenSet, tokens2: TokenSet) -> float:
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    return len(intersection) / len(union) if union else 0.0


def compute_soft_metrics(
    gold_spans: List[Span],
    pred_spans: List[Span],
    malicious_labels: Set[str],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Per-example soft span matching with Jaccard similarity."""
    gold_tok = [(i, set(tokenize(t)), l) for i, (t, l) in enumerate(gold_spans)]
    pred_tok = [(j, set(tokenize(t)), l) for j, (t, l) in enumerate(pred_spans)]

    total_gold = len(gold_spans)
    total_pred = len(pred_spans)

    # Precision direction: pred → gold
    soft_match_pred = 0
    soft_match_label_pred = 0
    for j, tp, lp in pred_tok:
        best_i, best_score = None, 0.0
        for i, tg, _ in gold_tok:
            s = jaccard(tp, tg)
            if s > best_score:
                best_score, best_i = s, i
        if best_score >= threshold:
            soft_match_pred += 1
            _, _, lg = gold_tok[best_i]
            if lp == lg:
                soft_match_label_pred += 1

    # Recall direction: gold → pred
    soft_match_gold = 0
    soft_match_label_gold = 0
    for i, tg, lg in gold_tok:
        best_j, best_score = None, 0.0
        for j, tp, _ in pred_tok:
            s = jaccard(tp, tg)
            if s > best_score:
                best_score, best_j = s, j
        if best_score >= threshold:
            soft_match_gold += 1
            _, _, lp = pred_tok[best_j]
            if lp == lg:
                soft_match_label_gold += 1

    def _prf(num_p, num_r, denom_p, denom_r):
        p = num_p / denom_p if denom_p else 0.0
        r = num_r / denom_r if denom_r else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f1

    sp, sr, sf = _prf(soft_match_pred, soft_match_gold, total_pred, total_gold)
    lp, lr, lf = _prf(soft_match_label_pred, soft_match_label_gold, total_pred, total_gold)

    return {
        "soft_span_match_precision": sp,
        "soft_span_match_recall": sr,
        "soft_span_match_f1": sf,
        "soft_span_match_with_label_precision": lp,
        "soft_span_match_with_label_recall": lr,
        "soft_span_match_with_label_f1": lf,
        "total_gold_spans": total_gold,
        "total_predicted_spans": total_pred,
    }


def compute_micro_average(
    dataset: List[Tuple[List[Span], List[Span]]],
    malicious_labels: Set[str],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Micro-average soft span metrics over a full dataset."""
    agg = dict(
        pred=0, gold=0, label_pred=0, label_gold=0,
        total_gold=0, total_pred=0,
    )

    for gold_spans, pred_spans in dataset:
        m = compute_soft_metrics(gold_spans, pred_spans, malicious_labels, threshold)
        agg["pred"] += m["soft_span_match_precision"] * m["total_predicted_spans"]
        agg["gold"] += m["soft_span_match_recall"] * m["total_gold_spans"]
        agg["label_pred"] += m["soft_span_match_with_label_precision"] * m["total_predicted_spans"]
        agg["label_gold"] += m["soft_span_match_with_label_recall"] * m["total_gold_spans"]
        agg["total_gold"] += m["total_gold_spans"]
        agg["total_pred"] += m["total_predicted_spans"]

    def _f1(p, r):
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    sp = agg["pred"] / agg["total_pred"] if agg["total_pred"] else 0.0
    sr = agg["gold"] / agg["total_gold"] if agg["total_gold"] else 0.0
    lp = agg["label_pred"] / agg["total_pred"] if agg["total_pred"] else 0.0
    lr = agg["label_gold"] / agg["total_gold"] if agg["total_gold"] else 0.0

    return {
        "micro_soft_span_match_precision": sp,
        "micro_soft_span_match_recall": sr,
        "micro_soft_span_match_f1": _f1(sp, sr),
        "micro_soft_span_match_with_label_precision": lp,
        "micro_soft_span_match_with_label_recall": lr,
        "micro_soft_span_match_with_label_f1": _f1(lp, lr),
        "total_gold_spans": agg["total_gold"],
        "total_predicted_spans": agg["total_pred"],
    }
