"""Span-level evaluation utilities for GLiNER-based models."""
import warnings
from collections import defaultdict
from functools import partial
from typing import List, Literal, Union

import numpy as np
import torch


class UndefinedMetricWarning(UserWarning):
    pass


def _prf_divide(numerator, denominator, metric, modifier, average, warn_for, zero_division="warn"):
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.true_divide(numerator, denominator)
        result[denominator == 0] = 0.0 if zero_division in ["warn", 0] else 1.0

    if denominator == 0 and zero_division == "warn" and metric in warn_for:
        axis0 = "label" if average == "samples" else "sample"
        msg = (
            f"{metric.title()} is ill-defined and being set to 0.0 "
            f"due to no {modifier} {axis0}. "
            "Use `zero_division` parameter to control this behavior."
        )
        warnings.warn(msg, UndefinedMetricWarning, stacklevel=3)

    return result


def extract_tp_actual_correct(y_true, y_pred):
    entities_true = defaultdict(set)
    entities_pred = defaultdict(set)

    for type_name, (start, end), idx in y_true:
        entities_true[type_name].add((start, end, idx))
    for type_name, (start, end), idx in y_pred:
        entities_pred[type_name].add((start, end, idx))

    target_names = sorted(set(entities_true) | set(entities_pred))
    tp_sum = pred_sum = true_sum = np.array([], dtype=np.int32)

    for t in target_names:
        et = entities_true.get(t, set())
        ep = entities_pred.get(t, set())
        tp_sum = np.append(tp_sum, len(et & ep))
        pred_sum = np.append(pred_sum, len(ep))
        true_sum = np.append(true_sum, len(et))

    return pred_sum, tp_sum, true_sum, target_names


def flatten_for_eval(y_true, y_pred):
    all_true, all_pred = [], []
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        all_true.extend([t + [i] for t in true])
        all_pred.extend([p + [i] for p in pred])
    return all_true, all_pred


def compute_prf(y_true, y_pred, average="micro"):
    y_true, y_pred = flatten_for_eval(y_true, y_pred)
    pred_sum, tp_sum, true_sum, _ = extract_tp_actual_correct(y_true, y_pred)

    if average == "micro":
        tp_sum = np.array([tp_sum.sum()])
        pred_sum = np.array([pred_sum.sum()])
        true_sum = np.array([true_sum.sum()])

    kw = dict(average=average, warn_for=["precision", "recall", "f-score"])
    precision = _prf_divide(tp_sum, pred_sum, "precision", "predicted", **kw)
    recall = _prf_divide(tp_sum, true_sum, "recall", "true", **kw)
    denom = precision + recall
    denom[denom == 0.0] = 1
    f_score = 2 * (precision * recall) / denom

    return {"precision": precision[0], "recall": recall[0], "f_score": f_score[0]}


class Evaluator:
    def __init__(self, all_true, all_outs):
        self.all_true = all_true
        self.all_outs = all_outs

    @staticmethod
    def _get_entities_fr(ents):
        return [[lab, (s, e)] for s, e, lab in ents]

    @staticmethod
    def _get_entities_pr(ents):
        return [[lab, (s, e)] for s, e, lab, _ in ents]

    def transform_data(self):
        return (
            [self._get_entities_fr(t) for t in self.all_true],
            [self._get_entities_pr(p) for p in self.all_outs],
        )

    @torch.no_grad()
    def evaluate(self):
        all_true_typed, all_outs_typed = self.transform_data()
        precision, recall, f1 = compute_prf(all_true_typed, all_outs_typed).values()
        output_str = f"P: {precision:.2%}\tR: {recall:.2%}\tF1: {f1:.2%}\n"
        return output_str, f1


# ---------------------------------------------------------------------------
# Span overlap helpers
# ---------------------------------------------------------------------------

def is_nested(idx1, idx2):
    return (idx1[0] <= idx2[0] and idx1[1] >= idx2[1]) or (idx2[0] <= idx1[0] and idx2[1] >= idx1[1])


def has_overlapping(idx1, idx2, multi_label=False):
    if idx1[:2] == idx2[:2]:
        return not multi_label
    return not (idx1[0] > idx2[1] or idx2[0] > idx1[1])


def has_overlapping_nested(idx1, idx2, multi_label=False):
    if idx1[:2] == idx2[:2]:
        return not multi_label
    if (idx1[0] > idx2[1] or idx2[0] > idx1[1]) or is_nested(idx1, idx2):
        return False
    return True


def greedy_search(spans, flat_ner=True, multi_label=False):
    has_ov = partial(has_overlapping if flat_ner else has_overlapping_nested, multi_label=multi_label)
    new_list = []
    for b in sorted(spans, key=lambda x: -x[-1]):
        if not any(has_ov(b[:-1], existing) for existing in new_list):
            new_list.append(b)
    return sorted(new_list, key=lambda x: x[0])
