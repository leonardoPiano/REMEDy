"""Shared output-parsing utilities for model responses."""
import re

_RATIONALE_REGEX = re.compile(r"(?P<span>.*)\s\((?P<label>.*)\)")


def parse_answer_rationale(text: str) -> tuple[str, list[str]]:
    """Parse a model response into (global_label, rationale_spans).

    Expected format::

        global_label: MALICIOUS
        rationales: span1 (label1);; span2 (label2)

    Returns
    -------
    global_label : str
        ``"MALICIOUS"``, ``"BENIGN"``, or ``"DOUBTFUL"`` (defaults to ``"Benign"`` on parse failure).
    rationales : list[str]
        De-duplicated list of extracted span strings.
    """
    global_label = "Benign"
    rationales: list[str] = []

    m = re.search(r"global_label:\s*(.*)", text)
    if m:
        global_label = m.group(1).strip()

    m = re.search(r"rationales:\s*(.*)", text)
    if m:
        spans_raw = m.group(1).strip()
        if spans_raw != "[]":
            rationales = list(set(spans_raw.split(";;")))

    return global_label, rationales


def parse_answer_rationale_structured(text: str) -> list[dict]:
    """Parse rationale portion into a list of ``{"span": ..., "label": ...}`` dicts.

    Used when the full span+label structure is needed .
    """
    if "rationales" in text:
        idx = text.index("rationales") + len("rationales")
        text = text[idx + 2:]

    output = []
    for item in (x.strip() for x in text.split(";;") if x.strip()):
        m = _RATIONALE_REGEX.match(item)
        if m:
            output.append(m.groupdict())
    return output


def label_to_binary(label: str) -> int:
    """Map a string label to a binary int (1 = malicious/harmful, 0 otherwise)."""
    return 1 if label.upper() == "MALICIOUS" else 0
