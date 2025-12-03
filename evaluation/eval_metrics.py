from typing import List, Tuple, Dict, Set

# Define type aliases
Span = Tuple[str, str]  # (text, label)
TokenSet = Set[str]


def tokenize(text: str) -> List[str]:
    # Simple whitespace tokenizer
    return text.lower().split()


def jaccard(tokens1: TokenSet, tokens2: TokenSet) -> float:
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    return len(intersection) / len(union) if union else 0.0


def compute_soft_metrics(
    gold_spans: List[Span],
    pred_spans: List[Span],
    malicious_labels: Set[str],
    threshold: float = 0.5
) -> Dict[str, float]:

    gold_tokenized = [(i, set(tokenize(text)), label) for i, (text, label) in enumerate(gold_spans)]
    pred_tokenized = [(j, set(tokenize(text)), label) for j, (text, label) in enumerate(pred_spans)]

    total_gold = len(gold_spans)
    total_pred = len(pred_spans)

    # --- Calculate matches for precision (pred->gold) ---
    matched_gold_for_pred = set()
    matched_pred_for_pred = set()

    soft_span_match_pred = 0
    soft_span_match_with_label_pred = 0
    soft_span_match_with_intent_pred = 0



    for j, tokens_p, label_p in pred_tokenized:
        best_i = None
        best_score = 0

        for i, tokens_g, label_g in gold_tokenized:
            score = jaccard(tokens_p, tokens_g)
            if score > best_score:
                best_score = score
                best_i = i

        if best_score >= threshold:
            matched_gold_for_pred.add(best_i)
            matched_pred_for_pred.add(j)

            _, tokens_g, label_g = gold_tokenized[best_i]

            soft_span_match_pred += 1
            if label_p == label_g:
                soft_span_match_with_label_pred += 1

         
    # --- Calculate matches for recall (gold->pred) ---
    matched_pred_for_gold = set()
    matched_gold_for_gold = set()

    soft_span_match_gold = 0
    soft_span_match_with_label_gold = 0
   
    for i, tokens_g, label_g in gold_tokenized:
        best_j = None
        best_score = 0

        for j, tokens_p, label_p in pred_tokenized:
            score = jaccard(tokens_p, tokens_g)
            if score > best_score:
                best_score = score
                best_j = j

        if best_score >= threshold:
            matched_gold_for_gold.add(i)
            matched_pred_for_gold.add(best_j)

            _, tokens_p, label_p = pred_tokenized[best_j]

            soft_span_match_gold += 1
            if label_p == label_g:
                soft_span_match_with_label_gold += 1

          

    # Precision, Recall, F1 calculations for spans
    span_precision = soft_span_match_pred / total_pred if total_pred else 0.0
    span_recall = soft_span_match_gold / total_gold if total_gold else 0.0
    span_f1 = 2 * span_precision * span_recall / (span_precision + span_recall) if (span_precision + span_recall) else 0.0

    # Precision, Recall, F1 for label-aware match
    label_precision = soft_span_match_with_label_pred / total_pred if total_pred else 0.0
    label_recall = soft_span_match_with_label_gold / total_gold if total_gold else 0.0
    label_f1 = 2 * label_precision * label_recall / (label_precision + label_recall) if (label_precision + label_recall) else 0.0

    return {
        'soft_span_match_precision': span_precision,
        'soft_span_match_recall': span_recall,
        'soft_span_match_f1': span_f1,

        'soft_span_match_with_label_precision': label_precision,
        'soft_span_match_with_label_recall': label_recall,
        'soft_span_match_with_label_f1': label_f1,
     

        'total_gold_spans': total_gold,
        'total_predicted_spans': total_pred,
        
    }


def compute_micro_average(dataset: List[Tuple[List[Span], List[Span]]], malicious_labels: Set[str], threshold: float = 0.5) -> Dict[str, float]:
    total_soft_span_match_pred = 0
    total_soft_span_match_gold = 0
    total_soft_span_match_with_label_pred = 0
    total_soft_span_match_with_label_gold = 0
   

    total_gold = 0
    total_pred = 0


    for gold_spans, pred_spans in dataset:
        metrics = compute_soft_metrics(gold_spans, pred_spans, malicious_labels, threshold)

        # Aggregate matches
        total_soft_span_match_pred += metrics['soft_span_match_precision'] * metrics['total_predicted_spans']
        total_soft_span_match_gold += metrics['soft_span_match_recall'] * metrics['total_gold_spans']

        total_soft_span_match_with_label_pred += metrics['soft_span_match_with_label_precision'] * metrics['total_predicted_spans']
        total_soft_span_match_with_label_gold += metrics['soft_span_match_with_label_recall'] * metrics['total_gold_spans']


        # Count spans
        total_gold += metrics['total_gold_spans']
        total_pred += metrics['total_predicted_spans']

      
    def f1(p, r):
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    # Compute micro-averaged metrics
    span_precision = total_soft_span_match_pred / total_pred if total_pred else 0.0
    span_recall = total_soft_span_match_gold / total_gold if total_gold else 0.0
    span_f1 = f1(span_precision, span_recall)

    label_precision = total_soft_span_match_with_label_pred / total_pred if total_pred else 0.0
    label_recall = total_soft_span_match_with_label_gold / total_gold if total_gold else 0.0
    label_f1 = f1(label_precision, label_recall)

   
  

    return {
        'micro_soft_span_match_precision': span_precision,
        'micro_soft_span_match_recall': span_recall,
        'micro_soft_span_match_f1': span_f1,

        'micro_soft_span_match_with_label_precision': label_precision,
        'micro_soft_span_match_with_label_recall': label_recall,
        'micro_soft_span_match_with_label_f1': label_f1,

       'total_gold_spans': total_gold,
        'total_predicted_spans': total_pred,
      
    }