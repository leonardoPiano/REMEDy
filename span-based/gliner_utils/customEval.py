"""Custom evaluation callback for GLiNER training."""
import torch
from transformers import TrainerCallback
from .gliner_evaluator import Evaluator


def custom_evaluation(global_step, model, eval_dataloader):
    model.eval()
    all_predictions, all_trues = [], []

    for batch in eval_dataloader:
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(model.device)

        model_output = model(**batch)[0]
        if not isinstance(model_output, torch.Tensor):
            model_output = torch.from_numpy(model_output)

        decoded = model.decoder.decode(
            batch["tokens"],
            batch["id_to_classes"],
            model_output,
            flat_ner=False,
            threshold=0.90,
            multi_label=False,
        )
        all_predictions.extend(decoded)
        all_trues.extend(batch["entities"])

    _, f1 = Evaluator(all_trues, all_predictions).evaluate()
    return f1


class CustomEvalDataloaderCallback(TrainerCallback):
    """Trainer callback that runs a custom evaluation function at each logging step."""

    def __init__(self, custom_eval_dataloader, custom_eval_function):
        self.custom_eval_dataloader = custom_eval_dataloader
        self.custom_eval_function = custom_eval_function

    def on_log(self, args, state, control, logs=None, **kwargs):
        f1 = self.custom_eval_function(state.global_step, kwargs["model"], self.custom_eval_dataloader)
        if logs is not None:
            logs["eval_F1"] = f1
        state.log_history.append(logs or {})
