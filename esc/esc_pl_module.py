from typing import Dict, Union, List
import torch
import pytorch_lightning as pl

from transformers import AutoModelForQuestionAnswering, get_linear_schedule_with_warmup

from esc.rc_models import WSDXLNetForQuestionAnswering, SquadQAModel
from esc.utils.lamb_optimizer import Lamb
from esc.utils.optimizers import RAdam


SUPPORTED_MODELS = ["bart", "bert", "longformer", "roberta", "xlnet"]


class ESCModule(pl.LightningModule):
    def __init__(self, conf, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparams = conf
        if any([tm in self.hparams.transformer_model for tm in SUPPORTED_MODELS]):
            if self.hparams.squad_head:
                self.qa_model = SquadQAModel(self.hparams)
            elif "xlnet" in self.hparams.transformer_model:
                self.qa_model = WSDXLNetForQuestionAnswering.from_pretrained(
                    self.hparams.transformer_model, mem_len=1024
                )
            else:
                self.qa_model = AutoModelForQuestionAnswering.from_pretrained(self.hparams.transformer_model)
        else:
            raise NotImplementedError

        if getattr(self.hparams, "use_special_tokens", False):
            self.qa_model.resize_token_embeddings(self.hparams.vocab_size)

    def forward(
        self, sequences, attention_masks, start_positions=None, end_positions=None, *args, **kwargs
    ) -> Dict[str, torch.Tensor]:

        with open("data/batches.txt", "a") as f:
            f.write(f"{str(sequences.shape)}\n")

        model_input = {
            "input_ids": sequences,
            "attention_mask": attention_masks,
            "start_positions": start_positions,
            "end_positions": end_positions,
            "return_dict": True,
        }

        if "bart" not in self.hparams.transformer_model:
            model_input["token_type_ids"] = kwargs.get("token_type_ids", None)

        if self.hparams.squad_head and self.hparams.use_pmask:
            gloss_positions = kwargs.get("gloss_positions")
            p_mask = torch.ones_like(attention_masks)
            p_mask[:, 0] = 0  # so that the CLS can always be predicted for impossible predictions
            for i, sent_gloss_positions in enumerate(gloss_positions):
                for sgp, egp in sent_gloss_positions:
                    p_mask[i][sgp] = 0
                    p_mask[i][egp] = 0
            model_input["p_mask"] = p_mask

        outputs = self.qa_model(**model_input)

        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        output_dict = {
            "start_logits": start_logits,
            "end_logits": end_logits,
            "start_predictions": torch.argmax(start_logits, dim=1),
            "end_predictions": torch.argmax(end_logits, dim=1),
        }

        if start_positions is not None:
            output_dict["loss"] = outputs.loss

        return output_dict

    def training_step(self, batch, batch_idx) -> pl.TrainResult:
        forward_output = self.forward(**batch)
        result = pl.TrainResult(minimize=forward_output["loss"])
        result.log("train_loss", forward_output["loss"])
        return result

    def validation_step(self, batch, batch_idx: int, *args, **kwargs) -> pl.EvalResult:

        forward_output = self.forward(**batch)

        result = pl.EvalResult()

        result.dataset_identifier = batch["dataset_identifier"]

        result.log(f'{batch["dataset_identifier"]}_val_loss', forward_output["loss"])

        result.start_predictions = forward_output["start_predictions"]
        result.end_predictions = forward_output["end_predictions"]

        result.start_positions = batch["start_positions"]
        result.end_positions = batch["end_positions"]

        return result

    def validation_epoch_end(self, all_outputs: Union[pl.EvalResult, List[pl.EvalResult]]) -> pl.EvalResult:

        if type(all_outputs) != list:
            all_outputs = [all_outputs]

        final_output = all_outputs[0]

        for i, outputs in enumerate(all_outputs):

            correct_start_predictions = torch.eq(outputs.start_predictions, outputs.start_positions)
            correct_end_predictions = torch.eq(outputs.end_predictions, outputs.end_positions)
            predictions_len = torch.tensor(correct_start_predictions.size(0), dtype=torch.float)

            correct_predictions = torch.bitwise_and(correct_start_predictions, correct_end_predictions)
            correct_predictions = torch.sum(correct_predictions) / predictions_len

            in_bound_start_predictions = torch.bitwise_and(
                outputs.start_predictions >= outputs.start_positions, outputs.start_predictions <= outputs.end_positions
            )

            in_bound_end_predictions = torch.bitwise_and(
                outputs.end_predictions >= outputs.start_positions, outputs.end_predictions <= outputs.end_positions
            )

            in_bound_predictions = torch.bitwise_and(in_bound_start_predictions, in_bound_end_predictions)

            prefix = "_".join(list(set(outputs.dataset_identifier)))

            final_output.log(
                f"{prefix}_correct_start_predictions", torch.sum(correct_start_predictions) / predictions_len
            )
            final_output.log(f"{prefix}_correct_end_predictions", torch.sum(correct_end_predictions) / predictions_len)
            final_output.log(f"{prefix}_correct_predictions", correct_predictions)

            final_output.log(
                f"{prefix}_in_bound_start_predictions", torch.sum(in_bound_start_predictions) / predictions_len
            )
            final_output.log(
                f"{prefix}_in_bound_end_predictions", torch.sum(in_bound_end_predictions) / predictions_len
            )
            final_output.log(f"{prefix}_in_bound_predictions", torch.sum(in_bound_predictions) / predictions_len)

            val_loss = torch.mean(outputs[f"{prefix}_val_loss"])
            final_output.log(f"{prefix}_val_loss", val_loss)

            if prefix == "wsd" or len(all_outputs) == 1:
                final_output.checkpoint_on = correct_predictions
                final_output.early_stop_on = correct_predictions

        return final_output

    def get_optimizer_and_scheduler(self):

        no_decay = self.hparams.no_decay_params

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, self.hparams.learning_rate)
        elif self.hparams.optimizer == "lamb":
            optimizer = Lamb(optimizer_grouped_parameters, self.hparams.learning_rate)
        elif self.hparams.optimizer == "radam":
            optimizer = RAdam(optimizer_grouped_parameters, self.hparams.learning_rate)
            return optimizer, None
        else:
            raise NotImplementedError

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=self.hparams.num_training_steps,
        )

        return optimizer, lr_scheduler

    def configure_optimizers(self):
        optimizer, lr_scheduler = self.get_optimizer_and_scheduler()
        if lr_scheduler is None:
            return optimizer
        return [optimizer], [{"interval": "step", "scheduler": lr_scheduler}]
