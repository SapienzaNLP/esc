import argparse
import time
from typing import NamedTuple, List, Optional, Tuple
from tqdm import tqdm

import numpy as np

import torch
from torch.utils.data import DataLoader

from esc.utils.definitions_tokenizer import get_tokenizer
from esc.utils.wordnet import synset_from_offset
from esc.utils.wsd import WSDInstance
from esc.esc_dataset import WordNetDataset, OxfordDictionaryDataset
from esc.esc_pl_module import ESCModule
from esc.utils.commons import list_elems_in_dir


class InstancePredictionReport(NamedTuple):
    sequence: List[int]
    possible_synsets: List[str]
    predicted_synsets: List[str]
    gold_synsets: Optional[List[str]]
    predicted_synsets_indices: List[Tuple[int, int]]
    gold_synsets_indices: Optional[List[Tuple[int, int]]]
    possible_synsets_indices: List[Tuple[int, int]]
    most_probable_start_index: int
    most_probable_end_index: int
    predicted_start_indices_logits: torch.FloatTensor
    predicted_end_indices_logits: torch.FloatTensor
    wsd_instance: Optional[WSDInstance] = None


class ScoresReport(NamedTuple):
    precision: float
    recall: float
    f1: float


class PredictionReport(NamedTuple):
    instances_prediction_reports: List[InstancePredictionReport]
    scores_report: ScoresReport


def probabilistic_prediction(
    start_logits: torch.FloatTensor,
    end_logits: torch.FloatTensor,
    glosses_indices: List[Tuple[int, int]],
    possible_offsets: List[str],
    probabilistic_type: str,
) -> List[str]:
    start_logits_lp = torch.log_softmax(start_logits, dim=0).squeeze()
    end_logits_lp = torch.log_softmax(end_logits, dim=0).squeeze()

    if probabilistic_type == "probabilistic":
        glosses_probs = [start_logits_lp[si] + end_logits_lp[ei] for si, ei in glosses_indices]
    elif probabilistic_type == "start":
        glosses_probs = [start_logits_lp[si] for si, _ in glosses_indices]
    elif probabilistic_type == "end":
        glosses_probs = [end_logits_lp[ei] for _, ei in glosses_indices]
    elif probabilistic_type == "max":
        glosses_probs = [max(start_logits_lp[si], end_logits_lp[ei]) for si, ei in glosses_indices]
    elif probabilistic_type == "split":
        glosses_start_probs = [start_logits_lp[si] for si, _ in glosses_indices]
        glosses_end_probs = [end_logits_lp[ei] for _, ei in glosses_indices]
        start_label_idx = np.argmax(glosses_start_probs).item()
        end_label_idx = np.argmax(glosses_end_probs).item()
        return list({possible_offsets[start_label_idx], possible_offsets[end_label_idx]})
    else:
        print(f"No matching prediction methods for {probabilistic_type}")
        raise NotImplementedError

    label_idx = torch.argmax(torch.tensor(glosses_probs)).item()

    return [possible_offsets[label_idx]]


def precision_recall_f1_accuracy_score(y_true: List[List[str]], y_pred: List[Optional[List[str]]]) -> ScoresReport:

    ok = notok = 0

    if len(y_true) != len(y_pred):
        print(
            "The number of predictions and the number of gold labels is not equal"
            f"predictions = {len(y_pred)}, gold labels = {len(y_true)}"
        )
        raise ValueError

    for yt, yp in zip(y_true, y_pred):

        if yp is None:
            continue

        local_ok = local_notok = 0
        for answer in yp:
            if answer in yt:
                local_ok += 1
            else:
                local_notok += 1

        ok += local_ok / len(yp)
        notok += local_notok / len(yp)

    precision = ok / (ok + notok) * 100
    recall = ok / len(y_true) * 100
    f1 = (2 * precision * recall) / (precision + recall)

    return ScoresReport(precision, recall, f1)


def predict(
    model: ESCModule, data_loader: DataLoader, device: int, prediction_type: str, evaluate: bool = False
) -> PredictionReport:

    instance_prediction_reports = []

    pbar = tqdm(total=len(data_loader.dataset))

    with torch.no_grad(), torch.cuda.amp.autocast():

        for batch in data_loader:

            if device >= 0:
                batch["sequences"] = model.transfer_batch_to_device(batch["sequences"], model.device)
                batch["attention_masks"] = model.transfer_batch_to_device(batch["attention_masks"], model.device)

            predictions = model(batch["sequences"], batch["attention_masks"])

            for i, seq in enumerate(batch["sequences"]):
                seq = seq.cpu().numpy().tolist()

                si = predictions["start_predictions"][i]
                ei = predictions["end_predictions"][i]
                start_logits_i = predictions["start_logits"][i]
                end_logits_i = predictions["end_logits"][i]

                if "xlnet" in model.hparams.transformer_model or model.hparams.squad_head:
                    ei = ei[0]
                    end_logits_i = end_logits_i.T[0]

                possible_offsets = batch["possible_offsets"][i]
                glosses_indices = batch["gloss_positions"][i]
                wsd_instance = None if "wsd_instances" not in batch else batch["wsd_instances"][i]

                predicted_offsets = probabilistic_prediction(
                    start_logits_i, end_logits_i, glosses_indices, possible_offsets, prediction_type
                )

                predicted_offsets_indices = [glosses_indices[possible_offsets.index(po)] for po in predicted_offsets]

                gold_labels, gold_labels_indices = None, None
                if evaluate:
                    gold_labels = batch["gold_labels"][i]
                    gold_labels_indices = [glosses_indices[possible_offsets.index(gl)] for gl in gold_labels]

                instance_prediction_report = InstancePredictionReport(
                    sequence=seq,
                    possible_synsets=possible_offsets,
                    predicted_synsets=predicted_offsets,
                    gold_synsets=gold_labels,
                    predicted_synsets_indices=predicted_offsets_indices,
                    gold_synsets_indices=gold_labels_indices,
                    possible_synsets_indices=glosses_indices,
                    most_probable_start_index=si,
                    most_probable_end_index=ei,
                    predicted_start_indices_logits=start_logits_i,
                    predicted_end_indices_logits=end_logits_i,
                    wsd_instance=wsd_instance,
                )

                instance_prediction_reports.append(instance_prediction_report)

            pbar.update(len(batch["sequences"]))

    if evaluate:
        scores_report = precision_recall_f1_accuracy_score(
            y_true=[ier.gold_synsets for ier in instance_prediction_reports],
            y_pred=[ier.predicted_synsets for ier in instance_prediction_reports],
        )
    else:
        scores_report = None

    return PredictionReport(instances_prediction_reports=instance_prediction_reports, scores_report=scores_report)


def process_prediction_result(
    model_path: str,
    dataset_path: str,
    prediction_type: str,
    prediction_report: PredictionReport,
    predictions_output_path: Optional[str] = None,
    errors_output_path: Optional[str] = None,
    tokenizer: Optional = None,
) -> None:

    scores_report = prediction_report.scores_report

    if scores_report is not None:
        print("*" * 50 + " Evaluation Report " + "*" * 50)
        print(f"- Model: {model_path}")
        print(f"- Dataset: {dataset_path}")
        print(f"- Prediction Type: {prediction_type}")
        print(
            "- Metrics: F1: {:.2f} | Recall: {:.2f} | Precision: {:.2f}".format(
                scores_report.f1, scores_report.recall, scores_report.precision
            )
        )

    if predictions_output_path is not None:

        with open(predictions_output_path, "w") as f:

            for instance_eval_report in prediction_report.instances_prediction_reports:
                wsd_instance = instance_eval_report.wsd_instance
                predicted_sense_keys = set()

                for pred_synset in instance_eval_report.predicted_synsets:
                    pred_synset = synset_from_offset(pred_synset)
                    pred_synset_lemmas = pred_synset.lemmas()
                    if len(pred_synset_lemmas) == 1:
                        sense_key = pred_synset_lemmas[0].key()
                    else:
                        valid_lemmas = [
                            l for l in pred_synset.lemmas() if l.name() == wsd_instance.annotated_token.lemma
                        ]

                        if len(valid_lemmas) == 0:
                            valid_lemmas = [
                                _l
                                for _l in pred_synset.lemmas()
                                if _l.name().lower() == wsd_instance.annotated_token.lemma
                            ]

                        if not len(valid_lemmas) == 1:
                            continue

                        sense_key = valid_lemmas[0].key()

                    predicted_sense_keys.add(sense_key)

                f.write(f'{wsd_instance.instance_id} {" ".join(predicted_sense_keys)}\n')

    if errors_output_path is not None:

        with open(errors_output_path, "w") as f:

            for instance_eval_report in prediction_report.instances_prediction_reports:

                if instance_eval_report.gold_synsets == instance_eval_report.predicted_synsets:
                    continue

                f.write("- Sequence: {}\n".format(tokenizer.decode(instance_eval_report.sequence)))
                f.write(
                    "- Gold: {}\n".format(
                        " | ".join(
                            [
                                "({}, {})".format(gs, synset_from_offset(gs).definition())
                                for gs in instance_eval_report.gold_synsets
                            ]
                        )
                    )
                )
                f.write(
                    "- Prediction: {}\n".format(
                        " | ".join(
                            [
                                "({}, {})".format(ps, synset_from_offset(ps).definition())
                                for ps in instance_eval_report.predicted_synsets
                            ]
                        )
                    )
                )
                f.write("- Possible predictions: {}\n".format(", ".join(instance_eval_report.possible_synsets)))
                f.write(
                    "- Gold indices: {} | Predicted indices: {}\n".format(
                        ", ".join([f"({si}, {ei})" for si, ei in instance_eval_report.gold_synsets_indices]),
                        ", ".join([f"({si}, {ei})" for si, ei in instance_eval_report.predicted_synsets_indices]),
                    )
                )
                f.write(
                    "- Most probable start index: {} | Most probable end index: {}\n".format(
                        instance_eval_report.most_probable_start_index, instance_eval_report.most_probable_end_index
                    )
                )

                start_indices_probabilities = torch.softmax(instance_eval_report.predicted_start_indices_logits, dim=-1)
                end_indices_probabilities = torch.softmax(instance_eval_report.predicted_end_indices_logits, dim=-1)
                glosses_indices_probabilities = [
                    "[ {} : {:.2f}%, {} : {:.2f}%]".format(
                        si, start_indices_probabilities[si] * 100, ei, end_indices_probabilities[ei] * 100
                    )
                    for si, ei in instance_eval_report.possible_synsets_indices
                ]

                f.write("- Glosses indices probabilities: {}\n".format(" | ".join(glosses_indices_probabilities)))

                f.write("\n\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # required
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--dataset-paths", type=str, required=True, nargs="+")
    parser.add_argument("--prediction-types", type=str, required=True, nargs="+")
    parser.add_argument("--evaluate", action="store_true", default=False)
    # default + not required
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--tokens-per-batch", type=int, default=4000)
    parser.add_argument("--output-errors", action="store_true", default=False)
    parser.add_argument("--oxford-test", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:

    import ntpath

    args = parse_args()

    known_raganato_paths = list_elems_in_dir("data/WSD_Evaluation_Framework/Evaluation_Datasets", only_dirs=True)

    if "all_datasets" in args.dataset_paths:
        dataset_paths = known_raganato_paths
    else:
        dataset_paths = args.dataset_paths

    wsd_model = ESCModule.load_from_checkpoint(args.ckpt)
    wsd_model.freeze()

    if args.device >= 0:
        wsd_model.to(torch.device(args.device))

    tokenizer = get_tokenizer(
        wsd_model.hparams.transformer_model, getattr(wsd_model.hparams, "use_special_tokens", False)
    )

    for prediction_type in args.prediction_types:

        for dataset_path in dataset_paths:

            dataset_path = dataset_path.replace(".data.xml", "").replace(".gold.key.txt", "")

            if dataset_path in known_raganato_paths:
                dataset_path = f"data/WSD_Evaluation_Framework/Evaluation_Datasets/{dataset_path}/{dataset_path}"

            dataset_tic = time.time()
            if args.oxford_test:
                dataset = OxfordDictionaryDataset(
                    dataset_path, tokenizer, args.tokens_per_batch, re_init_on_iter=False, is_test=True
                )
            else:
                dataset = WordNetDataset(
                    dataset_path, tokenizer, args.tokens_per_batch, re_init_on_iter=False, is_test=True
                )
            dataset_toc = time.time()

            data_loader = DataLoader(dataset, batch_size=None, num_workers=0)

            prediction_type = prediction_type.split("_")[-1]

            evaluation_tic = time.time()
            prediction_report = predict(
                model=wsd_model,
                data_loader=data_loader,
                device=args.device,
                prediction_type=prediction_type,
                evaluate=args.evaluate,
            )
            evaluation_toc = time.time()

            postprocessing_tic = time.time()
            predictions_path = "predictions/{}_predictions.txt".format(ntpath.basename(dataset_path))
            output_errors_path = (
                "models_errors/{}_{}.txt".format(dataset_path.split("/")[-1], prediction_type)
                if args.output_errors
                else None
            )

            process_prediction_result(
                args.ckpt,
                dataset_path,
                prediction_type,
                prediction_report,
                predictions_path,
                output_errors_path,
                tokenizer=dataset.tokenizer,
            )
            postprocessing_toc = time.time()

            print(
                "- Prediction time: Dataset initialization took: {:.2f}s"
                " | Prediction took: {:.2f}s"
                " | Post processing took: {:.2f}s".format(
                    dataset_toc - dataset_tic, evaluation_toc - evaluation_tic, postprocessing_toc - postprocessing_tic
                )
            )

        print()


if __name__ == "__main__":
    # transformers logging shut up!
    import logging

    logging.getLogger("transformers").setLevel(logging.ERROR)

    main()
