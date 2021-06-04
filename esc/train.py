import os
import argparse

import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data.dataloader import DataLoader

import pytorch_lightning as pl

from esc.utils.definitions_tokenizer import get_tokenizer
from esc.esc_dataset import WordNetDataset, OxfordDictionaryDataset, DatasetAlternator
from esc.esc_pl_module import ESCModule


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()
    parser.add_argument("--transformer_model", default="facebook/bart-large")
    parser.add_argument("--tokens_per_batch", type=int, default=700)
    parser.add_argument("--squad_head", action="store_true")
    parser.add_argument("--use_pmask", action="store_true")
    parser.add_argument("--add_glosses_noise", action="store_true", default=False)
    parser.add_argument("--fix_glosses", action="store_true", default=False)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--train_path", action="append")
    parser.add_argument(
        "--validation_path", default="data/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007"
    )
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--gradient_acc_steps", default=20, type=int)
    parser.add_argument("--gradient_clipping", default=10.0, type=float)
    parser.add_argument("--num_training_steps", default=300_000, type=int)
    parser.add_argument("--num_warmup_steps", default=0, type=int)
    parser.add_argument("--optimizer", default="radam", type=str)
    parser.add_argument("--learning_rate", default=0.00001, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--no_decay_params", default=["bias", "LayerNorm.weight"], action="append")
    parser.add_argument("--precision", default=16, type=int)
    parser.add_argument("--amp_level", default="O1", type=str)
    parser.add_argument("--validation_check_interval", default=2000, type=int)
    parser.add_argument("--run_name", type=str, default="default_name")
    parser.add_argument("--save_topk", type=int, default=5)
    parser.add_argument("--wandb_project", default="esc", required=False)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--use_oxford", action="store_true", default=False)
    parser.add_argument("--infinite_dataset", action="store_true", default=False)
    parser.add_argument("--no_wsd", action="store_true", default=False)
    parser.add_argument("--kshot", type=int, default=-1)
    parser.add_argument("--use_special_tokens", action="store_true", default=False)
    parser.add_argument("--start_from_ckpt", type=str)
    parser.add_argument("--poisson_lambda", type=int, default=1)
    return parser.parse_args()


def get_module(args: argparse.Namespace, tokenizer) -> ESCModule:
    module = ESCModule(args)
    if args.start_from_ckpt:
        starting_module = ESCModule.load_from_checkpoint(args.start_from_ckpt)
        tmp_path = f'/tmp/{args.start_from_ckpt.split("/")[-1]}'
        torch.save(starting_module.state_dict(), tmp_path)
        module.load_state_dict(torch.load(tmp_path))
        os.system(f"rm -rf {tmp_path}")
    elif args.use_special_tokens:
        module.qa_model.resize_token_embeddings(len(tokenizer))
    return module


def train(args: argparse.Namespace):

    tokenizer = get_tokenizer(args.transformer_model, args.use_special_tokens)
    if args.use_special_tokens:
        args.vocab_size = len(tokenizer)

    train_datasets, validation_datasets = [], []

    # WSD
    if not args.no_wsd:
        train_datasets.append(
            WordNetDataset(
                args.train_path,
                tokenizer,
                args.tokens_per_batch,
                re_init_on_iter=True,
                add_glosses_noise=args.add_glosses_noise,
                fix_glosses=args.fix_glosses,
                kshot=args.kshot,
                poisson_lambda=args.poisson_lambda,
            )
        )

        validation_datasets.append(
            WordNetDataset(
                args.validation_path,
                tokenizer,
                args.tokens_per_batch,
                re_init_on_iter=False,
                is_test=True,
            )
        )

    # OXFORD
    if args.use_oxford:
        train_datasets.append(
            OxfordDictionaryDataset(
                "data/preprocessed_data/train.processed.txt", tokenizer, args.tokens_per_batch, re_init_on_iter=True
            )
        )
        validation_datasets.append(
            OxfordDictionaryDataset(
                "data/preprocessed_data/val.processed.txt",
                tokenizer,
                args.tokens_per_batch,
                re_init_on_iter=False,
                is_test=True,
            )
        )

    train_dataset = (
        DatasetAlternator(train_datasets, is_infinite=args.infinite_dataset)
        if len(train_datasets) > 0
        else train_datasets[0]
    )

    train_dataloader = DataLoader(train_dataset, batch_size=None, num_workers=args.num_workers)
    validation_dataloader = [
        DataLoader(vd, batch_size=None, num_workers=args.num_workers) for vd in validation_datasets
    ]

    # WANDB
    wandb_logger = WandbLogger(name=args.run_name, project=args.wandb_project)

    wandb_logger.log_hyperparams(args)

    # CALLBACKS
    early_stopping_cb = EarlyStopping(mode="max", patience=args.patience)

    model_checkpoint = ModelCheckpoint(save_top_k=args.save_topk, verbose=True, mode="max", period=0)

    # WSD MODULE
    module = get_module(args, tokenizer)

    # TRAINER
    trainer = pl.Trainer(
        gpus=args.gpus,
        accumulate_grad_batches=args.gradient_acc_steps,
        gradient_clip_val=args.gradient_clipping,
        logger=wandb_logger,
        callbacks=[early_stopping_cb],
        checkpoint_callback=model_checkpoint,
        val_check_interval=args.validation_check_interval,
        max_steps=args.num_training_steps,
        precision=args.precision,
        amp_level=args.amp_level,
        weights_save_path=f"experiments/{args.run_name}",
    )

    # FIT
    trainer.fit(module, train_dataloader=train_dataloader, val_dataloaders=validation_dataloader)

    # save tokenizer if needed
    if args.use_special_tokens:
        tokenizer.save(f"experiments/{args.run_name}/tokenizer_with_st")


def main():
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "False"

    args = parse_args()
    print(args)
    train(args)


if __name__ == "__main__":
    main()
