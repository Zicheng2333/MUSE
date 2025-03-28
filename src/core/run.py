import os
import sys
src_path = os.path.abspath('../..')
sys.path.append(src_path)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging

from torch.utils.data import DataLoader

from model import MMLBackbone

from src.dataset.mimic4_dataset import MIMIC4Dataset
from src.dataset.utils import mimic4_collate_fn
from src.helper import Helper
from src.utils import count_parameters


def parse_arguments(parser):
    # parser.add_argument("--dataset", type=str, default="mimic4")
    # parser.add_argument("--dataset", type=str, default="eicu")
    # parser.add_argument("--task", type=str, default="readmission")
    parser.add_argument("--monitor", type=str, default="pr_auc")
    parser.add_argument("--dataset", type=str, default="adni")
    parser.add_argument("--task", type=str, default="y") #"readmission", "mortality", 'next_visit_diseases'
    #parser.add_argument("--monitor", type=str, default="auc_macro_ovo")
    parser.add_argument("--dev", action="store_true", default=False)
    parser.add_argument("--load_no_label", type=bool, default=False)
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--code_pretrained_embedding", type=bool, default=True)
    parser.add_argument("--code_layers", type=int, default=2)
    parser.add_argument("--code_heads", type=int, default=2)
    parser.add_argument("--bert_type", type=str, default="prajjwal1/bert-tiny")
    parser.add_argument("--rnn_layers", type=int, default=1)
    parser.add_argument("--rnn_type", type=str, default="GRU")
    parser.add_argument("--rnn_bidirectional", type=bool, default=True)
    parser.add_argument("--ffn_layers", type=int, default=2)
    parser.add_argument("--gnn_layers", type=int, default=2)
    parser.add_argument("--gnn_norm", type=str, default=None)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--monitor_criterion", type=str, default="max")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--no_train", type=bool, default=False)
    parser.add_argument("--note", type=str, default="mml_v19")
    parser.add_argument("--exp_name_attr", type=list, default=["dataset", "task", "note"])
    parser.add_argument("--official_run", action="store_true", default=True)
    parser.add_argument("--no_cuda", type=bool, default=False)
    args = parser.parse_args()
    return args

helper = Helper(parse_arguments)
args = helper.args

if args.dataset == "mimic4":
    train_set = MIMIC4Dataset(split="train", task=args.task, dev=args.dev, load_no_label=args.load_no_label)
    val_set = MIMIC4Dataset(split="val", task=args.task)
    test_set = MIMIC4Dataset(split="test", task=args.task)
    if args.task == 'next_visit_diseases':
        args.num_classes = 24
    else:
        args.num_classes = 1
    collate_fn = mimic4_collate_fn
    tokenizer = train_set.tokenizer
else:
    raise ValueError("Dataset not supported!")

train_loader = DataLoader(
    train_set,
    batch_size=args.batch_size,
    collate_fn=collate_fn,
    num_workers=4 if args.official_run else 0,
    shuffle=True
)
val_loader = DataLoader(
    val_set,
    batch_size=args.batch_size,
    collate_fn=collate_fn,
    num_workers=4 if args.official_run else 0,
    shuffle=False
)
test_loader = DataLoader(
    test_set,
    batch_size=args.batch_size,
    collate_fn=collate_fn,
    num_workers=4 if args.official_run else 0,
    shuffle=False
)

model = MMLBackbone(args, tokenizer)
model.to(args.device)
logging.info(model)
logging.info("Number of parameters: {}".format(count_parameters(model)))

if args.checkpoint:
    helper.load_checkpoint(model, args.checkpoint)

if not args.no_train:
    for epoch in range(args.epochs):

        logging.info("-------train: {}-------".format(epoch))
        scores = model.train_epoch(train_loader)
        for key in scores:
            helper.log(f"metrics/train/{key}", scores[key])
        helper.save_checkpoint(model, "last.ckpt")

        logging.info("-------val: {}-------".format(epoch))
        scores, _ = model.eval_epoch(val_loader, bootstrap=False)
        for key in scores.keys():
            helper.log(f"metrics/val/{key}", scores[key])
        helper.save_checkpoint_if_best(model, "best.ckpt", scores)

        # logging.info("-------test: {}-------".format(epoch))
        # scores, predictions = model.eval_epoch(test_loader, bootstrap=False)
        # for key in scores.keys():
        #     helper.log(f"metrics/test/{key}", scores[key])

        if not args.official_run:
            break

    helper.load_checkpoint(model, os.path.join(helper.model_saved_path, "best.ckpt"))

logging.info("-------final test-------")
#scores, predictions = model.eval_epoch(test_loader, bootstrap=True, output=True)
scores, predictions = model.eval_epoch(test_loader, bootstrap=True, output=False)
for key in scores.keys():
    helper.log(f"metrics/final_test/{key}", scores[key])
helper.save_predictions(predictions)
