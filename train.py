
import os
import json
import random
from tqdm import tqdm
from argparse import ArgumentParser

import numpy as np

import torch
from torch.optim import AdamW

from transformers import AutoTokenizer, AutoConfig, TrainingArguments, Trainer

from model import BertForDialogStateTracking, ElectraForDialogStateTracking, RobertaForDialogStateTracking
from dataset import DSTDataset

seed = 42

random.seed(seed)
np.random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  # type: ignore
torch.backends.cudnn.deterministic = True  # type: ignore
torch.backends.cudnn.benchmark = True  # type: ignore

def parse_args():

    parser = ArgumentParser()

    parser.add_argument("-td", "--train_dir", type=str, default="data/train.json")
    parser.add_argument("-ed", "--eval_dir", type=str, default="data/eval.json")
    parser.add_argument("-od", "--ontology_dir", type=str, default="data/ontology.json")
    parser.add_argument("-sd", "--save_dir", type=str, default="_roberta-result/")

    parser.add_argument("-pt", "--pretrained_tokenizer", type=str, default="klue/roberta-base")
    parser.add_argument("-pm", "--pretrained_model", type=str, default="klue/roberta-base")

    parser.add_argument("-e", "--num_epochs", type=int, default=3)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-ml", "--max_length", type=int, default=512)

    return parser.parse_args()

def train(args):

    os.makedirs(args.save_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_tokenizer)

    train_set = DSTDataset(
        data_dir=args.train_dir,
        ontology_dir=args.ontology_dir,
        tokenizer=tokenizer,
        max_length=args.max_length
    )

    eval_set = DSTDataset(
        data_dir=args.eval_dir,
        ontology_dir=args.ontology_dir,
        tokenizer=tokenizer,
        max_length=args.max_length
    )

    config = AutoConfig.from_pretrained(args.pretrained_model)
    config.slots = train_set.domain_slots
    config.num_labels = len(train_set.slot_gates)
    
    model = {
        "bert": BertForDialogStateTracking,
        "electra": ElectraForDialogStateTracking,
        "roberta": RobertaForDialogStateTracking,
    }[config.model_type].from_pretrained(args.pretrained_model, config=config)

    train_args = TrainingArguments(
        output_dir = args.save_dir,
        overwrite_output_dir = True,
        do_train = True,
        do_eval = True,
        evaluation_strategy="epoch",
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size = args.batch_size,
        learning_rate = 2e-5,
        num_train_epochs = args.num_epochs,
        logging_strategy = "steps",
        logging_steps = 10,
        save_strategy = "epoch",
        save_total_limit = 1,
        no_cuda = False,
    )

    trainer = Trainer(
        model,
        args = train_args,
        train_dataset = train_set,
        eval_dataset = eval_set,
        tokenizer = tokenizer,
    )

    trainer.train()

if __name__=="__main__":

    args = parse_args()
    train(args)
