
import os
import json
import random
from tqdm import tqdm
from argparse import ArgumentParser

import numpy as np

import torch

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

    parser.add_argument("-td", "--test_dir", type=str, default="data/eval.json")
    parser.add_argument("-od", "--ontology_dir", type=str, default="data/ontology.json")
    parser.add_argument("-sd", "--save_dir", type=str, default="_roberta-predictions.json")

    parser.add_argument("-pt", "--pretrained_tokenizer", type=str, default="_roberta-result/checkpoint-2757")
    parser.add_argument("-pm", "--pretrained_model", type=str, default="_roberta-result/checkpoint-2757")

    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-ml", "--max_length", type=int, default=512)

    return parser.parse_args()

def predict(args):

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_tokenizer)

    test_set = DSTDataset(
        data_dir=args.test_dir,
        ontology_dir=args.ontology_dir,
        tokenizer=tokenizer,
        max_length=args.max_length
    )

    config = AutoConfig.from_pretrained(args.pretrained_model)

    model = {
        "bert": BertForDialogStateTracking,
        "electra": ElectraForDialogStateTracking,
        "roberta": RobertaForDialogStateTracking,
    }[config.model_type].from_pretrained(args.pretrained_model, config=config)

    train_args = TrainingArguments(
        output_dir="tmp/",
        do_predict=True,
        per_device_eval_batch_size=args.batch_size,
        logging_strategy="steps",
        logging_steps=10,
        no_cuda=False,
    )

    trainer = Trainer(
        model,
        args=train_args,
    )

    outputs = trainer.predict(test_set)
    preds = outputs.predictions
    labels = outputs.label_ids

    gate_logits = preds[1]
    start_logits = preds[2]
    end_logits = preds[3]

    gate_labels = labels[0]
    position_labels = labels[1]

    result = []

    for sample_idx in tqdm(range(len(test_set))):

        source = tokenizer.decode(test_set[sample_idx]["input_ids"], skip_special_tokens=True)

        pred_state = []
        trg_state = []

        for slot_idx, slot in enumerate(test_set.domain_slots):

            gate_pred = gate_logits[slot][sample_idx].argmax(-1)
            gate_pred = test_set.slot_gates[gate_pred]

            if gate_pred == "none":
                pass
            elif gate_pred in ["dontcare", "yes", "no"]:
                pred_state.append(slot + "-" + gate_pred)
            else:
                start_pred = start_logits[slot][sample_idx].argmax(-1)
                end_pred = end_logits[slot][sample_idx].argmax(-1)

                pred = tokenizer.decode(test_set[sample_idx]["input_ids"][start_pred:end_pred], skip_special_tokens=True)
                pred_state.append(slot + "-" + pred)

            gate_label = gate_labels[sample_idx][slot_idx]
            gate_label = test_set.slot_gates[gate_label]

            if gate_label == "none":
                pass
            elif gate_label in ["dontcare", "yes", "no"]:
                trg_state.append(slot + "-" + gate_label)
            else:
                start_label = position_labels[sample_idx][slot_idx][0]
                end_label = position_labels[sample_idx][slot_idx][1]
                label = tokenizer.decode(test_set[sample_idx]["input_ids"][start_label:end_label], skip_special_tokens=True)
                trg_state.append(slot + "-" + label)

        result.append({
            "source": source,
            "target": sorted(trg_state),
            "predict": sorted(pred_state),
        })

    with open(args.save_dir, 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent='\t')

if __name__=="__main__":

    args = parse_args()
    predict(args)