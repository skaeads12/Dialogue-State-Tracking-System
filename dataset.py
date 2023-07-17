
import re
import json

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import PreTrainedTokenizer

def preprocess(string):

    days = {
        "하루": "1일",
        "이틀": "2일",
        "사흘": "3일",
        "나흘": "4일",
        "닷새": "5일",
        "엿새": "6일",
        "이레": "7일",
        "이흐레": "7일",
        "일주일": "7일",
        "여흐레": "8일",
        "아흐레": "9일",
        "열흘": "10일",
    }

    for day in days.keys():
        if day in string:
            string = string.replace(day, days[day])
    
    numerals = {
        "한": "1",
        "두": "2",
        "세": "3",
        "네": "4",
        "다섯": "5",
        "여섯": "6",
        "일곱": "7",
        "여덟": "8",
        "아홉": "9",
        "열": "10",
        "열한": "11",
        "열두": "12",
    }

    for numeral in numerals.keys():
        match = re.findall("(({})[ ]*([명분시]))".format(numeral), string=string)
        if match:
            for m in match:
                if "오후" in match:
                    string = str(int(numerals[m[1]]) + 12) + m[2]
                else:
                    string = string.replace(m[0], numerals[m[1]] + m[2])
                
    string = string.replace("혼자", "1명")

    match = re.findall("(([0-9]+)시[ ]*([0-9]+)분)", string=string)
    for m in match:
        m = list(m)
        if len(m[1]) == 1:
            m[1] = "0" + m[1]
        if len(m[2]) == 1:
            m[2] = "0" + m[2]
        string = string.replace(m[0], m[1] + " : " + m[2])

    match = re.findall("(([0-9]+)시[ 반]*)", string=string)
    for m in match:
        m = list(m)
        if len(m[1]) == 1:
            m[1] = "0" + m[1]

        if "반" in m[0]:
            string = string.replace(m[0], m[1] + " : 30")
        else:
            string = string.replace(m[0], m[1] + " : 00")

    match = re.findall("([서울 ]*([동서남북]+쪽))", string=string)
    for m in match:
        string = string.replace(m[0], " 서울 " + m[1]).replace("  ", " ").rstrip().strip()

    # errors = {
    #     "비싸고": "비싼",
    #     "비싸도": "비싼",
    #     "에어비앤비": "에어비엔비",
    #     "예술의 전당": "예술의전당",
    #     "회오리 라멘": "회오리라멘",
    #     "문화역서울284": "문화역서울 284",
    #     "그섬호텔": "그섬 호텔",
    #     "혜회역": "혜화역",
    #     "게스트하우스": "게스트 하우스",
    #     "한옥 게스트 하우스": "한옥 게스트하우스",
    # }

    # for error in errors.keys():
    #     string = string.replace(error, errors[error])

    return string

class DSTDataset(Dataset):

    def __init__(
        self,
        data_dir: str = None,
        ontology_dir: str = None,
        tokenizer: PreTrainedTokenizer = None,
        max_length: int = 512,
    ):
        self.data = json.load(open(data_dir, 'r'))
        self.ontology = json.load(open(ontology_dir, 'r'))
        self.domain_slots = list(self.ontology.keys())
        self.slot_gates = ["none", "dontcare", "span", "yes", "no"]
        # self.span_labels = ["B", "I", "O"]

        self.samples = list()

        for sample in self.data:

            history = list()

            for utterance in sample["dialogue"]:

                text = preprocess(utterance["text"])

                if utterance["role"] == "user":

                    self.samples.append({
                        "history": history.copy(),
                        "current_utterance": text,
                        "state": utterance["state"],
                    })

                history.append(text)

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.span_labeling_errors = list()

    def __len__(self):
        return len(self.samples)
    
    def get_dataloader(self, batch_size: int = 32, shuffle: bool = False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]

        history = sample["history"][::-1]
        current_utterance = sample["current_utterance"]
        state = sample["state"]

        current_utterance_ids = self.tokenizer(current_utterance).input_ids
        segment_ids = [0] * len(current_utterance_ids)

        history_ids = self.tokenizer(self.tokenizer.sep_token.join(history), add_special_tokens=False).input_ids
        history_ids = history_ids + [self.tokenizer.sep_token_id] if len(history_ids) != 0 else history_ids
        segment_ids += [1] * len(history_ids)

        gate_labels = [self.slot_gates.index("none") for _ in self.domain_slots]
        span_labels = [[0, 0] for _ in self.domain_slots]

        input_ids = current_utterance_ids + history_ids

        for s in state:
            domain, slot, value = s.split("-")
            slot_idx = self.domain_slots.index("-".join([domain, slot]))

            if value in ["dontcare", "yes", "no"]:
                gate_labels[slot_idx] = self.slot_gates.index(value)
            else:
                gate_labels[slot_idx] = self.slot_gates.index("span")

                value_ids = self.tokenizer(value, add_special_tokens=False).input_ids
                value_len = len(value_ids)

                for i in range(min(len(input_ids), self.max_length) - value_len):
                    if input_ids[i:i+value_len] == value_ids:
                        span_labels[slot_idx][0] = i
                        span_labels[slot_idx][1] = i + value_len

                        break

        attention_mask = [1] * len(input_ids)

        gap = max(0, self.max_length - len(input_ids))

        input_ids += [self.tokenizer.pad_token_id] * gap
        segment_ids += [0] * gap
        attention_mask += [0] * gap

        return {
            "input_ids": torch.LongTensor(input_ids[:self.max_length]),
            "segment_ids": torch.LongTensor(segment_ids[:self.max_length]),
            "attention_mask": torch.LongTensor(attention_mask[:self.max_length]),
            "gate_labels": torch.LongTensor(gate_labels),
            "span_labels": torch.LongTensor(span_labels),
        }

if __name__=="__main__":

    from tqdm import tqdm
    from transformers import AutoTokenizer

    data_dir = "data/eval.json"
    ontology_dir = "data/ontology.json"
    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
    max_length = 512

    dataset = DSTDataset(
        data_dir=data_dir,
        ontology_dir=ontology_dir,
        tokenizer=tokenizer,
        max_length=max_length
    )

    sample = dataset[0]
    for key, value in sample.items():
        print("{}: {}".format(key, value.shape))

    for i, sample in enumerate(tqdm(dataset)):
        
        if sample["input_ids"].shape[0] != 512 or sample["segment_ids"].shape[0] != 512 or sample["attention_mask"].shape[0] != 512 or sample["gate_labels"].shape[0] != 45 or sample["span_labels"].shape[1] != 2:
            for key, value in sample.items():
                print("{}: {}".format(key, value.shape))
            exit()

        if i < 10:
            for key, value in sample.items():
                print("{}: {}".format(key, value))
        else:
            exit()
