
from typing import Optional

import torch
from torch import nn

from transformers import (
    BertPreTrainedModel,
    ElectraPreTrainedModel,
    BertModel,
    ElectraModel,
    RobertaPreTrainedModel,
    RobertaModel,
)

class BertForDialogStateTracking(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.slots = config.slots
        self.num_labels = config.num_labels
        self.max_length = config.max_position_embeddings
        
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        for slot in self.slots:
            self.add_module("gate_" + slot, nn.Linear(config.hidden_size, self.num_labels))
            self.add_module("span_" + slot, nn.Linear(config.hidden_size, 2))
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        gate_labels: Optional[torch.Tensor] = None,
        span_labels: Optional[torch.Tensor] = None,
    ):
        
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=segment_ids,
        )

        sequence_output = outputs[0]
        pooled_output  = sequence_output[:, 0]
        
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        total_loss = 0
        per_slot_gate_logits = {}
        per_slot_start_logits = {}
        per_slot_end_logits = {}
        per_slot_example_loss = {}

        for i, slot in enumerate(self.slots):

            gate_logits = getattr(self, "gate_" + slot)(pooled_output)
            span_logits = getattr(self, "span_" + slot)(sequence_output)

            start_logits, end_logits = span_logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            per_slot_gate_logits[slot] = gate_logits
            per_slot_start_logits[slot] = start_logits
            per_slot_end_logits[slot] = end_logits

            loss_fct = nn.CrossEntropyLoss(reduction="none")
            
            gate_loss = loss_fct(gate_logits, gate_labels[:, i])
            start_loss = loss_fct(start_logits, span_labels[:, i, 0])
            end_loss = loss_fct(end_logits, span_labels[:, i, 1])
            span_loss = (start_loss + end_loss) / 2.0

            per_example_loss = 0.8 * gate_loss + 0.2 * span_loss
            total_loss += per_example_loss.sum()
            per_slot_example_loss[slot] = per_example_loss

        outputs = (total_loss,) + (per_slot_example_loss, per_slot_gate_logits, per_slot_start_logits, per_slot_end_logits,) + outputs[2:]

        return outputs
    
class ElectraForDialogStateTracking(ElectraPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.slots = config.slots
        self.num_labels = config.num_labels
        self.max_length = config.max_position_embeddings
        
        self.config = config
        
        self.electra = ElectraModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        for slot in self.slots:
            self.add_module("gate_" + slot, nn.Linear(config.hidden_size, self.num_labels))
            self.add_module("span_" + slot, nn.Linear(config.hidden_size, 2))
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        gate_labels: Optional[torch.Tensor] = None,
        span_labels: Optional[torch.Tensor] = None,
    ):
        
        outputs = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=segment_ids,
        )

        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0]
        
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        total_loss = 0
        per_slot_gate_logits = {}
        per_slot_start_logits = {}
        per_slot_end_logits = {}
        per_slot_example_loss = {}

        for i, slot in enumerate(self.slots):

            gate_logits = getattr(self, "gate_" + slot)(pooled_output)
            span_logits = getattr(self, "span_" + slot)(sequence_output)

            start_logits, end_logits = span_logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            per_slot_gate_logits[slot] = gate_logits
            per_slot_start_logits[slot] = start_logits
            per_slot_end_logits[slot] = end_logits

            loss_fct = nn.CrossEntropyLoss(reduction="none")
            
            gate_loss = loss_fct(gate_logits, gate_labels[:, i])
            start_loss = loss_fct(start_logits, span_labels[:, i, 0])
            end_loss = loss_fct(end_logits, span_labels[:, i, 1])
            span_loss = (start_loss + end_loss) / 2.0

            per_example_loss = 0.8 * gate_loss + 0.2 * span_loss
            total_loss += per_example_loss.sum()
            per_slot_example_loss[slot] = per_example_loss

        outputs = (total_loss,) + (per_slot_example_loss, per_slot_gate_logits, per_slot_start_logits, per_slot_end_logits,) + outputs[2:]

        return outputs

class RobertaForDialogStateTracking(RobertaPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.slots = config.slots
        self.num_labels = config.num_labels
        self.max_length = config.max_position_embeddings
        
        self.config = config

        self.roberta = RobertaModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        for slot in self.slots:
            self.add_module("gate_" + slot, nn.Linear(config.hidden_size, self.num_labels))
            self.add_module("span_" + slot, nn.Linear(config.hidden_size, 2))
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        gate_labels: Optional[torch.Tensor] = None,
        span_labels: Optional[torch.Tensor] = None,
    ):
        
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            # token_type_ids=segment_ids,
        )

        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0]
        
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        total_loss = 0
        per_slot_gate_logits = {}
        per_slot_start_logits = {}
        per_slot_end_logits = {}
        per_slot_example_loss = {}

        for i, slot in enumerate(self.slots):

            gate_logits = getattr(self, "gate_" + slot)(pooled_output)
            span_logits = getattr(self, "span_" + slot)(sequence_output)

            start_logits, end_logits = span_logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            per_slot_gate_logits[slot] = gate_logits
            per_slot_start_logits[slot] = start_logits
            per_slot_end_logits[slot] = end_logits

            loss_fct = nn.CrossEntropyLoss(reduction="none")

            gate_loss = loss_fct(gate_logits, gate_labels[:, i])
            start_loss = loss_fct(start_logits, span_labels[:, i, 0])
            end_loss = loss_fct(end_logits, span_labels[:, i, 1])
            span_loss = (start_loss + end_loss) / 2.0

            per_example_loss = 0.8 * gate_loss + 0.2 * span_loss
            total_loss += per_example_loss.sum()
            per_slot_example_loss[slot] = per_example_loss

        outputs = (total_loss,) + (per_slot_example_loss, per_slot_gate_logits, per_slot_start_logits, per_slot_end_logits,) + outputs[2:]

        return outputs

if __name__=="__main__":

    from transformers import AutoConfig

    pretrained_model = "monologg/koelectra-base-v3-discriminator"

    input_ids = torch.randint(0, 1000, (32, 512))
    gate_labels = torch.randint(0, 5, (32, 45,))
    span_labels = torch.randint(0, 512, (32, 45, 2))

    config = AutoConfig.from_pretrained(pretrained_model)
    config.num_labels = 5
    config.slots = ['관광-경치 좋은', '관광-교육적', '관광-도보 가능', '관광-문화 예술', '관광-역사적', '관광-이름', '관광-종류', '관광-주차 가능', '관광-지역', '숙소-가격대', '숙소-도보 가능', '숙소-수영장 유무', '숙소-스파 유무', '숙소-예약 기간', '숙소-예약 명수', '숙소-예약 요일', '숙소-이름', '숙소-인터넷 가능', '숙소-조식 가능', '숙소-종류', '숙소-주차 가능', '숙소-지역', '숙소-헬스장 유무', '숙소-흡연 가능', '식당-가격대', '식당-도보 가능', '식당-야외석 유무', '식당-예약 명수', '식당-예약 시간', '식당-예약 요일', '식당-이름', '식당-인터넷 가능', '식당-종류', '식당-주류 판매', '식당-주차 가능', '식당-지역', '식당-흡연 가능', '지하철-도착지', '지하철-출발 시간', '지하철-출발지', '택시-도착 시간', '택시-도착지', '택시-종류', '택시-출발 시간', '택시-출발지']
    model = RobertaForDialogStateTracking.from_pretrained(pretrained_model, config=config)

    outputs = model(input_ids=input_ids, gate_labels=gate_labels, span_labels=span_labels)

    loss, example_loss, gate_logits, start_logits, end_logits = outputs

    print(loss)
    loss.backward()

    print(example_loss)
    print(gate_logits)
    print(start_logits)
    print(end_logits)
