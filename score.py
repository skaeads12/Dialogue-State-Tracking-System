
import re
import json
import collections
from sklearn.metrics import precision_recall_fscore_support

F1Scores = collections.namedtuple("F1Scores", ["f1", "precision", "recall"])

def compute_f1(list_ref, list_hyp):
    """Compute F1 score from reference (grouth truth) list and hypothesis list.
    Args:
        list_ref: List of true elements.
        list_hyp: List of postive (retrieved) elements.
    Returns:
        A F1Scores object containing F1, precision, and recall scores.
    """

    ref = collections.Counter(list_ref)
    hyp = collections.Counter(list_hyp)
    true = sum(ref.values())
    positive = sum(hyp.values())
    true_positive = sum((ref & hyp).values())
    precision = float(true_positive) / positive if positive else 1.0
    recall = float(true_positive) / true if true else 1.0
    if precision + recall > 0.0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:  # The F1-score is defined to be 0 if both precision and recall are 0.
        f1 = 0.0

    return F1Scores(f1=f1, precision=precision, recall=recall)

eval_set = json.load(open("data/eval.json", 'r'))
ontology = json.load(open("data/ontology.json", 'r'))
predictions = json.load(open("_roberta-predictions.json", 'r'))

states = []
for dialog in eval_set:
    for utterance in dialog["dialogue"]:
        if utterance["role"] == "user":
            states.append(utterance["state"])

slots = list(ontology.keys())

jga = 0
_f1 = 0
errors = list()

for i, prediction in enumerate(predictions):
    
    if sorted(states[i]) == sorted(prediction["predict"]):
        jga += 1
    else:
        error = False
        error_list = list()

        if len(states[i]) == len(prediction["predict"]):
            for state, pred in zip(sorted(states[i]), sorted(prediction["predict"])):
                if state.replace(" ", "") == pred.replace(" ", ""):
                    pass
                else:
                    error_list.append({
                        "target": state.replace(" ", ""),
                        "predict": pred.replace(" ", ""),
                    })
                    error = True
        else:
            error = True

        if error:
            errors.append({
                "source": prediction["source"],
                "errors": error_list,
            })
        else:
            jga += 1

    target = [trg.replace(" ", "") for trg in states[i]]
    predict = [pred.replace(" ", "") for pred in prediction["predict"]]

    total_state = list(set(target) | set(predict))

    y_true = sorted([total_state.index(t) for t in target])
    y_pred = sorted([total_state.index(p) for p in predict])

    f1 = compute_f1(y_true, y_pred)
    _f1 += f1.f1

print("Joint Goal Accuracy: {}".format(jga / len(predictions)))
print("F1 Micro: {}".format(_f1 / len(predictions)))

with open("error.json", 'w') as f:
    json.dump(errors, f, ensure_ascii=False, indent='\t')
