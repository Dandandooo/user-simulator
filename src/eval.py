# Pre-made Evaluation metrics
from sklearn.metrics import accuracy_score, f1_score

# Tensor functions
import torch

def metrics(y_true, probs):
    y_true = torch.tensor(y_true, dtype=torch.float32)
    probs = torch.tensor(probs, dtype=torch.float32)
    return {
        'single_argmax_accuracy': single_argmax_accuracy(y_true, probs),
        'multi_argmax_accuracy': multi_argmax_accuracy(y_true, probs),
        # 'softmax_accuracy': softmax_accuracy(y_true, probs, 0.5),
        # 'softmax_f1': threshold_f1(y_true, probs, 0.5),
        'max_confidence': max_confidence(probs)
    }


def normalize_probs(probs, choice = 'softmax'):
    # WARN: This function got a wrong type during runtime
    match choice:
        case 'softmax':
            return torch.softmax(probs, dim=1)
        case 'sigmoid':
            return torch.sigmoid(probs)
        case _:
            raise ValueError(f"Invalid choice: {choice}")

def softmax_accuracy(y_true, probs, threshold) -> float:
    y_pred = normalize_probs(probs)
    y_pred = y_pred > threshold
    return accuracy_score(y_true, y_pred)

def threshold_f1(y_true, probs, threshold) -> float:
    y_pred = normalize_probs(probs)
    y_pred = y_pred > threshold
    return f1_score(y_true, y_pred)

def single_argmax_accuracy(y_true, probs) -> float:
    y_pred = normalize_probs(probs)
    y_pred = y_pred.argmax(dim=1)
    y_true = y_true.argmax(dim=1)
    return accuracy_score(y_true, y_pred)

def multi_argmax_accuracy(y_true, probs) -> float:
    score = 0
    for y, t in zip(probs, y_true):
        num_maxes = int(torch.sum(t).item())
        max_idxes = y.topk(num_maxes).indices
        for idx in max_idxes:
            if t[idx] == 1:
                score += 1/num_maxes

    return score / len(probs)

def max_confidence(probs) -> float:
    y_pred = normalize_probs(probs)
    y_pred = y_pred.max(dim=1)
    return y_pred.values.mean().item()
