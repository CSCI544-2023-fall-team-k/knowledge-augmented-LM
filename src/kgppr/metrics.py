import re
import string
import unicodedata
import logging

def normalize_text(s):
    s = unicodedata.normalize('NFD', s)

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_score(prediction, ground_truth):
    normalized_pred = normalize_text(prediction)
    normalized_gt = normalize_text(ground_truth)
    return normalized_pred == normalized_gt

def exact_matching(example, pred):
    logging.info(f"Gold Answers: {example.answer} / Prediction: {pred.answer}")
    assert(type(example.answer) is list)
    return max(em_score(pred.answer, ans) for ans in example.answer)
