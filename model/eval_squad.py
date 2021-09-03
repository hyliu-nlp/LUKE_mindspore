from dataset.build_dataset import build_dataset
import mindspore.dataset as ds
import os
import numpy as np
from mindspore.mindrecord import FileWriter
import json
from readingcomprehension.models.luke import LukeForReadingComprehension, LukeEntityAwareAttentionModel
import mindspore.common.dtype as mstype
from model.bert_model import BertConfig
from mindspore import context
from model.luke import LukeModel, EntityAwareEncoder
from mindspore import Tensor, context
from mindspore import dtype as mstype
import mindspore.ops as ops
import mindspore.nn as nn
from model.bert_model import BertOutput
from mindspore.common.initializer import TruncatedNormal
from mindspore.ops import composite as C
import mindspore
from mindspore.ops import operations as P
from mindspore.train.model import Model
import collections
import math
from collections import Counter
import string
import re
import json
import sys

class SquadExample():
    """extract column contents from raw data"""
    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=False):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

def read_squad_examples(input_file, is_training, version_2_with_negative=False):
    """Return list of SquadExample from input_data or input_file (SQuAD json file)"""
    with open(input_file, "r") as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    def token_offset(text):
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in paragraph_text:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)
        return (doc_tokens, char_to_word_offset)


    def process_one_example(qa, is_training, version_2_with_negative, doc_tokens, char_to_word_offset):
        qas_id = qa["id"]
        question_text = qa["question"]
        start_position = -1
        end_position = -1
        orig_answer_text = ""
        is_impossible = False
        if is_training:
            if version_2_with_negative:
                is_impossible = qa["is_impossible"]
            if (len(qa["answers"]) != 1) and (not is_impossible):
                raise ValueError("For training, each question should have exactly 1 answer.")
            if not is_impossible:
                answer = qa["answers"][0]
                orig_answer_text = answer["text"]
                answer_offset = answer["answer_start"]
                answer_length = len(orig_answer_text)
                start_position = char_to_word_offset[answer_offset]
                end_position = char_to_word_offset[answer_offset + answer_length - 1]
                actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                cleaned_answer_text = " ".join(tokenization.whitespace_tokenize(orig_answer_text))
                if actual_text.find(cleaned_answer_text) == -1:
                    return None
        example = SquadExample(
            qas_id=qas_id,
            question_text=question_text,
            doc_tokens=doc_tokens,
            orig_answer_text=orig_answer_text,
            start_position=start_position,
            end_position=end_position,
            is_impossible=is_impossible)
        return example


    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            doc_tokens, char_to_word_offset = token_offset(paragraph_text)
            for qa in paragraph["qas"]:
                one_example = process_one_example(qa, is_training, version_2_with_negative,
                                                  doc_tokens, char_to_word_offset)
                if one_example is not None:
                    examples.append(one_example)
    return examples

def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs

def get_nbest(prelim_predictions, features, example, n_best_size, do_lower_case):
    """get nbest predictions"""
    _NbestPrediction = collections.namedtuple(
        "NbestPrediction", ["text", "start_logit", "end_logit"])

    seen_predictions = {}
    nbest = []
    for pred in prelim_predictions:
        if len(nbest) >= n_best_size:
            break
        feature = features[pred.feature_index]
        if pred.start_index > 0:  # this is a non-null prediction
            tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            tok_text = " ".join(tok_tokens)

            # De-tokenize WordPieces that have been split off.
            tok_text = tok_text.replace(" ##", "")
            tok_text = tok_text.replace("##", "")

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)
            final_text = get_final_text(tok_text, orig_text, do_lower_case)
            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True
        else:
            final_text = ""
            seen_predictions[final_text] = True

        nbest.append(
            _NbestPrediction(
                text=final_text,
                start_logit=pred.start_logit,
                end_logit=pred.end_logit))

    # In very rare edge cases we could have no valid predictions. So we
    # just create a nonce prediction in this case to avoid failure.
    if not nbest:
        nbest.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

    assert len(nbest) >= 1
    return nbest

def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for (i, score) in enumerate(index_and_score):
        if i >= n_best_size:
            break
        best_indexes.append(score[0])
    return best_indexes

def get_prelim_predictions(features, unique_id_to_result, n_best_size, max_answer_length):
    """get prelim predictions"""
    _PrelimPrediction = collections.namedtuple(
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])
    prelim_predictions = []
    # keep track of the minimum score of null start+end of position 0
    for (feature_index, feature) in enumerate(features):
        if feature['unique_id'] not in unique_id_to_result:
            continue
        result = unique_id_to_result[feature['unique_id']]
        start_indexes = _get_best_indexes(result[1], n_best_size)
        end_indexes = _get_best_indexes(result[2], n_best_size)
        # if we could have irrelevant answers, get the min score of irrelevant
        for start_index in start_indexes:
            for end_index in end_indexes:
                # We could hypothetically create invalid predictions, e.g., predict
                # that the start of the span is in the question. We throw out all
                # invalid predictions.
                if start_index >= len(feature['tokens']):
                    continue
                if end_index >= len(feature['tokens']):
                    continue
                if start_index not in feature['token_to_orig_map']:
                    continue
                if end_index not in feature['token_to_orig_map']:
                    continue
                if not feature['token_is_max_context'].get(start_index, False):
                    continue
                if end_index < start_index:
                    continue
                length = end_index - start_index + 1
                if length > max_answer_length:
                    continue
                prelim_predictions.append(
                    _PrelimPrediction(
                        feature_index=feature_index,
                        start_index=start_index,
                        end_index=end_index,
                        start_logit=result[1][start_index],
                        end_logit=result[2][end_index]))

    prelim_predictions = sorted(
        prelim_predictions,
        key=lambda x: (x.start_logit + x.end_logit),
        reverse=True)
    return prelim_predictions

def get_predictions(all_examples, all_features, all_results, n_best_size, max_answer_length, do_lower_case):
    """Get final predictions"""
    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature['example_index']].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result[0]] = result
    all_predictions = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]
        prelim_predictions = get_prelim_predictions(features, unique_id_to_result, n_best_size, max_answer_length)
        nbest = get_nbest(prelim_predictions, features, example, n_best_size, do_lower_case)

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        all_predictions[example.qas_id] = nbest_json[0]["text"]
    return all_predictions

def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case):
    """Write final predictions to the json file and log-odds of null if needed."""

    all_predictions = get_predictions(all_examples, all_features, all_results,
                                      n_best_size, max_answer_length, do_lower_case)
    return all_predictions

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
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

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def evaluate(dataset, predictions):
    """do evaluation"""
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                if not ground_truths:
                    continue
                prediction = predictions[qa['id']]
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    print("f1:", f1)
    return {'exact_match': exact_match, 'f1': f1}

def SQuad_postprocess(dataset_file, all_predictions, output_metrics="output.json"):
    with open(dataset_file) as ds:
        dataset_json = json.load(ds)
        dataset = dataset_json['data']
    re_json = evaluate(dataset, all_predictions)
    print(json.dumps(re_json))
    with open(output_metrics, 'w') as wr:
        wr.write(json.dumps(re_json))

def f1_score(prediction, ground_truth):
    """calculate f1 score"""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

