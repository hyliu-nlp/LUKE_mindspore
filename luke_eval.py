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
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.model import Model
import collections
from src.dataset import SquadV1Processor
import os
import numpy as np
import json
from src.result_writer import Result, write_predictions
from src.squad_eval import EvalOpts as SQUAD_EVAL_OPTS
from src.squad_eval import main as evaluate_on_squad


def do_eval(dataset = None, load_checkpoint_path = None):
    config = BertConfig()
    Luke_model = LukeForReadingComprehension(config)
    Luke_model.set_train(False)
    model = Model(Luke_model)
    param_dict = load_checkpoint(load_checkpoint_path)
    load_param_into_net(Luke_model, param_dict)
    output = []
    model = Model(Luke_model)
    
    RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])
    columns_list = ["unique_id", "word_ids", "word_segment_ids", "word_attention_mask", "entity_ids", "entity_position_ids", "entity_segment_ids", "entity_attention_mask"]

    data_set = dataset
    for data in data_set.create_dict_iterator(num_epochs=1):
        input_data = []
        for i in columns_list:
            input_data.append(data[i])

        unique_id, word_ids,word_segment_ids,word_attention_mask,entity_ids,entity_position_ids,entity_segment_ids,entity_attention_mask = input_data
        #print(unique_id)
        logits = model.predict(word_ids,word_segment_ids,word_attention_mask,entity_ids,entity_position_ids,entity_segment_ids,entity_attention_mask)
        ids = unique_id.asnumpy()
        start = logits[0].asnumpy()
        end = logits[1].asnumpy()
        # print(len(ids))
        for i in range(len(ids)):  # eval_batch_size
            unique_id = int(ids[i])
            start_logits = [float(x) for x in start[i].flat]
            end_logits = [float(x) for x in end[i].flat]
            output.append(RawResult(
                unique_id=unique_id,
                start_logits=start_logits,
                end_logits=end_logits))
        #print(word_ids.shape)
    return output

def evaluate(outputs):
    all_results = []

    RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])
    for item in outputs:
        unique_id = int(item[0])
        start_logits = item[1]
        end_logits = item[2]
        all_results.append(RawResult(unique_id, start_logits, end_logits))

    local_data_path = '/cache/data'
    eval_json_dir = os.path.join(local_data_path, 'eval')
    json_file = os.path.join(local_data_path, 'eval', "dev-v1.1.json")

    processor = SquadV1Processor()
    examples = processor.get_dev_examples(eval_json_dir)

    features_path = os.path.join(local_data_path, 'eval', 'dev_data.npy')
    features = np.load(features_path)
    list_dict = []
    for item in features:
        dict_temp = json.loads(item)
        list_dict.append(dict_temp)
    features = np.array(list_dict)

    output_prediction_file = os.path.join("/cache/output/", "predictions_{}.json".format(""))
    output_nbest_file = os.path.join("/cache/output/", "nbest_predictions_{}.json".format(""))
    output_null_log_odds_file = None

    write_predictions(
        examples,
        features,
        #eval_ms,
        all_results,
        20,
        30,
        True,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        False,
        0.0,
    )   

    evaluate_on_squad(
        SQUAD_EVAL_OPTS(
            json_file,
            pred_file=output_prediction_file,
            na_prob_file=output_null_log_odds_file,
        )
    )


