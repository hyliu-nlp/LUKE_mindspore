# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
    SQuAD dataset
"""
import json

import numpy as np

from feature import convert_examples_to_features
from .dataset import SquadV1Processor
from entity_vocab import EntityVocab
import joblib
from wiki_link_db import WikiLinkDB
from word_tokenizer import AutoTokenizer


def _process():
    processor = SquadV1Processor()
    examples = processor.get_dev_examples("../dataset")
    segment_b_id = 1
    add_extra_sep_token = False

    model_redirect_mappings = joblib.load("wiki_entity/enwiki_20181220_redirects.pkl")
    link_redirect_mappings = joblib.load("wiki_entity/enwiki_20160305_redirects.pkl")
    tokenizer = AutoTokenizer.from_pretrained("studio-ousia/luke-large")
    wiki_link_db = WikiLinkDB("wiki_entity/enwiki_20160305.pkl")

    features = convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        entity_vocab=EntityVocab("../dataset/vocab_entity.jsonl"),
        wiki_link_db=wiki_link_db,
        model_redirect_mappings=model_redirect_mappings,
        link_redirect_mappings=link_redirect_mappings,
        max_seq_length=512,
        max_mention_length=30,
        doc_stride=128,
        max_query_length=64,
        min_mention_link_prob=0.01,
        segment_b_id=segment_b_id,
        add_extra_sep_token=add_extra_sep_token,
        is_training=True
    )

    return features


if __name__ == '__main__':
    evaluate = True
    features = _process()
    json_features = []
    for item in features:
        js = json.dumps(item.__dict__)
        json_features.append(js)
    arr_features = np.array(json_features)
    # test_feature = np.array(json_features[:2])
    if evaluate:
        np.save("raw_dev_data.npy", features)
        np.save("dev_data.npy", arr_features)
        # np.save("raw_dev_data1.npy", features[:2])
        # np.save("dev_data1.npy", test_feature)
        # logger.info("Save dev_data successfully!")

    else:
        np.save("raw_train_data.npy", features)
        np.save("train_data.npy", arr_features)
        # logger.info("Save train_data successfully!")
    # save_features(features, "features_save.json", "/root/zengwei/luke_dataProcess/data")
