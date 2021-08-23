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
Mindrecord file
"""
import json
import os
from mindspore.mindrecord import FileWriter
import numpy as np


def mindrecord_file_path(config, temp_dir, data_example):
    """查看temp_dir是否存在已经处理成mindrecord的文件，如果已经有则直接返回路径+文件名，没有则生成"""
    file_name = os.path.splitext(os.path.basename(config.data_path))[0]
    if os.path.exists(os.path.join(os.getcwd(), temp_dir)):
        print("MindRecord data already exist")
        data_path = os.path.join(os.getcwd(), temp_dir, file_name)
    else:
        print("Writing data to MindRecord file......")
        dir_path = os.path.join(os.getcwd(), temp_dir)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        data_path = os.path.join(dir_path, file_name)
        write_to_mindrecord(data_example, data_path + '.mindrecord', 1)
    return data_path


def write_to_mindrecord(data, path, shared_num=1):
    """generate mindrecord"""
    if not os.path.isabs(path):
        path = os.path.abspath(path)

    writer = FileWriter(path, shared_num)
    data_schema = {
        "word_ids": {"type": "int32", "shape": [-1]},
        "word_segment_ids": {"type": "int32", "shape": [-1]},
        "word_attention_mask": {"type": "int32", "shape": [-1]},
        "entity_ids": {"type": "int32", "shape": [-1]},
        "entity_position_ids": {"type": "int32", "shape": [-1]},
        "entity_segment_ids": {"type": "int32", "shape": [-1]},
        "entity_attention_mask": {"type": "int32", "shape": [-1]},
        "start_positions": {"type": "int32", "shape": [-1]},
        "end_positions": {"type": "int32", "shape": [-1]}
    }
    writer.add_schema(data_schema, "bert")
    for item in data:
        item['word_ids'] = np.array(item['word_ids'], dtype=np.int32)
        item['word_segment_ids'] = np.array(item['word_segment_ids'], dtype=np.int32)
        item['word_attention_mask'] = np.array(item['word_attention_mask'], dtype=np.int32)
        item['entity_ids'] = np.array(item['entity_ids'], dtype=np.int32)
        item['entity_position_ids'] = np.array(item['entity_position_ids'], dtype=np.int32)
        item['entity_segment_ids'] = np.array(item['entity_segment_ids'], dtype=np.int32)
        item['entity_attention_mask'] = np.array(item['entity_attention_mask'], dtype=np.int32)
        item['start_positions'] = np.array(item['start_positions'], dtype=np.int32)
        item['end_positions'] = np.array(item['end_positions'], dtype=np.int32)
        writer.write_raw_data([item])
    writer.commit()


if __name__ == "__main__":
    import numpy as np
    features = np.load("json_features.npy")
    list_dict = []
    for item in features:
        dict_temp = json.loads(item)
        list_dict.append(dict_temp)
        print(dict_temp["word_ids"])

    write_to_mindrecord(data=list_dict, path="./luke-squad-train.mindrecord", shared_num=1)
