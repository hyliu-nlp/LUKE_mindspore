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
"""squad class"""
import json
import pandas as pd


# from mindtext.common.data import DataSet, Instance


class Squad():
    """extract column contents from raw squad1.1"""

    def __init__(self, lower=False):
        """"""
        self.lower = lower

    def load(self, path):
        """input path and return raw DataSet samples """
        with open(path, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)["data"]
        ds = pd.DataFrame(columns=('id',
                                   'context',
                                   'question_text',
                                   'start_position',
                                   'end_position',
                                   'orig_answer_text',
                                   'is_impossible'))
        all_count = 0
        real_count = 0
        is_impossible = False
        for entry in input_data:
            for paragraph in entry["paragraphs"]:
                paragraph_text = paragraph["context"]
                for qa in paragraph["qas"]:
                    all_count += 1
                    if qa["answers"][0]["answer_start"] != -1:
                        ds = ds.append({'id': qa["id"],
                                        'context': paragraph_text,
                                        'question_text': qa["question"],
                                        'start_position': qa["answers"][0]["answer_start"],
                                        'end_position': qa["answers"][0]["answer_start"] + len(
                                            qa["answers"][0]["text"]) - 1,
                                        'orig_answer_text': qa["answers"][0]["text"],
                                        'is_impossible': is_impossible}, ignore_index=True)

                        real_count += 1

        print("all count:", all_count)
        print("real count:", real_count)
        return ds

    def preprocess(self):
        """dataset process"""
        return None


if __name__ == "__main__":
    testloader = Squad()

    datainfo = testloader.load("train-v1.1.json")
    print(datainfo.head(5))
