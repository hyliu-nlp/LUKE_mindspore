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
    luke for tagging and reading comprehension tasks
"""
import mindspore.nn as nn
import mindspore.ops as ops
from mindtext.modules.encoder.luke import LukeEntityAwareAttentionModel


class LukeForReadingComprehension(LukeEntityAwareAttentionModel):
    """Luke for reading comprehension task"""
    def __init__(self, config):
        super(LukeForReadingComprehension, self).__init__(config)

        self.qa_outputs = nn.Dense(self.config.hidden_size, 2)
        self.split = ops.Split(-1, 2)
        self.squeeze = ops.Squeeze(-1)

    def construct(
            self,
            word_ids,
            word_segment_ids,
            word_attention_mask,
            entity_ids,
            entity_position_ids,
            entity_segment_ids,
            entity_attention_mask,
    ):
        """LukeForReadingComprehension construct"""
        encoder_outputs = super(LukeForReadingComprehension, self).forward(
            word_ids,
            word_segment_ids,
            word_attention_mask,
            entity_ids,
            entity_position_ids,
            entity_segment_ids,
            entity_attention_mask,
        )

        word_hidden_states = encoder_outputs[0][:, : ops.shape(word_ids), :]
        logits = self.qa_outputs(word_hidden_states)
        start_logits, end_logits = self.split(logits)
        start_logits = self.squeeze(start_logits)
        end_logits = self.squeeze(end_logits)

        return start_logits, end_logits


if __name__ == "__main__":
    LukeForReadingComprehension(config='')
    # config = LukeConfig("config.json")
    # x = LukeForReadingComprehension(config)
    #
    # param_dict = load_checkpoint("luke.ckpt")
    # load_param_into_net(x, param_dict)
