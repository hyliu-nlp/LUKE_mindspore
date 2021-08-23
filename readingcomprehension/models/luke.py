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

from model.luke import LukeEntityAwareAttentionModel


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
        encoder_outputs = super(LukeForReadingComprehension, self).construct(
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


class LukeForReadingComprehensionWithLoss(nn.Cell):
    """LukeForReadingComprehensionWithLoss"""

    def __init__(self, net, loss):
        self.lukeforrc = net
        self.loss = loss
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
            start_positions=None,
            end_positions=None
    ):
        start_logits, end_logits = self.lukeforrc(word_ids,
                                                  word_segment_ids,
                                                  word_attention_mask,
                                                  entity_ids,
                                                  entity_position_ids,
                                                  entity_segment_ids,
                                                  entity_attention_mask)
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = self.squeeze(start_positions)
            if len(end_positions.size()) > 1:
                end_positions = self.squeeze(end_positions)

            ignored_index = start_logits.size(1)
            # *.clamp_算子
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            start_loss = self.loss(start_logits, start_positions)
            end_loss = self.loss(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,)
        else:
            outputs = tuple()

        return outputs + (start_logits, end_logits,)


if __name__ == "__main__":
    import mindspore.nn as nn
    model = LukeForReadingComprehension(config='')
    loss = nn.SoftmaxCrossEntropyWithLogits
    LukeForReadingComprehensionWithLoss(model, loss)
