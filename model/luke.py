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
    Encoder classes for LUKE.
"""
import math
import mindspore.ops as ops
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from .luke_embeddings import RobertaEmbeddings, EntityEmbeddings
from .bert_model import BertOutput, BertEncoderCell


class LukeModel(nn.Cell):
    """LukeModel"""

    def __init__(self, config):
        super(LukeModel, self).__init__()

        self.config = config

        self.encoder = BertEncoderCell(config.batch_size)
        self.pooler = BertPooler(config)

        if self.config.bert_model_name and "roberta" in self.config.bert_model_name:
            self.embeddings = RobertaEmbeddings(config)
            self.embeddings.token_type_embeddings.requires_grad = False
        else:
            self.embeddings = BertEmbeddings(config)
        self.entity_embeddings = EntityEmbeddings(config)

    def construct(self, word_ids, word_segment_ids, word_attention_mask,
                  entity_ids, entity_position_ids, entity_segment_ids, entity_attention_mask
                  ):
        """LukeModel construct"""
        word_seq_size = word_ids.size(1)

        embedding_output = self.embeddings(word_ids, word_segment_ids)

        attention_mask = self._compute_extended_attention_mask(word_attention_mask, entity_attention_mask)
        if entity_ids is not None:
            entity_embedding_output = self.entity_embeddings(entity_ids, entity_position_ids, entity_segment_ids)
            embedding_output = ops.Concat([embedding_output, entity_embedding_output], dim=1)

        encoder_outputs = self.encoder(embedding_output, attention_mask, [None] * self.config.num_hidden_layers)
        sequence_output = encoder_outputs[0]
        word_sequence_output = sequence_output[:, :word_seq_size, :]
        pooled_output = self.pooler(sequence_output)

        if entity_ids is not None:
            entity_sequence_output = sequence_output[:, word_seq_size:, :]
            return (word_sequence_output, entity_sequence_output, pooled_output,) + encoder_outputs[1:]

        return (word_sequence_output, pooled_output,) + encoder_outputs[1:]

    def _compute_extended_attention_mask(self, word_attention_mask, entity_attention_mask):
        attention_mask = word_attention_mask
        if entity_attention_mask is not None:
            attention_mask = self.concat([attention_mask, entity_attention_mask])
        extended_attention_mask = self.unsqueezee(self.unsqueezee(attention_mask, 1), 2)
        extended_attention_mask = extended_attention_mask.astype(mstype.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask


class LukeEntityAwareAttentionModel(LukeModel):
    """LukeEntityAwareAttentionModel"""

    def __init__(self, config):
        super(LukeEntityAwareAttentionModel, self).__init__(config)
        self.config = config
        self.encoder = EntityAwareEncoder(config)

    def construct(self, word_ids, word_segment_ids, word_attention_mask, entity_ids,
                  entity_position_ids, entity_segment_ids, entity_attention_mask):
        word_embeddings = self.embeddings(word_ids, word_segment_ids)
        entity_embeddings = self.entity_embeddings(entity_ids, entity_position_ids, entity_segment_ids)
        attention_mask = self._compute_extended_attention_mask(word_attention_mask, entity_attention_mask)

        return self.encoder(word_embeddings, entity_embeddings, attention_mask)


class EntityAwareSelfAttention(nn.Cell):
    """EntityAwareSelfAttention"""

    def __init__(self, config):
        super(EntityAwareSelfAttention, self).__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Dense(config.hidden_size, self.all_head_size)
        self.key = nn.Dense(config.hidden_size, self.all_head_size)
        self.value = nn.Dense(config.hidden_size, self.all_head_size)

        self.w2e_query = nn.Dense(config.hidden_size, self.all_head_size)
        self.e2w_query = nn.Dense(config.hidden_size, self.all_head_size)
        self.e2e_query = nn.Dense(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.concat = ops.Concat(1)
        self.concat2 = ops.Concat(2)
        self.concat3 = ops.Concat(3)
        self.sotfmax = ops.Softmax()

    def transpose_for_scores(self, x):
        new_x_shape = ops.shape(x)[:-1] + (self.num_attention_heads, self.attention_head_size)
        return ops.transpose(ops.reshape(x, *new_x_shape), (0, 2, 1, 3))

    def construct(self, word_hidden_states, entity_hidden_states, attention_mask):
        """EntityAwareSelfAttention construct"""
        word_size = word_hidden_states.size(1)

        w2w_query_layer = self.transpose_for_scores(self.query(word_hidden_states))
        w2e_query_layer = self.transpose_for_scores(self.w2e_query(word_hidden_states))
        e2w_query_layer = self.transpose_for_scores(self.e2w_query(entity_hidden_states))
        e2e_query_layer = self.transpose_for_scores(self.e2e_query(entity_hidden_states))

        key_layer = self.transpose_for_scores(self.key(self.concat([word_hidden_states, entity_hidden_states])))

        w2w_key_layer = key_layer[:, :, :word_size, :]
        e2w_key_layer = key_layer[:, :, :word_size, :]
        w2e_key_layer = key_layer[:, :, word_size:, :]
        e2e_key_layer = key_layer[:, :, word_size:, :]

        w2w_attention_scores = ops.matmul(w2w_query_layer, ops.transpose(w2w_key_layer, (-1, -2)))
        w2e_attention_scores = ops.matmul(w2e_query_layer, ops.transpose(w2e_key_layer, (-1, -2)))
        e2w_attention_scores = ops.matmul(e2w_query_layer, ops.transpose(e2w_key_layer, (-1, -2)))
        e2e_attention_scores = ops.matmul(e2e_query_layer, ops.transpose(e2e_key_layer, (-1, -2)))

        word_attention_scores = self.concat3([w2w_attention_scores, w2e_attention_scores])
        entity_attention_scores = self.concat3([e2w_attention_scores, e2e_attention_scores])
        attention_scores = self.concat2([word_attention_scores, entity_attention_scores])

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask

        attention_probs = self.softmax(attention_scores)
        attention_probs = self.dropout(attention_probs)

        value_layer = self.transpose_for_scores(
            self.value(self.concat([word_hidden_states, entity_hidden_states]))
        )
        context_layer = ops.matmul(attention_probs, value_layer)

        context_layer = ops.transpose(context_layer, (0, 2, 1, 3))
        new_context_layer_shape = ops.shape(context_layer)[:-2] + (self.all_head_size,)
        context_layer = ops.reshape(context_layer, *new_context_layer_shape)

        return context_layer[:, :word_size, :], context_layer[:, word_size:, :]


class EntityAwareAttention(nn.Cell):
    """EntityAwareAttention"""

    def __init__(self, config):
        super(EntityAwareAttention, self).__init__()
        self.self = EntityAwareSelfAttention(config)
        self.output = BertOutput(config.hidden_size, config.hidden_size)
        self.concat = ops.Concat(1)

    def construct(self, word_hidden_states, entity_hidden_states, attention_mask):
        word_self_output, entity_self_output = self.self(word_hidden_states, entity_hidden_states, attention_mask)
        hidden_states = self.concat([word_hidden_states, entity_hidden_states])
        self_output = self.concat([word_self_output, entity_self_output])
        output = self.output(hidden_states, self_output)
        return output[:, : word_hidden_states.size(1), :], output[:, word_hidden_states.size(1):, :]


class EntityAwareLayer(nn.Cell):
    """EntityAwareLayer"""

    def __init__(self, config):
        super(EntityAwareLayer, self).__init__()

        self.attention = EntityAwareAttention(config)
        self.intermediate = nn.Dense(config.hidden_size, config.hidden_size)
        self.output = BertOutput(config.hidden_size, config.hidden_size)
        self.concat = ops.Concat(1)

    def construct(self, word_hidden_states, entity_hidden_states, attention_mask):
        word_attention_output, entity_attention_output = self.attention(
            word_hidden_states, entity_hidden_states, attention_mask
        )
        attention_output = self.concat([word_attention_output, entity_attention_output])
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        return layer_output[:, : ops.shape(word_hidden_states)[1], :], \
               layer_output[:, ops.shape(word_hidden_states)[1]:, :]


class EntityAwareEncoder(nn.Cell):
    """EntityAwareEncoder"""

    def __init__(self, config):
        super(EntityAwareEncoder, self).__init__()
        self.layer = EntityAwareLayer(config)

    def construct(self, word_hidden_states, entity_hidden_states, attention_mask):
        word_hidden_states, entity_hidden_states = self.layer(word_hidden_states,
                                                              entity_hidden_states, attention_mask)
        return word_hidden_states, entity_hidden_states
