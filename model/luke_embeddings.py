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
    luke embeddings
"""
from mindspore.common.initializer import initializer
import mindspore.ops as ops
import mindspore.nn as nn
import mindspore.common.dtype as mstype

from model.bert_model import EmbeddingLookup, EmbeddingPostprocessor


class EntityEmbeddings(nn.Cell):
    """entity embeddings for luke model"""
    def __init__(self, config):
        super(EntityEmbeddings, self).__init__()
        self.config = config
        config.entity_vocab_size = 20
        config.entity_emb_size = config.hidden_size
        config.layer_norm_eps = 1e-6

        self.entity_embeddings = nn.Embedding(config.entity_vocab_size, config.entity_emb_size)

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.layer_norm = nn.LayerNorm((config.hidden_size, config.hidden_size), epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.unsqueezee = ops.ExpandDims()

    def construct(self, entity_ids, position_ids, token_type_ids=None):
        """EntityEmbeddings for luke"""
        if token_type_ids is None:
            token_type_ids = ops.zeros_like(entity_ids)

        entity_embeddings = self.entity_embeddings(entity_ids)
        #position_embeddings = self.position_embeddings(clamp(position_ids))
        position_embeddings = self.position_embeddings(position_ids)
        position_embedding_mask = self.unsqueezee((position_ids != -1), -1)
        position_embeddings = position_embeddings * position_embedding_mask
        position_embeddings = ops.reduce_sum(position_embeddings, -2)
        position_embeddings = position_embeddings / clamp(ops.reduce_sum(position_embedding_mask, -2), minimum=1e-7)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = entity_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


def clamp(x, minimum=0.0):
    mask = x > minimum
    x = x * mask + minimum
    return x


class RobertaEmbeddings(nn.Cell):
    def __init__(self, config):
        output_embedding_shape = [config.batch_size, config.seq_length,
                                  config.hidden_size]
        super(RobertaEmbeddings, self).__init__()
        self.bert_embedding_lookup = EmbeddingLookup(
            vocab_size=config.vocab_size,
            embedding_size=config.hidden_size,
            embedding_shape=output_embedding_shape,
            use_one_hot_embeddings=False,
            initializer_range=config.initializer_range)

        self.bert_embedding_postprocessor = EmbeddingPostprocessor(
            embedding_size=config.hidden_size,
            embedding_shape=output_embedding_shape,
            use_relative_positions=config.use_relative_positions,
            use_token_type=True,
            token_type_vocab_size=config.type_vocab_size,
            use_one_hot_embeddings=False,
            initializer_range=0.02,
            max_position_embeddings=config.max_position_embeddings,
            dropout_prob=config.hidden_dropout_prob)
        self.token_type_ids = initializer(
            "zeros", [config.batch_size, config.seq_length], mstype.int32).to_tensor()

    def construct(self, input_ids, segment_ids):
        word_embeddings, embedding_tables = self.bert_embedding_lookup(input_ids)
        embedding_output = self.bert_embedding_postprocessor(self.token_type_ids,
                                                             word_embeddings)
        return embedding_output

