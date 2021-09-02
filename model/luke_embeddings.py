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
# import numpy as np
from model.bert_model import EmbeddingLookup, EmbeddingPostprocessor
from mindspore import Tensor, context


class EntityEmbeddings(nn.Cell):
    """entity embeddings for luke model"""

    def __init__(self, config):
        super(EntityEmbeddings, self).__init__()
        # config.entity_vocab_size = 20
        # config.entity_emb_size = config.hidden_size
        # config.layer_norm_eps = 1e-6
        self.entity_emb_size = config.entity_emb_size
        self.hidden_size = config.hidden_size
        self.entity_embeddings = nn.Embedding(config.entity_vocab_size, config.entity_emb_size, padding_idx=0)

        if config.entity_emb_size != config.hidden_size:
            self.entity_embedding_dense = nn.Dense(config.entity_emb_size, config.hidden_size, has_bias=False)

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # TODO：[config.hidden_size] 和 torch有区别
        self.layer_norm = nn.LayerNorm([config.hidden_size], epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.unsqueezee = ops.ExpandDims()

    def construct(self, entity_ids, position_ids, token_type_ids=None):
        """EntityEmbeddings for luke"""

        if token_type_ids is None:
            token_type_ids = ops.zeros_like(entity_ids)

        entity_embeddings = self.entity_embeddings(entity_ids)
        if self.entity_emb_size != self.hidden_size:
            entity_embeddings = self.entity_embedding_dense(entity_embeddings)
        entity_position_ids_int = clamp(position_ids)

        # entity_position_ids_int = Tensor(entity_position_ids_int.asnumpy().astype(np.int32))
        position_embeddings = self.position_embeddings(entity_position_ids_int)

        # position_embeddings = self.position_embeddings(position_ids)
        position_ids = position_ids.astype(mstype.int32)
        position_embedding_mask = 1.0 * self.unsqueezee((position_ids != -1), -1)

        position_embeddings = position_embeddings * position_embedding_mask
        # return position_embeddings
        position_embeddings = ops.reduce_sum(position_embeddings, -2)
        position_embeddings = position_embeddings / clamp(ops.reduce_sum(position_embedding_mask, -2), minimum=1e-7)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = entity_embeddings + position_embeddings + token_type_embeddings

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


def clamp(x, minimum=0):
    mask = x > minimum
    x = x * mask + minimum
    return x


class RobertaEmbeddings(nn.Cell):
    def __init__(self, config):
        super(RobertaEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size,
                                            config.hidden_size,
                                            padding_idx=config.pad_token_id
                                            )
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
                                                  config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm([config.hidden_size],
                                      epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        # self.register_buffer("position_ids", nn.Range(config.max_position_embeddings).expand((1, -1)))
        # self.register_buffer("token_type_ids",
        #                      ops.Zeros(self.position_ids.size(), dtype=mstype.int64),  # dtype used to torch.long
        #                      persistent=False)
        # End copy
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size,
                                                padding_idx=self.padding_idx)

    def construct(self,
                  input_ids=None,
                  token_type_ids=None,
                  position_ids=None,
                  inputs_embeds=None,
                  past_key_values_length=0):
        if position_ids is None:
            if input_ids is not None:
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = create_position_ids_from_input_ids(inputs_embeds)

        input_shape = input_ids.shape
        # seq_length = input_shape[1]
        if token_type_ids is None:
            token_type_ids = ops.Zeros(input_shape, dtype=mstype.int64)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        position_ids = position_ids.astype(mstype.int32)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.
    Args:
       x: torch.Tensor x:
    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    # pad_id = np.array(padding_idx)
    # mask = Tensor(1 * np.array(input_ids.asnumpy() != pad_id))
    mask = input_ids != padding_idx
    mask = (1.0 * mask)

    # print(mask)
    # mask = input_ids.ne(padding_idx).int()  # 可能有问题
    cumsum = ops.CumSum()
    incremental_indices = (cumsum(mask, 1) + past_key_values_length) * mask
    return incremental_indices + padding_idx

