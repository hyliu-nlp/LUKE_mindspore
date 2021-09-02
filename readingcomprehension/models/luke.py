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
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.common import dtype as mstype
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.context import ParallelMode
from mindspore.communication.management import get_group_size
from mindspore import context
import mindspore.numpy as np

_grad_overflow = C.MultitypeFuncGraph("_grad_overflow")
GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0
grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()
clip_grad = C.MultitypeFuncGraph("clip_grad")


@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    """
    Clip gradients.

    Inputs:
        clip_type (int): The way to clip, 0 for 'value', 1 for 'norm'.
        clip_value (float): Specifies how much to clip.
        grad (tuple[Tensor]): Gradients.

    Outputs:
        tuple[Tensor], clipped gradients.
    """
    if clip_type not in (0, 1):
        return grad
    dt = F.dtype(grad)
    if clip_type == 0:
        new_grad = C.clip_by_value(grad, F.cast(F.tuple_to_array((-clip_value,)), dt),
                                   F.cast(F.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad, F.cast(F.tuple_to_array((clip_value,)), dt))
    return new_grad


class LukeForReadingComprehension(nn.Cell):
    """Luke for reading comprehension task"""

    def __init__(self, config):
        super(LukeForReadingComprehension, self).__init__()
        self.LukeEntityAwareAttentionModel = LukeEntityAwareAttentionModel(config)
        self.qa_outputs = nn.Dense(config.hidden_size, 2)
        self.split = ops.Split(-1, 2)
        self.squeeze = ops.Squeeze(-1)
        self.shape = ops.Shape()

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
            end_positions=None,
    ):
        """LukeForReadingComprehension construct"""
        encoder_outputs = self.LukeEntityAwareAttentionModel(
            word_ids,
            word_segment_ids,
            word_attention_mask,
            entity_ids,
            entity_position_ids,
            entity_segment_ids,
            entity_attention_mask,
        )

        word_hidden_states = encoder_outputs[0][:, : ops.shape(word_ids)[1], :]
        logits = self.qa_outputs(word_hidden_states)
        start_logits, end_logits = self.split(logits)
        start_logits = self.squeeze(start_logits)
        end_logits = self.squeeze(end_logits)
        return (start_logits, end_logits,)


#         return 1
#         if start_positions is not None and end_positions is not None:
#             if len(self.shape(start_positions)) > 1:
#                 start_positions = self.squeeze(start_positions)
#             if len(self.shape(end_positions)) > 1:
#                 end_positions = self.squeeze(end_positions)

#             ignored_index = ops.shape(start_logits)[1]
#             start_positions = C.clip_by_value(start_positions, 0, ignored_index)
#             end_positions = C.clip_by_value(end_positions, 0, ignored_index)

#             loss_fct = nn.SoftmaxCrossEntropyWithLogits(sparse = True)
#             #loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
#             start_loss = loss_fct(start_logits, start_positions)
#             end_loss = loss_fct(end_logits, end_positions)
#             total_loss = (start_loss + end_loss) / 2
#             outputs = (total_loss,)
#         else:
#             outputs = tuple()
#         return outputs + (start_logits, end_logits,)


class LukeForReadingComprehensionWithLoss(nn.Cell):
    """LukeForReadingComprehensionWithLoss"""

    def __init__(self, net, loss):
        self.net = net
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
        start_logits, end_logits = self.net(word_ids,
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


class LukeSquadCell(nn.Cell):

    def __init__(self, network, optimizer, scale_update_cell=None):
        super(LukeSquadCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.reducer_flag = False
        self.allreduce = P.AllReduce()
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        self.grad_reducer = None
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, mean, degree)
        self.is_distributed = (self.parallel_mode != ParallelMode.STAND_ALONE)
        self.cast = P.Cast()
        self.gpu_target = False
        if context.get_context("device_target") == "GPU":
            self.gpu_target = True
            self.float_status = P.FloatStatus()
            self.addn = P.AddN()
            self.reshape = P.Reshape()
        else:
            self.alloc_status = P.NPUAllocFloatStatus()
            self.get_status = P.NPUGetFloatStatus()
            self.clear_status = P.NPUClearFloatStatus()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.base = Tensor(1, mstype.float32)
        self.less_equal = P.LessEqual()
        self.hyper_map = C.HyperMap()
        self.loss_scale = None
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=mstype.float32))

    def construct(self,
                  input_ids,
                  input_mask,
                  token_type_id,
                  start_position,
                  end_position,
                  unique_id,
                  is_impossible,
                  sens=None):
        """LukeSquad"""
        weights = self.weights
        init = False
        loss = self.network(input_ids,
                            input_mask,
                            token_type_id,
                            start_position,
                            end_position,
                            unique_id,
                            is_impossible)
        if sens is None:
            scaling_sens = self.loss_scale
        else:
            scaling_sens = sens
        if not self.gpu_target:
            init = self.alloc_status()
            init = F.depend(init, loss)
            clear_status = self.clear_status(init)
            scaling_sens = F.depend(scaling_sens, clear_status)
        grads = self.grad(self.network, weights)(input_ids,
                                                 input_mask,
                                                 token_type_id,
                                                 start_position,
                                                 end_position,
                                                 unique_id,
                                                 is_impossible,
                                                 self.cast(scaling_sens,
                                                           mstype.float32))
        grads = self.hyper_map(F.partial(grad_scale, scaling_sens), grads)
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        if not self.gpu_target:
            init = F.depend(init, grads)
            get_status = self.get_status(init)
            init = F.depend(init, get_status)
            flag_sum = self.reduce_sum(init, (0,))
        else:
            flag_sum = self.hyper_map(F.partial(_grad_overflow), grads)
            flag_sum = self.addn(flag_sum)
            flag_sum = self.reshape(flag_sum, (()))
        if self.is_distributed:
            flag_reduce = self.allreduce(flag_sum)
            cond = self.less_equal(self.base, flag_reduce)
        else:
            cond = self.less_equal(self.base, flag_sum)
        overflow = cond
        if sens is None:
            overflow = self.loss_scaling_manager(self.loss_scale, cond)
        if not overflow:
            self.optimizer(grads)
        return (loss, cond)


if __name__ == "__main__":
    import mindspore.nn as nn

    model = LukeForReadingComprehension(config='')
    loss = nn.SoftmaxCrossEntropyWithLogits
    LukeForReadingComprehensionWithLoss(model, loss)
