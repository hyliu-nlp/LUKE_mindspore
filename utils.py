import math
from abc import ABC

from mindspore.nn.learning_rate_schedule import LearningRateSchedule, WarmUpLR, PolynomialDecayLR
from mindspore.train.callback import Callback
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, AdamW
import mindspore.nn as nn
import numpy as np
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore.common import dtype as mstype

# gradient_accumulation_steps = 3
# num_train_epochs = 2
#
#
# def _create_optimizer(self, model):
#     param_optimizer = list(model.named_parameters())
#     no_decay = ["bias", "LayerNorm.weight"]
#     optimizer_parameters = [
#         {
#             "params": [p for n, p in param_optimizer if p.requires_grad and not any(nd in n for nd in no_decay)],
#             "weight_decay": self.args.weight_decay,
#         },
#         {
#             "params": [p for n, p in param_optimizer if p.requires_grad and any(nd in n for nd in no_decay)],
#             "weight_decay": 0.0,
#         },
#     ]
#     return nn.AdamWeightDecay(optimizer_parameters,
#                               learning_rate=1e-5,
#                               eps=1e-6, beta1=0.9,
#                               beta2=0.98)
#
#
# def _create_scheduler(self, optimizer, steps_per_epoch):
#     num_train_steps = steps_per_epoch // gradient_accumulation_steps * num_train_epochs
#     warmup_proportion = 0.06
#     warmup_steps = int(num_train_steps * warmup_proportion)
#
#     return get_linear_schedule_with_warmup(optimizer, warmup_steps, num_train_steps)
#     if self.args.lr_schedule == "warmup_constant":
#         return get_constant_schedule_with_warmup(optimizer, warmup_steps)
#
#     raise RuntimeError("Unsupported scheduler: " + self.args.lr_schedule)


class BertLearningRate(LearningRateSchedule):
    """
    Warmup-decay learning rate for Bert network.
    """

    def __init__(self, learning_rate, end_learning_rate, warmup_steps, decay_steps, power):
        super(BertLearningRate, self).__init__()
        self.warmup_flag = False
        if warmup_steps > 0:
            self.warmup_flag = True
            self.warmup_lr = WarmUpLR(learning_rate, warmup_steps)
        self.decay_lr = PolynomialDecayLR(learning_rate, end_learning_rate, decay_steps, power)
        self.warmup_steps = Tensor(np.array([warmup_steps]).astype(np.float32))

        self.greater = P.Greater()
        self.one = Tensor(np.array([1.0]).astype(np.float32))
        self.cast = P.Cast()

    def construct(self, global_step):
        decay_lr = self.decay_lr(global_step)
        if self.warmup_flag:
            is_warmup = self.cast(self.greater(self.warmup_steps, global_step), mstype.float32)
            warmup_lr = self.warmup_lr(global_step)
            lr = (self.one - is_warmup) * decay_lr + is_warmup * warmup_lr
        else:
            lr = decay_lr
        return lr


# class LukeLearningRate(LearningRateSchedule):
class LossCallBack(Callback):
    """
    Monitor the loss in training.
    If the loss in NAN or INF terminating training.
    Note:
        if per_print_times is 0 do not print loss.
    Args:
        per_print_times (int): Print loss every times. Default: 1.
    """

    def __init__(self, dataset_size=-1):
        super(LossCallBack, self).__init__()
        self._dataset_size = dataset_size

    def step_end(self, run_context):
        """
        Print loss after each step
        """
        cb_params = run_context.original_args()
        if self._dataset_size > 0:
            percent, epoch_num = math.modf(cb_params.cur_step_num / self._dataset_size)
            if percent == 0:
                percent = 1
                epoch_num -= 1
            print("epoch: {}, current epoch percent: {}, step: {}, outputs are {}"
                  .format(int(epoch_num), "%.3f" % percent, cb_params.cur_step_num, str(cb_params.net_outputs)),
                  flush=True)
        else:
            print("epoch: {}, step: {}, outputs are {}".format(cb_params.cur_epoch_num, cb_params.cur_step_num,
                                                               str(cb_params.net_outputs)), flush=True)
