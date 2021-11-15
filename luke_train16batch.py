from readingcomprehension.models.luke import LukeForReadingComprehension, LukeForReadingComprehensionWithLoss, LukeEntityAwareAttentionModel, LukeSquadCell
import mindspore.common.dtype as mstype
from model.bert_model import BertConfig
from mindspore import context, save_checkpoint
from model.luke import LukeModel, EntityAwareEncoder
from mindspore import Tensor, context
from mindspore import dtype as mstype
import mindspore.ops as ops
import mindspore.nn as nn
from model.bert_model import BertOutput
from mindspore.common.initializer import TruncatedNormal
from mindspore.ops import composite as C
import mindspore
from mindspore.ops import operations as P
from mindspore.train.model import Model
from tqdm import tqdm
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.model import Model
import collections
from mindspore.train.callback import Callback
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.model import Model
from mindspore.nn.learning_rate_schedule import LearningRateSchedule, PolynomialDecayLR, WarmUpLR
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
import time
import numpy as np
import argparse
from dataset import create_dataset, device_id, device_num
from mindspore.context import ParallelMode
from mindspore.communication import init
import os
import moxing as mox
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from luke_eval import do_eval, evaluate


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

        self.greater = mindspore.ops.Greater()
        self.one = Tensor(np.array([1.0]).astype(np.float32))
        self.cast = mindspore.ops.Cast()

    def construct(self, global_step):
        decay_lr = self.decay_lr(global_step)
        if self.warmup_flag:
            is_warmup = self.cast(self.greater(self.warmup_steps, global_step), mstype.float32)
            warmup_lr = self.warmup_lr(global_step)
            lr = (self.one - is_warmup) * decay_lr + is_warmup * warmup_lr
        else:
            lr = decay_lr
            
        
        return lr
class LossCallBack(Callback):
    """
    Monitor the loss in training.

    If the loss is NAN or INF terminating training.

    Note:
        If per_print_times is 0 do not print loss.

    Args:
        per_print_times (int): Print loss every times. Default: 1.
    """

    def __init__(self, per_print_times=1, rank_ids=0):
        super(LossCallBack, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self.rank_id = rank_ids
        self.time_stamp_first = get_ms_timestamp()

    def step_end(self, run_context):
        """Monitor the loss in training."""
        time_stamp_current = get_ms_timestamp()
        cb_params = run_context.original_args()
        print("time: {}, epoch: {}, step: {}, outputs are {}".format(time_stamp_current - self.time_stamp_first,
                                                                     cb_params.cur_epoch_num,
                                                                     cb_params.cur_step_num,
                                                                     str(cb_params.net_outputs)))
        with open("./loss_{}.log".format(self.rank_id), "a+") as f:
            f.write("time: {}, epoch: {}, step: {}, loss: {}".format(
                time_stamp_current - self.time_stamp_first,
                cb_params.cur_epoch_num,
                cb_params.cur_step_num,
                str(cb_params.net_outputs[0].asnumpy())))
            f.write('\n')
def get_ms_timestamp():
    t = time.time()
    return int(round(t * 1000))

def luke_train(args):
    epoch=1
    train_batch_size= 16
    eval_batch_size = 2
    local_data_path = '/cache/data'

    # set graph mode and parallel mode
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False)
    context.set_context(device_id=device_id)

    if device_num > 1:
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        init()
        local_data_path = os.path.join(local_data_path, str(device_id))
    
    # data download
    print('Download data.')
    mox.file.copy_parallel(src_url=args.data_url, dst_url=local_data_path)

    # create dataset
    print('Create train and evaluate dataset.')
    train_dataset = create_dataset(dataset_path=local_data_path, do_train=True,
                                   repeat_num=1, batch_size=train_batch_size)
    eval_dataset = create_dataset(dataset_path=local_data_path, do_train=False,
                                  repeat_num=1, batch_size=eval_batch_size)
    
    # create model
    luke_config = BertConfig()
    LUKEModel = LukeForReadingComprehension(luke_config)
    # load_pretrain_ckpt
    ckptpath = os.path.join(local_data_path, "luke-large.ckpt")

    param_dict = load_checkpoint(ckptpath)
    load_param_into_net(LUKEModel,param_dict)
    lukesquad = LukeForReadingComprehensionWithLoss(LUKEModel)
    lr_schedule = BertLearningRate(learning_rate=15e-6,
                                   end_learning_rate=15e-6 * 0,
                                   warmup_steps=int(train_dataset.get_dataset_size() * epoch * 0.1),
                                   decay_steps=train_dataset.get_dataset_size() * epoch,
                                   power=1.0)
    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=2 ** 32, scale_factor=2, scale_window=1000)
    params = lukesquad.trainable_params()
    num_train_steps=14629
    warmup_steps = int(epoch * num_train_steps * 0.06)
    optimizer = mindspore.nn.AdamWeightDecay(params,
                                         learning_rate=lr_schedule,
                                         beta1=0.9,
                                         beta2=0.98,
                                         eps=1e-06)
    netwithgrads = LukeSquadCell(lukesquad,optimizer=optimizer, scale_update_cell=update_cell)
    model = Model(netwithgrads)
    loss_monitor = LossCallBack()

    model.train(epoch,train_dataset,callbacks=[loss_monitor],dataset_sink_mode=False)

    save_checkpoint(model.train_network.network.net, "/cache/ft.ckpt")
    mox.file.copy_parallel(src_url="/cache/ft.ckpt", dst_url="obs://llddy/LUKE_mindspore/output")
    print("Model save successfully!")

    #outputs = do_eval(eval_dataset, load_checkpoint_path = "/cache/ft.ckpt")
    #evaluate(outputs)
    #print("evaluate successfully!")
    #outputs = do_eval(eval_dataset, load_checkpoint_path = "./output/ft.ckpt", eval_batch_size = 32)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Luke train.')
    parser.add_argument('--data_url', required=True, default=None, help='Location of data.')
    parser.add_argument('--train_url', required=True, default=None, help='Location of training outputs.')
    #parser.add_argument('--ckpt_url', required=True, default=None, help='Location of training outputs.')
    args_opt, unknown = parser.parse_known_args()

    luke_train(args_opt)
    print('ResNet50 training success!')
