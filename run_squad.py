from mindspore import load_checkpoint, load_param_into_net, Model
from mindspore.nn import AdamWeightDecay, DynamicLossScaleUpdateCell
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor
import mindspore.nn as nn
from dataset.build_dataset import build_dataset
from readingcomprehension.models.luke import LukeForReadingComprehensionWithLoss, LukeSquadCell, \
    LukeForReadingComprehension
from utils import BertLearningRate, LossCallBack


def do_train(dataset=None, network=None, load_checkpoint_path="", save_checkpoint_path="", epoch_num=1):
    steps_per_epoch = dataset.get_dataset_size()
    epoch_num = 3
    # opimizer
    lr_schedule = BertLearningRate(learning_rate=1e-5,
                                   end_learning_rate=1e-6,
                                   warmup_steps=int(steps_per_epoch * epoch_num * 0.1),
                                   decay_steps=steps_per_epoch * epoch_num,
                                   power=1)
    param_optimizer = network.trainable_params()
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_parameters = [  # 这里可能有问题，直接照搬的
        {
            "params": [p for n, p in param_optimizer if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in param_optimizer if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamWeightDecay(optimizer_parameters, lr_schedule, eps=1e-6)

    # load checkpoint into network
    ckpt_config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch, keep_checkpoint_max=1)
    ckpoint_cb = ModelCheckpoint(prefix="squad",
                                 directory=None if save_checkpoint_path == "" else save_checkpoint_path,
                                 config=ckpt_config)
    param_dict = load_checkpoint(load_checkpoint_path)
    load_param_into_net(network, param_dict)

    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=2 ** 32, scale_factor=2, scale_window=1000)
    netwithgrads = LukeSquadCell(network, optimizer=optimizer, scale_update_cell=update_cell)
    model = Model(netwithgrads)
    callbacks = [TimeMonitor(dataset.get_dataset_size()), LossCallBack(dataset.get_dataset_size()), ckpoint_cb]
    model.train(epoch_num, dataset, callbacks=callbacks)


if __name__ == "__main__":
    from config import get_config

    config_path = "./luke_squad.yaml"
    config = get_config(config_path)

    input_file = "./dataset/luke-squad-train.mindrecord"
    dataset_train = build_dataset(input_file=input_file,
                                  batch_size=2)

    model_config = config.luke_net_cfg
    model = LukeForReadingComprehension(model_config)
    netwithloss = LukeForReadingComprehensionWithLoss(model, nn.SoftmaxCrossEntropyWithLogits)

    do_train(dataset_train, netwithloss)
