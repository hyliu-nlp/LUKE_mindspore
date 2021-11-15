import mindspore.dataset as ds
import os
import numpy as np
from mindspore.mindrecord import FileWriter
import json

device_id = int(os.getenv('DEVICE_ID'))
device_num = int(os.getenv('RANK_SIZE'))

def create_dataset(dataset_path, do_train, repeat_num=1, batch_size=2):
    """
    Create a train or eval dataset.

    Args:
        dataset_path (str): The path of dataset.
        do_train (bool): Whether dataset is used for train or eval.
        repeat_num (int): The repeat times of dataset. Default: 1.
        batch_size (int): The batch size of dataset. Default: 32.

    Returns:
        Dataset.
    """
    if do_train:
        dataset_path = os.path.join(dataset_path, 'train', "train_features.mindrecord")
        do_shuffle = True
    else:
        dataset_path = os.path.join(dataset_path, 'eval', "dev_features.mindrecord")
        do_shuffle = False

    columns_list = ['word_ids',  'word_segment_ids', 'word_attention_mask',
               'entity_ids', 'entity_position_ids', 'entity_segment_ids',
               'entity_attention_mask', 'start_positions', 'end_positions']

    print(dataset_path)
    if device_num == 1 or not do_train:
        dataset = ds.MindDataset(dataset_path, num_parallel_workers=8)
    else:
        dataset = ds.MindDataset(dataset_path, columns_list=columns_list, num_parallel_workers=8, 
                              num_shards=device_num, shard_id=device_id)
    # apply batch operations
    dataset = dataset.batch(batch_size)

    # apply dataset repeat operation
    dataset = dataset.repeat(repeat_num)

    return dataset