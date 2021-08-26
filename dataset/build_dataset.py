import mindspore.dataset as ds


def build_dataset(input_file,
                  batch_size,
                  epoch_count=1,
                  rank_size=1,
                  rank_id=0,
                  bucket=None,
                  shuffle=True):
    data_set = ds.MindDataset(input_file,
                              columns_list=['word_ids', 'word_segment_ids', 'word_attention_mask',
                                            'entity_ids', 'entity_position_ids', 'entity_segment_ids',
                                            'entity_attention_mask', 'start_positions', 'end_positions'],
                              shuffle=shuffle,
                              num_shards=rank_size,
                              shard_id=rank_id,
                              num_parallel_workers=1)
    #print(data_set.get_dataset_size())
    #data_set = data_set.batch(batch_size, drop_remainder=False)
    return data_set
