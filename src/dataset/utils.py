import torch


def mimic4_collate_fn(data):
    #labvectors = [b["labvectors"] for b in data]
    #for i, lv in enumerate(labvectors):
    #    print(f"Sample {i}: labvectors shape {lv.shape}")


    data = {k: [d[k] for d in data] for k in data[0]}

    data["age"] = torch.stack(data["age"])
    data["gender"] = torch.stack(data["gender"])
    data["ethnicity"] = torch.stack(data["ethnicity"])
    data["types"] = torch.nn.utils.rnn.pad_sequence(
        data["types"], batch_first=True, padding_value=0
    )
    data["codes"] = torch.nn.utils.rnn.pad_sequence(
        data["codes"], batch_first=True, padding_value=0
    )
    data["codes_flag"] = torch.tensor(data["codes_flag"])

    data["labvectors"] = torch.nn.utils.rnn.pad_sequence(
        data["labvectors"], batch_first=True, padding_value=0
    )
    data["labvectors_flag"] = torch.tensor(data["labvectors_flag"])

    # data["discharge"] = data["discharge"]
    data["discharge_flag"] = torch.tensor(data["discharge_flag"])

    data["label"] = torch.stack(data["label"])
    data["label_flag"] = torch.tensor(data["label_flag"])

    return data


'''def mimic4_collate_fn(data):
    """
    将一个 list 的 dict 数据转换为 dict 的 batch，
    并对变长序列使用 pad_sequence，同时确保每个张量是连续的（contiguous），以避免后续 resize 出错。
    """
    # 将每个字段组成列表
    data = {k: [d[k] for d in data] for k in data[0]}
    
    # 对于固定形状的字段，直接 stack
    data["age"] = torch.stack(data["age"])
    data["gender"] = torch.stack(data["gender"])
    data["ethnicity"] = torch.stack(data["ethnicity"])
    
    # 对于变长序列字段，使用 pad_sequence 并确保每个张量是连续的
    data["types"] = torch.nn.utils.rnn.pad_sequence(
        [x.contiguous() for x in data["types"]], batch_first=True, padding_value=0
    )
    data["codes"] = torch.nn.utils.rnn.pad_sequence(
        [x.contiguous() for x in data["codes"]], batch_first=True, padding_value=0
    )
    data["codes_flag"] = torch.tensor(data["codes_flag"])
    
    data["labvectors"] = torch.nn.utils.rnn.pad_sequence(
        [x.contiguous() for x in data["labvectors"]], batch_first=True, padding_value=0
    )
    data["labvectors_flag"] = torch.tensor(data["labvectors_flag"])
    
    # discharge 字段通常是文本（字符串），不需要 pad；如果是张量，可以在 dataset 中处理
    # discharge_flag 直接转换为 tensor
    data["discharge_flag"] = torch.tensor(data["discharge_flag"])
    
    data["label"] = torch.stack(data["label"])
    data["label_flag"] = torch.tensor(data["label_flag"])

    return data'''


def eicu_collate_fn(data):
    data = {k: [d[k] for d in data] for k in data[0]}

    data["age"] = torch.stack(data["age"])
    data["gender"] = torch.stack(data["gender"])
    data["ethnicity"] = torch.stack(data["ethnicity"])
    data["types"] = torch.nn.utils.rnn.pad_sequence(
        data["types"], batch_first=True, padding_value=0
    )
    data["codes"] = torch.nn.utils.rnn.pad_sequence(
        data["codes"], batch_first=True, padding_value=0
    )
    data["codes_flag"] = torch.tensor(data["codes_flag"])

    data["labvectors"] = torch.nn.utils.rnn.pad_sequence(
        data["labvectors"], batch_first=True, padding_value=0
    )
    data["labvectors_flag"] = torch.tensor(data["labvectors_flag"])

    data["apacheapsvar"] = torch.stack(data["apacheapsvar"])
    data["apacheapsvar_flag"] = torch.tensor(data["apacheapsvar_flag"])

    data["label"] = torch.stack(data["label"])
    data["label_flag"] = torch.tensor(data["label_flag"])

    return data
