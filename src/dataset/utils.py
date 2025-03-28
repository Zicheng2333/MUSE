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
