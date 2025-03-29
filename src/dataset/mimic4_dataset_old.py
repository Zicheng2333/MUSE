import os

import torch
from torch.utils.data import Dataset
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../MUSE/src')))

#from xzc.MUSE.src.dataset.tokenizer import MIMIC4Tokenizer
#from xzc.MUSE.src.utils import processed_data_path, read_txt, load_pickle

#TODO 0328 复现MUSE修改的
from tokenizer import MIMIC4Tokenizer
from src.utils import processed_data_path, read_txt, load_pickle

class MIMIC4Dataset(Dataset):
    def __init__(self, split, task, load_no_label=False, dev=False, return_raw=False):
        if dev:
            assert split == "train"
        if load_no_label:
            assert split == "train"
        self.load_no_label = load_no_label
        self.split = split
        self.task = task
        self.all_hosp_adm_dict = load_pickle(os.path.join(processed_data_path, "mimic4/hosp_adm_dict_v2.pkl"))
        included_admission_ids = read_txt(
            os.path.join(processed_data_path, f"mimic4/task:{task}/{split}_admission_ids.txt"))
        self.no_label_admission_ids = []
        if load_no_label:
            no_label_admission_ids = read_txt(
                os.path.join(processed_data_path, f"mimic4/task:{task}/no_label_admission_ids.txt"))
            self.no_label_admission_ids = no_label_admission_ids
            included_admission_ids += no_label_admission_ids #这个id是入院id而不是患者id
        self.included_admission_ids = included_admission_ids
        if dev:
            self.included_admission_ids = self.included_admission_ids[:10000]
        self.return_raw = return_raw
        self.tokenizer = MIMIC4Tokenizer()

    def __len__(self):
        return len(self.included_admission_ids)

    def __getitem__(self, index):
        admission_id = self.included_admission_ids[index]
        hosp_adm = self.all_hosp_adm_dict[admission_id]

        age = str(hosp_adm.age)
        gender = hosp_adm.gender
        ethnicity = hosp_adm.ethnicity
        types = hosp_adm.trajectory[0]
        codes = hosp_adm.trajectory[1]
        codes_flag = True

        labvectors = hosp_adm.labvectors
        labvectors_flag = True
        if labvectors is None:
            labvectors = torch.zeros(1, 116)
            labvectors_flag = False
        else:
            labvectors = torch.FloatTensor(labvectors)

        discharge = hosp_adm.discharge
        discharge_flag = True
        if discharge is None:
            discharge = ""
            discharge_flag = False

        label = getattr(hosp_adm, self.task)
        label_flag = True
        if label is None:
            label = 0.0
            label_flag = False
        else:
            label = float(label)

        if not self.return_raw:
            age, gender, ethnicity, types, codes = self.tokenizer(
                age, gender, ethnicity, types, codes
            )
            label = torch.tensor(label)

        return_dict = dict()
        return_dict["id"] = admission_id

        return_dict["age"] = age
        return_dict["gender"] = gender
        return_dict["ethnicity"] = ethnicity
        return_dict["types"] = types
        return_dict["codes"] = codes
        return_dict["codes_flag"] = codes_flag

        return_dict["labvectors"] = labvectors
        return_dict["labvectors_flag"] = labvectors_flag

        return_dict["discharge"] = discharge
        return_dict["discharge_flag"] = discharge_flag

        return_dict["label"] = label
        return_dict["label_flag"] = label_flag

        return return_dict


if __name__ == "__main__":
    output_filename = "dataset_info.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        # 第一部分：使用 return_raw=True 获取原始数据的统计信息
        dataset = MIMIC4Dataset(split="train", task="mortality", load_no_label=True, return_raw=True)
        f.write("=== Dataset (raw) statistics ===\n")
        f.write("Total items: {}\n".format(len(dataset)))
        item = dataset[0]
        f.write("Item[0] id: {}\n".format(item["id"]))
        f.write("Item[0] age: {}\n".format(item["age"]))
        f.write("Item[0] gender: {}\n".format(item["gender"]))
        f.write("Item[0] ethnicity: {}\n".format(item["ethnicity"]))
        f.write("Item[0] number of types: {}\n".format(len(item["types"])))
        f.write("Item[0] number of codes: {}\n".format(len(item["codes"])))
        f.write("Item[0] labvectors shape: {}\n".format(item["labvectors"].shape))
        f.write("Item[0] discharge: {}\n".format(item["discharge"]))
        f.write("Item[0] label: {}\n\n".format(item["label"]))

        # 第二部分：使用默认加载方式（非 raw 数据）及 DataLoader 的统计信息
        from torch.utils.data import DataLoader
        from src.dataset.utils import mimic4_collate_fn

        dataset = MIMIC4Dataset(split="train", task="mortality", load_no_label=True)
        f.write("=== Dataset (processed) statistics ===\n")
        f.write("Total items: {}\n".format(len(dataset)))
        item = dataset[0]
        f.write("Item[0] id: {}\n".format(item["id"]))
        f.write("Item[0] age: {}\n".format(item["age"]))
        f.write("Item[0] gender: {}\n".format(item["gender"]))
        f.write("Item[0] ethnicity: {}\n".format(item["ethnicity"]))
        f.write("Item[0] types shape: {}\n".format(item["types"].shape))
        f.write("Item[0] codes shape: {}\n".format(item["codes"].shape))
        f.write("Item[0] label shape: {}\n\n".format(item["label"].shape))

        data_loader = DataLoader(dataset, batch_size=32, collate_fn=mimic4_collate_fn, shuffle=True)
        batch = next(iter(data_loader))
        f.write("=== Batch statistics ===\n")
        f.write("Batch age: {}\n".format(batch["age"]))
        f.write("Batch gender: {}\n".format(batch["gender"]))
        f.write("Batch ethnicity: {}\n".format(batch["ethnicity"]))
        f.write("Batch types shape: {}\n".format(batch["types"].shape))
        f.write("Batch codes shape: {}\n".format(batch["codes"].shape))
        f.write("Batch codes_flag: {}\n".format(batch["codes_flag"]))
        f.write("Batch labvectors shape: {}\n".format(batch["labvectors"].shape))
        f.write("Batch labvectors_flag: {}\n".format(batch["labvectors_flag"]))
        f.write("Batch discharge: {}\n".format(batch["discharge"]))
        f.write("Batch discharge_flag: {}\n".format(batch["discharge_flag"]))
        f.write("Batch label: {}\n".format(batch["label"]))
        f.write("Batch label_flag: {}\n".format(batch["label_flag"]))

    print("数据集的统计信息和一个完整的数据样本已经保存到文件：{}".format(output_filename))