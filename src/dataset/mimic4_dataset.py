import os
import sys
import torch
import pickle
from torch.utils.data import Dataset
import pandas as pd

# If you have the MIMIC4Tokenizer from your original code:
from tokenizer import MIMIC4Tokenizer

def load_code_mapping(parquet_file_path):
    """
    Load your dict_note_code.parquet file into a dictionary.
    Each row has columns: [code, code_type, bge_embedding, index].
    We'll store them in code_map[index] = (code_string, code_type, bge_embedding).
    """
    df = pd.read_parquet(parquet_file_path)

    code_map = {}
    for _, row in df.iterrows():
        idx = row["index"]  # integer index
        code_str = row["code"]  # e.g. "ccs_215"
        code_type = row["code_type"]  # e.g. "ccs"
        bge_emb = row["bge_embedding"]  # list or array of floats
        code_map[idx] = (code_str, code_type, bge_emb)

    return code_map


class MIMIC4Dataset(Dataset):
    """
    A dataset that mimics the original MIMIC4Dataset structure, but reads
    your 20 .pkl files with this format:

        data[i] = {
            'patient_id': ...,
            'demographics': {'age': 52, 'gender':'F','race':'WHITE',...},
            'visits': [ {
                'visit_number': 1,
                '30_days_readmission': 0 or 1 or None,
                'in_hospital_mortality': 0 or 1 or None,
                'next_visit_diseases': [... or None],
                'ccs_events': [...], 'icd9_events': [...], etc.
                'dis_embeddings': [{ 'embedding': array(...) }, ...],
                'rad_embeddings': [{ 'embedding': array(...) }, ...],
                'lab_events': [ { 'code_index':..., 'relative_time':..., 'standardized_value':...}, ...]
            }, ... ]
        }

    Splits by file index:
      - 1–14 → 'train'
      - 15–16 → 'dev'
      - 17–20 → 'test'

    We keep three flags: codes_flag, labvectors_flag, and discharge_flag,
    similar to your original dataset code.
    """

    def __init__(
        self,
        data_dir,
        task,
        split="train",
        parquet_code_map_path='/root/autodl-tmp/reproduce/other_files/dict_note_code.parquet',
        return_raw=False,
        dev=False
    ):
        super().__init__()
        self.data_dir = data_dir
        self.parquet_code_map_path = parquet_code_map_path
        self.task = task
        self.split = split
        self.return_raw = return_raw
        self.tokenizer = MIMIC4Tokenizer()

        # ----------------------------------------------------------------
        # Load the code_map from dict_note_code.parquet
        # This gives us code_map[index] = (code_str, code_type, bge_embedding)
        # ----------------------------------------------------------------
        self.code_map = load_code_mapping(self.parquet_code_map_path)

        # Decide which .pkl files to load based on 'split'
        if split == "train":
            file_indices = range(1, 15)   # 1 through 14
        elif split == "val":
            file_indices = range(15, 17) # 15, 16
        elif split == 'test':
            file_indices = range(17, 21) # 17 to 20
        else:
            raise NotImplementedError

        self.samples = []
        for idx in file_indices:
            pkl_path = os.path.join(self.data_dir, f"processed_patient_records_batch_{idx}.pkl")
            if not os.path.isfile(pkl_path):
                print(f"[Warning] File not found: {pkl_path}, skipping.")
                continue

            with open(pkl_path, "rb") as f:
                data_list = pickle.load(f)

            # Flatten out each patient's visits
            for entry in data_list:
                patient_id = entry["patient_id"]
                demos = entry.get("demographics", {})
                age = str(demos.get("age", ""))  # keep as string for tokenizer
                gender = demos.get("gender", "")
                ethnicity = demos.get("race", "")

                visits = entry["visits"]
                for visit_info in visits:
                    visit_number = visit_info.get("visit_number", None)
                    if visit_number is None:
                        raise ValueError("visit_number is missing")

                    # Construct a unique ID for this admission
                    admission_id = f"{patient_id}_{visit_number}"

                    # ------------------------------------------------
                    # Retrieve the label for the requested task
                    # ------------------------------------------------
                    if self.task == "readmission":
                        label_val = visit_info.get("30_days_readmission", None)
                    elif self.task == "mortality":
                        label_val = visit_info.get("in_hospital_mortality", None)
                    elif self.task == "next_visit_diseases":
                        label_val = visit_info.get("next_visit_diseases", None)
                    else:
                        raise ValueError("task is invalid")

                    # Skip if label is missing or None
                    if label_val is None:
                        continue
                    label_flag = True

                    # For binary tasks, store as float; for multi-label, store as FloatTensor
                    if self.task in ["readmission", "mortality"]:
                        label_tensor = torch.tensor(float(label_val), dtype=torch.float)
                    elif self.task == 'next_visit_diseases':  # "next_visit_diseases"
                        label_tensor = torch.FloatTensor(label_val)
                    else:
                        raise ValueError("task is invalid")

                    # ------------------------------------------------
                    # Gather all code events
                    # ------------------------------------------------
                    all_code_events = []
                    for event_name in [
                        "ccs_events",
                        "icd9_events",
                        "icd10_events",
                        "drg_APR_events",
                        "drg_HCFA_events",
                        "mimic_events",
                        "phecode_events",
                        "rxnorm_events",
                        "dis_codes",
                        "rad_codes"
                    ]:
                        if event_name not in visit_info:
                            continue
                        for item in visit_info[event_name]:
                            cindex = item["code_index"]
                            # Attempt lookup in self.code_map
                            if cindex in self.code_map:
                                original_code, code_type, _bge_emb = self.code_map[cindex]
                            else:
                                # fallback if missing
                                original_code, code_type = f"UNK_{cindex}", "UNK"
                            # we won't store the bge_embedding here, but you could if you wanted
                            all_code_events.append((code_type, original_code))

                    # Just store them unsorted, or sort if you prefer
                    code_types = [x[0] for x in all_code_events]
                    code_values = [x[1] for x in all_code_events]
                    codes_flag = (len(code_values) > 0)

                    # ------------------------------------------------
                    # Gather lab events → shape [num_lab_events, 3]:
                    #   [code_index, relative_time, standardized_value]
                    # ------------------------------------------------
                    lab_list = []
                    if "lab_events" in visit_info:
                        for lab_item in visit_info["lab_events"]:
                            cindex = lab_item["code_index"]
                            rtime = lab_item["relative_time"]
                            val = lab_item["standardized_value"]
                            lab_list.append([cindex, rtime, val])

                    if len(lab_list) > 0:
                        labvectors = torch.FloatTensor(lab_list)
                        labvectors_flag = True
                    else:
                        labvectors = torch.zeros(0, 3)
                        labvectors_flag = False

                    # ------------------------------------------------
                    # Gather note embeddings: dis_embeddings + rad_embeddings
                    # Combine them into one [N, embed_dim] tensor if present
                    # ------------------------------------------------
                    discharge_embeddings = []
                    if "dis_embeddings" in visit_info:
                        for emb_dict in visit_info["dis_embeddings"]:
                            emb_array = emb_dict["embedding"]
                            discharge_embeddings.append(torch.tensor(emb_array))

                    # if "rad_embeddings" in visit_info:
                    #     for emb_dict in visit_info["rad_embeddings"]:
                    #         emb_array = emb_dict["embedding"]
                    #         discharge_embeddings.append(torch.tensor(emb_array))

                    if len(discharge_embeddings) > 0:
                        discharge = torch.stack(discharge_embeddings, dim=0)
                        discharge_flag = True
                    else:
                        # shape [0]
                        discharge = torch.zeros(0)
                        discharge_flag = False

                    # ------------------------------------------------
                    # Build final sample
                    # ------------------------------------------------
                    sample = {
                        "id": admission_id,
                        "age": age,
                        "gender": gender,
                        "ethnicity": ethnicity,
                        "types": code_types,
                        "codes": code_values,
                        "codes_flag": codes_flag,

                        "labvectors": labvectors,
                        "labvectors_flag": labvectors_flag,

                        "discharge": discharge,
                        "discharge_flag": discharge_flag,

                        "label": label_tensor,
                        "label_flag": label_flag,
                    }
                    self.samples.append(sample)

        # If dev=True, optionally truncate for debugging
        if dev and self.split == "train":
            self.samples = self.samples[:10000]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]
        if self.return_raw:
            # Return the raw data
            return item

        # Otherwise tokenize the typical textual fields:
        # (age, gender, ethnicity, types, codes)
        age, gender, ethnicity, code_types, code_values = self.tokenizer(
            item["age"],
            item["gender"],
            item["ethnicity"],
            item["types"],
            item["codes"]
        )

        return_dict = {
            "id": item["id"],
            "age": age,
            "gender": gender,
            "ethnicity": ethnicity,
            "types": code_types,
            "codes": code_values,
            "codes_flag": item["codes_flag"],

            "labvectors": item["labvectors"],
            "labvectors_flag": item["labvectors_flag"],

            "discharge": item["discharge"],
            "discharge_flag": item["discharge_flag"],

            "label": item["label"],
            "label_flag": item["label_flag"],
        }
        return return_dict


# -------------------------------------------------------------------
# Usage example:
# -------------------------------------------------------------------
if __name__ == "__main__":
    data_dir = "/path/to/folder/containing/processed_patient_records_batch_1.pkl/etc"
    parquet_code_map_path = "/path/to/dict_note_code.parquet"

    # Create a dataset for 30_days_readmission, in 'train' mode
    train_dataset = MIMIC4Dataset(
        data_dir=data_dir,
        parquet_code_map_path=parquet_code_map_path,
        task="30_days_readmission",
        split="train",
        return_raw=False,
        dev=False
    )

    print("Train dataset length:", len(train_dataset))
    sample = train_dataset[0]
    print("Sample keys:", sample.keys())
    print("Sample ID:", sample["id"])
    print("Codes flag:", sample["codes_flag"])
    print("Labvectors flag:", sample["labvectors_flag"])
    print("Discharge flag:", sample["discharge_flag"])
    print("Label:", sample["label"])