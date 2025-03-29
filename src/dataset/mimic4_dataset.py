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
    This version aggregates each patient's visits (from the 1st to the second last)
    into a single data point. We then use the second-to-last or last visit
    to provide labels:

      - 'readmission': from visits[-2]["30_days_readmission"]
      - 'next_visit_diseases': from visits[-2]["next_visit_diseases"]
      - 'mortality': from visits[-1]["in_hospital_mortality"]

    We require each patient to have at least 2 visits in their 'visits' list.
    If a patient has fewer than 2 visits, we skip them.
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
        #   code_map[index] = (code_str, code_type, bge_embedding)
        # ----------------------------------------------------------------
        self.code_map = load_code_mapping(self.parquet_code_map_path)

        # Decide which .pkl files to load based on 'split'
        if split == "train":
            file_indices = range(1, 15)   # 1 through 14
        elif split == "val":
            file_indices = range(15, 17) # 15, 16
        elif split == "test":
            file_indices = range(17, 21) # 17 to 20
        else:
            raise NotImplementedError(f"Unknown split: {split}")

        self.samples = []
        for idx in file_indices:
            pkl_path = os.path.join(self.data_dir, f"processed_patient_records_batch_{idx}.pkl")
            if not os.path.isfile(pkl_path):
                print(f"[Warning] File not found: {pkl_path}, skipping.")
                continue

            with open(pkl_path, "rb") as f:
                data_list = pickle.load(f)

            # Each 'entry' is one patient
            for entry in data_list:
                patient_id = entry["patient_id"]
                demos = entry.get("demographics", {})
                # We'll keep basic demographics from the 1st visit for convenience
                # or you can store them from all visits. That is up to you.
                age = str(demos.get("age", ""))  # keep as string for tokenizer
                gender = demos.get("gender", "")
                ethnicity = demos.get("race", "")

                visits = entry["visits"]
                # Sort by visit_number if not guaranteed sorted
                visits = sorted(visits, key=lambda x: x.get("visit_number", 0))

                if len(visits) < 2:
                    # Skip if patient has fewer than 2 visits
                    continue

                # The input visits are everything except the final one:
                # visits[0 ... n-2] inclusive
                input_visits = visits[:-1]  # from 1st to second last

                # Identify the last and second-to-last visits for labeling
                last_visit = visits[-1]       # Nth
                second_last_visit = visits[-2]  # (N-1)th

                # -------------------------------------------
                # LABEL
                # -------------------------------------------
                label_val = None
                label_flag = True
                if self.task == "readmission":
                    # from second-to-last visit
                    label_val = second_last_visit.get("30_days_readmission", None)
                elif self.task == "next_visit_diseases":
                    # from second-to-last visit
                    label_val = second_last_visit.get("next_visit_diseases", None)
                elif self.task == "mortality":
                    # from last visit
                    label_val = last_visit.get("in_hospital_mortality", None)
                else:
                    raise ValueError("task is invalid")

                # If label is missing or None, skip
                if label_val is None:
                    continue

                # For binary tasks, store as float; for multi-label, store as FloatTensor
                if self.task in ["readmission", "mortality"]:
                    label_tensor = torch.tensor(float(label_val), dtype=torch.float)
                elif self.task == 'next_visit_diseases':
                    label_tensor = torch.FloatTensor(label_val)
                else:
                    raise ValueError("task is invalid")

                # -------------------------------------------
                # Now gather codes, labs, discharge from the
                # visits in input_visits
                # -------------------------------------------
                all_codes = []
                # We can flatten them all
                for v in input_visits:
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
                        if event_name not in v:
                            continue
                        for item in v[event_name]:
                            cindex = item["code_index"]
                            if cindex in self.code_map:
                                orig_code, code_type, _bge = self.code_map[cindex]
                            else:
                                orig_code, code_type = f"UNK_{cindex}", "UNK"
                            all_codes.append((code_type, orig_code))

                code_types = [x[0] for x in all_codes]
                code_values = [x[1] for x in all_codes]
                codes_flag = (len(all_codes) > 0)

                # -------------------------------------------
                # labvectors (flattened)
                # shape [sum_of_all_lab_events, 3]
                # -------------------------------------------
                all_lab_list = []
                for v in input_visits:
                    if "lab_events" in v:
                        for lab_item in v["lab_events"]:
                            cindex = lab_item["code_index"]
                            rtime = lab_item["relative_time"]
                            val = lab_item["standardized_value"]
                            all_lab_list.append([cindex, rtime, val])

                if len(all_lab_list) > 0:
                    labvectors = torch.FloatTensor(all_lab_list)
                    labvectors_flag = True
                else:
                    labvectors = torch.zeros(1, 3)
                    labvectors_flag = False

                # -------------------------------------------
                # discharge embeddings
                # We'll combine dis_embeddings (and rad_embeddings if you want)
                # from all input visits
                # -------------------------------------------
                discharge_embeddings = []
                for v in input_visits:
                    if "dis_embeddings" in v:
                        for emb_dict in v["dis_embeddings"]:
                            emb_array = emb_dict["embedding"]
                            discharge_embeddings.append(torch.tensor(emb_array))

                    # If you want rad_embeddings as well, uncomment:
                    # if "rad_embeddings" in v:
                    #     for emb_dict in v["rad_embeddings"]:
                    #         emb_array = emb_dict["embedding"]
                    #         discharge_embeddings.append(torch.tensor(emb_array))

                if len(discharge_embeddings) > 0:
                    discharge = torch.stack(discharge_embeddings, dim=0)
                    discharge_flag = True
                else:
                    discharge = torch.zeros(0)
                    discharge_flag = False

                # -------------------------------------------
                # Construct one sample for this patient
                # We do one sample per patient if >=2 visits.
                # "id" can just be the patient_id, or you could
                # do patient_id + '_' + str(len(visits)) if you want
                # a unique "admission" style ID.
                # -------------------------------------------
                sample = {
                    "id": f"{patient_id}",
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