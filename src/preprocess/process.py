#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd

# ----------------------------------------------------------------------------------
# Adjust this path to point to your project directory if needed.
# e.g., src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# or set it based on your environment
import sys
src_path = os.path.abspath('../..')
sys.path.append(src_path)

# Import utility functions from your own 'src/utils.py'
# (Ensure that your PYTHONPATH or sys.path is set correctly so this import works)
from src.utils import create_directory, dump_pickle, raw_data_path, processed_data_path, set_seed


def extract_discharge_instructions(note):
    """
    Extract the 'Discharge Instructions' section from the discharge summary note.
    """
    start_index = note.find("Discharge Instructions:")
    end_index = note.find("Followup Instructions:")

    if start_index != -1 and end_index != -1:
        instructions = note[start_index + len("Discharge Instructions:"):end_index].strip()
    else:
        return float('nan')

    instructions = instructions.replace("___", "").replace("\n", " ").strip()
    return instructions


def extract_brief_hospital_course(note):
    """
    Extract the 'Brief Hospital Course' section from the discharge summary note.
    """
    start_index = note.find("Brief Hospital Course:")
    end_index = note.find("Medications on Admission:")

    if start_index != -1 and end_index != -1:
        instructions = note[start_index + len("Brief Hospital Course:"):end_index].strip()
    else:
        return float('nan')

    instructions = instructions.replace("___", "").replace("\n", " ").strip()
    return instructions


def main():
    # Set random seed
    set_seed(seed=42)

    # ----------------------------------------------------------------------------
    # Define your input and output paths
    # (Adjust them to match your folder structure)
    # For example, if raw_data_path points to "/root/autodl-tmp/data":
    input_path = os.path.join(raw_data_path, "hosp")  # originally MIMIC-IV/hosp
    print("raw path:", raw_data_path)
    print("input_path:", input_path)

    # Output path
    output_path = os.path.join(processed_data_path, "mimic4")
    create_directory(output_path)
    print("Output path:", output_path)

    # ----------------------------------------------------------------------------
    # 1. Patients
    # Uncomment and adapt if you prefer dynamic path loading:
    # patients = pd.read_csv(os.path.join(input_path, 'patients.csv'))
    patients = pd.read_csv("/root/autodl-tmp/data/MIMIC/hosp/patients.csv")
    print("Patients shape:", patients.shape)
    print(patients.head())

    patients['dod'] = pd.to_datetime(patients['dod'])
    print("Unique subject_id in patients:", patients.subject_id.nunique())

    patients = patients[['subject_id', 'gender', 'anchor_age', 'anchor_year', 'dod']]
    print(patients.head())

    # ----------------------------------------------------------------------------
    # 2. Admissions
    # admissions = pd.read_csv(os.path.join(input_path, 'admissions.csv'))
    admissions = pd.read_csv("/root/autodl-tmp/data/MIMIC/hosp/admissions.csv")
    print("Admissions shape:", admissions.shape)
    print(admissions.head())

    print("Unique hadm_id in admissions:", admissions.hadm_id.nunique())

    admissions['admittime'] = pd.to_datetime(admissions['admittime'])
    admissions['dischtime'] = pd.to_datetime(admissions['dischtime'])
    print(admissions.head())

    admissions = admissions[['subject_id', 'hadm_id', 'admittime', 'dischtime', 'hospital_expire_flag', 'race']]
    print(admissions.head())

    # ----------------------------------------------------------------------------
    # Merge patients and admissions
    df = admissions.merge(patients, on='subject_id', how='inner')
    print(df.head())

    df['admit_age'] = df.apply(lambda x: x.admittime.year - x.anchor_year + x.anchor_age, axis=1)
    df["duration"] = (df.dischtime - df.admittime).dt.total_seconds() / 60

    df = df.sort_values(['subject_id', 'admittime'], ascending=True).reset_index(drop=True)
    print(df.head())

    df['readmit_in_days'] = (df.groupby('subject_id').admittime.shift(-1) - df.dischtime).dt.days
    df['die_in_days'] = (df.dod - df.dischtime).dt.days

    admittime = df[["hadm_id", "admittime"]]
    print("admittime sample:\n", admittime.head())

    df = df[[
        "subject_id", "hadm_id", "hospital_expire_flag", "readmit_in_days", "die_in_days",
        "race", "gender", "admit_age", "duration"
    ]]
    print(df.head())

    df.to_csv(os.path.join(output_path, 'patients_admissions_tmp.csv'), index=False)

    # ----------------------------------------------------------------------------
    # 3. Prescriptions
    prescriptions_raw = pd.read_csv(os.path.join(input_path, 'prescriptions.csv'), dtype={'drug': str})
    print("Prescriptions raw shape:", prescriptions_raw.shape)
    print(prescriptions_raw.head())

    prescriptions = prescriptions_raw[['subject_id', 'hadm_id', 'starttime', 'drug']]
    print("Prescriptions subset shape:", prescriptions.shape)
    print(prescriptions.head())

    prescriptions = prescriptions.merge(admittime, on="hadm_id", how="inner")
    prescriptions['starttime'] = pd.to_datetime(prescriptions['starttime'])
    prescriptions["timestamp"] = (prescriptions.starttime - prescriptions.admittime).dt.total_seconds() / 60

    prescriptions = prescriptions[["subject_id", "hadm_id", "timestamp", "drug"]]
    prescriptions = prescriptions.dropna().drop_duplicates()

    print("Unique drugs:", prescriptions.drug.nunique())
    print("Prescriptions final shape:", prescriptions.shape)

    prescriptions = prescriptions.sort_values(['subject_id', 'hadm_id', 'timestamp', 'drug'], ascending=True)\
                                 .reset_index(drop=True)
    prescriptions.to_csv(os.path.join(output_path, 'prescriptions_tmp.csv'), index=False)

    # ----------------------------------------------------------------------------
    # 4. Diagnoses ICD
    diagnoses_icd = pd.read_csv(os.path.join(input_path, 'diagnoses_icd.csv'))
    print("diagnoses_icd shape:", diagnoses_icd.shape)
    print(diagnoses_icd.head())

    d_icd_diagnoses = pd.read_csv(os.path.join(input_path, 'd_icd_diagnoses.csv'))
    print("d_icd_diagnoses shape:", d_icd_diagnoses.shape)
    print(d_icd_diagnoses.head())

    diagnoses_icd = diagnoses_icd.merge(d_icd_diagnoses, on=["icd_code", "icd_version"])
    diagnoses_icd = diagnoses_icd.dropna().drop_duplicates()

    print("Unique long_title in diagnoses:", diagnoses_icd.long_title.nunique())
    print("Diagnoses shape after cleaning:", diagnoses_icd.shape)

    diagnoses_icd = diagnoses_icd.sort_values(['subject_id', 'hadm_id', 'seq_num'], ascending=True)\
                                 .reset_index(drop=True)
    diagnoses_icd = diagnoses_icd[["subject_id", "hadm_id", "long_title"]]

    diagnoses_icd.to_csv(os.path.join(output_path, 'diagnoses_icd_tmp.csv'), index=False)

    # ----------------------------------------------------------------------------
    # 5. Procedures ICD
    procedures_icd = pd.read_csv(os.path.join(input_path, 'procedures_icd.csv'))
    print("procedures_icd shape:", procedures_icd.shape)
    print(procedures_icd.head())

    d_icd_procedures = pd.read_csv(os.path.join(input_path, 'd_icd_procedures.csv'))
    print("d_icd_procedures shape:", d_icd_procedures.shape)
    print(d_icd_procedures.head())

    procedures_icd = procedures_icd.merge(d_icd_procedures, on=["icd_code", "icd_version"])
    procedures_icd = procedures_icd.dropna().drop_duplicates()

    print("Unique icd_code in procedures:", procedures_icd.icd_code.nunique())
    print("Procedures shape after cleaning:", procedures_icd.shape)

    procedures_icd = procedures_icd.merge(admittime, on="hadm_id", how="inner")
    procedures_icd['chartdate'] = pd.to_datetime(procedures_icd['chartdate'])
    procedures_icd["timestamp"] = (procedures_icd.chartdate - procedures_icd.admittime).dt.total_seconds() / 60

    procedures_icd = procedures_icd.sort_values(['subject_id', 'hadm_id', 'timestamp', 'seq_num'], ascending=True)\
                                   .reset_index(drop=True)
    procedures_icd = procedures_icd[["subject_id", "hadm_id", "timestamp", "long_title"]]

    procedures_icd.to_csv(os.path.join(output_path, 'procedures_icd_tmp.csv'), index=False)

    # ----------------------------------------------------------------------------
    # 6. Labevents
    selected_labs = [
        'Hematocrit',
        'Platelet',
        'WBC',
        'Bilirubin',
        'pH',
        'Bicarbonate',
        'Creatinine',
        'Lactate',
        'Potassium',
        'Sodium',
    ]

    d_labitems = pd.read_csv(os.path.join(input_path, 'd_labitems.csv'))
    print("d_labitems shape:", d_labitems.shape)
    print(d_labitems.head())

    # Make sure label is not NaN
    d_labitems = d_labitems.dropna(subset=["label"])
    d_labitems = d_labitems[d_labitems.label.apply(lambda x: any(lab in x for lab in selected_labs))]

    labevents_raw = pd.read_csv(os.path.join(input_path, 'labevents.csv'))
    print("labevents_raw shape:", labevents_raw.shape)
    print(labevents_raw.head())

    labevents = labevents_raw[['subject_id', 'hadm_id', 'specimen_id', 'storetime', 'itemid', 'valuenum', 'flag']]
    print("labevents subset shape:", labevents.shape)
    print(labevents.head())

    labevents = labevents.merge(d_labitems, on="itemid", how="inner")
    labevents = labevents.merge(admittime, on="hadm_id", how="inner")

    labevents['storetime'] = pd.to_datetime(labevents['storetime'])
    labevents["timestamp"] = (labevents.storetime - labevents.admittime).dt.total_seconds() / 60

    labevents = labevents[["subject_id", "hadm_id", "timestamp", "itemid", "valuenum", "flag"]]
    labevents = labevents.dropna(subset=["subject_id", "hadm_id", "timestamp", "itemid", "valuenum"])\
                         .drop_duplicates()

    print("Labevents shape after cleaning:", labevents.shape)

    labevents = labevents.sort_values(['subject_id', 'hadm_id', 'timestamp', 'itemid'], ascending=True)\
                         .reset_index(drop=True)
    print("Mean # of timestamps per admission (approx):",
          labevents.groupby(["subject_id", "hadm_id"]).timestamp.nunique().mean())

    labevents.to_csv(os.path.join(output_path, 'labevents_tmp.csv'), index=False)

    # ----------------------------------------------------------------------------
    # 7. Discharge instructions (version 1)
    # NOTE: You might need to adjust note_input_path to your actual location.
    note_input_path = os.path.join(raw_data_path, "/root/autodl-tmp/data/mimic-iv-note/2.2/note")
    discharge = pd.read_csv(os.path.join(note_input_path, 'discharge.csv'))
    print("discharge shape:", discharge.shape)
    print(discharge.head())

    discharge["text"] = discharge["text"].apply(extract_discharge_instructions)
    discharge = discharge[["subject_id", "hadm_id", "text"]].dropna()
    print("discharge instructions shape after extraction:", discharge.shape)

    discharge.to_csv(os.path.join(output_path, 'discharge_tmp.csv'), index=False)

    # ----------------------------------------------------------------------------
    # 8. Discharge instructions (version 2)
    discharge_v2 = pd.read_csv(os.path.join(note_input_path, 'discharge.csv'))
    print("discharge_v2 shape:", discharge_v2.shape)
    print(discharge_v2.head())

    discharge_v2["text"] = discharge_v2["text"].apply(extract_brief_hospital_course)
    discharge_v2 = discharge_v2[["subject_id", "hadm_id", "text"]].dropna()
    print("discharge_v2 shape after extraction:", discharge_v2.shape)

    discharge_v2.to_csv(os.path.join(output_path, 'discharge_v2_tmp.csv'), index=False)

    # ----------------------------------------------------------------------------
    # 9. Normalize lab values
    labevents = pd.read_csv(os.path.join(output_path, 'labevents_tmp.csv'))
    print("Loaded labevents_tmp for normalization:", labevents.shape)
    print(labevents.head())

    # Create normalized values (Z-score per itemid)
    labevents["normalized_valuenum"] = labevents.groupby('itemid')['valuenum']\
        .transform(lambda x: (x - x.mean()) / x.std())

    # Drop rows where normalization might fail (e.g., single or constant values for itemid)
    labevents = labevents.dropna(subset=["normalized_valuenum"])
    print("Labevents shape after dropping NaNs in normalized_valuenum:", labevents.shape)

    labevents.to_csv(os.path.join(output_path, 'labevents_tmp.csv'), index=False)
    print("All processing complete.")


if __name__ == "__main__":
    main()