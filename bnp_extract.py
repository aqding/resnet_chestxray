import os
import pandas as pd
import datetime
import numpy as np
import math

import pickle
import os

import argparse
from scipy import stats as stats

parser = argparse.ArgumentParser()
# parser.add_argument('--feature', type=str, required=True, help='Feature to extract. Input bnp or creatinine')
# parser.add_argument('--avg_val_window', type=int, default=2, help='The window size to calculate average feature value')
# parser.add_argument('--norm', type=str, default='none', help='The type of normalization to use. Input none, zscore, log, or both.')
# # parser.add_argument('--norm', dest='norm', action='store_true')
# # parser.add_argument('--unnorm', dest='norm', action='store_false')
# # parser.set_defaults(norm=False)
# args=parser.parse_args()
parser.add_argument('--min_points', type=int, default = 1, help='Minimum number of BNP points within the window')
args = parser.parse_args()
print(args)

current_dir = os.path.dirname(__file__)

mimiccxr_path = '/data/vision/polina/projects/chestxray/data_v2/mimic-cxr-jpg-chest-radiographs-with-structured-labels-2.0.0/'\
'mimic-cxr-2.0.0-metadata.csv'

feature_path = os.path.join(current_dir, 'mimic_processing/mimic_analysis/data/bnp.pkl')

cxr_timestamps = pd.read_csv(mimiccxr_path)
cxr_timestamps['identifier'] = cxr_timestamps.apply(lambda row: \
                                                   'p'+str(row['subject_id'])+'_s'+str(row['study_id'])+'_'+row['dicom_id'], axis=1)

def convert_to_datetime(indate, intime):
    #Takes in a date and a time and converts it to a datetime object.

    date = str(indate)
    time = str(intime).split(".")[0]
    while len(time) < 6:
        time = '0' + time
    datetime_obj = datetime.datetime.strptime(date + " " + time, '%Y%m%d %H%M%S')

    return datetime_obj

def get_timestamps(cxr_timestamps):

    cxr_timestamps['datetime'] = cxr_timestamps.apply(lambda row: \
                                                             convert_to_datetime(row.StudyDate, row.StudyTime),
                                                             axis=1)
    cxr_times = cxr_timestamps.groupby('subject_id')['datetime'].apply(list).to_dict()
    return cxr_times

def get_identifiers(cxr_timestamps):
    cxr_identifiers = cxr_timestamps.groupby('subject_id')['identifier'].apply(list).to_dict()
    return cxr_identifiers

cxr_patient_dict = get_timestamps(cxr_timestamps)
cxr_identifier_dict = get_identifiers(cxr_timestamps)

cxr_sorted_identifiers = {}
for patient_id in cxr_patient_dict:
    times = np.array(cxr_patient_dict[patient_id])
    identifiers = np.array(cxr_identifier_dict[patient_id])
    
    sorted_indices = np.argsort(times)
    sorted_identifiers = identifiers[sorted_indices]
    
    cxr_sorted_identifiers[patient_id] = sorted_identifiers.tolist()
    
for patient_id in cxr_patient_dict:
    cxr_patient_dict[patient_id].sort()
    
print(f'Number of CXR patients: {len(cxr_patient_dict)}')

with open(feature_path, 'rb') as f:
    feature_data = pickle.load(f)

def patient_to_timestamp(data):
    patient_dict = {}
    for patient in data:
        timestamps = list(data[patient].keys())
        patient_dict[patient] = [datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S') for time in timestamps]
    
    return patient_dict
            
def timestamp_to_value(data):
    patient_value_dict = {}
    for patient in data:
        patient_value_dict[patient] = {}
        for timestamp in data[patient].keys():
            new_time = datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
            patient_value_dict[patient][new_time] = data[patient][timestamp]
    return patient_value_dict

feature_timestamps = patient_to_timestamp(feature_data)
feature_timestamp_values = timestamp_to_value(feature_data)

print(f'Number of patients: {len(feature_timestamps)}')

overlapping_patients = set()

for patient_id in cxr_patient_dict:
    if patient_id in feature_timestamps:
        overlapping_patients.add(patient_id)

print(f'Number of overlapping patients: {len(overlapping_patients)}')
# if 13031024 in overlapping_patients:
#     print("Found patient")
# else:
#     print("No patient")
all_bnp_vals = []
for patient in overlapping_patients:
    for timestamp in feature_timestamp_values[patient]:
        all_bnp_vals.append(feature_timestamp_values[patient][timestamp])
        
mean = np.mean(all_bnp_vals)
std = np.std(all_bnp_vals)
logged = np.log(all_bnp_vals)
mean_log_val = np.mean(logged)

def window_timestamps(time, timestamps):
    timestamp_seconds = [x for x in timestamps]
    differences = np.absolute(np.array(timestamp_seconds) - np.array(time))
    valid_times = []
    for i in range(len(differences)):
        if differences[i].total_seconds() < 60*60*24:
            valid_times.append(timestamps[i])
    return valid_times

data = {}

for patient_id in overlapping_patients:
    for i in range(len(cxr_patient_dict[patient_id])):
        window_val = window_timestamps(cxr_patient_dict[patient_id][i], feature_timestamps[patient_id])
        if len(window_val) > args.min_points - 1:
            window_val = [feature_timestamp_values[patient_id][timestamp] for timestamp in window_val]
            score = np.mean(window_val)
            data[cxr_sorted_identifiers[patient_id][i]] = score
#             data[cxr_sorted_identifiers[patient_id][i]] = np.log(score) - mean_log_val

print(len(data))

with open('bnp_data/48h_window/bnp_features_2_points.pkl', 'wb') as f:
    pickle.dump(data, f)

