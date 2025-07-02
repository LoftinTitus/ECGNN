import os
import csv
import math
from math import sqrt


# Data loading functions
def load_csv(file_path):
    
    data = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
    for line in lines[1]:
        if line.strip():  # Check if the line is not empty
            values = line.strip().split(',')
            data.append(float(v) for v in values)
    return data
    

def load_all_data(folder_path):
    all_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            file_data = load_csv(file_path)
            all_data.extend(file_data)
    return all_data

# If needed: Add a missing data handling function, interpolating or filling missing values

def data_scaling(data):

    col2 = [row[1] for row in data]  # Extract the second column
    col3 = [row[2] for row in data]  # Extract the third column

    mean2 = sum(col2) / len(col2)
    mean3 = sum(col3) / len(col3)

    variance2 = sum((x - mean2) ** 2 for x in col2) / len(col2)
    variance3 = sum((x - mean3) ** 2 for x in col3) / len(col3)

    scaled_dataset = []
    for row in data:
        scaled_row2 = (row[1] - mean2) / sqrt(variance2) if variance2 != 0 else 0
        scaled_row3 = (row[2] - mean3) / sqrt(variance3) if variance3 != 0 else 0
        scaled_dataset.append([row[0], scaled_row2, scaled_row3])

    return scaled_dataset


def data_segmentation(scaled_dataset, segment_length):
    segments = []
    for i in range(0, len(scaled_dataset), segment_length):
        segment = scaled_dataset[i:i + segment_length]
        if len(segment) == segment_length:
            segments.append(segment)
    return segments