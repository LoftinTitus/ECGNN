import os
import csv

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

