import os
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
from io import BytesIO
import json
import boto3
from sklearn.model_selection import train_test_split

# Load configuration file
with open('config.json') as config_file:
    config = json.load(config_file)

# Configuration settings
s3_bucket_name = config.get("s3_bucket_name")
file_path_excel = config.get("file_path_excel")
data_percentage = config.get("data_percentage", 1)
train_test_split_percentage = config.get("train_test_split_percentage", 0.8)
output_dir_train = config.get("output_dir_train")
output_dir_test = config.get("output_dir_test")
views = config.get("views", ["CC", "MLO", "US"])
output_excel_name_train = config.get("output_excel_name_train", "train_labels.xlsx")
output_excel_name_test = config.get("output_excel_name_test", "test_labels.xlsx")

# Initialize S3 client
s3 = boto3.client('s3')

# Load metadata
obj = s3.get_object(Bucket=s3_bucket_name, Key=file_path_excel)
metadata = pd.read_excel(BytesIO(obj['Body'].read()))

# Sample a subset of the data
metadata = metadata.sample(frac=data_percentage, random_state=42).reset_index(drop=True)

# Split metadata
train_metadata, test_metadata = train_test_split(metadata, train_size=train_test_split_percentage, random_state=42)

def convert_dcm_to_png_from_s3(s3_key, output_png_path):
    """Convert a DICOM file from S3 to PNG and save locally."""
    try:
        obj = s3.get_object(Bucket=s3_bucket_name, Key=s3_key)
        dcm = pydicom.dcmread(BytesIO(obj['Body'].read()))
        pixel_array = dcm.pixel_array
        plt.imsave(output_png_path, pixel_array, cmap='gray')
        return True
    except Exception as e:
        print(f"Error converting {s3_key}: {e}")
        return False

def process_metadata(metadata, output_dir):
    csv_data = []
    for _, row in metadata.iterrows():
        patient_id = row['patient_id'].replace('D2-', '').zfill(4)
        subtype = row['subtype']

        for view in views:
            file_path = row.get('file_path')
            if pd.isna(file_path):
                print(f"No file path found for {view} view in patient {patient_id}. Skipping.")
                continue

            # Define the output path for the PNG file
            output_png_path = os.path.join(output_dir, f'{patient_id}_{view}.png')

            # Attempt conversion from S3
            if convert_dcm_to_png_from_s3(file_path, output_png_path):
                entry = {
                    'patient_id': patient_id,
                    f'{view}_file': output_png_path,
                    'subtype': subtype
                }
                csv_data.append(entry)
            
    return csv_data

# Process and save train metadata
train_csv_data = process_metadata(train_metadata, output_dir_train)
df_train = pd.DataFrame(train_csv_data)
df_train_pivot = df_train.pivot_table(index='patient_id', values=[f'{view}_file' for view in views if f'{view}_file' in df_train.columns], aggfunc='first').reset_index()
df_train_pivot['subtype'] = train_metadata.groupby('patient_id')['subtype'].first().values
df_train_pivot['target'] = df_train_pivot['subtype'].apply(lambda x: 1 if x == "Luminal A" else 0)

# Process and save test metadata
test_csv_data = process_metadata(test_metadata, output_dir_test)
df_test = pd.DataFrame(test_csv_data)
df_test_pivot = df_test.pivot_table(index='patient_id', values=[f'{view}_file' for view in views if f'{view}_file' in df_test.columns], aggfunc='first').reset_index()
df_test_pivot['subtype'] = test_metadata.groupby('patient_id')['subtype'].first().values
df_test_pivot['target'] = df_test_pivot['subtype'].apply(lambda x: 1 if x == "Luminal A" else 0)

print(f"DICOM files from S3 converted and saved in {output_dir_train} for training and {output_dir_test} for testing.")
