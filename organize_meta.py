import os
import pandas as pd

# Load clinical data
clinical_data = pd.read_excel("s3://ella-dlbiomarkers/D2_Clinical")

# Define directory and files
dicom_dir = "/home/ellaquan/Project/Extracted_DICOM_D2"

# Define view and laterality mapping
view_map = {
    '1-1': ('L', 'CC'),
    '1-2': ('L', 'MLO'),
    '1-3': ('R', 'CC'),
    '1-4': ('R', 'MLO')
}

# Create a list to store the new metadata
organized_data_with_clinical = []

# Iterate over all DICOM files in the directory
for file_name in os.listdir(dicom_dir):
    if file_name.endswith(".dcm"):
        # Extract the ID and view information from the file name
        parts = file_name.split('_')
        patient_id = parts[0]  # "D2-0001"
        exam_id = parts[1].replace('.dcm', '')  # "1-1", "1-2", etc.
        
        # Get the laterality and view based on exam_id
        laterality, view = view_map.get(exam_id, ('Unknown', 'Unknown'))
        
        # Match with the clinical data
        match = clinical_data[clinical_data['ID1'] == patient_id]
        
        if not match.empty:
            # Add the relevant clinical data columns to the metadata
            for _, row in match.iterrows():
                organized_data_with_clinical.append({
                    'patient_id': patient_id,
                    'exam_id': exam_id,
                    'laterality': laterality,
                    'view': view,
                    'file_path': os.path.join(dicom_dir, file_name),
                    'years_to_cancer': 0,
                    'years_to_last_followup': 10,
                    'split_group': 'train',
                    'Age': row['Age'],
                    'number': row['number'],
                    'abnormality': row['abnormality'],
                    'classification': row['classification'],
                    'subtype': row['subtype'],
                    'target': row['target']
                })

# Convert to DataFrame
organized_df_with_clinical = pd.DataFrame(organized_data_with_clinical)

# Save the organized data to XLSX
output_xlsx_path = "/Users/ellaquan/Project/organized_metadata_with_clinical.xlsx"
organized_df_with_clinical.to_excel(output_xlsx_path, index=False)

print(f"Organized metadata with clinical data saved to {output_xlsx_path}")
