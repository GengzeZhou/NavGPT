import os
import glob
import json

def merge_json_files(base_dir):
    merged_data = []

    # Iterate through subdirectories
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        
        # Check if the path is a directory
        if os.path.isdir(subdir_path):
            # Find all JSON files in the 'preds' subdirectory
            json_files = glob.glob(os.path.join(subdir_path, "preds", "*.json"))
            
            # Merge JSON data
            for file_path in json_files:
                with open(file_path, 'r') as json_file:
                    data = json.load(json_file)

                    # Merge the data from this file into the merged_data dictionary
                    for sample in data:
                        merged_data.append(sample)
                        

    # Save the merged JSON data to a file
    with open(os.path.join(base_dir, f"{exp_name}.json"), "w") as output_file:
        json.dump(merged_data, output_file, indent=4)

base_dir = "../datasets/R2R/exprs/"
exp_name = "4-R2R_val_unseen_instr"
path = os.path.join(base_dir, exp_name)
merge_json_files(path)