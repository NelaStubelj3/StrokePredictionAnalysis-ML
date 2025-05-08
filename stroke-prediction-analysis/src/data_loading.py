import pandas as pd
import os   
def load_data():
    file_path = os.path.abspath("../data/raw/healthcare-dataset-stroke-data.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None