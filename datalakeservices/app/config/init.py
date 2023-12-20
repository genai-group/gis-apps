#!/usr/bin/python

#%%
import os

print(f"Loading all required Python modules...")

from config.variables import *
from config.modules import *
from config.clients import *
from config.functions import *

# Loading the Template Yaml File(s)
# from config.templates.data_ingest import *

# Loading the YAML files from the config/templates directory
template_dir = './config/templates'
data_ingest = yaml.safe_load(open(f"{template_dir}/data_ingest.yml", "r").read())

# airline_flight_manifest_0_hours
try:
    data_dir = './config/data'
    manifest_data_0 = json.loads(open(f"{data_dir}/airline_flight_manifest_0_hours.json", "r").read())   
    manifest_data_24 = json.loads(open(f"{data_dir}/airline_flight_manifest_24_hours.json", "r").read())
    manifest_data_72 = json.loads(open(f"{data_dir}/airline_flight_manifest_72_hours.json", "r").read()) 
    print(f"Successfully loaded data for the manifest data files (0, 24 adn 72 hours).")
except Exception as e:
    print(f"Error loading data: {e}")
    raise

# Reading in the fake_passport data
try:
    fake_passport_hour_0 = json.loads(open(f"{data_dir}/fake_passport_hour_0.json", "r").read())
    fake_passport_hour_24 = json.loads(open(f"{data_dir}/fake_passport_hour_24.json", "r").read())
    fake_passport_hour_72 = json.loads(open(f"{data_dir}/fake_passport_hour_72.json", "r").read())
    print(f"Successfully loaded data for the fake_passport data files (0, 24 adn 72 hours).")
except Exception as e:
    print(f"Error loading fake_passport: {e}")
    raise

# Reading in the fake_customs_report data at the 0, 24, and 72 hours
try:
    fake_customs_report_hour_0 = json.loads(open(f"{data_dir}/fake_customs_report_hour_0.json", "r").read())
    fake_customs_report_hour_24 = json.loads(open(f"{data_dir}/fake_customs_report_hour_24.json", "r").read())
    fake_customs_report_hour_72 = json.loads(open(f"{data_dir}/fake_customs_report_hour_72.json", "r").read())
    print(f"Successfully loaded data for the fake_customs_report data files (0, 24 adn 72 hours).")
except Exception as e:
    print(f"Error loading fake_customs_report: {e}")
    raise

# Reading ISO json object
try:
    iso_data = json.loads(open(f"{data_dir}/iso.json", "r").read())
    print(f"Successfully loaded data for the ISO data file.")
except Exception as e:
    print(f"Error loading ISO data: {e}")
    raise

# Reading in the International Government IDs - Sheet1.csv
try:
    government_ids = pd.read_csv(f"{data_dir}/International Government IDs - Sheet1.csv")
    government_ids = government_ids.to_dict(orient='records')
    government_ids = {obj['Country']:{'Name':obj['Name'], 'Description':obj['Description']} for obj in government_ids}
    print(f"Successfully loaded data for the International Government IDs.")
except Exception as e:
    print(f"Error loading International Government IDs: {e}")
    raise
