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
# template_dir = './config/templates'
template_dir = './config/templates'
# data_ingest = yaml.safe_load(open(f"{template_dir}/data_ingest.yml", "r").read())

# airline_flight_manifest_0_hours
try:
    print(f"NEW DIRECTORY: {template_dir}")
    print(f"files: {os.listdir(template_dir)}")
    print(f"files: {os.listdir(template_dir)}")
except:
    pass

try:
    data_dir = 'config/data'
    manifest_data_0 = json.loads(open(f"{data_dir}/fake_airline_manifest_0_hours.json", "r").read())   
    manifest_data_24 = json.loads(open(f"{data_dir}/fake_airline_manifest_24_hours.json", "r").read())
    manifest_data_72 = json.loads(open(f"{data_dir}/fake_airline_manifest_72_hours.json", "r").read()) 
    print(f"Successfully loaded data for the fake_airline_manifest files (0, 24 adn 72 hours).")
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

# Global Terrorism Index
try:
    gti = pd.read_csv(f"{data_dir}/global_terrorism_index.csv")
    gti.set_index('country', inplace=True)
    gti_dict = gti.to_dict('index')
    print(f"Successfully loaded data for the Global Terrorism Index data file.")
except Exception as e:
    print(f"Errors loading the global terrorism index")
    raise

# Reading ISO json object
try:
    iso_data = json.loads(open(f"{data_dir}/iso.json", "r").read())
    iso_data = pd.DataFrame(iso_data)
    iso_data.set_index('name', inplace=True)
    iso_dict = iso_data.to_dict('index')
    # Add global terrorist index
    for k,v in iso_dict.items():
        try:
            iso_dict[k]['terrorism_index'] = gti_dict[k]
        except:
            pass
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

# IATA International Airport Codes
try:
    airport_codes = pd.read_csv(f"{data_dir}/international_airport_codes.csv")
    airport_codes = airport_codes.to_dict('records')
except Exception as e:
    print(f"Errors reading in IATA international airport codes.")    

# World Cities
try:
    world_cities = pd.read_csv(f"{data_dir}/worldcities.csv")   
except Exception as e:
    print(f"Errors reading in world cities.")
