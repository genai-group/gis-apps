#!/usr/bin/python

#%%
import os

print(f"Loading all required Python modules...")

from config.config_variables import *
from config.config_modules import *
from config.config_clients import *
from config.config_functions import *

# Loading the YAML files from the config/templates directory
template_dir = './config/templates'


################################################
####    Loading the Synthetic Data Files    ####
################################################

if GIS_ENVIRONMENT == 'flask-local':
    data_dir = 'app/config/data/synthetic'

if GIS_ENVIRONMENT == 'local':
    data_dir = './config/data/synthetic'

try:
    manifest_data_0 = open_file(f"{data_dir}/fake_airline_manifest_0_hours.json")
    manifest_data_24 = open_file(f"{data_dir}/fake_airline_manifest_24_hours.json")
    manifest_data_72 = open_file(f"{data_dir}/fake_airline_manifest_72_hours.json")
    # manifest_data_0 = json.loads(open(f"{data_dir}/fake_airline_manifest_0_hours.json", "r").read())   
    # manifest_data_24 = json.loads(open(f"{data_dir}/fake_airline_manifest_24_hours.json", "r").read())
    # manifest_data_72 = json.loads(open(f"{data_dir}/fake_airline_manifest_72_hours.json", "r").read()) 
    print(f"Successfully loaded data for the fake_airline_manifest files (0, 24 adn 72 hours).")
except Exception as e:
    print(f"Error loading data: {e}")
    raise

# Reading in the fake_passport data
try:
    fake_passport_0 = open_file(f"{data_dir}/fake_passport_0_hours.json")
    fake_passport_24 = open_file(f"{data_dir}/fake_passport_24_hours.json")
    fake_passport_72 = open_file(f"{data_dir}/fake_passport_72_hours.json")
    # fake_passport_hour_0 = json.loads(open(f"{data_dir}/fake_passport_hour_0.json", "r").read())
    # fake_passport_hour_24 = json.loads(open(f"{data_dir}/fake_passport_hour_24.json", "r").read())
    # fake_passport_hour_72 = json.loads(open(f"{data_dir}/fake_passport_hour_72.json", "r").read())
    print(f"Successfully loaded data for the fake_passport data files (0, 24 adn 72 hours).")
except Exception as e:
    print(f"Error loading fake_passport: {e}")
    raise

# Reading in the fake_customs_report data at the 0, 24, and 72 hours
try:
    fake_customs_report_hour_0 = open_file(f"{data_dir}/fake_customs_report_0_hours.json")
    fake_customs_report_hour_24 = open_file(f"{data_dir}/fake_customs_report_24_hours.json")
    fake_customs_report_hour_72 = open_file(f"{data_dir}/fake_customs_report_72_hours.json")
    # fake_customs_report_hour_0 = json.loads(open(f"{data_dir}/fake_customs_report_hour_0.json", "r").read())
    # fake_customs_report_hour_24 = json.loads(open(f"{data_dir}/fake_customs_report_hour_24.json", "r").read())
    # fake_customs_report_hour_72 = json.loads(open(f"{data_dir}/fake_customs_report_hour_72.json", "r").read())
    print(f"Successfully loaded data for the fake_customs_report data files (0, 24 adn 72 hours).")
except Exception as e:
    print(f"Error loading fake_customs_report: {e}")
    raise


###############################################
####    Loading the External Data Files    ####
###############################################

if GIS_ENVIRONMENT == 'flask-local':
    data_dir = 'app/config/data/external'

if GIS_ENVIRONMENT == 'local':
    data_dir = 'config/data/external'

# Global Terrorism Index
try:
    gti = open_file(f"{data_dir}/global_terrorism_index.csv")
    # gti = pd.read_csv(f"{data_dir}/global_terrorism_index.csv")
    gti.set_index('country', inplace=True)
    gti_dict = gti.to_dict('index')
    print(f"Successfully loaded data for the Global Terrorism Index data file.")
except Exception as e:
    print(f"Errors loading the global terrorism index")
    raise

# Reading ISO json object
try:
    iso_data = open_file(f"{data_dir}/iso.json", "r").read()
    # iso_data = json.loads(open(f"{data_dir}/iso.json", "r").read())
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
    government_ids = open_file(f"{data_dir}/International Government IDs - Sheet1.csv")
    # government_ids = pd.read_csv(f"{data_dir}/International Government IDs - Sheet1.csv")
    government_ids = government_ids.to_dict(orient='records')
    government_ids = {obj['Country']:{'Name':obj['Name'], 'Description':obj['Description']} for obj in government_ids}
    print(f"Successfully loaded data for the International Government IDs.")
except Exception as e:
    print(f"Error loading International Government IDs: {e}")
    raise

# IATA International Airport Codes
try:
    airport_codes = open_file(f"{data_dir}/international_airport_codes.csv")
    # airport_codes = pd.read_csv(f"{data_dir}/international_airport_codes.csv")
    airport_codes = airport_codes.to_dict('records')
except Exception as e:
    print(f"Errors reading in IATA international airport codes.")    

# World Cities
try:
    world_cities = open_file(f"{data_dir}/worldcities.csv")
    # world_cities = pd.read_csv(f"{data_dir}/worldcities.csv")   
except Exception as e:
    print(f"Errors reading in world cities.")


######################################################
####    Loading the Customer Sample Data Files    ####
######################################################

if GIS_ENVIRONMENT == 'flask-local':
    data_dir = 'app/config/data/sample_data'

if GIS_ENVIRONMENT == 'local':
    data_dir = 'config/data/sample_data'

