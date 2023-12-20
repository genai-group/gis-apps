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
    data_0 = json.loads(open(f"{data_dir}/airline_flight_manifest_0_hours.json", "r").read())   
    data_24 = json.loads(open(f"{data_dir}/airline_flight_manifest_24_hours.json", "r").read())
    data_72 = json.loads(open(f"{data_dir}/airline_flight_manifest_72_hours.json", "r").read()) 
    print(f"Successfully loaded data for the manifest files.")
except Exception as e:
    print(f"Error loading data: {e}")
    raise
