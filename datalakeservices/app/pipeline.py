#!/usr/bin/python

#%%
from config.init import *

#%% 
# Reading in the data
data = manifest_data_0

#%%
# Reading in the template file
template_dir = './config/templates'
parse_config = yaml.safe_load(open(f"{template_dir}/fake_airline_manifest.yml", "r").read())

#%%
# Filter data file
if 'primary_key' in parse_config['template']:
    data = manifest_data_0[parse_config['template']['primary_key']]

#%%
# Rename Fields
if 'rename_fields' in parse_config:
    data = rename_properties(data, parse_config['rename_fields'])

#%%
# Add hashes to each object
data = hashify(data, namespace='passenger')

#%%


#%%
passengers = manifest_data_0['passenger_info']


#%%

# Building the parser:

data = manifest_data_0['passenger_info']

"""
{
    'full_name': 'Ryan Smith',
    'gender': 'Male',
    'date_of_birth': '1960-03-20',
    'country_of_citizenship': 'Ireland',
    'government_id_type': 'Passport',
    'passport_information': '217-45-9511'
}

"""

schema = """full_name [name], country_of_citizenship [country], passport_information [passport_id]; passport_id -> name, passport_id -> country"""

# Split into statements
statements = schema.split(';')

statement = statements[0]

for statement in statements:
    commands = statement.split(',')
    for command in commands:
        print(f"command: {command.strip()}")
        # search for the namespace
        if '[' in command:
            namespace = command.split('[')[1].split(']')[0]
            variable = command.split('[')[0].strip()
            print(f"namespace: {namespace}")
            # print(f"variable: {variable}")
        if '->' in command:
            # This is a relation
            parent = command.split('->')[0].strip()
            # print(f"relation: {parent}")
            # This is an attribute
            children = command.split('->')[1].strip()
            children = children.split(',')
            print(f"children: {children}")
            for child in children:
                define_relation = 'has_' + child.strip().lower().replace(' ','_')
                print(f"parent: {parent} => {define_relation} => {child}")
                define_relation = None

# %%
