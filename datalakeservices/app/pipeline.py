#!/usr/bin/python

#%%
from config.init import *

#%%
passengers = manifest_data_0['passenger_info']
hashify(passengers, namespace='passenger')

#%%

# Building the parser:

data = manifest_data_0['passenger_info']


{
    'full_name': 'Ryan Smith',
    'gender': 'Male',
    'date_of_birth': '1960-03-20',
    'country_of_citizenship': 'Ireland',
    'government_id_type': 'Passport',
    'passport_information': '217-45-9511'
}

schema = '""full_name [name], country_of_citizenship [country], passport_information [passport_id]; passport_id -> full_name, country""'

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
            print(f"variable: {variable}")
        if '->' in command:
            # This is a relation
            parent = command.split('->')[0].strip()
            print(f"relation: {parent}")
            # This is an attribute
            children = command.split('->')[1].strip()
            children = children.split(',')
            for child in children:
                print(f"paernt: {parent}: child {child}")
