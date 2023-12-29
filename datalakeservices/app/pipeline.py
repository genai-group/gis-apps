#!/usr/bin/python

#%%
from config.init import *

#%% 
# Reading in the data
data = manifest_data_0

#%%
# Reading in the template file
template_dir = './config/templates'

#%%
# Flight
parse_config = yaml.safe_load(open(f"{template_dir}/fake_airline_manifest_flight.yml", "r").read())

#%%
# Filter data file
if 'primary_key' in parse_config['template']:
    data = manifest_data_0[parse_config['template']['primary_key']]

#%%
# Passengers
parse_config = yaml.safe_load(open(f"{template_dir}/fake_airline_manifest_passengers.yml", "r").read())

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

# Load Passengers into Neo4j
neo4j_objects = map_func(lambda x: {k:v for k,v in x.items() if k in ['_guid','name']}, data)

load_statement = ', '.join([f'`{k}`: line.`{k}`' for k in neo4j_objects[0].keys() if k not in ['_relationship']])
load_statement = f'UNWIND $objects AS line MERGE (obj:Object {{ {load_statement} }})'

#%%
# Load objects given the schema
with neo4j_client.session() as session:
    session.run(load_statement, objects=neo4j_objects)
    logging.info(f'Loaded Objects into Neo4j: {neo4j_objects}')
    print(f"Loaded Objects into Neo4j: {len(neo4j_objects)}")


#%%
# Load data into MongoDB Collection
mongo_collection.insert_many(data)

#%%
# Standardize Fields
neo4j_objects = standardize_objects(data, parse_config)

#%%
# Load Neo4j Nodes
load_statement = ', '.join([f'`{k}`: line.`{k}`' for k in neo4j_objects[0].keys() if k not in ['_relationship']])
load_statement = f'UNWIND $objects AS line MERGE (obj:Object {{ {load_statement} }})'

#%%
# Load objects given the schema
with neo4j_client.session() as session:
    session.run(load_statement, objects=neo4j_objects)
    logging.info(f'Loaded Objects into Neo4j: {neo4j_objects}')
    print(f"Loaded Objects into Neo4j: {len(neo4j_objects)}")

#%%
# Load Neo4j Relationships
neo4j_relationships = map_func(lambda x: {k:v for k,v in x.items() if k in ['_guid','_source','_relationship']}, neo4j_objects)

# Load Neo4j Relationships
#%%
load_statement = ', '.join([f'`{k}`: line.`{k}`' for k in neo4j_relationships[0].keys()])
load_statement = 'UNWIND $objects AS line MATCH (obj1:Object{_guid:line.`_source`}) MATCH (obj2:Object{_guid:line.`_guid`}) MERGE (obj1)-[owns:HAS{' + str(load_statement) + '}]-(obj2) RETURN *'

with neo4j_client.session() as session:
    session.run(load_statement, objects=neo4j_relationships)
    logging.info(f'Loaded Objects into Neo4j: {neo4j_relationships}')



# Load objects given the schema

#%%
# Get entities
if 'entities' in parse_config.keys():
    entities = parse_config['entities']
    if len(entities) > 0:
        for entity in entities:
            print(f"entity: {entity}")
            'standardize_' in entity

#%%
def standardize_objects(objects List[Dict], parse_config: List[Dict]) -> List[Dict]:
    """
        Standardize objects based on the parse_config

        Args:
            objects (list): List of objects to standardize
            parse_config (dict): The parse config file

        Returns:
            list: List of standardized objects

    """
    # Assertions

    if 'standardize_fields' in parse_config.keys():            
        for standardize_field in parse_config['standardize_fields']:
            if standardize_field['transform'] in globals().keys():
                print(map_func(lambda x: {'namespace':standardize_field['field'], 'label': x[standardize_field['field']], '_guid': hashify(globals()[(standardize_field['transform'])](x[standardize_field['field']]), namespace=standardize_field['field'])['_guid'], '_source':x['_guid'], '_relationship':'has_' + standardize_field['field']}, data))

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
