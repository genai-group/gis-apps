#!/usr/bin/python

#%%
from config.init import *

#%% 
# Reading in the data
data = manifest_data_0

#%%
# Getting ready to copy/paste data
passenger_info = data['passenger_info']

# Adding flight manifest and flight number to each persons data
new_data = []
for obj in passenger_info:
    obj['flight_manifest_id'] = data['flight_information']['flight_manifest_id']
    obj['flight_number'] = data['flight_information']['flight_number']
    obj['time_to_departure'] = data['flight_information']['time_to_departure']
    new_data.append(obj)

#%%
# Reading in the template file
template_dir = './config/templates'

##################################
####    Flight Information    ####
##################################

#%%
# Flight
parse_config = yaml.safe_load(open(f"{template_dir}/fake_airline_manifest_flight copy.yml", "r").read())

#%%
# Filter data file
if 'primary_key' in parse_config['template']:
    data = manifest_data_0[parse_config['template']['primary_key']]
    if not isinstance(data, list):
        data = [data]       

#%%
# Drop fields
drop_fields = filter_func(lambda x: 'drop' in x.keys(), parse_config['fields'])
drop_fields = filter_func(lambda x: str(x['drop']).lower() == 'true', drop_fields)
drop_fields = map_func(lambda x: x['field'], drop_fields)

if len(drop_fields) > 0:
    clean_objects = []
    for obj in data:
        clean_objects.append({k:v for k,v in obj.items() if k not in drop_fields})
    data = clean_objects

#%%
# Hashify
data = hashify(data, _namespace='flight_manifest', parse_config=parse_config)
if not isinstance(data, list):
    data = [data]       

#%%
# Load data into MongoDB Collection
mongo_collection.insert_many(data)

#%%
# Entities
entities = filter_func(lambda x: 'is_entity' in x.keys(), parse_config['fields'])
entities = filter_func(lambda x: str(x['is_entity']).lower() == 'true', entities)   
entities = map_func(lambda x: x['field'], entities)

# Making sure data is a list
if not isinstance(data, list):
    data = [data]

#%%
if len(entities) > 0:
    for entity in entities:
        # Loading an array of entities into Neo4j
        neo4j_objects = map_func(lambda x: {k:v for k,v in x.items() if k in ['_guid', entity]}, data)
        neo4j_objects = standardize_objects(neo4j_objects, parse_config)
        neo4j_objects = prepare_objects_for_load(neo4j_objects, entity, parse_config)
        pp(neo4j_objects)
        if len(neo4j_objects) > 0:
            # buiding the load statements
            load_statement = ', '.join([f'`{k}`: line.`{k}`' for k in neo4j_objects[0].keys() if k not in ['_edge']])
            load_statement = f'UNWIND $objects AS line MERGE (obj:Object {{ {load_statement} }})'
            # Load objects given the schema
            with neo4j_client.session() as session:
                session.run(load_statement, objects=neo4j_objects)
                logging.info(f'Loaded Objects into Neo4j: {neo4j_objects}')
                print(f"Loaded Objects into Neo4j: {len(neo4j_objects)}")

            # Load Neo4j Relationships
            neo4j_edges = map_func(lambda x: {k:v for k,v in x.items() if k in ['_guid','_source','_edge', '_created_at']}, neo4j_objects)

            # Load Neo4j Relationships
            #%%
            load_statement = ', '.join([f'`{k}`: line.`{k}`' for k in neo4j_edges[0].keys()])
            load_statement = 'UNWIND $objects AS line MATCH (obj1:Object{_guid:line.`_source`}) MATCH (obj2:Object{_guid:line.`_guid`}) MERGE (obj1)-[owns:HAS{' + str(load_statement) + '}]-(obj2) RETURN *'

            with neo4j_client.session() as session:
                session.run(load_statement, objects=neo4j_edges)
                logging.info(f'Loaded Objects into Neo4j: {neo4j_edges}')






#%%
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



#####################################
####    Passenger Information    ####
#####################################
    
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

