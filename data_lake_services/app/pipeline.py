#!/usr/bin/python

#%%
from config.config_init import *

#####################################
####    Start of the Pipeline    ####
#####################################

# New Data
#%%
data = manifest_data_0

#%%
# Airline
template_data_directory = './config/templates/data'
parse_config = yaml.safe_load(open(f"{template_data_directory}/fake_airline_manifest_flight.yml", "r").read())

#%%
# Process template
template = process_template(parse_config)

#%% 
# Process data
data = process_data(data, parse_config, template)

#%%
# Load data
load_data(data, parse_config, template)


#%%
# Airline
parse_config = yaml.safe_load(open(f"{template_data_directory}/fake_airline_manifest_passengers.yml", "r").read())

data = manifest_data_0
passenger_info = data['passenger_info']

# Adding flight manifest and flight number to each persons data
new_data = []
for obj in passenger_info:
    obj['flight_manifest_id'] = data['flight_information']['flight_manifest_id']
    obj['flight_number'] = data['flight_information']['flight_number']
    obj['time_to_departure'] = data['flight_information']['time_to_departure']
    new_data.append(obj)

data = new_data
del new_data

#%%
# Process template, data, and load data
template = process_template(parse_config)
data = process_data(data, parse_config, template)
load_data(data, parse_config, template)

##################################
####    Flight Information    ####
##################################

#%%
# Flight
parse_config = yaml.safe_load(open(f"{template_data_directory}/fake_airline_manifest_flight.yml", "r").read())

# Filter data file
if 'primary_key' in parse_config['template']:
    data = manifest_data_0[parse_config['template']['primary_key']]
    if not isinstance(data, list):
        data = [data]       

# Drop fields
drop_fields = filter_func(lambda x: 'drop' in x.keys(), parse_config['fields'])
drop_fields = filter_func(lambda x: str(x['drop']).lower() == 'true', drop_fields)
drop_fields = map_func(lambda x: x['field'], drop_fields)

if len(drop_fields) > 0:
    clean_objects = []
    for obj in data:
        clean_objects.append({k:v for k,v in obj.items() if k not in drop_fields})
    data = clean_objects

# Hashify
data = hashify(data, _namespace='flight_manifest', parse_config=parse_config)
if not isinstance(data, list):
    data = [data]       


#########################
####    Load Data    ####
#########################




#####################################
####    Passenger Information    ####
#####################################


#%%
# Passengers
parse_config = yaml.safe_load(open(f"{template_data_directory}/fake_airline_manifest_passengers.yml", "r").read())


# Filter data file
if 'primary_key' in parse_config['template']:
    data = manifest_data_0[parse_config['template']['primary_key']]
    if not isinstance(data, list):
        data = [data]       

# Getting ready to copy/paste data

data = manifest_data_0
passenger_info = data['passenger_info']

# Adding flight manifest and flight number to each persons data
new_data = []
for obj in passenger_info:
    obj['flight_manifest_id'] = data['flight_information']['flight_manifest_id']
    obj['flight_number'] = data['flight_information']['flight_number']
    obj['time_to_departure'] = data['flight_information']['time_to_departure']
    new_data.append(obj)

data = new_data
del new_data

# Drop fields
drop_fields = filter_func(lambda x: 'drop' in x.keys(), parse_config['fields'])
drop_fields = filter_func(lambda x: str(x['drop']).lower() == 'true', drop_fields)
drop_fields = map_func(lambda x: x['field'], drop_fields)

if len(drop_fields) > 0:
    clean_objects = []
    for obj in data:
        clean_objects.append({k:v for k,v in obj.items() if k not in drop_fields})
    data = clean_objects

# Hashify
data = hashify(data, _namespace='passenger', parse_config=parse_config)
if not isinstance(data, list):
    data = [data]       


#########################
####    Load Data    ####
#########################

#%%
# Load data into MongoDB Collection
mongo_collection.insert_many(data)

# Load the objecty _guids into the Neo4j graph
#%%
neo4j_objects = map_func(lambda x: {k:v for k,v in x.items() if k in ['_guid']}, data)
updated_neo4j_objects = []
for obj in neo4j_objects:
    obj['_source'] = 0
    updated_neo4j_objects.append(obj)

neo4j_objects = copy.deepcopy(updated_neo4j_objects)
del updated_neo4j_objects

if len(neo4j_objects) > 0:
    # buiding the load statements
    load_statement = ', '.join([f'`{k}`: line.`{k}`' for k in neo4j_objects[0].keys() if k not in ['_edge']])
    load_statement = f'UNWIND $objects AS line MERGE (obj:Object {{ {load_statement} }})'

with neo4j_client.session() as session:
    session.run(load_statement, objects=neo4j_objects)
    logging.info(f'Loaded Objects into Neo4j: {neo4j_objects}')
    print(f"Loaded Objects into Neo4j: {len(neo4j_objects)}")

# Entities
entities = filter_func(lambda x: 'is_entity' in x.keys(), parse_config['fields'])
entities = filter_func(lambda x: str(x['is_entity']).lower() == 'true', entities)   
entities = map_func(lambda x: x['field'], entities)

# Making sure data is a list
if not isinstance(data, list):
    data = [data]

if len(entities) > 0:
    for entity in entities:
        # Loading an array of entities into Neo4j
        neo4j_objects = map_func(lambda x: {k:v for k,v in x.items() if k in ['_guid', entity]}, data)
        neo4j_objects = standardize_objects(neo4j_objects, parse_config)
        neo4j_objects = prepare_entities_for_load(neo4j_objects, entity, parse_config, include_created_at=False)
        pp(neo4j_objects)
        if len(neo4j_objects) > 0:
            # buiding the load statements
            load_statement = ', '.join([f'`{k}`: line.`{k}`' for k in neo4j_objects[0].keys() if k not in ['_edge', '_source']])
            load_statement = f'UNWIND $objects AS line MERGE (obj:Object {{ {load_statement} }})'
            # Load objects given the schema
            with neo4j_client.session() as session:
                session.run(load_statement, objects=neo4j_objects)
                logging.info(f'Loaded Objects into Neo4j: {neo4j_objects}')
                print(f"Loaded Objects into Neo4j: {len(neo4j_objects)}")

            # Load Source Neo4j Relationships
            neo4j_edges = map_func(lambda x: {k:v for k,v in x.items() if k in ['_guid','_source','_edge', '_created_at']}, neo4j_objects)

            # replace _edge with "has_source" for _edge in all objects in neo4j_edges
            updated_neo4j_edges = []
            for obj in neo4j_edges:
                obj['_edge'] = 'has_source'
                updated_neo4j_edges.append(obj)

            neo4j_edges = copy.deepcopy(updated_neo4j_edges)
            del updated_neo4j_edges

            # Load Neo4j Relationships
            load_statement = ', '.join([f'`{k}`: line.`{k}`' for k in neo4j_edges[0].keys()])
            # load_statement = 'UNWIND $objects AS line MATCH (obj1:Object{_guid:line.`_source`}) MATCH (obj2:Object{_guid:line.`_guid`}) MERGE (obj1)-[owns:OWNS{' + str(load_statement) + '}]-(obj2) RETURN *'
            # load_statement = 'UNWIND $objects AS line MATCH (obj1:Object{_guid:line.`_source`}) MATCH (obj2:Object{_guid:line.`_guid`}) MERGE (obj1)-[owns:OWNS{' + str(load_statement) + '}]-(obj2) RETURN *'
            load_statement = 'UNWIND $objects AS line MATCH (obj1:Object{_guid:line.`_guid`}) MATCH (obj2:Object{_guid:line.`_source`}) MERGE (obj1)-[owns:' + str(neo4j_edges[0]['_edge']) + '{' + str(load_statement) + '}]-(obj2) RETURN *'

            with neo4j_client.session() as session:
                session.run(load_statement, objects=neo4j_edges)
                logging.info(f'Loaded Objects into Neo4j: {neo4j_edges}')


#%%
                
alias_fields = filter_func(lambda x: 'alias' in x.keys(), parse_config['fields'])
alias_fields = filter_func(lambda x: len(str(x['alias'])) > 0, alias_fields)
alias_fields = {x['field']: x['alias'] for x in alias_fields}

standardized_fields = filter_func(lambda x: 'standardize' in x.keys(), parse_config['fields'])
standardized_fields = {x['field']: x['standardize'] for x in standardized_fields}

#%%
# Load edges from the parse_config file
edge_objects = parse_config['edges']

edges_to_load = []
for edge_object in edge_objects:
    parents = edge_object['parents']
    children = edge_object['children']
    edge_type = edge_object['type']
    edge_direction = edge_object['direction']
    if 'properties' in edge_object.keys():
        edge_properties = edge_object['properties']
    else:
        edge_properties = []

    for parent in parents:
        for child in children:
            if str(edge_direction).lower() == 'out':
                # if child in alias_fields.keys():
                #     has_variable = 'has_' + str(alias_fields[child]).lower().replace(' ', '_')
                # else:
                has_variable = 'has_' + str(child).lower().replace(' ', '_')
                edge_triple = {'_child': child, '_parent': parent, '_edge':has_variable}
                for property in edge_properties:
                    edge_triple[property] = property #edge_object['properties'][property]
                edges_to_load.append(edge_triple)

# Load the edges_to_load into the Neo4j database
#%%
final_triples_to_load = []
for triple in edges_to_load:
    for object in data:
        temp_object = {}
        if triple['_child'] in standardized_fields.keys():
            child = globals()[standardized_fields[triple['_child']]](object[triple['_child']])
        else:
            child = object[triple['_child']]
        child_guid = hashify(child, _namespace=triple['_child'], parse_config=parse_config)['_guid']
        if triple['_parent'] in standardized_fields.keys():
            parent = globals()[standardized_fields[triple['_parent']]](object[triple['_parent']])
        else:
            parent = object[triple['_parent']]
        parent_guid = hashify(parent, _namespace=triple['_parent'], parse_config=parse_config)['_guid']
        temp_object = {'_child': child_guid, '_parent': parent_guid, '_edge': triple['_edge']}
        final_triples_to_load.append(temp_object)

        print(f"parent: {parent}")
        print(f"parent_guid: {parent_guid}")
        print(f"triple['_edge']: {triple['_edge']}")
        print(f"child: {child}")
        print(f"child_guid: {child_guid}")

#%%
# Load Neo4j Relationships by edge name
for _edge in set(map_func(lambda x: x['_edge'], final_triples_to_load)):
    neo4j_relationships = filter_func(lambda x: x['_edge'] == _edge, final_triples_to_load)
    load_statement = ', '.join([f'`{k}`: line.`{k}`' for k in neo4j_relationships[0].keys() if k not in ['_child','_parent']])
    load_statement = 'UNWIND $objects AS line MATCH (obj1:Object{_guid:line.`_parent`}) MATCH (obj2:Object{_guid:line.`_child`}) MERGE (obj1)-[owns:' + str(neo4j_relationships[0]['_edge']) + '{' + str(load_statement) + '}]-(obj2) RETURN *'
    pp(f"load_statement: {load_statement}")

    with neo4j_client.session() as session:
        session.run(load_statement, objects=neo4j_relationships)
        logging.info(f'Loaded Objects into Neo4j: {neo4j_relationships}')



########################
####    OLD CODE    ####
########################



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

