#!/usr/bin/python

#%%
import os

#%%
# GIS_ENVIRONMENT = 'local'
GIS_ENVIRONMENT = 'flask-local'

MINIO_ENDPOINT_URL = os.environ.get('MINIO_ENDPOINT_URL', 'localhost:9000')
MINIO_ACCESS_KEY = os.environ.get('MINIO_ACCESS_KEY', 'minio')
MINIO_SECRET_KEY = os.environ.get('MINIO_SECRET_KEY', 'minio123')

# %%
KAFKA_BOOTSTRAP_SERVERS = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
KAFKA_GROUP_ID = os.environ.get('KAFKA_GROUP_ID', 'my-group')
KAFKA_TOPIC = os.environ.get('KAFKA_TOPIC', 'my-topic')

# Neo4j Credentials
NEO4J_URI = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.environ.get('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD', 'password')

# TypeDB Credentials
TYPEDB_URI = os.environ.get('TYPEDB_HOST', 'localhost') + ':' + os.environ.get('TYPEDB_PORT', '1729')

UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')