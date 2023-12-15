#!/usr/bin/python

#%%
import os

MINIO_ENDPOINT_URL = os.environ.get('MINIO_ENDPOINT_URL', 'localhost:9000')
MINIO_ACCESS_KEY = os.environ.get('MINIO_ACCESS_KEY', 'minio')
MINIO_SECRET_KEY = os.environ.get('MINIO_SECRET_KEY', 'minio123')

# %%
