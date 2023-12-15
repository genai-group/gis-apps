#!/usr/bin/python

# Generic Modules
import os
import re
import json
import numpy as np
import pandas as pd
from pandas.api import types
from datetime import datetime, timedelta

# Multiprcessing
import ray

# MinIO
from minio import Minio
from minio.error import S3Error

# Pytest
import pytest

# Typing
from typing import List, Dict, Tuple, Union, Any, Optional, Coroutine, Callable, Awaitable, Iterable, AsyncIterable, TypeVar, Generic, Iterator

# Connect to PostgreSQL
import psycopg2
from psycopg2 import pool, sql

# AWS
import boto3
from botocore.exceptions import ClientError

# Mongo
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, PyMongoError

# LXML
from lxml import etree
