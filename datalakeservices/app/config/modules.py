#!/usr/bin/python

# Generic Modules
import os
import re
import json
import uuid
import unicodedata
import numpy as np
import pandas as pd
from pandas.api import types
from dateutil.parser import parse
from datetime import datetime, timedelta, timezone

# Ashnchronous packages
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

# Milvus
from pymilvus import (
    connections as milvus_connections,
    utility as milvus_utility,
    FieldSchema as MilvusFieldSchema,
    CollectionSchema as MilvusCollectionSchema,
    DataType as MilvusDataType,
    Collection as MilvusCollection,
)

# Neo4j
from neo4j import GraphDatabase

# Multiprcessing
import ray

# MinIO
from minio import Minio
from minio.error import S3Error

# Pytest
import pytest

# Redis
import redis

# Typing
from typing import List, Dict, Tuple, Union, Any, Optional, Coroutine, Callable, Awaitable, Iterable, AsyncIterable, TypeVar, Generic, Iterator

# Connect to PostgreSQL
import psycopg2
from psycopg2 import pool, sql

# Kafka & Zookeeper
from confluent_kafka import Producer, Consumer, KafkaException
from confluent_kafka.admin import AdminClient, NewTopic

# AWS
import boto3
from botocore.exceptions import ClientError

# Mongo
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure, PyMongoError

# LXML
from lxml import etree
