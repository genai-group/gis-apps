#!/usr/bin/python

# Generic Modules
import os
import re
import json
import time
import uuid
import pytz
import copy
import random
import sspipe
import hashlib
import logging
import asyncio
import aio_pika
import textwrap
import threading
import xmltodict
import unicodedata
import numpy as np
import pandas as pd
import logging.config
import logging.handlers
from pandas.api import types
from titlecase import titlecase
from pprint import pprint as pp
from dateutil.parser import parse
from urllib.parse import quote_plus
from datetime import datetime, timedelta, timezone, date
from cryptography.fernet import Fernet, InvalidToken

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

# PyXB => .xsd files
# import pyxb.utils.domutils as domutils

# Neo4j
import neo4j
from neo4j import GraphDatabase

# Pika - RabbitMQ
import pika
from pika import URLParameters
from pika.adapters.blocking_connection import BlockingChannel, BlockingConnection

# Multiprcessing
import ray

# Celery
from celery import Celery

# Spacy
import spacy

# PyYaml
import yaml

# TypeDB
# from typedb.client import TypeDB, TypeDBOptions, SessionType, TransactionType
# from typedb.driver import *
# from typedb.driver import TypeDB, SessionType, TransactionType
# from typedb.common.exception import TypeDBClientException, TypeDBClientError

# MinIO
# from minio import Minio
# from minio.error import S3Error

# Phone numbers
import phonenumbers
from phonenumbers import geocoder

# Pytest
import pytest

# Redis
import redis

# Typing
from typing import List, Dict, Tuple, Union, Any, Optional, Coroutine, Callable, Awaitable, Iterable, AsyncIterable, TypeVar, Generic, Iterator, Callable

# Connect to PostgreSQL
import psycopg2
from psycopg2 import pool, sql

# Kafka & Zookeeper
# from confluent_kafka import Producer, Consumer, KafkaException
# from confluent_kafka.admin import AdminClient, NewTopic

# Swagger
from flask_swagger_ui import get_swaggerui_blueprint
from flask_cors import CORS
from werkzeug.utils import secure_filename


# Vault
import hvac

# AWS
import boto3
from botocore.exceptions import ClientError

# Mongo
import pymongo
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure, PyMongoError

# LXML
from lxml import etree
