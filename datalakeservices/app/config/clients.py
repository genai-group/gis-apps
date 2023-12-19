#!/usr/bin/python

#%%
from config.variables import *
from config.modules import *

######################################
####    Ray - multi-processing    ####
######################################

# ray.init(dashboard_host="0.0.0.0",dashboard_port=6379)

######################
####    AWS S3    ####
######################

#%%
def s3_connect():
    try:
        s3_client = boto3.client('s3')
        print("S3 Client Connected")
    except Exception as e:
        print(f"S3 Client Connection Error: {e}")
        s3_client = None
    return s3_client

#####################
####    MinIO    ####
#####################

# Install MinIO

# curl -O https://dl.min.io/server/minio/release/darwin-arm64/minio
# chmod +x ./minio
# sudo mv ./minio /usr/local/bin/

# start the minio server
# minio server ~/data/minio

def minio_connect(endpoint_url: str, access_key: str, secret_key: str):
    """
    Connect to a MinIO client.

    Args:
        endpoint_url (str): The endpoint URL of the MinIO server.
        access_key (str): The access key for MinIO.
        secret_key (str): The secret key for MinIO.

    Returns:
        boto3.client: A boto3 client object if the connection is successful, None otherwise.
    """
    try:
        # Asserting that necessary parameters are provided
        assert endpoint_url, "Endpoint URL must be provided"
        assert access_key, "Access key must be provided"
        assert secret_key, "Secret key must be provided"

        # Creating a boto3 client for MinIO
        minio_client = boto3.client('s3', endpoint_url=endpoint_url,
                                    aws_access_key_id=access_key,
                                    aws_secret_access_key=secret_key,
                                    use_ssl=False)  # Set to True if your MinIO server uses SSL
        print("MinIO Client Connected")
        return minio_client
    except Exception as e:
        print(f"MinIO Client Connection Error: {e}")
        return None

##########################
####    PostgreSQL    ####
##########################

# Connect to PostgreSQL
# Run Postgres Locally
# brew services start postgresql

#%%
# Global variable for connection pool
connection_pool = None

def initialize_connection_pool():
    global connection_pool
    try:
        # Get database connection details from environment variables
        db_host = os.environ.get('POSTGRES_DB_HOST', 'localhost')
        db_port = os.environ.get('POSTGRES_DB_PORT', '5432')
        # db_name = os.environ.get('POSTGRES_DB_NAME')
        db_name = 'postgres'
        # db_user = os.environ.get('POSTGRES_DB_USER')
        db_user = 'postgres'
        # db_password = os.environ.get('POSTGRES_DB_PASSWORD')
        db_password = '12345asdf'

        # Initialize the connection pool => auto scale threads when needed
        connection_pool = psycopg2.pool.ThreadedConnectionPool(1, 10,
                                                            host=db_host,
                                                            port=db_port,
                                                            dbname=db_name,
                                                            user=db_user,
                                                            password=db_password)
        print("Connection pool created successfully!")
    except psycopg2.Error as e:
        print(f"Error initializing connection pool: {e}")

def get_connection():
    if connection_pool:
        return connection_pool.getconn()
    else:
        print("Connection pool is not initialized.")
        return None

def release_connection(conn):
    if connection_pool:
        connection_pool.putconn(conn)

def connect_to_postgres():
    try:
        # Get a connection from the pool
        conn = get_connection()
        if conn:
            # Do something with the connection
            print(f"Connected to PostgreSQL: {conn.status}")
            # Release the connection back to the pool
            release_connection(conn)
        else:
            print("Failed to obtain a connection.")
    except psycopg2.Error as e:
        print(f"Error connecting to PostgreSQL: {e}")

# Initialize the connection pool
initialize_connection_pool()

# Example usage
connect_to_postgres()


def create_table():
    try:
        # Connect to the PostgreSQL database
        conn = psycopg2.connect(
            host='localhost',
            port='5432',
            dbname='your_database_name',
            user='your_username',
            password='your_password'
        )

        # Create a cursor object to execute SQL queries
        cursor = conn.cursor()

        # Define the SQL query to create the table
        create_table_query = '''
        CREATE TABLE IF NOT EXISTS your_table_name (
            column1 datatype1,
            column2 datatype2,
            ...
        )
        '''

        # Execute the SQL query to create the table
        cursor.execute(create_table_query)

        # Commit the changes to the database
        conn.commit()

        # Close the cursor and connection
        cursor.close()
        conn.close()

        print("Table created successfully!")
    except psycopg2.Error as e:
        print(f"Error creating table: {e}")


#######################
####    MongoDB    ####
#######################

"""
docker pull --platform linux/arm64 mongo
docker run --name mongodb -d -p 27017:27017 --platform linux/arm64 mongo

RUN WITH USERNAME AND PASSWORD
docker run --name mongodb -d -p 27017:27017 -e MONGO_INITDB_ROOT_USERNAME=myuser -e MONGO_INITDB_ROOT_PASSWORD=mypassword --platform linux/arm64 mongo
"""


# %%

def connect_to_mongodb(host: str = 'localhost', 
                       port: int = 27017, 
                       username: Optional[str] = None, 
                       password: Optional[str] = None, 
                       db_name: str = 'mydatabase') -> Database:
    """
    Establishes a connection to a MongoDB database.

    Parameters:
    host (str): The hostname or IP address of the MongoDB server. Defaults to 'localhost'.
    port (int): The port number on which MongoDB is running. Defaults to 27017.
    username (Optional[str]): The username for MongoDB authentication. Defaults to None.
    password (Optional[str]): The password for MongoDB authentication. Defaults to None.
    db_name (str): The name of the database to connect to. Defaults to 'mydatabase'.

    Returns:
    Database: A MongoDB database object.

    Raises:
    AssertionError: If the provided host, port, or db_name are not valid.
    ConnectionFailure: If the connection to the MongoDB server fails.
    PyMongoError: For other PyMongo related errors.

    Example:
    >>> mongodb_client = connect_to_mongodb('localhost', 27017, 'user', 'pass', 'mydb')
    >>> print(mongodb_client.name)
    mydb
    """

    # Input validation
    assert isinstance(host, str) and host, "Host must be a non-empty string."
    assert isinstance(port, int) and port > 0, "Port must be a positive integer."
    assert isinstance(db_name, str) and db_name, "Database name must be a non-empty string."

    try:
        # Create a MongoDB client instance
        # client = MongoClient(host, port, username=username, password=password)
        client = MongoClient('mongodb://localhost:27017/')

        # Access the specified database
        db = client[db_name]

        # Attempt a simple operation to verify the connection
        db.command('ping')

        print(f"Connected to MongoDB: {db_name}")

        return db
    except ConnectionFailure as e:
        raise ConnectionFailure(f"Failed to connect to MongoDB: {e}")
    except PyMongoError as e:
        raise PyMongoError(f"An error occurred with PyMongo: {e}")

# Example usage
mongo_client = connect_to_mongodb()


#################################
####    Kafka & Zookeeper    ####
#################################


def create_kafka_admin_client(bootstrap_servers: str = "localhost:9092",
                              client_id: Optional[str] = None) -> AdminClient:
    """
    Create a Kafka AdminClient with specified parameters.

    Parameters:
    bootstrap_servers (str): Comma-separated list of broker addresses (default: "localhost:9092").
    client_id (Optional[str]): An optional identifier for the client.

    Returns:
    AdminClient: A Kafka AdminClient instance.

    Raises:
    AssertionError: If any of the input parameters are not in expected format.
    Exception: If there is an error in creating the Kafka AdminClient.
    """

    # Validate input parameters
    assert isinstance(bootstrap_servers, str), "bootstrap_servers must be a string"
    if client_id is not None:
        assert isinstance(client_id, str), "client_id must be a string"

    # Prepare configuration
    config = {"bootstrap.servers": bootstrap_servers}
    if client_id is not None:
        config["client.id"] = client_id

    # Attempt to create a Kafka AdminClient
    try:
        return AdminClient(config)
    except Exception as e:
        raise Exception(f"Error in creating Kafka AdminClient: {e}")

# Creating the kafa admin client
kafka_client = create_kafka_admin_client("localhost:9092", "my_client_id")


####    Neo4j Client    ####

def connect_to_neo4j():
    try:
        client = GraphDatabase.driver(
            os.environ.get("NEO4J_URI"),
            auth=(os.environ.get("NEO4J_USER"), os.environ.get("NEO4J_PASSWORD")),
            max_connection_lifetime=3600*24*30,
            keep_alive=True
        )
        print(f"Connected to Neo4j: {client}")
        return client
    except Exception as e:
        print(f"Errors loading Neo4j Client: {e}")
        return None
    
neo4j_client = connect_to_neo4j()


############################
####    Redis Client    ####
############################

def connect_to_redis():
    try:
        client = redis.Redis(host=os.environ.get('REDIS_HOST'), port=int(os.environ.get('REDIS_PORT')))
        print(f"Connected to Redis: {client}")
        return client
    except Exception as e:
        print(f"Errors loading Redis Client: {e}")
        return None
    
redis_client = connect_to_redis()    


#############################
####    Milvus Client    ####
#############################

# Instructions to start Milvus
# cd ~/git/gis-apps
# docker-compose -f docker-compose-milvus.yml up 

def milvus_connect_to_server(host: str = 'localhost', port: str = '19530') -> None:
    """
    Connect to a Milvus server.

    Args:
        host (str, optional): Hostname or IP address of the Milvus server. Defaults to 'localhost'.
        port (str, optional): Port number of the Milvus server. Defaults to '19530'.

    Returns:
        Milvus: Milvus server instance.
    """
    assert isinstance(host, str), "Host must be a string"
    assert isinstance(port, str), "Port must be a string"

    try:
        return milvus_connections.connect("default", host="localhost", port="19530")
    except Exception as e:
        raise ConnectionError(f"Failed to connect to Milvus server: {e}")

#############################
####    Milvus Client    ####
#############################

def milvus_create_collection(collection_name: str, description: str) -> None:
    """
    Create a new collection in the Milvus database with the given schema.

    Parameters:
    collection_name (str): Name of the collection to be created.
    description (str): Description of the collection.

    Raises:
    AssertionError: If inputs are not in expected format.
    Exception: For issues encountered while creating the collection.
    """
    assert isinstance(collection_name, str), "Collection name must be a string"
    assert isinstance(description, str), "Description must be a string"

    # Create FieldSchema objects from field_definitions
    fields = []
    fields.append(MilvusFieldSchema(name="gluid", dtype=MilvusDataType.VARCHAR, is_primary=True, max_length=100))
    fields.append(MilvusFieldSchema(name="namespace", dtype=MilvusDataType.VARCHAR, is_primary=False, max_length=100))
    fields.append(MilvusFieldSchema(name="vector", dtype=MilvusDataType.FLOAT_VECTOR, dim=300))
    fields.append(MilvusFieldSchema(name="created_at", dtype=MilvusDataType.INT64))

    # Create a CollectionSchema
    schema = MilvusCollectionSchema(fields, description)

    # Create a Collection
    try:
        milvus_collection = MilvusCollection(name=collection_name, schema=schema)
        print(f"Milvus Collection created: {milvus_collection}")
        return milvus_collection
    except Exception as e:
        raise Exception(f"Error in creating collection {collection_name}: {e}")

# Create Milvus Collection

def milvus_create_index(collection_name: str = 'gis_main', field_name: str = 'gluid', vector_len: int = 300) -> None:
    """
    Create an index for a specified field in a Milvus collection.

    Parameters:
    collection_name (str): Name of the collection on which the index is to be created.
    field_name (str): Name of the field in the collection to be indexed.
    index_params (Dict): Parameters of the index including index type, metric type, and other configurations.

    Raises:
    AssertionError: If inputs are not in expected format.
    Exception: For issues encountered while creating the index.
    """
    assert isinstance(collection_name, str), "Collection name must be a string"
    assert isinstance(field_name, str), "Field name must be a string"

    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": vector_len},
    }

    try:
        collection = MilvusCollection(name=collection_name)
        collection.create_index(field_name, index_params)
    except Exception as e:
        raise Exception(f"Error in creating index on {field_name} in collection {collection_name}: {e}")

# Create Milvus Index

######################################
####    Making Connections    ########
######################################

#%%

# S3
s3_connect()

# PostgreSQL
postgres_client = connect_to_postgres()

# MongoDB
mongodb_client = connect_to_mongodb()

# Kafka
kafka_client = create_kafka_admin_client()

# Neo4j
neo4j_client = connect_to_neo4j()

# Redis
redis_client = connect_to_redis()

# Load Milvus
milvus_connect_to_server()
milvus_collection = milvus_create_collection("gis_main", "gis_main holds vectors for GIS Data Lake.")
milvus_create_index("gis_main", "vector")
milvus_collection.load()