#!/usr/bin/python

#%%
from config.config_variables import *
from config.config_modules import *

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

########################
####    RabbitMQ    ####
########################

def connect_to_rabbitmq(host: str = 'localhost', user: Optional[str] = '', password: Optional[str] = '', connection_parameters: Optional[URLParameters] = None) -> pika.BlockingConnection:
    """
    Create and return a connection to RabbitMQ.

    Args:
    host (str): The hostname for RabbitMQ, e.g., 'localhost'.
    user (str, optional): The username for RabbitMQ. Default is taken from RABBITMQ_USERNAME environment variable.
    password (str, optional): The password for RabbitMQ. Default is taken from RABBITMQ_PASSWORD environment variable.
    connection_parameters (URLParameters, optional): Additional connection parameters. 
    Default is None, which means the connection will use only the host, user, and password.

    Returns:
    BlockingConnection: A pika BlockingConnection instance.

    Raises:
    AssertionError: If the host is not provided or is empty.
    pika.exceptions.AMQPConnectionError: If the connection to RabbitMQ fails.

    Example:
    connection = connect_to_rabbitmq('localhost', 'user', 'password')
    """
    assert host, "RabbitMQ host must be provided."
    assert isinstance(host, str), "RabbitMQ host must be a string."
    assert isinstance(user, str), "RabbitMQ username must be a string."
    assert isinstance(password, str), "RabbitMQ password must be a string."

    # Construct the URL for the connection
    url = f'amqp://{user}:{password}@{host}:5672/'

    # Use connection parameters if provided, else create from URL
    if connection_parameters:
        parameters = connection_parameters
    else:
        parameters = pika.URLParameters(url)

    # Create and return the connection
    return pika.BlockingConnection(parameters)

async def connect_to_rabbitmq_async(host: str = 'localhost', 
                                    user: Optional[str] = None, 
                                    password: Optional[str] = None, 
                                    connection_parameters: Optional[URLParameters] = None) -> aio_pika.Connection:
    """
    Asynchronously create and return a connection to RabbitMQ.

    Args:
        host (str): The hostname for RabbitMQ, e.g., 'localhost'.
        user (Optional[str]): The username for RabbitMQ. Defaults to environment variable RABBITMQ_USERNAME or 'guest'.
        password (Optional[str]): The password for RabbitMQ. Defaults to environment variable RABBITMQ_PASSWORD or 'guest'.
        connection_parameters (Optional[URLParameters]): Additional connection parameters. Default is None.

    Returns:
        aio_pika.Connection: An aio_pika Connection instance.

    Raises:
        AssertionError: If the host is not provided or is empty.
        aio_pika.exceptions.AMQPConnectionError: If the connection to RabbitMQ fails.
        Exception: For any other unexpected errors.
    """
    assert host, "RabbitMQ host must be provided."

    user = user or os.environ.get('RABBITMQ_USERNAME', 'guest')
    password = password or os.environ.get('RABBITMQ_PASSWORD', 'guest')

    url = f'amqp://{user}:{password}@{host}/'

    if connection_parameters:
        parameters = connection_parameters
    else:
        parameters = URLParameters(url)

    try:
        return await aio_pika.connect_robust(parameters)
    except aio_pika.exceptions.AMQPConnectionError as e:
        raise ConnectionError(f"Failed to connect to RabbitMQ: {e}")
    except Exception as e:
        raise Exception(f"Unexpected error while connecting to RabbitMQ: {e}")

async def rabbitmq_create_channel_async(connection: aio_pika.Connection) -> aio_pika.Channel:
    """
    Asynchronously create and return a channel using the provided RabbitMQ connection.

    Args:
        connection (aio_pika.Connection): An aio_pika Connection instance.

    Returns:
        aio_pika.Channel: An aio_pika Channel instance.

    Raises:
        AssertionError: If the connection is not provided or is not an aio_pika Connection instance.
        Exception: For any errors in channel creation.
    """
    assert isinstance(connection, aio_pika.Connection), "A valid aio_pika.Connection must be provided."

    try:
        channel = await connection.channel()
        print(f"channel: {dir(channel)}")
        return channel
    except Exception as e:
        raise Exception(f"Error creating channel asynchronously: {e}")

async def rabbitmq_create_queue_async(channel: aio_pika.Channel, queue_name: str) -> str:
    """
    Asynchronously create a new RabbitMQ queue.

    Args:
        channel (aio_pika.Channel): An aio_pika Channel instance.
        queue_name (str): The name of the queue to be created.

    Returns:
        str: The name of the created queue.

    Raises:
        AssertionError: If the channel is not provided or is not an aio_pika Channel instance.
        Exception: For any errors in queue creation.
    """
    assert isinstance(channel, aio_pika.Channel), "A valid aio_pika.Channel must be provided."

    try:
        await channel.declare_queue(queue_name, durable=True)
        print(f"Queue: {queue_name}")
        return queue_name
    except Exception as e:
        raise Exception(f"Error creating queue '{queue_name}' asynchronously: {e}")

async def rabbitmq_create_exchange_async(channel: aio_pika.Channel, exchange_name: str, exchange_type: str) -> str:
    """
    Asynchronously create a new RabbitMQ exchange.

    Args:
        channel (aio_pika.Channel): An aio_pika Channel instance.
        exchange_name (str): The name of the exchange to be created.
        exchange_type (str): The type of the exchange (e.g., 'direct', 'topic', 'fanout').    

    Returns:
        str: The name of the created exchange.

    Raises:
        AssertionError: If the channel is not provided or is not an aio_pika Channel instance.
        Exception: For any errors in exchange creation.
    """
    assert isinstance(channel, aio_pika.Channel), "A valid aio_pika.Channel must be provided."

    try:
        await channel.declare_exchange(name=exchange_name, type=exchange_type, durable=True)
        print(f"exchange_name: {exchange_name}")
        return exchange_name
    except Exception as e:
        raise Exception(f"Error creating exchange '{exchange_name}' asynchronously: {e}")

async def rabbitmq_create_binding_async(channel: aio_pika.Channel, queue_name: str, exchange_name: str, routing_key: Optional[str] = '') -> None:
    """
    Asynchronously create a new binding between a queue and an exchange.

    Args:
        channel (aio_pika.Channel): An aio_pika Channel instance.
        queue_name (str): The name of the queue.
        exchange_name (str): The name of the exchange.
        routing_key (Optional[str]): The routing key for the binding. Default is an empty string.

    Returns:
        None

    Raises:
        AssertionError: If the channel is not provided or is not an aio_pika Channel instance.
        Exception: For any errors in binding creation.
    """
    assert isinstance(channel, aio_pika.Channel), "A valid aio_pika.Channel must be provided."

    try:
        queue = await channel.get_queue(queue_name)
        exchange = await channel.get_exchange(exchange_name)
        await queue.bind(exchange, routing_key)
        return True
    except Exception as e:
        raise Exception(f"Error creating binding between '{queue_name}' and '{exchange_name}' asynchronously: {e}")

async def rabbitmq_create_consumer_async(channel: aio_pika.Channel, queue_name: str, callback) -> None:
    """
    Asynchronously create a consumer for a RabbitMQ queue.

    Args:
        channel (aio_pika.Channel): An aio_pika Channel instance.
        queue_name (str): The name of the queue.
        callback (callable): A callback function to handle incoming messages.

    Returns:
        None

    Raises:
        AssertionError: If the channel is not provided or is not an aio_pika Channel instance.
        Exception: For any errors in consumer setup.
    """
    assert isinstance(channel, aio_pika.Channel), "A valid aio_pika.Channel must be provided."

    try:
        queue = await channel.get_queue(queue_name)
        await queue.consume(callback)
        return True
    except Exception as e:
        raise Exception(f"Error creating consumer for queue '{queue_name}' asynchronously: {e}")

async def setup_rabbitmq_pipeline_async(rabbitmq_connection: aio_pika.Connection,
                                        queue_name: str, exchange_name: str, exchange_type: str,
                                        routing_key: Optional[str], callback) -> None:
    """
    Asynchronously set up the RabbitMQ pipeline.

    Args:
    rabbitmq_connection (aio_pika.Connection): An aio_pika Connection instance.
    queue_name (str): The name of the queue to be created.
    exchange_name (str): The name of the exchange to be created.
    exchange_type (str): The type of the exchange.
    routing_key (str, optional): The routing key for the binding.
    callback (callable): A callback function to handle incoming messages.

    Raises:
    Exception: If any part of the setup fails.
    """
    try:
        channel = await rabbitmq_create_channel_async(rabbitmq_connection)
        print(f"channel: {dir(channel)}")
        await rabbitmq_create_queue_async(channel, queue_name)
        await rabbitmq_create_exchange_async(channel, exchange_name, exchange_type)
        await rabbitmq_create_binding_async(channel, queue_name, exchange_name, routing_key)
        await rabbitmq_create_consumer_async(channel, queue_name, callback)

        print("RabbitMQ pipeline setup complete.")
    except Exception as e:
        raise Exception(f"Error in setting up RabbitMQ pipeline asynchronously: {e}")

async def rabbitmq_store_config_async(db_client: AsyncIOMotorClient, config_data: dict, collection_name: str) -> None:
    """
    Asynchronously store RabbitMQ configuration data in MongoDB.

    Args:
    db_client (AsyncIOMotorClient): Asynchronous MongoDB client instance.
    config_data (dict): Configuration data to be stored.
    collection_name (str): The name of the MongoDB collection.

    Raises:
    Exception: If storing data fails.
    """
    try:
        db = db_client['rabbitmq_config_db']
        collection = db[collection_name]
        await collection.insert_one(config_data)
    except Exception as e:
        raise Exception(f"Error storing RabbitMQ configuration asynchronously: {e}")

async def rabbitmq_get_config_async(db_client: AsyncIOMotorClient, collection_name: str) -> Optional[dict]:
    """
    Asynchronously retrieve RabbitMQ configuration data from MongoDB.

    Args:
    db_client (AsyncIOMotorClient): Asynchronous MongoDB client instance.
    collection_name (str): The name of the MongoDB collection.

    Returns:
    Optional[dict]: Configuration data if found, else None.

    Raises:
    Exception: If retrieval fails.
    """
    try:
        db = db_client['rabbitmq_config_db']
        collection = db[collection_name]
        return await collection.find_one()
    except Exception as e:
        raise Exception(f"Error retrieving RabbitMQ configuration asynchronously: {e}")

async def rabbitmq_log_action_async(db_client: AsyncIOMotorClient, log_data: dict, log_collection: str = "rabbitmq_logs") -> None:
    """
    Asynchronously log actions or changes related to RabbitMQ configurations in MongoDB.

    Args:
    db_client (AsyncIOMotorClient): Asynchronous MongoDB client instance.
    log_data (dict): Log data to be stored.
    log_collection (str): The name of the MongoDB log collection.

    Raises:
    Exception: If logging fails.
    """
    try:
        db = db_client['rabbitmq_config_db']
        collection = db[log_collection]
        await collection.insert_one(log_data)
    except Exception as e:
        raise Exception(f"Error logging RabbitMQ action asynchronously: {e}")

async def sample_callback(message: aio_pika.IncomingMessage):
    async with message.process():
        print("Received message:", message.body.decode())
        # Further processing can be done here


##########################
####    PostgreSQL    ####
##########################

# Connect to PostgreSQL
# Run Postgres Locally
# brew services start postgresql

#%%
# Global variable for connection pool
connection_pool = None

def initialize_connection_pool(host: str = 'localhost'):
    global connection_pool
    try:
        # Get database connection details from environment variables
        # db_host = os.environ.get('POSTGRES_DB_HOST', 'localhost')
        db_host = os.environ.get('POSTGRES_DB_HOST', host)
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

def create_table():
    try:
        # Connect to the PostgreSQL database
        conn = psycopg2.connect(
            # host='localhost',
            host='postgres-container',
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

# %%

# def connect_to_mongodb(host: str = 'localhost',   # def connect_to_mongodb(host: str = 'localhost', 
#                        port: int = 27017,
#                        username: Optional[str] = None, 
#                        password: Optional[str] = None, 
#                        db_name: str = 'mydatabase') -> Database:
#     """
#     Establishes a connection to a MongoDB database.

#     Parameters:
#     host (str): The hostname or IP address of the MongoDB server. Defaults to 'localhost'.
#     port (int): The port number on which MongoDB is running. Defaults to 27017.
#     username (Optional[str]): The username for MongoDB authentication. Defaults to None.
#     password (Optional[str]): The password for MongoDB authentication. Defaults to None.
#     db_name (str): The name of the database to connect to. Defaults to 'mydatabase'.

#     Returns:
#     Database: A MongoDB database object.

#     Raises:
#     AssertionError: If the provided host, port, or db_name are not valid.
#     ConnectionFailure: If the connection to the MongoDB server fails.
#     PyMongoError: For other PyMongo related errors.

#     Example:
#     >>> mongodb_client = connect_to_mongodb('localhost', 27017, 'user', 'pass', 'mydb')
#     >>> print(mongodb_client.name)
#     mydb
#     """

#     # Input validation
#     assert isinstance(host, str) and host, "Host must be a non-empty string."
#     assert isinstance(port, int) and port > 0, "Port must be a positive integer."
#     assert isinstance(db_name, str) and db_name, "Database name must be a non-empty string."

#     try:
#         # Create a MongoDB client instance
#         # client = MongoClient(host, port, username=username, password=password)
#         # client = MongoClient('mongodb://localhost:27017/')
#         client = MongoClient(f'mongodb://{host}:27017/')

#         # Access the specified database
#         db = client[db_name]

#         # Attempt a simple operation to verify the connection
#         db.command('ping')

#         print(f"Connected to MongoDB: {db_name}")

#         return db
#     except ConnectionFailure as e:
#         raise ConnectionFailure(f"Failed to connect to MongoDB: {e}")
#     except PyMongoError as e:
#         raise PyMongoError(f"An error occurred with PyMongo: {e}")

def connect_to_mongodb(host: str = 'localhost', port: int = 27017, 
                       username: Optional[str] = None, password: Optional[str] = None, 
                       db_name: str = 'mydatabase') -> MongoClient:
    """
    Establishes a connection to a MongoDB database.
    ... [rest of the docstring] ...
    """

    if not host or not isinstance(host, str):
        raise ValueError("Host must be a non-empty string.")
    if not isinstance(port, int) or port <= 0:
        raise ValueError("Port must be a positive integer.")
    if not db_name or not isinstance(db_name, str):
        raise ValueError("Database name must be a non-empty string.")

    try:
        # Form the connection string
        if username and password:
            client = MongoClient(f'mongodb://{username}:{password}@{host}:{port}/{db_name}')
        else:
            client = MongoClient(f'mongodb://{host}:{port}/{db_name}')

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


#################################
####    Kafka & Zookeeper    ####
#################################


def create_kafka_admin_client(bootstrap_servers: str = "localhost:9092",   # def create_kafka_admin_client(bootstrap_servers: str = "localhost:9092",
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

############################
####    Neo4j Client    ####
############################

def connect_to_neo4j(host: str = 'bolt://localhost:7687'):
    try:
        client = GraphDatabase.driver(
            host,
            # "bolt://neo4j-container:7687",
            # os.environ.get("NEO4J_URI"),
            auth=(os.environ.get("NEO4J_USER"), os.environ.get("NEO4J_PASSWORD")),
            max_connection_lifetime=3600*24*30,
            keep_alive=True
        )
        print(f"Connected to Neo4j: {client}")
        return client
    except Exception as e:
        print(f"Errors loading Neo4j Client: {e}")
        return None
    
# neo4j_client = connect_to_neo4j()


#####################
####    Spacy    ####
#####################

def connect_to_spacy():
    try:
        nlp = spacy.load('en_core_web_lg')
        print(f"Connected to Spacy: {nlp}")
        return nlp
    except Exception as e:
        print(f"Errors loading Spacy Client: {e}")
        return None

############################
####    Redis Client    ####
############################

def connect_to_redis(host: str = 'localhost'):
    try:
        client = redis.Redis(host=host, port=6379)    # client = redis.Redis(host=os.environ.get('REDIS_HOST'), port=int(os.environ.get('REDIS_PORT')))     int(os.environ.get('REDIS_PORT'))
        print(f"Connected to Redis: {client}")
        return client
    except Exception as e:
        print(f"Errors loading Redis Client: {e}")
        return None


#############################
####    Milvus Client    ####
#############################

# Instructions to start Milvus
# cd ~/git/gis-apps
# docker-compose -f docker-compose-milvus.yml up 

def milvus_connect_to_server(host: str = 'localhost', port: str = '19530') -> None:      # def milvus_connect_to_server(host: str = 'localhost', port: str = '19530') -> None:
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
        return milvus_connections.connect("default", host=host, port="19530")    # return milvus_connections.connect("default", host="localhost", port="19530")
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
    fields.append(MilvusFieldSchema(name="_guid", dtype=MilvusDataType.VARCHAR, is_primary=True, max_length=100))
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

def milvus_create_index(collection_name: str = 'gis_main', field_name: str = '_guid', vector_len: int = 300) -> None:
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
        milvus_collection = MilvusCollection(name=collection_name)
        milvus_collection.create_index(field_name, index_params)
    except Exception as e:
        raise Exception(f"Error in creating index on {field_name} in collection {collection_name}: {e}")

############################
####    Bloom Filter    ####
############################

class BloomFilter:
    """
    A simple Bloom filter implementation using Redis.
    """
    def __init__(self, redis_host: str, redis_port: int, redis_db: int = 0):
        self.redis_client = redis.StrictRedis(host=os.environ.get("REDIS_HOST"), port=os.environ.get("REDIS_PORT"), db=redis_db)
        self.num_hashes = 6  # Number of hash functions to use
        self.bit_size = 10000000  # Size of the bitmap

    def _hashes(self, key: str) -> List[int]:
        """
        Generates multiple hash values for a given key.
        """
        hashes = []
        for i in range(self.num_hashes):
            hash_val = int(hashlib.md5(f"{key}_{i}".encode()).hexdigest(), 16) % self.bit_size
            hashes.append(hash_val)
        return hashes

    def add(self, key: str) -> None:
        """
        Adds a key to the Bloom filter.
        """
        hashes = self._hashes(key)
        pipeline = self.redis_client.pipeline()
        for hash_val in hashes:
            pipeline.setbit("bloom_filter", hash_val, 1)
        pipeline.execute()

    def check(self, key: str) -> bool:
        """
        Checks if a key might be in the Bloom filter.
        """
        hashes = self._hashes(key)
        for hash_val in hashes:
            if not self.redis_client.getbit("bloom_filter", hash_val):
                return False
        return True

def connect_to_bloomfilter():
    try:
        bloom_filter = BloomFilter(redis_host=os.environ.get("REDIS_HOST"), redis_port=os.environ.get("REDIS_PORT"), redis_db=0)
        print(f"Connected to BloomFilter: {bloom_filter}")
        return bloom_filter
    except Exception as e:
        print(f"Errors loading BloomFilter Client: {e}")
        return None


######################
####    TypeDB    ####
######################

# def connect_to_typedb(host: str = 'localhost', 
#                       port: int = 27017, 
#                       username: Optional[str] = None, 
#                       password: Optional[str] = None, 
#                       db_name: str = 'gis_main') -> Database:
#     """
#     Establishes a connection to a MongoDB database.

#     Parameters:
#     host (str): The hostname or IP address of the TypeDB server. Defaults to 'localhost'.
#     port (int): The port number on which TypeDB is running. Defaults to 1729.
#     username (Optional[str]): The username for TypeDB authentication. Defaults to None.
#     password (Optional[str]): The password for TypeDB authentication. Defaults to None.
#     db_name (str): The name of the database to connect to. Defaults to 'gis_main'.

#     Returns:
#     Database: A TypeDB database object.

#     Raises:
#     AssertionError: If the provided host, port, or db_name are not valid.

#     Example:
#     >>> typedb_client = connect_to_mongodb('localhost', 1729, 'user', 'pass', 'mydb')
#     >>> print(typedb_client.name)
    
#     """

#     # Input validation
#     assert isinstance(host, str) and host, "Host must be a non-empty string."
#     assert isinstance(port, int) and port > 0, "Port must be a positive integer."
#     assert isinstance(db_name, str) and db_name, "Database name must be a non-empty string."

#     try:
#         # Create a TypeDB client instance
#         client = TypeDB.core_driver(TYPEDB_URI)
#         print(f"Connected to TypeDB: {client}")

#         return client
    
#     except Exception as e:
#         print(f"Error connecting to TypeDB: {e}")
#         return None

# Connect to TypeDB and create a database 'gis-main'
    
# def create_typedb_database(typedb_client: type, database_name: str = 'gis_main') -> None:
#     """
#     Create a new database in the TypeDB server.

#     Parameters:
#     database_name (str): Name of the database to be created.

#     Raises:
#     AssertionError: If inputs are not in expected format.
#     Exception: For issues encountered while creating the database.
#     """
#     assert isinstance(database_name, str), "Database name must be a string"

#     try:
#         # Create a TypeDB client instance
#         client = TypeDB.core_driver(TYPEDB_URI)

#         # Create a database
#         client.databases.create(database_name)
#         print(f"TypeDB Database created: {database_name}")
    
#     except Exception as e:
#         # print(f"Error creating TypeDB Database: {e}")
#         return None    


############################
####    Vault Server    ####
############################
    
def connect_to_vault(url: str = 'http://127.0.0.1:8200') -> hvac.Client:
    try:
        vault_client = hvac.Client(url=os.environ.get("VAULT_ADDR"), token=os.environ.get("VAULT_TOKEN"))
        print(f"Connected to Vault: {vault_client}")
        return vault_client
    except Exception as e:
        print(f"Errors loading Vault Client: {e}")
        return None

################################
####    Creating Clients    ####
################################

#%%

# S3
s3_client = s3_connect()

# PostgreSQL

if GIS_ENVIRONMENT == 'flask-local':
    try:
        # Initialize the connection pool
        initialize_connection_pool('postgres-container')
        postgres_client = connect_to_postgres('postgres-container')
        print("PostgreSQL client connected to container.")
    except Exception as e:
        pass

if GIS_ENVIRONMENT == 'local':
    try:
        initialize_connection_pool('localhost')
        postgres_client = connect_to_postgres('localhost')
        print("PostgreSQL client connected locally.")
    except Exception as e:
        pass

# Kafka

# kafka_connected = False

# try:
#     kafka_client = create_kafka_admin_client("kafka:9092", "my_client_id")
#     kafka_connected = True
#     print("Kafka client connected to container.")
# except Exception as e:
#     pass

# if not kafka_connected:
#     try:
#         kafka_client = create_kafka_admin_client("localhost:9092", "my_client_id")
#         kafka_connected = True
#         print("Kafka client connected locally.")
#     except Exception as e:
#         pass

# MongoDB

if GIS_ENVIRONMENT == 'flask-local':
    try:
        mongodb_client = connect_to_mongodb('mongodb-container')
        print("MongoDB client connected to container.")
    except Exception as e:
        print(f"Error connecting to MongoDB with Flask API: {e}")
        pass

if GIS_ENVIRONMENT == 'local':
    try:
        mongodb_client = connect_to_mongodb('localhost')
        print("MongoDB client connected locally.")
    except Exception as e:
        pass    

# If connected to MongoDB, create a collection
mongo_collection = mongodb_client['gis_main']

# Creating an index on the _guid field
if '_guid' not in list(mongo_collection.index_information()):
    mongo_collection.create_index([("_guid", 1)], unique=True)

# Kafka
# kafka_connected = False

# try:
#     kafka_client = create_kafka_admin_client("kafka:9092")
#     kafka_connected = True
#     print("Kafka client connected to container.")
# except Exception as e:
#     pass

# if not kafka_connected:
#     try:
#         kafka_client = create_kafka_admin_client("localhost:9092")
#         kafka_connected = True
#         print("Kafka client connected locally.")
#     except Exception as e:
#         pass

# Spacy
nlp = connect_to_spacy()

# Neo4j
if GIS_ENVIRONMENT == 'flask-local':    
    try:
        neo4j_client = connect_to_neo4j('bolt://neo4j-container:7687')
        print("Neo4j client connected to container.")
    except Exception as e:
        pass

if GIS_ENVIRONMENT == 'local':
    try:
        neo4j_client = connect_to_neo4j('bolt://localhost:7687')
        print("Neo4j client connected locally.")
    except Exception as e:
        pass

# Redis
if GIS_ENVIRONMENT == 'flask-local':
    try:
        redis_client = connect_to_redis('redis-container')
        redis_connected = True
        print("Redis client connected to container.")
    except Exception as e:
        pass

if GIS_ENVIRONMENT == 'local':
    try:
        redis_client = connect_to_redis('localhost')
        redis_connected = True
        print("Redis client connected locally.")
    except Exception as e:
        pass

# Load Milvus
if GIS_ENVIRONMENT == 'flask-local':
    try:
        milvus_connect_to_server('milvus-container')
        print("Milvus client connected to container.")
    except Exception as e:
        pass

if GIS_ENVIRONMENT == 'local':
    try:
        milvus_connect_to_server('localhost')
        print("Milvus client connected locally.")
    except Exception as e:
        pass    

# Creating Milvus collection
milvus_collection = milvus_create_collection("gis_main", "gis_main holds vectors for GIS Data Lake.")
milvus_create_index("gis_main", "vector")
milvus_collection.load()

# Load RabbitMQ
# if GIS_ENVIRONMENT == 'flask-local':
#     try:
#         rabbitmq_connection = connect_to_rabbitmq('rabbitmq-container')
#         print("RabbitMQ connection created successfully with container.")
#         rabbitmq_channel = rabbitmq_create_channel(rabbitmq_connection)
#         print(f"RabbitMQ channel created successfully with container.")
#     except Exception as e:
#         pass

# if GIS_ENVIRONMENT == 'local':
#     try:
#         rabbitmq_connection = connect_to_rabbitmq('localhost')
#         print("RabbitMQ connection created successfully locally.")
#         rabbitmq_channel = rabbitmq_create_channel(rabbitmq_connection)
#         print(f"RabbitMQ channel created successfully with container.")
#     except Exception as e:
#         pass        

# if GIS_ENVIRONMENT == 'local':
#     queue_name = 'example_queue'
#     exchange_name = 'example_exchange'
#     exchange_type = 'direct'
#     routing_key = 'example_routing_key'

#     try:
#         print(f"os.environ.get('RABBITMQ_USERNAME'): {os.environ.get('RABBITMQ_USERNAME')}")
#         rabbitmq_connection = connect_to_rabbitmq('localhost', os.environ.get('RABBITMQ_USERNAME', 'rabbit'), os.environ.get('RABBITMQ_PASSWORD', 'r@bb!tM@'))
#         print("RabbitMQ connection created successfully locally.")
#         setup_rabbitmq_pipeline_async(rabbitmq_connection,
#                                         queue_name, 
#                                         exchange_name, 
#                                         exchange_type, 
#                                         routing_key, 
#                                         sample_callback)
        
#     except Exception as e:
#         print(f"Error connecting to RabbitMQ locally: {e}")
#         pass

# if GIS_ENVIRONMENT == 'flask-local':
#     queue_name = 'example_queue'
#     exchange_name = 'example_exchange'
#     exchange_type = 'direct'
#     routing_key = 'example_routing_key'

#     try:
#         print(f"os.environ.get('RABBITMQ_USERNAME'): {os.environ.get('RABBITMQ_USERNAME')}")
#         rabbitmq_connection = connect_to_rabbitmq('rabbitmq-container', os.environ.get('RABBITMQ_USERNAME', 'rabbit'), os.environ.get('RABBITMQ_PASSWORD', 'r@bb!tM@'))
#         print("RabbitMQ connection created successfully with container.")
#         setup_rabbitmq_pipeline_async(rabbitmq_connection,
#                                         queue_name, 
#                                         exchange_name, 
#                                         exchange_type, 
#                                         routing_key, 
#                                         sample_callback)

#     except Exception as e:
#         print(f"Error connecting to RabbitMQ with Flask API: {e}")
#         pass

########################
####    RabbitMQ    ####
########################

if GIS_ENVIRONMENT == 'local':
    rabbitmq_connection = connect_to_rabbitmq('localhost', os.environ.get('RABBITMQ_USERNAME', 'rabbit'), os.environ.get('RABBITMQ_PASSWORD', 'r@bb!tM@'))
    print("RabbitMQ connection created successfully locally.")
elif GIS_ENVIRONMENT == 'flask-local':
    rabbitmq_connection = connect_to_rabbitmq('rabbitmq-container', os.environ.get('RABBITMQ_USERNAME', 'rabbit'), os.environ.get('RABBITMQ_PASSWORD', 'r@bb!tM@'))
    print("RabbitMQ connection created successfully with container.")


async def main(rabbitmq_connection: aio_pika.Connection):
    queue_name = 'example_queue'
    exchange_name = 'example_exchange'
    exchange_type = 'direct'
    routing_key = 'example_routing_key'

    try:
        print(f"os.environ.get('RABBITMQ_USERNAME'): {os.environ.get('RABBITMQ_USERNAME')}")
        # Assuming connect_to_rabbitmq is an async function
        print("RabbitMQ connection created successfully locally.")
        await setup_rabbitmq_pipeline_async(rabbitmq_connection,
                                            queue_name, 
                                            exchange_name, 
                                            exchange_type, 
                                            routing_key, 
                                            sample_callback)
        
    except Exception as e:
        print(f"Error connecting to RabbitMQ locally: {e}")

# def run_main():
#     try:
#         loop = asyncio.get_running_loop()
#     except RuntimeError:  # No running event loop
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)

#     if loop.is_running():
#         print("Running inside existing event loop")
#         task = asyncio.ensure_future(main(rabbitmq_connection))
#         loop.run_until_complete(task)
#     else:
#         print("Starting new event loop")
#         asyncio.run(main(rabbitmq_connection))

def start_async_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

def run_main_async(rabbitmq_connection: aio_pika.Connection):
    # Create a new event loop
    new_loop = asyncio.new_event_loop()

    # Start the new event loop in a separate thread
    threading.Thread(target=start_async_loop, args=(new_loop,), daemon=True).start()

    # Now, we can use this new event loop to run our coroutine
    asyncio.run_coroutine_threadsafe(main(rabbitmq_connection), new_loop)

# Building RabbitMQ Objects
run_main_async(rabbitmq_connection)


# Load Vault
if GIS_ENVIRONMENT == 'flask-local':
    try:
        vault_client = connect_to_vault('http://vault-container:8200')
        print("Vault client connected to container.")
    except Exception as e:
        pass

if GIS_ENVIRONMENT == 'local':
    try:
        vault_client = connect_to_vault('http://127.0.0.1:8200')
        vault_connected = True
        print("Vault client connected locally.")
    except Exception as e:
        pass

# Bloom Filter
bloom_filter = connect_to_bloomfilter()

# TypeDB
# typedb_client = connect_to_typedb()
# create_typedb_database(typedb_client, 'gis_main')
