#!/usr/bin/python

#%%
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

        return db
    except ConnectionFailure as e:
        raise ConnectionFailure(f"Failed to connect to MongoDB: {e}")
    except PyMongoError as e:
        raise PyMongoError(f"An error occurred with PyMongo: {e}")

# Example usage
mongo_client = connect_to_mongodb()

