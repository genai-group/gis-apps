#!/usr/bin/python

#%%
from config.variables import *
from config.modules import *
from config.clients import *

###############################
####    Basic Functions    ####
###############################

#%%

def map_func(func, *iterables):
    """
    Map a function to a list of iterables.
    """
    
    try:
        return list(map(func, *iterables))
    
    except Exception as e:
        raise ValueError(f"Failed to map function '{func}' to iterables '{iterables}'. Error: {e}")
        return []

def filter_func(func, iterable):
    """
    Filter an iterable using a function.
    """
    try:
        return list(filter(func, iterable))
    
    except Exception as e:
        raise ValueError(f"Failed to filter iterable '{iterable}' using function '{func}'. Error: {e}")
        return []

## Datetime functions

"""
assert to_unix("2023-08-09 12:00:00") == 1691582400
assert from_unix(1691582400) == datetime(2023, 8, 9, 12, 0, tzinfo=timezone.utc)
"""

def to_unix(date_str: str, tz_str: Optional[str] = None) -> int:
    """
    Convert a date string to a timezone aware UNIX timestamp using dateutil parsing.

    :param date_str: String representation of the date.
    :param tz_str: Timezone string. If None, it's treated as UTC.
    :return: UNIX timestamp.
    """

    try:
        # Convert string to datetime object using dateutil parsing
        if isinstance(date_str, str):
            dt = parse(date_str)
        else:
            dt = date_str

        # If the parsed datetime is naive, attach the appropriate timezone
        if dt.tzinfo is None:
            if tz_str:
                import pytz
                tz = pytz.timezone(tz_str)
                dt = tz.localize(dt)
            else:
                dt = dt.replace(tzinfo=timezone.utc)
        
        # Convert datetime to UNIX timestamp
        return int(dt.timestamp())
    
    except Exception as e:
        raise ValueError(f"Failed to convert date string '{date_str}' to UNIX timestamp. Error: {e}")
        return None

def from_unix(unix_timestamp: int, tz_str: Optional[str] = None) -> datetime:
    """
    Convert a UNIX timestamp to a timezone aware datetime object.

    :param unix_timestamp: UNIX timestamp to convert.
    :param tz_str: Timezone string. If None, it's returned as UTC.
    :return: Timezone aware datetime object.
    """

    try:
        # Convert UNIX timestamp to datetime
        dt = datetime.fromtimestamp(unix_timestamp, tz=timezone.utc)

        # If timezone is provided, adjust datetime
        if tz_str:
            import pytz
            tz = pytz.timezone(tz_str)
            dt = dt.astimezone(tz)

        return dt
    
    except Exception as e:
        raise ValueError(f"Failed to convert UNIX timestamp '{unix_timestamp}' to datetime. Error: {e}")
        return None


############################
####    S3 Functions    ####
############################

def s3_create_bucket(bucket_name: str, region: Optional[str] = None) -> None:
    """
    Create a new S3 bucket in a specified region.

    Args:
        bucket_name (str): The name of the bucket to create.
        region (str, optional): The AWS region in which to create the bucket. Defaults to None.

    Returns:
        None
    """
    try:
        s3_client = boto3.client('s3', region_name=region)
        if region is None:
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            location = {'LocationConstraint': region}
            s3_client.create_bucket(Bucket=bucket_name, CreateBucketConfiguration=location)
        print(f"Bucket '{bucket_name}' created")
    except ClientError as e:
        print(f"Error creating bucket: {e}")

def s3_list_items(bucket_name: str) -> None:
    """
    List all items in a specified S3 bucket.

    Args:
        bucket_name (str): The name of the bucket to list items from.

    Returns:
        None
    """
    try:
        s3_client = boto3.client('s3')
        contents = s3_client.list_objects_v2(Bucket=bucket_name).get('Contents', [])
        for item in contents:
            print(item['Key'])
    except ClientError as e:
        print(f"Error listing items in bucket: {e}")

def empty_and_delete_bucket(bucket_name: str) -> None:
    """
    Empty and delete a specified S3 bucket.

    Args:
        bucket_name (str): The name of the bucket to empty and delete.

    Returns:
        None
    """
    try:
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(bucket_name)
        bucket.objects.all().delete()
        bucket.delete()
        print(f"Bucket '{bucket_name}' emptied and deleted")
    except ClientError as e:
        print(f"Error in emptying and deleting bucket: {e}")

def upload_file(bucket_name: str, file_path: str, object_name: Optional[str] = None) -> None:
    """
    Upload a file to an S3 bucket.

    Args:
        bucket_name (str): The name of the bucket to upload to.
        file_path (str): The file path to upload.
        object_name (str, optional): The object name in the bucket. Defaults to file_path if None.

    Returns:
        None
    """
    if object_name is None:
        object_name = file_path

    try:
        s3_client = boto3.client('s3')
        with open(file_path, 'rb') as file:
            s3_client.upload_fileobj(file, bucket_name, object_name)
        print(f"File '{file_path}' uploaded to '{bucket_name}/{object_name}'")
    except ClientError as e:
        print(f"Error uploading file: {e}")

def s3_download_file(bucket_name: str, object_name: str, file_path: Optional[str] = None) -> None:
    """
    Download a file from an S3 bucket.

    Args:
        bucket_name (str): The name of the S3 bucket.
        object_name (str): The object name in the bucket to download.
        file_path (str, optional): The local file path to save the downloaded file. Defaults to object_name if None.

    Returns:
        None
    """
    if file_path is None:
        file_path = object_name

    try:
        s3_client = boto3.client('s3')
        with open(file_path, 'wb') as file:
            s3_client.download_fileobj(bucket_name, object_name, file)
        print(f"File '{object_name}' downloaded from '{bucket_name}' to '{file_path}'")
    except ClientError as e:
        print(f"Error downloading file: {e}")

def s3_update_bucket_policy(bucket_name: str, policy: dict) -> None:
    """
    Update the policy of an S3 bucket.

    Args:
        bucket_name (str): The name of the S3 bucket.
        policy (dict): The policy in dictionary format to apply to the bucket.

    Returns:
        None
    """
    policy_json = json.dumps(policy)
    try:
        s3_client = boto3.client('s3')
        s3_client.put_bucket_policy(Bucket=bucket_name, Policy=policy_json)
        print(f"Policy updated for bucket '{bucket_name}'")
    except ClientError as e:
        print(f"Error updating bucket policy: {e}")

def s3_generate_presigned_url(bucket_name: str, object_name: str, expiration: int = 3600) -> None:
    """
    Generate a presigned URL for an S3 object.

    Args:
        bucket_name (str): The name of the S3 bucket.
        object_name (str): The object name in the bucket for which to generate the URL.
        expiration (int, optional): Time in seconds for the presigned URL to remain valid. Defaults to 3600 seconds (1 hour).

    Returns:
        None
    """
    try:
        s3_client = boto3.client('s3')
        url = s3_client.generate_presigned_url('get_object',
                                               Params={'Bucket': bucket_name, 'Key': object_name},
                                               ExpiresIn=expiration)
        print(f"Presigned URL: {url}")
    except ClientError as e:
        print(f"Error generating presigned URL: {e}")


###############################
####    Minio Functions    ####
###############################

import boto3
from botocore.exceptions import ClientError
from typing import Optional

def minio_create_bucket(bucket_name: str, endpoint_url: str, access_key: str, secret_key: str) -> None:
    """
    Create a new bucket in MinIO.

    Args:
        bucket_name (str): The name of the bucket to create.
        endpoint_url (str): The endpoint URL of the MinIO server.
        access_key (str): MinIO access key.
        secret_key (str): MinIO secret key.

    Returns:
        None
    """
    assert bucket_name, "Bucket name must be provided"
    assert endpoint_url, "Endpoint URL must be provided"
    assert access_key, "Access key must be provided"
    assert secret_key, "Secret key must be provided"

    try:
        minio_client = boto3.client('s3', endpoint_url=endpoint_url,
                                    aws_access_key_id=access_key,
                                    aws_secret_access_key=secret_key)
        minio_client.create_bucket(Bucket=bucket_name)
        print(f"Bucket '{bucket_name}' created")
    except ClientError as e:
        print(f"Error creating bucket: {e}")

def minio_list_items(bucket_name: str, endpoint_url: str, access_key: str, secret_key: str) -> None:
    """
    List all items in a specified MinIO bucket.

    Args:
        bucket_name (str): The name of the bucket to list items from.
        endpoint_url (str): The endpoint URL of the MinIO server.
        access_key (str): MinIO access key.
        secret_key (str): MinIO secret key.

    Returns:
        None
    """
    assert bucket_name, "Bucket name must be provided"
    assert endpoint_url, "Endpoint URL must be provided"
    assert access_key, "Access key must be provided"
    assert secret_key, "Secret key must be provided"

    try:
        minio_client = boto3.client('s3', endpoint_url=endpoint_url,
                                    aws_access_key_id=access_key,
                                    aws_secret_access_key=secret_key)
        contents = minio_client.list_objects_v2(Bucket=bucket_name).get('Contents', [])
        for item in contents:
            print(item['Key'])
    except ClientError as e:
        print(f"Error listing items in bucket: {e}")

def minio_empty_and_delete_bucket(bucket_name: str, endpoint_url: str, access_key: str, secret_key: str) -> None:
    """
    Empty and delete a specified MinIO bucket.

    Args:
        bucket_name (str): The name of the bucket to empty and delete.
        endpoint_url (str): The endpoint URL of the MinIO server.
        access_key (str): MinIO access key.
        secret_key (str): MinIO secret key.

    Returns:
        None
    """
    assert bucket_name, "Bucket name must be provided"
    assert endpoint_url, "Endpoint URL must be provided"
    assert access_key, "Access key must be provided"
    assert secret_key, "Secret key must be provided"

    try:
        s3_resource = boto3.resource('s3', endpoint_url=endpoint_url,
                                     aws_access_key_id=access_key,
                                     aws_secret_access_key=secret_key)
        bucket = s3_resource.Bucket(bucket_name)
        bucket.objects.all().delete()
        bucket.delete()
        print(f"Bucket '{bucket_name}' emptied and deleted")
    except ClientError as e:
        print(f"Error in emptying and deleting bucket: {e}")

def minio_upload_file(bucket_name: str, file_path: str, endpoint_url: str, access_key: str, secret_key: str, object_name: Optional[str] = None) -> None:
    """
    Upload a file to a MinIO bucket.

    Args:
        bucket_name (str): The name of the bucket to upload to.
        file_path (str): The file path to upload.
        endpoint_url (str): The endpoint URL of the MinIO server.
        access_key (str): MinIO access key.
        secret_key (str): MinIO secret key.
        object_name (str, optional): The object name in the bucket. Defaults to file_path if None.

    Returns:
        None
    """
    assert bucket_name, "Bucket name must be provided"
    assert file_path, "File path must be provided"
    assert endpoint_url, "Endpoint URL must be provided"
    assert access_key, "Access key must be provided"
    assert secret_key, "Secret key must be provided"

    object_name = object_name if object_name else file_path

    try:
        minio_client = boto3.client('s3', endpoint_url=endpoint_url,
                                    aws_access_key_id=access_key,
                                    aws_secret_access_key=secret_key)
        with open(file_path, 'rb') as file:
            minio_client.upload_fileobj(file, bucket_name, object_name)
        print(f"File '{file_path}' uploaded to '{bucket_name}/{object_name}'")
    except ClientError as e:
        print(f"Error uploading file: {e}")


def minio_download_file(bucket_name: str, object_name: str, endpoint_url: str, access_key: str, secret_key: str, file_path: Optional[str] = None) -> None:
    """
    Download a file from a MinIO bucket.

    Args:
        bucket_name (str): The name of the MinIO bucket.
        object_name (str): The object name in the bucket to download.
        endpoint_url (str): The endpoint URL of the MinIO server.
        access_key (str): MinIO access key.
        secret_key (str): MinIO secret key.
        file_path (str, optional): The local file path to save the downloaded file. Defaults to object_name if None.

    Returns:
        None
    """
    assert bucket_name, "Bucket name must be provided"
    assert object_name, "Object name must be provided"
    assert endpoint_url, "Endpoint URL must be provided"
    assert access_key, "Access key must be provided"
    assert secret_key, "Secret key must be provided"

    file_path = file_path if file_path else object_name

    try:
        minio_client = boto3.client('s3', endpoint_url=endpoint_url,
                                    aws_access_key_id=access_key,
                                    aws_secret_access_key=secret_key)
        with open(file_path, 'wb') as file:
            minio_client.download_fileobj(bucket_name, object_name, file)
        print(f"File '{object_name}' downloaded from '{bucket_name}' to '{file_path}'")
    except ClientError as e:
        print(f"Error downloading file: {e}")





##################################
####    Postgres Functions    ####
##################################


# Postgres Functions

# def create_table_from_json(conn, json_data, parent_table=None, parent_key=None):
#     """
#     Recursively creates tables from a JSON object.

#     Args:
#     conn (psycopg2.connection): PostgreSQL database connection.
#     json_data (dict): A JSON object.
#     parent_table (str, optional): Name of the parent table for nested objects.
#     parent_key (str, optional): Primary key of the parent table for linking.
#     """
#     if isinstance(json_data, dict):
#         # Determine table name
#         table_name = parent_table + "_child" if parent_table else "root_table"
#         columns = []
#         foreign_key = None
#         if parent_table and parent_key:
#             foreign_key = parent_key + "_id"
#             columns.append(foreign_key + " INTEGER REFERENCES " + parent_table + "(" + parent_key + ")")

#         # Process each key in the JSON object
#         for key, value in json_data.items():
#             if isinstance(value, dict):
#                 # Recursive call for nested objects
#                 create_table_from_json(conn, value, table_name, key)
#             elif isinstance(value, list):
#                 # Handle lists (arrays)
#                 for item in value:
#                     if isinstance(item, dict):
#                         create_table_from_json(conn, item, table_name, key)
#             else:
#                 # Simple data types
#                 columns.append(key + " " + get_sql_data_type(value))

#         # Create table
#         create_table_query = "CREATE TABLE IF NOT EXISTS " + table_name + " (" + ", ".join(columns) + ")"
#         execute_sql(conn, create_table_query)

#     elif isinstance(json_data, list):
#         # Handle JSON arrays
#         for item in json_data:
#             if isinstance(item, dict):
#                 create_table_from_json(conn, item, parent_table, parent_key)

def get_sql_data_type(value):
    """
    Maps a Python data type to an SQL data type.
    
    Args:
    value: A Python value.
    
    Returns:
    str: An SQL data type.
    """
    if isinstance(value, int):
        return "INTEGER"
    elif isinstance(value, float):
        return "REAL"
    elif isinstance(value, bool):
        return "BOOLEAN"
    else:
        return "TEXT"

def execute_sql(conn, query):
    """
    Executes an SQL query using the given connection.

    Args:
    conn (psycopg2.connection): PostgreSQL database connection.
    query (str): SQL query to execute.
    """
    with conn.cursor() as cursor:
        cursor.execute(query)
    conn.commit()

# Example usage
conn = connect_to_postgres()
json_data = '{"name": "John", "age": 30, "address": {"street": "123 Main St", "city": "Anytown"}, "hobbies": ["reading", "hiking"]}'
create_table_from_json(conn, json.loads(json_data))

def normalize_name(name: str) -> Optional[str]:
    """
    Normalizes a name string by performing various transformations.

    This function performs the following operations:
    1. Trims leading and trailing whitespaces.
    2. Converts the name to lowercase.
    3. Removes diacritics and accents from characters.
    4. Removes punctuation and special characters.
    5. Normalizes whitespaces to single spaces.
    
    Parameters:
    name (str): The name string to be normalized.

    Returns:
    Optional[str]: The normalized name as a string or None if the input is invalid.

    Raises:
    ValueError: If the input is not a string or is an empty string.
    """

    try:
        # Input Validation
        if not isinstance(name, str) or not name:
            raise ValueError("Name must be a non-empty string")

        # Trimming, Case Normalization, and Whitespace Normalization
        name = name.strip().lower()

        # Removing Diacritics
        name = ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')

        # Removing Punctuation and Special Characters
        name = re.sub(r"[^\w\s]", '', name)

        # Whitespace Normalization (Again)
        name = re.sub(r"\s+", ' ', name)

        return name

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example usage
try:
    normalized_name = normalize_name("Dr. María-José O'Neill")
    print(normalized_name)  # Output: 'maria jose oneill'
except ValueError as ve:
    print(f"ValueError: {ve}")


########################################
####    Creating the Feature Map    ####
########################################

def read_sample(df: pd.DataFrame, sample_size: int = 100) -> pd.DataFrame:
    """
    Reads a random sample of rows from a pandas DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to sample from.
    sample_size (int): The number of rows to sample. Default is 100.

    Returns:
    pd.DataFrame: A DataFrame containing the sampled rows.

    Raises:
    ValueError: If the provided DataFrame is empty or the sample size is non-positive.
    """

    assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame"
    assert isinstance(sample_size, int) and sample_size > 0, "Sample size must be a positive integer"

    try:
        return df.sample(n=min(sample_size, len(df)), random_state=1)
    except ValueError as e:
        raise ValueError("Error in sampling data: " + str(e))

def identify_column_type(column: pd.Series) -> str:
    """
    Identify the column type based on a sample of data using pandas.api.types,
    with specific handling for text and categorical data.

    Parameters:
    column (pd.Series): A pandas Series representing a column from a DataFrame.

    Returns:
    str: The identified data type of the column.

    Raises:
    TypeError: If the input is not a pandas Series.
    Exception: For errors in identifying column types.
    """

    assert isinstance(column, pd.Series), "Input must be a pandas Series"

    try:
        if types.is_numeric_dtype(column):
            if types.is_float_dtype(column):
                return 'float'
            elif types.is_integer_dtype(column):
                return 'integer'
            else:
                return 'numeric'
        elif types.is_datetime64_any_dtype(column):
            return 'datetime'
        elif types.is_timedelta64_dtype(column):
            return 'time delta'
        elif types.is_object_dtype(column):
            sample_str = column.dropna().astype(str)
            if sample_str.str.contains(r"^\s*\+?\d[\d -]{8,12}\d\s*$").any():
                return 'phone'
            elif sample_str.str.contains(r"^\s*[^@]+@[^@]+\.[^@]+\s*$").any():
                return 'email'
            elif sample_str.str.contains(r"^\s*\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b\s*$").any():
                return 'ip address'
            else:
                unique_vals = sample_str.nunique()
                if unique_vals == len(column):
                    return 'text'
                elif unique_vals < 10:
                    return 'text_categorical_low_cardinality'
                else:
                    return 'text_categorical_high_cardinality'
        else:
            return 'unknown'
    except Exception as e:
        raise TypeError("Error in identifying column type: " + str(e))
        
def create_json_object(df: pd.DataFrame, column_types: dict) -> str:
    """
    Creates a JSON object with column names, types, and categories.

    Parameters:
    df (pd.DataFrame): The DataFrame whose columns are being analyzed.
    column_types (dict): A dictionary with column names as keys and their data types as values.

    Returns:
    str: A JSON string representation of the column information.

    Raises:
    TypeError: If the inputs are not of the correct type.
    """

    assert isinstance(df, pd.DataFrame), "First argument must be a pandas DataFrame"
    assert isinstance(column_types, dict), "Second argument must be a dictionary"

    try:
        json_object = {"columns": []}
        for col, col_type in column_types.items():
            json_object["columns"].append({
                "column_name": col,
                "column_type": col_type
            })
        return json.dumps(json_object, indent=4)
    except Exception as e:
        raise TypeError("Error in creating JSON object: " + str(e))

def categorize_columns(df_sample: pd.DataFrame) -> dict:
    """
    Categorizes columns of the DataFrame based on their data types.

    Parameters:
    df_sample (pd.DataFrame): A pandas DataFrame sample whose columns are to be categorized.

    Returns:
    dict: A dictionary with column names as keys and determined data types as values.

    Raises:
    TypeError: If the input is not a pandas DataFrame.
    Exception: For errors in categorizing column types.
    """
    
    assert isinstance(df_sample, pd.DataFrame), "Input must be a pandas DataFrame"

    column_types = {}
    try:
        for col in df_sample.columns:
            column_type = identify_column_type(df_sample[col])
            column_types[col] = column_type
        return column_types
    except Exception as e:
        raise Exception(f"Error in categorizing columns: {str(e)}")


"""

try:
    df_sample = read_sample(df)
    column_types = categorize_columns(df_sample)
    json_output = create_json_object(df, column_types)
    print(json_output)
except Exception as e:
    print("An error occurred:", e)

"""                        



########################################
####    Auto Feature Engineering    ####
########################################


def prepare_objects(object: Union[Dict, List]) -> Union[Dict, List]:
    """
    Recursively prepares a JSON object for feature engineering.

    Args:
    object (Union[Dict, List]): A JSON object.

    Returns:
    Union[Dict, List]: A JSON object prepared for feature engineering.
    """
    if isinstance(object, dict):
        object = list(object)
    elif isinstance(object, pd.DataFrame):
        object = object.to_dict('records')
    elif isinstance(object, list):
        print(f"Object is already a list.")
    else:
        raise ValueError(f"Object type '{type(object)}' is not supported.")
    
    # Process each item in the JSON object
    for item in object:
        if isinstance(item, dict):
            # Recursive call for nested objects
            prepare_objects(item)
        elif isinstance(item, list):
            # Handle lists (arrays)
            for item in value:
                if isinstance(item, dict):
                    prepare_objects(item)
        else:
            # Simple data types
            object[key] = value    
    if isinstance(object, dict):
        # Process each key in the JSON object
        for key, value in object.items():
            if isinstance(value, dict):
                # Recursive call for nested objects
                object[key] = prepare_objects(value)
            elif isinstance(value, list):
                # Handle lists (arrays)
                for item in value:
                    if isinstance(item, dict):
                        prepare_objects(item)
            else:
                # Simple data types
                object[key] = value

    elif isinstance(object, list):
        # Handle JSON arrays
        for item in object:
            if isinstance(item, dict):
                prepare_objects(item)

    return object


########################
#### XML to JSON    ####
########################

def parse_element_stream(element_iter: Iterator[etree.Element]) -> Dict[str, Any]:
    """
    Parses an XML element stream, preserving its attributes, text, and children.

    Parameters:
    element_iter (Iterator[etree.Element]): An iterator over XML elements.

    Returns:
    Dict[str, Any]: A dictionary representation of the XML element.
    """
    def add_child(parent_dict: Dict[str, Any], child: etree.Element):
        child_dict = {"@attributes": child.attrib, "#text": child.text or ""}
        if child.tag in parent_dict:
            if not isinstance(parent_dict[child.tag], list):
                parent_dict[child.tag] = [parent_dict[child.tag]]
            parent_dict[child.tag].append(child_dict)
        else:
            parent_dict[child.tag] = child_dict

    root_dict = {}
    for event, elem in element_iter:
        if event == 'start':
            current_dict = {"@attributes": elem.attrib, "#text": elem.text or ""}
            if not root_dict:
                root_dict = current_dict
            else:
                add_child(root_dict, elem)
        elif event == 'end':
            elem.clear()
    return root_dict

@ray.remote
def xml_to_json_stream(xml_data: Union[str, bytes]) -> str:
    try:
        context = etree.iterparse(etree.BytesIO(xml_data) if isinstance(xml_data, bytes) else xml_data, events=('start', 'end'))
        xml_dict = parse_element_stream(context)
        return json.dumps(xml_dict, ensure_ascii=False, indent=2)
    except etree.XMLSyntaxError as e:
        raise ValueError(f"Invalid XML data: {e}")

# Example usage
"""

xml_str = "<root><child id='1'>Hello</child><child id='2'>World</child></root>"
try:
    json_str = xml_to_json_stream(xml_str)
    print(json_str)
except ValueError as e:
    print(e)

"""


#################################
####    MongoDB Functions    ####
#################################

def mongodb_connect_to_database(db_name: str = 'gis') -> Database:
    """
    Connect to or create a MongoDB database.

    Args:
        db_name (str, optional): Name of the database to connect to or create. Defaults to 'gis'.

    Returns:
        Database: MongoDB Database object.
    """
    assert isinstance(db_name, str), "Database name must be a string"

    try:
        client = MongoClient('mongodb://localhost:27017/')
        return client[db_name]
    except Exception as e:
        raise ConnectionError(f"Failed to connect to the database: {e}")

def mongodb_create_collection(db: Database, collection_name: str = 'base') -> Collection:
    """
    Create a collection in the given database if it doesn't exist.

    Args:
        db (Database): The database in which to create the collection.
        collection_name (str, optional): The name of the collection. Defaults to 'base'.

    Returns:
        Collection: MongoDB Collection object.
    """
    assert isinstance(collection_name, str), "Collection name must be a string"

    try:
        return db[collection_name]
    except Exception as e:
        raise RuntimeError(f"Failed to create collection: {e}")

def mongodb_bulk_load_objects(collection: Collection, objects: List[dict]) -> None:
    """
    Bulk loads objects into a collection.

    Args:
        collection (Collection): The collection to load objects into.
        objects (List[dict]): A list of dictionaries representing the objects to be loaded.

    Raises:
        ValueError: If objects is not a list of dictionaries.
    """
    if not all(isinstance(item, dict) for item in objects):
        raise ValueError("Objects must be a list of dictionaries")

    try:
        collection.insert_many(objects)
    except Exception as e:
        raise RuntimeError(f"Failed to bulk load objects: {e}")

def mongodb_search_by_guid(collection: Collection, guid: str) -> Optional[dict]:
    """
    Searches the collection for a specific GUID.

    Args:
        collection (Collection): The collection to search.
        guid (str): The GUID to search for.

    Returns:
        Optional[dict]: The found document or None.
    """
    assert isinstance(guid, str), "GUID must be a string"

    try:
        return collection.find_one({"guid": guid})
    except Exception as e:
        raise RuntimeError(f"Failed to search for GUID: {e}")

def mongodb_destroy_objects(collection: Collection, ids: Optional[List[str]] = None) -> None:
    """
    Destroys objects in a collection based on an array of IDs.

    Args:
        collection (Collection): The collection from which to remove objects.
        ids (Optional[List[str]], optional): List of IDs to remove. If None, removes all objects.

    Raises:
        ValueError: If ids is not a list of strings.
    """
    try:
        if ids is None:
            collection.delete_many({})
        else:
            if not all(isinstance(id_, str) for id_ in ids):
                raise ValueError("IDs must be a list of strings")
            collection.delete_many({"_id": {"$in": ids}})
    except Exception as e:
        raise RuntimeError(f"Failed to destroy objects: {e}")

def mongodb_delete_collection(db: Database, collection_name: str) -> None:
    """
    Deletes a MongoDB collection.

    Args:
        db (Database): The database containing the collection.
        collection_name (str): The name of the collection to delete.

    Raises:
        ValueError: If collection_name is not a string.
    """
    assert isinstance(collection_name, str), "Collection name must be a string"

    try:
        db.drop_collection(collection_name)
    except Exception as e:
        raise RuntimeError(f"Failed to delete collection: {e}")

## Async functions to load and delete data from MongoDB

async def mongodb_async_bulk_load_objects(collection: Collection, objects: List[dict]) -> None:
    """
    Asynchronously bulk loads objects into a collection.

    Args:
        collection (Collection): The collection to load objects into.
        objects (List[dict]): A list of dictionaries representing the objects to be loaded.

    Raises:
        ValueError: If objects is not a list of dictionaries.
    """
    if not all(isinstance(item, dict) for item in objects):
        raise ValueError("Objects must be a list of dictionaries")

    try:
        # Motor uses the same syntax as PyMongo for bulk operations
        await collection.insert_many(objects)
    except Exception as e:
        raise RuntimeError(f"Failed to bulk load objects asynchronously: {e}")

async def mongodb_async_destroy_objects(collection: Collection, ids: Optional[List[str]] = None) -> None:
    """
    Asynchronously destroys objects in a collection based on an array of IDs.

    Args:
        collection (Collection): The collection from which to remove objects.
        ids (Optional[List[str]], optional): List of IDs to remove. If None, removes all objects.

    Raises:
        ValueError: If ids is not a list of strings.
    """
    try:
        if ids is None:
            await collection.delete_many({})
        else:
            if not all(isinstance(id_, str) for id_ in ids):
                raise ValueError("IDs must be a list of strings")
            await collection.delete_many({"_id": {"$in": ids}})
    except Exception as e:
        raise RuntimeError(f"Failed to destroy objects asynchronously: {e}")
    

###############################
####    Kafka Functions    ####
###############################

def kafka_create_topic(topic_name: str, num_partitions: int = 1, replication_factor: int = 1, kafka_client: AdminClient = kafka_client) -> None:
    """
    Create a new topic in Kafka.

    Parameters:
    kafka_client (AdminClient): The Kafka AdminClient instance.
    topic_name (str): Name of the topic to be created.
    num_partitions (int): Number of partitions for the topic (default: 1).
    replication_factor (int): Replication factor for the topic (default: 1).

    Raises:
    AssertionError: If any of the input parameters are not in expected format.
    KafkaException: If there is an error in creating the topic.
    """
    assert isinstance(topic_name, str), "Topic name must be a string"
    assert isinstance(num_partitions, int), "Number of partitions must be an integer"
    assert isinstance(replication_factor, int), "Replication factor must be an integer"

    new_topic = NewTopic(topic_name, num_partitions=num_partitions, replication_factor=replication_factor)
    try:
        kafka_client.create_topics([new_topic])
    except KafkaException as e:
        raise KafkaException(f"Error in creating topic {topic_name}: {e}")

def kafka_delete_topic(topic_name: str, kafka_client: AdminClient = kafka_client) -> None:
    """
    Delete a topic in Kafka.

    Parameters:
    kafka_client (AdminClient): The Kafka AdminClient instance.
    topic_name (str): Name of the topic to be deleted.

    Raises:
    AssertionError: If topic_name is not a string.
    KafkaException: If there is an error in deleting the topic.
    """
    assert isinstance(topic_name, str), "Topic name must be a string"

    try:
        kafka_client.delete_topics([topic_name])
    except KafkaException as e:
        raise KafkaException(f"Error in deleting topic {topic_name}: {e}")

def kafka_list_topics(kafka_client: AdminClient = kafka_client):
    """
    List all topics in the Kafka cluster.

    Parameters:
    kafka_client (AdminClient): The Kafka AdminClient instance.

    Returns:
    dict: A dictionary of topics and their metadata.

    Raises:
    KafkaException: If there is an error in listing the topics.
    """
    try:
        return kafka_client.list_topics().topics
    except KafkaException as e:
        raise KafkaException(f"Error in listing topics: {e}")

def kafka_consume_message(bootstrap_servers: str, group_id: str, topic_name: str):
    """
    Consume a message from a Kafka topic.

    Parameters:
    bootstrap_servers (str): Comma-separated list of broker addresses.
    group_id (str): The consumer group to which the consumer belongs.
    topic_name (str): The topic to consume messages from.

    Returns:
    str or None: The consumed message or None if no message is received.

    Raises:
    Exception: If there is an error in consuming the message.
    """
    consumer = Consumer({"bootstrap.servers": bootstrap_servers, "group.id": group_id})
    consumer.subscribe([topic_name])
    try:
        message = consumer.poll(1.0)
        consumer.close()
        if message is None:
            print("No message received")
        else:
            return message
    except Exception as e:
        raise Exception(f"Error in consuming message: {e}")