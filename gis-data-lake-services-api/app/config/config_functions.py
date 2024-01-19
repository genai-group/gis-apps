#!/usr/bin/python

#%%
from config.config_variables import *
from config.config_modules import *
from config.config_clients import *

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

def chunk_text(text: str, character_split_threshold: int = 500) -> List[str]:
    """
    Chunk a string into 500 character chunks with overlap of 100 characters using langchain.

    Args:
    text (str): The text to be chunked.

    Returns:
    List[str]: A list of text chunks.
    """

    # assertions
    assert isinstance(text, str), "text must be a string"

    try:
        # chunk the text into 500 character chunks with overlap of 100 characters using langchain
        text_chunks = textwrap.wrap(obj[_text], width=character_split_threshold)
        return text_chunks
    except Exception as e:
        print(f"Errors chunking text: {e}")
        return []

###########################
####    Data Ingest    ####
###########################

# import os
# import json
# import pandas as pd
# import xmltodict

def open_file(filename: str):
    """
    Opens a file and reads its contents. If the file is a JSON, CSV, Excel, XML, XSD, or TXT file, 
    it returns the contents. If the file is CSV, Excel, XML, XSD, or certain TXT files, 
    it converts the contents to JSON format.

    Parameters:
    filename (str): The path to the file.

    Returns:
    Union[dict, list, None]: The contents of the file, possibly converted to a JSON-like structure.
    """
    assert isinstance(filename, str), "Filename must be a string"

    # Get the file extension
    _, file_extension = os.path.splitext(filename)

    try:
        if file_extension == '.json':
            with open(filename, 'r') as file:
                return json.load(file)

        elif file_extension in ['.csv', '.xlsx']:
            # Read the file using pandas and convert to JSON
            if file_extension == '.csv':
                data = pd.read_csv(filename)
            else:  # .xlsx
                data = pd.read_excel(filename)

            return data.to_dict(orient='records')

        elif file_extension in ['.xml', '.xsd']:
            # Read the file and convert XML/XSD to dict
            with open(filename, 'r') as file:
                return xmltodict.parse(file.read())

        # elif file_extension in ['.xsd']:
        #     return domutils.StringToDOM(filename)            

        elif file_extension == '.txt':
            # Read the text file (assuming simple key-value pairs)
            with open(filename, 'r') as file:
                data = {}
                for line in file:
                    key, value = line.strip().split(':')
                    data[key.strip()] = value.strip()
                return data

        else:
            print("Unsupported file format. Supported formats are JSON, CSV, Excel, XML, XSD, and TXT.")
            return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def is_json_nested(json_obj: Union[Dict[str, Any], List[Any]]) -> bool:
    """
    Determines if a JSON object is nested.

    Parameters:
    json_obj (Union[Dict[str, Any], List[Any]]): The JSON object, which can be a dictionary or a list.

    Returns:
    bool: True if the JSON object is nested, False otherwise.
    """

    def _is_nested(obj: Any) -> bool:
        if isinstance(obj, dict):
            return any(_is_nested(v) for v in obj.values())
        elif isinstance(obj, list):
            return any(_is_nested(item) for item in obj)
        return False

    return _is_nested(json_obj)

def flatten_json(json_obj: Union[Dict[str, Any], List[Any]]) -> Dict[str, Any]:
    """
    Flattens a nested JSON object into a single-level object with dot-separated keys.

    Parameters:
    json_obj (Union[Dict[str, Any], List[Any]]): The JSON object to flatten.

    Returns:
    Dict[str, Any]: The flattened JSON object.
    """
    assert isinstance(json_obj, (dict, list)), "Input must be a dictionary or a list"

    def _flatten(current_obj: Union[Dict[str, Any], List[Any]], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        items = []
        if isinstance(current_obj, dict):
            for k, v in current_obj.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                items.extend(_flatten(v, new_key, sep=sep).items())
        elif isinstance(current_obj, list):
            for i, v in enumerate(current_obj):
                new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
                items.extend(_flatten(v, new_key, sep=sep).items())
        else:
            items.append((parent_key, current_obj))

        return dict(items)

    return _flatten(json_obj)

"""

data = 'config/data/fake_airline_manifest_0_hours.json'
data = 'config/data/iso.json'
data = 'config/data/worldcities.csv'

data = fake_customs_report_hour_0

temp_data = open_file(data)
if is_json_nested(temp_data):
    temp_data = flatten_json(temp_data)

temp_data

"""


###############################
####   File Fingerprint    ####
###############################    

def generate_fingerprint(json_data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> str:
    """
    Generate a fingerprint for the JSON data by creating a sorted string of unique keys.
    This function handles both dictionaries and arrays of dictionaries by considering the keys
    of objects within the array.

    Args:
    json_data (Union[Dict[str, Any], List[Dict[str, Any]]]): The JSON data (or list of JSON data) 
    for which the fingerprint is to be generated.

    Returns:
    str: A SHA256 hash representing the fingerprint of the JSON schema.

    Raises:
    TypeError: If the input is not a dictionary or a list of dictionaries.
    """
    assert isinstance(json_data, (dict, list)), "Input must be a dictionary or a list of dictionaries."

    keys_set: Set[str] = set()

    def recurse_extract_keys(data: Any, parent_key: str = '') -> None:
        if isinstance(data, dict):
            for key, value in data.items():
                full_key = f"{parent_key}.{key}" if parent_key else key
                keys_set.add(full_key)
                recurse_extract_keys(value, full_key)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    recurse_extract_keys(item, parent_key)

    try:
        if isinstance(json_data, list):
            # Process each dictionary in the list separately
            for item in json_data:
                assert isinstance(item, dict), "List must only contain dictionaries."
                recurse_extract_keys(item)
        else:
            # Process a single dictionary
            recurse_extract_keys(json_data)

        sorted_keys = sorted(keys_set)
        keys_string = ','.join(sorted_keys)
        fingerprint = hashlib.sha256(keys_string.encode()).hexdigest()
        return fingerprint
    except Exception as e:
        raise RuntimeError(f"Error in generating fingerprint: {e}")

def verify_json_schema(original_json: Dict[str, Any], new_json: Dict[str, Any]) -> bool:
    """
    Compare the fingerprint of the original JSON and new JSON to verify if they have the same schema.

    Args:
    original_json (Dict[str, Any]): The original JSON data.
    new_json (Dict[str, Any]): The new JSON data to compare against the original.

    Returns:
    bool: True if both JSON data have the same schema, False otherwise.

    Raises:
    RuntimeError: If there's an error during the fingerprint generation.
    """
    try:
        original_fingerprint = generate_fingerprint(original_json)
        new_fingerprint = generate_fingerprint(new_json)

        return original_fingerprint == new_fingerprint
    except Exception as e:
        raise RuntimeError(f"Schema verification failed: {e}")

## Datetime functions

"""
assert to_unix("2023-08-09 12:00:00") == 1691582400
assert from_unix(1691582400) == datetime(2023, 8, 9, 12, 0, tzinfo=timezone.utc)
"""

# def to_unix(date_str: str, tz_str: Optional[str] = None) -> int:
#     """
#     Convert a date string to a timezone aware UNIX timestamp using dateutil parsing.

#     :param date_str: String representation of the date.
#     :param tz_str: Timezone string. If None, it's treated as UTC.
#     :return: UNIX timestamp.
#     """

#     try:
#         # Convert string to datetime object using dateutil parsing
#         if isinstance(date_str, str):
#             dt = parse(date_str)
#         else:
#             dt = date_str

#         # If the parsed datetime is naive, attach the appropriate timezone
#         if dt.tzinfo is None:
#             if tz_str:
#                 import pytz
#                 tz = pytz.timezone(tz_str)
#                 dt = tz.localize(dt)
#             else:
#                 dt = dt.replace(tzinfo=timezone.utc)
        
#         # Convert datetime to UNIX timestamp
#         return int(dt.timestamp())
    
#     except Exception as e:
#         raise ValueError(f"Failed to convert date string '{date_str}' to UNIX timestamp. Error: {e}")
#         return None

def to_unix(date_input: Union[datetime, date, str]) -> int:
    """
    Convert a datetime object, a date object, or a string representation of a datetime 
    to a Unix timestamp.

    Args:
    date_input (datetime | date | str): The datetime object, date object, 
                                        or a string representation of a datetime.

    Returns:
    int: The Unix timestamp corresponding to the provided datetime.

    Raises:
    ValueError: If the date_input is neither a datetime object, a date object, nor a string.
    """
    # Check if the input is a datetime object, a date object, or a string
    if not isinstance(date_input, (datetime, date, str)):
        raise ValueError("date_input must be either a datetime object, a date object, or a string")

    # If the input is a date object, convert it to a datetime object at midnight
    if isinstance(date_input, date) and not isinstance(date_input, datetime):
        date_input = datetime(date_input.year, date_input.month, date_input.day)

    # If the input is a string, parse it to a datetime object
    if isinstance(date_input, str):
        date_input = parse(date_input)

    # Check if the datetime object is timezone aware
    if date_input.tzinfo is not None and date_input.tzinfo.utcoffset(date_input) is not None:
        # Convert to UTC
        date_input = date_input.astimezone(pytz.utc)

    # Convert to Unix timestamp
    timestamp = int(date_input.timestamp())
    return timestamp

def from_unix(unix_timestamp: int, tz_str: Optional[str] = None) -> datetime:
    """
    Convert a UNIX timestamp to a timezone aware datetime object.

    Args:
    unix_timestamp (int): UNIX timestamp to convert.
    tz_str (Optional[str]): Timezone string. If None, it's returned as UTC.

    Returns:
    datetime: Timezone aware datetime object.

    Raises:
    ValueError: If conversion fails or invalid timezone is provided.
    """
    try:
        # Convert UNIX timestamp to datetime in UTC
        dt = datetime.fromtimestamp(unix_timestamp, tz=timezone.utc)

        # If timezone is provided, adjust datetime to that timezone
        if tz_str:
            tz = pytz.timezone(tz_str)
            dt = dt.astimezone(tz)

        return dt

    except Exception as e:
        raise ValueError(f"Failed to convert UNIX timestamp '{unix_timestamp}' to datetime. Error: {e}")


# def from_unix(unix_timestamp: int, tz_str: Optional[str] = None) -> datetime:
#     """
#     Convert a UNIX timestamp to a timezone aware datetime object.

#     :param unix_timestamp: UNIX timestamp to convert.
#     :param tz_str: Timezone string. If None, it's returned as UTC.
#     :return: Timezone aware datetime object.
#     """

#     try:
#         # Convert UNIX timestamp to datetime
#         dt = datetime.fromtimestamp(unix_timestamp, tz=timezone.utc)

#         # If timezone is provided, adjust datetime
#         if tz_str:
#             import pytz
#             tz = pytz.timezone(tz_str)
#             dt = dt.astimezone(tz)

#         return dt
    
#     except Exception as e:
#         raise ValueError(f"Failed to convert UNIX timestamp '{unix_timestamp}' to datetime. Error: {e}")
#         return None

## Cryptography

def generate_key():
    """
    Generate a Fernet encryption key and print it.
    It's recommended to store this key securely (e.g., environment variables, secrets manager, etc.)
    """
    try:
        key = Fernet.generate_key()
        return key.decode()
    except Exception as e:
        logging.error(f"An error occurred in generate_key: {e}", exc_info=True)
        return None

def serialize_data(data: Any) -> bytes:
    """
    Serialize different data types into bytes.

    Parameters:
    - data (Any): The data to be serialized.

    Returns:
    - bytes: The serialized data.
    """
    if isinstance(data, (dict, list, str, int, float)):
        # Directly serializable to JSON
        return json.dumps(data).encode('utf-8')
    else:
        # For other data types like numpy arrays, pandas dataframes, images, etc.
        return pickle.dumps(data)

def encrypt_data(data: Any) -> Optional[bytes]:
    """
    Encrypt any data type using the Fernet symmetric encryption.

    Parameters:
    - data (Any): The data to be encrypted.

    Returns:
    - Optional[bytes]: The encrypted data, or None if an error occurs.
    """
    try:
        key = os.environ.get("SECRET_KEY_ENV_VAR")
        if not key:
            raise ValueError("crypto environment variable must be set.")
    
        cipher = Fernet(key)
        serialized_data = serialize_data(data)
        encrypted_data = cipher.encrypt(serialized_data)
        return encrypted_data
    
    except Exception as e:
        print(f"Encryption failed: {e}")
        return None

def deserialize_data(serialized_data: bytes) -> Any:
    """
    Deserialize bytes into the original data type.

    Parameters:
    - serialized_data (bytes): The serialized data.

    Returns:
    - Any: The deserialized data.
    """
    try:
        return json.loads(serialized_data)
    except json.JSONDecodeError:
        return pickle.loads(serialized_data)

def decrypt_data(encrypted_data: bytes) -> Optional[Any]:
    """
    Decrypt data previously encrypted with the `encrypt_data` function.

    Parameters:
    - encrypted_data (bytes): The encrypted data.

    Returns:
    - Optional[Any]: The decrypted data or None if decryption fails.
    """
    try:
        key = os.environ.get("SECRET_KEY_ENV_VAR")
        if not key:
            raise ValueError(f"{os.environ.get('SECRET_KEY_ENV_VAR')} environment variable must be set.")
        
        cipher = Fernet(key)
        decrypted_data = cipher.decrypt(encrypted_data)
        return deserialize_data(decrypted_data)

    except InvalidToken:
        print("Decryption failed due to an invalid token. Key mismatch or corrupted data.")
        return None
    except Exception as e:
        print(f"Decryption failed: {e}")
        return None


####################################################################################################################################
#####    GIS Lock & Unlock Functions (Truly Random Number Generation based on hardware-based random number generators (RNGs)    ####
####################################################################################################################################

def generate_near_truly_random_arrays(n, low=1, high=1000000):
    """
    Generates a set of n 2D numpy arrays with n rows and n columns, filled with truly random integers.

    Parameters:
    - n (int): The number of arrays and the size of each array.
    - low (int): The lowest integer to be included.
    - high (int): One above the highest integer to be included.

    Returns:
    - List[np.ndarray]: A list containing n numpy arrays with integer values.
    """
    arrays = []
    for _ in range(n):
        # Generate truly random bytes and convert them to integers
        random_bytes = os.urandom(n * n * 4)  # 4 bytes for each integer
        random_integers = np.frombuffer(random_bytes, dtype=np.uint32) % (high - low) + low
        array = random_integers.reshape((n, n))
        arrays.append(array)
    
    return arrays

def hashify(data, _namespace: str = '', template: dict = {}, hash_length: int = 20, _created_at: str = '') -> str:
    """
    Generate a short unique hash of a given string or JSON object using the SHA-256 hashing algorithm.

    Parameters:
        data (str or JSON-like object): The string or JSON object to be hashed.
        _namespace (str, optional): The namespace of the object to be hashed. Defaults to ''.
        parse_config (dict, optional): The parse configuration specifying fields and transformations. Defaults to {}.
        hash_length (int, optional): The length of the hash to be returned. Defaults to 20.
        _created_at (str, optional): The name of the property to be removed prior to hashing. Defaults to ''.
        
    Returns:
        [Optional] str: The short hash of the input string or JSON object.
        [Optional] dict: The input JSON object with the short hash added as a property.

    Raises:
        ValueError: If an error occurs during hashing.
    """

    # Preparing the field aliases - This is a dictionary of field names and their aliases so that the field names can be changed prior to being loaded into the graph

    if 'alias_fields' in template.keys():
        field_aliases = template['alias_fields']
    else:
        field_aliases = {}

    try:
        if not isinstance(data, list):
            data = [data]
        hash_list = []
        for obj in data:
            if isinstance(obj, str):
                temp_obj = {'_label': obj}
                original_obj = copy.deepcopy(obj)
            else:
                original_obj = copy.deepcopy(obj)
                temp_obj = copy.deepcopy(obj)
            # This is the line that might need adjusting.
            if isinstance(temp_obj, str):
                temp_obj = {'value': temp_obj}
            # Convert JSON-like objects to string
            if isinstance(obj, dict):
                # removing the created_at property prior to hashing the object
                if _created_at in obj.keys():
                    obj.pop(_created_at)
                input_str = json.dumps(obj, sort_keys=True)
            else:
                input_str = obj

            # Calculate the hash
            hash_object = hashlib.sha256(str(input_str).encode())  # Calculate the SHA-256 hash of the string
            short_hash = hash_object.hexdigest()[:hash_length]  # Shorten the hash to the specified length
            if not isinstance(temp_obj, dict):
                temp_obj = {'_label': temp_obj}
            temp_obj['_hash'] = short_hash

            # _namespace
            if len(_namespace) > 0:
                # Use the alias if the field_aliases property is present in the parse_config
                if len(field_aliases) > 0:
                    if _namespace in field_aliases.keys():
                        original_namespace = copy.deepcopy(_namespace)
                        _namespace = field_aliases[_namespace]

                # original code after alias has been implemented
                namespace_short_hash = _namespace.lower().replace(' ','_') + '___' + short_hash
                temp_obj['_guid'] = namespace_short_hash
                temp_obj['_namespace'] = _namespace

            hash_list.append(temp_obj)

        if len(hash_list) == 1:
            if isinstance(original_obj, str):
                if len(_namespace) > 0:
                    return hash_list[0]
                else:
                    return hash_list[0]['_hash']
            else:
                return hash_list[0]
        else:                
            return hash_list

    except Exception as e:
        # Log the error using the logging module
        logging.error(f"An error occurred in hashify: {e}", exc_info=True)
        raise ValueError(f"Input error: {e}")





########################################
####    Find Dates in the object    ####
########################################
    
def is_datetime_string(s: str) -> bool:
    """
    Checks if the given string matches common datetime formats.

    Args:
    s (str): The string to check.

    Returns:
    bool: True if the string matches a datetime pattern, False otherwise.
    """
    datetime_patterns = [
        r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
        r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
        r'\d{2}-\d{2}-\d{4}',  # DD-MM-YYYY
        # Additional patterns can be added here
    ]
    return any(re.fullmatch(pattern, s) for pattern in datetime_patterns)

def is_unix_timestamp(val: Any) -> bool:
    """
    Checks if the given value is a Unix timestamp.

    Args:
    val (Any): The value to check.

    Returns:
    bool: True if the value is a Unix timestamp, False otherwise.
    """
    try:
        return isinstance(val, int) and 0 <= val <= 4102444800
    except ValueError:
        return False

def find_datetime_values(obj: Union[Dict, List], path: str = "") -> None:
    """
    Recursively searches for datetime values in a nested dictionary or list.

    Args:
    obj (Union[Dict, List]): The object to search through.
    path (str): The current path in the object.

    Returns:
    None: This function prints the paths and values of datetime-related data it finds.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            current_path = f"{path}.{k}" if path else k
            if isinstance(v, (dict, list)):
                find_datetime_values(v, current_path)
            else:
                if isinstance(v, datetime):
                    print(f"Found datetime object at {current_path}: {v}")
                elif isinstance(v, str):
                    if is_datetime_string(v):
                        print(f"Found datetime string at {current_path}: {v}")
                elif is_unix_timestamp(v):
                    print(f"Found Unix timestamp at {current_path}: {v}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            current_path = f"{path}[{i}]"
            find_datetime_values(item, current_path)

# Example usage
example_obj = {
    "date_str": "2023-04-01",
    "timestamp": 1672502400,
    "nested": {
        "another_date": datetime.now(),
        "list_example": ["2023/12/31", 42, {"deep_nested_date": "01-01-2023"}]
    }
}

find_datetime_values(example_obj)


############################
####    Bloom Filter    ####
############################

def bloom_filter_check_and_load(objects: List[Dict[str, Any]] = [], 
                                key: str = '_guid', 
                                bloom: BloomFilter = bloom_filter) -> Any:    
    """
    Filters out items that are possibly in the Bloom filter.
    Loads the new item keys into the Bloom filter and returns True when that is completed.
    """
    new_objects = []
    
    try:
        for obj in objects:
            assert key in obj, "Each object should have a 'key' property."
            if not bloom.check(obj[key]):
                new_objects.append(obj)
                bloom.add(obj[key])

        return new_objects
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

"""

NOTE: Example Bloom Filter Usage:
google_form_output = bloom_filter_check_and_load(google_form_output, key='Job Posting URL', bloom=bloom_filter)

"""

def bloomfilter_remove(objects: List[Dict[str, Any]] = [], 
                       bloom: BloomFilter = bloom_filter) -> Any:    
    """
    Filters out items that are possibly in the Bloom filter.
    Loads the new item keys into the Bloom filter and returns True when that is completed.
    """
    new_objects = []
    
    try:
        for obj in objects:
            obj['uuid'] = hashify(str(obj))
            if not bloom.check(obj['uuid']):
                new_objects.append(obj)

        return new_objects
    
    except Exception as e:
        print(f"An error occurred when removing items with bloomfilter_remove: {e}")
        return []

def bloom_filter_clear(redis_client: Any = redis_client):
    """
    Clears the entire Bloom filter.
    """

    try:
        redis_client.delete("bloom_filter")
        print(f"Bloom filter cleared.")
        return True
    
    except Exception as e:
        print(f"An error occurred when clearing the bloom filter: {e}")
        return False
    
########################################
####    Transformation Functions    ####
########################################

def standardize_date(date_input) -> int:
    """
    Standardizes a date, datetime, or timestamp to a Unix timestamp.

    Parameters:
    date_input: Some date, datetime, or timestamp.

    Returns:
    int: A unix timestamp.

    Raises:
    ValueError: If the input is not a date.
    """

    try:
        date_value = to_unix(date_input)
        return date_value
    
    except Exception as e:
        logging.error(f"An error occurred in standardize_date: {e}", exc_info=True)

def standardize_geography(input_data: str) -> Dict[str, Any]:
    """
    Standizes a location name to a standardized location object.

    Parameters:
    input_data (str): The location name to be standardized.

    Returns:
    Dict[str, Any]: A standardized location object.
    """
    try:
        return(input_data)
    except Exception as e:
        logging.error(f"An error occurred in standardize_geography: {e}", exc_info=True)
        return None

def standardize_phone(phone_number: str) -> str:
    """
    Standardizes an international phone number according to ISO guidelines (E.164 format)
    and provides geographic information based on the country and area codes.

    Parameters:
    phone_number (str): The phone number to be standardized.

    Returns:
    str: A standardized phone number along with geographic information.

    Raises:
    ValueError: If the input is not a valid phone number.
    """
    # Validate input
    assert isinstance(phone_number, str), "Input must be a string."

    try:
        # Parse and standardize the phone number
        parsed_number = phonenumbers.parse(phone_number)
        standardized_number = phonenumbers.format_number(parsed_number, phonenumbers.PhoneNumberFormat.E164)
        
        # Extract geographic information
        country_info = geocoder.description_for_number(parsed_number, 'en')

        return f"Standardized Number: {standardized_number}, Country: {country_info}"

    except phonenumbers.NumberParseException as e:
        raise ValueError(f"Invalid phone number: {e}")

def standardize_name(name: str) -> str:
    """
    Standardizes a person's name by converting to lowercase, replacing special 
    punctuation with space, and removing accent marks from letters. Additionally, 
    it trims leading and trailing whitespaces.

    Parameters:
    name (str): The name to be standardized.

    Returns:
    str: The standardized name.

    Raises:
    AssertionError: If the input is not a string.
    """

    # Ensure the input is a string
    assert isinstance(name, str), "Input must be a string."

    try:
        # Convert to lowercase
        name = name.lower()

        # Normalize to decompose accent characters
        name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')

        # Replace special punctuation with space
        name = re.sub(r"[^\w\s]", ' ', name)

        # Collapse all spaces and tabs to a single space
        name = re.sub(r"\s+", ' ', name)

        # Trim leading and trailing whitespaces
        name = name.strip()

        return name

    except Exception as e:
        raise ValueError(f"Error processing the name: {e}")

def standardize_passport_id(passport_id: str, length: int = 12, separator: str = '-') -> str:
    """
    Standardizes the given passport ID to a specified format.
    
    Args:
    passport_id (str): The passport ID to be standardized.
    length (int): The total length of the standardized ID (including separators).
    separator (str): The separator character to be used in the standardized ID.

    Returns:
    str: The standardized passport ID.

    Raises:
    ValueError: If the passport ID contains non-alphanumeric characters or is too long to be formatted.
    """
    # Remove any non-alphanumeric characters
    passport_id = re.sub(r"[^a-zA-Z0-9]", '', passport_id)

    # Ensure passport ID is alphanumeric
    if not passport_id.isalnum():
        raise ValueError("Passport ID must be alphanumeric.")

    # Remove existing non-alphanumeric characters
    clean_id = ''.join(filter(str.isalnum, passport_id))

    # Check if the clean ID can be formatted to the specified length
    if len(clean_id) > length:
        raise ValueError("Passport ID is too long to be standardized to the specified length.")

    # Padding the ID with zeros if necessary
    standardized_id = clean_id.ljust(length, '0')

    # Inserting separators
    try:
        return separator.join(standardized_id[i:i + 4] for i in range(0, length, 4))
    except Exception as e:
        raise RuntimeError(f"Error in formatting: {e}")

# Example usage
# print(standardize_passport_id("A1234567"))

def standardized_functions():
    functions = filter_func(lambda x: 'standardize_' in x, globals().keys())
    return functions

def standardize_objects(objects: List[Dict], parse_config: Dict, _created_at: str = '') -> List[Dict]:
    """
    Standardize objects based on the parse_config.

    Args:
        objects (List[Dict]): List of objects to standardize.
        parse_config (Dict): The parse configuration specifying fields and transformations.
        _created_at (str): A property name representing _created_at unix timestamp.

    Returns:
        List[Dict]: List of standardized objects.

    Raises:
        KeyError: If 'standardize_fields' is not in parse_config.
        ValueError: If 'transform' function is not found in globals.
        RuntimeError: If an error occurs during processing.
    """
    # Assertions
    assert isinstance(objects, list), "objects must be a list"
    assert isinstance(parse_config, dict), "parse_config must be a dictionary"
    assert 'fields' in parse_config, "parse_config must contain 'standardize_fields'"

    # standardize_fields
    standardize_fields = filter_func(lambda x: 'standardize' in x.keys(), parse_config['fields'])
    # standardize_fields = filter_func(lambda x: str(x['standardize']).lower() == 'true', standardize_fields)
    standardize_fields = {x['field']: x['standardize'] for x in standardize_fields}

    try:
        standardized_objects = []
        for obj in objects:
            # standardize all of the values for which there is a key in the standardize_fields dictionary
            for field in obj.keys():
                if field in standardize_fields.keys():
                    obj[field] = globals()[standardize_fields[field]](obj[field])
            standardized_objects.append(obj)

        return standardized_objects

    except Exception as e:
        raise RuntimeError(f"Error during processing: {e}")

def prepare_entities_for_load(objects, _namespace: str, template: Dict, _created_at: str = '', include_created_at: bool = True) -> List[Dict]:
    """
    Prepare objects for hashing and loading into the database.

    Args:
        objects (List[Dict]): List of objects to prepare.
        template (Dict): The template for the object.
        _created_at (int): A property name representing _created_at unix timestamp.

    Returns:
        List[Dict]: List of prepared objects.
    """
    prepared_objects = []
    original_namespace = copy.deepcopy(_namespace)
 
    try:
        for obj in objects:
            hashed_object = hashify(obj[_namespace], _namespace=_namespace, template=template)
            original_namespace = copy.deepcopy(_namespace)
            # Use the alias if the field_aliases property is present in the template
            if 'alias_fields' in template.keys():
                if original_namespace in template['alias_fields'].keys():
                    updated_namespace = template['alias_fields'][original_namespace]
            prepared_obj = {
                # '_namespace': updated_namespace,
                '_label': obj[original_namespace],
                '_guid': hashed_object['_guid'],
                # '_hash': hashed_object['_hash'],
                '_source': obj['_guid'],
                '_edge': 'has_' + original_namespace
            }
            # Adding the _created_at property
            if len(_created_at) > 0 and str(include_created_at).lower() == 'true':
                if _created_at in obj.keys():
                    prepared_obj['_created_at'] = obj[_created_at]
                else: 
                    prepared_obj['_created_at'] = to_unix(datetime.now())
            else: 
                prepared_obj['_created_at'] = to_unix(datetime.now())

            if not str(include_created_at).lower() == 'true':
                prepared_obj = {k:v for k,v in prepared_obj.items() if k != '_created_at'}

            prepared_objects.append(prepared_obj)    

        return prepared_objects  

    except Exception as e:
        raise RuntimeError(f"Error during processing: {e}")


######################
####    Search    ####
######################

"""

query = "flight_manifest___e4268fb2489004f749ff"

"""

def search(query_str: str, neo4j_client: neo4j._sync.driver.BoltDriver = neo4j_client, mongodb_client: pymongo.database.Database = mongodb_client, expand: bool = False, foaf: bool = False) -> List[Dict]:
    """
    Search GIS Data Lake for a given query.

    Args:
        query_str (str): The search query.

    Returns:
        List[Dict]: A list of search results.
    """
    assert isinstance(query_str, str), "query must be a string"

    try:
        # Search for a GUID
        if bool(re.match(r".*___.*[A-Za-z0-9]{20}.*", query_str)) and 'match' not in query_str.lower() and 'return' not in query_str.lower():
            results = list(mongo_collection.find({'_guid':query_str}))
            results = json.loads(json_util.dumps(results))
            logging.info(f"MongoDB _guid search results: {results}")

        elif 'find' in query_str.lower():
            results = list(query_str)
            results = json.loads(json_util.dumps(results))
            logging.info(f"Full MongoDB _guid search results: {results}")

        # Search Neo4j for a label
        elif 'match' in query_str.lower() and 'return' in query_str.lower():
            with neo4j_client.session() as session:
                results = list(session.run(query_str).data())
                results = json.loads(json.dumps(results))
                logging.info(f"Neo4j search results: {results}")

        else:
            logging.info(f"Search query not recognized: {query_str}")

        return {'results': results}
        
    except Exception as e:
        logging.error(f"An error occurred in search: {e}", exc_info=True)
        return {}


####################################
####    Similarity Functions    ####
####################################
    
def transliterate_and_standardize_name(name: str) -> str:
    """Transliterate (if needed) and standardize the name."""
    # Transliterate to Latin script
    name = unidecode(name)
    # Standardize
    name = name.lower()
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
    name = re.sub(r"[^\w\s]", ' ', name)
    name = re.sub(r"\s+", ' ', name)
    return name.strip()

def name_similarity(names: List[str], similarity_threshold: float = 0.8) -> bool:
    """
    Compares an array of names, possibly in different languages, to test if they could represent the same entity.

    Parameters:
    names (List[str]): An array of names to be compared.
    similarity_threshold (float): The threshold for considering names as similar.

    Returns:
    bool: True if the names are likely to represent the same entity, False otherwise.
    """

    # Transliterate and standardize all names
    standardized_names = [transliterate_and_standardize_name(name) for name in names]
    print(f"standardized_names: {standardized_names}")

    # Compare each name with every other name
    for i in range(len(standardized_names)):
        for j in range(i + 1, len(standardized_names)):
            similarity = difflib.SequenceMatcher(None, standardized_names[i], standardized_names[j]).ratio()
            print(f"similarity: {similarity} | Threshold: {similarity_threshold}")
            if similarity < similarity_threshold:
                return False

    return True

"""

names_to_compare = ['Иван Петров', 'Ivanar Petrov']
print(name_similarity(names_to_compare))

"""




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
# create_table_from_json(conn, json.loads(json_data))

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

# def kafka_create_topic(topic_name: str, num_partitions: int = 1, replication_factor: int = 1, kafka_client: AdminClient = kafka_client) -> None:
#     """
#     Create a new topic in Kafka.

#     Parameters:
#     kafka_client (AdminClient): The Kafka AdminClient instance.
#     topic_name (str): Name of the topic to be created.
#     num_partitions (int): Number of partitions for the topic (default: 1).
#     replication_factor (int): Replication factor for the topic (default: 1).

#     Raises:
#     AssertionError: If any of the input parameters are not in expected format.
#     KafkaException: If there is an error in creating the topic.
#     """
#     assert isinstance(topic_name, str), "Topic name must be a string"
#     assert isinstance(num_partitions, int), "Number of partitions must be an integer"
#     assert isinstance(replication_factor, int), "Replication factor must be an integer"

#     new_topic = NewTopic(topic_name, num_partitions=num_partitions, replication_factor=replication_factor)
#     try:
#         kafka_client.create_topics([new_topic])
#     except KafkaException as e:
#         raise KafkaException(f"Error in creating topic {topic_name}: {e}")

# def kafka_delete_topic(topic_name: str, kafka_client: AdminClient = kafka_client) -> None:
#     """
#     Delete a topic in Kafka.

#     Parameters:
#     kafka_client (AdminClient): The Kafka AdminClient instance.
#     topic_name (str): Name of the topic to be deleted.

#     Raises:
#     AssertionError: If topic_name is not a string.
#     KafkaException: If there is an error in deleting the topic.
#     """
#     assert isinstance(topic_name, str), "Topic name must be a string"

#     try:
#         kafka_client.delete_topics([topic_name])
#     except KafkaException as e:
#         raise KafkaException(f"Error in deleting topic {topic_name}: {e}")

# def kafka_list_topics(kafka_client: AdminClient = kafka_client):
#     """
#     List all topics in the Kafka cluster.

#     Parameters:
#     kafka_client (AdminClient): The Kafka AdminClient instance.

#     Returns:
#     dict: A dictionary of topics and their metadata.

#     Raises:
#     KafkaException: If there is an error in listing the topics.
#     """
#     try:
#         return kafka_client.list_topics().topics
#     except KafkaException as e:
#         raise KafkaException(f"Error in listing topics: {e}")

# def kafka_consume_message(bootstrap_servers: str, group_id: str, topic_name: str):
#     """
#     Consume a message from a Kafka topic.

#     Parameters:
#     bootstrap_servers (str): Comma-separated list of broker addresses.
#     group_id (str): The consumer group to which the consumer belongs.
#     topic_name (str): The topic to consume messages from.

#     Returns:
#     str or None: The consumed message or None if no message is received.

#     Raises:
#     Exception: If there is an error in consuming the message.
#     """
#     consumer = Consumer({"bootstrap.servers": bootstrap_servers, "group.id": group_id})
#     consumer.subscribe([topic_name])
#     try:
#         message = consumer.poll(1.0)
#         consumer.close()
#         if message is None:
#             print("No message received")
#         else:
#             return message
#     except Exception as e:
#         raise Exception(f"Error in consuming message: {e}")

################################
####    Milvus Functions    ####
################################

# Milvus Delete Collection
def milvus_delete_collection(collection_name: str) -> None:
    """
    Delete a collection from the Milvus database.

    Parameters:
    collection_name (str): Name of the collection to be deleted.

    Raises:
    AssertionError: If collection_name is not a string.
    Exception: For issues encountered while deleting the collection.
    """
    assert isinstance(collection_name, str), "Collection name must be a string"

    try:
        milvus_utility.drop_collection(collection_name)
        print(f"Successfully deleted collection: {collection_name}")
    except Exception as e:
        raise Exception(f"Error in deleting collection {collection_name}: {e}")

# List Collections
def milvus_list_collections() -> List[str]:
    """
    List all collections in the Milvus database.

    Raises:
    Exception: For issues encountered while listing the collections.
    """
    try:
        collections = milvus_utility.list_collections()
        return collections
    except Exception as e:
        raise Exception(f"Error in listing collections: {e}")

# Milvus Delete Index
def milvus_delete_index(field_name: str, collection_name: str = 'gis_main') -> None:
    """
    Delete an index from a specified field in a Milvus collection.

    Parameters:
    collection_name (str): Name of the collection from which the index is to be deleted.
    field_name (str): Name of the field in the collection from which the index is to be deleted.

    Raises:
    AssertionError: If inputs are not in expected format.
    Exception: For issues encountered while deleting the index.
    """
    assert isinstance(collection_name, str), "Collection name must be a string"
    assert isinstance(field_name, str), "Field name must be a string"

    try:
        milvus_collection = MilvusCollection(name=collection_name)
        milvus_collection.drop_index(name=field_name)
    except Exception as e:
        raise Exception(f"Error in deleting index on {field_name} in collection {collection_name}: {e}")

# Load vectors into Milvus
def milvus_load_vectors(vectors: List[List[float]], _guids: List[str], namespace: List[str], created_at: List[float], collection: str = 'gis_main') -> None:
    """
    Load vectors into a Milvus collection.

    Parameters:
    collection (str): Name of the collection to load vectors into.
    vectors (List[List[float]]): A list of vectors to be loaded.
    _guids (List[str]): A list of globally unique IDs for the vectors.

    Raises:
    AssertionError: If inputs are not in expected format.
    Exception: For issues encountered while loading the vectors.
    """
    assert isinstance(collection, str), "Collection name must be a string"
    assert isinstance(vectors, list), "Vectors must be a list"
    assert isinstance(_guids, list), "IDs must be a list"
    assert isinstance(namespace, list), "Namespace must be a list"
    assert isinstance(created_at, list), "Created_at must be a list"
    assert len(vectors) == len(_guids), "Vectors and IDs must have the same length"
    assert len(vectors) == len(namespace), "Vectors and Namespace must have the same length"
    assert len(vectors) == len(created_at), "Vectors and Created_at must have the same length"

    try:
        data = [
            _guids,
            namespace,
            vectors,
            created_at
        ]    
        milvus_collection = MilvusCollection(collection)
        milvus_collection.insert(data)
        milvus_collection.flush()
    except Exception as e:
        raise Exception(f"Error in loading vectors into collection {collection}: {e}")

"""

# Fake data to load into Milvus
vectors = [[random.random() for _ in range(300)] for _ in range(100)]
_guids = [str(uuid.uuid4()) for _ in range(100)]
namespace = ["person" for _ in range(100)]
created_at = [to_unix(datetime.now()) for _ in range(100)]

# Load vectors into Milvus
milvus_load_vectors(vectors, _guids, namespace, created_at, collection="gis_main")

"""

# Delete vectors from Milvus

def milvus_delete_vectors(guilds: List[str], collection: str = 'gis_main') -> None:
    """
    Delete vectors from a Milvus collection.

    Parameters:
    collection (str): Name of the collection to delete vectors from.
    guilds (List[str]): A list of globally unique IDs for the vectors.
    
    Raises:
    AssertionError: If inputs are not in expected format.
    Exception: For issues encountered while deleting the vectors.
    """
    assert isinstance(collection, str), "Collection name must be a string"
    assert isinstance(guilds, list), "IDs must be a list"

    try:
        # Set up Milvus collection
        milvus_collection = MilvusCollection(collection)
        
        # Delete vectors
        expr = f"_guid in {guilds}"
        milvus_collection.delete(expr)
        milvus_collection.flush()
        print(f"Successfully deleted vectors from collection {collection}")
    except Exception as e:
        raise Exception(f"Error in deleting vectors from collection {collection}: {e}")

"""

milvus_delete_vectors(_guids, collection="gis_main")

"""

"""
namespace = ""
collection = 'gis_main'
offset = 0
opt_k = 10
"""

# Search vectors in Milvus
def milvus_search(vectors: List[List[float]], namespace: str = '', top_k: int = 10, collection: str = 'gis_main', offset: int = 0) -> List[List[int]]:
    """
    Search for vectors in a Milvus collection.

    Parameters:
    collection (str): Name of the collection to search.
    vectors (List[List[float]]): A list of vectors to search for.
    namespace (str): Namespace of the vectors to search for.
    top_k (int): Number of results to return (default: 10).

    Raises:
    AssertionError: If inputs are not in expected format.
    Exception: For issues encountered while searching for the vectors.
    """
    assert isinstance(collection, str), "Collection name must be a string"
    assert isinstance(vectors, list), "Vectors must be a list"
    assert isinstance(top_k, int), "Top_k must be an integer"

    try:
        # Search Parameters
        search_params = {
            "metric_type": "L2", 
            "offset": offset, 
            "ignore_growing": False, 
            "params": {"nprobe": top_k}
        }

        milvus_collection = MilvusCollection(collection)

        # loading collection into memory
        milvus_collection.load()

        # Returning similar vectors
        if len(namespace) > 0:
            results = milvus_collection.search(data=vectors, 
                                               anns_field='vector', 
                                               expr = f"namespace == '{namespace}'",
                                               limit = top_k, 
                                               param = search_params)
        else:
            results = milvus_collection.search(data=vectors, 
                                               anns_field='vector', 
                                               limit = top_k, 
                                               param = search_params)
            
        return results

    except Exception as e:
        raise Exception(f"Error in searching for vectors in collection {collection}: {e}")

###############################
####    Redis Functions    ####
###############################

def redis_load_objects(objects: List[Dict[str, Union[any, any]]]) -> bool:
    """
    Load a list of dictionary objects into Redis.

    Parameters:
    objects (List[Dict[str, Union[str, int]]]): A list of objects to be loaded, each containing a '_guid' key.

    Returns:
    bool: True if objects loaded successfully, False otherwise.

    Raises:
    AssertionError: If the input is not a list, is empty, or objects lack '_guid'.
    """

    assert isinstance(objects, List) and objects, "Input must be a non-empty list"

    try:
        with redis_client.pipeline() as pipe:
            for obj in objects:
                assert "_guid" in obj, "Each object must contain a '_guid' key"
                _guid = obj["_guid"]
                serialized_obj = json.dumps(obj)
                pipe.set(_guid, serialized_obj)
            pipe.execute()
        return True

    except redis.RedisError as e:
        print(f"Redis error loading objects: {e}")
        return False

def redis_retrieve_objects(_guids: List[str]) -> List[Dict[str, Union[str, int]]]:
    """
    Retrieve a list of objects from Redis based on their _guid values.

    Parameters:
    _guids (List[str]): A list of _guid values to fetch the corresponding objects.

    Returns:
    List[Dict[str, Union[str, int]]]: A list of retrieved objects.

    Raises:
    AssertionError: If the input is not a list or is empty.
    """

    assert isinstance(_guids, List) and _guids, "Input must be a non-empty list"

    try:
        retrieved_objects = []
        for _guid in _guids:
            obj_str = redis_client.get(_guid)
            if obj_str:
                retrieved_objects.append(json.loads(obj_str))
        return retrieved_objects

    except redis.RedisError as e:
        print(f"Redis error retrieving objects: {e}")
        return []   

def redis_delete_objects(_guids: List[str]) -> bool:
    """
    Remove a list of objects from Redis based on their _guid values.

    Parameters:
    _guids (List[str]): A list of _guid values of objects to be removed.

    Returns:
    bool: True if objects removed successfully, False otherwise.

    Raises:
    AssertionError: If the input is not a list or is empty.
    """

    assert isinstance(_guids, List) and _guids, "Input must be a non-empty list"

    try:
        redis_client.delete(*_guids)
        return True

    except redis.RedisError as e:
        print(f"Redis error removing objects: {e}")
        return False     

def redis_delete_all() -> bool:
    """
    Remove all values from Redis.

    Returns:
    bool: True if all values removed successfully, False otherwise.
    """

    try:
        redis_client.flushall()
        return True

    except redis.RedisError as e:
        print(f"Redis error flushing all values: {e}")
        return False

# Test Redis Functions with Fake Data
"""

objects = [
    {"_guid": "1", "name": "John", "age": 30},
    {"_guid": "2", "name": "Jane", "age": 25},
    {"_guid": "3", "name": "Bob", "age": 40}
]
redis_load_objects(objects)
redis_retrieve_objects(["1", "2", "3"])
redis_delete_objects(["1", "2", "3"])
redis_delete_all()

"""

###############################
####    Neo4j Functions    ####
###############################

# def neo4j_query(node: Dict[str, Union[str, int]]) -> None:

# query_str = query

# neo4j_query = f"MATCH (obj:Object{{_guid:'{query_str}'}})-[r]->(obj2:Object) RETURN obj, r, obj2"

# query_str2 = query_str

# query_str = f"MATCH (obj:Object{{_guid:'{query_str}'}})-[]->(obj2:Object) RETURN obj, r, obj2"

# with neo4j_client.session() as session:
#     results = list(session.run(query_str).data())
#     return results

################################
####    Process Template    ####
################################

#%%

def process_template(parse_config: Dict) -> Dict:
    """
    Process the different components of the parse_config file.

    Args:
        parse_config (Dict): parse_config dictionary

    Returns:
        Dict: updated parse_config dictionary

    """
    assert isinstance(parse_config, dict), "parse_config must be a dictionary"

    try:
        name = parse_config['template']['name']
    except Exception as e:
        name = ''

    try:
        primary_key = parse_config['template']['primary_key']
    except Exception as e:
        primary_key = ''

    try:
        namespace = parse_config['template']['namespace']
    except Exception as e:
        namespace = ''

    try:
        drop_fields = filter_func(lambda x: 'drop' in x.keys(), parse_config['fields'])
        drop_fields = filter_func(lambda x: str(x['drop']).lower() == 'true', drop_fields)
        drop_fields = map_func(lambda x: x['field'], drop_fields)
    except Exception as e:
        drop_fields = []

    try:
        alias_fields = filter_func(lambda x: 'alias' in x.keys(), parse_config['fields'])
        alias_fields = filter_func(lambda x: len(str(x['alias'])) > 0, alias_fields)
        alias_fields = {x['field']: x['alias'] for x in alias_fields}
    except Exception as e:
        alias_fields = {}

    try:
        standardized_fields = filter_func(lambda x: 'standardize' in x.keys(), parse_config['fields'])
        standardized_fields = {x['field']: x['standardize'] for x in standardized_fields}
    except Exception as e:
        standardized_fields = {}

    try:    
        embedding_fields = filter_func(lambda x: 'is_embedding' in x.keys(), parse_config['fields'])    
        embedding_fields = filter_func(lambda x: str(x['is_embedding']).lower() == 'true', embedding_fields)
        embedding_fields = map_func(lambda x: x['field'], embedding_fields)
    except Exception as e:
        embedding_fields = []

    try:    
        datetime_fields = filter_func(lambda x: 'is_datetime' in x.keys(), parse_config['fields'])
        datetime_fields = filter_func(lambda x: str(x['is_datetime']).lower() == 'true', datetime_fields)
        datetime_fields = map_func(lambda x: x['field'], datetime_fields)
    except Exception as e:
        datetime_fields = []

    try:    
        entity_fields = filter_func(lambda x: 'is_entity' in x.keys(), parse_config['fields'])
        entity_fields = filter_func(lambda x: str(x['is_entity']).lower() == 'true', entity_fields)
        entity_fields = map_func(lambda x: x['field'], entity_fields)
    except Exception as e:
        entity_fields = []

    try:
        calculated_entity_fields = parse_config['calculated_entities']
        calculated_entity_fields = map_func(lambda x: {x['alias']: x['fields']}, calculated_entity_fields)
    except Exception as e:
        calculated_entity_fields = []

    try:
        edges = parse_config['edges']
    except Exception as e:
        edges = []

    # Output dict
    template_dict = {
        'name': name,
        'primary_key': primary_key,
        'namespace': namespace,
        'drop_fields': drop_fields,
        'alias_fields': alias_fields,
        'standardized_fields': standardized_fields,
        'embedding_fields': embedding_fields,
        'datetime_fields': datetime_fields,
        'entity_fields': entity_fields,
        'calculated_entity_fields': calculated_entity_fields,
        'edges': edges
    }

    return template_dict


############################
####    Process Data    ####
############################

def process_data(data: Union[List[Dict], Dict], parse_config, template: Dict) -> Union[List[Dict], Dict]:
    """
    Process the different components of the parse_config file.

    Args:
        data (Union[List[Dict], Dict]): data to process
        template (Dict): template_output dictionary which contains the parsed configuration

    Returns:
        Union[List[Dict], Dict]: processed data

    """
    assert isinstance(data, (list, dict)), "data must be a list of dictionaries or a dictionary"
    assert isinstance(template, dict), "parse_config must be a dictionary"

    # Name
    if 'name' in template.keys():
        template_name = template['name']
    else:
        template_name = ''

    # Primary Key
    if 'primary_key' in template.keys():
        primary_key = template['primary_key']
        if len(primary_key) > 0:
            if isinstance(data, dict):
                if primary_key in data.keys():
                    data = data[primary_key]
    else:
        primary_key = ''

    # Namespace
    if 'namespace' in template.keys():
        namespace = template['namespace']
    else:
        namespace = ''

    # Ensure data is a list
    if not isinstance(data, list):
        data = [data]

    # Drop fields
    drop_fields = template['drop_fields']
    if len(drop_fields) > 0:
        clean_objects = []
        for obj in data:
            clean_objects.append({k:v for k,v in obj.items() if k not in drop_fields})
        data = clean_objects

    # Verify the data schema fingerprint is consisent with template fingerprint
    # if len(data) > 0:
    #     data_schema_fingerprint = generate_fingerprint(data)
    #     # Check if fingerprint already exists in redis
    #     template_fingerprint = redis_client.get(template_name)
    #     if template_fingerprint is None:
    #         redis_client.set(template_name, data_schema_fingerprint)
    #         logging.info(f"Added fingerprint to redis: {template_name}")
    #     else:
    #         # Check if the fingerprint matches what was stored in redis
    #         if str(data_schema_fingerprint) != str(template_fingerprint.decode()):
    #             raise Exception(f"Data schema fingerprint does not match what is stored in redis for template: {template_name}")

    # Hashify the data
    data = hashify(data, _namespace=namespace, template=template)

    return data

#########################
####    Load Data    ####
#########################

def load(data: Union[List[Dict], Dict], parse_config: Dict, template: Dict) -> None:
    """
    Load data into GIS Data Lake databases

    Args:
        data (Union[List[Dict], Dict]): data to load
        template (Dict): template_output dictionary which contains the parsed configuration

    Returns:
        Status of job load (success or failure for each database)

    """
    assert isinstance(data, (list, dict)), "data must be a list of dictionaries or a dictionary"
    assert isinstance(template, dict), "parse_config must be a dictionary"

    # Ensure data is a list
    if not isinstance(data, list): 
        data = [data]

    ###########
    # MongoDB #
    ###########
    
    try:
        mongo_collection.insert_many(data)
    except Exception as e:
        pass

    #########
    # Neo4j #
    #########

    try:
        neo4j_objects = map_func(lambda x: {k:v for k,v in x.items() if k in ['_guid']}, data)
        updated_neo4j_objects = []
        for obj in neo4j_objects:
            obj['_label'] = 'source'
            updated_neo4j_objects.append(obj)

        neo4j_objects = copy.deepcopy(updated_neo4j_objects)
        del updated_neo4j_objects

        if len(neo4j_objects) > 0:
            load_statement = ', '.join([f'`{k}`: line.`{k}`' for k in neo4j_objects[0].keys() if k not in ['_edge']])
            load_statement = f'UNWIND $objects AS line MERGE (obj:Object {{ {load_statement} }})'

        with neo4j_client.session() as session:
            session.run(load_statement, objects=neo4j_objects)
            logging.info(f'Loaded Objects into Neo4j: {neo4j_objects}')
            print(f"Loaded Objects into Neo4j: {len(neo4j_objects)}")

        # Neo4j - Entities
        entities = template['entity_fields']
        if len(entities) > 0:
            for entity in entities:
                # Loading an array of entities into Neo4j
                neo4j_objects = map_func(lambda x: {k:v for k,v in x.items() if k in ['_guid', entity]}, data)
                neo4j_objects = standardize_objects(neo4j_objects, parse_config, template)
                neo4j_objects = prepare_entities_for_load(neo4j_objects, entity, template, include_created_at=False)
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
                    load_statement = load_statement.replace('`_edge`:', '`_label`:')
                    load_statement = 'UNWIND $objects AS line MATCH (obj1:Object{_guid:line.`_source`}) MATCH (obj2:Object{_guid:line.`_guid`}) MERGE (obj1)-[owns:OWNS{' + str(load_statement.replace("`_guid`: line.`_guid`, `_source`: line.`_source`,","")) + '}]-(obj2) RETURN *'
                    #  => THIS IS THE WORKING SOLUTION => load_statement = 'UNWIND $objects AS line MATCH (obj1:Object{_guid:line.`_guid`}) MATCH (obj2:Object{_guid:line.`_source`}) MERGE (obj1)-[owns:' + str(neo4j_edges[0]['_edge']) + '{' + str(load_statement) + '}]-(obj2) RETURN *'

                    with neo4j_client.session() as session:
                        session.run(load_statement, objects=neo4j_edges)
                        logging.info(f'Loaded Objects into Neo4j: {neo4j_edges}')

        # Building additional edge relationships as outlined in the template
        alias_fields = template['alias_fields']
        standardized_fields = template['standardized_fields']
        edge_objects = template['edges']

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
                        has_variable = 'has_' + str(child).lower().replace(' ', '_')
                        edge_triple = {'_child': child, '_parent': parent, '_edge':has_variable}
                        for property in edge_properties:
                            edge_triple[property] = property #edge_object['properties'][property]
                        edges_to_load.append(edge_triple)

        # Load the edges_to_load into the Neo4j database
        final_triples_to_load = []
        for triple in edges_to_load:
            for object in data:
                temp_object = {}
                if triple['_child'] in standardized_fields.keys():
                    child = globals()[standardized_fields[triple['_child']]](object[triple['_child']])
                else:
                    child = object[triple['_child']]
                child_guid = hashify(child, _namespace=triple['_child'], template=template)['_guid']
                if triple['_parent'] in standardized_fields.keys():
                    parent = globals()[standardized_fields[triple['_parent']]](object[triple['_parent']])
                else:
                    parent = object[triple['_parent']]
                parent_guid = hashify(parent, _namespace=triple['_parent'], template=template)['_guid']
                temp_object = {'_child': child_guid, '_parent': parent_guid, '_edge': triple['_edge']}
                final_triples_to_load.append(temp_object)

        # Load Neo4j Relationships by edge name
        for _edge in set(map_func(lambda x: x['_edge'], final_triples_to_load)):
            neo4j_relationships = filter_func(lambda x: x['_edge'] == _edge, final_triples_to_load)
            load_statement = ', '.join([f'`{k}`: line.`{k}`' for k in neo4j_relationships[0].keys() if k not in ['_child','_parent']])
            load_statement = load_statement.replace('`_edge`:', '`_label`:')
            load_statement = 'UNWIND $objects AS line MATCH (obj1:Object{_guid:line.`_parent`}) MATCH (obj2:Object{_guid:line.`_child`}) MERGE (obj1)-[owns:OWNS{' + str(load_statement.replace("`_guid`: line.`_guid`, `_source`: line.`_source`,","")) + '}]-(obj2) RETURN *'

            with neo4j_client.session() as session:
                session.run(load_statement, objects=neo4j_relationships)
                logging.info(f'Loaded Objects into Neo4j: {neo4j_relationships}')


        # Calculate entities
        calculated_entities = template['calculated_entity_fields']
        if len(calculated_entities) > 0:
            for calculated_entity in calculated_entities:
                calculated_entity_title = list(calculated_entity.keys())[0]
                calculated_entity_fields = calculated_entity[calculated_entity_title]
    except Exception as e:
        print(f"Errors loading Neo4j Database: {e}")


#############################################
####    RabbitMQ / Pika Functionality    ####
#############################################

def declare_queue(channel: BlockingChannel, queue_name: str):
    """
    Declare a queue on the provided channel.

    Args:
    channel (BlockingChannel): A pika BlockingChannel instance.
    queue_name (str): The name of the queue to declare.

    Raises:
    AssertionError: If the channel or queue_name is not provided.
    """
    assert isinstance(channel, BlockingChannel), "A valid BlockingChannel must be provided."
    assert queue_name, "Queue name must be provided."

    try:
        channel.queue_declare(queue=queue_name)
        print(f"Declared queue: {queue_name}")
    except Exception as e:
        raise Exception(f"Error declaring queue: {e}")

def set_consumer_callback(channel: BlockingChannel, queue_name: str, callback: Callable):
    """
    Set a callback function for consuming messages from a queue.

    Args:
    channel (BlockingChannel): A pika BlockingChannel instance.
    queue_name (str): The name of the queue to consume messages from.
    callback (Callable): The callback function to be called when a message is received.

    Raises:
    AssertionError: If any of the arguments are not provided or are not of the correct type.
    """
    assert isinstance(channel, BlockingChannel), "A valid BlockingChannel must be provided."
    assert queue_name, "Queue name must be provided."
    assert callable(callback), "Callback must be a callable function."

    try:
        channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=True)
        print(f"Set callback for queue: {queue_name}")
    except Exception as e:
        raise Exception(f"Error setting callback for queue: {e}")

def publish_message(channel: BlockingChannel, queue_name: str, message: str):
    """
    Publish a message to a specified queue.

    Args:
    channel (BlockingChannel): A pika BlockingChannel instance.
    queue_name (str): The name of the queue to publish the message to.
    message (str): The message to be published.

    Raises:
    AssertionError: If any of the arguments are not provided.
    """
    assert isinstance(channel, BlockingChannel), "A valid BlockingChannel must be provided."
    assert queue_name, "Queue name must be provided."
    assert message or isinstance(message, str), "Message must be provided and must be a string."

    try:
        channel.basic_publish(exchange='', routing_key=queue_name, body=message)
        print(f"Published message to queue: {queue_name}")
    except Exception as e:
        raise Exception(f"Error publishing message to queue: {e}")

def start_consuming(channel: BlockingChannel):
    """
    Start consuming messages from the queue.

    Args:
    channel (BlockingChannel): A pika BlockingChannel instance.

    Raises:
    AssertionError: If the channel is not provided.
    """
    assert isinstance(channel, BlockingChannel), "A valid BlockingChannel must be provided."

    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        channel.stop_consuming()


    
