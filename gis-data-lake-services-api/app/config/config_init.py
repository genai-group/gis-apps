#!/usr/bin/python

#%%
import os

print(f"Loading all required Python modules...")

from config.config_variables import *
from config.config_modules import *
from config.config_clients import *
from config.config_functions import *

# Loading the YAML files from the config/templates directory
template_dir = './config/templates'


################################################
####    Loading the Synthetic Data Files    ####
################################################

if GIS_ENVIRONMENT == 'flask-local':
    data_dir = 'app/config/data/synthetic'

if GIS_ENVIRONMENT == 'local':
    data_dir = 'config/data/synthetic'

try:
    manifest_data_0 = open_file(f"{data_dir}/fake_airline_manifest_0_hours.json")
    manifest_data_24 = open_file(f"{data_dir}/fake_airline_manifest_24_hours.json")
    manifest_data_72 = open_file(f"{data_dir}/fake_airline_manifest_72_hours.json")
    print(f"Successfully loaded data for the fake_airline_manifest files (0, 24 adn 72 hours).")
except Exception as e:
    print(f"Error loading data: {e}")
    raise

# Reading in the fake_passport data
try:
    fake_passport_0 = open_file(f"{data_dir}/fake_passport_0_hours.json")
    fake_passport_24 = open_file(f"{data_dir}/fake_passport_24_hours.json")
    fake_passport_72 = open_file(f"{data_dir}/fake_passport_72_hours.json")
    print(f"Successfully loaded data for the fake_passport data files (0, 24 adn 72 hours).")
except Exception as e:
    print(f"Error loading fake_passport: {e}")
    raise

# Reading in the fake_customs_report data at the 0, 24, and 72 hours
try:
    fake_customs_report_hour_0 = open_file(f"{data_dir}/fake_customs_report_0_hours.json")
    fake_customs_report_hour_24 = open_file(f"{data_dir}/fake_customs_report_24_hours.json")
    fake_customs_report_hour_72 = open_file(f"{data_dir}/fake_customs_report_72_hours.json")
    print(f"Successfully loaded data for the fake_customs_report data files (0, 24 adn 72 hours).")
except Exception as e:
    print(f"Error loading fake_customs_report: {e}")
    raise


###############################################
####    Loading the External Data Files    ####
###############################################

if GIS_ENVIRONMENT == 'flask-local':
    data_dir = 'app/config/data/external'

if GIS_ENVIRONMENT == 'local':
    data_dir = 'config/data/external'

# Global Terrorism Index
try:
    gti = open_file(f"{data_dir}/global_terrorism_index.csv")
    gti = pd.DataFrame(gti)
    gti.set_index('country', inplace=True)
    gti_dict = gti.to_dict('index')
    print(f"Successfully loaded data for the Global Terrorism Index data file.")
except Exception as e:
    print(f"Errors loading the global terrorism index")
    raise

# Reading ISO json object
try:
    iso_data = open_file(f"{data_dir}/iso.json")
    iso_data = pd.DataFrame(iso_data)
    iso_data.set_index('name', inplace=True)
    iso_dict = iso_data.to_dict('index')
    # Add global terrorist index
    for k,v in iso_dict.items():
        try:
            iso_dict[k]['terrorism_index'] = gti_dict[k]
        except:
            pass
    print(f"Successfully loaded data for the ISO data file.")
except Exception as e:
    print(f"Error loading ISO data: {e}")
    raise

# languages
try:
    spoken_languages = open_file(f"{data_dir}/languages.json")
    language_list = spoken_languages['language_list']
    print(f"Successfully loaded data for the Languages list: {spoken_languages}")
except Exception as e:
    print(f"Errors loading spoken languages: {e}")
    raise

# Reading in the International Government IDs - Sheet1.csv
try:
    government_ids = open_file(f"{data_dir}/International Government IDs - Sheet1.csv")
    government_ids = pd.DataFrame(government_ids)
    government_ids = government_ids.to_dict(orient='records')
    government_ids = {obj['Country']:{'Name':obj['Name'], 'Description':obj['Description']} for obj in government_ids}
    print(f"Successfully loaded data for the International Government IDs.")
except Exception as e:
    print(f"Error loading International Government IDs: {e}")
    raise

# IATA International Airport Codes
try:
    airport_codes = open_file(f"{data_dir}/international_airport_codes.csv")
    print(f"Successfully loaded data for the Airport Codes.")
except Exception as e:
    print(f"Errors reading in IATA international airport codes.")    

# World Cities
try:
    world_cities = open_file(f"{data_dir}/worldcities.csv")
except Exception as e:
    print(f"Errors reading in world cities.")


######################################################
####    Loading the Customer Sample Data Files    ####
######################################################

if GIS_ENVIRONMENT == 'flask-local':
    data_dir = 'app/config/data/sample_data'

if GIS_ENVIRONMENT == 'local':
    data_dir = 'config/data/sample_data'

try:
    raw_pax_list_3 = open_file(f'{data_dir}/RAW PAX LIST 3.txt')
    print(f"Successfully loaded data for the RAW PAX LIST 3.txt file.")
except Exception as e:
    print(f"Error loading RAW PAX LIST 3.txt: {e}")
    pass

try:
    atsg_passenger_export = open_file(f'{data_dir}/ATSG Passenger Export.xsd')
    print(f"Successfully loaded data for the ATSG Passenger Export.xsd file.")
except Exception as e:
    print(f"Error loading ATSG Passenger Export.xsd: {e}")
    pass

try:
    raw_pnrgov = open_file(f'{data_dir}/RAW PNRGOV ESGBSC.txt')
    print(f"Successfully loaded data for the RAW PNRGOV ESGBSC.txt file.")
except Exception as e:
    print(f"Error loading RAW PNRGOV ESGBSC.txt: {e}")
    pass

try:
    raw_paxlist = open_file(f'{data_dir}/RAW PAXLIST 2.txt')
    print(f"Successfully loaded data for the RAW PAXLIST 2.txt file.")
except Exception as e:
    print(f"Error loading RAW PAXLIST 2.txt: {e}")
    pass

try:
    atsg_export = open_file(f'{data_dir}/ATSG EXPORT with raw PNRGOV and API.xml')
    print(f"Successfully loaded data for the ATSG EXPORT with raw PNRGOV and API.xml file.")
except Exception as e:
    print(f"Error loading ATSG EXPORT with raw PNRGOV and API.xml: {e}")
    pass

try:
    raw_pnrgov_khnpul = open_file(f'{data_dir}/RAW PNRGOV KHNPUL.txt')
    print(f"Successfully loaded data for the RAW PNRGOV KHNPUL.txt file.")
except Exception as e:
    print(f"Error loading RAW PNRGOV KHNPUL.txt: {e}")
    pass

try:
    raw_api_paxlist = open_file(f'{data_dir}/RAW API PAXLIST 1.txt')
    print(f"Successfully loaded data for the RAW API PAXLIST 1.txt file.")
except Exception as e:
    print(f"Error loading RAW API PAXLIST 1.txt: {e}")
    pass

try:
    raw_pnrgov_kxugjf = open_file(f'{data_dir}/RAW PNRGOV KXUGJF.txt')
    print(f"Successfully loaded data for the RAW PNRGOV KXUGJF.txt file.")
except Exception as e:
    print(f"Error loading RAW PNRGOV KXUGJF.txt: {e}")
    pass

#########################
####    Trade.gov    ####
#########################

def get_bsp_count(subscription_key):
    """
    This function returns the count of business service providers from trade.gov
    
    Parameters:
    subscription_key (str): The subscription key for the trade.gov API
    
    Returns:
    dict: The count of business service providers
    """
    assert isinstance(subscription_key, str), f"subscription_key must be a string"
    url = "https://data.trade.gov/business_service_providers/v1/count"
    headers = {
        'Accept': 'application/json',
        'subscription-key': subscription_key
    }

    try:
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            return response.json()
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return f"Error with Trade.gov call: {e}"

def search_business_service_providers(q: Optional[str] = None, 
                                      categories: Optional[str] = None, 
                                      ita_offices: Optional[str] = None, 
                                      offset: Optional[int] = None, 
                                      size: Optional[int] = None,
                                      subscription_key: str = '') -> Dict[str, Any]:
    """
    Search business service providers from the Trade.gov API.

    Args:
        q (Optional[str]): Search query.
        categories (Optional[str]): Categories filter.
        ita_offices (Optional[str]): ITA offices filter.
        offset (Optional[int]): Offset for pagination.
        size (Optional[int]): Size of the result set.

    Returns:
        Dict[str, Any]: The JSON response from the API.
    """
    headers = {
        'Accept': 'application/json',
        'subscription-key': subscription_key
    }

    url = 'https://data.trade.gov/business_service_providers/v1/search'
    params = {
        'q': q,
        'categories': categories,
        'ita_offices': ita_offices,
        'offset': offset,
        'size': size
    }

    try:
        response = requests.get(url, params={k: v for k, v in params.items() if v is not None}, headers=headers)
        response.raise_for_status()  # Raises HTTPError for bad requests (4XX, 5XX)
        return response.json()
    except requests.RequestException as e:
        raise SystemExit(f"Request failed: {e}")

