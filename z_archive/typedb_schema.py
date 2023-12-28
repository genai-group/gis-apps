#!/usr/bin/python

# Loading all required Python modules...
import os
from config.modules import *
from config.variables import *
from config.clients import *
from config.functions import *


#############################
####    TypeDB Schema    ####
#############################

def define_schema(driver, database_name):
    define_query = """
        define

        flight sub entity,
            has flight_number,
            has departure_time,
            has arrival_time,
            has departure_airport,
            has arrival_airport,
            has airline;

        passenger sub entity,
            has name,
            has passport_number,
            has seat_number;

        on_flight sub relation,
            relates boarded_flight,
            relates boarded_passenger;

        flight_number sub attribute, value string;
        departure_time sub attribute, value datetime;
        arrival_time sub attribute, value datetime;
        departure_airport sub attribute, value string;
        arrival_airport sub attribute, value string;
        airline sub attribute, value string;
        name sub attribute, value string;
        passport_number sub attribute, value string;
        seat_number sub attribute, value string;
    """

    with driver.session(database_name, SessionType.SCHEMA) as session:
        with session.transaction(TransactionType.WRITE) as transaction:
            transaction.query().define(define_query)
            transaction.commit()

# def main(database_name, flight_info, passengers, uri="localhost:1729"):
#     # Create a TypeDB client
#     with TypeDB.core_driver(uri) as client:
#         # Check if the database exists, if not create it
#         if not client.databases().contains(database_name):
#             client.databases().create(database_name)

#         # Define schema
#         with client.session(database_name, SessionType.SCHEMA) as session:
#             define_schema(session)

#         # Insert data
#         with client.session(database_name, SessionType.DATA) as session:
#             # Assuming insert_data function is defined elsewhere
#             insert_data(session, flight_info, passengers)

def main():
    with TypeDB.core_driver(TYPEDB_URI) as client:
        # Database check and creation logic here...

        # Define schema
        with client.session('gis_main', SessionType.SCHEMA) as session:
            define_schema(session)

        # Load data
        # Here you would implement or call your data loading function
        # For example, load_data(client)

# Main script execution
if __name__ == '__main__':
    main()

"""

## Building the schema and loading the data into TypeDB

data = manifest_data_0

flight_info = data['flight_information']
passengers = data['passenger_info']

main("gis_main", flight_info=flight_info, passengers=passengers)

"""
# %%
