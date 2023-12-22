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

def define_schema(session):
    with session.transaction(TransactionType.WRITE) as transaction:
        transaction.query().define("""
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
        """)
        transaction.commit()

def main(database_name, flight_info, passengers, uri=TYPEDB_URI):
    # Create a TypeDB client
    with TypeDB.core_driver(uri) as client:
        # Check if the database exists, if not create it
        # database_manager = client.databases()
        # if not database_manager.contains(database_name):
        #     database_manager.create(database_name)

        # Define schema
        with client.session(database_name, SessionType.SCHEMA) as session:
            define_schema(session)

        # Insert data
        with client.session(database_name, SessionType.DATA) as session:
            insert_data(session, flight_info, passengers)

"""
main("gis_main", flight_info=flight_info, passengers=passengers)

"""