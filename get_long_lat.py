# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 11:49:19 2017

@author: astachn1
"""

import os
import numpy as np
from datetime import datetime
import re
import pandas as pd
import pyodbc
import sqlite3
from sqlite3 import OperationalError
import urllib.parse
import json
import certifi
import urllib3
import googlemaps

def insert_new (cursor, table, data):	
	placeholders = ", ".join(["?" for x in range(len(data))])	
	sql = "INSERT OR REPLACE INTO %s VALUES (%s);" % (table, placeholders)	
	cursor.execute(sql, data)
    
def get_row (table, primary_key, pk_id, cursor):
    '''Retrieves all row date for student_id.'''
    try:
        SQL = '''SELECT * FROM %s WHERE %s = %s''' % (table, primary_key, pk_id)
        cursor.execute(SQL)
        return cursor.fetchone()
    except OperationalError as msg:
        print ('Command skipped: ', msg)

def drop_row (table_name, pk_id_name, pk_id, cursor):
    '''docstring'''
    sql = "DELETE FROM %s WHERE %s = %s" (table_name, pk_id_name, pk_id)
    cursor.execute(sql)
    
def update_table (table, primary_key, pk_id, field, value, cursor):
    '''Updates a single field in a table.'''
    sql = f"UPDATE {table} SET {field} = '{value}' WHERE {primary_key} = {pk_id};"
    cursor.execute(sql)
    
def check_database_for_id (student_id, cursor):
    '''Requests the most recent item in the table.'''
    SQL = '''SELECT count(1) FROM students WHERE student_id = %s''' % (student_id)
    cursor.execute(SQL)
    return cursor.fetchone()[0]

def get_pd_row(df, primary_key, pk_id, fields):
    '''docstring'''
    row = []
    row.append(pk_id)
    # Iterate through fields and append items
    for field in fields:
        item = df[df[primary_key] == pk_id][field].item()
        # If the item is NaN, replace with None
        if type(item) == float and np.isnan(item):
            item = None
        row.append(item)
    return row

def parse_address (address):
    '''Function attempts to remove unit/apt info from an address.'''
    # Common strings to identify unit/apt info
    drop_right = ['apt', 'apartment', 'unit', '#', 'suite']
    drop_left = ['floor']
    # Remove caps
    ret_addr = address.lower()
    # Remove periods and commas (replace with empty string)
    for txt in [',', '.']:
        ret_addr = re.sub(r"{0}".format(re.escape(txt)), '', ret_addr)
    # Drop text for drop_right keywords
    for txt in drop_right:
        ret_addr = re.split(txt, ret_addr)[0]
    # Drop text for drop_left keywords
    for txt in drop_left:
        if txt in ret_addr:
            ret_addr = re.split(txt, ret_addr)[0]
            ret_addr = ret_addr.split()
            ret_addr = ret_addr[:-1]
            ret_addr = ' '.join(ret_addr)
    return ret_addr

def get_long_lat (address, city, state):
    '''Function retrieves longitude and latitude from the Google Maps api
    when passed a street address.'''
    address = urllib.parse.quote_plus(address)
    city = urllib.parse.quote_plus(city)
    state = urllib.parse.quote_plus(state)
    # api key belonging to Zander Stachniak
    api_key = 'AIzaSyA0yJtHJVGobbcZUMfc45Cq370QmvaqsCw'
    # secure https request URL
    gmaps_api_URL = 'https://maps.googleapis.com/maps/api/geocode/json?address={0},+{1},+{2}&key={3}'.format(address, city, state, api_key)
    # Submit request
    request = http.request('GET', gmaps_api_URL)
    # Convert json to python dictionary
    result = json.loads(request.data)
    # So long as result came back OK, return longitude and latitude
    if result['status'] == 'OK':
        longitude = result['results'][0]['geometry']['location']['lng']
        latitude = result['results'][0]['geometry']['location']['lat']
    else:
        longitude = 0
        latitude = 0
    return longitude, latitude

def get_travel_time (loc1, loc2):
    '''A function that returns the expected duration of travel time in seconds 
    between two locations using 1) driving and 2) public transit.'''
    driving_time = gmaps.directions(loc1, loc2, mode='driving')[0]['legs'][0]['duration']['value']
    transit_time = gmaps.directions(loc1, loc2, mode='transit')[0]['legs'][0]['duration']['value']
    return driving_time, transit_time

def access_or_update_travel_time (student_id, site_id, cursor):
    '''Function first attempts to pull cached travel time from SQLlite
    database. Failing to find cached travel time, function will pull the
    locations of the student and site, query the Google Maps API to find
    travel time, then upload to database and return as function call.'''
    
    # Query database using student and site IDs
    sql_query = "SELECT driving_time, transit_time FROM travel WHERE student_id = %s AND site_id = %s" % (student_id, site_id)
    cursor.execute(sql_query)
    result = cursor.fetchone()
    
    # If result exists
    if result:
        # Separate out the data
        driving_time, transit_time = result
        
    # If result does not exist
    else:
        # Get student location
        student_query = "SELECT latitude, longitude FROM students WHERE id = %s" % (student_id)
        cursor.execute(student_query)
        student_loc = cursor.fetchone()
        # Get site location
        site_query = "SELECT latitude, longitude FROM locations WHERE id = %s" % (site_id)
        cursor.execute(site_query)
        site_loc = cursor.fetchone()
        # Get travel time
        driving_time, transit_time = get_travel_time (student_loc, site_loc)
        # Upload to database
        insert_tuple = (student_id, site_id, driving_time, transit_time)
        insert_new(cursor, 'travel', insert_tuple)
    
    # Return from function
    return driving_time, transit_time

def executeSQL (command, cursor, p=False):
    '''Executes SQL command when passed a command and a cursor.
    If p is set to True, function will also print results of the query.'''
    try:
        cursor.execute(command.strip('\n'))
        if p == True:
            print('\nQuery: \n{0}\n'.format(command.strip('\n')))
            rows = cursor.fetchall()
            for row in rows:
                print(' | '.join(str(x) for x in row))
    except OperationalError as msg:
        print ('Command skipped: ', msg)

def get_latest (folder_path, name_stump, **kwargs):
    '''Scans folder_path for all files that contain the name_stump.
    For those files, function parses file name to look for date edited,
    then opens and returns the file with the latest date.'''
    # Gather optional keyword arguments
    date_format = kwargs.pop('date_format', '%Y-%m-%d')
    verbose = kwargs.pop('verbose', False)
    converters = kwargs.pop('converters', None)
    #Set up empty lists
    files = []
    dates = []
    if verbose:
        print('Scanning: {}'.format(folder_path))
    for file_name in os.listdir(folder_path):
        subname = os.path.join(folder_path, file_name)
        #Checks for standard naming conventions and ignores file fragments
        if os.path.isfile(subname) and name_stump in subname and '~$' not in subname:
            #Ignore files that end in '_(2).xlsx'
            if (subname[(subname.find('.')-3):subname.find('.')]) == '(2)':
                pass
            else:
                files.append(subname)
                #Parses file names and converts to datetimes
                date = subname[(subname.find('.')-10):subname.find('.')]
                date_time = datetime.strptime(date, date_format).date()
                dates.append(date_time)
    # Find most recent file
    latest = max(dates)
    for file in files:
        if str(latest.strftime(date_format)) in file:
            latest_file = file
    # Verbose printing
    if verbose:
        print('Found {} file(s).'.format(len(files)))
        print('Latest file: {}'.format(latest.strftime('%Y-%m-%d')))
    # Return excel file
    return pd.read_excel(latest_file, converters=converters)

def update_data (df, table_name, unique_ids, df_fields, db_fields, df_pk, cursor):
    '''A function to process data and add to database'''
    # Iterate through IDs
    for unique_id in unique_ids:
        # Test if student_id exists in database
        db_data = get_row(table_name, 'id', unique_id, cursor)
        if db_data:
            # Get data
            df_data = get_pd_row(df, df_pk, unique_id, df_fields)
            # If any of the identifying data has changed, update database
            for i, (df_field, db_field) in enumerate(zip(df_fields, db_fields), 1):
                if db_data[i] != df_data[i]:
                    # Update field
                    update_table(table_name, 'id', unique_id, db_field, df_data[i], cursor)
                    # If it was the address that changed...
                    if df_field == 'Address1':
                        # If address is not blank
                        if df[df[df_pk] == unique_id]['Address1'].item():
                            # Parse address to remove unit/apt number
                            corrected_address = parse_address(df[df[df_pk] == unique_id]['Address1'].item())
                            # Get longitude and latitude
                            long, lat = get_long_lat(corrected_address, df[df[df_pk] == unique_id]['City'].item(), df[df[df_pk] == unique_id]['State'].item())
                            # Update longitude and latitude
                            update_table(table_name, 'id', unique_id, 'corrected_address', corrected_address, cursor)
                            update_table(table_name, 'id', unique_id, 'longitude', long, cursor)
                            update_table(table_name, 'id', unique_id, 'latitude', lat, cursor)
                            # Remove any associated travel times from 'travel'
                            if table_name == 'students':
                                pk = 'student_id'
                            else:
                                pk = 'site_id'
                            drop_row ('travel', pk, unique_id, cursor)
             
        # For students not already in the database
        else:
            # Gather Data
            df_data = get_pd_row(df, df_pk, unique_id, df_fields)      
            # If address is not blank
            if df[df[df_pk] == unique_id]['Address1'].item():
                # Get longitude and latitude
                corrected_address = parse_address(df[df[df_pk] == unique_id]['Address1'].item())
                long, lat = get_long_lat(corrected_address, df[df[df_pk] == unique_id]['City'].item(), df[df[df_pk] == unique_id]['State'].item())
            else:
                corrected_address, long, lat = (None, None, None)
            # Combine all insert data into tuple
            df_data.extend((corrected_address, long, lat))
            # Insert whole new row
            insert_new(cursor, table_name, df_data)

'''////////////////////////////////////////////////////////////
// Table Definitions.                                        //
////////////////////////////////////////////////////////////'''
students_table = '''CREATE TABLE students (
        id                          VARCHAR(7),
        name                        VARCHAR(50),
        campus                      VARCHAR(6),
        start_term                  NUMBER,
        run_term                    NUMBER,
        program                     VARCHAR(50),
        subplan                     VARCHAR(50),
        address                     VARCHAR(100),
        city                        VARCHAR(50),
        state                       VARCHAR(2),
        postal                      VARCHAR(10),
        corrected_address           VARCHAR(100),
        longitude                   NUMBER,
        latitude                    NUMBER,
        CONSTRAINT id_PK PRIMARY KEY (id)
);'''

faculty_table = '''CREATE TABLE faculty(
        id                          VARCHAR(7),
        name                        VARCHAR(200),
        address                     VARCHAR(100),
        city                        VARCHAR(50),
        state                       VARCHAR(2),
        postal                      VARCHAR(10),
        type                        VARCHAR(100),
        corrected_address           VARCHAR(100),
        longitude                   NUMBER,
        latitude                    NUMBER,
        CONSTRAINT id_PK PRIMARY KEY (id)
);'''

locations_table = '''CREATE TABLE locations (
        id                          VARCHAR(7),
        name                        VARCHAR(200),
        address                     VARCHAR(100),
        city                        VARCHAR(50),
        state                       VARCHAR(2),
        postal                      VARCHAR(10),
        type                        VARCHAR(100),
        corrected_address           VARCHAR(100),
        longitude                   NUMBER,
        latitude                    NUMBER,
        CONSTRAINT id_PK PRIMARY KEY (id)
);'''

travel_table = '''CREATE TABLE travel (
        student_id                  VARCHAR(7),
        site_id                     VARCHAR(200),
        driving_time                NUMBER,
        transit_time                NUMBER,
        CONSTRAINT id_PK PRIMARY KEY (student_id, site_id)
);'''



# Create a PoolManager that verifies certificates when making requests
http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED', ca_certs=certifi.where())

# Initialize the googlemaps Client with API key
gmaps = googlemaps.Client(key='AIzaSyCDIQVn-Kf91vMi3yXWGC_RjeejX-K20Kw')

# Open connection to database
database = 'W:\\csh\\Nursing Administration\\Data Management\\Location\\location.db'
conn = sqlite3.connect(database)
cursor = conn.cursor()

# To Drop and Re-add Tables
'''
del_students = "DROP TABLE IF EXISTS students"
cursor.execute(del_students)
cursor.execute(students_table)
del_locations = "DROP TABLE IF EXISTS locations"
cursor.execute(del_locations)
cursor.execute(locations_table)
del_faculty = "DROP TABLE IF EXISTS faculty"
cursor.execute(del_faculty)
cursor.execute(faculty_table)
del_travel = "DROP TABLE IF EXISTS travel"
cursor.execute(del_travel)
cursor.execute(travel_table)
'''

# Gather Student Data
# Open most recent student list
path = 'W:\\csh\\Nursing\\Student Records'
student_list = get_latest(path, 'Student List', converters={'Emplid':str})
# Drop duplicated student IDs - this would only cause problems
student_list.drop_duplicates(subset='Emplid', inplace=True)
# Define fields
student_ids = student_list['Emplid'].unique()
student_df_fields = ['Student Name', 'Campus', 'Admit Term', 'Run Term', 'Maj Desc', 'Maj Subplan Desc', 'Address1', 'City', 'State', 'Postal']
student_db_fields = ['name', 'campus', 'start_term', 'run_term', 'program', 'subplan', 'address', 'city', 'state', 'postal']
# Define the run_term of the student list
run_term = int(student_list['Run Term'][0])
# Update data
update_data(student_list, 'students', student_ids, student_df_fields, student_db_fields, 'Emplid', cursor)

# Gather Site Data
# Set Access db connection
access_conn_str = (
    r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
    r'DBQ=W:\\csh\\Nursing\\Agreement Request Center (ARC)\\Backend\\AffiliationAgreements_Backend.accdb;'
    )
# Connect and open cursor
cnxn = pyodbc.connect(access_conn_str)
crsr = cnxn.cursor()
# Get all site data
sites_cols = ['site_id', 'Name', 'Address1', 'Address2', 'City', 'State', 'Postal', 'Type']
sql = "SELECT * FROM sites"
crsr.execute(sql)
sites_df = pd.DataFrame.from_records(crsr.fetchall(), columns=sites_cols)
sites_df['site_id'] = sites_df['site_id'].astype('str')
# Close Access connection
crsr.close()
cnxn.close()
# Define fields
site_ids = sites_df['site_id'].unique()
site_df_fields = ['Name', 'Address1', 'City', 'State', 'Postal', 'Type']
site_db_fields = ['name', 'address', 'city', 'state', 'postal', 'type']
# Update data
update_data(sites_df, 'locations', site_ids, site_df_fields, site_db_fields, 'site_id', cursor)

# Gather Faculty Data
path = 'W:\\csh\\Nursing\\Faculty\\Employee List.xlsx'
faculty_list = pd.read_excel(path)
# Drop inactive and staff
faculty_list = faculty_list[(faculty_list['Status'] == 'Active') & (faculty_list['Track'] != 'Staff')]
# Drop faculty without ID number
faculty_list.dropna(subset=['Empl ID'], inplace=True)
# Rename address column
faculty_list.rename(columns={'Address': 'Address1'}, inplace=True)
# Define fields
faculty_ids = faculty_list['Empl ID'].unique()
faculty_df_fields = ['Last-First', 'Address1', 'City', 'State', 'Zip', 'Track']
faculty_db_fields = ['name', 'address', 'city', 'state', 'postal', 'type']
# Update data
update_data(faculty_list, 'faculty', faculty_ids, faculty_df_fields, faculty_db_fields, 'Empl ID', cursor)

# Close cursor, commit, and close connection
cursor.close()
conn.commit()
conn.close() 
