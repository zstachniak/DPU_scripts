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
import dpu.scripts as dpu
from dpu.file_locator import FileLocator
import click
import folium
from folium.plugins import HeatMap
from itertools import product
# bokeh imports
from bokeh.plotting import figure, show
from bokeh.models import HoverTool, Range1d, BoxZoomTool, PanTool, ResetTool
from bokeh.embed import file_html
from bokeh.resources import CDN
from bokeh.models.widgets import Panel, Tabs

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

def get_long_lat (address, city, state, http, api_key):
    '''Function retrieves longitude and latitude from the Google Maps api
    when passed a street address.'''
    address = urllib.parse.quote_plus(address)
    city = urllib.parse.quote_plus(city)
    state = urllib.parse.quote_plus(state)
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

def get_travel_time (loc1, loc2, gmaps):
    '''A function that returns the expected duration of travel time in seconds 
    between two locations using 1) driving and 2) public transit.'''
    try:
        driving_time = gmaps.directions(loc1, loc2, mode='driving')[0]['legs'][0]['duration']['value']
    except:
        driving_time = None
    try:
        transit_time = gmaps.directions(loc1, loc2, mode='transit')[0]['legs'][0]['duration']['value']
    except:
        transit_time = None
    return driving_time, transit_time

def access_or_update_travel_time (student_id, site_id, cursor, gmaps):
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
        driving_time, transit_time = get_travel_time (student_loc, site_loc, gmaps)
        # Upload to database
        insert_tuple = (student_id, site_id, driving_time, transit_time)
        dpu.insert_new(cursor, 'travel', insert_tuple)
    
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

def update_data (df, table_name, unique_ids, df_fields, db_fields, df_pk, cursor, conn, http, api_key):
    '''A function to process data and add to database'''
    # Iterate through IDs
    for unique_id in unique_ids:
        # Test if student_id exists in database
        db_data = dpu.get_row(table_name, 'id', unique_id, cursor)
        if db_data:
            # Get data
            df_data = get_pd_row(df, df_pk, unique_id, df_fields)
            # If any of the identifying data has changed, update database
            for i, (df_field, db_field) in enumerate(zip(df_fields, db_fields), 1):
                if db_data[i] != df_data[i]:
                    # Update field
                    dpu.update_table(table_name, 'id', unique_id, db_field, df_data[i], cursor)
                    # If it was the address that changed...
                    if df_field == 'Address1':
                        # If address is not blank
                        addr = df[df[df_pk] == unique_id]['Address1'].item()
                        if addr and type(addr) == str:
                            # Parse address to remove unit/apt number
                            corrected_address = parse_address(addr)
                            # Get longitude and latitude
                            long, lat = get_long_lat(corrected_address, df[df[df_pk] == unique_id]['City'].item(), df[df[df_pk] == unique_id]['State'].item(), http, api_key)
                            # Update longitude and latitude
                            dpu.update_table(table_name, 'id', unique_id, 'corrected_address', corrected_address, cursor)
                            dpu.update_table(table_name, 'id', unique_id, 'longitude', long, cursor)
                            dpu.update_table(table_name, 'id', unique_id, 'latitude', lat, cursor)
                            # Remove any associated travel times from 'travel'
                            if table_name == 'students':
                                pk = 'student_id'
                            else:
                                pk = 'site_id'
                            dpu.drop_row ('travel', pk, unique_id, cursor)
            # If the long/lat failed last time (zero values), try updating
            if db_data[-1] == 0 and db_data[-2] == 0:
                addr = df[df[df_pk] == unique_id]['Address1'].item()
                if addr and type(addr) == str:
                    # Parse address to remove unit/apt number
                    corrected_address = parse_address(addr)
                    # Get longitude and latitude
                    long, lat = get_long_lat(corrected_address, df[df[df_pk] == unique_id]['City'].item(), df[df[df_pk] == unique_id]['State'].item(), http, api_key)
                    # Update longitude and latitude
                    dpu.update_table(table_name, 'id', unique_id, 'corrected_address', corrected_address, cursor)
                    dpu.update_table(table_name, 'id', unique_id, 'longitude', long, cursor)
                    dpu.update_table(table_name, 'id', unique_id, 'latitude', lat, cursor)
                    # Remove any associated travel times from 'travel'
                    if table_name == 'students':
                        pk = 'student_id'
                    else:
                        pk = 'site_id'
                    dpu.drop_row ('travel', pk, unique_id, cursor)
             
        # For those not already in the database
        else:
            # Gather Data
            df_data = get_pd_row(df, df_pk, unique_id, df_fields)
            # If address is not blank
            addr = df[df[df_pk] == unique_id]['Address1'].item()
            if addr and type(addr) == str:
                # Get longitude and latitude
                corrected_address = parse_address(addr)
                long, lat = get_long_lat(corrected_address, df[df[df_pk] == unique_id]['City'].item(), df[df[df_pk] == unique_id]['State'].item(), http, api_key)
            else:
                corrected_address, long, lat = (None, None, None)
            # Combine all insert data into tuple
            df_data.extend((corrected_address, long, lat))
            # Insert whole new row
            dpu.insert_new(cursor, table_name, df_data)
    conn.commit()

def gather_site_travel_data (site_id, student_df, cursor, sort_by='driving_time'):
    '''docstring'''
    # Gather data from the travel table
    student_ids = student_df['student_id'].values
    sql = "SELECT * FROM travel WHERE site_id = {} AND student_id IN ({seq})".format(site_id, seq=','.join(['?']*len(student_ids)))
    cursor.execute(sql, student_ids)
    # Fetch data into dataframe
    temp_df = pd.DataFrame(cursor.fetchall(), columns=travel_cols)
    # Convert seconds to minutes
    temp_df['driving_time'] = temp_df['driving_time'] / 60
    temp_df['transit_time'] = temp_df['transit_time'] / 60
    # Merge with student data for additional info
    temp_df = pd.merge(temp_df, student_df, how='left', on='student_id')
    # Sort values and reset the index
    temp_df.sort_values(by=sort_by, inplace=True)
    temp_df.reset_index(drop=True, inplace=True)
    return temp_df

def gather_student_travel_data (student_id, site_df, cursor, sort_by='driving_time'):
    '''docstring'''
    # Gather data from the travel table
    site_ids = site_df['site_id'].values
    sql = "SELECT * FROM travel WHERE student_id = {} AND site_id IN ({seq})".format(student_id, seq=','.join(['?']*len(site_ids)))
    cursor.execute(sql, site_ids)
    # Fetch data into dataframe
    temp_df = pd.DataFrame(cursor.fetchall(), columns=travel_cols)
    # Convert seconds to minutes
    temp_df['driving_time'] = temp_df['driving_time'] / 60
    temp_df['transit_time'] = temp_df['transit_time'] / 60
    # Merge with site data for additional info
    temp_df = pd.merge(temp_df, site_df, how='left', on='site_id')
    # Sort values and reset the index
    temp_df.sort_values(by=sort_by, ascending=False, inplace=True)
    temp_df.reset_index(drop=True, inplace=True)
    return temp_df

def plot_student_transit (df, student_name, **kwargs):
    '''Function to plot the transit times for each student. bokeh does not
    support horizontal bars out-of-box, so we define our own simple
    implementation using rectangles.'''
    
    # Ensure that no NaNs remain (incompatible with JSON output)
    df.fillna(0, inplace=True)
    
    # Gather measurements for bar sizes
    ht = kwargs.pop('height', 0.3)
    y_offset = ht / 2
    
    # Privacy enabler
    privacy_guard = kwargs.pop('privacy_guard', False)
    if privacy_guard:
        student_name = '***'
    
    # Define tools
    pan_tool = PanTool()
    box_zoom_tool = BoxZoomTool()
    reset_tool = ResetTool()
    # The ordering defines which tools is active at the outset
    tools = [pan_tool, box_zoom_tool, reset_tool]
    
    # Initialize figure
    p = figure(plot_width=600, plot_height=400, tools=tools, title="Travel Time for {}".format(student_name), y_range=df['name'].tolist())
    
    # Initialize a counter
    j = 0.5
    # Iterate through driving and transit times
    for dt, tt in zip(df['driving_time'], df['transit_time']):
        # Plot rectangles
        p.rect(x=dt/2, y=j + y_offset, width=dt, height=ht, fill_color='blue', fill_alpha=0.6, legend="Driving Time")
        p.rect(x=tt/2, y=j - y_offset, width=tt, height=ht, line_color='green', fill_color='green', fill_alpha=0.6, legend="Transit Time")
        j += 1
    
    # Define legend location
    p.legend.location = "bottom_right"
    
    # Set as html object
    html = file_html(p, CDN)
    return html

def plot_site_transit (df, site_name, **kwargs):
    '''A function to build html bokeh charts showing travel distance to
    clinical sites.'''
    
    # Gather optional keyword arguments
    height = kwargs.pop('height', 400)
    width = kwargs.pop('width', 400)
    y_max = kwargs.pop('y_max', 120)
    privacy_guard = kwargs.pop('privacy_guard', False)
    
    # Define a collection of active tools
    hover = HoverTool(tooltips=[
        ('Name', '@name'),
        ('Student ID', '@student_id'),
        ('Program', '@program'),
        ('Campus', '@campus'),
        ('Driving Time', '@driving_time'),
        ('Transit Time', '@transit_time'),
    ])
    pan_tool = PanTool()
    box_zoom_tool = BoxZoomTool()
    reset_tool = ResetTool()
    # The ordering defines which tool is active at the outset
    if privacy_guard:
        tools = [pan_tool, box_zoom_tool, reset_tool]
    else:
        tools = [hover, pan_tool, box_zoom_tool, reset_tool]
    
    # Initialize the figure for panel 1
    p1 = figure(plot_width=width, plot_height=height, tools=tools, title="Travel Time to {}".format(site_name))
    # Plot circles
    p1.circle(x='index', y='driving_time', size=15, fill_alpha=0.6, source=df)
    # Set a y range using a Range1d
    p1.y_range = Range1d(0, y_max)
    # Set axis labels
    p1.xaxis.axis_label = 'Student Index'
    p1.yaxis.axis_label = 'Driving Time (in minutes)'
    # Set as panel object
    tab1 = Panel(child=p1, title="Driving")
    
    # Initialize the figure for panel 2
    p2 = figure(plot_width=width, plot_height=height, tools=tools, title="Travel Time to {}".format(site_name))
    # Plot circles
    p2.circle(x='index', y='transit_time', size=15, fill_alpha=0.6, source=df)
    # Set a y range using a Range1d
    p2.y_range = Range1d(0, y_max)
    # Set axis labels
    p2.xaxis.axis_label = 'Student Index'
    p2.yaxis.axis_label = 'Transit Time (in minutes)'
    # Set as panel object
    tab2 = Panel(child=p2, title="Transit")
    
    # Set as Tabs object
    tabs = Tabs(tabs=[tab1, tab2])
    # Set as html object
    html = file_html(tabs, CDN)
    return html

def plot_map(cursor, **kwargs):
    '''A wrapper function to control overall plotting of folium map.
    
    # Icon Sets (http://fontawesome.io/icons/)
    '''
    
    # Gather optional keyword arguments
    center_map = kwargs.pop('center_map', [41.8781, -87.6298])
    student_list = kwargs.pop('student_list', None)
    site_list = kwargs.pop('site_list', None)
    dpu_list = kwargs.pop('dpu_list', None)
    fac_list = kwargs.pop('fac_list', None)
    outfile = kwargs.pop('outfile', 'index.html')
    privacy_guard = kwargs.pop('privacy_guard', False)
    
    # Center base map (default is Chicago)
    m = folium.Map(location=center_map)
    
    if dpu_list is not None:
        # Gather icon keyword arguments
        dpu_icon = kwargs.pop('dpu_icon', 'home')
        dpu_color = kwargs.pop('dpu_color', 'blue')
        dpu_icon_color = kwargs.pop('dpu_icon_color', 'white')
        # Create feature group
        depaul = folium.map.FeatureGroup(name='DePaul Campuses')
        # Gather location and popup identifiers
        locations = zip(dpu_list['latitude'], dpu_list['longitude'])
        identifier = zip(dpu_list['name'], dpu_list['site_id'])
        # Plot markers
        for point, ident in zip(locations, identifier):
            depaul.add_child(folium.Marker(point, popup=folium.Popup('{0} ({1})'.format(*ident), parse_html=True), icon=folium.Icon(color=dpu_color, icon_color=dpu_icon_color, icon=dpu_icon)))
        # Add as child on its own layer
        m.add_child(depaul)
        
    if fac_list is not None:
        # Gather icon keyword arguments
        fac_icon = kwargs.pop('fac_icon', 'user')
        fac_color = kwargs.pop('fac_color', 'green')
        fac_icon_color = kwargs.pop('fac_icon_color', 'white')
        # Create feature group
        fac = folium.map.FeatureGroup(name='Faculty')
        # Gather location and popup identifiers
        locations = zip(fac_list['latitude'], fac_list['longitude'])
        identifier = zip(fac_list['name'], fac_list['id'])
        # Plot markers
        for point, ident in zip(locations, identifier):
            fac.add_child(folium.Marker(point, popup=folium.Popup('{0} ({1})'.format(*ident), parse_html=True), icon=folium.Icon(color=fac_color, icon_color=fac_icon_color, icon=fac_icon)))
        # Add as child on its own layer
        m.add_child(fac)
        
    if site_list is not None:
        # Gather icon keyword arguments
        site_icon = kwargs.pop('site_icon', 'briefcase')
        site_color = kwargs.pop('site_color', 'red')
        site_icon_color = kwargs.pop('site_icon_color', 'white')
        # Create feature group
        sites = folium.map.FeatureGroup(name='Clinical Sites')
        # Gather location and popup identifiers
        locations = zip(site_list['latitude'], site_list['longitude'])
        identifier = zip(site_list['name'], site_list['site_id'])
        # Plot markers
        for point, ident in zip(locations, identifier):
            # Gather travel data
            temp_site_df = gather_site_travel_data(ident[1], student_list, cursor)
            # Build a bokeh html graph
            html = plot_site_transit(temp_site_df, ident[0], privacy_guard=privacy_guard)
            # Add site as child
            sites.add_child(folium.Marker(point, popup=folium.Popup(folium.IFrame(html, width=500, height=500), max_width=500), icon=folium.Icon(color=site_color, icon_color=site_icon_color, icon=site_icon)))
        # Add as child on its own layer
        m.add_child(sites)

    if student_list is not None:
        # Gather icon keyword arguments
        stud_icon = kwargs.pop('stud_icon', 'user')
        stud_color = kwargs.pop('stud_color', 'blue')
        stud_icon_color = kwargs.pop('stud_icon_color', 'white')
        # Create feature group
        students = folium.map.FeatureGroup(name='Students')
        # Gather location and popup identifiers
        locations = list(zip(student_list['latitude'], student_list['longitude']))
        identifier = zip(student_list['name'], student_list['student_id'], student_list['program'], student_list['campus'])
        # Plot markers
        for point, ident in zip(locations, identifier):
            # Gather travel data
            temp_student_df = gather_student_travel_data (ident[1], site_list, cursor)
            if temp_student_df.empty:
                # Display only student data
                html = '{0} ({1}): {2} at {3}'.format(*ident)
            else:
                # Build a bokeh html graph
                html = plot_student_transit(temp_student_df, ident[0], privacy_guard=privacy_guard)
            # Add student as child
            students.add_child(folium.Marker(point, popup=folium.Popup(folium.IFrame(html, width=700, height=500), max_width=700), icon=folium.Icon(color=stud_color, icon_color=stud_icon_color, icon=stud_icon)))
        # Add as child on its own layer
        m.add_child(students)
        # Check if student heatmap is requested (default True)
        student_heatmap = kwargs.pop('student_heatmap', True)
        if student_heatmap:
            # Plot a heatmap of student addresses
            heatmap = HeatMap(locations, name='Student Heatmap')
            # Add as child on its own layer
            m.add_child(heatmap)
    
    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)
    # Save to outfile
    m.save(outfile)

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
# Define Database Fields
student_df_fields = ['Student Name', 'Campus', 'Admit Term', 'Run Term', 'Maj Desc', 'Maj Subplan Desc', 'Address1', 'City', 'State', 'Postal']
student_db_fields = ['name', 'campus', 'start_term', 'run_term', 'program', 'subplan', 'address', 'city', 'state', 'postal']
site_df_fields = ['Name', 'Address1', 'City', 'State', 'Postal', 'Type']
site_db_fields = ['name', 'address', 'city', 'state', 'postal', 'type']
faculty_df_fields = ['Last-First', 'Address1', 'City', 'State', 'Zip', 'Track']
faculty_db_fields = ['name', 'address', 'city', 'state', 'postal', 'type']

# Define standard column names (for import into Pandas)
student_cols = ['student_id', 'name', 'campus', 'start_term', 'run_term', 'program', 'subplan', 'address', 'city', 'state', 'postal', 'corrected_address', 'longitude', 'latitude']
site_cols = ['site_id', 'name', 'address', 'city', 'state', 'postal', 'type', 'corrected_address', 'longitude', 'latitude']
faculty_cols = ['id', 'name', 'address', 'city', 'state', 'postal', 'type', 'corrected_address', 'longitude', 'latitude']
travel_cols = ['student_id', 'site_id', 'driving_time', 'transit_time']

@click.command()
@click.option(
        '--api-key', '-a',
        envvar="API_KEY",
        help="Your Google Maps API key. Code will default to first check to see if API_KEY is already an environment variable and use that. If there is no environment variable, or if you enter this your first time, code will automatically store for you as an environment variable."
)
@click.option(
        '--term', '-t', type=str,
        help='Term for which to chart location data',
)
@click.option(
        '--cr', '-cr', type=str,
        help='Course list for which to chart location data',
)
@click.option(
        '--prog', '-p', type=str,
        help='Program for which to chart location data',
)
@click.option(
        '--sites', '-s', type=str,
        help='A path to a file listing sites you want to use.',
)
@click.option(
        '--f_name', '-f', '-o', type=str,
        help='Output file name.',
)
def main (api_key, term, cr, prog, sites, f_name):
    '''Main function call.'''
    # Ensure API key
    if not api_key:
        raise "You did not specify an API_KEY and there is no key saved in your environment variables. Use --api-key or -a to specify key."
    
    # Initialize File Locator
    FL = FileLocator()       
    
    # Create a PoolManager that verifies certificates when making requests
    http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED', ca_certs=certifi.where())
    
    # Initialize the googlemaps Client with API key
    gmaps = googlemaps.Client(key=api_key)
    
    # Open connection to database
    db_path = os.path.join(os.path.sep, FL.location, 'location.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print('Updating student data...', end='', flush=True)
    # Gather student data
    student_list = dpu.get_student_list()
    # Drop duplicated student IDs - this would only cause problems
    student_list.drop_duplicates(subset='Emplid', inplace=True)
    # Update student data
    student_ids = student_list['Emplid'].unique()
    #run_term = int(student_list['Run Term'][0])
    update_data(student_list, 'students', student_ids, student_df_fields, student_db_fields, 'Emplid', cursor, conn, http, api_key)
    print('complete!')
    print('Updating site data...', end='', flush=True)
    # Gather site data
    # Data currently stored in ARC database
    arc_path = os.path.join(os.path.sep, FL.arc, 'Backend', 'AffiliationAgreements_Backend.accdb')   
    access_conn_str = (r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)}; DBQ=%s;' % arc_path)
    # Connect and open cursor
    cnxn = pyodbc.connect(access_conn_str)
    crsr = cnxn.cursor()
    # Get all site data
    sites_cols = ['site_id', 'Name', 'Address1', 'Address2', 'City', 'State', 'Postal', 'Type']
    sql = "SELECT * FROM sites"
    crsr.execute(sql)
    sites_df = pd.DataFrame.from_records(crsr.fetchall(), columns=sites_cols)
    sites_df['site_id'] = sites_df['site_id'].astype('str')
    site_ids = sites_df['site_id'].unique()
    all_site_names = sites_df[sites_df['Type'] == 'Hospital']['Name'].unique().tolist()
    # Close Access connection
    crsr.close()
    cnxn.close()
    # Update data
    update_data(sites_df, 'locations', site_ids, site_df_fields, site_db_fields, 'site_id', cursor, conn, http, api_key)
    print('complete!')
    print('Updating faculty data...', end='', flush=True)
    # Gather Faculty Data
    faculty_list = dpu.get_employee_list()
    # Drop inactive and staff
    faculty_list = faculty_list[(faculty_list['Status'] == 'Active') & (faculty_list['Track'] != 'Staff')]
    # Drop faculty without ID number
    faculty_list.dropna(subset=['Empl ID'], inplace=True)
    # Rename address column
    faculty_list.rename(columns={'Address': 'Address1'}, inplace=True)
    faculty_ids = faculty_list['Empl ID'].unique()
    # Update data
    update_data(faculty_list, 'faculty', faculty_ids, faculty_df_fields, faculty_db_fields, 'Empl ID', cursor, conn, http, api_key)
    print('complete!')
    
    print('Gathering student and faculty roster...', end='', flush=True)
    # If no term provided, guess the current term
    TermDescriptions = dpu.get_term_descriptions()
    if not term:
        term = dpu.guess_current_term(TermDescriptions)

    # Gather schedule
    schedule = dpu.get_schedule(term, TermDescriptions)
    # Filter schedule
    if prog:
        schedule = schedule[schedule['Program'] == prog].copy(deep=True)
        if prog == 'MENP':
            MS_progs = ['BS-Health Sciences Combined', 'MS-Generalist Nursing']
            possible_ids = student_list[(student_list['Campus'] == 'LPC') & (student_list['Maj Desc'].isin(MS_progs))]['Emplid'].unique().tolist()
        elif prog == 'RFU':
            possible_ids = student_list[student_list['Campus'] == 'RFU']['Emplid'].unique().tolist()
        elif prog == 'DNP':
            possible_ids = student_list[student_list['Maj Desc'] == 'Doctor of Nursing Practice']['Emplid'].unique().tolist()
        elif prog == 'RN-MS':
            RN_progs = ['BS-Nursing  RN to MS', 'MS-Nursing RN to MS']
            possible_ids = student_list[student_list['Maj Desc'].isin(RN_progs)]['Emplid'].unique().tolist()
    else:
        schedule = schedule[schedule['Program'].isin(['MENP', 'RFU'])]
        MS_progs = ['BS-Health Sciences Combined', 'MS-Generalist Nursing']
        possible_ids = student_list[student_list['Maj Desc'].isin(MS_progs)]['Emplid'].unique().tolist()
    if cr:
        possible_courses = [cr]
    else:
        possible_courses = schedule['Cr'].unique().tolist()
    
    # Need to gather roster data (student IDs and faculty IDs only) as array
    roster = dpu.get_student_roster()
    # Filter roster
    cln_types = ['PRA', 'CLN']
    roster = roster[(roster['Term'] == term) & (roster['Student ID'].isin(possible_ids)) & (roster['Role'] == 'PI') & (roster['Cr'].isin(possible_courses)) & ( (roster['Type'].isin(cln_types)) | ((roster['Cr'] == '301') & (roster['Type'] == 'LAB')))].copy(deep=True)
    # Takes care of 301 issue (due to changing course times)
    roster = roster.sort_values(by=['Faculty_ID', 'Start Date'], ascending=[True, False]).drop_duplicates(subset=['Term', 'Student ID', 'Cr', 'Sec']).copy(deep=True)
    print('complete!')
    # Gather student and faculty IDs
    requested_students = np.unique(roster['Student ID'].values)
    requested_faculty = np.unique(roster['Faculty_ID'].values)
    
    # All DePaul Campuses
    sql = "SELECT id FROM locations WHERE type = 'DePaul'"
    cursor.execute(sql)
    requested_dpu = np.ravel(cursor.fetchall())
    
    print('Gathering sites...', end='', flush=True)
    # Gather requested sites    
    if sites:        
        try:
            with open(sites, 'r') as infile:
                site_names = infile.readlines()
            site_names = [x.strip() for x in site_names]
        except:
            raise 'Could not open file {}'.format(sites)
        # Find best string match for each site name provided in file
        best_matches = []
        for site_name in site_names:
            best_matches.append(dpu.find_best_string_match(site_name, all_site_names))
        best_matches = list(set(best_matches))
        requested_sites = sites_df[sites_df['Name'].isin(best_matches)]['site_id'].unique().tolist()
    # Default to all hospitals in IL
    else:
        requested_sites = sites_df[(sites_df['State'] == 'IL') & (sites_df['Type'] == 'Hospital')]['site_id'].unique().tolist()
    print('complete!')
    print('Updating student travel times...', end='', flush=True)
    # Update all the possible travel times
    for student_id, site_id in product(requested_students, requested_sites):
        _, _ = access_or_update_travel_time(student_id, site_id, cursor, gmaps)
    print('complete!')
    # Then pull the students, sites
    sql = "SELECT * FROM students WHERE id IN ({seq})".format(seq=','.join(['?']*len(requested_students)))
    cursor.execute(sql, requested_students)
    students = pd.DataFrame(cursor.fetchall(), columns=student_cols)
    
    sql = "SELECT * FROM locations WHERE id IN ({seq})".format(seq=','.join(['?']*len(requested_sites)))
    cursor.execute(sql, requested_sites)
    sites = pd.DataFrame(cursor.fetchall(), columns=site_cols)
    
    sql = "SELECT * FROM locations WHERE id IN ({seq})".format(seq=','.join(['?']*len(requested_dpu)))
    cursor.execute(sql, requested_dpu)
    depaul = pd.DataFrame(cursor.fetchall(), columns=site_cols)
    
    sql = "SELECT * FROM faculty WHERE id IN ({seq})".format(seq=','.join(['?']*len(requested_faculty)))
    cursor.execute(sql, requested_faculty)
    faculty = pd.DataFrame(cursor.fetchall(), columns=faculty_cols)
    print('Building the chart...', end='', flush=True)
    # Plot the map
    if not f_name:
        f_name = 'location_chart'
    today = datetime.strftime(datetime.today(), '%Y-%m-%d')
    f_name = os.path.join(os.path.sep, FL.location, '{} {}.html'.format(f_name, today))
    plot_map(cursor, student_list=students, site_list=sites, dpu_list=depaul, fac_list=faculty, outfile=f_name)
    print('complete!')
    # Close cursor, commit, and close connection
    cursor.close()
    conn.commit()
    conn.close() 

if __name__ == "__main__":
    main()
