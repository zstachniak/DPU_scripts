# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 12:46:45 2017

@author: astachn1
"""

import os
import pandas as pd
import numpy as np
import sqlite3
import urllib.parse
import json
import certifi
import urllib3
import googlemaps
import folium
from folium.plugins import HeatMap
from itertools import product
# bokeh imports
from bokeh.plotting import figure, show
from bokeh.models import HoverTool, Range1d, BoxZoomTool, PanTool, ResetTool
from bokeh.embed import file_html
from bokeh.resources import CDN
from bokeh.models.widgets import Panel, Tabs

def insert_new (cursor, table, data):	
	placeholders = ", ".join(["?" for x in range(len(data))])	
	sql = "INSERT OR REPLACE INTO %s VALUES (%s);" % (table, placeholders)	
	cursor.execute(sql, data)

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
    try:
        driving_time = gmaps.directions(loc1, loc2, mode='driving')[0]['legs'][0]['duration']['value']
    except:
        driving_time = None
    try:
        transit_time = gmaps.directions(loc1, loc2, mode='transit')[0]['legs'][0]['duration']['value']
    except:
        transit_time = None
    return driving_time, transit_time

def access_or_update_travel_time (student_id, site_id, cursor, **kwargs):
    '''Function first attempts to pull cached travel time from SQLlite
    database. Failing to find cached travel time, function will pull the
    locations of the student and site, query the Google Maps API to find
    travel time, then upload to database and return as function call.'''
    
    table = kwargs.pop('table', 'students')
    
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
        student_query = "SELECT latitude, longitude FROM %s WHERE id = %s" % (table, student_id)
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

def gather_site_travel_data (site_id, student_df, sort_by='driving_time'):
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

def gather_student_travel_data (student_id, site_df, sort_by='driving_time'):
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

def plot_map(**kwargs):
    '''A wrapper function to control overall plotting of folium map.
    
    # Icon Sets (http://fontawesome.io/icons/)
    '''
    
    # Gather optional keyword arguments
    center_map = kwargs.pop('center_map', [41.8781, -87.6298])
    student_list = kwargs.pop('student_list', None)
    site_list = kwargs.pop('site_list', None)
    dpu_list = kwargs.pop('dpu_list', None)
    fac_list = kwargs.pop('fac_list', None)
    outfile = os.path.join('W:\\csh\\Nursing Administration\\Data Management\\Location', kwargs.pop('outfile', 'index.html'))
    privacy_guard = kwargs.pop('privacy_guard', False)
    
    # Center base map (default is Chicago)
    m = folium.Map(location=center_map)
    
    if dpu_list is not None:
        # Gather icon keyword arguments
        dpu_icon = kwargs.pop('dpu_icon', 'home')
        dpu_color = kwargs.pop('dpu_color', 'blue')
        dpu_icon_color = kwargs.pop('dpu_icon_color', 'white')
        # Create feature group
        dpu = folium.map.FeatureGroup(name='DePaul Campuses')
        # Gather location and popup identifiers
        locations = zip(dpu_list['latitude'], dpu_list['longitude'])
        identifier = zip(dpu_list['name'], dpu_list['site_id'])
        # Plot markers
        for point, ident in zip(locations, identifier):
            dpu.add_child(folium.Marker(point, popup=folium.Popup('{0} ({1})'.format(*ident), parse_html=True), icon=folium.Icon(color=dpu_color, icon_color=dpu_icon_color, icon=dpu_icon)))
        # Add as child on its own layer
        m.add_child(dpu)
        
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
            temp_site_df = gather_site_travel_data(ident[1], student_list)
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
            temp_student_df = gather_student_travel_data (ident[1], site_list)
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

# Create a PoolManager that verifies certificates when making requests
http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED', ca_certs=certifi.where())

# Initialize the googlemaps Client with API key
gmaps = googlemaps.Client(key='AIzaSyCDIQVn-Kf91vMi3yXWGC_RjeejX-K20Kw')

# Open connection to database
database = 'W:\\csh\\Nursing Administration\\Data Management\\Location\\location.db'
conn = sqlite3.connect(database)
cursor = conn.cursor()

# Define standard column names (for import into Pandas)
student_cols = ['student_id', 'name', 'campus', 'start_term', 'run_term', 'program', 'subplan', 'address', 'city', 'state', 'postal', 'corrected_address', 'longitude', 'latitude']
site_cols = ['site_id', 'name', 'address', 'city', 'state', 'postal', 'type', 'corrected_address', 'longitude', 'latitude']
faculty_cols = ['id', 'name', 'address', 'city', 'state', 'postal', 'type', 'corrected_address', 'longitude', 'latitude']
travel_cols = ['student_id', 'site_id', 'driving_time', 'transit_time']

'''////////////////////////////////////////////////////////////
// Gather a list of student and site ids to plot.            //
////////////////////////////////////////////////////////////'''
# Winter 443 Students
clinical_roster = pd.read_excel('W:\\csh\\Nursing Administration\\Clinical Placement Files\\2017-2018\\Winter\\Clinical Roster.xlsx', converters={'Student ID':str})
requested_students = clinical_roster[clinical_roster['Cr'] == 443]['Student ID'].values

# All hospitals in Illinois
sql = "SELECT id FROM locations WHERE type = 'Hospital' AND state = 'IL'"
cursor.execute(sql)
requested_sites = np.ravel(cursor.fetchall())

# All DePaul Campuses
sql = "SELECT id FROM locations WHERE type = 'DePaul'"
cursor.execute(sql)
requested_dpu = np.ravel(cursor.fetchall())

# Selected Faculty
requested_faculty = np.array(['0739469', '0987298', '1188510', '0000001', '0000002'])

'''////////////////////////////////////////////////////////////
// The main plotting call.            //
////////////////////////////////////////////////////////////'''
# Update all the possible travel times
for student_id, site_id in product(requested_students, requested_sites):
    _, _ = access_or_update_travel_time(student_id, site_id, cursor, table='faculty')

# Then pull the students, sites
sql = "SELECT * FROM students WHERE id IN ({seq})".format(seq=','.join(['?']*len(requested_students)))
cursor.execute(sql, requested_students)
students = pd.DataFrame(cursor.fetchall(), columns=student_cols)

sql = "SELECT * FROM locations WHERE id IN ({seq})".format(seq=','.join(['?']*len(requested_sites)))
cursor.execute(sql, requested_sites)
sites = pd.DataFrame(cursor.fetchall(), columns=site_cols)

sql = "SELECT * FROM locations WHERE id IN ({seq})".format(seq=','.join(['?']*len(requested_dpu)))
cursor.execute(sql, requested_dpu)
dpu = pd.DataFrame(cursor.fetchall(), columns=site_cols)

sql = "SELECT * FROM faculty WHERE id IN ({seq})".format(seq=','.join(['?']*len(requested_faculty)))
cursor.execute(sql, requested_faculty)
faculty = pd.DataFrame(cursor.fetchall(), columns=faculty_cols)

# Plot the map
plot_map(student_list=students, site_list=sites, dpu_list=dpu, fac_list=faculty, outfile='443.html')

'''
# 443 Instructors
requested_sites = np.array(['74', '165', '78', '279', '75', '55', '59', '305', '330', '65'])
requested_students = np.array(['0739469', '0987298', '1188510', '0000001', '0000002'])

faculty['student_id'] = faculty['id']
faculty['program'] = 'Faculty'
faculty['campus'] = 'LPC'

plot_map(student_list=faculty, site_list=sites, outfile='443 Instructors.html')
'''

# Close cursor, commit, and close connection
cursor.close()
conn.commit()
conn.close() 
