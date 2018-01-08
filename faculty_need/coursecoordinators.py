# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 09:44:10 2017

@author: astachn1
"""

import os
from datetime import datetime
import pandas as pd
import win32com.client as win32
import re

#############################################################
# Functions
#############################################################
def get_dir_paths (pathname, pattern, ignore_list):
    '''Given a starting pathname and a pattern, function will return
    and subdirectories that match the regex pattern.'''
    dirs = []
    # Iterate through subfiles
    for name in os.listdir(pathname):
        subname = os.path.join(pathname, name)
        # Checks if subpath is a directory and that it matches pattern
        if os.path.isdir(subname) and re.fullmatch(pattern, name) and name not in ignore_list:
            dirs.append(subname)
    return dirs

def get_latest (pathname):
    '''Scans the folder for all files that match typical naming conventions.
    For those files, function parses file name to look for date edited,
    then returns the file name with the latest date.'''
    
    #print('Scanning: {}'.format(pathname))
    
    #Set up empty lists
    files = []
    dates = []
    
    for name in os.listdir(pathname):
        subname = os.path.join(pathname, name)
        
        #Checks for standard naming conventions and ignores file fragments
        if os.path.isfile(subname) and 'Fiscal Schedule' in subname and '~$' not in subname:
            #Ignore files that end in '_(2).xlsx'
            if (subname[(subname.find('.')-3):subname.find('.')]) == '(2)':
                pass
            else:
                files.append(subname)
                
                #Parses file names and converts to datetimes
                date = subname[(subname.find('.')-10):subname.find('.')]
                date_time = datetime.strptime(date, '%m-%d-%Y').date()
                dates.append(date_time)
                #print('Adding: {}'.format(subname))
    
    #If only one file, return that one
    if len(files) == 1:
        return files[0]
    
    #If multiple files, return the one that contains the latest date
    else:
        latest = max(dates)
        #print(latest.strftime('%m-%d-%Y'))
        for file in files:
            if str(latest.strftime('%m-%d-%Y')) in file:
                return file

def cat_sched (file):
    Summer = pd.read_excel(file, sheetname='Summer', header=0,converters={'Cr':str, 'Term':str})
    Fall = pd.read_excel(file, sheetname='Fall', header=0,converters={'Cr':str, 'Term':str})
    Winter = pd.read_excel(file, sheetname='Winter', header=0,converters={'Cr':str, 'Term':str})
    Spring = pd.read_excel(file, sheetname='Spring', header=0,converters={'Cr':str, 'Term':str})
    Faculty = pd.read_excel(file, sheetname='Faculty', header=0)
    
    #Drop NaNs
    Faculty = Faculty.dropna(subset=['Name'])
    
    #Bind the quarter schedules into a single dataframe
    frames = [Summer, Fall, Winter, Spring]
    Schedule = pd.concat(frames)

    #If faculty member is full-time, mark as such
    fulltimers = Faculty['Name'].tolist()
    def FT_PT (faculty):
        if faculty == 'TBA':
            return 'TBA'
        elif faculty in fulltimers:
            return 'FT'
        else:
            return 'PT'
    Schedule['FT_PT'] = Schedule.apply(lambda x: FT_PT(x['Faculty']), axis=1)

    return Schedule, Faculty

def find_coord (row):
    if row['Type'] == 'COORD':
        coord_courses.add(row['Cr'])
        coord_faculty.add(row['Faculty'])

def find_sections (row):
    if row['Cr'] == course and row['Type'] != 'LEC' and row['Type'] != 'COORD':
        coord_programs.add(row['Program'])

def remove_files (folder):
    '''Removes all files in a given folder.'''
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

def map_contact (fac, element):
    '''Map an element of the Employee List'''
    return_item = Faculty[Faculty['Last-First'] == fac][element].astype(list).values
    try:
        return_item = return_item[0]
    except:
        return_item = ''
    return return_item

def send_email (recipient, subject, body, attachments):
    '''Sends email using the Outlook client'''
    outlook = win32.Dispatch('outlook.application')
    mail = outlook.CreateItem(0)
    mail.To = recipient
    mail.Subject = subject
    mail.Body = body
    if attachments:
        for attachment in attachments:
            mail.Attachments.Add(Source = attachment)
    mail.Send()

#############################################################
# Grab Data
#############################################################
# Set the starting path
starting_path = 'W:\\csh\\Nursing\\Schedules'
# Set the directory pattern to match against
pattern = '^[0-9]+-[0-9]+$'
# List of directories to ignore. These will not function like more recent ones
ignore_dirs = ['2011-2012', '2012-2013', '2013-2014']
# Get a list of schedule directories
dirs = get_dir_paths (starting_path, pattern, ignore_dirs)

# Initiate frames to hold full schedule
schedule_frames = []
# Populate frames
for directory in dirs:
    sched_file_path = get_latest(directory)
    schedule, faculty = cat_sched(sched_file_path)
    schedule_frames.append(schedule)
# Concatenate frames
schedule = pd.concat(schedule_frames)

# Get Employee List
Faculty = pd.read_excel('W:\\csh\\Nursing\\Faculty\\Employee List.xlsx', header=0, converters={'Cell Phone':str})

#Read in term descriptions
TermDescriptions = pd.read_excel('W:\\csh\\Nursing\\Schedules\\Term Descriptions.xlsx', header=0, converters={'Term':str})

#############################################################
# Main
#############################################################
# Ask user for term number
term = str(input('Term: '))

# Initial setup
output_path = 'W:\\csh\\Nursing\\Schedules\\Template\\Course Coordinators'
output_fields = ['Term', 'NSG', 'Cr', 'Sec', 'Time', 'Clinical Site', 'Unit', 'Faculty', 'Fac Conf', 'Primary Email', 'Secondary Email', 'Cell Phone']
contact_fields = ['Primary Email', 'Secondary Email', 'Cell Phone']

# Remove any courses in the wrong term or in DNP or RN to MS program
schedule = schedule[(schedule['Term'] == term) & (schedule['Program'].isin(['MENP', 'RFU']))]

# Create a list of all courses and faculty with a coordinator
coord_courses = set()
coord_faculty = set()

schedule.apply(find_coord, axis=1)
coord_courses = list(coord_courses)
coord_faculty = list(coord_faculty)

# Remove any courses that do not have a coordinator
schedule = schedule[schedule['Cr'].isin(coord_courses)]

# Remove all previous files
remove_files(output_path)

# Map contact information
for field in contact_fields:
    schedule[field] = schedule.apply(lambda x: map_contact(x['Faculty'], field), axis=1)

for course in coord_courses:
    coord_programs = set()
    schedule.apply(find_sections, axis=1)
    coord_programs = list(coord_programs)
    for program in coord_programs:
        # Get coordinator faculty name
        faculty = schedule[(schedule['Cr'] == course) & (schedule['Type'] == 'COORD') & (schedule['Program'] == program)]['Faculty'].astype(list).values[0]
        # Return only the labs or clns that match the program
        schedule_out = schedule[(schedule['Cr'] == course) & (schedule['Type'] != 'COORD') & (schedule['Type'] != 'LEC') & (schedule['Program'] == program)]
        # Keep only the needed columns
        schedule_out = schedule_out[output_fields]
        # Build output filename
        output_file = '{0}\\{1} - {2} - {3}.xlsx'.format(output_path, faculty, course, program)
        # Output file of lab/cln sections
        schedule_out.to_excel(output_file, index=False)
    

# Iterate through coordinators
for coordinator in coord_faculty:
    # Get coordinator's email
    try:
        recipient = Faculty[Faculty['Last-First'] == coordinator]['Primary Email'].astype(list).values[0]
    except:
        recipient = Faculty[Faculty['Last-First'] == coordinator]['Secondary Email'].astype(list).values[0]
    # Write subject
    subject = 'Course Coordinator: Your Clinical or Lab Faculty'
    # Write body
    body = '''Dear {},
    
    Attached please find a list of your clinical or lab courses along with faculty contact info. If there is no confirmation date listed for a faculty member (field = 'Fac Conf'), we do not recommend contacting that person at this time. They have likely only given us a tentative 'yes.'
    
    This is an automated email - if something looks wrong, please reply and let me know.
    
    -Zander Stachniak'''.format(coordinator.split(',')[1].strip())
    # Get attachments    
    attachments = []
    for file in os.listdir(output_path):
        file_path = os.path.join(output_path, file)
        if coordinator in file_path:
            attachments.append(file_path)
    # Send email
    try:
        send_email(recipient, subject, body, attachments)
    except Exception as e:
        print('Exception occured for {0}: {1}'.format(coordinator, e))
    