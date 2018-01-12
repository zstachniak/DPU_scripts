# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 12:31:05 2017

@author: astachn1
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 09:44:10 2017

@author: astachn1
"""

import os
from datetime import datetime
import pandas as pd
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

# Read in faculty on verge of being terminated
to_be_terminated = pd.read_excel('W:\\csh\\Nursing\\Schedules\\Template\\Adjunct.xlsx', header=0, converters={'Empl ID':str})
to_be_terminated['Empl ID'] = to_be_terminated['Empl ID'].str.zfill(7)

# Read in do not hire list
do_not_hire = pd.read_excel('W:\\csh\\Nursing\\Schedules\\Template\\Do Not Hire List.xlsx', header=0, converters={'ID':str})
do_not_hire['ID'] = do_not_hire['ID'].str.zfill(7)
do_not_hire.dropna(inplace=True)

#############################################################
# Main
#############################################################

# Merge with faculty to be terminated
on_verge = to_be_terminated.merge(Faculty, how='left', on='Empl ID')
# Merge with schedule
on_verge = on_verge.merge(schedule, how='left', left_on='Last-First', right_on='Faculty')
# Merge with term descriptions
on_verge = on_verge.merge(TermDescriptions, how='left', on='Term')
# Check if on do not hire list
def check_do_not_hire_list (row):
    if row['Empl ID'] in do_not_hire['ID'].unique():
        return 'X'
on_verge['On Do Not Hire List?'] = on_verge.apply(check_do_not_hire_list, axis=1)
# Keep only needed vars
on_verge = on_verge[['Empl ID', 'Last-First', 'Last Check Dt', 'On Do Not Hire List?', 'Credentials', 'Long Description', 'Program', 'Cr', 'Sec', 'Type']]
# Output to excel
on_verge.to_excel('W:\\csh\\Nursing\\Schedules\\Template\\On_Verge.xlsx', index=False)
