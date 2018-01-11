# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 09:23:14 2018

@author: astachn1
"""

import pandas as pd
import os
from datetime import datetime


#############################################################
# Functions
#############################################################

def get_latest (pathname, filename, **kwargs):
    '''Scans the pathname folder for all files that match the typical 
    naming conventions of "filename". For those files, function parses 
    file name to look for a suffix to indicate date edited, then returns
    the file name with the latest date.
    
    @ Optional Keyword Arguments:
    ----------------------
    date_format: indicate a date format for files (default: '%Y-%m-%d')
    num_files: number of files to return (if more than one, in ordered list)
    ignore_suffix: an additional file suffix to ignore
    '''
    # Gather optional keyword arguments
    date_format = kwargs.pop('date_format', '%Y-%m-%d')
    num_files= kwargs.pop('num_files', 1)
    ignore_suffix = kwargs.pop('ignore_suffix', None)    
    #Set up storage dict
    file_dict = {}
    # Iterate through objects in directory
    for name in os.listdir(pathname):
        # Concatenate the path with the object name
        subname = os.path.join(pathname, name)
        #Checks for standard naming conventions and ignores file fragments
        if os.path.isfile(subname) and filename in subname and '~$' not in subname:
            # Split into name and extension
            f_name, ext = os.path.splitext(subname)
            if ignore_suffix:
                # Ignore files that end in user-identified suffix
                if f_name.endswith(ignore_suffix):
                    continue
            else:
                # Determine the expected string length of date
                date_length = len(datetime.now().strftime(date_format))                    
                # Gather expected date portion
                date_suffix = f_name[-date_length:]
                # Attempt to convert to datetime and add to file_dict
                try:
                    date_time = datetime.strptime(date_suffix, date_format).date()
                    file_dict[date_time] = subname
                except:
                    print('{} does not meet expected date convention and has been skipped.'.format(f_name))         
    # If no files found, exit function
    if len(file_dict) == 0:
        return 'No files were found'
    # Gather the most recent date in dictionary keys
    latest = max(list(file_dict.keys()))
    # If only one file is requested OR there is only one file found, return
    if num_files == 1 or len(file_dict) == 1:
        return file_dict[latest]
    else:
        file_list = [file_dict.pop(latest)]
        while len(file_list) < num_files and len(file_dict) != 0:
            # Gather the new latest file
            latest = max(list(file_dict.keys()))
            # Pop the latest file (i.e., remove from dictionary)
            file_list.append(file_dict.pop(latest))
        return file_list
    
    
    


def cat_sched (file):
    Summer = pd.read_excel(file, sheet_name='Summer', header=0,converters={'Cr':str, 'Term':str})
    Fall = pd.read_excel(file, sheet_name='Fall', header=0,converters={'Cr':str, 'Term':str})
    Winter = pd.read_excel(file, sheet_name='Winter', header=0,converters={'Cr':str, 'Term':str})
    Spring = pd.read_excel(file, sheet_name='Spring', header=0,converters={'Cr':str, 'Term':str})
    Faculty = pd.read_excel(file, sheet_name='Faculty', header=0)
    
    #Drop NaNs
    Faculty = Faculty.dropna(subset=['Name'])
    Faculty = Faculty[Faculty['Name'] != 'Null']
    
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


folders = {'reports': 'W:\\csh\\Nursing\\Clinical Placements\\Castle Branch and Health Requirements\\Reporting\\Downloaded Reports',
           'students': 'W:\\csh\\Nursing\\Student Records',
           'schedule': 'W:\\csh\\Nursing\\Schedules',
           'clinical': 'W:\\csh\\Nursing Administration\\Clinical Placement Files',
           }

# Get the latest reports
noncompliant_files = get_latest(folders['reports'], 'Noncompliant', num_files=2)
compliant_files = get_latest(folders['reports'], 'Compliant', num_files=2)

files = {'students': get_latest(folders['students'], 'Student List'),
         'faculty': 'W:\\csh\\Nursing\\Faculty\\Employee List.xlsx',
         'terms': 'W:\\csh\\Nursing\\Schedules\\Term Descriptions.xlsx',
         'noncompliant_curr': noncompliant_files[0],
         'noncompliant_prev': noncompliant_files[1],
         'compliant_curr': compliant_files[0],
         'compliant_prev': compliant_files[1],
         }




# Get the two most recent reports
noncompliant_curr = pd.read_csv(files['noncompliant_curr'], header=0)
noncompliant_prev = pd.read_csv(files['noncompliant_prev'], header=0)
# Get a change-file
nomcompliant_chng = pd.concat([noncompliant_curr, noncompliant_prev], ignore_index=True)
nomcompliant_chng.drop_duplicates(inplace=True, keep=False)




compliant_curr = pd.read_excel(files['compliant_curr'], header=0)
compliant_prev = pd.read_excel(files['compliant_prev'], header=0)


# Get the latest student list
students = pd.read_excel(files['students'], header=0, converters={'Emplid':str, 'Admit Term':str, 'Latest Term Enrl': str, 'Run Term': str,})
# Get the faculty list
faculty = pd.read_excel(files['faculty'], header=0, converters={'Empl ID': str,})
# Get term descriptions
TermDescriptions = pd.read_excel(files['terms'], header=0, converters={'Term':str})



# Need a dedicated folder where reports are dropped with constant name + date
    # get latest 


# Grab most recent clinical roster
    # Determine the term
    # also grab previous clinical roster



# Grab schedules specifically for the current and previous term
    


# Clean data
    # Drop classifications that don't matter
        # careful - DNP students could be stuck as alumni or inactive?
    # Separate into student/faculty based on account name
    # Attempt to connect to real people
        # start with student list and attempt to perfect match emails
        # for those who don't attempt match first and last name
        # for those who don't do fuzzy match on name and keep if > threshold
        # finally, consider failure
        # same with faculty, but start with secondary email, then try primary
        
# Add data
    # for students, add their clinical course and location for curr and prev
        # program, start date, advisor
        # maybe add instructor info?
    # faculty, add teaching load current and prev
    # add phone numbers
    
# Consider graphing?
    # number of 

# to highlight new data, grab the most recent report as well and do a full
    # removal of full duplicates. should be left with only the new stuff












TermDescriptions = pd.read_excel('W:\\csh\\Nursing\\Schedules\\Term Descriptions.xlsx', header=0, converters={'Term':str})

# Initial path
starting_path = 'W:\\csh\\Nursing Administration\\Clinical Placement Files'
quarters = ['Fall', 'Winter', 'Spring', 'Summer']
    
    
# Get all directories that match regex
subfolders = [f.name for f in os.scandir(starting_path) if f.is_dir() and re.match(r'\d{4}-\d{4}', f.name)]
# Sort the directories
subfolders.sort()
# Assume correct academic year is the last one
ay = subfolders[-1]
starting_path = os.path.join(starting_path, ay)
# Now search for correct quarter
subfolders = [f.name for f in os.scandir(starting_path) if f.is_dir()]
# Assume last quarter is correct one
for q in reversed(quarters):
    if q in subfolders:
        break
starting_path = os.path.join(starting_path, q)
    
# Read in latest preceptors file
file = get_latest(starting_path)
clinical_roster = pd.read_excel(file, header=0, converters={'Term':str, 'Cr':str, 'Empl ID':str, 'Hours':str})




