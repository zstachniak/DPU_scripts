# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 09:23:14 2018

@author: astachn1
"""

import pandas as pd
import os
from datetime import datetime
from fuzzywuzzy import fuzz, process
import re

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
    
def guess_current_term ():
    '''A function that uses the current date to estimate what the current
    academic term should be. Assumes the following cross-over points:
    1/1 - winter
    3/25 - spring
    6/15 - summer
    9/1 - fall
    '''
    # Get the current date and year
    today = datetime.today()
    current_year = today.year
    # Determine which cross-over points have been reached to assign quarter
    if today > datetime(current_year, 9, 1):
        quarter = 'Autumn'
    elif today > datetime(current_year, 6, 15):
        quarter = 'Summer'
    elif today > datetime(current_year, 3, 25):
        quarter = 'Spring'
    else:
        quarter = 'Winter'
    # Assign academic year from current year + quarter
    if quarter == 'Autumn':
        ay = f'{current_year}-{current_year + 1}'
    else:
        ay = f'{current_year - 1}-{current_year}'
    # Get current term from TermDescriptions
    term = TermDescriptions[(TermDescriptions['Academic Year'] == ay) & (TermDescriptions['Quarter'] == quarter)]['Term'].item()
    return term

def get_cln (starting_path, term):
    '''A function to gather clinical rosters from path & term.'''
    # Gather academic year and quarter
    ay = TermDescriptions[TermDescriptions['Term'] == term]['Academic Year'].item()
    q = TermDescriptions[TermDescriptions['Term'] == term]['Quarter'].item()
    # Update file path
    folder = os.path.join(starting_path, ay, q)
    file = get_latest(folder, 'Clinical Roster')
    # Read data
    clinical_roster = pd.read_excel(file, header=0, converters={'Term':str, 'Cr':str, 'Empl ID':str, 'Hours':str})
    return clinical_roster

def get_schedule (starting_path, term):
    '''A function to gather a quarterly schedule from path & term.'''
    # Gather quarter
    q = TermDescriptions[TermDescriptions['Term'] == term]['Quarter'].item()
    # If Summer, increment term to account for fiscal year changeover
    if q == 'Summer':
        term = str(int(term) + 5)
    # Gather academic year
    ay = TermDescriptions[TermDescriptions['Term'] == term]['Academic Year'].item()
    # Update file path
    folder = os.path.join(starting_path, ay)
    file = get_latest(folder, 'Fiscal Schedule', date_format='%m-%d-%Y')
    # Read data
    schedule = pd.read_excel(file, sheet_name=q, header=0,converters={'Cr':str, 'Term':str})
    return schedule




def find_best_string_match (query, choices, **kwargs):
    '''This function takes a single query and a list of possible
    choices and ranks the choices to find the most likely match.
    Rankings are calculated via fuzzywuzzy ratios, and can be
    passed directly by the user via optional keyword.'''
    # Optional argument to test only certain scorers
    scorers = kwargs.pop('scorers', [fuzz.ratio, fuzz.partial_ratio, fuzz.token_sort_ratio, fuzz.token_set_ratio])
    # Initialize a dictionary to store scoring
    score_mapping = {}
    for key in choices:
        score_mapping[key] = []
    # Test for each scorer
    for scorer in scorers:
        # Store temporary results as list of tuples
        temp_results = process.extract(query, choices, scorer=scorer, limit=None)
        # Add scores to mapping
        for (key, score) in temp_results:
            score_mapping[key].append(score)
    # Sum all results for each key
    for key in score_mapping.keys():
        score_mapping[key] = sum(score_mapping[key])
    # Determine the maximum scored
    result = max(score_mapping, key=lambda key: score_mapping[key])
    return result


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



# combine with indicator
test = pd.merge(noncompliant_curr, noncompliant_prev, on=noncompliant_curr.columns.tolist(), how='outer', indicator=True)
# Can keep both, left_only, or right_only
test = test.query("_merge == 'both'").drop('_merge', 1)












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



current_term = guess_current_term()
prev_term = str(int(current_term) - 5)




clinical_roster = get_cln (folders['clinical'], current_term)
schedule = get_schedule (folders['schedule'], current_term)


















