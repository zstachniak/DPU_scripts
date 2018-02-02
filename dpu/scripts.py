# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 15:16:16 2018

@author: astachn1
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from fuzzywuzzy import fuzz, process
from dpu.file_locator import FileLocator
from PyPDF2 import PdfFileMerger
FL = FileLocator()
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
    verbose: if True, will print statements about progress (default: False)
    '''
    # Gather optional keyword arguments
    date_format = kwargs.pop('date_format', '%Y-%m-%d')
    num_files= kwargs.pop('num_files', 1)
    ignore_suffix = kwargs.pop('ignore_suffix', None)
    verbose = kwargs.pop('verbose', False)
    #Set up storage dict
    file_dict = {}
    if verbose:
        print('Checking {}...'.format(pathname))
    # Iterate through objects in directory
    for name in os.listdir(pathname):
        # Concatenate the path with the object name
        subname = os.path.join(pathname, name)
        # Split into name and extension
        f_name, ext = os.path.splitext(subname)
        # Gather basename
        basename = os.path.basename(subname)
        if verbose:
            print('{}...'.format(basename), end='')
        #Checks for standard naming conventions and ignores file fragments
        if os.path.isfile(subname) and filename in basename and '~$' not in subname:
            if ignore_suffix:
                # Ignore files that end in user-identified suffix
                if f_name.endswith(ignore_suffix):
                    print('ignoring suffix.')
                    continue
            # Determine the expected string length of date
            date_length = len(datetime.now().strftime(date_format))                    
            # Gather expected date portion
            date_suffix = f_name[-date_length:]
            # Attempt to convert to datetime and add to file_dict
            try:
                date_time = datetime.strptime(date_suffix, date_format).date()
                file_dict[date_time] = subname
                if verbose:
                    print('type match.')
            except:
                if verbose:
                    print('does not meet expected date convention and has been skipped.')
        else:
            if verbose:
                print('does not meet file requirements')
    # If no files found, exit function
    if len(file_dict) == 0:
        if verbose:
            print('No files were found')
        return
    # Gather the most recent date in dictionary keys
    latest = max(list(file_dict.keys()))
    # If only one file is requested OR there is only one file found, return
    if num_files == 1 or len(file_dict) == 1:
        if verbose:
            print('Success! One file found!')
        return file_dict[latest]
    else:
        file_list = [file_dict.pop(latest)]
        while len(file_list) < num_files and len(file_dict) != 0:
            # Gather the new latest file
            latest = max(list(file_dict.keys()))
            # Pop the latest file (i.e., remove from dictionary)
            file_list.append(file_dict.pop(latest))
        if verbose:
            print('Success! Multiple files found!')
        return file_list

def get_term_descriptions ():
    '''A function that returns a Pandas dataframe of all terms.'''
    TermDescriptions = pd.read_excel(os.path.join(FL.schedule, 'Term Descriptions.xlsx'), header=0, converters={'Term':str})
    return TermDescriptions

def guess_current_term (TermDescriptions):
    '''A function that uses the current date to estimate what the current
    academic term should be. Assumes the following cross-over points:
    12/1 - winter
    3/25 - spring
    6/10 - summer
    9/1 - fall
    '''
    # Get the current date and year
    today = datetime.today()
    current_year = today.year
    # Determine which cross-over points have been reached to assign quarter
    # For end of year dates, we increment year by one to reach correct ay
    if today > datetime(current_year, 12, 1):
        quarter = 'Winter'
        current_year += 1
    elif today > datetime(current_year, 9, 1):
        quarter = 'Autumn'
        current_year += 1
    elif today > datetime(current_year, 6, 10):
        quarter = 'Summer'
    elif today > datetime(current_year, 3, 25):
        quarter = 'Spring'
    else:
        quarter = 'Winter'
    # Assign academic year from current year + quarter
    ay = f'{current_year - 1}-{current_year}'
    # Get current term from TermDescriptions
    term = TermDescriptions[(TermDescriptions['Academic Year'] == ay) & (TermDescriptions['Quarter'] == quarter)]['Term'].item()
    return term

def get_cln (term, TermDescriptions):
    '''A function to gather clinical rosters from path & term.'''
    # Starting Path
    starting_path = FL.cln_roster
    # Gather academic year and quarter
    ay = TermDescriptions[TermDescriptions['Term'] == term]['Academic Year'].item()
    q = TermDescriptions[TermDescriptions['Term'] == term]['Quarter'].item()
    # Update file path
    folder = os.path.join(starting_path, ay, q)
    file = get_latest(folder, 'Clinical Roster')
    # Read data
    clinical_roster = pd.read_excel(file, header=0, converters={'Term':str, 'Cr':str, 'Student ID':str})
    return clinical_roster

def get_dir_of_schedule (term, TermDescriptions):
    '''A function to output a full directory path to a schedule.'''
    # Define schedule starting path
    starting_path = FL.schedule
    # Gather quarter
    q = TermDescriptions[TermDescriptions['Term'] == term]['Quarter'].item()
    # If Summer, increment term to account for fiscal year changeover
    if q == 'Summer':
        term = str(int(term) + 5)
    # Gather academic year
    ay = TermDescriptions[TermDescriptions['Term'] == term]['Academic Year'].item()
    # Update file path
    folder = os.path.join(starting_path, ay)
    return folder

def get_schedule (term, TermDescriptions):
    '''A function to gather a quarterly schedule from path & term.'''
    # Define schedule starting path
    starting_path = FL.schedule
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

def get_student_roster ():
    # Define schedule starting path
    starting_path = FL.rosters
    # Find most recent
    file = get_latest(starting_path, 'NSG_STDNT_ROSTER')
    # Read data
    rosters = pd.read_excel(file, skiprows=[0], header=0, converters={'Term':str, 'Student ID':str, 'Catalog':str, 'Section':str, 'Class Nbr':str, 'Faculty_ID':str})
    # Remove extra space in course numbers
    rosters['Catalog'] = rosters['Catalog'].apply(lambda x: x.strip())
    rosters.rename(columns={'Catalog':'Cr', 'Section':'Sec'}, inplace=True)
    return rosters

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

def build_File (pathname, tempdict, file_format='csv'):
    '''Iterates recursively through a folder and combines files into a
    dictionary pair of filename (key) and dataframe (value). Works with
    csv and excel formats.'''
    # Iterate through objects in pathname folder
    #print('Scanning: {}'.format(pathname))
    for name in os.listdir(pathname):
        subname = os.path.join(pathname, name)
        # If object in folder is a file
        if os.path.isfile(subname):
            #print('Adding: {}'.format(subname))
            # Path naming
            base = os.path.basename(subname)
            dfname = os.path.splitext(base)[0]
            # Read file using user-defined format
            if file_format == 'csv':
                tempdict[dfname] = pd.read_csv(subname,delimiter=',',na_values='nan',)
            elif file_format == 'excel':
                tempdict[dfname] = pd.read_excel(subname,skiprows=0,header=1,na_values='nan')
        # If object in folder is a folder, recursive call to function
        elif os.path.isdir(subname):
            build_File(subname, tempdict)
        else:
            pass
    return tempdict

def archive_old_reports (report_fullpath, report_basename, archive_folder_name, **kwargs):
    '''Function to move all old reports in a folder into a separate
    archive folder.'''
    keep_min = kwargs.pop('keep_min', 0)
    # Gather all files that match basename within the path
    all_files = get_latest(report_fullpath, report_basename, num_files=float('inf'))
    # If no old reports are found, exit function
    if not all_files:
        return
    # If a single file, coerce to list
    if type(all_files) == str:
        all_files = [all_files]
    # Test if destination exists; if not, make folder
    destination = os.path.join(report_fullpath, archive_folder_name)
    if not os.path.isdir(destination):
        os.makedirs(destination)
    # While there remain more than keep_min number of reports
    while len(all_files) > keep_min:
        # Pop oldest first
        old_path = all_files.pop()
        # Create new path name
        file_name = os.path.basename(old_path)
        new_path = os.path.join(destination, file_name)
        # If new path already exists, delete it
        if os.path.exists(new_path):
            os.remove(new_path)
        # Rename (i.e., move) old reports
        os.rename(old_path, new_path)

def recursive_char (i, **kwargs):
    '''A function that takes an integer value and returns a list of 
    integers representing a letter character in unicode.'''
    upper = kwargs.get('upper', True)
    if upper:
        char_offset = ord('A')
    else:
        char_offset = ord('a')
    # Initialize list for sequential storage
    int_list = []
    # Take floor of integer
    floor = i // 26
    # Base case: floor == 0
    if floor == 0:
        # Append int with the char_offset
        int_list.append((i % 26) + char_offset)
    # Recursive case
    else:
        # Append first digit with char_offset
        # We subtract 1 due to indexing
        int_list.append((i // 26) + char_offset - 1)
        # Subtract out first digit
        i -= (floor * 26)
        # Recursive step
        int_list += recursive_char(i, **kwargs)
    return int_list

def char_counter_from_int (i, **kwargs):
    '''Wrapper function around recursive char that takes an integer 
    value and returns a character representation. Starts with 0 
    represented by A. At 27, will become AA, then AB, etc. Implements
    recursion to allow for any length of integer.
    
    Optional Keyword Arguments:
    upper: True will return uppercase, False lowercase (default:True)
    '''
    int_list = recursive_char(i, **kwargs)
    char_list = [chr(x) for x in int_list]
    return ''.join(char_list)

def true_date (date, day_of_week, date_is_max=False, **kwargs):
    '''Function is designed to return a "true" date when user passes
    a first possible date and a day of the week pattern. For example,
    although 1/2/18 is the first possible meeting date, a Monday pattern
    course represented by the user as "0" would not actually meet until
    1/8/18. Function can work backwards from a max date if user supplies 
    that argument. In addition, user can request that a range of dates be
    passed through keyword arguments.'''
    # Gather optional keyword arguments
    return_range = kwargs.pop('return_range', False)
    num_recurrence = kwargs.pop('num_recurrence', 10)
    skip_weeks = kwargs.pop('skip_weeks', 0)
    # Account for last day as ending week
    num_recurrence -= 1
    # Apply skip weeks
    if skip_weeks:
        date += timedelta(days=(7 * skip_weeks))
    if day_of_week is not None:
        # Take modulus to determine offset delta
        delta = (day_of_week - date.weekday()) % 7
    else:
        # If no identifiable day of the week, return date/range unchanged
        delta = 0
    # If the date is a max, need a negative offset delta
    if date_is_max:
        delta = ((delta - 7) % 7) * -1
        if return_range:
            num_recurrence *= -1
    # If a range is requested
    if return_range:
        # Multiply delta by occurences for the range_delta
        range_delta = delta + (num_recurrence * 7)
        # Apply both timedeltas within a list
        range_dates = [date + timedelta(days=x) for x in [delta, range_delta]]
        # Sort list so that range is in proper order
        range_dates.sort()
        return range_dates
    # If no range requested
    else:
        # Return date with timedelta applied
        return date + timedelta(days=delta)

def check_if_dir_is_empty (folder_path):
    '''A simple check to see if a directory is empty. Returns T/F'''
    # Walk through directory
    for root, dirs, files in os.walk(folder_path):
        # If files or directories exist, return False
        if files or dirs:
            return False
        # Else, return True
        else:
            return True

def empty_dir(folder_path):
    '''Delete all contents of a directory. Based on https://stackoverflow.com/questions/185936/how-to-delete-the-contents-of-a-folder-in-python.'''
    # Make sure you don't make a big mistake
    if (folder_path == '/' or folder_path == "\\"):
        return
    else:
        # Walk through directory
        for root, dirs, files in os.walk(folder_path, topdown=False):
            # Remove all files first
            for name in files:
                os.remove(os.path.join(root, name))
            # Then remove and directories
            for name in dirs:
                os.rmdir(os.path.join(root, name))

def ensure_empty_dir (folder_path):
    '''Wrapper function to ensure that a directory exists and is empty.'''
    # Check if staging dir exists
    if os.path.isdir(folder_path):
        # If exists, check that staging dir is empty
        if not check_if_dir_is_empty(folder_path):
            # If not empty, empty it
            empty_dir(folder_path)
    # Make directory if doesn't exist
    else:
        os.mkdir(folder_path)

def merge_PDFs (files, output_path):
    '''Merges order list of PDF file names into a single file.'''
    # Initialize PdfFileMerger class
    merger = PdfFileMerger()
    # Cycle through and add PDFs to merger
    for file in files:
        merger.append(file)
    # Write as single PDF
    merger.write(output_path)
    merger.close()