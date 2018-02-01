# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 09:23:14 2018

@author: astachn1
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta, time
from fuzzywuzzy import fuzz, process
import dpu.scripts as dpu
from dpu.file_locator import FileLocator
import click

def separate_cln_dupes (df, groupby='Student ID'):
    '''Separates out clinicals when one student has two courses.'''
    # Sort by course number
    df.sort_values(by='Cr', inplace=True)
    # Gather list of duplicates (i.e., student has more than one clinical)
    dupe_truth_value = df.duplicated(subset=[groupby], keep='first')
    dupes = df.loc[dupe_truth_value].copy(deep=True)
    dupes.drop(labels=['Term'], axis=1, inplace=True)
    # Drop duplicates from original list
    df.drop_duplicates(subset=[groupby], keep='first', inplace=True)
    # Merge two lists 
    df = pd.merge(df, dupes, how='left', on=groupby, suffixes=['_1', '_2'])
    return df

def apply_dates_times (row, internships, roster_type='student'):
    '''Pandas apply function that returns clinical date range and
    time range.'''
    # If internship
    if row['Cr'] == '443' and roster_type == 'student':
        # Gather dates from internship document
        dates = internships[internships['Student ID'] == row['Student ID']]['Date'].item().split('-')
        # Process and format
        dates = [datetime.strptime(x.strip(), '%m/%d/%y') for x in dates]
        dates = '{} - {}'.format(*map(date_format, dates))
        times = None
    # For all others
    else:
        # Gather Meeting Pattern
        pat = row['Pat']
        # If NSG 301
        if row['Cr'] == '301':
            first, last = true_date(row['Start Date'], pat, return_range=True, num_recurrence=4, skip_weeks=6)
        else:
            # Gather Dates
            first, last = true_date(row['Start Date'], pat, return_range=True)
        # Format dates
        dates = '{} - {}'.format(*map(date_format, [first, last]))
        # If pattern is Null (i.e., no course time or pattern)
        if pd.isnull(pat) or pat == 'BYAR':
            times = None
        else:
            # Gather Times
            start = datetime.time(row['Mtg Start'])
            end = datetime.time(row['Mtg End'])
            times = '{} {} - {}'.format(day_of_week_abbr[pat], *map(time_format, [start, end]))
    # Return string-formated, concatenated dates and times
    return pd.Series([dates, times])

# STORAGE of days_of_week hash
day_of_week_hash = {'MON': 0,
                    'TUE': 1,
                    'WED': 2,
                    'THUR': 3,
                    'F': 4,
                    'SAT': 5,}
day_of_week_abbr = {'MON': 'Mo',
                    'TUE': 'Tu',
                    'WED': 'We',
                    'THUR': 'Th',
                    'F': 'Fr',
                    'SAT': 'Sa',}

# Date and Time format lambda functions
date_format = lambda x: datetime.strftime(x, '%m/%d/%Y')
time_format = lambda x: time.strftime(x, '%I:%M%p')

def true_date (date, pattern, date_is_max=False, **kwargs):
    '''Function is designed to return a "true" date when user passes
    a first possible date and a day of the week pattern. For example,
    although 1/2/18 is the first possible meeting date, a MON pattern
    course would not actually meet until 1/8/18. Function can work
    backwards from a max date if user supplies that argument. In
    addition, user can request that a range of dates be passed through
    keyword arguments.'''
    # Gather optional keyword arguments
    return_range = kwargs.pop('return_range', False)
    num_recurrence = kwargs.pop('num_recurrence', 10)
    skip_weeks = kwargs.pop('skip_weeks', 0)
    # Account for last day as ending week
    num_recurrence -= 1
    # Apply skip weeks
    if skip_weeks:
        date += timedelta(days=(7 * skip_weeks))
    # Gather day of week as integer
    day_of_week = day_of_week_hash.get(pattern, None)
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

def clean_student_roster (current_term, internships):
    '''Simple function to load and clean the student roster.'''
    # Download student rosters
    roster = dpu.get_student_roster()
    # Clinical course types
    cln_types = ['CLN', 'PRA']
    # Range of Anesthesia courses that we simply don't care about
    ignore_courses = [str(x) for x in range(500, 517)] + [str(x) for x in range(600, 617)]
    # Filter rosters
    roster = roster[((roster['Type'].isin(cln_types)) | ((roster['Cr'] == '301') & (roster['Type'] == 'LAB'))) & (~roster['Cr'].isin(ignore_courses)) & (roster['Role'] != 'SI')].copy(deep=True)
    # Drop unneeded fields
    roster.drop(labels=['Class Nbr', 'Role', 'Mode', 'Type', 'Subject', 'Student Name', 'Student Major'], axis=1, inplace=True)
    # Drop any true duplicates
    roster.drop_duplicates(inplace=True)
    # Sort by faculty and drop duplicates (excluding faculty)
    # Takes care of 301 issue (due to changing course times)
    roster = roster.sort_values(by=['Faculty_ID', 'Start Date']).drop_duplicates(subset=['Term', 'Student ID', 'Cr', 'Sec']).copy(deep=True)
    # Figure out next term value
    next_term = str(int(current_term) + 5)
    # Gather string-formatted and concatenated dates and times
    roster[['Dates', 'Times']] = roster.apply(apply_dates_times, axis=1, args=(internships,))
    # Drop more unneeded fields
    roster.drop(labels=['Start Date', 'End Date', 'Pat', 'Mtg Start', 'Mtg End'], axis=1, inplace=True)
    # Gather roster for next term
    roster_next_term = roster[roster['Term'] == next_term].copy(deep=True)
    roster_next_term = separate_cln_dupes(roster_next_term)
    roster_next_term['Cln Next Term?'] = True
    roster_next_term.drop(labels=['Term'], axis=1, inplace=True)
    # Drop next term from current roster
    roster = roster[roster['Term'] == current_term].copy(deep=True)
    roster = separate_cln_dupes(roster)
    roster['Cln This Term?'] = True
    # Perform outer join so that we retain students with cln next term
    roster = pd.merge(roster, roster_next_term, how='outer', on='Student ID', suffixes=['_curr', '_next'])
    # Put student ID at start
    roster.insert(0, 'Emplid', roster['Student ID'])
    roster.drop(labels=['Student ID'], axis=1, inplace=True)
    # Fill NaNs
    roster['Cln Next Term?'].fillna(value=False, inplace=True)
    return roster

def contains_CLN (row):
    '''Simple function to return if any clinical or lab exists in groupby'''
    return any([x in row['Type'].unique() for x in ['CLN', 'PRA', 'LAB']])

def get_cln_site (row):
    '''Concatenate all clinical sites (if exist) in groupby'''
    # Convert courses and sections to list
    cr = row['Cr'].tolist()
    sec = row['Sec'].tolist()
    sites = []
    # Get site from global(schedule)
    for c, s in zip(cr, sec):
        res = schedule[(schedule['Cr'] == c) & (schedule['Sec'] == s)]
        # Ignore Nulls and NaNs
        if len(res) == 1:
            if type(res['Clinical Site'].item()) == str:
                sites.append(res['Clinical Site'].item())
    # Return concatenated list of unique sites
    return ', '.join(np.unique(sites))

def cr_sec (row):
    '''Concatenate courses and sections in groupby'''
    # Convert courses and sections to list
    cr = row['Cr'].tolist()
    sec = row['Sec'].tolist()
    # Return list of combined Cr-Sec
    return ', '.join(['-'.join([c, s]) for c, s in zip(cr, sec)])

def group_faculty (df, term='This'):
    '''Takes list of non-unique faculty teaching assignments and converts
    to unique through series of groupby operations.'''
    # Reindex
    df = df.set_index(['Faculty_ID'])
    # Teaching a clinical this term?
    df['Cln ' + term + ' Term?'] = df.groupby(by='Faculty_ID').apply(contains_CLN)
    # All Cr-Sec pairs
    df['Courses'] = df.groupby(by='Faculty_ID').apply(cr_sec)
    # All clinical sites will be attending
    df['Cln Sites'] = df.groupby(by='Faculty_ID').apply(get_cln_site)
    # Drop unneeded and reset index
    df.drop(labels=['Term', 'Cr', 'Sec', 'Type', 'Class Nbr'], axis=1, inplace=True)
    df.reset_index(inplace=True)
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    return df

def clean_faculty_roster (current_term):
    '''Simple function to load and clean the faculty roster.'''
    # Download student rosters
    roster = dpu.get_student_roster()
    # Range of Anesthesia courses that we simply don't care about
    ignore_courses = [str(x) for x in range(500, 517)] + [str(x) for x in range(600, 617)]
    # Filter rosters
    roster = roster[(~roster['Cr'].isin(ignore_courses)) & (roster['Role'] != 'SI')].copy(deep=True)
    # Drop unneeded fields
    roster.drop(labels=['Student ID', 'Student Name', 'Student Major', 'Subject', 'Role', 'Mode'], axis=1, inplace=True)
    # Drop any true duplicates
    roster.drop_duplicates(inplace=True)
    # Sort by faculty and drop duplicates (excluding faculty)
    # Takes care of 301 issue (due to changing course times)
    roster = roster.sort_values(by='Faculty_ID').drop_duplicates(subset=['Term', 'Cr', 'Sec', 'Type']).copy(deep=True)
    #
    # NOTE: NaN faculty would still exist here
    #
    # Figure out next term value
    next_term = str(int(current_term) + 5)
    # Gather string-formatted and concatenated dates and times
    #roster[['Dates', 'Times']] = roster.apply(apply_dates_times, axis=1, args=(internships,), roster_type='faculty')
    # Drop more unneeded fields
    roster.drop(labels=['Start Date', 'End Date', 'Pat', 'Mtg Start', 'Mtg End'], axis=1, inplace=True)
    # Gather roster for next term
    roster_next_term = roster[roster['Term'] == next_term].copy(deep=True)
    # Drop next term from current roster
    roster = roster[roster['Term'] == current_term].copy(deep=True)
    # Groupby faculty
    roster = group_faculty(roster)
    # If courses exist for next term, groupby the faculty
    if len(roster_next_term) > 0:
        roster_next_term = group_faculty(roster_next_term, term='Next')
    # Else, create a blank df
    else:
        roster_next_term = roster[roster['Faculty_ID'] == 0]
        roster_next_term = roster_next_term.rename(columns={'Cln This Term?': 'Cln Next Term?'}).copy(deep=True)
    # Perform outer join so that we retain faculty with cln next term
    roster = pd.merge(roster, roster_next_term, how='outer', on='Faculty_ID', suffixes=['_curr', '_next'])
    # Fill NaNs
    roster['Cln Next Term?'].fillna(value=False, inplace=True)
    # Rename ID field for ease
    roster = roster.rename(columns={'Faculty_ID': 'Emplid'}).copy(deep=True)
    return roster

def get_historical_student_data (hist_path, students):
    # Get a good handful of most recent files
    stud_files = dpu.get_latest(hist_path, 'Student List', num_files=16)
    # Iterate through files and concatenate all together
    for i, file in enumerate(stud_files):
        if i == 0:
            historical_students = pd.read_excel(file, header=0, converters={'Emplid':str, 'Admit Term':str, 'Latest Term Enrl': str, 'Run Term': str,})
        else:
            historical_students = pd.concat([historical_students, pd.read_excel(file, header=0, converters={'Emplid':str, 'Admit Term':str, 'Latest Term Enrl': str, 'Run Term': str,})], ignore_index=True)
    # Keep only necessary data
    historical_students = historical_students[['Emplid', 'Last Name', 'First Name', 'Email']]
    # Drop all true duplicates
    historical_students.drop_duplicates(inplace=True)
    # Merge with students
    historical_students = historical_students.merge(students, on='Emplid', how='left', indicator=True, suffixes=['', 'y'])
    # Remove any exact matches to current students (this does not help us)
    historical_students = historical_students[historical_students['_merge'] == 'left_only']
    # Keep only necessary columns
    historical_students = historical_students.iloc[:,0:4]
    # Reset index
    historical_students.reset_index(drop=True, inplace=True)
    return historical_students

def read_cb (file):
    '''Function to load Castle Branch report'''
    to_datetime = lambda d: datetime.strptime(d, '%m/%d/%Y')
    temp = pd.read_csv(file, header=0, converters={'Order Submission Date': to_datetime})
    temp.rename(columns={'Email Address':'Email'}, inplace=True)
    return temp

def instructor_contact_info (row, suffix, faculty, req_list, schedule):
    '''Function to return pd.Series element of requested values.'''
    # Get ID data
    row_field = 'Faculty_ID' + '_' + suffix
    x = row[row_field]
    # Attempt exact ID comparison
    res = faculty[faculty['Empl ID'] == x]
    # If results are found, return those results
    if len(res) == 1:
        return pd.Series([res[x].item() for x in req_list])
    else:
        # Assume faculty is unhired. Attempt to get faculty name from
        # schedule and do exact match on name.
        # Prepare search terms
        cr_x = 'Cr' + '_' + suffix
        sec_x = 'Sec' + '_' + suffix
        # Attempt to gather match
        res = schedule[(schedule['Cr'] == row[cr_x]) & (schedule['Sec'] == row[sec_x])]
        # If results are found, gather name
        if len(res) == 1:
            row_name = res['Faculty'].item()
            # Attempt exact name comparison
            res = faculty[faculty['Last-First'] == row_name]
            # If results are found, return those results
            if len(res) == 1:
                return pd.Series([res[x].item() for x in req_list])
            else:
                # Else, attempt to find the most closely matched name
                # Gather all faculty names into a list
                all_faculty_names = faculty['Last-First'].unique()
                # Attempt to find the most closely matched name
                best_guess = dpu.find_best_string_match(row_name, all_faculty_names)
                # Attempt exact name comparison with the best_guess
                res = faculty[faculty['Last-First'] == best_guess]
                # If results are found, return those results
                if len(res) == 1:
                    return pd.Series([res[x].item() for x in req_list])
                else:
                    return pd.Series([None for x in req_list])
        else:
            return pd.Series([None for x in req_list])

def clinical_info (row, suffix, req_list, schedule, internships):
    '''Function to return pd.Series element of requested values.'''
    # Prepare search terms
    cr_x = 'Cr' + '_' + suffix
    sec_x = 'Sec' + '_' + suffix
    
    if row[cr_x] == '443':
        # Attempt comparison on Emplid
        res = internships[internships['Student ID'] == row['Emplid']]
    else:
        # Attempt comparison on Cr-Sec
        res = schedule[(schedule['Cr'] == row[cr_x]) & (schedule['Sec'] == row[sec_x])]
    # If results are found, return those results
    if len(res) == 1:
        return pd.Series([res[x].item() for x in req_list])
    else:
        return pd.Series([None for x in req_list])  

def true_match (row, df, left_fields, **kwargs):
    '''Function to handle true matches. If only left_fields are passed
    function will assume field names match. If right fields are passed
    function will process as differently named.'''
    right_fields = kwargs.pop('right_fields', None)
    if right_fields:
        for left, right in zip(left_fields, right_fields):
            df = df[df[left].apply(lambda x: x.lower().strip()) == row[right].lower().strip()]
    else:
        for field in left_fields:
            df = df[df[field].apply(lambda x: x.lower().strip()) == row[field].lower().strip()]
    return df

def partial_name_match (query, possibilities):
    '''If names are set matches, return other name'''
    temp = process.extract(query, possibilities, scorer=fuzz.token_set_ratio)
    if temp[0][1] == 100:
        return temp[0][0]
    else:
        return query

def gather_output (result):
    '''Standardize gathering data'''
    # If more than one result, simply take the first
    res = result.iloc[0]
    # Return required fields
    return res['Last Name'], res['First Name'], res['Email']

def match_students (row, df, output_dict, **kwargs):
    '''Match student IDs with their CB lookup values.'''
    # Gather optional keyword arguments
    historic = kwargs.pop('historic', None)
    # Save student ID
    student = row['Emplid']
    # Check to see that student has not already been matched
    if student not in output_dict.keys():
        # Attempt a true match on email address
        res = true_match(row, df, ['Email'])
        if len(res) >= 1:
            # If successful, add to dictionary
            output_dict[student] = gather_output(res)
            return
        else:
            # Else, attempt a true match on Last, First
            res = true_match(row, df, ['Last Name', 'First Name'])
            if len(res) >= 1:
                # If successful, add to dictionary
                output_dict[student] = gather_output(res)
                return
            else:
                # If user requests, test against possible historic data
                if historic is not None:
                    # Search for previous (i.e., different) matches by ID
                    previous = true_match(row, historic, ['Emplid'])
                    if len(previous) > 0:
                        # If a match exists, try out each match
                        for idx in range(len(previous)):
                            prev = previous.iloc[idx]
                            # Attempt a true match on email address
                            res = true_match(prev, df, ['Email'])
                            if len(res) >= 1:
                                # If successful, add to dictionary
                                output_dict[student] = gather_output(res)
                                return
                            # Else, attempt a true match on Last, First
                            else:
                                res = true_match(prev, df, ['Last Name', 'First Name'])
                                if len(res) >= 1:
                                    # If successful, add to dictionary
                                    output_dict[student] = gather_output(res)
                                    return
                
                # Finally, try a match using partial first name
                # Should help with issues of middle name
                res = true_match(row, df, ['Last Name'])
                if len(res) >= 1:
                    # Gather partial name match
                    partial = partial_name_match(row['First Name'].lower().strip(), res['First Name'].apply(lambda x: x.lower().strip()).tolist())
                    res = res[res['First Name'].apply(lambda x: x.lower().strip()) == partial]
                    # If successful, add to dictionary
                    if len(res) >= 1:
                        output_dict[student] = gather_output(res)
                        return
                
                # If still unsuccessful, give up
                else:
                    return

def match_faculty (row, df, output_dict, **kwargs):
    '''Match student IDs with their CB lookup values.'''
    # Gather optional keyword arguments
    historic = kwargs.pop('historic', None)
    # Save faculty ID
    faculty = row['Emplid']
    # Check to see that faculty has not already been matched
    if faculty not in output_dict.keys():
        # Attempt a true match on primary email address
        if not pd.isnull(row['Primary Email']):
            res = true_match(row, df, ['Email'], right_fields=['Primary Email'])
        else:
            res = []
        if len(res) >= 1:
            # If successful, add to dictionary
            output_dict[faculty] = gather_output(res)
            return
        else:
            # Else, attempt a true match on secondary email address
            if not pd.isnull(row['Secondary Email']):
                res = true_match(row, df, ['Email'], right_fields=['Secondary Email'])
            else:
                res = []
            if len(res) >= 1:
                # If successful, add to dictionary
                output_dict[faculty] = gather_output(res)
                return
            else:
                # Else, attempt a true match on Last, First
                res = true_match(row, df, ['Last Name', 'First Name'])
                if len(res) >= 1:
                    # If successful, add to dictionary
                    output_dict[faculty] = gather_output(res)
                    return
                else:

                    # If user requests, test against possible historic data
                    if historic is not None:
                        # Search for previous (i.e., different) matches by ID
                        previous = true_match(row, historic, ['Emplid'])
                        if len(previous) > 0:
                            # If a match exists, try out each match
                            for idx in range(len(previous)):
                                prev = previous.iloc[idx]
                                # Attempt a true match on email address
                                res = true_match(prev, df, ['Email'])
                                if len(res) >= 1:
                                    # If successful, add to dictionary
                                    output_dict[faculty] = gather_output(res)
                                    return
                                # Else, attempt a true match on Last, First
                                else:
                                    res = true_match(prev, df, ['Last Name', 'First Name'])
                                    if len(res) >= 1:
                                        # If successful, add to dictionary
                                        output_dict[faculty] = gather_output(res)
                                        return
                
                    # Finally, try a match using partial first name
                    # Should help with issues of middle name
                    res = true_match(row, df, ['Last Name'])
                    if len(res) >= 1:
                        # Gather partial name match
                        partial = partial_name_match(row['First Name'].lower().strip(), res['First Name'].apply(lambda x: x.lower().strip()).tolist())
                        res = res[res['First Name'].apply(lambda x: x.lower().strip()) == partial]
                        # If successful, add to dictionary
                        if len(res) >= 1:
                            output_dict[faculty] = gather_output(res)
                            return
                    
                    # If still unsuccessful, give up
                    else:
                        return

def check_compliance (person, df, df_changelog, df_next_action, test='noncompliant'):
    '''A wrapper function to check for compliance for a single person.
    Returns status along with incomplete and next due items.'''
    # Set various status outcomes based on test type
    if test == 'noncompliant':
        status_result = 'No'
    elif test == 'compliant':
        status_result = 'Yes'
        reqs_incomplete = None
    elif test == 'dna_compliant':
        status_result = 'No'
        reqs_incomplete = 'Has met D&A but not registered for health req tracker'
    elif test == 'dna_noncompliant':
        status_result = 'No'
        reqs_incomplete = 'Has not registered for an account'
        
    # Attempt true match between person and df
    match = true_match(person, df, ['Email'])
    # If a match exists...
    if len(match) >= 1:
        # If more than one result...
        if len(match) > 1:
            # Noncompliant ONLY
            if test == 'noncompliant':
                # Concatenate all incomplete requirements
                reqs_incomplete = ', '.join(match['Requirements Incomplete'].tolist())
            # Check if this is an update from last time
            changed = all([(match.values[i] == df_changelog.values).all(1).any() for i in range(len(match.values))])
        elif len(match) == 1:
            # Noncompliant ONLY
            if test == 'noncompliant':
                # Gather all incomplete requirements
                reqs_incomplete = match['Requirements Incomplete'].item()
            # Check if this is an update from last time
            changed = (match.values == df_changelog.values).all(1).any()
        # Check next due
        next_due = true_match(person, df_next_action, ['Email'])
        # If a match exists...
        if len(next_due) > 1:        
            # Convert due dates to datetime objects
            next_due.loc[:, 'Requirement Due Date'] = next_due['Requirement Due Date'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y'))
            #next_due['Requirement Due Date'] = next_due['Requirement Due Date'].copy(deep=True).apply(lambda x: datetime.strptime(x, '%m/%d/%Y'))
            # Get the nearest due date
            next_due_date = next_due['Requirement Due Date'].min()
            # Keep only nearest requirement
            next_due = next_due[next_due['Requirement Due Date'] == next_due_date]
            # If more than one result...
            if len(next_due) > 1:
                # Concatenate the results
                next_due_req = ', '.join(next_due['Requirement Name'].tolist())
            else:
                # Else, take single result
                next_due_req = next_due['Requirement Name'].item()
        elif len(next_due) == 1:
            # Convert due date to datetime object
            next_due_date = datetime.strptime(next_due['Requirement Due Date'].item(), '%m/%d/%Y')
            # Take single result
            next_due_req = next_due['Requirement Name'].item()
        else:
            # If no next due items, return None
            next_due_req = None
            next_due_date = None
        # Return required fields
        return pd.Series([changed, status_result, reqs_incomplete, next_due_req, next_due_date])
    # If no results, return empty Series
    else:
        return pd.Series()

def determine_status (row, id_dict, noncompliant_df, compliant_df, noncompliant_changelog, compliant_changelog, tracker_1, tracker_2, next_action_date, account):
    '''Gather CB compliance stats'''
    # Get ID
    key = id_dict.get(row['Emplid'], None)
    # If ID doesn't exist, can't identify student
    if not key:
        # If unable to identify student
        return pd.Series([None, 'No', ['Could Not Identify CB Account'], None, None])
    # Store connection key
    conn = pd.Series(data={'Last Name':key[0], 'First Name':key[1], 'Email':key[2]})
    
    # Check for noncompliance in tracker_1
    result = check_compliance(conn, noncompliant_df[noncompliant_df['To-Do List Name'].isin(tracker_1)].copy(deep=True), noncompliant_changelog, next_action_date, test='noncompliant')
    if len(result) != 0:
        return result
    # Check for compliance in tracker_1
    result = check_compliance(conn, compliant_df[compliant_df['To-Do List Name'].isin(tracker_1)].copy(deep=True), compliant_changelog, next_action_date, test='compliant')
    if len(result) != 0:
        return result
    
    if account == 'faculty':
        # Check for non-compliance in tracker_2
        result = check_compliance(conn, noncompliant_df[noncompliant_df['To-Do List Name'].isin(tracker_2)].copy(deep=True), noncompliant_changelog, next_action_date, test='noncompliant')
        if len(result) != 0:
            return result
        # Check for compliance in tracker_2
        result = check_compliance(conn, compliant_df[compliant_df['To-Do List Name'].isin(tracker_2)].copy(deep=True), compliant_changelog, next_action_date, test='compliant')
        if len(result) != 0:
            return result
        
    elif account == 'student':
        # Check for compliance in tracker_2
        result = check_compliance(conn, compliant_df[compliant_df['To-Do List Name'].isin(tracker_2)].copy(deep=True), compliant_changelog, next_action_date, test='dna_compliant')
        if len(result) != 0:
            return result
        # Check for non-compliance in tracker_2
        result = check_compliance(conn, noncompliant_df[noncompliant_df['To-Do List Name'].isin(tracker_2)].copy(deep=True), noncompliant_changelog, next_action_date, test='dna_noncompliant')
        if len(result) != 0:
            return result

def archive_old_reports (report_path, report_basename, archive_folder):
    '''Function to move all old reports into an archive folder.'''
    # Gather all files that match basename within the path
    all_files = dpu.get_latest(report_path, report_basename, num_files=float('inf'))
    # If no old reports are found, exit function
    if not all_files:
        return
    # If a single file, coerce to list
    if type(all_files) == str:
        all_files = [all_files]
    # Test if destination exists; if not, make folder
    destination = os.path.join(report_path, archive_folder)
    if not os.path.isdir(destination):
        os.makedirs(destination)
    # Iterate through all old reports
    for old_path in all_files:
        # Create new path name
        file_name = os.path.basename(old_path)
        new_path = os.path.join(destination, file_name)
        # If new path already exists, delete it
        if os.path.exists(new_path):
            os.remove(new_path)
        # Rename (i.e., move) old reports
        os.rename(old_path, new_path)

def output_report (df, column_names, file_name, date_of_report, output_path):
    '''Function that primarily applies formatting to excel report.'''
    # Re-order columns
    new_order = [col[0] for col in column_names]
    df = df[new_order]
    # File name
    f_name = file_name + '_' + date_of_report.strftime('%Y-%m-%d') + '.xlsx'
    f_name = os.path.join(output_path, f_name)
    # Initialize a writer
    writer = pd.ExcelWriter(f_name, engine='xlsxwriter')
    # Write data
    df.to_excel(writer, index=False, sheet_name='report')
    # Access the worksheet
    workbook = writer.book
    worksheet = writer.sheets['report']
    # Set zoom
    worksheet.set_zoom(90)
    # Set column sizes
    for i, col in enumerate(column_names):
        c = char_counter_from_int(i)
        worksheet.set_column('{0}:{1}'.format(c, c), col[1])
    # Conditional formatting
    # Add a format. Light red fill with dark red text.
    red = workbook.add_format({'bg_color': '#FFC7CE',
                                   'font_color': '#9C0006'})
    # Add a format. Green fill with dark green text.
    green = workbook.add_format({'bg_color': '#C6EFCE',
                                   'font_color': '#006100'})
    # Add a format. Yellow fill with black text.
    yellow = workbook.add_format({'bg_color': '#FFFF99',
                                   'font_color': '#000000',
                                   'num_format': 'mm/dd/yyyy'})
    # Add a format. Date
    date_format = workbook.add_format({'num_format': 'mm/dd/yyyy'})
    
    # Define our range for the color formatting
    number_rows = len(df.index)
    compliant_range = "D2:D{}".format(number_rows+1)
    changes_range = "C2:C{}".format(number_rows+1)
    nextdue_range = "G2:G{}".format(number_rows+1)
    
    # Highlight Noncompliant in Red
    worksheet.conditional_format(compliant_range, {'type': 'cell',
                                                   'criteria': 'equal to',
                                                   'value': '"No"',
                                                   'format': red})
    # Highlight changes in Green
    worksheet.conditional_format(changes_range, {'type': 'cell',
                                                   'criteria': 'equal to',
                                                   'value': 'TRUE',
                                                   'format': green})
    # Highlight next due in Yellow
    worksheet.conditional_format(nextdue_range, {'type': 'date',
                                                   'criteria': 'between',
                                                   'minimum': date_of_report,
                                                   'maximum': date_of_report + timedelta(days=60),
                                                   'format': yellow})
    # Format date for rest of column
    worksheet.conditional_format(nextdue_range, {'type': 'date',
                                                   'criteria': 'greater than',
                                                   'value': date_of_report + timedelta(days=60),
                                                   'format': date_format})
    
    # Freeze panes on top row
    worksheet.freeze_panes(1, 0)
    # Apply autofilters
    worksheet.autofilter('A1:{0}{1}'.format(char_counter_from_int(len(column_names)), number_rows+1))
    # Wrap text formatter
    wrap_text = workbook.add_format({'text_wrap': 1, 'valign': 'top', 'bold': True, 'bottom': True})
    # Wrap text on first row
    for i, col in enumerate(column_names):
        worksheet.write(0, i, col[0], wrap_text)
    # Apply changes
    writer.save()

def recursive_char (i, **kwargs):
    '''A function that takes an integer value and returns a character
    representation. Starts with 0-26 represented by A-Z. At 27, will
    become AA, then AB, etc. Implements recursion to allow for any
    length of integer.'''
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
    '''A function that takes an integer value and returns a character
    representation. Starts with 0-26 represented by A-Z. At 27, will
    become AA, then AB, etc. Implements recursion to allow for any
    length of integer.
    
    Optional Keyword Arguments:
    upper: True will return uppercase, False lowercase (default:True)
    '''
    int_list = recursive_char(i, **kwargs)
    char_list = [chr(x) for x in int_list]
    return ''.join(char_list)

def prepare_columns (prev_date):
    tru_order = [('Cln This Term?', 8),
                 ('Cln Next Term?', 8),
                 ('Changed Since {}'.format(prev_date), 15),
                 ('Compliant', 10),
                 ('Requirements Incomplete', 20),
                 ('Next Due', 20),
                 ('Next Due Date', 15),
                 ('Emplid', 10),
                 ('Last Name', 15),
                 ('First Name', 10),]
    fac_order = [('Primary Email', 20),
                 ('Secondary Email', 20),
                 ('Cell Phone', 20),
                 ('Courses_curr', 20),
                 ('Cln Sites_curr', 40),
                 ('Courses_next', 20),
                 ('Cln Sites_next', 40),]
    stu_order = [('Maj Desc', 15),
                 ('Campus', 8),
                 ('Email', 20),
                 ('Best Phone', 15),
                 ('Term', 5),
                 ('Cr_1_curr', 6),
                 ('Sec_1_curr', 6),
                 ('Dates_1_curr', 25),
                 ('Times_1_curr', 25),
                 ('Clinical Site_1_curr', 25),
                 ('Unit_1_curr', 11),
                 ('Instructor_1_curr Last-First', 20),
                 ('Instructor_1_curr Primary Email', 20),
                 ('Instructor_1_curr Secondary Email', 20),
                 ('Instructor_1_curr Cell Phone', 15),
                 ('Cr_2_curr', 6),
                 ('Sec_2_curr', 6),
                 ('Dates_2_curr', 25),
                 ('Times_2_curr', 25),
                 ('Clinical Site_2_curr', 25),
                 ('Unit_2_curr', 11),
                 ('Instructor_2_curr Last-First', 20),
                 ('Instructor_2_curr Primary Email', 20),
                 ('Instructor_2_curr Secondary Email', 20),
                 ('Instructor_2_curr Cell Phone', 15),
                 ('Cr_1_next', 6),
                 ('Sec_1_next', 6),
                 ('Dates_1_next', 25),
                 ('Times_1_next', 25),
                 ('Clinical Site_1_next', 25),
                 ('Unit_1_next', 11),
                 ('Instructor_1_next Last-First', 20),
                 ('Instructor_1_next Primary Email', 20),
                 ('Instructor_1_next Secondary Email', 20),
                 ('Instructor_1_next Cell Phone', 15),
                 ('Cr_2_next', 6),
                 ('Sec_2_next', 6),
                 ('Dates_2_next', 25),
                 ('Times_2_next', 25),
                 ('Clinical Site_2_next', 25),
                 ('Unit_2_next', 11),
                 ('Instructor_2_next Last-First', 20),
                 ('Instructor_2_next Primary Email', 20),
                 ('Instructor_2_next Secondary Email', 20),
                 ('Instructor_2_next Cell Phone', 15),]
    return (tru_order + fac_order), (tru_order + stu_order)

@click.command()
@click.option(
        '--prev_date',
        help='The date you want to use for previous report (basically, this is what the changelog will be built from. Should be in %Y-%m-d format (e.g. 2018-01-24).',
)
def main (prev_date):
    '''Main function call.'''
    # Call to FileLocator class
    FL = FileLocator()
    
    # Check if a prev_date was supplied
    if prev_date:
        # Test for proper date formatting
        try:
            datetime.strptime(prev_date, '%Y-%m-%d')
        except ValueError:
            print('Date provided was not in a valid format. Please retry using %Y-%m-d format (e.g. 2018-01-24).')
        
        # Gather absolute paths to previous
        nc_prev = os.path.abspath(os.path.join(os.sep, FL.health_req_report, 'Downloaded Reports', 'Noncompliant ' + prev_date + '.csv'))
        cc_prev = os.path.abspath(os.path.join(os.sep, FL.health_req_report, 'Downloaded Reports', 'Compliant ' + prev_date + '.csv'))
        # Make sure both files exist
        if os.path.exists(nc_prev) and os.path.exists(cc_prev):
            num_files = 1
        else:
            raise ValueError('Date provided is in a valid format, but compliance files do not exist using that date.')
    else:
        num_files = 2

    # Get the latest reports
    noncompliant_files = dpu.get_latest(os.path.join(FL.health_req_report, 'Downloaded Reports'), 'Noncompliant', num_files=num_files)
    compliant_files = dpu.get_latest(os.path.join(FL.health_req_report, 'Downloaded Reports'), 'Compliant', num_files=num_files)
    
    if prev_date:
        # Add previous files
        noncompliant_files = [noncompliant_files, nc_prev]
        compliant_files = [compliant_files, cc_prev]
        if noncompliant_files[0] == noncompliant_files[1] or compliant_files[0] == compliant_files[1]:
            raise 'Previous date provided is same as date of most recent compliance files. Download more recent reports and try again.'
    else:
        prev_date = noncompliant_files[1].rstrip('.csv')[-10:]
    
    # Get the two most recent reports
    noncompliant_curr = read_cb(noncompliant_files[0])
    noncompliant_prev = read_cb(noncompliant_files[1])
    compliant_curr = read_cb(compliant_files[0])
    compliant_prev = read_cb(compliant_files[1])
    # Get change logs
    noncompliant_changelog = pd.merge(noncompliant_curr, noncompliant_prev, on=noncompliant_curr.columns.tolist(), how='outer', indicator=True).query("_merge != 'both'").drop('_merge', 1)
    compliant_changelog = pd.merge(compliant_curr, compliant_prev, on=compliant_curr.columns.tolist(), how='outer', indicator=True).query("_merge != 'both'").drop('_merge', 1)
    # Get next action date file
    nad_file = dpu.get_latest(os.path.join(FL.health_req_report, 'Downloaded Reports'), 'Next_Action_Date', num_files=1)
    to_datetime = lambda d: datetime.strptime(d, '%m/%d/%Y')
    to_string = lambda d: datetime.strftime(d, '%m/%d/%Y')
    next_action_date = pd.read_csv(nad_file, header=0, converters={'Order Submission Date': to_datetime, 'Requirement Due Date': to_datetime})
    next_action_date.rename(columns={'Email Address':'Email'}, inplace=True)
    # Drop all but earliest next requirement
    next_action_date.sort_values(by='Requirement Due Date').drop_duplicates(subset='Email', keep='first', inplace=True)
    # Put back as string
    next_action_date['Requirement Due Date'] = next_action_date['Requirement Due Date'].apply(lambda x: to_string(x))
    
    # Get the latest student list
    students = pd.read_excel(dpu.get_latest(FL.students, 'Student List'), header=0, converters={'Emplid':str, 'Admit Term':str, 'Latest Term Enrl': str, 'Run Term': str,})
    students.drop_duplicates(subset='Emplid', inplace=True)
    # Get the faculty list
    faculty = pd.read_excel(os.path.join(FL.faculty, 'Employee List.xlsx'), header=0, converters={'Empl ID': str,})
    # Ignore all but necessary faculty columns
    faculty = faculty[['Empl ID', 'Last-First', 'Primary Email', 'Secondary Email', 'Cell Phone']]
    # Get term descriptions
    TermDescriptions = dpu.get_term_descriptions()
    # Get current term
    current_term = dpu.guess_current_term(TermDescriptions)
    # Figure out next term value
    next_term = str(int(current_term) + 5)
    # Get internship roster
    internships = dpu.get_cln(current_term, TermDescriptions)
    # Try to get next term's roster, if exists
    try:
        cln_2 = dpu.get_cln(next_term, TermDescriptions)
        internships = pd.concat(internships, cln_2)
    except:
        pass
    # Drop all but required information
    internships = internships[['Term', 'Cr', 'Clinical Site', 'Unit', 'Date', 'Student ID']].copy(deep=True)
    internships = internships[internships['Cr'] == '443']
    # Get schedule
    global schedule
    schedule = dpu.get_schedule(current_term, TermDescriptions)
    # Get clinical roster
    roster = clean_student_roster(current_term, internships)
    # Merge with student data
    roster = pd.merge(roster, students[['Emplid', 'Last Name', 'First Name', 'Maj Desc', 'Campus', 'Email', 'Best Phone']], how='left', on='Emplid')
    # Combine with faculty data
    # List of the required columns
    req_list = ['Last-First', 'Primary Email', 'Secondary Email', 'Cell Phone']
    # Naming convention for suffixes
    name_con = ['1_curr', '2_curr', '1_next', '2_next']
    # Iterate through suffixes (i.e., through instructors)
    for suffix in name_con:
        # Create new column names
        inst = 'Instructor' + '_' + suffix
        new_cols = [inst + ' ' + x for x in req_list]
        # Apply search function
        roster[new_cols] = roster.apply(instructor_contact_info, axis=1, args=(suffix, faculty, req_list, schedule))
    # Drop Faculty ID Fields
    id_fields = ['Faculty_ID_' + suffix for suffix in name_con]
    roster.drop(labels=id_fields, axis=1, inplace=True)
    # Combine with schedule data
    # List of the required columns
    req_list = ['Clinical Site', 'Unit']
    # Iterate through suffixes
    for suffix in name_con:
        # Create new column names
        new_cols = [x + '_' + suffix for x in req_list]
        # Apply search function
        roster[new_cols] = roster.apply(clinical_info, axis=1, args=(suffix, req_list, schedule, internships))
    # Gather historical student data
    historical_students = get_historical_student_data(FL.hist_students, students)
    
    # Get a faculty roster for health_req report
    faculty_roster = clean_faculty_roster(current_term)
    # Merge with faculty contact info
    faculty_roster = pd.merge(faculty_roster, faculty, left_on='Emplid', right_on='Empl ID')
    # Separate out names
    faculty_roster['Last Name'] = faculty_roster['Last-First'].apply(lambda x: x.split(', ')[0])
    faculty_roster['First Name'] = faculty_roster['Last-First'].apply(lambda x: x.split(', ')[1])
    
    # Collection of tracker names
    all_trackers = np.concatenate((compliant_curr['To-Do List Name'].unique(), noncompliant_curr['To-Do List Name'].unique()))
    all_trackers = np.unique(all_trackers)
    # Breakdown into types
    dna_trackers = []
    student_trackers = []
    faculty_trackers = []
    for tracker in all_trackers:
        if 'Disclosure & Authorization' in tracker:
            dna_trackers.append(tracker)
        elif 'DE34' in tracker:
            faculty_trackers.append(tracker)
        elif 'DE69' in tracker:
            student_trackers.append(tracker)
        
    '''Here we attempt to connect all students in the clinical roster with
    an account in Castle Branch. To do so, we make a dictionary lookup.
    # dict[Emplid] = (Last Name, First Name, Email)
    '''
    cb_to_dpu = {}
    # Attempt to match students, starting with full accounts
    # Sometimes the dna account is all we get, even though they have full
    roster.apply(match_students, axis=1, args=(noncompliant_curr[noncompliant_curr['To-Do List Name'].isin(student_trackers)], cb_to_dpu), historic=historical_students);
    roster.apply(match_students, axis=1, args=(compliant_curr[compliant_curr['To-Do List Name'].isin(student_trackers)], cb_to_dpu), historic=historical_students);
    roster.apply(match_students, axis=1, args=(noncompliant_curr[noncompliant_curr['To-Do List Name'].isin(dna_trackers)], cb_to_dpu));
    roster.apply(match_students, axis=1, args=(compliant_curr[compliant_curr['To-Do List Name'].isin(dna_trackers)], cb_to_dpu));
    # Faculty matching    
    cb_to_fac = {}
    faculty_roster.apply(match_faculty, axis=1, args=(noncompliant_curr[noncompliant_curr['To-Do List Name'].isin(faculty_trackers)], cb_to_fac));
    faculty_roster.apply(match_faculty, axis=1, args=(compliant_curr[compliant_curr['To-Do List Name'].isin(faculty_trackers)], cb_to_fac));
    faculty_roster.apply(match_faculty, axis=1, args=(noncompliant_curr[noncompliant_curr['To-Do List Name'].isin(student_trackers)], cb_to_fac));
    faculty_roster.apply(match_faculty, axis=1, args=(compliant_curr[compliant_curr['To-Do List Name'].isin(student_trackers)], cb_to_fac));
    
    # New column names
    fields = ['Changed Since ' + prev_date, 'Compliant', 'Requirements Incomplete', 'Next Due', 'Next Due Date']
    # Gather compliance status
    #roster[fields] = roster.apply(determine_status, axis=1, args=(cb_to_dpu, noncompliant_curr, compliant_curr, noncompliant_changelog, compliant_changelog, student_trackers, dna_trackers, next_action_date))
    roster[fields] = roster.apply(determine_status, axis=1, args=(cb_to_dpu, noncompliant_curr, compliant_curr, noncompliant_changelog, compliant_changelog, student_trackers, dna_trackers, next_action_date), account='student')
    # Gather compliance status
    faculty_roster[fields] = faculty_roster.apply(determine_status, axis=1, args=(cb_to_fac, noncompliant_curr, compliant_curr, noncompliant_changelog, compliant_changelog, faculty_trackers, student_trackers, next_action_date), account='faculty')
    
    # Gather column names
    
    faculty_cols, student_cols = prepare_columns (prev_date)
    
    # Archive old reports
    archive_old_reports(FL.health_req_report, 'student_report', 'Archived Student Reports')
    archive_old_reports(FL.health_req_report, 'faculty_report', 'Archived Faculty Reports')
    
    # Output to file
    date_of_current = noncompliant_files[0].rstrip('.csv')[-10:]
    date_of_current = datetime.strptime(date_of_current, '%Y-%m-%d')
    output_report (roster, student_cols, 'student_report', date_of_current, FL.health_req_report)
    output_report (faculty_roster, faculty_cols, 'faculty_report', date_of_current, FL.health_req_report)
    
    #output_report(roster, date_of_current, FL.health_req_report, student_cols)
    #output_fac_report(faculty_roster, date_of_current, FL.health_req_report, faculty_cols)

if __name__ == "__main__":
    main()
