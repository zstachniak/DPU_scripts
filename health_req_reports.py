# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 09:23:14 2018

@author: astachn1
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from fuzzywuzzy import fuzz, process
import dpu.scripts as dpu
from dpu.file_locator import FileLocator
import click

'''
Need to add Program of Study
Filter out obvious non-necessary course types, numbers
use excel schedule as backup for unhired faculty
Field for: [Currently in Clinical] & [Registered for CLN next Quarter]


'''
def separate_cln_dupes (df):
    '''Separates out clinicals when one student has two courses.'''
    # Sort by course number
    df.sort_values(by='Cr', inplace=True)
    # Gather list of duplicates (i.e., student has more than one clinical)
    dupe_truth_value = df.duplicated(subset=['Student ID'], keep='first')
    dupes = df.loc[dupe_truth_value].copy(deep=True)
    dupes.drop(labels=['Term'], axis=1, inplace=True)
    # Drop duplicates from original list
    df.drop_duplicates(subset=['Student ID'], keep='first', inplace=True)
    # Merge two lists 
    df = pd.merge(df, dupes, how='left', on='Student ID', suffixes=['_1', '_2'])
    return df

def clean_student_roster (current_term):
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
    roster = roster.sort_values(by='Faculty_ID').drop_duplicates(subset=['Term', 'Student ID', 'Cr', 'Sec']).copy(deep=True)
    # Figure out next term value
    next_term = str(int(current_term) + 5)
    # Gather roster for next term
    roster_next_term = roster[roster['Term'] == next_term].copy(deep=True)
    roster_next_term = separate_cln_dupes(roster_next_term)
    roster_next_term['Cln Next Term?'] = 'Yes'
    roster_next_term.drop(labels=['Term'], axis=1, inplace=True)
    # Drop next term from current roster
    roster = roster[roster['Term'] == current_term].copy(deep=True)
    roster = separate_cln_dupes(roster)
    roster['Cln This Term?'] = 'Yes'
    # Perform outer join so that we retain students with cln next term
    roster = pd.merge(roster, roster_next_term, how='outer', on='Student ID', suffixes=['_curr', '_next'])
    # Put student ID at start
    roster.insert(0, 'Emplid', roster['Student ID'])
    roster.drop(labels=['Student ID'], axis=1, inplace=True)
    # Fill NaNs
    roster['Cln Next Term?'].fillna(value='No', inplace=True)
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

def clinical_info (row, suffix, req_list, schedule):
    '''Function to return pd.Series element of requested values.'''
    # Prepare search terms
    cr_x = 'Cr' + '_' + suffix
    sec_x = 'Sec' + '_' + suffix
    # Attempt exact ID comparison
    res = schedule[(schedule['Cr'] == row[cr_x]) & (schedule['Sec'] == row[sec_x])]
    # If results are found, return those results
    if len(res) == 1:
        return pd.Series([res[x].item() for x in req_list])
    else:
        return pd.Series([None for x in req_list])  

def true_match (row, df, field_list):
    '''Function to handle true matches'''
    for field in field_list:
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

def determine_status (row, id_dict, noncompliant_df, compliant_df, noncompliant_changelog, compliant_changelog, student_trackers, dna_trackers, next_action_date):
    '''Gather CB compliance stats'''
    # Get ID
    key = id_dict.get(row['Emplid'], None)
    # If ID doesn't exist, can't identify student
    if not key:
        # If unable to identify student
        return pd.Series([None, 'No', ['Could Not Identify Student'], None, None])
    # Store connection key
    conn = pd.Series(data={'Last Name':key[0], 'First Name':key[1], 'Email':key[2]})
    
    # Check for compliance
    compliant = true_match(conn, compliant_df[compliant_df['To-Do List Name'].isin(student_trackers)], ['Email'])
    if len(compliant) == 1:
        # Check if this is an update from last time
        changed = (compliant.values == compliant_changelog.values).all(1).any()
        # Return required fields
        return pd.Series([changed, 'Yes', None, compliant['Name of Next Requirement Due'].item(), datetime.strptime(compliant['Next Action Date'].item(), '%m/%d/%Y')])
        
    # Check for noncompliance
    noncompliant = true_match(conn, noncompliant_df[noncompliant_df['To-Do List Name'].isin(student_trackers)], ['Email'])
    if len(noncompliant) >= 1:
        # If more than one result, simply take the first
        noncompliant = noncompliant.iloc[0]
        # Check if this is an update from last time
        changed = (noncompliant.values == noncompliant_changelog.values).all(1).any()
        # Check 
        next_due = true_match(conn, next_action_date, ['Email'])
        if len(next_due) == 1:
            next_due_req = next_due['Requirement Name'].item()
            next_due_date = datetime.strptime(next_due['Requirement Due Date'].item(), '%m/%d/%Y')
        else:
            next_due_req = None
            next_due_date = None
        # Return required fields
        return pd.Series([changed, 'No', noncompliant['Requirements Incomplete'], next_due_req, next_due_date])
    
    # Check for students who have only completed D&A
    dna = true_match(conn, compliant_df[compliant_df['To-Do List Name'].isin(dna_trackers)], ['Email'])
    if len(dna) == 1:
        # Check if this is an update from last time
        changed = (dna.values == compliant_changelog.values).all(1).any()
        # Return required fields
        return pd.Series([changed, 'No', 'Student has met D&A but not registered for health req tracker', None, None])
    
    # Check for students who have no account
    dna = true_match(conn, noncompliant_df[noncompliant_df['To-Do List Name'].isin(dna_trackers)], ['Email'])
    if len(dna) == 1:
        # Check if this is an update from last time
        changed = (dna.values == noncompliant_changelog.values).all(1).any()
        # Return required fields
        return pd.Series([changed, 'No', 'Student has not registered for an account', None, None])

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

def output_report (df, date_of_report, output_path, column_names):
    '''Function that primarily applies formatting to excel report.'''
    # File name
    f_name = 'student_report_' + date_of_report.strftime('%Y-%m-%d') + '.xlsx'
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
    worksheet.set_column('A:B', 8)
    worksheet.set_column('C:C', 15)
    worksheet.set_column('D:D', 10)
    worksheet.set_column('E:F', 20)
    worksheet.set_column('G:G', 15)
    worksheet.set_column('H:H', 10)
    worksheet.set_column('I:I', 15)
    worksheet.set_column('J:J', 10)
    worksheet.set_column('K:K', 15)
    worksheet.set_column('L:L', 8)
    worksheet.set_column('M:M', 20)
    worksheet.set_column('N:N', 15)
    worksheet.set_column('O:Q', 5)
    worksheet.set_column('R:R', 20)
    worksheet.set_column('S:S', 10)
    worksheet.set_column('T:V', 20)
    worksheet.set_column('W:W', 15)
    worksheet.set_column('X:Y', 5)
    worksheet.set_column('Z:Z', 20)
    worksheet.set_column('AA:AA', 10)
    worksheet.set_column('AB:AD', 20)
    worksheet.set_column('AE:AE', 15)
    worksheet.set_column('AF:AG', 5)
    worksheet.set_column('AH:AH', 20)
    worksheet.set_column('AI:AI', 10)
    worksheet.set_column('AJ:AL', 20)
    worksheet.set_column('AM:AM', 15)
    worksheet.set_column('AN:AO', 5)
    worksheet.set_column('AP:AP', 20)
    worksheet.set_column('AQ:AQ', 10)
    worksheet.set_column('AR:AT', 20)
    worksheet.set_column('AU:AU', 15)
    
    # Conditional formatting
    # Add a format. Light red fill with dark red text.
    format1 = workbook.add_format({'bg_color': '#FFC7CE',
                                   'font_color': '#9C0006'})
    # Add a format. Green fill with dark green text.
    format2 = workbook.add_format({'bg_color': '#C6EFCE',
                                   'font_color': '#006100'})
    # Add a format. Yellow fill with black text.
    format3 = workbook.add_format({'bg_color': '#FFFF99',
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
                                                   'format': format1})
    # Highlight changes in Green
    worksheet.conditional_format(changes_range, {'type': 'cell',
                                                   'criteria': 'equal to',
                                                   'value': 'TRUE',
                                                   'format': format2})
    # Highlight next due in Yellow
    worksheet.conditional_format(nextdue_range, {'type': 'date',
                                                   'criteria': 'less than',
                                                   'value': date_of_report + timedelta(days=60),
                                                   'format': format3})
    # Format date for rest of column
    worksheet.conditional_format(nextdue_range, {'type': 'date',
                                                   'criteria': 'greater than',
                                                   'value': date_of_report + timedelta(days=60),
                                                   'format': date_format})
    
    # Apply autofilters
    worksheet.autofilter('A1:AF{}'.format(number_rows+1))
    # Wrap text on first row
    wrap_text = workbook.add_format({'text_wrap': 1, 'valign': 'top'})
    
    for i, col in column_names:
        worksheet.write_row((1, {}).format(i), col, wrap_text)
    
    worksheet.set_row(0, 30, wrap_text)
    # Apply changes
    writer.save()

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
    # Get schedule
    schedule = dpu.get_schedule(current_term, TermDescriptions)
    # Get clinical roster
    roster = clean_student_roster(current_term)
    # Merge with student data
    roster = pd.merge(roster, students[['Emplid', 'Last Name', 'First Name', 'Maj Desc', 'Campus', 'Email', 'Best Phone']], how='left', on='Emplid')
    # Combine with faculty data
    # List of the required columns
    req_list = ['Primary Email', 'Secondary Email', 'Cell Phone']
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
    req_list = ['Clinical Site', 'Unit', 'Time']
    # Iterate through suffixes
    for suffix in name_con:
        # Create new column names
        new_cols = [x + '_' + suffix for x in req_list]
        # Apply search function
        roster[new_cols] = roster.apply(clinical_info, axis=1, args=(suffix, req_list, schedule))
    # Gather historical student data
    historical_students = get_historical_student_data(FL.hist_students, students)
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
    
    # New column names
    fields = ['Changed Since ' + noncompliant_files[1].rstrip('.csv')[-10:], 'Compliant', 'Requirements Incomplete', 'Next Due', 'Next Due Date']
    # Gather compliance status
    roster[fields] = roster.apply(determine_status, axis=1, args=(cb_to_dpu, noncompliant_curr, compliant_curr, noncompliant_changelog, compliant_changelog, student_trackers, dna_trackers, next_action_date))
    
    # Revise order of columns
    cols = roster.columns.tolist()
    new_order = cols[6:7] + cols[11:12] + cols[42:47] + cols[0:1] + cols[12:18] + cols[1:4] + cols[30:33] + cols[18:21] + cols[4:6] + cols[33:36] + cols[21:24] + cols[7:9] + cols[36:39] + cols[24:27] + cols[9:11] + cols[39:42] + cols[27:30]
    roster = roster[new_order]
    
    # Archive old reports
    archive_old_reports(FL.health_req_report, 'student_report', 'Archived Student Reports')
    
    # Output to file
    date_of_current = noncompliant_files[0].rstrip('.csv')[-10:]
    date_of_current = datetime.strptime(date_of_current, '%Y-%m-%d')
    output_report(roster, date_of_current, FL.health_req_report, roster.columns)

if __name__ == "__main__":
    main()
