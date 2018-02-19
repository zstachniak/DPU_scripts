# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 15:32:27 2018

@author: astachn1
"""

import pandas as pd
import os
from datetime import datetime, time
import dpu.scripts as dpu
from dpu.file_locator import FileLocator
import click

def apply_dates_times (row):
    '''Pandas apply function that returns clinical date range and
    time range.'''
    # If internship, return None
    if row['Cr'] in ['442', '443']:
        dates = None
        times = None
    # For all others
    else:
        # Gather Meeting Pattern
        pat = row['Pat']
        # Convert pattern to day of week integer
        day_of_week = day_of_week_hash.get(pat, None)
        # If NSG 301
        if row['Cr'] == '301':
            first, last = dpu.true_date(row['Start Date'], day_of_week, return_range=True, num_recurrence=4, skip_weeks=6)
        else:
            # Gather Dates
            first, last = dpu.true_date(row['Start Date'], day_of_week, return_range=True)
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

def get_col_widths(df, offset=0):
    '''Returns max of lengths of column name and values, left to right.'''
    return [max([len(str(s)) for s in df[col].values] + [len(col)]) + offset for col in df.columns]

def output_report (df, column_names, file_name, date_of_report, output_path):
    '''Function that primarily applies formatting to excel report.'''
    # Re-order columns
    df = df[column_names]
    # Sort by Student Name
    df = df.sort_values(by=['Cr', 'Sec', 'Student Full Name'])
    # File name
    f_name = file_name + '_' + date_of_report.strftime('%Y-%m-%d') + '.xlsx'
    f_name = os.path.join(output_path, f_name)
    # Initialize a writer
    writer = pd.ExcelWriter(f_name, engine='xlsxwriter')
    # Write data
    df.to_excel(writer, index=False, sheet_name='Clinical Roster')
    # Access the worksheet
    workbook = writer.book
    worksheet = writer.sheets['Clinical Roster']
    
    # Set column widths
    for i, width in enumerate(get_col_widths(df, offset=2)):
        worksheet.set_column(i, i, width)
    
    # Get num rows and columns
    nrows = len(df)
    ncols = len(column_names)
    # Gather header as a dictionary (for hacky way to ignore overwriting)
    header = [{'header': di} for di in df.columns.tolist()]
    
    # Add a table to the worksheet (uses table style Light 1)
    worksheet.add_table('A1:{}{}'.format(dpu.char_counter_from_int(ncols - 1), nrows + 1), {'autofilter': 1, 'style': 'Table Style Light 1', 'header_row': 1, 'columns': header})

    # Freeze panes on top row and first 3 columns
    worksheet.freeze_panes(1, 3)
    
    # Apply changes
    writer.save()

@click.command()
@click.option(
        '--term', '-t', type=str,
        help='Term for which to send out course coordinator emails',
)
def main (term):
    '''Main function call.'''
    # Initialize File Locator
    FL = FileLocator()
    # Gather term descriptions
    TermDescriptions = dpu.get_term_descriptions()
    # If term not given, guess
    if not term:
        term = dpu.guess_current_term(TermDescriptions)
        
    # Gather faculty
    Faculty = dpu.get_employee_list()
    Faculty.rename(columns={'Empl ID':'Faculty ID', 'Last Name': 'Faculty Last Name', 'First Name': 'Faculty First Name', 'Last-First': 'Faculty Full Name', 'Primary Email': 'Faculty Primary Email', 'Secondary Email': 'Faculty Secondary Email', 'Cell Phone': 'Faculty Phone'}, inplace=True)
    # Gather student list
    students = dpu.get_student_list()
    students.rename(columns={'Emplid':'Student ID', 'Student Name': 'Student Full Name', 'Last Name': 'Student Last Name', 'First Name': 'Student First Name', 'Email': 'Student Email', 'Best Phone': 'Student Phone'}, inplace=True)
    # Gather schedule
    schedule = dpu.get_schedule(term, TermDescriptions)
    # Filter schedule
    schedule = schedule[schedule['Program'].isin(['MENP', 'RFU'])]
    
    # Gather Roster
    roster = dpu.get_student_roster()
    roster.rename(columns={'Faculty_ID': 'Faculty ID'}, inplace=True)
    # Filter roster
    cln_types = ['PRA', 'CLN']
    MENP_courses = schedule['Cr'].unique().tolist()
    roster = roster[(roster['Term'] == term) & (roster['Role'] == 'PI') & (roster['Cr'].isin(MENP_courses)) & ( (roster['Type'].isin(cln_types)) | ((roster['Cr'] == '301') & (roster['Type'] == 'LAB')))].copy(deep=True)
    # Takes care of 301 issue (due to changing course times)
    roster = roster.sort_values(by=['Faculty ID', 'Start Date'], ascending=[True, False]).drop_duplicates(subset=['Term', 'Student ID', 'Cr', 'Sec']).copy(deep=True)
    # Gather string-formatted and concatenated dates and times
    roster[['Dates', 'Times']] = roster.apply(apply_dates_times, axis=1)
    # Drop Unneeded
    roster.drop(labels=['Student Name', 'Student Major', 'Subject', 'Type', 'Class Nbr', 'Role', 'Mode', 'Start Date', 'End Date', 'Pat', 'Mtg Start', 'Mtg End'], axis=1, inplace=True)
    
    # Merge together
    roster = roster.merge(schedule[['Cr', 'Sec', 'Title', 'Clinical Site', 'Unit', 'Max Cap', 'Confirmed']], how='left', on=['Cr', 'Sec'])
    roster = roster.merge(students[['Student ID', 'Student Last Name', 'Student First Name', 'Student Full Name', 'Student Email', 'Student Phone', 'Campus']], how='left', on='Student ID')
    roster = roster.merge(Faculty[['Faculty ID', 'Faculty Last Name', 'Faculty First Name', 'Faculty Full Name', 'Faculty Primary Email', 'Faculty Secondary Email', 'Faculty Phone']], how='left', on='Faculty ID')
    
    # Column names and order
    column_names = ['Term', 'Campus', 'Cr', 'Sec', 'Clinical Site', 'Unit', 'Dates', 'Times', 'Student ID', 'Student Last Name', 'Student First Name', 'Student Full Name', 'Student Email', 'Student Phone', 'Faculty ID', 'Faculty Last Name', 'Faculty First Name', 'Faculty Full Name', 'Faculty Primary Email', 'Faculty Secondary Email', 'Faculty Phone', 'Title', 'Max Cap', 'Confirmed']
        
    # Gather date
    date_of_report = datetime.strptime(dpu.get_latest(FL.rosters, 'NSG_STDNT_ROSTER').rstrip('.xlsx')[-10:], '%Y-%m-%d')
    
    # Ensure Output path
    starting_path = FL.cln_roster
    # Gather academic year and quarter
    ay = TermDescriptions[TermDescriptions['Term'] == term]['Academic Year'].item()
    q = TermDescriptions[TermDescriptions['Term'] == term]['Quarter'].item()
    # Update file path
    output_path = os.path.join(starting_path, ay, q)
    dpu.ensure_dir(output_path)
    
    # Output file
    output_report(roster, column_names, 'Clinical Roster', date_of_report, output_path)
    
if __name__ == "__main__":
    main()