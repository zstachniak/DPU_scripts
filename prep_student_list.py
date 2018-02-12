# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 15:35:51 2018

@author: astachn1
"""

import pandas as pd
import os
from datetime import datetime
import dpu.scripts as dpu
from dpu.file_locator import FileLocator
import click

def determine_campus (row, rfu_ids):
    '''Function to return campus based on student group.'''
    if row['Emplid'] in rfu_ids:
        return 'RFU'
    else:
        return 'LPC'

def get_col_widths(df, offset=0):
    '''Returns max of lengths of column name and values, left to right.'''
    return [max([len(str(s)) for s in df[col].values] + [len(col)]) + offset for col in df.columns]

def output_report (df, column_names, file_name, date_of_report, output_path):
    '''Function that primarily applies formatting to excel report.'''
    # Re-order columns
    df = df[column_names]
    # Sort by Student Name
    df = df.sort_values(by='Student Name')
    # File name
    f_name = file_name + '_' + date_of_report.strftime('%Y-%m-%d') + '.xlsx'
    f_name = os.path.join(output_path, f_name)
    # Initialize a writer
    writer = pd.ExcelWriter(f_name, engine='xlsxwriter')
    # Write data
    df.to_excel(writer, index=False, sheet_name='student_list')
    # Access the worksheet
    workbook = writer.book
    worksheet = writer.sheets['student_list']
    
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
def main ():
    '''Main function call.'''
    # Initialize File Locator
    FL = FileLocator()
    
    # Gather and read data using latest files, then concatenate together
    grads = pd.read_excel(dpu.get_latest(FL.students_downloaded, 'NSG_Graduate_Students'), converters={'Emplid':str, 'Admit Term':str, 'Maj Adv Emplid':str, 'Run Term':str})
    majors = pd.read_excel(dpu.get_latest(FL.students_downloaded, 'NSG_Majors'), converters={'Emplid':str, 'Admit Term':str, 'Maj Adv Emplid':str, 'Run Term':str})
    df = pd.concat([grads, majors])
    
    # Gather IDs of students in RFU student group
    rfns = pd.read_excel(dpu.get_latest(FL.students_downloaded, 'RFNS_STDNTGRP'), skiprows=0, header=1, converters={'ID':str})
    rfu_ids = rfns['ID'].unique().tolist()
    
    # Pandas apply function to determine campus
    df['Campus'] = df.apply(determine_campus, axis=1, args=(rfu_ids,))

    # Archive old student list
    dpu.archive_old_reports(FL.students, 'Student List', 'Historical Student Lists')
    
    # Column names and order
    column_names = ['Emplid', 'Student Name', 'Campus', 'Admit Term', 'Admit Term Desc', 'Acad Level Desc', 'Maj Desc', 'Maj Subplan Desc', 'Maj Adv Name', 'Latest Term Enrl', 'Term Enrl Desc', 'Enrl Hrs', 'Qty Crses', 'Dpu Hrs', 'Transfer Hrs', 'Total Credhrs', 'Term GPA', 'Cum GPA', 'Prefix', 'First Name', 'Middle Name', 'Last Name', 'Suffix', 'Gender', 'Ethnic Group Desc', 'Orig Cntry Des', 'Email', 'Address1', 'Address2', 'City', 'State', 'Postal', 'Best Phone', 'Main Phone', 'Cell Phone', 'Run Term']
        
    # Gather date
    date_of_report = datetime.strptime(dpu.get_latest(FL.students_downloaded, 'NSG_Graduate_Students').rstrip('.xlsx')[-10:], '%Y-%m-%d')

    # Output file
    output_report (df, column_names, 'Student List', date_of_report, FL.students)

if __name__ == "__main__":
    main()
    