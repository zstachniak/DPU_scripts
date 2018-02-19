# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 09:44:10 2017

@author: astachn1
"""

import os
import pandas as pd
import win32com.client as win32
import click
import dpu.scripts as dpu
from dpu.file_locator import FileLocator

#############################################################
# Functions
#############################################################

def find_coord (row, coord_courses, coord_faculty):
    if row['Type'] == 'COORD':
        coord_courses.add(row['Cr'])
        coord_faculty.add(row['Faculty'])

def find_sections (row, course, coord_programs):
    if row['Cr'] == course and row['Type'] != 'LEC' and row['Type'] != 'COORD':
        coord_programs.add(row['Program'])

def map_contact (row, element, Faculty):
    '''Map an element of the Employee List'''
    return_item = Faculty[Faculty['Last-First'] == row['Faculty']][element].astype(list).values
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
# Main
#############################################################
@click.command()
@click.option(
        '--term', '-t', type=str,
        help='Term for which to send out course coordinator emails',
)
def main(term):
    '''Main function.'''

    FL = FileLocator()
    
    # Gather term descriptions and faculty
    TermDescriptions = dpu.get_term_descriptions()
    Faculty = dpu.get_employee_list()

    # Gather schedule based on term (guess term if not provided)
    if not term:
        term = dpu.guess_current_term(TermDescriptions)
    schedule = dpu.get_schedule(term, TermDescriptions)
    
    # Initial setup
    output_path = os.path.join(os.path.sep, FL.schedule, 'Template', 'Course Coordinators')
    output_fields = ['Term', 'NSG', 'Cr', 'Sec', 'Time', 'Clinical Site', 'Unit', 'Faculty', 'Fac Conf', 'Primary Email', 'Secondary Email', 'Cell Phone']
    contact_fields = ['Primary Email', 'Secondary Email', 'Cell Phone']

    # Remove any courses in the wrong term or in DNP or RN to MS program
    schedule = schedule[(schedule['Term'] == term) & (schedule['Program'].isin(['MENP', 'RFU']))]

    # Create a set of all courses and faculty with a coordinator
    coord_courses = set()
    coord_faculty = set()
    schedule.apply(find_coord, axis=1, args=(coord_courses, coord_faculty))
    coord_courses = list(coord_courses)
    coord_faculty = list(coord_faculty)

    # Remove any courses that do not have a coordinator
    schedule = schedule[schedule['Cr'].isin(coord_courses)]

    # Remove all previous files
    dpu.ensure_empty_dir(output_path)

    # Map contact information
    for field in contact_fields:        
        schedule[field] = schedule.apply(map_contact, axis=1, args=(field, Faculty))

    for course in coord_courses:
        coord_programs = set()
        schedule.apply(find_sections, axis=1, args=(course, coord_programs))
        coord_programs = list(coord_programs)
        for program in coord_programs:
            # Get coordinator faculty name
            faculty = schedule[(schedule['Cr'] == course) & (schedule['Type'] == 'COORD') & (schedule['Program'] == program)]['Faculty'].astype(list).values[0]
            # Return only the labs or clns that match the program
            schedule_out = schedule[(schedule['Cr'] == course) & (schedule['Type'] != 'COORD') & (schedule['Type'] != 'LEC') & (schedule['Program'] == program)].copy(deep=True)
            # Keep only the needed columns
            schedule_out = schedule_out[output_fields]
            # Build output filename
            f_name = '{0} - {1} - {2}.xlsx'.format(faculty, course, program)
            output_file = os.path.join(os.path.sep, output_path, f_name)
            # Output file of lab/cln sections
            schedule_out.to_excel(output_file, index=False)
    
    # Iterate through coordinators
    for coordinator in coord_faculty:
        # Ignore undefined faculty
        if coordinator not in ['TBA', 'TBD']:
            # Get coordinator's email
            try:
                recipient = Faculty[Faculty['Last-First'] == coordinator]['Primary Email'].item()
            except:
                recipient = Faculty[Faculty['Last-First'] == coordinator]['Secondary Email'].item()
            # Write subject
            subject = 'Course Coordinator: Your Clinical or Lab Faculty'
            # Write body
            body = '''Dear {},
            
            Attached please find a list of your clinical or lab courses along with faculty contact info. If there is no confirmation date listed for a faculty member (field = 'Fac Conf'), we do not recommend contacting that person at this time. They have likely only given us a tentative 'yes.'
            
            This is an automated email - if something looks wrong, please reply and let me know.'''.format(coordinator.split(',')[1].strip())
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

if __name__ == "__main__":
    main()
    