# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 13:27:31 2018

@author: astachn1
"""

import pandas as pd
import dpu.scripts as dpu
import click

def print_row (row, long_term, Faculty):
    try:
        name = Faculty[Faculty['Empl ID'] == row['ID']]['Last-First'].item()    
    except:
        name = None
    print('{0} {1}: NSG {2}-{3} {4} {5} {6} {7}'.format(row['Term'], long_term, row['Catalog'], row['Section'], row['Component'], row['Mode'], row['ID'], name))

@click.command()
@click.option(
        '--faculty_id', '-id', type=str,
        help='Faculty ID number for which to search.',
)
@click.option(
        '--cr', '-cr', type=str,
        help='Course for which to search.',
)
@click.option(
        '--cr_type', '-crt', type=str,
        help='Course type for which to search',
)
@click.option(
        '--mode', '-m', type=str,
        help='Modality for which to search (P, HB, OL)',
)
@click.option(
        '--num_recent', '-n', type=int,
        help='The number of recent courses to return (in reverse order).',
)
def main (faculty_id, cr, cr_type, mode, num_recent):
    '''Main function call.'''
    
    # Guess current term
    TermDescriptions = dpu.get_term_descriptions()

    # Download historical course info
    course_data = dpu.get_class_data()
    # Filter out non-primary instructors
    course_data = course_data[course_data['Role'] == 'PI'].copy(deep=True)
    # Takes care of 301 issue (due to changing course times)
    course_data = course_data.sort_values(by=['ID', 'Start Date'], ascending=[True, False]).drop_duplicates(subset=['Term', 'Class Nbr']).copy(deep=True)
    
    # Gather min and max terms
    min_term = course_data['Term'].min()
    min_long_term = TermDescriptions[TermDescriptions['Term'] ==  min_term]['Long Description'].item()
    max_term = course_data['Term'].max()
    max_long_term = TermDescriptions[TermDescriptions['Term'] ==  max_term]['Long Description'].item()
    
    Faculty = dpu.get_employee_list()
    if faculty_id:
        # Get faculty name for print statement
        first = Faculty[Faculty['Empl ID'] == faculty_id]['First Name'].item()
        last = Faculty[Faculty['Empl ID'] == faculty_id]['Last Name'].item()
        print(f'Searching ID {faculty_id} between {min_long_term} and {max_long_term}.')
    else:
        if not cr:
            raise "Must submit either a faculty ID or a course for which to perform the search."
        print(f'Searching for last offering of NSG {cr} between {min_long_term} and {max_long_term}.')

    # If report appears to be outdate, suggest so to user
    curr_term = dpu.guess_current_term(TermDescriptions)
    if int(curr_term) > int(max_term):
        print('Your report may be out of date. To update or increase range of search, download a new NSG_CLASS_SCHEDULE_MULTI_TERM report.')
    
    # Do Pandas filter for ID and other optional attributes
    if faculty_id:
        course_data = course_data[course_data['ID'] == faculty_id].copy(deep=True)
    if cr:
        course_data = course_data[course_data['Catalog'] == cr].copy(deep=True)
    if cr_type:
        course_data = course_data[course_data['Component'] == cr_type].copy(deep=True)
    if mode:
        course_data = course_data[course_data['Mode'] == mode].copy(deep=True)
        
    # Test if any records found
    if len(course_data) == 0:
        raise 'No records found. To increase range of search, download a new NSG_CLASS_SCHEDULE_MULTI_TERM report.'
    
    # Sort by term
    course_data.sort_values(by=['Term', 'Catalog', 'Section'], ascending=[False, True, True], inplace=True)
    
    # Get number of unique terms, and sort in reverse order
    unique_terms = course_data['Term'].unique().tolist()
    unique_terms.sort(reverse=True)
    
    if faculty_id and cr:
        print(f'NSG {cr} was last taught by {first} {last} on... (in reverse order)')
    elif not cr:
        print(f'Last course taught by {first} {last} on... (in reverse order)')
    elif not faculty_id:
        print(f'NSG {cr} was last taught on... (in reverse order)')
    # Default to return a single term of info
    if not num_recent:
        num_recent = 1
    
    # Iterate through terms
    counter = 0
    while counter < num_recent:
        # get term from array using counter as index
        temp_term = unique_terms[counter]
        long_term = TermDescriptions[TermDescriptions['Term'] == temp_term]['Long Description'].item()
        # filter for that term
        temp_courses = course_data[course_data['Term'] == temp_term]
        # print each one
        temp_courses.apply(print_row, axis=1, args=(long_term, Faculty))        
        counter += 1

if __name__ == "__main__":
    main()
