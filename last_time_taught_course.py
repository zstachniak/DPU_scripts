# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 13:27:31 2018

@author: astachn1
"""

import pandas as pd
import dpu.scripts as dpu
import click

def print_row (row, long_term):    
    print('{0} {1}: NSG {2}-{3} {4}'.format(row['Term'], long_term, row['Catalog'], row['Section'], row['Component']))

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
        '--num_recent', '-n', type=int,
        help='The number of recent courses to return (in reverse order).',
)
def main (faculty_id, cr, num_recent):
    '''Main function call.'''
    
    if not faculty_id:
        raise "Must submit a faculty ID for which to perform the search."
    
    # Guess current term
    TermDescriptions = dpu.get_term_descriptions()
    # Get faculty name for print statement
    Faculty = dpu.get_employee_list()
    first = Faculty[Faculty['Empl ID'] == faculty_id]['First Name'].item()
    last = Faculty[Faculty['Empl ID'] == faculty_id]['Last Name'].item()
    
    # Download historical course info
    course_data = dpu.get_class_data()
    
    # Gather min and max terms
    min_term = course_data['Term'].min()
    min_long_term = TermDescriptions[TermDescriptions['Term'] ==  min_term]['Long Description'].item()
    max_term = course_data['Term'].max()
    max_long_term = TermDescriptions[TermDescriptions['Term'] ==  max_term]['Long Description'].item()

    # Print to user the min and max terms included in search
    # and suggest way to re-download
    print(f'Searching ID {faculty_id} between {min_long_term} and {max_long_term}.')
    print('If you need to increase range of search, download a new NSG_CLASS_SCHEDULE_MULTI_TERM report.')
    
    # Do Pandas filter for ID, Cr
    course_data = course_data[(course_data['ID'] == faculty_id) & (course_data['Role'] == 'PI')].copy(deep=True)
    if cr:
        course_data = course_data[course_data['Catalog'] == cr].copy(deep=True)
    # Test if any records found
    if len(course_data) == 0:
        print('No records found.')
    
    # Sort by term
    course_data.sort_values(by=['Term', 'Catalog', 'Section'], ascending=[False, True, True], inplace=True)
    
    # Get number of unique terms, and sort in reverse order
    unique_terms = course_data['Term'].unique().tolist()
    unique_terms.sort(reverse=True)
    
    if cr:
        print(f'NSG {cr} was last taught by {first} {last} on... (in reverse order)')
    else:
        print(f'Last course taught by {first} {last} on... (in reverse order)')
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
        temp_courses.apply(print_row, axis=1, args=(long_term,))        
        counter += 1

if __name__ == "__main__":
    main()
