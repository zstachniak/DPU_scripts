# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 09:42:38 2018

@author: astachn1
"""

import dpu.scripts as dpu
import click
import os
import pandas as pd

def filter_schedule_for_cln (df, program_list):
    '''Filters schedule to only include clinical courses within a program.'''
    # Gather list of clinical courses
    cln_list = df[(df['Program'].isin(program_list)) & (df['Type'] == 'PRA')]['Cr'].unique().tolist()
    # Add lonely 301
    cln_list.append('301')
    # Keep only clinical courses
    df = df[(df['Program'].isin(program_list)) & (df['Cr'].isin(cln_list)) & (~df['Type'].isin(['LEC', 'COORD']))].copy(deep=True)
    return df

cln_progression = {'301': '332',
                   '302': '301',
                   '303': '302',
                   '307': '303', 
                   '440': '307',
                   '441': '307',
                   '442': '440',
                   '472': '440',
                   '443': '442'}

def get_prev_enrl (row, df, prev_term):
    '''docstring'''
    # First, determine appropriate campus from section number
    campus = row['Sec'][0]
    prev_cr = cln_progression.get(row['Cr'], None)
    # Get previous course
    while int(row['Term']) - int(prev_term) != 5:
        prev_cr = cln_progression.get(prev_cr, None)
        prev_term = str(int(prev_term) + 5)
    # Return count
    return len(df[(df['Cr'] == prev_cr) & (df['Sec'].apply(lambda x: x.startswith(campus)))])

#############################################################
# Main
#############################################################
@click.command()
@click.option(
        '--term',
        help='The 4 digit term for which you want a report, e.g. "1005".',
)
@click.option(
        '--campus',
        help='Either "LPC" or "RFU".',
)
def main (term, campus):
    '''docstring'''
    
    # Get Term Descriptions
    TermDescriptions = dpu.get_term_descriptions()
    
    # If a term is not passed, guess current
    if not term:
        term = dpu.guess_current_term(TermDescriptions)
    # Ensure term is type str
    if type(term) is not str:
        term = str(term)
    
    # If a campus is not passed, use both
    if not campus:
        campus = ['MENP', 'RFU']
    else:
        # In schedule, 'LPC' is represented by 'MENP'
        if campus == 'LPC':
            campus = 'MENP'
        # Convert string to list so code execution can be parallel
        campus = [campus]
    
    # Get schedule
    schedule = dpu.get_schedule(term, TermDescriptions)
    
    # Filter out non-clinical courses
    schedule = filter_schedule_for_cln(schedule, campus)
    
    # Reorder and filter out unncessary data
    schedule = schedule[['Term', 'Cr', 'Sec', 'Time', 'Clinical Site', 'Unit', 'Max Cap', 'Confirmed', 'Faculty']]
    
    # Get previous clinical roster
    prev_term = str(int(term) - 5)
    try:
        prev_cln_roster = dpu.get_cln(prev_term, TermDescriptions)
    except:
        prev_term = str(int(prev_term) - 5)
        prev_cln_roster = dpu.get_cln(prev_term, TermDescriptions)
    # Get previous enrollment numbers
    schedule['Prev Cr Enrl'] = schedule.apply(get_prev_enrl, axis=1, args=(prev_cln_roster, prev_term))
    
    # Get output folder
    output_folder = os.path.join(dpu.get_dir_of_schedule(term, TermDescriptions), 'Charts')
    
    # Output file name
    output_file = os.path.join(output_folder, 'Clinical Review {}.xlsx'.format(term))
    
    # Output result
    schedule.to_excel(output_file, index=False)

if __name__ == "__main__":
    main()
