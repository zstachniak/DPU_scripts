# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 09:42:38 2018

@author: astachn1
"""

import dpu.scripts as dpu
import click





























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
    
    # If a term is not passed, guess current
    if not term:
        term = dpu.guess_current_term()
    # Ensure term is type str
    if type(term) is not str:
        term = str(term)
    
    # If a campus is not passed, use both
    if not campus:
        campus = ['LPC', 'RFU']
    # Convert string to list so code execution can be parallel
    else:
        campus = [campus]
    
    # Define schedule starting path
    path = 'W:/csh/Nursing/Schedules'
    
    # Get schedule
    schedule = dpu.get_schedule(path, term)
    
    
    

if __name__ == "__main__":
    main()
