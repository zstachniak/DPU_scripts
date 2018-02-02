# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 12:37:01 2018

@author: astachn1
"""
import numpy as np
import os
import click
from datetime import datetime
import dpu.scripts as dpu
from dpu.file_locator import FileLocator

@click.command()
@click.option(
        '--term', '-t', type=str,
        help='Term for which to gather faculty emails',
)
@click.option(
        '--cr', '-cr', type=str,
        help='Course list which to gather faculty emails',
)
@click.option(
        '--cr_type', '-crt', type=str,
        help='Course type for which to gather faculty emails',
)
@click.option(
        '--prog', '-p', type=str,
        help='Program for which to gather faculty emails',
)
@click.option(
        '--track', '-tr', type=str,
        help='Faculty track for which to gather faculty emails',
)
@click.option(
        '--email_type', '-e', type=str,
        help='Email type list for which to gather faculty emails',
)
@click.option(
        '--f_name', '-f', '-o', type=str,
        help='File name.',
)
def main(term, cr, cr_type, prog, track, email_type, f_name):
    '''Main function.'''
    # Initialize FileLocator
    FL = FileLocator()
    # Today's date
    today = datetime.strftime(datetime.today(), '%Y-%m-%d')
    # If no term provided, guess the current term
    if not term:
        terms = dpu.get_term_descriptions()
        term = dpu.guess_current_term(terms)
    # Default file name
    if not f_name:
        f_name = 'faculty'
    # Get emails
    emails = dpu.get_faculty_emails(term, course=cr, course_type=cr_type, program=prog, faculty_track=track, email_type=email_type)
    # Set output location
    output_folder = os.path.abspath(os.path.join(os.sep, FL.faculty, 'temp_email_lists'))
    # Ensure location exists
    dpu.ensure_dir(output_folder)
    # Set output file name
    output_file = os.path.abspath(os.path.join(os.sep, output_folder, 'emails_{0}_{1}.csv'.format(f_name, today)))
    # Output
    np.savetxt(output_file, emails, delimiter=";", fmt='%s')

if __name__ == "__main__":
    main()
