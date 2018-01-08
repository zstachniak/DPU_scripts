# -*- coding: utf-8 -*-
"""
Created on Thu May 18 16:21:44 2017

@author: astachn1
"""

import os
import pandas as pd

def build_File (pathname, tempdict, file_format='csv'):
    '''Iterates recursively through a folder and combines files into a
    dictionary pair of filename (key) and dataframe (value). Works with
    csv and excel formats.'''
    # Iterate through objects in pathname folder
    #print('Scanning: {}'.format(pathname))
    for name in os.listdir(pathname):
        subname = os.path.join(pathname, name)
        # If object in folder is a file
        if os.path.isfile(subname):
            #print('Adding: {}'.format(subname))
            # Path naming
            base = os.path.basename(subname)
            dfname = os.path.splitext(base)[0]
            # Read file using user-defined format
            if file_format == 'csv':
                tempdict[dfname] = pd.read_csv(subname,delimiter=',',na_values='nan',)
            elif file_format == 'excel':
                tempdict[dfname] = pd.read_excel(subname,skiprows=0,header=1,na_values='nan')
        # If object in folder is a folder, recursive call to function
        elif os.path.isdir(subname):
            build_File(subname, tempdict)
        else:
            pass
    return tempdict

'''
# Combine all ATI data (1190 .csv files) into a single Pandas dataframe
# containing 608,023 observations over 24 features (data is in long format).
# After removal of full duplicates, 339,364 observations remain.
'''
# Initialize dictionary of dfs
frames = {}
#Build a dictionary of dataframes
frames = build_File('Raw Data', frames, file_format='excel')
# Concatenate all data frames (ignore index)
df = pd.concat(frames, ignore_index=True)
# Remove full duplicates
df = pd.DataFrame.drop_duplicates(df, keep='last')
# Write to file
df.to_excel('2016 Clinical Roster.xlsx', columns=['Course', 'Term', 'Term.1', 'Class Nbr', 'Emplid', 'Last', 'First Name', 'Class Descr'])



student_list = df.drop_duplicates(subset=['Emplid'])
student_list.to_excel('2016 List of Students.xlsx', columns=['Emplid', 'Last', 'First Name'])


