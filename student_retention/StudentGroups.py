# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 11:18:45 2017

@author: astachn1
"""

import os
from datetime import datetime
import pandas as pd

def build_File (pathname, file_format='csv', skiprows=0, headerrow=1):
    '''Iterates recursively through a folder and combines fields in files into a dictionary
    pair of filename (key) and dataframe (value). Defaults to csv format. Also accepts
    excel files specified with 'excel'. '''
    
    #print('Scanning: {}'.format(pathname))
    
    #Instantiate dictionary of dfs
    tempdict = {}
    
    for name in os.listdir(pathname):
        subname = os.path.join(pathname, name)
        
        if os.path.isfile(subname):
            #print('Adding: {}'.format(subname))
            #read contents
            base = os.path.basename(subname)
            dfname = os.path.splitext(base)[0]
            if file_format == 'csv':
                tempdict[dfname] = pd.read_csv(subname,delimiter=',',na_values='nan',skiprows=skiprows, header=headerrow)
            elif file_format == 'excel':
                tempdict[dfname] = pd.read_excel(subname, skiprows=skiprows, header=headerrow, na_values='nan', converters={'Emplid':str, 'Empl ID':str, 'ID':str})
            
        elif os.path.isdir(subname):
            build_File(subname)
            
        else:
            pass
    
    return tempdict

def get_latest (pathname):
    '''Scans the folder for all files that match the typical naming conventions.
    For those files, function parses file name to look for date edited,
    then returns the file name with the latest date.'''
    
    #print('Scanning: {}'.format(pathname))
    
    #Set up empty lists
    files = []
    dates = []
    
    for name in os.listdir(pathname):
        subname = os.path.join(pathname, name)
        
        #Checks for standard naming conventions and ignores file fragments
        if os.path.isfile(subname) and 'Student List' in subname and '~$' not in subname:
            #Ignore files that end in '_(2).xlsx'
            if (subname[(subname.find('.')-3):subname.find('.')]) == '(2)':
                pass
            else:
                files.append(subname)
                
                #Parses file names and converts to datetimes
                date = subname[(subname.find('.')-10):subname.find('.')]
                date_time = datetime.strptime(date, '%Y-%m-%d').date()
                dates.append(date_time)
                #print('Adding: {}'.format(subname))
    
    #If only one file, return that one
    if len(files) == 1:
        filename = files[0]
    
    #If multiple files, return the one that contains the latest date
    else:
        latest = max(dates)
        #print(latest.strftime('%m-%d-%Y'))
        for file in files:
            if str(latest.strftime('%Y-%m-%d')) in file:
                filename = file
            
    return pd.read_excel(filename, header=0, converters={'Emplid':str})

# Build Student Groups df
frames = build_File('W:\\csh\\Nursing Administration\\Data Management\\Student Groups\\2017 Audit\\Data', file_format='excel')
groups = pd.concat(frames, ignore_index=True)

# Build Active Student df
studentlist = get_latest('W:\\csh\\Nursing\\Student Records')

# Build Admitted Students df
frames = build_File('W:\\csh\\Nursing Administration\\Data Management\\Student Groups\\2017 Audit\\Admissions Data', file_format='excel', skiprows=0, headerrow=0)
admits = pd.concat(frames, ignore_index=True)

# If ID in student groups not in active or admit student list, mark as such
activelist = studentlist['Emplid'].tolist()
admitcodes = ['ADMT', 'MATR']
admitlist = admits[admits['Program Action'].isin(admitcodes)]['Empl ID'].tolist()
allactive = activelist + admitlist

def Cat_Students (student):
    if student in allactive:
        return 'Active'
    else:
        return 'Terminate'

groups['Action'] = groups.apply(lambda x: Cat_Students(x['ID']), axis=1)

# Output the students to terminate from student groups
groups.to_csv('W:\\csh\\Nursing Administration\\Data Management\\Student Groups\\2017 Audit\\groups.csv')
