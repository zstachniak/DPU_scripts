# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 10:12:10 2016

@author: astachn1
"""

import os
import pandas as pd

#Instantiate dictionary of dfs (must be done globally)
tempdict = {}

def build_File (pathname, file_format='csv'):
    '''Iterates recursively through a folder and combines fields in files into a dictionary
    pair of filename (key) and dataframe (value). Defaults to csv format. Also accepts
    excel files specified with 'excel'. '''
    
    #print('Scanning: {}'.format(pathname))
    
    for name in os.listdir(pathname):
        subname = os.path.join(pathname, name)
        
        if os.path.isfile(subname):
            #print('Adding: {}'.format(subname))
            #read contents
            base = os.path.basename(subname)
            dfname = os.path.splitext(base)[0]
            if file_format == 'csv':
                tempdict[dfname] = pd.read_csv(subname,delimiter=',',na_values='nan',)
            elif file_format == 'excel':
                tempdict[dfname] = pd.read_excel(subname,skiprows=0,header=1,na_values='nan')
            
        elif os.path.isdir(subname):
            build_File(subname)
            
        else:
            pass
    
    return tempdict
    
###############################################################################
#Use the build_File function to iterate through all downloaded ATI data and
#concatenate to a single dataframe. Then, export to CSV.
###############################################################################
#Build a dictionary of dataframes
frames = build_File('W:\\csh\\Nursing Administration\\Data Management\\ATI\\ATI_Raw_Data')
#Concatenate all data frames (ignore index)
result = pd.concat(frames, ignore_index=True)
#Remove full duplicates
result = pd.DataFrame.drop_duplicates(result, keep='last')
#Write to csv
#result.to_csv('W:\\csh\\Nursing Administration\\Data Management\\DataWarehouse\\Testing\\ATI_Temp.csv')

###############################################################################
#Create list of all ATI user IDs with other identifying information. Remove 
#duplicates, then export as CSV. Will manually review and update all Empl IDs.
###############################################################################
#Pull all possible identifiers
UserID = result[['User ID', 'User Name', 'Student ID', 'Last Name', 'First Name']]
#Remove duplicates based on User ID
UserID = UserID.drop_duplicates(['User ID'], keep='last')
#Output for manual editing
UserID.to_csv('W:\\csh\\Nursing Administration\\Data Management\\DataWarehouse\\Testing\\ATIUserID.csv')

###############################################################################
#Read list of ATI user IDs and Empl IDs into a dictionary. Iterate through ATI
#testing data. For every ATI user ID (key), set the student id to Empl ID 
#(value). Remove unnecessary columns. Output cleaned file.
###############################################################################
#Read data from csv files (note that the empl ID had to be manually mapped)
ID = pd.read_csv('W:\\csh\\Nursing Administration\\Data Management\\DataWarehouse\\Testing\\ATIUserID.csv',delimiter=',',na_values='nan', dtype={'emplID':str})
#Create dictionary of key-value pairs
IDdict = dict(zip(ID['User ID'], ID['emplID']))
#Assign Student ID the value of dictionary key User ID
result['Student ID'] = result['User ID'].map(IDdict)
#Write to csv
result.to_csv('W:\\csh\\Nursing Administration\\Data Management\\DataWarehouse\\Testing\\ATI.csv', sep=',', encoding='utf-8')

###############################################################################
#Use the build_File function to iterate through all downloaded grad data and
#concatenate to a single dataframe. Then, export to CSV.
###############################################################################
#Build a dictionary of dataframes
frames = build_File('W:\\csh\\Nursing Administration\\Data Management\\DataWarehouse\\OG_Data\\NSG_GRADS','excel')
#Concatenate all data frames (ignore index)
result = pd.concat(frames, ignore_index=True)
#Remove full duplicates
result = pd.DataFrame.drop_duplicates(result, keep='last')
#Write to csv
result.to_csv('W:\\csh\\Nursing Administration\\Data Management\\DataWarehouse\\Testing\\Grad.csv')
