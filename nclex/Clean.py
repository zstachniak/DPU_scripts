# -*- coding: utf-8 -*-
"""
@author: Alexander Stachniak
"""

import os
import pandas as pd
#from passlib.context import CryptContext
import numpy as np
import pickle

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
frames = build_File('W:\\csh\\Nursing Administration\\Data Management\\DataWarehouse\\OG_Data\\ATI_Data\\ATI_Raw_Data', frames, file_format='csv')
#Concatenate all data frames (ignore index)
ATI_raw_data = pd.concat(frames, ignore_index=True)
#Remove full duplicates
ATI_raw_data = pd.DataFrame.drop_duplicates(ATI_raw_data, keep='last')

'''
# Read a list of ATI User IDs and Student IDs into a dictionary. Map
# ATI User IDs to Student IDs in the ATI Data df.
'''
#Read data from csv files (empl ID had to be manually mapped ahead of time)
IDs = pd.read_csv('W:\\csh\\Nursing Administration\\Data Management\\DataWarehouse\\OG_Data\\ATI_Data\\ATIUserID.csv', delimiter=',', na_values='nan', dtype={'emplID':str})
#Create dictionary of key-value pairs
IDdict = dict(zip(IDs['User ID'], IDs['emplID']))
#Assign Student ID the value of dictionary key User ID
ATI_raw_data['Student ID'] = ATI_raw_data['User ID'].map(IDdict)
# Keep only necessary fields
ATI_raw_data = ATI_raw_data[['Assessment', 'Assessment ID', 'Booklet ID', 'Date Taken', 'National Mean', 'National Percentile', 'Proficiency Level', 'Program Mean', 'Program Percentile', 'Score', 'Section', 'Student ID']].copy(deep=True)

'''
# Combine all graduation data (57 Excel files) into a single dataframe.
'''
# Initialize dictionary of dfs
frames = {}
#Build a dictionary of dataframes
frames = build_File('W:\\csh\\Nursing Administration\\Data Management\\DataWarehouse\\OG_Data\\NSG_GRADS', frames, file_format='excel')
#Concatenate all data frames (ignore index)
Grad_data = pd.concat(frames, ignore_index=True)
# Convert Student ID to string format
Grad_data['ID'] = Grad_data['ID'].astype(str)
# Multiple steps to remove trailing ".0" and fill first zero where needed
Grad_data['ID'] = [s.rstrip("0") for s in Grad_data['ID']]
Grad_data['ID'] = [s.rstrip(".") for s in Grad_data['ID']]
Grad_data['ID'] = Grad_data['ID'].str.zfill(7)
# Drop students in the wrong degree program (students who take more than
# one program will be duplicated unnecessarily).
Grad_data = Grad_data[Grad_data['Degree'] == 'MS']
Grad_data = Grad_data[Grad_data['Acad Plan'].isin(['MS-NURSING', 'MS-GENRNSG'])]
Grad_data = Grad_data[Grad_data['Sub-Plan'] != 'ANESTHESIS']

'''
# Quarters are defined by a 4 digit integer that represents a unique value.
# Quarters typically count up by fives, but there are several inconsistencies
# which must be accounted for. Function will return an integer representing
# total number of quarters taken by each student to graduate.
'''
def qtrs(admit, grad):
    if admit >= 860:
        return ((grad - admit)/5) + 1
    elif admit > 620:
        return ((grad - admit)/5)
    elif admit >= 600:
        return ((grad - admit)/5) - 2
    elif admit >=570:
        return ((grad - admit)/5) - 3
    else:
        return ((grad - admit)/5)
Grad_data['Qtrs to Grad'] = Grad_data.apply(lambda x: qtrs(x['Admit Term'], x['Compl Term']), axis=1)
# Keep only necessary fields
Grad_data = Grad_data[['ID', 'Admit Term', 'Compl Term', 'Confer Dt', 'GPA', 'Qtrs to Grad']].copy(deep=True)

'''
# Read all NCLEX data into a dataframe.
'''
# Read data from Excel file
NCLEX_data = pd.read_excel('W:\\csh\\Nursing Administration\\Data Management\\DataWarehouse\\OG_Data\\NCLEX_Results\\NCLEX.xlsx', header=0, converters={'Empl ID':str})
# Fill first zero where needed
NCLEX_data['Empl ID'] = NCLEX_data['Empl ID'].str.zfill(7)
# Drop any repeat test takers from data
NCLEX_data = NCLEX_data[NCLEX_data['Repeater'] != 'Yes']
# Keep only needed fields
NCLEX_data = NCLEX_data[['Empl ID', 'Result']].copy(deep=True)

'''
# Combine all grade data (multiple Excel files) into a single dataframe.
'''
# Initialize dictionary of dfs
frames = {}
#Build a dictionary of dataframes
frames = build_File('W:\\csh\\Nursing Administration\\Data Management\\DataWarehouse\\OG_Data\\Class_Grades', frames, file_format='excel')
#Concatenate all data frames (ignore index)
Grades_data = pd.concat(frames, ignore_index=True)
# Convert Student ID to string format
Grades_data['Student ID'] = Grades_data['Student ID'].astype(str)
# Fill first zero where needed
Grades_data['Student ID'] = Grades_data['Student ID'].str.zfill(7)
# Drop date where no grade exists (often indicates a component of a course)
Grades_data.dropna(axis=0, subset=['Grade'], inplace=True)
# Convert Catalog number to string
Grades_data['Catalog'] = Grades_data['Catalog'].astype(str)
# Strip whitespaces from catalog numbers
Grades_data['Catalog'] = Grades_data['Catalog'].str.strip()
# Convert Faculty ID to string format
Grades_data['Faculty ID'] = Grades_data['ID'].astype(str)
Grades_data.drop(['ID'], axis=1, inplace=True)
# Multiple steps to remove trailing ".0" and fill first zero where needed
Grades_data['Faculty ID'] = [s.rstrip("0") for s in Grades_data['Faculty ID']]
Grades_data['Faculty ID'] = [s.rstrip(".") for s in Grades_data['Faculty ID']]
Grades_data['Faculty ID'] = Grades_data['Faculty ID'].str.zfill(7)
# Make new column to represent GPA from course grade
Grades_data['Course GPA'] = Grades_data['Grd Points'] / Grades_data['Unit Taken']
# Drop students in the wrong program
Grades_data = Grades_data[(Grades_data['Student Major'] == 'MS-Nursing') | (Grades_data['Student Major'] == 'MS-Generalist Nursing')]
# Drop unneeded grades: we do not want to deal with withdrawals and such,
# but want to focus on having one grade for each course
necessary_grades = Grades_data['Grade'].unique().tolist()
unnecessary_grades = ['W', 'R', 'RG', 'WA', 'IN', 'FX', 'PA']
necessary_grades = [x for x in necessary_grades if x not in unnecessary_grades]
Grades_data = Grades_data[Grades_data['Grade'].isin(necessary_grades)]
# Drop where faculty is not primary
Grades_data = Grades_data[Grades_data['Role'] == 'PI']
# Drop unnecessary fields
Grades_data.drop(['Student Name', 'Student Major', 'Acad Group', 'Subject', 'Class Nbr', 'Course ID', 'Subject', 'Grd Points', 'Unit Taken', 'Role', 'Faculty ID'], axis=1, inplace=True)
# Convert Term to string format
Grades_data['Term'] = Grades_data['Term'].astype(str)
# Fill first zero where needed
Grades_data['Term'] = Grades_data['Term'].str.zfill(4)
# Drop grades for Independent study, which won't apply here
mode_types = ['P', 'OL', 'HB']
Grades_data = Grades_data[Grades_data['Mode'].isin(mode_types)]
# Drop duplicate grade data (if a student retook a course, only want last)
Grades_data.sort_values(by=['Term'], inplace=True)
Grades_data.drop_duplicates(subset=['Student ID', 'Catalog'], keep='last', inplace=True)

'''
# Anonymize the data: obfuscate the IDs by assigning an integer value to each unique ID after randomizing the order.
'''
# Generate list of unique IDs
unique_ids = pd.concat([ATI_raw_data['Student ID'], NCLEX_data['Empl ID'], Grad_data['ID'], Grades_data['Student ID']]).unique()
# Shuffle ids to increase anonymity
np.random.shuffle(unique_ids)
# Create dictionary that assigns sequential value to ID
id_dict = {}
counter = 0
for unique_id in unique_ids:
    if unique_id not in id_dict:
        id_dict[unique_id] = counter
        counter += 1
# Save dictionary for future use
with open('ATI\\ids.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(id_dict, f, pickle.HIGHEST_PROTOCOL)

# Map to DFs
ATI_raw_data['Student ID'] = ATI_raw_data['Student ID'].map(id_dict)
NCLEX_data['Empl ID'] = NCLEX_data['Empl ID'].map(id_dict)
Grad_data['ID'] = Grad_data['ID'].map(id_dict)
Grades_data['Student ID'] = Grades_data['Student ID'].map(id_dict)

# Rename some columns
NCLEX_data.rename(columns={'Empl ID': 'Student ID'}, inplace=True)
Grad_data.rename(columns={'ID': 'Student ID'}, inplace=True)

'''
# Output data
'''
# ATI
ATI_raw_data.to_csv('ATI\\PA_data.csv')
ATI_raw_data.to_pickle('ATI\\PA_data.pickle')
# NCLEX_data
NCLEX_data.to_csv('ATI\\NCLEX_data.csv')
NCLEX_data.to_pickle('ATI\\NCLEX_data.pickle')
# Grad_data
Grad_data.to_csv('ATI\\Grad_data.csv')
Grad_data.to_pickle('ATI\\Grad_data.pickle')
# Grades_data
Grades_data.to_csv('ATI\\Grades_data.csv')
Grades_data.to_pickle('ATI\\Grades_data.pickle')
