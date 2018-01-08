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
frames = build_File('OG_Data\ATI_Data\ATI_Raw_Data', frames, file_format='csv')
#Concatenate all data frames (ignore index)
ATI_raw_data = pd.concat(frames, ignore_index=True)
#Remove full duplicates
ATI_raw_data = pd.DataFrame.drop_duplicates(ATI_raw_data, keep='last')

'''
# Read a list of ATI User IDs and Student IDs into a dictionary. Map
# ATI User IDs to Student IDs in the ATI Data df.
'''
#Read data from csv files (empl ID had to be manually mapped ahead of time)
IDs = pd.read_csv('OG_Data\ATI_Data\ATIUserID.csv', delimiter=',', na_values='nan', dtype={'emplID':str})
#Create dictionary of key-value pairs
IDdict = dict(zip(IDs['User ID'], IDs['emplID']))
#Assign Student ID the value of dictionary key User ID
ATI_raw_data['Student ID'] = ATI_raw_data['User ID'].map(IDdict)

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

'''
# Read all NCLEX data into a dataframe.
'''
# Read data from Excel file
NCLEX_data = pd.read_excel('OG_Data\\NCLEX_Results\\NCLEX.xlsx', header=0, converters={'Empl ID':str})
# Fill first zero where needed
NCLEX_data['Empl ID'] = NCLEX_data['Empl ID'].str.zfill(7)
# Drop unnecessary fields
NCLEX_data.drop(['Last Name', 'First Name', 'Candidate ID', 'Time Delivered', 'Year', 'Quarter'],axis=1,inplace=True)
# Add days elapsed since graduation
NCLEX_data['Date Delivered'] = pd.to_datetime(NCLEX_data['Date Delivered'])
NCLEX_data['Days Elapsed'] = NCLEX_data['Date Delivered'] - NCLEX_data['Graduation Date']
NCLEX_data['Days Elapsed'] = NCLEX_data['Days Elapsed'].dt.days
# Drop any repeat test takers from data
NCLEX_data = NCLEX_data[NCLEX_data['Repeater'] != 'Yes']

#'''
## Combine the NCLEX and Grad data into a single dataframe of 908 observations.
#'''
#Student_data = pd.merge(NCLEX_data[['Empl ID', 'Result', 'Days Elapsed']], Grad_data[['ID', 'GPA', 'Qtrs to Grad', 'Degree']], how='inner', left_on='Empl ID', right_on='ID', sort=True, copy=True)
## Drop students in the wrong degree program (students who take more than
## one program will be duplicated unnecessarily).
#Student_data = Student_data[Student_data['Degree'] == 'MS']
#
#'''
## Combine Student_data with ATI_raw_data
#'''
#ATI = pd.merge(Student_data, ATI_raw_data, how='inner', left_on='Empl ID', right_on='Student ID', sort=True, copy=True)
## Drop unnecessary fiels
#ATI.drop(['ID', 'At Risk Score', 'Benchmark', 'Benchmark Type', 'Class', 'First Name', 'High Risk Score', 'Institution', 'Last Name', 'Program', 'Standard Score', 'Student ID', 'User ID', 'User Name'], axis=1, inplace=True)

'''
# Combine all grade data (58 Excel files) into a single dataframe.
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

#'''
## Reshape Grades_data from long to wide format, but avoid merging at this time
#'''
#Grades_data_wide = Grades_data.pivot(index='Student ID', columns='Catalog')
## Rename columns
##Grades_data_wide.columns = [' '.join(col).strip() for col in Grades_data_wide.columns.values]

'''
# Create a list of all Student IDs to be considered.
# Remove unnecessary observations from all dataframes
'''
ID_data = pd.merge(NCLEX_data[['Empl ID']], ATI_raw_data[['Student ID']], how='inner', left_on='Empl ID', right_on='Student ID', sort=True, copy=True)
ID_data.drop_duplicates(inplace=True)
ID_data.drop(['Empl ID'], axis=1, inplace=True)
# Remove unnecessary data
ATI_raw_data = ATI_raw_data[ATI_raw_data['Student ID'].isin(ID_data['Student ID'])]
NCLEX_data = NCLEX_data[NCLEX_data['Empl ID'].isin(ID_data['Student ID'])]
Grad_data = Grad_data[Grad_data['ID'].isin(ID_data['Student ID'])]
Grades_data = Grades_data[Grades_data['Student ID'].isin(ID_data['Student ID'])]

'''
# Combine NCLEX_data (only the target variable) with ATI_raw_data
'''
ATI = pd.merge(NCLEX_data[['Empl ID', 'Result']], ATI_raw_data, how='inner', left_on='Empl ID', right_on='Student ID', sort=True, copy=True)
# Drop unnecessary fiels
ATI.drop(['Empl ID', 'At Risk Score', 'Benchmark', 'Benchmark Type', 'First Name', 'High Risk Score', 'Institution', 'Last Name', 'Program', 'Standard Score', 'User ID', 'User Name'], axis=1, inplace=True)

'''
# Combine the NCLEX and Grad data into a single dataframe.
'''
Student_data = pd.merge(NCLEX_data[['Empl ID', 'Result', 'Days Elapsed']], Grad_data[['ID', 'GPA', 'Qtrs to Grad', 'Degree']], how='inner', left_on='Empl ID', right_on='ID', sort=True, copy=True)
Student_data.drop(['ID'], axis=1, inplace=True)

'''
# Anonymize the data

Originally, I determined to use a complicated encryption hashing and salting technique to anonymize the data, but I have determined that the level of encryption is unnecessary. Instead, I will simply obfuscate the IDs by assigning an integer value to each unique ID after randomizing the order. This is a much simpler functionality, and will make working with the data easier without sacrificing anonymity.
'''
## Set the privacy_context
#privacy_context = CryptContext(schemes = ['pbkdf2_sha512'])
## Save the privacy_context
#with open('ATI\\context.txt', 'w') as outfile:
#    outfile.write(privacy_context.to_string())
## Load the privacy_context
##privacy_context = CryptContext.from_path('ATI\\context.txt')
## Anonymize student IDs
#ATI['ID'] = ATI.apply(lambda x: privacy_context.hash(x['Empl ID']), axis=1)
## Drop unnecessary fields
#ATI.drop(['Empl ID'], axis=1, inplace=True)

# Generate list of unique IDs
unique_ids = ID_data['Student ID'].unique()
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
ATI['Student ID'] = ATI['Student ID'].map(id_dict)
Student_data['Student ID'] = Student_data['Empl ID'].map(id_dict)
Student_data.drop(['Empl ID'], axis=1, inplace=True)
Grades_data['Student ID'] = Grades_data['Student ID'].map(id_dict)

'''
# Output data
'''
# ATI
ATI.to_csv('ATI\\ATI_data.csv')
ATI.to_pickle('ATI\\ATI_data.pickle')
# Student_data
Student_data.to_csv('ATI\\Student_data.csv')
Student_data.to_pickle('ATI\\Student_data.pickle')
# Grades_data
Grades_data.to_csv('ATI\\Grades_data.csv')
Grades_data.to_pickle('ATI\\Grades_data.pickle')
