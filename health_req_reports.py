# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 09:23:14 2018

@author: astachn1
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from fuzzywuzzy import fuzz, process
import re
import dpu.scripts as dpu

folders = {'reports': 'W:\\csh\\Nursing\\Clinical Placements\\Castle Branch and Health Requirements\\Reporting\\Downloaded Reports',
           'students': 'W:\\csh\\Nursing\\Student Records',
           'schedule': 'W:\\csh\\Nursing\\Schedules',
           'clinical': 'W:\\csh\\Nursing Administration\\Clinical Placement Files',
           }

# Get the latest reports
noncompliant_files = dpu.get_latest(folders['reports'], 'Noncompliant', num_files=2)
compliant_files = dpu.get_latest(folders['reports'], 'Compliant', num_files=2)

files = {'students': dpu.get_latest(folders['students'], 'Student List'),
         'faculty': 'W:\\csh\\Nursing\\Faculty\\Employee List.xlsx',
         'terms': 'W:\\csh\\Nursing\\Schedules\\Term Descriptions.xlsx',
         'noncompliant_curr': noncompliant_files[0],
         'noncompliant_prev': noncompliant_files[1],
         'compliant_curr': compliant_files[0],
         'compliant_prev': compliant_files[1],
         }


to_datetime = lambda d: datetime.strptime(d, '%m/%d/%Y')

# Get the two most recent reports
noncompliant_curr = pd.read_csv(files['noncompliant_curr'], header=0, converters={'Order Submission Date': to_datetime})
noncompliant_prev = pd.read_csv(files['noncompliant_prev'], header=0, converters={'Order Submission Date': to_datetime})
# Get a change log
changelog = pd.merge(noncompliant_curr, noncompliant_prev, on=noncompliant_curr.columns.tolist(), how='outer', indicator=True).query("_merge != 'both'").drop('_merge', 1)




compliant_curr = pd.read_csv(files['compliant_curr'], header=0, converters={'Order Submission Date': to_datetime, 'Date of Compliance': to_datetime})
compliant_prev = pd.read_csv(files['compliant_prev'], header=0, converters={'Order Submission Date': to_datetime, 'Date of Compliance': to_datetime})


# Get the latest student list
students = pd.read_excel(files['students'], header=0, converters={'Emplid':str, 'Admit Term':str, 'Latest Term Enrl': str, 'Run Term': str,})
students.drop_duplicates(subset='Emplid', inplace=True)
# Get the faculty list
faculty = pd.read_excel(files['faculty'], header=0, converters={'Empl ID': str,})
# Get term descriptions
TermDescriptions = dpu.get_term_descriptions()
# Get current term
current_term = dpu.guess_current_term(TermDescriptions)
# Get clinical roster and schedule
schedule = dpu.get_schedule(current_term, TermDescriptions)
clinical_roster = dpu.get_cln(current_term, TermDescriptions)


# Drop unneeded columns
clinical_roster.drop(labels=clinical_roster.columns[12:], axis=1, inplace=True)
clinical_roster.drop(labels=clinical_roster.columns[8:11], axis=1, inplace=True)
# Gather list of duplicates (i.e., student has more than one clinical course)
dupe_truth_value = clinical_roster.duplicated(subset=['Student ID'], keep='first')
dupes = clinical_roster.loc[dupe_truth_value]
dupes.drop(labels=['Term'], axis=1, inplace=True)
# Drop duplicates from original list
clinical_roster.drop_duplicates(subset=['Student ID'], keep='first', inplace=True)
# Merge two lists 
clinical_roster = pd.merge(clinical_roster, dupes, how='left', on='Student ID', suffixes=['_1', '_2'])
# Put student ID at start
clinical_roster.insert(0, 'Emplid', clinical_roster['Student ID'])
clinical_roster.drop(labels=['Student ID'], axis=1, inplace=True)
# Merge with student data
clinical_roster = pd.merge(clinical_roster, students[['Emplid', 'Last Name', 'First Name', 'Campus', 'Email', 'Best Phone']], how='left', on='Emplid')
# Reorder columns
cols = ['Emplid', 'Last Name', 'First Name', 'Campus', 'Email', 'Best Phone', 'Term', 'Cr_1', 'Sec_1', 'Clinical Site_1', 'Unit_1', 'Date_1', 'Time_1', 'Instructor_1', 'Cr_2', 'Sec_2', 'Clinical Site_2', 'Unit_2', 'Date_2', 'Time_2', 'Instructor_2']
clinical_roster = clinical_roster.reindex(columns= cols)


# Get instructor data










student_trackers = [x for x in compliant_curr['To-Do List Name'].unique() if 'DE69' in x and 'Disclosure & Authorization' not in x]
faculty_trackers = [x for x in compliant_curr['To-Do List Name'].unique() if 'DE69' in x and 'Disclosure & Authorization' not in x]


all_trackers = np.concatenate((compliant_curr['To-Do List Name'].unique(), noncompliant_curr['To-Do List Name'].unique()))
all_trackers = np.unique(all_trackers)



dna_trackers = []
student_trackers = []
faculty_trackers = []
for tracker in all_trackers:
    if 'Disclosure & Authorization' in tracker:
        dna_trackers.append(tracker)
    elif 'DE34' in tracker:
        faculty_trackers.append(tracker)
    elif 'DE69' in tracker:
        student_trackers.append(tracker)

student_trackers = [x for x in all_trackers if 'DE34' not in x]









t = compliant_curr[compliant_curr['To-Do List Name'].isin(student_trackers)]
u = noncompliant_curr[noncompliant_curr['To-Do List Name'].isin(student_trackers)]


q = pd.concat([t, u])
q[q.duplicated(subset=['Email Address'], keep=False)]



















# Sort by To-Do List and then by Order Submission Date
q.sort_values(by=['To-Do List Status', 'Order Submission Date'], ascending=False, inplace=True)
# In case of duplicates, first we would drop "compliant"
# Second we would drop the least recent order
q.drop_duplicates(subset=['Email Address'], keep='first', inplace=True)



test = pd.merge(students, q, how='left', left_on='Email', right_on='Email Address')


# do a merge on email address

test = pd.merge(students, t, how='left', left_on='Email', right_on='Email Address')

test = pd.merge(test, u, how='left', left_on='Email', right_on='Email Address')


test[test.duplicated(subset=['Emplid'], keep=False)]



# try merging by email
# then try matching by email with just D & A
# if still none, try matching with fuzzywuzzy

e = 'leanderson8@outlook.com'
e = 'christinebalderas91@gmail.com'

matches = q[q['Email Address'] == e]


students[students['Email'] == e]['Emplid'].item()

def match_student (row):
    '''docstring'''
    # First, try a true match of email address
    try:
        Emplid = students[students['Email'] == row['Email Address']]['Emplid'].item()
    except:
        # Second, try a true match of first and last name
        try:
            Emplid = students[(students['Last Name'].lower() == row['Last Name'].lower()) & (students['First Name'].lower() == row['First Name'].lower())]['Emplid'].item()
        except:
            Emplid = None
    return Emplid
    
    

    


        
q = pd.concat([t, u])
q = q[['First Name', 'Last Name', 'Email Address']]
q.drop_duplicates(subset=['Email Address'], inplace=True)
q['Emplid'] = q.apply(match_student, axis=1)





cln_ids = clinical_roster['Student ID'].unique()


def check_clinical (row):
    if row['Emplid'] in cln_ids:
        return True
    else:
        return False
    
    try:
        #site = clinical_roster[clinical_roster['Student ID'] == row['Emplid']]['Clinical Site'].item()
        site = clinical_roster.loc[clinical_roster['Student ID'] == row['Emplid'], 'Clinical Site'].values
        return site
    except:
        pass


test = students.copy(deep=True)
test['cln'] = test.apply(check_clinical, axis=1)

test2 = test[test['cln'] == True]




all_student_trackers = [x for x in all_trackers if 'DE34' not in x]

t = compliant_curr[compliant_curr['To-Do List Name'].isin(all_student_trackers)]
u = noncompliant_curr[noncompliant_curr['To-Do List Name'].isin(all_student_trackers)]

q = pd.concat([t, u])
q['Emplid'] = q.apply(match_student, axis=1)


def check_compliance (row):
    if row['Emplid'] in q['Emplid']:
        result = q[(q['Emplid'] == row['Emplid']) & (q['To-Do List Name'].isin(student_trackers)) & (~q['To-Do List Status'].isin(['Compliant', 'Complete']))]
        if not result.empty:
            status = 'Noncompliant'
            num_reqs_due = result['Number of Requirements Incomplete']
            reqs_due = result['Requirements Incomplete']
        else:
            result = q[(q['Emplid'] == row['Emplid']) & (q['To-Do List Name'].isin(student_trackers)) & (q['To-Do List Status'].isin(['Compliant', 'Complete']))]
        if not result.empty:
            status = 'Compliant'
            num_reqs_due = 0
            reqs_due = 'N/A'
        else:
            result = q[(q['Emplid'] == row['Emplid']) & (q['To-Do List Name'].isin(dna_trackers))]
        if not result.empty:
            status = 'Noncompliant'
            num_reqs_due = 1
            reqs_due = 'Never set up health requirement tracker'
    else:
        status = 'Unknown'
        num_reqs_due = 1
        reqs_due = 'Locate tracker'
        
    return pd.Series({'Status': status, 'Num Reqs Due': num_reqs_due, 'Reqs Due': reqs_due})
        
    
test2[['Status', 'Num Reqs Due', 'Reqs Due']] = test2.apply(check_compliance, axis=1)








stuff = clinical_roster[clinical_roster['Student ID'] == '0695385']['Clinical Site']
stuff = clinical_roster[clinical_roster['Student ID'] == '1824625']['Clinical Site']



'''
start with student list

ask if student is in clinical 
    gather this data
    
if not in clinical, ignore?

if in clinical:
    check if non-compliant
    
    if not, check if compliant
    
    if not, check if D & A
    
    if not, raise flag?


'''





    
    























    


# Clean data
    # Drop classifications that don't matter
        # careful - DNP students could be stuck as alumni or inactive?
    # Separate into student/faculty based on account name
    # Attempt to connect to real people
        # start with student list and attempt to perfect match emails
        # for those who don't attempt match first and last name
        # for those who don't do fuzzy match on name and keep if > threshold
        # finally, consider failure
        # same with faculty, but start with secondary email, then try primary
        
# Add data
    # for students, add their clinical course and location for curr and prev
        # program, start date, advisor
        # maybe add instructor info?
    # faculty, add teaching load current and prev
    # add phone numbers
    
# Consider graphing?
    # number of 

# to highlight new data, grab the most recent report as well and do a full
    # removal of full duplicates. should be left with only the new stuff






















