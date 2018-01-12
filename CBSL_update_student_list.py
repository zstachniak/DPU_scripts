# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 09:48:17 2017

@author: astachn1
"""

import pandas as pd

# Read files
partner_assignments = pd.read_excel('W:\\csh\\Nursing\\Community Engagement (CbSL)\\Hours Tracking\\Student Partner Assignments.xlsx', header=0, converters={'Emplid':str, 'Term':str})

student_list = pd.read_excel('W:\\csh\\Nursing\\Community Engagement (CbSL)\\Hours Tracking\\Rev V10\\Student List.xlsx', header=0, converters={'Emplid':str, 'Run Term Desc':str, 'Admit Term':str})

terms_list = pd.read_excel('W:\\csh\\Nursing\\Community Engagement (CbSL)\\Hours Tracking\\Rev V10\\Term Descriptions.xlsx', header=0, converters={'Term':str})

# Remove unnecessary term info
partner_assignments = partner_assignments[partner_assignments['Term'] == partner_assignments['Term'].max()]

student_list = student_list[student_list['Run Term Desc'] == student_list['Run Term Desc'].max()]

# Remove unnecessary programs from student list
student_list = student_list[student_list['Maj Desc'].isin(['MS-Generalist Nursing', 'BS-Health Sciences Combined'])]

# Add current term description
student_list['Run Term Description'] = terms_list[terms_list['Term'] == student_list['Run Term Desc'].max()]['Description'].values[0]

# Merge to create new Partner Assignments list based on current students
df = pd.merge(student_list[['Run Term Desc', 'Run Term Description', 'Emplid', 'Student Name', 'Campus']], partner_assignments[['Community Partner', 'Community Partner Location', 'Program (if Applicable)', 'Project', 'Notes', 'Emplid']], how='left', on='Emplid', sort=True, copy=True)

# Change 'Nan' to 'Not Yet Assigned' for Community Partner
df['Community Partner'].fillna(value='Not Yet Assigned', inplace=True);

# Output to excel
df.to_excel('W:\\csh\\Nursing\\Community Engagement (CbSL)\\Hours Tracking\\Rev V10\\Updated_Student_Partner_List.xlsx')
