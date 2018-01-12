# -*- coding: utf-8 -*-
"""
Created on Tue May 30 09:09:49 2017

@author: astachn1
"""

import pandas as pd
import numpy as np

# Read in community survey choices
admits = pd.read_excel('W:\\csh\\Nursing\\Student Records\\Admitted Student Lists\\DNP Admits.xlsx', header=0, converters={'Empl ID':str})
admits['Empl ID'] = admits['Empl ID'].str.zfill(7)
admits.drop_duplicates(subset='Empl ID', inplace=True)

current = pd.read_excel('W:\\csh\\Nursing\\Student Records\\Student List 2017-03-27.xlsx', header=0, converters={'Emplid':str})
current['Emplid'] = current['Emplid'].str.zfill(7)
current.drop_duplicates(subset='Emplid', inplace=True)

graduates = pd.read_excel('W:\\csh\\Nursing\\Student Records\\Graduate Contact List.xlsx', header=0, converters={'Emplid':str})
graduates['Emplid'] = graduates['Emplid'].str.zfill(7)
graduates = graduates[graduates['Maj Desc'] ==  'Doctor of Nursing Practice']
graduates.drop_duplicates(subset='Emplid', inplace=True)


retention = admits.merge(graduates[['Emplid', 'Graduation Date']], how='left', left_on='Empl ID', right_on='Emplid')
retention.drop(['Emplid'], axis=1, inplace=True)

retention = retention.merge(current[['Emplid', 'Latest Term Enrl']], how='left', left_on='Empl ID', right_on='Emplid')
retention.drop(['Emplid'], axis=1, inplace=True)

def status (row):
    if row['Graduation Date'] is not pd.NaT:
        return 'Graduated'
    elif row['Latest Term Enrl'] is not np.nan:
        return 'Current Student'
    else:
        return 'Inactive'

retention['Status'] = retention.apply(status, axis=1)

retention['Status'].value_counts()

retention.to_excel('W:\\csh\\Nursing\\Student Records\\Admitted Student Lists\\DNP Retention.xlsx')

