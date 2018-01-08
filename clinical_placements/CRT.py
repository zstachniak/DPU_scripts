# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 08:22:38 2017

@author: astachn1
"""

import pandas as pd

df = pd.read_csv('W:\\csh\\Nursing Administration\\Clinical Placement Files\\2017-2018\\Winter\\443\\Northwestern CRT.csv', header=0, skiprows=0, converters={'ID':str})
df.columns = ['Name', 'ID', 'Campus', 'Email', 'Phone', 'Reference', 'Units of Interest', 'Description', 'Able to Apply', 'Able to Interview']
df['ID'] = df['ID'].str.zfill(7)

GPA = pd.read_excel('W:\\csh\\Nursing\\Student Records\\Student List 2017-11-02.xlsx', converters={'Emplid':str})

df2 = pd.merge(df, GPA[['Emplid', 'Cum GPA']], how='left', left_on='ID', right_on='Emplid')

df2.to_excel('W:\\csh\\Nursing Administration\\Clinical Placement Files\\2017-2018\\Winter\\443\\Northwestern CRT.xlsx')
