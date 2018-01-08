# -*- coding: utf-8 -*-
"""
Created on Mon May 15 09:10:10 2017

@author: astachn1
"""

import pandas as pd

# Read in community survey choices
df = pd.read_excel('W:\\csh\\Nursing Administration\\Clinical Placement Files\\2017-2018\\Winter\\443\\443 Preferences Cleaned.xlsx', header=0, converters={'ID':str})
df['ID'] = df['ID'].str.zfill(7)

df.columns

units = list(df.columns[7:14])

# Function to return the item with the correct rank
def rank_it (row):      
    for item in units:
        if row[item] == rank:
            return item

for rank in range(1,8):
    df['Ranked Unit {}'.format(rank)] = df.apply(rank_it, axis=1)
    
df.to_excel('W:\\csh\\Nursing Administration\\Clinical Placement Files\\2017-2018\\Winter\\443\\Updated Survey Data.xlsx')
