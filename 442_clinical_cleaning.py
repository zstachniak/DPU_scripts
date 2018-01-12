# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 11:05:30 2017

@author: astachn1
"""

import pandas as pd

# Read in community survey choices
community = pd.read_excel('W:\\csh\\Nursing Administration\\Clinical Placement Files\\2017-2018\\Fall\\442\\442ResponseDataCleaned.xlsx', header=0, converters={'Empl ID':str})
community['Empl ID'] = community['Empl ID'].str.zfill(7)

# Read in clinical placements
clinical = pd.read_excel('W:\\csh\\Nursing Administration\\Clinical Placement Files\\2017-2018\\Fall\\Clinical Roster Fall 2017.xlsx', header=0, converters={'Cr':str, 'Sec':str, 'Empl ID':str})
clinical['Empl ID'] = clinical['Empl ID'].str.zfill(7)
# Keep only 472 placements
clinical = clinical[clinical['Cr'] == '472']
# Keep only day of the week
Days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
def day (row):
    snip = row['Time'][:2]
    for day in Days:
        if snip in day:
            return day
clinical['472 Day'] = clinical.apply(day, axis=1)

# Merge the two dataframes
df = pd.merge(community, clinical[['472 Day', 'Empl ID']], how='inner', on='Empl ID')

'''
# Rank the days and sites (put in a more useable format)
# For this to work, need to rename columns to the names of the days & sites.
'''
# Function to return the item with the correct rank
def rank_it (row):      
    for item in items_list:
        if row[item] == rank:
            return item

# Set days of week
items_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
# Apply the function to rank days of the week
for rank in range(1,6):
    df['Ranked Day {}'.format(rank)] = df.apply(rank_it, axis=1)

# First read all selected sites from the dataframe, add to a set
# (no duplicates), convert to a list, and then sort.
sites_list = set()
def find_sites(row):
    sites = row['Five Sites']
    sites = sites.split(',')
    for site in sites:
        sites_list.add(site)
# Replace NaN with None in 'Five Sites'
df['Five Sites'].fillna(value='None', inplace=True)
# Apply the function to find all sites
df.apply(find_sites, axis=1)
# Convert to list
items_list = list(sites_list)
# Remove None value from list
items_list.remove('None')
# Apply the function to rank sites
for rank in range(1,6):
    df['Ranked Site {}'.format(rank)] = df.apply(rank_it, axis=1)
        
# Output file
df.to_excel('W:\\csh\\Nursing Administration\\Clinical Placement Files\\2017-2018\\Fall\\442\\ResponseDataFinal.xlsx')
