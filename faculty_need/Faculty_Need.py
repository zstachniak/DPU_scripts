# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 16:47:02 2017

@author: astachn1
"""

import os
from datetime import datetime
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

#############################################################
# Functions
#############################################################
def get_dir_paths (pathname, pattern, ignore_list):
    '''Given a starting pathname and a pattern, function will return
    and subdirectories that match the regex pattern.'''
    dirs = []
    # Iterate through subfiles
    for name in os.listdir(pathname):
        subname = os.path.join(pathname, name)
        # Checks if subpath is a directory and that it matches pattern
        if os.path.isdir(subname) and re.fullmatch(pattern, name) and name not in ignore_list:
            dirs.append(subname)
    return dirs

def get_latest (pathname):
    '''Scans the folder for all files that match typical naming conventions.
    For those files, function parses file name to look for date edited,
    then returns the file name with the latest date.'''
    
    #print('Scanning: {}'.format(pathname))
    
    #Set up empty lists
    files = []
    dates = []
    
    for name in os.listdir(pathname):
        subname = os.path.join(pathname, name)
        
        #Checks for standard naming conventions and ignores file fragments
        if os.path.isfile(subname) and 'Fiscal Schedule' in subname and '~$' not in subname:
            #Ignore files that end in '_(2).xlsx'
            if (subname[(subname.find('.')-3):subname.find('.')]) == '(2)':
                pass
            else:
                files.append(subname)
                
                #Parses file names and converts to datetimes
                date = subname[(subname.find('.')-10):subname.find('.')]
                date_time = datetime.strptime(date, '%m-%d-%Y').date()
                dates.append(date_time)
                #print('Adding: {}'.format(subname))
    
    #If only one file, return that one
    if len(files) == 1:
        return files[0]
    
    #If multiple files, return the one that contains the latest date
    else:
        latest = max(dates)
        #print(latest.strftime('%m-%d-%Y'))
        for file in files:
            if str(latest.strftime('%m-%d-%Y')) in file:
                return file




def cat_sched (file):
    Summer = pd.read_excel(file, sheetname='Summer', header=0,converters={'Cr':str, 'Term':str})
    Fall = pd.read_excel(file, sheetname='Fall', header=0,converters={'Cr':str, 'Term':str})
    Winter = pd.read_excel(file, sheetname='Winter', header=0,converters={'Cr':str, 'Term':str})
    Spring = pd.read_excel(file, sheetname='Spring', header=0,converters={'Cr':str, 'Term':str})
    Faculty = pd.read_excel(file, sheetname='Faculty', header=0)
    Fac_TEs = pd.read_excel(sched_file_path, sheetname='Faculty Workload', header=None)
    
    #Drop NaNs
    Faculty = Faculty.dropna(subset=['Name'])
    
    #Bind the quarter schedules into a single dataframe
    frames = [Summer, Fall, Winter, Spring]
    Schedule = pd.concat(frames)

    # Gather faculty TE data
    max_x = int(np.floor((Fac_TEs.shape[0] - 17) / 8))
    idx = [x + ((x-1) * 8) for x in range(1,max_x)]
    cols = ['Name', 'Base TE', 'Scholarship TE', 'Admin Release', 'Other Release', 'Adjusted Target', 'Actual TE']
    Fac_TEs = Fac_TEs.iloc[idx, [0, 13, 14, 15, 16, 17, 18]]
    Fac_TEs.columns = cols
    Fac_TEs[cols[1:]] = Fac_TEs[cols[1:]].apply(pd.to_numeric)
    Fac_TEs.dropna(subset=['Name'], inplace=True)

    #If faculty member is full-time, mark as such
    fulltimers = Faculty['Name'].tolist()
    def FT_PT (faculty):
        if faculty == 'TBA':
            return 'TBA'
        elif faculty in fulltimers:
            return 'FT'
        else:
            return 'PT'
    Schedule['FT_PT'] = Schedule.apply(lambda x: FT_PT(x['Faculty']), axis=1)

    return Schedule, Faculty, Fac_TEs

#############################################################
# Grab Data
#############################################################
# Set the starting path
starting_path = 'W:\\csh\\Nursing\\Schedules'
# Set the directory pattern to match against
pattern = '^[0-9]+-[0-9]+$'
# List of directories to ignore. These will not function like more recent ones
ignore_dirs = ['2011-2012', '2012-2013', '2013-2014']
# Get a list of schedule directories
dirs = get_dir_paths (starting_path, pattern, ignore_dirs)

# Initiate frames to hold full schedule
schedule_frames = []
fac_te_frames = []
# Populate frames
for directory in dirs:
    sched_file_path = get_latest(directory)
    schedule, faculty, fac_tes = cat_sched(sched_file_path)
    schedule_frames.append(schedule)
    fac_te_frames.append(fac_tes)
# Concatenate frames
schedule = pd.concat(schedule_frames)

# Get Employee List
Faculty = pd.read_excel('W:\\csh\\Nursing\\Faculty\\Employee List.xlsx', header=0, converters={'Cell Phone':str})

#Read in term descriptions
TermDescriptions = pd.read_excel('W:\\csh\\Nursing\\Schedules\\Term Descriptions.xlsx', header=0, converters={'Term':str})

#############################################################
# Grab Data
#############################################################







#Set abbreviations for programs
program_types_abbr = {'MENP': 'MENP at LPC',
                     'RFU': 'MENP at RFU',
                     'DNP': 'DNP',
                     'RN to MS': 'RN to MS',
                     None: 'All'
                     }



def historical_line (schedule_list, fac_te_list, course_type_list=None, faculty_types_list=None, course_list=None):
    '''Creates a historical line graph that shows the Faculty TEs per
    fiscal year. User is able to define the following parameters that affect
    output: program_list, course_type_list, faculty_types_list, course_list.
    '''
    
    #Initiate Title; will build upon string for user-defined parameters
    chart_title = 'Total Faculty TEs Required for Course Coverage'
    
    #Create list of course types
    if course_type_list == None:
        course_types = ['COORD', 'LEC', 'LAB', 'PRA']
    else:
        course_types = []
        if type(course_type_list) is str:
            course_types.append(course_type_list)
            chart_title = chart_title + '\nCourse Type(s): ' + course_type_list
        else:
            for ct in course_type_list:
                course_types.append(ct)
            chart_title = chart_title + '\nCourse Type(s): ' + ', '.join(course_type_list)
    course_types.sort()
    
    #Create list of faculty types
    if faculty_types_list == None:
        faculty_types = ['FT','PT','TBA']
    else:
        faculty_types = []
        if type(faculty_types_list) is str:
            faculty_types.append(faculty_types_list)
            chart_title = chart_title + '\nFaculty Type(s): ' + faculty_types_list
        else:
            for ft in faculty_types_list:
                faculty_types.append(ft)
            chart_title = chart_title + '\nFaculty Type(s): ' + ', '.join(faculty_types_list)
    faculty_types.sort()
    
    #Set defaults
    plt.rcdefaults()
    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    
    fiscal_years = []
    
    x_pos = np.arange(len(schedule_list))
    
    TE = []
    f_te = []

    for schedule, fac_te in zip(schedule_list, fac_te_list):
        #Create list of terms
        terms = schedule['Term'].unique().tolist()
        terms.sort()
        
        #Create list of courses
        if course_list == None:
            courses = schedule['Cr'].unique().tolist()
        else:
            courses = []
            if type(course_list) is str:
                courses.append(course_list)
            else:
                for c in course_list:
                    courses.append(c)
        courses.sort()
        
        #Create list of fiscal years
        fiscal_years.append(TermDescriptions.loc[TermDescriptions['Term'] == terms[0], 'Fiscal Year'].item())
        
        #Cycle through terms and append total TEs
        TE.append(schedule[(schedule['Term'].isin(terms)) & (schedule['Cr'].isin(courses)) & (schedule['FT_PT'].isin(faculty_types)) & (schedule['Type'].isin(course_types))].sum()['TE'])
                
        # Append faculty TEs
        f_te.append(fac_te['Adjusted Target'].sum())
               
    ax.plot(x_pos, TE, '-o', label='TEs Needed for Coverage')
    for x, y in zip(x_pos, TE):
        ax.annotate('{0:.2f}'.format(y), xy=(x,y), xytext=(0,6), textcoords='offset points', size='small', ha='center')
    ax.plot(x_pos, f_te, '-o', label='Actual Full-Time Workload')
    for x, y in zip(x_pos, f_te):
        ax.annotate('{0:.2f}'.format(y), xy=(x,y), xytext=(0,-14), textcoords='offset points', size='small', ha='center')
    # Annotate percentages
    per_ful = [(f_te[x] / TE[x]) * 100 for x in range(len(TE))]
    for x, y, p in zip(x_pos, f_te, per_ful):
        ax.annotate('{0:.2f}%'.format(p), xy=(x,y), xytext=(0,-22), textcoords='offset points', size='small', ha='center')
    
    #Prepare fiscal_years for x_axis label
    fiscal_years = list(set(fiscal_years))
    fiscal_years.sort()
            
    #Chart formatting
    ax.set_title(chart_title)
    ax.set_ylabel('Faculty TEs')
    ax.set_ylim([0, 600])
    ax.set_xlabel('Fiscal Year')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(fiscal_years) 
    ax.legend(loc='upper left')
        
    #plt.show()
    return fig

historical_line(schedule_frames, fac_te_frames, course_type_list=None, faculty_types_list=None, course_list=None)
historical_line(schedule_frames, fac_te_frames, course_type_list=['LEC', 'COORD'], faculty_types_list=None, course_list=None)


# the end of the block, even if an Exception occurs.
Path = 'W:\\csh\\Nursing\\Schedules\\Template'
with PdfPages('{0}\\Charts\\Faculty Need.pdf'.format(Path)) as pdf:
    x1 = historical_line(schedule_frames, fac_te_frames, course_type_list=None, faculty_types_list=None, course_list=None)
    pdf.savefig(x1)
    plt.close(x1)
    x2 = historical_line(schedule_frames, fac_te_frames, course_type_list=['LEC', 'COORD'], faculty_types_list=None, course_list=None)
    pdf.savefig(x2)
    plt.close(x2)


def pie_by_faculty_type (schedule_list):
    '''Creates two rows of pie charts that break down Faculty TEs by full-time,
    part-time, and TBA faculty where row 1 corresponds to Lecture and
    Coordination course types and row 2 corresponds to Lab and Clinical course
    types. Each row contains four pie charts, one for each quarter in the
    fiscal year. User may supply a list of programs.'''
       
    #Create list of faculty types
    faculty_types = ['FT','PT','TBA']
    
    #Set defaults
    plt.rcdefaults()
    #plt.rcParams['font.size'] = 8
    colors = ["#6885d0","#64a85c","#b88f3e"]
    fig, ax = plt.subplots()
    
    fiscal_years = []

    for schedule in schedule_list:
        #Create list of terms
        terms = schedule['Term'].unique().tolist()
        terms.sort()
        
        #Create list of fiscal years
        fiscal_years.append(TermDescriptions.loc[TermDescriptions['Term'] == terms[0], 'Fiscal Year'].item())
    
    #Row 0: LEC and COORD
    y = 0
    
    num_years = len(fiscal_years)
    
    #Initiate the subplots as callable 2d array
    x = [[plt.subplot2grid((2,num_years), (j,i), colspan=1) for i in range(len(fiscal_years))] for j in range(2)]
    
    for i, year, schedule in zip(range(len(fiscal_years)), fiscal_years, schedule_list):
        terms = TermDescriptions[TermDescriptions['Fiscal Year'] == year]['Term'].values
        TE = []
        fac_labels = []
        fac_colors = []
        for f, faculty_type in enumerate(faculty_types):
            TE_sum = schedule[(schedule['Type'].isin(['LEC', 'COORD'])) & (schedule['Term'].isin(terms)) & (schedule['FT_PT'] == faculty_type)].sum()['TE']
            #Only track the faculty types which have a non-zero TE sum
            if TE_sum != 0:
                TE.append(TE_sum)
                fac_labels.append(faculty_type)
                fac_colors.append(colors[f])
            
        #Plot pie
        x[y][i].pie(TE, labels=fac_labels, autopct='%1.0f%%', pctdistance=0.4, startangle=90, colors=fac_colors)
        
        #Row 0 Pie Chart formatting
        x[y][i].axis('equal')
        x[y][0].set_ylabel('Lecture')
        x[y][i].set_xlabel(year)
        #Plot center circles on top row
        center_circle = plt.Circle((0,0), 0.65, color='white', fc='white', linewidth=0.25)
        x[y][i].add_artist(center_circle)
        
    #Row 1: CLN and LAB
    y += 1
    
    for i, year, schedule in zip(range(len(fiscal_years)), fiscal_years, schedule_list):
        terms = TermDescriptions[TermDescriptions['Fiscal Year'] == year]['Term'].values
        TE = []
        fac_labels = []
        fac_colors = []
        for f, faculty_type in enumerate(faculty_types):
            TE_sum = schedule[(schedule['Type'].isin(['LAB', 'PRA'])) & (schedule['Term'].isin(terms)) & (schedule['FT_PT'] == faculty_type)].sum()['TE']
            #Only track the faculty types which have a non-zero TE sum
            if TE_sum != 0:
                TE.append(TE_sum)
                fac_labels.append(faculty_type)
                fac_colors.append(colors[f])
            
        #Plot pie
        x[y][i].pie(TE, labels=fac_labels, autopct='%1.0f%%', pctdistance=0.4, startangle=90, colors=fac_colors)
        #Plot center circles on bottom row
        center_circle = plt.Circle((0,0), 0.65, color='white', fc='white', linewidth=0.25)
        x[y][i].add_artist(center_circle)
        
        #Row 1 Pie Chart formatting
        x[y][i].axis('equal')
        x[y][0].set_ylabel('LAB & CLN')
        
    #Legend Formatting
    descriptions = ['Full-Time', 'Part-Time', 'TBA']
    proxies = []
    for d, description in enumerate(descriptions):
        proxies.append(plt.bar(0, 0, color=colors[d], label=description))
    plt.legend(proxies, descriptions, loc='lower right', ncol=3)

    #Plot Formatting
    plt.suptitle('Workload Coverage', size=16)
    plt.tight_layout()
    #plt.show()
    return fig

# Showing steady erosion of full-time instruction
pie_by_faculty_type(schedule_frames[:-1])

pie_by_faculty_type(schedule_frames)


faculty['Name'].count()
fac_te = faculty['Adjusted TE Target'].sum()


# per fiscal year
    # lec vs non-lec
        # te's to cover lecture
        # current num_faculty and current te_load
# next year is stable state
    # what load is needed
    # what is current te_load

cr_types = [['LEC', 'COORD'], ['LAB', 'PRA']]
fiscal_years = []

for schedule in schedule_frames:
    #Create list of terms
    terms = schedule['Term'].unique().tolist()
    terms.sort()
    
    #Create list of fiscal years
    fiscal_years.append(TermDescriptions.loc[TermDescriptions['Term'] == terms[0], 'Fiscal Year'].item())
    
    
    
    

f, ax = plt.subplots()
years = np.arange(len(fiscal_years))
colors=["white","white","#b88f3e", "#b88f3e"]
width = 0.35

for year, fy, schedule in zip(years, fiscal_years, schedule_frames):
    TE_sum = []
    fac_te = []

    # Total num TEs to cover
    for cr_type, rate in zip(cr_types, [0.90, 0.10]):
        TE_sum.append(schedule[schedule['Type'].isin(cr_type)].sum()['TE'])

        if fy != "FY '19":
            fac_te.append(schedule[(schedule['FT_PT'] == 'FT') & (schedule['Type'].isin(cr_type))].sum()['TE'])
        else:
            fac_te.append(faculty['Adjusted TE Target'].sum() * rate)
        
    for y, i, b in zip([TE_sum[0], TE_sum[1], fac_te[0], fac_te[1]], range(4), [0, TE_sum[0], 0, TE_sum[0]]):
        ax.bar(year, y, width, color=colors[i], bottom=b, edgecolor='black')
        ax.annotate('{0:.2f}'.format(y), xy=(year,y+b), xytext=(0,0), textcoords='offset points', size='x-small', ha='center')

ax.set_xticks(years + width)
ax.set_xticklabels(fiscal_years)
#Legend Formatting
descriptions = ['Lecture', 'Lab/Cln', 'Full-Time TEs']
proxies = []
for d, description in enumerate(descriptions):
    proxies.append(plt.bar(0, 0, color=colors[d], label=description))
plt.legend(proxies, descriptions, loc='best', ncol=3)
plt.show()








