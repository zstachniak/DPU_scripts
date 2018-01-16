# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 11:43:20 2017

@author: astachn1
"""
import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import click
import re

#############################################################
# Functions
#############################################################

def get_latest (pathname):
    '''Scans the folder for all files that match the typical naming conventions.
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
    Summer = pd.read_excel(file, sheet_name='Summer', header=0,converters={'Cr':str, 'Term':str})
    Fall = pd.read_excel(file, sheet_name='Fall', header=0,converters={'Cr':str, 'Term':str})
    Winter = pd.read_excel(file, sheet_name='Winter', header=0,converters={'Cr':str, 'Term':str})
    Spring = pd.read_excel(file, sheet_name='Spring', header=0,converters={'Cr':str, 'Term':str})
    Faculty = pd.read_excel(file, sheet_name='Faculty', header=0)
    
    #Drop NaNs
    Faculty = Faculty.dropna(subset=['Name'])
    Faculty = Faculty[Faculty['Name'] != 'Null']
    
    #Bind the quarter schedules into a single dataframe
    frames = [Summer, Fall, Winter, Spring]
    Schedule = pd.concat(frames)

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

    return Schedule, Faculty

def bar_faculty_workload (faculty_schedule, program_list=None):
    '''Creates a bar chart which shows the full-time faculty TE variance.
    Chart will only work for FY '17 or later.'''
    
    #Initiate Title; will build upon string for user-defined parameters
    chart_title = 'TE Variance for Full-Time Faculty'
    
    #Create list of programs
    if program_list == None:
        programs = ['DNP', 'MENP', 'RFU', 'RN to MS']
    else:
        programs = []
        if type(program_list) is str:
            programs.append(program_list)
            chart_title = chart_title + '\n' + program_types_abbr[program_list]
        else:
            chart_title = chart_title + '\n'
            for p in program_list:
                programs.append(p)
                chart_title = chart_title + program_types_abbr[program_list]
    programs.sort()
    
    #Remove faculty not in program_list
    faculty_schedule = faculty_schedule[(faculty_schedule['Program'].isin(programs))]

    #Set defaults
    plt.rcdefaults()
    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    
    #Set x_position by number of faculty
    x_pos = np.arange(len(faculty_schedule['Name']))
    #Set width of bars
    width = 0.75
    
    #Create bar chart
    ax.bar(x_pos, faculty_schedule['Variance'], width, align='center', color='cornflowerblue')
    
    #Ensure y-axis covers -2 to 2 as a bare minimum
    ylim = ax.get_ylim()
    if ylim[0] > -2:
        ymin = -2
    else:
        ymin = ylim[0]
    if ylim[1] < 2:
        ymax = 2
    else:
        ymax = ylim[1]
    ax.set_ylim([ymin, ymax])
    
    #Chart Labels
    ax.set_title(chart_title)
    ax.set_ylabel('TE Variance')
    ax.set_xlabel('Full-Time Faculty')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(faculty_schedule['Name'], rotation=45, fontsize=6, ha='right')
    
    #Horizontal lines to show acceptable variance
    ax.axhline(y=1, linestyle='--', color='red', linewidth=1)
    ax.axhline(y=0, linestyle='-', color='black', linewidth=1)
    ax.axhline(y=-1, linestyle='--', color='red', linewidth=1)
    
    plt.tight_layout()
    #plt.show()
    return fig

def pie_by_faculty_type (schedule, program_list=None):
    '''Creates two rows of pie charts that break down Faculty TEs by full-time,
    part-time, and TBA faculty where row 1 corresponds to Lecture and
    Coordination course types and row 2 corresponds to Lab and Clinical course
    types. Each row contains four pie charts, one for each quarter in the
    fiscal year. User may supply a list of programs.'''
    
    #Create list of programs
    if program_list == None:
        programs = ['DNP', 'MENP', 'RFU', 'RN to MS']
    else:
        programs = []
        if type(program_list) is str:
            programs.append(program_list)
        else:
            for p in program_list:
                programs.append(p)
    programs.sort()
    
    #Create list of terms
    terms = schedule['Term'].unique().tolist()
    terms.sort()
    
    #Create list of faculty types
    faculty_types = ['FT','PT','TBA']
    
    #Set defaults
    plt.rcdefaults()
    #plt.rcParams['font.size'] = 8
    colors = ["#6885d0","#64a85c","#b88f3e"]
    fig, ax = plt.subplots()
    
    #Row 0: LEC and COORD
    y = 0
    
    #Initiate the subplots as callable 2d array
    x = [[plt.subplot2grid((2,4), (j,i), colspan=1) for i in range(len(terms))] for j in range(2)]
    
    for i, term in enumerate(terms):
        TE = []
        fac_labels = []
        fac_colors = []
        for f, faculty_type in enumerate(faculty_types):
            TE_sum = schedule[(schedule['Program'].isin(programs)) & (schedule['Type'].isin(['LEC', 'COORD'])) & (schedule['Term'] == term) & (schedule['FT_PT'] == faculty_type)].sum()['TE']
            #Only track the faculty types which have a non-zero TE sum
            if not np.isnan(TE_sum):
                TE.append(TE_sum)
                fac_labels.append(faculty_type)
                fac_colors.append(colors[f])
            
        #Plot pie
        x[y][i].pie(TE, labels=fac_labels, autopct='%1.0f%%', pctdistance=0.4, startangle=90, colors=fac_colors)
        
        #Row 0 Pie Chart formatting
        x[y][i].axis('equal')
        x[y][0].set_ylabel('Lecture')
        x[y][i].set_xlabel(TermDescriptions.loc[TermDescriptions['Term'] == term, 'Quarter'].item())
        #Plot center circles on top row
        center_circle = plt.Circle((0,0), 0.65, color='white', fc='white', linewidth=0.25)
        x[y][i].add_artist(center_circle)
        
    #Row 1: CLN and LAB
    y += 1
    
    for i, term in enumerate(terms):
        TE = []
        fac_labels = []
        fac_colors = []
        for f, faculty_type in enumerate(faculty_types):
            TE_sum = schedule[(schedule['Program'].isin(programs)) & (schedule['Type'].isin(['LAB', 'PRA'])) & (schedule['Term'] == term) & (schedule['FT_PT'] == faculty_type)].sum()['TE']
            #Only track the faculty types which have a non-zero TE sum
            if not np.isnan(TE_sum):
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
    plt.suptitle('Workload Coverage: {0}, {1}'.format(program_types_abbr[program_list], TermDescriptions.loc[TermDescriptions['Term'] == term, 'Fiscal Year'].item()), size=16)
    plt.tight_layout()
    #plt.show()
    return fig

def historical_line (schedule_list, program_list=None, course_type_list=None, faculty_types_list=None, course_list=None):
    '''Creates a historical line graph that shows the Faculty TEs per
    fiscal year. User is able to define the following parameters that affect
    output: program_list, course_type_list, faculty_types_list, course_list.
    '''
    
    #Initiate Title; will build upon string for user-defined parameters
    chart_title = 'Total Faculty TEs Required for Course Coverage'
    
    #Create list of programs
    if program_list == None:
        programs = ['DNP', 'MENP', 'RFU', 'RN to MS']
    else:
        programs = []
        if type(program_list) is str:
            programs.append(program_list)
            chart_title = chart_title + '\n' + program_types_abbr[program_list]
        else:
            chart_title = chart_title + '\n'
            for p in program_list:
                programs.append(p)
                chart_title = chart_title + program_types_abbr[program_list]
    programs.sort()
    
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
    
    for program in programs:
        TE = []
    
        for schedule in schedule_list:
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
            TE.append(schedule[(schedule['Program'] == program) & (schedule['Term'].isin(terms)) & (schedule['Cr'].isin(courses)) & (schedule['FT_PT'].isin(faculty_types)) & (schedule['Type'].isin(course_types))].sum()['TE'])
            
        #Plot each line
        ax.plot(x_pos, TE, '-o', label=program_types_abbr[program])
    
    #Prepare fiscal_years for x_axis label
    fiscal_years = list(set(fiscal_years))
    fiscal_years.sort()
            
    #Chart formatting
    ax.set_title(chart_title)
    ax.set_ylabel('Faculty TEs')
    ax.set_xlabel('Fiscal Year')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(fiscal_years) 
    ax.legend(loc='upper left')
        
    #plt.show()
    return fig

def stacked_bar_by_course (schedule, program_list=None, term_list=None, course_list=None, course_type_list=None):
    '''Creates stacked bar chart(s) that indicate the number of TEs assigned
    to full-time, part-time, or TBA faculty for each course and course type
    in a given program and term. In addition to a schedule, user may pass 
    either a string or a list of strings for program, term, course, or 
    course type, and the charts will be focused accordingly.'''
    
    #Create list of programs
    if program_list == None:
        programs = schedule['Program'].unique().tolist()
    else:
        programs = []
        if type(program_list) is str:
            programs.append(program_list)
        else:
            for p in program_list:
                programs.append(p)
    programs.sort()
       
    #Create list of terms
    if term_list == None:
        terms = schedule['Term'].unique().tolist()
    else:
        terms = []
        if type(term_list) is str:
            terms.append(term_list)
        else:
            for t in term_list:
                terms.append(t)
    terms.sort()
    
    #Loop through programs and terms and create all requested charts
    for program in programs:
        for term in terms:
            
            #Create list of courses based on program and term
            if course_list == None:
                courses = schedule[(schedule['Program'] == program) & (schedule['Term'] == term)]['Cr'].unique().tolist()
            else:
                courses = []
                if type(course_list) is str:
                    courses.append(course_list)
                else:
                    for c in course_list:
                        courses.append(c)
            courses.sort()
            cr = np.arange(len(courses))
            
            #Create list of course types based on program and term
            if course_type_list == None:
                course_types = schedule[(schedule['Program'] == program) & (schedule['Term'] == term)]['Type'].unique().tolist()
            else:
                course_types = []
                if type(course_type_list) is str:
                    course_types.append(course_type_list)
                else:
                    for ct in course_type_list:
                        course_types.append(ct)
            course_types.sort()
            
            if courses == []:
                break
            
            #Set chart defaults and initiate subplots
            plt.rcdefaults()
            plt.style.use('ggplot')
            fig, ax = plt.subplots()
            colors = ["#6885d0","#64a85c","#b88f3e"]
            
            #Build stacked bars for each course
            for i, course in enumerate(courses):
                #For each course type, build a list of x_position values
                #Add padding to ensure they do not overlap across courses
                ctype = np.arange(len(course_types))
                ctype = ctype + (i*len(ctype)) + i
                
                #Initiate empty lists
                FT_TE = []
                PT_TE = []
                TBA_TE = []
                
                #Cycle through course types and store sum of TEs
                for course_type in course_types:
                    FT_TE.append(schedule[(schedule['Program'] == program) & (schedule['Term'] == term) & (schedule['Cr'] == course) & (schedule['Type'] == course_type) & (schedule['FT_PT'] == 'FT')].sum()['TE'])
                    PT_TE.append(schedule[(schedule['Program'] == program) & (schedule['Term'] == term) & (schedule['Cr'] == course) & (schedule['Type'] == course_type) & (schedule['FT_PT'] == 'PT')].sum()['TE'])
                    TBA_TE.append(schedule[(schedule['Program'] == program) & (schedule['Term'] == term) & (schedule['Cr'] == course) & (schedule['Type'] == course_type) & (schedule['FT_PT'] == 'TBA')].sum()['TE'])
                        
                #Plot each stack, one at a time
                FT = ax.bar(ctype, FT_TE, color=colors[0], label='FT')
                PT = ax.bar(ctype, PT_TE, color=colors[1], bottom=FT_TE, label='PT')
                TBA = ax.bar(ctype, TBA_TE, color=colors[2], bottom=[x + y for x, y in zip(FT_TE,PT_TE)], label='TBA')
    
            #Create a list of course type abbreviations
            #Add a blank string to account for padding
            m_lab = list(map(course_types_abbr.get, course_types))
            m_lab.append('')
            
            #Set the minor ticks and labels            
            minor_label = m_lab * len(courses)
            minor_tick = list(range((len(cr)*len(m_lab))))
               
            #Add chart label elements
            ax.set_title('TEs by Course for {0} in {1} Quarter'.format(program, TermDescriptions.loc[TermDescriptions['Term'] == term, 'Quarter'].item()))
            ax.set_ylabel('TEs')
            ax.set_xlabel('Course')
            ax.set_xticks(cr*len(m_lab))
            ax.set_xticklabels(courses, ha='left', size=10)
            xax= ax.get_xaxis()
            xax.set_tick_params(which='major', pad=15)
            ax.set_xticks(minor_tick, minor=True)
            ax.set_xticklabels(minor_label, ha='left', size=5, rotation=35, minor=True)
            ax.legend((FT[0], PT[0], TBA[0]), ('FT', 'PT', 'TBA'), loc='upper left')
            
            #Plot it!
            #plt.show()
            return(fig)

def stacked_bar (schedule, program_list=None, term_list=None, course_list=None):
    '''Creates stacked bar chart(s) that indicate the number of TEs assigned
    to full-time, part-time, or TBA faculty for each course in a given
    program and term. In addition to a schedule, user may pass either a
    string or a list of strings for program, term, or course, and the charts
    created will be focused accordingly.'''
    
    chart_title = 'TE Coverage by Course: '
    
    #Create list of programs
    if program_list == None:
        programs = schedule['Program'].unique().tolist()
    else:
        programs = []
        if type(program_list) is str:
            programs.append(program_list)
        else:
            for p in program_list:
                programs.append(p)
    programs.sort()
       
    #Create list of terms
    if term_list == None:
        terms = schedule['Term'].unique().tolist()
        chart_title = chart_title + TermDescriptions.loc[TermDescriptions['Term'] == terms[0], 'Fiscal Year'].item()
    else:
        terms = []
        if type(term_list) is str:
            terms.append(term_list)
            chart_title = chart_title + TermDescriptions.loc[TermDescriptions['Term'] == term_list, 'Quarter'].item() + ' ' + TermDescriptions.loc[TermDescriptions['Term'] == term_list, 'Fiscal Year'].item()
        else:
            chart_title = chart_title + '\n'
            for t in term_list:
                terms.append(t)
                chart_title = chart_title + TermDescriptions.loc[TermDescriptions['Term'] == t, 'Quarter'].item() + ' '
            chart_title = chart_title + TermDescriptions.loc[TermDescriptions['Term'] == terms[0], 'Fiscal Year'].item()
    terms.sort()
    
    #Loop through terms and create all requested charts
    for i,term in enumerate(terms):
        
        #Create list of courses based on program and term
        if course_list == None:
            courses = schedule[(schedule['Program'].isin(programs)) & (schedule['Term'] == term)]['Cr'].unique().tolist()
        else:
            courses = []
            if type(course_list) is str:
                courses.append(course_list)
            else:
                for c in course_list:
                    courses.append(c)
        courses.sort()
        
        #For each course, build a list of x_position values
        #Add padding to ensure they do not overlap
        cr = np.arange(len(courses))
        
        #Set chart defaults and initiate subplots
        plt.rcdefaults()
        plt.style.use('ggplot')
        fig, ax = plt.subplots()
        colors = ["#6885d0","#64a85c","#b88f3e"]
        
        #Initiate empty lists
        FT_TE = []
        PT_TE = []
        TBA_TE = []
        
        #Build stacked bars for each course
        for course in courses:
            
            FT_TE.append(schedule[(schedule['Program'].isin(programs)) & (schedule['Term'] == term) & (schedule['Cr'] == course) & (schedule['FT_PT'] == 'FT')]['TE'].values.sum())
            PT_TE.append(schedule[(schedule['Program'].isin(programs)) & (schedule['Term'] == term) & (schedule['Cr'] == course) & (schedule['FT_PT'] == 'PT')]['TE'].values.sum())
            TBA_TE.append(schedule[(schedule['Program'].isin(programs)) & (schedule['Term'] == term) & (schedule['Cr'] == course) & (schedule['FT_PT'] == 'TBA')]['TE'].values.sum())
        
        #Plot each stack, one at a time
        FT = ax.bar(cr, FT_TE, color=colors[0], label='FT')
        PT = ax.bar(cr, PT_TE, color=colors[1], bottom=FT_TE, label='PT')
        TBA = ax.bar(cr, TBA_TE, color=colors[2], bottom=[x + y for x, y in zip(FT_TE,PT_TE)], label='TBA')
           
    #Add chart label elements
    ax.set_title(chart_title)
    ax.set_ylabel('TEs')
    ax.set_xlabel('Course')
    ax.set_xticks(cr)
    ax.set_xticklabels(courses, ha='center', size=6, rotation=35)
    ax.legend((FT[0], PT[0], TBA[0]), ('Full-Time', 'Part-Time', 'TBA'), loc='upper left')
    
    #Plot it!
    #plt.show()
    return(fig)

def Create_Charts (FY_Schedule, FY_Faculty_Schedule, FY_Historical_Schedule, Program, Terms, Path):
    '''
    '''
    
    #If path does not exist, make path
    if not os.path.exists('{0}\\Charts\\'.format(Path)):
        os.makedirs('{0}\\Charts\\'.format(Path))

    # Create the PdfPages object to which we will save the pages:
    # The with statement makes sure that the PdfPages object is closed properly at
    # the end of the block, even if an Exception occurs.
    with PdfPages('{0}\\Charts\\{1} Charts.pdf'.format(Path, program_types_abbr[Program])) as pdf:
        x1 = bar_faculty_workload(FY_Faculty_Schedule, program_list=Program)
        pdf.savefig(x1)
        plt.close(x1)
        
        x2 = historical_line(FY_Historical_Schedule)
        pdf.savefig(x2)
        plt.close(x2)
        
        x3 = pie_by_faculty_type(FY_Schedule, program_list=Program)
        pdf.savefig(x3)
        plt.close(x3)
        
        x4 = stacked_bar(FY_Schedule, program_list=Program, term_list=Terms[0])
        pdf.savefig(x4)
        plt.close(x4)
        
        x5 = stacked_bar(FY_Schedule, program_list=Program, term_list=Terms[1])
        pdf.savefig(x5)
        plt.close(x5)
        
        x6 = stacked_bar(FY_Schedule, program_list=Program, term_list=Terms[2])
        pdf.savefig(x6)
        plt.close(x6)
        
        x7 = stacked_bar(FY_Schedule, program_list=Program, term_list=Terms[3])
        pdf.savefig(x7)
        plt.close(x7)
    
        #Set the file's metadata via the PdfPages object:
        d = pdf.infodict()
        d['Title'] = '{0} Charts'.format(program_types_abbr[Program])
        d['Author'] = 'Zander Stachniak'
        d['ModDate'] = datetime.now()

def chart_wrapper (Most_Recent_FY_Schedule, Most_Recent_Faculty, FYHistory, Most_Recent_Path, **kwargs):
    '''A simple wrapper to keep code DRY'''
    # Optional definition of programs
    programs = kwargs.pop('programs', ['MENP', 'RFU', 'DNP', 'RN to MS', None])
    # Gather terms
    Most_Recent_Terms = Most_Recent_FY_Schedule['Term'].unique().tolist()
    # Create the charts
    for program in programs:
        Create_Charts(Most_Recent_FY_Schedule, Most_Recent_Faculty, FYHistory, program, Most_Recent_Terms, Most_Recent_Path)

'''
# With the function below, we iterate through all current faculty and 
# output their schedule for the fiscal year given.
'''

def output_faculty_schedule (current_faculty_df, FY_df, FY_year_list=None):
    '''For each faculty member in the provided current_faculty_df, function
    will collect all teaching assignments in the FY_year_list, then output
    those assignments to an excel file.
    '''
    
    # This is the path where all output will be saved
    Path = 'W:\\csh\\Nursing Administration\\Faculty Data\\Faculty Workload'
    
    # This is the current Date, which will be appended to the Path
    Date = datetime.now().strftime("%Y-%m-%d")
    
    # Create a list of terms to query, based on Fiscal Year provided
    term_list = TermDescriptions[TermDescriptions['Fiscal Year'].isin(FY_year_list)]['Term']
    
    # Initiatlize a term dictionary for easier representation
    term_dict = dict(zip(TermDescriptions['Term'], TermDescriptions['Long Description']))

    # Iterate through all current faculty
    for faculty in current_faculty_df['Name']:
        # Create a df for each faculty
        faculty_df = FY_df[(FY_df['Faculty'] == faculty) & (FY_df['Term'].isin(term_list))]
        # Drop unneeded columns
        faculty_df = faculty_df[['Term', 'Cr', 'Sec', 'Type', 'Time', 'Mode', 'Faculty', 'TE', 'Program', 'Notes']]
        # Map term to long description for easier representation
        faculty_df['Term'] = faculty_df['Term'].map(term_dict)
        
        # If path does not exist, make path
        if not os.path.exists('{0}\\Tables\\{1}\\{2}\\'.format(Path, FY_year_list[0], Date)):
            os.makedirs('{0}\\Tables\\{1}\\{2}\\'.format(Path, FY_year_list[0], Date))
            
        # Write out to excel
        faculty_df.to_excel('{0}\\Tables\\{1}\\{2}\\{3}.xlsx'.format(Path, FY_year_list[0], Date, faculty), sheet_name='Workload Projection')

#############################################################
# Main Function Call
#############################################################
@click.command()
@click.option(
        '--fy',
        help='Fiscal Year for which to run charting',
)
def main(fy):
    '''Main function.'''
    
    # Path to schedules
    sched_path = 'W:\\csh\\Nursing\\Schedules\\'
    
    # Set as global variables (used by many charts)
    global TermDescriptions
    global course_types_abbr
    global program_types_abbr
    
    #Read in term descriptions
    TermDescriptions = pd.read_excel('W:\\csh\\Nursing\\Schedules\\Term Descriptions.xlsx', header=0, converters={'Term':str})
    # Simplify years for easier access
    years = TermDescriptions[['Academic Year', 'Fiscal Year']]
    years.drop_duplicates(subset=['Fiscal Year'], keep='first', inplace=True)
    
    #Set abbreviations for course types
    course_types_abbr = {'LEC': 'Lc',
                         'LAB': 'Lb',
                         'PRA': 'Pr',
                         'COORD': 'Co'                     
                         }
    
    #Set abbreviations for programs
    program_types_abbr = {'MENP': 'MENP at LPC',
                         'RFU': 'MENP at RFU',
                         'DNP': 'DNP',
                         'RN to MS': 'RN to MS',
                         None: 'All'
                         }
    
    
    
    # Create a dictionary to store data temporarily
    sched_dict = {}
    
    # If user supplies a specific Fiscal Year
    if fy:
        # Ensure that fiscal year follows appropriate formatting
        if re.match(r"\d{2}", fy):
            fy = "FY '" + str(fy)
        elif re.match(r"'\d{2}", fy):
            fy = "FY " + fy
        elif re.match(r"FY '\d{2}", fy):
            pass
        else:
            raise Exception("Fiscal Year must be in appropriate format, e.g. FY '18.")
        # Determine academic year from fiscal year
        try:
            ay = years[years['Fiscal Year'] == fy]['Academic Year'].item()
        except:
            print("Must provide a valid Fiscal Year")
        
    # If user does not supply Fiscal Year as an argument...
    else:        
        # Get all directories that match regex
        subfolders = [f.name for f in os.scandir('W:\\csh\\Nursing\\Schedules') if f.is_dir() and re.match(r'\d{4}-\d{4}', f.name)]
        # Sort the directories
        subfolders.sort()
        # Assume correct academic year is the last one
        ay = subfolders[-1]
        # Determine fiscal year from academic year
        fy = years[years['Academic Year'] == ay]['Fiscal Year'].item()
        
    # Collect data
    # Convert fy to integer for mathematical checks
    fy_int = int(fy[-2:])
    counter = 0
    # The lowest FY for which data is available is FY '15
    while fy_int >= 15 and counter <= 5:
        # Create nested dictionary
        sched_dict[fy] = {}
        # Build a temporary path to the schedule
        sched_dict[fy]['folder'] = sched_path + ay
        # Get the latest build of the schedule document
        sched_dict[fy]['file_path'] = get_latest(sched_dict[fy]['folder'])
        # Get the schedule and faculty data
        sched_dict[fy]['schedule'], sched_dict[fy]['faculty'] = cat_sched(sched_dict[fy]['file_path'])
        # Increment Fiscal Year and counter
        fy_int -= 1
        counter += 1
        # update fy and ay
        fy = "FY '" + str(fy_int)
        ay = years[years['Fiscal Year'] == fy]['Academic Year'].item()
    
    # Gather a sorted list of fiscal years
    fiscal_years = list(sched_dict.keys())
    fiscal_years.sort()
    
    #Make List of dataframes
    FYHistory = [sched_dict[x]['schedule'] for x in fiscal_years]
        
    #Concatenate Recent FY Data
    FYConcat = pd.concat(FYHistory)
        
    # Output charts and faculty schedule for highest two fiscal years
    for year in fiscal_years[-2:]:
        chart_wrapper(sched_dict[year]['schedule'], sched_dict[year]['faculty'], FYHistory, sched_dict[year]['folder'])
        output_faculty_schedule(sched_dict[year]['faculty'], FYConcat, FY_year_list=[year])

if __name__ == "__main__":
    main()
