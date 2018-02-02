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
import dpu.scripts as dpu
from dpu.file_locator import FileLocator
from win32com import client

#############################################################
# Functions
#############################################################

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
def output_faculty_schedule (output_path, sched_dict, FY_df, FY_year_list):
    '''For each faculty member in the provided current_faculty_df, function
    will collect all teaching assignments in the FY_year_list, then output
    those assignments to an excel file.
    '''
    # Order and sizing of columns for report
    col_order = [('Term', 20),
                 ('Cr', 6),
                 ('Sec', 6),
                 ('Type', 7),
                 ('Time', 20),
                 ('Mode', 8),
                 ('Faculty', 20),
                 ('TE', 6),
                 ('Program', 10),
                 ('Notes', 40),]

    # Order and sizing of columns for report
    wkl_order = [('Base TE', 10),
                 ('Scholarship Release', 19),
                 ('Admin Release', 14),
                 ('Additional TE Changes', 21),
                 ('Adjusted TE Target', 17),
                 ('Actual TE', 10),
                 ('Variance', 10),
                 ('Notes', 40),]
    
    text_blobs = ["The following represents your projected workload for the next fiscal year (Summer through Spring). Your projected workload is determined by your program director. You are not required to take any action at this time, but we do ask faculty to review their workload and detailed course assignments carefully. If you have any questions or concerns, please contact your program director.",
              "Workload is measured in 'Teaching Equivalencies' or 'TEs.' In general, 1 TE is equal to teach a single four-quarter hour course. For courses with multiple components (e.g. Lecture, Lab, Clinical), workload is broken down based on the ratios previously determined for lecture to clinical to lab. 'Prog.' refers to the program for which you are teaching the course. We must track this for budgetary reasons, but it does not affect workload.",
              "All full-time faculty are considered to have a base workload of 9 TEs. Tenure-Track faculty receive a 3 TE reduction for scholarship and all Advanced Practice Nurse full-time faculty receive a 1 TE reduction for practice. Faculty who hold administrative positions may receive additional recurring release time based on what was negotiated with the Dean’s office. In cases where there was a non-zero workload balance in the previous year, an adjustment will be made to the next year and will be reflected in the 'Additional TE Changes' column. In other rare circumstances, one-time TE modifications may be granted (e.g. faculty granted leave for one quarter). After all changes to the base workload have been applied, the projected workload for the year is then subtracted to determine any variance. You can see this calculation in the top row on the right of the workload calculations. A green '0.00' under Variance TE is a good thing! Low values of variance may be carried over to the next year (whether positive or negative). High values should be discussed with your program director.",
              "Please review your course assignment and schedule for the upcoming quarters. Although we are limited in our ability to shift course times, please contact your program director if you foresee issues or wish to discuss modifications.", 
              "The majority of all courses have a 'preferred teaching modality,' either face-to-face, hybrid, or online. Preferred modality is determined by the program director, and modality may vary across programs for a single course (e.g. NSG 540 is taught in all three programs). There may be cases where a preferred modality has not been established, in which case you will see an abbreviation for 'unknown.'",
              "The preferred modality for your courses can be found in the 'Mode' column of your course assignments. If an instructor wishes to teach in a modality other than the preferred modality, the instructor must request approval from the program director.",
              "For courses in which a D2L course master exists, the course master will automatically be copied over to your section in D2L. A master course is developed to be taught in a particular modality (online, hybrid, or face-to-face) and integrates best practices in instructional design, content expertise, and extensive teaching experience. The master course contains a syllabus, course readings, videos, assignments, and rubrics. Instructors have the freedom to edit their course as they see fit. If an instructor has questions about course masters, or wishes to know what is contained in a course master, please contact Lisa Torrescano.",
              ]
    
    # Get current fiscal year
    FY_year = FY_year_list[-1]
    
    # Get current faculty df
    current_faculty_df = sched_dict[FY_year]['faculty']
    
    # Initiatlize a term dictionary for easier representation
    term_dict = dict(zip(TermDescriptions['Term'], TermDescriptions['Long Description']))
    
    # Ensure empty output paths
    expected_path = os.path.abspath(os.path.join(os.sep, output_path, FY_year))
    excel_path = os.path.abspath(os.path.join(os.sep, expected_path, 'tables'))
    dpu.ensure_empty_dir(expected_path)    
    dpu.ensure_empty_dir(excel_path)
    
    sheet_names = ['Workload Projection']

    # Iterate through all current faculty
    for faculty in current_faculty_df['Name']:
        # Gather workload data
        wrkld = current_faculty_df[current_faculty_df['Name'] == faculty].copy(deep=True)
        wrkld.drop(labels=['Name', 'Track', 'APN', 'Program', 'Contract', "Thelma's Notes"], axis=1, inplace=True)
        
        # File name
        full_file_name = os.path.abspath(os.path.join(os.sep, excel_path, faculty)) + '.xlsx'
        # Initialize a writer
        writer = pd.ExcelWriter(full_file_name, engine='xlsxwriter')
        
        # Write to excel starting on nonzero row
        startrow = 3
        wrkld.to_excel(writer, index=False, sheet_name='Workload Projection', startrow=startrow)
            
        # Access the worksheet
        workbook = writer.book
        worksheet = writer.sheets['Workload Projection']
        
        # Page formatting (landscape, narrow margins, fit to one page)
        worksheet.set_landscape()
        worksheet.set_margins(left=0.3, right=0.3, top=0.6, bottom=0.3)
        worksheet.fit_to_pages(1, 1)
        
        # Set Header and Footer
        worksheet.set_header('&L{}&CProjected Workload for {}&R&D'.format(faculty, FY_year))
        worksheet.set_footer('&CPage &P of &N')

        # Formatters
        wrap_text = workbook.add_format({'text_wrap': 1, 'align': 'center', 'valign': 'center'})
        top_row = workbook.add_format({'text_wrap': 1, 'valign': 'top', 'bold': True, 'bottom': True})
        merge_format = workbook.add_format({'text_wrap': 1, 'valign': 'top'})
        merge_heading = workbook.add_format({'text_wrap': 1, 'valign': 'top', 'bold': True})
        # Light red fill with dark red text.
        red = workbook.add_format({'bg_color': '#FFC7CE',
                                       'font_color': '#9C0006'})
        # Green fill with dark green text.
        green = workbook.add_format({'bg_color': '#C6EFCE',
                                       'font_color': '#006100'})
        
        # Set column sizes
        for i, col in enumerate(wkl_order):
            c = dpu.char_counter_from_int(i)
            worksheet.set_column('{0}:{1}'.format(c, c), col[1])
        
        # Define our range(s) for color formatting
        number_rows = len(wrkld.index)
        variance_range = "G{}:G{}".format(startrow + 2, startrow + 1 + number_rows)

        # Highlight Overages or Underages in red
        worksheet.conditional_format(variance_range, {'type': 'cell',
                                                       'criteria': 'equal to',
                                                       'value': 0,
                                                       'format': green})
        worksheet.conditional_format(variance_range, {'type': 'cell',
                                                       'criteria': 'not equal to',
                                                       'value': 0,
                                                       'format': red})
                
        # Set height and formatting of data
        worksheet.set_row(startrow + 1, 30, wrap_text)
        
        # Insert and merge text
        last_col = dpu.char_counter_from_int( len(wkl_order) -1 )
        worksheet.merge_range('A1:{}1'.format(last_col), 'Projected Workload', merge_heading)
        worksheet.set_row(1, 45)
        worksheet.merge_range('A2:{}2'.format(last_col), text_blobs[0], merge_format)
        
        worksheet.set_row(6, 45)
        worksheet.merge_range('A7:{}7'.format(last_col), text_blobs[1], merge_format)
        
        worksheet.set_row(8, 105)
        worksheet.merge_range('A9:{}9'.format(last_col), text_blobs[2], merge_format)
        
        worksheet.set_row(10, 30)
        worksheet.merge_range('A11:{}11'.format(last_col), text_blobs[3], merge_format)
        
        worksheet.merge_range('A13:{}13'.format(last_col), 'Course Modalities', merge_heading)
        worksheet.set_row(13, 45)
        worksheet.merge_range('A14:{}14'.format(last_col), text_blobs[4], merge_format)
        worksheet.set_row(15, 30)
        worksheet.merge_range('A16:{}16'.format(last_col), text_blobs[5], merge_format)
        
        worksheet.merge_range('A18:{}18'.format(last_col), 'Course Masters', merge_heading)
        worksheet.set_row(18, 45)
        worksheet.merge_range('A19:{}19'.format(last_col), text_blobs[6], merge_format)
        
        # Iterate over fiscal years
        for FY, indication in zip(reversed(FY_year_list[-2:]), ['Projected', 'Previously Taught']):
            # Create a list of terms to query, based on Fiscal Year
            term_list = TermDescriptions[TermDescriptions['Fiscal Year'] == FY]['Term']
            # Create a df for each faculty
            faculty_df = FY_df[(FY_df['Faculty'] == faculty) & (FY_df['Term'].isin(term_list))]
            # Drop unneeded columns
            faculty_df = faculty_df[['Term', 'Cr', 'Sec', 'Type', 'Time', 'Mode', 'Faculty', 'TE', 'Program', 'Notes']]
            # Map term to long description for easier representation
            faculty_df['Term'] = faculty_df['Term'].map(term_dict)
        
            # Sheet name
            sheet_name = '{} Courses'.format(FY)
            sheet_names.append(sheet_name)
            # Write current year data
            faculty_df.to_excel(writer, index=False, sheet_name=sheet_name)
            # Access the worksheet
            worksheet = writer.sheets[sheet_name]
            
            # Page formatting (landscape, narrow margins, fit to one page)
            worksheet.set_landscape()
            worksheet.set_margins(left=0.3, right=0.3, top=0.6, bottom=0.3)
            worksheet.fit_to_pages(1, 1)
            
            # Set Header and Footer
            worksheet.set_header('&L{}&C{} Courses for {}&R&D'.format(faculty, indication, FY))
            worksheet.set_footer('&CPage &P of &N')
            
            # Set column sizes
            for i, col in enumerate(col_order):
                c = dpu.char_counter_from_int(i)
                worksheet.set_column('{0}:{1}'.format(c, c), col[1])
    
            # Freeze panes on top row
            worksheet.freeze_panes(1, 0)
            # Apply autofilters
            number_rows = len(faculty_df.index)
            worksheet.autofilter('A1:{0}{1}'.format(dpu.char_counter_from_int(len(col_order) - 1), number_rows+1))
            
            # Set column sizes
            for i, col in enumerate(col_order):
                worksheet.write(0, i, col[0], top_row)

        # Apply changes
        writer.save()
        
        # Convert the excel file to PDF
        
        convert_excel_to_pdf(excel_path, expected_path, faculty, sheet_names=sheet_names)
  
def convert_excel_to_pdf (input_path, output_path, file_name, sheet_names):
    '''A function to handle conversion of excel to PDF.'''
    # Connect file_name to path_name
    excel_file_name = os.path.abspath(os.path.join(os.sep, input_path, file_name)) + '.xlsx'
    pdf_file_name = os.path.abspath(os.path.join(os.sep, output_path, file_name)) + '.pdf'
    # Call excel client as invisible
    excel = client.Dispatch("Excel.Application")
    excel.Visible = 0
    # Open the Excel workbook
    wb = excel.Workbooks.Open(excel_file_name)
    # Open all sheets
    wb.Worksheets(sheet_names).Select()
    # Set PDF output variables
    xlTypePDF = 0
    xlQualityStandard = 0
    # Export to PDF
    excel.ActiveSheet.ExportAsFixedFormat(xlTypePDF, pdf_file_name, xlQualityStandard, True, True)
    # Close the workbook and quit Excel
    wb.Close(False)
    excel.Quit()
    wb = None
    excel = None

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
    
    # Call to FileLocator class
    FL = FileLocator()
    
    # Set as global variables (used by many charts)
    global TermDescriptions
    global course_types_abbr
    global program_types_abbr
    
    #Read in term descriptions
    TermDescriptions = dpu.get_term_descriptions()
    # Simplify years for easier access
    years = TermDescriptions[['Academic Year', 'Fiscal Year']].copy(deep=True)
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
        sched_dict[fy]['folder'] = os.path.join(FL.schedule, ay)
        # Get the latest build of the schedule document
        sched_dict[fy]['file_path'] = dpu.get_latest(sched_dict[fy]['folder'], 'Fiscal Schedule', date_format='%m-%d-%Y', ignore_suffix='(2)')
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
    FYConcat = pd.concat(FYHistory, ignore_index=True)

    # Output charts and faculty schedule for highest two fiscal years
    for year in fiscal_years[-2:]:
        chart_wrapper(sched_dict[year]['schedule'], sched_dict[year]['faculty'], FYHistory, sched_dict[year]['folder'])
        
    output_faculty_schedule(FL.faculty_workload, sched_dict, FYConcat, fiscal_years)

if __name__ == "__main__":
    main()
