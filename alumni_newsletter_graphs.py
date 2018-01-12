# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 12:59:58 2017

@author: astachn1
"""
import matplotlib.pyplot as plt

chart_location = 'W:\\csh\\Nursing\\Digital Editing\\Projects\\Alumni Newsletter\\Issue 2 - Fall 2017\\Charts'

# Set student values
#StudentBody = [344, 111, 23]
StudentBody = [440, 4, 127, 25]
TotalStudents = sum(StudentBody)
Programs = ['MENP', '3+2', 'DNP', 'RN-MS']
colors = ["white","white","white","white"]

# Set faculty values
FacultyBody = [14, 4, 10]
TotalFaculty = sum(FacultyBody)
FacultyType = ['Non-Tenure-Track', 'Tenured', 'Tenure-Track']
colors = ["white","white","white"]

# Set Graduation values
Graduates = [181, 174, 165, 135]
Years = [2017, 2016, 2015, 2014]

def donut (values_list, values_text, values_colors, filename):
    '''
    '''
    
    values_total = sum(values_list)
    
    #Set defaults for plot
    plt.rcdefaults()
    # Set backaground color
    plt.rcParams['figure.facecolor'] = '#0068ac'
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['font.size'] = 20
    fig, ax = plt.subplots()
    
    # Plot the pie chart
    wedges, texts = ax.pie(values_list, labels=values_text, autopct=None, pctdistance=0.4, startangle=90, colors=values_colors, labeldistance=1.2)
    ax.axis('equal')
    
    # Line widths and colors
    for w in wedges:
        w.set_linewidth(4)
        w.set_edgecolor('#0068ac')
    for t in texts:
        t.set_horizontalalignment('center')
        
    # Center circle
    center_circle = plt.Circle((0,0), 0.65, color='#0068ac', fc='#0068ac', linewidth=0.25)
    ax.add_artist(center_circle)
    
    # Annotate with Total
    ax.annotate(str('{0}'.format(values_total)), xy=(0,0), xytext=(0,-0.1), ha='center', size=60)
    
    # Plot
    plt.tight_layout()
    #plt.show()
    plt.savefig('{0}\\{1}'.format(chart_location, filename), dpi=600, facecolor='#0068ac', bbox_inches='tight')
    #return fig
    
    
donut(StudentBody, Programs, colors, 'studentbody')
#donut(FacultyBody, FacultyType, colors, 'facultybody')

def grad_bar (graduates, years, filename):
    '''
    '''

    #Set defaults for plot
    plt.rcdefaults()
    # Set backaground color
    plt.rcParams['figure.facecolor'] = '#0068ac'
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['font.size'] = 20
    fig, ax = plt.subplots()
    
    plt.ylim(120, 180)
    ax.bar(years, graduates, color='white', edgecolor='white')
    ax.patch.set_facecolor('#0068ac')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('#0068ac')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('#0068ac')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    plt.yticks([125, 150, 175, 200])
    
    #plt.show()
    plt.savefig('{0}\\{1}'.format(chart_location, filename), dpi=600, facecolor='#0068ac', bbox_inches='tight')

grad_bar(Graduates, Years, 'grad')
