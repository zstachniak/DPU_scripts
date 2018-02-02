# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:01:51 2017

@author: astachn1
"""

import dpu.scripts as dpu
from dpu.file_locator import FileLocator
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from scipy import stats
from sklearn.ensemble import IsolationForest
import os
import click
import re

school_colors = {'DePaul University': '#396E93',
               'Millikin University': '#F0B67F',
               'Rush University': '#80AA69',
               'University of Illinois at Chicago': '#4C443C',
               'Elmhurst College': '#694D75'
               }

campus_pass_colors = {'LPC': sns.color_palette("Blues")[3],
                      'RFU': sns.color_palette("BuGn_r")[1]
                      }

campus_fail_colors = {'LPC': sns.color_palette("coolwarm", 7)[6],
                      'RFU': sns.color_palette("coolwarm", 7)[6]
                      }



def qtrs(admit, grad):
    '''Compute number of quarters taken to graduate by subtracting admit 
    quarter from graduation quarter. The quarters typically count by fives,
    but there are several inconsistencies, which must be accounted for based
    on the range of the terms.'''
    admit = int(admit)
    grad = int(grad)
    if admit >= 860:
        return ((grad - admit)/5) + 1
    elif admit > 620:
        return ((grad - admit)/5)
    elif admit >= 600:
        return ((grad - admit)/5) - 2
    elif admit >=570:
        return ((grad - admit)/5) - 3
    else:
        return ((grad - admit)/5)

def historical_line (df, x_field, y_field, groupby_field, **kwargs):
    '''Creates a line graph that displays a pass rates by year.
    
    
    x_field: e.g., "Year"
    y_field: e.g., "Pass Rate"
    groupby: e.g., "Master's Entry Program"
    
    
    show_vals: options are "all" or "last". Default to None.
    '''
    # Gather optional keyword arguments
    title = kwargs.pop('title', "{} by {} for {}".format(y_field, x_field, groupby_field))
    y_lim = kwargs.pop('y_lim', None)
    years = kwargs.pop('years', df["Year"].unique().tolist())
    show_vals = kwargs.pop('show_vals', None)
    groupby_vals = kwargs.pop('groupby_vals', df[groupby_field].unique().tolist())
    
    # Coerce years into a list
    years = coerce_to_list(years)
    years.sort()
    
    #Set defaults
    plt.rcdefaults()
    plt.style.use('seaborn')
    fig, ax = plt.subplots()
    
    #Set x_position values based on years
    yr = np.arange(len(years))
    
    #Iterate through schools and years to capture pass rates
    for group in groupby_vals:
        rate = []
        for year in years:
            rate.append(df[(df[groupby_field] == group) & (df[x_field] == year)].sum()[y_field])
        
        #Replace 0 values with NaNs so that programs that didn't exist yet
        #will not show up in the chart.
        rate = [np.nan if x == 0 else x for x in rate]
        
        #Plot each line
        ax.plot(yr, rate, '-o', color=school_colors[group], label=group)
        
        if show_vals == 'all':
            for i,j in zip(yr, rate):
                ax.annotate(str(j), xy=(i,j), xytext=(0,8), textcoords='offset points')
        elif show_vals == 'last':
            i = (len(yr)-1)
            j = rate[i]
            ax.annotate(str(j), xy=(i,j), xytext=(0,8), textcoords='offset points')
    
    #Chart formatting
    ax.set_title(title)
    ax.set_ylabel(y_field)
    ax.set_xlabel(x_field)
    ax.set_xticks(yr)
    ax.set_xticklabels(years) 
    ax.legend(loc='lower left')
    axes = plt.gca()
    axes.set_ylim(y_lim)
        
    return fig

#historical_line(IL_Competitors, "Year", "Pass Rate", "Master's Entry Program", groupby_vals=['DePaul University'], show_vals='all', y_lim=[70,101])

def stacked_bar_NCLEX (historical_rates, **kwargs):
    '''Creates stacked bar chart(s) that compare school pass rates.'''
    # Gather optional keyword arguments
    schools = kwargs.pop('school_list', historical_rates["Master's Entry Program"].unique().tolist())
    years = kwargs.pop('year_list', historical_rates["Year"].unique().tolist())
    
    #Set defaults
    plt.rcdefaults()
    plt.style.use('seaborn')
    fig, ax = plt.subplots()
    failcolor = "#ED6A5E"
    
    #Set x_position values based on years
    yr = np.arange(len(years))
    sc = np.arange(len(schools))
    
    #Initiate counters
    patch_legend = []
    pass_n = {}
    fail_n = {}
    for i,school in enumerate(schools):
        pass_n[school] = 0
        fail_n[school] = 0

    #Loop through programs and terms and create all requested charts
    for i, year in enumerate(years):
        
        sc = np.arange(len(schools))
        sc = sc + (i*len(sc)) + i
            
        #Build stacked bars for each year
        for j, school in enumerate(schools):
            p = historical_rates[(historical_rates["Master's Entry Program"] == school) & (historical_rates["Year"] == year)].sum()['Passed']
            pass_n[school] += p
            Pass_bar = ax.bar(sc[j], p, color=school_colors[school], label='Pass')
            
            f = historical_rates[(historical_rates["Master's Entry Program"] == school) & (historical_rates["Year"] == year)].sum()['Failed']
            fail_n[school] += f
            Fail_bar = ax.bar(sc[j], f, color=failcolor, bottom=p, label='Fail')
            
            #Use first set of patches to form a legend
            if len(patch_legend) <= 1:
                patch_legend.append(Pass_bar)
                
            #Annotate
            if (p+f) != 0:
                ax.annotate(str('{0:.0f}%'.format((p/(p+f))*100)), xy=(sc[j],p+f), xytext=(0,6), textcoords='offset points', ha='center', size=8)
        
    description_legend = ['{0} Pass/Fail: {1}/{2}={3:.0f}%'.format(x, pass_n[x], (pass_n[x]+fail_n[x]), ((pass_n[x]/(pass_n[x]+fail_n[x]))*100)) for x in schools]

    # Create a list of abbreviations
    m_lab = [re.sub(r'[^A-Z]', '', x) for x in schools]
    # Add a blank string to account for padding
    m_lab.append('')
    
    #Set the minor ticks and labels            
    minor_label = m_lab * len(years)
    minor_tick = list(range((len(yr)*len(m_lab))))
    
    #shrink chart height by 20% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.8])

    #Add chart label elements
    ax.set_title('Number of Passes and Failures by Year')
    ax.set_ylabel('Number of Candidates')
    ax.set_xlabel('Year')
    ax.set_xticks(yr*len(m_lab))
    ax.set_xticklabels(years, ha='left', size=10)
    xax = ax.get_xaxis()
    xax.set_tick_params(which='major', pad=15)
    ax.set_xticks(minor_tick, minor=True)
    ax.set_xticklabels(minor_label, ha='center', size=5, rotation=35, minor=True)
    
    #Add legend
    ax.legend(patch_legend, description_legend, loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, ncol=2)
    
    #Plot it!
    #plt.show()
    return(fig)

#stacked_bar_NCLEX(IL_Competitors)
#stacked_bar_NCLEX(IL_Competitors, school_list=['DePaul University','Rush University'], year_list=[2010,2011,2012,2013,2014,2015,2016,2017])
#stacked_bar_NCLEX(IL_Competitors, school_list='DePaul University')

def coerce_to_list (x):
    '''A simple function that will coerce a single string to a list.'''
    if type(x) is str:
        return [x]
    else:
        return x

def scatter_trend (df, x_field, y_field, groupby_field, **kwargs):
    '''Creates a simple scatter graph that also includes a trend line for
    each groupby_val requested.
    
    @ Parameters:
    ----------------------------
    df: dataframe
    x_field: Named dataframe column that will represent x-axis
    y_field: Named dataframe column that will represent y-axis
    groupby_field: Named dataframe column that will represent different lines
    
    @ Optional Keyword Arguments:
    ----------------------------
    title: Title for plot
    y_lim: a list that represents [min, max] of y_ticks
    x_vals: a list of values to filter x_field of df by
    y_vals: a list of values to filter y_field of df by
    groupby_vals: a list of values to filter groupby_field of df by
    groupby_colors: a list of colors that should match the groupby_vals.
        Defaults to an hls seaborn palette.
    '''
    # Gather optional keyword arguments
    title = kwargs.pop('title', 'Scatterplot with Trendline')
    y_lim = kwargs.pop('y_lim', None)
    x_vals = coerce_to_list(kwargs.pop('x_vals', df[x_field].unique().tolist()))
    y_vals = coerce_to_list(kwargs.pop('y_vals', df[y_field].unique().tolist()))
    groupby_vals = coerce_to_list(kwargs.pop('groupby_vals', df[groupby_field].unique().tolist()))
    groupby_colors = kwargs.pop('groupby_colors', sns.color_palette("hls", len(groupby_vals)))
     
    # Filter data if necessary
    for field, vals in zip([x_field, y_field, groupby_field], [x_vals, y_vals, groupby_vals]):
        df = df[df[field].isin(vals)].copy(deep=True)
    
    # Set defaults
    plt.rcdefaults()
    plt.style.use('seaborn')
    fig, ax = plt.subplots()
    
    # Iterate through groupby
    for val, col in zip(groupby_vals, groupby_colors):
        x = df[df[groupby_field] == val][x_field]
        y = df[df[groupby_field] == val][y_field]
        # Plot scatter
        ax.plot(x, y, 'o', color=col, label='{0} Pass Rate'.format(val))
        # Fit and plot trendline
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), '--', color=col, label='Slope: {0:.2f}'.format(z[0]))
        
    #Chart formatting
    ax.set_title(title)
    ax.set_ylabel(y_field)
    ax.set_xlabel(x_field)
    ax.set_xticks(x_vals)
    ax.set_xticklabels(x_vals)
    axes = plt.gca()
    if y_lim:
        axes.set_ylim(y_lim)
    ax.legend(loc='lower left')
    
    #Plot it
    return(fig)

#scatter_trend(IL_Competitors, 'Year', 'Pass Rate', "Master's Entry Program", title="Yearly Pass Rates of IL Master's Entry Programs", x_vals=np.arange(2010,2018), groupby_vals='DePaul University', groupby_colors=['#396E93'], y_lim=[70,101])
#scatter_trend(IL_Competitors, 'Year', 'Pass Rate', "Master's Entry Program", title="Yearly Pass Rates of IL Master's Entry Programs", x_vals=np.arange(2010,2018), groupby_vals=['DePaul University', 'Rush University'], groupby_colors=['#396E93', '#80AA69'], y_lim=[70,101])

def histogram_NCLEX (df, field, **kwargs):
    '''Given a user-specified field, creates dual charts with the top
    chart being a histogram of the data distribution and the bottom
    chart plotting the pass rates for the same bin sizes as the 
    histogram. Together, this should give an idea of both the performance
    of a group and the relative importance of that group.
    
    
    @ Optional Keyword Arguments:
    ----------------------------
    years: A list of years (string format) to use. Default is to use all.
    ignore_outliers: if True, function will run an Isolation Forest
        to determine values that are outliers in the given field and
        will remove those data points before graphing.
    contamination: The expected percentage of outliers in the data frame.
    '''
    # Gather optional keyword arguments
    years = kwargs.pop('years', df["Year"].unique().tolist())
    ignore_outliers = kwargs.pop('ignore_outliers', False)
    contamination = kwargs.pop('contamination', 0.01)
    
    # Coerce years into a list
    if type(years) is str:
        years = [years]
    # Sort list
    years.sort()
    
    # Subset dataframe based on years requested
    df = df.loc[df['Year'].isin(years)].copy()
    # Drop NaNs in field
    df.dropna(subset=[field], inplace=True)
    
    # Set plot defaults
    plt.rcdefaults()
    plt.style.use('seaborn')
    fig, ax = plt.subplots()
    
    # Initiatlize a subplot grid
    x1 = plt.subplot2grid((2,4), (0,0), colspan=4)
    x2 = plt.subplot2grid((2,4), (1,0), colspan=4)
    
    # Remove outliers if requested
    if ignore_outliers:
        df = remove_outliers(df, field, contamination=contamination)
    
    # Determine Mean and Standard Deviation
    mu = df[field].mean()
    sigma = df[field].std()
    
    # Set mins, maxes, and bin widths
    if field == 'GPA':
        field_min = 3.0
        field_max = 4.0
        counter = 0.1
    elif field == 'Days Elapsed':
        field_min = 0
        field_max = df[field].max()
        counter = 7
    elif field == 'Qtrs to Grad':
        field_min = df[field].min()
        field_max = df[field].max()
        counter = 1
    else:
        return print('Field not in dataframe.')
    
    #Set field_range, bins, and x_positions
    field_range = np.arange(field_min,field_max,counter)
    field_bins = np.arange(field_min,field_max + counter,counter)
    x_pos = np.arange(len(field_range))
    
    #Plot histogram to x1 with probability density
    n, bins, patches = x1.hist(df[field], normed=1, edgecolor='black', alpha=0.75, bins=field_bins)
    
    # add a 'best fit' line to x1
    y = mlab.normpdf(bins, mu, sigma)
    l = x1.plot(bins, y, 'r--', linewidth=1)
    
    #Chart formatting for x1
    x1.set_ylabel('Probability Density')
    x1_title = str('{0}: '.format(field) + r'$\mu=$' + '{0:.2f},'.format(mu) + r' $\sigma=$' + '{0:.2f}'.format(sigma) + '\n' + 'Testing Year(s): ' + ', '.join(years))
    x1.set_title(x1_title)
    x1.grid(True)

    # Initialize a list to store pass rates for x2   
    pass_rate = []
    
    # Loop through bins and append pass rates for each
    for x in field_range:
        if df[(df[field] >= x) & (df[field] < (x + counter))].count()['Result'] == 0:
            pass_rate.append(0)
        else:
            pass_rate.append(df[(df[field] >= x) & (df[field] < (x + counter)) & (df['Result'] == 'pass')].count()['Result'] / df[(df[field] >= x) & (df[field] < (x + counter))].count()['Result'])
    
    # Plot histogram with pass rates as y axis
    x2.bar(x_pos, pass_rate, align='edge', edgecolor='black', width=1, alpha=0.75)
    
    # Adjustments to x_labels for better visualization
    x_labels = np.append(field_range, (field_range.max() + counter))
    if field == 'Days Elapsed' or field == 'Qtrs to Grad':
        x_labels = x_labels.astype(int)
    
    # Chart formatting for x2
    x2.set_title("Pass Rates by {}".format(field))
    x2.set_ylabel('Pass Rate')
    yvals = x2.get_yticks()
    x2.set_yticklabels(['{:3.0f}%'.format(x*100) for x in yvals])
    x2.set_xlabel('{}'.format(field))
    x2.set_xticks(np.append(x_pos, (x_pos.max() + 1)))
    x2.set_xticklabels(x_labels)
    # Make x1 tick labels invisible
    plt.setp(x1.get_xticklabels(), visible=False)
    #For Days Elapsed (where cramped), remove every other x_label
    if field == 'Days Elapsed':
        for label in x2.xaxis.get_ticklabels()[::2]:
            label.set_visible(False)

    #Plot it
    plt.tight_layout()
    #plt.show()
    return fig

#histogram_NCLEX(NCLEX_df, 'Days Elapsed', ignore_outliers=True)
#histogram_NCLEX(NCLEX_df, 'Days Elapsed', ignore_outliers=True, years='2017')
#histogram_NCLEX(NCLEX_df, 'GPA', ignore_outliers=False)
#histogram_NCLEX(NCLEX_df, 'Qtrs to Grad', ignore_outliers=True)

def remove_outliers (df, field, contamination=0.01, verbose=False):
    '''Function will run an Isolation Forest to determine values that 
    are outliers in the given field and will remove those data points 
    before returning a new dataframe.'''
    # Use a deep copy of data to avoid making changes to original
    X = df[field].copy(deep=True)
    X = X.values.reshape(-1, 1)
    # Prepare and fit the Isolation Forest
    IsoFor = IsolationForest(bootstrap=True, n_jobs=-1, contamination=contamination)
    IsoFor.fit(X)
    # Make predictions
    y_pred = IsoFor.predict(X)
    if verbose:
        num_outliers = np.unique(y_pred, return_counts=True)[1][0]
        print('{} outliers detected and removed from dataframe.'.format(num_outliers))
    # Truth value of non_outliers (equal to 1)
    non_outliers = y_pred == 1
    # Return new df
    return df[non_outliers].copy(deep=True)

def ANOVA_boxplot (df, field, groupby, **kwargs):
    '''Function to plot side by side boxplot comparisons of the field 
    separated by groupby. Function will also output statistical significance
    test of the difference as part of the plot title.
    
    @ Parameters:
    ----------------------------
    df: The original dataframe
    field: the field of comparison in the df
    groupby: the groupby values
    
    @ Optional Keyword Arguments:
    ----------------------------
    ignore_outliers: if True, function will run an Isolation Forest
        to determine values that are outliers in the given field and
        will remove those data points before graphing.
    contamination: The expected percentage of outliers in the data frame.
    '''
    # Gather optional keyword arguments
    ignore_outliers = kwargs.pop('ignore_outliers', False)
    contamination = kwargs.pop('contamination', 0.01)
    
    # Remove outliers if requested
    if ignore_outliers:
        df = remove_outliers(df, field, contamination=contamination)
    
    #Set defaults
    plt.rcdefaults()
    plt.style.use('seaborn')
    fig, ax = plt.subplots()
    
    # Drop NaNs
    df.dropna(subset=[field], inplace=True)
    
    # Gather all possible values, 
    values = df[groupby].unique().tolist()
    # Then separate df by the groupby values
    frames = []
    for val in values:
        frames.append(df[(df[groupby] == val)][field])
    
    # Plot a boxplot (outliers black dots)    
    x = ax.boxplot(frames, labels=values, notch=True, patch_artist=True, sym='k.')
    # Set transparency (looks nicer)
    for patch in x['boxes']:
        patch.set_alpha(0.75)
        
    # Statistical test to determine if the groupby values are different. 
    F, p = stats.f_oneway(frames[0], frames[1])
    
    # Set chart options
    title = '{}: Analysis of Variance'.format(field)
    title += '\nF-Statistic: {0:.2f}, p-value: {1:.2E}'.format(F, p)
    ax.set_title(title)
    ax.set_ylabel(field)
    ax.set_xlabel(groupby)
    
    #plt.show()
    return fig

#ANOVA_boxplot(NCLEX_df, 'GPA', 'Result')
#ANOVA_boxplot(NCLEX_df, 'Days Elapsed', 'Result', ignore_outliers=True)
#ANOVA_boxplot(NCLEX_df, 'Qtrs to Grad', 'Result', ignore_outliers=True)

def stacked_bar_campus (df, year=None, quarter_list=None):
    '''Creates stacked bar chart(s) that indicate the number of candidates
    who took the NCLEX by each quarter in a calendar year. Chart also breaks
    the data up by campus.'''
    
    #Create list of campuses
    campuses = df['Campus'].unique().tolist()
    
    #Create list of years
    if year == None:
        year = max(df['Year'].unique().tolist())
    
    #Create list of quarters
    if quarter_list == None:
        quarters = df["Quarter"].unique().tolist()
    else:
        quarters = []
        if type(quarter_list) is str:
            quarters.append(quarter_list)
        else:
            for q in quarter_list:
                quarters.append(q)
    quarters.sort()
    
    #Set defaults
    plt.rcdefaults()
    plt.style.use('seaborn')
    fig, ax = plt.subplots()
    
    #Set x_position values based on years
    qtr = np.arange(len(quarters))
    
    #Initiate counters
    patch_legend = []
    pass_n = {}
    fail_n = {}
    for campus in campuses:
        pass_n[campus] = 0
        fail_n[campus] = 0
    
    #Loop through programs and terms and create all requested charts
    for i, quarter in enumerate(quarters):
        cmp = np.arange(len(campuses))
        cmp = cmp + (i*len(cmp)) + i
            
        #Build stacked bars for each year
        for j,campus in enumerate(campuses):
            
            p = df[(df["Campus"] == campus) & (df["Year"] == year) & (df["Quarter"] == quarter) & (df["Result"] == 'pass')].count()['Result']
            pass_n[campus] += p
            Pass_bar = ax.bar(cmp[j], p, color=campus_pass_colors[campus], label='Pass')
                  
            f = df[(df["Campus"] == campus) & (df["Year"] == year) & (df["Quarter"] == quarter) & (df["Result"] == 'fail')].count()['Result']
            fail_n[campus] += f
            Fail_bar = ax.bar(cmp[j], f, color=campus_fail_colors[campus], bottom=p, label='Fail')
            
            #Use first set of patches to form a legend
            if len(patch_legend) <= 1:
                patch_legend.append(Pass_bar)
                
            #Annotate
            if (p+f) != 0:
                ax.annotate(str('{0:.2f}%'.format((p/(p+f))*100)), xy=(cmp[j],p+f), xytext=(0,6), textcoords='offset points', ha='center', size=8)
        
    description_legend = ['{0} Pass/Fail: {1}/{2}={3:.2f}%'.format(x, pass_n[x], (pass_n[x]+fail_n[x]), ((pass_n[x]/(pass_n[x]+fail_n[x]))*100)) for x in campuses]
    
    #Create a list of campus abbreviations
    #Add a blank string to account for padding
    m_lab = campuses
    m_lab.append('')
    
    #Set the minor ticks and labels            
    minor_label = m_lab * len(quarters)
    minor_tick = list(range((len(qtr)*len(m_lab))))

    #Add chart label elements
    ax.set_title('Passes and Failures by Quarter for {}'.format(year))
    ax.set_ylabel('Number of Candidates')
    ax.set_xlabel('Quarter')
    ax.set_xticks(qtr*len(m_lab))
    ax.set_xticklabels(quarters, ha='left', size=10)
    xax = ax.get_xaxis()
    xax.set_tick_params(which='major', pad=15)
    ax.set_xticks(minor_tick, minor=True)
    ax.set_xticklabels(minor_label, ha='center', size=5, rotation=35, minor=True)
    
    #Add legend
    ax.legend(patch_legend, description_legend, loc='upper left')
    
    #Plot it!
    #plt.show()
    return(fig)
    
#stacked_bar_campus(NCLEX_df, year='2017', quarter_list=None)

def stacked_bar_cohort (df, year=None, quarter_list=None, sortby='campus'):
    '''Creates stacked bar chart(s) that indicate the number of candidates
    who took the NCLEX by each quarter in a calendar year. Chart also breaks
    the data up by campus or cohort.'''
    
    #Create list of years
    if year == None:
        year = max(df['Year'].unique().tolist())
        
    #Create list of cohorts
    if sortby == 'Campus':
        cohorts = df[(df['Year'] == year)]['Campus'].unique().tolist()
        #pass_colormap = campus_pass_colors[cohort]
        #fail_colormap = campus_fail_colors[cohort]
    elif sortby == 'Compl Term':
        cohorts = df[(df['Year'] == year)]['Compl Term'].unique().tolist()
        pass_colormap = sns.color_palette("GnBu_d")
        fail_colormap = sns.color_palette("Reds")
    cohorts.sort()
    
    if len(cohorts) % 2 == 0:
        colormap = sns.color_palette("coolwarm", (len(cohorts) * 3 + 1))
    else:
        colormap = sns.color_palette("coolwarm", (len(cohorts) * 3))
    
    #Create list of quarters
    if quarter_list == None:
        quarters = df["Quarter"].unique().tolist()
    else:
        quarters = []
        if type(quarter_list) is str:
            quarters.append(quarter_list)
        else:
            for q in quarter_list:
                quarters.append(q)
    quarters.sort()
    
    #Set defaults
    plt.rcdefaults()
    plt.style.use('seaborn')
    fig, ax = plt.subplots()
    
    #Set x_position values based on years
    qtr = np.arange(len(quarters))
    
    #Initiate counters
    patch_legend = []
    pass_n = {}
    fail_n = {}
    for k,cohort in enumerate(cohorts):
        pass_n[cohort] = 0
        fail_n[cohort] = 0
    
    #Loop through programs and terms and create all requested charts
    for i, quarter in enumerate(quarters):
        
        cmp = np.arange(len(cohorts))
        cmp = cmp + (i*len(cmp)) + i
            
        #Build stacked bars for each year
        for j,cohort in enumerate(cohorts):
            p = df[(df[sortby] == cohort) & (df["Year"] == year) & (df["Quarter"] == quarter) & (df["Result"] == 'pass')].count()['Result']
            pass_n[cohort] += p
            Pass_bar = ax.bar(cmp[j], p, color=colormap[j], label='Pass')
                  
            f = df[(df[sortby] == cohort) & (df["Year"] == year) & (df["Quarter"] == quarter) & (df["Result"] == 'fail')].count()['Result']
            fail_n[cohort] += f
            Fail_bar = ax.bar(cmp[j], f, color=colormap[-(j+1)], bottom=p, label='Fail')
            
            #Use first set of patches to form a legend
            if len(patch_legend) <= (len(cohorts)-1):
                patch_legend.append(Pass_bar)
                
            #Annotate
            if (p+f) != 0:
                ax.annotate(str('{0:.2f}%'.format((p/(p+f))*100)), xy=(cmp[j],p+f), xytext=(0,6), textcoords='offset points', ha='center', size=8)
        
    description_legend = ['{0} Pass/Fail: {1}/{2}={3:.2f}%'.format(x, pass_n[x], (pass_n[x]+fail_n[x]), ((pass_n[x]/(pass_n[x]+fail_n[x]))*100)) for x in cohorts]
    
    #Create a list of cohort abbreviations
    #Add a blank string to account for padding
    m_lab = cohorts
    m_lab.append('')
    
    #Set the minor ticks and labels            
    minor_label = m_lab * len(quarters)
    minor_tick = list(range((len(qtr)*len(m_lab))))

    #Add chart label elements
    ax.set_title('Passes and Failures by Quarter for {}'.format(year))
    ax.set_ylabel('Number of Candidates')
    ax.set_xlabel('Quarter')
    ax.set_xticks(qtr*len(m_lab))
    ax.set_xticklabels(quarters, ha='left', size=10)
    xax = ax.get_xaxis()
    xax.set_tick_params(which='major', pad=15)
    ax.set_xticks(minor_tick, minor=True)
    ax.set_xticklabels(minor_label, ha='center', size=5, rotation=35, minor=True)
    
    #Add legend
    ax.legend(patch_legend, description_legend, loc='upper left')
    
    #Plot it!
    #plt.show()
    return(fig)
    
#stacked_bar_cohort(NCLEX_df, year=None, quarter_list=None, sortby='Campus')
#stacked_bar_cohort(NCLEX_df, year=None, quarter_list=None, sortby='Compl Term')





#############################################################
# Main
#############################################################
@click.command()
@click.option(
        '--max_year',
        help='The max year through which you want to focus your inquiry, e.g. "2018"',
)
@click.option(
        '--min_year',
        help='The min year through which you want to focus your inquiry, e.g. "2010"',
)
def main(max_year, min_year):
    '''Main function.'''
    FL= FileLocator()
    today = datetime.today()
    # If the user did not pass a max_year, use current year
    if not max_year:
        max_year = str(today.max_year)
        
    # Gather list of yearly IL pass rates
    il_path = os.path.abspath(os.path.join(os.sep, FL.nclex, 'Illinois Yearly Pass Rates for All Schools', 'CompetitorPassRates.xlsx'))
    IL_Competitors = pd.read_excel(il_path, sheet_name='Sheet1', header=0)
    
    # DPU NCLEX pass data
    f_path = os.path.abspath(os.path.join(os.sep, FL.nclex, 'NCLEX.xlsx'))
    NCLEX = pd.read_excel(f_path, header=0, converters={'Empl ID':str, 'Year':str})
    #Fill first zero where needed
    NCLEX['Empl ID'] = NCLEX['Empl ID'].str.zfill(7)
    #Drop unnecessary fields
    NCLEX.drop(['Last Name','First Name','Time Delivered', 'Candidate ID'],axis=1,inplace=True)
    #Add days elapsed since graduation
    NCLEX['Date Delivered'] = pd.to_datetime(NCLEX['Date Delivered'])
    NCLEX['Days Elapsed'] = NCLEX['Date Delivered'] - NCLEX['Graduation Date']
    NCLEX['Days Elapsed'] = NCLEX['Days Elapsed'].dt.days
    # Standardize Result
    NCLEX['Result'] = NCLEX['Result'].map(lambda x: x.lower())
    # Remove repeat test-takers
    NCLEX = NCLEX[NCLEX['Repeater'] == 'No']
    
    # Read Grad Data
    g_path = os.path.abspath(os.path.join(os.sep, FL.grad))
    f = dpu.get_latest(g_path, 'NSG_GRAD_REVISED')
    Grad_data = pd.read_excel(f, skiprows=0, header=1, na_values='nan', converters={'ID':str, 'Admit Term':str, 'Compl Term': str})
    # Drop students in the wrong degree program (students who take more than
    # one program will be duplicated unnecessarily).
    Grad_data = Grad_data[Grad_data['Degree'] == 'MS']
    Grad_data = Grad_data[Grad_data['Acad Plan'].isin(['MS-NURSING', 'MS-GENRNSG'])]
    Grad_data = Grad_data[Grad_data['Sub-Plan'] != 'ANESTHESIS']
    # Determine number of quarters between admit and completion
    Grad_data['Qtrs to Grad'] = Grad_data.apply(lambda x: qtrs(x['Admit Term'], x['Compl Term']), axis=1)

    # Combine NCLEX and Graduates into Temp dataframe
    NCLEX_df = pd.merge(NCLEX[['Empl ID', 'Campus', 'Result', 'Days Elapsed', 'Year', 'Quarter', 'Graduation Date']], Grad_data[['ID', 'GPA', 'Qtrs to Grad', 'Compl Term']], how='left', left_on='Empl ID', right_on='ID', sort=True, copy=True)
    # Drop degree and ID fields
    NCLEX_df = NCLEX_df.drop(['ID'],axis=1)
    '''We do not have the testing dates for students who take NCLEX out of
    state. There are also a few oddities, where students test before they
    technically receive their degrees. To avoid these outlier cases from
    affecting our data, we will impute all NaN, 0, and negative values 
    using the mean of the group as a whole, rounded to the nearest unit
    value.'''
    elapsed = round(NCLEX_df.mean()['Days Elapsed'],0)
    NCLEX_df['Days Elapsed'].fillna(elapsed, inplace=True)
    NCLEX_df.loc[NCLEX_df['Days Elapsed'] <= 0, 'Days Elapsed'] = elapsed
    
    # If the user did not pass a min_year, use earliest year in data
    if not min_year:
        all_years = NCLEX_df['Year'].unique().tolist()
        all_years.sort()
        min_year = all_years[0]
    
    # Build a list of all years to consider
    year_list_int = np.arange(int(min_year), int(max_year)+1)
    year_list_str = [str(x) for x in year_list_int]

    # Build all the graphs
    graphs = [
            historical_line(IL_Competitors, "Year", "Pass Rate", "Master's Entry Program", groupby_vals=['DePaul University'], show_vals='all', y_lim=[70,101], years=year_list_int),
            
            historical_line(IL_Competitors, "Year", "Pass Rate", "Master's Entry Program", y_lim=[70,101], years=year_list_int),
                                                                                                                                                                                                    
            scatter_trend(IL_Competitors, 'Year', 'Pass Rate', "Master's Entry Program", title="Yearly Pass Rates of IL Master's Entry Programs", groupby_vals=['DePaul University', 'Rush University'], groupby_colors=['#396E93', '#80AA69'], y_lim=[70,101], x_vals=year_list_int),
                          
            stacked_bar_NCLEX(IL_Competitors, school_list=['DePaul University','Rush University'], year_list=year_list_int),
            
            histogram_NCLEX(NCLEX_df, 'GPA', ignore_outliers=False),
            
            ANOVA_boxplot(NCLEX_df, 'GPA', 'Result'),
            
            histogram_NCLEX(NCLEX_df, 'Days Elapsed', ignore_outliers=True),
            
            ANOVA_boxplot(NCLEX_df, 'Days Elapsed', 'Result', ignore_outliers=True)
            
            ]
    
    # Open a PDF file
    o_path = os.path.abspath(os.path.join(os.sep, FL.nclex, 'Reports', 'NCLEX_Charts {}.pdf'.format(today.strftime('%Y-%m-%d'))))
    with PdfPages(o_path) as pdf:
        # Write all the figures to file
        for graph in graphs:
            pdf.savefig(graph)
            plt.close(graph)
        #Set the file's metadata via the PdfPages object:
        d = pdf.infodict()
        d['Title'] = 'NCLEX Charts'
        d['Author'] = os.getlogin()
        d['ModDate'] = today

if __name__ == "__main__":
    main()
