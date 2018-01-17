# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:01:51 2017

@author: astachn1
"""

import dpu.scripts as dpu
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from scipy import stats
from sklearn.ensemble import IsolationForest

IL_Competitors = pd.read_excel('W:\\csh\\Nursing Administration\\Data Management\\NCLEX Improvement Plan\\Illinois Yearly Pass Rates for All Schools\\CompetitorPassRates.xlsx', sheet_name='Sheet1', header=0)

school_abbr = {'DePaul University': 'DPU',
               'Millikin University': 'Mlkn',
               'Rush University': 'Rush',
               'University of Illinois at Chicago': 'UIC',
               'Elmhurst College': 'Elm'
               }

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

#Read NCLEX
NCLEX = pd.read_excel('W:\\csh\\Nursing Administration\\Data Management\\DataWarehouse\\OG_Data\\NCLEX_Results\\NCLEX.xlsx',header=0,converters={'Empl ID':str, 'Year':str})
#Fill first zero where needed
NCLEX['Empl ID'] = NCLEX['Empl ID'].str.zfill(7)
#Drop unnecessary fields
NCLEX.drop(['Last Name','First Name','Time Delivered', 'Candidate ID'],axis=1,inplace=True)
#Add days elapsed since graduation
NCLEX['Date Delivered'] = pd.to_datetime(NCLEX['Date Delivered'])
NCLEX['Days Elapsed'] = NCLEX['Date Delivered'] - NCLEX['Graduation Date']
NCLEX['Days Elapsed'] = NCLEX['Days Elapsed'].dt.days
# Standardize Result
result_map = {'FAIL': 'Fail',
              'Fail': 'Fail',
              'fail': 'Fail',
              'PASS': 'Pass',
              'Pass': 'Pass',
              'pass': 'Pass'}
NCLEX['Result'] = NCLEX['Result'].map(result_map)
# Remove repeat test-takers
NCLEX = NCLEX[NCLEX['Repeater'] == 'No']

#Read Grad Data
f = dpu.get_latest('W:/csh/Nursing Administration/Data Management/DataWarehouse/OG_Data/NSG_GRADS', 'NSG_GRAD_REVISED')
Grad_data = pd.read_excel(f, skiprows=0, header=1, na_values='nan', converters={'ID':str, 'Admit Term':str, 'Compl Term': str})
# Drop students in the wrong degree program (students who take more than
# one program will be duplicated unnecessarily).
Grad_data = Grad_data[Grad_data['Degree'] == 'MS']
Grad_data = Grad_data[Grad_data['Acad Plan'].isin(['MS-NURSING', 'MS-GENRNSG'])]
Grad_data = Grad_data[Grad_data['Sub-Plan'] != 'ANESTHESIS']
'''Compute number of quarters taken to graduate by subtracting admit quarter
# from graduation quarter. The quarters typically count by fives, but there
# are several inconsistencies, which must be accounted for based on the range
# of the terms.'''
def qtrs(admit, grad):
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
Grad_data['Qtrs to Grad'] = Grad_data.apply(lambda x: qtrs(x['Admit Term'], x['Compl Term']), axis=1)

# Combine NCLEX and Graduates into Temp dataframe
NCLEX_df = pd.merge(NCLEX[['Empl ID', 'Campus', 'Result', 'Days Elapsed', 'Year', 'Quarter', 'Graduation Date']], Grad_data[['ID', 'GPA', 'Qtrs to Grad', 'Compl Term']], how='left', left_on='Empl ID', right_on='ID', sort=True, copy=True)
# Drop degree and ID fields
NCLEX_df = NCLEX_df.drop(['ID'],axis=1)
'''We do not have the testing dates for students who take NCLEX out of state.
#There are also a few oddities, where students test before they technically
#receive their degrees. To avoid these five outlier cases from affecting 
#our model, we will impute all NaN, 0, and negative values using the mean
#of the group as a whole, rounded to the nearest unit value.'''
elapsed = round(NCLEX_df.mean()['Days Elapsed'],0)
NCLEX_df['Days Elapsed'].fillna(elapsed, inplace=True)
NCLEX_df.loc[NCLEX_df['Days Elapsed'] <= 0, 'Days Elapsed'] = elapsed

def historical_line_NCLEX (historical_rates, school_list=None, year_list=None, show_vals=None):
    '''Creates a line graph that displays the NCLEX pass rates of IL Master's
    Entry to Nursing Practice programs over time. User can select schools
    or particular years.'''
    
    #Create list of programs
    if school_list == None:
        schools = historical_rates["Master's Entry Program"].unique().tolist()
    else:
        schools = []
        if type(school_list) is str:
            schools.append(school_list)
        else:
            for s in school_list:
                schools.append(s)
    schools.sort()
    
    #Create list of years
    if year_list == None:
        years = historical_rates["Year"].unique().tolist()
    else:
        years = []
        if type(year_list) is str:
            years.append(year_list)
        else:
            for y in year_list:
                years.append(y)
    years.sort()
    
    #Set defaults
    plt.rcdefaults()
    plt.style.use('seaborn')
    fig, ax = plt.subplots()
    
    #Set x_position values based on years
    yr = np.arange(len(years))
    
    #Iterate through schools and years to capture pass rates
    for school in schools:
        Pass_Rate = []
        for year in years:
            Pass_Rate.append(historical_rates[(historical_rates["Master's Entry Program"] == school) & (historical_rates["Year"] == year)].sum()['Pass Rate'])
        
        #Replace 0 values with NaNs so that programs that didn't exist yet
        #will not show up in the chart.
        Pass_Rate = [np.nan if x == 0 else x for x in Pass_Rate]
        
        #Plot each line
        ax.plot(yr, Pass_Rate, '-o', color=school_colors[school], label=school)
        
        if show_vals == 'all':
            for i,j in zip(yr, Pass_Rate):
                ax.annotate(str(j), xy=(i,j), xytext=(0,8), textcoords='offset points')
        elif show_vals == 'last':
            i = (len(yr)-1)
            j = Pass_Rate[i]
            ax.annotate(str(j), xy=(i,j), xytext=(0,8), textcoords='offset points')
    
    #Chart formatting
    ax.set_title("Yearly Pass Rates of IL Master's Entry Programs")
    ax.set_ylabel('Pass Rate')
    ax.set_xlabel('Calendar Year')
    ax.set_xticks(yr)
    ax.set_xticklabels(years) 
    ax.legend(loc='lower left')
    
    axes = plt.gca()
    axes.set_ylim([70,101])
        
    #plt.show()
    return fig

#historical_line_NCLEX(IL_Competitors, school_list='DePaul University')
#historical_line_NCLEX(IL_Competitors, school_list='DePaul University', show_vals='last')
#historical_line_NCLEX(IL_Competitors, school_list='DePaul University', show_vals='all')
#historical_line_NCLEX(IL_Competitors, show_vals='last')

def stacked_bar_NCLEX (historical_rates, school_list=None, year_list=None):
    '''Creates stacked bar chart(s) that indicate the number of TEs assigned
    to full-time, part-time, or TBA faculty for each course in a given
    program and term. In addition to a schedule, user may pass either a
    string or a list of strings for program, term, or course, and the charts
    created will be focused accordingly.'''
    
    #Create list of programs
    if school_list == None:
        schools = historical_rates["Master's Entry Program"].unique().tolist()
    else:
        schools = []
        if type(school_list) is str:
            schools.append(school_list)
        else:
            for s in school_list:
                schools.append(s)
    schools.sort()
    
    #Create list of years
    if year_list == None:
        years = historical_rates["Year"].unique().tolist()
    else:
        years = []
        if type(year_list) is str:
            years.append(year_list)
        else:
            for y in year_list:
                years.append(y)
    years.sort()
    
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

    #Create a list of course type abbreviations
    #Add a blank string to account for padding
    m_lab = list(map(school_abbr.get, schools))
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

def scatter_trend_NCLEX (historical_rates, school_list=None, year_list=None):
    '''
    '''
    
    #Create list of schools
    if school_list == None:
        schools = historical_rates["Master's Entry Program"].unique().tolist()
    else:
        schools = []
        if type(school_list) is str:
            schools.append(school_list)
        else:
            for s in school_list:
                schools.append(s)
    schools.sort()
    
    #Create list of years
    if year_list == None:
        years = historical_rates["Year"].unique().tolist()
    else:
        years = []
        if type(year_list) is str:
            years.append(year_list)
        else:
            for y in year_list:
                years.append(y)
    years.sort()
    
    #Remove Years if necessary
    historical_rates = historical_rates[historical_rates['Year'].isin(years)]
    
    #Set defaults
    plt.rcdefaults()
    plt.style.use('seaborn')
    fig, ax = plt.subplots()
    
    for school in schools:
        
        x = historical_rates[historical_rates["Master's Entry Program"] == school]['Year']
        y = historical_rates[historical_rates["Master's Entry Program"] == school]['Pass Rate']

        #Plot
        ax.plot(x, y, 'o', color=school_colors[school], label='%s Pass Rate'%(school))
        
        #Fit and plot trendline
        z = np.polyfit(x,y,1)
        p = np.poly1d(z)
        ax.plot(x,p(x),'--', color=school_colors[school], label='%s Slope: %.2f'%(school_abbr[school],z[0]))
        
    #Chart formatting
    ax.set_title("Yearly Pass Rates of IL Master's Entry Programs")
    ax.set_ylabel('Pass Rate')
    ax.set_xlabel('Calendar Year')
    ax.set_xticks(years)
    ax.set_xticklabels(years)
    axes = plt.gca()
    axes.set_ylim([70,101])
    ax.legend(loc='lower left')
    
    #Plot it
    #plt.show()
    return(fig)

#scatter_trend_NCLEX(IL_Competitors, school_list='DePaul University', year_list=[2010,2011,2012,2013,2014,2015,2016,2017])
#scatter_trend_NCLEX(IL_Competitors, school_list=['DePaul University', 'Rush University'], year_list=[2010,2011,2012,2013,2014,2015,2016,2017])
#scatter_trend_NCLEX(IL_Competitors, school_list=['DePaul University', 'University of Illinois at Chicago', 'Rush University'], year_list=[2010,2011,2012,2013,2014,2015,2016,2017])

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
    ignore_outliers = kwargs.pop('ignore_outliers', True)
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
    
    # Remove outliers, if requested
    if ignore_outliers:
        # Use a deep copy of data to avoid making changes to original
        X = df[field].copy(deep=True)
        X = X.values.reshape(-1, 1)
        # Prepare and fit the Isolation Forest
        IsoFor = IsolationForest(bootstrap=True, n_jobs=-1, contamination=contamination)
        IsoFor.fit(X)
        # Make predictions
        y_pred = IsoFor.predict(X)
        num_outliers = np.unique(y_pred, return_counts=True)[1][0]
        #print('{} outliers detected and removed from analysis.'.format(num_outliers))
        # Truth value of non_outliers (equal to 1)
        non_outliers = y_pred == 1
        # Remove outliers
        df = df[non_outliers]
    
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
            pass_rate.append(df[(df[field] >= x) & (df[field] < (x + counter)) & (df['Result'] == 'Pass')].count()['Result'] / df[(df[field] >= x) & (df[field] < (x + counter))].count()['Result'])
    
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

def NCLEX_boxplot (df, column, groupby):
    '''
    '''
    #Set defaults
    plt.rcdefaults()
    plt.style.use('seaborn')
    fig, ax = plt.subplots()
    
    values = df[groupby].unique().tolist()
    
    frames = []
    
    for i in range(len(values)):
        frames.append(df[(df[groupby] == values[i])][column])
        
    #ax.boxplot(frames, labels=values, notch=True, patch_artist=True)
    x = ax.boxplot(frames, labels=values, notch=True, patch_artist=True, sym='k.')
    
    for patch in x['boxes']:
        patch.set_alpha(0.75)
        
    title = '{}: Analysis of Variance'.format(column)
    
    # Calculate statistical significance
    F, p = stats.f_oneway(frames[0], frames[1])
    title += '\nF-Statistic: {0:.2f}, p-value: {1:.2E}'.format(F, p)
    
    ax.set_title(title)
    ax.set_ylabel(column)
    ax.set_xlabel(groupby)
    
    #plt.show()
    return fig
    
#NCLEX_boxplot(NCLEX_df, 'GPA', 'Result')

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
            
            p = df[(df["Campus"] == campus) & (df["Year"] == year) & (df["Quarter"] == quarter) & (df["Result"] == 'Pass')].count()['Result']
            pass_n[campus] += p
            Pass_bar = ax.bar(cmp[j], p, color=campus_pass_colors[campus], label='Pass')
                  
            f = df[(df["Campus"] == campus) & (df["Year"] == year) & (df["Quarter"] == quarter) & (df["Result"] == 'Fail')].count()['Result']
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
            p = df[(df[sortby] == cohort) & (df["Year"] == year) & (df["Quarter"] == quarter) & (df["Result"] == 'Pass')].count()['Result']
            pass_n[cohort] += p
            Pass_bar = ax.bar(cmp[j], p, color=colormap[j], label='Pass')
                  
            f = df[(df[sortby] == cohort) & (df["Year"] == year) & (df["Quarter"] == quarter) & (df["Result"] == 'Fail')].count()['Result']
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

# Create the PdfPages object to which we will save the pages:
# The with statement makes sure that the PdfPages object is closed properly at
# the end of the block, even if an Exception occurs.
with PdfPages('W:\\csh\\Nursing Administration\\Data Management\\NCLEX Improvement Plan\Reports\\NCLEX_Charts 2018.pdf') as pdf:
    
    x1 = historical_line_NCLEX(IL_Competitors, school_list='DePaul University', show_vals='all')
    pdf.savefig(x1)
    plt.close(x1)
    
    x13 = stacked_bar_campus(NCLEX_df, year=None, quarter_list=None)
    pdf.savefig(x13)
    plt.close(x13)
    
    x17 = stacked_bar_cohort(NCLEX_df, year=None, quarter_list=None, sortby='Compl Term')
    pdf.savefig(x17)
    plt.close(x17)
    
    x14 = historical_line_NCLEX(IL_Competitors, school_list='DePaul University', show_vals=None)
    pdf.savefig(x1)
    plt.close(x1)
    
    x2 = historical_line_NCLEX(IL_Competitors)
    pdf.savefig(x2)
    plt.close(x2)
    
    x3 = scatter_trend_NCLEX(IL_Competitors, school_list='DePaul University', year_list=[2010,2011,2012,2013,2014,2015,2016, 2017])
    pdf.savefig(x3)
    plt.close(x3)
    
    x4 = scatter_trend_NCLEX(IL_Competitors, school_list=['DePaul University', 'Rush University'], year_list=[2010,2011,2012,2013,2014,2015,2016])
    pdf.savefig(x4)
    plt.close(x4)
    
    x5 = scatter_trend_NCLEX(IL_Competitors, school_list=['DePaul University', 'University of Illinois at Chicago', 'Rush University'], year_list=[2010,2011,2012,2013,2014,2015,2016])
    pdf.savefig(x5)
    plt.close(x5)
    
    x6 = stacked_bar_NCLEX(IL_Competitors, school_list=['DePaul University','Rush University'], year_list=[2010,2011,2012,2013,2014,2015,2016])
    pdf.savefig(x6)
    plt.close(x6)
    
    x7 = histogram_NCLEX(NCLEX_df, field='GPA', year_list=['2010', '2011', '2012', '2013', '2014', '2015', '2016'])
    pdf.savefig(x7)
    plt.close(x7)
    
    x8 = NCLEX_boxplot(NCLEX_df, 'GPA', 'Result')
    pdf.savefig(x8)
    plt.close(x8)
    
    x9 = histogram_NCLEX(NCLEX_df, field='Days Elapsed', year_list=['2010', '2011', '2012', '2013', '2014', '2015', '2016'])
    pdf.savefig(x9)
    plt.close(x9)
    
    x10 = NCLEX_boxplot(NCLEX_df, 'Days Elapsed', 'Result')
    pdf.savefig(x10)
    plt.close(x10)
    
    x11 = histogram_NCLEX(NCLEX_df, field='Qtrs to Grad', year_list=['2010', '2011', '2012', '2013', '2014', '2015', '2016'])
    pdf.savefig(x11)
    plt.close(x11)
    
    x12 = NCLEX_boxplot(NCLEX_df, 'Qtrs to Grad', 'Result')
    pdf.savefig(x12)
    plt.close(x12)
    
    #Set the file's metadata via the PdfPages object:
    d = pdf.infodict()
    d['Title'] = 'NCLEX Charts'
    d['Author'] = 'Zander Stachniak'
    d['ModDate'] = datetime.now()










