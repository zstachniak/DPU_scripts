# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 15:38:18 2018

@author: astachn1
"""

import dpu.scripts as dpu
from dpu.file_locator import FileLocator
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.ensemble import IsolationForest
from scipy import stats

def coerce_to_list (x):
    '''A simple function that will coerce a single string to a list.'''
    if type(x) is str:
        return [x]
    else:
        return x
    
def save_plt_figure (fig, output_folder, file_name, **kwargs):
    '''Simple wrapper to save matplotlib figure'''
    # Gather optional keyword arguments
    face_color = kwargs.pop('face_color', '#0068ac')
    dpi = kwargs.pop('dpi', 600)
    # Generate full output path
    output_path = os.path.abspath(os.path.join(os.path.sep, output_folder, file_name))
    # Save figure
    fig.savefig(output_path, dpi=dpi, facecolor=face_color, bbox_inches='tight')

def donut_center_text (values, val_text, val_colors, **kwargs):
    '''Constructs a donut chart with the total of all values displayed in
    the center.'''
    face_color = kwargs.pop('face_color', '#0068ac')
    text_color = kwargs.pop('text_color', 'white')
    font_size = kwargs.pop('font_size', 20)
    pctdistance = kwargs.pop('pctdistance', 0.4)
    startangle = kwargs.pop('startangle', 90)
    labeldistance = kwargs.pop('labeldistance', 1.2)

    # Set defaults for plot
    plt.rcdefaults()
    # Set backaground color
    plt.rcParams['figure.facecolor'] = face_color
    plt.rcParams['text.color'] = text_color
    plt.rcParams['font.size'] = font_size
    # Build figure
    fig, ax = plt.subplots()
    
    # Plot the pie chart
    wedges, texts = ax.pie(values, labels=val_text, autopct=None, pctdistance=pctdistance, startangle=startangle, colors=val_colors, labeldistance=labeldistance)
    ax.axis('equal')
    
    # Line widths and colors
    for w in wedges:
        w.set_linewidth(4)
        w.set_edgecolor(face_color)
    for t in texts:
        t.set_horizontalalignment('center')
        
    # Center circle
    center_circle = plt.Circle((0,0), 0.65, color=face_color, fc=face_color, linewidth=0.25)
    ax.add_artist(center_circle)
    
    # Annotate with Total
    values_total = sum(values)
    ax.annotate(str('{0}'.format(values_total)), xy=(0,0), xytext=(0,-0.1), ha='center', size=60)
    
    # Plot
    plt.tight_layout()
    return fig

def simple_bar_with_spines (x_vals, y_vals, **kwargs):
    '''Constructs a simple, but pretty bar chart with spines on left and
    bottom axes.'''
    face_color = kwargs.pop('face_color', '#0068ac')
    text_color = kwargs.pop('text_color', 'white')
    font_size = kwargs.pop('font_size', 20)
    y_lim = kwargs.pop('y_lim', None)
    y_ticks = kwargs.pop('y_ticks', None)
    
    #Set defaults for plot
    plt.rcdefaults()
    # Set backaground color
    plt.rcParams['figure.facecolor'] = face_color
    plt.rcParams['text.color'] = text_color
    plt.rcParams['font.size'] = font_size
    # Build figure
    fig, ax = plt.subplots()
    # Set y_lim
    if y_lim:
        plt.ylim(y_lim)
    ax.bar(x_vals, y_vals, color=text_color, edgecolor=text_color)
    # Set face and spine colors
    ax.patch.set_facecolor(face_color)
    ax.spines['bottom'].set_color(text_color)
    ax.spines['top'].set_color(face_color)
    ax.spines['left'].set_color(text_color)
    ax.spines['right'].set_color(face_color)
    # Set tick colors
    ax.tick_params(axis='x', colors=text_color)
    ax.tick_params(axis='y', colors=text_color)
    
    # Set y_ticks
    if y_ticks:
        plt.yticks(y_ticks)
    # Plot and return
    return fig

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
        ax.plot(yr, rate, '-o', label=group)
        
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