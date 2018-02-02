# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 15:38:18 2018

@author: astachn1
"""

import dpu.scripts as dpu
from dpu.file_locator import FileLocator
import matplotlib.pyplot as plt
import os

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
