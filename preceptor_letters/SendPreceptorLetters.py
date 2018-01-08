# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 14:49:40 2017

@author: astachn1
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 09:44:10 2017

@author: astachn1
"""

import os
from datetime import datetime
import pandas as pd
import win32com.client as win32
import re
from time import sleep
import errno

#############################################################
# Official Organization Name Mappings
#############################################################
org_name = {'Advocate Lutheran General': 'Advocate Health and Hospitals Corporation',
           'Advocate Lutheran General Hospital': 'Advocate Health and Hospitals Corporation',
           'Lutheran General': 'Advocate Health and Hospitals Corporation',
           'Highland Park Hospital': 'NorthShore University HealthSystem',
           'Holy Cross': 'Sinai Health System',
           'Holy Cross Hospital': 'Sinai Health System',
           'Mt. Sinai': 'Sinai Health System',
           'Mt. Sinai Hospital': 'Sinai Health System',
           "Lurie Children's": "Ann & Robert H. Lurie Children's Hospital",
           "Lurie Children's Hospital": "Ann & Robert H. Lurie Children's Hospital",
           'MacNeal Hospital': 'VHS of Illinois, Inc',
           'MacNeal': 'VHS of Illinois, Inc',
           'Mercy Hospital': 'Mercy Hopsital and Medical Center',
           'NorthShore Evanston': 'NorthShore University HealthSystem',
           'NorthShore': 'NorthShore University HealthSystem',
           'Northwestern': 'Northwestern Memorial HealthCare',
           'Skokie Hospital': 'NorthShore University HealthSystem',
           'Skokie Hospital (NorthShore)': 'NorthShore University HealthSystem',
           "St. Anthony's": "OSF Saint Anthony Medical Center",
           "RIC": "Rehabilitation Insitute of Chicago",
           'Shirley Ryan AbilityLab (RIC)': 'Rehabilitation Insitute of Chicago',
           "St. Francis": "Presence RHC Corporation",
           "St. Francis Hospital": "Presence RHC Corporation",
           "West Suburban Medical Center": "West Suburban Medical Center",
           "West Suburban": "West Suburban Medical Center"}

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
        if os.path.isfile(subname) and 'Clinical Roster' in subname and '~$' not in subname:
            #Ignore files that end in '_(2).xlsx'
            if (subname[(subname.find('.')-3):subname.find('.')]) == '(2)':
                pass
            else:
                files.append(subname)
                
                #Parses file names and converts to datetimes
                date = subname[(subname.find('.')-10):subname.find('.')]
                date_time = datetime.strptime(date, '%Y-%m-%d').date()
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
            if str(latest.strftime('%Y-%m-%d')) in file:
                return file

def remove_files (folder):
    '''Removes all files in a given folder.'''
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

def send_email (recipient, subject, body, attachments):
    '''Sends email using the Outlook client'''
    outlook = win32.Dispatch('outlook.application')
    mail = outlook.CreateItem(0)
    mail.To = recipient
    mail.Subject = subject
    mail.Body = body
    if attachments:
        for attachment in attachments:
            mail.Attachments.Add(Source = attachment)
    mail.Send()

def email_preceptor (row):
    '''Emails each preceptor in the file.'''
    # Get coordinator's email
    recipient = row['Preceptor Email']
    # Write subject
    subject = 'DePaul University Preceptorship'
    # Write body   
    body = '''To: {0} {1}
    Re:	DePaul University Preceptorship

    Dear {2}:
    Thank you for agreeing to serve as a Preceptor for DePaul University student {3} {4} (the "Student"), for the time period {5}. 
    
    As a Preceptor you will provide a supervised learning experience for the Student, who will be simultaneously enrolled in the course NSG {6}-{7}: {8} (the "Course"). The Student's specific objectives for the learning experience and the Course will be jointly planned and approved by the University, the Preceptor, and the Student at the initiation of the learning experience. The learning experience must provide the Student with a minimum of {9} hours of direct, supervised clinical practice consistent with the State of Illinois Nurse Practice Act and in keeping with your typical responsibilities in your work setting. Evaluating the Student's achievement of the approved objectives will be a joint responsibility of the University, the Preceptor, and the Student at the conclusion of the learning experience.
    
    Your specific responsibilities as Preceptor are outlined in the attached document, entitled "Preceptor, Student, and Faculty Responsibilities."  We would also encourage you to review the Affiliation Agreement between DePaul University and your site for more information about the rights and responsibilities of various parties with respect to the placement. Please be sure to share this letter with your immediate supervisor. In the event that you cannot fulfill these responsibilities and your immediate supervisor assigns another preceptor for the Student, this letter applies to anyone who may be assigned as preceptor for the Student.
    
    I again thank you for agreeing to participate in this program and provide an invaluable learning experience for {10} {11}.  If you have any questions or concerns, please do not hesitate to contact me.
    
    Sincerely,
    
    Alexander Stachniak
    Coordinator of Data Management'''.format(row['Preceptor First Name'], row['Preceptor Last Name'], row['Preceptor First Name'], row['Student First Name'], row['Student Last Name'], row['Date'], row['Cr'], row['Sec'], row['Course Title'], row['Hours'], row['Student First Name'], row['Student Last Name'])
    
    # Get attachments    
    attachments = [duties]
    # Send email
    try:
        send_email(recipient, subject, body, attachments)
    except Exception as e:
        print('Exception occured for {0}: {1}'.format(recipient, e))
    
    # Map clinical site to official organization name
    org = [org_name[k] for k in [row['Clinical Site']]][0]
    
    # Save a copy of the email
    save_path = ending_path + '\\' + org + '\\' + 'Preceptor Letters of Agreement' + '\\' + TermDescriptions.loc[TermDescriptions['Term'] == row['Term'], 'Academic Year'].item() + '\\' + TermDescriptions.loc[TermDescriptions['Term'] == row['Term'], 'Quarter'].item()   
    title = row['Preceptor Last Name'] + ', ' + row['Preceptor First Name']
    # Sleep for 1 second to allow time for Outlook to refresh
    sleep(1)
    save_email('Preceptor', save_path, title)

def test_path (save_path):
    '''Creates path. If exception = Path Exists, ignore. Otherwise raise.'''
    try:
        os.makedirs(save_path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def save_email (folder, save_path, title):
    '''Saves the last email in a particular folder.
    Additional documentation: https://msdn.microsoft.com/en-us/library/microsoft.office.interop.outlook.mailitem_methods.aspx
    '''
    outlook = win32.Dispatch('outlook.application').GetNamespace("MAPI")
    inbox = outlook.Folders("astachn1@depaul.edu").Folders(folder)
    messages = inbox.Items
    message = messages.GetLast()
    test_path(save_path)
    message.SaveAs(save_path + '\\' + title + '.msg')
    
#############################################################
# Grab Data
#############################################################
# Set the starting path
starting_path = 'W:\\csh\\Nursing Administration\\Clinical Placement Files\\2017-2018\\Winter'
ending_path = 'W:\\csh\\Nursing\\Affiliation Agreements'
# Read in latest preceptors file
file = get_latest(starting_path)
preceptors = pd.read_excel(file, header=0, converters={'Term':str, 'Cr':str, 'Empl ID':str, 'Hours':str})
# Drop any rows without a preceptor email
preceptors.dropna(subset=['Preceptor Email'], inplace=True)
# Drop any rows with "TBD" as preceptor email
preceptors = preceptors[~preceptors['Preceptor Email'].isin(['TBD', 'TBA'])]
# Drop any rows where preceptor email was already sent or will be sent by RFU
preceptors = preceptors[preceptors['Preceptor Letter Sent'].isnull()]
#Read in term descriptions
TermDescriptions = pd.read_excel('W:\\csh\\Nursing\\Schedules\\Term Descriptions.xlsx', header=0, converters={'Term':str})
# Set path for attachment
duties = 'W:\\csh\\Nursing\\Preceptor-Mentor Requests\\MENP\\Preceptor, Student, and Faculty Responsibilities for MENP Program.pdf'

#############################################################
# Main
#############################################################
preceptors.apply(email_preceptor, axis=1)
        