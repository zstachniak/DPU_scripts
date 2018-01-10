# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 14:27:05 2017

@author: astachn1
"""

import win32com.client as win32
import sqlite3
from sqlite3 import OperationalError
from datetime import datetime
import pyodbc
import sys

'''////////////////////////////////////////////////////////////
// Functions.                                                //
////////////////////////////////////////////////////////////'''
def insert_new (cursor, table, data):	
	placeholders = ", ".join(["?" for x in range(len(data))])	
	sql = "INSERT INTO %s VALUES (%s);" % (table, placeholders)	
	cursor.execute(sql, data)

def executeSQL (command, cursor, p=False):
    '''Executes SQL command when passed a command and a cursor.
    If p is set to True, function will also print results of the query.'''
    try:
        cursor.execute(command.strip('\n'))
        if p == True:
            print('\nQuery: \n{0}\n'.format(command.strip('\n')))
            rows = cursor.fetchall()
            for row in rows:
                print(' | '.join(str(x) for x in row))
    except OperationalError as msg:
        print ('Command skipped: ', msg)

def get_last_request (cursor):
    '''Requests the most recent item in the table.'''
    SQL = '''SELECT request_id, MAX(request_date) FROM request_log'''
    cursor.execute(SQL)
    return cursor.fetchone()

def get_last_report (cursor):
    '''Requests the most recent report date in the table.'''
    SQL = "SELECT MAX(send_date) FROM send_log WHERE send_type = 'Weekly Report'"
    cursor.execute(SQL)
    return cursor.fetchone()
  
def verify_sender (sender_email):
    '''Verifies that the sender has the authority to receive the
    requested information.
    '''
    pass

def determine_sender_email (sender_info):
    ''' Returns the true email address of the sender, necessary
    because of the formatting of internal emails:
    
    Example:
        DePaul: '/O=DEPAUL/OU=FIRST ADMINISTRATIVE GROUP/CN=RECIPIENTS/CN=ASTACHN1'
        Outside: 'zstachniak@gmail.com'
    '''
    if dpu_credential in sender_info:
        # user ID appears in last '/CN=' field
        sender_email = sender_info.split('/CN=')[-1] + '@depaul.edu'
    else:
        sender_email = sender_info
    return sender_email

def determine_request_fields(body):
    '''Separates request info in email body into dictionary mapping.'''
    temp_dict = {}
    # Split on new line
    for ln in message.Body.split('\r\n'):
        # Split on tab and strip
        key = ln.split('\\t')[0].strip()
        value = ln.split('\\t')[1].strip()
        temp_dict[key] = value
    return temp_dict

def submit_request_to_access_db(request_dict, cursor, connection):
    '''docstring'''    
    # Initialize lists
    params = []
    values = []
    holders = []
    # Iterate through possible params
    for key in request_param_list:
        # Map Program
        if key == 'Program':
            value = programs_of_study[request_dict[key]]
        # Map Request_Type
        elif key == 'Request_Type':
            value = request_types[request_dict[key]]
        # Map Status from Request_Type
        elif key == 'Status':
            value = request_status[request_dict['Request_Type']]
        elif key == 'Date_Submitted':
            value = datetime.strptime(request_dict[key], '%m/%d/%Y')
        else:
            # Get value, or '' as default
            value = request_dict.get(key, '')
        # if value is anything other than '', use it
        if value != '':
            params.append(key)
            values.append(value)
            holders.append('?')
    # Join lists
    params = ', '.join(params)
    holders = ', '.join(holders)
    # Build SQL string
    ssql = "INSERT INTO Requests (" + params + ") VALUES (" + holders + ");"
    # Execute SQL statement
    cursor.execute(ssql, values)
    # Commit database
    connection.commit()

def weekly_report ():
    '''Creates a weekly report of all affiliation agreements set to expire
    in the next six months.
    '''
    pass

def build_email(request_dict, submission_status):
    '''docstring'''
    
    if request_dict['Request_Type'] == 'Student':
        name = 'Chrisine'
        recipient = 'CSCHLODE@depaul.edu'
        subject = 'Preceptor/Mentor Request'
    else:
        name = 'Tricia'
        recipient = 'PBACLAWS@depaul.edu'
        subject = 'Affiliation Agreement Request'
    
    sub_items = ''
    for key in request_param_list:
        sub_items += key + ': ' + request_dict.get(key, '') + '\n'
        
    body = """Hi {0},

A new {1} was submitted. {2}

{3}

Sincerely,
@alex
Digital Personal Assistant""".format(name, subject, submission_status, sub_items)

    return recipient, subject, body

def send_email (recipient, subject, body, **kwargs):
    '''Sends email using the Outlook client'''
    Cc = kwargs.pop('Cc', None)
    outlook = win32.Dispatch('outlook.application')
    mail = outlook.CreateItem(0)
    mail.To = recipient
    if Cc:
        mail.CC = Cc
    mail.Subject = subject
    mail.Body = body
    mail.Send()

def log_run_status (cursor, run_date, run_status, run_log):
    '''Logs every attempted running of this program, along with error
    messages to indicate failures.'''
    ssql = "INSERT INTO run_status (run_date, run_status, run_log) VALUES (?, ?, ?)"
    cursor.execute(ssql, (run_date, run_status, run_log))

def get_latest_successful_run (cursor):
    '''Returns the max date where program ran successfully.'''
    ssql = "SELECT MAX(run_date) FROM run_status WHERE run_status = 1"
    cursor.execute(ssql)
    return cursor.fetchone()[0]

def process_updates (cursor, most_recent_notification):
    '''Sends email to update faculty of the progress of their requests.'''
    ssql = "SELECT * FROM Request_Status_Route_Table WHERE Route_Date >= ?"
    cursor.execute(ssql, most_recent_notification)
    # Fetch all recent updates
    rows = cursor.fetchall()
    # Iterate through updates
    for row in rows:
        request_id = row[1]
        status = row[2]
        route_date = row[3]
        # Get additional information from Requests
        ssql = "SELECT Requester_First_Name, Requester_Email, Site, Request_Type FROM Requests WHERE ID = ?"
        cursor.execute(ssql, request_id)
        [requester_first_name, requester_email, requester_site, request_type] = cursor.fetchone()
        # If the request originated from a faculty member, update progress
        if request_type == 1:
            # Prepare email
            subject = 'Agreement Request Update'
            body = """Dear {0},
On {1}, your affiliation agreement request for {2} was routed forward to stage: {3}.
Sincerely,
@alex
Digital Personal Assistant""".format(requester_first_name, route_date, requester_site, request_route_stages[status])
            # Send email
            send_email(requester_email, subject, body, Cc='pbaclaws@depaul.edu')

'''////////////////////////////////////////////////////////////
// Hardcoded information.                                    //
////////////////////////////////////////////////////////////'''
application = 'outlook.application'
namespace='MAPI'
folder = 'Automation'
user = 'astachn1@depaul.edu'
dpu_credential = '/O=DEPAUL'
database = 'W:\\csh\\Nursing\\Agreement Request Center (ARC)\\Scripts\\monitor.db'
request_table = 'request_log'
send_table = 'send_log'
# Access
access_conn_str = (
    r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
    r'DBQ=W:\\csh\\Nursing\\Agreement Request Center (ARC)\\Backend\\AffiliationAgreements_Backend.accdb;'
    )
programs_of_study = {'MENP': 1,
                     'DNP': 2,
                     'RN-MS': 3,
                     'MS-APN': 4,
                     'BSN': 5}
request_types = {'Student': 2,
                'Faculty / Staff': 1}
request_status = {'Student': 1,
                  'Faculty / Staff': 4}
request_param_list = ('Requester_Last_Name','Requester_First_Name','Requester_ID','Requester_Email','Program','Course','Needed_By','Date_Submitted','Survey_ID','Preceptor_Last_Name','Preceptor_First_Name','Preceptor_Credentials','Preceptor_Email','Preceptor_Phone','Preceptor_License_Number','Preceptor_License_State','Site','Address_1','Address_2','City','State','Zip_Code','Site_Contact', 'Request_Type', 'Status')

sub_fail = "Unfortunately, I failed to successfully add the request to the ARC database. I have copied Zander so that he can figure out where I've gone wrong. In the meantime, you can manually add this request. :("
sub_success = "I successfully added the request to the ARC database!"

request_route_stages = {1: 'Open', 2: 'Closed', 3: 'Processing', 4: 'Intake', 5: 'Executed', 6: 'Completed', 7: 'Negotiating'}

'''////////////////////////////////////////////////////////////
// Open connection to database.                              //
////////////////////////////////////////////////////////////'''
# Open connection
conn = sqlite3.connect(database)
cursor = conn.cursor()

'''////////////////////////////////////////////////////////////
// Send weekly report, if necessary.                         //
////////////////////////////////////////////////////////////'''

'''
# Get the current date and time
now = datetime.now()
# Get last report
last_report = get_last_report(cursor)
# If a report has not been sent in the past week...
if now - last_report >= 7:
    # Send new weekly report
    
    # Log the sending
    insert_new(cursor,
               send_table,
               [now.strftime('%Y-%m-%d %H-%M-%S'),
                'Weekly Report'])
'''
    
'''////////////////////////////////////////////////////////////
// OUTER PROGRAM LOOP.                                       //
////////////////////////////////////////////////////////////'''
try:

    '''////////////////////////////////////////////////////////////
    // Get most recent message from database and Outlook folder. //
    ////////////////////////////////////////////////////////////'''
    # Request most recent message from database
    (last_id, last_date) = get_last_request(cursor)
    # Establish Outlook connection
    outlook = win32.Dispatch(application).GetNamespace(namespace)
    inbox = outlook.Folders(user).Folders(folder)
    messages = inbox.Items
    # Get the most recent message
    message = messages.GetLast()

    '''////////////////////////////////////////////////////////////
    // Process requests for all emails received since last_id.   //
    ////////////////////////////////////////////////////////////'''
    # Loop while message is more recent than last log   
    while message.EntryID != last_id:
        # Gather information
        sender = determine_sender_email(message.SenderEmailAddress)
        # Process Preceptor / Mentor requests
        if message.Subject == '@alex: New Request':
            request_dict = determine_request_fields(message.Body)
            # Upload request to access database
            # Connect and open cursor
            cnxn = pyodbc.connect(access_conn_str)
            crsr = cnxn.cursor()
            try:
                submit_request_to_access_db(request_dict, crsr, cnxn)
                recipient, subject, body = build_email(request_dict, sub_success)
                send_email(recipient, subject, body)
            except:
                recipient, subject, body = build_email(request_dict, sub_success)
                send_email(recipient, subject, body, Cc='astachn1@depaul.edu')
        
        # Add request to email_log table
        insert_new(cursor, 
                   request_table, 
                   [message.EntryID,
                    message.ReceivedTime.strftime('%Y-%m-%d %H-%M-%S'),
                    message.Subject.split(':')[1].strip(),
                    sender])
        
        # Loop through messages until hit the last_id
        message = messages.GetPrevious()
    
    '''////////////////////////////////////////////////////////////
    // Send notifications for updates since last successful run. //
    ////////////////////////////////////////////////////////////'''
    # Get last successful run
    last_success = get_latest_successful_run(cursor)
    process_updates(crsr, last_success)
    
    # Log a successful run
    log_run_status(cursor, datetime.now(), 1, '')
    
# Log an unsuccessful run with error message
except:
    e_log = '{}: {}'.format(sys.exc_info()[0].__name__, sys.exc_info()[1])
    log_run_status(cursor, datetime.now(), 0, e_log)
    
'''////////////////////////////////////////////////////////////
// Close database connections.                                //
////////////////////////////////////////////////////////////'''
crsr.close()
cnxn.close()
cursor.close()
conn.commit()
conn.close() 

'''////////////////////////////////////////////////////////////
// Table Definitions.                                        //
////////////////////////////////////////////////////////////'''
CREATE_TABLE = '''CREATE TABLE request_log (
        request_id                  VARCHAR(200),
        request_date                DATE,
        request_type                VARCHAR(40),
        sender                      VARCHARD(40),
        CONSTRAINT email_log PRIMARY KEY (request_id)
);'''

CREATE_TABLE = '''CREATE TABLE send_log (
        send_date                   DATE,
        send_type                   VARCHAR(40)
);'''

CREATE_TABLE = '''CREATE TABLE run_status (
        run_date                    DATE,
        run_status                  NUMBER,
        run_log                     VARCHAR(200)
);'''
