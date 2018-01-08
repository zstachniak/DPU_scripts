# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 14:37:07 2016

@author: astachn1
"""

## Import analysis modules
import pandas as pd
#pd.set_option('display.max_rows', 25)
import numpy as np
#np.set_printoptions(threshold=np.nan)

from numpy import nan, isnan, mean, std, hstack, ravel
from sklearn.cross_validation import train_test_split, cross_val_score, KFold, LeaveOneOut, LeavePOut, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Binarizer, Imputer, LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA, KernelPCA
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, roc_curve, auc, make_scorer
from sklearn.grid_search import GridSearchCV
from sklearn import neighbors, datasets, svm, preprocessing, linear_model
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier

## Import visualization modules
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.colors import ListedColormap

## Import SciPy
from scipy.sparse import issparse
import scipy.stats as stats

def plot_decision_regions(X,y,classifier,resolution=0.02):
    '''Plots the decision regions of a classifier. Function is credit to
    Sebastian Raschka, Python Machine Learning.'''
    
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:,0].min() - 1, X[:,0].max() + 1
    x2_min, x2_max = X[:,1].min() - 1, X[:,1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)

#Read ATI Data
ATI = pd.read_csv('W:\\csh\\Nursing Administration\\Data Management\\DataWarehouse\\Testing\\ATI.csv',delimiter=',',na_values='nan',usecols=['Assessment', 'Assessment ID', 'Date Taken', 'National Mean', 'National Percentile', 'Proficiency Level', 'Program Mean', 'Program Percentile', 'Score', 'Section', 'Student ID', 'User ID'],dtype={'Student ID':str, 'Assessment ID':str, 'User ID':str},parse_dates=['Date Taken'])

#Rename section headings that are 'overall' scores
def overall(Assessment, Section):
    if Section == Assessment:
        return 'Overall'
    else:
        return Section
ATI['Section'] = ATI.apply(lambda x: overall(x['Assessment'], x['Section']), axis=1)

#Keep only the overall scores
ATI = ATI[(ATI['Section']=='Overall')]
ATI = ATI.drop('Section', 1)

#Drop any rows where Student ID is NaN
#ATI['Student ID'].isnull().any()
ATI = ATI.dropna(subset=['Student ID'])

#Convert Proficiency Levels to integer values.
#Impute missing values.
#print(ATI['Proficiency Level'].unique())    #list unique values
proficiency_map = {'Below Level 1': 0,
                   'Level 1': 1,
                   'Level 2': 2,
                   'Level 3': 3                   
                   }

ATI['Proficiency Level'] = ATI['Proficiency Level'].map(proficiency_map)

#Create categories for all Assessments
def Categorize (string):
    kw_dict = {
               'Mental Health': 'Psych',
               'Comprehensive': 'Comprehensive',
               'Medical-Surgical': 'MedSurg',
               'Medical Surgical': 'MedSurg',
               'Fundamentals': 'Fundamentals',
               'Essential Academic Skills': 'Fundamentals',
               'Pharmacology': 'Pharm',
               'Maternal': 'OB',
               'Community Health': 'Community',
               'Leadership': 'Leadership',
               'Care of Children': 'Peds',
               'Critical Thinking': 'Critical Thinking'
               }
    
    for key in kw_dict:
        if key in string:
            return kw_dict[key]
ATI['Category'] = ATI.apply(lambda x: Categorize(x['Assessment']), axis=1)

#Treat duplicate Assessment Categories
ATI = ATI.sort(columns=['Date Taken'])
# 1. Take most recent attempt (last)
ATI = ATI.drop_duplicates(subset=['Student ID', 'Category'], keep='last')
# 2. Take first attempt (first)
#ATI = ATI.drop_duplicates(subset=['Student ID', 'Category'], keep='first')
# 3. Average

#Drop true duplicates
ATI = ATI.drop_duplicates()

#Drop NaN in National Percentage (seems to only accompany duplicates)
ATI = ATI.dropna(axis=0,how='any',subset=['National Percentile'])

#Treat duplicate Assessment IDs taken on the same date (these represent errors)
# Take highest score
ATI = ATI.drop_duplicates(subset=['Student ID', 'Assessment ID', 'Date Taken'], keep='last')

#Treat duplicate Assessment IDs taken on separate dates
# 1. Take most recent attempt (last)
ATI = ATI.drop_duplicates(subset=['Student ID', 'Assessment ID'], keep='last')
# 2. Take first attempt (first)
#ATI = ATI.drop_duplicates(subset=['Student ID', 'Assessment ID'], keep='first')
# 3. Average

#Drop tests with no program mean
ATI = ATI.dropna(axis=0, subset=['Program Mean'])

#Create train and test set for imputing Proficiency Level
#First, create df that contains data with class labels for Proficiency Level
x = ATI[(ATI['Proficiency Level'].notnull())]
#Drop non-numeric data
x = x.drop(['Date Taken', 'Student ID', 'User ID', 'Assessment', 'Assessment ID', 'Category'],1)
x = x.dropna()
y = x['Proficiency Level'].as_matrix()
x = x.drop('Proficiency Level',1)
#Use train test split for cross-validation
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#Start with Support Vector Machines (SVM)
#Start by pre-scaling the X data
x_train_scaled = preprocessing.scale(x_train)
x_test_scaled = preprocessing.scale(x_test)
##Use grid search to determine best parameters
#SVM_params = [
#              {'kernel':['rbf'],
#               'gamma':[1e-1,1e-2,1e-3,1e-4],
#               'C':[1, 10, 100, 1000],
#               'class_weight':['balanced']
#               },
#               {'kernel':['sigmoid'],
#               'gamma':[1e-1,1e-2,1e-3,1e-4],
#               'C':[1, 10, 100, 1000],
#               'class_weight':['balanced']
#               },
#              {'kernel':['linear'],
#               'C':[1, 10, 100, 1000],
#               'class_weight':['balanced']
#               },
#               {'kernel':['poly'],
#                'degree':[2,3,4],
#               'gamma':[1e-1,1e-2,1e-3,1e-4],
#               'C':[1, 10, 100, 1000],
#               'class_weight':['balanced']
#               }
#              ]
##SVM_params = {'kernel':['linear','poly','rbf','sigmoid'],
##              'class_weight':['balanced']
##              }
#svm_clf = svm.SVC()
#gs = GridSearchCV(svm_clf,param_grid=SVM_params,scoring='f1_macro')
#gs.fit(x_train_scaled,y_train)
#gs.best_score_
#gs.best_estimator_
#gs.scorer_
#est = gs.best_estimator_
#est.fit(x_train_scaled,y_train)
#y_pred = est.predict(x_test_scaled)
#confusion_matrix(y_test, y_pred)
#print (classification_report(y_test, y_pred))

#K Nearest Neighbors
#Use grid search to determine best parameters
k = list(range(3,33,3)) #create a range for k
params = {'n_neighbors':k,
          'weights':['uniform', 'distance'],
          'algorithm':['ball_tree', 'kd_tree', 'brute'],
          'p':[1,2,3]
          }
'''Scoring Options:
Valid options are ['accuracy', 'adjusted_rand_score', 'average_precision', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'neg_log_loss', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_median_absolute_error', 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc']
'''
knn_clf = neighbors.KNeighborsClassifier()
knn_gs = GridSearchCV(knn_clf,param_grid=params,scoring='f1_macro')
knn_gs.fit(x_train,y_train)
knn_gs.best_score_
knn_gs.best_estimator_
knn_gs.scorer_
knn_est = knn_gs.best_estimator_
knn_est.fit(x_train,y_train)
y_pred = knn_est.predict(x_test)
confusion_matrix(y_test, y_pred)
print (classification_report(y_test, y_pred))

#Define the best fit model for imputing Proficiency Level
best_fit = knn_est

#Impute NaNs for Proficiency Level
x_impute = ATI[(ATI['Proficiency Level'].isnull())]
x_impute = x_impute.drop(['Date Taken', 'Student ID', 'User ID', 'Assessment', 'Assessment ID','Category'],1)
x_impute['Proficiency Level'] = best_fit.predict(x_impute[['National Mean','National Percentile','Program Mean','Program Percentile','Score']])

#Add imputed values to ATI df using fillna
ATI['Proficiency Level'].fillna(x_impute['Proficiency Level'], inplace=True)

#Check Shape
rows, columns = ATI.shape

#Review Numeric
numeric = ATI.describe(include=['number']).T.reset_index()
numeric.rename(columns={'index':'feature'},inplace=True)
numeric.insert(1,'missing',(rows - numeric['count'])/ float(rows))

numeric.head(15)

#Review Discrete
discrete = ATI.describe(include=['object']).T.reset_index()
discrete.rename(columns={'index':'feature'},inplace=True)
discrete.insert(1,'missing',(rows - discrete['count'])/ float(rows))

discrete.head(15)

#Read NCLEX
NCLEX = pd.read_excel('W:\\csh\\Nursing Administration\\Data Management\\DataWarehouse\\OG_Data\\NCLEX_Results\\NCLEX.xlsx',header=0,converters={'Empl ID':str})
#Fill first zero where needed
NCLEX['Empl ID'] = NCLEX['Empl ID'].str.zfill(7)
#Drop unnecessary fields
NCLEX.drop(['Last Name','First Name','DOB','In-Out','Time Delivered','Year','Quarter'],axis=1,inplace=True)
#Add days elapsed since graduation
NCLEX['Days Elapsed'] = NCLEX['Date Delivered'] - NCLEX['Graduation Date']
NCLEX['Days Elapsed'] = NCLEX['Days Elapsed'].dt.days

#NCLEX.head()

#Read Grad Data
Graduates = pd.read_csv('W:\\csh\\Nursing Administration\\Data Management\\DataWarehouse\\Testing\\Grad.csv', delimiter=',',na_values='nan',usecols=['ID','Degree','Acad Plan','Sub-Plan','Admit Term','Compl Term','Confer Dt','GPA'],dtype={'ID':str},parse_dates=['Confer Dt'])
#Multiple steps to remove trailing ".0" and fill first zero where needed
Graduates['ID'] = [s.rstrip("0") for s in Graduates['ID']]
Graduates['ID'] = [s.rstrip(".") for s in Graduates['ID']]
Graduates['ID'] = Graduates['ID'].str.zfill(7)

'''Compute number of quarters taken to graduate by subtracting admit quarter from
# graduation quarter. The quarters typically count by fives, but there are several
# inconsistencies, which must be accounted for based on the range of the terms.'''
def qtrs(admit, grad):
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
Graduates['Qtrs to Grad'] = Graduates.apply(lambda x: qtrs(x['Admit Term'], x['Compl Term']), axis=1)

'''
# For reporting purposes only
'''
'''
df = pd.merge(Graduates, ATI[['Student ID', 'Date Taken', 'Assessment ID', 'Assessment', 'Category', 'Score','National Mean', 'National Percentile', 'Program Mean', 'Program Percentile', 'Proficiency Level']], how='inner', left_on='ID', right_on='Student ID', sort=True, copy=True)
df = df.drop('ID', 1)

# Fix some stuff
df.loc[df['Student ID'] == '1315669', 'Days Elapsed'] = 31.0
df = df.drop(df[df['Student ID'] == '0858995'].index)
df = df.drop(df[df['Degree'] != 'MS'].index)
df = df.drop('Degree',axis=1)
df.loc[df['Student ID'] == '0917861', 'Qtrs to Grad'] = 22
df = df.drop(['Acad Plan', 'Sub-Plan', 'Admit Term', 'GPA', 'Qtrs to Grad', 'Days Elapsed'],axis=1)

# Aggregate by year
mask_2014 = (df['Date Taken'] >= '2014-1-1') & (df['Date Taken'] < '2015-1-1')
df_2014 = df.loc[mask_2014]
mask_2015 = (df['Date Taken'] >= '2015-1-1') & (df['Date Taken'] < '2016-1-1')
df_2015 = df.loc[mask_2015]
mask_2016 = (df['Date Taken'] >= '2016-1-1') & (df['Date Taken'] < '2017-1-1')
df_2016 = df.loc[mask_2016]

df_2014['Proficiency Level'].value_counts()
df_2015['Proficiency Level'].value_counts()
df_2016['Proficiency Level'].value_counts()

cohorts = [930, 940, 950, 960, 970, 980]

for item in cohorts:
    cohort = df[(df['Compl Term'] == item) & (df['Category'] == 'Comprehensive')]
    cohort = cohort.sort(columns=['Date Taken'])
    attempt1 = cohort.drop_duplicates(subset=['Student ID', 'Category'], keep='first')
    pass1 = attempt1[attempt1['Proficiency Level'] >= 2]
    attempt2 = cohort.drop_duplicates(subset=['Student ID', 'Category'], keep='last')
    pass2 = attempt2[attempt2['Proficiency Level'] >= 2]
    
    print('Cohort: {}'.format(item))
    print('N: {}'.format(len(attempt1)))
    print('Attempt 1: {}/{} = {}%'.format(len(pass1),len(attempt1), (len(pass1)/len(attempt1))))
    print('Attempt 2: {}/{} = {}%'.format(len(pass2),len(attempt2), (len(pass2)/len(attempt2))))
'''

#Combine NCLEX and Graduates into Temp dataframe
Temp = pd.merge(NCLEX[['Empl ID', 'Result', 'Days Elapsed']], Graduates[['ID', 'GPA', 'Qtrs to Grad', 'Degree']], how='inner', left_on='Empl ID', right_on='ID', sort=True, copy=True)

#Combine Temp with ATI for final dataframe
df = pd.merge(Temp, ATI[['Student ID', 'Date Taken', 'Assessment ID', 'Assessment', 'Category', 'Score','National Mean', 'National Percentile', 'Program Mean', 'Program Percentile', 'Proficiency Level']], how='left', left_on='Empl ID', right_on='Student ID', sort=True, copy=True)
df = df.drop('ID', 1)
df = df.drop('Student ID', 1)

'''We do not have the testing dates for students who take NCLEX out of state.
#There are also a few oddities, where students test before they technically
#receive their degrees. To avoid these five outlier cases from affecting 
#our model, we will impute all NaN, 0, and negative values using the mean
#of the group as a whole, rounded to the nearest unit value.'''
elapsed = round(df.mean()['Days Elapsed'],0)
df['Days Elapsed'].fillna(elapsed, inplace=True)
df.loc[df['Days Elapsed'] <= 0, 'Days Elapsed'] = elapsed

'''Manually update the Days Elapsed for one student who showed as an outlier.
# Upon further inspection, the graduation date was an error.'''
df.loc[df['Empl ID'] == '1315669', 'Days Elapsed'] = 31.0

'''Based on a review of the distribution of Days Elapsed, one outlier was detected.
# One student did not test until 483 days after graduation, with the next highest
# at 191. Values were checked for accuracy. Given the extremeness of the outlier,
# a decision was made to drop the observation entirely to increase model performance.

#Distribution of Qtrs to Grad
h = sorted(df['Days Elapsed'])
fit = stats.norm.pdf(h, np.mean(h), np.std(h))
plt.plot(h,fit,'-o')
plt.hist(h,normed=True)
plt.show()
'''
df = df.drop(df[df['Empl ID'] == '0858995'].index)

#Drop the entries for non-MS degrees (these represent students who earned multiple
#degrees, and we only care about the MENP degree)
df = df.drop(df[df['Degree'] != 'MS'].index)
df = df.drop('Degree',axis=1)

#Manually update the Qtrs to Grad for one student who stopped out and restarted
#and currently displays two separate values
df.loc[df['Empl ID'] == '0917861', 'Qtrs to Grad'] = 22

#Count Number of unique students
df['Empl ID'].nunique()

#Drop unused columns
df = df.drop(['Date Taken', 'Assessment ID', 'Assessment'],axis=1)

#Calculate distance to National Mean
df['Dist_National_Mean'] = df.apply(lambda x: x['Score'] - x['National Mean'],axis=1)

#Calculate distance to Program Mean
df['Dist_Program_Mean'] = df.apply(lambda x: x['Score'] - x['Program Mean'],axis=1)

#Drop NaNs
df = df.dropna(axis=0, subset=['Score'])

#Drop National Mean and Program Mean, which are standard across many scores
# and so not appropriate to include in our model.
df = df.drop(['National Mean', 'Program Mean'], axis=1)

#Consider dropping Community (30.13%) and Leadership (8.28%).
#Later imputation of values may be skewing results.
df['Category'].value_counts()
df = df.drop(df[df['Category'] == 'Community'].index)
df = df.drop(df[df['Category'] == 'Leadership'].index)

#Long to Wide reshape
df_wide_pivoted = df.set_index(['Empl ID','Result','Days Elapsed','GPA','Qtrs to Grad','Category']).unstack('Category')
df_wide_pivoted_reindex = df_wide_pivoted.reset_index(level=['Result','Days Elapsed','GPA','Qtrs to Grad'])
df_wide_pivoted_reindex.columns = [' '.join(col).strip() for col in df_wide_pivoted_reindex.columns.values]
list(df_wide_pivoted_reindex)

#Impute Missing Values
x = df_wide_pivoted_reindex.drop('Result',axis=1)
y = df_wide_pivoted_reindex['Result']
imp = Imputer(missing_values='NaN', strategy='mean', axis=0, copy=False)
imp.fit_transform(x)

'''At this point, we output to CSV as a means of avoiding replicating the above
steps. We output in two formats, as long and as wide.'''
df.to_csv('W:\\csh\\Nursing Administration\\Data Management\\ATI\\ATI_Long.csv')
#Re-combine for outputing wide format
frames=[x,y]
out = pd.concat(frames, axis=1)
out.to_csv('W:\\csh\\Nursing Administration\\Data Management\\ATI\\ATI_Wide.csv')

#Binarize Result
le = LabelEncoder()
df_wide_pivoted_reindex['Result'] = le.fit_transform(ravel(df_wide_pivoted_reindex['Result']))

#Set y as Result as matrix type
y = df_wide_pivoted_reindex['Result'].as_matrix()

#Use train test split for cross-validation (stratified for class imbalance)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=25, stratify=y)

#Scale Data using StandardScaler so that the same scale is applied to both
#training and testing set.
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

'''
For each student, and for each ATI subject area, we have the following six data points:

Raw Score
National Percentile
Program Percentile
Proficiency Level
Distance from National Mean
Distance from Program Mean

As can be shown, all of these data points for a specific subject area are highly correlated to one another. This makes intuitive sense, as the raw score would be used to calculate all other data points.

In the correlation matrix below the noticeable diagonal bands indicate the high correlation between the six data points for each subject.
'''
#Assessing correlation and covariance
import seaborn as sns
#Calculate correlation matrix
corr = df_wide_pivoted_reindex.corr()
#Create a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
#Set up plot
f, ax = plt.subplots(figsize=(11,9))
#Generate a colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
#Plot
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

'''
This high degree of correlation is perhaps better represented by taking only one subject at a time. In the plot below, the correlation between the six data points for the comprehensive exam are plotted. Notice the legend is between the absolute values 0.90 and 1.00. This range is confirmed if we print the actual correlation values.
'''
#Take only the columns related to the Comprehensive test
comp = df_wide_pivoted_reindex.loc[:, df_wide_pivoted_reindex.columns.to_series().str.contains('Comprehensive').tolist()]
#Calculate correlation matrix
corr = comp.corr()
#Create a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
#Set up plot
f, ax = plt.subplots(figsize=(11,9))
#Generate a colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
#Plot
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
#Print correlation matrix
corr

'''
Given the high amount of correlation, it will be necessary to remove features from our dataset before we can make any predictions. Otherwise, we risk major issues due to covariance. Removing features should also help with computation time and with interpretability.
'''

'''
We will start by attempting to discover feature importance using random forests. When using all data points, it is difficult to uncover which of the six is most effective for each subject area.

If we cycle through one subject at a time, however, a strong pattern emerges where Distance from National Mean is the strongest predictor for all but MedSurg and Critical Thinking (it is a close second for both). Given this information, Distance from National Mean seems like a good choice for future predictive models. This choice also makes intuitive sense when we consider that the NCLEX is a nationally standardized exam.
'''
#Assessing all Feature Importances with Random Forests
feat_labels = df_wide_pivoted_reindex.columns[1:]
#fit forest
forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
#print importances
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
#plot importances
plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]),
        importances[indices],
        color='lightblue',
        align='center')
plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

#Do the same, but looping through Subject Areas
subject_list = ['Fundamentals', 'Comprehensive', 'Pharm', 'OB', 'Peds', 'MedSurg', 'Critical Thinking', 'Psych']
for subject in subject_list:
    #Pull out subject as X
    Subject_train = X_train.loc[:, X_train.columns.to_series().str.contains(subject).tolist()]
    #Capture labels
    feat_labels = Subject_train.columns[:]
    #Fit forest
    forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
    forest.fit(Subject_train, y_train)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    #print importances
    for f in range(Subject_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
    #plot importances
    plt.title('Feature Importances')
    plt.bar(range(Subject_train.shape[1]),
            importances[indices],
            color='lightblue',
            align='center')
    plt.xticks(range(Subject_train.shape[1]), feat_labels[indices], rotation=90)
    plt.xlim([-1, Subject_train.shape[1]])
    plt.tight_layout()
    plt.show()

'''
We now define a new X_train and X_test for future modeling. 
'''
#Define new X_train and X_test beased on Dist_National_Mean
NatX_train = X_train.loc[:, X_train.columns.to_series().str.contains('Dist_National_Mean').tolist()]
frames = [NatX_train, X_train[['Days Elapsed', 'GPA', 'Qtrs to Grad']]]
NewX_train = pd.concat(frames, axis=1)

NatX_test = X_test.loc[:, X_test.columns.to_series().str.contains('Dist_National_Mean').tolist()]
frames = [NatX_test, X_test[['Days Elapsed', 'GPA', 'Qtrs to Grad']]]
NewX_test = pd.concat(frames, axis=1)

'''
At this point, we can run our random forest classifier again to see if there are additional features we can remove. Of note is "Qtrs to Grad" which exhibits very little predictive power. It is suggested that we remove this feature from future models. If considered logically, there is an extreme amount of noise in our data between students who took 7 quarters to graduate and those who took 8. The vast majority of students that needed an extra quarter to graduate will not be adequately captured. Furthermore, once we move past 8 quarters, there are so few samples that these cases are effectively outliers. 
'''
#Assessing all Feature Importances with Random Forests
feat_labels = NewX_train.columns[:]
#fit forest
forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
forest.fit(NewX_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
#print importances
for f in range(NewX_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
#plot importances
plt.title('Feature Importances')
plt.bar(range(NewX_train.shape[1]),
        importances[indices],
        color='lightblue',
        align='center')
plt.xticks(range(NewX_train.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, NewX_train.shape[1]])
plt.tight_layout()
plt.show()

'''
Drop Quarters to Graduate from our training and test set, then standardize for models which need the same units.
'''
#Drop Quarters to Graduate
NewX_train = NewX_train.drop('Qtrs to Grad', 1)
NewX_test = NewX_test.drop('Qtrs to Grad', 1)

#Scale data
scaler = preprocessing.StandardScaler().fit(NewX_train)
NewX_train_std = scaler.transform(NewX_train)
NewX_test_std = scaler.transform(NewX_test)

'''
We can evaluate model performance using a learning curve.
'''

pipe_lr = Pipeline([
        ('scl', StandardScaler()),
        ('clf', LogisticRegression(
                penalty='l2', random_state=0))])
train_sizes, train_scores, test_scores = learning_curve(estimator = pipe_lr, X = NewX_train, y = y_train, train_sizes = np.linspace(0.1, 1.0, 10), cv = 10, n_jobs = -1, scoring='precision')

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('precision score')
plt.legend(loc='best')
plt.show()

'''
Grid Search
'''

pipe_svc = Pipeline([
        ('scl', StandardScaler()),
        ('clf', SVC(random_state=1))])
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'clf__C': param_range,
               'clf__kernel': ['linear']},
            {'clf__C': param_range,
             'clf__gamma': param_range,
             'clf__kernel': ['rbf']}]
gs = GridSearchCV(estimator = pipe_svc, param_grid = param_grid, scoring = 'precision', cv = 10, n_jobs = -1)
gs = gs.fit(NewX_train, y_train)
print(gs.best_score_)
print(gs.best_params_)

clf = gs.best_estimator_
clf.fit(NewX_train, y_train)
print('Test precision: %.3f' % clf.score(NewX_test, y_test))

'''
Grid search with nested cross-validation
'''
gs = GridSearchCV(estimator = pipe_svc, param_grid = param_grid, scoring = 'precision', cv = 2, n_jobs = -1)
scores = cross_val_score(gs, NewX_train, y_train, scoring='precision', cv=5)
print('CV precision: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

'''
Bagging
'''
tree = DecisionTreeClassifier(criterion='entropy', max_depth=None, random_state=1)
bag = BaggingClassifier(base_estimator=tree, n_estimators=500, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1, random_state=1)
bag = bag.fit(NewX_train, y_train)
y_test_pred = bag.predict(NewX_test)
confusion_matrix(y_test, y_test_pred)
print(classification_report(y_test, y_test_pred))

'''
AdaBoost
'''
tree = DecisionTreeClassifier(criterion='entropy', max_depth=None, random_state=1)
ada = AdaBoostClassifier(base_estimator=tree, n_estimators=500, learning_rate=0.1, random_state=0)
ada = ada.fit(NewX_train, y_train)
y_test_pred = ada.predict(NewX_test)
confusion_matrix(y_test, y_test_pred)
print(classification_report(y_test, y_test_pred))

'''
AdaBoost with a different scoring method (chaging class of interest to 0)
'''
scorer = make_scorer(precision_score, pos_label=0)
pipe_svc = Pipeline([
        ('scl', StandardScaler()),
        ('clf', SVC(random_state=1))])
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'clf__C': param_range,
               'clf__kernel': ['linear']},
            {'clf__C': param_range,
             'clf__gamma': param_range,
             'clf__kernel': ['rbf']}]
gs = GridSearchCV(estimator = pipe_svc, param_grid = param_grid, scoring = scorer, cv = 10, n_jobs = -1)
gs = gs.fit(NewX_train, y_train)
print(gs.best_score_)
print(gs.best_params_)

clf = gs.best_estimator_
clf.fit(NewX_train, y_train)
print('Test precision: %.3f' % clf.score(NewX_test, y_test))









'''
We can attempt to use Principal Component Analysis to reduce the number of dimensions further (and reduce feature space). However, there does not appear to be a clear reason to do so. If we look at the step plot of the explained variance, only the first principal component covers a large amount of the variance, and there is no clear knee afterwards.
'''
#Principal Component Analysis
pca = PCA(n_components=None)
NewX_train_pca = pca.fit_transform(NewX_train_std)
NewX_test_pca = pca.transform(NewX_test_std)
pca.explained_variance_ratio_
#Cumulative variance
cum_var_exp = np.cumsum(pca.explained_variance_ratio_)
#Plot it
plt.bar(range(1,len(NewX_train.columns)+1), pca.explained_variance_ratio_, alpha=0.5, align='center', label='individual explained variance')
plt.step(range(1,len(NewX_train.columns)+1), cum_var_exp, where='mid', label='cumulative explained variance')
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Components')
plt.legend(loc='best')
plt.show()

'''
Out of curiousity, we can train a logistic regression model and analyze performance with only 2 principal components. As expected, performance is not optimal.
'''
# PCA using 2 components
pca = PCA(n_components=2)
NewX_train_pca = pca.fit_transform(NewX_train_std)
NewX_test_pca = pca.transform(NewX_test_std)
# train Logistic Regression model
lr = LogisticRegression()
lr.fit(NewX_train_pca, y_train)
# Plot decision regions
plot_decision_regions(NewX_train_pca, y_train, classifier=lr)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='best')
plt.show()

'''
Linear Discriminant Analysis, or LDA.
'''
lda = LinearDiscriminantAnalysis()
NewX_train_lda = lda.fit_transform(NewX_train_std, y_train)
NewX_test_lda = lda.transform(NewX_test_std)

lr = LogisticRegression()
lr = lr.fit(NewX_train_lda, y_train)

lr_pred = lr.predict(NewX_test_lda)
confusion_matrix(y_test, lr_pred)
print (classification_report(y_test, lr_pred))

# This plot will not work unless our class has more than 2 possible values
plot_decision_regions(NewX_train_lda, y_train, classifier=lr)
plt.xlabel('LD1')
plt.ylabl('LD2')
plt.legend(loc='best')
plt.show()

'''
Kernel PCA
'''

kpca = KernelPCA(n_components=2, kernel='rbf', gamma=.01)
NewX_kpca = kpca.fit_transform(NewX_train_std)

# train Logistic Regression model
lr = LogisticRegression()
lr.fit(NewX_kpca, y_train)
# Plot decision regions
plot_decision_regions(NewX_kpca, y_train, classifier=lr)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='best')
plt.show()
















'''At this point, we test several models for the best classification method.
# Models include: logistic regression, Linear SVC, SVC, and KNN.
'''


'''Scoring Options:
Valid options are ['accuracy', 'adjusted_rand_score', 'average_precision', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'neg_log_loss', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_median_absolute_error', 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc']
'''

#Fit logistic regression model using grid search over C and tolerance
log_params = {'C':[1, 10, 100, 1000],
              'tol':[1e-1,1e-2,1e-3,1e-4],
              'penalty':['l1', 'l2'],
              'class_weight':['balanced', {0:.88,1:.12}]
              }
log = linear_model.LogisticRegression()
log_gs = GridSearchCV(log,param_grid=log_params,cv=5,scoring='f1_macro')
log_gs.fit(x_train_scaled,y_train)
log_gs.best_score_
log_gs.best_estimator_
log_gs.scorer_
log_est = log_gs.best_estimator_
log_est.fit(x_train_scaled,y_train)
log_y_pred = log_est.predict(x_test_scaled)
confusion_matrix(y_test, log_y_pred)
print (classification_report(y_test, log_y_pred))

#Fit LinearSVC model
LinearSVC_params = {'C':[1, 10, 100, 1000],
                    'tol':[1e-1,1e-2,1e-3,1e-4],
                    'class_weight':['balanced', {0:.88,1:.12}]
                    }
LinearSVC = svm.LinearSVC(class_weight='balanced')
LinearSVC_gs = GridSearchCV(LinearSVC,param_grid=LinearSVC_params,cv=5,scoring='f1_weighted')
LinearSVC_gs.fit(x_train_scaled,y_train)
LinearSVC_gs.best_score_
LinearSVC_gs.best_estimator_
LinearSVC_gs.scorer_
LinearSVC_est = LinearSVC_gs.best_estimator_
LinearSVC_est.fit(x_train_scaled,y_train)
LinearSVC_y_pred = LinearSVC_est.predict(x_test_scaled)
confusion_matrix(y_test, LinearSVC_y_pred)
print (classification_report(y_test, LinearSVC_y_pred))

#Fit SVC model
SVC_params = [
              {'kernel':['rbf'],
               'gamma':[1e-1,1e-2,1e-3,1e-4],
               'C':[1, 10, 100, 1000],
               'class_weight':['balanced', {0:.88,1:.12}]
               },
               {'kernel':['sigmoid'],
               'gamma':[1e-1,1e-2,1e-3,1e-4],
               'C':[1, 10, 100, 1000],
               'class_weight':['balanced', {0:.88,1:.12}]
               },
              {'kernel':['linear'],
               'C':[1, 10, 100, 1000],
               'class_weight':['balanced', {0:.88,1:.12}]
               },
               {'kernel':['poly'],
                'degree':[2,3,4],
               'gamma':[1e-1,1e-2,1e-3,1e-4],
               'C':[1, 10, 100, 1000],
               'class_weight':['balanced', {0:.88,1:.12}]
               }
              ]
SVC_clf = svm.SVC()
SVC_gs = GridSearchCV(SVC_clf,param_grid=SVC_params,cv=5,scoring='f1_weighted')
SVC_gs.fit(x_train_scaled,y_train)
SVC_gs.best_score_
SVC_gs.best_estimator_
SVC_gs.scorer_
SVC_est = SVC_gs.best_estimator_
SVC_est.fit(x_train_scaled,y_train)
SVC_y_pred = SVC_est.predict(x_test_scaled)
confusion_matrix(y_test, SVC_y_pred)
print (classification_report(y_test, SVC_y_pred))

#K Nearest Neighbors
#Use grid search to determine best parameters
k = list(range(3,33,3)) #create a range for k
knn_params = {'n_neighbors':k,
          'weights':['uniform', 'distance'],
          'algorithm':['ball_tree', 'kd_tree', 'brute'],
          'p':[1,2,3]
          }
knn_clf = neighbors.KNeighborsClassifier()
knn_gs = GridSearchCV(knn_clf,param_grid=knn_params,cv=5,scoring='average_precision')
knn_gs.fit(x_train_scaled,y_train)
knn_gs.best_score_
knn_gs.best_estimator_
knn_gs.scorer_
knn_est = knn_gs.best_estimator_
knn_est.fit(x_train_scaled,y_train)
knn_y_pred = knn_est.predict(x_test_scaled)
confusion_matrix(y_test, knn_y_pred)
print (classification_report(y_test, knn_y_pred))

#Random Forest
RF_params = {'n_estimators':[5,10,15,20,25],
             'criterion':['gini','entropy'],
             'max_features':['sqrt','log2'],
             'class_weight':['balanced', {0:.88,1:.12}]
             }
RF_clf = RandomForestClassifier(n_jobs=4)
RF_gs = GridSearchCV(RF_clf,param_grid=RF_params,cv=5,scoring='roc_auc')
RF_gs.fit(x_train_scaled,y_train)
RF_gs.best_score_
RF_gs.best_estimator_
RF_gs.scorer_
RF_est = RF_gs.best_estimator_
RF_est.fit(x_train_scaled,y_train)
RF_y_pred = RF_est.predict(x_test_scaled)
confusion_matrix(y_test, RF_y_pred)
print (classification_report(y_test, RF_y_pred))

#AdaBoost
Ada_params = {'n_estimators':[50,100,150,200,250,300,350,400],
              'learning_rate':[1.0,2.0,3.0]
              }
Ada_clf = AdaBoostClassifier()
Ada_gs = GridSearchCV(Ada_clf,param_grid=Ada_params,cv=5,scoring='roc_auc')
Ada_gs.fit(x_train_scaled,y_train)
Ada_gs.best_score_
Ada_gs.best_estimator_
Ada_gs.scorer_
Ada_est = Ada_gs.best_estimator_
Ada_est.fit(x_train_scaled,y_train)
Ada_y_pred = Ada_est.predict(x_test_scaled)
confusion_matrix(y_test, Ada_y_pred)
print (classification_report(y_test, Ada_y_pred))




# Get data to plot ROC Curve
fp, tp, th = roc_curve(y_test, knn_y_pred)
roc_auc = auc(fp, tp)

# Plot ROC Curve
plt.title('ROC Curve')
plt.plot(fp, tp, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()




