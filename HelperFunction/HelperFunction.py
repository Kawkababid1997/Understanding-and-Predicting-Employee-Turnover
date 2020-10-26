
## Import Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score
from sklearn.metrics import recall_score, confusion_matrix, precision_recall_curve
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

# Load the Data
def load_data(file_directory, format = 'txt'):
    """
    loads data from excel, csv, tsv, or txt file
    
    Parameters
    ----------
    file_directory: specify the filepath.
    format: specify "excel", "csv", "tsv", or "txt"
    """
    if format == 'csv':
        return pd.read_csv(file_directory)
    elif format =='excel':
        return pd.read_excel(file_directory)
    elif format == 'tsv':
        return pd.read_csv(file_directory, sep='\t')
    elif format == 'txt':
        return pd.read_table(file_directory)
    else:
        raise ValueError('Invalid file format. Please specify "excel", "csv", "tsv", or "txt"')
        

# Rename the HR DataFrame column for better readablity
def renaming_column(df):
    """
    Rename columns for better readablity 
    Parameters
    ----------
    df: Specify the hr_dataset 
    """
    return df.rename(columns={'satisfaction_level':'satisfaction',
                   'last_evaluation': 'evaluation',
                   'number_project': 'projectCount',
                   'average_montly_hours': 'averageMonthlyHours',
                   'time_spend_company': 'yearsAtCompany',
                   'Work_accident': 'workAccident',
                   'promotion_last_5years': 'promotion',
                   'sales': 'department', 
                   'left': 'turnover'})
        
## Heatmap        
def correlation_map(data):
    """
    Returns a corrolation heatmap
    Parameters
    ----------
    data : rectangular dataset
    2D dataset that can be coerced into an ndarray. If a Pandas DataFrame
    is provided, the index/column information will be used to label the
    columns and rows.
    """
    
    fig = plt.figure(figsize=(12,10))
    sns.heatmap(data, annot=True, 
                xticklabels = data.columns.values,
                yticklabels = data.columns.values,
                cmap = 'Blues', fmt = '.2g')

# Plot Feature
def plot_feature(kind, data = None or [], title = None or [],
                 x=None or [], y = None or [], hue = None, xlabel = None, ylabel = None, legend = None or [], 
                 label = None or []):
    if kind == "distplot":
        sns.set(style="white")
        if len(data) and len(title) == 3: 
            fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize=(15,6))
            sns.distplot(data[0], kde=False, color = 'g', ax=axes[0])
            axes[0].set_title(title[0])
            axes[0].set_ylabel(ylabel)
            sns.distplot(data[1], kde=False, color = 'r', ax=axes[1])
            axes[1].set_title(title[1])
            axes[1].set_ylabel(ylabel)
            sns.distplot(data[2], kde=False, color = 'b', ax=axes[2])
            axes[2].set_title(title[2])
            axes[2].set_ylabel(ylabel)
            
    elif kind == 'lmplot':
        sns.lmplot(x= x, y=y, data=data, fit_reg=False, hue=hue)
    
    elif kind == 'kde':
        sns.set(style="white")
        if len(legend) == 2 and len(data) == 2:
            fig = plt.figure(figsize=(15,4))
            ax = sns.kdeplot(data[0], color='b', shade = True, label = legend[0])
            ax = sns.kdeplot(data[1], color = 'r', shade = True, label = legend[1])
            plt.title(title)
            
    elif kind == 'bar':
        plt.figure(figsize =(10,6))
        ax = sns.barplot(x=x, y = y, hue = hue, data = data, estimator = lambda x: len(x) / len(data) * 100)
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f'),
                           (p.get_x() + p.get_width()/2., p.get_height()/6),
                           ha = 'center', va = 'center',
                           xytext = (0,9), 
                           textcoords = 'offset points')
        ax.set(ylabel = ylabel)
    
    elif kind == 'simple bar':
        sns.set(style="white")
        plt.figure(figsize =(8,6))
        names = sns.barplot(x=x, y = y, alpha = 0.6)
        for p in names.patches:
            names.annotate(format(p.get_height(), '.0f'),
                           (p.get_x() + p.get_width()/2., p.get_height()/2),
                           ha = 'center', va = 'center',
                           xytext = (0,9), 
                           textcoords = 'offset points')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
    elif kind == 'barh':
        sns.set(style="whitegrid")
        f, ax = plt.subplots(figsize=(13,7))
        sns.set_color_codes("pastel")
        sns.barplot(x = x[0], y = ylabel.lower(), data = data, label = x[0], color = 'b')
        sns.set_color_codes("muted")
        name2 = sns.barplot(x = x[1], y = ylabel.lower(), data = data, label = x[1], color = 'r')
        ax.set(ylabel=ylabel, xlabel=xlabel, title = title)
        ax.legend(ncol=2, loc = 'lower right', frameon = True)
        sns.despine(left=True, bottom=True)
    
    
    elif kind == 'heatmap':
        fig, axes = plt.subplots(nrows = 1, ncols =4, figsize = (14,4))
        fig.tight_layout(w_pad=0)

        labels = ['TN', 'FP','FN', 'TP']
        labels = np.asarray(labels).reshape(2,2)

        sns.heatmap(confusion_matrix(x, x), annot=labels, fmt = '', 
                    cbar = False, ax = axes[0])
        axes[0].set_ylabel('Predicted Label')
        axes[0].set_xlabel("True Label")
        axes[0].set_title('Confusion Matrix')
        sns.heatmap(confusion_matrix(x, y[0]), annot = True, cbar = False, fmt = 'g', 
                    cmap = 'Blues', ax = axes[1])
        axes[1].set_xlabel("True Label")
        axes[1].set_title(title[0])

        sns.heatmap(confusion_matrix(x, y[1]), annot = True, cbar = False, fmt = 'g', 
                    cmap = 'Blues', ax = axes[2])
        axes[2].set_xlabel("True Label")
        axes[2].set_title(title[1])

        sns.heatmap(confusion_matrix(x, y[2]), annot = True, cbar = False, fmt = 'g', 
                    cmap = 'Blues', ax = axes[3])
        axes[3].set_xlabel("True Label")
        axes[3].set_title(title[2])
    
    elif kind == 'feature_importance':
        sns.set(style = 'whitegrid')
        f, ax = plt.subplots(figsize = (13, 7))
        sns.barplot(x = x, y = y, data = data, label = 'Total', color = 'b')
       
        
## Preprocessing Data for Machine Learning
def label_encoding(df, cat_var=None, num_var=None):
    """
    Performs one-hot encoding on all categorical variables and combines result with continuous variable
    Parameters
    ------------
    df: Select the dataframe
    cat_var: categorical variables
    num_var: numerical variables
    """
    categorical_df = pd.get_dummies(df[cat_var], drop_first=True)
    numerical_df = df[num_var]
    return pd.concat([categorical_df, numerical_df], axis = 1)


# Train the model and print summary
def train_and_summarize(model, x_train, x_test, y_train, y_test):
    """
    Train the sklearn-estimator and print a short summary of the model performance
    Parameters
    ------------
    model: name of the sklearn estimator
    x_train
    x_test
    y_train
    y_test
    """
    model = model.fit(x_train, y_train)
    print(f"\n---{type(model).__name__} Model---")
    auc = roc_auc_score(y_test, model.predict(x_test))
    print(f"{type(model).__name__} AUC = %2.2f" % auc)
    print(classification_report(y_test, model.predict(x_test)))


