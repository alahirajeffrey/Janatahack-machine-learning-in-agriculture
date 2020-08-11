# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 17:54:43 2020

@author: Jeffrey Alahira
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import seaborn as sns


def read_csv(file_name):
    """
    Function to read in csv file
    filename: name of csv file to read
    """
    
    train = pd.read_csv(file_name)
    
    return train

## read in csv file
train = read_csv('train_yaOffsB.csv')

## check shape of dataframe
print(train.shape)

## check for missing values
print(train.isnull().sum())


def drop_columns(dataframe, column):
    """
    function to drop columnns in dataframe
    dataframe: name of dataframe 
    column: column to drop
    """
    
    dataframe.drop(column, inplace = True, axis = 1)
    return dataframe

## drop ID column
train = drop_columns(train, "ID")

## dataframe to hold initial data before cleaning
logged_df = train

print (train.shape)


def plot_distribution(dataframe, column, plot_title):
    """
    function to plot the distribution of columns in dataframe\n
    dataframe: name of dataframe \n
    column: column to drop\n
    plot_title: title of the plot\n
    """
    
    plot.hist(dataframe[column])
    plot.title(plot_title)
    plot.grid (True)
    
    plot.savefig(plot_title + '.png')


## check distribution of Estimated_Insects_Count column
plot_distribution(
    train, 'Estimated_Insects_Count', 'Plot of Estimated_Insects_Count column before transformation')

## transform Estimated_Insects_Count column and plot
logged_df['Estimated_Insects_Count'] = np.log(logged_df['Estimated_Insects_Count'])

plot_distribution(
    logged_df, 'Estimated_Insects_Count', 'Plot of Estimated_Insects_Count column after log transformation')

print(train.columns)


## check distribution of Number_Doses_Week column
plot_distribution(
    train, 'Number_Doses_Week', 'Plot of Number_Doses_Week column before transformation')

## transform Number_Doses_Week column and plot
logged_df['Number_Doses_Week'] = np.log(logged_df['Estimated_Insects_Count'])

plot_distribution(
    logged_df, 'Number_Doses_Week', 'Plot of Number_Doses_Week column after log transformation')


## check distribution of Estimated_Insects_Count column
plot_distribution(
    train, 'Number_Weeks_Used', 'Plot of Number_Weeks_Used column before transformation')

## transform Estimated_Insects_Count column using minmaxscaler due to
## the presence 0 values

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,5))

logged_df['Number_Weeks_Used'] = scaler.fit_transform(
    logged_df['Number_Weeks_Used'].values.reshape(-1,1))

plot_distribution(
    logged_df, 'Number_Weeks_Used', 'Plot of Number_Weeks_Used column after scaling and log transformation')


## check distribution of Estimated_Insects_Count column
plot_distribution(
    train, 'Number_Weeks_Quit', 'Plot of Number_Weeks_Quit column before transformation')

## transform Estimated_Insects_Count column using minmaxscaler due to
## the presence 0 values

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (1,5))

logged_df['Number_Weeks_Quit'] = scaler.fit_transform(
    logged_df['Number_Weeks_Quit'].values.reshape(-1,1))

## perform log transformation
logged_df['Number_Weeks_Quit'] = np.log(logged_df['Number_Weeks_Quit'])

plot_distribution(
    train, 'Number_Weeks_Quit', 'Plot of Number_Weeks_Quit column after scaling and log transformation')


## save logged_df to csv file
logged_df.to_csv('data_after_log_transformation.csv')


from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=3)

def fill_missing_values(dataframe, column):
    """
    function to fill missing values using KNNImputer\n
    datafame: dataframe containing missing values
    column: column to fill
    """
    dataframe[column]= imputer.fit_transform(
        dataframe[column].values.reshape(-1,1))
    
    return dataframe

## fill missing values
filled_df = fill_missing_values(logged_df,'Number_Weeks_Used' )

## typecast target variable to categorical
filled_df['Crop_Damage'] = filled_df['Crop_Damage'].astype('category')

## save filled_df to csv filing
filled_df.to_csv('data_after_filling_missing_values.csv')




