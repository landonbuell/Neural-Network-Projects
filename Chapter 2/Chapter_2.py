"""
Landon Buell
Neural Network Projects with Python
Chapter 2
26 April 2020
"""

            #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing as preprocessing

            #### DEFINTIIONS ####

def cols_with_0 (dataframe):
    """ Print number of 0's in each col of frame """
    print("Number of rows with 0 values:")
    for col in df.columns:
        missing_rows = df.loc[df[col]==0].shape[0]
        print('\t'+col+':'+str(missing_rows))

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':
    """  Diabetes Predictions """

    df = pd.read_csv('diabetes.csv')
    print(df.head())

    # visualize data sets
    df.hist()
    #plt.show()

    print(df.describe())

    cols_with_0(df)

    columns = ['Glucose','BloodPressure','SkinThickness',
                'Insulin','BMI']
    
    # Replace '0-s' w/ np.nan
    for col in columns:
        df[col] = df[col].replace(0,np.nan)

    # Replace np.nan w/ mean of col
    for col in columns:
        df[col] = df[col].fillna(df[col].mean())

    # scale to mean=0,var=1
    df_scaled = preprocesing.scale(df)
    df_scaled = pd.DataFrame(data=df_scaled,columns=df.columns)
    df_scaled['Outcomes'] = df['Outcome']
    df = df_scaled

        
