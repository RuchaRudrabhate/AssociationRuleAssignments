# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 12:37:52 2023

@author: arudr
"""
#assignment 2
'''
Problem Statement: - 
The Departmental Store, has gathered the data of the products it sells on a Daily basis.
Using Association Rules concepts, provide the insights on the rules and the plots.
2.) Groceries.csv


'''
#business objective -
#max - sales of the groceries
#min -

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:/5-Recommendation/groceries.csv')
df

df.columns
df.dtypes
'''
citrus fruit           object       categorical     
semi-finished bread    object
margarine              object
ready soups            object
Unnamed: 4             object
Unnamed: 5             object
Unnamed: 6             object
Unnamed: 7             object
Unnamed: 8             object
Unnamed: 9             object
Unnamed: 10            object
Unnamed: 11            object
Unnamed: 12            object
Unnamed: 13            object
Unnamed: 14            object
Unnamed: 15            object
Unnamed: 16            object
Unnamed: 17            object
Unnamed: 18            object
Unnamed: 19            object
Unnamed: 20            object
Unnamed: 21            object
Unnamed: 22            object
Unnamed: 23            object
Unnamed: 24            object
Unnamed: 25            object
Unnamed: 26            object
Unnamed: 27            object
Unnamed: 28            object
Unnamed: 29            object
Unnamed: 30            object
Unnamed: 31            object
dtype: object

'''

#treating nan values by using modes of columns
df['citrus fruit'].mode()
df['citrus fruit'] = np.where(df['citrus fruit']==np.nan,df['citrus fruit'].mode(),np.where(df['citrus fruit']!=np.nan,df['citrus fruit'],df['citrus fruit']))

df2 = df
df['semi-finished bread'].mode()
df2['semi-finished bread'] = np.where(df['semi-finished bread']==np.nan,df['semi-finished bread'].mode(),np.where(df['semi-finished bread']!=np.nan,df['semi-finished bread'],df['semi-finished bread']))

df.fillna(df.mode().iloc[0])
df.fillna(df.mode().iloc[:,:],inplace=True,axis = 1)

