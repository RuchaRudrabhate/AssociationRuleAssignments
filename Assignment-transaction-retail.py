# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 22:57:06 2023

@author: arudr
"""
#assignment 5

'''
Problem Statement: - 
A retail store in India, has its transaction data, and it would like to know the buying pattern of the 
consumers in its locality, you have been assigned this task to provide the manager with rules 
on how the placement of products needs to be there in shelves so that it can improve the buying
patterns of consumes and increase customer footfall. 

'''

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:/5-Recommendation/transactions_retail1.csv')
df
df.dtypes
'''
'HANGING'    object
'HEART'      object
'HOLDER'     object
'T-LIGHT'    object
'WHITE'      object
NA           object
dtype: object'''
df.columns
'''
Index([''HANGING'', ''HEART'', ''HOLDER'', ''T-LIGHT'', ''WHITE'', 'NA'], dtype='object')'''
df.describe()
'''
'HANGING'  'HEART' 'HOLDER' 'T-LIGHT' 'WHITE'     NA
count     557040   538818   525799    436182  222350  83150
unique       949     1177     1099       856     489    184
top        'BAG'  'HEART'    'RED'      'OF'   'SET'  'SET'
freq       41795    16186    18724     20683   14738  16194
'''

#here we can see that data is already normalised

#find pdf and cdf for each column

#1 'HANGING'
counts, bin_edges = np.histogram(df['HANGING'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.show();

#2 'HEART'
counts, bin_edges = np.histogram(df['HEART'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.show();

#3 'HOLDER'
counts, bin_edges = np.histogram(df['HOLDER'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.show();

##4 'T-LIGHT'
counts, bin_edges = np.histogram(df['T-LIGHT'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.show();

#store this processed data as a csv file
df.to_csv('C:/5-Recommendation/transaction_retail_processed.csv')

#now we can apply association rules 

from mlxtend.frequent_patterns import apriori, association_rules
phone =[]

with open('C:/5-Recommendation/transaction_retail_processed.csv') as f:df=f.read()

#splitting the data into separate transactions usinusing separator it is comma separated
#we can use new line charater
df = df.split('\n')

#now let us separate out each item from the phone list

df_list =[]
for i in df:
    df_list.append(i.split(','))
    
#split function will separate each item from each list, whenever it will find 
all_df_list = [i for item in df_list for i in item ]

#you will get all the items occured in all transactions

from collections import Counter 
item_frequencies = Counter(all_df_list)

item_frequencies = sorted(item_frequencies.items(),key = lambda x:x[1])
#when execute this, item frequencies will be in sorted form, in the form of 
#iem name with count
#let us separate out items and their count

items = list(reversed([i[0] for i in item_frequencies]))

#when we execute this, ietm frequencies will be in sorted form 
# in the form of tuple
#item name with count 
#let us separate out items and their count
items = list(reversed([i[1] for i in item_frequencies]))
frequencies = list(reversed([i[1] for i in item_frequencies]))

import matplotlib.pyplot as plt

plt.bar(height = frequencies[0:11],x=list(range(0,11)))
plt.xticks(list(range(0,11)),items[0:11])

#plt.xtricks, you can specify the rotation for the trick
#label in degrees or with keywords

plt.xlabel("items")
plt.ylabel("count")
plt.show()

import pandas as pd

#now let us try to establish association rule mining
#we  have phone list in the format, we need to convert it in dataframe
df_series = pd.DataFrame(pd.Series(df_list))

df_series = df_series.iloc[:11,:]

df_series.columns = ['HEART']

#now we will have to apply 1 hot encoding, before that in
#one column there are various items separated by ',
#let us separate it with '*
x = df_series['HEART'].str.join(sep = '*')

frequent_itemsets = apriori(x, min_support = 0.0075,max_len = 4,use_colnames = True)

frequent_itemsets.sort_values('support',ascending = False, inplace = True)

rules = association_rules(frequent_itemsets,metric='lift',min_threshold=1)
rules.head(20)
rules.sort_values('lift',ascending=False).head(10)