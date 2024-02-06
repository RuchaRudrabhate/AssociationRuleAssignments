# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 22:57:06 2023

@author: arudr
"""

#assignment 4

'''
Problem Statement: - 
A Mobile Phone manufacturing company wants to launch its three
 brand new phone into the market, but before going with its traditional
 marketing approach this time it want to analyze the data of its 
 previous model sales in different regions and you have been hired 
 as an Data Scientist to help them out, use the Association rules 
 concept and provide your insights to the company’s marketing team
 to improve its sales.
4.) myphonedata.csv

'''

#business objective
#maximize --> sales of new phones across different regoins
#minimize-->

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

phones = pd.read_csv('C:/5-Recommendation/myphonedata.csv')
phones
phones.columns
'''
phones.columns
Out[7]: Index(['red', 'white', 'green', 'yellow', 'orange', 'blue'], dtype='object')
'''

phones.dtypes
'''
Out[8]: 
red       int64 
white     int64
green     int64
yellow    int64
orange    int64
blue      int64
dtype: object 
'''
phones.shape
#(11, 6)
phones.describe()
'''
  red      white      green     yellow     orange       blue
count  11.000000  11.000000  11.000000  11.000000  11.000000  11.000000
mean    0.545455   0.636364   0.181818   0.090909   0.181818   0.545455
std     0.522233   0.504525   0.404520   0.301511   0.404520   0.522233
min     0.000000   0.000000   0.000000   0.000000   0.000000   0.000000
25%     0.000000   0.000000   0.000000   0.000000   0.000000   0.000000
50%     1.000000   1.000000   0.000000   0.000000   0.000000   1.000000
75%     1.000000   1.000000   0.000000   0.000000   0.000000   1.000000
max     1.000000   1.000000   1.000000   1.000000   1.000000   1.000000
'''

#here we can see that data is already normalised

#find pdf and cdf for each column
#1 red
counts, bin_edges = np.histogram(phones['red'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.show();

##Inference
'''
from pdf we can say that about 0.2 to 0.9 i.e 20 to 90% red phones have 0.1% sales 
and sales decreases from 0.4 to 0.0 for 20% phones and increases
from 0.0 to 0.6 for phones > 80% 
'''

#2 white
counts, bin_edges = np.histogram(phones['white'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.show();

#inference
'''
from pdf we can say that about 0.2 to 0.9 i.e 20 to 90% white phones have 0.1% sales 
and sales decreases from 0.3 to 0.0 for 20% sales and increases
from 0.0 to 0.75 for phones > 80%

from CDF -> we say that cdf is constant for 90% of sales and it is
0.3 sales and it increases later on up to 1.0 sales
 
'''

#3 green
counts, bin_edges = np.histogram(phones['green'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.show();

#4 yellow
counts, bin_edges = np.histogram(phones['yellow'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.show();

#5 orange
counts, bin_edges = np.histogram(phones['orange'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.show();


#6 blue
counts, bin_edges = np.histogram(phones['blue'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.show();

###############################
#outliers treatment
sns.boxplot(phones['red'])
sns.boxplot(phones['white'])
sns.boxplot(phones['green'])
sns.boxplot(phones['yellow'])
sns.boxplot(phones['orange'])
sns.boxplot(phones['blue'])

#from boxplot we can say that only green yellow and orange have 
#outliers and we need to treat them
#1  green
iqr = phones['green'].quantile(0.75)-phones['green'].quantile(0.25)
iqr
q1=phones['green'].quantile(0.25)
q3=phones['green'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
phones['green'] =  np.where(phones['green']>u_limit,u_limit,np.where(phones['green']<l_limit,l_limit,phones['green']))
sns.boxplot(phones['green'])
            
#2  yellow
iqr = phones['yellow'].quantile(0.75)-phones['yellow'].quantile(0.25)
iqr
q1=phones['yellow'].quantile(0.25)
q3=phones['yellow'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
phones['yellow'] =  np.where(phones['yellow']>u_limit,u_limit,np.where(phones['yellow']<l_limit,l_limit,phones['yellow']))
sns.boxplot(phones['yellow'])

#2  orange
iqr = phones['orange'].quantile(0.75)-phones['orange'].quantile(0.25)
iqr
q1=phones['orange'].quantile(0.25)
q3=phones['orange'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
phones['orange'] =  np.where(phones['orange']>u_limit,u_limit,np.where(phones['orange']<l_limit,l_limit,phones['orange']))
sns.boxplot(phones['orange'])


#now data set do not have outliers
#lets describe data again

phones.describe()      
'''
 red      white  green  yellow  orange       blue
count  11.000000  11.000000   11.0    11.0    11.0  11.000000
mean    0.545455   0.636364    0.0     0.0     0.0   0.545455
std     0.522233   0.504525    0.0     0.0     0.0   0.522233
min     0.000000   0.000000    0.0     0.0     0.0   0.000000
25%     0.000000   0.000000    0.0     0.0     0.0   0.000000
50%     1.000000   1.000000    0.0     0.0     0.0   1.000000
75%     1.000000   1.000000    0.0     0.0     0.0   1.000000
max     1.000000   1.000000    0.0     0.0     0.0   1.000000
'''

#data is normalised

#store this processed data as a csv file
phones.to_csv('C:/5-Recommendation/myphonedata_processed.csv')

#now we can apply association rules 

from mlxtend.frequent_patterns import apriori, association_rules
phone =[]

with open('C:/5-Recommendation/myphonedata_processed.csv') as f:phone=f.read()

#splitting the data into separate transactions usinusing separator it is comma separated
#we can use new line charater
phone = phone.split('\n')

#now let us separate out each item from the phone list

phone_list =[]
for i in phone:
    phone_list.append(i.split(','))
    
#split function will separate each item from each list, whenever it will find 
all_phone_list = [i for item in phone_list for i in item ]

#you will get all the items occured in all transactions

from collections import Counter 
item_frequencies = Counter(all_phone_list)

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
phones_series = pd.DataFrame(pd.Series(phone_list))

phones_series = phones_series.iloc[:11,:]

phones_series.columns = ['red']

#now we will have to apply 1 hot encoding, before that in
#one column there are various items separated by ',
#let us separate it with '*
x = phones_series['red'].str.join(sep = '*')

frequent_itemsets = apriori(x, min_support = 0.0075,max_len = 4,use_colnames = True)

frequent_itemsets.sort_values('support',ascending = False, inplace = True)

rules = association_rules(frequent_itemsets,metric='lift',min_threshold=1)
rules.head(20)
rules.sort_values('lift',ascending=False).head(10)

