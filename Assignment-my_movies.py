# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 15:14:12 2023

@author: arudr
"""

#assignment 3
'''
Problem Statement: - 
A film distribution company wants to target audience based on 
their likes and dislikes, you as a Chief Data Scientist Analyze the 
data and come up with different rules of movie list so that the 
business objective is achieved.
3.) my_movies.csv
'''

#Business Objective
#Maximize - to maximize the likes for films, performance and quality of films
#Minimize - num of dislikes

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

movies = pd.read_csv('C:/5-Recommendation/my_movies.csv')
movies

movies.columns
'''
Index(['Sixth Sense', 'Gladiator', 'LOTR1', 'Harry Potter1', 'Patriot',
       'LOTR2', 'Harry Potter2', 'LOTR', 'Braveheart', 'Green Mile'],
      dtype='object')
'''
movies.shape
# (10, 10)
movies.dtypes
'''
Sixth Sense      int64      nominal, quantitative       relevent as it is film name
Gladiator        int64      nominal, quantitative       relevent as it is film name
LOTR1            int64      nominal, quantitative       relevent as it is film name
Harry Potter1    int64      nominal, quantitative       relevent as it is film name
Patriot          int64      nominal, quantitative       relevent as it is film name
LOTR2            int64      nominal, quantitative       relevent as it is film name
Harry Potter2    int64      nominal, quantitative       relevent as it is film name
LOTR             int64      nominal, quantitative       relevent as it is film name
Braveheart       int64      nominal, quantitative       relevent as it is film name
Green Mile       int64      nominal, quantitative       relevent as it is film name
dtype: object
'''

movies.describe()

'''
Sixth Sense  Gladiator      LOTR1  ...       LOTR  Braveheart  Green Mile
count    10.000000  10.000000  10.000000  ...  10.000000   10.000000   10.000000
mean      0.600000   0.700000   0.200000  ...   0.100000    0.100000    0.200000
std       0.516398   0.483046   0.421637  ...   0.316228    0.316228    0.421637
min       0.000000   0.000000   0.000000  ...   0.000000    0.000000    0.000000
25%       0.000000   0.250000   0.000000  ...   0.000000    0.000000    0.000000
50%       1.000000   1.000000   0.000000  ...   0.000000    0.000000    0.000000
75%       1.000000   1.000000   0.000000  ...   0.000000    0.000000    0.000000
max       1.000000   1.000000   1.000000  ...   1.000000    1.000000    1.000000

[8 rows x 10 columns]
'''

#all the values of mean , max, min are in the range of 0 to 1

#Data prreprocessing --> feature engineering and data cleaning
#pairplot
plt.close();
sns.set_style("whitegrid");
sns.pairplot(movies, height=3);
plt.show()

#pdf and cdf
#1 Sixth Sense
counts, bin_edges = np.histogram(movies['Sixth Sense'], bins=10, 
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
from pdf we can say that about 0.2 to 0.9 i.e 20 to 90% Sixth Sense films have 0.1% likes 
and like decreases from 0.4 to 0.0 for 20% films and increases
from 0.0 to 0.6 for films > 80% 
'''

#2 Gladiator
counts, bin_edges = np.histogram(movies['Gladiator'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.show();

#inference ->
'''
from pdf we can say that about 0.2 to 0.9 i.e 20 to 90% Sixth Sense films have 0.1% likes 
and like decreases from 0.3 to 0.0 for 20% films and increases
from 0.0 to 0.75 for films > 80%

from CDF -> we say that cdf is constant for 90% of films and it is
0.3 likes and it increases later on up to 1.0 likes
 
'''

#3 LOTR1
counts, bin_edges = np.histogram(movies['LOTR1'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.show();


#4 Harry Potter1
counts, bin_edges = np.histogram(movies['Harry Potter1'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.show();

#5 Patriot
counts, bin_edges = np.histogram(movies['Patriot'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.show();

#6 LOTR2
counts, bin_edges = np.histogram(movies['LOTR2'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.show();

#7 Harry Potter2
counts, bin_edges = np.histogram(movies['Harry Potter2'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.show();

#8 LOTR
counts, bin_edges = np.histogram(movies['LOTR'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.show();

#9 Braveheart
counts, bin_edges = np.histogram(movies['Braveheart'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.show();

#10 Green Mile
counts, bin_edges = np.histogram(movies['Green Mile'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.show();


## outliers treatment

movies.columns

sns.boxplot(movies['Sixth Sense'])
sns.boxplot(movies['Gladiator'])
sns.boxplot(movies['LOTR1'])
sns.boxplot(movies['Harry Potter1'])
sns.boxplot(movies['Patriot'])
sns.boxplot(movies['LOTR2'])
sns.boxplot(movies['Harry Potter2'])
sns.boxplot(movies['LOTR'])
sns.boxplot(movies['Braveheart'])
sns.boxplot(movies['Green Mile'])

#only sixth sense, gladiator, patriot do not have outliers 

#1  LOTR1
iqr = movies['LOTR1'].quantile(0.75)-movies['LOTR1'].quantile(0.25)
iqr
q1=movies['LOTR1'].quantile(0.25)
q3=movies['LOTR1'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
movies['LOTR1'] =  np.where(movies['LOTR1']>u_limit,u_limit,np.where(movies['LOTR1']<l_limit,l_limit,movies['LOTR1']))
sns.boxplot(movies['LOTR1'])


#2  Harry Potter1
iqr = movies['Harry Potter1'].quantile(0.75)-movies['Harry Potter1'].quantile(0.25)
iqr
q1=movies['Harry Potter1'].quantile(0.25)
q3=movies['Harry Potter1'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
movies['Harry Potter1'] =  np.where(movies['Harry Potter1']>u_limit,u_limit,np.where(movies['Harry Potter1']<l_limit,l_limit,movies['Harry Potter1']))
sns.boxplot(movies['Harry Potter1'])

#3  Harry Potter2
iqr = movies['Harry Potter2'].quantile(0.75)-movies['Harry Potter2'].quantile(0.25)
iqr
q1=movies['Harry Potter2'].quantile(0.25)
q3=movies['Harry Potter2'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
movies['Harry Potter2'] =  np.where(movies['Harry Potter2']>u_limit,u_limit,np.where(movies['Harry Potter2']<l_limit,l_limit,movies['Harry Potter2']))
sns.boxplot(movies['Harry Potter2'])

#4  LOTR2
iqr = movies['LOTR2'].quantile(0.75)-movies['LOTR2'].quantile(0.25)
iqr
q1=movies['LOTR2'].quantile(0.25)
q3=movies['LOTR2'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
movies['LOTR2'] =  np.where(movies['LOTR2']>u_limit,u_limit,np.where(movies['LOTR2']<l_limit,l_limit,movies['LOTR2']))
sns.boxplot(movies['LOTR2'])


#5  LOTR
iqr = movies['LOTR'].quantile(0.75)-movies['LOTR'].quantile(0.25)
iqr
q1=movies['LOTR'].quantile(0.25)
q3=movies['LOTR'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
movies['LOTR'] =  np.where(movies['LOTR']>u_limit,u_limit,np.where(movies['LOTR']<l_limit,l_limit,movies['LOTR']))
sns.boxplot(movies['LOTR'])

#6 Braveheart
iqr = movies['Braveheart'].quantile(0.75)-movies['Braveheart'].quantile(0.25)
iqr
q1=movies['Braveheart'].quantile(0.25)
q3=movies['Braveheart'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
movies['Braveheart'] =  np.where(movies['Braveheart']>u_limit,u_limit,np.where(movies['Braveheart']<l_limit,l_limit,movies['Braveheart']))
sns.boxplot(movies['Braveheart'])

#7  Green Mile
iqr = movies['Green Mile'].quantile(0.75)-movies['Green Mile'].quantile(0.25)
iqr
q1=movies['Green Mile'].quantile(0.25)
q3=movies['Green Mile'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
movies['Green Mile'] =  np.where(movies['Green Mile']>u_limit,u_limit,np.where(movies['Green Mile']<l_limit,l_limit,movies['Green Mile']))
sns.boxplot(movies['Green Mile'])


#Now data set do not have outliers

#now lets describe our dataset again as
movies.describe()
'''
Sixth Sense  Gladiator  LOTR1  ...  LOTR  Braveheart  Green Mile
count    10.000000  10.000000   10.0  ...  10.0        10.0        10.0
mean      0.600000   0.700000    0.0  ...   0.0         0.0         0.0
std       0.516398   0.483046    0.0  ...   0.0         0.0         0.0
min       0.000000   0.000000    0.0  ...   0.0         0.0         0.0
25%       0.000000   0.250000    0.0  ...   0.0         0.0         0.0
50%       1.000000   1.000000    0.0  ...   0.0         0.0         0.0
75%       1.000000   1.000000    0.0  ...   0.0         0.0         0.0
max       1.000000   1.000000    0.0  ...   0.0         0.0         0.0

[8 rows x 10 columns]
'''

#data seems to be standardized

#store this processed data as a csv file
movies.to_csv('C:/5-Recommendation/my_movies_processed.csv')

#now we can apply association rules 

from mlxtend.frequent_patterns import apriori, association_rules
movie =[]

with open('C:/5-Recommendation/my_movies_processed.csv') as f:movie=f.read()

#splitting the data into separate transactions usinusing separator it is comma separated
#we can use new line charater
movie = movie.split('\n')

#now let us separate out each item from the movies list

movie_list =[]
for i in movie:
    movie_list.append(i.split(','))
    
#split function will separate each item from each list, whenever it will find 
all_movie_list = [i for item in movie_list for i in item ]

#you will get all the items occured in all transactions

from collections import Counter 
item_frequencies = Counter(all_movie_list)

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
#we  have movies list in the format, we need to convert it in dataframe
movies_series = pd.DataFrame(pd.Series(movie_list))

movies_series = movies_series.iloc[:10,:]

movies_series.columns = ['Sixth Sense']

#now we will have to apply 1 hot encoding, before that in
#one column there are various items separated by ',
#let us separate it with '*
x = movies_series['Sixth Sense'].str.join(sep = '*')

frequent_itemsets = apriori(x, min_support = 0.0075,max_len = 4,use_colnames = True)

frequent_itemsets.sort_values('support',ascending = False, inplace = True)

rules = association_rules(frequent_itemsets,metric='lift',min_threshold=1)
rules.head(20)
rules.sort_values('lift',ascending=False).head(10)

