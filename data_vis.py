#Data visualisation in python

#*********#
#Histograms#
#*********#
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm

#Loading data
df = pd.read_csv('/population_data.csv',names = ["age",
"workclass",
"fnlwgt",
"education",
"education-num",
"marital-status",
"occupation",
"relationship",
"race",
"sex",
"capital-gain",
"capital-loss",
"hoursperweek",
"nativecountry","salary"])

# Fit a normal distribution to age data
mu, std = norm.fit(df['age'])

# Plotting histogram
plt.hist(df['age'], bins=range(min(df['age']), max(df['age']) + 5, 5),  alpha=0.5, color='c')

plt.title('Histogram showing age distribution\n mu = %.2f,  Standard Deviation = %.2f' % (mu, std))
plt.xlabel('Age')
plt.ylabel('Population')
plt.show()

# Fit a normal distribution to hours data
mu, stddev = norm.fit(df['hoursperweek'])

# Plotting histogram
plt.hist(df['hoursperweek'], bins=range(min(df['hoursperweek']), max(df['hoursperweek']) + 5, 5),  alpha=0.6, color='g')

plt.title('Histogram showing hours-per-week\nmu = %.2f,  Standard Deviation = %.2f' % (mu, stddev))
plt.xlabel('hours-per-week')
plt.ylabel('Population')
plt.show()



#*********#
#Bar charts#
#********#
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

# loading data
df = pd.read_csv('/population_data.csv',names = ["age",
"workclass",
"fnlwgt",
"education",
"educationnum",
"maritalstatus",
"occupation",
"relationship",
"race",
"sex",
"capitalgain",
"capitalloss",
"hours-per-week",
"nativecountry","salary"])


cnt_df=df.groupby(['nativecountry','salary']).size().unstack().reset_index()
cnt_df.columns=['country','<=50k','>50k']
cnt_df.fillna(0,inplace = True)
# bar chart
f, axis = plt.subplots(2, 1, sharex=True)
cnt_df.plot(kind='bar', ax=axis[0])
cnt_df.plot(kind='bar', ax=axis[1])
axis[0].set_ylim(5000, 25000)
axis[1].set_ylim(0, 650)
axis[1].legend().set_visible(False)

axis[0].spines['bottom'].set_visible(False)
axis[1].spines['top'].set_visible(False)
axis[0].xaxis.tick_top()
axis[0].tick_params(labeltop='off')
axis[1].set_xticklabels(cnt_df['country'], rotation = 45, ha="right")
axis[1].xaxis.tick_bottom()
d = 0.01
kwargs = dict(transform=axis[0].transAxes, color='k', clip_on=False)
axis[0].plot((-d,+d),(-d,+d), **kwargs)
axis[0].plot((1-d,1+d),(-d,+d), **kwargs)
kwargs.update(transform=axis[1].transAxes)
axis[1].plot((-d,+d),(1-d,1+d), **kwargs)
axis[1].plot((1-d,1+d),(1-d,1+d), **kwargs)

plt.xlabel('nativecountry')
axis[0].set_ylabel('Population in number')
axis[1].set_ylabel('Population in number')
axis[0].set_title('Bar Chart showing number of people from different countries \n having salary <=50K and >50K')
plt.show()

race_df=df.groupby(['race','salary']).size().unstack().reset_index()
race_df.columns=['race','<=50k','>50k']
race_df.fillna(0,inplace = True)
# bar chart, creating sub plots
f, axis = plt.subplots(2, 1, sharex=True)
race_df.plot(kind='bar', ax=axis[0])
race_df.plot(kind='bar', ax=axis[1])
axis[0].set_ylim(5000, 25000)
axis[1].set_ylim(0, 3000)
axis[1].legend().set_visible(False)

axis[0].spines['bottom'].set_visible(False)
axis[1].spines['top'].set_visible(False)
axis[0].xaxis.tick_top()
axis[0].tick_params(labeltop='off')
axis[1].set_xticklabels(race_df['race'], rotation = 45, ha="right")
axis[1].xaxis.tick_bottom()
d = 0.01
kwargs = dict(transform=axis[0].transAxes, color='k', clip_on=False)
axis[0].plot((-d,+d),(-d,+d), **kwargs)
axis[0].plot((1-d,1+d),(-d,+d), **kwargs)
kwargs.update(transform=axis[1].transAxes)
axis[1].plot((-d,+d),(1-d,1+d), **kwargs)
axis[1].plot((1-d,1+d),(1-d,1+d), **kwargs)
plt.xlabel("Race")
axis[0].set_ylabel('Population in number')
axis[1].set_ylabel('Population in number')
axis[0].set_title('Bar Chart showing number of people from different racial backgrounds \nhaving salary <=50K and >50K')
plt.show()

work_df=df.groupby(['workclass','salary']).size().unstack().reset_index()
work_df.columns=['workclass','<=50k','>50k']
work_df.fillna(0,inplace = True)
# bar chart, creating subplots
f, axis = plt.subplots(2, 1, sharex=True)
work_df.plot(kind='bar', ax=axis[0])
work_df.plot(kind='bar', ax=axis[1])
axis[0].set_ylim(2000, 20000)
axis[1].set_ylim(0, 2000)
axis[1].legend().set_visible(False)

axis[0].spines['bottom'].set_visible(False)
axis[1].spines['top'].set_visible(False)
axis[0].xaxis.tick_top()
axis[0].tick_params(labeltop='off')
axis[1].set_xticklabels(work_df['workclass'], rotation = 45, ha="right")
axis[1].xaxis.tick_bottom()
d = 0.01
kwargs = dict(transform=axis[0].transAxes, color='k', clip_on=False)
axis[0].plot((-d,+d),(-d,+d), **kwargs)
axis[0].plot((1-d,1+d),(-d,+d), **kwargs)
kwargs.update(transform=axis[1].transAxes)
axis[1].plot((-d,+d),(1-d,1+d), **kwargs)
axis[1].plot((1-d,1+d),(1-d,1+d), **kwargs)
plt.xlabel('workclass')
axis[0].set_ylabel('Population in number')
axis[1].set_ylabel('Population in number')
axis[0].set_title('Bar Chart showing number of people from different workclasses \n having salary <=50K and >50K')
plt.show()

edu_df=df.groupby(['education','salary']).size().unstack().reset_index()
edu_df.columns=['education','<=50k','>50k']
edu_df.fillna(0,inplace = True)
# bar chart, creating subplots
f, axis = plt.subplots(2, 1, sharex=True)
edu_df.plot(kind='bar', ax=axis[0])
edu_df.plot(kind='bar', ax=axis[1])
axis[0].set_ylim(2000, 20000)
axis[1].set_ylim(0, 2000)
axis[1].legend().set_visible(False)

axis[0].spines['bottom'].set_visible(False)
axis[1].spines['top'].set_visible(False)
axis[0].xaxis.tick_top()
axis[0].tick_params(labeltop='off')

axis[1].xaxis.tick_bottom()
axis[1].set_xticklabels(edu_df["education"], rotation = 45, ha="right")
d = 0.01
kwargs = dict(transform=axis[0].transAxes, color='k', clip_on=False)
axis[0].plot((-d,+d),(-d,+d), **kwargs)
axis[0].plot((1-d,1+d),(-d,+d), **kwargs)
kwargs.update(transform=axis[1].transAxes)
axis[1].plot((-d,+d),(1-d,1+d), **kwargs)
axis[1].plot((1-d,1+d),(1-d,1+d), **kwargs)
plt.xlabel('education')
axis[0].set_ylabel('Population in number')
axis[1].set_ylabel('Population in number')
axis[0].set_title('Bar Chart showing number of people from different educational backgrounds having \n salary <=50K and >50K')
plt.show()


#Scatter plot#
#*********************************************************************************#

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties 
# loading data
df = pd.read_csv('/population_data.csv',names = ["age",
"workclass",
"fnlwgt",
"education",
"educationnum",
"maritalstatus",
"occupation",
"relationship",
"race",
"sex",
"capitalgain",
"capitalloss",
"hoursperweek",
"native-country","salary"])

cnt_df=df.groupby(['education','age','salary']).size().unstack().reset_index()
cnt_df.columns=['education','age','less50k','more50k']
cnt_df.fillna(0,inplace = True)
print(cnt_df)

cnt_df['age'].astype(int)

grouped = cnt_df.groupby('education')
print(grouped)
# Ploting scatterplot
fig, ax = plt.subplots()
for name, group in grouped:
    ax.plot(group.age, group.less50k, marker='o', linestyle='', ms=4, label=name)

for name, group in grouped:    
    ax.plot(group.age, group.more50k, marker='+', linestyle='', ms=7, label=name)
    
fp = FontProperties()
fp.set_size('small')
ax.legend( prop=fp)
plt.title('Scatterplot showing population with education and salary details\n(+ indicates population with salary >50K. o indicates population with salary <=50K.)')
plt.ylabel('Population in numbers')
plt.xlabel('Age')

plt.show()


s_df=df.groupby(['education','age','sex']).size().unstack().reset_index()
s_df.columns=['education','age','Male','Female']
s_df.fillna(0,inplace = True)
print(s_df)

s_df['age'].astype(int)

grouped = s_df.groupby('education')
print(grouped)
# Plotting scatterplot
fig, ax = plt.subplots()

for name, group in grouped:
    ax.plot(group.age, group.Male, marker='+', linestyle='', ms=7, label=name)

for name, group in grouped:
    ax.plot(group.age, group.Female, marker='o', linestyle='', ms=5, label=name)
    
fp = FontProperties()
fp.set_size('small')
ax.legend( prop=fp)
plt.title('Scatterplot showing  population with Age and Education details\n(+ indicates male population. o indicates female population)')
plt.ylabel('Population in numbers')
plt.xlabel('Age')
plt.show()

w_df=df.groupby(['race','educationnum','sex']).size().unstack().reset_index()

w_df.columns=['race','educationnum','Male','Female']
w_df.fillna(0,inplace = True)

grouped = w_df.groupby('race')

# Ploting scatterplot
fig, ax = plt.subplots()
for name, group in grouped:
    print(group)
    ax.plot(group.educationnum, group.Male, marker='+', linestyle='', ms=7, label= name)

for name, group in grouped:
    ax.plot(group.educationnum, group.Female, marker='o', linestyle='', ms=5, label=name)

fp = FontProperties()
fp.set_size('small')
ax.legend( prop=fp)
plt.title('Scatterplot showing population with Race and Education details\n(+ indicates male population. o indicates female population)')
plt.ylabel('Population in numbers')
plt.xlabel('Education')
plt.show()

w_df=df.groupby(['workclass','age','sex']).size().unstack().reset_index()

w_df.columns=['workclass','age','Male','Female']
w_df.fillna(0,inplace = True)

grouped = w_df.groupby('workclass')

# Plotting scatterplot
fig, ax = plt.subplots()

for name, group in grouped:
    print(group)
    ax.plot(group.age, group.Male, marker='+', linestyle='', ms=10, label=name)

for name, group in grouped:
    ax.plot(group.age, group.Female, marker='o', linestyle='', ms=5, label=name)

fp = FontProperties()
fp.set_size('small')
ax.legend( prop=fp)
plt.title('Scatterplot showing population with age and workclass details\n(+ indicates male population. o indicates female population)')
plt.ylabel('Population in numbers')
plt.xlabel('Age')

plt.show()

#Bubble plot

#1. Bubble plot showing number of people with salary <=50K or >50K \n with respect to workclass #and hours-per-week(Yellow indicates salary<=50K,Blue indicates salary>50K)
#********************************************************************************#
df1 = df[['hoursperweek','workclass','salary']]

df1=df1[df1['salary'].str.contains("<=50K")]
print(df1)
s_df=df1.groupby(['hoursperweek','workclass']).size()

s_df.columns=['hoursperweek','workclass','<=50k']



xlabels, ylabels = s_df.index.levels
xs,ys=s_df.index.labels

plt.xticks(np.arange(0, 100, step=5))
plt.yticks(range(0,9), ylabels)

    
plt.scatter(xs,ys, s=s_df,c='b',alpha=0.5) 

#plt.show()

df2 = df[['hoursperweek','workclass','salary']]
df2=df2[df2['salary'].str.contains(">50K")]
r = df2[df2['workclass'].str.contains('Never-worked')]
df2.append(r)
s_df2=df2.groupby(['hoursperweek','workclass']).size()

s_df.columns=['hoursperweek','workclass','>50k']


#s_df = s_df.unstack()
print(s_df2)
xlabels, ylabels = s_df.index.levels
xs,ys=s_df.index.labels


# plotting bubble plot     
plt.scatter(xs,ys, s=s_df2,c='y',alpha=0.5) 
plt.xticks(np.arange(0, 100, step=5))


plt.xlabel('hours-per-week')
plt.ylabel('workclass')
plt.title('Bubble plot showing number of people with salary <=50K or >50K \n with respect to workclass and hours-per-week\n(Blue indicates salary<=50K,Yellow indicates salary>50K)')
plt.legend()
plt.show()
