#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import date


# In[14]:


df = pd.read_csv('top20_deathtoll.csv')
df.head(10)


# In[16]:


fig, ax = plt.subplots(figsize = (4.5,6))
ax.barh(df['Country_Other'], df['Total_Deaths'], height = 0.8)


# In[32]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig, ax = plt.subplots(figsize = (4.5,6))
ax.barh(df['Country_Other'], df['Total_Deaths'], height = 0.8)
direction = ['right','left','top','bottom']
for i in direction:
    ax.spines[i].set_visible(False)


# In[42]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig, ax = plt.subplots(figsize = (10,10))
ax.barh(df['Country_Other'], df['Total_Deaths'], height = 0.8, color = 'pink')
direction = ['right','left','top','bottom']
for i in direction:
    ax.spines[i].set_visible(False)
ax.set_xticks([0,100000,200000,300000])
ax.xaxis.tick_top()
ax.tick_params(left = False, top = False)


# In[52]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig, ax = plt.subplots(figsize = (10,10))
ax.barh(df['Country_Other'], df['Total_Deaths'], height = 0.8, color = 'pink')
direction = ['right','left','top','bottom']
for i in direction:
    ax.spines[i].set_visible(False)
ax.set_xticks([0,100000,200000,300000])
ax.xaxis.tick_top()
ax.tick_params(left = False, top = False)
ax.text(x = -80000, y = 23.5, s = 'The Death Toll World wide Is 1.5M+', size = 19, weight = 'bold', color = 'pink')
ax.text(x = -80000, y = 22.5, s = 'Top 20 countries by death toll (December 2020)', size = 15, color = 'purple')


# In[54]:


df = pd.read_excel('Sample - Superstore.xls', sheet_name = 'Orders')
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Year'] = df['Order Date'].dt.year


# In[55]:


df.head(10)


# In[82]:


fig, ax0 = plt.subplots(figsize = [10,5])
sum_df = df.groupby('Year').Sales.sum()
ax0.bar(sum_df.index, sum_df.values, width = 0.3, color = '#BB98B8')
direction = ['right','left','top','bottom']
for i in direction:
    ax0.spines[i].set_visible(False)
ax0.tick_params(left = False, bottom = False)
ax0.set_xticks([2018, 2019, 2020, 2021])
for p in ax0.patches: # viết số trên cột
    width, height = p.get_width(), p.get_height() # lấy chiều rộng, chiều cao của cột
    x, y = p.get_xy()
    ax0.annotate('{}'.format(int(height)),\
                 (x,height + 20000)) # viết số trên cột
profit_df = df.groupby('Year').Profit.sum()
ax0.set_yticklabels([str(int(x/1000)) + '$' for x in ax0.get_yticks()]) #thay đổi parameters sang chữ
ax1 = ax0.twinx()
ax1.plot(profit_df.index,profit_df.values, color = 'red', linewidth = 2, marker = '*')
ax0.spines['top'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax0.spines[['left','right','bottom']].set_color('red')
ax1.spines[['left','right','bottom']].set_color('red')
ax0.set_ylabel('Revenue (Thousand)')
ax1.set_ylabel('Profit (Thousand)')
ax0.tick_params(color = 'black')
ax1.tick_params(color = 'black')
ax0.set_yticklabels([str(int(x/1000)) + '$' for x in ax0.get_yticks()])
ax1.set_yticklabels([str(int(x/1000)) + '$' for x in ax1.get_yticks()])


# In[84]:


df3 = pd.DataFrame()
df3['Month'] = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
df3['Received'] = [160,184,241,149,180,161,132,202,160,139,149,177]
df3['Processed'] = [160,184,237,148,181,150,123,156,126,104,124,140]
df3.head()


# In[104]:


fig, ax3 = plt.subplots(figsize = [10,5])
ax3.plot(df3.Month, df3.Received, color = 'red', linewidth = 2)
ax3.set_yticks([0, 50, 100, 150, 200, 250, 300])
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

ax3.plot(df3.Month, df3.Processed, color = 'blue', linewidth = 2)


# In[ ]:




