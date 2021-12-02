#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install numpy')
get_ipython().system('pip install pandas')


# In[3]:


get_ipython().system('pip install matplotlib.pyplot')
get_ipython().system('pip install seaborn')
get_ipython().system('pip install plotly.express')
get_ipython().system('pip install make_subplots')
get_ipython().system('pip install datetime')


# In[4]:


get_ipython().system('pip install plotly.subplots')
get_ipython().system('pip install matplotlib')


# In[5]:


get_ipython().system('pip install plotly')


# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime


# In[10]:


covid_df=pd.read_csv("/home/barish/my_project_dir/my_project_env/covid_19_india.csv")


# In[11]:


covid_df.head(10)


# In[12]:


covid_df.info()


# In[13]:


covid_df.describe()


# In[14]:


vaccine_df=pd.read_csv("/home/barish/my_project_dir/my_project_env/covid_vaccine_statewise.csv")


# In[15]:


vaccine_df.head(5)


# In[16]:


covid_df.drop(["Sno","Time","ConfirmedIndianNational","ConfirmedForeignNational"],inplace = True, axis =1 )


# In[18]:


covid_df.head(5)


# In[22]:


covid_df['Date']=pd.to_datetime(covid_df['Date'],format='%Y-%m-%d')


# In[23]:


covid_df.head()


# In[26]:


# total number of active cases
covid_df['Active_cases']= covid_df['Confirmed'] - covid_df['Cured']+covid_df['Deaths']
covid_df.tail()


# In[27]:


statewise = pd.pivot_table(covid_df,values =["Confirmed","Deaths","Cured"],
                           index = "State/UnionTerritory",aggfunc=max)


# In[28]:


statewise["Recovery Rate"]= statewise["Cured"]*100/statewise["Confirmed"]


# In[29]:


statewise["Mortality Rate"]= statewise["Deaths"]*100/statewise["Confirmed"]


# In[30]:


statewisse =statewise.sort_values(by = "Confirmed",ascending = False)


# In[35]:


statewise.style.background_gradient(cmap ="cubehelix")


# In[37]:


# Top 10 Active Cases States
top_10_active_cases =covid_df.groupby(by='State/UnionTerritory').max()[['Active_cases','Date']].sort_values(by=['Active_cases'],ascending = False).reset_index()


# In[38]:


fig=plt.figure(figsize=(15,9))


# In[47]:


plt.title("Top 10 States with most number of Active Cases in India",size =20)


# In[49]:


plt.xlabel("States")
plt.ylabel("Total Active Cases")
plt.show()


# In[58]:


# Top 10 Active Cases States
top_10_active_cases =covid_df.groupby(by='State/UnionTerritory').max()[['Active_cases','Date']].sort_values(by=['Active_cases'],ascending = False).reset_index()
fig=plt.figure(figsize=(15,9))
plt.title("Top 10 States with most number of Active Cases in India",size =20)
ax=sns.barplot(data=top_10_active_cases.iloc[:10],y="Active_cases",x="State/UnionTerritory")
plt.xlabel("States")
plt.ylabel("Total Active Cases")
plt.show()


# In[62]:


#Top States with Highest Deaths
top_10_states_deaths = covid_df.groupby(by= 'State/UnionTerritory').max()[['Deaths','Date']].sort_values(by =['Deaths'],ascending =False).reset_index()
fig =plt.figure(figsize=(18,5))
plt.title("Top 10 States with most number of Deaths",size = 25)
ax= sns.barplot(data=top_10_states_deaths.iloc[:12],y= "Deaths",x="State/UnionTerritory",linewidth =2,edgecolor='black')
plt.xlabel( "States")
plt.ylabel("Total Death Cases")
plt.show()


# In[79]:


# Growth 
fig = plt.figure(figsize=(12,6))

ax = sns.lineplot(data = covid_df[covid_df['State/UnionTerritory'].isin(['Maharashtra','Karnataka', 'Kerala','Tamil Nadu' ,'Uttar Pradesh'])],x='Date',y='Active_cases',hue ='State/UnionTerritory')

ax.set_title("Top 5 Affected States in India",size =16)


# In[80]:


vaccine_df.head()


# In[81]:


vaccine_df.rename(columns= {'Updated On': 'Vaccine_Date'}, inplace= True)


# In[82]:


vaccine_df.head(10)


# In[83]:


vaccine_df.info()


# In[84]:


vaccine_df.isnull().sum()


# In[86]:


vaccination = vaccine_df.drop(columns=['Sputnik V (Doses Administered)','AEFI','18-44 Years (Doses Administered)','45-60 Years (Doses Administered)','60+ Years (Doses Administered)'],axis =1)


# In[87]:


vaccination.head()


# In[90]:


# Male vs Female Vaccination

male = vaccination["Male(Individuals Vaccinated)"].sum()
female = vaccination["Female(Individuals Vaccinated)"].sum()
px.pie(names=["Male","Female"],values=[male,female],title = "Male and Female Vaccination")
    


# In[91]:


#Remove rows where state = India
vaccine = vaccine_df[vaccine_df.State!='India']


# In[92]:


vaccine


# In[94]:


vaccine.rename(columns ={"Total Individuals Vaccinated": "Total"},inplace =True)
vaccine.head()


# In[95]:


# State with most number of vaccinations
max_vaccine = vaccine.groupby('State')['Total'].sum().to_frame('Total')
max_vaccine = max_vaccine.sort_values('Total',ascending =False)[:5]
max_vaccine


# In[100]:


fig = plt.figure(figsize = (10,5))
plt.title(" Top 5 Vaccinated States in India", size=20)
x = sns.barplot(data = max_vaccine.iloc[:10],y= max_vaccine.Total,x = max_vaccine.index,linewidth=2,edgecolor ='red')
plt.xlabel("States")
plt.ylabel("Vaccination")
plt.show()


# In[102]:


# State with minimum number of vaccinations
min_vaccine = vaccine.groupby('State')['Total'].sum().to_frame('Total')
min_vaccine = min_vaccine.sort_values('Total',ascending =True)[:5]
min_vaccine


# In[103]:


fig = plt.figure(figsize = (10,5))
plt.title(" Last 5 Vaccinated States in India", size=20)
x = sns.barplot(data = min_vaccine.iloc[:10],y= min_vaccine.Total,x = min_vaccine.index,linewidth=2,edgecolor ='black')
plt.xlabel("States")
plt.ylabel("Vaccination")
plt.show()


# In[ ]:




