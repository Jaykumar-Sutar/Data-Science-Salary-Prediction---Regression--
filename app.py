# %% [markdown]
# ## 1. Problem Statement:
# - To perform exploratory data analysis to find important insights into salaries of data science professionals.

# %% [markdown]
# ## 2) Data Collection.
# * The Dataset is collected from https://www.kaggle.com/datasets/saurabhshahane/data-science-jobs-salaries
# * The data consists of 11 column and 607 rows.

# %% [markdown]
# ### 2.1 Import Data and Required Packages
# #### Importing Pandas, Numpy, Matplotlib, Seaborn and Warings Library.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
warnings.filterwarnings("ignore")
import pycountry

# %% [markdown]
# #### Import the CSV Data as Pandas DataFrame

# %%
df = pd.read_csv('data/ds_cleaned.csv')

# %% [markdown]
# #### Show Top 5 Records

# %%
df.head(5)

# %% [markdown]
# #### Shape of the dataset

# %%
df.shape

# %% [markdown]
# #### Data Science Job Salaries Dataset contains 11 columns
# 1. work_year : The year the salary was paid.
# 
# 2. experience_level : The experience level in the job during the year
# 
# 3. employment_type : The type of employment for the role
# 
# 4. job_title : The role worked in during the year.
# 
# 5. salary : The total gross salary amount paid.
# 
# 6. salary_currency : The currency of the salary paid as an ISO 4217 currency code.
# 
# 7. salaryinusd : The salary in USD
# 
# 8. employee_residence : Employee's primary country of residence in during the work year as an ISO 3166 country code.
# 
# 9. remote_ratio : The overall amount of work done remotely
# 
# 10. company_location : The country of the employer's main office or contracting branch
# 
# 11. company_size : The median number of people that worked for the company during the year

# %% [markdown]
# #### Summary of the dataset
# 
# - The described method will help to see how data has been spread for numerical values.
# - We can clearly see the minimum value, mean values, different percentile values, and maximum values.

# %%
df.describe()

# %% [markdown]
# #### Check Datatypes in the dataset
# #### info() is used to check the Information about the data and the datatypes of each respective attribute

# %%
df.info() 

# %% [markdown]
# #### Insights
# - Most of the data is categorical, As data has 7 object and 4 numeric feature.
# - There are no missing values

# %%
numeric_features = [feature for feature in df.columns if df[feature].dtype != 'O']
categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']

# print columns
print('We have {} numerical features : {}'.format(len(numeric_features), numeric_features))
print('\nWe have {} categorical features : {}'.format(len(categorical_features), categorical_features))

# %% [markdown]
# ### Univariate Analysis
# - The term univariate analysis refers to the analysis of one variable prefix “uni” means “one.” The purpose of univariate analysis is to understand the distribution of values for a single variable.
# -Other Type of Analysis are
#     - Bivariate Analysis: The analysis of two variables.
#     - Multivariate Analysis: The analysis of two or more variables.

# %%
plt.figure(figsize=(15, 10))
plt.suptitle('Univariate Analysis of Numerical Features', fontsize=20, fontweight='bold', alpha=0.8, y=1.)

for i in range(0, len(numeric_features)):
    plt.subplot(2, 2, i+1)
    sns.kdeplot(x=df[numeric_features[i]], color='blue')
    plt.xlabel(numeric_features[i])
    plt.tight_layout()

# %%
unwanted_categories = {'experience_level', 'job_title', 'salary_currency','Updated_Job_Title'}
categorical_features = [element for element in categorical_features if element not in unwanted_categories]

# %%
# categorical columns
plt.figure(figsize=(15, 10))
plt.suptitle('Univariate Analysis of Categorical Features', fontsize=20, fontweight='bold', alpha=0.8, y=1.)

for i in range(0, len(categorical_features)):
    plt.subplot(3, 2, i+1)
    sns.countplot(x=df[categorical_features[i]])
    plt.xlabel(categorical_features[i])
    plt.tight_layout()

# %% [markdown]
# #### Insights
# - Higher number of the employees are located in US which justifies the fact that most of the companies are of US.
# - Almost all of the employee are full time employees indicating it is difficult to get a part-time job in data science domain.
# - Majority of the companies are Medium Sized companies.
# - Most of the companies are flexible as most of them have high remote ratio
# 

# %% [markdown]
# #### Job Title - How many job titles are there in our dataset ?

# %%
print('how many job titles in the dataset: ',df['job_title'].value_counts().size)
top_job_titles = df.groupby('job_title').size().reset_index().sort_values(by=0,ascending = False)
top_job_titles.head()

# %%
plt.subplots(figsize=(14,7))
sns.set_style("darkgrid")
sns.barplot(x='job_title',y=0,data = top_job_titles[:10],palette = 'hls')
plt.title('Top 10 Jobs Titles')
plt.xlabel('Job Title')
plt.ylabel('Counts')
plt.xticks(rotation=45)
plt.show()

# %% [markdown]
# #### Inference
# - Data scientist, data engineer and data analyst ranked top 3 frequent job titles.

# %% [markdown]
# #### Employment Type - What kind of employment type is most frequent ?

# %%
emp_type = df.employment_type.value_counts()
explode = [0,0.1,0.1, 0.3]
f,ax=plt.subplots(1,2,figsize=(20,10))
sns.countplot(x=df['employment_type'],data=df,palette ='bright',ax=ax[0],saturation=0.95)
for container in ax[0].containers:
    ax[0].bar_label(container,color='black',size=20)
    
plt.pie(x=df['employment_type'].value_counts(),labels=emp_type.index,explode=explode,autopct='%1.2f%%',shadow=True)
plt.show()

# %% [markdown]
# #### Insights
# - It is observed that most of the Employees are doing Full Time Jobs
# - As observed lowest number of he Employees are doing Freelance.

# %% [markdown]
# #### Location - Where most of Data Science Companies are located ??

# %%
plt.figure(figsize = (14,10))
sns.set_context("talk")
sns.set_style("darkgrid")
locations = df.company_location.value_counts().head(5)
ax = sns.barplot(x = locations.values , y = locations.index , data = df, ec = "black", palette="Set2_r" )
ax.set_xlabel('No. of Counts')
ax.set_ylabel('Company Location')
ax.set_title("Most of Data Science Companies are locations", size = 20)

# %% [markdown]
# #### Insights
# - It is observed that most of Data Science Companies are located at USA

# %% [markdown]
# #### Employee Location - Where do most data scientist live ??

# %%
residence = df.groupby('employee_residence').size().sort_values(ascending =False).reset_index().head()
plt.subplots(figsize=(14,7))
sns.barplot(x="employee_residence", y = 0,data= residence,ec = "black",palette="CMRmap")

# %% [markdown]
# #### Insights
# - Most of the employees reside in US which conincides with the fact that most of the companies are of US.
# - India is 3rd country where most employees are located

# %%
# clean
df.company_size.replace(['M','L',"S"], ['Medium', 'Large' ,'Small'],inplace = True)

# %%
plt.subplots(figsize=(14,7))
sns.countplot(x=df.company_size, data= df,ec = "black",palette="CMRmap")
plt.title("Distribution of Company Size", weight="bold",fontsize=20, pad=20)
plt.ylabel("Count", weight="bold", fontsize=20)
plt.xlabel("Company Size", weight="bold", fontsize=16)
plt.show()

# %% [markdown]
# #### Insights
# - It can be easily seen that company size mostly consists of 'medium size' and the 'large size' ranked the next.

# %%
# clean
df.remote_ratio.replace([100,50,0], ['Remote', 'Hybrid' ,'On-site'],inplace = True)
plt.figure(figsize=(14,8)) 
sns.scatterplot(data=df,y=df.company_location.sort_values(),x=df.employee_residence.sort_values(),color="g", hue=df['remote_ratio'])
plt.xticks(rotation='vertical',size=10)
plt.yticks(size=10)
plt.xlabel("Employee Residence",size=15,c="r")
plt.ylabel('Company Location',size=15,c="r")
plt.title("Company Location VS Employee Residence for type of work(Remote, Hybrid or On-site)",size=15,c="b")
plt.show()


# %% [markdown]
# #### Insights
# - It is observed that most of the remote employees were from different Countries.

# %% [markdown]
# #### work_year - What trend Data science jobs are following ?

# %%
work_yr = df.work_year.value_counts()
explode = [0.1,0,0.3]
f,ax=plt.subplots(1,2,figsize=(20,10))
sns.countplot(x=df['work_year'],data=df,palette ='bright',ax=ax[0],saturation=0.95)
for container in ax[0].containers:
    ax[0].bar_label(container,color='black',size=20)
    
plt.pie(x=df['work_year'].value_counts(),labels=work_yr.index,explode=explode,autopct='%1.2f%%',shadow=True)
plt.show()

# %% [markdown]
# #### Insights
# - Amoung all 3 years most of the salaries were paid in 2022.Year - 2022 is 52.39% of whole distribution of year.

# %% [markdown]
# #### salary_in_usd - What is average salary Data scientist receiving ?

# %%
fig, axs = plt.subplots(1, 2, figsize=(15, 7))
plt.subplot(121)
sns.boxplot(df['salary_in_usd'],color='hotpink')
plt.subplot(122)
sns.histplot(data=df,x=df['salary_in_usd'],kde=True)
plt.show()

# %% [markdown]
# #### Insights
# - We can see that salary mostly distributed between 100k and 150k.
# - We can trea Above 300K as outliers.

# %% [markdown]
# ####  Employment Type by Experience Level

# %% [markdown]
# - EN, which refers to Entry-level / Junior
# - MI, which refers to Mid-level / Intermediate
# - SE, which refers to Senior-level / Expert
# - EX, which refers to Executive-level / Director

# %%
df['experience_level'] = df['experience_level'].replace('EN','Entry-level/Junior')
df['experience_level'] = df['experience_level'].replace('MI','Mid-level/Intermediate')
df['experience_level'] = df['experience_level'].replace('SE','Senior-level/Expert')
df['experience_level'] = df['experience_level'].replace('EX','Executive-level/Director')

# %%
plt.figure(figsize=(14,7)) 
sns.countplot(x="employment_type",hue="experience_level", data=df,ec = "black",palette="Set2")

# %% [markdown]
# #### Insights
# - Almost all of the employee are full time employees indicating it is difficult to get a part-time job in data science domian.
# - We can see that type of Part-Time consists of Entry-level and Mid-level.
# - Additionally, type of Freelance consists of Mid-level and Senior-level.

# %% [markdown]
# ####  What's the relation between salary and experience ?

# %%
plt.figure(figsize=(14,7)) 
sns.barplot(x = df['experience_level'], y = df['salary_in_usd'],ec = "black",palette="Set1")

# %% [markdown]
# #### Insights
# - It's noticeable that as experience level increases so does salary.
# - Entry-level/junior are geting lowest salary.

# %% [markdown]
# #### Is there any relation between  salary and job title ?

# %%
plt.figure(figsize=(14,7))
sns.barplot(x = 'Updated_Job_Title', y = 'salary_in_usd', data = df,ec = "black",palette="Set2")
plt.xticks(rotation= 45)
plt.show()

# %% [markdown]
# #### Insights
# - Managers are earning more compared to other roles which is expected.
# - It is intresting to note that Data Engineers are earning a bit more compared to ML Engineer and Data Scientist.
# - Salaries of ML Engineer varies more compared to other roles.

# %% [markdown]
# #### Is there any relation between company location and experience level ?

# %%
plt.figure(figsize=(14,7))
sns.barplot(x = 'employee_residence', y = 'salary_in_usd', data = df,ec = "black",palette="Set2")

# %% [markdown]
# #### Insights
# - It can be observed that employess of US are earning more compared to employees of other locations.
# - Employees residing in India are earning comparively less when compared to other employees of other nations .

# %% [markdown]
# #### Is there any relation between salary and company size ?

# %%
plt.figure(figsize=(14,7))
sns.barplot(x = 'company_size', y = 'salary_in_usd', data = df,ec = "black", palette="Set2_r")

# %% [markdown]
# #### Insights
# - Medium and Large size companies are almost giving almost equal salary.
# - Small size companies are paying less.

# %% [markdown]
# #### Multivariate Analysis

# %%
plt.figure(figsize=(14,7))
sns.barplot(x = 'company_size', y = 'salary_in_usd', hue = 'experience_level', data = df,ec = "black",palette="Set1")

# %% [markdown]
# #### Insights
# - Whether Large, Small or Medium organisation,Executive-level/Director are getting highest salary.

# %%
US_data = df[df['employee_residence'] == 'US']

# %%
US_data_scientist = US_data[US_data['Updated_Job_Title']  == 'Data Scientist'] 
US_data_engineer = US_data[US_data['Updated_Job_Title']  == 'Data Engineer']
US_data_analyst = US_data[US_data['Updated_Job_Title'] == 'Data Analyst']
US_ml_engineer = US_data[US_data['Updated_Job_Title'] == 'Machine Learning Engineer']
US_manager = US_data[US_data['Updated_Job_Title'] == 'Manager']

# %%
plt.figure(figsize=(15,8))
plt.subplot(3,2,1)
sns.distplot(US_data_scientist['salary'], color='red', kde=True, label='train')
plt.title('US Data Scienist Salary Distributions')
plt.subplot(3,2,2)
sns.distplot(US_data_engineer['salary'], color='red', kde=True, label='train')
plt.title('US Data Engineer Salary Distributions')
plt.subplot(3,2,3)
sns.distplot(US_data_analyst['salary'], color='red', kde=True, label='train')
plt.title('US Data Analyst Salary Distributions')
plt.subplot(3,2,4)
sns.distplot(US_ml_engineer['salary'], color='red', kde=True, label='train')
plt.title('US ML Engineer Salary Distributions')
plt.subplot(3,2,5)
sns.distplot(US_manager['salary'], color='red', kde=True, label='train')
plt.title('US Manager Salary Distributions')
plt.tight_layout()

# %% [markdown]
# #### Insights
# - We can see for all designations salary mostly distributed between 100k and 300k.
# - We can treat Above 300K as outliers.

# %%
sns.pairplot(df);

# %% [markdown]
# #### How has the average salary of data science jobs change over time?

# %%
avg_sal_change=df.groupby('work_year')['salary_in_usd'].mean()
plt.figure(figsize=(12,8))
sns.lineplot(x=avg_sal_change.index,y=avg_sal_change.values,linewidth=3)

xticks=[2020,2021,2022]
plt.xticks(xticks,xticks)
plt.xlabel('Year')
plt.ylabel('Average Salary')
plt.title('Average Salary of Data Science Jobs change', size=20);

# %% [markdown]
# #### Insights
# - The average salaries increased from approximately 96000USD in 2020 to 100000USD in 2021 and finally 125000USD in 2022.
# - This shows a positive trend and shows that data science jobs are becoming more valuable as the years pass.

# %%
#experienced employees number in different data science job profiles histogram
plt.figure(figsize=(12,8))
ax = sns.histplot(x = "Updated_Job_Title" , hue = "experience_level" , data = df  , kde = True )
plt.title("Job department with Experience level")
plt.xticks(rotation=45);
for i in ax.containers:     #to set a label on top of the bars.
    ax.bar_label(i,)

# %% [markdown]
# #### Insights
# - Data Scientist(36) job profile has maximum number of entry level employees.after that data analyst(17) and ML engineer(15) comes.
# - Data Scientist(73) job profile has maximum number of Mid-level employees.after that data engineer(61) and data analyts(40) comes.
# - manager(9) job profile has maximum number of Executive-level employees.after that data engineer(5) and data analyts(5) comes.

# %% [markdown]
# #### Conclusions-
# - Majority of the Data science professionals are working as full time employee.
# - Experience level plays a significant role on salary.
# - Most common job_title in data science fields are:
#     - Data Scientist
#     - Data Engineer
#     - Data Analyst
#     - Machine Learning Engineer
#     - Managers
#     - Others
#     
# - Among job titles Manager are paid more compared to other job titles.
# - Majority of data points are from US and hence it becomes a little difficult to do any analysis based on location
# - Salary is directly proporptional to experience_level that means higher the experience, higher the salary.
# - Data Science Salaries are higher in US comparing to other western countries, India on the other hand has lower salaries


