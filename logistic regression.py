#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import necessary libraries
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# ## **Data Exploration**

# ### **Loading Data Set And Performing Statistical Analysis**

# In[4]:


#loading the data sets 
train_df = pd.read_csv("C:\\Users\\abhin\\Downloads\\Titanic_train.csv")
test_df = pd.read_csv("C:\\Users\\abhin\\Downloads\\Titanic_test.csv")


# In[5]:


#Displaying a few rows of the data set 
train_df.head()


# In[6]:


#data sets information
train_df.info()


# In[7]:


train_df.describe()


# In[8]:


train_df.skew(numeric_only = True)


# In[9]:


train_df.kurt(numeric_only = True)


# ### **General Observstions**

# #### **Statistical Summary of Train set**
# 
#  - There are 891 entries in the train data set which is 80% of the raw data set that gives about <400 entries should be present in the test data set.
# 
#  - The avg of survived is 0.38 i.e, 38% considering there are 891 people are present .around 300 - 350 people survived.
# 
#      > - Pclass(passenger class) avg is 2.3 indicating that most of them are 2nd and 3rd classes with a few 1st class passengers
#      > - varible age avg is 29.6 with a few missing values which can bring a little change in the mean
#      > - the avg of sibling/spouse is 0.52 indicates that 1 in 2 travelled with sibling or spouse which means that most of the passengers came with family.
#      > - parch with avg of 0.38 says that 1 in 3 people are have a children or a parent with them.
#      > - as per fare paid by the passenger's avg is 32.2 which above the 75th of the data suggests that a few passenger's paid a hefty amount of fare than most of them probably first class passengers
# 
#  - Age range consists from 0.42(min) a few months old child to 80(max) years old an elderly
# 
#      > - the siblings/spouse range from 1(which is 75%percentile ,probably most of them are spouses) to 8 which indicates siblings
#      > - while parents and child max value is 6  . which is probably indicates children
#       
#    #### **From Skewness and Kurtosis**
# 
#  > - Variables like sib/sp,par/ch and fare are highly right skewed while Age,survived are nearly symmetric with pclass being slightly left skewed
# 
#  >  - likewise sib/sp,par/ch and fare are leptokurtic indicating heavy tails and plausible sharp peak.survived,pclass are platykurtic suggests light tails and flatter peak
# 
#  >  - the passengerid variable is identifier so its common to have a uniform distribution 

# ### **Data visualization**

# In[13]:


#Identifying numerical columns in both sets
tr_numeric_cols = train_df.select_dtypes(include = ['int64','float64']).columns
t_numeric_cols = test_df.select_dtypes(include = ['int64','float64']).columns
print("train numerical columns:",tr_numeric_cols)
print("test numerical columns:",t_numeric_cols)


# In[14]:


#creating histograms of numerical columns
train_df.hist(bins = 30,figsize = (15,10))
plt.suptitle("Histogram of numerical columns")
plt.show()

#Now we create boxplot of numerical columns
plt.figure(figsize=(15,15),facecolor="pink")
for i,col in enumerate(tr_numeric_cols):
    plt.subplot(4,2, i + 1) #Adjusting the columns as per the data
    sns.boxplot(y =train_df[col],color ="lightgreen")
    plt.suptitle("Boxplot of numerical columns")
    
plt.tight_layout()
plt.show()


# In[15]:


#let's check the corelation between the numerical columns
corr_matrix =train_df[tr_numeric_cols].corr()

#visualizing a corelation heatmap for numerical columns
plt.figure(figsize = (10,8))
sns.heatmap(corr_matrix,annot = True, cmap = "coolwarm",fmt = ".2f")
plt.show()


# In[16]:


#now let's create a pair plots
sns.pairplot(train_df[tr_numeric_cols])
plt.show()


# ### **Analysis from Visualization**
# 
# #### **Histogram**
#  - PassengerId shows a uniform distribution, which is obvious. since PassengerId is typically a unique identifier for each passenger.
#  - Survived variable is a binary column (0 = did not survive, 1 = survived). The histogram indicates that a significant number of passengers did not survive which is about 60-70%, with a smaller but substantial percentage who did.
# 
#  - Pclass represents the passenger's class (1 = 1st class, 2 = 2nd class, 3 = 3rd class). The distribution shows that the majority of passengers were in the 3rd class(which has low fare), followed by the 1st and 2nd classes.
# 
#  - Age distribution is right-skewed, indicating that most passengers were younger, with fewer older passengers. There is a peak around the 20-30 age range most lighly to be 25-75th percentile.
# 
#  - Sib/Sp column represents the number of siblings/spouses aboard. The distribution shows that most passengers traveled alone (0), with a decreasing number as the count of siblings/spouses increases(there's high chance that passengers with 1 are indicating most of the values are spouses).
# 
#  - The fare distribution is highly right-skewed, indicating that most passengers paid lower fares(3rd class are most of them), with a few paying significantly higher amounts. This could reflect the different classes and possibly the presence of luxury accommodations.

# #### **Boxplots**
# 
#  - Survived and Pclass suggests categorical distributions rather than continuous distributions. suvived being 0 = not survived,1= survived.while Pclass as 1,2,3 categories.
# 
#  - Age almost a roughly normal distribution with some outliers at the higher end (above ~60 years old high chance being actual ). The median age is around 30.
# 
#  - Most passengers had very few siblings/spouses onboard with some having more than 5.
# 
#  - Similar to Sib/Sp, most passengers had no parents/children with them, with a few having 2 and 3,4.
# 
#  - Some passengers paid significantly higher fares,most likely it's first-class passengers for more lavish life in the ship.
# 
#   **Insights**
# * Pclass and Survived are categorical columns.
# * Fare and few are highly skewed due to extreme values.
# * Most passengers were between 20-40 years old.

# #### **Heatmap and Pairplot**
# 
#  - As expected from passengerid column having no correlation with any other,since it a unique identifier.
# 
#  - From surivived variable a moderate correlation with Pclass(0.36) and fare(0.2) which indicates that those who paid high fares and were high class(1st class) are most likely to survive.while it has less correlation with age means it doesn't matter whether passenger is young or not.
# 
#  - Obviously Pclass is highly correlated with fare .because of high class passengers and a significant correlation with age it probably elder passengers in high class.
# 
#  - A high correlation between sib/sp and par/ch(0.41).suggests that families are travelling together.
# 
#  - Fare having correlation with Pclass reflects that fare differences in classes.

# ## **Data Preprocessing**

# ### **Handling Missing Values**

# In[22]:


#Identifying the null values in the both sets
print("train:",train_df.isnull().sum())
print("test:",test_df.isnull().sum())


# In[23]:


#A few columns have missing values especially the 'Cabin' column have approximately 80% of missing values 
#since it disrupts the actual analysis let's drop the column
train_df.drop(columns =['Cabin'],inplace =True)
test_df.drop(columns =['Cabin'],inplace =True)


# In[24]:


#the other two column which has missing values of a few let's replace them with median and mode of each column
train_df.fillna(train_df['Age'].median(),inplace = True) #fornumerical columns
train_df.fillna(train_df['Embarked'].mode()[0],inplace =True) #for categorical columns
test_df.fillna(test_df['Age'].median(),inplace = True) #since there are no null values in test set column let's ignore it.


# In[25]:


print(train_df.isnull().sum())
print(test_df.isnull().sum())


# ### **Encoding Categorical Columns**
#  - Before we proceed with this process there are few aspects we need to address which is about object data type columns there are two types one is  categorical columns and other is just string data types which has unique values.
#  - Especially the name column has unique values for each row which doesn't provide predictive power to the model.
#    > There are few reasons why they don't prrovide any help in predicting.
#    >- Lack of reusable information : since they all are unique values,it can't learn any pattern.
#    >- Noise : the unique values increases noise instead of contributing to the betterment of model.
#    >- Overfitting : rather than learning pattern in feature with the unique values
#    
#    - However the ticket column may contain useful info that could be provide some patterns but others features like Pclass,fare and embarked fills it's place.
#    

# In[27]:


#Droping Name and Ticket columns
train_df.drop(columns=['Name','Ticket'],inplace = True)
test_df.drop(columns =['Name','Ticket'],inplace = True)


# In[28]:


#One-hot encoding to the categorical columns
tr_categoric_cols = train_df.select_dtypes(include =['object']).columns
t_categoric_cols = test_df.select_dtypes(include =['object']).columns
train_encoded =pd.get_dummies(train_df,columns= tr_categoric_cols,drop_first=True)
test_encoded =pd.get_dummies(test_df,columns = t_categoric_cols,drop_first=True)
print(train_encoded.head())
print(test_encoded.head())


# In[29]:


#Coverting the boolean columns to integers
train_encoded.astype(int , errors = 'ignore' )
train_encoded['Sex_male'] =train_encoded['Sex_male'].astype('int64')
train_encoded['Embarked_C'] =train_encoded['Embarked_C'].astype('int64')
train_encoded['Embarked_Q'] =train_encoded['Embarked_Q'].astype('int64')
train_encoded['Embarked_S'] =train_encoded['Embarked_S'].astype('int64')
test_encoded.astype(int , errors = 'ignore' )
test_encoded['Sex_male'] =test_encoded['Sex_male'].astype('int64')
test_encoded['Embarked_Q'] =test_encoded['Embarked_Q'].astype('int64')
test_encoded['Embarked_S'] =test_encoded['Embarked_S'].astype('int64')


# In[30]:


print(train_encoded.dtypes)
print(test_encoded.dtypes)


# In[31]:


missing_cols = set(train_encoded.columns) - set(test_encoded.columns)

# Add missing columns with default value 0
for col in missing_cols:
    test_encoded[col] = 0

# Ensure the column order in test matches the train data
test_encoded = test_encoded[train_encoded.columns]


# In[32]:


test_encoded.columns


# ## **Builing The Model**

# In[34]:


#Model Fitting (Training the Logistic Regression Model)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#Ensure that train and test datasets have the same features
X_train = train_encoded.drop(columns=["Survived"])  # Training features
y_train = train_encoded["Survived"]  # Target variable

#Aligning test dataset with train dataset
missing_cols = set(X_train.columns) - set(test_encoded.columns)
for col in missing_cols:
    test_encoded[col] = 0  # Add missing columns with default value 0

X_test = test_encoded[X_train.columns]  

#Initializing and training the model at max iterations
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)


# ## **Evaluating The Model**

# In[36]:


#Importing the necessary metrics to evaluate
from sklearn.metrics import accuracy_score,roc_auc_score,classification_report

#Evaluating model on the training dataset
y_train_pred = model.predict(X_train)
y_train_probs = model.predict_proba(X_train)[:, 1]  #Probability estimates for positive class
print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
print("Training ROC-AUC Score:", roc_auc_score(y_train, y_train_probs))
print("Classification Report on Train Data:\n", classification_report(y_train, y_train_pred))

#Predicting on the test dataset 
test_predictions = model.predict(X_test)
test_probs = model.predict_proba(X_test)[:, 1]  #Probability estimates for positive class

#Saving predicted values in another file
test_encoded["Predicted_Survived"] = test_predictions
test_encoded.to_csv("test_predictions.csv", index=False)

print("Predictions saved to 'test_predictions.csv'")


# ### **Visualization of ROC-Curve**

# In[38]:


from sklearn.metrics import roc_curve,roc_auc_score

#Calculating ROC curve,since we already have predicted prob of y_train
fpr, tpr, thresholds = roc_curve(y_train, y_train_probs)

#Calculating AUC score
auc_score = roc_auc_score(y_train, y_train_probs)

#Plotting the ROC Curve
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color="skyblue", lw=2, label=f"ROC Curve (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")  
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve for Logistic Regression")
plt.legend(loc="lower right")
plt.grid()
plt.show()


# In[75]:


#import the statsmodels to interpret the feature and their significance with predicting target variable
import statsmodels.api as sm
sm.Logit(y_train,X_train).fit().summary()


# ### **Interpreting The Coefficients of Features**
# 
#  - A high positive coefficient means increase in chance of predicting survival rate
#  - Similarly high negative coefficient means decrease in chance of predicting survival rate
#  - In between(i.e, around 0.1 something) suggests weak impact to make any prediction
# 
#  > - PassengerId with 0.000084 has very small impact.it's most likely irrelevant since it just a unique identifier.
# 
#  > - Pclass (ticket class) coef is	-1.063975 some selected classes has a lower survival chance.
# 
#  > - Age(-0.038296) some elder passengers has a lower chance of survival.
# 
#  > - While passengers with siblings/spouses on board has coef of -0.312287	with decreased survival chance.
# 
#  >  - Likewise passengers with parents/children on board has coef of -0.081570	who has more family members aboard most likely have slightly decreased survival chance.
# 
#  > - Fare(0.002154) theres high chances that higher fares slightly increases survival probability (wealthier passengers).
# 
#  > - Sex_male with -2.619694coefficient indicates that males were much less likely to survive than females.
# 
#  > - Embarked_C(0.076031) & Embarked_Q has a slight positive effect on survival.while Embarked_S has negative effect.
# 
#  **since its logistic regression  we can further analyze which might change probablity in some variables.**

# ### **Significance of Feature in Predicting the Survival probability**
# 
#  - To keep it simple a few feature suggests that high significance inpredicting while some shows a moderate significance and the rest of them are irrelevent since significance of a feature is calculated with p_values,coefficients and etc.
#  - Just like coefficients the p_values shows the significance of feature in predicting survival probability.
#    A low p-value (< 0.05) suggests that the features are significant in predicting survival chances .
#    A high p-value (> 0.05) means the feature is not statistically significant and may not contribute much to the model.
#    
# **High-Significance**
#  > - Gender specific feature "sex_male" has increaced survival probablity for females .Chivalry a characteristics of men might had a special impact on the probability of females survival probability.
#  > - Pclass a feature depend on fare variable that suggests money puts impact on survival probability higher class(1st class) passengers has most likely to surivie
#  > - Likewise Age has high significance in survival probability.youngsters has high chances
#    
# **Moderate-Significance**
#  > - Fare feature has some impact on suvival probability ,high fares increases chances of survival suggesting that there are special previlages for those who pays high fare or it might be that there were special precations taken in case of emergency for high class passengers.
#  > - Features like sib/sp and par/ch has also have a moderate significance that shows passengers with family members have to struggle a little to survive
# 
# **Least Significance**
#  > - From the Embarked feature has some positive and some negative with few no significance while passengerid has no impact on survival probability.

# In[ ]:




