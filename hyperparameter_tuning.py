import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder #for encode categorical variables 
from imblearn.over_sampling import SMOTE #for oversampling means synthetic technique where we can solve the class imbalance issue in target column to balance the dataset
from sklearn.model_selection import train_test_split, cross_val_score #for train and test split of entire data for cross validation score accuracy of model
from sklearn.tree import DecisionTreeClassifier #this is tree based model it has robustness para d na mag standardize sa data 
from sklearn.ensemble import RandomForestClassifier #model
from xgboost import XGBClassifier #model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV #hyperparameter tuning with gridsearchcv for randomforest
import pickle #to save some files and use it later 

################################################# 
#For data loading and exploration
df = pd.read_csv("../data/churn.csv")
df.shape #for number of rows and columns
df.head() #DIsplayfirst 5 rows
df.info() #checking data types and missing values
df.columns #checking the columns


#dropping customerID as this not required for modeling
df = df.drop(columns=["customerID"])
df.head(2)

#numerical features list
numerical_features_list = ['tenure', 'MonthlyCharges', 'TotalCharges']

for col in df.columns:#for unique values checking if this categorical or numerical values
 if col not in numerical_features_list: #not displaying numerical features
    print(col, df[col].unique()) #for unique values checking if this categorical or numerical values 
    print('-'*50)#for separating the results of unique values 

print(df.isnull().sum())


#Missing Values
len(df[df["TotalCharges"] == " "]) #checking the missing values in TotalCharges column or filtering the dataframe 
df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0"})#replacing empty values with 0
df["TotalCharges"] = df["TotalCharges"].astype(float) #converting to float datatypes 


#checking the class distribution of target variable
df['Churn'].value_counts() #we can see there's an imbalance of this dataset no= 5174, yes= 1869 because of that we cannot directly train our model we need to perform upper sampling or down sampling either to increase minority class or decrease majority class

#INSIGTS
#1. REMOVING CUSTOMER ID AS IT IS NOT REQUIRED FOR MODELLING
#2. NO MISSING VALUES IN THE DATASET
#3. MISSING VALUES IN TOTAL CHARGES COLUMN WERE REPLACED WITH 0 
#4. CLASS IMBALANCE IDENTIFIED ON TARGET COLUMN WHICH IS THE CHURN
#################################################################

#EXPLORATORY DATA ANALYSIS 
df.describe()

#1. Numerical analysis - Understand the distribution of numerical features
def plot_histogram(df, column_name): #using function  
  plt.figure(figsize = (5,3))
  sns.histplot(df[column_name], kde=True)
  plt.title(f"Distribution of {column_name}")

  #understand the mean and median values for columns 
  col_mean = df[column_name].mean()
  col_median = df[column_name].median()

  #add vertical lines for mean and median values 
  plt.axvline(col_mean, color='red', linestyle='--', linewidth=1, label='Mean')
  plt.axvline(col_median, color='blue', linestyle='-', linewidth=1, label= 'Median')

  plt.legend()
  plt.show()

plot_histogram(df, "tenure") #tenure distribution
plot_histogram(df, "MonthlyCharges") #monthly charges distribution
plot_histogram(df, "TotalCharges") #total charges distribution



#boxplot use to identify outliers
#boxplot for numerical features
def plot_boxplot(df, column_name):
    plt.figure(figsize = (5,3))
    sns.boxplot(y=df[column_name])
    plt.title(f"Boxplot of {column_name}")
    plt.ylabel(column_name)
    plt.show()
plot_boxplot(df, "tenure") #tenure distribution
plot_boxplot(df, "MonthlyCharges") #monthly charges distribution
plot_boxplot(df, "TotalCharges") #total charges distribution


# Correlation heatmap for each numerical column
plt.figure(figsize=(8, 6))
sns.heatmap(df[["tenure", "MonthlyCharges", "TotalCharges"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.yticks(rotation=0)  # Adjust rotation as needed
plt.show()

#CATEGORICAL ANALYSIS
#Countplot for categorical features
object_cols = df.select_dtypes(include='object').columns.to_list()
object_cols = ['SeniorCitizen'] + object_cols
for col in object_cols:
   plt.figure(figsize = (5,3))
   sns.countplot(x=df[col])
   plt.title(f"Countplot of {col}")
   plt.show()


#######Data Preprocessing
##Label encoding of target column
df["Churn"] = df["Churn"].replace({"No": 0, "Yes": 1})
df.head(2)
print(df['Churn'].value_counts())

##Label Encoding of categorical features
#identify columns with object datatype
object_columns = df.select_dtypes(include='object')
print(object_columns)

#initialize a dictionary to save the encoders 
encoders ={}

#apply label encoding and store the encoders
for column in object_columns:
   label_encoder = LabelEncoder()
   df[column] = label_encoder.fit_transform(df[column])
   encoders[column] = label_encoder

#save the encoders to a pickle file
with open("encoder.pkl", "wb") as f:
   pickle.dump(encoders, f)

encoders
df.head(2)


#Training and testing data split
#splitting features and target
x = df.drop(columns = ['Churn']) #features
y = df['Churn'] #target 
print(x)

#split training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)
#all train data features will go here x_train
#all test data features will go here x_test
#all train data target will go here y_train
#all test data target will go here y_test
#using train_test_split function to split the data into train and test sets using the x, y variables
#test_size = 0.2 means 20% of the data will be used for testing and the other percent the 80% will be used for training
#random_state = 42 means that the split will be the same every time the code is run meaning for reproducibility if you change it, it will be different every time
print(y_train.shape) #5634 the remaining 1366 are for testing
print(y_train.value_counts())#there's an imbalance of data 


#Synthetic Minority OverSampling Technique (SMOTE)
smote = SMOTE(random_state = 42)
x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)#this smote algorithm x_train, y_train identify the minority class and oversample it to balance the dataset

print(y_train_smote.shape)#so gidungangan ang minority class para mabalance ang dataset
print(y_train_smote.value_counts()) #now we have a balanced dataset

######Model Training
#Training with default hyperparameters
#dictionary of models
models ={
   "Decision Tree" : DecisionTreeClassifier(random_state=42),
   "Random Forest" : RandomForestClassifier(random_state=42),
   "XGBoost" : XGBClassifier(random_state=42)
} #need also cross validation because it give you more rounded results instead of just giving accuracy one fold, it's going to give accuracy like multiple folds

#dictionary to store the cross validation results 
cv_scores = {}

#perform 5-fold cross validation for each model
for model_name, model in models.items():#if you run this model_name it takes the key means Decision Tree and other two model next variable model one is DecisionTreeClassifier and the other model with classifier
   print(f"Training {model_name} with default parameters")
   scores = cross_val_score(model, x_train_smote, y_train_smote, cv=5, scoring="accuracy") #this is where cross validation happens 
   cv_scores[model_name] = scores
   print(f"{model_name} cross-validation accuracy: {np.mean(scores):.2f}")
   print("-"*50)

#Random forest gives the highest accuracy compared to other models with default parameters
param_grid ={
   'n_estimators':[100, 200, 300],
   'max_depth':[None, 10, 20],
   'min_samples_split':[2,5,10]
}

grid_search = GridSearchCV(
   RandomForestClassifier(random_state=42),
   param_grid,
   cv=5,
   scoring='accuracy',
   n_jobs=-1
)

grid_search.fit(x_train_smote, y_train_smote)

print("Best parameters:", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)

#use the best model to found by GridSearchCV
best_rfc = grid_search.best_estimator_

#train best model on training data
best_rfc.fit(x_train_smote, y_train_smote)

###Model Evaluation
#Evaluate test data
y_test_pred = best_rfc.predict(x_test)
print("Accuracy Score:\n", accuracy_score(y_test, y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print("Classification Report:\n", classification_report(y_test, y_test_pred))


#save the trained model as a pickle file
model_data = {"model": best_rfc, "features_name": x.columns.tolist(),}
with open("customer_churn_model.pkl", "wb") as f:
   pickle.dump(model_data, f)

#####Load the save model and build a predictive system
#load the save model and features name 

with open("customer_churn_model.pkl", "rb") as f:
   model_data = pickle.load(f)

   loaded_model = model_data["model"]
   feature_names = model_data["features_name"]

print(loaded_model)
print(feature_names)

input_data = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 1,
    'PhoneService': 'No',
    'MultipleLines': 'No phone service',
    'InternetService': 'DSL',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 29.85,
    'TotalCharges': 29.85
}

input_data_df = pd.DataFrame([input_data])
with open("encoder.pkl", "rb") as f:
   encoders = pickle.load(f)
print(input_data_df.head())

#encode categorical features using the saved encoders
for column, encoder in encoders.items():
   input_data_df[column] = encoder.transform(input_data_df[column])

#make a prediction
model_prediction = loaded_model.predict(input_data_df)
pred_prob = loaded_model.predict_proba(input_data_df)
print(model_prediction)

#results
print(f"Prediction: {'Churn' if model_prediction[0] == 1 else 'No Churn'}")
print(f"Prediction Probability: {pred_prob}")



#To do for improvements:
#1. Implement Hyperparameter Tuning 
#2. Try Model Selection
#3. Try Downsampling
#4. Try to address overfitting
#5. Try Stratified K-Fold Cross Validation