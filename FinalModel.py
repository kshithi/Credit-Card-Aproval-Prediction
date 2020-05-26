import  numpy as np
import pandas as pd 

################################################################STEP1:Load data set#################################################################

df = pd.read_csv('data.csv')
test_df = pd.read_csv('testdata.csv')#this is the test data set which havent 'A16' column(final class)


#################################################################STEP2:Handle missing values##############################################

df = df.replace(['?'],np.NaN)
test_df = test_df.replace(['?'],np.NaN)


# Impute the missing values with mean imputation.fillna only fills up the numeric columns
df.fillna(df.mean(), inplace=True) 
test_df.fillna(df.mean(), inplace=True) 

'''
There are still some missing values in non-numeric columns.Since they are non-numerical data ,mean imputation does not work here.
Therefore,impute these missing values with the most frequent values as present in the respective columns.
'''

# Iterate over each column of df
for col in df:
    # Check if the column is of object type
    if df[col].dtypes == 'object':
        # Impute with the most frequent value
        df = df.fillna(df[col].value_counts().index[0])

for col in test_df:
    # Check if the column is of object type
    if test_df[col].dtypes == 'object':
        # Impute with the most frequent value
        test_df = test_df.fillna(test_df[col].value_counts().index[0])
        


#########################################################STEP3:Converte all the non-numeric values into numeric ones###########################

#Instantiate LabelEncoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# Iterate over all the values of each column and extract their dtypes
for col in df:
    # Compare if the dtype is object
    if df[col].dtypes =='object' or df[col].dtypes =='bool':
    # Use LabelEncoder to do the numeric transformation
        df[col]=le.fit_transform(df[col])

for col in test_df:
    # Compare if the dtype is object
    if test_df[col].dtypes =='object' or test_df[col].dtypes =='bool':
    # Use LabelEncoder to do the numeric transformation
        test_df[col]=le.fit_transform(test_df[col])



#########################################################STEP4:Split data into train set and test data###############################

#convert the DataFrame to a NumPy array
df = df.values
test_df = test_df.values

# Segregate features and labels into separate variables
X,y = df[:,0:15] , df[:,15]


# Split into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.33, random_state=15)


###############################################################STEP5:Rescaling the values#################################################

# Instantiate MinMaxScaler and use it to rescale X_train and X_test
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.fit_transform(X_test)

rescaled_testdata = scaler.fit_transform(test_df)

###########################################################STEP6:Generalize Random Forest Model#########################################

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=2100)
model.fit(rescaledX_train, y_train)



# Use classifier to predict instances from the test set and store it
y_pred = model.predict(rescaledX_test)


# Get the accuracy score of Decision Tree model and print it
from sklearn.metrics import accuracy_score
print("Accuracy of RandomForestClassifier: ",accuracy_score(y_test,y_pred))

# Print the confusion matrix of the Decision Tree model
from sklearn import metrics 
from sklearn.metrics import confusion_matrix
print("Confusion matrix of RandomForestClassifier: ", confusion_matrix(y_test, y_pred))

#Perdict classes for test data set
testdata_predictions = model.predict(rescaled_testdata)
mydict ={0:'Failure',1:'Success'}
testdata_predictions = [mydict[i] for i in testdata_predictions]
print(testdata_predictions)

#If you want to print predictions into a file,

#testdata_predictions= pd.DataFrame(testdata_predictions, columns=['predictions']).to_csv('predictions.csv', index=False)


