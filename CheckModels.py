import  numpy as np
import pandas as pd 

################################################################STEP1:Load data set#################################################################

df = pd.read_csv('data.csv')

################################################################STEP2.Check the Class Distribution#########################################

import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x="A16", data=df, palette="Greens_d")
plt.title("Class Distribution")


#################################################################STEP3:Handle missing values##############################################
'''
print (df.info())
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 552 entries, 0 to 551
Data columns (total 16 columns):
 #   Column  Non-Null Count  Dtype
---  ------  --------------  -----
 0   A1      552 non-null    object
 1   A2      552 non-null    object
 2   A3      552 non-null    object
 3   A4      552 non-null    object
 4   A5      552 non-null    float64
 5   A6      552 non-null    object
 6   A7      552 non-null    int64
 7   A8      552 non-null    bool
 8   A9      552 non-null    object
 9   A10     552 non-null    float64
 10  A11     552 non-null    bool
 11  A12     552 non-null    int64
 12  A13     552 non-null    bool
 13  A14     552 non-null    object
 14  A15     552 non-null    object
 15  A16     552 non-null    object
dtypes: bool(3), float64(2), int64(2), object(9)

coloumn A5,A7,A10,A12 contain numeric values.Others are non-numeric values
'''

# Replace the '?'s with NaN
df = df.replace(['?'],np.NaN)

'''
Count the number of NaNs in the dataset
print(df.isnull().sum())
print(df.isnull().values.sum())

A1      8
A2     10
A3      4
A4      4
A5      0
A6      6
A7      0
A8      0
A9      6
A10     0
A11     0
A12     0
A13     0
A14    10
A15     0
A16     0
dtype: int64
48

'''
# Impute the missing values with mean imputation.fillna only fills up the numeric columns
df.fillna(df.mean(), inplace=True) 

'''
print(df.isnull().sum())
print(df.isnull().values.sum())

A1      8
A2     10
A3      4
A4      4
A5      0
A6      6
A7      0
A8      0
A9      6
A10     0
A11     0
A12     0
A13     0
A14    10
A15     0
A16     0
dtype: int64
48
There are still some missing values in non-numeric columns.Since they are non-numerical data ,mean imputation does not work here.
Therefore,impute these missing values with the most frequent values as present in the respective columns.
'''

# Iterate over each column of df
for col in df:
    # Check if the column is of object type
    if df[col].dtypes == 'object':
        # Impute with the most frequent value
        df = df.fillna(df[col].value_counts().index[0])

'''
print(df.isnull().sum())
print(df.isnull().values.sum())
A1     0
A2     0
A3     0
A4     0
A5     0
A6     0
A7     0
A8     0
A9     0
A10    0
A11    0
A12    0
A13    0
A14    0
A15    0
A16    0
dtype: int64
0
The missing values are now successfully handled.
'''

#################################################################STEP4:Converte all the non-numeric values into numeric ones###########################
#Instantiate LabelEncoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# Iterate over all the values of each column and extract their dtypes
for col in df:
    # Compare if the dtype is object
    if df[col].dtypes =='object' or df[col].dtypes =='bool':
    # Use LabelEncoder to do the numeric transformation
        df[col]=le.fit_transform(df[col])
'''
print(df.head())
 A1   A2  A3  A4     A5  A6    A7  A8  A9   A10  A11  A12  A13  A14  A15  A16
0   1  144   2   1   0.00  13     0   1   8  1.25    1    1    0   42    0    1
1   0  295   2   1   4.46  11   560   1   4  3.04    1    6    0  108    0    1
2   0   82   2   1   0.50  11   824   0   4  1.50    1    0    0   71    0    1
3   1  116   2   1   1.54  13     3   1   8  3.75    1    5    1    1    0    1
4   1   87   2   1  11.25   2  1208   1   8  2.50    1   17    0   40    0    1
All non-numerical values are converted into numerical values
'''

###############################################################STEP5:Split data into train set and test data###############################


#convert the DataFrame to a NumPy array
df = df.values


# Segregate features and labels into separate variables
X,y = df[:,0:15] , df[:,15]


from sklearn.model_selection import train_test_split
# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.33, random_state=15)


###############################################################STEP5:Rescaling the values#################################################
from sklearn.preprocessing import MinMaxScaler
# Instantiate MinMaxScaler and use it to rescale X_train and X_test
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.fit_transform(X_test)


###########################################################STEP6:Use various classification models and check their accuracy##########################
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GaussianNB(),
    LogisticRegression()]

log_cols=["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)

for clf in classifiers:
    clf.fit(rescaledX_train, y_train)
    name = clf.__class__.__name__
    
    print("="*30)
    print(name)
    
    print('****Results****')
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print("Accuracy: {:.4%}".format(acc))
    

    
print("="*30)


plt.show()