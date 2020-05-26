import  numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler



df = pd.read_csv('data.csv')

df = df.replace(['?'],np.NaN)

df.fillna(df.mean(), inplace=True) 

# Iterate over each column of df
for col in df:
    # Check if the column is of object type
    if df[col].dtypes == 'object':
        # Impute with the most frequent value
        df = df.fillna(df[col].value_counts().index[0])

#Instantiate LabelEncoder
le = LabelEncoder()

# Iterate over all the values of each column and extract their dtypes
for col in df:
    # Compare if the dtype is object
    if df[col].dtypes =='object' or df[col].dtypes =='bool':
    # Use LabelEncoder to do the numeric transformation
        df[col]=le.fit_transform(df[col])

#convert the DataFrame to a NumPy array
df = df.values

# Segregate features and labels into separate variables
X,y = df[:,0:15] , df[:,15]


# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.33, random_state=15)


# Instantiate MinMaxScaler and use it to rescale X_train and X_test
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.fit_transform(X_test)

model = RandomForestClassifier(random_state=2100)
model.fit(rescaledX_train, y_train)



#plot feature importance
std = np.std([tree.feature_importances_ for tree in model.estimators_],axis=0)
indices = np.argsort(model.feature_importances_)
plt.title("Feature importances")
plt.barh(range(X.shape[1]), model.feature_importances_[indices],color="r", xerr=std[indices], align="center")
plt.yticks(range(X.shape[1]),  {'A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13','A14','A15'})
plt.ylim([-1, X.shape[1]])
plt.show()
