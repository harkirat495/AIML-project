import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
# train_data = pd.read_csv("C:/Users/Admin/Downloads/train.csv")
# test_data = pd.read_csv("C:/Users/Admin/Downloads/test.csv")
# gender_submission_data = pd.read_csv("C:/Users/Admin/Downloads/gender_submission.csv")
print(train_data.head())

print(train_data.isnull().sum()) #

sns.countplot(x='Survived', data=train_data)
plt.show()

sns.countplot(x='Survived', hue='Sex', data=train_data)
plt.show()

sns.countplot(x='Survived', hue='Pclass', data=train_data)
plt.show()

sns.countplot(x='Survived', hue='Embarked', data=train_data)
plt.show()

train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True) #handle missing values

train_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})
train_data['Embarked'] = train_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

X = train_data.drop('Survived', axis=1)
y = train_data['Survived']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print("Shape of X_train_scaled:", X_train_scaled.shape)
print("Shape of X_val_scaled:", X_val_scaled.shape)

svm_model = SVC()
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_val_scaled)
accuracy_svm = accuracy_score(y_val, y_pred_svm)
print("SVM Accuracy:", accuracy_svm)

logistic_model = LogisticRegression()
logistic_model.fit(X_train_scaled, y_train)
y_pred_lr = logistic_model.predict(X_val_scaled)
accuracy_lr = accuracy_score(y_val, y_pred_lr)
print("Logistic Regression Accuracy:", accuracy_lr)

tree_model = DecisionTreeClassifier()
tree_model.fit(X_train_scaled, y_train)
y_pred_tree = tree_model.predict(X_val_scaled)
accuracy_tree = accuracy_score(y_val, y_pred_tree)
print("Decision Tree Accuracy:", accuracy_tree)