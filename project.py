import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, log_loss, hinge_loss
import re

data = pd.read_csv('emails.csv', delimiter=';')
data.drop(data.columns[[2,3,4]], axis=1, inplace=True)
data.rename(columns={'v1': 'Value', 'v2': 'Content'}, inplace=True)

# Array of values ('spam' / 'ham') - labels
values = data['Value']

# Making the labels easier to process with 1 (spam) and 0 (ham) values
binValues = [1 if value == 'spam' else 0 for value in values]

# Array of the emails
emails = data['Content']

# Feature Engineering
# Calculate the count of special characters for each email
data['special_character_count'] = emails.apply(lambda x: sum(c.isalnum() is False for c in x))

# Calculate the count of uppercase letters for each email
data['uppercase_letter_count'] = emails.apply(lambda x: sum(c.isupper() for c in x))

# Define a list of typical spam-related keywords
spam_keywords = ['free', 'win', 'money', 'lottery', 'click', 'discount', 'guarantee', 'prize', 'download',
                'available now', 'sign up', 'order now', 'offer', 'thousands', 'password']

# Creating a new feature indicating the count of spam-related keywords in each email (initializing with 0)
data['spam_keyword_count'] = 0

for keyword in spam_keywords:
    data['spam_keyword_count'] += emails.str.count(keyword, flags=re.IGNORECASE)
 
    
# Combine the special character count, uppercase letter count, and spam keyword count as features
X = data[['special_character_count', 'uppercase_letter_count', 'spam_keyword_count']]
y = binValues

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# First model (Logistic Regression as ML method)
# Initialize and train a Logistic Regression classifier
clf_log = LogisticRegression()
clf_log.fit(X_train, y_train)

# Make predictions on the training and validation sets using Logistic Regression
y_train_pred_log = clf_log.predict(X_train)
y_val_pred_log = clf_log.predict(X_val)

# Predicted probabilities for logistic loss
y_train_pred_prob_log = clf_log.predict_proba(X_train)
y_val_pred_prob_log = clf_log.predict_proba(X_val)

# Calculate the accuracy score as well as training and validation errors using log loss
accuracy_score_log = accuracy_score(y_val, y_val_pred_log)
train_error_log = log_loss(y_train, y_train_pred_prob_log)
val_error_log = log_loss(y_val, y_val_pred_prob_log)

print("Logistic Regression:")
print(f"Accuracy Score: {accuracy_score_log:.4f}")
print(f"Training Error: {train_error_log:.4f}")
print(f"Validation Error: {val_error_log:.4f}")
print("\n")

# Second model (Support Vector Machine - SVM as ML method)
# Initialize and train an SVM classifier
clf_svm = SVC()
clf_svm.fit(X_train, y_train)

# Make predictions on the training and validation sets
y_train_pred_svm = clf_svm.predict(X_train)
y_val_pred_svm = clf_svm.predict(X_val)

# Calculate the accuracy score as well as training and validation errors using hinge loss
accuracy_score_svm = accuracy_score(y_val, y_val_pred_svm)
train_error_svm = hinge_loss(y_train, clf_svm.decision_function(X_train))
val_error_svm = hinge_loss(y_val, clf_svm.decision_function(X_val))

print("SVC:")
print(f"Accuracy Score: {accuracy_score_svm:.4f}")
print(f"Training Error: {train_error_svm:.4f}")
print(f"Validation Error: {val_error_svm:.4f}")