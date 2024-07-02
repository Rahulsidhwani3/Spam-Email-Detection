import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
data = pd.read_csv('spam.csv', encoding='latin-1')
data = data[['Category', 'Message']]
data.columns = ['label', 'message']

# Creating a dummy 'category' feature for illustration purposes
# In practice, this should be replaced with actual category data
data['category'] = data['label'].apply(lambda x: 'email' if x == 'ham' else 'advertisement')

# Data preprocessing
data['label'] = data['label'].map({'ham': 0, 'spam': 1})
X_message = data['message']
X_category = data['category']
y = data['label']

# Feature extraction
vectorizer_message = CountVectorizer()
vectorizer_category = CountVectorizer()

X_message = vectorizer_message.fit_transform(X_message)
X_category = vectorizer_category.fit_transform(X_category)

# Combine the features
import scipy.sparse as sp
X = sp.hstack((X_message, X_category))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = MultinomialNB()
model.fit(X_train, y_train)

# Model prediction
y_pred = model.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')
