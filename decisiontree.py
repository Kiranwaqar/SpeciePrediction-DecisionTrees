import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score,f1_score,classification_report

# Load the dataset
df = pd.read_csv('iris.csv')

# Split the dataset into features and target variable
X = df.drop('species', axis=1)
y = df['species']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree with pruning
clf = DecisionTreeClassifier(max_depth=3,random_state=42)

# Fit the model on the training data
clf.fit(X_train, y_train)

# Make prediction
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Accuracy: {accuracy:.2f}')
print(f'F1 Score: {f1:.2f}')
print("\nClassification Report:\n",classification_report(y_test,y_pred))

# Visualize the decision tree
plt.figure(figsize=(8,8))
plot_tree(clf,feature_names=X.columns,class_names=clf.classes_,filled=True)
plt.title("Decision Tree - Iris Dataset")
plt.show()