import tensorflow
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

X = iris.data  # these are the values in each attribute
y = iris.target  # these are the numeric values representing the label

# training the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# the classifier to use
my_classifier = KNeighborsClassifier()

# we use these to train the classifier
# Essentially these X_train and y_train are just data. You use these to match the features to the label
my_classifier.fit(X_train, y_train)

# getting the predictions via the X_test data.
predictions = my_classifier.predict(X_test)

# comparing the predictions that we get from X_test and comparing it to the actual y_test value
print(accuracy_score(y_test, predictions))
