import tensorflow
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.spatial import distance
import random


def euc(a, b):
    return distance.euclidean(a, b)


# Learning how to make your own KNeighbourClassifier(basic version)
# Pros
# 1) relatively simple to implement
# Cons
# 1) Even number will not work
# 2) Odd number will have a higher priority of working, however in encountering things like 1 3 3, it is still unable
# to make a decision
# 3) Computationally intensive as for each data point, we have to compared to all N other data points to find K in total
# 4) Hard to represent relationships between features
class ScrappyKNN():
    def fit(self, X_train, y_train):
        # this is like storing the data
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self, row):
        best_dist = euc(row, self.X_train[0])  # save the best distance recorded so far
        best_index = 0  # save the best index
        for i in range(len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]


iris = datasets.load_iris()

X = iris.data  # these are the values in each attribute
y = iris.target  # these are the numeric values representing the label

# training the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# the classifier to use
# my_classifier = KNeighborsClassifier()
my_classifier = ScrappyKNN()

# we use these to train the classifier
# Essentially these X_train and y_train are just data. You use these to match the features to the label
my_classifier.fit(X_train, y_train)

# getting the predictions via the X_test data.
predictions = my_classifier.predict(X_test)

# comparing the predictions that we get from X_test and comparing it to the actual y_test value
print(accuracy_score(y_test, predictions))
