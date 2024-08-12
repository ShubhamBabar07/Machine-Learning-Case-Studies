# Iris Classification Case Study by comparing KNN Algorithm & Decision Tree Algorithm

from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def AccuracyDecisionTree():
    iris = load_iris()

    data = iris.data
    target = iris.target

    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.5)

    classifier = tree.DecisionTreeClassifier()

    classifier.fit(data_train, target_train)

    predictons = classifier.predict(data_test)

    accuracy = accuracy_score(target_test, predictons)

    return accuracy


def AccuracyKNeighbor():
    iris = load_iris()

    data = iris.data
    target = iris.target

    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.5)

    classifier = KNeighborsClassifier()

    classifier.fit(data_train, target_train)

    predictons = classifier.predict(data_test)

    accuracy = accuracy_score(target_test, predictons)

    return accuracy


def main():

    print("---------------- Iris Classification Case Study ----------------")

    Accuracy = AccuracyDecisionTree()
    print("Accuracy of Decision tree Classifier is : ",Accuracy*100,"%")

    Accuracy = AccuracyKNeighbor()
    print("Accuracy of K Neighbor Classifier is : ",Accuracy*100,"%")


if __name__ == "__main__":
    main()

