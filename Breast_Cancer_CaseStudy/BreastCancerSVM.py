# BreastCancer case study by Support Vector Machine Algorithm

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

def BreastCancerSVM():

    # Load data
    cancer = datasets.load_breast_cancer()

    # print the names of the 13 features
    print("Features :", cancer.feature_names)

    # print the label type of cancer('malignant', 'benign')
    print("Labels :", cancer.target_names)

    # print data(feature)shape
    print("Shape of dataset is :", cancer.data.shape)

    # print the cancer data features (top 5 records)
    print("First 5 records are :")
    print(cancer.data[0:5])

    # print the cancer labels (0: malignant, 1: benign)
    print("Target of dataset :",cancer.target[0:5])

    # split dataset into training set and test test
    X_train, X_test, Y_train, Y_test = train_test_split(cancer.data, cancer.target, test_size=0.3, random_state=109) # 70 % training & 30 % test

    # create a SVM classifier
    clf = svm.SVC(kernel='linear') # Linear kernel 

    # train model
    clf.fit(X_train, Y_train)

    # predict the response
    y_pred = clf.predict(X_test)

    # Model Accuracy : how often the classfier correct ?
    print("Accuracy :", metrics.accuracy_score(Y_test, y_pred)*100)


def main():

    print("----------- BreastCancer case study by Support Vector Machine -----------")

    BreastCancerSVM()


if __name__ == "__main__":
    main()
