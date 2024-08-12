# PlayPredictor by using KNN

import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

def PlayPredictor(datapath):

    # Step 1 : Load data
    data = pd.read_csv(datapath, index_col=0)
    print("Size of Actual dataset :",len(data))

    # step 2 : clean, prepare, manipulate data
    feature_names = ["Whether", "Temperature"]
    print("Names of features :", feature_names)

    label_names = ["Play"]
    print("Name of label :", label_names)

    whether = data.Whether
    Temperature = data.Temperature
    play = data.Play

    # creating labelEncoder
    le = preprocessing.LabelEncoder()

    # Converting strings features into numbers
    whether_encoded = le.fit_transform(whether)
    print("whether_encoded : ",whether_encoded)

    temp_encoded = le.fit_transform(Temperature)
    print("temp_encoded :",temp_encoded)

    # converting  string labels into numbers
    label = le.fit_transform(play)
    print("label_encoded :",label)

    # combining weather and temp into single list of tuples
    features = list(zip(whether_encoded, temp_encoded))

    # step 3 : Train data
    model = KNeighborsClassifier(n_neighbors=3)

    # train the model using training sets
    model.fit(features, label)

    # step 4 : test data
    predicted = model.predict([[0,2]]) # 0: overcast, 2:mild
    print("Prediction for (0: overcast, 2:mild) :",predicted)

    predicted = model.predict([[2,1]]) # 2: sunny, 1:hot
    print("Prediction for (2: sunny, 1:hot) :",predicted)


def main():
    print("----------- PlayPredictor Case Study by KNN -----------")

    PlayPredictor("PlayPredictor.csv")


if __name__ == "__main__":
    main()