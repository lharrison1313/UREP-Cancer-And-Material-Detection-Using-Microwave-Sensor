import csv
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.metrics import confusion_matrix
from itertools import combinations


#TODO confusion matrix plotting
#TODO try removing different combinations of features
#TODO only focus on D vs Rest, and E vs F
#TODO do majority votes give best cases or single classifier


def crossValidate(dataset,expectedValues, classifier, iterator, scale, minimum, maximum, printResults):
    X = np.array(dataset)
    y = np.array(expectedValues)

    # scalling data
    if (scale == True):
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(minimum, maximum))
        X = min_max_scaler.fit_transform(X)

    #getting scores and predictions
    scores = cross_val_score(classifier,X,y,cv = iterator)
    predictions = cross_val_predict(classifier,X,y,cv = iterator)

    #getting number of correct classifications
    correct = 0
    for x,y in zip(predictions,expectedValues):
        if x == y:
            correct += 1

    if(printResults):
        #printing results
        print("Correct Classifications: " + str(correct) + "/" + str(len(expectedValues)))
        print("Percentage Correctly Classified: " + str(scores.mean()))

        #printing confusion matrix
        print(confusion_matrix(expectedValues,predictions))

    return scores,predictions,correct


#removes outliers from dataset using specified outlier detector
def removeOutliers(data, detector, printResults):
    outliers = 0
    dataWOoutliers = []
    X = np.array(data)
    detector.fit(X)
    predictions = detector.predict(X)
    for x in predictions:
        if x == 1:
            outliers += 1

    if printResults:
        print("number of outliers: " + str(outliers))
        print(predictions)

    for i in range(len(predictions)):
        if predictions[i] == 0:
            dataWOoutliers.append(data[i])
    return dataWOoutliers


#parses a csv file into a dataset
def parseCsv(csvFile):
    # initializing 2d array
    data = []

    # parsing csv into 2d array
    keys = ["0", "1", "2", "3", "4","5"]
    with open(csvFile) as csvfile:
        reader = csv.DictReader(csvfile, keys)
        for row in reader:
            data.append([row["1"],row["2"],row["3"],row["4"],row["5"]])

    # removing header info from datalist
    data = data[1:]

    # converting datalist values from strings to floats
    rows = 0
    for x in data:
        data[rows][0] = float(x[0])
        data[rows][1] = float(x[1])
        data[rows][2] = float(x[2])
        data[rows][3] = float(x[3])
        data[rows][4] = float(x[4])
        rows += 1
    return(data)

#returns an array of sensor combinations for a given dataset
def getCombos(dataset,r):
    comboSet = []
    for row in dataset:
        comboSet.append(list(combinations(row,r)))

    outputSet = [[] for i in range(len(comboSet[0]))]
    for i in range(len(comboSet)):
        for j in range(len(comboSet[0])):
            outputSet[j].append(comboSet[i][j])

    return outputSet

def getBestCombination(datasetArray, expectedValues, classifier, iterator, scale, minimum, maximum, printResults):
    maxI = 0
    maxJ = 0
    maxC = 0
    dataset = []

    for i in range(len(datasetArray[0])):
        for j in range(len(datasetArray[0][i])):
            if printResults:
                print("combo set: " + str(i) + str(j))

            # creating data set for current combination
            dataset.clear()
            for set in datasetArray:
                dataset += set[i][j]

            # using majority vote or loocv based on input parameters
            c = crossValidate(dataset, expectedValues, classifier, iterator, scale, minimum, maximum, printResults)[2]

            # checking if new maximum
            if c > maxC:
                maxC = c
                maxI = i
                maxJ = j

    # creating set with best prediction results
    bestSet = []
    for set in datasetArray:
        bestSet += set[i][j]

    # printing results
    print("Best results: i=" + str(maxI) + " j=" + str(maxJ) + " correct classifications=" + str(maxC) + "/" + str(len(expectedValues)))
    return bestSet









