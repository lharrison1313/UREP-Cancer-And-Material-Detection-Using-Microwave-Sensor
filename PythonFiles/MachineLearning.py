import csv
import numpy as np
from sklearn import preprocessing
from itertools import combinations

#TODO confusion matrix plotting
#TODO try removing different combinations of features
#TODO only focus on D vs Rest, and E vs F
#TODO do majority votes give best cases or single classifier

#leave one out cross validation
def loocv(dataset,expectedValues,classifier,scale,minimum,maximum,printResults):
    X = np.array(dataset)
    y = np.array(expectedValues)
    clf = classifier
    results = []
    classes = createClassList(expectedValues)
    misses = 0

    #scalling data
    if(scale == True):
        min_max_scaler = preprocessing.MinMaxScaler(feature_range =(minimum,maximum))
        X = min_max_scaler.fit_transform(X)

    #performing loocv
    for x in range(len(X)):
        A = np.delete(X,x,0)
        b = np.delete(y,x)
        clf.fit(A,b)
        prediction = clf.predict([X[x]])
        results.append(prediction[0])

    #building confusion matrix
    confusionMatrix, misses = buildConfusionMatrix(classes,results,expectedValues)


    if(printResults):
        print("********************************************************************")
        print("expected: " + str(expectedValues))
        print("outcome: " + str(results))
        print("number of correct classifications " + str(len(X)-misses) + "/" + str(len(X)))
        print("Confusion Matrix: ")
        print(confusionMatrix[0])
        print(confusionMatrix[1])

    return results, confusionMatrix, misses, len(expectedValues)-misses

def majorityVote(dataset,expectedValues,classifiers,scale,minimum,maximum,printResults):
    classes = createClassList(expectedValues)

    #classifiers voting
    votes = []
    for c in classifiers:
        votes.append(loocv(dataset,expectedValues,c,scale,minimum,maximum,False)[0])


    #tallying votes
    subvotes = []
    majority = []
    for col in range(len(votes[0])):
        subvotes.clear()
        count0 = 0
        count1 = 0
        for row in range(len(classifiers)):
            subvotes.append(votes[row][col])
        for x in subvotes:
            if x == 0:
                count0 += 1
            else:
                count1 += 1
        if count0 > count1:
            majority.append(0)
        else:
            majority.append(1)


    #building confusion matrix
    confusionMatrix,misses = buildConfusionMatrix(classes,majority,expectedValues)

    #printing results
    if(printResults == True):
        print("********************************************************************")
        for x in range(len(classifiers)):
            print("classifier " + str(x) + " votes:" + str(votes[x]))
        print("Majority votes: " + str(majority))
        print("Expected: " + str(expectedValues))
        print("Correctly Classified: " + str(len(expectedValues)-misses) + "/" + str(len(expectedValues)))
        print("Confusion Matrix: ")
        print(confusionMatrix[0])
        print(confusionMatrix[1])

    return majority, confusionMatrix, misses, len(expectedValues)-misses



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

def createClassList(expectedValues):
    classes = []
    # creating list of classes
    for e in expectedValues:
        inlist = False
        for c in classes:
            if c == e:
                inlist = True
        if not inlist:
            classes.append(e)
    return classes

def buildConfusionMatrix(classes, predictions,actual):
    confusionMatrix = [[0 for x in range(len(classes))] for x in range(len(classes))]
    misses = 0
    for i in range(len(predictions)):
        if(predictions[i] != actual[i]):
            for c in classes:
                if c == predictions[i]:
                    confusionMatrix[actual[i]][c] += 1
            misses+=1
        else:
            for c in classes:
                if c == predictions[i]:
                    confusionMatrix[c][c] += 1
    return confusionMatrix, misses









