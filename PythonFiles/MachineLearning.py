from sklearn import svm
import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from pyod.models import abod
from pyod.models import iforest
from pyod.models import knn
from itertools import combinations

#D = cardboard
#E = Plastic
#F = wood

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
        count1 = 0
        count2 = 0
        for row in range(len(classifiers)):
            subvotes.append(votes[row][col])
        for x in subvotes:
            if x == 1:
                count1 += 1
            else:
                count2 += 1
        if count1 > count2:
            majority.append(1)
        elif count2 > count1:
            majority.append(2)
        else:
            majority.append(0)

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
    return majority, confusionMatrix, misses, len(expectedValues)-misses


#removes outliers from dataset using specified outlier detector
def removeOutliers(data, detector):
    outliers = 0
    dataWOoutliers = []
    X = np.array(data)
    detector.fit(X)
    predictions = detector.predict(X)
    for x in predictions:
        if x == 1:
            outliers += 1

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
        for c in classes:
            if c == predictions[i]:
                confusionMatrix[c][c] += 1
    return confusionMatrix, misses


#machine learning parameters and classifiers
high = 1
low = -1
scale = True
svm1 = svm.SVC(kernel="rbf", decision_function_shape='ovr', gamma="scale", C=100) #creating svm object
rf = RandomForestClassifier(n_estimators=10, random_state=10) #creating rf object
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 3), random_state=1) #creating mlp object
clfs = [rf,svm1,mlp]
clf = svm1

#outlier detectors
angleBased = abod.ABOD(method="fast")
isolationForrest = iforest.IForest(n_estimators=100, behaviour="new")
kNearestNeighbors = knn.KNN(method="median")
detector = kNearestNeighbors

#datasets
cardboard = parseCsv("C:\\Users\\Luke\\Documents\\GitHub\\UREP_Cancer_Detection_Array_Microwave_Sensor\\results\\Deltas\\DDeltas.csv")
wood = parseCsv("C:\\Users\\Luke\\Documents\\GitHub\\UREP_Cancer_Detection_Array_Microwave_Sensor\\results\\Deltas\\EDeltas.csv")
plastic = parseCsv("C:\\Users\\Luke\\Documents\\GitHub\\UREP_Cancer_Detection_Array_Microwave_Sensor\\results\\Deltas\\BDeltas.csv")
plastic = plastic+parseCsv("C:\\Users\\Luke\\Documents\\GitHub\\UREP_Cancer_Detection_Array_Microwave_Sensor\\results\\Deltas\\FDeltas.csv")

#standard expected values
expected = [[1 for x in range(50)],[2 for x in range(50)]]

#sensor combo datasets row = rValue col = dataset
#number of datasets for each rValue
# 1 -> 5
# 2 -> 10
# 3 -> 10
# 4 -> 5
cardboardCombos = []
woodCombos = []
plasticCombos = []

for i in range(1,len(cardboard[0])):
    cardboardCombos.append(getCombos(cardboard,i))
    woodCombos.append(getCombos(wood,i))
    plasticCombos.append(getCombos(plastic,i))

#removing outliers
cardboard = removeOutliers(cardboard, detector)
wood = removeOutliers(wood, detector)
plastic = removeOutliers(plastic, detector)


#creating expected values
cardboardEV = [[0 for i in range(len(cardboard))],[1 for i in range(len(cardboard))]]
woodEV = [[ 0for i in range(len(wood))],[1 for i in range(len(wood))]]
plasticEV = [[0 for i in range(len(plastic))],[1 for i in range(len(plastic))]]

'''
#combo predictions E vs F
maxI = 0
maxJ = 0
maximum = 0
for i in range(len(cardboardCombos)):
    for j in range(len(cardboardCombos[i])):
        print(str(i) + str(j))
        c = majorityVote(woodCombos[i][j]+plasticCombos[i][j],expected[0]+expected[1],clfs,True,low,high,True)[2]
        if(c > maximum):
            maximum = c
            maxI = i
            maxJ = j
print("Best results: i=" + str(maxI) + " j=" + str(maxJ) + " max=" + str(maximum))


#combo predictions D vs Rest
maxI = 0
maxJ = 0
maximum = 0
for i in range(len(cardboardCombos)):
    for j in range(len(cardboardCombos[i])):
        print(str(i) + str(j))
        c = loocv(cardboardCombos[i][j]+woodCombos[i][j]+plasticCombos[i][j],expected[0]+expected[1]+expected[1],clf,True,low,high,True)[2]
        if(c > maximum):
            maximum = c
            maxI = i
            maxJ = j
print("Best results: i=" + str(maxI) + " j=" + str(maxJ) + " max=" + str(maximum))


#Majority vote predictions
print("D vs Rest")
majorityVote(cardboard+wood+plastic, cardboardEV[0]+plasticEV[1]+woodEV[1], clfs, scale, low, high)


print("E vs F")
majorityVote(wood+plastic,woodEV[0]+plasticEV[1],clfs,scale,low,high)

'''

#single model predictions
#D vs Rest
print("Classifying D")
loocv(cardboard+wood+plastic, cardboardEV[0]+plasticEV[1]+woodEV[1], clf, scale, low, high, True)


#E vs F
print("Classifying E vs F")
loocv(wood+plastic, woodEV[0]+plasticEV[1], clf, scale, low, high, True)










