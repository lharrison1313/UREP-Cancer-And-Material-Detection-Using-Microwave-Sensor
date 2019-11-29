from sklearn import svm
import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from pyod.models import abod
from pyod.models import iforest
from pyod.models import knn

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
    misses = 0
    miss1 = 0
    miss2 = 0
    miss3 = 0

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
        if(prediction[0]!=y[x]):
            if(y[x] == 1):
                miss1 += 1
            elif(y[x] == 2):
                miss2 +=1
            else:
                miss3 +=1
            misses+=1
        results.append(prediction[0])

    if(printResults):
        print("********************************************************************")
        print("expected: " + str(expectedValues))
        print("outcome: " + str(results))
        print("number of correct classifications " + str(len(X)-misses) + "/" + str(len(X)))
        print("misclassified 1's: " + str(miss1))
        print("misclassified 2's: " + str(miss2))
        print("misclassified 3's: " + str(miss3))

    return results, misses, miss1, miss2

def majorityVote(dataset,expectedValues,classifiers,scale,minimum,maximum):
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

    misses = 0
    for i in range(len(majority)):
        if(majority[i] != expectedValues[i]):
            misses+=1

    print("********************************************************************")
    for x in range(len(classifiers)):
        print("classifier " + str(x) + " votes:" + str(votes[x]))
    print("Majority votes: " + str(majority))
    print("Expected: " + str(expectedValues))
    print("Correctly Classified: " + str(len(expectedValues)-misses) + "/" + str(len(expectedValues)))
    return misses, len(expectedValues)-misses


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

#removing outliers
cardboard = removeOutliers(cardboard, detector)
wood = removeOutliers(wood, detector)
plastic = removeOutliers(plastic, detector)

#creating expected values
cardboardEV = [[1 for i in range(len(cardboard))],[2 for i in range(len(cardboard))]]
woodEV = [[1 for i in range(len(wood))],[2 for i in range(len(wood))]]
plasticEV = [[1 for i in range(len(plastic))],[2 for i in range(len(plastic))]]

'''
#Majority vote predictions
print("D vs Rest")
majorityVote(cardboard+wood+plastic, cardboardEV[0]+plasticEV[1]+woodEV[1], clfs, scale, low, high)

print("E vs F")
majorityVote(wood+plastic,woodEV[0]+plasticEV[1],clfs,scale,low,high)



#single model predictions
#D vs Rest
print("Classifying D")
loocv(cardboard+wood+plastic, cardboardEV[0]+plasticEV[1]+woodEV[1], clf, scale, low, high, True)


#E vs F
print("Classifying E vs F")
loocv(wood+plastic, woodEV[0]+plasticEV[1], clf, scale, low, high, True)
'''









