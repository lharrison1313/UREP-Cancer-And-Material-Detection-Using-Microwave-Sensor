import MachineLearning as ml
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from pyod.models import abod
from pyod.models import iforest
from pyod.models import knn

def getBestCombination(datasetArray, expectedValues, classifier, scale, minimum, maximum, printResults, majorityVote):
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
            if majorityVote:
                c = ml.majorityVote(dataset, expectedValues, classifier, scale, minimum, maximum, printResults)[3]
            else:
                c = ml.loocv(dataset, expectedValues, classifier, scale, minimum, maximum, printResults)[3]

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
    print("Best results: i=" + str(maxI) + " j=" + str(maxJ) + " max=" + str(maxC))
    return bestSet


#D = cardboard
#E = Plastic
#F = wood

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
cardboard = ml.parseCsv("C:\\Users\\Luke\\Documents\\GitHub\\UREP_Cancer_Detection_Array_Microwave_Sensor\\results\\Deltas\\DDeltas.csv")
wood = ml.parseCsv("C:\\Users\\Luke\\Documents\\GitHub\\UREP_Cancer_Detection_Array_Microwave_Sensor\\results\\Deltas\\EDeltas.csv")
plastic = ml.parseCsv("C:\\Users\\Luke\\Documents\\GitHub\\UREP_Cancer_Detection_Array_Microwave_Sensor\\results\\Deltas\\BDeltas.csv")
plastic = plastic+ml.parseCsv("C:\\Users\\Luke\\Documents\\GitHub\\UREP_Cancer_Detection_Array_Microwave_Sensor\\results\\Deltas\\FDeltas.csv")

#standard expected values
expected = [[0 for x in range(50)],[1 for x in range(50)]]

#sensor combo datasets row = rValue col = dataset
# r -> number of datasets
# 1 -> 5
# 2 -> 10
# 3 -> 10
# 4 -> 5
# 5 -> 1
cardboardCombos = []
woodCombos = []
plasticCombos = []
woodPlasticCombos = []

for i in range(1,len(cardboard[0])+1):
    cardboardCombos.append(ml.getCombos(cardboard,i))
    woodCombos.append(ml.getCombos(wood,i))
    plasticCombos.append(ml.getCombos(plastic,i))

#removing outliers
cardboard = ml.removeOutliers(cardboard, detector, False)
wood = ml.removeOutliers(wood, detector, False)
plastic = ml.removeOutliers(plastic, detector, False)


#creating expected values after outlier removal
cardboardEV = [[0 for i in range(len(cardboard))],[1 for i in range(len(cardboard))]]
woodEV = [[0 for i in range(len(wood))],[1 for i in range(len(wood))]]
plasticEV = [[0 for i in range(len(plastic))],[1 for i in range(len(plastic))]]


#combo predictions D vs Rest
print("Combo DvR svm")
getBestCombination([cardboardCombos,plasticCombos,woodCombos],expected[0]+expected[1]+expected[1],svm1,True,low,high,False,False)
print("\nCombo DvR rf")
getBestCombination([cardboardCombos,plasticCombos,woodCombos],expected[0]+expected[1]+expected[1],rf,True,low,high,False,False)
print("\nCombo DvR mlp")
getBestCombination([cardboardCombos,plasticCombos,woodCombos],expected[0]+expected[1]+expected[1],mlp,True,low,high,False,False)
print("\nCombo DvR majority")
getBestCombination([cardboardCombos,plasticCombos,woodCombos],expected[0]+expected[1]+expected[1],clfs,True,low,high,False,True)

#combo preictions E vs F
print("\nCombo EvF svm")
getBestCombination([plasticCombos,woodCombos],expected[0]+expected[1],svm1,True,low,high,False,False)
print("\nCombo EvF rf")
getBestCombination([plasticCombos,woodCombos],expected[0]+expected[1],rf,True,low,high,False,False)
print("\nCombo EvF mlp")
getBestCombination([plasticCombos,woodCombos],expected[0]+expected[1],mlp,True,low,high,False,False)
print("\nCombo EvF majority")
getBestCombination([plasticCombos,woodCombos],expected[0]+expected[1],clfs,True,low,high,False,True)

#5 sensor predictions
#D vs Rest
print("\nDvR svm")
ml.loocv(cardboard+wood+plastic, cardboardEV[0]+plasticEV[1]+woodEV[1], svm1, scale, low, high, True)
print("\nDvR rf")
ml.loocv(cardboard+wood+plastic, cardboardEV[0]+plasticEV[1]+woodEV[1], rf, scale, low, high, True)
print("\nDvR mlp")
ml.loocv(cardboard+wood+plastic, cardboardEV[0]+plasticEV[1]+woodEV[1], mlp, scale, low, high, True)
print("\nDvR Majority vote")
ml.majorityVote(cardboard+wood+plastic, cardboardEV[0]+plasticEV[1]+woodEV[1], clfs, scale, low, high,True)


#E vs F
print("\nEvF svm")
ml.loocv(wood+plastic, woodEV[0]+plasticEV[1], svm1, scale, low, high, True)
print("\nEvF rf")
ml.loocv(wood+plastic, woodEV[0]+plasticEV[1], rf, scale, low, high, True)
print("\nEvF mlp")
ml.loocv(wood+plastic, woodEV[0]+plasticEV[1], mlp, scale, low, high, True)
print("\nEvF Majority vote")
ml.majorityVote(wood+plastic,woodEV[0]+plasticEV[1],clfs,scale,low,high,True)

