import MachineLearning as ml
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.ensemble import VotingClassifier
from pyod.models import abod
from pyod.models import iforest
from pyod.models import knn

#D = cardboard
#E = wood
#F = Plastic

#machine learning parameters and classifiers
high = 1
low = -1
scale = True
gnb = GaussianNB()
dt = DecisionTreeClassifier()
KNearest = KNeighborsClassifier(n_neighbors=5)
svm1 = svm.SVC(kernel="rbf", decision_function_shape='ovr', gamma="scale", C=100) #creating svm object
rf = RandomForestClassifier(n_estimators=10, random_state=1) #creating rf object
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,3), random_state=1) #creating mlp object
majority = VotingClassifier(estimators=[("svm",svm1),("rf",rf),("mlp",mlp)], voting = "hard") #creating majority vote object


#Cross validation iterators
skf = StratifiedKFold(n_splits=5)
loo = LeaveOneOut()

#outlier detectors
angleBased = abod.ABOD(method="fast")
isolationForrest = iforest.IForest(n_estimators=10, behaviour="new")
kNearestNeighbors = knn.KNN(method="median",n_neighbors=5)
detector = kNearestNeighbors

#datasets
cardboard = ml.parseCsv("C:\\Users\\Luke\\Documents\\GitHub\\UREP_Cancer_Detection_Array_Microwave_Sensor\\results\\Deltas\\DDeltas.csv")
wood = ml.parseCsv("C:\\Users\\Luke\\Documents\\GitHub\\UREP_Cancer_Detection_Array_Microwave_Sensor\\results\\Deltas\\EDeltas.csv")
plastic = ml.parseCsv("C:\\Users\\Luke\\Documents\\GitHub\\UREP_Cancer_Detection_Array_Microwave_Sensor\\results\\Deltas\\BDeltas.csv")
plastic = plastic+ml.parseCsv("C:\\Users\\Luke\\Documents\\GitHub\\UREP_Cancer_Detection_Array_Microwave_Sensor\\results\\Deltas\\FDeltas.csv")


#standard expected values
expected = [[0 for x in range(50)],[1 for x in range(50)]]

#removing outliers
cardboard = ml.removeOutliers(cardboard, detector, True)
wood = ml.removeOutliers(wood, detector, True)
plastic = ml.removeOutliers(plastic, detector, True)

#creating expected values after outlier removal
cardboardEV = [[0 for i in range(len(cardboard))],[1 for i in range(len(cardboard))]]
woodEV = [[0 for i in range(len(wood))],[1 for i in range(len(wood))]]
plasticEV = [[0 for i in range(len(plastic))],[1 for i in range(len(plastic))]]

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

#LOO MLP
#ml.getBestCombination([cardboardCombos,woodCombos,plasticCombos], cardboardEV[0]+woodEV[1]+plasticEV[1], mlp, loo, True, low, high,True)
#ml.getBestCombination([woodCombos,plasticCombos], woodEV[0]+plasticEV[1], mlp, loo, True, low, high, True)

#SKF MLP
#ml.getBestCombination([cardboardCombos,woodCombos,plasticCombos], cardboardEV[0]+woodEV[1]+plasticEV[1], mlp, skf, True, low, high,True)
#ml.getBestCombination([woodCombos,plasticCombos], woodEV[0]+plasticEV[1], mlp, skf, True, low, high, True)

#LOO Decision Tree
#ml.getBestCombination([cardboardCombos,woodCombos,plasticCombos], cardboardEV[0]+woodEV[1]+plasticEV[1], dt, loo, True, low, high,True)
#ml.getBestCombination([woodCombos,plasticCombos], woodEV[0]+plasticEV[1], dt, loo, True, low, high, True)

#SKF Decision Tree
#ml.getBestCombination([cardboardCombos,woodCombos,plasticCombos], cardboardEV[0]+woodEV[1]+plasticEV[1], dt, skf, True, low, high,True)
#ml.getBestCombination([woodCombos,plasticCombos], woodEV[0]+plasticEV[1], dt, skf, True, low, high, True)

#LOO Gaussian Nieve Bayes
#ml.getBestCombination([cardboardCombos,woodCombos,plasticCombos], cardboardEV[0]+woodEV[1]+plasticEV[1], gnb, loo, True, low, high,True)
#ml.getBestCombination([woodCombos,plasticCombos], woodEV[0]+plasticEV[1], gnb, loo, True, low, high, True)

#SKF Gaussian Nieve Bayes
#ml.getBestCombination([cardboardCombos,woodCombos,plasticCombos], cardboardEV[0]+woodEV[1]+plasticEV[1], gnb, skf, True, low, high,True)
#ml.getBestCombination([woodCombos,plasticCombos], woodEV[0]+plasticEV[1], gnb, skf, True, low, high, True)

#LOO KNN
#ml.getBestCombination([cardboardCombos,woodCombos,plasticCombos], cardboardEV[0]+woodEV[1]+plasticEV[1],KNearest , loo, True, low, high,True)
#ml.getBestCombination([woodCombos,plasticCombos], woodEV[0]+plasticEV[1], KNearest, loo, True, low, high, True)

#SKF KNN
#ml.getBestCombination([cardboardCombos,woodCombos,plasticCombos], cardboardEV[0]+woodEV[1]+plasticEV[1],KNearest , skf, True, low, high,True)
#ml.getBestCombination([woodCombos,plasticCombos], woodEV[0]+plasticEV[1], KNearest, skf, True, low, high, True)

#LOO Random Forrest
#ml.getBestCombination([cardboardCombos,woodCombos,plasticCombos], cardboardEV[0]+woodEV[1]+plasticEV[1], rf, loo, True, low, high,True)
#ml.getBestCombination([woodCombos,plasticCombos], woodEV[0]+plasticEV[1], rf, loo, True, low, high, True)

#SKF Random Forrest
#ml.getBestCombination([cardboardCombos,woodCombos,plasticCombos], cardboardEV[0]+woodEV[1]+plasticEV[1], rf, skf, True, low, high,True)
#ml.getBestCombination([woodCombos,plasticCombos], woodEV[0]+plasticEV[1], rf, skf, True, low, high, True)

#loo SVM
#ml.getBestCombination([cardboardCombos,woodCombos,plasticCombos], cardboardEV[0]+woodEV[1]+plasticEV[1], svm1, loo, True, low, high,True)
#ml.getBestCombination([woodCombos,plasticCombos], woodEV[0]+plasticEV[1], svm1, loo, True, low, high, True)

#SKF SVM
#ml.getBestCombination([cardboardCombos,woodCombos,plasticCombos], cardboardEV[0]+woodEV[1]+plasticEV[1], svm1, skf, True, low, high,True)
#ml.getBestCombination([woodCombos,plasticCombos], woodEV[0]+plasticEV[1], svm1, skf, True, low, high, True)




