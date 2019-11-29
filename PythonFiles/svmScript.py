import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm as svm
from sklearn import preprocessing as pp
from matplotlib import style
import xlrd 
style.use("ggplot")

#asking for input location
print("enter excel sheet file name:")
loc = input()

#taking dataset from excel file
wb = xlrd.open_workbook(loc)
sheet = wb.sheet_by_index(0)

title = sheet.cell_value(0,1)
sub = sheet.cell_value(1,1)

xTitle = sheet.cell_value(4,0)
yTitle = sheet.cell_value(4,1)

c1 = 5
inputArray = []

while sheet.cell_value(c1,0) != 0 and sheet.cell_value(c1,1) != 0:
    inputArray.append([sheet.cell_value(c1,0),sheet.cell_value(c1,1)])
    c1 += 1

c2 = 5
expectedValues = []

while sheet.cell_value(c2,2) != -1:
        expectedValues.append(sheet.cell_value(c2,2))
        c2 += 1

c3 = 5
while sheet.cell_value(c3,3) != 0 and sheet.cell_value(c3,4) != 0:
    inputArray.append([sheet.cell_value(c3,3),sheet.cell_value(c3,4)])
    c3 += 1

#testStart is the index of the 1st test case in input array
testStart = c1 - 5

#scaling data between 0 and 1
min_max_scaler = pp.MinMaxScaler()
X_min_max = min_max_scaler.fit_transform(inputArray)

#getting test value (last element of scaled x array)
Test = X_min_max[testStart:]
print('predicting points:')
print(Test)


#removing test value from x array
X_min_max = X_min_max[0:testStart]

#predicting
clf = svm.SVC(kernel='linear', C=1000)
clf.fit(X_min_max,expectedValues)

prediction = clf.predict(Test)
print('prediction: (class A = 1 class B = 0) ')
print(prediction)

# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])

# plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')

#displaying scatter plot of values and decision function
plt.scatter(X_min_max[:,0],X_min_max[:,1], c=expectedValues)
plt.xlabel(xTitle)
plt.ylabel(yTitle)
plt.title(sub, fontsize = 12)
plt.suptitle(title, fontsize = 18)
plt.show()

