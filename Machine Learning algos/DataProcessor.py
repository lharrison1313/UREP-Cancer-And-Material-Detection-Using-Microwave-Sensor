import csv
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


#function used for finding the minimum of a single sensor
def findMin(dataset):
    minimum = dataset[0][1]
    for x in dataset:
        if(x[1] < minimum):
            minimum = x[1]
            minimumf = x[0]
    return minimumf,minimum

#prints name of data set and each array in the dataset
def printData(name,dataset):
    print(name)
    for data in dataset:
        print(data)

#returns an array of 5 minimums 1 for each sensor format: {f1,f2,f3,f4,f5}
def processData(csvFile):
    #initializing 2d array
    datalist = [[0 for x in range(2)] for y in range(10004)]

    #parsing csv into 2d array
    keys = ["Frequency", "Formatted Data"]
    with open(csvFile) as csvfile:
        reader = csv.DictReader(csvfile,keys)
        rows = 0
        for row in reader:
           datalist[rows][0] = row["Frequency"]
           datalist[rows][1] = row["Formatted Data"]
           rows += 1
           
    #removing header info from datalist
    datalist = datalist[3:]

    #converting datalist values from strings to floats
    rows = 0
    for x in datalist:
        datalist[rows][0] = float(x[0])
        datalist[rows][1] = float(x[1])
        rows+=1

    #splitting datalist values into 5 sets of sensor data
    #change splicing values to change intervals
    f1set = datalist[0:1119]
    f2set = datalist[1119:3589]
    f3set = datalist[3589:5740]
    f4set = datalist[5740:7175]
    f5set = datalist[7175:9086]

    f1 = findMin(f1set)
    f2 = findMin(f2set)
    f3 = findMin(f3set)
    f4 = findMin(f4set)
    f5 = findMin(f5set)


    #plotting values
    xvalues=[0 for x in range(len(datalist))]
    yvalues=[0 for x in range(len(datalist))]
    i = 0
    for x in datalist:
        xvalues[i] = datalist[i][0] 
        yvalues[i] = datalist[i][1] 
        i+=1
    fig, ax = plt.subplots()
    ax.plot(xvalues,yvalues,"b")
    ax.plot(f1[0],f1[1],"go")
    ax.plot(f2[0],f2[1],"go")
    ax.plot(f3[0],f3[1],"go")
    ax.plot(f4[0],f4[1],"go")
    ax.plot(f5[0],f5[1],"go")
    ax.grid()
    plt.show()
    
    return f1[0], f2[0], f3[0], f4[0], f5[0]

#returns deltas and dataset of a folder containing sensor data and a empty sensor file
def buildResults(datasetFilePath,emptySensorFilePath):
    
    filelist = os.listdir(datasetFilePath)
    dataset = []
    deltas = []

    for file in filelist:
        dataset.append(processData(datasetFilePath + file))

    mt = processData(emptySensorFilePath)
    
    for data in dataset:
        deltas.append(np.subtract(np.asarray(mt),np.asarray(data)))

    return dataset, mt, deltas




print("Please enter folder path for dataset")
datasetFilePath = input()
print("Please enter name for dataset")
datasetName = input()
print("please enter file path for empty sensor data")
emptySensorFilePath = input()

results = buildResults(datasetFilePath,emptySensorFilePath)

printData(datasetName + " data",results[0])
printData("empty sensor", results[1])
printData(datasetName + " deltas", results[2] )


#example input        
#C:\Users\Luke\Documents\DNA_Hybridization\SensorData\Wood\
#wood
#C:\Users\Luke\Documents\DNA_Hybridization\SensorData\Empty_Sensor\M1(EMPTY).CSV

    
