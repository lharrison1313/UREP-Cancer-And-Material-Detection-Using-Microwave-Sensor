import csv
import os
import numpy as np
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

def parseCsv(csvFile):
    # initializing 2d array
    data = []

    # parsing csv into 2d array
    keys = ["Frequency", "Formatted Data"]
    with open(csvFile) as csvfile:
        reader = csv.DictReader(csvfile, keys)

        for row in reader:
            data.append([row["Frequency"],row["Formatted Data"]])

    # removing header info from datalist
    data = data[3:]

    # converting datalist values from strings to floats
    rows = 0
    for x in data:
        data[rows][0] = float(x[0])
        data[rows][1] = float(x[1])
        rows += 1
    return data


#returns an array of peak resonance frequencies based on the number of input partitions
def processData(csvFile,inputPartitions):
    data = parseCsv(csvFile)

    # splitting datalist values into different partitions
    frequencies = []
    for x in range(len(inputPartitions)):
        if(x == 0):
            frequencies.append(findMin(data[0:inputPartitions[x]]))
        else:
            frequencies.append(findMin(data[inputPartitions[x-1]:inputPartitions[x]]))

    #plotting all values
    xvalues=[0 for x in range(len(data))]
    yvalues=[0 for x in range(len(data))]
    i = 0
    for x in data:
        xvalues[i] = data[i][0]
        yvalues[i] = data[i][1]
        i+=1
    plt.plot(xvalues,yvalues,"black")

    #plotting peak resonance frequencies
    for x in frequencies :
        plt.plot(x[0],x[1],"ro")

    #setting labels
    plt.xlabel("Frequency [GHz]")
    plt.ylabel("S21 [DB]")
    #plt.savefig()

    #plotting partitions
    for xvals in inputPartitions:
        plt.axvline(x=data[xvals][0])
    plt.show()

    #putting x values in output array
    output = []
    for x in frequencies:
        output.append(x[0])
    
    return output


#returns deltas and dataset of a folder containing sensor data and a empty sensor file
def buildResults(datasetFilePath,emptySensorFilePath,partitions):

    filelist = os.listdir(datasetFilePath)
    peakFrequencies = []
    deltas = []
    tempPartitions = partitions

    for file in filelist:
        peakFrequencies.append(processData(datasetFilePath + file,partitions))

    mt = processData(emptySensorFilePath,partitions)
    
    for data in peakFrequencies:
        deltas.append(np.subtract(np.asarray(mt),np.asarray(data)))

    return peakFrequencies, mt, deltas




defaultPartitions = [1119, 3589, 5740, 7175, 9086]
print("Please enter folder path for dataset")
datasetFilePath = input() + "\\"
print("Please enter name for dataset")
datasetName = input()
print("please enter file path for empty sensor data")
emptySensorFilePath = input()

results = buildResults(datasetFilePath,emptySensorFilePath,defaultPartitions)
printData(datasetName + " data",results[0])
printData("empty sensor", results[1])
printData(datasetName + " deltas", results[2] )


#example input        
#C:\Users\Luke\Documents\DNA_Hybridization\SensorData\Wood
#wood
#C:\Users\Luke\Documents\DNA_Hybridization\SensorData\Empty_Sensor\M1(EMPTY).CSV

    
