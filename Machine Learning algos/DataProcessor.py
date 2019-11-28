import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory


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
def processData(csvFile,inputPartitions,figOutputFile):
    data = parseCsv(csvFile)

    # splitting datalist values into different partitions
    frequencies = []
    for x in range(len(inputPartitions)):
        if(x == 0):
            frequencies.append(findMin(data[0:inputPartitions[x]]))
        else:
            frequencies.append(findMin(data[inputPartitions[x-1]:inputPartitions[x]]))

    #plotting all values
    xvalues=[]
    yvalues=[]
    i = 0
    for x in data:
        xvalues.append(data[i][0])
        yvalues.append(data[i][1])
        i += 1
    plt.plot(xvalues, yvalues, "black")

    #plotting peak resonance frequencies
    for x in frequencies :
        plt.plot(x[0], x[1], "ro")

    #setting labels
    plt.xlabel("Frequency [GHz]")
    plt.ylabel("S21 [DB]")
    plt.savefig(figOutputFile)

    #plotting partitions
    for i in inputPartitions:
        plt.axvline(x=data[i][0])
    plt.show()
    print("done0")
    #putting x values in output array
    output = []
    for x in frequencies:
        output.append(x[0])
    
    return output

#returns deltas and dataset of a folder containing sensor data and a empty sensor file
def buildResults( datasetName, datasetFilePath, emptySensorFilePath, partitions, figOutputFile):

    filelist = os.listdir(datasetFilePath)
    peakFrequencies = []
    deltas = []
    dataNum = 1

    for file in filelist:
        tempPartitions = partitions.copy()
        keep = False
        while not keep:
            peakFrequencies.append(processData(datasetFilePath + file, tempPartitions, figOutputFile+"/"+datasetName+str(dataNum)))
            print("done1")
            tempPartitions, keep = changePartitions(tempPartitions)
        dataNum += 1

    keep = False
    while not keep:
        mt = processData(emptySensorFilePath, tempPartitions, figOutputFile+"/"+"EmptySensor")
        tempPartitions, keep = changePartitions(tempPartitions)
    
    for data in peakFrequencies:
        deltas.append(np.subtract(np.asarray(mt), np.asarray(data)),)

    return peakFrequencies, mt, deltas

def changePartitions(currentPartitions):
    newPartitions = currentPartitions.copy()
    keep = False
    done1 = False
    done2 = False
    while(not done1):
        print("done2")
        response1 = input("Would you like to edit the partitions for this sample? y/n\n")
        if response1 == "y":
            while not done2:
                print("Current Partitions")
                numParts = 0
                for parts in newPartitions:
                    print(str(numParts)+": "+ str(parts))
                    numParts += 1

                response2 = input("enter partition number or type 'done' to stop editing partition\n")
                if str(response2) == "done":
                    done2 = True
                elif int(response2) >= 0 and int(response2) < len(currentPartitions):
                    response3 = input("enter new value for partition or type 'done' to stop editing partition\n")
                    if(str(response3) == "done"):
                        print("backing up")
                    elif(int(response3) >= 0 and int(response3) <= 10000):
                        newPartitions[int(response2)] = int(response3)
                    else:
                        print("Invalid Input")
                else:
                    print("Invalid Input")
            done1 = True
        elif response1 == "n":
            keep = True
            done1 = True
        else:
            print("Invalid Input")
        return newPartitions,keep




defaultPartitions = [1119, 3589, 5740, 7175, 9086]
root = Tk()
root.withdraw()
print("Please enter folder path for dataset")
dataSetFilePath = askdirectory()+"/"
print(dataSetFilePath)
dataSetName = input("Please enter name for dataset:\n")
print("please enter file path for empty sensor data")
emptySensorFilePath = askopenfilename()
print(emptySensorFilePath)
print("please enter file path for peak resonance frequencies figures")
outputFilePathFigs = askdirectory()
print(outputFilePathFigs)
print("please enter folder path for deltas output csv file")
outputFilePathCsv = askdirectory()
print(outputFilePathCsv)
root.destroy()

results = buildResults(dataSetName, dataSetFilePath, emptySensorFilePath, defaultPartitions, outputFilePathFigs)
printData(dataSetName + " data", results[0])
printData("empty sensor", results[1])
printData(dataSetName + " deltas", results[2])

#placing deltas in csv file
df = pd.DataFrame.from_records(results[2])
df.to_csv(outputFilePathCsv+"/"+dataSetName+"Deltas.csv")

