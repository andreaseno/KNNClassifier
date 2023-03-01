
# resources
# https://scikit-learn.org/stable/modules/cross_validation.html
# https://machinelearningmastery.com/loocv-for-evaluating-machine-learning-algorithms/
# https://stats.stackexchange.com/questions/353904/do-i-need-data-separation-in-knn
# https://www.educba.com/python-print-table/
# https://medium.com/@kunal_gohrani/different-types-of-distance-metrics-used-in-machine-learning-e9928c5e26c7
# https://becominghuman.ai/machine-learning-series-day-4-k-nn-8e9ba757a419


from random import seed
from random import randrange
from csv import reader
from math import sqrt
import matplotlib.pyplot as plt
import pandas as pd
import copy
import time
import numpy as np
import seaborn

#time total wallclock
wallclockStart = time.perf_counter()
#Load the dataset for visualization using pandas
fname = 'digit-recognizer/train.csv'
data = pd.read_csv(fname)
print('First 5 rows of the dataframe')
print(data.head())

# observer shape
print('(rows by collumns): ', end='')
print(data.shape)
print(data.describe().T)
numrows = data.shape[0]

def display_row(i,listData):
    displayList = copy.deepcopy(listData[i])
    displayList.pop(0)

    display=np.array(displayList)
    display = display.reshape(28,28)
    plt.imshow(display, cmap='gray')
    plt.show()
    
listData = data.values.tolist()
for i in range(20):
    display_row(i,listData)

# calulate distance
# 0 for chebyshev, 1 for manhattan, 2 for euclidean
def distance(row1, row2, distType):
	stepDist = 1
	if(distType == 0):
		distance = list()
		# calculates the minkowski metric using chebyshev distance
		for i in range(1,len(row1),stepDist):
			# print(row1[i])
			# print(row2[i])
			distance.append(abs(row1[i] - row2[i]))

		return max(distance)
	elif(distType == 1):
		distance = 0.0
		# calculates the minkowski metric using manhattan distance
		for i in range(1,len(row1),stepDist):
			# print(row1[i])
			# print(row2[i])
			distance += abs(row1[i] - row2[i])

		return distance
	elif (distType == 2):
		distance = 0.0
		# calculates the minkowski metric using euclidean distance
		for i in range(1,len(row1),stepDist):
			# print(row1[i])
			# print(row2[i])
			distance += (row1[i] - row2[i])**2

		return sqrt(distance)

# retreive the k nearest neighbors to the test row
def getKNN(train, testRow, k, distType):
	# list to store distances on k nearby neighbors
	distances = list()
	# calculate distance to n neighbors from the test_row
	for trainRow in train:
		dist = distance(testRow, trainRow, distType)
		# appends a tuple containing 2 elements one for the row and one for the distance
		distances.append((trainRow, dist))
	# sorts the list based on the second index of 
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(k):
		neighbors.append(distances[i][0])
	return neighbors

def predicted_class(trainingData, testRow, k, distType):
	# gets the nearest k neighbors to the current point
	neighbors = getKNN(trainingData, testRow, k, distType)
	# create a list containing labels of the k nearest neighbors
	nearestLabels = [row2[0] for row2 in neighbors]
	# assign label to the current test point based on what label is in the majority
	return max(set(nearestLabels), key=nearestLabels.count)

# kNN Algorithm
def k_nearest_neighbors(trainingData, testData, k, distType):
    # create a list to store the predicted values 
	predictions = []
    # for each node in the test data predict the outcome based
    # on the surrounding neighbors labels
	for row in testData:
		predictions.append(predicted_class(trainingData, row, k, distType))
    
	return(predictions)

# Calculate how accurate the trained algorithm is
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, n_folds, k, distType):
    # separate out the dataset into n_folds number of lists 
	folds = []
	dataset_copy = list(dataset)
    #  calculate the size of each fold using the dataset size and the number of folds
	fold_size = int(len(dataset) / n_folds)
    # separate the dataset into n equal folds
	for _ in range(n_folds):
		fold = []
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		folds.append(fold)
    # list to store accuracy of each fold
	scores = []
    # for every fold in the split, perform training on n_folds-1 folds and then
    # use the last fold for validation
	for fold in folds:
		train_set = list(folds)
        # remove fold for validation
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
        #  list of actual outcomes used to report accuracy
		actual = []
        # copy all data from the removed fold to the test data
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
            # add real outcome for current row to actual
			actual.append(row[0])
            # sets the outcome value to none
			row_copy[0] = None
        
		predicted = k_nearest_neighbors(train_set, test_set, k, distType)

        # calculate accuracy of the predictions
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores

def test_with_distance_type(diabetesData, testSet, distType):
	tablerow = list()
	# evaluate algorithm
	nFolds = 9
	kNumNeighbors = 1
	# preturns list of predicted values for the outcomes of each row
	scores = evaluate_algorithm(diabetesData, nFolds, kNumNeighbors, distType)
	
	if(extraInfo): print('List of Scores: ', scores)
	tot = (sum(scores)/float(len(scores)))
	if(extraInfo): print('Average Accuracy: ', tot)
	tablerow.append(tot)

	totalcorrect=0
	falseP = 0
	falseN = 0
	trueP = 0
	trueN = 0
	totaliterations=len(testSet)
	correctOutcome=0
	# my accuracy calculation
	for i in range(len(testSet)):
		# select random row from dataset
		rowIndex = randrange(len(testSet))
		row = copy.deepcopy(testSet[rowIndex])
		if (debug): print("row number is ",rowIndex," and the row is ",row)
		correctOutcome=row[0]
		confMatrixAct.append(correctOutcome)
		del row[0]
		label = predicted_class(testSet, row, kNumNeighbors, distType)
		confMatrixPred.append(label)
		if (label == correctOutcome): 
			totalcorrect+=1

		if(debug): print()

	if(extraInfo):
		print("percentage correct for test set size of %s is %s" % (totaliterations, (totalcorrect/totaliterations)*100))
	tablerow.append((totalcorrect/totaliterations)*100)


	return tablerow

#lets debug print statements be bounded behind if statements
debug = False
# toggle for extra tables and info
extraInfo = False
accuracytable=list()
confMatrixPred = list()
confMatrixAct = list()

# x is data subset size and y*1000 is the threshhold value
x = 800
y = 60
for seednum in range(1,2):
	tablerow=list()
	tablerow.append(seednum)
	seed(seednum)

	numrows = data.shape[0]
	testSize = x
	testSet = list()
	trainSet = list()

	# build test set
	remMatrix = [0]*numrows
	rowIndex = randrange(len(listData))
	for i in range(testSize):
		# select random row from dataset
		while(remMatrix[rowIndex]!=0):
			rowIndex = randrange(len(listData))
		remMatrix[rowIndex]=1
		# remove from training data and add to test data (holdout)
		row = copy.deepcopy(listData[rowIndex])
		testSet.append(row)
	# build train set
	rowIndex = randrange(len(listData))
	for i in range (testSize):
		# select random row from dataset
		while(remMatrix[rowIndex]!=0):
			rowIndex = randrange(len(listData))
		remMatrix[rowIndex]=1
		# remove from training data and add to test data (holdout)
		row = copy.deepcopy(listData[rowIndex])
		trainSet.append(row)

	trainDimMatrix = [0]*len(trainSet[0])
	for i in range(len(trainSet)):
		for j in range(1,len(trainSet[i])):
			trainDimMatrix[j]+=trainSet[i][j]
	jcount = 0
	lowThreshold = y*1000
	highThreshold = testSize*255 - lowThreshold

	for i in range(len(trainSet)):
		for j in range(len(trainSet[i])-1,0,-1):
			if ((trainDimMatrix[j]<=lowThreshold or trainDimMatrix[j]>=highThreshold)):
				del trainSet[i][j]


	TestDimMatrix = [0]*len(testSet[0])
	for i in range(len(testSet)):
		for j in range(1,len(testSet[i])):
			TestDimMatrix[j]+=testSet[i][j]
	for i in range(len(testSet)):
		for j in range(len(testSet[i])-1,0,-1):
			if ((TestDimMatrix[j]<=lowThreshold or TestDimMatrix[j]>=highThreshold)):
				del testSet[i][j]

	distType = 2
	if(extraInfo):
		if(distType == 0):
			print("\nTesting with chebyshev distance:\n")
		elif(distType == 1):
			print("\nTesting with manhattan distance:\n")
		elif(distType == 2):
			print("\nTesting with euclidean distance:\n")
	tablerow.extend(test_with_distance_type(trainSet, testSet, distType))
	accuracytable.append(tablerow)

from tabulate import tabulate
print()
print (tabulate(accuracytable, headers=["Seed\nNumber", "Validation\nAccuracy", "Test\nAccuracy"]))
print()

# Print confusion matrix
y_actu = pd.Series(confMatrixAct, name='Actual')
y_pred = pd.Series(confMatrixPred, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)
hm = seaborn.heatmap(data = df_confusion, cmap='Blues', annot=True)
plt.show()

# Test accuracy on differing number of k on one class
seed(3)

# build a set only containing the number 9
kvarying = list()
for i in range(1000):
    if(listData[i][0]==9): 
        row = copy.deepcopy(listData[i])
        kvarying.append(row)
        
testSet = list()
# build test set
remMatrix = [0]*numrows
rowIndex = randrange(len(listData))
for i in range(800):
    # select random row from dataset
    while(remMatrix[rowIndex]!=0):
        rowIndex = randrange(len(listData))
    remMatrix[rowIndex]=1
    # remove from training data and add to test data (holdout)
    row = copy.deepcopy(listData[rowIndex])
    testSet.append(row)        

# perform dimension reduction
kvaryingDimMatrix = [0]*len(testSet[0])
for i in range(len(testSet)):
    for j in range(1,len(testSet[i])):
        kvaryingDimMatrix[j]+=testSet[i][j]
for i in range(len(testSet)):
    for j in range(len(testSet[i])-1,0,-1):
        if ((kvaryingDimMatrix[j]<=20000 or kvaryingDimMatrix[j]>=len(testSet)*255 - 20000)):
            del testSet[i][j]
for i in range(len(kvarying)):
    for j in range(len(kvarying[i])-1,0,-1):
        if ((kvaryingDimMatrix[j]<=20000 or kvaryingDimMatrix[j]>=len(testSet)*255 - 20000)):
            del kvarying[i][j]
# report the results of the dimension reduction
dimcount=0
for i in range(len(kvaryingDimMatrix)) :
		if(kvaryingDimMatrix[i]<=20000 or kvaryingDimMatrix[i]>=len(testSet)*255 - 20000):
			dimcount+=1
print("The total number of features before dimension reduction is: ", data.shape[1])
print("The total number of features after dimension reduction is: ", data.shape[1]-dimcount)

# test the accuracy
accuracyvals = list()
kvals = list()
for testK in range(1,6):
    totalcorrect=0
    totaliterations=len(kvarying)
    correctOutcome=0
    # my accuracy calculation
    for i in range(len(kvarying)):
        # select random row from dataset
        rowIndex = randrange(len(kvarying))
        row = copy.deepcopy(kvarying[rowIndex])
        correctOutcome=row[0]
        del row[0]
        label = predicted_class(testSet, row, testK, 2)
        if (label == correctOutcome): 
            totalcorrect+=1
    print("percentage correct for k = %s is %s" % (testK, (totalcorrect/totaliterations)*100))
    accuracyvals.append((totalcorrect/totaliterations)*100) 
    kvals.append(testK)
# plot the results
plt.plot(kvals, accuracyvals)
plt.show()

wallclockEnd = time.perf_counter()
ms = (wallclockEnd-wallclockStart)
print("exec_time for .py file is ",ms," secs.")