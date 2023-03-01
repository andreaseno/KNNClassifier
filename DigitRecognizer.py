from random import seed
from random import randrange
from csv import reader
from math import sqrt
import matplotlib.pyplot as plt
import pandas as pd
import copy
import time
import numpy as np


# resources
# https://scikit-learn.org/stable/modules/cross_validation.html
# https://machinelearningmastery.com/loocv-for-evaluating-machine-learning-algorithms/
# https://stats.stackexchange.com/questions/353904/do-i-need-data-separation-in-knn
# https://www.educba.com/python-print-table/
# https://medium.com/@kunal_gohrani/different-types-of-distance-metrics-used-in-machine-learning-e9928c5e26c7
# https://becominghuman.ai/machine-learning-series-day-4-k-nn-8e9ba757a419
# ??should I be normalizing my data??

plt.style.use('ggplot')


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
        
		# start = time.perf_counter()
 
		# # preturns list of predicted values for the outcomes of each row
		# print("checkpoint6 trainset size is ",len(train_set)," testsize size is ",len(test_set))
		predicted = k_nearest_neighbors(train_set, test_set, k, distType)
		print("checkpoint7")
	
		# end = time.perf_counter()
		
		# # find elapsed time in seconds
		# ms = (end-start) * 10**6
		# if (extraInfo):print(f"exec_time for KNN is {ms:.03f} micro secs.")
		# with open('KNNRuntimes.txt', 'a') as f:
		# 	f.write(str(ms)+'\n')
		
        # calculate accuracy of the 
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores

def test_with_distance_type(diabetesData, testSet, distType):
	tablerow = list()
	# evaluate algorithm
	nFolds = 5
	kNumNeighbors = 6
	print("checkpoint5")
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
		del row[0]
		# print("row number is ",rowIndex," and the row is ",row)
		label = predicted_class(testSet, row, kNumNeighbors, distType)
		# print(' Predicted: %s, Correct Oucome: %s' % ( label, correctOutcome))
		if (label == correctOutcome): 
			totalcorrect+=1
			if (label == 1):
				trueP+=1
			else:
				trueN+=1
		else:
			if (label == 1):
				falseP+=1
			else:
				falseN+=1
			

		if(debug): print()

	if(extraInfo):
		print("percentage correct for test set size of %s is %s" % (totaliterations, (totalcorrect/totaliterations)*100))
		print("False Negatives: ",falseN)
		print("False Positives: ",falseP)
		print("True Negatives:  ",trueN)
		print("True Positives:  ",trueP)
	tablerow.append((totalcorrect/totaliterations)*100)
	tablerow.append(falseN)
	tablerow.append(falseP)
	tablerow.append(trueN)
	tablerow.append(trueP)

	return tablerow

#lets debug print statements be bounded behind if statements
debug = False
# toggle for extra tables and info
extraInfo = True
accuracytable=list()
print("checkpoint1")
fname = 'digit-recognizer/train.csv'
data = pd.read_csv(fname)
listData = data.values.tolist()

num = 4
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
# for i in range(len(listData)):
# 	if (listData[i][0]==num):
# num+=1
displayList = copy.deepcopy(listData[1])
# print(display)
displayList.pop(0)
print(displayList)
display=np.array(displayList)
display = display.reshape(28,28)
# print(display)
plt.imshow(display, cmap='gray')
plt.show()
		# break
	# if (num <=10): break
# print(display)

x = 230
y = 20
for seednum in range(1,6):
	tablerow=list()
	tablerow.append(seednum)
	
	seed(seednum)
	
	
	print("checkpoint2")

	#Load the dataset for visualization using matplotlib and pandas
	
	# listData = data.values.tolist()
	print("checkpoint3")
	# if(extraInfo):
		#Print the first 5 rows of the dataframe.
		# print('First 5 rows of the dataframe')
		# print(data.head())

		# # observer shape
		# print('(rows by collumns): ', end='')
		# print(data.shape)
		# print(data.describe().T)

		# data.hist()
		# plt.show()
	numrows = data.shape[0]
	print("numcols = ",data.shape[1])
	# for i in range(len(listData)):
	# 	# print("row before ",listData[i])
	# 	temp = listData[i][0]
	# 	# print("type=",type(listData[i])," type=",type(listData[i]))
	# 	# listData[i] = listData[i][::5]
	# 	listData[i][0] = temp
	# 	# print("row after  ",listData[i])

	
	# remove rows for test set
	# testSize = int(0.1*numrows)
	testSize = x
	testSet = list()
	trainSet = list()

	# for i in range(testSize):
	# 	# select random row from dataset
	# 	rowIndex = randrange(len(listData))
	# 	# remove from training data and add to test data (holdout)
	# 	row = listData.pop(rowIndex)
	# 	testSet.append(row)
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

	print("test set size is", len(testSet))

	rowIndex = randrange(len(listData))
	for i in range (testSize):
		# select random row from dataset
		while(remMatrix[rowIndex]!=0):
			rowIndex = randrange(len(listData))
		remMatrix[rowIndex]=1
		# remove from training data and add to test data (holdout)
		row = copy.deepcopy(listData[rowIndex])
		trainSet.append(row)



	print("train set size is", len(trainSet))
	trainDimMatrix = [0]*len(trainSet[0])
	# print(trainRangeMatrix)
	for i in range(len(trainSet)):
		for j in range(1,len(trainSet[i])):
			# if (trainSet[i][j]!=0):
			trainDimMatrix[j]+=trainSet[i][j]
	jcount = 0
	lowThreshold = y*1000
	highThreshold = numrows*255 - lowThreshold

	for i in range(len(trainDimMatrix)) :
		if((trainDimMatrix[i]<=lowThreshold or trainDimMatrix[i]>=highThreshold)):
			# print("range is ",trainRangeMatrix[i][1]-trainRangeMatrix[i][0])
			# print(i," is irrelevant")
			jcount+=1
	print("there are ",data.shape[1],"total dimesions with",data.shape[1]-jcount," relevant dimensions and irrelevant =  ",jcount)
	for i in range(len(trainSet)):
		for j in range(len(trainSet[i])-1,0,-1):
			# if (trainRangeMatrix[j][1]-trainRangeMatrix[j][0]<255):
				# print("range is ",trainRangeMatrix[j][1]-trainRangeMatrix[j][0])
			if ((trainDimMatrix[j]<=lowThreshold or trainDimMatrix[j]>=highThreshold)):
				del trainSet[i][j]


	TestDimMatrix = [0]*len(testSet[0])
	for i in range(len(testSet)):
		for j in range(1,len(testSet[i])):
			# if (trainSet[i][j]!=0):
			TestDimMatrix[j]+=testSet[i][j]
	# jcount = 0
	# lowThreshold = 30000
	# highThreshold = numrows*255 - lowThreshold

	# for i in range(len(TestDimMatrix)) :
	# 	if(TestDimMatrix[i]<=lowThreshold or TestDimMatrix[i]>=highThreshold):
	# 		print(i," is irrelevant")
	# 		jcount+=1
	# print("there are ",data.shape[1],"total dimesions with",data.shape[1]-jcount," relevant dimensions and irrelevant =  ",jcount)
	for i in range(len(testSet)):
		for j in range(len(testSet[i])-1,0,-1):
			if ((TestDimMatrix[j]<=lowThreshold or TestDimMatrix[j]>=highThreshold)):
				del testSet[i][j]

	# for i in range(1,len(trainSet)):
	# 	if (len(trainSet[i-1])!=len(trainSet[i])):
	# 		print("error")


	distType = 2
	if(extraInfo):
		if(distType == 0):
			print("\nTesting with chebyshev distance:\n")
		elif(distType == 1):
			print("\nTesting with manhattan distance:\n")
		elif(distType == 2):
			print("\nTesting with euclidean distance:\n")
	print("checkpoint4")
	tablerow.extend(test_with_distance_type(trainSet, testSet, distType))
	accuracytable.append(tablerow)
averages = copy.deepcopy(accuracytable[0])

averages[0]=("Mean Value:")
for value in range(1,len(averages)):
	averages[value]=0
for row in range(len(accuracytable)):
	for value in range(1,len(accuracytable[row])):
		averages[value]+=float(accuracytable[row][value])



for value in range(1,len(averages)):
	averages[value]= averages[value]/len(accuracytable)
accuracytable.append(averages)

from tabulate import tabulate
print()
print (tabulate(accuracytable, headers=["Seed\nNumber", "Validation\nAccuracy", "Test\nAccuracy", "False\nNegatives", "False\nPositives", "True\nNegatives", "True\nPositives"]))
print()