# resources
# https://scikit-learn.org/stable/modules/cross_validation.html
# https://machinelearningmastery.com/loocv-for-evaluating-machine-learning-algorithms/
# https://stats.stackexchange.com/questions/353904/do-i-need-data-separation-in-knn
# https://www.educba.com/python-print-table/
# https://medium.com/@kunal_gohrani/different-types-of-distance-metrics-used-in-machine-learning-e9928c5e26c7
# https://becominghuman.ai/machine-learning-series-day-4-k-nn-8e9ba757a419
# ??should I be normalizing my data??


from random import seed
from random import randrange
from csv import reader
from math import sqrt
import matplotlib.pyplot as plt
import pandas as pd
import copy
import time
import seaborn

plt.style.use('ggplot')

#time total wallclock
wallclockStart = time.perf_counter()

fname = 'KNNPima/diabetes.csv'

#Load the dataset for visualization using matplotlib and pandas
data = pd.read_csv(fname)
#Print the first 5 rows of the dataframe.
print('First 5 rows of the dataframe')
print(data.head())

# observer shape
print('(rows by collumns): ', end='')
print(data.shape)
print(data.describe().T)
numrows = data.shape[0]

seaborn.pairplot(data, hue='Outcome')
plt.show()

# calulate distance
# 0 for chebyshev, 1 for manhattan, 2 for euclidean
def distance(row1, row2, distType):
	if(distType == 0):
		distance = list()
		# calculates the minkowski metric using chebyshev distance
		for i in range(len(row1)-1):
			# print(row1[i])
			# print(row2[i])
			distance.append(abs(row1[i] - row2[i]))

		return max(distance)
	elif(distType == 1):
		distance = 0.0
		# calculates the minkowski metric using manhattan distance
		for i in range(len(row1)-1):
			# print(row1[i])
			# print(row2[i])
			distance += abs(row1[i] - row2[i])

		return distance
	elif (distType == 2):
		distance = 0.0
		# calculates the minkowski metric using euclidean distance
		for i in range(len(row1)-1):
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
	nearestLabels = [row2[-1] for row2 in neighbors]
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
			actual.append(row[-1])
            # sets the outcome value to none
			row_copy[-1] = None
         
		# returns list of predicted values for the outcomes of each row
		predicted = k_nearest_neighbors(train_set, test_set, k, distType)

        # calculate accuracy of the 
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores

def test_with_distance_type(diabetesData, testSet, distType):
	tablerow = list()
	# evaluate algorithm
	nFolds = 9
	kNumNeighbors = 5 
	# returns list of predicted values for the outcomes of each row
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
		correctOutcome=row[8]
		del row[8]
		label = predicted_class(testSet, row, kNumNeighbors, distType)
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

# toggle for extra tables and info
extraInfo = False
listOfTables = list()
for distType in range(3):
	accuracytable=list()
	for seednum in range(1,11):
		tablerow=list()
		tablerow.append(seednum)

		seed(seednum)

        # turn pandas dataframe into a list of lists
		diabetesData = data.values.tolist()

        # remove rows for test set
		testSize = int(0.1*numrows)
		testSet = list()

		for i in range (testSize):
			# select random row from dataset
			rowIndex = randrange(len(diabetesData))
            # remove from training data and add to test data (holdout)
			row = diabetesData.pop(rowIndex)
			testSet.append(row)
            
		tablerow.extend(test_with_distance_type(diabetesData, testSet, distType))
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
    
	listOfTables.append(accuracytable)

from tabulate import tabulate
#time total wallclock
wallclockEnd = time.perf_counter()
for i in range(len(listOfTables)):
	print()
	if(i == 0):
		print("\nTesting with Chebyshev distance:\n")
	elif(i == 1):
		print("\nTesting with Manhattan distance:\n")
	elif(i == 2):
		print("\nTesting with Euclidean distance:\n")
	print (tabulate(listOfTables[i], headers=["Seed\nNumber", "Validation\nAccuracy", "Test\nAccuracy", "False\nNegatives", "False\nPositives", "True\nNegatives", "True\nPositives"]))
	print()

ms = (wallclockEnd-wallclockStart)
print("exec_time for .py file is ",ms," secs.")