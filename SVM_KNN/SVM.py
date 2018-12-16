# -*-coding=utf-8 -*-
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import numpy as np
import scipy.io as sio
import random
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import time
import os
from datetime import datetime


def load_dataset():
	if whichDataset == "pavia":
		pavia = "./data/PaviaU.mat"
		paviaGT = "./data/PaviaU_gt.mat"
		rawData = sio.loadmat(pavia)
		rawDataGT = sio.loadmat(paviaGT)
		img = rawData['paviaU']
		imgGT = rawDataGT["paviaU_gt"]
		return img, imgGT
	elif whichDataset == 'salinas':
		salinas = "./data/Salinas_corrected.mat"
		salinasGT = "./data/Salinas_gt.mat"
		rawData = sio.loadmat(salinas)
		rawDataGT = sio.loadmat(salinasGT)
		img = rawData['salinas_corrected']
		imgGT = rawDataGT['salinas_gt']
		return img, imgGT
	# Pour le datatset indian pines il faut reduire le nombre de classes 
	elif whichDataset == "indianPines":
		idnianPines = "./data/Indian_pines_corrected.mat"
		idnianPinesGT = "./data/Indian_pines_gt.mat"
		rawData = sio.loadmat(idnianPines)
		rawDataGT = sio.loadmat(idnianPinesGT)
		img = rawData['indian_pines_corrected']
		imgGT = rawDataGT['indian_pines_gt']
		# print(img.shape)
		# print(imgGT.shape)
		# print(np.max(imgGT))

		newGT = np.zeros([imgGT.shape[0], imgGT.shape[1]], dtype=int) # On initialise une matrice de 0 de la même taille que la matrice indian pines (145X145)
		classesNumber = np.max(imgGT) # Le nombre de classe maximum du dataset indian pines (16)
		originNum = np.zeros(shape=[classesNumber + 1], dtype=int) # Une matrice avec une seul ligne de 17 entier

		for i in range(imgGT.shape[0]): # Pour chaque ligne (145)
			for j in range(imgGT.shape[1]): # Pour chaque colonne (145)
				for k in range(1, classesNumber + 1): # pour k de 1 à 17 on compte les occurences de chaque classes
					if imgGT[i][j] == k: 
						originNum[k] += 1
		# print originNum
		# [   0   46 1428  830  237  483  730   28  478   20  972 2455  593  205 1265  386   93]

		index = 0
		numberOfClassesToKeep = 9 # On garde 9 classes
		dataNum = np.zeros(shape=[numberOfClassesToKeep], dtype=int)
		dataLabel = np.zeros(shape=[numberOfClassesToKeep], dtype=int)
		minClass = sorted(originNum,reverse=True)[numberOfClassesToKeep] # selectionne la 9eme classes la plus petite
		for i in range(len(originNum)): # parcours originNum et selectionne les classes voulu
			if originNum[i] > minClass:
				dataNum[index] = originNum[i]
				dataLabel[index] = i
				index += 1
		# Parcours imgGT et recopie seulement les classe voulu dans newGT
		for i in range(imgGT.shape[0]):
			for j in range(imgGT.shape[1]):
				if imgGT[i, j] in dataLabel:
					for k in range(len(dataLabel)):
						if imgGT[i][j] == dataLabel[k]:
							newGT[i, j] = k + 1
							continue
		imgGT = newGT
		return img, imgGT

def preprocess_split_dataset(img, imgGT, output):
	img = (img - float(np.min(img))) # preprocess l'image 
	img = img / np.max(img) # preprocess l'image 
	sampleNum = 200 # 
	deepth = img.shape[2]
	classNumber = np.max(imgGT) # le nombre de classe

	allData = {} 
	trainData = {}
	testData = {}

	# Les dictionnaire contiennent une liste pour chaque classes
	for i in range(1, classNumber+1):
		allData[i] = []
		trainData[i] = []
		testData[i] = []

	added = 0
	denied = 0
	# Pour chaque pixel, on l'ajoute à la liste de toute les 
	# données dans la bonne classe si une classe lui est attribué dans GT
	for i in range(imgGT.shape[0]):
		for j in range(imgGT.shape[1]):
			for k in range(1,classNumber+1):
				if imgGT[i,j] == k:
					allData[k].append(img[i,j])
					added += len(img[i,j])
				else:
					denied += len(img[i,j])


	allPixels = deepth * img.shape[0] * img.shape[1]
	ratio = (added * 100 ) / allPixels
	output.write("Dimensionality has been reduce from "+str(allPixels) 
								+ " pixels to " + str(added) + " pixels (" + str(ratio) + "% data of the total)")
	output.write("\n")

	# Pour chaque pixels qui est maintenant dans allData, 
	# on l'ajoute soit a train soit a test où k est l'indice de la classe
	for i in range(1,classNumber+1):
		indexies = random.sample(range(len(allData[i])),sampleNum) # selectionne un sample de 200 element de chaque liste
		for k in range(len(allData[i])):
			if k not in indexies:
				testData[i].append(allData[i][k])
			else:
				trainData[i].append(allData[i][k]) # le sample de 200 correspond aux données d'entrainements

	train = []
	trainLabel = []
	test = []
	testLabel = []

	# Remplir la liste d'entrainement et ses labels
	for i in range(1, len(trainData)+1):
		for j in range(len(trainData[i])):
			train.append(trainData[i][j])
			trainLabel.append(i)

	# Remplir la liste de test et ses labels
	for i in range(1, len(testData)+1):
		for j in range(len(testData[i])):
			test.append(testData[i][j])
			testLabel.append(i)
	return train, trainLabel, test, testLabel

def classify(img, imgGT, train, trainLabel, test, testLabel, whichDataset, output):

	clf = KNeighborsClassifier(n_neighbors=10)
	clf.fit(train, trainLabel) # Entraine un SVM
	classesNumber = np.max(imgGT)

	confMatrix = np.zeros((classesNumber, classesNumber)) # La matrice de confusion de taille nombreClasse X nombreClasse
	# Les lignes de resMat sont les resultats predits, les collones sont les verités de terrain
	for i in range(len(test)):
		results = clf.predict(test[i].reshape(-1, len(test[i])))
		confMatrix[results-1, testLabel[i]-1] += 1

	output.write("CONFUSION MATRIX : \n")
	output.write(str(np.int_(confMatrix)))
	output.write("\n")
	print(np.int_(confMatrix))

	totalPredicted = np.sum(np.trace(confMatrix))
	total = np.sum(confMatrix)
	totalAccu = (totalPredicted * 100) / total

	output.write(str(totalPredicted) + " pixels has been predicted correctly for a total of " + str(total) 
								+ " pixels (" + str(totalAccu) + "%)")
	output.write("\n")

	for i in range(len(confMatrix)): # La précision de chaque classe
		accuracy = confMatrix[i,i]/sum(confMatrix[:,i])
		print("accuracy ", i+1, " : ", accuracy)
		output.write("accuracy " + str(i+1) + ":" + str(accuracy))
		output.write('\n')


whichDataset = "indianPines"

time = str(datetime.now().strftime('%d%m%Y_%H%M%S'))
if not os.path.exists("./output"):
	os.makedirs("./output")
output = open("output/" + str(whichDataset) + str(time) + ".txt","w")


img, imgGT = load_dataset()
train, trainLabel, test, testLabel = preprocess_split_dataset(img, imgGT, output)
classify(img, imgGT, train, trainLabel, test, testLabel, whichDataset, output)

