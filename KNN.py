from functools import reduce
import math

class KNN:
	@staticmethod
	def BLOCK_DISTANCE(a, b):
		'''Or distance of Manhattan. Considers only 1-dim movement.'''
		distances_1d = [abs(x - y) for x, y in zip(a, b)]
		print(distances_1d)
		return sum(distances_1d)

	@staticmethod
	def EUCLIDEAN_DISTANCE(a, b):
		'''The shortest route between two points.'''
		distances_1d = [(x - y) ** 2 for x, y in zip(a, b)]
		return sum(distances_1d) ** (1/2)

	@staticmethod
	def NO_WEIGHT(acc, i):
		'''Just counts the frequency of every label.'''
		(label, _) = i
		if (label in acc):
			acc[label] += 1
		else:
			acc[label] = 1
		return acc

	@staticmethod
	def WEIGHTED(acc, i):
		'''Weights the values by their distance.'''
		(label, distance) = i
		if (label in acc):
			acc[label] += 1 / (distance ** 2)
		else:
			acc[label] = 1 / (distance ** 2)
		return acc

	def __init__(self):
		self.samples = []
		self.distance = KNN.EUCLIDEAN_DISTANCE
		self.k = 3
		self.type = KNN.NO_WEIGHT
		self.adaptive = False
	
	def calculateRadius(self, sample):
		'''Calculates the radius for the adaptive distance.
		If the distance for the classification is changed this function should be called again.'''
		distances = [(s[-1], self.distance(sample[0:-1], s[0:-1])) for s in self.samples]
		distances.sort(key = lambda i: i[1])
		for i in distances:
			if i[0] != sample[-1]:
				return i[1]
		return math.inf

	def train(self, training):
		'''Trains the kNN.'''
		self.samples = training
		self.sampleRadius = [self.calculateRadius(i) for i in self.samples]

	def classify(self, sample):
		'''Classifies a single sample.'''
		closest = self.find_closest(sample)
		votes = reduce(self.type, closest, dict())
		chosen = max(votes.keys(), key = lambda i: votes[i])
		return chosen
	
	def test(self, testSamples):
		'''Tests the kNN efficiency in classifying its samples.'''
		realResults = [i[-1] for i in testSamples]
		print(realResults)
		testResults = [self.classify(i) for i in testSamples]
		print(testResults)

	def find_closest(self, sample):
		'''Finds the closest k samples.'''
		if self.adaptive:
			distances = [(s[-1], self.distance(sample[0:-1], s[0:-1]) / self.sampleRadius[i])\
						for s in self.samples]
		else:
			distances = [(s[-1], self.distance(sample[0:-1], s[0:-1])) for s in self.samples]
		distances.sort(key = lambda i: i[1])
		return distances[0:self.k]

a = KNN()
a.train([[0, 1, 0], [1, 9, 1], [4, 8, 0], [2, 4, 1], [6, 3, 0]])
a.test([[0, 1, 0], [4, 3, 1], [9, 4, 0]])