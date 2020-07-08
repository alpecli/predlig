import numpy as np
import sklearn
import sys
import multiprocessing
from sklearn import svm
from sklearn.model_selection import cross_validate
from sklearn import tree
from sklearn import neighbors
from sklearn import naive_bayes
from sklearn import ensemble
from predlig.classifier import ParallelClassifier

class SupervisedLinkPrediction:
	"""
	A supervised learning link predictor.
	This class builds a classifier model based on the classification dataset provider, and uses this model to predict the existence
	or absence of a link in a set of pairs of nodes.
	"""
	def __init__(self, dataset, folds_number, classifier = "NB", classifier_params = {}, metric = "precision"):
		self.dataset = dataset
		self.classifier_params = classifier_params
		self.metric = metric
		self.folds_number = folds_number
		self.classifier = ParallelClassifier(classifier, classifier_params, metric)
		 
	def set_classifier(self, classifier, classifier_params, metric):
		self.classifier.set_classifier(classifier, classifier_params, metric)
	
	def set_folds_number(self, folds_number):
		self.folds_number = folds_number
	
	def apply_classifier(self):
		number_examples, number_attributes = self.dataset.shape
		examples = self.dataset[:,range(number_attributes-2)]
		examples_classes = self.dataset[:,[-2]]
		
		train_test_folds = []
		
		fold = np.zeros([number_examples])
		map_folds = {}
		
		for example in range(number_examples):
			fold[example] = self.dataset[example][-1]
			dict.setdefault(map_folds, fold[example], set())
			map_folds[fold[example]].add(example)
			
		for fold_examples in map_folds.values():
			train_test_folds.append((list(set(range(number_examples)) - fold_examples), list(fold_examples)))
		
		score = self.classifier.get_final_score(examples, examples_classes, train_test_folds)
		return score
