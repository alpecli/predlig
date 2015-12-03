import numpy as np
import sklearn
import sys
import multiprocessing
from sklearn import svm
from sklearn import cross_validation
from sklearn import tree
from sklearn import neighbors
from sklearn import naive_bayes
from sklearn import ensemble
from sklearn import neural_network

class ParallelClassifier:
	"""
	A machine learning classifier class.
	It stores the kind of classifier to be used with it parameters and the metric that will be
	used to evaluate the classifier performance. For each fold previously defined, this class
	will create a process with a classifier responsible for training and testing the correspoding
	fold of the dataset. The final performance is the mean of the performance of each classifier
	process.
	"""
	def __init__(self, classifier, classifier_params, metric):
		"""
		Defines the classifier, its parameters and the metric that will be used to
		evaluate the classifier.
		"""
		self.classifier = classifier
		self.classifier_params = classifier_params
		self.classifiers = {
		 "CART": tree.DecisionTreeClassifier, 
		 "SVM": svm.SVC, 
		 "KNN": neighbors.KNeighborsClassifier, 
		 "NB": naive_bayes.GaussianNB, 
		 "RFC": ensemble.RandomForestClassifier,
		 "MLP": neural_network.MLPClassifier
		 }
		self.metric = metric
		self.metrics = {
			"accuracy": sklearn.metrics.accuracy_score,
			"precision": sklearn.metrics.precision_score,
			"recall": sklearn.metrics.recall_score,
			"f1": sklearn.metrics.f1_score,
			"roc_auc": sklearn.metrics.roc_auc_score
		 }
	
	def set_classifier(self, classifier, classifier_params, metric):
		"""
		Defines the classifier, its parameters and the evaluation performance metric.
		"""
		self.classifier = classifier
		self.classifier_params = classifier_params
		self.metric = metric 
		
	def get_classifier_score(self, dataset, pair_class, fold, out_score):
		"""
		Returns the performance of a classifier process.
		"""
		trained_classifier = self.classifiers[self.classifier](**self.classifier_params).fit(dataset[fold[0]], pair_class[fold[0]])
		score = self.metrics[self.metric](pair_class[fold[1]], trained_classifier.predict(dataset[fold[1]]))
		out_score.put(score)
	
	def get_final_score(self, dataset, pair_class, folds):
		"""
		Returns the mean performance of the classifier processes created for each fold of
		the dataset.
		"""
		out_q = multiprocessing.Queue()
		procs = []
		nprocs = len(folds)
			
		for fold in folds:
			p = multiprocessing.Process(target=self.get_classifier_score, args=(dataset, pair_class, fold, out_q))
			procs.append(p)
			p.start()
				
		result = []
		for i in range(nprocs):
			result.append(out_q.get())
				
		for p in procs:
			p.join()
			
		score = sum(result)/len(result)
		
		return score
