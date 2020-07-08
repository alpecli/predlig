from predlig.classifier import *

class WrapperFeatureSelection:
	def __init__(self):
		#raise NotImplementedError('Need to override this method')
		pass
	
	def first_solutions_generation(self):
		"""
		Returns if it is the first iteration of the feature selection strategy implemented.
		"""
		raise NotImplementedError('Need to override this method')
		
	def generate_initial_possible_solutions(self):
		"""
		Generates the first possible solutions for the feature selection strategy implemented.
		"""
		raise NotImplementedError('Need to override this method')
		
	def generate_new_possible_solutions(self):
		"""
		Generates a new set of possible solutions, usually the second or later genetation of the strategy.
		"""
		raise NotImplementedError('Need to override this method')
	
	def reached_stopping_criteria(self):
		"""
		Returns if the algorithm has reached the stopping criteria of the strategy.
		"""
		raise NotImplementedError('Need to override this method')
	
	def evaluate_possible_solutions(self):
		"""
		Evaluates the performance of the classification algorithm for each possible solution of the current generation.
		"""
		raise NotImplementedError('Need to override this method')
	
	def generate_possible_solutions(self):
		"""
		Generates the new generation of possible solutions, depending on the current state
		of the algorithm.
		"""
		if self.first_solutions_generation():
			self.generate_initial_possible_solutions()
		else:
			self.generate_new_possible_solutions()
	
	def perform_feature_selection(self):
		"""
		Performs the feature selection strategy indeed, by calling each function
		at a proper time.
		"""
		while not self.reached_stopping_criteria():
			self.generate_possible_solutions()
			self.evaluate_possible_solutions()
		
class BackwardFeatureElimination(WrapperFeatureSelection):
	def __init__(self, number_variables, classifier, classifier_params, metric, records, classes, folds):
		self.number_variables = number_variables
		self.classifier = ParallelClassifier(classifier, classifier_params, metric)
		self.records = records
		self.classes = classes
		self.folds = folds
		
		self.variable_subset = range(self.number_variables)
		self.best_score = None
		self.best_var_subset = []
		self.best_temp_score = None
		self.best_temp_var_subset = []
		self.possible_solutions = []
		
	def first_solutions_generation(self):
		return (self.best_temp_score == None)
	
	def generate_initial_possible_solutions(self):
		for variable in range(self.number_variables):
			variable_subset = list(self.variable_subset)
			variable_subset.remove(variable)
			self.possible_solutions.append(variable_subset)
			
		self.variable_subset = range(self.number_variables)
	
	def generate_new_possible_solutions(self):
		self.possible_solutions = []
		self.best_temp_score = None
		for variable in self.variable_subset:
			variable_subset = list(self.variable_subset)
			variable_subset.remove(variable)
			self.possible_solutions.append(variable_subset)
	
	def reached_stopping_criteria(self):
		return ((self.best_score != None and self.best_temp_score < self.best_score) or len(self.variable_subset) <= 1)
	
	def evaluate_possible_solutions(self):		
		for variable_subset in self.possible_solutions:
			records_subset = self.records[:, variable_subset]
			score = self.classifier.get_final_score(records_subset, self.classes, self.folds)
			if self.best_temp_score == None or score >= self.best_temp_score:
				self.best_temp_score = score
				self.best_temp_var_subset = list(variable_subset)
		
		if (self.best_score <= self.best_temp_score):
			self.best_score = self.best_temp_score
			self.best_var_subset = list(self.best_temp_var_subset)
			self.variable_subset = list(self.best_var_subset)

class ForwardFeatureSelection(WrapperFeatureSelection):
	def __init__(self, number_variables, classifier, classifier_params, metric, records, classes, folds):
		self.number_variables = number_variables
		self.classifier = ParallelClassifier(classifier, classifier_params, metric)
		self.records = records
		self.classes = classes
		self.folds = folds
		
		self.variable_subset = range(self.number_variables)
		self.best_score = None
		self.best_var_subset = []
		self.best_temp_score = None
		self.best_temp_var_subset = []
		self.possible_solutions = []
		
	def first_solutions_generation(self):
		return (self.best_temp_score == None)
	
	def generate_initial_possible_solutions(self):
		for variable in range(self.number_variables):
			variable_subset = set(self.best_var_subset)
			variable_subset.add(variable)
			variable_subset = list(variable_subset)
			self.possible_solutions.append(variable_subset)
	
	def generate_new_possible_solutions(self):
		self.possible_solutions = []
		self.best_temp_score = None
		for variable in self.variable_subset:
			variable_subset = set(self.best_var_subset)
			variable_subset.add(variable)
			variable_subset = list(variable_subset)
			self.possible_solutions.append(variable_subset)
			
	def reached_stopping_criteria(self):
		return  ((self.best_score != None and (self.best_temp_score < self.best_score or len(self.best_temp_var_subset) > len(self.best_var_subset))) or len(self.best_var_subset) >= self.number_variables)
		'''if self.best_score != None:
			return False
		elif self.best_temp_score == self.best_score:
			
		else:
			return True'''
	
	def evaluate_possible_solutions(self):
		for variable_subset in self.possible_solutions:
			records_subset = self.records[:, variable_subset]
			score = self.classifier.get_final_score(records_subset, self.classes, self.folds)
			if self.best_temp_score == None or score > self.best_temp_score:				
				self.best_temp_score = score
				self.best_temp_var_subset = list(variable_subset)
		
		if self.best_score == None or self.best_score < self.best_temp_score:
			self.best_score = self.best_temp_score
			self.best_var_subset = list(self.best_temp_var_subset)		
			list(set(self.variable_subset) - set(self.best_var_subset))
			self.variable_subset = list(set(self.variable_subset) - set(self.best_var_subset))

class BidirectionalFeatureSelection(WrapperFeatureSelection):
	def __init__(self, number_variables, classifier, classifier_params, metric, records, classes, folds):
		self.number_variables = number_variables
		self.classifier = ParallelClassifier(classifier, classifier_params, metric)
		self.records = records
		self.classes = classes
		self.folds = folds
		
		self.variable_subset = range(self.number_variables)
		self.best_score = None
		self.best_var_subset = []
		self.best_temp_score = None
		self.best_temp_var_subset = []
		self.worst_score = None
		self.worst_var_subset = []
		self.possible_solutions = []
		
	def first_solutions_generation(self):
		return (self.best_score == None)
	
	def generate_initial_possible_solutions(self):
		for variable in self.variable_subset:
			variable_subset = set(self.best_var_subset)
			variable_subset.add(variable)
			variable_subset = list(variable_subset)
			self.possible_solutions.append(variable_subset)
	
	def generate_new_possible_solutions(self):
		self.possible_solutions = []
		for variable in self.variable_subset:
			variable_subset = set(self.best_var_subset)
			variable_subset.add(variable)
			variable_subset = list(variable_subset)
			self.possible_solutions.append(variable_subset)
	
	def reached_stopping_criteria(self):
		return ((self.best_score != None and (self.best_temp_score < self.best_score or len(self.best_temp_var_subset) > len(self.best_var_subset))) or len(self.variable_subset) <= 1)
	
	def evaluate_possible_solutions(self):
		self.best_temp_score = None
		self.worst_score = None
		for variable_subset in self.possible_solutions:
			records_subset = self.records[:, variable_subset]
			score = self.classifier.get_final_score(records_subset, self.classes, self.folds)
			
			if self.best_temp_score == None or score > self.best_temp_score:
				self.best_temp_score = score
				self.best_temp_var_subset = list(variable_subset)
				
			if self.worst_score == None or score < self.worst_score:
				self.worst_score = score
				self.worst_var_subset = list(variable_subset)
		
		if self.best_score < self.best_temp_score:
			self.best_score = self.best_temp_score
			self.best_var_subset = list(self.best_temp_var_subset)
			self.variable_subset = list(set(self.variable_subset) - set(self.worst_var_subset) - set(self.best_var_subset))
