import networkx as nx 
from networkx.algorithms import bipartite 
import random 
import numpy as np
import sys
import sklearn
import predlig.features
import predlig.spvlinkprediction as slp
from predlig.wrapper import *
from predlig.evolutionary import *
#from hypothesis_test import *

class CDConstructor:
	"""
	A Classification Dataset Constructor.
	
	This class extracts a sample of pairs of vertices from the graph that represents the original data, according to the defined parameters.
	It can be used for both future and missing link prediction tasks, represented as 0 or 1 in the 'task' variable, respectively.
	"""
	def __init__(self, dataset, sample_size, sample_proportions, task = "missing_link_prediction", future_dataset = None, k_fold = 10):
		self.dataset = dataset
		self.future_dataset = future_dataset
		self.sample_size = sample_size
		self.positive_examples_size = sample_proportions[0]
		self.negative_examples_size = sample_proportions[1]
		self.edges = set()
		self.positive_examples = set()
		self.negative_examples = set()
		self.original_graph = None
		self.graph_training = None
		self.graph_test = None
		self.task = task
		self.attributes_list = {}
		self.ordered_attributes_list = []
		self.k_fold = k_fold
		self.sample_dataset = None
		self.classification_dataset = None
		self.folds_indexes = {}
		
	def set_attributes_list(self, attributes_list):
		"""
		Stores the attributes list defined by the user in the attributes_list attribute
		of the class. It also keeps stored the sorted attribute list in another variable,
		in order to keep the same order of the attributes in the classification dataset.
		"""
		self.attributes_list = attributes_list
		self.ordered_attributes_list = sorted(self.attributes_list.keys())
	
	def get_future_link_graph(self):
		"""
		Returns the graph based on the future links dataset. It is used in the
		future link prediction task.
		"""
		Graph = nx.Graph()
		for line in open(self.future_dataset).readlines():
			line = line.strip()
			col = line.split(';')
			node1_id = int(col[0])
			node2_id = int(col[1])
			Graph.add_edge(node1_id, node2_id)
		return Graph
		
	def set_test_graph(self, graph_test):
		"""
		Defines the graph used for testing.
		"""
		self.graph_test = graph_test
	
	def is_subpath(self, path, first_node, second_node):
		"""
		Returns if a pair of nodes is part of a path.
		"""
		return ",".join(map(str, [first_node, second_node])) in ",".join(map(str, path)) or ",".join(map(str, [second_node, first_node])) in ",".join(map(str, path))
		
	def nodes_in_path(self, paths, first_node, second_node):
		""" 
		Verifies if a pair of nodes is part of some set of paths
		"""
		for path in paths:
			if self.is_subpath(path, first_node, second_node) or self.is_subpath(path, second_node, first_node):
				return True
		return False
	
	def get_pair_of_neighbors_nodes(self, Graph):
		"""
		Returns a random pair of neighbors from the graph.
		"""
		return random.sample(Graph.edges(), 1)[0]
	
	def get_total_edges(self):
		"""
		Returns the total edges in the collected sample.
		"""
		return self.positive_examples.union(self.negative_examples)
	
	def get_missing_link_positive_pair_of_nodes(self, paths):
		"""
		Returns a positive class pair of nodes for the missing link prediction task.
		"""
		while True:
			first_node, second_node = self.get_pair_of_neighbors_nodes(self.graph_training)
			if self.nodes_in_path(paths, first_node, second_node) or (first_node == second_node) or(first_node, second_node) in self.positive_examples:
				continue
				
			self.graph_training.remove_edge(first_node, second_node)
			if not nx.has_path(self.graph_training, first_node, second_node):
				self.graph_training.add_edge(first_node, second_node)
				continue
			
			return (first_node, second_node)
	
	def get_future_link_positive_pair_of_nodes(self, paths):
		"""
		Returns a positive class pair of nodes for the future link prediction task.
		"""
		while True:
			first_node, second_node = random.sample(self.graph_test.edges(), 1)[0]
			if self.graph_training.has_edge(first_node, second_node) or (first_node == second_node) or (first_node, second_node) in self.positive_examples or not self.graph_training.has_node(first_node) or not self.graph_training.has_node(second_node) or not nx.has_path(self.graph_training, first_node, second_node):
				continue
			return (first_node, second_node)
			
	
	def get_positive_examples(self):
		"""
		Extracts the pairs of nodes of the positive class from the graph. Those pairs of nodes have link(s) between them in the missing
		link prediction, or have link(s) in the future period of time in the future link prediction.
		"""
		paths = set()
		get_pair_of_nodes = self.get_future_link_positive_pair_of_nodes if self.task == "future_link_prediction" else self.get_missing_link_positive_pair_of_nodes
		while len(self.positive_examples) < (self.sample_size * self.positive_examples_size):
			edge = get_pair_of_nodes(paths)
			if edge not in self.positive_examples:
				paths.add(tuple(nx.shortest_path(self.graph_training, edge[0], edge[1])))
				self.positive_examples.add(edge)
	
	def get_missing_link_negative_pair_of_nodes(self):
		"""
		Returns a negative class pair of nodes for the missing link prediction task.
		"""
		while True:
			first_node, second_node = random.sample(self.graph_training.nodes(), 2)
			if not self.graph_training.has_edge(first_node, second_node) and not ((first_node, second_node) in self.negative_examples) and not (first_node == second_node) and nx.has_path(self.graph_training, first_node, second_node):
				return (first_node, second_node)
	
	def get_future_link_negative_pair_of_nodes(self):
		"""
		Returns a negative class pair of nodes for the future link prediction task.
		"""
		while True:
			first_node, second_node = self.get_missing_link_negative_pair_of_nodes()
			if not self.graph_test.has_edge(first_node, second_node):
				return (first_node, second_node)
	
	def get_negative_examples(self):
		"""
		Extracts the pairs of nodes of the negative class from the graph. Those pairs of nodes have no link between them in the missing
		link prediction, or have no link in the future period of time in the future link prediction.
		"""
		get_pair_of_nodes = self.get_future_link_negative_pair_of_nodes if self.task == "future_link_prediction" else self.get_missing_link_negative_pair_of_nodes
		while len(self.negative_examples) < (self.sample_size * self.negative_examples_size):
			self.negative_examples.add(get_pair_of_nodes())
			
	def get_sample(self):
		if self.sample_dataset == None:
			self.sample_dataset = self.extract_sample()
		return self.sample_dataset
	
	def set_sample(self, sample_file):
		self.sample_dataset = []
		for line in open(sample_file).readlines():
			self.sample_dataset.append([int(field) for field in line.split(';')])
		
		self.set_graphs()
	
	def save_sample(self, sample_file):
		writer = open(sample_file, 'w')
		for line in self.sample_dataset:
			writer.write("%s\n" %(";".join([str(column) for column in line])))
		writer.close()
		
	def set_graphs(self):
		self.original_graph = nx.Graph()
		dataset = open(self.dataset)
		for line in dataset.readlines():
			line = line.strip()
			col = line.split(';')
			iid = "p%s" %(int(col[0]))
			self.original_graph.add_node(iid, {'bipartite': 0})
			pid = int(col[1])
			self.original_graph.add_node(pid, {'bipartite': 1})
			self.original_graph.add_edge(iid, pid)
		
		dataset.close()

		nodes = set(n for n,d in self.original_graph.nodes(data=True) if d['bipartite'] == 1)
		
		self.graph_training = nx.Graph()
		for node in nodes:
			for intermediate_node in list(nx.all_neighbors(self.original_graph, node)):
				for second_node in list(set(nx.all_neighbors(self.original_graph, intermediate_node)) - set([node])):
					self.graph_training.add_edge(node, second_node)		
		
		self.graph_test = self.graph_training.copy() if self.task == "missing_link_prediction" else self.get_future_link_graph()
		
		edges = set()
		
		if (self.task == "missing_link_prediction") and (self.sample_dataset != None):
			for line in [line for line in self.sample_dataset if line[2] == 1]:
				node_1, node_2 = int(line[0]), int(line[1])
				try:
					self.graph_training.remove_edge(node_1, node_2)
				except:
					print(node_1, node_2)
					pass
		
	def extract_sample(self):
		"""
		Extracts the whole sample from the graph representing the original dataset. 
		"""
		
		self.set_graphs()
		
		self.get_positive_examples()
		self.get_negative_examples()
	
		sample_dataset = []
		line = 0
		fold_half_size = (self.sample_size/self.k_fold)/2
		
		for edge in self.positive_examples:
			fold = int(line/fold_half_size)
			sample_dataset.append([edge[0], edge[1], 1, fold])
			line += 1
			self.folds_indexes.setdefault(fold, set())
			self.folds_indexes[fold].add(line)
			
		line = 0
		for edge in self.negative_examples:
			fold = int(line/fold_half_size)
			sample_dataset.append([edge[0], edge[1], 0, fold])
			self.folds_indexes[fold].add(len(self.positive_examples) + line)
			line += 1
			
		return sample_dataset
	
	def get_fold_list(self):
		train_test_folds = []
		for fold, lines in self.folds_indexes.items():
			train_test_folds.append((list(set(range(self.sample_size)) - lines), list(lines)))
		
		return train_test_folds

	def normalize_attributes(self):
		"""
		Normalizes the values of the attributes by extracting the mean of the attribute and dividing the result by the standard deviation. 
		"""
		attributes_number = self.classification_dataset.shape[1] - 2
		examples = range(self.classification_dataset.shape[0])
		for attribute in range(attributes_number):
			std = np.std(self.classification_dataset[[examples],[attribute]])
			if std == 0:
				self.classification_dataset[[examples],[attribute]] = 0
			else:
				self.classification_dataset[[examples],[attribute]] = (self.classification_dataset[[examples],[attribute]] - self.classification_dataset[[examples],[attribute]].mean())/np.std(self.classification_dataset[[examples],[attribute]])
	
	def get_classification_dataset(self):
		if self.classification_dataset is None:
			self.classification_dataset = self.calculate_classification_dataset()
		return self.classification_dataset
	
	def set_classification_dataset(self, attributes_file, dataset_file):
		attributes_lines = open(attributes_file).readlines()
		
		for line in attributes_lines:
			fields = line.strip().split(';')
			attribute_name = fields[0]
			parameters = {}
			if fields[1] != "":
				for parameter in fields[1:]:
					parameter_name, parameter_value = parameter.split(':')
					parameters[parameter_name] = float(parameter_value)
			self.attributes_list[attribute_name] = parameters
			self.ordered_attributes_list.append(attribute_name)
		
		dataset_lines = open(dataset_file).readlines()
		self.classification_dataset = np.zeros((len(dataset_lines), len(attributes_lines) + 2))
		counter = 0
		self.folds_indexes = {}
		for line in open(dataset_file).readlines():
			self.classification_dataset[counter] = [float(field) for field in line.split(';')]
			fold = self.classification_dataset[counter][-1]
			self.folds_indexes.setdefault(fold, set())
			self.folds_indexes[fold].add(counter)
			counter += 1	
	
	def calculate_classification_dataset(self):
		"""
		Calculates the attributes for each example of the sample, and returns it as a matrix ready for applying the classification
		algorithms, in order to perform the link prediction.
		"""
		attributes_calculator = features.FeatureConstructor(self.graph_training)
		if self.attributes_list == {}:
			self.ordered_attributes_list = sorted(attributes_calculator.attributes_map.keys())
			for attribute in self.ordered_attributes_list:
				self.attributes_list[attribute] = {}
		
		classification_dataset = np.zeros((self.sample_size, len(self.attributes_list) + 2))
		line_count = 0
		
		for line in self.sample_dataset:
			first_node, second_node, pair_class, pair_fold = line
			attributes_calculator.set_nodes(first_node, second_node)
			column = 0
			for function in self.ordered_attributes_list:
				parameters = self.attributes_list[function]
				classification_dataset[line_count][column] = attributes_calculator.attributes_map[function](**parameters)
				column += 1
			classification_dataset[line_count][-2] = pair_class
			classification_dataset[line_count][-1] = pair_fold
			
			line_count += 1
		
		return classification_dataset
	
	def save_classification_dataset(self, dataset_file):
		writer = open(dataset_file + "_attributes", "w")
		for attribute in self.ordered_attributes_list:
			line = attribute + ";"
			line += ";".join(["%s:%s" %(parameter_name, parameter_value) for parameter_name, parameter_value in self.attributes_list[attribute]])
			writer.write(line + "\n")
		writer.close()
		
		writer = open(dataset_file, 'w')
		for line in self.classification_dataset:
			writer.write(";".join([str(field) for field in line]) + "\n")
		writer.close()
