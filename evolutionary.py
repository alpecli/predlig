from wrapper import *
from deap import base
from deap import creator
from deap import tools
from operator import attrgetter

import copy
import numpy as np
import random
from classifier import *

class EvolutionaryFeatureSelection(WrapperFeatureSelection):
	def __init__(self, number_variables, classifier, classifier_params, metric, records, classes, folds, pop_number, gen_number, crossover_rate, mutation_rate, selected_crossover = "generational_mode", ss_gap = 10, selection_function = "roulette", selection_parameters = {}, linear_normalization_generation = None):
		self.number_variables = number_variables
		self.classifier = ParallelClassifier(classifier, classifier_params, metric)
		self.records = records
		self.classes = classes
		self.folds = folds
		self.pop_number = pop_number
		self.pop = None
		self.gen_number = gen_number
		self.current_generation = 0
		self.crossover_rate = crossover_rate
		self.mutation_rate = mutation_rate
		
		self.selected_crossover = selected_crossover
		self.ss_gap = ss_gap
		
		self.crossover_functions = {"generational_mode": self.generational_mode_crossover,
									"steady_state_mode": self.steady_state_crossover}
		
		self.selection_function = selection_function
		self.selection_parameters = selection_parameters
		self.selection_functions = {"tournament": tools.selTournament, 
									"roulette": tools.selRoulette,
									"random": tools.selRandom}
		
		self.linear_normalization_generation = linear_normalization_generation
		
		self.best_solution = None
		self.offspring = []
		
		creator.create("FitnessMulti", base.Fitness, weights=(0.7, -0.3,))
		creator.create("Individual", list, fitness=creator.FitnessMulti)

		self.toolbox = base.Toolbox()
		# Attribute generator
		self.toolbox.register("attr_bool", random.randint, 0, 1)
		# Structure initializers
		self.toolbox.register("individual", tools.initRepeat, creator.Individual,
			self.toolbox.attr_bool, self.number_variables)
		self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
		
		self.toolbox.register("evaluate", self.evalMulti)
		self.toolbox.register("mate", tools.cxTwoPoint)
		self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
		self.toolbox.register("select", self.selection_functions[self.selection_function], **self.selection_parameters)
		
	def evalMulti(self, individual):
		count = 0
		subvar = list()
		for i in individual:
			if i == 1:
				subvar.append(count)
			count += 1

		if len(subvar) == 0:
			return 0, 100,
		else:
			r = self.records[:, subvar]
			score = self.classifier.get_final_score(r, self.classes, self.folds)
			return score, sum(individual),
	
	def first_solutions_generation(self):
		return (self.current_generation == 0)
	
	def reached_stopping_criteria(self):
		return (self.current_generation >= self.gen_number)
	
	def generate_initial_possible_solutions(self):
		self.pop = self.toolbox.population(n=self.pop_number)
		fitnesses = list(map(self.toolbox.evaluate, self.pop))
		for ind, fit in zip(self.pop, fitnesses):
			ind.fitness.values = fit
		
		self.current_generation = 1
		self.offspring[:] = self.pop
	
	def generate_new_possible_solutions(self):
		
		if self.linear_normalization_generation != None:
			if self.gen_number - self.current_generation <= self.linear_normalization_generation:
				self.toolbox.register("select", tools.selRandom)
		
		self.crossover_functions[self.selected_crossover]()
		
		self.current_generation += 1
	
	def evaluate_possible_solutions(self):
		
		invalid_ind = [ind for ind in self.offspring if not ind.fitness.valid]
		fitnesses = map(self.toolbox.evaluate, invalid_ind)
		for ind, fit in zip(invalid_ind, fitnesses):
			ind.fitness.values = fit
		
		if self.offspring != []:
			self.pop[:] = self.offspring
		
		self.steady_state_crossover()
		
		self.best_solution = tools.selBest(self.pop, 1)[0]
		
	def steady_state_crossover(self):
		self.offspring = list(map(self.toolbox.clone, self.pop))
		lista1 = list(self.offspring)
		children = []
		parents = tools.selRandom(self.offspring, self.ss_gap)
		for parent1, parent2 in zip(parents[::2], parents[1::2]):
			child1, child2 = self.toolbox.clone(parent1), self.toolbox.clone(parent2)
			self.toolbox.mate(child1, child2)

			if random.random() < self.mutation_rate:
				self.toolbox.mutate(child1)

			if random.random() < self.mutation_rate:
				self.toolbox.mutate(child2)
				
			del child1.fitness.values
			del child2.fitness.values
			
			children.append(child1)
			children.append(child2)
			
		for child in children:
			self.offspring.remove(tools.selRandom(self.offspring, 1)[0])
			self.offspring.append(child)
		
	def generational_mode_crossover(self):
		# Select the next generation individuals
		self.offspring = self.toolbox.select(self.pop, len(self.pop))
				
		# Clone the selected individuals
		self.offspring = list(map(self.toolbox.clone, self.offspring))
		for child1, child2 in zip(self.offspring[::2], self.offspring[1::2]):
			if random.random() < self.crossover_rate:
				self.toolbox.mate(child1, child2)
				del child1.fitness.values
				del child2.fitness.values

		for mutant in self.offspring:
			if random.random() < self.mutation_rate:
				self.toolbox.mutate(mutant)
				del mutant.fitness.values
		
		
if __name__ == "__main__":
	F = EvolutionaryFeatureSelection()
