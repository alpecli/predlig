from sample import *
from evolutionary import *
from wrapper import *

sample = CDConstructor("flickr_al.txt", 10000, (0.5, 0.5))
sample.set_classification_dataset('flickr_final_version_attributes', 'flickr_final_version')
table = sample.get_classification_dataset()
number_of_attributes = len(sample.ordered_attributes_list)
message = "Dataset Flickr\n"
classifiers = {"NB": {}, "CART": {"random_state": 10}, "KNN": {"n_neighbors": 5}, "RFC": {"n_estimators": 10}, "SVM": {"gamma": 0.01, "C": 100.}}
metrics = ["precision", "f1", "roc_auc"]
for metric in metrics:
	message += "Metric: " + metric + "\n"
	message +=  "Classifier/Strategy;FFS;BaFE;BiFE\n"
	for classifier, classifier_params in classifiers.items():
		BiFE = BidirectionalFeatureSelection(number_of_attributes, classifier, classifier_params, metric, table[:, range(number_of_attributes)], table[:, number_of_attributes], sample.get_fold_list())
		BaFE = BackwardFeatureElimination(number_of_attributes, classifier, classifier_params, metric, table[:, range(number_of_attributes)], table[:, number_of_attributes], sample.get_fold_list())
		FFS = ForwardFeatureSelection(number_of_attributes, classifier, classifier_params, metric, table[:, range(number_of_attributes)], table[:, number_of_attributes], sample.get_fold_list())
		FFS.perform_feature_selection()
		BiFE.perform_feature_selection()
		BaFE.perform_feature_selection()
		message += "%s;%f/%s;%f/%s;%f/%s\n" %(classifier, FFS.best_score, FFS.best_var_subset, BaFE.best_score, BaFE.best_var_subset, BiFE.best_score, BiFE.best_var_subset)

print message

message = "Dataset Flickr\n"
for metric in metrics:
	message += "Metric: " + metric + "\n"
	message +=  "Classifier/Strategy;ES1;ES2;ES3;ES4\n"
	for classifier, classifier_params in classifiers.items():
		FS1 = EvolutionaryFeatureSelection(number_of_attributes, classifier, classifier_params, metric, table[:, range(number_of_attributes)], table[:, number_of_attributes], sample.get_fold_list(), 20, 100, 0.65, 0.05)
		FS2 = EvolutionaryFeatureSelection(number_of_attributes, classifier, classifier_params, metric, table[:, range(number_of_attributes)], table[:, number_of_attributes], sample.get_fold_list(), 20, 100, 0.65, 0.05, selection_function = "tournament", selection_parameters = {"tournsize": 3})
		FS3 = EvolutionaryFeatureSelection(number_of_attributes, classifier, classifier_params, metric, table[:, range(number_of_attributes)], table[:, number_of_attributes], sample.get_fold_list(), 50, 100, 0.65, 0.05, "steady_state_mode")
		FS4 = EvolutionaryFeatureSelection(number_of_attributes, classifier, classifier_params, metric, table[:, range(number_of_attributes)], table[:, number_of_attributes], sample.get_fold_list(), 30, 100, 0.65, 0.05, linear_normalization_generation = 10)
		FS1.perform_feature_selection()
		FS2.perform_feature_selection()
		FS3.perform_feature_selection()
		FS4.perform_feature_selection()
		message += classifier
		message += ";%s/%s" %(FS1.best_solution.fitness.values[0], [i for i in range(len(FS1.best_solution)) if FS1.best_solution[i] == 1])
		message += ";%s/%s" %(FS2.best_solution.fitness.values[0], [i for i in range(len(FS2.best_solution)) if FS2.best_solution[i] == 1])
		message += ";%s/%s" %(FS3.best_solution.fitness.values[0], [i for i in range(len(FS3.best_solution)) if FS3.best_solution[i] == 1])
		message += ";%s/%s\n" %(FS4.best_solution.fitness.values[0], [i for i in range(len(FS4.best_solution)) if FS4.best_solution[i] == 1])

print message
