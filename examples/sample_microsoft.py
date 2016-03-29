from sample import *

sample = CDConstructor("microsoft_training.txt", 10000, (0.5, 0.5), "future_link_prediction", "microsoft_test.txt")
sample.get_sample()
sample.save_sample('microsoft_sample.txt')
sample.get_classification_dataset()
sample.save_classification_dataset("microsoft_classification")
