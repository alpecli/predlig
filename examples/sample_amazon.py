from sample import *

sample = CDConstructor("amazon_al.txt", 10000, (0.5, 0.5))
dataset = sample.get_sample()
sample.save_sample('amazon_sample.txt')
sample.get_classification_dataset()
sample.save_classification_dataset("amazon_classification")
