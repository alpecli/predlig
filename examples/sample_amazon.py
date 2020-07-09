from context import predlig
from predlig.sample import *

if __name__ == "__main__":
	sample = CDConstructor("../datasets/amazon_al.txt", 100, (0.5, 0.5))
	dataset = sample.get_sample()
	sample.save_sample('amazon_sample.txt')
	sample.get_classification_dataset()
	sample.save_classification_dataset("amazon_classification")
