from context import predlig
from predlig.sample import *

sample = CDConstructor("flickr_al.txt", 10000, (0.5, 0.5))
dataset = sample.get_sample()
sample.save_sample('flickr_sample.txt')
sample.get_classification_dataset()
sample.save_classification_dataset('flickr_final_version')
