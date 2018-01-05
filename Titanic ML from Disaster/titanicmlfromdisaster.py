
import pandas as pd
import numpy as np

#Some of the feature extraction ideas have either been inspired or taken as is from Sina's best titanic working classifier kernel. 
#Will update through comments which snippet is taken and which is not

#Since the
#description of the competition mentions that women and
# children have a more chance to survive, we will engineer the features accordingly.

#Also, we will check if the existence of cabins has any impact on the outcome or not