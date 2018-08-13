import csv
from sklearn import preprocessing
import numpy as np
import scipy
ohe = preprocessing.OneHotEncoder()
le = preprocessing.LabelEncoder()


with open("dataset.txt", "r") as f:
	reader = csv.reader(f)
	line = [row for row in reader]

with open("label.txt", "r") as f:
	reader = csv.reader(f)
	label = [row for row in reader]

label = numpy.array(label)

#number of attributes
m = len(line[1])
#m_temp = m
#number of instances
n = len(line)

info = np.zeros((11), dtype=numpy.float)


#1 Statistical meta-features:
#1.a mean standard deviation



#1.b mean covariance



#1.c mean linear correlation coefficient



#1.d mean skewness



#1.e mean kurtosis



#2 Infomation-theoretic meta-features:
#2.a class entropy



#2.b mean attribute entropy



#2.c mean joint entropy



#2.d mean mutual infomation 



#2.e equivalent number of attributes



#2.f signal to noise ratio



