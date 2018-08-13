import csv
from sklearn import preprocessing
import numpy   
ohe = preprocessing.OneHotEncoder()
le = preprocessing.LabelEncoder()


with open("abalone.txt", "r") as f:
	reader = csv.reader(f)
	line = [row for row in reader]

with open("label.txt", "r") as f:
	reader = csv.reader(f)
	label = [row for row in reader]

label = numpy.array(label)
#print(label)

#number of attributes
m = len(line[1])
#number of instances
n = len(line)

flag_writer2 = 0
flag_writer3 = 0

#check if there is missing value(flag = 1)
flag = 0
for i in range(0,n):
	for j in range(0,m):
		if line[i][j] == '?':
			flag = 1

def process(x):

	#use OneHotEncoder to deal with categorical features			
	a = ''
	ohe.fit(x)
	#copy the class value
	for i in range(0,n):
		#print(x[i][m-1])
		a += x[i][m-1]
		x[i][m-1] = 0
	#print(type(x[1][1]))
	print(x)
	#print(a)

	'''
	b = []
	for i in range(0,m-1):
		if label[:,i] == '1':
			b += '1'
		else:
			b += '0' 
	b += '0'
	b = numpy.array(b)
	b = numpy.array(b,dtype=bool)
	print(b)
	preprocessing.OneHotEncoder(categorical_features=b)
	x = ohe.transform(x).toarray()
	'''
	y = x
	for j in range(0,m):
		y = ohe.transform(x).toarray()
		if label[:,i] == '0':
			cnt_j = 0
			for k in range(cnt_j,j):
				b[i][k] = x[i][k]
			b = ohe.fit_transform(b).toarray()
			

	print(x)

	#scale the dataset
	scaler = preprocessing.MinMaxScaler()
	#preprocessing.MinMaxScaler(copy=True, feature_range=(0, 1))
	x = scaler.fit_transform(x)

	#put back the class value
	m_x = len(x[0])
	for i in range(0,n):
		x[i][m_x-1] = a[i]
	print(x)
		
	if flag == 0:
		with open("dataset1.txt","w") as f:
			writer = csv.writer(f)
			writer.writerows(x)

	if flag_writer2 == 1:
		with open("dataset2.txt","w") as f:
			writer2 = csv.writer(f)
			writer2.writerows(x)

	if flag_writer3 == 1:
		with open('dataset3.txt','w') as f:
			writer3 = csv.writer(f)
			writer3.writerows(x)



#if there are missing values	
#there imputation methods
#1 delete all the instances with a missing value
if flag == 1:
	cnt = 0
	for i in range(0,n):
		flag2 = 0
		for j in range(0,m):
		
			if line[i][j] == '?':
				flag2 = 1
				break
		if flag2 == 1:
			continue
		x[cnt] = line[i]
		cnt += 1
	n = cnt + 1
	flag = 0

#use LabelEncoder to deal with categorical features
x = numpy.array(line)
for i in range(0, m):
#	print(label[:,i], '\n')
	if label[:,i] == '1':			
		a = le.fit_transform(x[:,i])
	#	print(a)
		for j in range(0, n):
			x[j][i] = a[j]
			#print(x[j][i], ' ')


#3 define the missing values as a new attribute value(if discrete) or else just use mean(continuous)
mean_imputer = preprocessing.Imputer(missing_values='?', strategy='mean')
y = ''
if flag == 1:
	y = mean_imputer(x)
	for j in range(0,m):
		#if continuous 
		if label[:,j] == '0':
			for i in range(0,n):
				x[i][j] = y[i][j]
	flag_writer3 = 1
	process(x)


#2 replace missing values with the mean(continuous) or mode(discrete)
mode_imputer = preprocessing.Imputer(missing_values='?', strategy='most_frequent')
y = ''
if flag == 1:
	for i in range(0,n):
		for j in range(0,m):
			if line[i][j] == '?':
				x[i][j] = '?'
	y = mode_imputer(x)
	for j in range(0,m):
		if label[:,j] == '1':
			for i in range(0,n):
				x[i][j] = y[i][j]
	y = mean_imputer(x)
	for j in range(0,m):
		#if continuous 
		if label[:,j] == '0':
			for i in range(0,n):
				x[i][j] = y[i][j]
	flag_writer2 = 1
	process(x)

#if there are no missing values
if flag == 0:
	#print(x)
	process(x)
	
