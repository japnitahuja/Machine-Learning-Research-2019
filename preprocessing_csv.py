import csv
from sklearn import preprocessing
import numpy   
import scipy
ohe = preprocessing.OneHotEncoder(handle_unknown='ignore')
le = preprocessing.LabelEncoder()


with open("dataset.txt", "r") as f:
	reader = csv.reader(f)
	line = [row for row in reader]

with open("label.txt", "r") as f:
	reader = csv.reader(f)
	label = [row for row in reader]

label = numpy.array(label)
#print(label)

#number of attributes
m = len(line[0])
#m_temp = m
#number of instances
n = len(line)

print(n)
print(m)

flag_writer2 = 0
flag_writer3 = 0

#check if there is missing value(flag = 1)
flag = 0
for i in range(0,n):
	#print(i)
	for j in range(0,m):
		#print(j)
		if line[i][j] == '?':
			flag = 1
			break


def process(x):

	#use OneHotEncoder to deal with categorical features
	'''		
	a = ''
	#copy the class value
	for i in range(0,n):
		#print(x[i][m-1])
		global m
		a += x[i][m-1]
		x[i][m-1] = 0
	'''
	#print(type(x[1][1]))
	#print(x)
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
	global m
	x_temp = numpy.empty((n,0))
	
	for j in range(0,m-1):
		y = [0 for i in range(0,n)]
		if label[:,j] == '1':
			for i in range(0,n):
				#print("x ij")
				#print(x[i][j])
				y[i] = x[i][j]
				#print("yi")
				#print(y[i]) 
			y = numpy.array(y)
			#print("y1")
			#print(y)
			#reshape the array to a 2D matrix
			#print("y2")
			y = numpy.reshape(y,(-1,1))
			#print(y)
			#print(y)
			ohe.fit(y)
			#print(x_temp)
			#y1 = ohe.transform(y)#.toarray()
			#print(y1)
			#number of attributes gotten from ohe(y)
			'''
			b = scipy.sparse.csr_matrix.getnnz(y1,axis = 0)
			#print(b)
			len_y = len(b)
			'''
			#print(len_y)
			'''
			y = toarray()
			print(y)
			
			len_y = numpy.ndarray.shape(y)
			print(len_y)
			len_y = len_y[1]
			print(len_y)
			'''
			y = ohe.transform(y).toarray()
			'''
			m_temp = m
			m_temp += len_y-1
			#print(m)
			#print(m_temp)
			#print(x)
			x_temp = numpy.zeros((n,m_temp), dtype=numpy.float)

			for k in range(0,j):
				for i in range(0,n):
					x_temp[i][k] = x[i][k]
			for k in range(0,len_y):
				for i in range(0,n):
					x_temp[i][k+j] = y[i][k]
			for k in range(j+1,m):
				for i in range(0,n):
					x_temp[i][k+len_y-1] = x[i][k]
					'''
			#print(x_temp)
			shape = y.shape
			
			#x_temp = numpy.column_stack((x_temp,[1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8]))
			#print(shape)
			
			for k in range(0,shape[1]):
				#print("y:")
				#print(y)
				#y1 = numpy.reshape(y,(-1,1))
				#print(y1)
				#print("xtemp")
				#print(x_temp.shape)
				#print(x_temp)
				#print("yshape")
				#print(y.shape)
				#numpy.concatenate((x_temp,numpy.concatenate(y)[:,None]),axis=1)
				
				x_temp=numpy.column_stack((x_temp,y[:,k]))
			
	cnt = 0		
	for j in range(0,m-1):
		if label[:,j] == '1':
			x = numpy.delete(x,j-cnt,axis=1)
			cnt += 1
	#print(n)
	#x_class = numpy.empty((n,1))
	#print(x_class.shape)
	#print(len(x[:,1]))
	#print(x.shape)
	x_class = x[:,m-1-cnt]
	x = numpy.delete(x,m-1-cnt,axis=1)
	x = numpy.column_stack((x,x_temp))
	m = len(x[0])
	#print(x_class.shape)
	#print(x.shape)
	x = numpy.column_stack((x,x_class))

	#print(x)

	#scale the dataset
	scaler = preprocessing.MinMaxScaler()
	#preprocessing.MinMaxScaler(copy=True, feature_range=(0, 1))
	x = scaler.fit_transform(x)

	#put back the class value
	m_x = len(x[0])
	for i in range(0,n):
		x[i][m_x-1] = a[i]
	#print(x)
		
	if flag == 0:
		print("1")
		with open("dataset1.txt","w") as f:
			writer = csv.writer(f)
			writer.writerows(x)

	global flag_writer2
	if flag_writer2 == 1:
		print("2")
		with open("dataset2.txt","w") as f:
			writer2 = csv.writer(f)
			writer2.writerows(x)
		flag_writer2 = 0

	global flag_writer3
	if flag_writer3 == 1:
		print("3")
		with open('dataset3.txt','w') as f:
			writer3 = csv.writer(f)
			writer3.writerows(x)
		flag_writer3 = 0

x = numpy.array(line)


#if there are missing values	
#there imputation methods

#use LabelEncoder to deal with categorical features

#print(x)
#print(m)
#print(n)


#3 define the missing values as a new attribute value(if discrete) or else just use mean(continuous)
#mean
m_fix = m
mean_imputer = preprocessing.Imputer(strategy='mean')
y = ''
if flag == 1:
	for i1 in range(0,m):
		#print(i1)
		
		if label[:,i1] == '1':			
			a = le.fit_transform(x[:,i1])
		#	print(a)
			for j1 in range(0,n):
				x[j1][i1] = a[j1]
		
	#print(x)
	#print(n)
	#print(m)
	#print("XXXXXXXX")
	#print(x)
	for i in range(0,n):
		#print(i)
		for j in range(0,m):
			#print(x[i][j])
			if x[i][j] == '?':
			#	print('before', x[i][j])
				x[i][j] = numpy.nan
			#	print('after', x[i][j])				


	y = mean_imputer.fit_transform(x)

	#print(y)
	for j in range(0,m):
		#if continuous 
		if label[:,j] == '0':
			for i in range(0,n):
				x[i][j] = y[i][j]
	
	flag_writer3 = 1
	process(x)
#end
m = m_fix

#mode
#2 replace missing values with the mean(continuous) or mode(discrete)
mode_imputer = preprocessing.Imputer(strategy='most_frequent')
if flag == 1:
	#print('2')
	#print(m)
	#print(n)
	temp = line[0][0]
	line[0][0] = 100
	x = numpy.array(line)
	x[0][0] = temp
	line[0][0] = temp
	#print(x)
	'''
	x[0][0]="jjj"
	print(x)
	'''
	for i in range(0,n):
		#print(i)
		for j in range(0,m):
			if x[i][j] == '?':
				#print('before',x[i][j])
				x[i][j] = numpy.nan
				#print('after',x[i][j])


	for i1 in range(0,m):
		#print(i1)
		if label[:,i1] == '1':			
			a = le.fit_transform(x[:,i1])
		#	print(a)
			for j1 in range(0,n):
				x[j1][i1] = a[j1]
				#print(x[j][i], ' ')
			#print(j1)
	y = ''
	if flag == 1:
		y = mean_imputer.fit_transform(x)

		#print(y)
		for j in range(0,m):
			#if continuous 
			if label[:,j] == '0':
				for i in range(0,n):
					x[i][j] = y[i][j]

	for i in range(0,n):
		#print(i)
		for j in range(0,m):
			if line[i][j] == '?' and label[:,j] == '1':
				x[i][j] = numpy.nan
	#print(n)
	#print(m)
	y = mode_imputer.fit_transform(x)
	for j in range(0,m):
		if label[:,j] == '1':
			for i in range(0,n):
				x[i][j] = y[i][j]
	flag_writer2 = 1
	process(x)

#if there are no missing values
if flag == 0:
	x = numpy.array(line)
	m = len(line[0])
	for i1 in range(0,m):
		#print(i1)
		if label[:,i1] == '1':			
			a = le.fit_transform(x[:,i1])
		#	print(a)
			for j1 in range(0,n):
				x[j1][i1] = a[j1]
	process(x)
	

#1 delete all the instances with a missing value

'''
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
	print(x)
	process(x)
'''
m = m_fix
if flag == 1:
	#print('1')
	cnt = 0
	for i in range(0,n):
		flag2 = 0
		for j in range(0,m):
			if line[i][j] == '?':
				 #line[i] = '?'
				 x = numpy.delete(x,i-cnt,axis=0)
				 cnt += 1
				 break
	n -= cnt
	#print(n)
	#print('x length ',len(x[:,1]))
	#print('x shape',x.shape)
	'''
	y = numpy.empty((cnt,m), str)
	for i in range(0,n):
		if line[0] != '?':
			y = numpy.append(y,numpy.array(line[i]),axis = 0)
	'''
	flag = 0
	process(x)
