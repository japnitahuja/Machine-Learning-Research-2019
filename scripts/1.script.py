#ecoc script

import os
import numpy as np 
import csv
from itertools import product

def ecoc(k):
	file_name = 'dataset' + str(k) + '.txt'
	with open(file_name,'r') as f:
		reader = csv.reader(f)
		line = [row for row in reader]
		f.close()

	file_name1 = 'dataset' + str(k) + "/metadata.txt"
	op = open(file_name1,"w")

	#remove empty lines
	n = len(line)
	i = 0
	while(i < n):
		#print(i)
		if line[i] == []:
			line.remove(line[i])
			n -= 1
		i += 1

	x = np.array(line)
	m = len(line[0])
	n = len(line)
	'''	
	print(m)
	print(n)
	print(x[0][0])
	print(x[0][m-1])
	print(x[n-1][m-1])
	print(x)
	print(line)
	'''

	nclass = 0
	for i in range(0,n):
		#print(i)
		if float(x[i][m-1]) >= nclass:
			nclass = float(x[i][m-1])
	nclass = int(nclass)
	nclass += 1
	print(nclass)

	if nclass < 3:
		print("#class < 3")
		return

	if nclass > 15:
		print('#class > 15')
		return

	#get all the permutations of the classes
	code = list()		
	for combo in product(range(0,2),repeat = nclass):
		all0 = list()
		all1 = list()
		for i in range(0,nclass):
			all0.append(0)
			all1.append(1)
		all0 = tuple(all0)
		all1 = tuple(all1)
		if combo not in code and combo != all0 and combo != all1:
			code.append(combo)
			#print(combo)
		#print(code)

	ncode = len(code)
	code = np.array(code)
	#print(code)
	print(ncode)
	'''
	#get labels for all the classes
	label = [[]]
	for i in range(0,m):
		a = bin(2**i)
		for j in range(0,m):
			b = a % 2
			label[i].append(b)
			a = a / 2
	print(label)
	'''
	op.write("number of classes, number of instances, number of attributes\n")
	op.write(str(nclass) + ' ' + str(n) + ' ' + str(m)+ "\n")
	#change the class labels for all instances

	for i in range(0,ncode):
		print(code[i])
		op.write(str(i) + "," + str(code[i]) + "\n")
		x=np.array(line)
		for j in range(0,n):
			x[j][m-1] = float(code[i][int(float(x[j][m-1]))])
		i_str = str(i)
		file_name2 = 'dataset' + str(k)
		path = os.path.join(os.getcwd(), file_name2)
		os.makedirs(path, exist_ok=True)
		file_name = "dataset_ecoc" + i_str  + ".txt"
		print(file_name)
		path = os.path.join(os.getcwd(), file_name2,file_name)
		with open(path, 'w') as f:
			writer = csv.writer(f)
			writer.writerows(x)
			f.close()
	op.close()




home = os.getcwd()

for i in os.listdir(home):
    path = os.path.join(home, i)
  
    if os.path.isdir(path):
        os.chdir(path)
 
        if os.path.exists(os.path.join(home, i,"dataset1.txt")):
            print(os.path.join(home, i,"dataset1.txt"))
            path = os.path.join(os.getcwd(), "dataset1")
            os.makedirs(path, exist_ok=True)
            ecoc(1)
        if os.path.exists(os.path.join(home, i,"dataset2.txt")):
            print(os.path.join(home, i,"dataset2.txt"))
            path = os.path.join(os.getcwd(), "dataset2")
            os.makedirs(path, exist_ok=True)
            ecoc(2)
        if os.path.exists(os.path.join(home, i,"dataset3.txt")):
            print(os.path.join(home, i,"dataset3.txt"))
            path = os.path.join(os.getcwd(), "dataset3")
            os.makedirs(path, exist_ok=True)
            ecoc(3)
            

            

            
