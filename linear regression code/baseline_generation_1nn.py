import csv
import os
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier


home = os.getcwd()
train_dt_path = os.path.join(home,"predicted_dt")
train_rf_path = os.path.join(home,"predicted_rf")
test_dt_path = os.path.join(home,"predicted_dt_test")
test_rf_path = os.path.join(home,"predicted_rf_test")

for k in range(1):
	all_f = []
	classifier = KNeighborsClassifier(n_neighbors=1)
	for n0 in range(10):
		for fold in range(10):
			train_file = "train_y_" + str(n0) + "_" + str(fold) + ".txt"
			y_file = "test_y_" + str(n0) + "_" + str(fold) + ".txt"
			x_acc = "pred_" + str(n0) + '_' + str(fold) + ".txt"

			print(y_file)

			with open(y_file,"r") as f:
				reader = csv.reader(f)
				test_y = [row for row in reader]

			with open(train_file,"r") as f:
				reader = csv.reader(f)
				train_y = [row for row in reader]

			train_x_file = "train_x_" + str(n0) + "_" + str(fold) + ".txt"
			test_x_file = "test_x_" + str(n0) + "_" + str(fold) + ".txt"

			with open(train_x_file,"r") as f:
				reader = csv.reader(f)
				train_mf = [row for row in reader]

			with open(test_x_file,"r") as f:
				reader = csv.reader(f)
				test_mf = [row for row in reader]

			os.chdir(train_dt_path)
			with open(x_acc,"r") as f:
				reader = csv.reader(f)
				train_dt_acc = [row for row in reader]

			os.chdir(train_rf_path)
			with open(x_acc,"r") as f:
				reader = csv.reader(f)
				train_rf_acc = [row for row in reader]

			os.chdir(test_dt_path)
			with open(x_acc,"r") as f:
				reader = csv.reader(f)
				test_dt_acc = [row for row in reader]

			os.chdir(test_rf_path)
			with open(x_acc,"r") as f:
				reader = csv.reader(f)
				test_rf_acc = [row for row in reader]

			os.chdir(home)

			n = len(train_mf)
			m = len(train_mf[0])

			train_x = []
			for i in range(n):
				row = []
				for j in range(m):
					row.append(train_mf[i][j])
				row.append(train_dt_acc[i][0])
				row.append(train_rf_acc[i][0])
				train_x.append(row)

			for i in range(len(train_x)):
				for j in range(len(train_x[0])):
					if train_x[i][j] == "inf":
						train_x[i][j] = 100000.0

			train_x = np.array(train_x).astype(np.float64)
			train_y = np.array(train_y).reshape(len(train_y))
			train_y = np.array(train_y).astype(np.float64)

			n = len(test_mf)
			test_x = []
			for i in range(n):
				row = []
				for j in range(m):
					row.append(test_mf[i][j])
				row.append(test_dt_acc[0][i])
				row.append(test_rf_acc[0][i])
				test_x.append(row)

			for i in range(len(test_x)):
				for j in range(len(test_x[0])):
					if test_x[i][j] == "inf":
	 					test_x[i][j] = 100000.0

			test_x = np.array(test_x)
			test_x = test_x.astype(np.float64)
			test_y = np.array(test_y).reshape(len(test_y))
			test_y = test_y.astype(np.float64)
			
			classifier.fit(train_x,train_y)
			acc = classifier.score(test_x,test_y)

			all_f.append(acc)

	result = sum(all_f)/len(all_f)

	with open("baseline_1nn.txt","a") as f:
		f.write(str(k))
		f.write(str(result))
		
		f.write("\n")
		writer = csv.writer(f)
		writer.writerow(all_f)
		