#the first layer: hill-climbing version of feature selection and linear regression
import numpy as np 
import csv
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import os

home = os.getcwd()
#need to change to either decision tree or random forest
path = os.path.join(home,"predicted_rf_test")
path_train = os.path.join(home,"predicted_rf")

reg = linear_model.LinearRegression()
mse = mean_squared_error

def sse(true, pred,n):
	#print(true)
	#print(pred)
	a = mse(true, pred, multioutput = 'raw_values')
	return (a*n)

for n0 in range(10):
	for fold in range(10):
		train_x_file = "train_x_" + str(n0) + "_" + str(fold) + ".txt"
		#need to change to either random forest or decision tree
		train_y_file = "train_rf_" + str(n0) + "_" + str(fold) + ".txt"
		'''
		test_x_file = "test_x_" + str(n0) + "_" + str(fold) + ".txt"
		test_y_file = "test_z_" + str(n0) + "_" + str(fold) + ".txt"
		'''
		print(train_x_file)

		with open(train_x_file,"r") as f:
			reader = csv.reader(f)
			line = [row for row in reader]
		with open(train_y_file,"r") as f:
			reader = csv.reader(f)
			count = 0
			for row in reader:
				line[count].append(row[0])
				count += 1

		x = np.array(line)
		n = len(line)
		m = len(line[0]) - 1
		#print(m)
		#print(n)


		for i in range(n):
			for j in range(m+1):
				if line[i][j] == "inf":
					line[i][j] = 1000000.0

		feat_potential = []
		for i in range(m):
			feat_potential.append(i)
		feat_old = []
		feat_new = []
		feat_old_coef = []
		feat_new_coef = []
		feat_old_true = []
		feat_old_pred = []
		flag = 1
		score = []
		score_min = None
		mean_sd = []
		y_pred = []
		y_true = []
		old_sse = None
		coef_temp = []
		feat_all = []
		#the hill-climbing linear regression
		while flag:
			for feature in feat_potential:
				y_pred_temp = []
				y_true_temp = []
				y_pred_temp = []
				y_true_temp = []
				feat_temp = []
				for i in range(len(feat_old)):
					feat_temp.append(feat_old[i])
				feat_temp.append(feature)
				data_m = [[] for i in range(n)]
				data_class = []
				#print(feat_temp)
				for i in range(n):
					for j in range(m):
						if j in feat_temp:
							data_m[i].append(line[i][j])
					data_class.append(line[i][m])
				#print(data_class)
				data_m = np.array(data_m)
				if len(data_m[0]) == 1:
					data_m.reshape(-1,1)
				#print(data_m)
				#print(len(data_class))


				#print(data_m)


				data_m = data_m.astype(np.float64)
				data_class = np.array(data_class)
				data_class = data_class.astype(np.float64)
				reg.fit(data_m,data_class)
				r_temp = pearsonr(data_class,reg.predict(data_m))
				score.append(r_temp[0]**2)
				
				#sd_temp = reg.predict(data_m)
				#sd_temp = sd_temp.reshape(-1,1)
				#sd_temp = list(sd_temp)
				#for i in range(len(data_class)):
					#sd_temp[i].append(data_class[i])
					
				sd_temp = np.append(reg.predict(data_m),data_class, axis = 0)
				mean_sd.append(np.mean(np.std(sd_temp)))
				coef_temp.append(reg.coef_)
				y_pred_temp = reg.predict(data_m)
				y_true_temp = (data_class)
				y_pred.append(list(y_pred_temp))
				y_true.append(list(y_true_temp))
				feat_all.append(feat_temp)
			#print(score)
			for i in range(len(score)):
				if score_min == None:
					score_min = i
				elif score[i] < score[score_min]:
					score_min = i
			#print(score_min)
			feat_new = feat_all[score_min]
			#print(coef_temp)
			#print(score_min)
			feat_new_coef = coef_temp[score_min]

			#print(old_sse)
			#print(y_true_temp)
			#print(y_pred_temp)
			if old_sse == None:
				old_sse = sse(y_true[score_min],y_pred[score_min],n)
			else: 
				f_s = (sse(y_true[score_min],y_pred[score_min],n) - old_sse)/mse(y_true[score_min],y_pred[score_min],multioutput = 'uniform_average')
			if feat_old == []:
				feat_old = feat_new
				feat_old_coef = feat_new_coef
				feat_old_true = y_true[score_min]
				feat_old_pred = y_pred[score_min]
				feat_potential.pop(feature)
				flag = 1
			else:
				if f_s > 4.0:
					feat_old = feat_new
					feat_old_coef = feat_new_coef
					feat_old_true = y_true[score_min]
					feat_old_pred = y_pred[score_min]
					flag = 1
					#print(len(feat_potential),i,feature)
					feat_potential.pop(feature)
				else:
					flag = 0
			#print(feat_old)
			#print("feat_new:  ",feat_new)
			if len(feat_potential) == 0:
				flag = 0
		#print(score)
		
		score_f = []
		score_f.append(n0)
		score_f.append(fold)
		score_f.append(score)
		'''
		with open("score_rf.txt","a") as f:
			writer = csv.writer(f)
			writer.writerow(score_f)
			'''
			

		data = [[] for i in range(n)]
		data_c = []
		for i in range(n):
			for j in feat_old:
				data[i].append(line[i][j])
			data_c.append(line[i][m])
		reg.fit(data,data_c)

		#save the test data predicted results
		test_x_file = "test_x_" + str(n0) + '_' + str(fold) + ".txt"
		with open(test_x_file,"r") as f:
			reader = csv.reader(f)
			test_x_line = [row for row in reader]
		test_x = [[] for i in range(len(test_x_line))]
		for i in range(len(test_x_line)):
			for j in feat_old:
				test_x[i].append(test_x_line[i][j])
		test_x = np.array(test_x).astype(np.float64)
		test_pred = reg.predict(test_x)
		os.chdir(path)
		test_pred_file = "pred_" + str(n0) + '_' + str(fold) + ".txt"
		with open(test_pred_file,"w") as f:
			writer = csv.writer(f)
			writer.writerow(test_pred)
		os.chdir(home)


		#save the r^2 and standard deviation. 
		result = []
		result.append(n0)
		result.append(fold)
		result.append(score[score_min])
		result.append(mean_sd[score_min])	
		with open("result_rf.txt","a") as f:
			writer = csv.writer(f)
			writer.writerow(result)
			#output the coefficient for the linear regression model and the result for feature selection
			#writer.writerow(feat_old_coef)
			#writer.writerow(feat_old)
			#writer.write(score[score_min])

		#save the train data predicted results
		os.chdir(path_train)
		pred_result = y_pred[score_min]
		pred_result = np.array(pred_result).reshape(-1,1)
		out_file = "pred_" + str(n0) + '_' + str(fold) + ".txt"
		with open(out_file,"w") as f:
			writer = csv.writer(f)
			writer.writerows(pred_result)
		os.chdir(home)

		