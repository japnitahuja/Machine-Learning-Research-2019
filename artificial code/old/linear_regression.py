import numpy as np 
import csv
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

debug = True

reg = linear_model.LinearRegression()
mse = mean_squared_error

def sse(true, pred):
	a = mse(true, pred, multioutput = 'raw_values')
	return sum(a)

with open("complexity_cluster_format_normal.txt", "r") as f:
	reader = csv.reader(f)
	cluster_wise_complexity = [row for row in reader]
	cluster_wise_complexity = np.array(cluster_wise_complexity).astype(np.float64)

with open("cluster_wise_accuracy.txt", "r") as f:
	reader = csv.reader(f)
	cluster_wise_accuracy = [row for row in reader]
	cluster_wise_accuracy = np.array(cluster_wise_accuracy).astype(np.float64)

with open("complexity_baseline_format_normal.txt", "r") as f:
	reader = csv.reader(f)
	baseline_complexity = [row for row in reader]
	baseline_complexity = np.array(baseline_complexity).astype(np.float64)
'''
with open("baseline_accuracy.txt", "r") as f:
	reader = csv.reader(f)
	baseline_accuracy = [row for row in reader]
	baseline_accuracy = np.array(baseline_accuracy).astype(np.float64)
'''
with open("hybrid_overall_acc.txt", "r") as f:
	reader = csv.reader(f)
	hybrid_overall_acc = [row for row in reader]

baseline_accuracy = hybrid_overall_acc[2][:]
baseline_accuracy = np.array(baseline_accuracy).astype(np.float64)
print(baseline_accuracy)



'''
#calculate mean baseline complexity
baseline_complexity_mean = [[0 for i in range(22)] for j in range(10)]
for feature in range(22):
	for i in range(10):
		for j in range(10):
			baseline_complexity_mean[i][feature] += baseline_complexity[i*10+j][feature]
	baseline_complexity[i][feature] /= 10

#calculate mean baseline accuracy
baseline_accuracy_mean = [0 for i in range(10)]
for i in range(10):
	for j in range(10):
		baseline_accuracy_mean[i] += baseline_accuracy[i*10+j] 
	baseline_accuracy_mean /= 10
'''

#calculate difference between baseline complexity and cluster wise complexity
for feature in range(22):
	for n0 in range(100):
		for cluster in range(4):
			cluster_wise_complexity[n0*4+cluster][feature] -= baseline_complexity[n0][feature]

#calculate difference between baseline accuracy and cluster wise accuracy
for n0 in range(100):
	for cluster in range(4):
		cluster_wise_accuracy[n0*4+cluster][3] -= baseline_accuracy[n0]

#form the dataset for linear regression model
line = []
for i in cluster_wise_complexity:
	line.append(i)
for i in range(len(cluster_wise_accuracy)):
	line[i] += cluster_wise_accuracy[i][3]

n = len(line)
m = len(line[0]) - 1
#print(m)
#print(n)


feat_potential = []
for i in range(m):
	feat_potential.append(i)
feat_old = []
feat_new = []
feat_old_coef = []
feat_new_coef = []
feat_old_true = []
feat_old_pred = []
y_true = []
y_pred = []
flag = 1
score = []
score_max = None
old_sse = None
coef_temp = []
feat_all = []
#the hill-climbing linear regression
while flag:
	y_pred_temp = []
	y_true_temp = []
	y_pred_temp = []
	y_true_temp = []
	for feature in feat_potential:
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
		data_m = data_m.astype(np.float64)
		data_class = np.array(data_class)
		data_class = data_class.astype(np.float64)
		reg.fit(data_m,data_class)
		r_temp = pearsonr(data_class,reg.predict(data_m))
		score.append(r_temp[0]**2)
		coef_temp.append(reg.coef_)
		y_pred_temp.append(reg.predict(data_m))
		y_true_temp.append(data_class)
		y_pred.append(y_pred_temp)
		y_true.append(y_true_temp)
		feat_all.append(feat_temp)
	'''
	if debug:
		print("score")
		print(score)
		'''
	for i in range(len(score)):
		if score_max == None:
			score_max = i
		elif score[i] > score[score_max]:
			score_max = i
	feat_new = feat_all[score_max]
	if debug:
		#print(coef_temp)
		print("score_max", score_max, score[score_max])
	feat_new_coef = coef_temp[score_max]

	if old_sse == None:
		old_sse = sse(y_true_temp,y_pred_temp)
	else: 
		f_s = (old_sse - sse(y_true_temp,y_pred_temp))/mse(y_true_temp,y_pred_temp,multioutput = 'uniform_average')
		old_sse = sse(y_true_temp,y_pred_temp)
		if debug:
			print("f_score", f_s)

	if debug:
		print("feat_old: ", feat_old)

	if feat_old == []:
		feat_old = feat_new
		feat_old_coef = feat_new_coef
		feat_old_true = y_true_temp[score_max]
		feat_old_pred = y_pred_temp[score_max]
		for i in range(len(feat_potential)):
			if feat_potential[i] in feat_old:
				feat_potential.pop(i)
				break
		flag = 1
	else:
		if f_s > 4.0:
			feat_old = feat_new
			feat_old_coef = feat_new_coef
			feat_old_true = y_true[score_max]
			feat_old_pred = y_pred[score_max]
			flag = 1
			for i in range(len(feat_potential)):
				if feat_potential[i] in feat_old:
					feat_potential.pop(i)
					break
		else:
			flag = 0
	if debug:
		print("feat_new:  ",feat_new)
	if len(feat_potential) == 0:
		flag = 0

#output the coefficient for the linear regression model and the result for feature selection
with open("coefficient.txt","w") as f:
	writer = csv.writer(f)
	writer.writerow(feat_old_coef)
	writer.writerow(feat_old)
	temp = []
	temp.append(score[score_max])
	writer.writerow(temp)
