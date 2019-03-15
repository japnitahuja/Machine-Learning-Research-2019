import csv

with open("result_baseline.txt", "r") as f:
	reader = csv.reader(f)
	line = [row for row in reader]

with open("result_cluster_wise.txt", "r") as f:
	reader = csv.reader(f)
	line2 = [row for row in reader]

result_baseline = [[] for i in range(8)]
result_cluster_wise = [[[] for j in range(4)] for i in range(8)]

for i in range(len(line)):
	for j in range(len(line[0])):
		result_baseline[j].append(line[i][j])

for i in range(len(line)):
	cluster = i % 4
	for j in range(len(line[0])):
		result_cluster_wise[j][cluster].append(line[i][j])


with open("result_baseline_format.csv", "w") as f:
	writer = csv.writer(f)
	writer.writerows(result_baseline)

with open("result_cluster_wise_format.csv", "w") as f:
	writer = csv.writer(f)
	for i in range(8):
		for j in range(4):
			writer.writerow(result_cluster_wise[i][j])
