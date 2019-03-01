import csv

with open("cluster_wise_accuracy.txt", "r") as f:
	reader = csv.reader(f)
	line = [row for row in reader]

print(line)
cnt = 0
temp = []
while True:
	temp.append(line[cnt][3])
	if cnt % 4 == 3:
		print(temp)
		temp = []
	if cnt == 40:
		break
	cnt += 1
