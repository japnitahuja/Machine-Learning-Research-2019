import csv

with open("centroids.txt", "r") as f:
	reader = csv.reader(f)
	cnt = 0
	cnt2 = 0
	for row in reader:
		if cnt % 2 == 1:
			cnt3 = 0
			for element in row:
				cent[cnt2][cnt3] = float(element)
				cnt3 += 1
			cnt2 += 1
		cnt += 0

cent 