import os 
import csv

home = os.getcwd()

f1 = open("file.txt",'a')

cnt = 0
count = 0
for i in os.listdir(home):

	path = os.path.join(home, i)
	folder = "main_folder"
	main_folder = os.path.join(home,folder)

    #go into every dataset folder
	if os.path.isdir(path) and i != folder:
		os.chdir(path)
		#print("yes,there is a dataset folder")
		dataset_name = i
		f1.write(dataset_name + ',' + str(count + 1) + ',')

		#go to dataset1&2&3 folder
		flag = 0
		for k in range(1,4):
			folder_name = "dataset" + str(k)
			cnt = 0
			
			#go into dataset1/2/3
			if os.path.exists(os.path.join(path, folder_name)):
				path_d = os.path.join(path, folder_name)
				os.chdir(path_d)
				print(path_d)
				flag = 1

				#copy the metadata.txt to the big folder
				if k == 1:
					d = []
					file_name = "metadata.txt"
					with open(file_name,"r") as f:
						reader = csv.reader(f)
						for row in reader:
							if len(row) != 0:
								d.append(row)

					folder_name = "metadata"
					os.chdir(os.path.join(home, folder_name))
					file_name = str(count+1) + ".txt"
					with open(file_name, 'w') as f:
						writer = csv.writer(f)
						writer.writerows(d)
					os.chdir(path_d)

				#check every ecoc file
				for j in os.listdir(path_d):
					#print(j)
					if j.startswith("dataset_ecoc"):
						#calculate the name of the file
						no_file = j[:(len(j)-4)]
						no_file = no_file[12:]
						#print(no_file)
						newname = str(int(no_file) + count) + ".txt"
						#print(newname)
						cnt += 1
						#print(newname)
						#print(j)

						#copy the ecoc file
						d = []
						with open(j,'r') as f:
							reader = csv.reader(f)
							for row in reader:
								if len(row) != 0:
									d.append(row)

						#paste the dataset to the main folder
						os.chdir(main_folder)
						with open(newname,'w') as f:
							writer = csv.writer(f)
							writer.writerows(d)

					#go back to the ecoc folder(dataset1/2/3)
					os.chdir(path_d)
				count += cnt
				#print(count)
		if flag == 0:
			for k in range(1,4):
				file_name = "dataset" + str(k) + ".txt"
				if os.path.exists(os.path.join(path, file_name)):
					newname = str(k + count) + ".txt" 
					cnt += 1

					d = []
					with open(file_name,'r') as f:
						reader = csv.reader(f)
						for row in reader:
							if len(row) != 0:
								d.append(row)

					os.chdir(main_folder)
					with open(newname,'w') as f:
						writer = csv.writer(f)
						writer.writerows(d)
					os.chdir(path)
			count += cnt

		f1.write(str(count) + '\n')

f1.close()