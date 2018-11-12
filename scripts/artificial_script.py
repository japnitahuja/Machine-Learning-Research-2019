import os
import shutil

home = os.getcwd()
new_folder = os.path.join(home,"artificial_datasets_all")
os.makedirs(new_folder, exist_ok=True)

main_folder =  os.path.join(new_folder,"main_folder")
os.makedirs(main_folder, exist_ok=True)

img_folder = os.path.join(new_folder,"img_folder")
os.makedirs(img_folder, exist_ok=True)

artificial_dataset_folder = os.path.join(home,"artificial datasets")
count = 0

meta_data_folder = os.path.join(new_folder,"metadata")
os.makedirs(meta_data_folder, exist_ok=True)

#clear file
file_numbering_path = os.path.join(meta_data_folder,"0.numbering.txt")
file_numbering = open(file_numbering_path,"w")
file_numbering.close()


for folder in os.listdir(artificial_dataset_folder):
    
    if folder.startswith("."):
        continue

    
    
    if "img" not in folder:
        file_numbering = open(file_numbering_path,"a")
        file_numbering.write(folder+":"+str(count)+"\n")
        file_numbering.close()

        folder_path = os.path.join(artificial_dataset_folder,folder)
        for dataset in os.listdir(folder_path):

            if dataset.startswith("."):
                continue
    
            
            new_dataset = str(int(dataset.strip(".txt")) + count) + ".txt"
            
            jpg_file_name = str(int(dataset.strip(".txt"))) + ".png"
         
            new_jpg_file_name = new_dataset.strip(".txt") + "_img" + ".png"
            new_jpg_path = os.path.join(img_folder,new_jpg_file_name)
    
            temp = folder.split("_")
            jpg_folder = ""

            for i in range(len(temp)):
                jpg_folder += temp[i] + "_"
                if i == len(temp) - 2:
                    jpg_folder += "img" + "_"
                    
            jpg_folder = jpg_folder[:-1] 
            jpg_folder = os.path.join(artificial_dataset_folder,jpg_folder)
            jpg_img_path = os.path.join(jpg_folder,jpg_file_name)

            os.rename(jpg_img_path, new_jpg_path)

            #shutil.copy(jpg_img_path,os.path.join(main_folder) )

            file = open(os.path.join(folder_path,dataset),"r")
            file_op = os.path.join(main_folder,new_dataset)
            file_op = open(file_op,"w")
            
            for i in file:
                file_op.write(i)
            
            file.close()
            file_op.close()
            
        count += 200
    print(count)
       
