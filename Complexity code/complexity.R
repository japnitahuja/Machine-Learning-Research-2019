library(ECoL)


home = getwd()

dataset_folder = file.path(home ,"main_folder")

for (dataset in list.files(dataset_folder))
{
  print(dataset)
  mydata <- read.csv(file.path(dataset_folder,dataset), header = FALSE)
  file_op <- file.path(home , "/complexity.txt")
  write.table(dataset, file_op,sep = ",",append=T)
  write.table(complexity(mydata[,1:2],mydata[,3],type="class"), file_op,sep = ",",append=T)
}


  





