library(ECoL)


home = getwd()

dataset_folder = file.path(home ,"baseline_datasets")
count <- 0
for (dataset in list.files(dataset_folder))
{
  count <- count + 1
  print(dataset)
  mydata <- read.csv(file.path(dataset_folder,dataset), header = FALSE)
  file_op <- file.path(home , "/complexity_baseline.txt")
  m = length(mydata)
  error = try(complexity(mydata[,1:(m-1)],mydata[,m],type="class"))
  print(class(error))
  if (class(error) != "try-error")
  {
    print("DONE")
    write.table(dataset, file_op,sep = ",",append=T)
    write.table(complexity(mydata[,1:(m-1)],mydata[,m],type="class"), file_op,sep = ",",append=T)
  }
  print(count)
  
}


  





