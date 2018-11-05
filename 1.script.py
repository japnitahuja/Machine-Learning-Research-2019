import os

home = os.getcwd()

for i in os.listdir(home):
    path = os.path.join(home, i)
  
    if os.path.isdir(path):
        os.chdir(path)
       
        if os.path.exists(os.path.join(path, "dataset1.txt")):
            print(os.path.join(home, i,"dataset1.txt"))
            path = os.path.join(os.getcwd(), "dataset1")
            os.makedirs(path, exist_ok=True)
            os.system('python ecoc1.py')
        if os.path.exists(os.path.join(home, i,"dataset2.txt")):
            print(os.path.join(home, i,"dataset2.txt"))
            path = os.path.join(os.getcwd(), "dataset2")
            os.makedirs(path, exist_ok=True)
            os.system('python ecoc2.py')
        if os.path.exists(os.path.join(home, i,"dataset3.txt")):
            print(os.path.join(home, i,"dataset3.txt"))
            path = os.path.join(os.getcwd(), "dataset3")
            os.makedirs(path, exist_ok=True)
            os.system('python ecoc3.py')
            

            

            
