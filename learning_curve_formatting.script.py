file = open("learning_curves.txt").read().split("\n")
op = open("learning_curves_UCI_dt.txt","w")
count = 0
for i in file:
    if count == 0:
        i = i.strip(".txt")
        index = i.strip("\n")
        count += 1
        
    elif count == 1:
        count += 1
        continue
    
    elif count == 2:
        if ".txt" in i:
            count = 4
        else:
            count += 1
            acc = i[:-1]
        
    elif count == 3:
        count = 0
        op.write(index + "," + acc + "\n")
        continue

    elif count == 4:
        count = 0
        continue

op.close()

"""
op_r = open("learning_curves_UCI_dt.txt","r")

for i in op_r:
    print(i.split(","))

op_r.close()
"""
