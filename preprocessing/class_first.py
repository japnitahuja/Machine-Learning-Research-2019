#Rename the original file to dataset(original)
# class first fix
f = open("dataset(original).txt","r")
new_f = open("dataset.txt","w")
for i in f:
    s=""
    i = i.strip("\n")
    i = i.split(",")
    for x in range(1,len(i)):
        s += str(i[x]) + ","
    s += i[0] + "\n"
    new_f.write(s)
f.close()
new_f.close()
