#Rename the original file to dataset(original)
f = open("dataset(original).txt","r")
new_f = open("dataset.txt","w")
for i in f:
    s=""
    i = i.split(",")
    for x in range(1,len(i)):
        if "\n" in i[x]:
            s += str(i[x])
        else:
            s += str(i[x]) + ","
    new_f.write(s)
f.close()
new_f.close()
