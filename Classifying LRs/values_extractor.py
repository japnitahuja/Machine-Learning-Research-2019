import json
from pprint import pprint

with open('_values.json') as data_file:    
    data = json.load(data_file)

data_file.close()
temp = []

opfile = open("values.txt","w")
    
for i in range(len(data)):
    name = data[i]["dataset"]
    a = data[i]["result"]["Random Forest"]["coefficients"][0]
    b = data [i]["result"]["Random Forest"]["coefficients"][1]
    alpha = data [i]["result"]["Random Forest"]["coefficients"][2]
    values = str(a) + "," + str(b) + "," + str(alpha) + "," + str(name) + "\n"
    opfile.write(values)
    print(i)
opfile.close()
