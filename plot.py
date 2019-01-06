import matplotlib.pyplot as plt
import numpy as np


#as deiksoyme prota to sinolo
data = open("/home/thanasis/Scala/outliers_detection/src/main/resources/" + "data" + ".csv")

x,y = [] , []

for line in data:
	d = line.split(",")
	if (len(d) ==2) and (d[1] not in ('\n','')) and (d[0] not in ('\n','')):
		x.append(float(d[0]))
		y.append(float(d[1]))

plt.figure(1)
plt.title("Data")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x,y,'go')
plt.show()

#meta to clustercccccccccccing
data = open("/home/thanasis/Scala/outliers_detection/results/Method B/" + "results"+ ".csv")

clusters = []
for i in range(5):
	clusters.append([[],[]])
for line in data:
	#print line
	d = line.split(",")
	if (len(d) ==4) and (d[1] not in ('')) and (d[0] not in ('')):
		clusters[int(d[2])][0].append(float(d[0]))
		clusters[int(d[2])][1].append(float(d[1]))
		#print d[2]
plt.figure(1)
plt.title("Clusters")
plt.xlabel("x")
plt.ylabel("y")
colors=[ 'r','b','g','y','c']
i=0
for cluster in clusters:
	print len(cluster[0])
	plt.plot(cluster[0],cluster[1],colors[i]+'o')
	i+=1
plt.show()


data = open("/home/thanasis/Scala/outliers_detection/results/Method A/" + "results" + ".csv")
x,y = [] , []
out_x , out_y = [] , []
for line in data:
	d = line.split(",")
	if (len(d) ==4) and (d[1] not in ('')) and (d[0] not in ('')):
		if  d[3] == "false\n":
			x.append(float(d[0]))
			y.append(float(d[1]))
		else:
			out_x.append(float(d[0]))
			out_y.append(float(d[1]))

fig = plt.figure(1)
fig.add_subplot(121) 
plt.title("Outliers- Method A")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x,y,'go')
plt.plot(out_x,out_y,'ro')


data = open("/home/thanasis/Scala/outliers_detection/results/Method B/" + "results" + ".csv")
x,y = [] , []
out_x , out_y = [] , []
for line in data:
	d = line.split(",")
	if (len(d) ==4) and (d[1] not in ('')) and (d[0] not in ('')):
		if  d[3] == "false\n":
			x.append(float(d[0]))
			y.append(float(d[1]))
		else:
			out_x.append(float(d[0]))
			out_y.append(float(d[1]))

#plt.figure(1)
fig.add_subplot(122)
plt.title("Outliers - Method B")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x,y,'go')
plt.plot(out_x,out_y,'ro')
plt.show()
