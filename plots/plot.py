import matplotlib.pyplot as plt
import numpy as np

# initial data
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

# clusters
data = open("/home/thanasis/Scala/outliers_detection/results/Method A1/" + "results"+ ".csv")

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
	plt.plot(cluster[0],cluster[1],colors[i]+'o')
	i+=1
plt.show()


# centers and circles
data = open("/home/thanasis/Scala/outliers_detection/results/Method A1/" + "results_scaled"+ ".csv")

centers = [[1.099180859665593,1.0851321460091121],
	[-1.0959456410773094,-1.0852360927261495],
	[1.103477705823583,-1.0942022329939314],
	[-0.004818114478098345,-0.009928735132038988],
	[-1.1018933644993796,1.1042378934634924]]
thresholds = [0.5039445441549318,
	0.4896088633757686,
	0.5083215872934504,
	0.5077786941120794,
	0.5178250954932317]

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
# no circles
plt.figure(1)
plt.title("Clusters (normalized data)")
plt.xlabel("x")
plt.ylabel("y")
colors=[ 'r','b','g','y','c']
i=0
for cluster in clusters:
	plt.plot(cluster[0],cluster[1],colors[i]+'o')
	i+=1
plt.show()

# with circles
plt.figure(1)
plt.title("Clusters - Method A1 (with centers and threshold)")
plt.xlabel("x")
plt.ylabel("y")
colors=[ 'r','b','g','y','c']
i=0
for cluster in clusters:
	plt.plot(cluster[0],cluster[1],colors[i]+'o')
	plt.plot(centers[i][0], centers[i][1], '+', mew=5,  ms=10, color = 'black')
	plt.gcf().gca().add_artist(plt.Circle((centers[i][0],centers[i][1]),thresholds[i], color='black', fill=False, zorder=100))
	i+=1
plt.show()

# circles+outliers
data = open("/home/thanasis/Scala/outliers_detection/results/Method A1/" + "results_scaled" + ".csv")
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

plt.figure(1)
plt.title("Outliers - Method A1 (with centers and threshold)")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x,y,'go')
plt.plot(out_x,out_y,'ro')
i=0
for cluster in clusters:
	plt.plot(centers[i][0], centers[i][1], '+', mew=5,  ms=10, color = 'black')
	plt.gcf().gca().add_artist(plt.Circle((centers[i][0],centers[i][1]),thresholds[i], color='black', fill=False, zorder=100))
	i+=1

plt.show()



# outliers - method a1
data = open("/home/thanasis/Scala/outliers_detection/results/Method A1/" + "results" + ".csv")
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

plt.figure(1)
plt.title("Outliers - Method A1 (1930 outliers)")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x,y,'go')
plt.plot(out_x,out_y,'ro')
plt.show()


# outliers - method a2
data = open("/home/thanasis/Scala/outliers_detection/results/Method A2/" + "results" + ".csv")
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

plt.figure(1)
plt.title("Outliers - Method A2 (2500 outliers)")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x,y,'go')
plt.plot(out_x,out_y,'ro')
plt.show()




# outliers - method b
data = open("/home/thanasis/Scala/outliers_detection/results/Method B/" + "results" + ".csv")
x,y = [] , []
out_x , out_y = [] , []
for line in data:
	d = line.split(",")
	if (len(d) ==3) and (d[1] not in ('')) and (d[0] not in ('')):
		if  d[2] == "false\n":
			x.append(float(d[0]))
			y.append(float(d[1]))
		else:
			out_x.append(float(d[0]))
			out_y.append(float(d[1]))

plt.figure(1)
plt.title("Outliers - Method B (4310 outliers)")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x,y,'go')
plt.plot(out_x,out_y,'ro')
plt.show()

# compare
# outliers - method a1
data = open("/home/thanasis/Scala/outliers_detection/results/Method A1/" + "results" + ".csv")
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
fig.add_subplot(131) 
plt.title("Outliers - Method A1 (1930 outliers)")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x,y,'go')
plt.plot(out_x,out_y,'ro')


# outliers - method a2
data = open("/home/thanasis/Scala/outliers_detection/results/Method A2/" + "results" + ".csv")
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
fig.add_subplot(132)
plt.title("Outliers - Method A2 (2500 outliers)")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x,y,'go')
plt.plot(out_x,out_y,'ro')


# outliers - method b
data = open("/home/thanasis/Scala/outliers_detection/results/Method B/" + "results" + ".csv")
x,y = [] , []
out_x , out_y = [] , []
for line in data:
	d = line.split(",")
	if (len(d) ==3) and (d[1] not in ('')) and (d[0] not in ('')):
		if  d[2] == "false\n":
			x.append(float(d[0]))
			y.append(float(d[1]))
		else:
			out_x.append(float(d[0]))
			out_y.append(float(d[1]))

#plt.figure(1)
fig.add_subplot(133)
plt.title("Outliers - Method B (4310 outliers)")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x,y,'go')
plt.plot(out_x,out_y,'ro')
plt.show()

