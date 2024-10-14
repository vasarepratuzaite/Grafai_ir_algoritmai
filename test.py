import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

f = open("data.log", 'r')
f.readline()
n = 100
p = 0.01
mG = []
wG = []
avgwG = []
no_of_CC = []
nTa = []
mTa = []
wTa = []
avgwTa = []
nTb = []
mTb = []
wTb = []
avgwTb = []
T = []
for i in range(103):
	all = f.readline().split(' ')
	mG.append(int(all[1]))
	wG.append(float(all[2]))
	avgwG.append(float(all[3]))
	no_of_CC.append(int(all[4]))
	nTa.append(int(all[5]))
	mTa.append(int(all[6]))
	wTa.append(float(all[7]))
	avgwTa.append(float(all[8]))
	nTb.append(int(all[9]))
	mTb.append(int(all[10]))
	wTb.append(float(all[11]))
	avgwTb.append(float(all[12]))
	T.append(int(all[13]))
f.close()

plt.xlabel("wG")
plt.ylabel("avgwTb")
plt.scatter(wG, avgwTb, s=10)
# yplot = sorted(wG)
# yplot.reverse()
# zplot = sorted(avgwTa)
# z = np.polyfit(yplot, zplot, 9)
# p = np.poly1d(z)
f = lambda x,a : -np.log(x)+a
a = np.amax(avgwTa)
x = sorted(wG)
y = f(x,a)
plt.plot(x,y,"r--")
# plt.yticks(np.arange(min(no_of_CC), max(no_of_CC)+1, 1.0))
plt.show()