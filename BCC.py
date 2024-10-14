from operator import itemgetter
from numpy import random as rnd
from numpy import amin
from numpy import matrix
from math import inf
from copy import deepcopy
from copy import copy
import matplotlib.pyplot as plt
import networkx as nx
from string import ascii_lowercase
from matplotlib.patches import FancyArrowPatch
import time

class Graph:
	def __init__(self, V=[str(i) for i in range(5)]):
		self.n = len(V)
		self.Vnames = V
		self.adjlist = dict()
		for i in self.Vnames:
			self.adjlist[i] = dict()
		self.update_adjlist()
		
	def add_node(self, node='z'):
		self.n = self.n + 1
		self.adjlist[node] = dict()
		self.Vnames.append(node)
		
	def remove_node(self, v):
		for i in self.adjlist:
			if v in self.adjlist[i]:
				del self.adjlist[i][v]
		del self.adjlist[v]
		for i in range(len(self.Vnames)):
			if self.Vnames[i] == v:
				del self.Vnames[i]
				self.n = self.n - 1
				break
	
	def remove_edge(self, v, u):
		if v in self.adjlist and u in self.adjlist:
			if v in self.adjlist[u] and u in self.adjlist[v]:
				del self.adjlist[v][u]
				del self.adjlist[u][v]

	def add_edge(self, u, v, w=1):
		self.adjlist[v][u] = w
		self.adjlist[u][v] = w
		self.update_adjlist()
		
	def get_matrix(self, From=None):
		if From == None:
			adjmatrix = [[0]*self.number_of_nodes() for i in range(self.number_of_nodes())]
			for i in range(len(self.Vnames)):
				for j in range(i+1, len(self.Vnames)):
					if self.Vnames[j] in list(self.adjlist[self.Vnames[i]].keys()):
						adjmatrix[i][j] = self.adjlist[self.Vnames[i]][self.Vnames[j]]
						adjmatrix[j][i] = adjmatrix[i][j]
			return matrix(adjmatrix)
		else:
			adjmatrix = [[0]*From.number_of_nodes() for i in range(From.number_of_nodes())]
			# print(From.Vnames, From.number_of_nodes())
			for i in range(From.number_of_nodes()):
				v = From.Vnames[i]
				for j in range(i+1, From.number_of_nodes()):
					u = From.Vnames[j]
					if v in self.adjlist and u in self.adjlist[v]:
						adjmatrix[i][j] = self.adjlist[v][u]
						adjmatrix[j][i] = adjmatrix[i][j]
			return matrix(adjmatrix)
			
	def to_Gnp(self, p=.5, maxw=10):
		for i in self.adjlist:
			for j in self.Vnames:
				if i == j:
					continue
				if rnd.random() <= p:
					w = rnd.randint(maxw) + 1
					self.adjlist[i][j] = w
					self.adjlist[j][i] = w
		self.update_adjlist()
		
	def update_adjlist(self):
		AdjL = dict(sorted(self.adjlist.items(), key=itemgetter(0)))
		for i in AdjL:
			AdjL[i] = dict(sorted(AdjL[i].items(), key=itemgetter(1), reverse=True))
		self.adjlist = AdjL
	
	def edge_weight(self, u, v):
		return self.adjlist[u][v]
	
	def print_matrix(self):
		print(self.get_matrix())
	
	
	def print_list(self):
		for i in range(self.number_of_nodes()):
			print(i,":",self.adjlist[i],)
			
	def from_graph_edge_list(self, E, G):
		for i in E:
			self.adjlist[i[0]][i[1]] = G.adjlist[i[0]][i[1]]
			self.adjlist[i[1]][i[0]] = G.adjlist[i[1]][i[0]]
		
	def number_of_edges(self):
		e = 0
		for v in self.adjlist:
			e = e + len(self.adjlist[v])
		return e//2

	def number_of_nodes(self):
		return len(self.Vnames)

	def weight(self):
		w = 0
		for i in self.adjlist:
			for j in self.adjlist[i]:
				w = w + self.adjlist[i][j]
		return w/2
		
	def avg_weight(self):
		if self.number_of_edges()>0:
			return self.weight()/self.number_of_edges()
		return 0

	def DFS(self, cc, v, visited):
		visited[v] = True
		cc.append(v)
		for i in self.adjlist[v]:
			if visited[i] == False:
				cc = self.DFS(cc, i, visited)
		return cc
	
	def CC(self):
		visited = dict(zip(self.Vnames, [False]*self.number_of_nodes()))
		ccs = []
		for v in self.Vnames:
			if visited[v] == False:
				cc = []
				ccs.append(self.DFS(cc, v, visited))
		return ccs
		
	def CCs_to_graph_list(self):
		H = deepcopy(self)
		L = self.CC()
		Glist = []
		for l in L:
			to_remove = []
			G = Graph(l)
			for i in l:
				for j in H.adjlist[i]:
					G.adjlist[i][j] = H.adjlist[i][j]
			H.remove_node(i)
			G.update_adjlist()
			Glist.append(G)
		return Glist
		
	def __str__(self):
		string = ''
		for v in self.adjlist:
			i = 0
			string = string + str(v) + ': {'
			for u in self.adjlist[v]:
				i = i + 1
				string = string + str(u) + ':' + str(self.adjlist[v][u])
				if len(self.adjlist[v]) != i:
					string = string + ', '
			string = string + '}' + '\n'
		return string
		
	def min_index(self, r):
		if len(self.adjlist[r]) > 0:
			return list(self.adjlist[r].keys()).pop()
		else:
			return -1

	def min_weight(self, r):
		return list(self.adjlist[self.min_index(r)].values()).pop()
	
	def number_of_components(self):
		return len(self.CCs_to_graph_list())

	def print_info(self):
		print(self.number_of_nodes())
		print(self.number_of_edges())
		print(self.weight())
		print(self.avg_weight())
		print(self.number_of_components())	
		print(self)

def prim(G, r):
	H = deepcopy(G)
	H.update_adjlist()
	visited = dict(zip(H.Vnames, [False]*H.number_of_nodes()))
	visited[r] = True
	mst = []
	while len(mst) < G.number_of_nodes() - 1:
		minw = inf
		o = 0
		for v in [key for key,val in visited.items() if val==True]:
			minv = H.min_index(v)
			if minv == -1:
				o = o + 1
				continue
			if H.edge_weight(v, minv) < minw and visited[minv]==False:
				minw = H.edge_weight(v, minv)
				u = v
				minu = minv
		if o == len(H.adjlist):
			break
		if minw == inf:
			# print(H)
			for v in H.adjlist:
				if len(H.adjlist[v]) > 0:
					H.adjlist[v].popitem()
			# print(H)
			H = copy(H)
			continue
		mst.append((u,minu))
		visited[minu] = True
		H.remove_edge(u, minu)
	T = Graph(G.Vnames)
	T.from_graph_edge_list(mst, G)
	# print(G.n, len(mst))
	return T

def DFS_PRIM_avg(G):
	max = -1
	for F in G.CCs_to_graph_list():
			# print(F)
			Ti = prim(F, F.Vnames[0])
			# print(Ti)
			if Ti.avg_weight() > max:
				T = deepcopy(Ti)
				max = T.avg_weight()
	return T

def DFS_PRIM_big(G):
	max = -1
	for F in G.CCs_to_graph_list():
			Ti = prim(F, F.Vnames[0])
			if Ti.weight() > max:
				T = deepcopy(Ti)
				max = T.weight()
	return T

def test(times=100, p=.5, datafile="data.log"):
	f = open(datafile, 'w')
	upper = "n(G) m(G) w(G) avgw(G) CC(G) n(Ta) m(Ta) w(Ta) avgw(Ta) n(Tb) m(Tb) w(Tb) avgw(Tb) time\n"
	f.write(upper)
	f.close()
	for i in range(2, times+1):
		G = Graph([str(v) for v in range(i)])
		G.to_Gnp(p,10)
		start = int(round(time.time()*1000))
		for j in range(100):
			T1 = DFS_PRIM_avg(G)
			T2 = DFS_PRIM_big(G)
		end = int(round(time.time()*1000))
		writedata(G,T1,T2,i,'data.log','a',time)
		time = end-start

def writedata(G, T1, T2, n=0, filename=None, mode='w', time=0.0):
	f = open(filename,mode)
	if mode == 'w':
		upper = "n(G) m(G) w(G) avgw(G) CC(G) n(Ta) m(Ta) w(Ta) avgw(Ta) n(Tb) m(Tb) w(Tb) avgw(Tb) time\n"
		f.write(upper)
	f.write(str(n))
	f.write(" ")
	f.write(str(G.number_of_edges()))
	f.write(" ")
	f.write(str(G.weight()))
	f.write(" ")
	f.write(str(G.avg_weight()))
	f.write(" ")
	f.write(str(len(G.CCs_to_graph_list())))
	f.write(" ")
	f.write(str(T1.number_of_nodes()))
	f.write(" ")
	f.write(str(T1.number_of_edges()))
	f.write(" ")
	f.write(str(T1.weight()))
	f.write(" ")
	f.write(str(T1.avg_weight()))
	f.write(" ")
	f.write(str(T2.number_of_nodes()))
	f.write(" ")
	f.write(str(T2.number_of_edges()))
	f.write(" ")
	f.write(str(T2.weight()))
	f.write(" ")
	f.write(str(T2.avg_weight()))
	f.write(" ")
	f.write(str(time))
	f.write("\n")
	f.close()

n = 100
# test(n, .15)
G = Graph([i for i in range(n)])
for i in range(10):
	G.to_Gnp(.01)
	start = int(round(time.time()*1000))
	for j in range(10):
		T1 = DFS_PRIM_avg(G)
		T2 = DFS_PRIM_big(G)
	end = int(round(time.time()*1000))
	writedata(G, T1, T2, n, mode='a', filename = "data.log", time=end-start)
# test(n, 0.001, "data0.log")
# test(n, 0.01, "data1.log")
# test(n, 0.1, "data2.log")
# test(n, 0.5, "data3.log")

# print(G)
# M1 = T1.get_matrix(G)
# M2 = T2.get_matrix(G)

# Gt = nx.from_numpy_matrix(G.get_matrix())
# T1t = nx.from_numpy_matrix(M1)
# T2t = nx.from_numpy_matrix(M2)

# pos = nx.spring_layout(Gt, weight='weight')
		
# nx.draw_networkx_nodes(Gt, pos, node_size = 1)
# nx.draw_networkx_edges(Gt, pos, width = 3)
# nx.draw_networkx_edges(T1t, pos, edge_color = 'r', width = 2)
# nx.draw_networkx_edges(T2t, pos, edge_color = 'g')
# #nx.draw_networkx_labels(Gt, pos, labels = dict(zip([i for i in range(n)],G.Vnames)))
# #labels=dict([((u, v,), d['weight']) for u, v, d in Gt.edges(data=True)])
# #nx.draw_networkx_edge_labels(Gt, pos, edge_labels = labels)
# savefile = "./graphs/graph"+str(n)+".png"
# plt.savefig(savefile, dpi = 800)
# plt.close()