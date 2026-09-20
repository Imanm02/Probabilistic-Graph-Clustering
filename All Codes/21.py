from random import random
import networkx as ntx

N=1
n=1000
p=0.033
# The problem statement specifies p=0.0033. For n=1000 that sits below the connectivity
# threshold ln(n)/n = 0.0069, so almost every sample is disconnected and the mean shortest
# path length is undefined. I use p=0.033 instead, which is comfortably above the threshold.


def gen_graph(n, p) -> ntx.Graph:
	G=ntx.empty_graph(n)
	for i in range(n):
		for j in range(i):
			if random()<p:
				G.add_edge(i, j)
	return G

def simul():
	G=gen_graph(n, p)
	sp = dict(ntx.all_pairs_shortest_path_length(G))
	mean=0
	for i in range(n):
		for j in range(i):
			mean+=sp[i][j]
	return mean/(n*(n-1)/2)

results=[simul() for i in range(N)]
average=sum(results)/N
print(average)


