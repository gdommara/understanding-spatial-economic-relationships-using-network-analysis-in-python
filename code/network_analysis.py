# import libraries
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd 
from pandas import Series, DataFrame
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from shapely.geometry import Point, MultiPoint

print('NetworkX version: {}'.format(nx.__version__))

# # Read csv of binominal data for each coin for time period
# # Rerun code for each time period
# coins = pd.read_csv("../data/780-850 binom2.csv", delimiter = ';')
# coins_grouped = coins.groupby("county").sum()
# coins_grouped = coins_grouped.drop(['weight', 'diameter','id'], 1))
# cross_table = pd.DataFrame(data = None, index= coins_grouped.index, columns =coins_grouped.index)
# 
# # Create a comparison weight matrix, +1 weight for where each has a 'true' find
# length = len(coins_grouped)
# for row in range(0,length):
#     for column in range(length):
#         value = 0
#         county_row = cross_table.iloc[row].index[row]
#         county_column = cross_table.iloc[column].index[column]
#         if county_column == county_row:
#             cross_table[county_column][county_row]= 'nan'
#         else:
#             for i in range(len(coins_grouped.columns)):
#                 if coins_grouped.ix[county_row][i] > 0 and coins_grouped.ix[county_column][i] > 0:
#                     value += 1
#             cross_table[county_column][county_row]= value
# 
# # export the data
# cross_table.to_csv("780-850 UNIPARTITE NETWORK.csv")


plt.figure()

# read coordinates from file
coords_df = pd.read_csv('../data/County GPX.csv', sep='\t', usecols=['county','lat','long'], index_col='county')

# create dictionary of coordinates 
# (correcting/swapping values of lat and long in the values of the dictionary)
coords_dict = coords_df.T.to_dict('list')
for i in coords_dict.values():
    x = i[0]
    y = i[1]
    i[0] = y
    i[1] = x

# read three correlation matricies and create graphs
english = pd.read_csv('../data/970-1020 UNIPARTITE NETWORK.csv', index_col='county')
viking = pd.read_csv('../data/780-850 UNIPARTITE NETWORK.csv', index_col='county')
anglo_sax = pd.read_csv('../data/680-730 UNIPARTITE NETWORK.csv', index_col='county')

english = english.fillna(0)
viking = viking.fillna(0)
anglo_sax = anglo_sax.fillna(0)

EU = nx.Graph()
for i in english.index:
    EU.add_node(i)

VU = nx.Graph()
for i in viking.index:
    VU.add_node(i)

AU = nx.Graph()
for i in anglo_sax.index:
    AU.add_node(i)

# create new coordinate dictionaries for each graph  
EU_dict = {}
VU_dict = {}
AU_dict = {}
    
for x in coords_dict.keys():
    if x in EU.nodes():
        EU_dict[x] = coords_dict[x]
        
for x in coords_dict.keys():
    if x in AU.nodes():
        AU_dict[x] = coords_dict[x]
      
for x in coords_dict.keys():
    if x in VU.nodes():
        VU_dict[x] = coords_dict[x]

# set the coordinates as node attributes
nx.set_node_attributes(EU, 'loc', EU_dict)
nx.set_node_attributes(AU, 'loc', AU_dict)
nx.set_node_attributes(VU, 'loc', VU_dict)

# new lists of x and y coords
EU_lats = [i[0] for i in EU_dict.values()]
EU_longs = [i[1] for i in EU_dict.values()]
AU_lats = [i[0] for i in AU_dict.values()]
AU_longs = [i[1] for i in AU_dict.values()]
VU_lats = [i[0] for i in VU_dict.values()]
VU_longs = [i[1] for i in VU_dict.values()]

# function that add edges to graph
def make_edges(df,R):
    for i in df.index:
            for j in df.columns:
                #add an edge between county 
                #with the weight equal to the correlation
                if df.loc[i,j]>2:
                    R.add_edge(i, j, weight=df.loc[i,j]/100.0)
    return df,R
    
english,EU = make_edges(english,EU)
anglo_sax,AU = make_edges(anglo_sax,AU)
viking, VU = make_edges(viking,VU)

# set edge with and edge color value lists
edges_EU = EU.edges(data='weight')
edgec_EU = [(e[2]*100)**.8/5 for e in edges_EU]
edgew_EU = [(e[2]*100)**1.4/5 for e in edges_EU]
# size_EU = Series(EU.degree(weight='weight'))
# size_EU = (size_EU)*200

edges_AU = AU.edges(data='weight')
edgec_AU = [(e[2]*100)**.8/5 for e in edges_AU]
edgew_AU = [(e[2]*100)**1.4/5 for e in edges_AU]
# size_AU = Series(AU.degree(weight='weight'))
# size_AU = (size_AU)

edges_VU = VU.edges(data='weight')
edgec_VU = [(e[2]*100)**.8/5 for e in edges_VU]
edgew_VU = [(e[2]*100)**1.4/5 for e in edges_VU]
# size_VU = Series(VU.degree(weight='weight'))
# size_VU = (size_VU + .01)*2000

nx.draw(AU,pos=AU_dict,edge_cmap=plt.cm.Wistia,edge_color=edgec_AU,width=edgew_AU,node_size=80,node_color="darksalmon",with_labels=True,font_size=5)
# nx.draw(EU,pos=EU_dict,edge_cmap=plt.cm.Wistia,edge_color=edgec_EU,width=edgew_EU,node_size=80,node_color="darksalmon",with_labels=True,font_size=5)
# nx.draw(VU,pos=VU_dict,edge_cmap=plt.cm.Wistia,edge_color=edgec_VU,width=edgew_VU,node_size=80,node_color="darksalmon",with_labels=True,font_size=5)
        

m = Basemap(resolution='i',llcrnrlon=-8,llcrnrlat=49,urcrnrlon=3,urcrnrlat=57)
m.drawcoastlines()

plt.savefig("AU_text.png", dpi = 400)
plt.show()


## Network Analysis
from multiprocessing import Pool
import itertools

def partitions(nodes, n):
    "Partitions the nodes into n subsets"
    nodes_iter = iter(nodes)
    while True:
        partition = tuple(itertools.islice(nodes_iter,n))
        if not partition:
            return
        yield partition
        
def btwn_pool(G_tuple):
    return nx.betweenness_centrality_source(*G_tuple)
    
def between_parallel(G, processes = None):
    p = Pool(processes=processes)
    part_generator = 4*len(p._pool)
    node_partitions = list(partitions(G.nodes(), int(len(G)/part_generator)))
    num_partitions = len(node_partitions)
    bet_map = p.map(btwn_pool, zip([G]*num_partitions,[True]*num_partitions,
                    [None]*num_partitions,node_partitions))
    bt_c = bet_map[0]
    for bt in bet_map[1:]:
        for n in bt:
            bt_c[n] += bt[n]
    return bt_c
    

bt = between_parallel(AU)
top = 10
sorted(bt.items(), key = lambda v: -v[1])[:top]
 
print(nx.info(AU))
print(nx.info(VU))
print(nx.info(EU))


# Calculating Degree Centralities for weighted graphs
degree_AU = dict(AU.degree(weight='weight'))
degree_VU = dict(VU.degree(weight='weight'))
degree_EU = dict(EU.degree(weight='weight'))

# Printing top 10 counties having high degree centralities
print(sorted(degree_AU.items(), key = lambda v: -v[1])[:top])
print(sorted(degree_VU.items(), key = lambda v: -v[1])[:top])
print(sorted(degree_EU.items(), key = lambda v: -v[1])[:top])


# Calculating Betweenness centralities
btws_AU = nx.betweenness_centrality(AU, normalized=False)
btws_VU = nx.betweenness_centrality(VU, normalized=False)
btws_EU = nx.betweenness_centrality(EU, normalized=False)

# Printing top 10 counties having high betweenness centralities
print(sorted(btws_AU.items(), key = lambda v: -v[1])[:top])
print(sorted(btws_VU.items(), key = lambda v: -v[1])[:top])
print(sorted(btws_EU.items(), key = lambda v: -v[1])[:top])


