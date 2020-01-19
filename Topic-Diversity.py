# --- Hend Alrasheed - halrasheed@ksu.edu.sa
# --- Jan 19, 2020

# Analyze topic diversity using text networks analysis.
# Input: a conversation (set of text items such as replies to a tweet or comments to a news feed)
# Output: Text files: text graph & list of nodes with their frequencies
#         On  screen: list of communities with their properties & the sentiment analysis of the key words 
#-------------------------------------------------------------------------------------------------------- 

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
import sys
import string
import networkx as nx 
import matplotlib.pyplot as plt  
import community
from community import community_louvain
from collections import Counter

  
#------------------------------------------------------------------------------- 

def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

#------------------------------------------------------------------------------- 

def process_text(filename):

    # read the text
    file1 = open (filename,"r", encoding='utf-8')
    text1 = file1.read()
    file1.close()

    # convert text to words and process
    tokens = word_tokenize(text1)
    tokens = [w.lower() for w in tokens]
    table = str.maketrans('', '', string.punctuation)        
    stripped = [w.translate(table) for w in tokens]          
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]

    wnl = WordNetLemmatizer()
    lemmas = [wnl.lemmatize(word,get_wordnet_pos(word)) for word in words]
    lemmas = [word for word in lemmas if len(word) > 2]

    return lemmas

#-------------------------------------------------------------------------------                   

def word_count(lemmas):
    
  ls = []  
  for word in lemmas:
      word_count = lemmas.count(word) 
      ls.append((word,word_count))       
  word_dict = dict(ls)

  return word_dict
#-------------------------------------------------------------------------------                   

def return_synonyms(word): 

    synonyms = []

    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())

    return(set(synonyms))
                      
#-------------------------------------------------------------------------------                   

def create_text_graph(words, counts):

    parts_of_speech={'JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBR', 'RBS'}

    # add nodes to graph
    for x in words:
        tagged = nltk.pos_tag(x)
        if tagged[1][1] in parts_of_speech:
           G.add_node(x)
           G.nodes[x]['word_frequency'] = counts[x]
           g.add_node(x)
           g.nodes[x]['word_frequency'] = counts[x]
  
    # add edges
    for node1 in G:
        synset = return_synonyms(node1)
        for node2 in G:
            if node2 in synset:
               if node2 != node1:       
                  G.add_edge(node1,node2)
                  G[node1][node2]['weight'] = w1
                  g.add_edge(node1,node2)
                  g[node1][node2]['weight'] = w1

    print("Text Graph (synonyms level 1):")
    print("|V| = ", G.number_of_nodes())
    print("|E| = ", G.number_of_edges())

#-------------------------------------------------------------------------------    

def find_communities(graph):

    parts = community_louvain.best_partition(graph)
    nx.set_node_attributes(graph, parts, 'community_num')
    size = len(set(parts.values()))
    print("\nnum of comm = ", size)
    mod=community.modularity(parts,graph)
    print("Modularity: ", mod)
    return parts

#-------------------------------------------------------------------------------

def expand_graph():

    for u in list(g.nodes()):
            for v in list(g.nodes()):
                if u != v and v not in g.neighbors(u):
                   synset1 = return_synonyms(u)
                   synset2 = return_synonyms(v)
                   common = any(item in synset1 for item in synset2)
                   if common is True:
                      g.add_edge(u,v)
                      g[u][v]['weight'] = w2

    print("Text Graph (synonyms level 2):")
    print("|V| = ", g.number_of_nodes())
    print("|E| = ", g.number_of_edges())                 

#-------------------------------------------------------------------------------

def draw_graph(parts, graph):

    pos = nx.spring_layout(graph)
    values = [parts.get(node) for node in graph.nodes()]
    nx.draw_spring(graph, cmap=plt.get_cmap('jet'), node_color=values, node_size=700, with_labels = True)
  
    print("\n Communities: ")
    for com in set(parts.values()) :
        list_nodes = [nodes for nodes in parts.keys() if parts[nodes] == com]
        if len(list_nodes) > 1:
           print(list_nodes)

    plt.show()    

#-------------------------------------------------------------------------------        

def analyze_communities(partition,number_of_key_words):

   communities = [partition.get(node) for node in g.nodes()]
   size = len(set(partition.values()))
   print("\n\n")
   for com in set(partition.values()):
        list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
        if len(list_nodes) > 1:
           subgraph = g.subgraph(list_nodes)
           print("Community ", subgraph.nodes())
           print("|Vc| = ", len(subgraph))
           print("|Ec| = ", subgraph.number_of_edges())
           print("diam = ", nx.diameter(subgraph))
           print("Avg cc = ", nx.average_clustering(subgraph))
           s=0
           for u in list(subgraph.nodes()):
               s = s +  subgraph.nodes[u]['word_frequency']
           print("Com weight = ", s)

           deg = nx.degree_centrality(subgraph)
           sorted_ = {k: v for k, v in sorted(deg.items(), key=lambda item: item[1], reverse = True)}
           print("******\nSorted degree cent nodes:")
           print(list(sorted_))
           key_words = list(sorted_)[number_of_key_words:]

           bet = nx.betweenness_centrality(subgraph)
           sorted_ = {k: v for k, v in sorted(deg.items(), key=lambda item: item[1], reverse = True)}
           print("******\nSorted betweenness cent nodes:")
           print(list(sorted_))

           clos = nx.closeness_centrality(subgraph)
           sorted_ = {k: v for k, v in sorted(deg.items(), key=lambda item: item[1], reverse = True)}
           print("******\nSorted closeness cent nodes:")
           print(list(sorted_))
           
           print("--------------------------------------------")
           return key_words
                
#-------------------------------------------------------------------------------        

def clean_graph(): #remove nodes with deg=0 and freq=1

    cnt=0
    for u in list(g.nodes()):
        if g.degree[u] == 0 and g.nodes[u]['word_frequency'] == 1:
            #print(u, " will be deleted")    
            g.remove_node(u)
            cnt=cnt+1

    print(cnt, ' words have been removed')
    print("Text Graph (synonyms level 2) - after cleaning:")
    print("|V| = ", g.number_of_nodes())
    print("|E| = ", g.number_of_edges())

#-------------------------------------------------------------------------------

# sentiment analysis will be performed on the key nodes in each comm + singletons with high freq
def sentiment_analysis(central_words, T):

    frequency_threshold = T
    singles=[]

    for u in list(g.nodes()):
        if g.degree[u] == 0 and g.nodes[u]['word_frequency'] >= frequency_threshold:
            singles.append(u)

    analyser = SentimentIntensityAnalyzer()
    pos=0
    neg=0
    neutral=0
    total=0

    words = central_words + singles

    for w in set(words):
        score = analyser.polarity_scores(w)['compound']
        total = total + analyser.polarity_scores(w)['compound']
        if score > 0:
           pos = pos+1
        elif score < 0:
           neg = neg + 1
        else:
           neutral = neutral + 1

    print('unweighted polarity:')
    print('      pos: ', pos)
    print('      neg: ', neg)
    print('      neutral: ', neutral)
    print('Weighted polarity = ', total)

#-------------------------------------------------------------------------------

def print_graph_file(graph):

    directed_graph = nx.DiGraph()
    directed_graph = graph

    nx.write_edgelist(directed_graph,filename+"-graph.csv", delimiter=',')

    f = open(filename+"-nodeList.csv","w+")

    for u in list(g.nodes()):
        if g.degree[u] > 0 or g.nodes[u]['word_frequency'] > 1:
           f.write(u + "," + str(g.nodes[u]['word_frequency']) + "\n")

    f.close()
    
#-------------------------------------------------------------------------------
     
G = nx.Graph()
g = nx.Graph()

#edge weights /// w1 --> direct syn, w2 --> indirect syn
w1=1
w2=.5

filename=sys.argv[1]
c = int(sys.argv[2]) #number of central nodes to be included in the analysis
T = int(sys.argv[3]) #frequency threshold of singleton nodes

lemmas = process_text(filename)
word_dict = word_count(lemmas)
create_text_graph(lemmas, word_dict)
expand_graph()
clean_graph()
print_graph_file(g)
partition = find_communities(g)
key_words = analyze_communities(partition,c)
sentiment_analysis(key_words, T)

print("Done!")
                  
                    


        
  


