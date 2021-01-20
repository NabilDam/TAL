#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 14:29:13 2019

@author: 3414093
"""


from bs4 import BeautifulSoup
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.sklearn
import json
#import nltk 
"""
ps = nltk.stemmer.PorterStemmer()
ps.stem(token)
"""


def load_corpus(filename):
   
    with open(filename,"r") as f:
        soup = BeautifulSoup(f.read(),"xml")
    #stemmer = SnowballStemmer("french", ignore_stopwords=True)        
    
    return np.array([ d.text   for d in soup.find_all("TEXT")])


def load_R_model(filename):
    with open(filename, 'r') as j:
        data_input = json.load(j)
    data = {'topic_term_dists': data_input['phi'], 
            'doc_topic_dists': data_input['theta'],
            'doc_lengths': data_input['doc.length'],
            'vocab': data_input['vocab'],
            'term_frequency': data_input['term.frequency']}
    return data

def vectorizer(X):
    """
    Fonction qui vectorise les textes de chaques documents et effectue du préprocessing de données.
    """
    #token = r"\b[^\d\W]+\b/g"
    #stemmer = SnowballStemmer("french", ignore_stopwords=False)
    vectorizer = CountVectorizer(stop_words='english',token_pattern=r'[a-zA-Z]+')
    Xvec = vectorizer.fit_transform(X)

    return (Xvec,vectorizer.get_feature_names(),vectorizer)


def LDA(sparVec,n):
    
    """
        Algorithme LDA pour le cluster des données.
    """
    lda = LatentDirichletAllocation(n, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=1)

    lda.fit(sparVec)
    return lda
    
def cluster(lda,mot,n):
    
    
    for k in enumerate(lda.components_):
        
        mots = np.argsort(k[1])[::-1][:n]
        print('\ncluster : ',k[0],'\n')
        for i in mots: 
            print(mot[i],end=" ")
        print('\n')
            

def classe_nouveau_doc(doc,lda,mot,vectorizer):
    
    text = vectorizer.transform([doc])
    
    #models = [float(model[1]*text.T) for model in enumerate(lda.components_)]
    topic = lda.transform(text)
   
    
    clust = np.argsort(topic)[0][-5:]
    print("Le nouveau doc est dans le cluster ",clust[::-1])
    return clust
    

if __name__ == "__main__":
    corpus = load_corpus('ap/ap.txt')
    #vect = CountVectorizer(stop_words='english',token_pattern=r'[a-zA-Z]+')
    #X = vect.fit_transform(corpus)
    X ,vec,vectori= vectorizer(corpus)
    lda = LDA(X,40)
    len(vec)
    cluster(lda,vec,10)
    clust=classe_nouveau_doc("The Soviet Union had its roots in the 1917 October Revolution, \
                             when the Bolsheviks, led by Vladimir Lenin, overthrew the Russian \
                             Provisional Government which had replaced Tsar Nicholas II during World War I.\
                             In 1922, the Soviet Union was formed by a treaty which legalized the unification\
                             of the Russian, Transcaucasian, Ukrainian and Byelorussian \
                             republics that had occurred from \
                             1918. Following Lenin's death in 1924 and a brief power struggle \
                             Joseph Stalin came to power in the mid-1920s. Stalin committed  \
                             the state's ideology to Marxism–Leninism (which he created) and constructed \
                             a command economy which led to a period of rapid industrialization and \
                             collectivization. During his rule, political paranoia fermented and the \
                             Great Purge removed Stalin's opponents within and outside of the party \
                             via arbitrary arrests and persecutions of many people, resulting in at \
                             least 600,000 deaths. In 1933, a major famine struck the country,\
                             causing the deaths of some 3 to 7 million people. .",lda,vec,vectori)
    
    
    