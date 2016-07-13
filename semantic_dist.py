"""
Functions for computing semantic similarities.
"""
from collections import defaultdict

def pre_process(sentences, ignore=[]):
    """Takes in a list of sentence strings, remove words that occur only once 
    in all sentences and words in the ignore list"""
    texts = [ [word for word in sentence.lower().split() if word not in ignore]
              for sentence in sentences ]

    #remove words that only appear once
    frequency = defaultdict(int) #int here is the default constructor for missing keys

    for wlist in texts:
        for w in wlist:
            frequency[w]+=1

    texts = [[ w for w in wlist if frequency[w]>1 ] for wlist in texts ]
    return texts

def calc_inter_union(text1, text2):
    """Take two texts and compute a similarity score as follows:
    score = (# of words in both text1 and text2)/(# of words in either text1 or text2)
    """
    set1, set2 = set(text1), set(text2)
    inter, union = set1 & set2, set1 | set2

    if len(union)==0:
    	return 1.0	# since both texts are empty, they are identical
    else:
    	return 1.0* len(inter)/len(union)