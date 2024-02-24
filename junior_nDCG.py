'''
nDCG (Normalized Discounted Cumulative Gain) is a popular
metric in the ranking task that takes into account the order of items in the output.
'''

from typing import List
import numpy as np


def cumulative_gain(relevance: List[float], k: int) -> float:
    """Score is cumulative gain at k (CG@k)

    Parameters
    ----------
    relevance:  `List[float]`
        Relevance labels (Ranks)
    k : `int`
        Number of elements to be counted

    Returns
    -------
    score : float
    """
    score=np.sum(relevance[0:k])
    return score


def discounted_cumulative_gain(relevance: List[float], k: int, method: str = "standard") -> float:
    """Discounted Cumulative Gain

    Parameters
    ----------
    relevance : `List[float]`
        Video relevance list
    k : `int`
        Count relevance to compute
    method : `str`, optional
        Metric implementation method, takes the values 
        `standard` - adds weight to the denominator
        `industry` - adds weights to the numerator and denominator
        `raise ValueError` - for any value

    Returns
    -------
    score : `float`
        Metric score
    """
    if k>len(relevance):
        k=len(relevance)
    gains=[]

    if method == 'standard':
        for i in range(k):
            res=(relevance[i])/(np.log2(i+2))
            gains.append(res)
        score=np.sum(gains)
    elif method == 'industry':
        for i in range(k):
            res=(2 ** relevance[i]-1)/(np.log2(i+2))
            gains.append(res)
        score=np.sum(gains)    
    else:
        raise ValueError()
    
    return score


def normalized_dcg(relevance: List[float], k: int, method: str = "standard") -> float:
    """Normalized Discounted Cumulative Gain.

    Parameters
    ----------
    relevance : `List[float]`
        Video relevance list
    k : `int`
        Count relevance to compute
    method : `str`, optional
        Metric implementation method, takes the values
        `standard` - adds weight to the denominator
        `industry` - adds weights to the numerator and denominator
        `raise ValueError` - for any value

    Returns
    -------
    score : `float`
        Metric score
    """
    if k>len(relevance):
        k=len(relevance)

    gains=[]
    sorted_gains=[]
    sorted_relevance=sorted(relevance, reverse=True)

    if method == 'standard':
        for i in range(k):
            gains.append((relevance[i])/(np.log2(i+2)))
            sorted_gains.append((sorted_relevance[i])/(np.log2(i+2)))
    elif method == 'industry':
        for i in range(k):
            gains.append((2 ** relevance[i]-1)/(np.log2(i+2)))
            sorted_gains.append((2 ** sorted_relevance[i]-1)/(np.log2(i+2)))
    else:
        raise ValueError()
    
    dcg=np.sum(gains)
    idcg=np.sum(sorted_gains)
    score=dcg/idcg
    return score

def avg_ndcg(list_relevances: List[List[float]], k: int, method: str = 'standard') -> float:
    """Average nDCG

    Parameters
    ----------
    list_relevances : `List[List[float]]`
        Video relevance matrix for various queries
    k : `int`
        Count relevance to compute
    method : `str`, optional
        Metric implementation method, takes the values
        `standard` - adds weight to the denominator
        `industry` - adds weights to the numerator and denominator
        `raise ValueError` - for any value

    Returns
    -------
    score : `float`
        Metric score
    """
    list_ndcg = [normalized_dcg(relevances, k=k, method=method) for relevances in list_relevances]
    score = sum(list_ndcg) / len(list_relevances)
    return score
