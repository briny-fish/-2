import pandas as pd
import numpy as np
import csv
import os
import pickle as pkl
#['age', 'body type', 'bust size', 'category', 'fit', 'height', 'item_id',
#       'rating', 'rented for', 'review_date', 'review_summary', 'review_text',
#       'size', 'user_id', 'weight']
def readTrain():
    f = pd.read_csv('product_fit/train.txt')
    f = pd.DataFrame(f)
    return f
def readTest():
    f = pd.read_csv('product_fit/test.txt.txt')
    f = pd.DataFrame(f)
    return f
if __name__ == '__main__':
    f = readTrain()
    print(list(f['fit']))

