'''
Author: your name
Date: 2021-10-22 13:17:32
LastEditTime: 2021-10-24 22:47:39
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \k-means-and-fuzzy-c-means\src\test.py
'''

import numpy as np
from numpy.core.fromnumeric import size

data = 30*np.random.random_sample((200,2))
# print(data)
np.savetxt("test.csv",data,fmt="%.2f",delimiter=',')
