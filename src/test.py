import numpy as np
from numpy.core.fromnumeric import size

data = 30*np.random.random_sample((200,2))
# print(data)
np.savetxt("test.csv",data,fmt="%.2f",delimiter=',')
