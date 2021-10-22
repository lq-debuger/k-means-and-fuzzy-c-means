import numpy as np
import pandas as pd
import matplotlib as plt 

# 读取.csv文件
data_full = pd.read_csv("test.csv")
# 得到表格的列名
columns = list(data_full.columns)
data = data_full[columns]
# 分类数
c = 3
# 最大迭代数
MAX_ITER = 100