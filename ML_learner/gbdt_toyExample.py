import numpy as np
import pydotplus
from sklearn.ensemble import GradientBoostingRegressor

X = np.arange(1, 11).reshape(-1, 1)
y = np.array([5.16, 4.73, 5.95, 6.42, 6.88, 7.15, 8.95, 8.71, 9.50, 9.15])

gbdt = GradientBoostingRegressor(max_depth=4, criterion ='squared_error').fit(X, y)

print(gbdt)

from IPython.display import Image
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz

# 拟合训练5棵树
sub_tree = gbdt.estimators_[4, 0]
dot_data = export_graphviz(sub_tree, out_file=None, filled=True, rounded=True, special_characters=True, precision=2)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())