import pandas as pd
df = pd.read_csv('1.csv',index_col=0)


from graphviz import Digraph


class Node(object):
    """结点
       leaf_value ： 记录叶子结点值
       split_feature ：分裂节点对应的特征名
       split_value ： 分裂节点对应的特征的值
       left ： 左子树
       right ： 右子树
    """

    def __init__(self, leaf_value=None, split_feature=None, split_value=None, left=None, right=None):
        self.leaf_value = leaf_value
        self.split_feature = split_feature
        self.split_value = split_value
        self.left = left
        self.right = right

    def show(self):
        print(
            f'weight: {self.leaf_value}, split_feature: {self.split_feature}, split_value: {self.split_value}.')

    def visualize_tree(self):
        """
        使用graphviz递归绘制树
        """
        def add_nodes_edges(self, dot=None):
            if dot is None:
                dot = Digraph()
                dot.node(name=str(self),
                         label=f'{self.split_feature}<{self.split_value}')
            # Add nodes
            if self.left:
                if self.left.leaf_value:
                    dot.node(name=str(self.left),
                             label=f'leaf={self.left.leaf_value:.10f}')
                else:
                    dot.node(name=str(self.left),
                             label=f'{self.left.split_feature}<{self.left.split_value}')
                dot.edge(str(self), str(self.left))
                dot = add_nodes_edges(self.left, dot=dot)
            if self.right:
                if self.right.leaf_value:
                    dot.node(name=str(self.right),
                             label=f'leaf={self.right.leaf_value:.10f}')
                else:
                    dot.node(name=str(self.right),
                             label=f'{self.right.split_feature}<{self.right.split_value}')
                dot.edge(str(self), str(self.right))
                dot = add_nodes_edges(self.right, dot=dot)
            return dot

        dot = add_nodes_edges(self)
        return dot



# 树的结构参数
reg_lambda = 1 # 叶节点权重L2正则系数
min_samples_split = 1 # 分裂所需的最小样本个数
max_depth = 3 # 树的深度

# 建树过程参数
learning_rate = 0.1 # 学习率
n_estimators = 2 # 树的个数

# log损失函数
def log_loss_obj(preds, labels):
    # preds是建该树之前模型的输出，对于二分类问题需要的是概率，因此将该值经过Sigmoid转换
    probs = 1.0 / (1.0 + np.exp(-preds))
    grad = probs - labels
    hess = probs * (1.0 - probs)
    return grad, hess


def build_tree(df, feature_names, depth=1):
    df = df.copy()
    df['g'], df['h'] = log_loss_obj(df.y_pred, df.y)
    G, H = df[['g', 'h']].sum()
    Gain_max = float('-inf')

    # 终止条件 当前节点个数小于分裂所需最小样本个数，深度大于max_depth，叶节点只有一类样本无需再分
    if df.shape[0] > min_samples_split and depth <= max_depth and df.y.nunique() > 1:
        for feature in feature_names: # 遍历每个特征
            thresholds = sorted(set(df[feature])) # 特征取值排序
            for thresh_value in thresholds[1:]: # 遍历每个取值
                left_instance = df[df[feature] < thresh_value] # 划分到左右节点的样本
                right_instance = df[df[feature] >= thresh_value]
                G_left, H_left = left_instance[['g', 'h']].sum()
                G_right, H_right = right_instance[['g', 'h']].sum()

                Gain = G_left**2/(H_left+reg_lambda)+G_right**2 / \
                    (H_right+reg_lambda)-G**2/(H+reg_lambda) # 评价划分的增益效果
                if Gain >= Gain_max:
                    Gain_max = Gain
                    split_feature = feature # 最大增益对应的分裂特征
                    split_value = thresh_value # 最大增益对应的分裂值
                    left_data = left_instance # 最大增益对应的分裂后左节点样本
                    right_data = right_instance # 最大增益对应的分裂后右节点样本

        left = build_tree(left_data, feature_names,  depth+1) # 递归建左子树
        right = build_tree(right_data, feature_names, depth+1)# 递归建右子树
        return Node(split_feature=split_feature, split_value=split_value, left=left, right=right) # 返回分裂节点
    return Node(leaf_value=-G/(H+reg_lambda)*learning_rate) # 分裂终止，返回叶节点权重



def predict(x, tree):
    # 递归每个分裂点直到样本对应的叶节点
    # 终止条件：叶节点
    if tree.leaf_value is not None:
        return tree.leaf_value
    if x[tree.split_feature] < tree.split_value:
        return predict(x, tree.left)
    else:
        return predict(x, tree.right)

trees = []
y_pred = 0  # 初始预测值
for i in range(n_estimators):
    df['y_pred'] = y_pred
    tree = build_tree(df, feature_names=['x1', 'x2'])
    data_weight = df[['x1', 'x2']].apply(
        predict, tree=tree, axis=1)
    y_pred += data_weight
    trees.append(tree)