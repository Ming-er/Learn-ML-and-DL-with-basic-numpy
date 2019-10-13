import numpy as np

class leaf(object):
	#叶节点
	'''
	如果是分类树则返回值为相应特征空间对应的类别
	如果是回归树则返回值为相应特征空间对应的均值
	'''
	def __init__(self,value):
		self.value=value

class node(object):
	#内部结点
	#包含属性值：左子树、右子树、当前结点划分依据的特征及阈值
	def __init__(self,left,right,feature,threshold):
		self.left = left
		self.right = right
		self.feature = feature
		self.threshold = threshold

class decision_Tree(object):
	#决策树
	'''
	使用CART算法实现分类与回归决策树
	对于分类树，支持使用基尼系数与信息增益进行分类
	对于回归树，采用最小化平方损失函数来进行回归
	'''
	def __init__(self,classifier=True,rule='Gini',max_depth=None,detailed=False):
		self.depth = 0
		self.root = None
		self.classifier = classifier
		self.rule = rule
		self.max_depth = max_depth if max_depth else np.inf
		self.detailed = detailed

		if (not self.classifier and self.rule in ['Gini','Entropy']) or (self.classifier and self.rule == 'Mse'):
			raise ValueError('split rule is not supported')

	def fit(self,X,y):
		#模型拟合
		if self.classifier:
			self.n_examples = X.shape[0]
			self.n_classes = np.unique(y).shape[0]

		self.root = self.grow(X,y)

	def predict(self,X):
		#序列预测
		return np.array([self.travel(x,self.root) for x in X])

	def grow(self,X,y):
		#决策树的向下伸展
		if np.unique(y).shape[0] == 1:
			if self.classifier:
				tmp = np.zeros(self.n_classes)
				tmp[y[0]] = 1.0
				return leaf(tmp)
			return leaf(y[0])

		if self.max_depth <= self.depth:
			if self.classifier:
				hist = np.bincount(y)
				probs = hist/np.sum(hist)
				return leaf(probs)
			return leaf(np.mean(y))

		self.depth+=1;

		split_feature,split_threshold = self.find_Split(X,y)

		left_idx = np.argwhere(X[:,split_feature]<=split_threshold)
		right_idx = np.argwhere(X[:,split_feature]>split_threshold)

		left = self.grow(X[left_idx.flatten(),:],y[left_idx.flatten()])
		right = self.grow(X[right_idx.flatten(),:],y[right_idx.flatten()])

		return node(left,right,split_feature,split_threshold)

	def find_Split(self,X,y):
		#寻找分叉的特征和阈值
		M,N = X.shape
		best_gain,split_feature,split_threshold = (-np.inf,-1,None)

		for i in range(N):
			slices = X[:,i]
			thresholds = (slices[1:]+slices[:-1])/2

			gains = self.cal_Gain(slices,y,thresholds,i,self.detailed)
			max_gain = np.max(gains)
			argmax_gain = np.argmax(gains)

			if max_gain > best_gain:
				best_gain = max_gain
				split_feature = i

				split_threshold = thresholds[argmax_gain]
		if self.detailed:
			print("当前分割所能获得的最大信息增益为 {:.3f}".format(best_gain))
		return split_feature, split_threshold

	def cal_Gain(self,x,y,thresholds,i,detailed):
		#计算增益
		loss = eval('cal_'+self.rule)
		ans=[]

		for threshold in set(thresholds):
			idx_min = np.argwhere(x <= threshold)
			idx_max = np.argwhere(x > threshold)

			tmp = idx_min.shape[0]/(idx_min.shape[0]+idx_max.shape[0])

			former_gains = loss(y)
			later_gains = tmp*loss(y[idx_min.flatten()])+(1-tmp)*loss(y[idx_max.flatten()])
			ans.append(abs(former_gains-later_gains))

		return ans

	def travel(self,x,startnode):
		#从根向下遍历二叉树
		if isinstance(startnode,leaf):
			if self.classifier:
				return np.argmax(startnode.value)
			return startnode.value
		if x[startnode.feature] <= startnode.threshold:
			ans = self.travel(x,startnode.left)
		else:
			ans = self.travel(x,startnode.right)
		return ans
		
def cal_Entropy(y):
	#计算信息熵
	hist=np.bincount(y)
	probs=hist/np.sum(hist)
	return -sum([prob*np.log2(prob) for prob in probs.tolist() if prob > 0])

def cal_Gini(y):
	#计算基尼系数
	hist = np.bincount(y)
	probs = hist/np.sum(hist)
	return 1-np.sum(probs**2)

def cal_Mse(y):
	#计算平方误差
	return np.mean((y-np.mean(y)**2))

