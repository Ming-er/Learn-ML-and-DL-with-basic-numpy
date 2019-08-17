import numpy as np

class GMM(object):
	def __init__(self,clu=3,seed=None):
        #构造函数

		self.clu=clu
		self.num=None
		self.dim=None

		if seed:
			np.random.seed(seed)
        #初始化随机数种子
	def set_params(self,num,dim):
        #参数初始化

		self.num=num
		self.dim=dim

		self.pi=np.random.rand(self.clu)
        #pi对应每一个高斯的先验，size为(cluster,)
		self.pi=self.pi/self.pi.sum()
		self.Qi=np.zeros((self.num,self.clu))
        #Qi对应每一个样本在每一个高斯上的后验，size为(num,cluster)
        #
		minnum=np.min(self.x,axis=0)
		maxnum=np.max(self.x,axis=0)
		for d in range (self.dim):
			mu=np.arange(minnum[d],maxnum[d],(maxnum[d]-minnum[d])/self.clu)
			if d==0:
				self.mu=mu
			else:
				self.mu=np.append(self.mu,mu,axis=0)
		self.mu=self.mu.reshape(self.clu,self.dim)
		self.mu=self.mu.T
		#
        #mu对应每一个高斯的所有维度的，size为(cluster,dimension)
		self.sigma=np.array([np.identity(self.dim) for _ in range(self.clu)])
        #sigma对应每一个高斯的所有维度的协方差矩阵，size为(cluster,dimension,dimension)

		self.final_mu=None
		self.final_sigma=None
		self.final_pi=None
		self.final_elbo=-np.inf


	def logpro_of_gaussian(self,x_i,mu,sigma):
        #多维高斯分布的概率(对数)计算

		term_1=self.dim*np.log(2*np.pi)
		sgn,term_2=np.linalg.slogdet(sigma)
		term_3=np.dot((x_i-mu),np.linalg.solve(sigma,x_i-mu))

		res=-0.5*(term_1+term_2+term_3)

		#print(res)
		return res

	def log_sum_exp(self,logprob):
        #求解形如log(e**a+e**b+e**c)
        #其中logprob=[a,b,c]

		maxexp=np.max(logprob)
		logprob=logprob-maxexp
		prob=np.exp(logprob)

		return maxexp+np.log(np.sum(prob))

	def E_step(self):
        #E步,固定参数,改变Q

		eps=np.finfo(float).eps
		for i in range (self.num):
			x_i=self.x[i,:]
			pro_list=[]

			for c in range (self.clu):
				mu_c=self.mu[c,:]
				sigma_c=self.sigma[c,:,:]
				pi_c=self.pi[c]
				pro_list.append(np.log(pi_c)+self.logpro_of_gaussian(x_i,mu_c,sigma_c))


			pro_list_sum=self.log_sum_exp(pro_list)
			pro_list=[pro-pro_list_sum for pro in pro_list]
			pro_list=np.exp(pro_list)

			self.Qi[i,:]=pro_list

	def M_step(self):
    	#M步,固定Q,改变参数

		Q_c_sum=np.sum(self.Qi,axis=0)
    	#更新均值
		for c in range(self.clu):
			Q_c=self.Qi[:,c]
			tmp=np.dot(Q_c,self.x)
			mu_c=tmp/Q_c_sum[c] if Q_c_sum[c]>0 else np.zeros_like(tmp)
			self.mu[c,:]=mu_c
    	#更新先验
		self.pi=Q_c_sum/self.num
    	#更新方差
		for c in range(self.clu):
			mu_c=self.mu[c,:]
			n_c=Q_c_sum[c]

			cov=np.zeros((self.dim,self.dim))

			for i in range(self.num):
				Q_i_c=self.Qi[i,c]
				x_i=self.x[i,:]
				tmp=x_i-mu_c
				cov+=Q_i_c*np.outer(tmp,tmp)
			cov=cov/n_c if n_c > 0 else cov
			self.sigma[c,:,:]=cov

	def lower_bound_of_likelihood(self):
        #求詹森不等式放缩后的迭代下界

		sum=0
		eps=np.finfo(float).eps

		for i in range (self.num):
			x_i=self.x[i,:]

			for c in range(self.clu):
				mu_c=self.mu[c,:]
				sigma_c=self.sigma[c,:,:]
				Q_i_c=self.Qi[i,c]
				pi_c=self.pi[c]

				sum+=Q_i_c*(np.log(pi_c+eps)+self.logpro_of_gaussian(x_i,mu_c,sigma_c)-np.log(Q_i_c+eps))
		return sum

	def solve(self,x,max_iter=100,tol=1e-6):
    	#利用GMM拟合数据
		self.x=x
		num=x.shape[0]
		dim=x.shape[1]
		self.set_params(num,dim)
		prev_llb=-np.inf
		for iter in range(max_iter):
			try:
				self.E_step()
				self.M_step()
				cur_llb=self.lower_bound_of_likelihood()

				print("{}. Lower bound: {}".format(iter+1,cur_llb))
				converged=np.abs(cur_llb-prev_llb)<=tol
				if np.isnan(cur_llb) or converged:
					break
				prev_llb=cur_llb
				if cur_llb > self.final_elbo:
					self.final_elbo=cur_llb
					self.final_mu=self.mu
					self.final_pi=self.pi
					self.final_sigma=self.sigma
			except np.linalg.LinAlgError:
				print("Singular matrix: components collapsed")
				return -1
		return 0