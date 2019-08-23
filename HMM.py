import numpy as np

def log_sum_exp(log_probs,axis=None):
	#对于[a,b,c],求解log(e**a+e**b+e**c)

	max_prob=np.max(log_probs)

	log_probs=log_probs-max_prob
	exp_sum=np.exp(log_probs).sum(axis=axis)

	return max_prob + np.log(exp_sum)

class HMM(object):
	def __init__(self, A=None,B=None,pi=None,eps=None):
		#HMM模型的构造函数初始化，这里的HMM是离散的

		self.eps=np.finfo(float).eps if eps is None else eps
		'''
		A代表转移矩阵,B代表发射矩阵,pi代表各隐变量的先验

		A的size为(N,N),B的size为(N,V),pi的size为(1,N)
		O为实际的HMM观测序列
		T代表序列长度,I代表序列数目
		O的size为(I,T)
		'''
		self.A=A
		self.B=B
		self.pi=pi

		if self.pi is not None:
			self.pi[self.pi==0]=self.eps

		self.N=None
		self.V=None

		if self.A is not None:
			self.A[self.A==0]=self.eps
			self.N=self.A.shape[0]

		if self.B is not None:
			self.B[self.B==0]=self.eps
			self.V=self.B.shape[1]

		self.O=None
		self.T=None
		self.I=None

	def generate(self,step,latent_variables,observation_variables):
		#根据HMM参数，产生合法的HMM序列(包含隐变量序列与观测量序列)

		lat_seq=[]
		obs_seq=[]

		lat_index=np.random.multinomial(1,self.pi).argmax()
		lat_seq.append(latent_variables[lat_index])

		obs_index=np.random.multinomial(1,self.B[lat_index,:]).argmax()
		obs_seq.append(observation_variables[obs_index])

		for t in range(step-1):
			lat_index=np.random.multinomial(1,self.A[lat_index,:]).argmax()
			lat_seq.append(latent_variables[lat_index])

			obs_index=np.random.multinomial(1,self.B[lat_index,:]).argmax()
			obs_seq.append(observation_variables[obs_index])

		return lat_seq,obs_seq

	def _forward(self,obs_seq):
		#前向算法

		T=obs_seq.shape[0]
		#注意forward中是取对后的结果
		forward=np.zeros((self.N,T))
		eps=self.eps

		obs=obs_seq[0]
		for i in range(self.N):
			forward[i,0]=np.log(self.pi[i]+self.eps)+np.log(self.B[i,obs]+self.eps)

		for t in range(1,T):
			obs=obs_seq[t]

			for i in range (self.N):
				forward[i,t]=log_sum_exp([forward[_,t-1]+np.log(self.A[_,i]+self.eps)+np.log(self.B[i,obs]+self.eps) for _ in range(self.N)])

		return forward

	def _backward(self,obs_seq):
		#后向算法

		T=obs_seq.shape[0]
		#注意backward中是取对后的结果
		backward=np.zeros((self.N,T))
		eps=self.eps

		for i in range(self.N):
			backward[i,T-1]=0

		for t in reversed(range(T-1)):
			obs=obs_seq[t+1]

			for i in range (self.N):
				backward[i,t]=log_sum_exp([backward[_,t+1]+np.log(self.A[i,_]+self.eps)+np.log(self.B[_,obs]+self.eps) for _ in range(self.N)])

		return backward

	def log_likelihood(self,obs_seq):
		#评估问题

		if obs_seq.ndim == 1:
			obs_seq=obs_seq.reshape(1,-1)

		forward=self._forward(obs_seq[0])
		backward=self._backward(obs_seq[0])
		T=obs_seq.shape[1]

		log_prob_for=log_sum_exp(forward[:,T-1])
		obs=obs_seq[0,0]
		log_prob_back=log_sum_exp([backward[i,0]+np.log(self.pi[i]+self.eps)+np.log(self.B[i,obs]+self.eps) for i in range(self.N)])

		return log_prob_for,log_prob_back

	def viterbi(self,obs_seq):
		#解码问题

		T=obs_seq.shape[0]
		dp=np.zeros((self.N,T))
		path=np.zeros((self.N,T))
		obs=obs_seq[0]

		best_path=[]
		best_path_prob=0

		for i in range(self.N):
			dp[i,0]=np.log(self.pi[i]+self.eps)+np.log(self.B[i,obs]+self.eps)

		for t in range(1,T):
			obs=obs_seq[t]

			for j in range(self.N):
				dp[j,t]=np.max([dp[i,t-1]+np.log(self.A[i,j]+self.eps)+np.log(self.B[j,obs]+self.eps) for i in range(self.N)])
				path[j,t]=np.argmax([dp[i,t-1]+np.log(self.A[i,j]+self.eps)+np.log(self.B[j,obs]+self.eps) for i in range(self.N)])

		best_path_prob=np.max(dp[:,T-1])
		best_terminal=np.argmax(dp[:,T-1])
		best_path.append(best_terminal)

		for t in reversed(range(1,T)):
			best_terminal=int(path[best_terminal,t])
			best_path.append(best_terminal)

		best_path=best_path[::-1]
		return np.exp(best_path_prob),best_path

	def E_step(self):
		#E步

		gamma=np.zeros((self.I,self.N,self.T))
		xi=np.zeros((self.I,self.N,self.N,self.T))
		phi=np.zeros((self.I,self.N))

		for i in range(self.I):
			obs_seq=self.O[i]

			fwd=self._forward(obs_seq)
			bwd=self._backward(obs_seq)
			log_prob,_=self.log_likelihood(obs_seq)


			#求解phi
			for n in range(self.N):
				phi[i,n]=fwd[n,0]+bwd[n,0]-log_prob

			#求解gamma
			for t in range(self.T):
				for n in range(self.N):
					gamma[i,n,t]=fwd[n,t]+bwd[n,t]-log_prob

			#求解xi
			for t in range(self.T-1):
				obs_seq_t=obs_seq[t]

				for it in range(self.N):
					for ij in range(self.N):
						xi[i,it,ij,t]=fwd[it,t]+np.log(self.A[it,ij]+self.eps)+np.log(self.B[ij,obs_seq_t]+self.eps)+bwd[ij,t+1]-log_prob

		return phi,gamma,xi

	def M_step(self,phi,gamma,xi):
		#M步

		A=np.zeros((self.N,self.N))
		B=np.zeros((self.N,self.V))
		pi=np.zeros(self.N)

		#辅助概率矩阵
		count_gamma=np.zeros((self.I,self.N,self.V))
		count_xi=np.zeros((self.I,self.N,self.N))

		#计算count_gamma与count_xi
		for i in range(self.I):
			obs_seq=self.O[i,:]

			for ni in range(self.N):
				for v in range(self.V):
					if not (obs_seq==v).any():
						count_gamma[i,ni,v]=np.log(self.eps)
					else:
						count_gamma[i,ni,v]=log_sum_exp(gamma[i,ni,obs_seq==v])

				for nj in range(self.N):
					count_xi[i,ni,nj]=log_sum_exp(xi[i,ni,nj,:])

		#求解A、B矩阵
		for i in range(self.N):
			for j in range(self.N):
				A[i,j]=log_sum_exp(count_xi[:,i,j])-log_sum_exp(count_xi[:,i,:])

		print(np.exp(A))

		for i in range(self.N):
			for v in range(self.V):
				B[i,v]=log_sum_exp(count_gamma[:,i,v])-log_sum_exp(count_gamma[:,i,:])

		#求解pi
		pi=np.exp(log_sum_exp(phi,axis=0)-np.log(self.I+self.eps))

		return np.exp(A),np.exp(B),pi

	def fit(self,O,latent_variables,observation_variables,pi=None,tol=1e-3,verbose=True):
		'''
		学习问题
		鲍姆-韦尔奇算法
		是E-M算法的一种形式上的推广
		'''
		if O.ndim == 1:
			O.reshape(1,-1)

		self.O=O
		self.I,self.T=O.shape
		self.N=len(latent_variables)
		self.V=len(observation_variables)

		self.pi=pi
		if self.pi == None:
			self.pi=np.ones(self.N)
			self.pi=self.pi/np.sum(self.pi)


		self.A=np.ones((self.N,self.N))
		self.A=self.A/np.sum(self.A,axis=0)


		self.B=np.random.rand(self.N, self.V)
		self.B=self.B/np.sum(self.B,axis=1).reshape(self.N,-1)


		step,delta=0,np.inf

		pre_log_prob=np.sum([self.log_likelihood(obs_seq) for obs_seq in self.O])

		while abs(delta)>tol:
			phi,gamma,xi=self.E_step()
			self.A,self.B,self.pi=self.M_step(phi,gamma,xi)

			cur_log_prob=np.sum([self.log_likelihood(o) for o in self.O])
			delta=cur_log_prob-pre_log_prob

			pre_log_prob=cur_log_prob
			step+=1

			if verbose:
				fstr = "[Epoch {}] log_likelihood: {:.3f} Delta: {:.5f}"
				print(fstr.format(step, pre_log_prob, delta))

		return self.A,self.B,self.pi
