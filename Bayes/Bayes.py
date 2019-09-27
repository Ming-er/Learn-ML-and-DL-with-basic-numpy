import numpy as np
import math


class MultinomialNaiveBayes(object):

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes = None
        self.class_prior = None
        self.conditional_prob = None

    def fit(self, X_train, y_train):
        self.classes = np.unique(y_train)
        self.class_prior = self.calPriorProb(y_train)
        self.conditional_prob = self.calConditionalProb(X_train, y_train)

    def calPriorProb(self, targets):
        targets_class = np.unique(targets)
        num_samples = len(targets)
        targets_prior = {target: (np.sum(np.equal(targets, target))+self.alpha) /
                         (num_samples+self.alpha*len(targets_class)) for target in targets_class}
        return targets_prior

    def calFeaturesProb(self, features):
        features_class = np.unique(features)
        num_samples = len(features)
        feature_prior = {feature: (np.sum(np.equal(features, feature))+self.alpha) /
                         (num_samples+self.alpha*len(features_class)) for feature in features_class}
        return feature_prior

    def calConditionalProb(self, data, targets):
        targets_class = np.unique(targets)
        conditional_prob = {}
        for cla in targets_class:
            conditional_prob[cla] = {}
            for i in range(data.shape[0]):
                features = data[i][np.equal(targets, cla)]
                conditional_prob[cla][i] = self.calFeaturesProb(features)
        return conditional_prob

    def calProbProduct(self, cla, sample):
        prob = 1
        for index, feature in enumerate(sample):
            prob *= self.conditional_prob[cla][index][feature]
        prob *= self.class_prior[cla]
        return prob

    def predictSample(self, sample):
        maxprob = -1
        for cla in self.classes:
            tmpprob = self.calProbProduct(cla, sample)
            if(tmpprob > maxprob):
                maxprob = tmpprob
                label = cla
        return label

    def predict(self, X):
        if(X.ndim == 1):
            return self.predictSample(X)
        else:
            labels = [self.predictSample(X[index])
                      for index in range(X.shape[0])]
            return labels


class GaussianNaiveBayes(MultinomialNaiveBayes):

    def GaussianPDF(self,x, mu, sigma):
        return np.sqrt(2*math.pi*sigma)*math.exp(-0.5*(x-mu)**2/sigma)

    def calConditionalProb(self, data, targets):
        targets_class = np.unique(targets)
        conditional_prob = {}
        for cla in targets_class:
            conditional_prob[cla] = {}
            for i in range(data.shape[0]):
                features = data[i][np.equal(targets, cla)]
                mu = np.mean(features)
                sigma = np.std(features)**2
                conditional_prob[cla][i] = (mu, sigma)
        return conditional_prob

    def calProbProduct(self, cla, sample):
        prob = 1
        for index, feature in enumerate(sample):
            prob *= self.GaussianPDF(feature,
                                self.conditional_prob[cla][index][0], self.conditional_prob[cla][index][1])
        prob *= self.class_prior[cla]
        return prob


class BernoulliNaiveBayes(MultinomialNaiveBayes):

    def __init__(self, threshold=None,alpha=1.0):
        self.classes = None
        self.class_prior = None
        self.conditional_prob = None
        self.alpha=alpha
        self.threshold = None

    def fit(self, X_train, y_train):
        self.classes = np.unique(y_train)
        self.class_prior = self.calPriorProb(y_train)
        if not self.threshold:
            self.threshold = np.array([np.mean(X_train[i])
                                       for i in range(X_train.shape[0])]).reshape(X_train.shape[0], -1)
        boolean = X_train > self.threshold
        X_train = self.convert(boolean)
        self.conditional_prob = self.calConditionalProb(X_train, y_train)

    def convert(self, boolean):
        boolean[np.equal(boolean,False)] = 0
        boolean[np.equal(boolean,True)] = 1
        return boolean

    def predictSample(self, sample):
        maxprob = -1
        boolean = sample > self.threshold.T
        zero_one = self.convert(boolean)
        for cla in self.classes:
            tmpprob = self.calProbProduct(cla, zero_one.tolist()[0])
            if(tmpprob > maxprob):
                maxprob = tmpprob
                label = cla
        return label