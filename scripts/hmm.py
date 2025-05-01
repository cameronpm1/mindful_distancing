import numpy as np
from scipy.stats import multivariate_normal


class hmm():

    def __init__(
            self,
    ):
        self.emissprob_means = np.load('params/emissprob_means.npy')
        self.emissprob_covs = np.load('params/emissprob_covs.npy')
        self.startprob = np.load('params/startprob.npy')
        self.transprob = np.load('params/startprob.npy')
        self.states = len(self.startprob)

        self.predictions = np.zeros((self.states,))
        self.predictions_history = []

        self.behaviors = ['crouch', 'walk_left', 'walk_right', 'clear']

    def predict(
            self,
            obs : list[list[float]]
    ):
        '''
        takes current observed state o_t, and provides predictions for 
        future hidden state x_{t+1}

        input
        -----
        obs:list[list[float]]
            observed state of system (movenet pose angles)

        output
        ------
        list[float]
            predicted probabilities of future state [P[crouch], P[walk left], P[walk right], P[clear]]
        '''

        #if first step, initialize probabilities to P[o|x]
        if sum(self.predictions) == 0:
            for i in range(self.states):
                self.predictions[i] = multivariate_normal.pdf(obs,self.emissprob_means[i],self.emissprob_covs[i])
        else:
            new_predictions = np.zeros((self.states,))
            for i in range(self.states):
                new_predictions[i] = np.dot(probs,self.transprob[:,i]) * multivariate_normal.pdf(obs,self.emissprob_means[i],self.emissprob_covs[i])
            self.predictions = new_predictions

        self.predictions = self.predictions/np.linalg.norm(self.predictions)
        self.predictions_history.append(np.argmax(self.predictions))
        
        return self.behaviors[self.predictions_history[-1]]

