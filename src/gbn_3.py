# #!/usr/bin/env python3

import os
import numpy as np
import pickle
import tensorflow as tf
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import PGBN_sampler
import model as model, sample as sample, encoder as encoder
from load_dataset import load_dataset, Sampler


class gbn_3_model():
    def __init__(self, Vocabsize, args, N):         #train_data = tf.placeholder(tf.int32, [Setting['V'], args.minibatch])
        with tf.variable_scope('Topic'):

            self.Setting = {}
            self.Setting['V'] = Vocabsize
            self.Setting['K'] = args.theta_size
            self.Setting['H'] = args.theta_size
            self.Setting['N'] = N
            self.Setting['Num_class'] = 10
            self.Setting['Num_layers']= len(self.Setting['K'])

            # online setting
            self.Setting['SweepTimes'] = 100;
            self.Setting['Burnin']  = 10 ;     self.Setting['Collection'] = 10
            # Setting['Iterall'] = Setting['SweepTimes'] * Setting['N'] / Setting['Minibatch'];
            self.Setting['Iterall'] = self.Setting['SweepTimes'] * self.Setting['N'] ;
            self.Setting['tao0FR'] = 0   ;  self.Setting['kappa0FR'] = 0.9
            self.Setting['tao0'] = 20    ;  self.Setting['kappa0'] = 0.7
            self.Setting['epsi0'] = 1    ;  self.Setting['FurCollapse'] = 1  # 1 or 0
            self.Setting['flag'] = 0

            # load setting
            self.V = self.Setting['V']
            self.H = self.Setting['H']
            self.K = self.Setting['K']
            self.N = self.Setting['N']
            self.T = len(self.Setting['K'])
            self.real_min = np.float32(2.2e-10)

            # superparams
            self.Supara = {}
            self.Supara['ac'] = 1            ; self.Supara['bc'] = 1
            self.Supara['a0pj'] = 0.01       ; self.Supara['b0pj'] = 0.01
            self.Supara['e0cj'] = 1          ; self.Supara['f0cj'] = 1
            self.Supara['e0c0'] = 1          ; self.Supara['f0c0'] = 1
            self.Supara['a0gamma'] = 1       ; self.Supara['b0gamma'] = 1
            self.Supara['eta'] = np.ones(self.T)*0.1  # 0.01

            # params
            self.Eta = []
            for t in range(self.T):  # 0:T-1
                self.Eta.append(self.Supara['eta'][t])
            r_k = np.ones([self.K[self.T-1],1])/self.K[self.T-1]    ;  gamma0 = 1 ;  c0 = 1

            self.NDot = [0] * self.T
            self.Xt_to_t1 = [0] * self.T
            self.WSZS = [0] * self.T
            self.EWSZS = [0] * self.T

            self.ForgetRate = np.power((self.Setting['tao0FR'] + np.linspace(1, self.Setting['Iterall'], self.Setting['Iterall'])),-self.Setting['kappa0FR'])
            self.epsit = np.power((self.Setting['tao0'] + np.linspace(1, self.Setting['Iterall'], self.Setting['Iterall'])), -self.Setting['kappa0'])
            self.epsit = self.Setting['epsi0'] * self.epsit / self.epsit[0]

            # build graph
            Batch_Size = args.batch_size

            self.input_x = tf.placeholder(tf.float32, shape=[None,self.V])   # N*V
            #self.input_x = tf.clip_by_value(self.input_x,0.0,50.0)
            x_vn = tf.transpose(self.input_x)
            self.phi1 = tf.placeholder(tf.float32, shape = [self.V, self.K[0]])
            self.phi2 = tf.placeholder(tf.float32, shape = [self.K[0], self.K[1]])
            self.phi3 = tf.placeholder(tf.float32, shape = [self.K[1], self.K[2]])

            # upward
            self.h_1 = self.encoder_left(self.input_x, 0)
            self.h_2 = self.encoder_left(self.h_1, 1)
            self.h_3 = self.encoder_left(self.h_2, 2)

            # downward
            self.k3, self.l3 = self.encoder_right(self.h_3, 2, 0 , 0)
            self.theta3 = self.reparameterization(self.k3,self.l3,2,Batch_Size)
            self.k2, self.l2 = self.encoder_right(self.h_2, 1, self.phi3, self.theta3)
            self.theta2 = self.reparameterization(self.k2,self.l2,1,Batch_Size)
            self.k1, self.l1 = self.encoder_right(self.h_1, 0,self.phi2, self.theta2)
            self.theta1 = self.reparameterization(self.k1,self.l1,0,Batch_Size)

            # loss
            Theta1Scale_prior = 1.0
            Theta2Shape_prior = 0.01
            Theta2Scale_prior = 1.0
            theta3_KL = tf.reduce_sum(self.KL_GamWei(np.float32(0.01), np.float32(1.0), self.k3, self.l3))
            theta2_KL = tf.reduce_sum(self.KL_GamWei(tf.matmul(self.phi3,self.theta3), np.float32(Theta2Shape_prior), self.k2, self.l2))
            theta1_KL = tf.reduce_sum(self.KL_GamWei(tf.matmul(self.phi2,self.theta2), np.float32(Theta1Scale_prior),self.k1, self.l1))

            tmp1 = x_vn * self.log_max(tf.matmul(self.phi1, self.theta1))
            tmp2 = tf.matmul(self.phi1, self.theta1)
            tmp3 = tf.lgamma( x_vn + 1)

            # Likelihood = tf.reduce_sum( x_vn * self.log_max(tf.matmul(self.phi1, self.theta1)) - tf.matmul(self.phi1, self.theta1) - tf.lgamma( x_vn + 1))
            # self.tm_Loss    = - (0.1*theta1_KL + Likelihood) / tf.to_float(Batch_Size) # * N
            # self.tm_train = tf.train.AdamOptimizer(args.tm_learning_rate).minimize(self.tm_Loss)

            Likelihood = tf.reduce_sum(x_vn * self.log_max(tf.matmul(self.phi1, self.theta1)) - tf.matmul(self.phi1, self.theta1) - tf.lgamma(x_vn + 1))
            self.tm_Loss = -(0.001 * theta3_KL + 0.01 * theta2_KL + 0.1 * theta1_KL + Likelihood) / tf.to_float(Batch_Size)  # * N
            # self.tm_train = tf.train.AdamOptimizer(args.tm_learning_rate).minimize(self.tm_Loss)

            Optimizer = tf.train.AdamOptimizer(args.tm_learning_rate)
            threshold = 0.001
            grads_vars = Optimizer.compute_gradients(self.tm_Loss)
            capped_gvs = []
            for grad, var in grads_vars:
                grad = tf.where(tf.is_nan(grad), threshold * tf.ones_like(grad), grad)
                grad = tf.where(tf.is_inf(grad), threshold * tf.ones_like(grad), grad)
                capped_gvs.append((tf.clip_by_value(grad, -threshold, threshold), var))
            self.tm_train = Optimizer.apply_gradients(capped_gvs)

    def weight_variable(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.01, dtype=tf.float32))

    def bias_variable(self, shape):
        return tf.Variable(tf.constant(0.1, shape=shape, dtype=tf.float32))

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

    def max_pool(self, x):
        # return tf.nn.max_pool(x, pool_size=[ 2, 2],strides=[2, 2], padding='SAME')
        return tf.reduce_max(x, axis=1)

    def log_max(self, input_x):
        return tf.log(tf.maximum(input_x, self.real_min))

    def encoder_left(self, input_x, i):  # i = 0:T-1 , input_x N*V
        # params
        H_dim = [self.V] + self.H
        W_h = self.weight_variable(shape=[H_dim[i], H_dim[i + 1]])
        b_h = self.bias_variable(shape=[H_dim[i + 1]])

        # feedforward
        if i == 0:
            output = tf.nn.softplus(tf.matmul(self.log_max(1 + input_x), W_h) + b_h)  # none * H_dim[i+1]
        else:
            output = tf.nn.softplus(tf.matmul(input_x, W_h) + b_h)  # none * H_dim[i+1]
        return tf.clip_by_value(output, -1.0, 1e4)
        #return output

    def encoder_right(self, input_x, i, phi, theta):  # i = 0:T-1 , input_x N*V
        # params
        H_dim = [self.V] + self.H
        K_dim = self.K
        W_k = self.weight_variable(shape=[H_dim[i + 1], 1])  # params k   H_dim*1
        b_k = self.bias_variable(shape=[1])
        W_l = self.weight_variable(shape=[H_dim[i + 1], K_dim[i]])  # params l   H_dim*K_dim
        b_l = self.bias_variable(shape=[K_dim[i]])

        # feedforward
        k_tmp = tf.reshape(tf.maximum(tf.exp(tf.matmul(input_x, W_k) + b_k), self.real_min), [-1, 1])  # none * 1
        k_tmp = tf.tile(k_tmp, [1, K_dim[i]])  # none * K_dim[i]
        l = tf.maximum(tf.exp(tf.matmul(input_x, W_l) + b_l), self.real_min)  # none * K_dim[i]

        if i != len(self.K) - 1:
            #         k = tf.maximum(k_tmp + tf.transpose(tf.matmul(phi, theta)),real_min)                  # none * K_dim[i]
            k = tf.maximum(k_tmp, self.real_min)  # none * K_dim[i]
        else:
            k = tf.maximum(k_tmp, self.real_min)  # none * K_dim[i]

        return tf.clip_by_value(tf.transpose(k),1e-2,1e4), tf.clip_by_value(tf.transpose(l),1e-2,1e4)  # K_dim[i] * none
        #return tf.transpose(k), tf.transpose(l)
    def init_phi(self):
        Phi = []
        for t in range(self.T): # 0:T-1
            if t == 0:
                Phi.append(0.2 + 0.8 * np.float32(np.random.rand(self.V, self.K[t])))
            else:
                Phi.append(0.2 + 0.8 * np.float32(np.random.rand(self.K[t-1], self.K[t])))
            Phi[t] = Phi[t] / np.maximum(self.real_min, Phi[t].sum(0)) # maximum every elements
        return Phi

    def reparameterization(self, Wei_shape, Wei_scale, i, batch_size):
        K_dim = self.K

        eps = tf.random_uniform(shape=[np.int32(K_dim[i]), batch_size], dtype=tf.float32,minval=0.2,maxval=0.8)  # none * K_dim[i]
        theta = Wei_scale * tf.pow(-self.log_max(1 - eps), 1 / Wei_shape)
        return theta  # K_dim[i] * none

    def KL_GamWei(self, Gam_shape, Gam_scale, Wei_shape, Wei_scale):  # K_dim[i] * none
        eulergamma = 0.5772

        KL_Part1 = eulergamma * (1 - 1 / Wei_shape) + self.log_max(Wei_scale / Wei_shape) + 1 + Gam_shape * self.log_max(
            Gam_scale)
        KL_Part2 = -tf.lgamma(Gam_shape) + (Gam_shape - 1) * (self.log_max(Wei_scale) - eulergamma / Wei_shape)
        KL = KL_Part1 + KL_Part2 - Gam_scale * Wei_scale * tf.exp(tf.lgamma(1 + 1 / Wei_shape))
        return KL

    def updatePhi(self, miniBatch, Phi, Theta, MBratio, MBObserved):
        Xt = np.array(np.transpose(miniBatch), order='C').astype('float64')

        for t in range(len(Phi)):  # t = 0:T-1
            Phi[t] = np.array(Phi[t], order='C').astype('float64')
            Theta[t] = np.array(Theta[t], order='C').astype('float64')

            if t == 0:
                self.Xt_to_t1[t], self.WSZS[t] = PGBN_sampler.Multrnd_Matrix(Xt, Phi[t], Theta[t])
            else:
                self.Xt_to_t1[t], self.WSZS[t] = PGBN_sampler.Crt_Multirnd_Matrix(self.Xt_to_t1[t - 1], Phi[t], Theta[t])

            self.EWSZS[t] = MBratio * self.WSZS[t]  # Batch_Num * WSZS[t]

            if (MBObserved == 0):
                self.NDot[t] = self.EWSZS[t].sum(0)
            else:
                self.NDot[t] = (1 - self.ForgetRate[MBObserved]) * self.NDot[t] + self.ForgetRate[MBObserved] * self.EWSZS[t].sum(0)  # 1*K

            tmp = self.EWSZS[t] + self.Eta[t]  # V*K
            tmp = (1 / np.maximum(self.NDot[t], self.real_min)) * (tmp - tmp.sum(0) * Phi[t])  # V*K
            tmp1 = (2 / np.maximum(self.NDot[t], self.real_min)) * Phi[t]
            tmp = Phi[t] + self.epsit[MBObserved] * tmp + np.sqrt(self.epsit[MBObserved] * tmp1) * np.random.randn(
                Phi[t].shape[0],
                Phi[t].shape[1])
            Phi[t] = PGBN_sampler.ProjSimplexSpecial(tmp, Phi[t], 0)

        return Phi

