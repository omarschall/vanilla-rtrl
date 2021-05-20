#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 12:02:11 2018

@author: omarschall
"""

import numpy as np
import matplotlib.pyplot as plt
try:
    from sklearn import linear_model
except ModuleNotFoundError:
    pass
import os
import pickle
from pdb import set_trace

def plot_smoothed_loss(mons, filter_size=100):
    
    losses = mons['loss_']
    smoothed_loss = np.convolve(losses, np.ones(filter_size)/filter_size, mode='valid')
    
    plt.plot(smoothed_loss)
    plt.plot([0, len(smoothed_loss)], [0.66, 0.66], '--', color='r')
    plt.plot([0, len(smoothed_loss)], [0.52, 0.52], '--', color='m')
    plt.plot([0, len(smoothed_loss)], [0.45, 0.45], '--', color='g')
    
    plt.ylim([0,1])

def classification_accuracy(data, y_hat):
    
    y_hat = np.array(y_hat)
    
    i_label = np.argmax(data['test']['Y'], axis=1)
    i_pred = np.argmax(y_hat, axis=1)
    
    acc = np.sum(i_label==i_pred)/len(i_label)
    
    return acc

def regress_vars(X_list, Y):
    
    X = np.concatenate(X_list, axis=1)
    model = linear_model.LinearRegression()
    
    model.fit(X, Y)
    r2 = model.score(X,Y)
    print('R2 = {}'.format(r2))
    
    A = get_vector_alignment(model.coef_.dot(X.T).T + model.intercept_,
                             Y)
    plt.plot(A, '.', alpha=0.1)
    plt.axhline(y=0, color='k', linestyle='--')
    plt.axhline(y=np.nanmean(A), color='b')
    
    return r2

def normalized_dot_product(a, b):
    
    return np.dot(a.flatten(),b.flatten())/np.sqrt(np.sum(a**2)*np.sum(b**2))

def get_spectral_radius(M):
    
    eigs, _ = np.linalg.eig(M)
    
    return np.amax(np.absolute(eigs))

def get_spectral_radii(Ms):
    
    r = []
    for i in range(Ms.shape[0]):
        
        r.append(get_spectral_radius(Ms[i,:,:]))
        
    return np.array(r)

def get_vector_alignment(v1, v2):

    alignment = []
    for i in range(v1.shape[0]):
        
        a = v1[i,:].flatten()
        b = v2[i,:].flatten()
        
        alignment.append(np.dot(a,b)/np.sqrt(np.sum(a**2)*np.sum(b**2)))
        
    return np.array(alignment)
    
def plot_filtered_signals(signals, filter_size=100, y_lim=None, plot_loss_benchmarks=True):
    
    fig = plt.figure(figsize=[8, 4])
    
    for signal in signals:
        smoothed_signal = np.convolve(signal, np.ones(filter_size)/filter_size, mode='valid')
        plt.plot(smoothed_signal)
    
    if plot_loss_benchmarks:
        plt.axhline(y=0.66, color='r', linestyle='--')
        plt.axhline(y=0.52, color='m', linestyle='--')
        plt.axhline(y=0.45, color='g', linestyle='--')
    
    if y_lim is not None:
        plt.ylim(y_lim)
    plt.xlabel('Time')
    
    return fig

def plot_results_from_job(job_name, rnn_signals, colors,
                          learn_alg_signals=[],
                          filter_size=100, plot_median=True,
                          alpha=0.05, plot_loss_benchmarks=True,
                          y_lim=[0, 1.5], n_seeds=1000):
    
    data_dir = os.path.join('/Users/omarschall/cluster_results/vanilla-rtrl/', job_name)
    
    rnn_colors = colors[:len(rnn_signals)]
    learn_alg_colors = colors[len(rnn_signals):]
    
    signals = {}
    for key in rnn_signals+learn_alg_signals:
        signals[key] = []
    
    for file_name in os.listdir(data_dir):
        
        print(file_name)
        
        if int(file_name.split('_')[-1])>n_seeds:
            continue
        
        with open(os.path.join(data_dir, file_name), 'rb') as f:
            result = pickle.load(f)
            
        for key, col in zip(rnn_signals, rnn_colors):
            
            smoothed_signal = np.convolve(result['rnn'].mons[key],
                                          np.ones(filter_size)/filter_size,
                                          mode='valid')
            plt.plot(smoothed_signal, col, alpha=alpha)
            signals[key].append(smoothed_signal)
            
        for key, col in zip(learn_alg_signals, learn_alg_colors):
            
            smoothed_signal = np.convolve(result['rnn'].learn_alg.mons[key],
                                          np.ones(filter_size)/filter_size,
                                          mode='valid')
            plt.plot(smoothed_signal, col, alpha=alpha)
            signals[key].append(smoothed_signal)
            
    for key, col in zip(rnn_signals+learn_alg_signals, colors):
        
        signals[key] = np.array(signals[key])
        
        plt.plot(np.median(signals[key], axis=0), col)
    
    if plot_loss_benchmarks:
        plt.axhline(y=0.66, color='r', linestyle='--')
        plt.axhline(y=0.52, color='m', linestyle='--')
        plt.axhline(y=0.45, color='g', linestyle='--')    
        
    plt.ylim(y_lim)
    plt.xlabel('Time')


























































