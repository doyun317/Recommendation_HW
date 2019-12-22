# CF_Algorithm

from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
import pandas as pd
import os

from surprise import SVD
from surprise import Dataset
from surprise import Reader

from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import KNNWithZScore

from surprise.model_selection import cross_validate
from surprise.model_selection import KFold

from collections import defaultdict

class _CF:
    def __init__(self,data):
      
        self.data = data
            
    def precision_recall_at_k(self,predictions):
    
        _k=5
        threshold=4
        
        # First map the predictions to each user.
        user_est_true = defaultdict(list)
                
        for uid, _, true_r, est, _ in predictions:
            user_est_true[uid].append((est, true_r))

        precisions = dict()
        recalls = dict()
        
        for uid, user_ratings in user_est_true.items():

            # Sort user ratings by estimated value
            user_ratings.sort(key=lambda x: x[0], reverse=True)

            # Number of relevant items
            n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

            # Number of recommended items in top k
            n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:_k])

            # Number of relevant and recommended items in top k
            n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                                  for (est, true_r) in user_ratings[:_k])

            # Precision@K: Proportion of recommended items that are relevant
            precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

            # Recall@K: Proportion of relevant items that are recommended
            recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

        return precisions, recalls
            
            
            
    def Basic_CF(self) :
        kf = KFold(n_splits = 5)
        sim_options = {'name':'cosine','user_based':True}
        algo = KNNBasic(k=40, min_k = 1, sim_options = sim_options)

        for trainset, testset in kf.split(self.data):
            algo.fit(trainset)
            predictions = algo.test(testset)
            
            precisions, recalls = self.precision_recall_at_k(predictions)

            P = sum(prec for prec in precisions.values()) / len(precisions)
            R = sum(rec for rec in recalls.values()) / len(recalls)
            F1 = 2*P*R/(P+R)

            print("Precision : ", P)
            print("Recall    : ", R)
            print("F1        : ",F1)

    
    def CFM(self) : 
        kf = KFold(n_splits = 5)
        sim_options = {'name':'cosine','user_based':True}
        algo = KNNWithMeans(k=40, min_k = 1, sim_options = sim_options)

        for trainset, testset in kf.split(self.data):
            algo.fit(trainset)
            predictions = algo.test(testset)
            
            precisions, recalls = self.precision_recall_at_k(predictions)

            P = sum(prec for prec in precisions.values()) / len(precisions)
            R = sum(rec for rec in recalls.values()) / len(recalls)
            F1 = 2*P*R/(P+R)

            print("Precision : ", P)
            print("Recall    : ", R)
            print("F1        : ",F1)
    
    def CFZ(self) :
        kf = KFold(n_splits = 5)
        sim_options = {'name':'cosine','user_based':True}
        algo = KNNWithZScore(k=40, min_k = 1, sim_options = sim_options)

        for trainset, testset in kf.split(self.data):
            algo.fit(trainset)
            predictions = algo.test(testset)
            precisions, recalls = self.precision_recall_at_k(predictions)

            P = sum(prec for prec in precisions.values()) / len(precisions)
            R = sum(rec for rec in recalls.values()) / len(recalls)
            F1 = 2*P*R/(P+R)

            print("Precision : ", P)
            print("Recall    : ", R)
            print("F1        : ",F1)

    def SVD(self) : 
        kf = KFold(n_splits = 5)
        algo = SVD()

        for trainset, testset in kf.split(self.data):
            algo.fit(trainset)
            predictions = algo.test(testset)
            precisions, recalls = self.precision_recall_at_k(predictions)

            P = sum(prec for prec in precisions.values()) / len(precisions)
            R = sum(rec for rec in recalls.values()) / len(recalls)
            F1 = 2*P*R/(P+R)

            print("Precision : ", P)
            print("Recall    : ", R)
            print("F1        : ",F1)
    
    
    def PMF(self) :
        kf = KFold(n_splits = 5)
        algo = SVD(n_factors = 100, n_epochs = 20, biased = False, lr_all = 0.005, reg_all = 0.02)

        for trainset, testset in kf.split(self.data):
            algo.fit(trainset)
            predictions = algo.test(testset)
            precisions, recalls = self.precision_recall_at_k(predictions)

            P = sum(prec for prec in precisions.values()) / len(precisions)
            R = sum(rec for rec in recalls.values()) / len(recalls)
            F1 = 2*P*R/(P+R)

            print("Precision : ", P)
            print("Recall    : ", R)
            print("F1        : ",F1)
    
    
    def PMFB(self) :
        kf = KFold(n_splits = 5)
        algo = SVD(n_factors = 100, n_epochs = 20, biased = True, lr_all = 0.005, reg_all = 0.02)

        for trainset, testset in kf.split(self.data):
            algo.fit(trainset)
            predictions = algo.test(testset)
            precisions, recalls = self.precision_recall_at_k(predictions)

            P = sum(prec for prec in precisions.values()) / len(precisions)
            R = sum(rec for rec in recalls.values()) / len(recalls)
            F1 = 2*P*R/(P+R)

            print("Precision : ", P)
            print("Recall    : ", R)
            print("F1        : ",F1)
            
            
            
            
            