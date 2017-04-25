#######################################################################################################################
#####                              Credits go to Aarshay Jain: https://github.com/aarshayj
#######################################################################################################################


#######################################################################################################################
#####                                         IMPORT STANDARD MODULES
#######################################################################################################################


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydot
import os
from scipy.stats.mstats import chisquare, mode
        
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier,AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.grid_search import GridSearchCV
from sklearn import metrics, cross_validation
from sklearn.feature_selection import RFE, RFECV
import io
import xgboost as xgb
from xgboost.sklearn import XGBClassifier


#######################################################################################################################
#####                                      GENERIC MODEL CLASS
#######################################################################################################################


class GenericModelClass(object):
	"""
	This class contains the generic classiciation functions and variable definitions applicable
	across all models.
	"""
	def __init__(self, model, data_train, data_test, target, predictors=[], cv_folds=5, scoring_metric='accuracy'):
		"""Initialization"""
		self.model = model                     # an instance of particular model class
		self.data_train = data_train           # training data
		self.data_test = data_test             # testing data
		self.target = target
		self.cv_folds = cv_folds
		self.predictors = predictors
		self.train_predictions = []
		self.train_pred_prob = []
		self.test_predictions = []
		self.test_pred_prob = []
		self.num_target_class = len(data_train[target].unique())

		# define scoring metric:
		self.scoring_metric = scoring_metric


		# grid-search objects:
		self.gridsearch_class = None
		self.gridsearch_result = None

		# Define a Series object to store generic classidication model outcomes:
		self.classification_output = pd.Series(index=['ModelID', 'Accuracy', 'CVScore_mean', 'CVScore_std', 'AUC',
			                                    'ActualScore(manual entry)', 'CVMethod', 'ConfusionMatrix', 'Predictors'])

		# not to be used for all but most
		self.feature_imp = None


	
