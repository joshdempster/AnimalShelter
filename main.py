import numpy as np
import pandas as pd
import sklearn as skl
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import log_loss, accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import os
import gzip
import random
from time import sleep
import copy


def get_dummy_expanded(df, verbose=False):
    '''return a dataframe with columns for each unique entry with a 1 if that entry 
    appears in any of the columns of df'''
    first = True
    if len(df.shape) == 1:
        return pd.get_dummies(df)
    for key in df:
        if verbose:
            print "Unique values for %s:" %key
            print df.groupby(key).size()
        if first:
            base = pd.get_dummies(df[key])
            base.replace(0, float('NaN'), inplace=True)#implements OR
            first = False
        else:
            right = pd.get_dummies(df[key])
            right.replace(0, float('NaN'), inplace=True)
            for key2 in right:
                if not key2 in base:
                    base[key2] = right[key2]
            base.update(right)
    base.fillna(0, inplace=True)
    return base


def split_data(df, ratio):
    '''split df and y randomly by rows, one containing ratio and the other 1-ratio'''
    msk = np.random.rand(len(df)) < ratio
    return msk
    
    
class AnimalPredictor(object):
    def __init__(self, trainframe, classifier):
        '''abstract class for prediction.
        Parameters: 
            `trainframe`: pandas.DataFrame
                Labeled data. Note that to conserve space this frame will be altered
                IN PLACE and should not be reused!
            `classifier`: scikit-learn classifier
                must support predict_proba 
                '''
        print 'Data example: '
        print trainframe[0:10]
        print 'Outcomes overall:'
        print pd.value_counts(trainframe['OutcomeType'].values, sort=False)
        
        self.trainframe = trainframe
        self.classifier = classifier
        
        self.y = trainframe['OutcomeType'].copy()
        self.length = trainframe.shape[0]
        for label in 'AnimalID', 'Name', 'DateTime', 'OutcomeType', 'OutcomeSubtype':
            try: 
                trainframe.drop(label, 1, inplace=True)
            except KeyError:
                pass
        self.clean_train_data()
        
    def clean_train_data(self):
        raise NotImplementedError, 'Base class, use predictor for specific animal'
        
    def train(self, partition=.1):
        '''Train on 1-partition, hold out partition for evaluation.'''
        self.mask = split_data(self.X, partition)
        X_train = self.X[~self.mask]
        y_train = self.y[~self.mask]
        X_eval = self.X[self.mask]
        y_eval = self.y[self.mask] 
        print '%i examples to train, %i to test' %(X_train.shape[0], 
                                                X_eval.shape[0])
        print '%i features' %X_train.shape[1]
        print 'Training...'
        self.classifier.fit(X_train, y_train)
        print 'Predicting...'
        y_predict = self.classifier.predict(X_eval)
        y_train_predict = self.classifier.predict(X_train)
        print 'training accuracy:', accuracy_score(y_train, 
                    self.classifier.predict(X_train))
        print 'evaluation accuracy:', accuracy_score(y_eval, y_predict)
        
        y_expand = pd.get_dummies(y_eval) 
        outcome_prob = self.classifier.predict_proba(X_eval)
        
        print 'evaluation log loss:', log_loss(y_expand, outcome_prob)
        
        for i in range(10):
            print y_predict[i], y_eval.as_matrix()[i]
        
        print self.classifier.classes_
        print outcome_prob[:10]
        
        
class CatPredictor(AnimalPredictor):
    '''A fancy predictor for cats!''' 
    def get_breed(self):
        breed = pd.DataFrame(self.trainframe.Breed.str.split('/').tolist(), 
            columns=['breed1', 'breed2'])
        purebred = []
        for i, row in enumerate(breed.as_matrix()[:]):
            printing = False
            if random.random() > 1.0:
                printing = True
                oldrow = copy.deepcopy(row)
                
            if row[0][-3:] == 'Mix' or row[0][:8] == 'Domestic':
                purebred.append(0)
            elif row[1] == 'Black' or row[1] == 'White':
                row[1] = None
                purebred.append(1)
            elif row[1] != None:
                purebred.append(0)
            else:
                purebred.append(1)
            row[0] = row[0].strip()
            if row[1]:
                row[1] = row[1].strip()
            if printing: 
                print 'random breed: row %r to %r, purebred %i' %(
                                        oldrow.tolist(), row.tolist(), purebred[-1])
        return breed, purebred

    def get_breed_dummies(self):
        print '\n getting breeds'
        breed, purebred = self.get_breed()
        self.breedcounts = breed['breed1'].value_counts()
        norm = self.breedcounts.iloc[0]
        frequency = [] #how common breed is
        for i, row in enumerate(breed.as_matrix()[:]):
            frequency.append(self.breedcounts.loc[row[0]])
            if self.breedcounts.loc[row[0]] < 5: 
                #too few examples to be informative
                if purebred[i]:
                    row[0] = 'RareBreed'
                else:
                    row[0] == 'Mix'
            try:
                if self.breedcounts.loc[row[1]] < 5:
                    row[1] == 'Mix'
            except TypeError:
                row[1] == 'Mix'
            except KeyError:
                row[1] == 'Mix'
            
        breed = get_dummy_expanded(breed) 
              
        print '\nRemaining breeds:'
        for i, key in enumerate(breed):
            print key
        breed['Purebred'] = np.array(purebred, dtype=np.float64)
        breed['frequency'] = np.array(frequency, dtype=np.float64)/norm
        return breed
        
    def get_color_dummies(self):
        print '\ngetting colors\n'
        color = pd.DataFrame(self.trainframe.Color.str.split('/').tolist(), 
            columns=['color1', 'color2'])
        purecolor = []
        for i, row in enumerate(color.as_matrix()[:]):
            if row[1]:
                purecolor.append(0)
            else:
                purecolor.append(1)
            if random.random() > 1.0:
                print 'random color: %s gives purecolor %i' %(row, purecolor[-1])
                
        count1 = color.color1.value_counts()
        count2 = color.color2.value_counts()
        self.colorcounts = count1.add(count2, fill_value=0)
        
        for row in color.as_matrix()[:]:
            if self.colorcounts.loc[row[0]] < 5:
                row[0] = 'RareColor'
            try:
                if self.colorcounts.loc[row[1]] < 5:
                    row[1] == 'RareColor'
            except TypeError:
                pass
        
        color = get_dummy_expanded(color)
        print 'Remaining colors:'
        for key in color:
            print key
        color['Purecolor'] = np.array(purecolor, dtype=np.float64)
        return color
        
    def get_age(self):
        print '\n getting ages'
        self.trainframe.AgeuponOutcome.fillna('0 years', inplace=True)
        age = pd.DataFrame(self.trainframe.AgeuponOutcome.str.split(' ').tolist(), 
            columns=['age', 'unit'])
        for row in age.as_matrix()[:]:
            printing = False
            if random.random() > 1.0:
                printing = True
                oldrow = copy.deepcopy(row)
            unit = row[1].lower().strip('s')
            if unit == 'year':
                row[0] = float(row[0])
            elif unit == 'month':
                row[0] = float(row[0])/12
            elif unit == 'week':
                row[0] = float(row[0])/52
            elif unit == 'day':
                row[0] = float(row[0])/365
            else: print 'Error: unit %s not handled' %unit
            if printing:
                print 'random age: %s %s gives %1.2f' %tuple(
                    oldrow.tolist()+[row[0]])
        age.drop('unit', 1, inplace=True)
        mean = int(np.mean(age.age.as_matrix()))
        agemax = max(age.age.as_matrix())
        print 'max age', agemax
        for i, row in enumerate(age.age.as_matrix()[:]):
            if row == 0:
                age.age[i] = mean
        age.age.as_matrix()[:] /= agemax
        return age
        
    def clean_train_data(self):
        print '\ncleaning data\n'
        breeds = self.get_breed_dummies()
        colors = self.get_color_dummies()
        ages = self.get_age()
        sex = pd.get_dummies(pd.DataFrame(
                    self.trainframe.SexuponOutcome.tolist(), columns=['sex']))
        
        print '\ncreating X\n'
        self.X = pd.DataFrame()
        for frame in breeds, colors, sex:
            for key in frame:
                self.X[key] = frame[key]
        self.X['Age'] = ages.age
        #self.X['HasName'] = self.trainframe['HasName']
        self.X.reindex()
       
         
class DogPredictor(CatPredictor):
    '''a faithful predictor for dogs!'''
    def get_breed(self):
        #slashes used both for Black/Tan breed and to separate breeds, clean that
        numpylist = self.trainframe.Breed.as_matrix()
        for i, v in enumerate(numpylist):
            numpylist[i] = v.replace('Black/Tan', 'BlackTan')
        breed = pd.DataFrame(self.trainframe.Breed.str.split('/').tolist(), 
            columns=['breed1', 'breed2'])
            
        purebred = []
        for i, row in enumerate(breed.as_matrix()[:]):
            printing = False
            if random.random() > 1.0:
                printing = True
                oldrow = copy.deepcopy(row)
                
            if row[0][-3:] == 'Mix':
                purebred.append(0)
                row[0] = row[0][:-3]
            elif row[1] == 'Black' or row[1] == 'White':
                row[1] = None
                purebred.append(1)
            elif row[1] != None:
                purebred.append(0)
            else:
                purebred.append(1)
            row[0] = row[0].strip()
            if row[1]:
                row[1] = row[1].strip()
            if printing: 
                print 'random breed: row %r to %r, purebred %i' %(
                                        oldrow.tolist(), row.tolist(), purebred[-1])
        return breed, purebred


if __name__ == '__main__':
    print 'Loading training data'
    trainframe = pd.read_csv('train.csv')
    trainframe['HasName'] = pd.isnull(trainframe.Name)
    print 'data shape: %i, %i' %trainframe.shape
    subset = trainframe[trainframe['AnimalType'] == 'Cat'].copy()   
    subset.reset_index(drop=True, inplace=True)
    
    svc = skl.svm.SVC(probability=True)
    predictor = CatPredictor(subset, svc)
    predictor.train()

  