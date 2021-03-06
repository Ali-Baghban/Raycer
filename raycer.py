# -*- coding: utf-8 -*-
"""Raycer.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1wsGYG_GZeE488dI18VVliDpPBxQEIHMn
"""

!pip install ray
#!pip install modin[ray]

"""**Import libs**"""

import pandas as pd
import ray
import operator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from google.colab import files
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import time

"""**Loading Dataset and Partitioning**"""

uploaded = files.upload()

n = int(input("Number of Partitions = "))
df = pd.read_csv('/content/car.data')
shuffled = df.sample(frac=1,random_state=1).reset_index()
shuffled.to_csv('shuffled.data')
shuffled_ten = df.sample(frac=0.1).reset_index()
shuffled_ten.to_csv('shuffled_ten.data')
#=================================================
split_size  = int(len(df)/n)
#=================================================
dataset = []
for i,chunk in enumerate(pd.read_csv('/content/shuffled.data' , chunksize=split_size)):
  dataset.append(chunk.astype('category'))

X,Y = [],[]

for i in range(n):
  X.append(dataset[i].iloc[:,2:shuffled.shape[1]])
  Y.append(dataset[i].iloc[:,shuffled.shape[1]:shuffled.shape[1]+1])

X_ten, Y_ten = shuffled_ten.iloc[:,1:shuffled.shape[1]-1].astype('category'), shuffled_ten.iloc[:,shuffled.shape[1]-1:shuffled.shape[1]].astype('category')

X_all, Y_all = shuffled.iloc[:,1:shuffled.shape[1]-1].astype('category'), shuffled.iloc[:,shuffled.shape[1]-1:shuffled.shape[1]].astype('category')



#kf = KFold(n_splits=10, random_state=1, shuffle=True)
#X_train, X_test, Y_train, Y_test = [],[],[],[]
#for i in range(s):
#  for train_index, test_index in kf.split(X[i]):
#     X_train, X_test = X[i][train_index], X[i][test_index]
#     y_train, y_test = Y[i][train_index], Y[i][test_index]

#====================================================
#ray.init()

"""**Traditional Racer**"""

class Racer():
  
  rules = {}
  final_rules = {}
  columns = {}
  classes = []

  # Fitness Value = alpha * accuracy + beta * coverage
  # accuracy coef : accuracy is the percent of covered instances which are correctly classified 
  # coverage coef : coverage is the percent of instances which are covered among the training set
  alpha = 0.5   
  beta = 0.5  

  rules_size = 0
  d_size = 0
  X = None
  Y = None
  def __init__(self, alpha, beta):  # initialize valiables
    self.alpha = alpha
    self.beta = beta
    self.rules = {}
    self.final_rules = {}
    self.columns = {}
    self.classes = []
    self.rules_size = 0
    self.d_size = 0
    
  
  def fit(self,X,Y):  
    
    keys = Y.keys()
    
    if Y[keys[0]].dtype.name != "category":
      raise ValueError(f'Y must be of type "category", "{Y.loc[0].dtype.name}" given')
    for i in Y[keys[0]].cat.categories:
      self.classes.append(i)
      self.rules[i] = []
    
    cols = X.columns.copy()
    self.d_size = X.shape[0]
    for c in cols:
      if X[c].dtype.name != "category":
        raise ValueError(f'All columns must be of type "category", "{X[c].dtype.name}" given')
      self.columns[c] = []
      for i in X[c].cat.categories:
        self.columns[c].append(i)
    
    self.X = X.copy()
    self.Y = Y.copy()
    # self.rules_size += len(self.classes)
    for key in self.columns:
      self.rules_size += len(self.columns[key])

    self.convert_rules()

    self.proccess_rules()


  
  def proccess_rules(self):
    for c in self.classes:
      extant_rules = self.rules[c].copy()
      extant_rules = [[i,0] for i in extant_rules]
      for i in range(len(extant_rules)):
        for j in range(i+1, len(extant_rules)):
          if extant_rules[i][1] == 0 and extant_rules[j][1] == 0:
            composed_rule = self.composition(extant_rules[i][0], extant_rules[j][0])
            if self.fitness(composed_rule, c) > self.fitness(extant_rules[i][0], c) and self.fitness(composed_rule, c) > self.fitness(extant_rules[j][0], c):
              extant_rules[i][0] = composed_rule
              extant_rules[j][1] = 1
              for x in range(len(extant_rules)):
                if x != i and self.rule_covers(composed_rule, extant_rules[x][0]):
                  extant_rules[x][1] = 1
      
      for i in range(len(extant_rules)):
        if extant_rules[i][1] == 0:
          extant_rules[i][0] = self.generalize(extant_rules[i][0], c)
      
      final = sorted(extant_rules, key=lambda x: self.fitness(x[0], c), reverse=True)
      self.final_rules[c] = [[i[0], self.fitness(i[0], c)] for i in final if i[1] == 0]
      # self.final_rules[c] = final.copy()

  
  def predict(self, X):
    cols = X.columns.copy()
    Y = []
    for c in cols:
      if X[c].dtype.name != "category":
        raise ValueError(f'All columns must be of type "category", "{X[c].dtype.name}" given')
    
    for i in range(X.shape[0]):
      rule = self.generate_rule(X.iloc[i])
      rules = []
      for c in self.classes:
        for j in range(len(self.final_rules[c])):
          if self.rule_covers(self.final_rules[c][j][0], rule):
            rules.append([c, self.final_rules[c][j][1], j])
      rules.sort(key=operator.itemgetter(1, 2), reverse=True)
      if len(rules)==0:
        Y.append(-1)
        continue
      Y.append(rules[0][0])
    
    return Y

    
  def generalize(self, rule, c):
    for i in range(len(rule)):
      if rule[i] == 0:
        new_rule = rule
        new_rule[i] = 1
        if self.fitness(new_rule, c) > self.fitness(rule, c):
          rule = new_rule
    return rule

  def composition(self, rule1, rule2):
    result = ""
    for i in range(self.rules_size):
      if rule1[i] == "1" or rule2[i] == "1":
        result += "1"
      else:
        result += "0"
    
    return result
  
  def fitness(self, rule, cls):
    return self.alpha * self.accuracy(rule, cls) + self.beta * self.coverage(rule)
  
  def accuracy(self, rule, cls):
    return self.n_correct(rule, cls) / self.n_covers(rule)
  
  def coverage(self, rule):
    return self.n_covers(rule) / self.d_size

  def n_correct(self, rule1, cls):
    num = 0
    for rule in self.rules[cls]:
      correct = True
      for i in range(self.rules_size):
        if rule1[i] != rule[i] and rule1[i] == "0":
          correct = False
      if correct:
        num += 1
    
    return num

  def n_covers(self, rule1):
    num = 0
    for cls in self.classes:
      for rule in self.rules[cls]:
        correct = True
        for i in range(self.rules_size):
          if rule1[i] != rule[i] and rule1[i] == "0":
            correct = False
        if correct:
          num += 1
    
    return num


  def rule_covers(self, rule1, rule2):
    for i in range(self.rules_size):
      if rule2[i] == "1" and rule1[i] == "0":
        return False
    return True


  def convert_rules(self):
    for i in range(self.X.shape[0]):
      self.generate_rules(self.X.iloc[i], self.Y.iloc[i])

  def generate_rule(self, input):
    rule = ""
    for key in self.columns:
      sub_rule = ""
      for i in self.columns[key]:
        if i == input[key]:
          sub_rule += "1"
        else:
          sub_rule += "0"
      rule += sub_rule
    return rule
  
  def generate_rules(self, input, output):
    rule = ""
    for key in self.columns:
      sub_rule = ""
      for i in self.columns[key]:
        if i == input[key]:
          sub_rule += "1"
        else:
          sub_rule += "0"
      rule += sub_rule
    rule_class = None
    for i in self.classes:
      if (i == output).bool():
        rule_class = i

    
    #self.rules[rule_class].append(rule)
    self.rules[rule_class].append(rule)
  def get_final_rules(self):
    return self.final_rules

Tracer = Racer(alpha=0.7,beta=0.3)
Tracer.fit(X_ten,Y_ten)
rules_ten = Tracer.get_final_rules()
#rules_ten

#Tracer.rules

start_time = time.time()
Aracer = Racer(alpha=0.7,beta=0.3)
Aracer.fit(X_all,Y_all)
response_time_traditional = time.time() - start_time

"""**Define Ray classes and remotes**"""

@ray.remote
class XRacer():
  
  rules = {}
  final_rules = {}
  columns = {}
  classes = []

  # Fitness Value = alpha * accuracy + beta * coverage
  # accuracy coef : accuracy is the percent of covered instances which are correctly classified 
  # coverage coef : coverage is the percent of instances which are covered among the training set
  alpha = 0.5   
  beta = 0.5  

  rules_size = 0
  d_size = 0
  X = None
  Y = None
  def __init__(self, alpha, beta, X, Y ):  # initialize valiables
    self.alpha = alpha
    self.beta = beta
    self.rules = {}
    self.final_rules = {}
    self.columns = {}
    self.classes = []
    self.rules_size = 0
    self.d_size = 0
    self.X = X
    self.Y = Y
  
  def fit(self):  
    X = self.X
    Y = self.Y
    keys = Y.keys()
    
    if Y[keys[0]].dtype.name != "category":
      raise ValueError(f'Y must be of type "category", "{Y.loc[0].dtype.name}" given')
    for i in Y[keys[0]].cat.categories:
      self.classes.append(i)
      self.rules[i] = []
    
    cols = X.columns.copy()
    self.d_size = X.shape[0]
    for c in cols:
      if X[c].dtype.name != "category":
        raise ValueError(f'All columns must be of type "category", "{X[c].dtype.name}" given')
      self.columns[c] = []
      for i in X[c].cat.categories:
        self.columns[c].append(i)
    
    self.X = X.copy()
    self.Y = Y.copy()
    # self.rules_size += len(self.classes)
    for key in self.columns:
      self.rules_size += len(self.columns[key])

    self.convert_rules()

    self.proccess_rules()


  
  def proccess_rules(self):
    for c in self.classes:
      extant_rules = self.rules[c].copy()
      extant_rules = [[i,0] for i in extant_rules]
      for i in range(len(extant_rules)):
        for j in range(i+1, len(extant_rules)):
          if extant_rules[i][1] == 0 and extant_rules[j][1] == 0:
            composed_rule = self.composition(extant_rules[i][0], extant_rules[j][0])
            if self.fitness(composed_rule, c) > self.fitness(extant_rules[i][0], c) and self.fitness(composed_rule, c) > self.fitness(extant_rules[j][0], c):
              extant_rules[i][0] = composed_rule
              extant_rules[j][1] = 1
              for x in range(len(extant_rules)):
                if x != i and self.rule_covers(composed_rule, extant_rules[x][0]):
                  extant_rules[x][1] = 1
      
      for i in range(len(extant_rules)):
        if extant_rules[i][1] == 0:
          extant_rules[i][0] = self.generalize(extant_rules[i][0], c)
      
      final = sorted(extant_rules, key=lambda x: self.fitness(x[0], c), reverse=True)
      self.final_rules[c] = [[i[0], self.fitness(i[0], c)] for i in final if i[1] == 0]
      # self.final_rules[c] = final.copy()

  
  def predict(self, X):
    cols = X.columns.copy()
    Y = []
    for c in cols:
      if X[c].dtype.name != "category":
        raise ValueError(f'All columns must be of type "category", "{X[c].dtype.name}" given')
    
    for i in range(X.shape[0]):
      rule = self.generate_rule(X.iloc[i])
      rules = []
      for c in self.classes:
        for j in range(len(self.final_rules[c])):
          if self.rule_covers(self.final_rules[c][j][0], rule):
            rules.append([c, self.final_rules[c][j][1], j])
      rules.sort(key=operator.itemgetter(1, 2), reverse=True)
      if len(rules)==0:
        Y.append(-1)
        continue
      Y.append(rules[0][0])
    
    return Y

    
  def generalize(self, rule, c):
    for i in range(len(rule)):
      if rule[i] == 0:
        new_rule = rule
        new_rule[i] = 1
        if self.fitness(new_rule, c) > self.fitness(rule, c):
          rule = new_rule
    return rule

  def composition(self, rule1, rule2):
    result = ""
    for i in range(self.rules_size):
      if rule1[i] == "1" or rule2[i] == "1":
        result += "1"
      else:
        result += "0"
    
    return result
  
  def fitness(self, rule, cls):
    return self.alpha * self.accuracy(rule, cls) + self.beta * self.coverage(rule)
  
  def accuracy(self, rule, cls):
    return self.n_correct(rule, cls) / self.n_covers(rule)
  
  def coverage(self, rule):
    return self.n_covers(rule) / self.d_size

  def n_correct(self, rule1, cls):
    num = 0
    for rule in self.rules[cls]:
      correct = True
      for i in range(self.rules_size):
        if rule1[i] != rule[i] and rule1[i] == "0":
          correct = False
      if correct:
        num += 1
    
    return num

  def n_covers(self, rule1):
    num = 0
    for cls in self.classes:
      for rule in self.rules[cls]:
        correct = True
        for i in range(self.rules_size):
          if rule1[i] != rule[i] and rule1[i] == "0":
            correct = False
        if correct:
          num += 1
    
    return num


  def rule_covers(self, rule1, rule2):
    for i in range(self.rules_size):
      if rule2[i] == "1" and rule1[i] == "0":
        return False
    return True


  def convert_rules(self):
    for i in range(self.X.shape[0]):
      self.generate_rules(self.X.iloc[i], self.Y.iloc[i])

  def generate_rule(self, input):
    rule = ""
    for key in self.columns:
      sub_rule = ""
      for i in self.columns[key]:
        if i == input[key]:
          sub_rule += "1"
        else:
          sub_rule += "0"
      rule += sub_rule
    return rule
  
  def generate_rules(self, input, output):
    rule = ""
    for key in self.columns:
      sub_rule = ""
      for i in self.columns[key]:
        if i == input[key]:
          sub_rule += "1"
        else:
          sub_rule += "0"
      rule += sub_rule
    rule_class = None
    for i in self.classes:
      if (i == output).bool():
        rule_class = i

    
    #self.rules[rule_class].append(rule)
    self.rules[rule_class].append(rule)
  def get_final_rules(self):
    return self.final_rules

#====================================================
machines = [XRacer.remote(alpha=0.7,beta=0.3,X=X[i],Y=Y[i]) for i in range(n)]
machines

"""**Call fit function**"""

start_time = time.time() 
#ray.get(machines[1].fit.remote())
ray.get([m.fit.remote() for m in machines])

#racer = Racer(alpha=0.7,beta=0.3,X=X[0],Y=Y[0])

rules = ray.get([m.get_final_rules.remote() for m in machines])
rules.append(rules_ten)
classes = Aracer.classes
rules_aggrigated = {}
for cls in classes:
  rules_aggrigated[cls] = []
  for j,c in enumerate(rules):
    rules_aggrigated[cls].append(rules[j][cls])
final_rules = {}
for c in rules_aggrigated:
  final_rules[c] = []
  for l,i in enumerate(rules_aggrigated[c]):
    for r in rules_aggrigated[c][l]:
      final_rules[c].append(r[0]) 

####################################     
total_rules = 0
for c in final_rules:
  total_rules = total_rules+len(final_rules[c])
total_rules

MRacer = Racer(alpha=0.7,beta=0.3)
MRacer.rules = final_rules
MRacer.classes = classes
MRacer.d_size = total_rules
MRacer.proccess_rules()
response_time = time.time() - start_time
Y_pred = MRacer.predict(X_all)
print("Accuracy in distributed mode : %.2f %%" % (accuracy_score(Y_pred, Y_all)*100))
print("Response time in distributed mode : %.2f" % (response_time))
Y_pred = Aracer.predict(X_ten)
print("Accuracy in traditional mode : %.2f %%" % (accuracy_score(Y_pred, Y_ten)*100))
print("Response time in traditional mode : %.2f" % (response_time_traditional))