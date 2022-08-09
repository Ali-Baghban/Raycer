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
    
  def feature_uniqer(X):
    rules_size = 0
    columns = {}
    cols = X.columns.copy()
    for c in cols:
        if X[c].dtype.name != "category":
            raise ValueError("Error !!!")
        columns[c] = []
        for i in X[c].cat.categories:
            columns[c].append(i)
    return columns


  def fit(self, X, Y, C):  
    keys = Y.keys()
    self.columns = C
    if Y[keys[0]].dtype.name != "category":
      raise ValueError(f'Y must be of type "category", "{Y.loc[0].dtype.name}" given')
    for i in Y[keys[0]].cat.categories:
      self.classes.append(i)
      self.rules[i] = []
    
    self.d_size = X.shape[0]
    
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
      for i in self.columns[key]:    ##### if exist
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

    
    self.rules[rule_class].append(rule)