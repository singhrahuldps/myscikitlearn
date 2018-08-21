import math
class entropyDecisionTreeClassifier:
    def __init__(self):
        self.dt = None

    class tree:
        def __init__(self):
            self.colname = None
            self.thisclass = None
            self.isdecisive = False
            self.availablecols = []
            self.trees = {}

    def entropy(self,column,findmax = False):
        elements = list(set(column))
        d={}
        for ele in elements:
            d[ele] = 0
        total = len(column)
        for i in column:
            d[i] += 1
        ent = 0
        for ele in d:
            val = d[ele]/total
            if val!=0:
                ent += -val * math.log(val,2)
        if findmax:
            maxclass = None
            maxv = 0
            for ele in d:
                if d[ele] > maxv:
                    maxv = d[ele]
                    maxclass = ele
            fin = False
            if maxv/total >= 0.95:
                fin = True
            return ent,maxclass,fin
        return ent

    def entropypercategory(self,column,y):
        categs = list(set(column))
        d={}
        count = len(column)
        for cat in categs:
            d[cat] = []
        for i in range(count):
            d[column[i]].append(y[i])
        ent = 0.0
        for cat in categs:
            ent += (len(d[cat])/count)*self.entropy(d[cat])
        return ent

    def retrievenewrows(self,x,y,col,cols):
        categs = list(set(x[col]))
        d={}
        for cat in categs:
            if len(cols) > 0:
                newx = x[cols].loc[x[col] == cat]
                newcols = list(newx)
            else:
                newx = []
                newcols = []
            newy = y.loc[x[col] == cat]
            d[cat] = [newx,newy,newcols]
        return d

    def builddecisiontree(self,x,y,xcols,ycol):
        a = self.tree()
        main_entropy,maxclass,fin = self.entropy(list(y[ycol].values),True)
        a.thisclass = maxclass
        if main_entropy == 0 or len(xcols) == 0 or fin:
            a.isdecisive = True
            return a
        best = None
        bestinfogain = -1
        index = 0
        for i,col in enumerate(xcols):
            ent = self.entropypercategory(list(x[col].values),list(y[ycol].values))
            infogain = main_entropy - ent
            if bestinfogain < infogain:
                bestinfogain = infogain
                best = col
                index = i
        a.colname = best
        a.availablecols = xcols[:index] + xcols[index+1:]
        newdict = self.retrievenewrows(x,y,best,a.availablecols)
        for cat in newdict:
            newx = newdict[cat][0]
            newy = newdict[cat][1]
            newcols = newdict[cat][2]
            a.trees[cat] = self.builddecisiontree(newx,newy,newcols,ycol)
        return a

    def predict_tree(self,dt,xtest):
        pred = []
        for index,row in xtest.iterrows():
            a = dt
            while(not a.isdecisive):
                val = row[a.colname]
                if val in a.trees:
                    a = a.trees[row[a.colname]]
                else:
                    break
            pred.append(a.thisclass)
        return pred

    def fit(self,x_train,y_train):
        self.dt = self.builddecisiontree(x_train,y_train,list(x_train),list(y_train)[0])

    def predict(self,x_test):
        return self.predict_tree(self.dt,x_test)

    def score(self,x_test,y_test):
        from sklearn.metrics import accuracy_score
        pred = self.predict(x_test)
        return accuracy_score(y_test,pred)