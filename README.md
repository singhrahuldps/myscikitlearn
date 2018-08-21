# myscikitlearn
My implementation of some Machine Learning Algorithms from scratch.

## Required libraries -
* Numpy
* Pandas
* Sklearn for accuracy_score

## Basic Usage
Download the zip, extract and rename the folder to **myscikitlearn**

Below is the gist for using various algorithms
```
clf = Classifier()
clf.fit(x_train, y_train)
pred = clf.predict(x_test)
```

## Current algorithms

#### Entropy based Decision Tree
`from myscikitlearn.tree import entropyDecisionTreeClassifier`

