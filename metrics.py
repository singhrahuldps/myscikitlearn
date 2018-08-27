import numpy as np

def accuracy_score(y_test,pred):
	return (y_test.values.reshape(len(y_test.values))==np.array(pred)).sum()/len(pred)