import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier

from data_loader import Data_loader



# Init data_loader
data_loader = Data_loader()
Xs_lsi, Ys_lsi, Xa_lsi,Ya_lsi = data_loader.load_LSI()

X, y = [], []
p = 2*data_loader.num_topics

# Fetch all duplicate data
print('\n Duplicate data \n')
for q1,q2 in zip(Xs_lsi, Ys_lsi):
    try:
        q12 = np.concatenate((q1,q2),axis=0)
        q12 = q12.reshape(p)
        X.append(q12)
        y.append(1)
    except:
        pass
        #print(q1.shape)
        #print(q2.shape)

# Fetch all NON duplicate data
print('\n Non Duplicate data \n')
for q1,q2 in zip(Xa_lsi, Ya_lsi):
    try:
        q12 = np.concatenate((q1,q2),axis=0)
        q12 = q12.reshape(p)
        X.append(q12)
        y.append(0)
    except:
        pass
        #print(q1.shape)
        #print(q2.shape)


# Cross validation
X = np.stack(X)
y = np.asarray(y)
print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

# Simple classifier (stack Xs,Ys --> 1, Xs,Ys --> 0)
#clf = ExtraTreesClassifier(criterion='entropy', n_estimators=30, max_depth=100, max_features='sqrt') # 0.78 test acc
#clf = RandomForestClassifier() # 0.749
#clf = SGDClassifier(penalty='l2') # 0.70
clf = MLPClassifier(hidden_layer_sizes=(400,), verbose=True, early_stopping=True) # 0.78 test acc
clf.fit(X_train, y_train)

predicted_train = clf.predict(X_train)
predicted_test = clf.predict(X_test)

accuracy_train = 1-np.mean(np.abs(predicted_train-y_train))
accuracy_test = 1-np.mean(np.abs(predicted_test-y_test))
print('Training accuracy',accuracy_train)
print('Testing accuracy',accuracy_test)