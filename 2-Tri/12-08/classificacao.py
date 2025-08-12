import numpy as np
import sys
import os
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import fetch_openml, make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Import simplificado da classe dataset
sys.path.append('../29-07')
from dataset import data_set

rng = np.random.RandomState(42)

# Usar a classe local ao invés do fetch_openml
print("Carregando dataset adult usando classe local...")
adult_path = os.path.join(os.path.dirname(__file__), '..', 'adultDataset', 'adult.csv')
result = data_set(adult_path)

xdata = np.array(result['dados'])
ytarg = np.array(result['classes'])

print(f"Dataset carregado com sucesso! Shape: {xdata.shape}")
print(f"Classes únicas: {np.unique(ytarg)}")

X_train, X_test, y_train, y_test = train_test_split(xdata, ytarg, test_size=0.25, random_state=rng)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


perceptron = Perceptron(max_iter=100, random_state=rng)
perceptron.fit(X_train, y_train)

# model_svc = SVC(probability=True, gamma='auto', random_state=rng)
# model_svc.fit(X_train, y_train)

# model_bayes = GaussianNB()
# model_bayes.fit(X_train, y_train)

# model_tree = DecisionTreeClassifier(random_state=rng, max_depth=10)
# model_tree.fit(X_train, y_train)

# model_knn = KNeighborsClassifier(n_neighbors=7)
# model_knn.fit(X_train, y_train)


print('Evaluating DS techniques:')
print('perceptron:', perceptron.score(X_test, y_test))
# print('svm:', model_svc.score(X_test, y_test))
# print('bayes:', model_bayes.score(X_test, y_test))
# print('tree:', model_tree.score(X_test, y_test))
# print('knn:', model_knn.score(X_test, y_test))




