import numpy as np
from random import shuffle
from sklearn import metrics  # pyright: ignore[reportMissingModuleSource]

from sklearn.datasets import fetch_openml # pyright: ignore[reportMissingModuleSource]
from sklearn.linear_model import Perceptron  # pyright: ignore[reportMissingModuleSource]
from sklearn.svm import SVC  # pyright: ignore[reportMissingModuleSource]
from sklearn.naive_bayes import GaussianNB  # pyright: ignore[reportMissingModuleSource]
from sklearn.neighbors import KNeighborsClassifier  # pyright: ignore[reportMissingModuleSource]
from sklearn.tree import DecisionTreeClassifier # pyright: ignore[reportMissingModuleSource]



def load_dataset():
    return fetch_openml(name='phoneme', cache=False, as_frame=False)


data = load_dataset()
xdata = data.data
ytarg = data.target


# embaralhar os dados
idx = list(range(len(ytarg)))
shuffle(idx)
part = int(len(ytarg)*0.6) # assumindo 60% treino

# xtr --> x_treino  ;  xte --> x_teste
xtr = xdata[ :part ]
ytr = ytarg[ :part ]
xte = xdata[ part: ]
yte = ytarg[ part: ]


rng = np.random.RandomState()

perceptron = Perceptron(max_iter=100,random_state=rng)
model_svc = SVC(probability=True, gamma='auto',random_state=rng)
model_bayes = GaussianNB()
model_tree = DecisionTreeClassifier(random_state=rng, max_depth=10)
model_knn = KNeighborsClassifier(n_neighbors=7)

# colocando todos classificadores criados em um dicionario
clfs = {    'perceptron':   perceptron,
            'svm':          model_svc,
            'bayes':        model_bayes,
            'trees':        model_tree,
            'knn':          model_knn
        }

ytrue = yte
print('Treinando cada classificador e encontrando o score')
for clf_name, classific in clfs.items():
    classific.fit(xtr, ytr)
    ypred = classific.predict(xte)
    matrconf = metrics.confusion_matrix(ytrue, ypred)
    acc = metrics.accuracy_score(ytrue, ypred)
    f1 = metrics.f1_score(ytrue, ypred, average='macro')
    print(clf_name, '-- f1:', f1)

