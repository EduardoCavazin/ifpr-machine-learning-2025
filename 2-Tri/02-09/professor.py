import numpy as np
from random import shuffle
from sklearn import metrics

from sklearn.datasets import fetch_openml

from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def load_dataset():
    data = fetch_openml(name='phoneme', cache=False, as_frame=False)    
    # manipular os dados faltantes

    # normalizacao.. 
    # normalizar os dados entre minimo e maximo
    return data


data = load_dataset()
alldata = data.data
alltarg = data.target


results = {
            'perceptron':   [],
            'svm':          [],
            'bayes':        [],
            'trees':        [],
            'knn':          []
}

rng = np.random.RandomState()

def get_cv_value(xdata, ytarg):

    part = int(len(ytarg)*0.8) # assumindo 80%
    parcial_result = {
                'perceptron':   [],
                'svm':          [],
                'bayes':        [],
                'trees':        [],
                'knn':          []
    }

    for crossv in range(5):

        # xtr --> x_treino  ;  xte --> x_teste
        xtr = xdata[ :part ]
        ytr = ytarg[ :part ]
        xte = xdata[ part: ]
        yte = ytarg[ part: ]


        perceptron = Perceptron(max_iter=100,random_state=rng)
        model_svc = SVC(probability=True, gamma='auto',random_state=rng)
        model_bayes = GaussianNB()
        model_tree = DecisionTreeClassifier(random_state=rng, max_depth=10)
        model_knn = KNeighborsClassifier(n_neighbors=7)

        # colocando todos classificadores criados em um dicionario
        clfs = {    
                    'perceptron':   perceptron,
                    'svm':          model_svc,
                    'bayes':        model_bayes,
                    'trees':        model_tree,
                    'knn':          model_knn
                }

        ytrue = yte
        #print('Treinando cada classificador e encontrando o score')
        for clf_name, classific in clfs.items():
            classific.fit(xtr, ytr)
            ypred = classific.predict(xte)
            f1 = metrics.f1_score(ytrue, ypred, average='macro')
            print(clf_name, '-- f1:', f1)
            parcial_result[clf_name].append( f1 )


        ytarg = list(ytarg[ part: ]) + list(ytarg[ :part ])
        xdata = list(xdata[ part: ]) + list(xdata[ :part ])

        print('####\n####')

    for clf_name, result in parcial_result.items():
        value = sum(result)/len(result)
        print(clf_name, '-->', value)
        parcial_result[clf_name] = value

    return parcial_result


def principal():

    for exec_id in range(3):
        # embaralhar os dados
        idx = list(range(len(alltarg)))
        shuffle(idx)
        xdata = alldata[ idx ]
        ytarg = alltarg[ idx ]
        ret = get_cv_value(xdata, ytarg)
        print(ret)


principal()

#for clf_name, result in results.items():
#    print(clf_name, '-->', sum(result)/len(result))