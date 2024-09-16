"""Created some day.

@authors:  salmon, gramfort, vernade
"""

#%%
from functools import partial  # useful for weighted distances
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import stats  # to use scipy.stats.mode
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn import neighbors
from sklearn import datasets

from tp_knn_source import (rand_gauss, rand_bi_gauss, rand_tri_gauss,
                           rand_checkers, rand_clown, plot_2d, ErrorCurve,
                           frontiere, LOOCurve)


import seaborn as sns
from matplotlib import rc

plt.close('all')
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Computer Modern Roman']})
params = {'axes.labelsize': 12,
          'font.size': 16,
          'legend.fontsize': 16,
          'text.usetex': False,
          'figure.figsize': (8, 6)}
plt.rcParams.update(params)

sns.set_context("poster")
sns.set_palette("colorblind")
sns.set_style("white")

#%%
############################################################################
#     Data Generation: example
############################################################################

# Q1
np.random.seed(42)  # fix seed globally

n = 100
mu = [1., 1.]
sigma = [1., 1.]
rand_gauss(n, mu, sigma)

n1 = 20
n2 = 20
mu1 = [1., 1.]
mu2 = [-1., -1.]
sigma1 = [0.9, 0.9]
sigma2 = [0.9, 0.9]
X1, y1 = rand_bi_gauss(n1, n2, mu1, mu2, sigma1, sigma2)

n1 = 50
n2 = 50
n3 = 50
mu1 = [1., 1.]
mu2 = [-1., -1.]
mu3 = [1., -1.]
sigma1 = [0.9, 0.9]
sigma2 = [0.9, 0.9]
sigma3 = [0.9, 0.9]
X2, y2 = rand_tri_gauss(n1, n2, n3, mu1, mu2, mu3, sigma1, sigma2, sigma3)

n1 = 50
n2 = 50
sigma1 = 1.
sigma2 = 5.
X3, y3 = rand_clown(n1, n2, sigma1, sigma2)

n1 = 150
n2 = 150
sigma = 0.1
X4, y4 = rand_checkers(n1, n2, sigma)

#%%
############################################################################
#     Displaying labeled data
############################################################################

plt.show()
plt.close("all")
plt.ion()
plt.figure(figsize=(15, 5))
plt.subplot(141)
plt.title('First data set')
plot_2d(X1, y1)

plt.subplot(142)
plt.title('Second data set')
plot_2d(X2, y2)

plt.subplot(143)
plt.title('Third data set')
plot_2d(X3, y3)

plt.subplot(144)
plt.title('Fourth data set')
plot_2d(X4, y4)

#%%
############################################################################
#     K-NN
############################################################################

# Q2 : Write your own implementation


class KNNClassifier(BaseEstimator, ClassifierMixin):
    """Home made KNN Classifier class."""

    #attributs : on donne à chaque objet de la classe une valeur qui lui est propre
    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors

    #methodes 
    def fit(self, X, y):
        self.X_ = X #self.X_ : attribut de notre classe , X : local variable
        self.y_ = y
        return self
    #fit definit deux nouveaux attributs

    #méthodes
    def predict(self, X):
        n_samples, n_features = X.shape 
        # TODO: Compute all pairwise distances between X and self.X_ using e.g. metrics.pairwise.pairwise_distances
        dist = metrics.pairwise.pairwise_distances(X, self.X_,metrics='euclidean',n_jobs=1)
        # Get indices to sort them
        idx_sort = np.argsort(dist, axis=1)
        # Get indices of neighbors
        idx_neighbors = idx_sort[:, :self.n_neighbors]
        # Get labels of neighbors
        y_neighbors = self.y_[idx_neighbors]
        # Find the predicted labels y for each entry in X
        # You can use the scipy.stats.mode function
        mode, _ = stats.mode(y_neighbors, axis=1)
        # the following might be needed for dimensionaality
        y_pred = np.asarray(mode.ravel(), dtype=int)
        return y_pred

# TODO : compare your implementation with scikit-learn

# Focus on dataset 2 for instance
X_train = X2[::2]
Y_train = y2[::2].astype(int)
X_test = X2[1::2]
Y_test = y2[1::2].astype(int)

#%%
def f(xx):
    """Classifier: needed to avoid warning due to shape issues"""
    return knn.predict(xx.reshape(1, -1))
knn = KNNClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
plt.figure()
frontiere(knn, X_train, Y_train, w=None, step=50, alpha_choice=1)


# TODO: use KNeighborsClassifier vs. KNNClassifier

#%%
# Q3 : test now all datasets
# From now on use the Scikit-Learn implementation

n_neighbors = 5  # the k in k-NN
knn = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
for X, y in [(X1, y1), (X2, y2), (X3, y3), (X4, y4)]:
    X_tr = X[::2]
    Y_tr = y[::2].astype(int)
    X_te = X[1::2]
    Y_te = y[1::2].astype(int)
    knn.fit(X_tr,Y_tr)
    y_pred = knn.predict(X_te)
    # Afficher les prédictions et les étiquettes réelles
    #print("Predictions:", y_pred)
    #print("True Labels:", Y_test)
    print(knn.score(X_te, Y_te))

# Créer une figure avec 4 sous-graphiques
plt.figure(figsize=(20, 5))

# for data in [data1, data2, data3, data4]:
for i, (X, y) in enumerate([(X1, y1), (X2, y2), (X3, y3), (X4, y4)], start=1):
    knn.fit(X, y)
    plt.subplot(1, 4, i)
    plot_2d(X, y)
    n_labels = np.unique(y).shape[0]
    frontiere(knn, X, y, w=None, step=50, alpha_choice=1,
              n_labels=n_labels, n_neighbors=n_neighbors)

# Afficher tous les sous-graphiques
plt.tight_layout()  # Ajuster l'espacement entre les sous-graphiques
plt.show()

#%%
# Q4: Display the result when varying the value of K

plt.figure(figsize=(12, 8))
plt.subplot(3, 5, 3)
plot_2d(X_train, Y_train)
plt.xlabel('Samples')
ax = plt.gca()
ax.get_yaxis().set_ticks([])
ax.get_xaxis().set_ticks([])

for n_neighbors in range(1, 11):
    knn = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train,Y_train)
    y_pred = knn.predict(X_test)
    accuracy = sklearn.metrics.accuracy_score(Y_test, y_pred)
    print(f"Accuracy: {accuracy:.3f}")
    plt.subplot(3, 5, 5 + n_neighbors)
    plt.xlabel(f'KNN with k={n_neighbors}, \n accu={accuracy:.3f}')

    n_labels = np.unique(y).shape[0]
    frontiere(knn, X, y, w=None, step=50, alpha_choice=1,
              n_labels=n_labels, colorbar=False, samples=False,
              n_neighbors=n_neighbors)

plt.draw()  # update plot
plt.tight_layout()


#%%
# Q5 : Scores on train data

for n_neighbors in range(1, 11):
    knn = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, Y_train)
    y_pred_train = knn.predict(X_train)
    accuracy_train = sklearn.metrics.accuracy_score(Y_train, y_pred_train)
    error_rate_train = 1 - accuracy_train
    print(f"K={n_neighbors} - Training Accuracy: {accuracy_train:.3f}, Training Error Rate: {error_rate_train:.3f}")

for n_neighbors in range(1, 11):
    knn = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_test, Y_test)
    y_pred_test = knn.predict(X_test)
    accuracy_test = sklearn.metrics.accuracy_score(Y_test, y_pred_test)
    error_rate_test = 1 - accuracy_test
    print(f"K={n_neighbors} - Testing Accuracy: {accuracy_test:.3f}, Testing Error Rate: {error_rate_test:.3f}")
#%%
# Q6 : Scores on left out data : Correction

collist = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
sigma = 0.1
plt.figure()
range_n_samples = [100, 500, 1000]
niter = len(range_n_samples)
for n in range(niter):
    n1 = n2 = range_n_samples[n]
    X_train, Y_train = rand_checkers(n1, n2, sigma)
    X_test, Y_test = rand_checkers(n1, n2, sigma)
    error_curve = ErrorCurve(k_range=range(1, 50))
    error_curve.fit_curve(X_train, Y_train, X_test, Y_test)
    error_curve.plot(color=collist[n % len(collist)], maketitle=False)
plt.legend(["training size : %d" % n for n in range_n_samples],
loc='upper left')
plt.draw()
#%%
############################################################################
#     Digits data
############################################################################

# Q8 : test k-NN on digits dataset

# The digits dataset
digits = datasets.load_digits()

print(type(digits))
# A Bunch is a subclass of 'dict' (dictionary)
# help(dict)
# see also "http://docs.python.org/3/library/stdtypes.html#mapping-types-dict"


# plot some images to observe the data
plt.figure()
for index, (img, label) in enumerate(list(zip(digits.images, digits.target))[10:20]):
    plt.subplot(2, 5, index + 1)
    plt.axis('off')
    plt.imshow(img, cmap=plt.cm.gray_r, interpolation='None')
    plt.title('%i' % label)
plt.draw()

# Check that the dataset is balanced
plt.figure()
plt.hist(digits.target, density=True)
plt.title("Labels frequency over the whole dataset")
plt.ylabel("Frequency")
plt.draw()

#%%
n_samples = len(digits.data)

X_digits_train = digits.data[:n_samples // 2]
Y_digits_train = digits.target[:n_samples // 2]
X_digits_test = digits.data[n_samples // 2:]
Y_digits_test = digits.target[n_samples // 2:]

# Check that the test dataset is balanced
plt.figure()
plt.hist(Y_digits_test, density=True)
plt.title("Labels frequency on the test dataset")


knn = neighbors.KNeighborsClassifier(n_neighbors=30)
knn.fit(X_digits_train, Y_digits_train)

score = knn.score(X_digits_test, Y_digits_test)
Y_digits_pred = knn.predict(X_digits_test)
print('Score : %s' % score)


#%%
# Q9 : Compute confusion matrix: use sklearn.metrics.confusion_matrix

Y_pred = knn.predict(X_test)
CM = metrics.confusion_matrix(Y_test, Y_pred)
print(CM)
CM_norm = 1.0 * CM / (CM.sum(axis=1)[:, np.newaxis])
print(CM_norm)
plt.matshow(CM)
# TODO : compute and show confusion matrix

#%%
# Q10 : Estimate k with cross-validation for instance

# Have a look at the class  'LOOCurve', defined in the source file.
plt.figure()
loo_curve = LOOCurve(k_range=list(range(1, 50, 5)) + list(range(100, 300, 100)))
loo_curve.fit_curve(X=digits.data, y=digits.target)
# TODO

# %%
