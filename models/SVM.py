import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

##copied data from from https://www.geeksforgeeks.org/implementing-svm-from-scratch-in-python/

class SVM:
    def __init__(self, config, features, classes, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        #training defaults
        self.learning_rate=0.001
        self.lambda_param=0.01
        self.n_iters=1000

        #NOTE currently not sepearted to training/classificaiton everything is in training
        ## X is (n_samples, n_features) 
        ## y must be (n_samples)
        # check if features are 2D vs 1D and flatte

        self.config = config
        self.X = features.reshape(2, -1) #flattens all the dimensions
        self.y = classes.T 
        return
    
    def fit(self):
        clf = SVC(kernel='linear')
        clf.fit(self.X, self.y.flatten())
        return clf
    # Plot decision boundary
    def plot_svm_decision_boundary(self, clf, X, y):
        # Create a mesh grid
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                            np.linspace(y_min, y_max, 500))

        # Predict over the grid
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot everything
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
        plt.title("SVM Decision Boundary")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        return

    def run_SVM(self, plt_decision = True):

        clf = self.fit(self.X, self.lambda_param)
        # Call the function
        self.plot_svm_decision_boundary(clf, self.X, self.y)

        plt.savefig('./svm')
        return
    
    # def pred_svm(self):
        # not created yet
    #     clf = SVC(kernel='linear')
    #     clf.fit(X, y)

    #     return print(clf.predict(samples))
