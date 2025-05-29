import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA

##copied data from from https://www.geeksforgeeks.org/implementing-svm-from-scratch-in-python/

class SVM:
    def __init__(self, config, features, classes, learning_rate=0.001, lambda_param=0.01, n_iters=1000):

        self.kernel = config.feature_extraction.svm_kernel

        #NOTE currently not sepearted to training/classificaiton everything is in training
        ## X is (n_samples, n_features) 
        ## y must be (n_samples)
        # check if features are 2D vs 1D and flatte

        self.config = config
        self.features = features
        self.X = features.reshape(features.shape[0], -1) #flattens all the dimensions and must be shape (n_samples, feats)
        self.y = classes.flatten() #must be shape (n_samples)
        return
    
    def fit_pca_svm(self,X):
        # Reduce to 2D you must or this wont work to visualise SVM
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)

        clf = SVC(kernel=self.kernel)
        clf.fit(X_reduced, self.y)

        return clf, X_reduced
    
    def fit_svm_pca(self, X, c):
        # Reduce to 2D you must or this wont work to visualise SVM
        clf = SVC(kernel=self.kernel)
        clf.fit(X, self.y)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        # Project the SVM decision function to PCA space
        x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
        y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                            np.linspace(y_min, y_max, 200))

        # Inverse transform PCA grid points back to original 100D space
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        grid_original = pca.inverse_transform(grid_points)

        # Evaluate SVM decision function 
        Z = clf.decision_function(grid_original)
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(8, 6))
        # Decision boundary and margins
        plt.contourf(xx, yy, Z, levels=[Z.min(), 0, Z.max()], colors=['#CCCCFF','#FFCCCC'], alpha=0.3)
        plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='k')

        # Scatter plot of PCA-reduced data
        plt.scatter(X_pca[self.y==0, 0], X_pca[self.y==0, 1], color=['blue'],label="Con", edgecolor='k', alpha=0.7)
        plt.scatter(X_pca[self.y==1, 0], X_pca[ self.y==1, 1], color=['red'],label="PAN", edgecolor='k', alpha=0.7)

        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.title("SVM Decision Boundary in PCA Space")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('./svm_pca'+str(c))
        plt.close()

        return clf
    
    # Plot decision boundary
    def plot_svm_decision_boundary(self, clf, X, y, c):
        # Plot the points only first two dimension
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=30)

        # Plot the decision boundary
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Create grid to evaluate model
        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = clf.decision_function(xy).reshape(XX.shape)
        
        # Plot decision boundary and margins
        # plt.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1],
        #             alpha=0.5, linestyles=['--', '-', '--'])
        plt.contour(XX, YY, Z, levels=[0], linewidths=2, colors='k')
        try:
            plt.contourf(XX, YY, Z, levels=[Z.min(), 0, Z.max()], colors=['#CCCCFF','#FFCCCC'], alpha=0.3)
        except:
            print('predicted as class zero')

        # Plot support vectors
        plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                    s=100, linewidth=1, facecolors='none')

        plt.title("SVM Decision Boundary with Support Vectors")
        plt.savefig('./pca_svm'+str(c))
        plt.close()

        return

    def run_PCA_SVM(self, plt_decision = True):

        for c in range(self.features.shape[3]):
            _one_channel = self.features[:,:,:,c,:]
            X = _one_channel.reshape(_one_channel.shape[0], -1)

            clf, X_reduced = self.fit_pca_svm(X)
            # Call the function
            self.plot_svm_decision_boundary(clf,X_reduced, self.y, c)

        return
    

    def run_SVM_PCA(self, plt_decision = True):
        ## this function instead runs the svm prior to PCA
        #self.X = features.reshape(features.shape[0], -1) #flattens all the dimensions and must be shape (n_samples, feats)
        #loop through each channel
        for c in range(self.features.shape[3]):
            _one_channel = self.features[:,:,:,c,:]
            X = _one_channel.reshape(_one_channel.shape[0], -1)
            self.fit_svm_pca(X, c)

        return
    # def pred_svm(self):
        # not created yet
    #     clf = SVC(kernel='linear')
    #     clf.fit(X, y)

    #     return print(clf.predict(samples))
