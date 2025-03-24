import matplotlib.pyplot as plt
import numpy as np
import os 
from scipy import stats
import sklearn
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import math
class Compare():
    def __init__(self, config, dataloader):
        self.config = config

        self.feature= config.feature_extraction.method

        self.control_str = config.data.control
        self.disease_str = config.data.disease
        
        self.dataloader = dataloader

        self.output_dir = config.output.loc
        
        self.output_sub_dir = config.feature_extraction.method
        os.makedirs(os.path.join(self.output_dir,self.output_sub_dir), exist_ok=True)


        self.plt_name = self.output_sub_dir+'_'+self.control_str+'_'+self.disease_str+'_comparison'

        self.comparison_type = config.feature_extraction.compare
        
        self.img_details_arr, self.img_torch, self.img_class_torch, self.img_features_torch, self.img_features_labels_arr = dataloader.dataset.img_details, dataloader.dataset.img_arr, dataloader.dataset.img_class, dataloader.dataset.img_features, dataloader.dataset.img_feat_labels

        pass

    def normalize_array(self, arr):
        norm_arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        return norm_arr

    def plot_stat_difference(self, normalize = True, alpha = 0.05):
        '''This function takes the feature arrays for control and disease and takes average of each feature.
         Then it plots again list of features (label). '''

        feature_labels = np.squeeze(self.img_features_labels_arr[0],0)
        feat_arr = np.squeeze(self.img_features_torch.cpu().detach().numpy(), axis=1)
        if normalize == True:
            feat_arr = self.normalize_array(feat_arr)

        classes = np.squeeze(self.img_class_torch.cpu().detach().numpy(), axis=0)

        ##separate into different classes
        #get index where classes = 0 for control
        index_control, index_disease = np.where(classes == 0)[0], np.where(classes == 1)[0]

        control_feats, control_std = np.mean(feat_arr[index_control],axis=0), np.std(feat_arr[index_control],axis=0)
        disease_feats, disease_std = np.mean(feat_arr[index_disease],axis=0), np.std(feat_arr[index_disease],axis=0)


        ##for each channel calculate the feature difference
        channels = feat_arr.shape[1]
        for c in range(channels):
            # Create the bar plot
            fig, ax = plt.subplots(constrained_layout=True)
            x = np.arange(len(control_feats[c]))
            width = 0.3
            bars1 = ax.bar(x - width/2, control_feats[c], width, yerr=control_std[c], capsize=5, label='control', color='green', alpha=0.7)
            bars2 = ax.bar(x + width/2, disease_feats[c], width, yerr=disease_std[c], capsize=5, label='disease', color='blue', alpha=0.7)

            for f in range(len(feature_labels)):
                vals_disease=feat_arr[index_disease][:,c,f]
                vals_control=feat_arr[index_control][:,c,f]
                t_stat, p_value = stats.ttest_ind(vals_disease, vals_control)
                print(p_value)
                if p_value < alpha:
                    ax.text(f, y=np.max(np.concat((vals_control,vals_disease))), s="*", ha='center', va='center', fontsize=15)

            # Add labels and title
            ax.set_ylabel('Feature Values')
            plt.xticks(rotation=45)
            ax.set_xticks(x)
            ax.set_xticklabels(feature_labels.tolist())  # Set custom x-axis labels
            
            plt.legend()
            plt.show()
            plt.title(self.plt_name+'_'+str(c))
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir,self.output_sub_dir,self.comparison_type[0]+'_channel'+str(c)+'_bar_graph'))
            plt.close()

        return
    
    def histogram_per_img(self, bins = 30):
        '''input image features are N,f,C,h,w
        N: number of samples
        f: features
        C: channel
        h and w: height and width of image '''
        feature_labels = np.squeeze(self.img_features_labels_arr[0],0)
        feat_arr = np.squeeze(self.img_features_torch.cpu().detach().numpy(), axis=1)

        classes = np.squeeze(self.img_class_torch.cpu().detach().numpy(), axis=0)

        ##separate into different classes
        #get index where classes = 0 for control
        index_control, index_disease = np.where(classes == 0)[0], np.where(classes == 1)[0]

        control_feats_arr, disease_feats_arr  = feat_arr[index_control],  feat_arr[index_disease]

        cont_feat_flatten= np.reshape(control_feats_arr, (control_feats_arr.shape[0],control_feats_arr.shape[1], int(np.prod(control_feats_arr.shape[2:]))))
        disease_feat_flatten=np.reshape(disease_feats_arr, (disease_feats_arr.shape[0],disease_feats_arr.shape[1], int(np.prod(disease_feats_arr.shape[2:]))))
        
        #plt individual
        for f in range(len(feature_labels)):
            feature_name = feature_labels[f]
            print(feature_name)
            kwargs = dict(histtype='stepfilled', alpha=0.3, bins=bins)

            for pat in range(cont_feat_flatten.shape[0]):
                if pat == 0:
                    plt.hist(self.normalize_array(cont_feat_flatten[pat,f,:]), color = 'g',label=self.control_str, **kwargs)
                else:
                    plt.hist(self.normalize_array(cont_feat_flatten[pat,f,:]), color = 'g', **kwargs)
   
            for pat in range(disease_feat_flatten.shape[0]):
                if pat == 0:
                    plt.hist(self.normalize_array(disease_feat_flatten[pat,f,:]), color = 'b', label=self.disease_str, **kwargs)
                else:
                    plt.hist(self.normalize_array(disease_feat_flatten[pat,f,:]), color = 'b', **kwargs)


            plt.title(feature_name, fontsize=10, pad=0)
            plt.show()
            plt.legend()
            plt.savefig(os.path.join(self.output_dir,self.output_sub_dir,feature_name+'_histogram'))
            plt.close()
        
        # #plt all
                
        # #fatten features in histogram
        # grid_size = int(np.ceil(math.sqrt(len(feature_labels))))
        # fig, axes = plt.subplots(grid_size,grid_size,sharex=True,sharey=True,figsize=(grid_size,grid_size))
        # axes = axes.ravel()

        # for f in range(len(feature_labels)):
        #     feature_name = feature_labels[f]
        #     print(feature_name)
        #     kwargs = dict(histtype='stepfilled', alpha=0.3, bins=20)

        #     for pat in range(cont_feat_flatten.shape[0]):
        #         axes[f].hist(cont_feat_flatten[pat,f,:], color = 'g', **kwargs)

        #     for pat in range(cont_feat_flatten.shape[0]):
        #         axes[f].hist(disease_feat_flatten[pat,f,:], color = 'b', **kwargs)

        #     axes[f].set_title(feature_name, fontsize=3, pad=0)

        return
    
    def run_PCA(self, scale_data=False):
        _feat_arr = np.squeeze(self.img_features_torch.cpu().detach().numpy(), axis=1)

        fig, axes = plt.subplots(1,_feat_arr.shape[2],figsize=(5*_feat_arr.shape[2],5))

        for c in range(_feat_arr.shape[2]):
            feat_arr = _feat_arr[:,:,c,:,:]
            classes = np.squeeze(self.img_class_torch.cpu().detach().numpy(), axis=0)

            flatten_feat_arr =  np.reshape(feat_arr, (feat_arr.shape[0], np.prod(feat_arr.shape[1:])))

            #Scale data
            if scale_data == True:
                scaler = StandardScaler()
                scaler.fit(flatten_feat_arr)
                flatten_feat_arr=scaler.transform(flatten_feat_arr)    

            feats_fit_pca = PCA(n_components=2).fit_transform(flatten_feat_arr)

            axes[c].set_xlim(-1,1)
            axes[c].set_ylim(-1,1)
            axes[c].set_xlabel("Component 1")
            axes[c].set_ylabel("Component 2")
            axes[c].grid()
            axes[c].set_title('Channel '+str(c))

            score=feats_fit_pca[:,0:2]
            xs = score[:,0]
            ys = score[:,1]
            scalex = 1.0/(xs.max() - xs.min())
            scaley = 1.0/(ys.max() - ys.min())
            axes[c].scatter(xs * scalex,ys * scaley, c = classes) # labels = [self.control_str, self.disease_str])
            axes[c].legend([self.control_str, self.disease_str])
            
        plt.show()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir,self.output_sub_dir,self.plt_name+'_PCA'))
        plt.close()
        return

    def _report(self):
        #returns the healthy vs the control values
        for compare_func in self.comparison_type:
            eval("self."+compare_func+"()")

        return