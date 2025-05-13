import matplotlib.pyplot as plt
import numpy as np
import os 
from scipy import stats
import sklearn

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import math
import umap.umap_ as umap

from scipy.optimize import curve_fit
from scipy.stats import f as fff
from scipy.stats import mannwhitneyu

from visualization import Histograms
from visualization import Visuals

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

        try:
            self.exponential_comparison = config.feature_extraction.compare_exponential
        except:
            self.exponential_comparison = False

        
        self.img_details_arr, self.img_torch, self.img_class_torch, self.img_features_torch, self.img_features_labels_arr = dataloader.dataset.img_details, dataloader.dataset.img_arr, dataloader.dataset.img_class, dataloader.dataset.img_features, dataloader.dataset.img_feat_labels

        pass

    def normalize_array(self, arr):
        norm_arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        return norm_arr
    
    def plot_stat_difference(self, normalize = True, alpha = 0.05):
        '''This function takes the feature arrays for control and disease and takes average of each feature.
         Then it plots again list of features (label). '''

        feature_labels = np.squeeze(self.img_features_labels_arr[0],0)
        
        #shape of feat array is pat, channels, features
        feat_arr = np.squeeze(self.img_features_torch.cpu().detach().numpy(), axis=1)
        
        if len(feat_arr.shape) == 4: #shape of feat array is pat, channels, features (image) - if featues is image, stack them all into an arary 
            feat_arr = feat_arr.reshape(feat_arr.shape[0], feat_arr.shape[1], feat_arr.shape[2]*feat_arr.shape[3])

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
            fig, ax = plt.subplots()#constrained_layout=True)
            x = np.arange(len(control_feats[c]))
            width = 0.3
            bars1 = ax.bar(x - width/2, control_feats[c], width, yerr=control_std[c], capsize=5, label='control', color='blue', alpha=0.7)
            bars2 = ax.bar(x + width/2, disease_feats[c], width, yerr=disease_std[c], capsize=5, label='disease', color='red', alpha=0.7)

            for f in range(len(feature_labels)):
                vals_disease=feat_arr[index_disease][:,c,f]
                vals_control=feat_arr[index_control][:,c,f]
                t_stat, p_value = stats.ttest_ind(vals_disease, vals_control)

                man_stat, p_man = mannwhitneyu(vals_disease, vals_control, alternative='two-sided')

                print(p_value)
                if p_value < alpha:
                    ax.text(f, y=np.max(np.concat((vals_control,vals_disease))), s="*", ha='center', va='center', fontsize=15)
                
                if p_man< alpha:
                    ax.text(f, y=np.max(np.concat((vals_control,vals_disease))), s="*", ha='center', va='center', fontsize=15)

            # Add labels and title
            ax.set_ylabel('Feature Values')
            plt.xticks(rotation=45)
            ax.set_xticks(x)
            ax.set_xticklabels(feature_labels.tolist())  # Set custom x-axis labels

            ax.set_ylim([0, 1])

            plt.legend()
            # plt.show()
            plt.title(self.plt_name+'_'+str(c))
            # plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir,self.output_sub_dir,self.comparison_type[0]+'_channel'+str(c)+'_bar_graph'))
            plt.close()

        ### plot each feature seperately
        channels = feat_arr.shape[2]
        for c in range(channels):
            # Create the bar plot
            for f in range(len(feature_labels)):
                fig, ax = plt.subplots()#constrained_layout=True)
                vals_disease=feat_arr[index_disease][:,c,f]
                vals_control=feat_arr[index_control][:,c,f]
                t_stat, p_value = stats.ttest_ind(vals_disease, vals_control)
                man_stat, p_man = mannwhitneyu(vals_disease, vals_control, alternative='two-sided')

                print(p_value)
                if p_value < alpha:
                    ax.text(f, y=np.max(np.concat((vals_control,vals_disease))), s="*", ha='center', va='center', fontsize=15)
                

                width = 0.3
                bars1 = ax.bar(1 - width/2, control_feats[c][f], width, yerr=control_std[c][f], capsize=5, label='control', color='blue', alpha=0.7)
                bars2 = ax.bar(1 + width/2, disease_feats[c][f], width, yerr=disease_std[c][f], capsize=5, label='disease', color='blue', alpha=0.7)


                # Add labels and title
                ax.set_ylabel('Feature Values')
                ax.set_xlabel(str(feature_labels[f]))
                
                if p_value < alpha:
                    text=str(p_value)+'*' 
                else:
                    text=str(p_value)

                plt.text(0.7,0.1,text, ha='center', va='center', transform=ax.transAxes)
                
                plt.legend()
                plt.show()
                plt.title(self.plt_name+'_'+str(c))
                # plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir,self.output_sub_dir,self.comparison_type[0]+'_channel'+str(c)+'_'+str(feature_labels[f])))
                plt.close()

        return
    
    def exp_func(self, x, a, b):
        return a * np.exp(b * x)
    
    def histogram_per_img(self):
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

        ###### plt individual features combined channels ######
        cont_feat_flatten= np.reshape(control_feats_arr, (control_feats_arr.shape[0],control_feats_arr.shape[1], int(np.prod(control_feats_arr.shape[2:]))))
        disease_feat_flatten=np.reshape(disease_feats_arr, (disease_feats_arr.shape[0],disease_feats_arr.shape[1], int(np.prod(disease_feats_arr.shape[2:]))))
        Histograms(self.config).get_all_channel_histogram(cont_feat_flatten,disease_feat_flatten,feature_labels)

        ######### plt individual channels ######
        cont_feat_flatten= np.reshape(control_feats_arr, (control_feats_arr.shape[0],control_feats_arr.shape[1],control_feats_arr.shape[2], int(np.prod(control_feats_arr.shape[3:]))))
        disease_feat_flatten=np.reshape(disease_feats_arr, (disease_feats_arr.shape[0],disease_feats_arr.shape[1],disease_feats_arr.shape[2], int(np.prod(disease_feats_arr.shape[3:]))))
        Histograms(self.config).get_individual_channel_histogram(cont_feat_flatten,disease_feat_flatten,feature_labels)
            
        return
    
    def run_PCA(self, scale_data=True):
        '''feat array input'''   
        try:     
            _feat_arr = np.squeeze(self.img_features_torch.cpu().detach().numpy(), axis=1)
        except:
            _feat_arr = np.squeeze(self.img_features_torch, axis=1)

        comparison_type='PCA'
        Visuals(self.config).plot_feature_analysis(self.img_class_torch, _feat_arr, comparison_type,scale_data)

        return
    
    def run_tSNE(self, scale_data=True,):
        '''feat array input'''        
        _feat_arr = np.squeeze(self.img_features_torch.cpu().detach().numpy(), axis=1)
        #if this shape is 3 then structure is [sample, channels, img_features], if its 4 then the features are per pixel and the output features are images

        comparison_type='tSNE'
        Visuals(self.config).plot_feature_analysis(self.img_class_torch, _feat_arr, comparison_type,scale_data)

        return
    
    def run_UMAP(self, scale_data=True):
        '''feat array input'''        
        _feat_arr = np.squeeze(self.img_features_torch.cpu().detach().numpy(), axis=1)

        comparison_type='UMAP'
        Visuals(self.config).plot_feature_analysis(self.img_class_torch, _feat_arr, comparison_type,scale_data)

        return

    def _report(self):
        #returns the healthy vs the control values
        for compare_func in self.comparison_type:
            eval("self."+compare_func+"()")

        return