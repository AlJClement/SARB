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
            bars1 = ax.bar(x - width/2, control_feats[c], width, yerr=control_std[c], capsize=5, label='control', color='green', alpha=0.7)
            bars2 = ax.bar(x + width/2, disease_feats[c], width, yerr=disease_std[c], capsize=5, label='disease', color='blue', alpha=0.7)

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
                bars1 = ax.bar(1 - width/2, control_feats[c][f], width, yerr=control_std[c][f], capsize=5, label='control', color='green', alpha=0.7)
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
    
    def histogram_per_img(self, bins = 50, alpha = 0.05):
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

        ### plt individual features combined channels
        fig, axes = plt.subplots(1,1,figsize=(5*cont_feat_flatten.shape[1],5))

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

            vals_control=cont_feat_flatten[:,f,:]
            vals_disease=disease_feat_flatten[:,f,:]
            t_stat, p_value = stats.ttest_ind(vals_disease.flatten(), vals_control.flatten())
            man_stat, p_man = mannwhitneyu(vals_disease.flatten(), vals_control.flatten(), alternative='two-sided')

            STR="p="+str(p_value)

            if self.exponential_comparison == True:
                x1 = np.linspace(0, 100,len(vals_control.flatten()))
                x2 = np.linspace(0, 100,len(vals_disease.flatten()))

                # Fit both datasets to exponential curve function
                popt1, pcov1 = curve_fit(self.exp_func, x1, vals_control.flatten(), p0=(1, 1))
                popt2, pcov2 = curve_fit(self.exp_func, x2, vals_disease.flatten(), p0=(1, 1))

                # RSS for individual fits
                rss1 = np.sum((vals_control.flatten() - self.exp_func(x1, *popt1))**2)
                rss2 = np.sum((vals_disease.flatten() - self.exp_func(x2, *popt2))**2)

                # Combine data
                t_all = np.concatenate([x1, x2])
                y_all = np.concatenate([vals_control.flatten(), vals_disease.flatten()])

                # Fit combined model
                popt_all, _ = curve_fit(self.exp_func, t_all, y_all, p0=[1, 1])
                rss_combined = np.sum((y_all - self.exp_func(t_all, *popt_all))**2)

                # F-test
                p = 2  # number of parameters per model
                n1, n2 = len(vals_control.flatten()), len(vals_disease.flatten())
                numerator = (rss_combined - (rss1 + rss2)) / p
                denominator = (rss1 + rss2) / (n1 + n2 - 2 * p)
                F_value = numerator / denominator

                # p-value
                df1 = p
                df2 = n1 + n2 - 2 * p
                p_value = 1 - fff.cdf(F_value, df1, df2)

                STR= STR+" F = {F_value:.4f}, p = {p_value:.4f}"

            if p_value<0.05:
                STR="t-test p="+str((p_value))+"**"+""
                STRmean="m_cont="+str(round(vals_control.mean(),3))
                STRmean_d="m_disease="+str(round(vals_disease.mean(),3))
            else:
                STR="t-test p="+str((p_value))
                STRmean="m_cont="+str(round(vals_control.mean(),3))
                STRmean_d="m_disease="+str(round(vals_disease.mean(),3))       
            
            man_stat, p_man = mannwhitneyu(vals_disease.flatten(), vals_control.flatten(), alternative='two-sided')
            if p_man<0.05:
                STR_MAN = "man p ="+str((p_man))+"**"+""                
            else:
                STR_MAN = "man p ="+str((p_man))                

            man_stat, p_man = mannwhitneyu(vals_disease.flatten(), vals_control.flatten(), alternative='two-sided')

            plt.text(0.05, 0.99, STR, horizontalalignment='left', verticalalignment='top',transform=axes.transAxes)
            plt.text(0.05, 0.95, STRmean, horizontalalignment='left', verticalalignment='top',transform=axes.transAxes)
            plt.text(0.05, 0.91, STRmean_d, horizontalalignment='left', verticalalignment='top',transform=axes.transAxes)
            plt.text(0.05, 0.86, STR_MAN, horizontalalignment='left', verticalalignment='top',transform=axes.transAxes)

            plt.title(feature_name, fontsize=10, pad=0)
            # plt.show()
            plt.legend()
            plt.savefig(os.path.join(self.output_dir,self.output_sub_dir,feature_name+'_histogram'))


        plt.close()
        
        ### plt individual channels
        #cont_feat_flatten= np.reshape(control_feats_arr, (control_feats_arr.shape[0],control_feats_arr.shape[1],control_feats_arr.shape[2], int(np.prod(control_feats_arr.shape[3:]))))
        #disease_feat_flatten=np.reshape(disease_feats_arr, (disease_feats_arr.shape[0],disease_feats_arr.shape[1],disease_feats_arr.shape[2], int(np.prod(disease_feats_arr.shape[3:]))))
        if len(feature_labels) == 1:
            cont_feat_flatten = np.expand_dims(cont_feat_flatten, axis=1)
            disease_feat_flatten = np.expand_dims(disease_feat_flatten, axis=1)
        else:
            pass
            
        for f in range(len(feature_labels)):
            fig, axes = plt.subplots(1,cont_feat_flatten.shape[2],figsize=(5*cont_feat_flatten.shape[2]+5,5))

            #loop in number of channels
            for c in range((cont_feat_flatten.shape[2])):
                print(c)
                feature_name = feature_labels[f]
                print(feature_name)
                kwargs = dict(histtype='stepfilled', alpha=0.3, bins=bins)
                
                for pat in range(cont_feat_flatten.shape[0]):
                    if pat == 0:
                        axes[c].hist(self.normalize_array(cont_feat_flatten[pat,f,c,:]), color = 'g',label=self.control_str, **kwargs)
                    else:
                        axes[c].hist(self.normalize_array(cont_feat_flatten[pat,f,c,:]), color = 'g', **kwargs)
    
                for pat in range(disease_feat_flatten.shape[0]):
                    if pat == 0:
                        axes[c].hist(self.normalize_array(disease_feat_flatten[pat,f,c,:]), color = 'b', label=self.disease_str, **kwargs)
                    else:
                        axes[c].hist(self.normalize_array(disease_feat_flatten[pat,f,c,:]), color = 'b', **kwargs)

                vals_control=cont_feat_flatten[:,f,c,:]
                vals_disease=disease_feat_flatten[:,f,c,:]
                t_stat, p_value = stats.ttest_ind(vals_disease.flatten(), vals_control.flatten())

                axes[c].set_title("channel "+str(c), fontsize=10, pad=0)
                axes[c].legend()
                if p_value<0.05:
                    STR="t-test p="+str((p_value))+"**"+""
                    STRmean="m_cont="+str(round(vals_control.mean(),3))
                    STRmean_d="m_disease="+str(round(vals_disease.mean(),3))                
                else:
                    STR="t-test p="+str((p_value))
                    STRmean="m_cont="+str(round(vals_control.mean(),3))
                    STRmean_d="m_disease="+str(round(vals_disease.mean(),3))

                if p_man<0.05:
                    STR_MAN = "man p ="+str((p_man))+"**"+""                
                else:
                    STR_MAN = "man p ="+str((p_man))                



                axes[c].text(0.05, 0.99, STR, horizontalalignment='left', verticalalignment='top',transform=axes[c].transAxes)
                axes[c].text(0.05, 0.95, STRmean, horizontalalignment='left', verticalalignment='top',transform=axes[c].transAxes)
                axes[c].text(0.05, 0.91, STRmean_d, horizontalalignment='left', verticalalignment='top',transform=axes[c].transAxes)
                axes[c].text(0.05, 0.87, STR_MAN, horizontalalignment='left', verticalalignment='top',transform=axes[c].transAxes)

                if self.exponential_comparison == True:
                    x1 = np.linspace(0, 100,len(vals_control.flatten()))
                    x2 = np.linspace(0, 100,len(vals_disease.flatten()))

                    # Fit both datasets to exponential curve function
                    popt1, pcov1 = curve_fit(self.exp_func, x1, vals_control.flatten(), p0=(1, 0.1))
                    popt2, pcov2 = curve_fit(self.exp_func, x2, vals_disease.flatten(), p0=(1, 0.1))

                    # RSS for individual fits
                    rss1 = np.sum((vals_control.flatten() - self.exp_func(x, *popt1))**2)
                    rss2 = np.sum((vals_disease.flatten() - self.exp_func(x, *popt2))**2)

                    # Combine data
                    t_all = np.concatenate([x, x])
                    y_all = np.concatenate([vals_control.flatten(), vals_disease.flatten()])

                    # Fit combined model
                    popt_all, _ = curve_fit(self.exp_func, t_all, y_all, p0=[1, 1])
                    rss_combined = np.sum((y_all - self.exp_func(t_all, *popt_all))**2)

                    # F-test
                    p = 2  # number of parameters per model
                    n1, n2 = len(vals_control.flatten()), len(vals_disease.flatten())
                    numerator = (rss_combined - (rss1 + rss2)) / p
                    denominator = (rss1 + rss2) / (n1 + n2 - 2 * p)
                    F_value = numerator / denominator

                    # p-value
                    df1 = p
                    df2 = n1 + n2 - 2 * p
                    p_value = 1 - fff.cdf(F_value, df1, df2)

                    STRmean_z=f"F = {F_value:.2f}, p = {p_value:.4}"

                    axes[c].text(0.05, 0.85, STRmean_z, horizontalalignment='left', verticalalignment='top',transform=axes[c].transAxes)


            plt.savefig(os.path.join(self.output_dir,self.output_sub_dir,feature_name+'_histogram_PERCHANNEL'))
            plt.close()
            
        return
    
    def run_PCA(self, scale_data=True):
        '''feat array input'''   
        try:     
            _feat_arr = np.squeeze(self.img_features_torch.cpu().detach().numpy(), axis=1)
        except:
            _feat_arr = np.squeeze(self.img_features_torch, axis=1)

        #if this shape is 3 then structure is [sample, channels, img_features], if its 4 then the features are per pixel and the output features are images

        if len(_feat_arr.shape)==4:
            feat_grid = False
        else:
            feat_grid = True

        for j in range((2)):
            #looping twice to save pca and normalised pca feature outputs
            # channels should be in shape 0 of features 
            fig, axes = plt.subplots(1,_feat_arr.shape[1],figsize=(5*_feat_arr.shape[1],5))

            for c in range(_feat_arr.shape[1]):
                if feat_grid == False:
                    feat_arr = _feat_arr[:,:,c,:]
                else:                
                    feat_arr = _feat_arr[:,:,c,:,:]
                classes = np.squeeze(self.img_class_torch.cpu().detach().numpy(), axis=0)

                flatten_feat_arr =  np.reshape(feat_arr, (feat_arr.shape[0], np.prod(feat_arr.shape[1:])))

                #Scale data
                if scale_data == True:
                    scaler = StandardScaler()
                    scaler.fit(flatten_feat_arr)
                    flatten_feat_arr=scaler.transform(flatten_feat_arr)    

                # feats_fit_pca = PCA(n_components=2).fit_transform(flatten_feat_arr)

                umap_model = umap.UMAP(n_components=2, n_neighbors=10, min_dist=0.7, random_state=42)
                feats_fit_pca = umap_model.fit_transform(flatten_feat_arr)

                axes[c].set_xlabel("Component 1")
                axes[c].set_ylabel("Component 2")
                axes[c].grid()
                axes[c].set_title('Channel '+str(c))

                class1 = feats_fit_pca[classes == 0] 
                class2 = feats_fit_pca[classes == 1] 
                scalex=1.0
                scaley=1.0

                if j == 1:
                    #standardise
                    scalex = 1.0/(feats_fit_pca[:,0].max() - feats_fit_pca[:,0].min())
                    scaley = 1.0/(feats_fit_pca[:,1].max() - feats_fit_pca[:,1].min())
                    axes[c].set_xlim(-1,1)
                    axes[c].set_ylim(-1,1)
                
                # Scatter plot
                axes[c].scatter(class1[:, 0]*scalex, class1[:, 1]*scaley, color='green', label=self.control_str)
                axes[c].scatter(class2[:, 0]*scalex, class2[:, 1]*scaley, color='blue', label=self.disease_str)

                axes[c].legend([self.control_str, self.disease_str])
                
            # plt.show()
            plt.tight_layout()
            if j == 1:
                plt.savefig(os.path.join(self.output_dir,self.output_sub_dir,self.plt_name+'_PCA_normalisedComponents'))
            else:
                plt.savefig(os.path.join(self.output_dir,self.output_sub_dir,self.plt_name+'_PCA'))
            plt.close()
        return
    
    def run_tSNE(self, scale_data=True):
        '''feat array input'''        
        _feat_arr = np.squeeze(self.img_features_torch.cpu().detach().numpy(), axis=1)

        #if this shape is 3 then structure is [sample, channels, img_features], if its 4 then the features are per pixel and the output features are images

        if len(_feat_arr.shape)==4:
            feat_grid = False

        for j in range((2)):
            #looping twice to save pca and normalised pca feature outputs 
            fig, axes = plt.subplots(1,_feat_arr.shape[2],figsize=(5*_feat_arr.shape[2],5))

            for c in range(_feat_arr.shape[2]):
                if feat_grid == False:
                    feat_arr = _feat_arr[:,:,c,:]
                else:                
                    feat_arr = _feat_arr[:,:,c,:,:]
                classes = np.squeeze(self.img_class_torch.cpu().detach().numpy(), axis=0)

                flatten_feat_arr =  np.reshape(feat_arr, (feat_arr.shape[0], np.prod(feat_arr.shape[1:])))

                #Scale data
                if scale_data == True:
                    scaler = StandardScaler()
                    scaler.fit(flatten_feat_arr)
                    flatten_feat_arr=scaler.transform(flatten_feat_arr)    

                tsne = TSNE(n_components=2, perplexity=5, random_state=42)
                feats_fit_tsne = tsne.fit_transform(flatten_feat_arr)

                axes[c].set_xlabel("Component 1")
                axes[c].set_ylabel("Component 2")
                axes[c].grid()
                axes[c].set_title('Channel '+str(c))

                class1 = feats_fit_tsne[classes == 0] 
                class2 = feats_fit_tsne[classes == 1] 
                scalex=1.0
                scaley=1.0

                if j == 1:
                    #normalise
                    axes[c].set_xlim(-1,1)
                    axes[c].set_ylim(-1,1)
                    scalex = 1.0/(feats_fit_tsne[:,0].max() - feats_fit_tsne[:,0].min())
                    scaley = 1.0/(feats_fit_tsne[:,1].max() - feats_fit_tsne[:,1].min())
                
                # Scatter plot
                axes[c].scatter(class1[:, 0]*scalex, class1[:, 1]*scaley, color='blue', label=self.control_str)
                axes[c].scatter(class2[:, 0]*scalex, class2[:, 1]*scaley, color='yellow', label=self.disease_str)

                axes[c].legend([self.control_str, self.disease_str])
                
            # plt.show()
            plt.tight_layout()
            if j == 1:
                plt.savefig(os.path.join(self.output_dir,self.output_sub_dir,self.plt_name+'_tsne_normalisedComponents'))
            else:
                plt.savefig(os.path.join(self.output_dir,self.output_sub_dir,self.plt_name+'_tsne'))
            plt.close()
        return
    
    def run_UMAP(self, scale_data=True):
        '''feat array input'''        
        _feat_arr = np.squeeze(self.img_features_torch.cpu().detach().numpy(), axis=1)

        #if this shape is 3 then structure is [sample, channels, img_features], if its 4 then the features are per pixel and the output features are images

        if len(_feat_arr.shape)==4:
            feat_grid = False

        for j in range((1)):#DONT LOOP BECAUSE UMAP VALUES CAN BE NEGATIVE SO WONT WORK

            #looping twice to save pca and normalised pca feature outputs 
            fig, axes = plt.subplots(1,_feat_arr.shape[2],figsize=(5*_feat_arr.shape[2],5))

            for c in range(_feat_arr.shape[2]):
                if feat_grid == False:
                    feat_arr = _feat_arr[:,:,c,:]
                else:                
                    feat_arr = _feat_arr[:,:,c,:,:]
                classes = np.squeeze(self.img_class_torch.cpu().detach().numpy(), axis=0)

                flatten_feat_arr =  np.reshape(feat_arr, (feat_arr.shape[0], np.prod(feat_arr.shape[1:])))

                #Scale data
                if scale_data == True:
                    scaler = StandardScaler()
                    scaler.fit(flatten_feat_arr)
                    flatten_feat_arr=scaler.transform(flatten_feat_arr)    

                umap_model = umap.UMAP(n_components=2, n_neighbors=10, min_dist=0.7, random_state=42)
                feats_fit_UMAP = umap_model.fit_transform(flatten_feat_arr)

                axes[c].set_xlabel("Component 1")
                axes[c].set_ylabel("Component 2")
                axes[c].grid()
                axes[c].set_title('Channel '+str(c))

                class1 = feats_fit_UMAP[classes == 0] 
                class2 = feats_fit_UMAP[classes == 1] 
                scalex=1.0
                scaley=1.0
                
                # Scatter plot
                axes[c].scatter(class1[:, 0]*scalex, class1[:, 1]*scaley, color='blue', label=self.control_str)
                axes[c].scatter(class2[:, 0]*scalex, class2[:, 1]*scaley, color='yellow', label=self.disease_str)

                axes[c].legend([self.control_str, self.disease_str])
                
            # plt.show()

            plt.savefig(os.path.join(self.output_dir,self.output_sub_dir,self.plt_name+'_UMAP'))
            plt.close()
        return

    def _report(self):
        #returns the healthy vs the control values
        for compare_func in self.comparison_type:
            eval("self."+compare_func+"()")

        return