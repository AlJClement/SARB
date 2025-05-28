import matplotlib.pyplot as plt
import os
import numpy as np
import math
import SimpleITK as sitk

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import math
import umap.umap_ as umap
from sklearn.preprocessing import scale, normalize

class Visuals():
    def __init__(self, config, log, sub_dir=''):
        self.output_path = config.output.loc
        os.makedirs(self.output_path, exist_ok=True)
        self.img_scale = config.data.img_scale
        self.sub_dir = sub_dir
        os.makedirs(os.path.join(self.output_path,self.sub_dir), exist_ok=True)
        self.dpi = config.data.dpi
        self.set_max = config.data.set_max
        self.set_max_feat = config.data.set_max_feat

        self.control_str = config.data.control
        self.disease_str = config.data.disease

        self.img_ext = '.svg'

        try:
            self.log = log
        except:
            ValueError('Check if log is not working')
        self.output_dir = config.output.loc
        
        self.output_sub_dir = config.feature_extraction.method
        os.makedirs(os.path.join(self.output_dir,self.output_sub_dir), exist_ok=True)

        self.dimension_reduction_components = config.feature_extraction.dimension_reduction_components

        return
    
    def plot_features_fromdict(self, name, orig_arr, feature_map, plot_list):
        '''plots all features per channel'''

        grid_size = int(np.ceil(math.sqrt(len(plot_list))))

        fig, axes = plt.subplots(grid_size,grid_size,sharex=True,sharey=True,figsize=(grid_size,grid_size))

        #flatten axes so you can access axes[i]
        axes = axes.ravel()
        i = 0
        for feat in axes:
            try:
                if i == 0:
                    #plot original array
                    axes[i].imshow(orig_arr,cmap='grey')
                    axes[i].set(title='image')
                else:
                    arr=sitk.GetArrayFromImage(feature_map[plot_list[i]])[0]
                    axes[i].imshow(arr)
                    axes[i].set_title(plot_list[i], fontsize=3, pad=0)

            except:
                pass
            i = i+1
        # plt.show()
        plt.tight_layout()  # Adjust layout to avoid overlap
        plt.savefig(os.path.join(self.output_path,self.sub_dir,name+self.img_ext), dpi=self.dpi)
        plt.savefig(os.path.join(self.output_path,self.sub_dir,name+'.png'), dpi=self.dpi)

        plt.close()

        return
    
    def save_channel0(self, arr,name, channel, folder_name = 'cache'):
        '''takes in one mat array dictionary and plots each svg just the channel given'''
        os.makedirs(os.path.join(self.output_path, folder_name), exist_ok=True)

        # Plot the original image
        plt.imshow(arr[:,:,channel], cmap='grey')
        plt.axis('off')

        plt.savefig(os.path.join(self.output_path,folder_name,name+'channel'+str(channel)+self.img_ext),dpi=self.dpi)
        plt.savefig(os.path.join(self.output_path,folder_name,name+'channel'+str(channel)+'.jpg'),dpi=self.dpi)

        plt.close()
        return
    
    def save_channel_rgb(self, arr,name, folder_name = 'cache'):
        '''takes in one mat array dictionary and plots each svg just the channel given'''
        os.makedirs(os.path.join(self.output_path, folder_name), exist_ok=True)
        # Plot the original image
        plt.close()
        s=500
        r=arr[:,:,1]*s
        g=arr[:,:,2]*s
        b=arr[:,:,3]*s
        __color = 'magma_r'

        # r= (r - r.min()) / (r.max() - r.min())
        # g= (g - g.min()) / (g.max() - g.min())
        # b= (b - b.min()) / (b.max() - b.min())
        # plt.imshow(r,cmap=__color)
        # plt.savefig(os.path.join(self.output_path,folder_name,name+'_channel_r'+'.png'),dpi=self.dpi)
        # plt.imshow(g,cmap=__color)
        # plt.savefig(os.path.join(self.output_path,folder_name,name+'_channel_g'+'.png'),dpi=self.dpi)
        # plt.imshow(b,cmap=__color)
        # plt.savefig(os.path.join(self.output_path,folder_name,name+'_channel_b'+'.png'),dpi=self.dpi)

        # plt.close()

        xx= r+b+g
        plt.imshow(xx, cmap=__color, vmin=0, vmax=1)

        # plt.colorbar() 
        plt.axis('off')

        plt.savefig(os.path.join(self.output_path,folder_name,name+'_channel_rgb'+'.png'),dpi=self.dpi)

        plt.close()
        return
        
    def plot_cache(self, mat_dict, folder_name = 'cache'):
        '''takes in dictionary of mat arrays, indexed by the scan name'''
        os.makedirs(os.path.join(self.output_path, folder_name), exist_ok=True)

        for name, arr in mat_dict.items():
            fig, axes = plt.subplots(1, arr.shape[0], figsize=(5*arr.shape[0], 5))

            #plot all channels in the array
            for i in range(arr.shape[0]):                               
                # Create a figure with subplots for each channel
                # Plot the original image
                arr[i] = self.img_scale[i]*arr[i]
                ax=axes[i].imshow(arr[i], cmap='grey')
                axes[i].set_title('channel '+str(i))
                axes[i].axis('off')

                # Set color limits based on the data range for each subplot
                _min =np.min(arr[i])

                if self.set_max == None:
                    _max = np.max(arr[i])

                    if _max > 3*np.mean(arr[i]):               
                        _max=np.mean(arr[i])
                else:
                    _max = np.float64(self.set_max[i])
                ax.set_clim(_min, _max)
                # Add a colorbar for each plot
                fig.colorbar(ax, ax=axes[i])
        
            plt.savefig(os.path.join(self.output_path,folder_name,name+self.img_ext),dpi=self.dpi)
            plt.savefig(os.path.join(self.output_path,folder_name,name+'.png'),dpi=self.dpi)


            plt.close()

        return
    
    def plot_perimg_channels_with_feature_per_channel(self, features_dict, img_arr):

        #dictionary of features per image
        #plots each image channel beside the feature

        num_plots = len(list(features_dict.keys()))
        fig, ax= plt.subplots(num_plots, 2, figsize=(num_plots, num_plots*2))
        
        j = 0

        for (f, f_arr) in features_dict.items():
            # Plot the original image and HOG features
            # Set color limits based on the data range for each subplot
            _min =np.min(img_arr[j]*self.img_scale[j],)
            _max = np.max(img_arr[j]*self.img_scale[j],)

            if self.set_max == None:
                _max = np.max(img_arr[j])

                if _max > 3*np.mean(img_arr[j]):               
                    _max=np.mean(img_arr[j])
            else:
                _max = np.float64(self.set_max[j])

            # if _max > 3*np.mean(img_arr[j]*self.img_scale[j],):               
            #     _max=np.mean(img_arr[j]*self.img_scale[j],)

            axes = ax[j][0].imshow(img_arr[j]*self.img_scale[j], cmap='grey', vmin=_min,vmax=_max) 
            ax[j][0].set_title('Image Channel '+str(j))
            ax[j][0].axis('off')
                            # Add a colorbar for each plot
            #axes.set_clim(_min, _max)
            fig.colorbar(axes, ax=ax[j][0])
            _feat_max = self.set_max_feat[j]

            axes_2 =ax[j][1].imshow(f_arr, vmin =f_arr.min(),vmax= _feat_max) 
            ax[j][1].set_title(f.split('_')[-1]+' Features')
            ax[j][1].axis('off')
            fig.colorbar(axes_2, ax=ax[j][1])

            j = j+1 

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path,self.sub_dir, list(features_dict.keys())[0].replace('_channel0','')+self.img_ext), dpi=self.dpi)
        plt.savefig(os.path.join(self.output_path,self.sub_dir, list(features_dict.keys())[0].replace('_channel0','')+'.jpg'), dpi=self.dpi)

        plt.close()
        return
    
    def plot_feature_analysis(self, img_class_torch, _feat_arr, comparison_type, scale_data = True):
        '''this function is used to plot PCA, UMAP or tSNE to comapre features.
        _feat_arr: feature array with channel, batches, features (type), features OR height_features and width_width [C, b, f, h_features, w_features] OR  [C, b, f, features]
        This is because features can be calculated per pixel (retaining shape h x w - ex. HOG) or just have the final feature values (ex. SimCLR).
        img_class_torch: an array of image class in shape [b, total images].
        '''
        num_channels= _feat_arr.shape[2]
        if num_channels < 6:
           num_channels=num_channels+4

        if len(_feat_arr.shape)==5:
            feat_grid = True
        else:        
            feat_grid = False

        for j in range((2)):
            #looping twice to save pca and normalised pca feature outputs
            # channels should be in shape 0 of features 
            fig, axes = plt.subplots(1, num_channels,figsize=(num_channels*4,1*4))

            for c in range(num_channels):
                if c == 5:
                    ## add 123 channel together.
                    if feat_grid == False:
                        feat_arr = _feat_arr[:,:,1:4,:]
                    else:                
                        feat_arr = _feat_arr[:,:,1:4,:,:]
                    flatten_feat_arr =  np.reshape(feat_arr, (feat_arr.shape[0], np.prod(feat_arr.shape[1:])))
                if c == 6:
                    ## add 13 channel together.
                    if feat_grid == False:
                        feat_arr = _feat_arr[:,:,1:4,:]
                    else:                
                        feat_arr = _feat_arr[:,:,1:4,:,:]
                    flatten_feat_arr =  np.reshape(feat_arr, (feat_arr.shape[0], np.prod(feat_arr.shape[1:])))
                if c == 7:
                    ## add 12 channel together.
                    if feat_grid == False:
                        feat_arr = _feat_arr[:,:,1:3,:]
                    else:                
                        feat_arr = _feat_arr[:,:,1:3,:,:]
                    flatten_feat_arr =  np.reshape(feat_arr, (feat_arr.shape[0], np.prod(feat_arr.shape[1:])))
      
                if c == 8:
                    ## add 23 channel together.
                    if feat_grid == False:
                        feat_arr = _feat_arr[:,:,2:4,:]
                    else:                
                        feat_arr = _feat_arr[:,:,2:4,:,:]
                    flatten_feat_arr =  np.reshape(feat_arr, (feat_arr.shape[0], np.prod(feat_arr.shape[1:])))
                if c < 5:    
                    if feat_grid == False:
                        feat_arr = _feat_arr[:,:,c,:]
                    else:                
                        feat_arr = _feat_arr[:,:,c,:,:]
                    flatten_feat_arr =  np.reshape(feat_arr, (feat_arr.shape[0], np.prod(feat_arr.shape[1:])))

                classes = np.squeeze(img_class_torch.cpu().detach().numpy(), axis=0)


                #Scale data
                if scale_data == True:
                    scaler = StandardScaler()
                    scaler.fit(flatten_feat_arr)
                    flatten_feat_arr=scaler.transform(flatten_feat_arr)    
                    #flatten_feat_arr = normalize(flatten_feat_arr,axis=0)

                if comparison_type == 'PCA':
                    if self.dimension_reduction_components=='ALL':
                        #do note set components default is all features are collected
                        pca = PCA()
                        pca.fit(flatten_feat_arr)
                    else:
                        # if you only did two then you dont need to select two for plotting
                        pca = PCA(n_components=self.dimension_reduction_components)
                        pca.fit(flatten_feat_arr)

                    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))  # 1 row, 2 columns
                    # First plot
                    ax1.plot(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_, color='blue')
                    ax1.set_title('explained variance ratio')
                    ax1.set_xlabel('componenet')
                    ax1.set_ylabel('ratio value')

                    # Second plot
                    ax2.plot(range(len(pca.singular_values_)), pca.singular_values_, color='green')
                    ax2.set_title('Singular Values')
                    ax2.set_xlabel('componenet')
                    ax2.set_ylabel('value')

                    ### printing to save
                    self.log.info('PCA Explained Variance Ratios: '+ str(pca.explained_variance_ratio_))
                    self.log.info('PCA Singular Values: '+ str(pca.singular_values_))

                    plt.savefig(os.path.join(self.output_dir,self.output_sub_dir,comparison_type+'_componentPLOTS_channel'+str(c)))

                    #this now takes the values and transforms on the two principal axis
                    feats_fit = pca.transform(flatten_feat_arr) 
                    #transform and get just the first two componenets
                    feats_fit=feats_fit[:, :2] 
                    plt.close()

                elif comparison_type == 'tSNE':
                    if self.dimension_reduction_components == 'ALL':
                        tsne = TSNE(perplexity=4, random_state=42)
                    else:
                        tsne = TSNE(n_components=self.dimension_reduction_components, perplexity=4, random_state=42)
                    #will always be two because its non parametric
                    feats_fit = tsne.fit_transform(flatten_feat_arr)
                    
                elif comparison_type == 'UMAP':
                    if self.dimension_reduction_components == 'ALL':
                        umap_model = umap.UMAP(n_neighbors=4, min_dist=0.7, random_state=42)
                    else:
                        umap_model = umap.UMAP(n_components=self.dimension_reduction_components, n_neighbors=4, min_dist=0.7, random_state=42)

                    feats_fit = umap_model.fit_transform(flatten_feat_arr)

                ##plot pca landmarks
                plt.tight_layout()

                axes[c].set_xlabel("Component 1")
                axes[c].set_ylabel("Component 2")
                axes[c].grid()

                if c == 5:
                    axes[c].set_title('Channel 1+2+3')
                else:
                    axes[c].set_title('Channel '+str(c))

                class1 = feats_fit[classes == 0] 
                class2 = feats_fit[classes == 1] 
                scalex=1.0
                scaley=1.0

                if j == 1:
                    #standardise
                    scalex = 1.0/(feats_fit[:,0].max() - feats_fit[:,0].min())
                    scaley = 1.0/(feats_fit[:,1].max() - feats_fit[:,1].min())
                    axes[c].set_xlim(-1,1)
                    axes[c].set_ylim(-1,1)
                
                axes[c].set_aspect('equal')
                
                # Scatter plot
                axes[c].scatter(class1[:, 0]*scalex, class1[:, 1]*scaley, color='blue', label=self.control_str)
                axes[c].scatter(class2[:, 0]*scalex, class2[:, 1]*scaley, color='red', label=self.disease_str)

                axes[c].legend([self.control_str, self.disease_str])
            plt.tight_layout()

            if j == 1:
                plt.savefig(os.path.join(self.output_dir,self.output_sub_dir,comparison_type+'_normalisedComponents'))
            else:
                plt.savefig(os.path.join(self.output_dir,self.output_sub_dir,comparison_type))
            plt.close()

        return