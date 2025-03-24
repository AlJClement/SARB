import matplotlib.pyplot as plt
import os
import numpy as np
import math
import SimpleITK as sitk
class Visuals():
    def __init__(self, config, sub_dir=''):
        self.output_path = config.output.loc
        os.makedirs(self.output_path, exist_ok=True)
        self.img_scale = config.data.img_scale
        self.sub_dir = sub_dir
        os.makedirs(os.path.join(self.output_path,self.sub_dir), exist_ok=True)
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
        plt.savefig(os.path.join(self.output_path,self.sub_dir,name), dpi=1000)
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
                _max = np.max(arr[i])

                if _max > 3*np.mean(arr[i]):               
                    _max=np.mean(arr[i])

                ax.set_clim(_min, _max)
                # Add a colorbar for each plot
                fig.colorbar(ax, ax=axes[i])
        
            plt.savefig(os.path.join(self.output_path,folder_name,name),dpi=1000)

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

            # if _max > 3*np.mean(img_arr[j]*self.img_scale[j],):               
            #     _max=np.mean(img_arr[j]*self.img_scale[j],)

            axes = ax[j][0].imshow(img_arr[j]*self.img_scale[j], cmap='grey', vmin=_min,vmax=_max) 
            ax[j][0].set_title('Image Channel '+str(j))
            ax[j][0].axis('off')
                            # Add a colorbar for each plot
            #axes.set_clim(_min, _max)
            fig.colorbar(axes, ax=ax[j][0])

            axes_2 =ax[j][1].imshow(f_arr, vmin =f_arr.min(),vmax= f_arr.max()) 
            ax[j][1].set_title(f.split('_')[-1]+' Features')
            ax[j][1].axis('off')
            fig.colorbar(axes_2, ax=ax[j][1])

            j = j+1 

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path,self.sub_dir, list(features_dict.keys())[0].replace('_channel0','')), dpi=1000)
        plt.close()
        return