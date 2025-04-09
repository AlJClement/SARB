import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog, canny
from skimage.filters import sobel
from skimage import exposure
from skimage import io
import os
from visualization import Visuals

class SkIMG():
    def __init__(self, config):
        self.extractor = ''

        self.config = config
        self.output_dir =self.config.output.loc
        self.out_sub_dir = config.feature_extraction.method
        
        self.plot = config.feature_extraction.plot

         # self.kernel=config.feature_extraction.kernel

        self.feat_list = config.feature_extraction.features

        if 'HOG' in self.feat_list:
            self.orientations = config.feature_extraction.orientations
            self.pixels_per_cell = config.feature_extraction.pixels_per_cell
            self.cell_per_block = config.feature_extraction.cells_per_block
            self.hog_intensity_rescale = config.feature_extraction.hog_intensity_rescale

        if 'CANNY' in self.feat_list:
            self.sigma = config.feature_extraction.sigma

        pass

    def HOG(self, name, img_arr):
        features_dict = {}
        for i in range(img_arr.shape[0]):
            # Extract HOG features and visualize them
            fd, hog_image = hog(img_arr[i], orientations=self.orientations, pixels_per_cell=(self.pixels_per_cell, self.pixels_per_cell), cells_per_block=(self.cell_per_block, self.cell_per_block), visualize=True)

            # Rescale histogram for better displayd
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, self.hog_intensity_rescale))
            features_dict[name+'_channel'+str(i)+'_'+self.out_sub_dir] = hog_image_rescaled

            try:
                all_hog_feat_arr = np.concat((all_hog_feat_arr,np.expand_dims(hog_image, axis = 0)),axis=0)
            except:
                all_hog_feat_arr = np.expand_dims(hog_image, axis = 0)


        if self.plot == True:
            Visuals(self.config, self.out_sub_dir).plot_perimg_channels_with_feature_per_channel(features_dict, img_arr)

        return all_hog_feat_arr
    
    def SOBEL(self, name, img_arr):
        features_dict = {}
        for i in range(img_arr.shape[0]):
            # Extract HOG features and visualize them
            sobel_img = sobel(img_arr[i])

            # Rescale histogram for better displayd
            features_dict[name+'_channel'+str(i)+'_'+self.out_sub_dir+'_sobel'] = sobel_img

            try:
                all_sobel_arr = np.concat((all_sobel_arr,np.expand_dims(sobel_img, axis = 0)),axis=0)
            except:
                all_sobel_arr = np.expand_dims(sobel_img, axis = 0)


        if self.plot == True:
            Visuals(self.config, self.out_sub_dir).plot_perimg_channels_with_feature_per_channel(features_dict, img_arr)

        return all_sobel_arr
    
    def CANNY(self, name, img_arr):
        features_dict = {} 
        for i in range(img_arr.shape[0]):
            # Extract HOG features and visualize them
            canny_im =  canny(img_arr[i], sigma=self.sigma)

            # Rescale histogram for better displayd
            features_dict[name+'_channel'+str(i)+'_'+self.out_sub_dir+'_canny'] = canny_im

            try:
                all_sobel_arr = np.concat((all_sobel_arr,np.expand_dims(canny_im, axis = 0)),axis=0)
            except:
                all_sobel_arr = np.expand_dims(canny_im, axis = 0)


        if self.plot == True:
            Visuals(self.config, self.out_sub_dir).plot_perimg_channels_with_feature_per_channel(features_dict, img_arr)

        return all_sobel_arr


    def extract_pixel_features_dict(self, name, img_arr):
        features_dict = {}
        for feat in self.feat_list:
            feat_extractor = eval('self.'+feat)
            feat_dict = feat_extractor(name, img_arr)
            
            features_dict[name+'_'+self.out_sub_dir+'_'+feat] = feat_dict

        return features_dict

    def extract_pixel_features(self, img_arr, pat_id):
        feat_name_list = []
        for feat in self.feat_list:
            feat_extractor = eval('self.'+feat)
            feat_arr = feat_extractor(pat_id, img_arr)
            
            try:
                all_features = np.concat((all_features,np.expand_dims(feat_arr, axis = 0)),axis=0)
            except:
                all_features = np.expand_dims(feat_arr, axis = 0)

            feat_name_list.append(feat)

        return all_features, feat_name_list

    def _get_feature_dict(self, dict_arr):
        # Put in dictionary of image arrays, add a new index for the dictionary of image features
        # Extract the features for each pixel
        feat_dict = {}
        for name, img_arr in dict_arr.items():
            print('calculating features for ', name)
            features_per_img = self.extract_pixel_features_dict(name, img_arr)

            feat_dict[name+'_features_'+self.config.feature_extraction.method] = features_per_img

        return feat_dict
    

    def _get_feature_arr(self, img_arr, pat_id):

        # Take input imge arr of multiple channels and caclulate features per channel
        features_per_channel,feat_names = self.extract_pixel_features(img_arr, pat_id)

        feat_names_arr=np.expand_dims(np.array(feat_names), axis =0)
        # feat_names = np.expand_dims(labels, axis=0)

        ### features_per_channel: SHAPE SHOULD BE [IMG, C, H, W]
        ### faeture_names_arr: SHAPE SHOULD BE [IMG, NAMES]

        return features_per_channel, feat_names_arr
    