import SimpleITK as sitk
import radiomics
import numpy as np
import six
import pyfeats

import matplotlib.pyplot as plt
import math
from visualization import Visuals

class Pyfeats():
    def __init__(self, config):
        self.extractor = ''

        self.plot = config.feature_extraction.plot

        self.config = config
        self.out_sub_dir = config.feature_extraction.method

        if config.feature_extraction.features[0] == 'ALL':
            self.feat_dict = {'fos':'mask)', 
                              'ngtdm_features':'mask, d=1)',
                              'glrlm_features':'mask, Ng=256)'
            }
            #self.feat_dict = {'glrlm_features':'mask, Ng=256)'}
            # , 
            #                   'glrlm_features':'mask, Ng=256)'
            # }
             #'glcm_features':'ignore_zeros=True)'
        else:
            self.feat_dict = config.feature_extraction.features
        
        pass

    def extract_features(self, img_arr):
        mask = np.ones(img_arr.shape)
        feature_arr, label_arr = np.array([]), np.array([])
        for feat, config in self.feat_dict.items():
            features, labels = eval("pyfeats."+feat+"(img_arr,"+config)
            feature_arr = np.append(feature_arr, features)
            label_arr = np.append(label_arr,labels)

        return feature_arr, label_arr

    
    def _get_feature_dict(self, dict_arr):
        # Put in dictionary of image arrays, add a new index for the dictionary of image features
        # Extract the features for each pixel
        feat_dict = {}
        for name, img_arr in dict_arr.items():
            print('calculating features for ', name)
            features_per_img = self.extract_pixel_features(name, img_arr)

            feat_dict[name+'_features_'+self.config.feature_extraction.method] = features_per_img

        return feat_dict
    

    def _get_feature_arr(self, mat_arr, pat_id):
        # Take input imge arr of multiple channels and caclulate features per channel
        feat_arr_allchannels,feat_names =np.array([]), np.array([])
        
        for c in range(mat_arr.shape[0]):
            feature_arr, labels = self.extract_features(mat_arr[c])
            features_per_channel = np.expand_dims(feature_arr, axis=0)

            try:
                feat_arr_allchannels = np.concat((feat_arr_allchannels,features_per_channel), axis=0)
            except:
                feat_arr_allchannels = features_per_channel

        feat_names = np.expand_dims(labels, axis=0)
        feat_arr_allchannels = np.expand_dims(feat_arr_allchannels, axis=0)

        return feat_arr_allchannels, feat_names