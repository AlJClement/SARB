import SimpleITK as sitk
import radiomics
import numpy as np
from radiomics import featureextractor
import matplotlib.pyplot as plt
import math
from visualization import Visuals

class Pyradiomics():
    def __init__(self, config):
        self.extractor = ''
        #set small bounding box for testing of radiomics, to reduce computational complexity
        self.x_bound, self.y_bound = config.feature_extraction.x_bound, config.feature_extraction.y_bound

        #drop featuers that are NOT 2D.
        self.features_to_drop = ['_Versions', '_Configuration', '_Mask-', '_Image-', '_shape_']

        self.plot = config.feature_extraction.plot

        self.config = config
        self.out_sub_dir = config.feature_extraction.method

        self.voxel_based = config.feature_extraction.voxel_based
        pass

    def extract_pixel_features(self, img_arr, pat_id):
        pyrad = featureextractor.RadiomicsFeatureExtractor()
        pyrad.enableAllFeatures()
        pyrad.enableFeatureClassByName({'firstorder': True, 'glcm': True})
        feat_map_arr_all_channels = None

        for i in range(img_arr.shape[0]):
            #feature_map = np.zeros_like(img_arr[i], dtype=object)  # To store features for each pixel
            img_arr_wchannel = np.expand_dims(img_arr[i,:self.x_bound,:self.y_bound],axis=0)
            image=sitk.GetImageFromArray(img_arr_wchannel)
            mask = sitk.Image(image.GetSize(), sitk.sitkUInt32)
            mask.CopyInformation(image)
            mask = sitk.Cast(mask, sitk.sitkUInt32)
            mask += 1  # Set all pixel values to 1
            
            feature_map = pyrad.execute(image, mask, voxelBased=self.voxel_based)

            rm_list = []
            for d in self.features_to_drop:
                _rm_list = [x for x in list(feature_map.keys()) if d in x]
                rm_list = rm_list + _rm_list

            plot_list = [word for word in list(feature_map.keys()) if word not in rm_list]

            if self.plot == True:
                if self.voxel_based == True:
                    print('plot:',pat_id+f'_channel{i}')
                    Visuals(self.config,self.out_sub_dir).plot_features_fromdict(pat_id+f'_channel{i}', img_arr_wchannel[0], feature_map, plot_list)
                else:
                    print('no img feat representation because features are calculated over entire image')

            label_arr = None        
            for feat_arr in plot_list:
                if label_arr ==None:
                    feat_map_arr = np.expand_dims(sitk.GetArrayFromImage(feature_map[feat_arr]),axis = 1)
                    label_arr = plot_list
                else:
                    feat_map_arr = np.concat((feat_map_arr, np.expand_dims(sitk.GetArrayFromImage(feature_map[feat_arr]),axis = 0)), axis=0)
            
            try:
                feat_map_arr_all_channels = np.concat((feat_map_arr_all_channels, feat_map_arr), axis=1)
            except:
                feat_map_arr_all_channels = feat_map_arr

        return feat_map_arr_all_channels, plot_list

    

    # def _get_feature_dict(self, dict_arr):
    #     # Put in dictionary of image arrays, add a new index for the dictionary of image features
    #     # Extract the features for each pixel
    #     feat_dict = {}
    #     for name, img_arr in dict_arr.items():
    #         print('calculating features for ', name)
    #         features_per_img = self.extract_pixel_features(name, img_arr)

    #         feat_dict[name+'_features_'+self.config.feature_extraction.method] = features_per_img

    #     return feat_dict
    

    def _get_feature_arr(self, img_arr, pat_id):
        # Take input imge arr of multiple channels and caclulate features per channel
        features_per_channel,feat_names = self.extract_pixel_features(img_arr, pat_id)

        feat_names_arr=np.expand_dims(np.array(feat_names), axis =0 )
        # feat_names = np.expand_dims(labels, axis=0)

        return features_per_channel, feat_names_arr
    



#     def calculate_ngtdm_image(image, neighborhood_size=3):
#     rows, cols = image.shape
#     ngtdm_image = np.zeros_like(image)  # Initialize an output image of the same size as input

#     # Iterate over the image pixels (excluding the borders)
#     for i in range(neighborhood_size, rows - neighborhood_size):
#         for j in range(neighborhood_size, cols - neighborhood_size):
#             # Extract the local neighborhood around the pixel (i, j)
#             neighborhood = image[i - neighborhood_size:i + neighborhood_size + 1, 
#                                  j - neighborhood_size:j + neighborhood_size + 1]
            
#             center_value = image[i, j]
            
#             # Compute the difference between the center pixel and its neighbors
#             differences = np.abs(neighborhood - center_value)
#             differences = differences[differences != 0]  # Exclude the center pixel
            
#             # Calculate the mean difference of the neighborhood
#             if len(differences) > 0:
#                 mean_diff = np.mean(differences)
#             else:
#                 mean_diff = 0  # In case there are no neighbors with non-zero difference

#             # Store the calculated NGTDM feature (mean difference) in the corresponding pixel
#             ngtdm_image[i, j] = mean_diff

#     return ngtdm_image

# # Apply the NGTDM function to the grayscale image
# ngtdm_image = calculate_ngtdm_image(gray_image)

# # Display the original image and the NGTDM feature image
# plt.figure(figsize=(10, 5))

# # Show original grayscale image
# plt.subplot(1, 2, 1)
# plt.imshow(gray_image, cmap='gray')
# plt.title("Original Grayscale Image")
# plt.axis('off')

# # Show the NGTDM feature image
# plt.subplot(1, 2, 2)
# plt.imshow(ngtdm_image, cmap='gray')
# plt.title("NGTDM Feature Image")
# plt.axis('off')

# plt.show()
