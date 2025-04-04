import scipy
import scipy.io as sio
import os
from visualization import Visuals
import numpy as np
import torch
from feature_extraction import *
import glob
import tqdm
from torch.utils.data import Dataset
from skimage.transform import resize

class SARB_dataloader(Dataset):
    def __init__(self, config, array_path = 'X_est_all'):
        
        self.mat_dir = config.data.mat_dir
        self.array_path = array_path
        if 'orig_cache' in config.data.visuals:
            self.plot = True
        self.visuals =  Visuals(config)
        self.normalize = config.data.normalize

        self.control_str = config.data.control
        self.disease_str = config.data.disease

        try:
            self.resample_size = config.data.resample
        except:
            self.resample_size = None

        self._feat_extractor = eval(f"{config.feature_extraction.method}")
        self.feat_extractor= self._feat_extractor(config)
        
        if config.data.patch_size == False:
            self.patch_size = False
        else:
            self.patch_size = config.data.patch_size 

        self.img_details, self.img_arr, self.img_class, self.img_features, self.img_feat_labels = self.get_numpy_dataset() 



        return


    def load_file(self,mat: str):
        mat_contents = sio.loadmat(mat)
        return mat_contents

    def one_files_to_arr(self,file_name:str):
        '''loads array from filename as str
        returns a mat_arr as the array'''
        mat_contents = self.load(os.path.join(self.mat_dir,file_name))
        mat_arr = mat_contents[self.array_path]
        if self.normalize == True:
            data_min = np.min(mat_arr, axis=(1,2), keepdims=True)
            data_max = np.max(mat_arr, axis=(1,2), keepdims=True)

            scaled_data = (mat_arr - data_min) / (data_max - data_min)
        return scaled_data
    
    def multiple_files_to_dict(self):
        '''loads array from filename for anythin in the init directory
            returns a mat_arr with the name of each file, in a dictionary
            dictionary containing the file name and the array'''
        mat_dict = {}
        for file_name in os.listdir(self.mat_dir):
            mat_contents = self.load_file(os.path.join(self.mat_dir,file_name))
            mat_arr = mat_contents[self.array_path]
            #move channels to first axis
            mat_arr = np.moveaxis(mat_arr, -1, 0)
            if self.normalize == True:
                # data_min = np.min(mat_arr, axis=(1,2), keepdims=True)
                # data_max = np.max(mat_arr, axis=(1,2), keepdims=True)
                # scaled_mat_arr = (mat_arr - data_min) / (data_max - data_min)

                ##normalise each channel
                normalized_channels = None
                for i in range(mat_arr.shape[0]):
                    arr = mat_arr[i]
                    norm_arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
                    norm_arr = np.expand_dims(norm_arr, axis=0)
                    if normalized_channels != None:
                        normalized_channels = np.append(normalized_channels, norm_arr, axis=0)
                    else:
                        normalized_channels = norm_arr

                channels = normalized_channels
            else:
                channels = mat_arr

            
            mat_dict[file_name.split('.')[0]] = channels

            
        if self.plot==True:
            self.visuals.plot_cache(mat_dict)
            
        return mat_dict
    
    def get_disease_class(self,folder):
        if folder == self.control_str:
            return 0
        elif folder == self.disease_str:
            return 1
        else:
            raise ValueError('Check disease/control group folders must match CONFIG')

    def split_image_into_patches(self, arr_image, patch_size):
        """
        Splits an image of shape (C, H, W) into non-overlapping patches.

        Args:
            image (numpy.ndarray): Input image with shape (C, H, W).
            patch_size (tuple): (patch_H, patch_W) specifying patch size.

        Returns:
            numpy.ndarray: Patches with shape (num_patches, C, patch_H, patch_W).
        """
        C, H, W = arr_image.shape  # Get channels, height, width
        patch_H, patch_W = patch_size

        # Ensure image dimensions are divisible by patch size
        H_crop, W_crop = H - (H % patch_H), W - (W % patch_W)
        arr_image = arr_image[:, :H_crop, :W_crop]  # Crop image to be perfectly divisible

        # Reshape into patches
        patches = arr_image.reshape(C, H_crop // patch_H, patch_H, W_crop // patch_W, patch_W)
        
        # Reorder dimensions to (num_patches, C, patch_H, patch_W)
        patches = patches.transpose(1, 3, 0, 2, 4).reshape(-1, C, patch_H, patch_W)

        return patches


    def get_numpy_dataset(self):
        '''loads arrays from file and puts into numpy dataset for dataloader
        img_details: is the filename containing important aquisiton details given by Mihoko
        img: is array of image
        img_class: classification of healthy (0) or disease (1)
        img_features: features generated from the features specified in configuration files
        img_features_labels: features names 
        
        '''
        mat_dict = {}

        for folder in os.listdir(self.mat_dir):
            #input data directories must have specific folders for disease and control as listed in the control file (ex. PAN and Cont)
            print('loading: ', folder)

            for i in tqdm.tqdm(range(len(glob.glob(os.path.join(self.mat_dir,folder)+'/*/*/*/**')))):
                file_name = sorted(glob.glob(os.path.join(self.mat_dir,folder)+'/*/*/*/**'))[i]
                print('loading: ',file_name)

                ### get patient info ### 
                pat_id = file_name.split('/')[6]+'_'+file_name.split('/')[7].replace('_result','')

                ### load mat as array ### 
                mat_contents = self.load_file(os.path.join(self.mat_dir,file_name))
                mat_arr = mat_contents[self.array_path]

                if self.resample_size != None:
                    mat_arr = resize(mat_arr, self.resample_size)

                #move channels to first axis
                mat_arr = np.moveaxis(mat_arr, -1, 0)
                if self.normalize == True:
                    ##normalise each channel
                    normalized_channels = None       
                    for i in range(mat_arr.shape[0]):
                        arr = mat_arr[i]
                        norm_arr = (((arr - np.min(arr)) / (np.max(arr) - np.min(arr)))*255)#.astype('uint8')
                        norm_arr = np.expand_dims(norm_arr, axis=0)
                        try:
                            normalized_channels = np.append(normalized_channels, norm_arr, axis=0)
                        except:
                            normalized_channels = norm_arr

                    channels = normalized_channels
                else:
                    channels = mat_arr

                ### image class ### 
                disease_class = self.get_disease_class(folder)

                ### calculate features ###
                #plot visual of image/channels
                mat_dict[pat_id] = channels
                if self.plot==True:
                    self.visuals.plot_cache(mat_dict)
                
                if self.patch_size == "None":
                    feats_arr, feat_label_arr = self.feat_extractor._get_feature_arr(channels, pat_id)

                    if'img_details' in locals() :
                        img_details = np.concatenate((img_details,np.expand_dims(np.array([pat_id, file_name]),axis=0)),0)
                        img = np.concatenate((img,np.array([mat_arr])),0)
                        img_class = np.concatenate((img_class, np.expand_dims(disease_class,axis=0)),0)
                        img_features = np.concatenate((img_features,np.expand_dims(feats_arr,axis=0)),0)
                        img_features_labels = np.concatenate((img_features_labels,np.expand_dims(feat_label_arr,axis=0)),0)

                    else:
                        img_details = np.expand_dims(np.array([pat_id, file_name]),axis=0)
                        img = np.array([mat_arr])
                        img_class = np.array([disease_class])
                        img_features = np.expand_dims(feats_arr,axis=0)
                        img_features_labels = np.expand_dims(feat_label_arr,axis=0)
                else:
                    ##patch the image and load each as a different sample
                    arr_patches = self.split_image_into_patches(mat_arr, (self.patch_size, self.patch_size))

                    # img_features_patches = self.split_image_into_patches(mat_arr, (self.patch_size, self.patch_size))
                    patches_dict = {}

                    for patch in tqdm.tqdm(range(int(mat_arr.shape[1]/self.patch_size))):
                        arr_patch = arr_patches[patch]

                        if self.normalize == True:
                            ##normalise each channel
                            normalized_channels = None       
                            for i in range(arr_patch.shape[0]):
                                arr = arr_patch[i]
                                norm_arr = (((arr - np.min(arr)) / (np.max(arr) - np.min(arr)))*255)#.astype('uint8')
                                norm_arr = np.expand_dims(norm_arr, axis=0)
                                try:
                                    normalized_channels = np.append(normalized_channels, norm_arr, axis=0)
                                except:
                                    normalized_channels = norm_arr

                            channels = normalized_channels
                        else:
                            channels = arr_patch
                        
                        patches_dict[pat_id+"patch"+str(patch)] = channels

                        if self.plot==True:
                            self.visuals.plot_cache(patches_dict)
                        
                        img_features_patch, feat_label_arr = self.feat_extractor._get_feature_arr(channels, pat_id)

                        if'img_details' in locals() :
                            img_details = np.concatenate((img_details,np.expand_dims(np.array([pat_id, file_name, patch]),axis=0)),0)
                            img = np.concatenate((img,np.array([arr_patch])),0)
                            img_class = np.concatenate((img_class, np.expand_dims(disease_class,axis=0)),0)
                            img_features = np.concatenate((img_features,np.expand_dims(img_features_patch,axis=0)),0)
                            img_features_labels = np.concatenate((img_features_labels,np.expand_dims(feat_label_arr,axis=0)),0)
                        else:
                            img_details = np.expand_dims(np.array([pat_id, file_name,patch]),axis=0)
                            img = np.array([arr_patch])
                            img_class = np.array([disease_class])
                            img_features = np.expand_dims(img_features_patch,axis=0)
                            img_features_labels = np.expand_dims(feat_label_arr,axis=0)


        #expand dimensions for torch indexing
        img_details_arr = np.expand_dims(img_details,axis=1)
        img_arr = np.expand_dims(img,axis=1)
        img_class_arr = np.expand_dims(img_class,axis=0)
        img_features_arr = np.expand_dims(img_features,axis=1)
        img_features_labels_arr = img_features_labels

        #convert to torch
        img_torch = torch.from_numpy(img_arr).float()
        img_class_torch = torch.from_numpy(img_class_arr).float()
        img_features_torch = torch.from_numpy(img_features_arr).float()

        return img_details_arr, img_torch, img_class_torch, img_features_torch, img_features_labels_arr

    
    
    def __getitem__(self, index):
        img_details = self.img_details[index]
        img_arr = self.img_arr[index]
        img_class = self.img_class[index]
        img_features = self.img_features[index]
        img_feat_labels = self.img_feat_labels[index]

        # if self.perform_aug==True:
        #     # aug_seq = Augmentation(self.cfg).augmentation_fn()

            # #convert back to torch friendly   
            # x = torch.from_numpy(np.expand_dims(aug_image,axis=0)).float()
            # y = torch.from_numpy(aug_ann_array).float()
            # landmarks = torch.from_numpy(np.expand_dims(aug_kps,axis=0)).float()
            # if self.save_aug == True:
            #     visuals(self.aug_path+'/'+id).heatmaps(aug_image, aug_ann_array)

        return img_details, img_arr, img_class, img_features, img_feat_labels
    
    def __len__(self):
        return len(self.img_details)