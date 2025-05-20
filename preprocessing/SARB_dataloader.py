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
import matplotlib.patches as patches


class SARB_dataloader(Dataset):
    def __init__(self, config, array_path = 'X_est_all'):

        self.img_size = config.data.resample
        self.normalize_feature_bb=True
        self.feat_classes = config.feature_extraction.feature_classnames

        self.mat_dir = config.data.mat_dir
        self.array_path = array_path
        if 'orig_cache' in config.data.visuals:
            self.plot = True
            self.save_channel0 = False
            self.save_channel_rgb = True
        self.output_path = config.output.loc
        os.makedirs(self.output_path, exist_ok=True)
        self.output_dir = self.output_path+'/feats'
        os.makedirs(os.path.join(self.output_dir), exist_ok=True)

        self.visuals =  Visuals(config)
        self.normalize = config.data.normalize

        self.control_str = config.data.control
        self.disease_str = config.data.disease

        self.orig_size = [2048,2048]
        #specify the number of samples to get (ex renal tubes have like 19 segmentations, so select first 4 for space)
        self.num_annotations_to_compare = 5

        try:
            self.resample_size = config.data.resample
        except:
            self.resample_size = None

        try:
            self.feat_name = config.feature_extraction.method
            self._feat_extractor = eval(f"{config.feature_extraction.method}")
            self.feat_extractor= self._feat_extractor(config)
        except:
            print('WARNING: make sure that you have set feature extractor to false otherwise it may be loading incorrectly and will cause downstream errors')
            self.feat_extractor= False
        
        if config.data.patch_size == False:
            self.patch_size = False
        else:
            self.patch_size = config.data.patch_size 

        #for preexisting feature loading and also for loading annotations as sepearte classses         
        try:
            self.load_existing=config.feature_extraction.load_existing#if this is true you dont need the images to be loaded ** avoid time consuming step.
            self.load_images= config.feature_extraction.load_images
        except:
            self.load_existing=False #if this is true you dont need the images to be loaded ** avoid time consuming step.
            self.load_images=True
        
        try:
            self.separate_classes= config.feature_extraction.separate_classes
            self.feature_extraction_class = config.feature_extraction.anatomy_class
        except:
            self.separate_classes= False 


        try:
            #see if annotation path is defined
            self.annotation_path = config.data.annotation_dir 
            if self.annotation_path == 'None':
                raise ValueError('annotation_dir should not be set to None just remove this from config.')
            self.max_annotations = config.data.max_annotations
            x=1
        except:
            self.annotation_path = None
            x=0

        if x == 1:
            self.img_details, self.img_arr, self.label_arr, self.img_class, self.img_features, self.img_feat_labels = self.get_numpy_dataset()
        else:
            #no annotations
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
                ##normalise each channel
                channels = self.get_normalize(mat_arr)
            else:
                channels = mat_arr

            
            mat_dict[file_name.split('.')[0]] = channels

            
        if self.plot==True:
            self.visuals.plot_cache(mat_dict)
        
        if self.save_channel0 ==True:
            self.visuals.save_channel0(mat_dict, channel = 0)

            
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

    def get_label_from_patid(self, pat_id):
        '''assumes yolo structure, label - bounding box [x center, y center, width, height]'''
        file_path = os.path.join(self.annotation_path,pat_id +'_boundingbox.txt')
        try:
            with open(file_path, "r") as file: ## if you get an error here its because there must be a matching annotation with each input image.
                lines = file.readlines()
                # Initialize an empty list to store the data
        except:
            raise ValueError('ERROR: You must have a matching txt annotation if you are loading annotations for all files in the data folder.')
        label_ls = []

        for line in lines:
            # Split each line by spaces and convert to float
            parts = line.strip().split()
            parts = [float(x) for x in parts]  # Convert to float for numerical operations

            # Append the data to the list
            label_ls.append(parts)
            label_arr=np.array(label_ls)  # Read the entire file
        return label_arr
    
    def get_feat_arr(self, channels, pat_id):
        if self.feat_extractor==False:
            print('FEATURE EXTRACTOR IS FALSE')
            feats_arr, feat_label_arr = None, None
        else:
            if self.load_existing == False:#if this is true you dont need the images to be loaded ** avoid time consuming step.
                feats_arr, feat_label_arr = self.feat_extractor._get_feature_arr(channels, pat_id)
                ## saving features 
                print('Saving .npz features')
                np.savez(self.output_dir+'/'+self.feat_name+pat_id+".npz", arr1=feats_arr, arr2=feat_label_arr)
            else:
                # Loop through items in the directory
                for item in os.listdir(self.load_existing):
                    if pat_id in item:
                        print(item)
                data = np.load(self.load_existing+'/'+item)
                feats_arr = data['arr1']
                feat_label_arr = data['arr2']
                print('loading feature:'+self.output_dir+'/'+self.feat_name+pat_id+".npz")
                

        return feats_arr, feat_label_arr 
    
    def get_img_crop_from_annotation(self, img, annotation, name='', plot_bb=True):
        ''' This function takes an input image and annotation bounding box, and returns the new image as the reshaped value in the config file.

        img: image or mat_arr, shape channel, height, width (c, h, w) 
        anotation: the annotation as a bounding box, array of shape (x,y,w,h) 
        
        all_cropped: output shape (channels, h, w) 
        '''

        x_bb, y_bb, w_bb, h_bb = map(float, annotation)

        #remove after clinicns send PAN samples
        if 'PAN' in name:
            rad=90
            x_bb, y_bb, w_bb, h_bb = map(float, annotation)
            x1 = int(x_bb)
            y1 = int(y_bb)
            x2 = int(x_bb+w_bb)
            y2 = int(y_bb+h_bb)
        else:
            # Calculate top-left and bottom-right coordinates
            x1 = int(x_bb - w_bb / 2)
            y1 = int(y_bb - h_bb / 2)
            x2 = int(x_bb + w_bb / 2)
            y2 = int(y_bb + h_bb / 2)

        y1 = 0 if y1 < 0 else y1
        y2 = 0 if y2 < 0 else y2
        x1 = 0 if x1 < 0 else x1
        x2 = 0 if x2 < 0 else x2

        #loop through each channel
        for c in range(img.shape[0]):

            img_channel = img[c,:,:]
            #resize image to new size
            img_channel = resize(img_channel, self.resample_size)

            fig, ax = plt.subplots()

            # Crop the object
            _tmp_img = img_channel[y1:y2, x1:x2]
            _tmp_img = resize(_tmp_img, self.resample_size)
            _cropped = np.expand_dims(_tmp_img,axis=0)

            if 'all_cropped' in locals():
                all_cropped = np.concatenate((all_cropped,_cropped),axis=0)
            else:
                all_cropped =_cropped

            
            if plot_bb == True:
                ax.imshow(img_channel,cmap='grey')
                # Create a Rectangle patch
                rect = patches.Rectangle((x1, y1),
                                        w_bb,
                                        h_bb,
                                        linewidth=2,
                                        edgecolor='red',
                                        facecolor='none')
                ax.add_patch(rect)
                plt.savefig('./output/cache/'+name)
            plt.close()


        return all_cropped

    def get_normalize(self, arr_patch):
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

        return normalized_channels
    
    def get_numpy_dataset(self):
        '''loads arrays from file and puts into numpy dataset for dataloader
        img_details: is the filename containing important aquisiton details given by Mihoko (list of string)]
        img: is array of image
        img_class: classification of healthy (0) or disease (1)
        img_features: features generated from the features specified in configuration files or loaded from prexisting save features (.npz files)
        img_features_labels: features name(s)
        
        '''
        mat_dict = {}
        ls_dir = os.listdir(self.mat_dir)

        ## clean search lists to remove DS_Store and anything folders not disease or control.
        cleaned_files = [item for item in [f for f in ls_dir if f != '.DS_Store'] if item == self.control_str or item == self.disease_str]
        if len(cleaned_files) == 0:
            raise ValueError('No disease or control str folders found. Please review your folder inputs in the configuration files!')


        ### Loops through
        print('loading features for patches of class: ', self.feature_extraction_class)

        for folder in cleaned_files:
            #input data directories must have specific folders for disease and control as listed in the control file (ex. PAN and Cont)
            print('loading folder: ', folder)

            for i in tqdm.tqdm(range(len(glob.glob(os.path.join(self.mat_dir,folder)+'/*/*/*/**')))):
                
                file_name = sorted(glob.glob(os.path.join(self.mat_dir,folder)+'/*/*/*/**'))[i]
                print('loading file: ',file_name)

                ### get patient info ### 
                pat_id = file_name.split('/')[6]+'_'+file_name.split('/')[7].replace('_result','')

                ### get bounding box annotations ### 
                if self.annotation_path!=None:
                    label_arr = self.get_label_from_patid(pat_id)
        
                ### load mat as array ### 
                if self.load_images == True:
                    mat_contents = self.load_file(os.path.join(self.mat_dir,file_name))
                    mat_arr = mat_contents[self.array_path]

                    if self.save_channel0 ==True:
                        self.visuals.save_channel0(mat_arr, pat_id, channel = 0)

                    if self.save_channel_rgb==True:
                        self.visuals.save_channel_rgb(mat_arr, pat_id)

                    if self.orig_size != None:
                        #note image is resampled to the square size first of 2048 by 2048 as done for annotations
                        mat_arr = resize(mat_arr, self.orig_size)

                    #move channels to first axis
                    mat_arr = np.moveaxis(mat_arr, -1, 0)
                    if self.normalize == True:
                        ##normalise each channel
                        channels=self.get_normalize(mat_arr)
                    else:
                        channels = mat_arr
                    
                     #plot visual of image/channels
                    mat_dict[pat_id] = channels
                    if self.plot==True:
                        self.visuals.plot_cache(mat_dict)

                ### get image class ### 
                disease_class = self.get_disease_class(folder)

                ### if image is split into patches (see config) split the image into patches, else continue ### 
                if self.patch_size == "None":
                    ### separate classes, is if you want to load annotations per each image as a different batch input ###
                    if self.separate_classes == True:
                        ## cut the image into croped images from the annotaiton file
                        # if mat arr doesnt exits, image was not loaded - raise error
                        patches_dict = {}
                        filtered_array = label_arr[label_arr[:, 0] == self.feature_extraction_class]

                        #check max number of annotations only take the first self.num_annotations_to_compare
                        if len(filtered_array)>self.num_annotations_to_compare:
                            filtered_array= filtered_array[:self.num_annotations_to_compare,:]

                        #loop through annotations, load OR calculate features
                        for ann in tqdm.tqdm(range(len(filtered_array))):
                            feat_class = self.feat_classes[self.feature_extraction_class]
                            crop_name = pat_id +'_featclass'+feat_class+'_bb_annotation_'+str(ann)
                            #bb is short for bounding box
                            bb = filtered_array[ann][1:]
                            #assumes image is cubic, resize 
                            bb=bb*self.img_size[0]

                            if self.load_images == True:
                                arr_patch = self.get_img_crop_from_annotation(mat_arr, bb, crop_name)
                                if self.normalize_feature_bb==True:
                                    #normalize the patch 
                                    arr_patch = self.get_normalize(arr_patch)
                            else:
                                arr_patch = None

                            #if load existing is False, calculate else load.
                            if self.load_existing == False:
                                img_features_patch, feat_label_arr=self.get_feat_arr(arr_patch, crop_name)
                            else:
                                ##load features and crop to new size
                                if 'feat_arr' in locals():
                                    pass
                                else:
                                    try:
                                        feat_arr, feat_label_arr  = self.get_feat_arr(channels, pat_id+'_'+file_name.split('/')[7].split('_')[0])
                                    except:
                                        channels=None
                                        feat_arr, feat_label_arr  = self.get_feat_arr(channels, pat_id+'_'+file_name.split('/')[7].split('_')[0])

                                #crop feat_arr to shape
                                img_features_patch = self.get_img_crop_from_annotation(feat_arr, bb)     
                
                                if self.annotation_path:
                                    labels_arr = np.expand_dims(bb,axis=0)
        
                            

                            if'img_details' in locals() :
                                img_details = np.concatenate((img_details,np.expand_dims(np.array([pat_id, file_name, ann]),axis=0)),0)
                                img = np.concatenate((img,np.array([arr_patch])),0)
                                img_class = np.concatenate((img_class, np.expand_dims(disease_class,axis=0)),0)
                                img_features = np.concatenate((img_features,np.expand_dims(img_features_patch,axis=0)),0)
                                img_features_labels = np.concatenate((img_features_labels,np.expand_dims(feat_label_arr,axis=0)),0)
                                labels_arr =np.concatenate((labels_arr,np.expand_dims(bb,axis=0)),0)

                            else:
                                img_details = np.expand_dims(np.array([pat_id, file_name,ann]),axis=0)
                                img = np.array([arr_patch])
                                img_class = np.array([disease_class])
                                img_features = np.expand_dims(img_features_patch,axis=0)
                                img_features_labels = np.expand_dims(feat_label_arr,axis=0)
                                labels_arr = np.expand_dims(bb,axis=0)

                    if self.separate_classes == False:
                        #for preexisting feature loading and also for loading annotations as sepearte classses        
                        feats_arr, feat_label_arr  = self.get_feat_arr(channels, pat_id)

                        if'img_details' in locals() :
                            img_details = np.concatenate((img_details,np.expand_dims(np.array([pat_id, file_name]),axis=0)),0)
                            img = np.concatenate((img,np.array([mat_arr])),0)
                            img_class = np.concatenate((img_class, np.expand_dims(disease_class,axis=0)),0)

                            if self.feat_extractor==False:
                                img_features = None
                                img_features_labels = None
                            else: 
                                img_features = np.concatenate((img_features,np.expand_dims(feats_arr,axis=0)),0)
                                img_features_labels = np.concatenate((img_features_labels,np.expand_dims(feat_label_arr,axis=0)),0)

                            if self.annotation_path!=None:
                                #add zeros cols if they done match
                                label_arr=np.expand_dims(label_arr,axis=0)
                                if label_arr.shape[1] < self.max_annotations:
                                    _label_arr = np.zeros([1,self.max_annotations,5])
                                    _label_arr[:,:label_arr.shape[1],:] = label_arr
                                    labels_arr = np.concatenate((labels_arr, _label_arr), axis=0)
                                else:
                                    labels_arr = label_arr

                        else:
                            img_details = np.expand_dims(np.array([pat_id, file_name]),axis=0)
                            img = np.array([mat_arr])
                            img_class = np.array([disease_class])
                            if self.feat_extractor==False:
                                img_features = None
                                img_features_labels = None
                            else:
                                img_features = np.expand_dims(feats_arr,axis=0)
                                img_features_labels = np.expand_dims(feat_label_arr,axis=0)
                            
                            if self.annotation_path!=None:
                                labels_arr = np.expand_dims(label_arr,axis=0)
                                #add zeros cols if they done match

                else:
                    ##patch the image and load each as a different sample
                    arr_patches = self.split_image_into_patches(mat_arr, (self.patch_size, self.patch_size))

                    # img_features_patches = self.split_image_into_patches(mat_arr, (self.patch_size, self.patch_size))
                    patches_dict = {}

                    for patch in tqdm.tqdm(range(int(mat_arr.shape[1]/self.patch_size))):
                        arr_patch = arr_patches[patch]
                        if self.normalize == True:
                            channels= self.normalize(self, arr_patch)
                        else:
                            channels= arr_patch
                        
                        patches_dict[pat_id+"patch"+str(patch)] = channels

                        if self.plot==True:
                            self.visuals.plot_cache(patches_dict)
                    
                        if self.feat_extractor==False:
                            img_features_patch, feat_label_arr=None, None
                        else:
                            img_features_patch, feat_label_arr = self.feat_extractor._get_feature_arr(channels, pat_id)

                        if'img_details' in locals() :
                            img_details = np.concatenate((img_details,np.expand_dims(np.array([pat_id, file_name, patch]),axis=0)),0)
                            img = np.concatenate((img,np.array([arr_patch])),0)
                            img_class = np.concatenate((img_class, np.expand_dims(disease_class,axis=0)),0)
                            if self.feat_extractor==False:
                                img_features, img_features_labels=None, None
                            else:
                                img_features = np.concatenate((img_features,np.expand_dims(img_features_patch,axis=0)),0)
                                img_features_labels = np.concatenate((img_features_labels,np.expand_dims(feat_label_arr,axis=0)),0)
                        else:
                            img_details = np.expand_dims(np.array([pat_id, file_name,patch]),axis=0)
                            img = np.array([arr_patch])
                            img_class = np.array([disease_class])
                            if self.feat_extractor==False:
                                img_features=None
                                img_feature_labels = None
                            else:
                                img_features = np.expand_dims(img_features_patch,axis=0)
                                img_features_labels = np.expand_dims(feat_label_arr,axis=0)

        #expand dimensions for torch indexing
        img_details_arr = np.expand_dims(img_details,axis=1)
        img_arr = np.expand_dims(img,axis=1)
        img_class_arr = np.expand_dims(img_class,axis=0)
        #convert to torch
        if (img_arr==None).any():
            img_torch = None
        else:
            img_torch = torch.from_numpy(img_arr).float()

        img_class_torch = torch.from_numpy(img_class_arr).float()

        if self.feat_extractor==False:
            img_features_arr, img_features_labels_arr, img_features_torch = None,None,None
        else:       
            img_features_arr = np.expand_dims(img_features,axis=1)
            img_features_labels_arr = img_features_labels
            img_features_torch = torch.from_numpy(img_features_arr).float()

        if self.annotation_path != None:
            labels_arr = np.expand_dims(labels_arr,axis=1)
            return img_details, img_arr, labels_arr, img_class_torch, img_features_torch, img_features_labels_arr
        else:
            return img_details_arr, img_torch, img_class_torch, img_features_torch, img_features_labels_arr
    
    
    def __getitem__(self, index):
        img_details = self.img_details[index]
        img_arr = self.img_arr[index]
        img_class = self.img_class[index]
        img_features = self.img_features[index]
        img_feat_labels = self.img_feat_labels[index]

        if self.annotation_path:
            label_arr = self.label_arr[index]
    
        # if self.perform_aug==True:
        #     # aug_seq = Augmentation(self.cfg).augmentation_fn()

            # #convert back to torch friendly   
            # x = torch.from_numpy(np.expand_dims(aug_image,axis=0)).float()
            # y = torch.from_numpy(aug_ann_array).float()
            # landmarks = torch.from_numpy(np.expand_dims(aug_kps,axis=0)).float()
            # if self.save_aug == True:
            #     visuals(self.aug_path+'/'+id).heatmaps(aug_image, aug_ann_array)

        if self.annotation_path:
            return img_details, img_arr, label_arr, img_class, img_features, img_feat_labels
        else:
            return img_details, img_arr, img_class, img_features, img_feat_labels
    
    def __len__(self):
        return len(self.img_details)