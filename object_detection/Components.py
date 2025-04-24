import matplotlib.pyplot as plt
import numpy as np
import os 

import seaborn as sns
import math
import yaml
import SimpleITK as sitk
import matplotlib.pyplot as plt
import cv2
import math
import torch

import matplotlib.patches as patches


class Components():

    def __init__(self, config, dataloader):
        self.dataset_yaml_name = '.yaml'

        self.config = config
        self.input_data_path = config.data.mat_dir

        self.channel = self.config.object_detection.channel
        self.compactness_threshold = self.config.object_detection.compactness_threshold
        self.lower_thresh = self.config.object_detection.threshold_lower #choose threshold value for structures, assumes its the same for g and r
        self.upper_thresh = self.config.object_detection.threshold_upper #choose threshold value for structures, assumes its the same for g and r
        self.upper_thresh_d = self.config.object_detection.threshold_upper_d #choose threshold value for structures, assumes its the same for g and r
        self.med_size = config.object_detection.med_size

        try:
            self.edge_pad_buffer = self.config.object_detection.edge_pad_buffer #adds padding around detected bounding box
        except:
            self.edge_pad_buffer = 0 

        self.size_g = self.config.object_detection.size_g  #chose pixel size for glomerulus thresholding
        self.size_r = self.config.object_detection.size_r  #chose pixel size for renal tubes thresholding
        self.control_str = config.data.control
        self.disease_str = config.data.disease

        self.dataloader = dataloader

        self.output_dir = config.output.loc
        
        self.output_sub_dir = 'object_detection_components'
        os.makedirs(os.path.join(self.output_dir,self.output_sub_dir), exist_ok=True)

        self.plt_name = self.output_sub_dir+'_'+self.control_str+'_'+self.disease_str+'_comparison'

        self.img_details_arr, self.img_torch, self.img_class_torch, self.img_features_torch, self.img_features_labels_arr = dataloader.dataset.img_details, dataloader.dataset.img_arr, dataloader.dataset.img_class, dataloader.dataset.img_features, dataloader.dataset.img_feat_labels
        
        if torch.is_tensor(self.img_torch)==True:
            self.img_torch= self.img_torch.numpy()
            self.img_class_torch=self.img_class_torch.numpy()
            
        self.img_size = config.data.resample[0]

        pass

    def is_within_range(self,value, reference_value, range_value):
        # Check if the absolute difference between value and reference_value is within the range
        return abs(value - reference_value) <= range_value

        
    def is_round(self, area, perimeter, compactness_threshold):
        compactness_threshold= self.compactness_threshold
        compactness = (4 * math.pi * area) / (perimeter ** 2)
        print('compactness:',compactness)
        return compactness >= compactness_threshold

    def plot_bounding_boxes(self, img_id, image=None, boxes=[]):
        fig, ax = plt.subplots()
        
        # If an image is provided, show it
        if image is not None:
            ax.imshow(image, cmap='gray')
        c=0
        # Add each bounding box to the plot
        for i, x_min, y_min, width, height in boxes:
            if i == 0:
                col='b'
            else:
                col='r'

            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=0.2, edgecolor=col, facecolor='none')
            ax.text(x_min+20, y_min+20, str(c), fontsize=5, ha='center', va='center', color='white')
            ax.add_patch(rect)
            c=c+1

        ax.set_title("Bounding Boxes")
        plt.savefig(self.output_dir+'/'+self.output_sub_dir+'/'+img_id+'_boundingbox.png',dpi=300)

    def _detect_objects(self):
        ### for every image in the dataloader open image and detect golmerulus and rentubes basedon connecteed components

        for pat_idx in range(len(self.img_torch)):
            name = self.img_details_arr[pat_idx][0]
            
            #convert image to sitk image 
            image = self.img_torch[pat_idx]

            image = image.reshape(image.shape[1],image.shape[2], image.shape[3])
            
            image_for_threshold = image[self.channel]
            
            try:
                pat_class = self.img_class_torch[0][pat_idx]
            except:
                pat_class = self.img_class_torch.numpy()[0][pat_idx]
            

            if pat_class==0:
                upper_threshold = self.upper_thresh 
            else:
                upper_threshold = self.upper_thresh_d
            # Create binary mask: 1 where value is in [lower, upper], else 0
            binary = ((image_for_threshold >= self.lower_thresh) & (image_for_threshold <= upper_threshold )).astype(np.uint8)

            # Flip 0s to 1s and 1s to 0s
            if self.channel==0:
                binary = 1 - binary
            binary_itk = sitk.GetImageFromArray(binary)
            binary_itk = sitk.Median(binary_itk, [self.med_size, self.med_size])  # 10x10 neighborhood for 2D image

            cc = sitk.ConnectedComponent(binary_itk)
            cc = sitk.RelabelComponent(cc, sortByObjectSize=True)

            # Extract stats
            stats = sitk.LabelShapeStatisticsImageFilter()
            stats.Execute(cc)
                ##Step 2: Relabel components by size (largest = label 1, next = 2, etc.)
            relabeled = sitk.RelabelComponent(cc, sortByObjectSize=True)

            try:
                #check if any labels identified
                stats.GetNumberOfPixels(20)
            except:
                print('RE-Thresholding with mean: ', image[self.channel].mean())
                upper_threshold = image[self.channel].mean()
                # Create binary mask: 1 where value is in [lower, upper], else 0
                binary = ((image_for_threshold >= self.lower_thresh) & (image_for_threshold <= upper_threshold )).astype(np.uint8)

                # Flip 0s to 1s and 1s to 0s
                if self.channel==0:
                    binary = 1 - binary

                binary_itk = sitk.GetImageFromArray(binary)
                binary_itk = sitk.Median(binary_itk, [self.med_size, self.med_size])  # 10x10 neighborhood for 2D image

                cc = sitk.ConnectedComponent(binary_itk)
                cc = sitk.RelabelComponent(cc, sortByObjectSize=True)

                # Extract stats
                stats = sitk.LabelShapeStatisticsImageFilter()
                stats.Execute(cc)
                    ##Step 2: Relabel components by size (largest = label 1, next = 2, etc.)
                relabeled = sitk.RelabelComponent(cc, sortByObjectSize=True)

            # Step 3: Extract top 5 labels (they will be labeled 1 to 5 after )
            top_labels = list(range(1, 40))  # labels 1 to 5

            plt.imshow(binary, cmap='gray')
            plt.savefig('test.png')
            _binary = binary * 255
            # Save the image
            cv2.imwrite('binary_image.png', _binary)
            # Step 4: Plot each of the top 5 components
            fig, axs = plt.subplots(2,10, figsize=(20, 5))

            #create empty lists to append the new bounding boxes too.
            g = []
            r = []
            rr,i=0,0

            for j, label in enumerate(top_labels):

                # Isolate label i using BinaryThreshold
                mask = sitk.BinaryThreshold(relabeled, lowerThreshold=label, upperThreshold=label)
                mask_np = sitk.GetArrayViewFromImage(mask)

                img_to_show = mask_np

                if i > 9:
                    rr = 1
                    i = 0

                #check that they are squarish h ~= w
                if stats.GetNumberOfPixels(label) > self.size_g:
                    g_bb = np.array(stats.GetBoundingBox(label))
                    size=stats.GetNumberOfPixels(label)
                    print(size)
                    # find which is bigger x or y and resize to that shape, move center down that direction.
                    if g_bb[2]>g_bb[3]:
                        if size >740000:
                            #assume it was all captured
                            pass
                        else:
                            g_bb[1] = g_bb[1]+(g_bb[2]-g_bb[3])/2
                            g_bb[3] = g_bb[2]
                    else:
                        if size >740000:
                            #assume it was all captured
                            pass
                        else:
                            g_bb[0] = g_bb[0]+(g_bb[3]-g_bb[2])/2
                            g_bb[2] = g_bb[3]

                    g_bb[2]=g_bb[2]+self.edge_pad_buffer
                    g_bb[3]=g_bb[3]+self.edge_pad_buffer

                    if len(g) == 1:
                        #there can only be 1 glomerulus
                        pass
                    else:
                        g.append(np.append(1,g_bb))


                        id = ': Glomerulus'

                        axs[rr][i].imshow(image_for_threshold, cmap='gray')
                        axs[rr][i].imshow(img_to_show, cmap='summer', alpha=0.1)
                        axs[rr][i].set_title(f'C{label}'+id)
                        axs[rr][i].axis('off')
                        i=i+1
                else:

                    if self.is_round(stats.GetNumberOfPixels(label),stats.GetPerimeter(label),0):
                        if self.img_class_torch[0][pat_idx] == 0:
                            size_r = self.size_r 
                        else:
                            size_r = self.size_r + 3000
                            
                        if stats.GetNumberOfPixels(label) > size_r:
                            r_bb = np.array(stats.GetBoundingBox(label))
                            r_bb[2]=r_bb[2]+self.edge_pad_buffer
                            r_bb[3]=r_bb[3]+self.edge_pad_buffer
                            r.append(np.append(0,r_bb))
                            id = ': Renal'
                            print(f"Component {label}:")
                            print(f" - Area (in pixels): {stats.GetNumberOfPixels(label)}")
                            print(f" - Physical Size (mm² or mm³): {stats.GetPhysicalSize(label)}")
                            print(f" - Centroid: {stats.GetCentroid(label)}")
                            print(f" - Bounding Box (x, y, width, height): {stats.GetBoundingBox(label)}")
                            print(f" - Perimeter: {stats.GetPerimeter(label)}")

                            if self.is_within_range(stats.GetBoundingBox(label)[2],stats.GetBoundingBox(label)[3],2000):
                                axs[rr][i].imshow(image_for_threshold, cmap='gray')
                                axs[rr][i].imshow(img_to_show, cmap='gray', alpha=0.1)
                                axs[rr][i].set_title(f'C{label}'+id)
                                axs[rr][i].axis('off')
                                i=i+1
                        


                    plt.tight_layout()
                    # plt.show()
                    # plt.close()
                    os.makedirs(self.output_dir+'/'+self.output_sub_dir, exist_ok=True)
                    try:
                        plt.savefig(self.output_dir+'/'+self.output_sub_dir+'/'+name+'_segments.png',dpi=300)
                    except:
                        name = name[0]
                        plt.savefig(self.output_dir+'/'+self.output_sub_dir+'/'+name+'_segments.png',dpi=300)


                    bounding_boxes = g+r
                    self.plot_bounding_boxes(name, image_for_threshold, bounding_boxes)
                    plt.close()
                

            ##save bb to text
            # Open file in write mode (this will create the file if it doesn't exist)
        
            file = os.path.join(self.output_dir,self.output_sub_dir,name+'_boundingboxs')
            with open(file+".txt", "+w") as file:
                for item in bounding_boxes:
                    file.write(",".join(str(num) for num in item)+'\n')

        return
    
