import matplotlib.pyplot as plt
import numpy as np
import os 
from scipy import stats
import sklearn
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import math
import yaml
from ultralytics import YOLO

class ObjectDetection():
    def __init__(self, config, dataloader):
        self.dataset_yaml_name = 'yolov5_config.yaml'

        self.config = config
        self.input_data_path = config.data.mat_dir


        self.control_str = config.data.control
        self.disease_str = config.data.disease
        
        self.dataloader = dataloader

        self.output_dir = config.output.loc
        
        self.output_sub_dir = 'object_detection'
        os.makedirs(os.path.join(self.output_dir,self.output_sub_dir), exist_ok=True)

        self.plt_name = self.output_sub_dir+'_'+self.control_str+'_'+self.disease_str+'_comparison'
        
        self.img_details_arr, self.img_torch, self.label_torch, self.img_class_torch, self.img_features_torch, self.img_features_labels_arr = dataloader.dataset.img_details, dataloader.dataset.img_arr, dataloader.dataset.label_arr, dataloader.dataset.img_class, dataloader.dataset.img_features, dataloader.dataset.img_feat_labels
        self.img_size = config.data.resample[0]
        pass

    def create_dataset_yaml(self, n_classes = 2, class_names = ['glomerulus', 'renal tube']):
        # yolov5.yaml
        # Example dictionary
        data = {
            'train': self.input_data_path+'/training',
            'val': self.input_data_path+'/validation',
            'nc': n_classes,
            'names': {
                0: class_names[0],
                1: class_names[1]
            }
        }

        # Specify the output file path
        output_file = os.path.join(self.input_data_path,self.dataset_yaml_name)

        with open(output_file, 'w') as file:
            yaml.dump(data, file, default_flow_style=False)

        return

    def _detect_objects(self):
        ### for every image in the dataloader open, run YOLO and print objects
        #create dataset yaml
        self.create_dataset_yaml()

        #Load pre-trained model
        model = YOLO('yolov5s.pt')  # You can choose 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt' 

        #Train
        model.train(
            data=os.path.join(self.input_data_path,self.dataset_yaml_name),  # Path to your dataset YAML file
            imgsz=self.img_size,  # Image size 
            batch=1,              # Batch size 
            epochs=50,            # Number of epochs (adjust as needed)
            lr0=0.01,
            cache=True,           # Cache images for faster training
        )

        # Validate
        results = model.val(data=self.dataset_yaml_name, imgsz=self.img_size)
        print("Validation Results:")
        print(results)  # This will show metrics like mAP, precision, recall, etc.

        # Save Model (saved in 'runs/train/expX')
        best_model_path = os.path.join(model.best_model)  # Path to the best model
        print(f"Saving to {best_model_path}")

        # Step 6: Perform inference using the best model
        img_path = '/path/to/test_image.jpg'  # Replace with your test image path
        img = cv2.imread(img_path)
        result = model(img_path)

        # Print and display results
        result.print()  # Print detection results
        result.show()   # Show the image with bounding boxes

        return