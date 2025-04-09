import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import numpy as np

from torchvision.transforms.functional import to_pil_image



class SimCLR(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

        # Preprocessing Transform (as used in SimCLR training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def feat_extractor(self,img_tensor):
            # Load Pretrained Backbone
            # Load a pre-trained ResNet50 model (or any other backbone used in SimCLR)
        model = models.resnet50(pretrained=True)

        # Remove the final fully connected layer to extract features
        # Extract features from the penultimate layer (just before classification)
        model = torch.nn.Sequential(*list(model.children())[:-1])

        # Put the model in evaluation mode
        model.eval()

        # Perform feature extraction
        with torch.no_grad():
            features = model(img_tensor.unsqueeze(0))  # Extract features

        return features

    def forward(self, x):
        self.backbone.fc = torch.nn.Sequential(torch.nn.Identity()) 
        return self.backbone(x)
    
    def _get_feature_arr(self, mat_arr, pat_id):
        features_per_channel=None
        ## max 3 channels so loop through each channel
        for i in range((mat_arr.shape[0])):
            _mat_arr=np.expand_dims(mat_arr[i,:,:],axis=0)

            image_mat_arr = torch.from_numpy(_mat_arr)#.permute(1,2,0)
            # img_PIL = Image.fromarray(mat_arr)
            img_pil = to_pil_image(image_mat_arr)
            img_RGB = img_pil.convert('RGB')
            image_tensor = self.transform(img_RGB)

            # EXTRACT 
            feature_per_channel = self.feat_extractor(image_tensor)
            np_feat = feature_per_channel.numpy().reshape(1,2048)
            try:
                features_per_channel = np.concat([features_per_channel,np_feat],axis=0)
            except:
                features_per_channel=np_feat
        feat_names_arr=np.array(["simclr features"])
        feat_names_arr=np.expand_dims(np.array(feat_names_arr), axis =0)

        features_per_channel=np.expand_dims(np.array(features_per_channel), axis =0)
        ### features_per_channel: SHAPE SHOULD BE [IMG, C, H, W] or [img, c, num_FEATS] DEPENDING ON the features being calculated
        ### faeture_names_arr: SHAPE SHOULD BE [IMG, NAMES]


        return features_per_channel, feat_names_arr
