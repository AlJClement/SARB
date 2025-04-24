import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
import matplotlib.patches as patches
from scipy.io import loadmat
from skimage.transform import resize
import json

def save_xml(path, name, new_boxes):
    # Start building the XML string
    width, height, depth = 2048, 2048, 3
    # Define your variables
    folder_name = "my-project-name"
    file_name = name
    image_path = f"/{folder_name}/{file_name}"

    xml_content = f"""<annotation>
    <folder>{folder_name}</folder>
    <filename>{file_name}</filename>
    <path>{image_path}</path>
    <source>
        <database>Unspecified</database>
    </source>
    <size>
        <width>{width}</width>
        <height>{height}</height>
        <depth>{depth}</depth>
    </size>
    """
    # Loop through objects list and add each object to XML content
    for obj in new_boxes:
        max_x = obj[1]+obj[3]
        max_y = obj[2]+obj[4]

        if obj[1]<0:
            obj[1]=0
        if obj[2]<0:
            obj[2]=0

        if max_x >2048:
            max_x = 2048
        if max_y >2048:
            max_y = 2048

        xml_content += f"""
        <object>
        <name>{obj[0]}</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
        <xmin>{obj[1]}</xmin>
        <ymin>{obj[2]}</ymin>
        <xmax>{max_x}</xmax>
        <ymax>{max_y}</ymax>
        </bndbox>
    </object>
    """
        # [id,x_min,y_min,width,height] = new_boxes
        # object_strings = f'<object>
        #     <name>{id}</name>
        #     <pose>Unspecified</pose>
        #     <truncated>0</truncated>
        #     <difficult>0</difficult>
        #     <bndbox>
        #         <xmin>{x_min}</xmin>
        #         <ymin>{y_min}</ymin>
        #         <xmax>{width}</xmax>
        #         <ymax>{height}</ymax>
        #     </bndbox>
        # </object>'

    xml_content += "</annotation>"

    # Save to a file
    with open(path+'/'+name[:-4]+".xml", "w", encoding="utf-8") as file:
        file.write(xml_content)

    return

def plot_bounding_boxes(name, output_dir, image=None, boxes=[], add_radius=[0,0]):

    file_name = os.path.join(output_dir+'/new_bb_'+name)

    fig, ax = plt.subplots()
    
    # If an image is provided, show it
    if image is not None:
        dpi = 100
        height, width = image.shape
        figsize = (width / dpi, height / dpi)

        # Create the figure
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])  # Full image without border
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(image, cmap='gray', aspect='auto')

        # Save the figure as PNG with no padding or borders
        plt.savefig(output_dir+'/'+name, dpi=dpi, bbox_inches='tight', pad_inches=0)
        # plt.close()
        # ax.imshow(image, cmap='gray')
        # ax.grid(False)
        # ax.set_axis_off()
        # plt.tight_layout()
        # plt.savefig(output_dir+'/'+name,dpi=300)

    c=0
    # Add each bounding box to the plot
    for i, x_min, y_min, width, height in boxes:
        if i == 0:
            col='orange'
        else:
            col='red'

        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=0.6, edgecolor=col, facecolor='none')
        ax.text(x_min+20, y_min+20, str(c), fontsize=5, ha='center', va='center', color='white')
        ax.add_patch(rect)
        c=c+1
    
    ax.set_title("Bounding Boxes")
    # plt.show()
    print(name)
    plt.savefig(output_dir+'/bb_'+name,dpi=300)
    plt.close()

    fig, ax = plt.subplots()
    # If an image is provided, show it
    if image is not None:
        ax.imshow(image, cmap='gray')
    c=0

    new_boxes = []
    if add_radius is not None:
        ## add radius arbitrary to the bounding box
        for i, x_min, y_min, width, height in boxes:
            if i == 0:
                col='orange'
            else:
                col='r'

            if i == 1:
                pass
            else:
                x, y = add_radius[0],add_radius[1]
                x_min=x_min - x
                y_min=y_min - y
                width=width + x*2
                height=height + y*2

                

            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=0.2, edgecolor=col, facecolor='none')
            ax.text(x_min+20, y_min+20, str(c), fontsize=5, ha='center', va='center', color='white')
            ax.add_patch(rect)
            c=c+1
            
            ##save bb to text
            with open(file_name+".txt", "a") as file:
                file.write(str(i)+" "+str(x_min)+" "+str(y_min)+" "+str(width)+" "+str(height)+' \n')
            
            new_boxes.append([i,x_min,y_min,width,height])

            
        save_xml(output_dir, name, new_boxes)

            # Construct COCO-style annotation (you can customize this more)
            # coco_annotations.append({
            #     "image_id": c,  # This would usually come from the image dataset
            #     "category_id": i,
            #     "bbox": [x_min, y_min, width, height],
            #     "area": width * height,
            #     "iscrowd": 0,
            # })
    
    ax.set_title("Bounding Boxes")
    # plt.show()
    print(name)
    plt.savefig(output_dir+'/radius_added_'+name,dpi=300)
    plt.close()


    # output_json_path = output_dir+"/"+name+"annotations_coco.json"
    # with open(output_json_path, 'w') as f:
    #     json.dump(coco_annotations, f, indent=4)
    return








add_radius = [90,90]
# dir = '/Users/allison/Desktop/nii_repo/SARB/output/object_detection_components/Contkid1_middlekidney_edited'
# data = '/Users/allison/Desktop/Cont/Contkid1_middlekidney'
dir = '/Users/allison/Desktop/nii_repo/SARB/output/object_detection_components/Contkid1_middlekidney_edited'
data = '/Users/allison/Desktop/20250221_Contkid1_middlekidney'
output_dir = dir +'/new'
os.makedirs(output_dir,exist_ok=True)

png_files = [filename for filename in os.listdir(dir) if filename.lower().endswith("x.png")]

all_files = []
for root, dirs, files in os.walk(data):
    for file in files:
        full_path = os.path.join(root, file)
        if '.mat' in full_path:
            all_files.append(full_path)


for name in png_files:
    # find file which goes with annotaton
    filtered = [item for item in all_files if name.split('_')[-2]+'_result' in item]

    data = loadmat(filtered[0])
    mat_arr = data['X_est_all']
    mat_arr = np.moveaxis(mat_arr, -1, 0)
    arr_new = resize(mat_arr[0],[2048,2048])

    # bb = np.loadtxt(dir+'/'+name[:-4]+'s.txt')
    # dir+'/'+name[:-4]+'s.txt'

    # Read the file line by line and split manually
    with open(dir+'/'+name[:-4]+'s.txt', 'r') as f:
        lines = f.readlines()

    data = []
    for line in lines:
        cols = line.strip().split()  # or split(',') if comma-separated
        # If a column has multiple values like '1,463,347', split again
        new_cols = []
        for col in cols:
            if ',' in col:
                new_cols.extend(col.split(','))
            else:
                new_cols.append(col)
        data.append([float(x) for x in new_cols])
    
    bb = np.array(data)
    
    plot_bounding_boxes(name, output_dir, arr_new, bb, add_radius)




    

