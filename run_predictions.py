import os
import numpy as np
import json
from PIL import Image
from scipy import ndimage

def template(path='template.jpg'):
    data_path = path
    I = Image.open(data_path)
    I = np.asarray(I)
    red = np.array([[I[i,j,0]*2/255.0-1 for j in range(I.shape[1])]for i in range(I.shape[0])])
    green = np.array([[I[i,j,1]*2/255.0-1 for j in range(I.shape[1])]for i in range(I.shape[0])])
    blue = np.array([[I[i,j,2]*2/255.0-1 for j in range(I.shape[1])]for i in range(I.shape[0])])
    
    return red, green, blue
    
def box(matrix):
    labeled_image, num_features = ndimage.label(matrix)
    # Find the location of all objects
    objs = ndimage.find_objects(labeled_image)

    boxes = []
    for ob in objs:
        boxes.append([int(ob[0].start), int(ob[0].stop), int(ob[1].start), int(ob[1].stop)])
        
    return(boxes)

def detect_red_light(I):
    '''
    This function takes a numpy array <I> and returns a list <bounding_boxes>.
    The list <bounding_boxes> should have one element for each red light in the 
    image. Each element of <bounding_boxes> should itself be a list, containing 
    four integers that specify a bounding box: the row and column index of the 
    top left corner and the row and column index of the bottom right corner (in
    that order). See the code below for an example.
    
    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''
    
    
    bounding_boxes = [] # This should be a list of lists, each of length 4. See format example below. 
    
    '''
    BEGIN YOUR CODE
    '''
    temp_all_channel = template(path='template.jpg')
    temp_red = temp_all_channel[0]  #use red channel
    ground_truth = np.sum(temp_red*temp_red)
    h = temp_red.shape[0]
    w = temp_red.shape[1]
    red_ch = I[:,:,0]
    conv_res = [[0]*(red_ch.shape[1]-w) for _ in range(I.shape[0]-h)]
    for i in range(red_ch.shape[0]-h):
        for j in range(red_ch.shape[1]-w):
            test_area = red_ch[i:i+h, j:j+w]
            #print(test_area)
            conv_res[i][j]=abs(np.sum(temp_red*test_area)-ground_truth) 
            #print(np.sum(temp_red*test_area))
            
    conv_res = conv_res/np.max(conv_res)
    conv_res = np.where(conv_res<0.01,1,0)  
    #ax = sns.heatmap(np.array(conv_res))
    #ax = sns.heatmap(np.array(red_ch))
    #plt.show()
        
    bounding_boxes = box(conv_res)
    '''
    END YOUR CODE
    '''
    
    for i in range(len(bounding_boxes)):
        assert len(bounding_boxes[i]) == 4
    
    return bounding_boxes

# set the path to the downloaded data: 
data_path = '../RedLights2011_Medium'

# set a path for saving predictions: 
preds_path = '../data/hw01_preds' 
os.makedirs(preds_path,exist_ok=True) # create directory if needed 

# get sorted list of files: 
file_names = sorted(os.listdir(data_path)) 

# remove any non-JPEG files: 
file_names = [f for f in file_names if '.jpg' in f] 

preds = {}
pass_idx = [12,15,19]
for i in range(len(file_names)):
    if i in pass_idx:
        continue
        
    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names[i]))
    
    # convert to numpy array:
    I = np.asarray(I)
    
    preds[file_names[i]] = detect_red_light(I)
    print(file_names[i],'done')

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds.json'),'w') as f:
    json.dump(preds,f)
