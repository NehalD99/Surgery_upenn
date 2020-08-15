import os
import json
import numpy as np
import matplotlib.pyplot as plt


import pdb
masks = np.load('masks.npy',allow_pickle=True)
cats = np.load('cates.npy',allow_pickle=True)
imags = np.load('imgs.npy',allow_pickle=True)
bboxes = np.load('bboxes.npy',allow_pickle=True)

images = imags[10:-3]
images = np.concatenate((imags[0:1], images))

catToName = {0:'nonsegmented', 1 :'artery', 2:'blood',3:'spinalcord' ,4:'bluntprobe',5:'scissor' }


#categories=[{"id": seqId+1, "name": seq["name"], "supercategory": 'None'}
 #                                     for seqId, seq in enumerate(cats)]

#pdb.set_trace()

numImages = images.shape[0]

numAnnotations = masks.shape[0]

os.makedirs("annotations", exist_ok=True)
os.makedirs("train2017", exist_ok=True)
maskcounter = 0
for i in range(numImages):
    image_id = i
    numboxes = len(bboxes[i])
    for j in range(numboxes):
        mask = masks[maskcounter]
        object_class_name = cats[i][j]
        maskcounter += 1
        
        plt.imsave("annotations/"+str(image_id) + '_' +str(catToName[object_class_name]) + '_' + str(maskcounter) + '.png', mask)
    plt.imsave("train2017/"+str(image_id)+ '.jpeg', np.transpose(images[i], (1, 2, 0)))


#json_data = { "info" : info_, "licenses":self.licenses}


#with open("Annotations.json","w") as jsonfile:
#    json.dump(json_data, jsonfile, sort_keys=True, indent=4)
