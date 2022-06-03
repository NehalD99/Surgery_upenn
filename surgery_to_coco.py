import tqdm 
from detectron2.data.detection_utils import read_image
import argparse
from detectron2.utils.file_io import PathManager
import os
import json
import shutil
import numpy as np
import cv2
import pdb
import datetime
from scipy.io import loadmat
import pycocotools.mask as mask_util
from pycocotools import mask


def coco_ann( l, h, w, x, y, cat_id,image_id, segm):

   
            coco_annotation = {}
            coco_annotation['image_id'] = image_id
            coco_annotation['id'] = l + 1
            coco_annotation['width'] = w
            coco_annotation['height'] = h
            coco_annotation['category_id'] = cat_id
            # bbox_pred = np.asarray(instances.pred_boxes[i].tensor[0])
            # bbox_pred = [bbox_pred[0], bbox_pred[1], bbox_pred[2] - bbox_pred[0], bbox_pred[3] - bbox_pred[1]]
            # compress to RLE format
            segm_pred = mask.encode(np.asfortranarray( segm).astype('uint8'))
            ############## POLYGON format saving  ########################
            # maskedArr = mask.decode(segm_pred)
            # contours,_ = cv2.findContours(maskedArr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # segment = []
            # valid_poly = 0
            # for contour in contours:
            #     if contour.size >= 6:
            #         segment.append(contour.astype(float).flatten().tolist())
            #         valid_poly += 1
            #         if valid_poly == 0:
            #             raise ValueError
            # coco_annotation['segmentation'] = segment

            ################ POLYGON format saving#########################
            # maskedArr = mask.decode(segm_pred)
            ########### RLE format saving #########
            coco_annotation['segmentation'] = segm_pred
            area = mask_util.area(segm_pred).item()
            coco_annotation['area'] = int(area)
            if isinstance(segm_pred, dict):  # RLE
                counts = segm_pred["counts"]
                if not isinstance(counts, str):
                    # make it json-serializable
                    coco_annotation['segmentation']['counts'] = counts.decode("ascii")
            ############ RLE format saving #########

            x1 = np.min(x)
            y1 = np.min(y)
            x2 = np.max(x)
            y2 = np.max(y)
            coco_bbox = [ x1, y1, x2-x1, y2-y1]
            coco_annotation['bbox'] = list(int(np.round(x)) for x in coco_bbox)
            coco_annotation["iscrowd"] = 0 # Polygon format uses 0 but rle uses 1, slightly confused here


            return coco_annotation  



def main(image_path='', annotation_path=''): 
    
        coco_annotations = []
        coco_images = []
        for filename in os.listdir(annotation_path):

            if filename.startswith('DCNNtrainSS_Frame') or filename.startswith("DCNNtrain_Frame"):
                pass
            else:
                continue
            framename = filename.split('_')[1].split('.')[0]
            if framename +'.jpg' not in os.listdir(image_path):
                continue

            path = os.path.join(image_path, framename +'.jpg')            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")


            #INSERT CODE TO GENERATE COCO FORMAT FOR PREDICTIONS
            coco_image = {
                "id": int(framename[-4:]),
                "width": img.shape[1],
                "height": img.shape[0],
                "file_name": framename + '.jpg',
            }


            # mask_dict = {}
            # if not "ordered" in filename:
            #     continue
            mat_path = os.path.join(annotation_path, filename)
            # read .mat file, 4 layers in total.
            # 2-nd layer for anatomical objects
            # 3-rd layer for surgery tools
            print(mat_path)
            anns_per_img = loadmat(mat_path)['DCNNtrain']
            if len(anns_per_img.shape)!=3:
                print("Shape of annotation matrix is not 3D")
                continue
            print(anns_per_img.shape)


            if 'SS' in filename:

                for i in range(anns_per_img.shape[-1]):

                    y, x = np.where(anns_per_img[:,:,i] != 0)
                    if y.size == 0:
                        continue

                    c_a = coco_ann(l = len(coco_annotations), h=img.shape[0], w=img.shape[1], x=x, y=y, cat_id=int(np.unique(anns_per_img[:,:,i])[-1]),image_id=int(coco_image["id"]), segm=anns_per_img[:,:,i])
                    
 
                    coco_annotations.append(c_a)    
                    coco_images.append(coco_image)  


            else:

                for i in np.unique(anns_per_img[:,:,1]):

                    y, x = np.where(1*(anns_per_img[:,:,1] == i)!= 0)
                    if i == 0 or y.size == 0 :
                        continue

                    c_a = coco_ann( l = len(coco_annotations),h=img.shape[0], w=img.shape[1], x=x, y=y, cat_id=int(i),image_id=int(coco_image["id"]), segm=1*(anns_per_img[:,:,1]==i))


                    coco_annotations.append(c_a)    
                    coco_images.append(coco_image)  

                for i in np.unique(anns_per_img[:,:,2]):

                    y, x = np.where(1*(anns_per_img[:,:,2] == i)!= 0)
                    if  i<1000 or y.size == 0 :
                        continue
                    c_a = coco_ann( l = len(coco_annotations),h=img.shape[0], w=img.shape[1], x=x, y=y, cat_id=int(i),image_id=int(coco_image["id"]), segm=1*(anns_per_img[:,:,2]==i))


                    coco_annotations.append(c_a)    
                    coco_images.append(coco_image)


        return coco_images, coco_annotations




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="")
    parser.add_argument("--annotations",default="")
    parser.add_argument("--output_data")
    args = parser.parse_args()

    os.makedirs(args.output_data, exist_ok=True)

    coco_images, coco_annotations = main(args.images, args.annotations)
# save coco json

    dataset_name = 'Surgery_COCO'
    output_file = args.output_data + '/' +  args.images.split('/')[-1] + '_coco.json'


    # These categories are instance specific
    categories = [{"id": 1, "name": "Cerebellum", "supercategory": "shape"}, {"id": 2, "name": "Arachnoid", "supercategory": "shape"},{"id": 3, "name": "CN8", "supercategory": "shape"}, {"id": 4, "name": "CN5", "supercategory": "shape"}, {"id": 5, "name": "CN7", "supercategory": "shape"}, {"id": 6, "name": "CN_9_10_11", "supercategory": "shape"},{"id": 7, "name": "SCA", "supercategory": "shape"}, {"id": 8, "name": "AICA", "supercategory": "shape"}, {"id": 9, "name": "SuperiorPetrosalVein", "supercategory": "shape"}, {"id": 10, "name": "Labrynthine", "supercategory": "shape"}, {"id": 11, "name": "Vein", "supercategory": "shape"}, {"id": 12, "name": "Brainstem", "supercategory": "shape"},{"id": 1001, "name": "Suction", "supercategory": "shape"}, {"id": 1003, "name": "Bipolar", "supercategory": "shape"}, {"id": 1004, "name": "Forcep", "supercategory": "shape"}, {"id": 1005, "name": "BluntProbe", "supercategory": "shape"}, {"id": 1006, "name": "Drill", "supercategory": "shape"}, {"id": 1007, "name": "Kerrison", "supercategory": "shape"}, {"id": 1008, "name": "Cottonoid", "supercategory": "shape"}, {"id": 1009, "name": "Scissors", "supercategory": "shape"}, {"id": 1012, "name": "Unknown", "supercategory": "shape"}]
    info = {
        "date_created": str(datetime.datetime.now()),
        "description": "Automatically generated COCO json file for Detectron2.",
    }
    coco_dict = {"info": info, "images": coco_images, "categories": categories, "licenses": None}
    if len(coco_annotations) > 0:

        coco_dict["annotations"] = coco_annotations

    # logger.info(f"Caching COCO format annotations at '{output_file}' ...")
    tmp_file = output_file + ".tmp"
    with PathManager.open(tmp_file, "w") as f:
        json.dump(coco_dict, f)
    shutil.move(tmp_file, output_file)
