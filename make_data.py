import os
from PIL import Image
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import re
import tqdm


class DataMaker:
    def __init__(self):
        self.pic_path = "./train_image_3"
        self.label_path = "./VideoC_label_3"

    def ImgDataMaker(self):
        files = sorted(os.listdir(self.pic_path))
        spliter = re.compile("([a-zA-Z]+)([0-9]+)")
        files_num = [int(spliter.match(filename).groups()[1]) for filename in files if ".png" in filename]
        sorted_files_num = sorted(files_num)

        img_list = []
        for file_num in sorted_files_num:
            filename = "Frame" + str(file_num).zfill(4) + ".png"
            im_path = os.path.join(self.pic_path, filename)
            I = np.array(Image.open(im_path).convert("RGB"))
            # plt.imshow(I)
            # plt.show()
            I = I.transpose(2, 0, 1)
            img_list.append(I)
        img_arr = np.array(img_list)
        return img_arr

    # submasks are masks get from group same pixel value into a mask
    def InfoMaker(self):
        files = sorted(os.listdir(self.label_path))
        spliter = re.compile("([a-zA-Z_]+)([0-9]+)")
        files_num = [int(spliter.match(filename).groups()[1]) for filename in files if '.mat' in filename and 'DCNNtrain' in filename]
        sorted_files_num = sorted(files_num)

        cates_list = []
        sub_masks_list = []
        bbox_list = []
        for file_num in tqdm.tqdm(sorted_files_num):
            filename = "DCNNtrain_Frame" + str(file_num).zfill(4) + ".mat"
            # mask_dict = {}
            # if not "ordered" in filename:
            #     continue
            im_path = os.path.join(self.label_path, filename)
            # read .mat file, 4 layers in total.
            # 2-nd layer for anatomical objects
            # 3-rd layer for surgery tools
            label_4 = loadmat(im_path)['DCNNtrain']
            label_anatomical = label_4[:, :, 1]
            label_tools = label_4[:, :, 2]

            # find the mask for label, both anatomical and instrumental objects
            mask_dict = self.submask(label_anatomical, label_tools)

            # get cate and sub_mask info from dict
            cates = list(mask_dict.keys())
            sub_masks = np.array(list(mask_dict.values()))
            # build empty list for each img to get sub_sub_mask and info
            cates_list_img = []
            sub_masks_list_img = []
            bbox_list_img = []
            # for probe, split sub-mask into multiple instance
            for ind, sub_mask in enumerate(sub_masks):
                cate = cates[ind]
                if cate > 1000:  # category of tools, not general code.
                    labeled_array, num_features = ndimage.label(sub_mask)
                    # split out non-connected instance
                    for ins_i in range(num_features):
                        ins_pixel = ins_i+1  # start from 1
                        sub_sub_mask = labeled_array.copy()
                        sub_sub_mask[sub_sub_mask != ins_pixel] = 0
                        # plt.imshow(sub_sub_mask)
                        # plt.show()
                        # append the cate and sub_sub_mask, if instance is valid
                        valid = np.sum(sub_sub_mask > 0) > 1000  # not general code.
                        if valid:
                            cates_list_img.append(cate)
                            sub_masks_list_img.append(sub_sub_mask)
                else:
                    cates_list_img.append(cate)
                    sub_masks_list_img.append(sub_mask)

            # build the bounding box
            for ins_mask in sub_masks_list_img:
                # find the x1,y1,x2,y2 of the bounding box
                y, x = np.where(ins_mask > 0)
                ins_bbox = np.array([x.min(), y.min(), x.max(), y.max()])
                bbox_list_img.append(ins_bbox)
            cates_list_img_arr = np.array(cates_list_img)
            sub_masks_list_img_arr = np.array(sub_masks_list_img)
            bbox_list_img_arr = np.array(bbox_list_img)
            # attach info in current .png file to list
            cates_list.append(cates_list_img_arr)
            sub_masks_list.append(sub_masks_list_img_arr)
            bbox_list.append(bbox_list_img_arr)
        return cates_list, sub_masks_list, bbox_list

    def submask(self, l_anatomical, l_tool):
        assert l_anatomical.shape == l_tool.shape
        ins_info = {}
        # creat a submask for each category
        h, w = l_anatomical.shape
        # first record anatomical objects
        for i in range(h):
            for j in range(w):
                pixel = l_anatomical[i, j]
                # if the pixel is not background
                if pixel != 0:
                    # set the cls str for sub-mask
                    cls = pixel
                    sub_mask = ins_info.get(cls)
                    # check if submask has been created
                    if sub_mask is None:
                        # create the new sub-mask
                        ins_info[cls] = np.zeros((h, w))
                    # fill the mask
                    ins_info[cls][i, j] = 1

        # Second record tools objects
        for i in range(h):
            for j in range(w):
                pixel = l_tool[i, j]
                # if the pixel is not background
                if pixel > 1000:
                    # set the cls str for sub-mask
                    cls = pixel
                    sub_mask = ins_info.get(cls)
                    # check if submask has been created
                    if sub_mask is None:
                        # create the new sub-mask
                        ins_info[cls] = np.zeros((h, w))
                    # fill the mask
                    ins_info[cls][i, j] = 1


        return ins_info

if __name__ == '__main__':
    datamk = DataMaker()


    # save labels
    cates_list, sub_masks_list, bbox_list = datamk.InfoMaker()
    # record invalid img/label index
    valid = np.array([len(cate) for cate in cates_list]) != 0
    invalid_idx_list = np.where(valid == False)[0]

    # delete invalid label
    invalid_idx_counter = 0
    for invalid in invalid_idx_list:
        cates_list.pop(invalid - invalid_idx_counter)
        sub_masks_list.pop(invalid - invalid_idx_counter)
        bbox_list.pop(invalid - invalid_idx_counter)
        invalid_idx_counter += 1
    # resulting numpy array
    cates_arr = np.array(cates_list)
    np.save("catesC_3.npy", cates_arr)
    del cates_arr; del cates_list

    masks_arr = np.concatenate(sub_masks_list)
    np.save("masksC_3.npy", masks_arr)
    del masks_arr; del sub_masks_list

    bbox_arr = np.array(bbox_list)
    np.save("bboxesC_3.npy", bbox_arr)
    del bbox_arr; del bbox_list

    # save imgs
    img_arr = datamk.ImgDataMaker()
    # delete invalid images
    img_arr = img_arr[valid]
    np.save("imgsC_3.npy", img_arr)
    print()