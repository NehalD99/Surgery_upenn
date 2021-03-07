# Surgery_upenn
 
HISTORY OF CHANGES



~~1) INSTALL MMDETECTIONv.1.0.0 FOR SOLO~~

  ~~FOLLOW INTRUCTIONS ON WEBSITE - https://github.com/WXinlong/SOLO~~

2) CONVERT SURGERY DATA TO COCO
   Steps -
   https://patrickwasp.com/create-your-own-coco-style-dataset/

   pip install the pycococreator using
   pip install git+git://github.com/waspinator/pycococreator.git@0.2.0
   from this website - https://github.com/waspinator/pycococreator
 
   Run make_data.py - to create masks, categories and images in the npy 
   Run makeAnnotate.py - to create the annotations and images folder required to create the json

   in the shapes_to_coco.py file under examples/shapes/ in pycococreator repo modify the categories as needed.

   Run this file by giving appropriate paths.

   The json file will be created.


