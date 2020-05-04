#!/usr/bin/python
# utils
import sys
sys.path.insert(0, '')
# from clothes_seg.src.mhpfcn_trtis import MHPFCNTRTIS
from clothes_seg.src.mhpfcn_trt import MHPFCNTRT

import cv2
import numpy as np 
import os 
from time import perf_counter
from tqdm import tqdm


from clothes_seg.lib.color_extract import extract_color_clustering, extract_color_inference


def main_video(stream):
    # init 
    model = MHPFCNTRT()

    cap = cv2.VideoCapture(stream)
    frame_length = 1*60*25  # min * second * fps

    imgs = []
    batch_size = 64

    for count in range(frame_length):
        ret, frame = cap.read()
        if ret:
            imgs.append(frame)
        else:
            break
        
        if count % batch_size == 0:
            padding = False
            mask_results = model.do_inference(imgs, padding)
            model.mask2img(mask_results)

            imgs = []

            for rs in mask_results:
                cv2.imshow('frame', rs)
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break
    cap.release()
    cv2.destroyAllWindows()

    return

# Debugging
def var(var):
    def print_array(var):
        print(type(var), " shape:",var.shape,"range: [{},{}]".format(var.min(),var.max()))

    vtype = type(var)
    if vtype == np.ndarray:
        print_array(var)
    elif vtype == list:
        print(type(var),len(var))
        if type(var[0]) == np.ndarray:
            for array in var:
                print_array(array)
        else:
            print(type(var[0]))
    else:
        pass
    exit()
    return

def main_folder(folder):

    # Foldeer

    img_list = sorted([os.path.join(folder,file) for file in os.listdir(folder) if file.endswith(".jpg")])

    raw_img_list = []
    for idx,img_name in enumerate(img_list):
        img = cv2.imread(img_name)
        img = img[:,:,::-1]   # bgr to rgb

        raw_img_list.append(img)


    # Inference
    model = MHPFCNTRT()
    batch_size = 64
    # for i in tqdm(range(0, len(raw_img_list), batch_size)):
    for i in range(0, len(raw_img_list), batch_size):
        
        imgs = raw_img_list[i:i + batch_size]
        file_names = img_list[i:i + batch_size]
        MHP_FCN(imgs, file_names, mask=True, overlay = True)

    return

def main(img_name):
    # img
    img = cv2.imread(img_name)
    img = img[:,:,::-1]   # bgr to rgb
    imgs = np.tile(img, [1, 1, 1, 1])
    imgs = list(imgs)
    # print(np.unique(img.reshape(int(img.size/3),3),axis=0))

    print('================ Process A ================')
    MHP_FCN(imgs, [img_name], mask=True, overlay = True)

    # Color filter - Clustering
    print('================ Process B ================')
    healthcheck_color_filter_clustering(imgs, plot=False, color_fmt='rgb')
    
    # Color filter - Inference
    print('================ Process C ================')
    healthcheck_color_filter_inference(imgs, color_fmt='rgb')

    return

def MHP_FCN(imgs:list, img_names: list, mask:bool=False, overlay:bool=False): 
    # imgs          (Input): [<HWC>,<HWC>,...], fmt: RGB, Range:[0,255]
    # img_name      (Input): [<str>,<str>,...]
    # mask_results  (Output): [<str>,<str>,...]

    # init 
    model = MHPFCNTRT()

    mask_results, resized_imgs = model.do_inference(imgs)

    model.mask2img(mask_results, imgs, mask, overlay)

    # output:
    # img_names = [i for i in img_names for _ in range(3)]
    model.write_img(mask_results, img_names)
    
    # show:
    # for mask_result in mask_results:
    #     mask_result = mask_result[:,:,[2,1,0]]
    #     cv2.imshow("",mask_result)
    #     cv2.waitKey()

    if mask_results:
        print('ClothesSeg [TRT] - MHPFCN : Healthy')
    else:
        print('ClothesSeg [TRT] - MHPFCN : Down')

    return 

def healthcheck_color_filter_clustering(imgs:list, plot:bool = False, color_fmt:str = 'hsv'):
    # imgs          (Input): [<HWC>,<HWC>,...], fmt: RGB, Range:[0,255]
    # color_fmt     (Input): 'hsv' or 'rgb'

    # init    
    model = MHPFCNTRT()

    mask_results, resized_imgs = model.do_inference(imgs)

    # only one img

    status = extract_color_clustering(resized_imgs[0], mask_results[0], color_fmt, plot)
    if status: # if any result
        print('ClothesSeg [Extractor (clustering)] - MHPFCN : Healthy')
        return True
    else:
        print('ClothesSeg [Extractor (clustering)] - MHPFCN : Down')
        return False
    # color_filter = ColorFilter(color_fmt)


def healthcheck_color_filter_inference(imgs:list, color_fmt:str = 'hsv'):
    # imgs          (Input): [<HWC>,<HWC>,...], fmt: RGB, Range:[0,255]
    # color_fmt     (Input): 'hsv' or 'rgb'

    # init    
    model = MHPFCNTRT()

    mask_results, resized_imgs = model.do_inference(imgs)
    
    status = extract_color_inference(resized_imgs, mask_results, color_fmt)
    if status: # if any result
        print('ClothesSeg [Extractor (inference)] - MHPFCN : Healthy')
        return True
    else:
        print('ClothesSeg [Extractor (inference)] - MHPFCN : Down')
        return False
    # color_filter = ColorFilter(color_fmt)

if __name__ == '__main__':
    # main('data/two_face.jpg')
    # main_folder('data/seg_net_data')
    # main_folder('data/person')
    # main_folder('data/video_extract')
    # main_video('data/testing/IP_Camera15_G61_it_Shatin_20190809150000_20190809160253_160737.mp4')
    # main('data/85_1_0_0.8878.jpg')
    main('data/two_face.jpg')
    # main('data/trump.jpg')
    # main('data/jacket.jpg')
    # main('data/colors.png')
    # main('data/colors2.png')
    # main('data/person/36.jpg')
