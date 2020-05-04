import numpy as np
import cv2
# from line_profiler import LineProfiler
import json
# profile = LineProfiler()

# Debugging
def var(var):
    def print_array(var):
        print(f"{type(var)}, shape: {var.shape}, range: [{var.min()},{var.max()}], dtype: {var.dtype}")

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

def read_json(obj, path):
    with open(path, 'r') as infile:
        data = json.load(infile)
        for key, value in data.items():
            setattr(obj, key, value)

class MHPFCNBase(object):
    """docstring for ResNetFCN"""
    def __init__(self, json):
        read_json(self, json)
        self.colors = [[0,0,0], [139,69,19], [222,184,135], [210,105,30], [255,255,0], [255,165,0], [0,255,0], 
                     [60,179,113], [107,142,35], [255,0,0], [245,222,179], [0,0,255], [0,255,255], [238,130,238], 
                     [128,0,128], [255,0,0], [255,0,255], [128,128,128], [128,128,128], [128,128,128]]


    def image_standardise(self, rgb_img: np.array): # input shape : NHWC 
        # ((inp / 255) - mean) / std
        mean = np.array([0.485, 0.456, 0.406]) # RGB 
        std = np.array([0.229, 0.224, 0.225])
        rgb_img = np.divide(rgb_img,255)
        rgb_img = np.divide(np.subtract(rgb_img, mean), std)  # (inp-mean)/std
        rgb_img = rgb_img.transpose((0,3,1,2)) # NHWC to NCHW
        return rgb_img # NCHW

    def utils_resize_image(self, img: np.ndarray, out_img: np.ndarray) -> (float, int, int):
        assert img.dtype == out_img.dtype, "Input images must have same dtype"
        left_pad = 0
        top_pad = 0
        h, w, _ = out_img.shape
        if img.shape[0] / img.shape[1] > h / w:
            resize_scale = h / img.shape[0]
            tmp_img = cv2.resize(img, (0, 0), fx=resize_scale, fy=resize_scale)
            left_pad = int((w - tmp_img.shape[1]) / 2)
            out_img[:, left_pad:left_pad + tmp_img.shape[1], :] = tmp_img
        else:
            resize_scale = w / img.shape[1]
            tmp_img = cv2.resize(img, (0, 0), fx=resize_scale, fy=resize_scale)
            top_pad = int((h - tmp_img.shape[0]) / 2)
            out_img[top_pad:top_pad + tmp_img.shape[0], :, :] = tmp_img
        return tmp_img, resize_scale, top_pad, left_pad, tmp_img.shape

    # @profile
    def _preprocess(self, inp: np.array, padding:bool = True): # input: NHWC BGR
        # resize to (360,640,3)  NCHW RGB
        # init
        ori_shapes = []
        pads = []
        resized_imgs = []
        infer_batch = np.zeros((len(inp), self.net_h , self.net_w, self.channels), dtype=np.uint8)  # channels = 3 for RGB  , NCHW

        for idx , img in enumerate(inp):
            if padding:
                resize_img , _ , top_pad, left_pad, scaled_shape = self.utils_resize_image(img, infer_batch[idx])
                pads.append((top_pad,left_pad))
                resized_imgs.append(resize_img)
            else:
                infer_batch[idx] = cv2.resize(img, (self.net_w, self.net_h), interpolation=cv2.INTER_LINEAR)  # INTER_NEAREST , INTER_LINEAR

        # normalize 
        # in: NHWC RGB 
        # out: NCHW RGB , mean = 0 , std = 1 
        infer_batch = self.image_standardise(infer_batch).astype(np.float32)
        return infer_batch, pads , resized_imgs

    
    def _inference(self, inp):
        pass
        ## return a list , since the NN output maybe more than one layer in different shape. 

    # @profile
    def _postprocess(self, masks:np.ndarray, pads:list, resized_imgs:list ): # inp:batch
        # threshold filter
        masks[masks < self.score_threshold] = -100  # can be zero / -ve value

        # resize to N,C,H,W (N,3,360,640) for 21 class
        n,c,_,_= masks.shape
        mask_batch = np.zeros((n,c,self.net_h,self.net_w),dtype=np.float32)
        for n,img in enumerate(masks):
            for c, class_ in enumerate(img):
                mask_batch[n][c] = self.mask_resize(class_, (self.net_w, self.net_h))
        # Argmax
        mask_batch = np.argmax(mask_batch, axis=1).astype(np.uint8) # axis = 1 , If NCHW  (batch, 21, 12, 20)
        mask_batch[np.isin(mask_batch, self.ignore_class)] = 0

        # del padding
        mask_batch = list(mask_batch)
        for idx,pad in enumerate(pads):
            top_pad, left_pad, h,w,_ = *pad , *resized_imgs[idx].shape
            if top_pad:
                mask_batch[idx] = mask_batch[idx][top_pad:top_pad + h,:] # top pad 
            else:
                mask_batch[idx] = mask_batch[idx][:,left_pad:left_pad + w] # left pad 

        return mask_batch

    def _postprocess_single(self, inp, ori_shape:tuple):
        inp[inp < self.score_threshold] = 0
        c,w,h = len(inp), *ori_shape
        mask = np.zeros((c,h,w),dtype=np.uint8)
        inp = inp.astype(np.uint8)

        for idx,class_ in enumerate(inp):
            mask[idx]=(self.mask_resize(class_, ori_shape))
        mask = np.argmax(mask, axis=0) # axis = 1 , If NCHW  (batch, 21, 12, 20)
        mask[np.isin(mask, self.ignore_class)] = 0
        return mask

    def mask_resize(self, mask, shape): # shape : (w,h)
        return cv2.resize(mask, shape, interpolation=cv2.INTER_LINEAR) # INTER_NEAREST,INTER_LINEAR,INTER_CUBIC

    # @profile
    def do_inference(self, inp:np.array): # suppose input is a batch, list of raw img array
        input_batch, pads, resized_imgs = self._preprocess(inp)
        multi_class_masks = self._inference(list(input_batch)) # NCHW (64,21,12,20)
        batch_layer_mask = self._postprocess(multi_class_masks, pads, resized_imgs) #NCHW (64,1,ori_H,ori_W)
        return batch_layer_mask, resized_imgs

    def mask(self, mask): # mask for single image
        r = mask.copy()
        g = mask.copy()
        b = mask.copy()
        for idx, color in enumerate(self.colors):
            layer_mask = (mask == idx)
            r[layer_mask] = color[0]
            g[layer_mask] = color[1]
            b[layer_mask] = color[2]
        mask = np.concatenate((r, g, b), axis=None).reshape(3, mask.shape[0], mask.shape[1])
        mask = mask.transpose((1, 2, 0))
        return mask


    def overlay(self, img, mask):  # overlay for single image
        layer = img.copy()
        for idx, color in enumerate(self.colors):
            if idx: # not background
                layer_mask = (mask == idx)
                layer[layer_mask] = color
        overlay = cv2.addWeighted(img,0.65,layer,0.45,0)
        return overlay

    def resize_img(self, inp):
        ori_shape = []
        std_inp = []
        for img in inp:
            ori_shape.append(img.shape)
            std_inp.append(cv2.resize(img,(self.net_w,self.net_h),interpolation = cv2.INTER_LINEAR))
        return std_inp, ori_shape

    def print_stats(self):
        profile.print_stats()

    def mask2img(self, mask_results, img, mask:bool=False, overlay:bool=False):
        if overlay | mask:
            for idx, mask_result in enumerate(mask_results):
                h,w,_ = img[idx].shape
                masked_result = self.mask_resize(mask_result,(w,h))

                if overlay:
                    overlay_ = self.overlay(img[idx],masked_result)   # (640, 960, 3) (640, 960)
                if mask:
                    mask_ = self.mask(masked_result)


                if overlay & mask:
                    mask_results[idx] = np.concatenate([mask_,overlay_],axis=1)
                elif overlay:
                    mask_results[idx] = overlay_
                elif mask:
                    mask_results[idx] = mask_
                else:
                    pass
        return


    def write_img(self, imgs, img_name):
        for idx, img in enumerate(imgs):
            img = img[:,:,[2,1,0]]  # rgb to bgr
            cv2.imwrite('{}_output.jpg'.format(img_name[idx].split('.')[0]), img)
        return