import cv2
# from skimage.color import rgb2hsv, hsv2rgb
# from .color_converter import rgb2hsv, hsv2rgb
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from sklearn.cluster import KMeans , DBSCAN
import numpy as np 
import json

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

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

class ColorFilter(object):
    def __init__(self, color_fmt):
        self.read_color_fmt(color_fmt)
        self.color_fmt = color_fmt

    def read_color_fmt(self, class_fmt):
        with open('clothes_seg/config/config_color.json', 'r') as infile:
            data = json.load(infile)
            for key, value in data[class_fmt].items():
                setattr(self, key, value)

    # def color_filtering_NHWC(self, images, colors): # Inference   image (N, H, W, C) , same size
    #     color_percent = np.zeros((len(colors),len(images)))
    #     for idx,color in enumerate(colors):
    #         color_pixel_array = np.where(np.all([np.logical_and(images[:,:,:,0]>=color[0]-self.color_range[0], images[:,:,:,0]<=color[0]+self.color_range[0]), # ch1 : H / R
    #                                              np.logical_and(images[:,:,:,1]>=color[1]-self.color_range[1], images[:,:,:,1]<=color[1]+self.color_range[1]), # ch2 : S / G
    #                                              np.logical_and(images[:,:,:,2]>=color[2]-self.color_range[2], images[:,:,:,2]<=color[2]+self.color_range[2])], # ch3 : V / B
    #                                              axis=0),
    #                                     1,0)  # if True 1 else 0 
    #         y = [x.shape[0]*x.shape[1] for x in images]  # H*W = size/chl , x: CHW format
            
    #         color_percent[idx] = np.sum(np.sum(color_pixel_array,axis=1),axis=1)/y*100 # count of color pixel / total pixel * 100%

    #     # print("Inference \n",color_percent)
    #     return color_percent.T

    # def color_filtering_HWC(self, masked_array: list, colors): # Inference   image [<img1>,<img2>,....] (H, W, C)
    #     print(masked_array[0].shape)
    #     print(masked_array[0][1][1])
    #     color_percent = np.zeros((len(colors),len(masked_array)))
    #     for x2,resized_img in enumerate(masked_array):
    #         print(resized_img.shape)
    #         resized_img = rgb_to_hsv(resized_img)
    #         print(resized_img[1][1])
    #         for idx,color in enumerate(colors):
    #             color_pixel_array = np.where(np.all([np.logical_and(resized_img[:,:,0]>=color[0]-self.color_range[0], resized_img[:,:,0]<=color[0]+self.color_range[0]), # ch1 : H / R
    #                                                  np.logical_and(resized_img[:,:,1]>=color[1]-self.color_range[1], resized_img[:,:,1]<=color[1]+self.color_range[1]), # ch2 : S / G
    #                                                  np.logical_and(resized_img[:,:,2]>=color[2]-self.color_range[2], resized_img[:,:,2]<=color[2]+self.color_range[2])], # ch3 : V / B
    #                                                  axis=0),
    #                                         1,0)  # if True 1 else 0 
    #             y = resized_img.shape[0]*resized_img.shape[1] # H*W = size/chl , x: CHW format
    #             color_percent[idx][x2] = np.sum(np.sum(color_pixel_array,axis=0),axis=0)/y*100 # count of color pixel / total pixel * 100%

    #     # print(colors)
    #     print("Inference \n",color_percent)
    #     return color_percent.T

    def color_filtering(self, masked_arrays: list, colors): # Inference   image [<img1>,<img2>,....] (H* W, C)
        color_percent = np.zeros((len(colors),len(masked_arrays)))
        for x2,array in enumerate(masked_arrays):
            for idx,color in enumerate(colors):
                color_pixel_array = np.where(np.all([np.logical_and(array[:,0]>=color[0]-self.color_range[0], array[:,0]<=color[0]+self.color_range[0]), # ch1 : H / R
                                                     np.logical_and(array[:,1]>=color[1]-self.color_range[1], array[:,1]<=color[1]+self.color_range[1]), # ch2 : S / G
                                                     np.logical_and(array[:,2]>=color[2]-self.color_range[2], array[:,2]<=color[2]+self.color_range[2])], # ch3 : V / B
                                                     axis=0),
                                            1,0)  # if True 1 else 0 
                y = array.shape[0]*array.shape[1] # H*W = size/chl , x: CHW format
                color_percent[idx][x2] = np.sum(np.sum(color_pixel_array,axis=0),axis=0)/(y/3)*100 # count of color pixel / total pixel(per Ch) * 100%
        # print("Inference \n",color_percent)
        return color_percent.T.astype(np.uint8)

    def color_match(self, masked_arrays:list, colors: np.array , target_percent: np.array): 
        # masked_array : list of np array , masked_pixel, clothes crop

        # image and pixel color matching , color_threshold: percent_std_dev, color_range: RGB +- 15

        color_percent = self.color_filtering(masked_arrays, colors)
        bl = np.all(np.logical_and(color_percent >= target_percent - self.color_threshold , color_percent <= target_percent + self.color_threshold),
                    axis=1)
        return color_percent, bl # match or not

    def dominantColors_Kmeans(self, image, plot, k=5):  # k = no_class
        #reshaping to a list of pixels
        img = image.reshape((image.shape[0] * image.shape[1], 3))
        
        #using k-means to cluster pixels
        kmeans = KMeans(n_clusters = k)
        idx = kmeans.fit_predict(img)

        #the cluster centers are our dominant colors.
        colors = kmeans.cluster_centers_
        colors.astype(int) if self.fmt == 'rgb' else colors
        #save labels
        labels = kmeans.labels_


        percent = np.array([ np.divide(sum(idx==class_)*100,len(idx)) for class_ in range(k)],dtype=np.int8)
        
        if plot:
            self.plot(img, idx)
        #returning after converting to integer from float
        return colors, percent

    # Caution: size of image < 250*250,3 as possible
    def dominantColors_DBSCAN(self, array:np.array, plot:bool):
        #reshaping to a list of pixels
        if len(array.shape)>2:
            array = array.reshape((array.shape[0] * array.shape[1], 3))
        dbs = DBSCAN(self.eps ,self.min_sample)
        idx = dbs.fit_predict(array)
        no_class = len(set(idx))

        if self.fmt == 'hsv':
            colors = np.array([np.average(array[idx==class_],axis = 0) for class_ in range(no_class)])
            if -1 in idx:                
                colors = colors[:-1]
                no_class -= 1

        elif self.fmt == 'rgb':
            colors = np.array([np.average(array[idx==class_],axis = 0) for class_ in range(no_class)],dtype=np.uint8)
        else:
            raise('Invalid color format')


        # remove the color < 5%
        percent = np.array([ np.divide(sum(idx==class_)*100,len(idx)) for class_ in range(no_class)],dtype=np.int8)
        order=(-percent).argsort()
        percent = percent[percent>=5]
        colors = colors[:len(percent)]

        if plot:
            # self.plot(array, idx)
            self.canvas_plot(array, idx)

        return colors, percent

    def canvas_plot(self, img, idx):
        fig = Figure()
        canvas = FigureCanvas(fig)
        fig.set_size_inches(20, 15)
        ax = fig.add_subplot(111, projection='3d')

        xs = img[:,0]
        ys = img[:,1]
        zs = img[:,2]
        ax.scatter(xs, ys, zs, marker='o', c=idx)

        ax.set_xlabel('{} Label'.format(self.label[0]))
        ax.set_ylabel('{} Label'.format(self.label[1]))
        ax.set_zlabel('{} Label'.format(self.label[2]))

        canvas.draw()
        s,(w,h) = canvas.print_to_buffer()
        print(w,h)
        img = np.fromstring(canvas.tostring_rgb(),dtype='uint8').reshape(h,w,3)
        img = img[:,:,[2,1,0]]
        cv2.imshow("",img)
        cv2.waitKey()        
        # plt.show()

    def plot(self, img, idx):
        fig = plt.figure()
        fig.set_size_inches(20, 15)
        ax = fig.add_subplot(111, projection='3d')

        xs = img[:,0]
        ys = img[:,1]
        zs = img[:,2]
        ax.scatter(xs, ys, zs, marker='o', c=idx)

        ax.set_xlabel('{} Label'.format(self.label[0]))
        ax.set_ylabel('{} Label'.format(self.label[1]))
        ax.set_zlabel('{} Label'.format(self.label[2]))
        plt.show()

    # def rescale(self, img):
    #     return rgb_to_hsv(img/self.scale) if self.fmt == 'hsv' else img

    def rgb_to_hsv(self,imgs:list):
        for idx, img in enumerate(imgs):
            if len(img.shape)==2:
                img= np.tile(img,[1,1,1])
            imgs[idx] = rgb_to_hsv(img).squeeze().astype(np.float16)
        return

    # def resize(self, image,size=224):  # rescale and cvt to hsv
    #     scale = image.shape[1]/image.shape[0]  # W/H
    #     image = cv2.resize(image,(int(size*scale),size), interpolation = cv2.INTER_LINEAR) #(resize shape = (w,h) )
    #     image = image.astype(np.uint8)
    #     return self.rescale(image)
        # return inp

    # convert color_list into RGB 
    def recolor(self, color_list):
        # conv = color_converter()
        rgb = []
        # var(color_list)
        for idx,color in enumerate(color_list):
            rgb.append(hsv_to_rgb(color).astype(np.uint8))

        return np.array(rgb).squeeze()

    def write_color(self, color_list,percent,json_path):
        file = {'color_list': color_list.astype(np.float16).tolist(),
                'percent': percent.tolist()}
        with open(json_path,'w') as outfile:
            json.dump(file,outfile)

    def read_color(self, json_path):
        with open(json_path,'r') as infile:
            file = json.load(infile)
        color_list,percent = file.values()
        color_list = np.array(color_list,dtype=np.float16)
        percent = np.array(percent)
        return color_list,percent
         
    def mask_to_array(self, imgs:list ,masks:list): # [<HWC>,<HWC>,...]
        pixel_list = []
        for img,mask in zip(imgs,masks):
            # print(img.shape,mask.shape)
            *_,c = img.shape
            mask = np.where(mask, True, False)
            z = (img.T * mask.T ).T
            z = z[~np.all(z == 0, axis=2)]

            pixel_list.append(z)
        return pixel_list

def extract_color_clustering(img:np.ndarray, mask_results:np.array ,color_fmt:str, plot:bool): #input: BGR img 
    # input img
    color_filter = ColorFilter(color_fmt)
    
    # masked_arrays=[imgs[0].reshape(int(imgs[0].size/3),3)]
    masked_arrays  = color_filter.mask_to_array([img],[mask_results])   # return pixel result , [(h*w , c)]

    # color_fmt
    if color_filter.fmt == 'hsv':
        color_filter.rgb_to_hsv(masked_arrays)
    # var(masked_arrays)

    # sampleing if population > 10000
    for idx, masked_array in enumerate(masked_arrays):
        if len(masked_array)>=10000:
            masked_arrays[idx] = masked_array[np.random.choice(masked_array.shape[0],size=10000,replace = False),:]
    # Extraction, Clustering , DBSCAN / K-Mean  ( For Debug )
    # color_list, percent = color_filter.dominantColors_Kmeans(masked_arrays[0],plot)
    color_list, percent = color_filter.dominantColors_DBSCAN(masked_arrays[0], plot)

    json_file = "clothes_seg/config/color.json"
    color_filter.write_color(color_list,percent,json_file)

    # Print Color ( For Debug )
    if color_fmt == 'hsv':
        print('========= HSV in RGB ========: \n',color_filter.recolor(color_list),'\n',percent,'%')
    else: 
        print('========= RGB ========: \n',color_list,'\n',percent,'%')

    assert len(color_list) == len(percent)

    if len(color_list) == 0:
        return False
    elif sum(percent)<20:
        return False
    else:
        return True

def extract_color_inference(imgs:list, mask_results:np.array, color_fmt): #input: BGR img 
    # input img
    color_filter = ColorFilter(color_fmt)

    json_file = "clothes_seg/config/color.json"
    color_list, percent = color_filter.read_color(json_file)

    # masked_arrays=[imgs[0].reshape(int(imgs[0].size/3),3),imgs[0].reshape(int(imgs[0].size/3),3),imgs[0].reshape(int(imgs[0].size/3),3)]
    masked_arrays  = color_filter.mask_to_array(imgs,mask_results)   # return pixel result , [(h*w , c)]
    # var(masked_arrays)

    # color_fmt
    if color_filter.fmt == 'hsv':
        color_filter.rgb_to_hsv(masked_arrays)

    # Matching
    color_percent, bl = color_filter.color_match(masked_arrays, color_list, percent)
    # bl = color_filter.color_match(img, color_list, percent)

    # print("Target % \n",percent)
    print(f'colors:\n{color_list}\npercent: {color_percent} (target: {percent})\nbool: {bl}')
    return bl

if __name__ == '__main__':
    file_name = 'data/trump.jpg'
    img = cv2.imread(file_name)
    rgb_img = img[:,:,[2,1,0]]
    extract_color(rgb_img, color_fmt = 'rgb', plot = True)

    ## highly recommend usd HSV instead of RGB. 
    # KMEAN (RGB)
    # ================== KMEAN (RGB) ==================
    # [[253.48762542 253.34876254 253.28749164]
    #  [ 72.14611212  70.09077758  76.21844485]
    #  [ 26.63396861  26.00840807  26.242713  ]
    #  [ 49.23721468  48.28720023  51.38832707]
    #  [107.79509202 101.9006135  104.87116564]] 

    # Percent of image - Clustering
    #  [45                 17                  10                  21                 5] 
    # Percent of image - Inference
    #  [44.24539877300614, 18.877300613496935, 15.380368098159508, 27.60122699386503, 4.920245398773006]


    # ===============================================================================================

    # KMEAN (HSV)
    # ================== KMEAN (HSV) ==================
    # HSV Color 
    # [[0.67881244 0.07842921 0.25975267]
    #  [0.00600736 0.00213144 0.99450272]
    #  [0.07255089 0.09488661 0.18018258]
    #  [0.60250057 0.51682981 0.28225705]
    #  [0.9309358  0.08319914 0.29192116]]
    # In RGB Color Form
    # [[ 61  61  66]
    #  [253 253 253]
    #  [ 45  43  41]
    #  [ 34  49  71]
    #  [ 74  68  70]]

    # Percent of image - Clustering
    #  [16                  45                  17                  4                   16] 
    # Percent of image - Inference
    #  [16.87730061349693,  44.858895705521476, 16.05521472392638,  4.288343558282208,  21.11042944785276]

    # ===============================================================================================

    # DBSCAN (RGB)
    # ================== DBSCAN (RGB) ==================
    # In RGB Color Form
    # [[253 253 253]
    #  [ 57  54  56]
    #  [ 41  55  83]
    #  [  0   0   0]] 

    # Percent of image - Clustering
    # [45                   49                  3                   0] 
    # Percent of image - Inference
    # [44.325153374233125,  28.380368098159508, 3.717791411042945,  2.4417177914110426]

    # ===============================================================================================

    # DBSCAN (HSV)
    # ================== DBSCAN (HSV) ==================
    # HSV Color 
    # [[3.18887352e-05 2.15804919e-05 9.97063976e-01]
    #  [8.17258231e-01 7.84095971e-02 2.68741079e-01]
    #  [6.11918612e-01 5.03692210e-01 3.28533497e-01]
    #  [6.62706294e-02 8.08771816e-02 1.83760526e-01]
    #  [           nan            nan            nan]]
    # In RGB Color Form
    # [[254 254 254]
    #  [ 68  63  68]
    #  [ 41  55  83]
    #  [ 46  44  43]
    #  [  1   0 101]]

    # Percent of image - Clustering
    # [44                  29                 3                  15                  0 ] True
    # Percent of image - Inference
    # [44.852760736196316, 28.14723926380368, 4.177914110429448, 15.969325153374234, 0.0]

    # ===============================================================================================



    # HSV may cluster out more accurate in differenet color in 3D graph.