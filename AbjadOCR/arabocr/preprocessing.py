import numpy as np
from PIL import Image
from skimage.morphology import skeletonize
from scipy.misc import imsave, imread, imresize
from arabocr.utils import AbjadUtils
import matplotlib.pyplot as plt
import cv2
import glob
import base64
import re
import os

class AbjadImage(object):

    @classmethod
    def convert_from_Flask(cls, img_data):
        img_str = re.search(b'base64,(.*)',img_data).group(1) 
        with open('converted_image.png','wb') as output: 
            output.write(base64.b64decode(img_str))

        output = imread('converted_image.png', mode = 'L')
        return output

    @classmethod
    def binarize(cls, img_data):
        img_data = cv2.threshold(img_data, 10, 255, cv2.THRESH_BINARY)[1]

        """
        height, width = img_data.shape
        i = 0

        while i < height:
            j = 0
            while j < width:
                if(img_data[i,j] != 0 and img_data[i,j] < 127):
                    img_data[i,j] = 0
                elif img_data[i,j] != 255 and img_data[i,j] >= 127:
                    img_data[i,j] = 255
                j += 1
            i += 1
        """
        return img_data

    @classmethod
    def clamp01(cls, img_data):
        return np.multiply(img_data, 1 / 255)

    @classmethod
    def invert_binary(cls, img_data):
        return np.invert(img_data)

    @classmethod
    def reverse_gray_rgb(cls, img_data):
        return (255 - img_data)

    @classmethod
    def deskew_image(cls, img_data):
        coords = np.column_stack(np.where(img_data > 0))
        angle = cv2.minAreaRect(coords)[-1]
 
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        (h, w) = img_data.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1)
        rotated = cv2.warpAffine(img_data, M, (w, h))

        return rotated

    @classmethod
    def remove_dots_diacritics(cls, img_data, minContourArea = 700):
        _, contours, hierarchy = cv2.findContours(img_data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        dots_diacritics_mask = np.zeros(img_data.shape[:2], dtype="uint8")

        for c in contours:
            if type(c) == np.ndarray:
                if(cv2.contourArea(c) < minContourArea):
                    cv2.drawContours(dots_diacritics_mask, [c], -1, 255, -1)
        
        img_data = (img_data - dots_diacritics_mask)

        return img_data, dots_diacritics_mask

    @classmethod
    def x_cord_contour(cls, contours):
        if cv2.contourArea(contours) > 10:
            M = cv2.moments(contours)
            return(int(M['m10']/M['m00'])) 

    @classmethod
    def sortkey_x_right_most(cls, contours):
        return contours[0,0,0]

    @classmethod
    def getArg_x_left_most(cls, cnt):
        return cnt[:, :, 0].argmin()

    @classmethod
    def getArg_x_right_most(cls, cnt):
        return cnt[:, :, 0].argmax()

    @classmethod
    def centerize_img(cls, img_data, img_width = 32, img_height = 32):
        #Centerize Image data
        _, contours, hierarchy = cv2.findContours(img_data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return print('')

    @classmethod
    def seperate_connected_components(cls, img_data, dots_diacritics_mask = [], sub_word_padder = 137, offset = 170, widht_increase = 30, x_only = True):
        if x_only == False: return 'Not yet implemented'

        img_height, img_width = img_data.shape
        mask = np.zeros((img_height, img_width), dtype="uint8")

        _, contours, hierarchy = cv2.findContours(img_data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = sorted(contours, key = AbjadImage.sortkey_x_right_most, reverse = True)

        consider_dd_mask = dots_diacritics_mask != []
        mask_dd = np.zeros((img_height, img_width), dtype="uint8")

        plt.imshow(dots_diacritics_mask, cmap='gray')
        plt.show()

        dots_diacritics_mask = np.asarray(dots_diacritics_mask)
        _, dd_contours, _ = cv2.findContours(dots_diacritics_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #AbjadUtils.DebugNumpyArray(img_data)
        #AbjadUtils.DebugNumpyArray(dots_diacritics_mask)
        #AbjadUtils.DebugNumpyArray(dd_contours)
        dd_centers = np.ones((len(dd_contours), 2), dtype="uint8") * (-1) # 0 => center_value | 1 => idx_connectivity
        j = 0
        #Prepare all DDs centers
        while j < len(dd_contours):
            curr_dd_cnt = dd_contours[j]
            dd_leftArgs = AbjadImage.getArg_x_left_most(curr_dd_cnt)
            dd_rightArgs = AbjadImage.getArg_x_right_most(curr_dd_cnt)
            center_x_dd = curr_dd_cnt[dd_leftArgs][0, 0] + np.abs(curr_dd_cnt[dd_rightArgs][0, 0] - curr_dd_cnt[dd_leftArgs][0, 0]) // 2
            dd_centers[j][0] = center_x_dd 
            j += 1

        i = 0
        prev_left_most = img_width

        while i < len(contours):
            cnt = contours[i]
            curr_right_most = 0
            curr_left_most = img_width

            for idx, item in enumerate(cnt):
                curr_right_most = max(curr_right_most, item[0][0])
                curr_left_most = min(curr_left_most, item[0][0])

            trans_x = img_width - (curr_right_most + offset)

            for idx, item in enumerate(cnt):
                #Try to keep holes...
                #... 

                if i == 0:
                    item[0][0] += trans_x
                else:
                    item[0][0] += (prev_left_most - curr_right_most) - sub_word_padder
                    

            #Try to get dots & diacritics if consider_dd_mask == True 
            #...
            idx_cnt_closest_to_dd = -1 # To compare with 'i'
            idx_concerned_dd = -1 # Being a 'j'

            if(consider_dd_mask): #Imbreakable, check all the DDs
                j = 0
                while j < len(dd_centers):
                    curr_dd_center = dd_centers[j][0]
                    # In case this DD found a surronding sub-word to it, then it belongs to it
                    if dd_centers[j][1] == -1 and curr_dd_center >= curr_left_most - widht_increase and curr_dd_center <= curr_right_most + widht_increase:
                        print('DD_CENTER', curr_dd_center, 'LEFT: ', curr_left_most, 'RIGHT: ', curr_right_most)
                        dd_centers[j][1] = i
                        dd_cnt = dd_contours[j]

                        print('WE HERE')

                        for idx, item in enumerate(dd_cnt):
                            if i == 0:
                                item[0][0] += trans_x
                            else:
                                item[0][0] += (prev_left_most - curr_right_most) - sub_word_padder

                        cv2.drawContours(mask_dd, [dd_cnt], -1, 255, -1)

                    j += 1

            prev_left_most = img_width

            for idx, item in enumerate(cnt):
                prev_left_most = min(prev_left_most, item[0][0])

            cv2.drawContours(mask, [cnt], -1, 255, -1)
            i += 1
        
        mask = mask + mask_dd
        plt.imshow(mask_dd, cmap='gray')
        plt.show()

        return mask

    @classmethod
    def word_baseline(cls, img_data):

        def line_seg_T(x):
            return x if x > 70 else 0

        image_data_rgb = cv2.cvtColor(img_data, cv2.COLOR_GRAY2RGB)
        
        height, width = img_data.shape

        img_cols_sum = np.sum(img_data, axis=1).tolist()
        
        line_seg_T = np.vectorize(line_seg_T) 
        result_cols_array = line_seg_T(img_cols_sum) 

        i = 0
        x_baseline = 0
        basline_desnity = 0

        while i < len(result_cols_array) - height ** 0.75:
  
            if(result_cols_array[i] > basline_desnity):
              x_baseline = i
              basline_desnity = result_cols_array[i]
        
            i += 1

        i = x_baseline
        x_descline = i
        localMinValue = 2 ** 10000 #INT_INFINITY

        while i < len(result_cols_array):
  
            if(result_cols_array[i] < localMinValue):
              x_descline = i
              localMinValue = result_cols_array[i]
      
            i += 1

        cv2.line(image_data_rgb, (0, x_baseline), (width, x_baseline), (255, 255, 70), 2)
        cv2.line(img_data, (0, x_baseline), (width, x_baseline), (255, 255, 255), 2)
        
        cv2.line(image_data_rgb, (0, x_descline), (width, x_descline), (255, 0, 30), 2)
        cv2.line(img_data, (0, x_descline), (width, x_descline), (255, 255, 255), 2)

        return img_data, image_data_rgb, x_baseline, x_descline

    @classmethod
    def v_line_subwords_seperator(cls, img_data):
        img_data_rgb = cv2.cvtColor(img_data, cv2.COLOR_GRAY2RGB)
        
        img_row_sum = np.sum(img_data, axis=0).tolist()
        height, _ = img_data.shape

        def line_seg_T(x):
            return x if x > 70 else 0

        line_seg_T = np.vectorize(line_seg_T) 
        result_cols_array = line_seg_T(img_row_sum) 

        i = 1

        while i < len(result_cols_array) - 1:
  
            if(result_cols_array[i] == 0 and (result_cols_array[i - 1] != 0)):
                cv2.line(img_data_rgb, (i, 0), (i, height), (0, 170, 255), 1)
                cv2.line(img_data, (i , 0), (i , height), (255, 255, 255), 1)

            i += 1

        return img_data, img_data_rgb
        
    @classmethod
    def get_skeleton(cls, img_data):
        img_data = np.multiply(img_data, 1/255)
        img_data = skeletonize(img_data) * 255
        img_data = np.array(img_data)
        img_data = img_data.astype(np.uint8)
        return img_data

    @classmethod
    def psp_segmentation(cls, img_data, min_char_width = 100, img_width = 32, img_height = 32, line_thickness = 1, max_considerable_height = 2000, min_baseline_desc_distance = 700):
        img_data_rgb = cv2.cvtColor(img_data, cv2.COLOR_GRAY2RGB)

        img_row_sum_img = np.sum(img_data, axis=0).tolist()
        height, width = img_data.shape

        plt.plot(img_row_sum_img)
        plt.show()
        
        x_baseline = -1
        x_descline = -1
        fine_dist = False

        _, _, x_baseline, x_descline = AbjadImage.word_baseline(img_data.copy())
        #print('x_baseline ', x_baseline)
        #print('x_descline ', x_descline)
        dist = np.abs(x_descline - x_baseline) > min_baseline_desc_distance

        i = 1
        prev_index = 0
        x_start = 0
        prev_value = img_row_sum_img[height]
        to_predict_chars = []
        going_down = True
        just_did = False

        while i < len(img_row_sum_img):
            if(going_down == True):
                if(img_row_sum_img[i] > prev_value and just_did == False):
                    going_down = False
                    just_did = True
                    x_start = i

            elif (img_row_sum_img[i] < prev_value and img_row_sum_img[i] < max_considerable_height):
                if(just_did == True):
                    just_did = False
                elif i - prev_index >= min_char_width:
                    pt = i
                    j = pt - 14
                    if j < 0: j = 0

                    while j < pt + 14:
                        j += 1
                        curr_idx = j - 1
                        if(img_row_sum_img[curr_idx] < img_row_sum_img[pt]):
                            pt = curr_idx

                    if dist == True and img_data[x_descline - (x_descline - x_baseline) // 3 : x_descline, pt].any() != 0:
                        #print('FOUND IT, pt == ', pt, ' Value(approx) = ', img_data[x_descline, pt])
                        cv2.line(img_data_rgb, (pt, x_descline), (pt, x_descline + 3), (10, 200, 32), line_thickness)

                    else:
                        cv2.line(img_data, (pt, 0), (pt, height), (10, 200, 32), line_thickness)
                        cv2.line(img_data_rgb,(pt,0),(pt,height),(10,200,32), line_thickness)
                  
                        #print('Made point in ', pt, ' for ', img_row_sum_img[pt])

                        curr_img = img_data[:, prev_index + line_thickness : pt - line_thickness]

                        #curr_img = AbjadImage.centerize_img(curr_img, img_width, img_height)

                        curr_img = imresize(curr_img, (img_width, img_height))
                        curr_img = AbjadImage.binarize(curr_img)

                        if curr_img.any() != 0:
                            #curr_img = AbjadImage.reverse_gray_rgb(curr_img)
                            #print('max: ', np.max(curr_img))
                            plt.imshow(curr_img, cmap='gray')
                            plt.show()

                            curr_img = curr_img.reshape(1, img_width, img_height, 1)
                            to_predict_chars = np.append(to_predict_chars, curr_img)
                        
                        prev_index = pt
                        going_down = True

            prev_value = img_row_sum_img[i]

            i += 1

        if prev_index != 0:
            cv2.line(img_data, (width - 1, 0), (width - 1, height), (10, 200, 32), line_thickness)
            cv2.line(img_data_rgb, (width - 1, 0), (width - 1, height), (10, 200, 32), line_thickness)
                  
            #print('Made point in ', width - 1, ' for ', img_row_sum_img[width - 1], 'FINAL VERTICAL LINE')

            curr_img = img_data[:, prev_index + line_thickness : width - (1 + line_thickness)]
            curr_img = imresize(curr_img, (img_width, img_height))
            curr_img = AbjadImage.binarize(curr_img)

            if curr_img.any() != 0:
                #curr_img = AbjadImage.reverse_gray_rgb(curr_img)
                #print('max: ', np.max(curr_img))
                plt.imshow(curr_img, cmap='gray')
                plt.show()

                curr_img = curr_img.reshape(1, img_width, img_height, 1)
                #curr_img = AbjadImage.reverse_gray_rgb(curr_img)
                to_predict_chars = np.append(to_predict_chars, curr_img)


        to_predict_chars = np.array(to_predict_chars)
        to_predict_chars = to_predict_chars.reshape((-1, img_width * img_height)).astype('float32')
        to_predict_chars = to_predict_chars.reshape((-1, img_width, img_height, 1)).astype('float32')

        #to_predict_chars = reversed(to_predict_chars)

        return img_data, img_data_rgb, to_predict_chars