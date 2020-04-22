import os
import math
import cv2 as cv
import numpy as np
import random
import traceback

def noisy(noise_typ,image):
    if noise_typ == "gauss":
        gauss = np.random.normal(0,1,image.size)
        gauss = gauss.reshape(image.shape[0],image.shape[1],image.shape[2]).astype('uint8')
        img_gauss = cv.add(image,gauss)
        return img_gauss
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.05
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
            for i in image.shape]
        out[coords] = 1
        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
        for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        noisy = np.random.poisson(image / 255.0 * 3) / 3 * 255
        return noisy
    elif noise_typ =="blur":
        blur_ammount = random.randrange(1, 7)
        # print(blur_ammount)
        if ((blur_ammount % 2) == 0):
            # print('blur minus 1')
            blur_ammount = blur_ammount - 1
        noise = cv.GaussianBlur(image, (blur_ammount, blur_ammount), cv.BORDER_DEFAULT)

        return noise

# [y1:y2, x1:x2], x1,y1 is top left x2, y2 is bottom right
def change_black_pixels_to_random_color(image):
    # if not three channel just ignore
    #print(image.shape)
    if (len(image.shape) == 3 and image.shape[2] == 3):
        # print('changing color')
        r = random.randrange(256)
        g = random.randrange(256)
        b = random.randrange(256)
        image[np.where((image<=[50, 50, 50]).all(axis=2))] = [r,g,b]
    return image

def get_corners(image):
    x1 = 0
    y1 = 0
    y2 = image.shape[0]
    x2 = image.shape[1]
    # skew side by this percentage
    x3 = x2 * random.randrange(21) * .01
    y3 = x3

    return x1, x2, x3, y1, y2, y3

def perspective_transform_overlay(direction, image, bg_image):
    x1, y1 = get_overlay_coords(image, bg_image)
    pts1, pts2 = get_perspective_transform_corners(direction, image)

    sub_bg_image = bg_image[y1:y1 + image.shape[0], x1: x1 + image.shape[1]]

    M = cv.getPerspectiveTransform(pts1,pts2)
    dst = cv.warpPerspective(image, M, (image.shape[1], image.shape[0]), sub_bg_image, borderMode=cv.BORDER_TRANSPARENT)

    bg_image[y1:y1 + image.shape[0], x1: x1 + image.shape[1]] = dst

    return bg_image, pts2, x1, y1

def get_perspective_transform_corners(direction, image):
    x1, x2, x3, y1, y2, y3 = get_corners(image)

    # top-left, top-right, bottom-left, bottom-right. [x, y] order
    pts1 = np.float32([[x1,y1],[x2,y1],[x1,y2],[x2,y2]])
    pts2 = np.float32([[x1,y1],[x2,y1],[x1,y2],[x2,y2]])

    if direction == 'left':
        pts2 = np.float32([[x3,y3],[x2,y1],[x3,y2-y3],[x2,y2]])
    elif direction == 'right':
        pts2 = np.float32([[x1,y1],[x2-x3,y3],[x1,y2],[x2-x3,y2-y3]])
    elif direction == 'top':
        pts2 = np.float32([[x3,y3],[x2-x3,y3],[x1,y2],[x2,y2]])
    elif direction == 'bottom':
        pts2 = np.float32([[x1,y1],[x2,y1],[x3,y2-y3],[x2-x3,y2-y3]])

    return pts1, pts2

def transparent_overlay(overlay_image, base_image):
    alpha = 0.8
    beta = (1 - alpha)
    x1, y1 = get_overlay_coords(overlay_image, base_image)

    blend_base = base_image[y1:y1 + overlay_image.shape[0], x1: x1 + overlay_image.shape[1]]
    blended = cv.addWeighted(overlay_image, alpha, blend_base, beta, 0.0)

    base_image[y1:y1 + overlay_image.shape[0], x1: x1 + overlay_image.shape[1]] = blended
    return base_image, x1, y1

def overlay(overlay_image, base_image):
    x1, y1 = get_overlay_coords(overlay_image, base_image)

    base_image[y1:y1 + overlay_image.shape[0], x1: x1 + overlay_image.shape[1]] = overlay_image
    return base_image, x1, y1

def get_overlay_coords(overlay_image, base_image):
    y_max =  base_image.shape[0] - overlay_image.shape[0]
    x_max = base_image.shape[1] - overlay_image.shape[1]
    #print('base image shape: ', base_image.shape)
    #print('overlay_image shape: ', overlay_image.shape)
    y1 = random.randrange(y_max) #if y_max > 0 else 0
    x1 = random.randrange(x_max) #if x_max > 0 else 0
    #print('x1', x1)
    #print('y1', y1)

    return x1, y1

def rotate_image(image, bg_image, angle):
    h, w = image.shape[:2]
    img_c = (w / 2, h / 2)

    rot = cv.getRotationMatrix2D(img_c, angle, 1)

    rad = math.radians(angle)
    sin = math.sin(rad)
    cos = math.cos(rad)
    b_w = int((h * abs(sin)) + (w * abs(cos)))
    b_h = int((h * abs(cos)) + (w * abs(sin)))

    rot[0, 2] += ((b_w / 2) - img_c[0])
    rot[1, 2] += ((b_h / 2) - img_c[1])

    #print('b_h: ', b_h)
    #print('b_w: ', b_w)
    #outImg = cv.warpAffine(image, rot, (b_w, b_h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_TRANSPARENT)
    #print('warped y: ', outImg.shape[0])
    #print('warped x: ', outImg.shape[1])

    x1, y1 = get_overlay_coords(np.zeros((b_h, b_w)), bg_image)

    sub_bg_image = bg_image[y1:y1 + b_h, x1: x1 + b_w]
    dst = cv.warpAffine(image, rot, (b_w, b_h), dst=sub_bg_image, flags=cv.INTER_LINEAR, borderMode=cv.BORDER_TRANSPARENT)

    bg_image[y1:y1 + b_h, x1: x1 + b_w] = dst
    return bg_image, x1, y1, b_w, b_h

# def combine_output_files(keep_path, add_path):

#Set up file pathing
dir = os.path.dirname(__file__)
bi_path = dir + '/sample_base_images/'
bi_list = os.listdir(bi_path)
ol_path = dir + '/sample_qr_images/'
ol_list = os.listdir(ol_path)

data_set_name = 'sample_composite_images'
out_path = dir + '/' + data_set_name +'/'

# random num ranges
bi_range = len(bi_list)
ol_range = len(ol_list)

# main loop
num_output_images = ol_range * 2
i = 0
print("ol range: ", ol_range)
print("num: ", num_output_images)
while(i < num_output_images):
    bi_rand = random.randrange(bi_range)
    bi_img = bi_list[bi_rand]
    
    if i < ol_range:
        index = i
        print("regular list i: ", i)
    else:
        index = random.randrange(ol_range)
        print("ol range: ", index)
        print("random list: ", index)

    ol_img = ol_list[index]

    i += 1
    # print('base image path: ', bi_path + bi_list[bi_rand])
    # print('overlay image path: ', ol_path + ol_list[ol_rand])
    base_image = cv.imread(bi_path + bi_img, cv.IMREAD_UNCHANGED).copy()
    base_image = cv.cvtColor(base_image, cv.COLOR_BGR2BGRA)
    try:
        overlay_image = cv.imread(ol_path + ol_img, cv.IMREAD_UNCHANGED).copy()
    except:
        print("path: ", ol_path)
        print("img: ", ol_img)
        print(traceback.format_exc())
        break

    image_out_path = out_path + ol_img + str(i) + ".png"

    # declare bounding box vars for writing to file
    y_max = 0
    y_min = 0
    x_max = 0
    x_min = 0

    # poor mans exception handling. at least it will write to a log
    no_errors_flag = True
    try:
        # check size of overlay vs base and set bounding box
        # bounding box is x_min, y_min, x_max, y_max
        while (base_image.shape[1] > 416 and base_image.shape[0] > 416):
            scale_percent = random.randrange(50, 90) # percent of original size
            width = int(base_image.shape[1] * scale_percent / 100)
            height = int(base_image.shape[0] * scale_percent / 100)
            dim = (width, height) 
            # print('base resize dims: ', dim)

            base_image = cv.resize(base_image, dim, interpolation = cv.INTER_AREA)

        resized = False
        while(overlay_image.shape[0] >= base_image.shape[0] or overlay_image.shape[1] >= base_image.shape[1]):
            #print('staring dim: ', overlay_image.shape)
            scale_percent = random.randrange(20, 70) # percent of original size
            width = int(overlay_image.shape[1] * scale_percent / 100)
            height = int(overlay_image.shape[0] * scale_percent / 100)
            dim = (width, height)
            # print('overlay resize dims: ', dim)

            resized = True
            overlay_image = cv.resize(overlay_image, dim, interpolation = cv.INTER_AREA)
            #print('resized dim: ', overlay_image.shape)

        if not resized:
            scale_percent = random.randrange(20, 101) # percent of original size
            width = int(overlay_image.shape[1] * scale_percent / 100)
            height = int(overlay_image.shape[0] * scale_percent / 100)
            dim = (width, height) 

            resized = True
            overlay_image = cv.resize(overlay_image, dim, interpolation = cv.INTER_AREA)
        
        # modify base image
        # rotate
        rotate = random.randrange(2)
        if rotate == 1:
            base_image = cv.rotate(base_image, random.randrange(3))

        # modify overlay image
        color_case = random.randrange(2)
        if color_case == 1:
            overlay_image = change_black_pixels_to_random_color(overlay_image)

        # convert image to bgra so that we can make perspective border transparent
        overlay_image = cv.cvtColor(overlay_image, cv.COLOR_BGR2BGRA)

        rotate_overlay = random.randrange(4)
        if rotate_overlay == 0:
            combined, x1, y1, w, h = rotate_image(overlay_image, base_image.copy(), random.randrange(45))
            y_min = y1
            x_min = x1
            y_max = y1 + h
            x_max = x1 + w
            overlay_image = combined[y_min: y_max, x_min: x_max]

        # combine
        perspective_case = random.randrange(8)
        # corners -> top-left, top-right, bottom-left, bottom-right. [x, y] order
        if perspective_case == 0:
            combined, corners, x, y = perspective_transform_overlay('left', overlay_image, base_image)
            y_min = y
            y_max = corners[3][1] + y
            x_min = corners[0][0] + x
            x_max = corners[3][0] + x
        elif perspective_case == 1:
            combined, corners, x, y  = perspective_transform_overlay('right', overlay_image, base_image)
            y_min = y
            y_max = corners[2][1] + y
            x_min = x
            x_max = corners[3][0] + x
        elif perspective_case == 2:
            combined, corners, x, y  = perspective_transform_overlay('top', overlay_image, base_image)
            y_min = corners[0][1] + y
            y_max = corners[3][1] + y
            x_min = x
            x_max = corners[3][0] + x
        elif perspective_case == 3:
            combined, corners, x, y  = perspective_transform_overlay('bottom', overlay_image, base_image)
            y_min = y
            y_max = corners[2][1] + y
            x_min = x
            x_max = corners[1][0] + x
        else:
            transparent = random.randrange(2)
            if transparent == 0:
                combined, x, y = transparent_overlay(overlay_image, base_image)
            else:
                combined, x, y = overlay(overlay_image, base_image)
            y_min  = y
            y_max = overlay_image.shape[0] + y
            x_min = x
            x_max = overlay_image.shape[1] + x
        # cv.rectangle(combined, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)

        # modify both
        noise = random.randrange(3)
        if noise == 0:
            # print('gauss')
            combined = noisy("gauss", combined)
        elif noise == 1:
            # print("s&p")
            combined = noisy("s&p", combined)

        # black and white/colormaps
        bw = random.randrange(3)
        #color_map = random.randrange(2)
        if bw == 1:
            # print('bw called')
            try:
                combined = cv.cvtColor(combined.astype('uint8'), cv.COLOR_BGR2GRAY)
            except Exception as e:
                print('bw failed: ', e)
        # elif color_map == 1:
        #     print('applying color map')
        #     combined = cv.cvtColor(combined.astype('uint8'), cv.COLOR_BGR2GRAY)
        #     combined = cv.applyColorMap(combined, random.randrange(12))

        #blur image
        blur = random.randrange(2)
        if blur == 1:
            # print('blur called')
            combined = noisy("blur", combined)

    except Exception as e:
        no_errors_flag = False
        log = open(dir + "/logfile.txt", "a+")
        log.write('base image path: ' + bi_path + str(bi_list[bi_rand]) + "\n")
        log.write('overlay image path: ' + ol_path + str(ol_list[index]) + "\n")
        log.write(str(e) + "\n")
        # log.write(traceback.format_exc())
        log.close
        
    #[display]
    # cv.imshow(ol_img, combined)

    if no_errors_flag:
        cv.imwrite(image_out_path, combined)
        data_set = open(dir + "/" + data_set_name + ".txt", "a+")
        data_set.write(image_out_path + " " + str(x_min) + "," + str(y_min) + "," + str(x_max) + "," + str(y_max)+ "," + str(80) + "\n")
        # image_path x_min, y_min, x_max, y_max, class_id  x_min, y_min ,..., class_id 
        data_set.close()

cv.waitKey(0)
cv.destroyAllWindows()