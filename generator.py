import cv2
import numpy as np
import sklearn
from scipy import ndimage

def generator(samples, batch_size=32, correction = 0.2, center_only = 0, mirror_image = 0):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            
            for batch_sample in batch_samples:
                #name = '../../../opt/carnd_p3/data_own/IMG/'+batch_sample[0].split('/')[-1]
                # name in model0.h5
                #name = '../../../opt/carnd_p3_own/data_own/IMG/'+batch_sample[0].split('\\')[-1]
                name = '../../../opt/carnd_p3_own/data_own_mountain/IMG/'+batch_sample[0].split('\\')[-1]
                #center_image = cv2.imread(name)
                center_image = ndimage.imread(name)
                center_angle = float(batch_sample[3])
                
                images.append(center_image)
                angles.append(center_angle)
                
                if (center_only == 0):                           
                    #left_name = '../../../opt/carnd_p3/data_own/IMG/'+batch_sample[1].split('/')[-1]
                    #left_name in model0.h5
                    #left_name = '../../../opt/carnd_p3_own/data_own/IMG/'+batch_sample[1].split('\\')[-1]
                    left_name = '../../../opt/carnd_p3_own/data_own_mountain/IMG/'+batch_sample[1].split('\\')[-1]
                    #left_image = cv2.imread(left_name)
                    left_image = ndimage.imread(left_name)
                    left_angle = float(batch_sample[3]) + correction

                    #right_name = '../../../opt/carnd_p3/data_own/IMG/'+batch_sample[2].split('/')[-1]
                    #right_name in model0.h5
                    #right_name = '../../../opt/carnd_p3_own/data_own/IMG/'+batch_sample[2].split('\\')[-1]
                    right_name = '../../../opt/carnd_p3_own/data_own_mountain/IMG/'+batch_sample[2].split('\\')[-1]
                    #right_image = cv2.imread(right_name)
                    right_image = ndimage.imread(right_name)
                    right_angle = float(batch_sample[3]) - correction
                
                    #images.extend(left_image, right_image)
                    images.append(left_image)
                    images.append(right_image)
                    
                    #angles.extend(left_angle, right_angle)
                    angles.append(left_angle)
                    angles.append(right_angle)
                    
                if (mirror_image == 1):
                    center_image_mirror = np.fliplr(center_image)
                    center_angle_mirror = - center_angle
                    left_image_mirror = np.fliplr(left_image)
                    left_angle_mirror = - left_angle
                    right_image_mirror = np.fliplr(right_image)
                    right_angle_mirror = - right_angle
                    
                    images.append(center_image_mirror)
                    images.append(left_image_mirror)
                    images.append(right_image_mirror)
                    
                    angles.append(center_angle_mirror)
                    angles.append(left_angle_mirror)
                    angles.append(right_angle_mirror)
                                     
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
