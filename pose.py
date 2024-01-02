import numpy as np
import math
import pickle
from scene import dataset_readers
from utils.camera_utils import cameraList_from_camInfos
from arguments import ModelParams


def find_iteration(N,step,stop):
    start = 0
    step = step
    iter = 0
    viewpoint_stack = None
    while True:
        
        if not viewpoint_stack:
            viewpoint_stack = [1] * N
            if step > stop - 1:
                answer = iter
                # print(answer)
                break
            if start == step:
                start = 0
                step += 1

            viewpoint_stack = viewpoint_stack[start::step]
            # print("step :" ,step) 
            start += 1  
        iter += 1
        viewpoint_stack.pop(0)

    return answer

def forlatex(N,step,stop):
    start = 0
    step = step
    iteration = 10000
    viewpoint_stack = None
    for i in range(0 , iteration):
        
        if not viewpoint_stack:
            viewpoint_stack = [1] * N
            if step > stop - 1:

                break
            if start == step:
                start = 0
                step += 1

            viewpoint_stack = viewpoint_stack[start::step]
            # print("step :" ,step) 
            start += 1  

        iter += 1
        view = viewpoint_stack.pop(0)
        print(view) # the purpose of this function



def make_T (R,cam_center):
     # Full code for calculating the translation matrix T given the world coordinates and rotation matrix R


    # Updated world coordinates X, Y, Z in a single array
    world_coords = cam_center

     
    
    T = np.linalg.inv(R).dot(world_coords)

    return -1 * T

 
def distatnce_based(args : ModelParams ,cam_info ,cam_centers , dist_range, change_to ): #return sub generated camera view , sample in this dist_range = (1,4) , and make to #1.331 view

    if change_to > dist_range[1]:
         print("Over_the_range")
         exit(-1)

    
    # cam_info = np.array(cam_info ,dtype="object")
    dist = np.linalg.norm(cam_centers, 2 , axis=0, keepdims=True) #유클리드와 맨허튼 비교하기

    index = np.where((dist >= dist_range[0]) & ( dist <= dist_range[1]))[1]
    # filtered_cam_info =cam_info[np.where((dist >= dist_range[0]) & ( dist <= dist_range[1]))[1]]
    # print(dist[0][np.where((dist >= dist_range[0]) & ( dist <= dist_range[1]))[1]])

    # print(filtered_cam_info)

    print("index num :", index.size )
    print("dist_range" , dist_range)
    print("change_to:" ,change_to )

    new_cams = []

    for cam_num in index:

        # print(cam_num)

        filtered_cam = cam_info[cam_num]

        # print(filtered_cam)

        uid = cam_info[cam_num].uid
        R = cam_info[cam_num].R
        T = None 
        new_FovY = cam_info[cam_num].FovY
        new_FovX = cam_info[cam_num].FovX
        

        #image

        image = filtered_cam.image
        image_path = None
        image_name = filtered_cam.image_name + '_NEW'
        width = filtered_cam.width
        height = filtered_cam.height

        # width, height = image.size

        # print("w, h" , w ,h)


        #x,y,z
         # centroid
        new_x = (change_to* cam_centers[0][cam_num]) / dist[0][cam_num]
        new_y = (change_to* cam_centers[1][cam_num]) / dist[0][cam_num]
        new_z = (change_to* cam_centers[2][cam_num]) / dist[0][cam_num]
        
        centers = np.array([new_x,new_y,new_z])

        # Transpose mat
        T = make_T(R, centers)

        #image

        # print("change_to" , change_to)
        new_width =  int( ( change_to * width )/dist[0][cam_num] )
        # print("width :", width , "-> new_width : ",new_width)
        new_height =  int( (change_to * height )/dist[0][cam_num] )

        new_size = (new_width, new_height)
        # Resize the image
        resized_img = image.resize(new_size)


        new_cam_info = dataset_readers.CameraInfo(uid=uid, R=R, T=T, FovY=new_FovY, FovX=new_FovX, image=resized_img,
                              image_path=image_path, image_name=image_name, width=new_width, height=new_height)

        new_cams.append(new_cam_info)
 
    # print(new_cams[-1])
    # print(filtered_cam_info[-1])


    # cameraList_from_camInfos(new_cams, 1.0 , args)
    

    return cameraList_from_camInfos(new_cams, 1.0 , args)



 
def distatnce_test(cam_centers , dist_range ): #return distacance ranged index
 
    print(dist_range)

    
    # cam_info = np.array(cam_info ,dtype="object")
    dist = np.linalg.norm(cam_centers, 2 , axis=0, keepdims=True) #유클리드와 맨허튼 비교하기
    print(dist.max())

    index = np.where((dist >= dist_range[0]) & ( dist <= dist_range[1]))[1]

    return index



def find_hard_negative(cam_info ,cam_centers ,avg_cam_center): #카메라 각 위치 , 카메라 위치 평균 값


        cam_vectors = cam_centers - avg_cam_center

    
        dist = np.linalg.norm(cam_vectors, 2 , axis=0, keepdims=True) #유클리드와 맨허튼 비교하기


        print(dist.mean()) #1 그룹



        Unit_vector = cam_vectors / dist

        # print("Unit_vector" , Unit_vector)

        # unit_dist = np.linalg.norm(Unit_vector, 2 , axis=0, keepdims=True) #유클리드와 맨허튼 비교하기

        # print("Unit_vector_dist" , unit_dist)

        dot_matrix = np.dot(Unit_vector.T , Unit_vector)
 
        
        # sum = np.sum(dot_matrix, axis=1, keepdims=False)  # Step 2
        # mean = np.mean(dot_matrix, axis=1, keepdims=False)  # Step 2
 
 

        mean = np.sum(dot_matrix, axis=0, keepdims=False)  # Step 2  

        percent = np.percentile(mean, [0, 10 ,25, 50, 75, 100])

        num = 1

        print(np.where( mean <= percent[num] ))
    


        outlighers = cam_info[ np.where( mean <= percent[num] )  ] #Low_d distance



        return outlighers 


def softmax(matrix):
    matrix = matrix 
    e_matrix = np.exp( (matrix - np.max(matrix, axis=0, keepdims=True)) )  # Step 1
    sum_e_matrix = np.sum(e_matrix, axis=0, keepdims=True)  # Step 2
    return e_matrix / sum_e_matrix

def min_max_normalize_matrix(matrix):
    min_vals = np.min(matrix, axis=0)
    max_vals = np.max(matrix, axis=0)
    normalized_matrix = (matrix - min_vals) / (max_vals - min_vals)
    return normalized_matrix

def Probablity_based(cam_centers ,avg_cam_center): #카메라 각 위치 , 카메라 위치 평균 값
      
        cam_vectors = cam_centers - avg_cam_center

        dist = np.linalg.norm(cam_vectors, 2 , axis=0, keepdims=True) #유클리드와 맨허튼 비교하기

        Unit_vector = cam_vectors / dist

        dot_matrix = np.dot(Unit_vector.T , Unit_vector)

        print(dot_matrix)

        # Prob = min_max_normalize_matrix(dot_matrix)
        
        # print( Prob )

        
        sum = np.sum(dot_matrix, axis=0, keepdims=False)  # Step 2


        print(sum)

        print(sum.shape)

        np.random.choice(sum.shape, 3, p=p)
        

def Angle_based(cam_centers ,avg_cam_center):
      
    cam_vectors = cam_centers - avg_cam_center

    dist = np.linalg.norm(cam_vectors, 2 , axis=0, keepdims=True) #유클리드와 맨허튼 비교하기

    Unit_vector = cam_vectors / dist

    dot_matrix = np.dot(Unit_vector.T , Unit_vector)
        

    min_angle = 40 #40도 이상카메라
    max_angle = 90 #90 도범위내 카메라를 고릅니다.

    if max_angle == 90:
        max_angle = 0

    min_angle =  min_angle * np.pi /180
    max_angle =  max_angle * np.pi /180

    
    print(math.cos(max_angle))
    print(math.cos(min_angle))
 

    # High_d = cam_info[ np.where(dist > dist.mean() )[1]  ] #high distance

    # Low_d = cam_info[ np.where(dist < dist.mean() )[1]  ] #Low_d distance    

     

    # filtered_array = dot_matrix[(dot_matrix > max_angle) & (dot_matrix < min_angle)]
    # print(np.where(dot_matrix > max_angle))
    print(dot_matrix[np.where(dot_matrix > min_angle)[1]])
                               
 
         

