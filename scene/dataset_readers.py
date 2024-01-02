#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import math 
import time
import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import pose


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    
    
    def change_train_cam(self,cameras):
        self.train_cameras = cameras




def find_hard_negative(cam_centers ,avg_cam_center): #카메라 각 위치 , 카메라 위치 평균 값


        cam_vectors = cam_centers - avg_cam_center

        
        print("cam_vectors.shape" , cam_vectors.shape)
    
        dist = np.linalg.norm(cam_vectors, 2 , axis=0, keepdims=True) #유클리드와 맨허튼 비교하기


        print(dist.mean()) #1 그룹

         

        Group_1 = cam_vectors[ :, np.where(dist > dist.mean() )[1]  ] #high distance

        Group_2 = cam_vectors[ :, np.where(dist <= dist.mean() )[1]  ]

        print(Group_1.shape)
        print(Group_2.shape)
         
        

         
        # cam_x = cam_vectors[0][ np.where(dist > dist.mean)]

        # print(cam_x.shape)


         

        # print(  cam_vectors[ dist >   ].shape )

        # import pickle
        # with open('./dist.pickle', 'wb') as f:
        #     pickle.dump(dist, f)

        # print("dist", dist)

        Unit_vector = cam_vectors / dist

        # print("Unit_vector" , Unit_vector)

        unit_dist = np.linalg.norm(Unit_vector, 2 , axis=0, keepdims=True) #유클리드와 맨허튼 비교하기

        # print("Unit_vector_dist" , unit_dist)

        dot_matrix = np.dot(Unit_vector.T , Unit_vector)

        # print(dot_matrix)

        

        
        sum = np.sum(dot_matrix, axis=1, keepdims=False)  # Step 2
        mean = np.sum(dot_matrix, axis=1, keepdims=False)  # Step 2

        min_angle = 40 #40도 이상카메라
        max_angle = 90 #90 도범위내 카메라를 고릅니다.

        min_angle =  min_angle * np.pi /180
        max_angle =  max_angle * np.pi /180

        
        print(math.cos(max_angle))
        print(math.cos(min_angle))

        if max_angle == 90:
            max_angle = 0
        
        # filtered_array = dot_matrix[(dot_matrix > max_angle) & (dot_matrix < min_angle)]

        # print(dot_matrix[np.where((dot_matrix > max_angle) and (dot_matrix < min_angle))])

        # print( dot_matrix[np.where((dot_matrix > min_angle))]) 
         


        return Group_1 , Group_2


    


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        
        cam_centers = np.hstack(cam_centers) 

        


        
        # print("cam_centers", cam_centers)


        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)

        # Angle_based(cam_centers ,avg_cam_center)
        # pose.distatnce_based(None,cam_info, cam_centers , (1,4) , 1.331)

        # print("avg_cam_center", avg_cam_center) #모든 카메라에 대해서 평균 값 구함.

        center = avg_cam_center  
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True) # 카메라 평균으로부터 떨어진 거리 (distatnce)

        # print("dist.shape", dist)
    
        
        diagonal = np.max(dist) #길이 가장 큰 것 반환

        # print("center.flatten()", center.flatten()) 

        # Group_1 , Group_2 = find_hard_negative(cam_centers ,avg_cam_center )
        return center.flatten(), diagonal #, Group_1 , Group_2

    cam_centers = []

    for cam in cam_info: #학습 카메라를 들고 옴.
        W2C = getWorld2View2(cam.R, cam.T)
        
        # print("W2C",W2C)
        C2W = np.linalg.inv(W2C)
        # print("C2W",C2W)

        # print("?:"  ,C2W[:3, 3:4])
        cam_centers.append(C2W[:3, 3:4])

        # print(cam_info[0])

        # print(cam_centers[0])


    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []

    # print("\ncam_extrinsics keys::",cam_extrinsics.keys())
    # print("\ncam_extrinsics 1:",cam_extrinsics[1])
    # print("\ncam_extrinsics 1:",type(cam_extrinsics[1]))



    for idx, key in enumerate(cam_extrinsics):

        # print("\nidx, key",idx, key)

        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]

        # print("\n\nimages_folder", images_folder)
        # print("\n\nextr.type():",type(extr))
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
        # print("image_path:::",images_folder)

        # print("os.path.basename(extr.name) :::",os.path.basename(extr.name))

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        # print("image_path:::",image_path)
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path): #make pcd
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8): # col -> camera 등 외부 파일 들고옴

    #파일 준비
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images

    
    print("cam_extrinsics",type(cam_extrinsics[1]))
    
    # print("cam_intrinsics",cam_intrinsics)


    #파일 내 Colmap 카메라 읽기 함수 호출
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    # print("cam_infos",cam_infos)


    #cam_infos 최종 객체 담긴 리스트
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
    # print(train_cam_infos)
    nerf_normalization = getNerfppNorm(train_cam_infos)
    
 

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           
                           )
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []


    #??
    nerf_normalization = getNerfppNorm(train_cam_infos)
    

    #  ply 없으면 랜덤 포인트로 만든다. 보통 sfm 하면 ply 생기나? 몰겠네

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}