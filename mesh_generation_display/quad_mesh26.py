# -*- coding: utf-8 -*-
import open3d as o3d
import numpy as np
from mesh_generation_funcs_v23 import *
from copy import deepcopy
import sys
np.set_printoptions(precision=3, threshold=sys.maxsize)

def main():
    # hseg=16   # horizontal segmentation number
    # vseg=6    # vertical segmentation number for upper part 
    # line_len=1.5
    # ang_h=6   # np.sin(np.pi/2.)    
    # rad_h=np.pi*ang_h/180
    # ang_v=9   # np.sin(np.pi/2.)    
    # rad_v=np.pi*ang_v/180

    # hseg=45+1   # horizontal segmentation number
    # vseg=22+1    # vertical segmentation number for upper part 
    # line_len=1.5
    # ang_h=2   # np.sin(np.pi/2.)    
    # rad_h=np.pi*ang_h/180
    # ang_v=2   # np.sin(np.pi/2.)    
    # rad_v=np.pi*ang_v/180

    hseg=30+1   # horizontal segmentation number
    vseg=15+1    # vertical segmentation number for upper part 
    line_len=1.5
    ang_h=3   # np.sin(np.pi/2.)    
    rad_h=np.pi*ang_h/180
    ang_v=3   # np.sin(np.pi/2.)    
    rad_v=np.pi*ang_v/180

    # hseg=18+1   # horizontal segmentation number
    # vseg= 9+1  # vertical segmentation number for upper part 
    # line_len=1.5
    # ang_h=5   # np.sin(np.pi/2.)    
    # rad_h=np.pi*ang_h/180
    # ang_v=5   # np.sin(np.pi/2.)    
    # rad_v=np.pi*ang_v/180

    # hseg=6+1  # horizontal angle range: 6*15=90
    # vseg=3+1  # vertical angle range: (3*15)*2=90 , vertical direction include upper part and lower part.
    # line_len=1.5
    # ang_h=15   # np.sin(np.pi/2.)    
    # rad_h=np.pi*ang_h/180
    # ang_v=15   # np.sin(np.pi/2.)    
    # rad_v=np.pi*ang_v/180

    # hseg=4
    # vseg=4
    # line_len=1.5
    # ang_h=30   # np.sin(np.pi/2.)    
    # rad_h=np.pi*ang_h/180
    # ang_v=15   # np.sin(np.pi/2.)    
    # rad_v=np.pi*ang_v/180    

    pcd = o3d.io.read_point_cloud("pcd3.pcd")
    # pcd = deepcopy(pcd1)
    pcd = pcd.voxel_down_sample(voxel_size=0.02)  # down sample
    FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    pcd.translate((0-1,0.7, 0), relative=True)
    R = pcd.get_rotation_matrix_from_xyz((0,0,np.pi/3-0.2))#绕y轴旋转90°
    # R = pcd.get_rotation_matrix_from_xyz((0,0,0))#绕y轴旋转90°
    # print("----------R=",R,"---type of R=",R.dtype)
    pcd.rotate(R)#旋转点位于x=20处，若不指定则默认为原始点云质心。
    cl, ind = pcd.remove_radius_outlier(nb_points=60, radius=0.3)
    pcd = pcd.select_by_index(ind)
    # pcd = pcd.voxel_down_sample(voxel_size=0.1)  # down sample
    mesh_pcd = deepcopy(pcd)
    pcd2 = deepcopy(pcd)
    pcd3 = deepcopy(pcd)

    line_set0,line_set1,line_set2,line_set3,line_set4,line_set5 = standard_points(hseg,vseg,ang_h,ang_v,line_len)
    mesh_point_total,keypoints_num_total=mesh_centroid_computation(hseg,vseg,ang_h,ang_v,line_len,pcd)

    mesh_point=mesh_point_total.reshape((hseg-1)*(vseg-1)*2,3)

    mesh_pcd.points = o3d.utility.Vector3dVector(mesh_point[1:])

    # mesh_point=mesh_point_seq.reshape((vseg-1),(hseg-1),3)
    # mesh_point_down=mesh_point_down_seq.reshape((vseg-1),(hseg-1),3)
    # mesh_points_connection = mesh_points_connection_seq.reshape(2,(hseg-1),3)

    print("----mesh_point_total",mesh_point_total)
    print("----keypoints_num_total",keypoints_num_total)

    # pcd2.translate((1,-0.7, 0), relative=True)

    mesh_horizontal_con,mesh_vertical_con,mesh_diagonal_con=quad_mesh_generation(hseg,vseg,mesh_point_total,keypoints_num_total)

    m=mesh_face_generation(hseg,vseg,mesh_point_total,keypoints_num_total)
    ###########################  start of display of all information ################################################
    #创建窗口对象
    vis = o3d.visualization.Visualizer()
    #设置窗口标题
    vis.create_window(window_name="kitti")
    #设置点云大小
    vis.get_render_option().point_size = 1
    # vis.get_render_option().line_size = 1
    #设置颜色背景为黑色
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])    

    #将矩形框加入到窗口中
    render_option = vis.get_render_option()
    render_option.point_size = 2.0
    # render_option.show_coordinate_frame = True #显示坐标系
    mesh_pcd.paint_uniform_color([1,0,0]) #设置点云颜色为红色
    pcd2.paint_uniform_color([0,0,1]) #设置点云颜色为红色
    # pcd1.paint_uniform_color([0,1,0]) #设置点云颜色为红色
    pcd3.paint_uniform_color([0,1,1]) #设置点云颜色为红色
    # vis.add_geometry(line_set0) 
    # vis.add_geometry(line_set1) 
    # vis.add_geometry(line_set2) 
    # vis.add_geometry(line_set3) 
    # vis.add_geometry(line_set4) 
    # vis.add_geometry(line_set5) 
    # vis.add_geometry(FOR1) 
    # vis.add_geometry(mesh_line_set0) 
    # vis.add_geometry(mesh_line_set1) 
    # vis.add_geometry(mesh_line_set0_down) 
    # vis.add_geometry(mesh_line_set1_down) 
    # vis.add_geometry(mesh_line_set_connection)     
    vis.add_geometry(mesh_pcd) 
    # vis.add_geometry(pcd1) 
    vis.add_geometry(pcd2) 
    # vis.add_geometry(pcd3) 
    vis.add_geometry(mesh_horizontal_con) 
    vis.add_geometry(mesh_vertical_con) 
    vis.add_geometry(mesh_diagonal_con)    
    vis.add_geometry(m) 
    # vis.add_geometry(m_down) 
    # vis.add_geometry(m_mid) 
    vis.run()  #显示窗口，会阻塞当前线程，直到窗口关闭
    vis.destroy_window()  #销毁窗口，该函数必须从主线程调用   

if __name__=="__main__":
    main()
