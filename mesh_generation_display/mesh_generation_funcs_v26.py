import open3d as o3d
import numpy as np
from copy import deepcopy

def standard_points(hseg,vseg,ang_h,ang_v,line_len):
    rad_h=np.pi*ang_h/180
    rad_v=np.pi*ang_v/180
    hs= np.array([[0]])
    hc= np.array([[1]])
    vs= np.array([[0]])
    vc= np.array([[1]])
    for j in range(1,128):       # be less hseg and vseg
        hs= np.append(hs,np.array([np.sin(rad_h*j)]))
        hc= np.append(hc,np.array([np.cos(rad_h*j)]))
        vs= np.append(vs,np.array([np.sin(rad_v*j)]))
        vc= np.append(vc,np.array([np.cos(rad_v*j)]))
    
        ######### start of computation the key points of space seperation  ##############################################################
    points_3dbox=np.array([[0,0,0]])
    for k in range(0,vseg):
        for i in range(0,hseg):
            # points_3dbox=np.append(points_3dbox,np.array([[line_len*vc[0]*hc[i],line_len*vc[0]*hs[i],line_len*vs[0]]]),axis=0)
            points_3dbox=np.append(points_3dbox,np.array([[line_len*vc[k]*hc[i],line_len*vc[k]*hs[i],line_len*vs[k]]]),axis=0)    
    ######### end of computation the key points of space seperation ##############################################################
    
    ######### start of computation the seperation line of space  ##############################################################
    #指明哪两个顶点之间相连
    lines_box=np.array([[0,0]])
    lines_hor=np.array([[0,0]])
    lines_ver=np.array([[0,0]])
    for k in range(0,vseg):  # [0,0,0] is connected to the box points
        for i in range(0,hseg):
            lines_box=np.append(lines_box,np.array([[0,k*hseg+i+1]]),axis=0)
    for k in range(0,vseg):   # horizontal connection lines
        for i in range(0,hseg-1):
            lines_hor=np.append(lines_hor,np.array([[k*hseg+i+1,k*hseg+i+2]]),axis=0)
    for k in range(0,vseg-1):   # vertical connection lines
        for i in range(0,hseg):
            lines_ver=np.append(lines_ver,np.array([[k*hseg+i+1,(k+1)*hseg+i+1]]),axis=0)
    # print("---lines_ver=",lines_ver)

    ######### start of computation the seperation line of symmetric space ##############################################################
    points_3dbox_sym=np.array([[0,0,0]])
    for idx in range(1,len(points_3dbox)):
        points_3dbox_sym=np.append(points_3dbox_sym,np.array([[points_3dbox[idx][0],points_3dbox[idx][1],-points_3dbox[idx][2]]]),axis=0)    
    lines_box_sym=np.array([[0,0]])
    lines_hor_sym=np.array([[0,0]])
    lines_ver_sym=np.array([[0,0]])
    for k in range(0,vseg):  # [0,0,0] is connected to the box points
        for i in range(0,hseg):
            lines_box_sym=np.append(lines_box_sym,np.array([[0,k*hseg+i+1]]),axis=0)
    for k in range(0,vseg):   # horizontal connection lines
        for i in range(0,hseg-1):
            lines_hor_sym=np.append(lines_hor_sym,np.array([[k*hseg+i+1,k*hseg+i+2]]),axis=0)
    for k in range(0,vseg-1):   # vertical connection lines
        for i in range(0,hseg):
            lines_ver_sym=np.append(lines_ver_sym,np.array([[k*hseg+i+1,(k+1)*hseg+i+1]]),axis=0)
    ######### end of computation the seperation line of space  ##############################################################

    ###########################  start of boundary line of the point clouds #################################
    #设置点与点之间线段的颜色
    colors0   = np.array([[0, 1, 0] for j in range(len(lines_box))])
    colors1  = np.array([[0, 1, 1] for j in range(len(lines_hor))])
    colors2  = np.array([[1, 1, 0] for j in range(len(lines_ver))])
    #创建Bbox候选框对象
    line_set0 = o3d.geometry.LineSet()
    line_set1 = o3d.geometry.LineSet()
    line_set2 = o3d.geometry.LineSet()
    line_set3 = o3d.geometry.LineSet()
    line_set4 = o3d.geometry.LineSet()
    line_set5 = o3d.geometry.LineSet()
    #将八个顶点连接次序的信息转换成o3d可以使用的数据类型
    line_set0.lines = o3d.utility.Vector2iVector(lines_box)
    line_set1.lines = o3d.utility.Vector2iVector(lines_hor)
    line_set2.lines = o3d.utility.Vector2iVector(lines_ver)
    line_set3.lines = o3d.utility.Vector2iVector(lines_box)
    line_set4.lines = o3d.utility.Vector2iVector(lines_hor)
    line_set5.lines = o3d.utility.Vector2iVector(lines_ver)
    #设置每条线段的颜色
    line_set0.colors = o3d.utility.Vector3dVector(colors0)
    line_set1.colors = o3d.utility.Vector3dVector(colors1)
    line_set2.colors = o3d.utility.Vector3dVector(colors2)
    line_set3.colors = o3d.utility.Vector3dVector(colors0)
    line_set4.colors = o3d.utility.Vector3dVector(colors1)
    line_set5.colors = o3d.utility.Vector3dVector(colors2)
    #把八个顶点的空间信息转换成o3d可以使用的数据类型
    line_set0.points = o3d.utility.Vector3dVector(points_3dbox)
    line_set1.points = o3d.utility.Vector3dVector(points_3dbox)
    line_set2.points = o3d.utility.Vector3dVector(points_3dbox)
    line_set3.points = o3d.utility.Vector3dVector(points_3dbox_sym)
    line_set4.points = o3d.utility.Vector3dVector(points_3dbox_sym)
    line_set5.points = o3d.utility.Vector3dVector(points_3dbox_sym)
    ###########################  end of boundary line of the point clouds ################################################
    return line_set0,line_set1,line_set2,line_set3,line_set4,line_set5

def mesh_centroid_computation(hseg,vseg,ang_h,ang_v,line_len,pcd):
    rad_h=np.pi*ang_h/180
    rad_v=np.pi*ang_v/180
        ###########################  start of centroid computation ################################################
    mesh_point=np.zeros((vseg-1,hseg-1,3))
    keypoints_num=np.zeros((vseg-1,hseg-1))
    mesh_point_down=np.zeros((vseg-1,hseg-1,3))
    keypoints_num_down=np.zeros((vseg-1,hseg-1))

    # test the corresponding between index j, k  and hseg,vseg
    for j in range(vseg-1):  # vertical
        for k in range(hseg-1):  # horizatal
            mesh_point[j][k]= np.array([0,0,0])+np.array([0,0,0])
    
    point_num=len(np.array(pcd.points))
    tpn=0
    for i in range(point_num):
        # print("--i=",i,"----pcd.points=",np.array(pcd.points[i]))
        len_proj=np.sqrt(np.square(pcd.points[i][0])+np.square(pcd.points[i][1]))
        if len_proj<=0.05:
            continue
        len_whol=np.sqrt(np.square(len_proj)        +np.square(pcd.points[i][2]))
        degree_h=np.degrees(np.arccos(np.array(pcd.points[i][0])/len_proj))
        degree_v=np.degrees(np.arcsin(np.array(pcd.points[i][2])/len_whol))
        if degree_v/ang_v>=0:
            tpn=tpn+1
            idx_h=int(np.floor(degree_h/ang_h))
            idx_v=int(np.floor(degree_v/ang_v))
            # print("------idx_h=",idx_h,"-----idx_v=",idx_v)
            keypoints_num[idx_v][idx_h]=keypoints_num[idx_v][idx_h]+1
            mesh_point[idx_v][idx_h]=mesh_point[idx_v][idx_h]+pcd.points[i]
            # print("--i=",i,"----pcd.points=",np.array(pcd.points[i]),"----idx_h=",idx_h,"-----idx_v=",idx_v,"--keypoints_num=",keypoints_num[idx_v][idx_h],"--mesh_point=",mesh_point[idx_v][idx_h])
        else:
            idx_h=int(np.floor(degree_h/ang_h))
            idx_v=int(np.floor(degree_v/ang_v))
            # print("---idx_h=",idx_h,"----idx_v=",idx_v)
            keypoints_num_down[idx_v][idx_h]=keypoints_num_down[idx_v][idx_h]+1
            mesh_point_down[idx_v][idx_h]=mesh_point_down[idx_v][idx_h]+pcd.points[i]
            # print("--i=",i,"----pcd.points=",np.array(pcd.points[i]),"----idx_h=",idx_h,"-----idx_v=",idx_v,"--keypoints_num=",keypoints_num[idx_v][idx_h],"--mesh_point_down=",mesh_point_down[idx_v][idx_h])
       
    # print("-----mesh_point=",mesh_point)
    # print("-----keypoints_num=",keypoints_num)
    # print("-----mesh_point_down=",mesh_point_down)
    # print("-----keypoints_num_down=",keypoints_num_down)
    for j in range(vseg-1):  # vertical
        for k in range(hseg-1):  # horizatal
            if keypoints_num[j][k]<0.01:
                continue
            else:
                mesh_point[j][k]= mesh_point[j][k]/keypoints_num[j][k]
    for j in range(-vseg+1,0):  # vertical
        for k in range(hseg-1):  # horizatal
            if keypoints_num_down[j][k]<0.01:
                continue
            else:
                mesh_point_down[j][k]= mesh_point_down[j][k]/keypoints_num_down[j][k]
            # print("---j=",j,"--k=",k,"--mesh_point[j][k]=",mesh_point2[j][k])
    # print("-----mesh_point=",mesh_point,"---mesh_point[2][4]=",mesh_point[2][4])
    # print("-----mesh_point_down2=",mesh_point_down,"---mesh_point_down[2][4]=",mesh_point_down[2][4])
    mesh_point_seq=mesh_point.reshape((hseg-1)*(vseg-1),3)
    mesh_point_down_seq=mesh_point_down.reshape((hseg-1)*(vseg-1),3)
    # print("------mesh_point_down_seq=",mesh_point_down_seq)

    mesh_points_connection=deepcopy(mesh_point[0:2])
    mesh_points_connection[1]=mesh_point_down[-1]
    # print("-----mesh_points_connection=",mesh_points_connection)
    mesh_points_connection_seq=mesh_points_connection.reshape((hseg-1)*2,3)
    keypoints_num_seq=keypoints_num.reshape((hseg-1)*(vseg-1))
    keypoints_num_down_seq=keypoints_num_down.reshape((hseg-1)*(vseg-1))
    # print("-----mesh_points_connection_seq=",mesh_points_connection_seq)

    mesh_point_upper=np.zeros((vseg-1,hseg-1,3))
    mesh_point_lower=np.zeros((vseg-1,hseg-1,3))
    keypoints_num_upper=np.zeros((vseg-1,hseg-1))
    keypoints_num_lower=np.zeros((vseg-1,hseg-1))
    
    keypoint_num_half=(vseg-1)*(hseg-1)
    mesh_point_upper_seq=np.zeros((keypoint_num_half,3))
    mesh_point_lower_seq=np.zeros((keypoint_num_half,3))
    keypoints_num_upper_seq=np.zeros(keypoint_num_half)
    keypoints_num_lower_seq=np.zeros(keypoint_num_half)

    for k in range(keypoint_num_half):
        print("----k=",k)
        mesh_point_upper_seq[k]=mesh_point_seq[keypoint_num_half-1-k]
        mesh_point_lower_seq[k]=mesh_point_down_seq[keypoint_num_half-1-k]
        keypoints_num_upper_seq[k]=keypoints_num_seq[keypoint_num_half-1-k]
        keypoints_num_lower_seq[k]=keypoints_num_down_seq[keypoint_num_half-1-k]
    
    mesh_point_upper=mesh_point_upper_seq.reshape((vseg-1),(hseg-1),3)
    mesh_point_lower=mesh_point_lower_seq.reshape((vseg-1),(hseg-1),3)
    keypoints_num_upper=keypoints_num_upper_seq.reshape((vseg-1),(hseg-1))
    keypoints_num_lower=keypoints_num_lower_seq.reshape((vseg-1),(hseg-1))

    mesh_point_total=np.append(mesh_point_upper,mesh_point_lower,axis=0)
    keypoints_num_total=np.append(keypoints_num_upper,keypoints_num_lower,axis=0)

    # print("----mesh_point_upper=",mesh_point_upper)
    # print("----mesh_point_lower=",mesh_point_lower)
    # print("----mesh_point_total=",mesh_point_total)
    # print("----keypoints_num_upper=",keypoints_num_upper)
    # print("----keypoints_num_lower=",keypoints_num_lower)
    # print("----keypoints_num_total=",keypoints_num_total)
    ###########################  end of centroid computation ################################################
    # return mesh_point_seq,mesh_point_down_seq,mesh_points_connection_seq,keypoints_num_seq,keypoints_num_down_seq
    return mesh_point_total,keypoints_num_total

# def quad_mesh_generation(hseg,vseg,ang_h,ang_v,mesh_point_total,keypoints_num_total):
def quad_mesh_generation(hseg,vseg,mesh_point_total,keypoints_num_total):
    # rad_h=np.pi*ang_h/180
    # rad_v=np.pi*ang_v/180
    keypoints_num=keypoints_num_total
    ###########################  start of horizontal and vertical mesh line of the point clouds #################################
    mesh_lines_hor=np.array([[0,0]])  # horizatal connection lines
    mesh_lines_ver=np.array([[0,0]])  # vertical connection lines
    mesh_lines_diag=np.array([[0,0]])  # vertical connection lines
    for k in range(0,(vseg-1)*2):  # [0,0,0] is connected to the box points  vseg=4; hseg=7;
        for j in range(0,hseg-1):
            if (j<(hseg-1)-1) and (keypoints_num[k][j]>=1) and (keypoints_num[k][j+1]>=1):
                # print("-----keypoints_num[k][j]=",keypoints_num[k][j],"-----keypoints_num[k][j+1]=",keypoints_num[k][j+1])
                mesh_lines_hor=np.append(mesh_lines_hor,np.array([[k*(hseg-1)+j,k*(hseg-1)+j+1]]),axis=0)
                # print("---k=",k,"---j=",j,"---mesh_lines_hor=",mesh_lines_hor)
            if (k<(vseg-1)*2-1) and (keypoints_num[k][j]>=1) and (keypoints_num[k+1][j]>=1):
                # print("-----keypoints_num[k][j]=",keypoints_num[k][j],"-----keypoints_num[k+1][j]=",keypoints_num[k+1][j])
                mesh_lines_ver=np.append(mesh_lines_ver,np.array([[k*(hseg-1)+j,(k+1)*(hseg-1)+j]]),axis=0)
    # print("****---------------------------------------------------------------------------------------------*******")
    mesh_colors0   = np.array([[1, 1, 1] for j in range(len(mesh_lines_hor))])
    mesh_line_set0 = o3d.geometry.LineSet()
    mesh_line_set0.lines = o3d.utility.Vector2iVector(mesh_lines_hor)
    mesh_line_set0.colors = o3d.utility.Vector3dVector(mesh_colors0)
    mesh_line_set0.points = o3d.utility.Vector3dVector(np.array(mesh_point_total.reshape((vseg-1)*2*(hseg-1),3)))
    
    mesh_colors1   = np.array([[1, 1, 1] for j in range(len(mesh_lines_ver))])
    mesh_line_set1 = o3d.geometry.LineSet()
    mesh_line_set1.lines = o3d.utility.Vector2iVector(mesh_lines_ver)
    mesh_line_set1.colors = o3d.utility.Vector3dVector(mesh_colors1)
    mesh_line_set1.points = o3d.utility.Vector3dVector(np.array(mesh_point_total.reshape((vseg-1)*2*(hseg-1),3)))
    ###########################  end of horizontal and vertical mesh line of the point clouds #################################

    ###########################  start of diagnal mesh line of the point clouds ##############################################
    flag_keypoint=keypoints_num_total
    for i in range((vseg-1)*2):
        for j in range(hseg-1):
            if keypoints_num_total[i][j]>0.2:
                flag_keypoint[i][j]=5
            else:
                flag_keypoint[i][j]=0
    print("----flag_keypoint=",flag_keypoint)

    for k in range(0,(vseg-1)*2-1):  # [0,0,0] is connected to the box points  vseg=4; hseg=7;
        for j in range(0,hseg-1-1):
            if flag_keypoint[k][j]+flag_keypoint[k+1][j+1]>9:  # 5+5=10
                if flag_keypoint[k+1][j]+flag_keypoint[k][j+1]<6:  # 5
                    # print("-----keypoints_num[k][j]=",keypoints_num[k][j],"-----keypoints_num[k][j+1]=",keypoints_num[k][j+1])
                    mesh_lines_diag=np.append(mesh_lines_diag,np.array([[k*(hseg-1)+j,(k+1)*(hseg-1)+j+1]]),axis=0)
            if flag_keypoint[k+1][j]+flag_keypoint[k][j+1]>=10:
                if flag_keypoint[k][j]+flag_keypoint[k+1][j+1]<=5:
                    # print("-----keypoints_num[k][j]=",keypoints_num[k][j],"-----keypoints_num[k][j+1]=",keypoints_num[k][j+1])
                    mesh_lines_diag=np.append(mesh_lines_diag,np.array([[k*(hseg-1)+j+1,(k+1)*(hseg-1)+j]]),axis=0)
    print("----------------mesh_lines_diag=",mesh_lines_diag)
    print("----------------mesh_lines_diag[:-1]=",mesh_lines_diag[:-1])
    
    mesh_colors2   = np.array([[1, 1, 1] for j in range(len(mesh_lines_diag))])
    mesh_line_set2 = o3d.geometry.LineSet()
    mesh_line_set2.lines = o3d.utility.Vector2iVector(mesh_lines_diag)
    mesh_line_set2.colors = o3d.utility.Vector3dVector(mesh_colors2)
    mesh_line_set2.points = o3d.utility.Vector3dVector(np.array(mesh_point_total.reshape((vseg-1)*2*(hseg-1),3)))

    mesh_horizontal_con=mesh_line_set0
    # mesh_horizontal_con=0
    mesh_vertical_con=mesh_line_set1
    # mesh_vertical_con=0
    mesh_diagonal_con=mesh_line_set2
    # mesh_diagonal_con=0
    
    return mesh_horizontal_con,mesh_vertical_con,mesh_diagonal_con

def mesh_face_generation(hseg,vseg,mesh_point_total,keypoints_num_total):
    keypoints_num=keypoints_num_total
    flag_keypoint=keypoints_num_total
    for i in range((vseg-1)*2):
        for j in range(hseg-1):
            if keypoints_num_total[i][j]>0.2:
                flag_keypoint[i][j]=5
            else:
                flag_keypoint[i][j]=0
    print("----flag_keypoint=",flag_keypoint)

    face_keypoints=np.array([[0,0,0]])
    for k in range(0,(vseg-1)*2-1):  # [0,0,0] is connected to the box points  vseg=4; hseg=7;
        for j in range(0,hseg-1-1):
            if (flag_keypoint[k][j]+flag_keypoint[k][j+1]+flag_keypoint[k+1][j]+flag_keypoint[k+1][j+1])>19:
                face_keypoints=np.append(face_keypoints,np.array([[  k*(hseg-1)+j,(k+1)*(hseg-1)+j,k*(hseg-1)+j+1]]),axis=0)
                face_keypoints=np.append(face_keypoints,np.array([[k*(hseg-1)+j+1,(k+1)*(hseg-1)+j,  k*(hseg-1)+j]]),axis=0)
                face_keypoints=np.append(face_keypoints,np.array([[  (k+1)*(hseg-1)+j,k*(hseg-1)+j+1,(k+1)*(hseg-1)+j+1]]),axis=0)
                face_keypoints=np.append(face_keypoints,np.array([[(k+1)*(hseg-1)+j+1,k*(hseg-1)+j+1,  (k+1)*(hseg-1)+j]]),axis=0)
            elif (flag_keypoint[k][j]+flag_keypoint[k+1][j+1])>9:
                if flag_keypoint[k+1][j]>4:
                    face_keypoints=np.append(face_keypoints,np.array([[  k*(hseg-1)+j,(k+1)*(hseg-1)+j+1,(k+1)*(hseg-1)+j]]),axis=0)
                    face_keypoints=np.append(face_keypoints,np.array([[  (k+1)*(hseg-1)+j,(k+1)*(hseg-1)+j+1,k*(hseg-1)+j]]),axis=0)
                elif flag_keypoint[k][j+1]>4:
                    face_keypoints=np.append(face_keypoints,np.array([[  k*(hseg-1)+j,(k+1)*(hseg-1)+j+1,k*(hseg-1)+j+1]]),axis=0)
                    face_keypoints=np.append(face_keypoints,np.array([[  k*(hseg-1)+j+1,(k+1)*(hseg-1)+j+1,k*(hseg-1)+j]]),axis=0)
            elif (flag_keypoint[k+1][j]+flag_keypoint[k][j+1])>9:
                if flag_keypoint[k][j]>4:
                    face_keypoints=np.append(face_keypoints,np.array([[  k*(hseg-1)+j,(k+0)*(hseg-1)+j+1,(k+1)*(hseg-1)+j]]),axis=0)
                    face_keypoints=np.append(face_keypoints,np.array([[  (k+1)*(hseg-1)+j,(k+0)*(hseg-1)+j+1,k*(hseg-1)+j]]),axis=0)
                elif flag_keypoint[k+1][j+1]>4:
                    face_keypoints=np.append(face_keypoints,np.array([[  (k+1)*(hseg-1)+j,(k+1)*(hseg-1)+j+1,k*(hseg-1)+j+1]]),axis=0)
                    face_keypoints=np.append(face_keypoints,np.array([[  k*(hseg-1)+j+1,(k+1)*(hseg-1)+j+1,(k+1)*(hseg-1)+j]]),axis=0)
    
    print("-----------face_keypoints=",face_keypoints)

                # face_keypoints=np.append(face_keypoints,np.array([[k*(hseg-1)+j,(k+1)*(hseg-1)+j+1]]),axis=0)
            # if flag_keypoint[k][j]+flag_keypoint[k+1][j+1]>9:  # 5+5=10
            #     if flag_keypoint[k+1][j]+flag_keypoint[k][j+1]<6:  # 5
            #         # print("-----keypoints_num[k][j]=",keypoints_num[k][j],"-----keypoints_num[k][j+1]=",keypoints_num[k][j+1])
            #         mesh_lines_diag=np.append(mesh_lines_diag,np.array([[k*(hseg-1)+j,(k+1)*(hseg-1)+j+1]]),axis=0)
            # if flag_keypoint[k+1][j]+flag_keypoint[k][j+1]>=10:
            #     if flag_keypoint[k][j]+flag_keypoint[k+1][j+1]<=5:
            #         # print("-----keypoints_num[k][j]=",keypoints_num[k][j],"-----keypoints_num[k][j+1]=",keypoints_num[k][j+1])
            #         mesh_lines_diag=np.append(mesh_lines_diag,np.array([[k*(hseg-1)+j+1,(k+1)*(hseg-1)+j]]),axis=0)
    # mesh_point=mesh_point_total.reshape((vseg-1)*2,(hseg-1),3)
    mesh_point_seq=mesh_point_total.reshape((vseg-1)*2*(hseg-1),3)
    print("------mesh_point=",mesh_point_seq)
#     points=mesh_point_seq
    colors = np.array([[0., 0., 0.8]]) 
    vert = [[0]*3]*len(mesh_point_seq)
    for k in range(len(mesh_point_seq)):
        vert[k]=[mesh_point_seq[k][0],mesh_point_seq[k][1],mesh_point_seq[k][2]]
        colors=np.append(colors,np.array([[0,0,0.8]]),axis=0)
    faces = [[0]*3]*len(face_keypoints)
    for k in range(len(face_keypoints)):
        faces[k]=[face_keypoints[k][0],face_keypoints[k][1],face_keypoints[k][2]]
#     # print("----vert=",vert)
#     # print("----faces=",faces)
#     # print("----colors=",colors)
    m=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vert),
                            o3d.utility.Vector3iVector(faces))
    m.vertex_colors=o3d.utility.Vector3dVector(colors[1:])



    # m=face_keypoints
    return m