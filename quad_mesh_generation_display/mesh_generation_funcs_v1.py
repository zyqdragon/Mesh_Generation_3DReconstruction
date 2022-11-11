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
            print("------idx_h=",idx_h,"-----idx_v=",idx_v)
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
    # print("-----mesh_points_connection_seq=",mesh_points_connection_seq)
    ###########################  end of centroid computation ################################################
    return mesh_point_seq,mesh_point_down_seq,mesh_points_connection_seq,keypoints_num,keypoints_num_down

def quad_mesh_generation(hseg,vseg,ang_h,ang_v,line_len,\
                         keypoints_num,keypoints_num_down,\
                         mesh_point_seq,mesh_point_down_seq,mesh_points_connection_seq,pcd):

    rad_h=np.pi*ang_h/180
    rad_v=np.pi*ang_v/180

    ###########################  start of mesh line of the point clouds of upper part#################################
    mesh_lines_hor=np.array([[0,0]])  # horizatal connection lines
    mesh_lines_ver=np.array([[0,0]])  # vertical connection lines
    for k in range(0,vseg-1):  # [0,0,0] is connected to the box points   vseg=4; hseg=7;
        for j in range(0,hseg-1):
            if (j<hseg-2) and (keypoints_num[k][j]>1) and (keypoints_num[k][j+1]>1):
                # print("-----keypoints_num[k][j]=",keypoints_num[k][j],"-----keypoints_num[k][j+1]=",keypoints_num[k][j+1])
                mesh_lines_hor=np.append(mesh_lines_hor,np.array([[k*(hseg-1)+j,k*(hseg-1)+j+1]]),axis=0)
                # print("---k=",k,"---j=",j,"---mesh_lines_hor=",mesh_lines_hor)
    # print("****---------------------------------------------------------------------------------------------*******")
    # print("-----keypoints_num=",keypoints_num,"---keypoints_num[1][2]=",keypoints_num[1][2])
    for k in range(0,vseg-1):  # [0,0,0] is connected to the box points   vseg=4; hseg=7;
        for j in range(0,hseg-1):
            # print("--k=",k,"--j=",j,"---keypoints_num[k][j]=",keypoints_num[k][j],"-----keypoints_num[k+1][j]=",keypoints_num[k+1][j])
            if (k<vseg-2) and (keypoints_num[k][j]>1) and (keypoints_num[k+1][j]>1):
                # print("-----keypoints_num[k][j]=",keypoints_num[k][j],"-----keypoints_num[k+1][j]=",keypoints_num[k+1][j])
                mesh_lines_ver=np.append(mesh_lines_ver,np.array([[k*(hseg-1)+j,(k+1)*(hseg-1)+j]]),axis=0)
                # print("banch1---k=",k,"---j=",j,"---mesh_lines_hor=",mesh_lines_ver)
            elif (k<vseg-2) and (j>hseg/2) and (keypoints_num[k][j]>1) and (keypoints_num[k+1][j-1]>1):
                mesh_lines_ver=np.append(mesh_lines_ver,np.array([[k*(hseg-1)+j,(k+1)*(hseg-1)+j-1]]),axis=0)
                # print("banch2---k=",k,"---j=",j,"---mesh_lines_hor=",mesh_lines_ver)
            elif (k<vseg-2) and (j<hseg/2) and (keypoints_num[k][j]>1) and (keypoints_num[k+1][j+1]>1):
                mesh_lines_ver=np.append(mesh_lines_ver,np.array([[k*(hseg-1)+j,(k+1)*(hseg-1)+j+1]]),axis=0)
                # print("banch3---k=",k,"---j=",j,"---mesh_lines_hor=",mesh_lines_ver,"--(k+1)*(hseg-1)+j+1=",(k+1)*(hseg-1)+j+1)
    mesh_colors0   = np.array([[1, 1, 1] for j in range(len(mesh_lines_hor))])
    mesh_line_set0 = o3d.geometry.LineSet()
    mesh_line_set0.lines = o3d.utility.Vector2iVector(mesh_lines_hor)
    mesh_line_set0.colors = o3d.utility.Vector3dVector(mesh_colors0)
    mesh_line_set0.points = o3d.utility.Vector3dVector(mesh_point_seq)
    mesh_colors1   = np.array([[1, 1, 1] for j in range(len(mesh_lines_ver))])
    mesh_line_set1 = o3d.geometry.LineSet()
    mesh_line_set1.lines = o3d.utility.Vector2iVector(mesh_lines_ver)
    mesh_line_set1.colors = o3d.utility.Vector3dVector(mesh_colors1)
    mesh_line_set1.points = o3d.utility.Vector3dVector(mesh_point_seq)
    # print("----mesh_point_seq=",mesh_point_seq)
    # print("----keypoints_num=",keypoints_num)

    ###########################  end of mesh line of the point clouds of upper part#################################

    # ###########################  start of mesh line of the point clouds of down part#################################
    mesh_lines_hor_down=np.array([[0,0]])  # horizatal connection lines
    mesh_lines_ver_down=np.array([[0,0]])  # vertical connection lines
    for k in range(0,vseg-1):  # [0,0,0] is connected to the box points   vseg=4; hseg=7;
        for j in range(0,hseg-1):
            if (j<hseg-2) and (keypoints_num_down[k][j]>1) and (keypoints_num_down[k][j+1]>1):
                # print("-----keypoints_num[k][j]=",keypoints_num[k][j],"-----keypoints_num[k][j+1]=",keypoints_num[k][j+1])
                mesh_lines_hor_down=np.append(mesh_lines_hor_down,np.array([[k*(hseg-1)+j,k*(hseg-1)+j+1]]),axis=0)
                # print("---k=",k,"---j=",j,"---mesh_lines_hor_down=",mesh_lines_hor_down)
    # print("****---------------------------------------------------------------------------------------------*******")
    for k in range(0,vseg-1):  # [0,0,0] is connected to the box points   vseg=4; hseg=7;
        for j in range(0,hseg-1):
            # print("--k=",k,"--j=",j,"---keypoints_num[k][j]=",keypoints_num[k][j],"-----keypoints_num[k+1][j]=",keypoints_num[k+1][j])
            if (k<vseg-2) and (keypoints_num_down[k][j]>1) and (keypoints_num_down[k+1][j]>1):
                # print("-----keypoints_num[k][j]=",keypoints_num[k][j],"-----keypoints_num[k+1][j]=",keypoints_num[k+1][j])
                mesh_lines_ver_down=np.append(mesh_lines_ver_down,np.array([[k*(hseg-1)+j,(k+1)*(hseg-1)+j]]),axis=0)
                # print("branch1---k=",k,"---j=",j,"---keypoints_num_down[k][j]=",keypoints_num_down[k][j],"--keypoints_num_down[k+1][j]=",keypoints_num_down[k+1][j],"---mesh_lines_ver_down=",mesh_lines_ver_down)
            elif (k<vseg-2) and (j<hseg/2) and (keypoints_num_down[k][j]<1)and (keypoints_num_down[k][j+1]>1) and (keypoints_num_down[k+1][j]>1):
                mesh_lines_ver_down=np.append(mesh_lines_ver_down,np.array([[k*(hseg-1)+j+1,(k+1)*(hseg-1)+j]]),axis=0)
                # print("branch2---k=",k,"---j=",j,"---keypoints_num_down[k][j]=",keypoints_num_down[k][j],"--keypoints_num_down[k+1][j-1]=",keypoints_num_down[k+1][j-1],"---mesh_lines_ver_down=",mesh_lines_ver_down)
            elif (k<vseg-2) and (j>hseg/2) and (keypoints_num_down[k][j]<1)and (keypoints_num_down[k][j-1]>1) and (keypoints_num_down[k+1][j]>1):
                mesh_lines_ver_down=np.append(mesh_lines_ver_down,np.array([[k*(hseg-1)+j-1,(k+1)*(hseg-1)+j]]),axis=0)
                # print("branch3---k=",k,"---j=",j,"---keypoints_num_down[k][j]=",keypoints_num_down[k][j],"--keypoints_num_down[k+1][j-1]=",keypoints_num_down[k+1][j-1],"---mesh_lines_ver_down=",mesh_lines_ver_down)
   
    mesh_colors0_down   = np.array([[1, 1, 1] for j in range(len(mesh_lines_hor_down))])
    mesh_line_set0_down = o3d.geometry.LineSet()
    mesh_line_set0_down.lines = o3d.utility.Vector2iVector(mesh_lines_hor_down)
    mesh_line_set0_down.colors = o3d.utility.Vector3dVector(mesh_colors0_down)
    mesh_line_set0_down.points = o3d.utility.Vector3dVector(mesh_point_down_seq)
    mesh_colors1_down   = np.array([[1, 1, 1] for j in range(len(mesh_lines_ver_down))])
    mesh_line_set1_down = o3d.geometry.LineSet()
    mesh_line_set1_down.lines = o3d.utility.Vector2iVector(mesh_lines_ver_down)
    mesh_line_set1_down.colors = o3d.utility.Vector3dVector(mesh_colors1_down)
    mesh_line_set1_down.points = o3d.utility.Vector3dVector(mesh_point_down_seq)
    # print("----mesh_line_set0_down.lines=",np.array(mesh_line_set0_down.lines))
    ###########################  end of mesh line of the point clouds of down part#################################

    # ###########################  start of mesh connection line between upper and down part#################################
    mesh_lines_connection=np.array([[0,0]])  # horizatal connection lines
    for k in range(0,hseg-1):  # [0,0,0] is connected to the box points   vseg=4; hseg=7;
        mesh_lines_connection=np.append(mesh_lines_connection,np.array([[k,k+hseg-1]]),axis=0)
    
    mesh_colors_connection  = np.array([[1, 1, 1] for j in range(hseg)])
    mesh_line_set_connection = o3d.geometry.LineSet()
    mesh_line_set_connection.lines = o3d.utility.Vector2iVector(mesh_lines_connection)
    mesh_line_set_connection.colors = o3d.utility.Vector3dVector(mesh_colors_connection)
    mesh_line_set_connection.points = o3d.utility.Vector3dVector(mesh_points_connection_seq)
    # print("------mesh_lines_connection=",mesh_lines_connection)
    ###########################  end of mesh connection line between upper and down part#################################
    return mesh_line_set0, mesh_line_set1,mesh_line_set0_down,mesh_line_set1_down,mesh_line_set_connection

def mesh_face_generation(vseg,hseg,mesh_line_set0,mesh_line_set1,mesh_line_set0_down,mesh_line_set1_down,mesh_line_set_connection,\
                         keypoints_num,keypoints_num_down,mesh_point_seq,mesh_point_down_seq,mesh_points_connection_seq):
    mesh_lines_ver=np.array(mesh_line_set0.lines)
    mesh_lines_hor=np.array(mesh_line_set1.lines)

    mesh_lines_ver_down=np.array(mesh_line_set0_down.lines)
    mesh_lines_hor_down=np.array(mesh_line_set1_down.lines)

    mesh_points_connection_seq

    mesh_lines_mid_conn=np.array(mesh_line_set_connection.lines)

    # mesh_point_seq=mesh_point.reshape((hseg-1)*(vseg-1),3)
    mesh_point=mesh_point_seq.reshape((vseg-1),(hseg-1),3)
    mesh_point_down=mesh_point_down_seq.reshape((vseg-1),(hseg-1),3)
    ###########################  start of mesh face generation of upper part #################################
    # print(mesh_lines_ver[1:])   # delete the first element i.e. [0,0]
    # print(mesh_lines_hor[1:])
    # print(mesh_lines_hor_down[1:])  
    # print(mesh_lines_ver_down[1:])
    # print(mesh_lines_connection)
    face_upper_connection=np.append(mesh_lines_hor[1:],mesh_lines_ver[1:],axis=0)

    mesh_quadrilateral_upper= np.zeros((vseg-1,hseg-1,6)) # left_side, bottom_side, right_side, top_side, diag_line; anti_diag_line of quadrilateral
    # print("---------mesh_quadrilateral_upper=",mesh_quadrilateral_upper)
    for k in range(len(face_upper_connection)):
        if face_upper_connection[k][1]>face_upper_connection[k][0]:
            row_num_start=int(np.floor(face_upper_connection[k][0]/(hseg-1)))
            col_num_start=int(np.mod(face_upper_connection[k][0],(hseg-1)))
            row_num_end=int(np.floor(face_upper_connection[k][1]/(hseg-1)))
            col_num_end=int(np.mod(face_upper_connection[k][1],(hseg-1)))
            # print("----k=",k,"---face_upper_connection=",face_upper_connection[k])
            # print("--row_num_start=",row_num_start,"---col_num_start=",col_num_start,"--row_num_end=",row_num_end,"---col_num_end=",col_num_end)
            if row_num_end==row_num_start and col_num_end-col_num_start==1:
                mesh_quadrilateral_upper[row_num_start][col_num_start][1]=1
                if row_num_start>=1:
                    mesh_quadrilateral_upper[row_num_start-1][col_num_start][3]=1
            elif row_num_end-row_num_start==1 and col_num_end==col_num_start:
                if col_num_start>=1:
                    mesh_quadrilateral_upper[row_num_start][col_num_start-1][0]=1
                mesh_quadrilateral_upper[row_num_start][col_num_start][2]=1
            elif row_num_end-row_num_start==1 and col_num_end-col_num_start==1:
                mesh_quadrilateral_upper[row_num_start][col_num_start][4]=1
            elif row_num_end-row_num_start==1 and col_num_start-col_num_end==1:
                mesh_quadrilateral_upper[row_num_start][col_num_start-1][5]=1
        else:
            print("---------wrong point--------")
    mesh_quadrilateral_upper_temp=mesh_quadrilateral_upper[0:vseg-2][0:hseg-2]
    # print("-----mesh_quadrilateral_upper=",mesh_quadrilateral_upper)
    # print("-----mesh_quadrilateral_upper_temp=",mesh_quadrilateral_upper_temp)
    mesh_quadrilateral_upper=mesh_quadrilateral_upper_temp
    # print("-----mesh_quadrilateral_upper2=",mesh_quadrilateral_upper)

    face_keypoints=np.array([[0,0,0]])   
    for k in range(0,vseg-2):   
        for j in range(0,hseg-2):
            # print("----mesh_quadrilateral_upper[k][j]=",mesh_quadrilateral_upper[k][j],"--mesh_quadrilateral_upper[k][j][5]=",mesh_quadrilateral_upper[k][j][5]-1)
            if np.sum(mesh_quadrilateral_upper[k][j])==4:  # all four sides of quadrilateral are valid 
                face_keypoints=np.append(face_keypoints,np.array([[  k*(hseg-1)+j+1,  (k+1)*(hseg-1)+j,    k*(hseg-1)+j]]),axis=0)
                face_keypoints=np.append(face_keypoints,np.array([[  k*(hseg-1)+j+1,(k+1)*(hseg-1)+j+1,(k+1)*(hseg-1)+j]]),axis=0)
                face_keypoints=np.append(face_keypoints,np.array([[    k*(hseg-1)+j,  (k+1)*(hseg-1)+j,  k*(hseg-1)+j+1]]),axis=0)
                face_keypoints=np.append(face_keypoints,np.array([[(k+1)*(hseg-1)+j,(k+1)*(hseg-1)+j+1,  k*(hseg-1)+j+1]]),axis=0)
            elif abs (mesh_quadrilateral_upper[k][j][4]-1)<0.001:
                if (mesh_quadrilateral_upper[k][j][0]==1) and (mesh_quadrilateral_upper[k][j][1]==1):
                    face_keypoints=np.append(face_keypoints,np.array([[      k*(hseg-1)+j,k*(hseg-1)+j+1,(k+1)*(hseg-1)+j+1]]),axis=0)
                    face_keypoints=np.append(face_keypoints,np.array([[(k+1)*(hseg-1)+j+1,k*(hseg-1)+j+1,      k*(hseg-1)+j]]),axis=0)
                elif (mesh_quadrilateral_upper[k][j][2]==1) and (mesh_quadrilateral_upper[k][j][3]==1):
                    face_keypoints=np.append(face_keypoints,np.array([[      k*(hseg-1)+j,(k+1)*(hseg-1)+j,(k+1)*(hseg-1)+j+1]]),axis=0)
                    face_keypoints=np.append(face_keypoints,np.array([[(k+1)*(hseg-1)+j+1,(k+1)*(hseg-1)+j,      k*(hseg-1)+j]]),axis=0)
            elif abs(mesh_quadrilateral_upper[k][j][5]-1)<0.001:
                if (mesh_quadrilateral_upper[k][j][1]==1) and (mesh_quadrilateral_upper[k][j][2]==1):
                    face_keypoints=np.append(face_keypoints,np.array([[  k*(hseg-1)+j+1,      k*(hseg-1)+j,(k+1)*(hseg-1)+j]]),axis=0)
                    face_keypoints=np.append(face_keypoints,np.array([[(k+1)*(hseg-1)+j,      k*(hseg-1)+j,  k*(hseg-1)+j+1]]),axis=0)
                elif (mesh_quadrilateral_upper[k][j][0]==1) and (mesh_quadrilateral_upper[k][j][3]==1):
                    face_keypoints=np.append(face_keypoints,np.array([[  k*(hseg-1)+j+1, (k+1)*(hseg-1)+j+1,(k+1)*(hseg-1)+j]]),axis=0)
                    face_keypoints=np.append(face_keypoints,np.array([[(k+1)*(hseg-1)+j, (k+1)*(hseg-1)+j+1,  k*(hseg-1)+j+1]]),axis=0)
            else:
                continue
                # print("--------------tp2-----------------")

    points=mesh_point_seq
    colors = np.array([[0., 0., 0.8]]) 
    vert = [[0]*3]*len(points)
    for k in range(len(points)):
        vert[k]=[points[k][0],points[k][1],points[k][2]]
        colors=np.append(colors,np.array([[0,0,0.8]]),axis=0)
        face_keypoints[1:]
    faces = [[0]*3]*len(face_keypoints)
    for k in range(len(face_keypoints)):
        faces[k]=[face_keypoints[k][0],face_keypoints[k][1],face_keypoints[k][2]]
    # print("----vert=",vert)
    # print("----faces=",faces)
    # print("----colors=",colors)
    m=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vert),
                            o3d.utility.Vector3iVector(faces))
    m.vertex_colors=o3d.utility.Vector3dVector(colors[1:])
    ###########################  end of mesh face generation of upper part #################################

    ###########################  start of mesh face generation of down part #################################
    face_down_connection=np.append(mesh_lines_hor_down[1:],mesh_lines_ver_down[1:],axis=0)
    mesh_quadrilateral_down= np.zeros((vseg-1,hseg-1,6)) # left_side, bottom_side, right_side, top_side, diag_line; anti_diag_line of quadrilateral
    for k in range(len(face_down_connection)):
        # print("---face_down_connection[k][1]=",face_down_connection[k][1],"---face_down_connection[k][0]=",face_down_connection[k][0])
        if face_down_connection[k][1]>face_down_connection[k][0]:
            row_num_start=int(np.floor(face_down_connection[k][0]/(hseg-1)))
            col_num_start=int(np.mod(face_down_connection[k][0],(hseg-1)))
            row_num_end=int(np.floor(face_down_connection[k][1]/(hseg-1)))
            col_num_end=int(np.mod(face_down_connection[k][1],(hseg-1)))
            # print("----k=",k,"---face_down_connection=",face_down_connection[k])
            # print("--row_num_start=",row_num_start,"---col_num_start=",col_num_start,"--row_num_end=",row_num_end,"---col_num_end=",col_num_end)
            if row_num_end==row_num_start and col_num_end-col_num_start==1:
                mesh_quadrilateral_down[row_num_start][col_num_start][1]=1
                if row_num_start>=1:
                    mesh_quadrilateral_down[row_num_start-1][col_num_start][3]=1
            elif row_num_end-row_num_start==1 and col_num_end==col_num_start:
                if col_num_start>=1:
                    mesh_quadrilateral_down[row_num_start][col_num_start-1][0]=1
                mesh_quadrilateral_down[row_num_start][col_num_start][2]=1
            elif row_num_end-row_num_start==1 and col_num_end-col_num_start==1:
                mesh_quadrilateral_down[row_num_start][col_num_start][4]=1
            elif row_num_end-row_num_start==1 and col_num_start-col_num_end==1:
                mesh_quadrilateral_down[row_num_start][col_num_start-1][5]=1
        else:
            print("---------wrong point--------")

    face_keypoints_down=np.array([[0,0,0]])   
    for k in range(0,vseg-2):   
        for j in range(0,hseg-2):
            # print("----mesh_quadrilateral_upper[k][j]=",mesh_quadrilateral_upper[k][j],"--mesh_quadrilateral_upper[k][j][5]=",mesh_quadrilateral_upper[k][j][5]-1)
            if np.sum(mesh_quadrilateral_down[k][j])==4:  # all four sides of quadrilateral are valid 
                face_keypoints_down=np.append(face_keypoints_down,np.array([[  k*(hseg-1)+j+1,  (k+1)*(hseg-1)+j,    k*(hseg-1)+j]]),axis=0)
                face_keypoints_down=np.append(face_keypoints_down,np.array([[  k*(hseg-1)+j+1,(k+1)*(hseg-1)+j+1,(k+1)*(hseg-1)+j]]),axis=0)
                face_keypoints_down=np.append(face_keypoints_down,np.array([[    k*(hseg-1)+j,  (k+1)*(hseg-1)+j,  k*(hseg-1)+j+1]]),axis=0)
                face_keypoints_down=np.append(face_keypoints_down,np.array([[(k+1)*(hseg-1)+j,(k+1)*(hseg-1)+j+1,  k*(hseg-1)+j+1]]),axis=0)
            elif abs (mesh_quadrilateral_down[k][j][4]-1)<0.001:
                if (mesh_quadrilateral_down[k][j][0]==1) and (mesh_quadrilateral_down[k][j][1]==1):
                    face_keypoints_down=np.append(face_keypoints_down,np.array([[      k*(hseg-1)+j,k*(hseg-1)+j+1,(k+1)*(hseg-1)+j+1]]),axis=0)
                    face_keypoints_down=np.append(face_keypoints_down,np.array([[(k+1)*(hseg-1)+j+1,k*(hseg-1)+j+1,      k*(hseg-1)+j]]),axis=0)
                elif (mesh_quadrilateral_down[k][j][2]==1) and (mesh_quadrilateral_down[k][j][3]==1):
                    face_keypoints_down=np.append(face_keypoints_down,np.array([[      k*(hseg-1)+j,(k+1)*(hseg-1)+j,(k+1)*(hseg-1)+j+1]]),axis=0)
                    face_keypoints_down=np.append(face_keypoints_down,np.array([[(k+1)*(hseg-1)+j+1,(k+1)*(hseg-1)+j,      k*(hseg-1)+j]]),axis=0)
            elif abs(mesh_quadrilateral_down[k][j][5]-1)<0.001:
                if (mesh_quadrilateral_down[k][j][1]==1) and (mesh_quadrilateral_down[k][j][2]==1):
                    face_keypoints_down=np.append(face_keypoints_down,np.array([[  k*(hseg-1)+j+1,      k*(hseg-1)+j,(k+1)*(hseg-1)+j]]),axis=0)
                    face_keypoints_down=np.append(face_keypoints_down,np.array([[(k+1)*(hseg-1)+j,      k*(hseg-1)+j,  k*(hseg-1)+j+1]]),axis=0)
                elif (mesh_quadrilateral_down[k][j][0]==1) and (mesh_quadrilateral_down[k][j][3]==1):
                    face_keypoints_down=np.append(face_keypoints_down,np.array([[  k*(hseg-1)+j+1, (k+1)*(hseg-1)+j+1,(k+1)*(hseg-1)+j]]),axis=0)
                    face_keypoints_down=np.append(face_keypoints_down,np.array([[(k+1)*(hseg-1)+j, (k+1)*(hseg-1)+j+1,  k*(hseg-1)+j+1]]),axis=0)
            else:
                continue
                # print("--------------tp2-----------------")

    points_down=mesh_point_down_seq
    colors_down = np.array([[0., 0., 0.8]]) 
    vert_down = [[0]*3]*len(points)
    for k in range(len(points_down)):
        vert_down[k]=[points_down[k][0],points_down[k][1],points_down[k][2]]
        colors_down=np.append(colors_down,np.array([[0,0,0.8]]),axis=0)
        face_keypoints_down[1:]
    faces_down = [[0]*3]*len(face_keypoints_down)
    for k in range(len(face_keypoints_down)):
        faces_down[k]=[face_keypoints_down[k][0],face_keypoints_down[k][1],face_keypoints_down[k][2]]
    m_down=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vert_down),
                            o3d.utility.Vector3iVector(faces_down))
    m_down.vertex_colors=o3d.utility.Vector3dVector(colors_down[1:])
    ###########################  end of mesh face generation of down part #################################

    # ###########################  start of mesh face generation of connection part #################################
    # print(mesh_lines_mid_conn)   # delete the first element i.e. [0,0]
    face_mid_connection=mesh_lines_mid_conn[1:]
    # print("--mesh_points_connection_seq=",mesh_points_connection_seq,"----mesh_points_connection_seq=",len(mesh_points_connection_seq))
    # print("----face_mid_connection=",face_mid_connection)

    # print("----mesh_point=",mesh_point)
    # print("---keypoints_num=",keypoints_num)
    # print("----mesh_point_down=",mesh_point_down)
    # print("---keypoints_num_down=",keypoints_num_down) 

    mesh_points_connection=(np.append(mesh_point[0],mesh_point_down[vseg-2],axis=0)).reshape(2,(hseg-1),3)
    # print("-----mesh_points_connection=",mesh_points_connection)
    keypoints_num_connection=(np.append(keypoints_num[0],keypoints_num_down[vseg-2],axis=0)).reshape(2,(hseg-1))
    # print("-----keypoints_num_connection=",keypoints_num_connection)

    knc=keypoints_num_connection
    for k in range(hseg-1):        
        # print("----k=",k,"---keypoints_num_connection[0][k]=",keypoints_num_connection[0][k])
        if keypoints_num_connection[0][k]>1:
            knc[0][k]=1
        if keypoints_num_connection[1][k]>1:
            knc[1][k]=1
    # print("-----knc=",knc)

    face_keypoints_mid=np.array([[0,0,0]])
    # thresh=sum(knc[0][k]+knc[1][k]+knc[0][k+1]+knc[1][k+1])
    for k in range(hseg-2):
        thresh=knc[0][k]+knc[1][k]+knc[0][k+1]+knc[1][k+1]
        if thresh<=2:
            continue
        elif thresh==4:
            face_keypoints_mid=np.append(face_keypoints_mid,np.array([[k,k+1,(hseg-1)+k],[(hseg-1)+k,k+1,k]]),axis=0)
            face_keypoints_mid=np.append(face_keypoints_mid,np.array([[(hseg-1)+k,k+1,(hseg-1)+k+1],[(hseg-1)+k+1,k+1,(hseg-1)+k]]),axis=0)
    # print("---------face_keypoints_mid=",face_keypoints_mid)

    points_mid=mesh_points_connection_seq
    colors_mid = np.array([[0., 0., 0.8]]) 
    vert_mid = [[0]*3]*len(points_mid)
    for k in range(len(points_mid)):
        vert_mid[k]=[points_mid[k][0],points_mid[k][1],points_mid[k][2]]
        colors_mid=np.append(colors_mid,np.array([[0,0,0.8]]),axis=0)
    faces_mid = [[0]*3]*len(face_keypoints_mid)
    for k in range(len(face_keypoints_mid)):
        faces_mid[k]=[face_keypoints_mid[k][0],face_keypoints_mid[k][1],face_keypoints_mid[k][2]]
    # print("----vert_down=",vert_mid)
    # print("----faces_down=",faces_mid)
    # print("----colors_down=",colors_mid)
    m_mid=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vert_mid),
                                    o3d.utility.Vector3iVector(faces_mid))
    m_mid.vertex_colors=o3d.utility.Vector3dVector(colors_mid[1:])
    ###########################  end of mesh face generation of connection part #################################

    ###############################test small mesh face generation ################################
    # points=mesh_point_seq
    # vert = [[0]*3]*4
    # vert[0]=[points[8][0],points[8][1],points[8][2]]
    # vert[1]=[points[9][0],points[9][1],points[9][2]]
    # vert[2]=[points[14][0],points[14][1],points[14][2]]
    # vert[3]=[points[15][0],points[15][1],points[15][2]]
    # faces=[[0, 1, 2], [2, 1, 0],[1, 3, 2],[2, 3,1]]    
    # colors = np.array([[0., 0., 0.8],[0., 0., 0.8],[0., 0., 0.8],[0., 0., 0.8]]) 
    # print("----vert=",vert[1])
    # print("----faces=",faces)
    # print("----colors=",colors)
    # m=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vert),
    #                         o3d.utility.Vector3iVector(faces))
    # m.vertex_colors=o3d.utility.Vector3dVector(colors)
    ###########################  end of small mesh face generation #################################
    return m,m_down,m_mid