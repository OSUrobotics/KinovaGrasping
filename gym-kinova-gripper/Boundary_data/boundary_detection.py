#!/usr/bin/env python3

import csv
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from sklearn.cluster import DBSCAN


def Two_point_euclidean_distance_calculator(point1, point2):
    if len(point1) != len(point2):
        print("Points Dimension Error")
        quit()
    else:
        Distance = 0
        for i in range(len(point1)):
            Distance += (point1[i] - point2[i])**2
        Distance = np.sqrt(Distance)
        return Distance


def Quaternion_dot_product_of_2_points(point1, point2, degree=False):
    if len(point1) != len(point2):
        print("Points Dimension Error")
        quit()
    else:
        if len(point1) == 3:
            # Create a rotation object from Euler angles specifying axes of rotation
            point1_rot = Rotation.from_euler('xyz', point1, degrees=degree)
            point2_rot = Rotation.from_euler('xyz', point2, degrees=degree)
            
            # Convert to quaternions 
            point1_quat = point1_rot.as_quat()
            point2_quat = point2_rot.as_quat()

        else:
            point1_quat = point1
            point2_quat = point2

        Distance = 0
        for i in range(len(point1_quat)):
            Distance += point1_quat[i]*point2_quat[i]
        return Distance


def erosion_dilation(x_list, y_list, outlier_radius = 0.005, minimum_points_to_form_cluster = 10):
    Original_length = len(x_list)
    idx_to_pop = []
    for idx in range(len(x_list)):
        x = x_list[idx]
        y = y_list[idx]

        neigh = 0
        for idx2 in range(len(x_list)):
            x_temp = x_list[idx2]
            y_temp = y_list[idx2]

            temp_dist = Two_point_euclidean_distance_calculator([x, y], [x_temp, y_temp])
            if temp_dist < outlier_radius:
                neigh += 1

        if neigh < minimum_points_to_form_cluster:
            idx_to_pop.append(idx)
    return idx_to_pop
    

def idx_to_pop(idx_to_pop, x_list, y_list=False):
    if len(idx_to_pop) == len(x_list):
        return x_list, y_list
    number_of_points_removed = 0
    print("Total number of Points: {}".format(len(x_list)))
    if type(x_list) == np.ndarray:
        for idx in range(len(idx_to_pop)):
            if number_of_points_removed > 0:
                idx -= number_of_points_removed
            x_list = np.delete(x_list, idx_to_pop[idx], axis=0)
            if y_list != False:
                y_list = np.delete(y_list, idx_to_pop[idx], axis=0)
            number_of_points_removed += 1
    elif type(x_list) == list:
        for idx in range(len(idx_to_pop)):
            if number_of_points_removed > 0:
                idx -= number_of_points_removed
            x_list.pop(idx_to_pop[idx])
            if y_list != False:
                y_list.pop(idx_to_pop[idx])
            number_of_points_removed += 1 
    else:
        print("Data Type Error While Poping Index")
        quit()   
    print("Total number of Outlier Points Dropped {}".format(number_of_points_removed))
    return x_list, y_list


def boundary_data_analysis(points_list, orientation_list, status_list, clustering_radius = 0.003, minimum_points_to_form_cluster = 10, plot_saving_dir = './', orientation_type = 'normal', plot_ref_x = 0, plot_ref_y = 0.035, show = True, save = False):
    legend = ["Success"]
    legend1 = ["Failure"]
    legend2 = ["Transition stage"]

    if orientation_type == 'top':
        Base_rotation = np.array([0, 0, 0])
    elif orientation_type == 'side':
        Base_rotation = np.array([-1.2, 0, 0])
    else:
        Base_rotation = np.array([-1.57, 0, -1.57])

    plot_heatmap_Sucess_X = []
    plot_heatmap_Sucess_y = []
    plot_heatmap_Fail_X = []
    plot_heatmap_Fail_y = []

    plot_succes_x = []
    plot_succes_y = []
    plot_fail_x = []
    plot_fail_y = []

    Success_bin_x   = []
    Failure_bin_x   = []
    Grey_area_bin_x = []

    Success_bin_y   = []
    Failure_bin_y   = []
    Grey_area_bin_y = []

    ConvexHull_points_x = []
    ConvexHull_points_y = []

    orientation_idx_grey_araea = []

    success_x_bf = []
    success_y_bf = []
    fail_x_bf = []
    fail_y_bf = []
    idx_suc = []
    idx_fail = []

    for idx in range(len(status_list)):
        if status_list[idx] == 1:
            success_x_bf.append(points_list[idx][0])
            success_y_bf.append(points_list[idx][1])
            idx_suc.append(idx)
        else:
            fail_x_bf.append(points_list[idx][0])
            fail_y_bf.append(points_list[idx][1])
            idx_fail.append(idx)

    orientation_suc_bf = np.zeros((len(idx_suc), 3))
    for i in range(len(idx_suc)):
        orientation_suc_bf[i][0] = orientation_list[idx_suc[i]][0]
        orientation_suc_bf[i][1] = orientation_list[idx_suc[i]][1]
        orientation_suc_bf[i][2] = orientation_list[idx_suc[i]][2]

    orientation_fail_bf = np.zeros((len(idx_fail), 3))
    for i in range(len(idx_fail)):
        orientation_fail_bf[i][0] = orientation_list[idx_fail[i]][0]
        orientation_fail_bf[i][1] = orientation_list[idx_fail[i]][1]
        orientation_fail_bf[i][2] = orientation_list[idx_fail[i]][2]


    plt.figure("Original Heatmap")
    plt.scatter(success_x_bf, success_y_bf)
    plt.scatter(fail_x_bf, fail_y_bf)

    idx_to_pop_list = erosion_dilation(success_x_bf, success_y_bf, outlier_radius = clustering_radius, minimum_points_to_form_cluster = minimum_points_to_form_cluster)
    Current_length = len(success_x_bf)
    success_x_bf, success_y_bf = idx_to_pop(idx_to_pop_list, success_x_bf, success_y_bf)
    if len(success_x_bf) != Current_length - len(idx_to_pop_list):
        boundary_data_analysis(points_list, orientation_list, status_list, minimum_points_to_form_cluster = minimum_points_to_form_cluster/2)
        quit()
    orientation_suc_bf = idx_to_pop(idx_to_pop_list, orientation_suc_bf)
    idx_to_pop_list = erosion_dilation(fail_x_bf, fail_y_bf, outlier_radius = clustering_radius, minimum_points_to_form_cluster = minimum_points_to_form_cluster)
    Current_length = len(fail_x_bf)
    fail_x_bf, fail_y_bf = idx_to_pop(idx_to_pop_list, fail_x_bf, fail_y_bf)
    if len(fail_x_bf) != Current_length - len(idx_to_pop_list):
        boundary_data_analysis(points_list, orientation_list, status_list, minimum_points_to_form_cluster = minimum_points_to_form_cluster/2)
        quit()
    orientation_fail_bf = idx_to_pop(idx_to_pop_list, orientation_fail_bf)

    status_list = []
    for _ in range(len(success_x_bf)):
        status_list.append(1)
    for _ in range(len(fail_x_bf)):
        status_list.append(0)

    
    points_list = np.zeros((len(status_list), 2))
    counter = 0
    for _ in range(len(success_x_bf)):
        points_list[counter][0] = success_x_bf[counter]
        points_list[counter][1] = success_y_bf[counter]
        counter += 1

    for i in range(len(fail_x_bf)):
        points_list[counter][0] = fail_x_bf[i]
        points_list[counter][1] = fail_y_bf[i]
        counter += 1

    for val_idx in range(len(status_list)):
        point_cord = points_list[val_idx]
        num_of_success = 0
        num_of_failure = 0
        neighbor_list = []

        for val_idx2 in range(len(status_list)):
            point_temp = points_list[val_idx2][:]
            neighbor_dist = Two_point_euclidean_distance_calculator(point_cord, point_temp)
            if neighbor_dist < clustering_radius:
                if status_list[val_idx2] == 1:
                    num_of_success += 1
                else:
                    num_of_failure += 1

        if num_of_failure == 0:
            Success_bin_x.append(point_cord[0])
            Success_bin_y.append(point_cord[1])           
        elif num_of_success == 0:
            Failure_bin_x.append(point_cord[0])
            Failure_bin_y.append(point_cord[1])
        else:
            Grey_area_bin_x.append(point_cord[0])
            Grey_area_bin_y.append(point_cord[1])
            orientation_idx_grey_araea.append(val_idx)

    orientation_idx_grey = np.zeros((len(orientation_idx_grey_araea), 3))
    for i in range(len(orientation_idx_grey_araea)):
        orientation_idx_grey[i][0] = orientation_list[orientation_idx_grey_araea[i]][0]
        orientation_idx_grey[i][1] = orientation_list[orientation_idx_grey_araea[i]][1]
        orientation_idx_grey[i][2] = orientation_list[orientation_idx_grey_araea[i]][2]

    Grey_area_bin = np.zeros((len(Grey_area_bin_x), 2))
    for i in range(len(Grey_area_bin_x)):
        Grey_area_bin[i][0] = Grey_area_bin_x[i]
        Grey_area_bin[i][1] = Grey_area_bin_y[i]

    Cluster_points = {}
    orientation_points = {}
    clustering = DBSCAN(eps=clustering_radius*10, min_samples=minimum_points_to_form_cluster/5).fit(Grey_area_bin)
    cluster_number = max(clustering.labels_) + 1
    print("Number of Clusters found: {}".format(cluster_number))

    for number in range(cluster_number):
        temp_point_list_x = []
        temp_point_list_y = []
        temp_orr_list = []
        for cluster_num in range(len(Grey_area_bin)):
            if clustering.labels_[cluster_num] == number:
                temp_point_list_x.append(Grey_area_bin[cluster_num][0])
                temp_point_list_y.append(Grey_area_bin[cluster_num][1])
                temp_orr_list.append(cluster_num)
        array_of_orrr = np.zeros((len(temp_orr_list), 3))
        array_of_points = np.zeros((len(temp_point_list_x), 2))
        for idx in range(len(temp_point_list_x)):
            array_of_points[idx][0] = temp_point_list_x[idx]
            array_of_points[idx][1] = temp_point_list_y[idx]
            array_of_orrr[idx][0]   = orientation_idx_grey[temp_orr_list[idx]][0]
            array_of_orrr[idx][1]   = orientation_idx_grey[temp_orr_list[idx]][1]
            array_of_orrr[idx][2]   = orientation_idx_grey[temp_orr_list[idx]][2]

        Cluster_points[number] = array_of_points
        orientation_points[number] = array_of_orrr
    for idx in range(len(status_list)):
        orientation = orientation_list[idx][:]
        x = float(points_list[idx][0])
        y = float(point_list[idx][1])

        Translation_error = Two_point_euclidean_distance_calculator([x, y], [plot_ref_x, plot_ref_y])
        
        rotation = np.zeros(3)
        if max(abs(orientation)) > 1:
            rotation[0] = float(orientation[0])
            rotation[1] = float(orientation[1])
            rotation[2] = float(orientation[2])
        else:
            rotation[0] = Base_rotation[0] + float(orientation[0])
            rotation[1] = Base_rotation[1] + float(orientation[1])
            rotation[2] = Base_rotation[2] + float(orientation[2])

        Quaternion_distance = Quaternion_dot_product_of_2_points(Base_rotation, rotation)

        if x < 0:
            Translation_error = -Translation_error
        
        if status_list[idx] == 1:
            plot_succes_x.append(Translation_error)
            plot_succes_y.append(Quaternion_distance)
            plot_heatmap_Sucess_X.append(x)
            plot_heatmap_Sucess_y.append(y)
        else:       
            plot_fail_x.append(Translation_error)
            plot_fail_y.append(Quaternion_distance)
            plot_heatmap_Fail_X.append(x)
            plot_heatmap_Fail_y.append(y)

    

    for key in Cluster_points:
        points_cluster = Cluster_points[key]
        orientation_idx_grey = orientation_points[key]
        for i in range(len(points_cluster)):
            orientation = orientation_idx_grey[i]
            x = float(points_cluster[i][0])
            y = float(points_cluster[i][1])

            Translation_error = Two_point_euclidean_distance_calculator([x, y], [plot_ref_x, plot_ref_y])
            
            rotation = np.zeros(3)
            if max(abs(orientation)) > 1:
                rotation[0] = float(orientation[0])
                rotation[1] = float(orientation[1])
                rotation[2] = float(orientation[2])
            else:
                rotation[0] = Base_rotation[0] + float(orientation[0])
                rotation[1] = Base_rotation[1] + float(orientation[1])
                rotation[2] = Base_rotation[2] + float(orientation[2])

            Quaternion_distance = Quaternion_dot_product_of_2_points(Base_rotation, rotation)

            if x < 0:
                Translation_error = -Translation_error
            
            ConvexHull_points_x.append(Translation_error)
            ConvexHull_points_y.append(Quaternion_distance)


    plt.figure("Clustered Points")
    plt.scatter(Success_bin_x,   Success_bin_y)
    plt.scatter(Failure_bin_x,   Failure_bin_y)
    plt.scatter(Grey_area_bin_x, Grey_area_bin_y)
    plt.legend([legend, legend1, legend2])
    for key in Cluster_points:
        points_cluster = Cluster_points[key]
        hull = ConvexHull(points_cluster)
        for simplex in hull.simplices:
            plt.plot(points_cluster[simplex, 0], points_cluster[simplex, 1], 'k-')

    plt.xlabel("X-Axis")
    plt.ylabel("Y-Axis")
    plt.title("Clustered Points")
    if save:
        plt.savefig(plot_saving_dir + "Clustered_Points.png")

    plt.figure("Boundary")
    plt.scatter(plot_succes_x, plot_succes_y)
    plt.scatter(plot_fail_x, plot_fail_y)
    plt.scatter(ConvexHull_points_x, ConvexHull_points_y)
    plt.xlabel("Translation Error")
    plt.ylabel("Rotational Error")
    plt.title("Boundary")
    plt.legend([legend, legend1, legend2])
    if save:
        plt.savefig(plot_saving_dir + "Boundary.png")

    plt.figure("Heatmap")
    plt.scatter(plot_heatmap_Sucess_X, plot_heatmap_Sucess_y)
    plt.scatter(plot_heatmap_Fail_X, plot_heatmap_Fail_y)
    legend = ["Success"]
    legend1 = ["Failure"]
    plt.legend([legend, legend1])
    for key in Cluster_points:
        points_cluster = Cluster_points[key]
        hull = ConvexHull(points_cluster)
        for simplex in hull.simplices:
            plt.plot(points_cluster[simplex, 0], points_cluster[simplex, 1], 'k-')
    plt.xlabel("X-Axis")
    plt.ylabel("Y-Axis")
    plt.title("Heatmap")
    if save:
        plt.savefig(plot_saving_dir + "Heatmap.png")


    plt.figure("Boundary without Convex Hull")
    plt.scatter(plot_succes_x, plot_succes_y)
    plt.scatter(plot_fail_x, plot_fail_y)
    plt.xlabel("Translation Error (Euclidean distance)")
    plt.ylabel("Rotational Error (Quaternion dot product)")
    plt.title("Boundary without Convex Hull")
    plt.legend([legend, legend1])
    if save:
        plt.savefig(plot_saving_dir + "Boundary_without_Convex_Hull.png")
    
    if show:
        plt.show()



if __name__ == "__main__":

    tot_x = np.load("total_x.npy", allow_pickle=True)
    s_x = np.load("success_x.npy", allow_pickle=True)
    s_y = np.load("success_y.npy", allow_pickle=True)
    f_x = np.load("fail_x.npy", allow_pickle=True)
    f_y = np.load("fail_y.npy", allow_pickle=True)
    idx = np.load("orientation_indexes.npy", allow_pickle=True)

    coords_filename = "BottleS_norm_rotation.txt"
    rot_data = []
    with open(coords_filename) as csvfile:
        checker=csvfile.readline()
        if ',' in checker:
            delim=','
        else:
            delim=' '
        reader = csv.reader(csvfile, delimiter=delim)
        for i in reader:
            k = []
            for a in range(len(i)):
                if i[a] == ' ' or i[a] == '':
                    continue
                k.append(i[a]) 
            i = k
            rot_data.append([float(i[0]), float(i[1]), float(i[2])])

    status_list = []
    for _ in range(len(s_x)):
        status_list.append(1)
    for _ in range(len(f_x)-1):
        status_list.append(0)

    total_x = []
    for idd in range(len(s_x)):
        total_x.append(s_x[idd])
    for idd in range(len(f_x)):
        total_x.append(f_x[idd])    

    total_y = []
    for idd in range(len(s_y)):
        total_y.append(s_y[idd])
    for idd in range(len(f_y)):
        total_y.append(f_y[idd])
    point_list = np.zeros((len(total_x), 2))
    for i in range(len(total_x)):
        point_list[i][0] = total_x[i]
        point_list[i][1] = total_y[i]
    orr_list = np.zeros((len(idx), 3))
    for i in range(len(idx)):
        orr_list[i][0] = idx[i]
        orr_list[i][1] = idx[i]
        orr_list[i][2] = idx[i]

    boundary_data_analysis(point_list, orr_list, status_list)