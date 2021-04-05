#!/usr/bin/env python3

import csv
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from sklearn.cluster import DBSCAN
from pathlib import Path

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
    if y_list == False:
        flagg_list = False
    else:
        flagg_list = True
    if type(x_list) == np.ndarray:
        for idx in range(len(idx_to_pop)):
            idx = idx_to_pop[idx]
            if number_of_points_removed > 0:
                idx = idx -  number_of_points_removed
            x_list = np.delete(x_list, idx, axis=0)
            if flagg_list:
                y_list = np.delete(y_list, idx, axis=0)
            number_of_points_removed += 1
    elif type(x_list) == list:
        for idx in range(len(idx_to_pop)):
            idx = idx_to_pop[idx]
            if number_of_points_removed > 0:
                idx = idx - number_of_points_removed
            x_list.pop(idx)
            if flagg_list:
                y_list.pop(idx)
            number_of_points_removed += 1 
    else:
        print("Data Type Error While Poping Index")
        quit()   
    print("Total number of Outlier Points Dropped {}".format(number_of_points_removed))
    return x_list, y_list


def boundary_data_analysis(points_list, orientation_list, status_list, clustering_radius = 0.005, minimum_points_to_form_cluster = 20, plot_saving_dir = './', orientation_type = 'normal', plot_ref_x = 0, plot_ref_y = 0.0233, show = True, save = False):
    legend = ["Success region"]
    legend1 = ["Failure region"]
    legend2 = ["Grey region"]

    if orientation_type == 'top':
        Base_rotation = np.array([0, 0, 0])
    elif orientation_type == 'side':
        Base_rotation = np.array([-1.2, 0, 0])
    else:
        Base_rotation = np.array([-1.57, 0, -1.57])

    if minimum_points_to_form_cluster < 5:
        print("Not working")
        quit()

    plot_heatmap_Sucess_X = []
    plot_heatmap_Sucess_Y = []
    plot_heatmap_Fail_X = []
    plot_heatmap_Fail_Y = []

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
            plot_heatmap_Sucess_X.append(points_list[idx][0])
            plot_heatmap_Sucess_Y.append(points_list[idx][1])
            idx_suc.append(idx)
        else:
            fail_x_bf.append(points_list[idx][0])
            fail_y_bf.append(points_list[idx][1])
            idx_fail.append(idx)
            plot_heatmap_Fail_X.append(points_list[idx][0])
            plot_heatmap_Fail_Y.append(points_list[idx][1])

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
    print("Performing Erosion Dilation on Success Points")
    idx_to_pop_list = erosion_dilation(success_x_bf, success_y_bf, outlier_radius = clustering_radius, minimum_points_to_form_cluster = minimum_points_to_form_cluster)
    Current_length = len(success_x_bf)
    print("Position Points")
    success_x_bf, success_y_bf = idx_to_pop(idx_to_pop_list, success_x_bf, success_y_bf)
    if len(success_x_bf) != Current_length - len(idx_to_pop_list):
        print("Erosion Dilation failed reducing minimum number of points required to form a cluster")
        boundary_data_analysis(points_list, orientation_list, status_list, minimum_points_to_form_cluster = minimum_points_to_form_cluster/2, plot_saving_dir=plot_saving_dir, save=save, show=show)
        quit()
    print("Orientation Points")
    orientation_suc_bf = idx_to_pop(idx_to_pop_list, orientation_suc_bf)
    print("Performing Erosion Dilation on Failure Points")
    idx_to_pop_list = erosion_dilation(fail_x_bf, fail_y_bf, outlier_radius = clustering_radius, minimum_points_to_form_cluster = minimum_points_to_form_cluster)
    Current_length = len(fail_x_bf)
    print("Position Points")
    fail_x_bf, fail_y_bf = idx_to_pop(idx_to_pop_list, fail_x_bf, fail_y_bf)
    if len(fail_x_bf) != Current_length - len(idx_to_pop_list):
        print("Erosion Dilation failed reducing minimum number of points required to form a cluster")
        boundary_data_analysis(points_list, orientation_list, status_list, minimum_points_to_form_cluster = minimum_points_to_form_cluster/2, plot_saving_dir=plot_saving_dir, save=save, show=show)
        quit()
    print("Orienatation Points")
    orientation_fail_bf = idx_to_pop(idx_to_pop_list, orientation_fail_bf)

    total_number_of_points = len(success_x_bf) + len(fail_x_bf)
    for val_idx in range(total_number_of_points):
        if val_idx < len(success_x_bf):
            point_cord = np.array([success_x_bf[val_idx], success_y_bf[val_idx]])
        else:
            point_cord = np.array([fail_x_bf[val_idx-len(success_x_bf)], fail_y_bf[val_idx-len(success_x_bf)]])
        num_of_success = 0
        num_of_failure = 0
        neighbor_list = []

        for val_idx2 in range(total_number_of_points):
            if val_idx2 < len(success_x_bf):
                point_temp = np.array([success_x_bf[val_idx2], success_y_bf[val_idx2]])
            else:
                point_temp = np.array([fail_x_bf[val_idx2-len(success_x_bf)], fail_y_bf[val_idx2-len(success_x_bf)]])
            neighbor_dist = Two_point_euclidean_distance_calculator(point_cord, point_temp)
            if neighbor_dist < clustering_radius/5:
                if val_idx2 < len(success_x_bf):
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
    clustering = DBSCAN(eps=clustering_radius, min_samples=minimum_points_to_form_cluster).fit(Grey_area_bin)
    if max(clustering.labels_) < 1:
        clustering = DBSCAN(eps=clustering_radius, min_samples=minimum_points_to_form_cluster/2).fit(Grey_area_bin)
    cluster_number = max(clustering.labels_) + 1
    print("Number of Clusters found: {} in Transisiton Stage".format(cluster_number))

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

    test_x = []
    test_y = []
    testf_x = []
    testf_y = []
    for idx in range(total_number_of_points):
        orientation = orientation_list[idx][:]
        if idx < len(success_x_bf):
            x = success_x_bf[idx]
            y = success_y_bf[idx]
        else:
            x = fail_x_bf[idx-len(success_x_bf)]
            y = fail_y_bf[idx-len(success_x_bf)]

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

        Quaternion_distance = np.rad2deg(np.arccos(Quaternion_dot_product_of_2_points(Base_rotation, rotation)))

        if x < 0:
            Translation_error = -Translation_error
        
        if idx < len(success_x_bf):
            plot_succes_x.append(Translation_error)
            plot_succes_y.append(Quaternion_distance)
            test_x.append(x)
            test_y.append(y)

        else:       
            plot_fail_x.append(Translation_error)
            plot_fail_y.append(Quaternion_distance)
            testf_x.append(x)
            testf_y.append(y)
    

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

            Quaternion_distance = np.rad2deg(np.arccos(Quaternion_dot_product_of_2_points(Base_rotation, rotation)))

            if x < 0:
                Translation_error = -Translation_error
            
            ConvexHull_points_x.append(Translation_error)
            ConvexHull_points_y.append(Quaternion_distance)
    plt.rcParams.update({'font.size': 15})
    plt.figure("After_Erosion_Dilation_Heatmap", figsize=(7.5,7.5))
    plt.scatter(test_x, test_y, color='blue')
    plt.scatter(testf_x, testf_y, color='red')
    plt.legend([legend, legend1])
    plt.xlabel("X-Axis")
    plt.ylabel("Y-Axis")
    plt.xlim((-0.1, 0.1))
    plt.title("After_Erosion_Dilation_Heatmap")
    if save:
        plt.savefig(plot_saving_dir + "Test_Heatmap.png")

    plt.figure("Clustered Points", figsize=(7.5,7.5))
    plt.scatter(Success_bin_x,   Success_bin_y, color='blue')
    plt.scatter(Failure_bin_x,   Failure_bin_y, color='red')
    plt.scatter(Grey_area_bin_x, Grey_area_bin_y, color='#9c30c7')
    plt.legend([legend, legend1, legend2])
    for key in Cluster_points:
        points_cluster = Cluster_points[key]
        hull = ConvexHull(points_cluster)
        for simplex in hull.simplices:
            plt.plot(points_cluster[simplex, 0], points_cluster[simplex, 1], 'k-')
    plt.xlabel("X-Axis")
    plt.ylabel("Y-Axis")
    plt.xlim((-0.1, 0.1))
    plt.title("Clustered Points")
    if save:
        plt.savefig(plot_saving_dir + "Clustered_Points.png")

    plt.figure("Boundary", figsize=(7.5,7.5))
    plt.scatter(plot_succes_x, plot_succes_y, color='blue')
    plt.scatter(plot_fail_x, plot_fail_y, color='red')
    plt.scatter(ConvexHull_points_x, ConvexHull_points_y, color='#9c30c7')
    plt.xlabel("Position Noise")
    plt.ylabel("Orientation noise in deg")
    plt.title("Analysis plot")
    plt.xlim((-0.1, 0.1))
    plt.legend([legend, legend1, legend2])
    if save:
        plt.savefig(plot_saving_dir + "Boundary.png")

    plt.figure("Original_Heatmap", figsize=(7.5,7.5))
    plt.scatter(plot_heatmap_Sucess_X, plot_heatmap_Sucess_Y, color='blue')
    plt.scatter(plot_heatmap_Fail_X, plot_heatmap_Fail_Y, color='red')
    plt.legend([legend, legend1])
    for key in Cluster_points:
        points_cluster = Cluster_points[key]
        hull = ConvexHull(points_cluster)
        for simplex in hull.simplices:
            plt.plot(points_cluster[simplex, 0], points_cluster[simplex, 1], 'k-')
    plt.xlabel("X-Axis")
    plt.ylabel("Y-Axis")
    plt.xlim((-0.1, 0.1))
    plt.title("Original_Heatmap")
    if save:
        plt.savefig(plot_saving_dir + "OG_Heatmap.png")


    plt.figure("Boundary without Boundary Points", figsize=(7.5,7.5))
    plt.scatter(plot_succes_x, plot_succes_y, color='blue')
    plt.scatter(plot_fail_x, plot_fail_y, color='red')
    plt.xlabel("Position Noise")
    plt.ylabel("Orientation noise in deg")
    plt.title("Analysis plot without Boundary Points")
    plt.legend([legend, legend1])
    plt.xlim((-0.1, 0.1))
    if save:
        plt.savefig(plot_saving_dir + "Boundary_without_Convex_Hull.png")
    
    if show:
        plt.show()



if __name__ == "__main__":
    # print(Quaternion_dot_product_of_2_points([15,15,15], [0,0,0], degree=True))# 0.9972303739988352
    # quit() #0.9767773152319457
    shapes_key = ["CubeS"]#, "CubeB"]
    #PID = ["naive", "expert"]
    orientation1 = ["normal"]
    for shape in shapes_key:
        filepath = "./Data_with_noise/"+str(shape)+"/normal/expert/"
        point_list = np.load(filepath+"object_cord.npy", allow_pickle=True)
        orr_list = np.load(filepath+"orientation_list.npy", allow_pickle=True)
        status_list = np.load(filepath+"lift_success_list.npy", allow_pickle=True)
        coord_save_path = Path(filepath+"plot/")
        coord_save_path.mkdir(parents=True, exist_ok=True)
        boundary_data_analysis(point_list, orr_list, status_list, plot_saving_dir=filepath+"plot/", save=True, show=False)