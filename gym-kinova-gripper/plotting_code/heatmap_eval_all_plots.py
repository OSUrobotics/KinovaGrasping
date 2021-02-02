import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib import pylab
import numpy as np
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from PIL import Image
import glob
import os, sys

def heatmap_freq(a,b,eval_num,plot_title,plot_name):

  title = plot_title
  cb_label = 'Frequency count of all grasp trials'

  x_range = 0.09 - (-0.09)
  y_range = 0.07

  x_bins = int(x_range / 0.002)
  y_bins = int(y_range / 0.002)

  print("x_range: ",x_range," y_range: ",y_range)
  print("x_bins: ",x_bins," y_bins: ",y_bins)

  h2, x_edges, y_edges = np.histogram2d(a, b, range=[[-0.09, 0.09], [0, 0.07]],bins=(x_bins,y_bins))

  fig, ax = plt.subplots()

  im = ax.imshow(h2.T, cmap=plt.cm.Oranges, interpolation='none', origin='lower',extent=[-0.09, 0.09, 0, 0.07])
  ax.set_aspect('equal', adjustable='box')

  fig.set_size_inches(11,8)
  #ax.grid(which='major')

  ax.xaxis.set_major_locator(MultipleLocator(0.01))
  ax.xaxis.set_minor_locator(MultipleLocator(0.001))

  ax.yaxis.set_major_locator(MultipleLocator(0.01))
  ax.yaxis.set_minor_locator(MultipleLocator(0.001))

  plt.title(title)
  plt.xlabel('X-axis initial coordinate position of object (meters)')
  plt.ylabel('Y-axis initial coordinate position of object (meters)')
  cb = plt.colorbar(mappable=im,shrink=0.6)
  cb.set_label(cb_label)

  #plt.show()
  plt.savefig("freq_plots/"+plot_name)
  plt.clf()

def heatmap_combined_fail(x_success,y_success,x_fail,y_fail,a,b,plot_title,img_name):

  title = plot_title
  cb_label = 'Success rate of grasp trials out of total trials (Negative is failure rate)'

  x_range = 0.09 - (-0.09)
  y_range = 0.07

  x_bins = int(x_range / 0.002)
  y_bins = int(y_range / 0.002)

  print("x_range: ",x_range," y_range: ",y_range)
  print("x_bins: ",x_bins," y_bins: ",y_bins)

  h, _, _ = np.histogram2d(x_success, y_success, range=[[-0.09, 0.09], [0, 0.07]],bins=(x_bins,y_bins))

  h2, x_edges, y_edges = np.histogram2d(a, b, range=[[-0.09, 0.09], [0, 0.07]],bins=(x_bins,y_bins))

  h3 = np.divide(h,h2)
  h3 = np.nan_to_num(h3)

  g, _, _ = np.histogram2d(x_fail, y_fail, range=[[-0.09, 0.09], [0, 0.07]],bins=(x_bins,y_bins))

  g2, x_edges, y_edges = np.histogram2d(a, b, range=[[-0.09, 0.09], [0, 0.07]],bins=(x_bins,y_bins))

  g3 = np.divide(g,g2)
  g3 = np.nan_to_num(g3)
  g3 = np.multiply(-1,g3)

  fig = plt.figure()
  ax = fig.add_subplot(111)

  plt.imshow(g3.T, cmap=plt.cm.RdBu,  origin='lower',extent=[-0.09, 0.09, 0, 0.07],vmin=-1, vmax=1)
  ax.set_aspect('equal', adjustable='box')

  print("h3: ",h3)
  print("g3: ",g3)

  fig.set_size_inches(11,8)

  ax.xaxis.set_major_locator(MultipleLocator(0.01))
  ax.xaxis.set_minor_locator(MultipleLocator(0.001))

  ax.yaxis.set_major_locator(MultipleLocator(0.01))
  ax.yaxis.set_minor_locator(MultipleLocator(0.001))

  plt.title(title)
  plt.xlabel('X-axis initial coordinate position of object (meters)')
  plt.ylabel('Y-axis initial coordinate position of object (meters)')
  cb = plt.colorbar(cmap=plt.cm.RdBu, format=PercentFormatter(1), shrink=0.6, ticks=[-1, -.5, 0, .5, 1])
  cb.ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
  cb.set_label(cb_label)

  plt.savefig("heatmap_plots/"+img_name)

  #plt.show()
  plt.clf()

def heatmap_combined_success(x_success,y_success,x_fail,y_fail,a,b,plot_title,img_name):

  title = plot_title
  cb_label = 'Percentage of successful grasp trials out of total trials'

  x_range = 0.09 - (-0.09)
  y_range = 0.07

  x_bins = int(x_range / 0.002)
  y_bins = int(y_range / 0.002)

  print("x_range: ",x_range," y_range: ",y_range)
  print("x_bins: ",x_bins," y_bins: ",y_bins)

  h, _, _ = np.histogram2d(x_success, y_success, range=[[-0.09, 0.09], [0, 0.07]],bins=(x_bins,y_bins))

  h2, x_edges, y_edges = np.histogram2d(a, b, range=[[-0.09, 0.09], [0, 0.07]],bins=(x_bins,y_bins))

  h3 = np.divide(h,h2)
  h3 = np.nan_to_num(h3)

  g, _, _ = np.histogram2d(x_fail, y_fail, range=[[-0.09, 0.09], [0, 0.07]],bins=(x_bins,y_bins))

  g2, x_edges, y_edges = np.histogram2d(a, b, range=[[-0.09, 0.09], [0, 0.07]],bins=(x_bins,y_bins))

  g3 = np.divide(g,g2)
  g3 = np.nan_to_num(g3)
  g3 = np.multiply(-1,g3)

  fig = plt.figure()
  ax = fig.add_subplot(111)

  plt.imshow(h3.T, cmap=plt.cm.RdBu,  origin='lower',extent=[-0.09, 0.09, 0, 0.07],vmin=-1, vmax=1)
  ax.set_aspect('equal', adjustable='box')

  fig.set_size_inches(11,8)

  ax.xaxis.set_major_locator(MultipleLocator(0.01))
  ax.xaxis.set_minor_locator(MultipleLocator(0.001))

  ax.yaxis.set_major_locator(MultipleLocator(0.01))
  ax.yaxis.set_minor_locator(MultipleLocator(0.001))

  plt.title(title)
  plt.xlabel('X-axis initial coordinate position of object (meters)')
  plt.ylabel('Y-axis initial coordinate position of object (meters)')
  cb = plt.colorbar(cmap=plt.cm.RdBu, format=PercentFormatter(1), shrink=0.6, ticks=[-1, -.5, 0, .5, 1])
  cb.ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
  cb.set_label(cb_label)

  plt.savefig("heatmap_plots/"+img_name)

  #plt.show()
  plt.clf()


def make_img_transparent(img_name):
  img = Image.open("heatmap_plots/"+img_name+".png")
  img = img.convert("RGBA")
  datas = img.getdata()

  newData = []
  for item in datas:
      if item[0] == 255 and item[1] == 255 and item[2] == 255:
          newData.append((255, 255, 255, 0))
      elif item[0] == 255 and item[1] == 253 and item[2] == 253:
          newData.append((255, 255, 255, 0))
      else:
          newData.append(item)

  img.putdata(newData)
  img.save("heatmap_plots/"+img_name+".png", "PNG",transparent=True)


def create_plots(success_datax,success_datay,fail_datax,fail_datay,total_datax,total_datay,eval_num):
    shapes = "CubeS"

    freq_plot_title = "Grasp Trial Frequency per Initial Coordinate Position of Object ("+shapes+") at Eval "+str(eval_num)
    heatmap_freq(total_datax,total_datay,eval_num,freq_plot_title,'freq_heatmap_'+str(eval_num)+'.jpg')

    fail_plot_title = "Grasp Trial Success Rate per Initial Coordinate Position of Object ("+shapes+") at Eval "+str(eval_num)
    heatmap_combined_fail(success_datax,success_datay,fail_datax,fail_datay,total_datax,total_datay,fail_plot_title,'fail_heatmap_'+str(eval_num)+'.png')

    success_plot_title = "Grasp Trial Success Rate per Initial Coordinate Position of Object ("+shapes+") at Eval "+str(eval_num)
    img_name = 'success_heatmap_'+str(eval_num)
    heatmap_combined_success(success_datax,success_datay,fail_datax,fail_datay,total_datax,total_datay,success_plot_title,'success_heatmap_'+str(eval_num)+'.png')
    make_img_transparent(img_name)

def convert_to_local_coords(Tfw,obj_coords):
    obj_local_coords = np.empty(1)
    for coord in obj_coords:
        obj_local = np.append(coord, 1)
        obj_local = np.matmul(Tfw, obj_local)
        obj_local_pos = obj_local[0:3]
        obj_local_coords = np.append(obj_local_pos,obj_local)
    return obj_local_coords

def get_1000_eval():
    ep_num = 999 #1000
    success_datax = []
    success_datay = []
    fail_datax = []
    fail_datay = []
    total_datax = []
    total_datay = []
    j = 0
    eval_num = 999 #1000

    # for each evaluation after 100 episodes
    for i in range(10): #20
        # if at the next 1000 episode
        print("eval_num+1000: ",eval_num+1000)
        if ep_num == eval_num+1000:
            print("Evaluate data at "+str(ep_num)+" episodes")
            create_plots(success_datax,success_datay,fail_datax,fail_datay,total_datax,total_datay,eval_num)

            success_datax = []
            success_datay = []
            fail_datax = []
            fail_datay = []
            total_datax = []
            total_datay = []
            j += 1
            eval_num += 1000

        success_file_x = "heatmap_eval_success"+str(ep_num)+"_x_arr.npy"
        print("success_file_x: ",success_file_x)
        success_file_y = "heatmap_eval_success"+str(ep_num)+"_y_arr.npy"
        #print("success_file_y: ",success_file_y)

        fail_file_x = "heatmap_eval_fail"+str(ep_num)+"_x_arr.npy"
        #print("fail_file_x: ",fail_file_x)
        fail_file_y = "heatmap_eval_fail"+str(ep_num)+"_y_arr.npy"
        #print("fail_file_y: ",fail_file_y)

        total_file_x = "heatmap_eval_total"+str(ep_num)+"_x_arr.npy"
        #print("total_file_x: ",total_file_x)
        total_file_y = "heatmap_eval_total"+str(ep_num)+"_y_arr.npy"
        #print("total_file_y: ",total_file_y)

        # "heatmap_eval_success"+str(ep_num)+"_x_arr.npy"
        success_tmpx = np.load(success_file_x)
        success_tmpy = np.load(success_file_y)

        fail_tmpx = np.load(fail_file_x)
        fail_tmpy = np.load(fail_file_y)

        total_tmpx = np.load(total_file_x)
        total_tmpy = np.load(total_file_y)

        if len(success_tmpx) > 0:
            success_datax = np.append(success_datax,success_tmpx)
            success_datay = np.append(success_datay,success_tmpy)

        if len(fail_tmpx) > 0:
            fail_datax = np.append(fail_datax,fail_tmpx)
            fail_datay = np.append(fail_datay,fail_tmpy)

        if len(total_tmpx) > 0:
            total_datax = np.append(total_datax,total_tmpx)
            total_datay = np.append(total_datay,total_tmpy)
        ep_num += 1000

def get_heatmap_data(data_dir,filename,tot_episodes,saving_freq):
    """Get boxplot data from saved numpy arrays
    data_dir: Directory location of data file
    filename: Name of data file
    tot_episodes: Total number of evaluation episodes
    saving_freq: Frequency in which the data files were saved
    """
    data = []
    for ep_num in np.linspace(start=saving_freq, stop=tot_episodes, num=int(tot_episodes/saving_freq), dtype=int):
        data_str = data_dir+filename+"_"+str(ep_num)+".npy"
        print("Eval file: ", data_str)
        data_file = np.load(data_str)[0]

        for eval_data in data_file:
            data.append(eval_data)

    return data

def main():
    heatmap_saving_dir = "./heatmap_plots"
    if not os.path.isdir(heatmap_saving_dir):
        os.mkdir(heatmap_saving_dir)

    freq_saving_dir = "./freq_plots"
    if not os.path.isdir(freq_saving_dir):
        os.mkdir(freq_saving_dir)

    # Code to test with
    data_dir = "./eval_plots/heatmap/"
    #file_list = ["finger_reward", "grasp_reward", "lift_reward", "total_reward"]
    tot_episodes = 2000
    saving_freq = 1000
    eval_freq = 200

    success_x = get_heatmap_data(data_dir, "eval_success_x", tot_episodes, saving_freq)
    success_y = get_heatmap_data(data_dir, "eval_success_y", tot_episodes, saving_freq)
    fail_x = get_heatmap_data(data_dir, "eval_fail_x", tot_episodes, saving_freq)
    fail_y = get_heatmap_data(data_dir, "eval_fail_y", tot_episodes, saving_freq)
    total_x = get_heatmap_data(data_dir, "eval_total_x", tot_episodes, saving_freq)
    total_y = get_heatmap_data(data_dir, "eval_total_y", tot_episodes, saving_freq)

    print("np.asarray(success_x).shape: ",np.asarray(success_x).shape)

    #for i in freq_vals:
    #    create_plots(success_x[i], success_y[i], fail_x[i], fail_y[i], total_x[i], total_y[i], ep_num)

    # Get numpy arrays and produce plots every 1000 episodes
    #get_1000_eval()

main()
