import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib import pylab
import numpy as np
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from PIL import Image
import glob
import os

def heatmap_freq(a,b,plot_title,plot_name):

  title = plot_title
  cb_label = 'Frequency count of all grasp trials'

  x_range = 0.09 - (-0.09)
  y_range = 0.07

  x_bins = int(x_range / 0.002)
  y_bins = int(y_range / 0.002)

  print("x_range: ",x_range," y_range: ",y_range)
  print("x_bins: ",x_bins," y_bins: ",y_bins)

  h2, x_edges, y_edges = np.histogram2d(a, b, range=[[-.09, 0.09], [0, 0.07]],bins=(x_bins,y_bins))

  fig, ax = plt.subplots()

  im = ax.imshow(h2.T, cmap=plt.cm.Oranges, interpolation='none', origin='lower',extent=[-.09, 0.09, 0, 0.07])
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

  h, _, _ = np.histogram2d(x_success, y_success, range=[[-.09, 0.09], [0, 0.07]],bins=(x_bins,y_bins))

  h2, x_edges, y_edges = np.histogram2d(a, b, range=[[-.09, 0.09], [0, 0.07]],bins=(x_bins,y_bins))

  h3 = np.divide(h,h2)
  h3 = np.nan_to_num(h3)

  g, _, _ = np.histogram2d(x_fail, y_fail, range=[[-.09, 0.09], [0, 0.07]],bins=(x_bins,y_bins))

  g2, x_edges, y_edges = np.histogram2d(a, b, range=[[-.09, 0.09], [0, 0.07]],bins=(x_bins,y_bins))

  g3 = np.divide(g,g2)
  g3 = np.nan_to_num(g3)
  g3 = np.multiply(-1,g3)

  fig = plt.figure()
  ax = fig.add_subplot(111)

  ''' Create your own colorbar
  cdict = {
    'red': ((0.0, 0.25, .25), (0.02, .59, .59), (1., 1., 1.)),
    'green': ((0.0, 0.0, 0.0), (0.02, .45, .45), (1., .97, .97)),
    'blue': ((0.0, 1.0, 1.0), (0.02, .75, .75), (1., 0.45, 0.45))
  }
  #cm = mpl.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)
  '''
  
  plt.imshow(g3.T, cmap=plt.cm.seismic,  origin='lower',extent=[-.09, 0.09, 0, 0.07],vmin=-1, vmax=1) #blues, failure
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
  #plt.cm.seismic
  #norm=mpl.colors.Normalize(vmin=-1, vmax=-.5),
  cb = plt.colorbar(cmap=plt.cm.Blues,format=PercentFormatter(1),shrink=0.6)
  cb.set_label(cb_label)
  #cb.set_clim(-1, -.5)

  plt.savefig("heatmap_plots/"+img_name)

  #plt.show()
  plt.clf()

def heatmap_combined_success(x_success,y_success,x_fail,y_fail,x_total,y_total,plot_title,img_name):

  title = plot_title
  cb_label = 'Percentage of successful grasp trials out of total trials'

  x_range = 0.09 - (-0.09)
  y_range = 0.07

  x_bins = int(x_range / 0.002)
  y_bins = int(y_range / 0.002)

  print("x_range: ",x_range," y_range: ",y_range)
  print("x_bins: ",x_bins," y_bins: ",y_bins)

  h, _, _ = np.histogram2d(x_success, y_success, range=[[-.09, 0.09], [0, 0.07]],bins=(x_bins,y_bins))

  h2, x_edges, y_edges = np.histogram2d(x_total, y_total, range=[[-.09, 0.09], [0, 0.07]],bins=(x_bins,y_bins))

  h2[h2==0] = 1
  h2[h2==-0] = 1
  h3 = np.divide(h,h2)
  h3 = np.nan_to_num(h3)

  g, _, _ = np.histogram2d(x_fail, y_fail, range=[[-.09, 0.09], [0, 0.07]],bins=(x_bins,y_bins))

  g2, x_edges, y_edges = np.histogram2d(x_total, y_total, range=[[-.09, 0.09], [0, 0.07]],bins=(x_bins,y_bins))

  g2[g2==0] = 1
  g2[g2==-0] = 1
  g3 = np.divide(g,g2)
  g3 = np.nan_to_num(g3)
  g3 = np.multiply(-1,g3)

  fig = plt.figure()
  ax = fig.add_subplot(111)

  plt.imshow(h3.T, cmap=plt.cm.seismic,  origin='lower',extent=[-.09, 0.09, 0, 0.07],vmin=-1, vmax=1) #blues, failure
  ax.set_aspect('equal', adjustable='box')

  #print("h3: ",h3)
  #print("g3: ",g3)

  fig.set_size_inches(11,8)

  ax.xaxis.set_major_locator(MultipleLocator(0.01))
  ax.xaxis.set_minor_locator(MultipleLocator(0.001))

  ax.yaxis.set_major_locator(MultipleLocator(0.01))
  ax.yaxis.set_minor_locator(MultipleLocator(0.001))

  plt.title(title)
  plt.xlabel('X-axis initial coordinate position of object (meters)')
  plt.ylabel('Y-axis initial coordinate position of object (meters)')
  cb = plt.colorbar(cmap=plt.cm.seismic,format=PercentFormatter(1),shrink=0.6)
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


def create_plots(success_datax,success_datay,fail_datax,fail_datay,total_datax,total_datay):
    shapes = "CubeS"

    if len(success_datax) > 0:
      print("success_datax,success_datay: ",success_datax[0],success_datay[0])
    if len(fail_datax) > 0:
      print("fail_datax,fail_datay: ",fail_datax[0],fail_datay[0])
    if len(total_datax) > 0:
      print("total_datax,total_datay: ",total_datax[0],total_datay[0])


    freq_plot_title = "Grasp Trial Frequency per Initial Pose of Object ("+shapes+") - Naive Controller"
    heatmap_freq(total_datax,total_datay,freq_plot_title,'freq_heatmap_.jpg')

    fail_plot_title = "Grasp Trial Success Rate per Initial Pose of Object ("+shapes+") - Naive Controller"
    heatmap_combined_fail(success_datax,success_datay,fail_datax,fail_datay,total_datax,total_datay,fail_plot_title,'fail_heatmap.png')

    success_plot_title = "Grasp Trial Success Rate per Initial Pose of Object ("+shapes+") - Naive Controller"
    img_name = 'success_heatmap'
    heatmap_combined_success(success_datax,success_datay,fail_datax,fail_datay,total_datax,total_datay,success_plot_title,'success_heatmap.png')
    make_img_transparent(img_name)

def main():
    #get_arrays()
    #get_1000_eval()
    #print("len(x_success): ",len(x_success))

    heatmap_saving_dir = "./heatmap_plots"
    if not os.path.isdir(heatmap_saving_dir):
       os.mkdir(heatmap_saving_dir)

    freq_saving_dir = "./freq_plots"
    if not os.path.isdir(freq_saving_dir):
       os.mkdir(freq_saving_dir)

    success_datax = np.load("heatmap_train_success_new_x_arr.npy")
    success_datay = np.load("heatmap_train_success_new_y_arr.npy")

    fail_datax = np.load("heatmap_train_fail_new_x_arr.npy")
    fail_datay = np.load("heatmap_train_fail_new_y_arr.npy")

    total_datax = np.load("heatmap_train_total_new_x_arr.npy")
    total_datay = np.load("heatmap_train_total_new_y_arr.npy")

    create_plots(success_datax,success_datay,fail_datax,fail_datay,total_datax,total_datay)

    #heatmap_freq(a,b)
    #heatmap_combined_fail(x_success,y_success,x_fail,y_fail,a,b,0,'fail_heatmap.png')

    #img_name = 'success_plot'
    #heatmap_combined_success(x_success,y_success,x_fail,y_fail,a,b,1,'success_plot')
    #make_img_transparent(filename,img_name)

main()
