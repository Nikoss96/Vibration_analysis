import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#from comparison import *
from scipy.spatial import distance
def get_letters(frame : pd.DataFrame()):
    print("hui")
    
def plot_signal(df,typik):
    if typik == "barh":
        df[[]].plot(x='period', kind='barh')
    elif typik == "bar":
        fig, axes = plt.subplots()
        df.bar(subplots=True, ax=axes)
    elif typik == "plot":
        fig, axes = plt.subplots()
        df.plot(subplots=True, ax=axes)
    elif typik == "hist":
        pass
def plot_mass(mass):
    plt.plot(mass)
    plt.show()

def euclidean_dist(mass1,mass2):
    dist = distance.euclidean(mass1, mass2)
    fig, axes = plt.subplots(1, 1, figsize=(12, 4))
    axes.plot(mass1, alpha=0.7, label="a", marker="o")
    axes.plot(mass2, alpha=0.7, label="b", marker="D")
    
    # draw connecting segments between a_i and b_i used for Euclidean distance calculation
    axes.vlines(np.arange(0, len(mass1), 1), mass1, mass2, alpha = 0.7)
    axes.legend()
    axes.set_title("Нормированный сигнал | Евклидово расстояние = {:0.1f}".format(dist))
    
    
def trim_df(df1, df2):
   min_length = min(len(df1), len(df2))
    
   df1_trimmed = df1.iloc[:min_length, :]
   df2_trimmed = df2.iloc[:min_length, :]

   return df1_trimmed, df2_trimmed

    
def cross_correlation(mass1,mass2):
    print("nothing")
def compare_axis(axis1,axis2):
    plt.plot(axis1, alpha=0.5, label='график_1')
    plt.plot(axis2, alpha=0.5, label='график_2')
    plt.legend(loc='upper right')
    plt.show()

def normalize_axis_frame(frame_main,frame_change,axis):
    avg1 = sum(frame_main[axis][0:5])//6
    avg2 = sum(frame_change[axis][0:5])//6
    delta = avg1 - avg2
    frame_main[axis] = frame_main[axis].apply(lambda x: (x - delta))
    return frame_main,frame_change

def main():
    frame1 = pd.read_csv("C:\\Users\\nikch\\Desktop\\0data.csv")
    frame2 = pd.read_csv("C:\\Users\\nikch\\Desktop\\1data.csv")
    frame3 = pd.read_csv("data\\0\\output_cut_0.csv")
    frame1 , frame2 = trim_df(frame1,frame2)
    #frame1,frame2 = normalize_axis_frame(frame1,frame2,axis = "X")
    frame1,frame2 = normalize_axis_frame(frame1,frame2,axis = "Z")
#    plot_signal(frame,typik = "plot")
    for i,j in zip(frame1,frame2):
        #print(frame1[i][0:5],frame2[j][0:5])
        euclidean_dist(list(frame1[i]),list(frame2[j]))
        #compare_axis(list(frame1[i]),list(frame2[j]))
    #frame3.plot()
    
    
if __name__ == "__main__":
    main()