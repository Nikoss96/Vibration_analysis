import matplotlib.pyplot as plt # matplot lib is the premiere plotting lib for Python: https://matplotlib.org/
import numpy as np # numpy is the premiere signal handling library for Python: http://www.numpy.org/
import scipy as sp # for signal processing
from scipy import signal
from scipy.spatial import distance
import IPython.display as ipd
import librosa
import random
import math
#import makelab
#from makelab import audio
#from makelab import signal
import os
import pandas as pd
from statistics import mean
from tensorflow.keras.utils import to_categorical


def pad_zeros_right(s, padding_length):
    # https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
    return np.pad(s, (0, padding_length), mode = 'constant', constant_values=0)

def pad_mean_right(s, padding_length):
    # https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
    return np.pad(s, (0, padding_length), mode = 'mean')

# def compare_and_plot_signals(a, b, distance_function = distance.euclidean, alignment_function = None):
def plot_signals_with_alignment(a, b, pad_function = None):
    if(len(a) != len(b) and pad_function is None):
        raise Exception(f"Signal 'a' and 'b' must be the same size; len(a)={len(a)} and len(b)={len(b)} or pad_function must not be None")
    elif(len(a) != len(b) and pad_function is not None):
        if(len(a) < len(b)):
            a = pad_function(a, len(b) - len(a))
        else:
            b = pad_function(b, len(a) - len(b))
    
    correlate_result = np.correlate(a, b, 'full')
    shift_positions = np.arange(-len(a) + 1, len(b))
    
    print("len(a)", len(a), "len(b)", len(b), "len(correlate_result)", len(correlate_result))

    fig, axes = plt.subplots(5, 1, figsize=(10, 18))
    
    axes[0].plot(a, alpha=0.7, label="a", marker="o")
    axes[0].plot(b, alpha=0.7, label="b", marker="D")
    axes[0].legend()
    axes[0].set_title("Raw Signals 'a' and 'b'")
    
    if len(shift_positions) < 20:
        # useful for debugging and showing correlation results
        print(shift_positions)
        print(correlate_result)

    best_correlation_index = np.argmax(correlate_result)
    shift_amount_debug = shift_positions[best_correlation_index]
    shift_amount = (-len(a) + 1) + best_correlation_index
    print("best_correlation_index", best_correlation_index, "shift_amount_debug", shift_amount_debug, "shift_amount", shift_amount)
    
    axes[1].stem(shift_positions, correlate_result, use_line_collection=True, label="Cross-correlation of a and b")
    axes[1].set_title(f"Cross-Correlation Result | Best Match Index: {best_correlation_index} Signal 'b' Shift Amount: {shift_amount}")
    axes[1].set_ylabel("Cross Correlation")
    axes[1].set_xlabel("'b' Signal Shift Amount")
    
    best_match_ymin = 0
    best_match_ymin_normalized = makelab.signal.map(best_match_ymin, axes[1].get_ylim()[0], axes[1].get_ylim()[1], 0, 1)
    best_match_ymax = correlate_result[best_correlation_index]
    best_match_ymax_normalized = makelab.signal.map(best_match_ymax, axes[1].get_ylim()[0], axes[1].get_ylim()[1], 0, 1)
    axes[1].axvline(shift_positions[best_correlation_index], ymin=best_match_ymin_normalized, ymax=best_match_ymax_normalized, 
                    linewidth=2, color='orange', alpha=0.8, linestyle='-.', 
                    label=f"Best match ({shift_amount}, {best_match_ymax:.2f})")
    axes[1].legend()
    
    b_shifted_mean_fill = makelab.signal.shift_array(b, shift_amount, np.mean(b))
    axes[2].plot(a, alpha=0.7, label="a", marker="o")
    axes[2].plot(b_shifted_mean_fill, alpha=0.7, label="b_shifted_mean_fill", marker="D")
    axes[2].legend()
    axes[2].set_title("Signals 'a' and 'b_shifted_mean_fill'")
    
    b_shifted_zero_fill = makelab.signal.shift_array(b, shift_amount, 0)
    axes[3].plot(a, alpha=0.7, label="a", marker="o")
    axes[3].plot(b_shifted_zero_fill, alpha=0.7, label="b_shifted_zero_fill", marker="D")
    axes[3].legend()
    axes[3].set_title("Signals 'a' and 'b_shifted_zero_fill'")
    
    b_shifted_roll = np.roll(b, shift_amount)
    axes[4].plot(a, alpha=0.7, label="a", marker="o")
    axes[4].plot(b_shifted_roll, alpha=0.7, label="b_shifted_roll", marker="D")
    axes[4].legend()
    axes[4].set_title("Signals 'a' and 'b_shifted_roll'")
    
    fig.tight_layout()
    
def compare_and_plot_signals_with_alignment(a, b, bshift_method = 'mean_fill', pad_function = None):
    '''Aligns signals using cross correlation and then plots
    
       bshift_method can be: 'mean_fill', 'zero_fill', 'roll', or 'all'. Defaults to 'mean_fill'
    '''
    
    if(len(a) != len(b) and pad_function is None):
        raise Exception(f"Signal 'a' and 'b' must be the same size; len(a)={len(a)} and len(b)={len(b)} or pad_function must not be None")
    elif(len(a) != len(b) and pad_function is not None):
        if(len(a) < len(b)):
            a = pad_function(a, len(b) - len(a))
        else:
            b = pad_function(b, len(a) - len(b))
    
    correlate_result = np.correlate(a, b, 'full')
    shift_positions = np.arange(-len(a) + 1, len(b))
    print("len(a)", len(a), "len(b)", len(b), "len(correlate_result)", len(correlate_result))
    
    euclid_distance_a_to_b = distance.euclidean(a, b)
    
    num_charts = 3
    chart_height = 3.6
    if bshift_method is 'all':
        num_charts = 5
    
    fig, axes = plt.subplots(num_charts, 1, figsize=(10, num_charts * chart_height))
    
    # Turn on markers only if < 50 points
    a_marker = None
    b_marker = None
    if len(a) < 50:
        a_marker = "o"
        b_marker = "D"
        
    axes[0].plot(a, alpha=0.7, label="a", marker=a_marker)
    axes[0].plot(b, alpha=0.7, label="b", marker=b_marker)
    axes[0].legend()
    axes[0].set_title(f"Raw Signals | Euclidean Distance From 'a' to 'b' = {euclid_distance_a_to_b:.2f}")
    
    if len(shift_positions) < 20:
        # useful for debugging and showing correlation results
        print(shift_positions)
        print(correlate_result)
    
    best_correlation_index = np.argmax(correlate_result)
    shift_amount_debug = shift_positions[best_correlation_index]
    shift_amount = (-len(a) + 1) + best_correlation_index
    print("best_correlation_index", best_correlation_index, "shift_amount_debug", shift_amount_debug, "shift_amount", shift_amount)
    
    #axes[1].plot(shift_positions, correlate_result)
    axes[1].stem(shift_positions, correlate_result, use_line_collection=True, label="Cross-correlation of a and b")
    axes[1].set_title(f"Cross-correlation result | Best match index: {best_correlation_index}; Signal 'b' shift amount: {shift_amount}")
    axes[1].set_ylabel("Cross Correlation")
    axes[1].set_xlabel("'b' Signal Shift Amount")
    
    best_match_ymin = 0
    best_match_ymin_normalized = makelab.signal.map(best_match_ymin, axes[1].get_ylim()[0], axes[1].get_ylim()[1], 0, 1)
    best_match_ymax = correlate_result[best_correlation_index]
    best_match_ymax_normalized = makelab.signal.map(best_match_ymax, axes[1].get_ylim()[0], axes[1].get_ylim()[1], 0, 1)
    axes[1].axvline(shift_positions[best_correlation_index], ymin=best_match_ymin_normalized, ymax=best_match_ymax_normalized, 
                    linewidth=2, color='orange', alpha=0.8, linestyle='-.', 
                    label=f"Best match ({shift_amount}, {best_match_ymax:.2f})")
    axes[1].legend()
    
    if bshift_method is 'mean_fill' or bshift_method is 'all':
        b_shifted_mean_fill = makelab.signal.shift_array(b, shift_amount, np.mean(b))
        euclid_distance_a_to_b_shifted_mean_fill = distance.euclidean(a, b_shifted_mean_fill)
        axes[2].plot(a, alpha=0.7, label="a", marker=a_marker)
        axes[2].plot(b_shifted_mean_fill, alpha=0.7, label="b_shifted_mean_fill", marker=b_marker)
        axes[2].legend()
        axes[2].set_title(f"Euclidean distance From 'a' to 'b_shifted_mean_fill' = {euclid_distance_a_to_b_shifted_mean_fill:.2f}")
    
    ax_idx = 0
    if bshift_method is 'zero_fill' or bshift_method is 'all':
        if bshift_method is 'zero_fill':
            ax_idx = 2
        else:
            ax_idx = 3
    
        b_shifted_zero_fill = makelab.signal.shift_array(b, shift_amount, 0)
        euclid_distance_a_to_b_shifted_zero_fill = distance.euclidean(a, b_shifted_zero_fill)
        axes[ax_idx].plot(a, alpha=0.7, label="a", marker=a_marker)
        axes[ax_idx].plot(b_shifted_zero_fill, alpha=0.7, label="b_shifted_zero_fill", marker=b_marker)
        axes[ax_idx].legend()
        axes[ax_idx].set_title(f"Euclidean distance From 'a' to 'b_shifted_zero_fill' = {euclid_distance_a_to_b_shifted_zero_fill:.2f}")
    
    
    if bshift_method is 'roll' or bshift_method is 'all':
        if bshift_method is 'roll':
            ax_idx = 2
        else:
            ax_idx = 4
        b_shifted_roll = np.roll(b, shift_amount)
        euclid_distance_a_to_b_shifted_roll = distance.euclidean(a, b_shifted_roll)
        axes[ax_idx].plot(a, alpha=0.7, label="a", marker=a_marker)
        axes[ax_idx].plot(b_shifted_roll, alpha=0.7, label="b_shifted_roll", marker=b_marker)
        axes[ax_idx].legend()
        axes[ax_idx].set_title(f"Euclidean distance From 'a' to 'b_shifted_roll' = {euclid_distance_a_to_b_shifted_roll:.2f}")
    
    fig.tight_layout()


def compare_signals(signal1, signal2):

    correlation = np.corrcoef(signal1.flatten(), signal2.flatten())[0, 1]


    mean_signal1 = np.mean(signal1)
    mean_signal2 = np.mean(signal2)

    std_signal1 = np.std(signal1)
    std_signal2 = np.std(signal2)
    
    energy_signal1 = np.sum(np.square(signal1))
    energy_signal2 = np.sum(np.square(signal2))
    
    return {
        'correlation': correlation,
        'mean_signal1': mean_signal1,
        'mean_signal2': mean_signal2,
        'std_signal1': std_signal1,
        'std_signal2': std_signal2,
        'energy_signal1': energy_signal1,
        'energy_signal2': energy_signal2
    }



def compare_keyboard(folder_path_mechanical,folder_path_membr,gap,start):
    X_mem, y_mem = [], []
    X_mec, y_mec = [], [] 
    
    res_compare = []
    for label in range(10):
        path = f'{folder_path_mechanical}/{label}/'
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            data = pd.read_csv(file_path)
            X_mec.append(data[['X', "Y", "Z"]].values)
            y_mec.append(label)
    X_mec = np.array(X_mec)
    y_mec = np.array(y_mec)
    
    y_mec = to_categorical(y_mec, num_classes=10)
    
    for label in range(10):
        path = f'{folder_path_membr}/{label}/'
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            data = pd.read_csv(file_path)
            X_mem.append(data[['X', "Y", "Z"]].values)
            y_mem.append(label)
    X_mem = np.array(X_mem)
    y_mem = np.array(y_mem)
    
    y_mem = to_categorical(y_mem, num_classes=10)
    
    for i in range(start*100,(start+1)*100):
        res_compare.append(compare_signals(X_mem[i],X_mec[i*gap]))
    return res_compare    
  
def make_cross_corr():
    buttons_all = []
    for i in range(10):
        if i == 0 :
           buttons_all.append(compare_keyboard(folder_path_mechanical = "data",folder_path_membr = "data_membr", gap = 1,start = i))
        else:
           buttons_all.append(compare_keyboard(folder_path_mechanical = "data",folder_path_membr = "data_membr", gap = 2,start = i))
    return buttons_all    




def print_avg_info(buttons_all):
    buttons_avg_info = []
    sum_corr = 0
    for i in range(10):
       buttons_avg_info.append({
            'correlation_mean': 0
        })
       
       for j in range(100):
            buttons_avg_info[i]["correlation_mean"] += buttons_all[i][j]["correlation"]
       buttons_avg_info[i]["correlation_mean"] /= 100
       print(f"Значение корреляции кнопки {i} между клавиатурами: "+ str(buttons_avg_info[i]["correlation_mean"]) + "\n")
       sum_corr += abs(buttons_avg_info[i]["correlation_mean"])
    print("Общее значение корреляции между клавиатурами: " + str(sum_corr/10))








def compare_keyboard_full(folder_path_mechanical,folder_path_membr,gap,start):
    X_mem, y_mem = [], []
    X_mec, y_mec = [], [] 
    
    res_compare = []
    for label in range(10):
        path = f'{folder_path_mechanical}/{label}/'
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            data = pd.read_csv(file_path)
            X_mec.append(data[['X', "Y", "Z"]].values)
            y_mec.append(label)
    X_mec = np.array(X_mec)
    y_mec = np.array(y_mec)
    
    y_mec = to_categorical(y_mec, num_classes=10)
    
    for label in range(10):
        path = f'{folder_path_membr}/{label}/'
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            data = pd.read_csv(file_path)
            X_mem.append(data[['X', "Y", "Z"]].values)
            y_mem.append(label)
    X_mem = np.array(X_mem)
    y_mem = np.array(y_mem)
    
    y_mem = to_categorical(y_mem, num_classes=10)
    
    for j in range(10):
        sample1, sample2 = [], []
        for i in range(start*100,(start+1)*100):
            sample1.append(X_mem[i])
            sample2.append(X_mec[i*gap])
        sample1 = np.array(sample1)
        sample2 = np.array(sample2)
        res_compare.append(compare_signals(sample1,sample2))
    
    return res_compare

def print_full_info(buttons_all):    
    buttons_avg_info = []
    sum_corr = 0
    for i in range(10):
       buttons_avg_info.append({
            'correlation': buttons_all[i][0]["correlation"]
        })

       print(f"Значение корреляции кнопки {i} между клавиатурами: "+ str(buttons_avg_info[i]["correlation"]) + "\n")
       sum_corr += abs(buttons_avg_info[i]["correlation"])
    print("Общее значение корреляции между клавиатурами: " + str(sum_corr/10))
    


def make_full_corr():
    buttons_all = []
    for i in range(10):
        if i == 0 :
           buttons_all.append(compare_keyboard_full(folder_path_mechanical = "data",folder_path_membr = "data_membr", gap = 1,start = i))
        else:
           buttons_all.append(compare_keyboard_full(folder_path_mechanical = "data",folder_path_membr = "data_membr", gap = 2,start = i))
    return buttons_all



def main():    
    
    buttons_all = make_full_corr()     
    print_full_info(buttons_all)    
    
    buttons_all = make_cross_corr()     
    print_avg_info(buttons_all) 
    
if __name__ == "__main__":
    main()