#!/usr/bin/env python
import inspect
import os
import random
import sys
import argparse
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.legend as lgd
import matplotlib.markers as mks
from matplotlib.gridspec import GridSpec


def get_log_parsing_script():
    dirname = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    return dirname + '/parse_log_pvt.sh'

def get_log_file_suffix():
    return '.log'

def get_chart_type_description_separator():
    return '  vs. '

def is_x_axis_field(field):
    x_axis_fields = ['Iters', 'Seconds']
    return field in x_axis_fields

def create_field_index():
    train_key = 'Train'
    test_key = 'Test'
    field_index = {train_key:{'Iters':0, 'Seconds':1, train_key + ' Bbox Loss':2,
                              train_key + ' Cls Loss':3},
                   test_key:{'Iters':0, 'Seconds':1, test_key + ' accuracy':2,
                             test_key + ' loss':3}}
    fields = set()
    for data_file_type in field_index.keys():
        fields = fields.union(set(field_index[data_file_type].keys()))
    fields = list(fields)
    fields.sort()
    return field_index, fields

def get_supported_chart_types():
    field_index, fields = create_field_index()
    num_fields = len(fields)
    supported_chart_types = []
    for i in xrange(num_fields):
        if not is_x_axis_field(fields[i]):
            for j in xrange(num_fields):
                if i != j and is_x_axis_field(fields[j]):
                    supported_chart_types.append('%s%s%s' % (
                        fields[i], get_chart_type_description_separator(),
                        fields[j]))
    return supported_chart_types

def get_chart_type_description(chart_type):
    supported_chart_types = get_supported_chart_types()
    chart_type_description = supported_chart_types[chart_type]
    return chart_type_description

def get_data_file_type(chart_type):
    description = get_chart_type_description(chart_type)
    data_file_type = description.split()[0]
    return data_file_type

def get_data_file(chart_type, path_to_log):
    return os.path.basename(path_to_log) + '.' + get_data_file_type(chart_type).lower()

def get_field_descriptions(chart_type):
    description = get_chart_type_description(chart_type).split(
        get_chart_type_description_separator())
    y_axis_field = description[0]
    x_axis_field = description[1]
    return x_axis_field, y_axis_field    

def get_field_indecies(chart_type, x_axis_field, y_axis_field):    
    data_file_type = get_data_file_type(chart_type)
    fields = create_field_index()[0][data_file_type]
    return fields[x_axis_field], fields[y_axis_field]

def load_data(data_file, field_idx0, field_idx1):
    data = [[], []]
    with open(data_file, 'r') as f:
        interval_sum = 0
        count = 0
        for line in f:
            line = line.strip()
            if line[0] != '#':
                fields = line.split()
                if int(fields[field_idx0]) % 500 == 0:
                    if int(fields[field_idx0]) == 0:
                        data[0].append(float(fields[field_idx0].strip()))
                        data[1].append(float(fields[field_idx1].strip()))
                    else:
                        interval_sum += float(fields[field_idx1].strip())
                        interval_sum /= count + 1
                        data[0].append(float(fields[field_idx0].strip()))
                        data[1].append(interval_sum)
                        interval_sum = 0
                        count = 0
                else:
                    interval_sum += float(fields[field_idx1].strip())
                    count += 1
    return data

def random_marker():
    markers = mks.MarkerStyle.markers
    num = len(markers.values())
    idx = random.randint(0, num - 1)
    return markers.values()[idx]

def get_data_label(path_to_log):
    label = path_to_log[path_to_log.rfind('/')+1 : path_to_log.rfind(
        get_log_file_suffix())]
    return label

def get_legend_loc(chart_type):
    x_axis, y_axis = get_field_descriptions(chart_type)
    loc = 'lower right'
    if y_axis.find('accuracy') != -1:
        pass
    if y_axis.find('loss') != -1 or y_axis.find('learning rate') != -1:
        loc = 'upper right'
    return loc

def plot_subfigure(ax, data_x, data_y, label, label_x, label_y):
    color = [random.random(), random.random(), random.random()]
    linewidth = 0.75

    ax.set_autoscaley_on(False)
    ax.plot(data_x, data_y, label = label, color = color, linewidth = linewidth)
    ax.set_xlabel(label_x, fontsize=10)
    ax.set_ylabel(label_y, fontsize=10)
    ax.set_title(label, fontsize=12)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(8) 



def plot_chart(path_to_png, path_to_log, width=2000, height=500):
    os.system('%s %s' % (get_log_parsing_script(), path_to_log))

    dpi = 100
    fig, ((ax1, ax2, ax3, ax4),(ax5, ax6, ax7, ax8)) = plt.subplots(nrows=2, ncols=4, figsize = (width/dpi, height/dpi), dpi=dpi)

    #plt.ylim([0, 1])
    chart_type = 4 # bbox loss
    for log_id in range(0,8):
        if log_id > 3:
            chart_type = 6 # classification loss
        else:
            chart_type 
        if log_id == 0 or log_id == 4:
            current_log = path_to_log + '.stage1_1rpn.log'
            #subfig = fig.add_subplot(gs[0:45, 5:25])
        elif log_id == 1 or log_id == 5:
            current_log = path_to_log + '.stage1_2rcnn.log'
            #subfig = fig.add_subplot(gs[0:45, 30:50])
        elif log_id == 2 or log_id == 6:
            current_log = path_to_log + '.stage2_1rpn.log'
            #subfig = fig.add_subplot(gs[0:45, 55:75])
        elif log_id == 3 or log_id == 7:
            current_log = path_to_log + '.stage2_2rcnn.log'
            #subfig = fig.add_subplot(gs[0:45, 80:100])

        data_file = get_data_file(chart_type, current_log)
        x_axis_field, y_axis_field = get_field_descriptions(chart_type)
        x, y = get_field_indecies(chart_type, x_axis_field, y_axis_field)
        data = load_data(data_file, x, y)
        label = get_data_label(current_log)

        if log_id == 0:
            plot_subfigure(ax1, data[0], data[1], 'Stage1 - RPN', x_axis_field, y_axis_field)
        elif log_id == 1:
            plot_subfigure(ax2, data[0], data[1], 'Stage1 - RCNN', x_axis_field, y_axis_field)
        elif log_id == 2:
            plot_subfigure(ax3, data[0], data[1], 'Stage2 - RPN', x_axis_field, y_axis_field)
        elif log_id == 3:
            plot_subfigure(ax4, data[0], data[1], 'Stage2 - RCNN', x_axis_field, y_axis_field)
        elif log_id == 4:
            plot_subfigure(ax5, data[0], data[1], 'Stage1 - RPN', x_axis_field, y_axis_field)
        elif log_id == 5:
            plot_subfigure(ax6, data[0], data[1], 'Stage1 - RCNN', x_axis_field, y_axis_field)
        elif log_id == 6:
            plot_subfigure(ax7, data[0], data[1], 'Stage2 - RPN', x_axis_field, y_axis_field)
        elif log_id == 7:
            plot_subfigure(ax8, data[0], data[1], 'Stage2 - RCNN', x_axis_field, y_axis_field)
	plt.ylim([0,1]) 
    #legend_loc = get_legend_loc(chart_type)
    #plt.legend(loc = legend_loc, ncol = 1) # ajust ncol to fit the space
    #plt.title(get_chart_type_description(chart_type))
    fig.suptitle('Faster R-CNN Training Log: Loss vs. Iteration\n')
    #plt.xlabel(x_axis_field)
    #plt.ylabel(y_axis_field)  
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(path_to_png)     
    plt.show()

def is_valid_chart_type(chart_type):
    return chart_type >= 0 and chart_type < len(get_supported_chart_types())

def parse_args():
    desc= """Be warned that the fields in the training log may change in the future.
You had better check the data files and change the mapping from field name to
field index in create_field_index before designing your own plots."""

    parser = argparse.ArgumentParser(description = desc)
    parser.add_argument('pngpath', metavar='png_path', help='Path of the png file to write')
    parser.add_argument('logpath', metavar='log_path', help='Path of the log file to read from')
    parser.add_argument('--width', dest='width', help='Width of the figure', type=int, default=2000)
    parser.add_argument('--height', dest='height', help='Height of the figure', type=int, default=500)

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    path_to_png = args.pngpath
    if not path_to_png.endswith('.png'):
        print 'Path must ends with png' % path_to_png
        exit            
    path_to_log = args.logpath
    if not os.path.exists(path_to_log):
        print 'Path does not exist: %s' % path_to_log
        exit
    
    ## plot_chart accpets multiple path_to_logs
    plot_chart(path_to_png, path_to_log, args.width, args.height)
