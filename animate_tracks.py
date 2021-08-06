import numpy as np

from matplotlib import pyplot as plt
# from matplotlib.patches import Rectangle

from IPython.display import display
import ipywidgets

import sys
sys.path.append("python/")
from utils import map_vis_without_lanelet


def animate_in_xy(actual_tracks_dict_idm_with_headway, lanes_rxy, t_0, t_end):
    scene = 'DR_CHN_Merging_ZS'
    map_file = 'maps/' + scene + '.osm'

    lane_c_rx, lane_c_ry = lanes_rxy['lane_c_rxy'][:,0], lanes_rxy['lane_c_rxy'][:,1]
    lane_d_rx, lane_d_ry = lanes_rxy['lane_d_rxy'][:,0], lanes_rxy['lane_d_rxy'][:,1]

    min_t, max_t = t_0, t_end # track[0,0], track[-1,0]

    def f(t):
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        fig, axes = plt.subplots(1, 1)
        axes.set_title('timstep : 100 ms')
        fig.set_size_inches(18, 10)
        lat_origin, lon_origin = 0., 0.
        map_vis_without_lanelet.draw_map_without_lanelet(map_file, axes, lat_origin, lon_origin)
        
        plt.ylim(930, 980)
        plt.xlabel("x[m]"); plt.ylabel("y[m]")
        plt.arrow(lane_c_rx[0]+10, lane_c_ry[0]+2, -5, -1, color='r', length_includes_head=True, head_width=1, head_length=1)
        plt.arrow(lane_d_rx[0]+10, lane_d_ry[0]+1, -5, -1, color='r', length_includes_head=True, head_width=1, head_length=1)
        # plt.plot(lane_c_rx, lane_c_ry, ".r", ms=0.5)
        # plt.plot(lane_d_rx, lane_d_ry, ".r", ms=0.5)
        plt.grid()
    
        for veh_id, track in actual_tracks_dict_idm_with_headway.items():
            
            if veh_id > 195 and veh_id < 225:

                idx = np.where(track[:,0]==t)[0] 
                
                # if len(idx_true) == 1 and veh_id != 120:
                #     plt.scatter(actual_tracks_dict_idm_with_headway[veh_id][idx_true, 1], 
                #                 actual_tracks_dict_idm_with_headway[veh_id][idx_true, 2], s=100, c='y')

                if len(idx) == 1:# and veh_id != 120:
                    plt.scatter(track[idx, 1], track[idx, 2], s=10, c='b')
                    # plt.plot(track[:, 1], track[:, 2], '--b', lw=1.2)
                    plt.text(track[idx, 1], track[idx, 2]+3, str(veh_id), fontsize=15, va="center", ha="center", color='b')
                    # plt.text(track[idx, 1], track[idx, 2]+5.5, track[idx,10][0], fontsize=12, va="center", ha="center", color='green')

                    # plt.text(track[idx, 1], track[idx, 2]+5, str("%.2f" % track[idx, 19]), fontsize=10, va="center", ha="center", color='green')
                    plt.text(track[idx, 1], track[idx, 2]+5, str("%.2f" % track[idx, 3]), fontsize=10, va="center", ha="center", color='green')
                    # plt.text(track[idx, 1], track[idx, 2]+7, track[idx,10][0], fontsize=12, va="center", ha="center", color='green')

    widget = ipywidgets.interactive(f, t=(min_t, max_t, 1))
    output = widget.children[-1]
    output.layout.height = '350px'
    display(widget)


def animate_in_xy_simulate(tag_switch, tracks_xy_dict, change_lane_switch, actual_tracks_dict_all_with_headway, mle_probs_final_all_veh, lanes_rxy, t_0, t_end):
    scene = 'DR_CHN_Merging_ZS'
    map_file = 'maps/' + scene + '.osm'

    lane_c_rx, lane_c_ry = lanes_rxy['lane_c_rxy'][:,0], lanes_rxy['lane_c_rxy'][:,1]
    lane_d_rx, lane_d_ry = lanes_rxy['lane_d_rxy'][:,0], lanes_rxy['lane_d_rxy'][:,1]

    min_t, max_t = t_0, t_end # track[0,0], track[-1,0]

    def f(t):        
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        fig, axes = plt.subplots(1, 1)
        fig.set_size_inches(18, 10)
        lat_origin, lon_origin = 0., 0.
        map_vis_without_lanelet.draw_map_without_lanelet(map_file, axes, lat_origin, lon_origin)
        
        plt.ylim(930, 980)
        plt.xlabel("x[m]"); plt.ylabel("y[m]")
        plt.arrow(lane_c_rx[0]+10, lane_c_ry[0]+2, -5, -1, color='r', length_includes_head=True, head_width=1, head_length=1)
        plt.arrow(lane_d_rx[0]+10, lane_d_ry[0]+1, -5, -1, color='r', length_includes_head=True, head_width=1, head_length=1)
        # plt.plot(lane_c_rx, lane_c_ry, "--r")
        # plt.plot(lane_d_rx, lane_d_ry, "--r")
        plt.grid()

        for veh_id, track in tracks_xy_dict.items():

            idx = np.where(track[:,0]==t)[0]
            
            if len(idx) != 0:
                if veh_id not in mle_probs_final_all_veh.keys():
                    COLOR = 'gray'
                else:
                    COLOR = 'blue'
                    
                # idx_true = np.where(actual_tracks_dict_all_with_headway[veh_id][:,0]==t)[0]

                # if len(idx_true) == 1: # and veh_id != 120:
                #     plt.scatter(actual_tracks_dict_all_with_headway[veh_id][idx_true, 1], 
                #                 actual_tracks_dict_all_with_headway[veh_id][idx_true, 2], s=100, c='y')
                #     plt.text(actual_tracks_dict_all_with_headway[veh_id][idx_true, 1],
                #              actual_tracks_dict_all_with_headway[veh_id][idx_true, 2]+3, str(veh_id), fontsize=15, va="center", ha="center", color='y')
                #     plt.text(actual_tracks_dict_all_with_headway[veh_id][idx_true, 1],
                #              actual_tracks_dict_all_with_headway[veh_id][idx_true, 2]+5.5, 
                #              actual_tracks_dict_all_with_headway[veh_id][idx_true,10][0], fontsize=15, va="center", ha="center", color='k')


                if len(idx) == 1: # and veh_id != 120:

                    if track[idx, 10][0] == 'c':
                        txt_posy = [+2, +4.25, +6.5]
                    elif track[idx, 10][0] == 'd':
                        txt_posy = [+2, -2, -4.25]

                    if tag_switch == 'on':

                        ## Plot position & id of each vehicle.
                        plt.scatter(track[idx, 1], track[idx, 2], s=30, c=COLOR) #c='b')
                        # plt.plot(track[:, 1], track[:, 2], '--b', lw=1.2)
                        plt.text(track[idx, 1], track[idx, 2]+txt_posy[0], str(veh_id), fontsize=12, va="center", ha="center", color=COLOR) #color='b')

                        ## Show tagged_lane & change_lane_switch.
                        plt.text(track[idx, 1]-1, track[idx, 2]+txt_posy[1], track[idx,10][0], fontsize=12, va="center", ha="center", color='k')

                        change_lane_switch[veh_id] = np.array(change_lane_switch[veh_id])
                        idx_switch = np.where(change_lane_switch[veh_id][:,0]==t)[0]
                        if change_lane_switch[veh_id][idx_switch,1][0] == 0:
                            plt.text(track[idx, 1]+1, track[idx, 2]+txt_posy[1], change_lane_switch[veh_id][idx_switch,1][0], fontsize=12, va="center", ha="center", color='k')                    
                        else:
                            plt.text(track[idx, 1]+1, track[idx, 2]+txt_posy[1], change_lane_switch[veh_id][idx_switch,1][0], fontsize=12, va="center", ha="center", color='r')

                        ## Show id of headway vehicle.
                        plt.text(track[idx, 1], track[idx, 2]+txt_posy[2], track[idx, 11][0], fontsize=12, va="center", ha="center", color='k')
                    
                    elif tag_switch == 'off':

                        ## Plot position & id of each vehicle.
                        change_lane_switch[veh_id] = np.array(change_lane_switch[veh_id])
                        idx_switch = np.where(change_lane_switch[veh_id][:,0]==t)[0]
                        if change_lane_switch[veh_id][idx_switch,1][0] == 0:
                            plt.scatter(track[idx, 1], track[idx, 2], s=30, c=COLOR) #c='b')
                            # plt.plot(track[:, 1], track[:, 2], '--b', lw=1.2)
                            plt.text(track[idx, 1], track[idx, 2]+txt_posy[0], str(veh_id), fontsize=12, va="center", ha="center", color=COLOR) #color='b')
                        else:
                            plt.scatter(track[idx, 1], track[idx, 2], s=30, c='r')
                            # plt.plot(track[:, 1], track[:, 2], '--r', lw=1.2)
                            plt.text(track[idx, 1], track[idx, 2]+txt_posy[0], str(veh_id), fontsize=12, va="center", ha="center", color='r')




                # if veh_id in samples_xy_dict.keys():
                #     sd_samples = samples_xy_dict[veh_id]
                    
                #     for j in range(0, 10): 
                #         sample_j = sd_samples[j::10, :]
                #         idx_j = int(np.where(sample_j[:,0]==t)[0])
                #         idx_j_end = min(idx_j+10, len(sample_j))

                #         plt.plot(sample_j[idx_j:idx_j_end:2,1], sample_j[idx_j:idx_j_end:2,2],'-r', lw=0.5) 

    widget = ipywidgets.interactive(f, t=(min_t, max_t, 2))
    output = widget.children[-1]
    output.layout.height = '350px'
    display(widget)



##########################################################
if __name__ == "__main__":

    from read_tracks_lanes import *

    centerlines_path = 'dataset/modified_centerlines_/'
    lanes_rxy, lanes_s, centerlines_dict = load_centerlines(centerlines_path)

    veh_tracks_path = '../processed/INTERACTION/DR_CHN_Merging_ZS/'
    actual_tracks_dict_all = load_tracks(veh_tracks_path, lanes_rxy, lanes_s, centerlines_dict)
    actual_tracks_dict_all_with_headway = construct_tracks_with_headway(actual_tracks_dict_all, lanes_rxy, lanes_s)

    # # Animate loaded tracks
    # t_0, t_end = 1000, 1300
    # animate_in_xy(actual_tracks_dict_all_with_headway, lanes_rxy, t_0, t_end)

    # # Animate simulated tracks
    # t_0, t_end = 1000, 1300
    # animate_in_xy_simulate(tracks_xy_halluc_dict_em, change_lane_switch, actual_tracks_dict_all_with_headway, mle_probs_final_all_veh, lanes_rxy, t_0, t_end)
