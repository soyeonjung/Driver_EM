#### moving codes/functions from EM_IDM_MOBIL.ipynb

import numpy as np
import pickle
import os
import re

import argparse
# import sys
# np.set_printoptions(threshold=sys.maxsize)

# from matplotlib import pyplot as plt
# import itertools
# from scipy.interpolate import interp1d, PchipInterpolator
from scipy import interpolate
from scipy.spatial.distance import cdist

from cubic_spline_planner import *  #import other python file


##########################################################
# Load & Construct vehicle tracks and centerlines

def interp_centerline(centerline_array, interp_num):
    ### inputs)
    ### centerline_array : shape(len, 2)
    ### interp_num : number of sample points from interpolated line
    
    interpSpl, u = interpolate.splprep([centerline_array[:,0], centerline_array[:,1]], s=10)
    u_interp = np.linspace(0, 1, num=interp_num)
    [centerline_x_interp, centerline_y_interp] = interpolate.splev(u_interp, interpSpl, der=0)
    centerline_interp = np.column_stack((centerline_x_interp[:], centerline_y_interp[:]))

    return centerline_interp  #shape(num=100, 2)

def tag_lane(track, centerlines_dict):
    ### inputs)
    ### track : shape(len(track), 9)
    ###  => columns: frame_id, x, y, vx, vy, psi_rad, agent_type, length, width
    ### centerlines_dict : dictionary of centerlines (keys: lanes)
    
    ### output
    ### track_tagged : shape(len(track), 7)
    ###  => columns: frame_id, x, y, agent_type, length, width, lane
    
    tags = []
    for p, _ in enumerate(track):
        min_dist_dict = dict()
        # lane_idx_dict = dict()
        for lane, centerline in centerlines_dict.items():                    
            min_dist_dict[lane] = np.min(cdist(np.array([track[p, 1:3]]), centerline, 'euclidean'))
            # lane_idx_dict[lane] = np.argmin(cdist(np.array([track[p, 1:3]]), centerline, 'euclidean'))
            
        lane_tag = min(min_dist_dict, key=min_dist_dict.get)
        tags.append([lane_tag]) #, lane_idx_dict[lane_tag]])

        # if p == 0 and lane_tag != 'c':
        #     print('does not start in lane c')

    tags = np.array(tags)
    track_tagged = np.column_stack((track[:, :3], track[:, 6:], tags))  
    # track_tagged = np.column_stack((track[:, :5], track[:, 6:], tags))  #vx,vy plot 해보려고
    return track_tagged

# def convert_xy2sd():
# # def convert_sd2xy():

def load_centerlines(centerlines_path):

    centerlines_listdir = os.listdir(centerlines_path)
    
    centerlines_dict = dict()
    lanes_list = []
    for _, file in enumerate(centerlines_listdir):
        lane = re.split('_', file)[1].split('.')[0]

        if lane == 'c' or lane == 'd':
            # print('lane', lane)

            centerline = open(os.path.join(centerlines_path, file), 'r')
            centerline_temp = [line.split(",") for line in centerline.readlines()]

            centerline_array = []
            for point in centerline_temp:        
                centerline_array.append((float(point[0]), float(point[1])))     
            centerline_array = np.array(centerline_array)

            centerline_interp = interp_centerline(centerline_array, 100)
            # print('centerline interpolation', len(centerline_array), 'to', len(centerline_interp)) #, centerline_interp)

            centerlines_dict[lane] = centerline_interp
            lanes_list.append(lane)

    # print('\ncenterlines_dict', centerlines_dict.keys())
    # print('lanes_list', lanes_list)


    #### lane_c ####
    lane_c_x = centerlines_dict['c'][:,0]  
    lane_c_y = centerlines_dict['c'][:,1]
    # ds = 0.1  # [m] distance of each intepolated points
    num = 1000  # number of intepolated points

    sp_c = Spline2D(lane_c_x, lane_c_y)
    lane_c_s = np.arange(0, sp_c.s[-1], sp_c.s[-1]/num)

    lane_c_rx, lane_c_ry = [], []
    # lane_c_ryaw, lane_c_rk = , [], []
    for i in range(len(lane_c_s)):
        xi, yi = sp_c.calc_position(lane_c_s[i])
        lane_c_rx.append(xi)
        lane_c_ry.append(yi)
    lane_c_rxy = np.column_stack((lane_c_rx, lane_c_ry)) 


    #### lane_d ####
    lane_d_x = centerlines_dict['d'][:,0]  
    lane_d_y = centerlines_dict['d'][:,1]

    sp_d = Spline2D(lane_d_x, lane_d_y)
    lane_d_s = np.arange(0, sp_d.s[-1], sp_d.s[-1]/num)

    lane_d_rx, lane_d_ry = [], []
    for i in range(len(lane_d_s)):
        xi, yi = sp_d.calc_position(lane_d_s[i])
        lane_d_rx.append(xi)
        lane_d_ry.append(yi)
    lane_d_rxy = np.column_stack((lane_d_rx, lane_d_ry))

    #### center of lane_c and lane_d
    lane_cd_s = np.mean([lane_c_s, lane_d_s], axis=0)
    lane_cd_rxy = np.mean([lane_c_rxy, lane_d_rxy], axis=0)

    lanes_rxy = {'lane_c_rxy': lane_c_rxy, 'lane_d_rxy': lane_d_rxy, 'lane_cd_rxy': lane_cd_rxy}
    lanes_s = {'lane_c_s': lane_c_s, 'lane_d_s': lane_d_s, 'lane_cd_s': lane_cd_s, 'sp_c': sp_c, 'sp_d': sp_d}

    return lanes_rxy, lanes_s, centerlines_dict


##########################################################
# Load & Construct vehicle tracks

# tracks_dict['veh_id'] = np.empty((len(df_id), 9), dtype=object)
# tracks_dict['veh_id'][:, 0] = df_id.frame_id  #t(100ms = 0.1s)
# tracks_dict['veh_id'][:, 1] = df_id.x
# tracks_dict['veh_id'][:, 2] = df_id.y
# tracks_dict['veh_id'][:, 3] = df_id.vx
# tracks_dict['veh_id'][:, 4] = df_id.vy
# tracks_dict['veh_id'][:, 5] = df_id.psi_rad
# tracks_dict['veh_id'][:, 6] = df_id.agent_type
# tracks_dict['veh_id'][:, 7] = df_id.length
# tracks_dict['veh_id'][:, 8] = df_id.width

def load_tracks(veh_tracks_path, lanes_rxy, lanes_s, centerlines_dict):
    # veh_tracks_path = '../processed/INTERACTION/DR_CHN_Merging_ZS/'

    veh_tracks_listdir = os.listdir(veh_tracks_path)

    lane_c_rxy, lane_d_rxy, lane_cd_rxy = lanes_rxy['lane_c_rxy'], lanes_rxy['lane_d_rxy'], lanes_rxy['lane_cd_rxy']
    lane_c_s, lane_d_s, lane_cd_s = lanes_s['lane_c_s'], lanes_s['lane_d_s'], lanes_s['lane_cd_s']
    
    actual_tracks_dict_all = dict()
    actual_tracks_dict_idm = dict()
    actual_tracks_dict_mobil = dict()

    new_id = 0
    t_span_file = [0,0]

    for i, file in enumerate(veh_tracks_listdir[:1]):
        with open(os.path.join(veh_tracks_path, file), 'rb') as f:
            veh_tracks_i = pickle.load(f)
        # print('veh_tracks_', i, type(veh_tracks_i), len(veh_tracks_i), min(list(veh_tracks_i.keys())), max(list(veh_tracks_i.keys())))

        t_span = [0, 0]
        for veh_id, track in veh_tracks_i.items():
            if track[0, 1] > 1125 and track[0, 1] < 1140 and track[0, 2] > 954 and track[0, 2] < 963:  #lanes c,d
            # if track[0, 1] > 1125 and track[0, 1] < 1140 and track[0, 2] > 960 and track[0, 2] < 963:  #lane c
                track_tagged = tag_lane(track, centerlines_dict)
                # print('track_tagged', track_tagged[:5])

                track_tagged_t_span = np.column_stack((track_tagged[:,0] + t_span_file[1], track_tagged[:,1:]))
                actual_tracks_dict_all[10000*i + veh_id] = track_tagged_t_span

                if len(set(track_tagged[:, 6])) == 1:
                    actual_tracks_dict_idm[10000*i + veh_id] = track_tagged_t_span
                else: 
                    actual_tracks_dict_mobil[10000*i + veh_id] = track_tagged_t_span            

                new_id += 1
                t_min = min((t_span[0], track_tagged_t_span[0,0]))
                t_max = max((t_span[1], track_tagged_t_span[-1,0]))
                t_span = [t_min, t_max]

        if veh_id == list(veh_tracks_i)[-1]:
            t_span_min = min((t_span_file[0], t_span[0]))
            t_span_max = max((t_span_file[1], t_span[1]))
            t_span_file = [t_span_min, t_span_max]
            # print('t_span', t_span, 't_span_file', t_span_file)


    # print('\nactual_tracks_dict_all', len(actual_tracks_dict_all), actual_tracks_dict_all[33].shape, actual_tracks_dict_all[33][:1,:]) #, actual_tracks_dict.keys())
    # print('actual_tracks_dict_idm', len(actual_tracks_dict_idm), actual_tracks_dict_idm[33].shape) #, actual_tracks_dict.keys())
    # print('actual_tracks_dict_mobil', len(actual_tracks_dict_mobil), 'veh_ids', actual_tracks_dict_mobil.keys())

    # print('\nt_span_file', t_span_file)
    

    ##########################################################
    ##########################################################
    ### x-y coord => s-d coord
    ### Reference: github.com/fjp/frenet
    ###           "Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenext Frame"

    for veh_id, track in actual_tracks_dict_all.items():
        veh_d, veh_s = [], []
        veh_d_cd, veh_s_cd = [], []
        for i in range(len(track)):
            if track[i, 6] == 'c': 
                lane_rxy, lane_s = lane_c_rxy, lane_c_s
            elif track[i, 6] == 'd': 
                lane_rxy, lane_s = lane_d_rxy, lane_d_s

            ## s-d coordinate about its lane
            veh_xyi = track[i, 1:3]
            distances = cdist(np.array([veh_xyi]), lane_rxy, 'euclidean')
            veh_s.append(lane_s[np.argmin(distances)])
            if np.argmin(distances) == 0:
                lane_xyi = lane_rxy[np.argmin(distances)+1, :]
                lane_xyi_prev = lane_rxy[np.argmin(distances), :]
            else:
                lane_xyi = lane_rxy[np.argmin(distances), :]
                lane_xyi_prev = lane_rxy[np.argmin(distances)-1, :]
            [lane_dxi, lane_dyi]  = np.subtract(lane_xyi, lane_xyi_prev)
            [veh_dxi, veh_dyi] = np.subtract(veh_xyi, lane_xyi_prev)
            signi = np.sign(lane_dxi * veh_dyi - lane_dyi * veh_dxi)
            veh_d.append(np.min(distances) * signi)

            ## s-d coordinate about lane_cd_rxy
            distances_cd = cdist(np.array([veh_xyi]), lane_cd_rxy, 'euclidean')
            veh_s_cd.append(lane_cd_s[np.argmin(distances_cd)])

            if np.argmin(distances_cd) == 0:
                lane_xyi_cd = lane_cd_rxy[np.argmin(distances_cd)+1, :]
                lane_xyi_prev_cd = lane_cd_rxy[np.argmin(distances_cd), :]
            else:
                lane_xyi_cd = lane_cd_rxy[np.argmin(distances_cd), :]
                lane_xyi_prev_cd = lane_cd_rxy[np.argmin(distances_cd)-1, :]
            [lane_dxi_cd, lane_dyi_cd]  = np.subtract(lane_xyi_cd, lane_xyi_prev_cd)
            [veh_dxi_cd, veh_dyi_cd] = np.subtract(veh_xyi, lane_xyi_prev_cd)
            signi_cd = np.sign(lane_dxi_cd * veh_dyi_cd - lane_dyi_cd * veh_dxi_cd)
            veh_d_cd.append(np.min(distances_cd) * signi_cd)

        track_xy_new = np.column_stack((track[:,0:3], veh_s, veh_d, track[:,3:], veh_s_cd, veh_d_cd))
        actual_tracks_dict_all[veh_id] = track_xy_new
        
    # print('actual_tracks_dict_all', len(actual_tracks_dict_all), actual_tracks_dict_all[116].shape, actual_tracks_dict_all[116][:2,:])
    
    return actual_tracks_dict_all

# actual_tracks_dict_all['veh_id'][:, 0] = frame_id  #t(100ms = 0.1s)
# actual_tracks_dict_all['veh_id'][:, 1:3] = x, y
# actual_tracks_dict_all['veh_id'][:, 3:5] = s, d
# actual_tracks_dict_all['veh_id'][:, 5:8] = agent_type, length, width
# actual_tracks_dict_all['veh_id'][:, 8] = lane
# actual_tracks_dict_all['veh_id'][:, 9:11] = s_cd, d_cd  #s-d coord. about lane_cd_xy

def smoothing_vs_vd(veh_vsvd):
    w = 3  ## sliding window length
    
    veh_vsvd = np.array(veh_vsvd)
    vs, vd = veh_vsvd[:,0], veh_vsvd[:,1]
    
    vs_smooth_0 = np.convolve(vs, np.ones(w), 'valid') / w
    vs_smooth_0 = np.concatenate(([vs[0]], vs_smooth_0, [vs[-1]]))    
    vs_smooth = np.convolve(vs_smooth_0, np.ones(w), 'valid') / w
    vs_smooth = np.concatenate(([vs_smooth_0[0]], vs_smooth, [vs_smooth_0[-1]]))

    vd_smooth_0 = np.convolve(vd, np.ones(w), 'valid') / w
    vd_smooth_0 = np.concatenate(([vd[0]], vd_smooth_0, [vd[-1]]))
    vd_smooth = np.convolve(vd_smooth_0, np.ones(w), 'valid') / w
    vd_smooth = np.concatenate(([vd_smooth_0[0]], vd_smooth, [vd_smooth_0[-1]]))
    
    return np.column_stack((vs_smooth, vd_smooth))

def construct_tracks_with_headway(actual_tracks_dict_all, lanes_rxy, lanes_s):

    lane_c_rxy, lane_d_rxy, lane_cd_rxy = lanes_rxy['lane_c_rxy'], lanes_rxy['lane_d_rxy'], lanes_rxy['lane_cd_rxy']
    lane_c_s, lane_d_s, lane_cd_s = lanes_s['lane_c_s'], lanes_s['lane_d_s'], lanes_s['lane_cd_s']
    
    actual_tracks_dict_all_with_headway = dict()
    for veh_id, track in actual_tracks_dict_all.items():
        # if veh_id == 11 or veh_id == 116:

            veh_vsvd = []
            track_headway = []
            track_headway_prev = [None]*9
            for i in range(len(track)):
                t = track[i,0]

                if i == 0:
                    veh_vsvd.append((track[i+1, 3:5] - track[i, 3:5])) # * 10) 
                else:
                    veh_vsvd.append((track[i, 3:5] - track[i-1, 3:5])) # * 10) 

                dist_headway, veh_id_headway, idx_headway, lane_headway = 150, None, None, None
                track_xy_sd_headway, vsvd_headway = [None]*4, [None]*2

                for veh_id_other, track_other in actual_tracks_dict_all.items():
                    if veh_id_other != veh_id and t >= track_other[0,0] and t <= track_other[-1,0] and track[i,8] in set(track_other[:,8]):

                        idx = np.where(track_other[:,0] == t)[0][0]      
                        dist_other = track_other[idx, 3] - track[i,3]  #dist_s

                        if dist_other > 0 and dist_other < dist_headway and track[i,8] == track_other[idx,8]:
                            veh_id_headway = veh_id_other
                            idx_headway = idx
                            dist_headway = dist_other
                            track_xy_sd_headway = track_other[idx, 1:5]
                            lane_headway = track_other[idx, 8]

                            if idx == 0:
                                vsvd_headway = track_other[idx+1, 3:5] - track_other[idx, 3:5] # * 10
                            else:
                                vsvd_headway = track_other[idx, 3:5] - track_other[idx-1, 3:5] # * 10

                if veh_id_headway is None and track_headway_prev is None:
                    track_headway.append([None]*10)
                elif veh_id_headway is not None:
                    track_headway_prev = np.concatenate([np.array([veh_id_headway]), track_xy_sd_headway, vsvd_headway, \
                                        np.array([idx_headway]), np.array([dist_headway]), np.array([lane_headway])], axis=0)
                    # print('track_headway_prev', track_headway_prev.shape)
                    track_headway.append(track_headway_prev) 
                elif veh_id_headway is None and track_headway_prev is not [None]*10:
                    track_headway.append(track_headway_prev)
            veh_vsvd = smoothing_vs_vd(veh_vsvd)

            track_headway = np.array(track_headway)        
            actual_tracks_dict_all_with_headway[veh_id] = np.column_stack((track[:,:5], veh_vsvd, track[:,5:9], \
                                                                           track_headway, track[:,9:]))    
    return actual_tracks_dict_all_with_headway

   
# actual_tracks_dict_all_with_headway['veh_id'][:, 0] = frame_id  #t(in 0.1s)
# actual_tracks_dict_all_with_headway['veh_id'][:, 1:3] = x, y
# actual_tracks_dict_all_with_headway['veh_id'][:, 3:7] = s, d, vs, vd   #(vs,vd : m/s)
# actual_tracks_dict_all_with_headway['veh_id'][:, 7:11] = agent_type, length, width, lane
# actual_tracks_dict_all_with_headway['veh_id'][:, 11:14] = headway_veh_id, headway_x, headway_y 
# actual_tracks_dict_all_with_headway['veh_id'][:, 14:18] = headway_s, headway_d, headway_vs, headway_vd
# actual_tracks_dict_all_with_headway['veh_id'][:, 18:21] = headway_idx, dist_headway, headway_lane
# actual_tracks_dict_all_with_headway['veh_id'][:, 21:23] = s_cd, d_cd  #s-d coord. about lane_cd_xy


##########################################################
if __name__ == "__main__":
    centerlines_path = 'dataset/modified_centerlines_/'
    lanes_rxy, lanes_s, centerlines_dict = load_centerlines(centerlines_path)

    # print('\nlist(lanes_rxy.keys())', list(lanes_rxy.keys()))
    # print('lanes_rxy[lane_cd_rxy].shape', lanes_rxy['lane_cd_rxy'].shape)
    # print('\nlist(lanes_s.keys())', list(lanes_s.keys()))
    # print('lanes_s[lane_cd_s].shape', lanes_s['lane_cd_s'].shape)
    # print('\nlist(centerlines_dict.keys())', list(centerlines_dict.keys()))
    # print('centerlines_dict[c].shape', centerlines_dict['c'].shape)
    
    
    veh_tracks_path = '../processed/INTERACTION/DR_CHN_Merging_ZS/'
    actual_tracks_dict_all = load_tracks(veh_tracks_path, lanes_rxy, lanes_s, centerlines_dict)
    actual_tracks_dict_all_with_headway = construct_tracks_with_headway(actual_tracks_dict_all, lanes_rxy, lanes_s)
    
    print('\nactual_tracks_dict_all_with_headway', len(actual_tracks_dict_all_with_headway))
    print('actual_tracks_dict_all_with_headway[116]', actual_tracks_dict_all_with_headway[116].shape, \
          '\n', actual_tracks_dict_all_with_headway[116][:1,:])
    
##########################################################
