from read_tracks_lanes import *
import pytest

import numpy as np
import pickle, os
# from matplotlib import pyplot as plt


def test_load_centerlines():
    centerlines_path = 'dataset/modified_centerlines_/'
    lanes_rxy, lanes_s, centerlines_dict = load_centerlines(centerlines_path)
    
    assert list(lanes_rxy.keys()) == ['lane_c_rxy', 'lane_d_rxy', 'lane_cd_rxy']
    assert lanes_rxy['lane_cd_rxy'].shape == (1000, 2)
    
    assert list(lanes_s.keys()) == ['lane_c_s', 'lane_d_s', 'lane_cd_s']
    assert lanes_s['lane_cd_s'].shape == (1000, 2) # Should spit out error! (1000,)
    
    assert list(centerlines_dict.keys()) == ['c', 'd']
    assert centerlines_dict['c'].shape == (1000, 2) # Should spit out error! (100, 2)
    
    
    
def test_load_construct():
    centerlines_path = 'dataset/modified_centerlines_/'
    lanes_rxy, lanes_s, centerlines_dict = load_centerlines(centerlines_path)
    
    veh_tracks_path = '../processed/INTERACTION/DR_CHN_Merging_ZS/'
    actual_tracks_dict_all = load_tracks(veh_tracks_path, lanes_rxy, lanes_s, centerlines_dict)
    assert actual_tracks_dict_all[116].shape[1] == 11
    assert type(actual_tracks_dict_all[116][0,8]) == str
    
    actual_tracks_dict_all_with_headway = construct_tracks_with_headway(actual_tracks_dict_all, lanes_rxy, lanes_s)
    assert actual_tracks_dict_all_with_headway[116].shape[1] == 22  # Should spit out error! Should be 23

    

# ##########################################################
# ##########################################################
# # actual_tracks_dict_all_with_headway['veh_id'][:, 0] = frame_id  #t(in 0.1s)
# # actual_tracks_dict_all_with_headway['veh_id'][:, 1:3] = x, y
# # actual_tracks_dict_all_with_headway['veh_id'][:, 3:7] = s, d, vs, vd   #(vs,vd : m/s)
# # actual_tracks_dict_all_with_headway['veh_id'][:, 7:11] = agent_type, length, width, lane
# # actual_tracks_dict_all_with_headway['veh_id'][:, 11:14] = headway_veh_id, headway_x, headway_y 
# # actual_tracks_dict_all_with_headway['veh_id'][:, 14:18] = headway_s, headway_d, headway_vs, headway_vd
# # actual_tracks_dict_all_with_headway['veh_id'][:, 18:21] = headway_idx, dist_headway, headway_lane
# # actual_tracks_dict_all_with_headway['veh_id'][:, 21:23] = s_cd, d_cd  #s-d coord. about lane_cd_xy

# ##########################################################
# ##########################################################