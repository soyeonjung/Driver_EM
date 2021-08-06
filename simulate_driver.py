import numpy as np
import random 

from driver_model_EM import *  #import other python file


def get_xy_sd_cd(sd_next_pred, xy_prev, veh_lane, lanes_rxy, lanes_s):

    ## get xy coordinates
    sp_c, sp_d = lanes_s['sp_c'], lanes_s['sp_d']
    if veh_lane == 'c':
        sp = sp_c
    elif veh_lane == 'd':
        sp = sp_d
        
    lane_xi, lane_yi = sp.calc_position(sd_next_pred[0])
    if lane_xi == None:
        # print('get_sd_cd, lane_xi==None')
        return xy_prev, sd_next_pred  ##고쳐야함
    
    else:
        yawi = sp.calc_yaw(sd_next_pred[0])
        xi_ret = lane_xi + sd_next_pred[1] * math.cos(yawi + math.pi / 2.0)
        yi_ret = lane_yi + sd_next_pred[1] * math.sin(yawi + math.pi / 2.0)
        veh_xyi = np.array([xi_ret, yi_ret])

        # s-d coordinate about lane_cd_rxy
        lane_cd_rxy = lanes_rxy['lane_cd_rxy']
        lane_cd_s = lanes_s['lane_cd_s'] 
        # print('lane_cd_rxy', lane_cd_rxy.shape, 'veh_xyi', veh_xyi.shape)
        # print('veh_xyi[:,None]', veh_xyi[None,:].shape)

        if veh_xyi.shape == (2,):
            distances_cd = cdist(veh_xyi[None,:], lane_cd_rxy, 'euclidean')
        if veh_xyi.shape == (2,1):
            distances_cd = cdist(veh_xyi.reshape((1,2)), lane_cd_rxy, 'euclidean')
        veh_s_cd = lane_cd_s[np.argmin(distances_cd)]

        if np.argmin(distances_cd) == 0:
            lane_xyi_cd = lane_cd_rxy[np.argmin(distances_cd)+1, :]
            lane_xyi_prev_cd = lane_cd_rxy[np.argmin(distances_cd), :]
        else:
            lane_xyi_cd = lane_cd_rxy[np.argmin(distances_cd), :]
            lane_xyi_prev_cd = lane_cd_rxy[np.argmin(distances_cd)-1, :]
        [lane_dxi_cd, lane_dyi_cd]  = np.subtract(lane_xyi_cd, lane_xyi_prev_cd)
        [veh_dxi_cd, veh_dyi_cd] = np.subtract(veh_xyi, lane_xyi_prev_cd)
        signi_cd = np.sign(lane_dxi_cd * veh_dyi_cd - lane_dyi_cd * veh_dxi_cd)
        veh_d_cd = np.min(distances_cd) * signi_cd

    return veh_xyi, [veh_s_cd, veh_d_cd]



def simulate_pos_bunch_em(tracks_dict, mle_probs_final_all_veh, params_range, lanes_rxy, lanes_s, t_0, t_end):    
    
    v_des_all_veh, eps_all_veh, pol_all_veh, lam_all_veh = dict(), dict(), dict(), dict() 
    M = 1 #Monte Carlo simulation for each vehicle
    
    for veh_id, mle_probs_final in mle_probs_final_all_veh.items():
        # print('mle_probs_final', mle_probs_final)
        
        v_des_all_veh[veh_id] = []
        eps_all_veh[veh_id] = []
        pol_all_veh[veh_id] = []
        lam_all_veh[veh_id] = []
        
        for m in range(M):
            v_des_all_veh[veh_id].append(np.random.choice(params_range[0], size=1, p=mle_probs_final[0])[0])
            eps_all_veh[veh_id].append(np.random.choice(params_range[1], size=1, p=mle_probs_final[1])[0])
            pol_all_veh[veh_id].append(np.random.choice(params_range[2], size=1, p=mle_probs_final[2])[0])
            lam_all_veh[veh_id].append(np.random.choice(params_range[3], size=1, p=mle_probs_final[3])[0])  
        print('veh_id', veh_id, 'v_des', v_des_all_veh[veh_id], 'eps', eps_all_veh[veh_id], 'pol', pol_all_veh[veh_id], 'lam', lam_all_veh[veh_id])
        
                
    tracks_halluc_dict = dict()
    change_lane_switch = dict()
    # tracks_samples_dict = dict()
    t_range = np.arange(t_0, t_end)
    
    rmse_list = []
    ll_list = []
    
    for t in t_range:
        print('##### t: %s ######' % t)
        for veh_id, track in tracks_dict.items():
            
            idx_t = np.where(tracks_dict[veh_id][:,0] == t)[0]

            if len(idx_t) == 1 and veh_id > 201 and veh_id in mle_probs_final_all_veh.keys():
            # if len(idx_t) == 1 and veh_id > 201:
                
                if veh_id not in tracks_halluc_dict.keys():   # if new to tracks_halluc_dict
                    print('----- veh_id %s -----, idx_t %s. New to tracks_halluc_dict.' % (veh_id, idx_t[0]))

                    track_0 = tracks_dict[veh_id][idx_t[0], :21]
                    tracks_halluc_dict[veh_id] = track_0[None,:]

                    # change_lane_switch[veh_id] = 0
                    change_lane_switch[veh_id] = [[t, 0]]

                    # tracks_samples_dict[veh_id] = []
                    rmse_list.append([tracks_dict[veh_id][idx_t[0], 1:3], tracks_dict[veh_id][idx_t[0], 1:3], veh_id, t])  #[s_true, s_pred, veh_id, t]      
                    
                else:   # if already exists in tracks_halluc_dict
                    print('===== veh_id %s =====. idx_t %s. Lane-change-switch is %s.' % (veh_id, idx_t[0], change_lane_switch[veh_id][-1][1]))

                    idx_own_0 = np.where(tracks_halluc_dict[veh_id][:,0] == t-1)[0]
                    # veh_id_hw = tracks_halluc_dict[veh_id][idx_own_0[0], 11]

                    if veh_id not in mle_probs_final_all_veh.keys():
                        veh_id_fake = random.choice(list(mle_probs_final_all_veh.keys()))

                    for m in range(M):
                        if veh_id not in mle_probs_final_all_veh.keys():
                            v_des = v_des_all_veh[veh_id_fake][m]
                            eps = eps_all_veh[veh_id_fake][m]
                            pol = pol_all_veh[veh_id_fake][m]
                            lam = lam_all_veh[veh_id_fake][m]

                        else:
                            v_des = v_des_all_veh[veh_id][m]
                            eps = eps_all_veh[veh_id][m]
                            pol = pol_all_veh[veh_id][m]
                            lam = lam_all_veh[veh_id][m]

                        ### prob of changing lane
                        prob_lane, _, veh_id_old, veh_id_new = get_lane_change_prob_EM(veh_id, tracks_halluc_dict[veh_id][idx_own_0[0],:], \
                                                                v_des, pol, lam, \
                                                                tracks_halluc_dict, mle_probs_final_all_veh, params_range)
                        print('   Veh_id of old_follower is %s. new_follower is %s. Prob-change-lane is %s.' % (veh_id_old, veh_id_new, prob_lane))
                        

                        ################################################################
                        ################################################################
                        if change_lane_switch[veh_id][-1][1] == 0:  ##if lane-change has not started or has finished
                            # print("\n...Lane-change switch is 0...")

                            if prob_lane >= 0.52: # and veh_id_old 
                                # print('   -----veh_id', veh_id, '----- , idx_t', idx_t)
                                print("   ...Initiated lane-change, prob_lane", prob_lane)

                                sd_next_pred, vsvd_next_pred, lane_pred = hallucinate_a_step_change(tracks_halluc_dict[veh_id][idx_own_0[0],:], \
                                    lanes_rxy, lanes_s)  #pd controller (lane change) 

                                print('   Lane_pred is %s. <= Previous_lane was %s. sd_next_pred is %s.' % (lane_pred, tracks_halluc_dict[veh_id][idx_own_0[0],10], sd_next_pred))
                                # print('sd_next_pred', sd_next_pred)
                                xy_next_pred, sd_cd_next_pred = get_xy_sd_cd(sd_next_pred, tracks_halluc_dict[veh_id][idx_own_0[0],1:3], lane_pred, lanes_rxy, lanes_s)

                                if lane_pred != tracks_halluc_dict[veh_id][idx_own_0[0],10]:  #if just crossed the line bet. two lanes (=lane_pred is not the same as prev lane)
                                    # print('   Lane_predicted has changed to ', lane_pred, ' from', tracks_halluc_dict[veh_id][idx_own_0[0],10])
                                    print('   ...Lane_predicted has changed.')

                                    if veh_id_new is not None:
                                        idx_hw_new = np.where(tracks_halluc_dict[veh_id_new][:,0] == t-1)
                                        print('   idx_hw_new', idx_hw_new)

                                    if veh_id_new is not None and idx_hw_new is not None and len(idx_hw_new) == 1:  #if a new follower exists
                                        idx_hw_new = idx_hw_new[0]

                                        vs_new_0 = tracks_halluc_dict[veh_id_new][idx_hw_new[0], 5]
                                        vsvd_next_pred[0] = (vsvd_next_pred[0] + vs_new_0) / 2

                                        dist_headway_pred = tracks_halluc_dict[veh_id_new][idx_hw_new[0], 14] - sd_next_pred[0]
                                        # print('   idx_headway of new_follower', idx_hw_new[0], ', dist_headway_pred', dist_headway_pred) 
                                        print('   New_follower exists. => dist_headway_pred', dist_headway_pred)
                                        track_headway_new = np.concatenate((tracks_halluc_dict[veh_id_new][idx_hw_new[0],11:19],  #DEBUG: headway_idx
                                                                            [dist_headway_pred], [lane_pred]))
                                                                            # [idx_hw_new[0]], [dist_headway_pred], [lane_pred]))
                                    else:  #if a new follower doesn't exist
                                        veh_id_lead, idx_lead, dist_lead = None, None, 100
                                        for veh_id_other, track_other in tracks_halluc_dict.items():
                                            if veh_id_other != veh_id and t-1 >= track_other[0,0] and t-1 <= track_other[-1,0]:
                                                idx_other = np.where(track_other[:,0] == t-1)[0]
                                                if idx_other is not None and len(idx_other) == 1 and track_other[idx_other[0], 10] == lane_pred:
                                                    dist_other = track_other[idx_other[0], 3] - sd_next_pred[0]
                                                    if dist_other > 0 and dist_other < dist_lead: #if veh_id_other is closest & in front of veh_id
                                                        veh_id_lead, idx_lead, dist_lead = veh_id_other, idx_other[0], dist_other
                                                        track_headway_new = np.concatenate(([veh_id_lead],
                                                                                            tracks_halluc_dict[veh_id_lead][idx_lead[0],1:7],
                                                                                            idx_lead, dist_lead, [lane_pred]))
                                                                                            # [idx_lead[0]], [dist_lead], [lane_pred]))
                                        if veh_id_lead is not None and idx_lead is not None:
                                            print('   New_follower doesnt exist. => veh_id_lead: %s, idx_lead: %s, dist_lead: %s, lane_pred: %s' % (veh_id_lead, idx_lead, dist_lead, lane_pred))

                                    if np.abs(sd_next_pred[1]) < 0.2 :  ##if lane has changed and merged into new lane smoothly
                                        change_lane_switch[veh_id].append([t, 0])
                                    else:                               ##if lane has changed but haven't merged into new lane yet 
                                        change_lane_switch[veh_id].append([t, 2])

                                        
                                else:   #if not crosed the line yet (=if lane_pred is the same as prev lane)
                                    print('   ...Lane_predicted did not change yet.')
                                    dist_headway_pred = tracks_halluc_dict[veh_id][idx_own_0[0], 14] - sd_next_pred[0]
                                    print('   dist_headway_predicted of own_vehicle', dist_headway_pred)
                                    track_headway_new = np.concatenate((tracks_halluc_dict[veh_id][idx_own_0[0], 11:19],
                                                                        [dist_headway_pred],
                                                                        [tracks_halluc_dict[veh_id][idx_own_0[0], 20]]))
                                    change_lane_switch[veh_id].append([t, 1])

                                track_pred = np.concatenate(([t], # t
                                                             xy_next_pred, #x,y
                                                             sd_next_pred, vsvd_next_pred, #s,d,vs,vd
                                                             tracks_halluc_dict[veh_id][idx_own_0[0], 7:10], #agent_type, length, width
                                                             [lane_pred], #lane
                                                             track_headway_new  #headway_veh_id, headway_x/y/s/d/vs/vd, headway_idx, dist_headway, headway_lane 
                                                             )) #s_cd, d_cd    
                                tracks_halluc_dict[veh_id] = np.vstack((tracks_halluc_dict[veh_id], track_pred)) 
                                # print(" ====> track_pred", track_pred, '\n')
                                             
                                
                            elif prob_lane < 0.52:        
                                # print('\nnot change lane... prob_lane:', prob_lane)
                                sd_next_pred, vsvd_next_pred, _ = hallucinate_a_step_no_change(tracks_halluc_dict[veh_id][idx_own_0[0],:], v_des)  #IDM
                                # print('sd_next_pred', sd_next_pred)
                                xy_next_pred, sd_cd_next_pred = get_xy_sd_cd(sd_next_pred, tracks_halluc_dict[veh_id][idx_own_0[0],1:3],
                                                                             tracks_halluc_dict[veh_id][idx_own_0[0],10], lanes_rxy, lanes_s)
                                # print('xy_next_pred', xy_next_pred.shape, xy_next_pred)
                                if xy_next_pred.shape == (2,1): 
                                    xy_next_pred = xy_next_pred.reshape((2,))
                                
                                veh_id_lead, idx_lead, dist_lead = None, None, 100
                                for veh_id_other, track_other in tracks_halluc_dict.items():
                                    if veh_id_other != veh_id and t-1 >= track_other[0,0] and t-1 <= track_other[-1,0]:
                                        idx_other = np.where(track_other[:,0] == t-1)
                                        if idx_other is not None and len(idx_other) == 1 and track_other[idx_other[0], 10] == tracks_halluc_dict[veh_id][idx_own_0[0],10]:
                                            dist_other = track_other[idx_other[0], 3] - sd_next_pred[0]
                                            if dist_other > 0 and dist_other < dist_lead: #if veh_id_other is closest & in front of veh_id
                                                veh_id_lead, idx_lead, dist_lead = veh_id_other, idx_other[0], dist_other
                                                # track_headway_new = np.concatenate(([veh_id_lead], tracks_halluc_dict[veh_id_lead][idx_lead[0],1:7],
                                                #                                     idx_lead, dist_lead, tracks_halluc_dict[veh_id_lead][idx_other[0],10]))
                                if veh_id_lead is not None and idx_lead is not None:
                                    print('   IDM. => veh_id_headway: %s, idx_hw: %s, dist_hw: %s' % (veh_id_lead, idx_lead, dist_lead))
                                    track_headway_new = np.concatenate(([veh_id_lead], tracks_halluc_dict[veh_id_lead][idx_lead[0],1:7],
                                                                        idx_lead, dist_lead, tracks_halluc_dict[veh_id_lead][idx_other[0],10]))
                                else:
                                    dist_headway_pred = tracks_dict[veh_id][idx_t[0], 14] - sd_next_pred[0]
                                    track_headway_new = np.concatenate((tracks_dict[veh_id][idx_t[0], 11:19], #headway_veh_id, headway_x/y/s/d/vs/vd, headway_idx
                                                                        [dist_headway_pred], #dist_headway
                                                                        [tracks_dict[veh_id][idx_t[0], 20]] #headway_lane))
                                                                        ))

                                track_pred = np.concatenate(([t], # t
                                                             xy_next_pred, #x,y
                                                             sd_next_pred, vsvd_next_pred, #s,d,vs,vd
                                                             tracks_halluc_dict[veh_id][idx_own_0[0], 7:11], #agent_type, length, width, lane
                                                             track_headway_new  #headway_veh_id, headway_x/y/s/d/vs/vd, headway_idx, dist_headway, headway_lane
                                                             )) #s_cd, d_cd                            
                                tracks_halluc_dict[veh_id] = np.vstack((tracks_halluc_dict[veh_id], track_pred)) 
                                change_lane_switch[veh_id].append([t, 0])

                            else:
                                print('Error... prob_lane', prob_lane)

                                
                        ################################################################
                        ################################################################        
                        elif change_lane_switch[veh_id][-1][1] == 1:  ##if lane-change has started and not crossed the line yet
                            # print('   -----veh_id', veh_id, '----- , idx_t', idx_t)
                            # print("   Lane-change switch is 1...")
                            
                            sd_next_pred, vsvd_next_pred, lane_pred = hallucinate_a_step_change(tracks_halluc_dict[veh_id][idx_own_0[0],:], \
                               lanes_rxy, lanes_s)  #pd controller (lane change) 
                            xy_next_pred, sd_cd_next_pred = get_xy_sd_cd(sd_next_pred, tracks_halluc_dict[veh_id][idx_own_0[0],1:3], lane_pred, lanes_rxy, lanes_s)
                            # print('   sd_next_pred is %s.' % (sd_next_pred))
                            print('   Lane_pred is %s. <= Previous_lane was %s. sd_next_pred is %s.' % (lane_pred, tracks_halluc_dict[veh_id][idx_own_0[0],10], sd_next_pred))

                            ####################################################################
                            # if lane_pred != tracks_halluc_dict[veh_id][idx_own_0[0],10]:  #if just crossed the line bet. two lanes (=if lane_pred is not the same as prev lane
                            #     # print('   Lane_predicted has changed to %s from %s.' % (lane_pred, tracks_halluc_dict[veh_id][idx_own_0[0],10]))
                            #     print('   ...Lane_predicted has changed.')

                            #     if veh_id_new is not None:
                            #         idx_hw_new = np.where(tracks_halluc_dict[veh_id_new][:,0] == t-1)
                            #         # print('   idx_hw_new', idx_hw_new)

                            #     if idx_hw_new is not None and len(idx_hw_new) == 1:  #if a new follower exists
                            #         idx_hw_new = idx_hw_new[0]

                            #         vs_new_0 = tracks_halluc_dict[veh_id_new][idx_hw_new[0], 5]
                            #         vsvd_next_pred[0] = (vsvd_next_pred[0] + vs_new_0) / 2

                            #         dist_headway_pred = tracks_halluc_dict[veh_id_new][idx_hw_new[0], 14] - sd_next_pred[0]
                            #         # print('   idx_headway of new_follower', idx_hw_new[0], ', dist_headway_pred', dist_headway_pred) 
                            #         print('   New_follower exists. => dist_headway_pred', dist_headway_pred)
                            #         track_headway_new = np.concatenate((tracks_halluc_dict[veh_id_new][idx_hw_new[0],11:19],
                            #                                             # [idx_hw_new[0]], [dist_headway_pred], [lane_pred]))
                            #                                             [dist_headway_pred], [lane_pred]))

                            #     else:  #if a new follower doesn't exist
                            #         veh_id_lead, idx_lead, dist_lead = None, None, 100
                            #         for veh_id_other, track_other in tracks_halluc_dict.items():
                            #             if veh_id_other != veh_id and t-1 >= track_other[0,0] and t-1 <= track_other[-1,0]:
                            #                 idx_other = np.where(track_other[:,0] == t-1)
                            #                 if idx_other is not None and len(idx_other) == 1 and track_other[idx_other[0], 10] == lane_pred:
                            #                     dist_other = track_other[idx_other[0], 3] - sd_next_pred[0]
                            #                     if dist_other > 0 and dist_other < dist_lead: #if veh_id_other is closest & in front of veh_id
                            #                         veh_id_lead, idx_lead, dist_lead = veh_id_other, idx_other[0], dist_other
                            #         print('   New_follower doesnt exist. => veh_id_lead', veh_id_lead, 'idx_lead', idx_lead, ', dist_lead', dist_lead, 'lane_pred', lane_pred) 
                            #         # if veh_id_lead is not None and idx_lead is not None:
                            #         track_headway_new = np.concatenate(([veh_id_lead], tracks_halluc_dict[veh_id_lead][idx_lead[0],1:7],
                            #                                             idx_lead, dist_lead, [lane_pred]))
                            #                                             # [idx_lead[0]], [dist_lead], [lane_pred]))
                                    
                            #     if np.abs(sd_next_pred[1]) < 0.1 :  ##if lane has changed and merged into new lane smoothly
                            #         change_lane_switch[veh_id].append([t, 0])
                            #     else:                               ##if lane has changed but haven't merged into new lane yet 
                            #         change_lane_switch[veh_id].append([t, 2])

                            # else:   #if not crossed the line yet (=if lane_pred is the same as prev lane)
                            #     print('   ...Lane_predicted did not change yet.')
                            #     dist_headway_pred = tracks_halluc_dict[veh_id][idx_own_0[0], 14] - sd_next_pred[0]
                            #     print('   dist_headway_pred of old_follower', dist_headway_pred) 
                            #     track_headway_new = np.concatenate((tracks_halluc_dict[veh_id][idx_own_0[0], 11:19],
                            #                                         [dist_headway_pred],
                            #                                         [tracks_halluc_dict[veh_id][idx_own_0[0], 20]]))
                            #     change_lane_switch[veh_id].append([t, 1])
                            ####################################################################

                            if veh_id_new is not None:
                                idx_hw_new = np.where(tracks_halluc_dict[veh_id_new][:,0] == t-1)
                                print('   idx_hw_new', idx_hw_new)

                            if veh_id_new is not None and idx_hw_new is not None and len(idx_hw_new) == 1:  #if a new follower exists
                                idx_hw_new = idx_hw_new[0]

                                vs_new_0 = tracks_halluc_dict[veh_id_new][idx_hw_new[0], 5]
                                vsvd_next_pred[0] = max(vsvd_next_pred[0], (vsvd_next_pred[0] + vs_new_0) / 2)

                                dist_headway_pred = tracks_halluc_dict[veh_id_new][idx_hw_new[0], 14] - sd_next_pred[0]
                                # print('   idx_headway of new_follower', idx_hw_new[0], ', dist_headway_pred', dist_headway_pred) 
                                print('   New_follower exists. => dist_headway_pred', dist_headway_pred)
                                track_headway_new = np.concatenate((tracks_halluc_dict[veh_id_new][idx_hw_new[0],11:19],
                                                                    [dist_headway_pred], [lane_pred]))
                                                                    # [idx_hw_new[0]], [dist_headway_pred], [lane_pred]))
                            else:  #if a new follower doesn't exist
                                veh_id_lead, idx_lead, dist_lead = None, None, 100
                                for veh_id_other, track_other in tracks_halluc_dict.items():
                                    if veh_id_other != veh_id and t-1 >= track_other[0,0] and t-1 <= track_other[-1,0]:
                                        idx_other = np.where(track_other[:,0] == t-1)
                                        if idx_other is not None and len(idx_other) == 1 and track_other[idx_other[0], 10] == lane_pred:
                                            dist_other = track_other[idx_other[0], 3] - sd_next_pred[0]
                                            if dist_other > 0 and dist_other < dist_lead: #if veh_id_other is closest & in front of veh_id
                                                veh_id_lead, idx_lead, dist_lead = veh_id_other, idx_other[0], dist_other
                                                track_headway_new = np.concatenate(([veh_id_lead],
                                                                                    tracks_halluc_dict[veh_id_lead][idx_lead[0],1:7],
                                                                                    idx_lead, dist_lead, [lane_pred]))
                                                                                    # [idx_lead[0]], [dist_lead], [lane_pred]))
                                if veh_id_lead is not None and idx_lead is not None:
                                    print('   New_follower doesnt exist. => veh_id_lead: %s, idx_lead: %s, dist_lead: %s, lane_pred: %s' % (veh_id_lead, idx_lead, dist_lead, lane_pred))
                            

                            if lane_pred != tracks_halluc_dict[veh_id][idx_own_0[0],10]:  #if just crossed the line bet. two lanes (=if lane_pred is not the same as prev lane
                                # print('   Lane_predicted has changed to %s from %s.' % (lane_pred, tracks_halluc_dict[veh_id][idx_own_0[0],10]))
                                print('   ...Lane_predicted has changed.')
                                if np.abs(sd_next_pred[1]) < 0.2 :  ##if lane has changed and merged into new lane smoothly
                                    change_lane_switch[veh_id].append([t, 0])
                                else:                               ##if lane has changed but haven't merged into new lane yet 
                                    change_lane_switch[veh_id].append([t, 2])
                            else:   #if not crossed the line yet (=if lane_pred is the same as prev lane)
                                print('   ...Lane_predicted did not change yet.')
                                change_lane_switch[veh_id].append([t, 1])
                            ####################################################################

                            track_pred = np.concatenate(([t], # t
                                                         xy_next_pred, #x,y
                                                         sd_next_pred, vsvd_next_pred, #s,d,vs,vd
                                                         tracks_halluc_dict[veh_id][idx_own_0[0], 7:10], #agent_type, length, width
                                                         [lane_pred], #lane
                                                         track_headway_new # #headway_veh_id, headway_x/y/s/d/vs/vd, headway_idx, #dist_headway #headway_lane 
                                                         )) #s_cd, d_cd    
                            tracks_halluc_dict[veh_id] = np.vstack((tracks_halluc_dict[veh_id], track_pred)) 
                            # print(" ====> track_pred", track_pred, '\n')
                               

                        ################################################################
                        ################################################################
                        elif change_lane_switch[veh_id][-1][1] == 2:  ##if lane has changed (just crossed the line), but haven't merged into new lane yet 
                            # print('   -----veh_id', veh_id, '----- , idx_t', idx_t)
                            # print("   ...Lane-change switch is 2...")

                            track_0 = tracks_halluc_dict[veh_id][idx_own_0[0],:]                            
                            if track_0[10] == 'c': 
                                # offset = track_0[4] ** 2 * ((track_0[4] > 0) - (track_0[4] < 0))
                                offset = track_0[4]
                            elif track_0[10] == 'd': 
                                # offset = track_0[4] ** 2 * ((track_0[4] > 0) - (track_0[4] < 0))
                                offset = track_0[4]
                            print('   track_0: s_lat=%s, v_lat=%s, lane=%s. offset=%s.' % (track_0[4], track_0[6], track_0[10], offset))

                            if veh_id_new is not None:
                                idx_hw_new = np.where(tracks_halluc_dict[veh_id_new][:,0] == t-1)
                                print('   idx_hw_new', idx_hw_new)

                            if veh_id_new is not None and idx_hw_new is not None and len(idx_hw_new) == 1:  #if a new follower exists   
                                idx_hw_new = idx_hw_new[0]
                                track_new_0 = tracks_halluc_dict[veh_id_new][idx_hw_new[0], :]
                                # print('   idx_hw_new, track_new_0',idx_hw_new, track_new_0)
                                v_lon_pred = max(track_0[5], (track_0[5] + track_new_0[5]) / 2) #vsvd_next_pred[0] #track_i[5]
                            else:
                                v_lon_pred = track_0[5]  #vsvd_next_pred[0] 
                            s_lon_pred = track_0[3] + track_0[5] #sd_next_pred[0]  #track_i[3] + track_i[5]
                            # sd_next_pred_idm, vsvd_next_pred_idm, _ = hallucinate_a_step_no_change(track_0, v_des)   
                            
                            kp, kd = 0.01, 0.005
                            a_lat_pred = - offset * kp - track_0[6] * kd
                            v_lat_pred = track_0[6] + a_lat_pred
                            s_lat_pred = track_0[4] + track_0[6] + 0.5 * a_lat_pred
                            print('   s_lat_pred', s_lat_pred)
                            
                            vsvd_next_pred = [v_lon_pred, v_lat_pred]
                            sd_next_pred = [s_lon_pred, s_lat_pred]
                            
                            xy_next_pred, _ = get_xy_sd_cd(sd_next_pred, tracks_halluc_dict[veh_id][idx_own_0[0],1:3],
                                                                         tracks_halluc_dict[veh_id][idx_own_0[0],10], lanes_rxy, lanes_s)
                            if xy_next_pred.shape == (2,1): 
                                xy_next_pred = xy_next_pred.reshape((2,))
                            dist_headway_pred = tracks_halluc_dict[veh_id][idx_own_0[0], 14] - sd_next_pred[0]

                            track_pred = np.concatenate(([t], # t
                                                         xy_next_pred, #x,y
                                                         sd_next_pred, vsvd_next_pred, #s,d,vs,vd
                                                         tracks_halluc_dict[veh_id][idx_own_0[0], 7:11], #agent_type, length, width, lane
                                                         tracks_halluc_dict[veh_id][idx_own_0[0], 11:19], #headway_veh_id, headway_x/y/s/d/vs/vd, headway_idx
                                                         [dist_headway_pred], #dist_headway
                                                         [tracks_halluc_dict[veh_id][idx_own_0[0], 20]] #headway_lane
                                                         )) #s_cd, d_cd                            
                            tracks_halluc_dict[veh_id] = np.vstack((tracks_halluc_dict[veh_id], track_pred)) 
                            
                            if np.abs(sd_next_pred[1]) < 0.2 :  ##if lane has changed and merged into new lane smoothly
                                change_lane_switch[veh_id].append([t, 0])
                            else:                               ##if lane has changed but haven't merged into new lane yet 
                                change_lane_switch[veh_id].append([t, 2])
                            
                        # print('sd_next_pred', type(sd_next_pred), len(sd_next_pred), 'tracks_dict[veh_id][idx_t[0],3:5]', type(tracks_dict[veh_id][idx_t[0],3:5]), len(tracks_dict[veh_id][idx_t[0],3:5]))
                        # ll_list.append(multi_normal(np.array(sd_next_pred), np.diag([eps]*2)).pdf(tracks_dict[veh_id][idx_t[0],3:5]))
                        # rmse_list.append([tracks_dict[veh_id][idx_t[0], 3:5], sd_next_pred, veh_id, t])  #[sd_true, sd_pred, veh_id, t]      
                        rmse_list.append([tracks_dict[veh_id][idx_t[0], 1:3], xy_next_pred, veh_id, t])  #[sd_true, sd_pred, veh_id, t]      

        print('\n')

        
    rmse_list = np.array(rmse_list)
    # ll_list = np.array(ll_list)
    
    sp_c, sp_d = lanes_s['sp_c'], lanes_s['sp_d']

    tracks_xy_halluc_dict = dict()
    for veh_id, track_halluc in tracks_halluc_dict.items():
        veh_s, veh_d = track_halluc[:,3], track_halluc[:,4]

        veh_x_ret, veh_y_ret = [], []
        for i in range(len(veh_s)):

            if track_halluc[i,10] == 'c': 
                sp = sp_c
            elif track_halluc[i,10] == 'd': 
                sp = sp_d

            lane_xi, lane_yi = sp.calc_position(veh_s[i])
            if lane_xi == None:
                terminate_i = i
                break

            yawi = sp.calc_yaw(veh_s[i])
            xi_ret = lane_xi + veh_d[i] * math.cos(yawi + math.pi / 2.0)
            yi_ret = lane_yi + veh_d[i] * math.sin(yawi + math.pi / 2.0)

            veh_x_ret.append(xi_ret)
            veh_y_ret.append(yi_ret)

        track_xy_ret = np.column_stack((track_halluc[:len(veh_x_ret),0], veh_x_ret, veh_y_ret, track_halluc[:len(veh_x_ret),3:]))
        tracks_xy_halluc_dict[veh_id] = track_xy_ret
        # print(len(tracks_halluc_dict[veh_id]), len(tracks_xy_halluc_dict[veh_id])
        
    return tracks_xy_halluc_dict, change_lane_switch, rmse_list, ll_list


##########################################################
if __name__ == "__main__":
    import pickle

    pass