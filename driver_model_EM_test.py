from driver_model_EM import *

import numpy as np
import pickle, os
import itertools
from matplotlib import pyplot as plt

# # actual_tracks_dict_all_with_headway[:][0] = frame_id  #t(in 0.1s)
# # actual_tracks_dict_all_with_headway[:][1:3] = x, y
# # actual_tracks_dict_all_with_headway[:][3:7] = s, d, vs, vd   #(vs,vd : m/s)
# # actual_tracks_dict_all_with_headway[:][7:11] = agent_type, length, width, lane
# # actual_tracks_dict_all_with_headway[:][11:14] = headway_veh_id, headway_x, headway_y 
# # actual_tracks_dict_all_with_headway[:][14:18] = headway_s, headway_d, headway_vs, headway_vd
# # actual_tracks_dict_all_with_headway[:][18:21] = headway_idx, dist_headway, headway_lane
# # actual_tracks_dict_all_with_headway[:][21:23] = s_cd, d_cd  #s-d coord. about lane_cd_xy



def test_hallucinate_a_step_no_change():
    pass
    
def test_hallucinate_a_step_change():
    pass

def get_lane_change_prob_EM():
    pass    

    
    

##########################################################
##########################################################
## EM_DriverModel.py

def get_a_IDM(xt, st, z):
    vt, rt, dt = xt[0], xt[1], xt[2]
    v_des = z
    a_max, b_pref, tau, d_min = 2.0, 2.0, 1.5, 5.0
    
    d_des = d_min + tau * vt - (vt * rt) / (2 * (a_max * b_pref)**0.5)
    d_des = min(max(d_des, 3.0), 50)
    
    if dt is None:
        dt = d_des #150
    else:
        dt = min(max(dt, 0.0001), 100)

    a_IDM_mean = a_max * (1 - (vt/v_des)**4 - (d_des/dt)**2)
#     print('a_IDM_mean', a_IDM_mean)
    a_IDM_mean = min(max(a_IDM_mean,-0.1), 0.1)

#     print('----- get_a_IDM -----')
#     print('vt', vt, 'rt', rt, 'dt', xt[2], 'dt_rev', dt, 'st', st, 'v_des', z, 'd_des', d_des)
    
    return a_IDM_mean
    
def get_s_IDM(xt, st, z):
    vt = xt[0]
    a_IDM_mean = get_a_IDM(xt, st, z)
    s_IDM_mean = st + vt + 0.5 * a_IDM_mean    
    return s_IDM_mean

def hallucinate_a_step_no_change(track_i, p_z):
#     print('hallucinate_a_step_no_change')

#     print('\ntrack_i: vels', track_i[5], ',lane:', track_i[11], ',None?:', track_i[11]==None)
    s_prev, d_prev = track_i[3], track_i[4]
    v_0 = track_i[5]
    
    if track_i[11] != None:  #if headway vehicle exists
        r_0, d_0 = track_i[5]-track_i[16], track_i[19]
    else:
        r_0 = v_0
        d_0 = None #150 - s_prev
            
    s_pred = get_s_IDM([v_0, r_0, d_0], s_prev, p_z)
    if type(s_pred) != float and s_pred.shape == (1,):
        s_pred = s_pred[0]
    s_next_pred = min(max(s_pred, s_prev + v_0 - 0.05), s_prev + v_0 + 0.05)
    sd_next_pred = [s_next_pred, d_prev]

    a_pred = get_a_IDM([v_0, r_0, d_0], s_prev, p_z)
    v_pred = v_0 + a_pred
    v_pred = min(max(v_0, v_0 - 0.1), v_0 + 0.1)
    if type(v_pred) != float and v_pred.shape == (1,):
        v_pred = v_pred[0]
    vsvd_next_pred = np.array([v_pred, track_i[6]]).reshape((2,))
#     print('----- get_a_IDM -----')
#     print('v_0', v_0, 'r_0', r_0, 'd_0', d_0, 's_0', s_prev, 'v_des', p_z)
#     print('a_pred', a_pred, 'vsvd_next_pred', vsvd_next_pred, 'sd_next_pred', sd_next_pred)
    
    return sd_next_pred, vsvd_next_pred, a_pred
    
    
def get_sd_other_lane(sd_next_pred, xy_prev, lane_pred, lanes_rxy):
        
    ## get xy coordinates
    if lane_pred == 'c':
        sp_curr = sp_d
    elif lane_pred == 'd':
        sp_curr = sp_c
        
    lane_xi, lane_yi = sp_curr.calc_position(sd_next_pred[0])
    if lane_xi == None:
        return xy_prev, sd_next_pred  ##고쳐야함
    else:
        yawi = sp_curr.calc_yaw(sd_next_pred[0])
        xi_ret = lane_xi + sd_next_pred[1] * math.cos(yawi + math.pi / 2.0)
        yi_ret = lane_yi + sd_next_pred[1] * math.sin(yawi + math.pi / 2.0)
        veh_xyi_other = np.array([xi_ret, yi_ret])
       
    
    ## s-d coordinate about the lane_pred 
    if lane_pred == 'c':
        lane_rxy, lane_s = lane_c_rxy, lane_c_s
    elif lane_pred == 'd':
        lane_rxy, lane_s = lane_d_rxy, lane_d_s

    distances = cdist(np.array([veh_xyi_other]), lane_rxy, 'euclidean')
    veh_s_other = lane_s[np.argmin(distances)]

    if np.argmin(distances) == 0:
        lane_xyi = lane_rxy[np.argmin(distances)+1, :]
        lane_xyi_prev = lane_rxy[np.argmin(distances), :]
    else:
        lane_xyi = lane_rxy[np.argmin(distances), :]
        lane_xyi_prev = lane_rxy[np.argmin(distances)-1, :]
    [lane_dxi, lane_dyi] = np.subtract(lane_xyi, lane_xyi_prev)
    [veh_dxi, veh_dyi] = np.subtract(veh_xyi_other, lane_xyi_prev)
    signi = np.sign(lane_dxi * veh_dyi - lane_dyi * veh_dxi)
    veh_d_other = np.min(distances) * signi

    return veh_xyi_other, [veh_s_other, veh_d_other]


def hallucinate_a_step_change(track_i, lanes_rxy, z):
#     print('hallucinate_a_step_change')
    
    kp, kd = 0.02, 0

    if track_i[10] == 'c': 
        lane_curr_rxy = lanes_rxy['lane_c_rxy']; lane_new_rxy = lanes_rxy['lane_d_rxy']
    elif track_i[10] == 'd': 
        lane_curr_rxy = lanes_rxy['lane_d_rxy']; lane_new_rxy = lanes_rxy['lane_c_rxy']

    veh_xyi = track_i[1:3]
    
    distances_curr = cdist(np.array([veh_xyi]), lane_curr_rxy, 'euclidean')
    min_dist_curr = np.min(distances_curr)
    lane_curr_rxyi = lane_curr_rxy[np.argmin(distances_curr)]  #closest lane position

    distances_bet_lanes = cdist(np.array([lane_curr_rxyi]), lane_new_rxy, 'euclidean')
    min_dist_bet_lanes = np.min(distances_bet_lanes)
    lane_new_rxyi = lane_new_rxy[np.argmin(distances_bet_lanes)]  #closest lane_new position
    min_dist_new = cdist([veh_xyi], [lane_new_rxyi], 'euclidean')[0][0]
#     print('veh_xyi', veh_xyi, 'lane_curr_rxyi', lane_curr_rxyi, 'lane_new_rxyi', lane_new_rxyi)   
#     print('min_dist_curr', min_dist_curr, 'min_dist_new', min_dist_new)

    if min_dist_curr > min_dist_new: 
        if track_i[10] == 'c': 
            lane_pred = 'd'
        elif track_i[10] == 'd': 
            lane_pred = 'c'
    else:
        lane_pred = track_i[10]    
#     print('current_lane', track_i[10], 'lane_pred', lane_pred)


# #     print('-----hallucinate_a_step_change----')
# #     print('veh_xyi', veh_xyi, 'lane_curr_rxyi', lane_curr_rxyi, 'lane_new_rxyi', lane_new_rxyi)
#     plt.figure(figsize=(10,5))
#     plt.plot(lanes_rxy['lane_c_rxy'][:,0], lanes_rxy['lane_c_rxy'][:,1], ".r", ms=0.5)
#     plt.scatter(lane_curr_rxyi[0], lane_curr_rxyi[1], s=20, c='r')
#     plt.text(lane_curr_rxyi[0]+3, lane_curr_rxyi[1], np.round(min_dist,2), fontsize=10, va="center", ha="center", color='r')
    
#     plt.plot(lanes_rxy['lane_d_rxy'][:,0], lanes_rxy['lane_d_rxy'][:,1], ".g", ms=0.5)
#     plt.scatter(lane_new_rxyi[0], lane_new_rxyi[1], s=20, c='g')
#     plt.text(lane_new_rxyi[0]+3, lane_new_rxyi[1], np.round(min_dist_new,2), fontsize=10, va="center", ha="center", color='g')

#     plt.scatter(veh_xyi[0], veh_xyi[1], s=20, c='b')
#     plt.text(veh_xyi[0]+1, veh_xyi[1], track_i[10][0], fontsize=15, va="center", ha="center", color='b')
#     plt.xlim(1000, 1140); plt.ylim(950, 970)
#     plt.show()

    if track_i[10] == 'c': 
        offset = track_i[4] - min_dist_bet_lanes 
    elif track_i[10] == 'd': 
        offset = track_i[4] + min_dist_bet_lanes 
#     print('curr_d_pos', track_i[4],  'curr_vd', track_i[6], 'offset', offset)

#     sd_next_pred, vsvd_next_pred, _ = hallucinate_a_step_no_change(track_i, z)    
    v_lon_pred = track_i[5] #vsvd_next_pred[0] #track_i[5]
    s_lon_pred = track_i[3] + track_i[5] #sd_next_pred[0]  #track_i[3] + track_i[5]
    
    a_lat_pred = - offset * kp - track_i[6] * kd
    v_lat_pred = track_i[6] + a_lat_pred
    s_lat_pred = track_i[4] + track_i[6] + 0.5 * a_lat_pred    

    if lane_pred != track_i[10]:  #if next (predicted) lane is different from the current lane
#         print('lane_pred != track_i[10]', track_i[10], '=>', lane_pred)
        _, sd_next_pred = get_sd_other_lane([s_lon_pred, s_lat_pred], veh_xyi, lane_pred, lanes_rxy)
#         print('      sd:', [s_lon_pred, s_lat_pred], '=>', sd_next_pred)
    else:
        sd_next_pred = [s_lon_pred, s_lat_pred]

    return sd_next_pred, [v_lon_pred, v_lat_pred], lane_pred


def get_accel_neighbors_EM(veh_id, track_ego, z_i, actual_tracks_dict_all_with_headway, mle_probs_final_all_veh, params_range):
#     print('--- get_accel_neighbors_EM ---')

#     if len(mle_probs_final_all_veh.keys()) > 0:
#         print('get_accel_neighbors: mle_probs_final_all_veh', mle_probs_final_all_veh)

    t = track_ego[0]
    v_des_range = params_range[0]
    
    ## compute a_ego
    if veh_id in mle_probs_final_all_veh.keys():
        v_des_probs = mle_probs_final_all_veh[veh_id][0]
        v_des_ego = np.random.choice(v_des_range, size=1, p=v_des_probs)
    else: 
        v_des_ego = z_i
        
    _, _, a_ego = hallucinate_a_step_no_change(track_ego, v_des_ego)
    
    ## find old/new followers           
    veh_id_old, idx_old, dist_old = None, None, 100
    veh_id_new, idx_new, dist_new = None, None, 100
    for veh_id_other, track_other in actual_tracks_dict_all_with_headway.items():
        if veh_id_other != veh_id and t >= track_other[0,0] and t <= track_other[-1,0]:
            idx = np.where(track_other[:,0] == t)[0]
            
            if len(idx) == 1:
                dist_other = track_ego[3] - track_other[idx[0], 3]  #dist_s => should be positive
#                 dist_other = distance.euclidean(track_ego[1:3], track_other[idx[0], 1:3])  #dist_xy
                
                if dist_other > 0: #if veh_id_other is behind veh_id_ego
                    if track_ego[10] == track_other[idx[0],10] and dist_other < dist_old:
                        veh_id_old, idx_old, dist_old = veh_id_other, idx[0], dist_other
                    elif track_ego[10] != track_other[idx[0],10] and dist_other < dist_new:
                        veh_id_new, idx_new, dist_new = veh_id_other, idx[0], dist_other
#     print('veh_id_old', veh_id_old, ', veh_id_new', veh_id_new)
    
    ## compute a_old, a_old_tilde, a_new, a_new_tilde
    if veh_id_old is not None:
        track_old = actual_tracks_dict_all_with_headway[veh_id_old][idx_old,:]        
        v_old, r_old, d_old = track_old[5], track_old[5] - track_ego[5], track_ego[3] - track_old[3]
#         if track_old[16] != track_ego[5] or track_ego[3] - track_old[3] != track_old[19]:
#             print('veh_id_old: ', veh_id_old, 'headway', track_old[11], 'vel', track_old[16], track_ego[5], 'dist', track_ego[3] - track_old[3], track_old[19])
        s_prev_old = track_old[3]
        if veh_id_old in mle_probs_final_all_veh.keys():
            v_des_old_probs = mle_probs_final_all_veh[veh_id_old][0]
            v_des_old = np.random.choice(v_des_range, size=1, p=v_des_old_probs)        
        else: 
            v_des_old = z_i
        a_old = get_a_IDM([v_old, r_old, d_old], s_prev_old, v_des_old)
        
        if track_old[11] is not None and track_ego[11] is not None:
            r_old_tilde, d_old_tilde = track_old[5] - track_ego[16], d_old + track_ego[19]
        else:
            r_old_tilde, d_old_tilde = track_old[5], 100 #- track_old[3]
        a_old_tilde = get_a_IDM([v_old, r_old_tilde, d_old_tilde], s_prev_old, v_des_old)
        
#         print('veh_id_ego', veh_id, ', track_ego[11]', track_ego[11])
#         print('veh_id_old', veh_id_old, ', track_old[11]', track_old[11], ', d_old', d_old, ', d_old_tilde', d_old_tilde)
    else:
        a_old, a_old_tilde = 0.0, 0.0
        
            
    if veh_id_new is not None: 
        track_new = actual_tracks_dict_all_with_headway[veh_id_new][idx_new,:]
        if track_new[11] is not None:
            v_new, r_new, d_new = track_new[5], track_new[5] - track_new[16], track_new[19]
        else:
            v_new, r_new, d_new = track_new[5], track_new[5], 100 #- track_new[3]
                
        s_prev_new = track_new[3]
        if veh_id_new in mle_probs_final_all_veh.keys():
            v_des_new_probs = mle_probs_final_all_veh[veh_id_new][0]
            v_des_new = np.random.choice(v_des_range, size=1, p=v_des_new_probs)
        else:
            v_des_new = z_i
        a_new = get_a_IDM([v_new, r_new, d_new], s_prev_new, v_des_new)
        
        r_new_tilde, d_new_tilde = track_new[5] - track_ego[5], track_ego[3] - track_new[3]     
        a_new_tilde = get_a_IDM([v_new, r_new_tilde, d_new_tilde], s_prev_new, v_des_new)
        
        if track_new[11] is not None:
            v_ego, r_ego_tilde, d_ego_tilde = track_ego[5], track_ego[5]-track_new[16], track_new[14]-track_ego[3]
        else:
            v_ego, r_ego_tilde, d_ego_tilde = track_ego[5], track_ego[5], 100 #- track_ego[3]
        s_prev_ego = track_ego[3]
        a_ego_tilde = get_a_IDM([v_ego, r_ego_tilde, d_ego_tilde], s_prev_ego, v_des_ego)
    
#         print('veh_id_new', veh_id_new, ', track_new[11]', track_new[11], ', d_new', d_new, ', d_new_tilde', d_new_tilde)
    else:
        a_new, a_new_tilde = 0.0, 0.0
        a_ego_tilde = a_ego  ### 고쳐야함! veh_id_new가 없어도 lead는 있을수 있으니까..
        
    return [a_ego, a_ego_tilde, a_old, a_new, a_old_tilde, a_new_tilde], veh_id_new
                                
        
def get_lane_change_prob_EM(veh_id, track_i, z_i, pol_i, lam_i, actual_tracks_dict_all_with_headway, mle_probs_final_all_veh, params_range):
#     print('\n--- get_lane_change_prob_EM ---')
    [a_ego, a_ego_tilde, a_old, a_new, a_old_tilde, a_new_tilde], veh_id_new = get_accel_neighbors_EM(veh_id, track_i, z_i, actual_tracks_dict_all_with_headway, \
                                                                                                      mle_probs_final_all_veh, params_range)
#     print('--------------------------------')
    a_thr, b_safe = 0.1, 3.0
    
    if -a_new_tilde <= b_safe:
        mobil = a_ego_tilde - a_ego + pol_i * (a_new_tilde - a_new + a_old_tilde - a_old) - a_thr
        sigmoid_mobil = 1/(1 + np.exp(- lam_i * mobil))

        return sigmoid_mobil, mobil, veh_id_new
    else:
        return 0, 0, veh_id_new
    
    
def EM(veh_id, track, actual_tracks_dict_all_with_headway, lanes_rxy, params_range, mle_probs_final_all_veh, max_iter_per_veh):

    idx_range = [range(len(p_range)) for p_range in params_range]
    [z_range, eps_range, pol_range, lam_range] = params_range
    
#     mle_probs = [np.ones_like(p_range) / len(p_range) for p_range in params_range]
    mle_probs = [np.random.dirichlet(np.ones_like(p_range)) for p_range in params_range]
    print('initial mle_probs', mle_probs)
    
    mle_probs_history= []
    marginal_ll_history = []
    prev_marginal_ll = -np.infty
    tol = 1 #0.1
    
    for it in range(1, max_iter_per_veh):
        print('\n----- iter', it, ' -----')
        [z_probs, eps_probs, pol_probs, lam_probs] = mle_probs
#         print('z_probs', z_probs, 'eps_probs', eps_probs, 'pol_probs', pol_probs, 'lam_probs', lam_probs)
        
        ## E-step : update weights (Q's)
        p_x_given_z = np.zeros((len(track)-1, len(z_range), len(eps_range), len(pol_range), len(lam_range)))
        p_xz = np.zeros((len(track)-1, len(z_range), len(eps_range), len(pol_range), len(lam_range)))

        for i in range(len(track)-1):
#             print('\ntimestep', track[i][0])
            sd_next_true = track[i+1, 3:5]

            for z_idx, eps_idx, pol_idx, lam_idx in itertools.product(*idx_range):

                ## if do not change lane
                sd_next_pred_no_change, _, _ = hallucinate_a_step_no_change(track[i], z_range[z_idx])  #IDM
                likeli_pos_no_change = multi_normal([sd_next_pred_no_change[0]], eps_range[eps_idx]).pdf([sd_next_true[0]]) #observation model

                ## if change lane
                sd_next_pred_change, _, _ = hallucinate_a_step_change(track[i], lanes_rxy, z_range[z_idx])  #pd controller (lane change) 
                likeli_pos_change = multi_normal(sd_next_pred_change, [eps_range[eps_idx], 0.001]).pdf(sd_next_true) #observation model

                ## probability of changing lane & likelihood
                prob_lane, _, _ = get_lane_change_prob_EM(veh_id, track[i], z_range[z_idx], pol_range[pol_idx], lam_range[lam_idx], \
                                                          actual_tracks_dict_all_with_headway, mle_probs_final_all_veh, params_range)  #MOBIL
                print('time:', track[i][0], ', [z, eps, pol, lam]:', z_range[z_idx], eps_range[eps_idx], \
                      pol_range[pol_idx], lam_range[lam_idx], ' => prob_lane:', prob_lane)
                
                p_x_given_z[i, z_idx, eps_idx, pol_idx, lam_idx] = likeli_pos_no_change * (1-prob_lane) + \
                                                                   likeli_pos_change * prob_lane  #p(x|z)
                
                p_xz[i, z_idx, eps_idx, pol_idx, lam_idx] = p_x_given_z[i, z_idx, eps_idx, pol_idx, lam_idx] * \
                                                            z_probs[z_idx] * eps_probs[eps_idx] * \
                                                            pol_probs[pol_idx] * lam_probs[lam_idx]  #p(x,z)=p(x|z)p(z)
                
        Q = p_xz / np.sum(p_xz, axis=(1,2,3,4))[:,None,None,None,None] # normalize the weights (n, 6, 5, 4, 5)

        marginal_ll = np.sum(np.log(np.sum(p_xz, axis=(1,2,3,4))))
        print('marginal likelihood :', marginal_ll)
        
        
        ## M-step
        theta_init = np.concatenate([z_probs, eps_probs, pol_probs, lam_probs]).ravel()
        ticks = [len(z_range), len(z_range)+len(eps_range), len(z_range)+len(eps_range)+len(pol_range), len(theta_init)]  
        arguments = (p_x_given_z, Q, params_range, ticks)
        
        theta_opt, f_opt, d = l_bfgs_b(obj_function, x0=theta_init, args=arguments, approx_grad=True, \
                                       maxfun=5, maxiter=10)  # bounds=[(0,1)]*len(theta_init), \
        
#         cons = ({'type': 'eq', 'fun': lambda theta, ticks: np.round(theta[:ticks[0]].sum()**2, 6) - 1, 'args': (ticks,)},
#                 {'type': 'eq', 'fun': lambda theta, ticks: np.round(theta[ticks[0]:ticks[1]].sum()**2, 6) - 1, 'args': (ticks,)},
#                 {'type': 'eq', 'fun': lambda theta, ticks: np.round(theta[ticks[1]:ticks[2]].sum()**2, 6) - 1, 'args': (ticks,)},
#                 {'type': 'eq', 'fun': lambda theta, ticks: np.round(theta[ticks[2]:].sum()**2, 6) - 1, 'args': (ticks,)})
#         result = minimize(obj_function, x0=theta_init, args=arguments, method='L-BFGS-B', \
#                           constraints=cons) #, bounds=[(0,1)]*len(theta_init), options={'maxiter': 20, 'disp': True})
#         theta_opt, f_opt = result.x, result.fun
        
        if marginal_ll - prev_marginal_ll > 0:
            print('updated...')
            marginal_ll_final = marginal_ll
            marginal_ll_history.append(marginal_ll)
            
            mle_probs_unnormalized = [theta_opt[:ticks[0]], theta_opt[ticks[0]:ticks[1]], theta_opt[ticks[1]:ticks[2]], theta_opt[ticks[2]:]]
            print('mle_probs_unnormalized', mle_probs_unnormalized)
            
            z_probs = theta_opt[:ticks[0]] / theta_opt[:ticks[0]].sum()
            eps_probs = theta_opt[ticks[0]:ticks[1]] / theta_opt[ticks[0]:ticks[1]].sum()
            pol_probs = theta_opt[ticks[1]:ticks[2]] / theta_opt[ticks[1]:ticks[2]].sum()
            lam_probs = theta_opt[ticks[2]:] / theta_opt[ticks[2]:].sum()
            mle_probs = [z_probs, eps_probs, pol_probs, lam_probs]
            mle_probs_history.append(mle_probs)
            
#             mle_probs = [theta_opt[:ticks[0]], theta_opt[ticks[0]:ticks[1]], theta_opt[ticks[1]:ticks[2]], theta_opt[ticks[2]:]]
            print('mle_probs_normalized', mle_probs)
            
            if marginal_ll - prev_marginal_ll < tol:
                print('Break in %s iterations' % it)
                break 
            prev_marginal_ll = marginal_ll
    
    mle_probs_final = mle_probs
    converged_iter = it
    return mle_probs_final, mle_probs_history, marginal_ll_final, marginal_ll_history, converged_iter



def obj_function(theta, p_x_given_z, Q, params_range, ticks):
#     print('\nobj_function...')

    idx_range = [range(len(p_range)) for p_range in params_range]
    [z_range, eps_range, pol_range, lam_range] = params_range
    z_probs, eps_probs, pol_probs, lam_probs = theta[:ticks[0]], theta[ticks[0]:ticks[1]], theta[ticks[1]:ticks[2]], theta[ticks[2]:]

    p_xz = np.zeros((len(p_x_given_z), len(z_range), len(eps_range), len(pol_range), len(lam_range)))   #(n, 6, 5, 4, 5)
    for z_idx, eps_idx, pol_idx, lam_idx in itertools.product(*idx_range):
        p_xz[:, z_idx, eps_idx, pol_idx, lam_idx] = z_probs[z_idx] * eps_probs[eps_idx] * \
                                                    pol_probs[pol_idx] * lam_probs[lam_idx] #* \
#                                                     p_x_given_z[:, z_idx, eps_idx, pol_idx, lam_idx]

#         print('p_xz.sum', p_xz.sum(axis=(1,2,3,4)))
#     print('Q * np.log(p_xz)', (Q * np.log(p_xz)).shape)
    ELBO = np.sum(Q * np.log(p_xz))
#     print('ELBO', ELBO)
    return -ELBO
    
    
def run_EM_plot(actual_tracks_dict_all_with_headway, lanes_rxy, max_iter_per_veh, params_range, veh_id_minmax):
        
    mle_probs_final_all_veh, mle_probs_history_all_veh =  dict(), dict()
    marginal_ll_final_all_veh, marginal_ll_history_all_veh =  dict(), dict()
    converged_iter_all_veh = dict()
    
    for veh_id, track in actual_tracks_dict_all_with_headway.items():
        if veh_id >= veh_id_minmax[0] and veh_id <= veh_id_minmax[1]:  #veh_id>= 203 and veh_id < 210:
            print('\n####### veh_id', veh_id, '#######')
            mle_probs_final, mle_probs_history, marginal_ll_final, marginal_ll_history, converged_iter = \
                EM(veh_id, track, actual_tracks_dict_all_with_headway, lanes_rxy, params_range, mle_probs_final_all_veh, max_iter_per_veh)  
            
            mle_probs_final_all_veh[veh_id] = mle_probs_final
            mle_probs_history_all_veh[veh_id] = mle_probs_history
            marginal_ll_final_all_veh[veh_id] = marginal_ll_final
            marginal_ll_history_all_veh[veh_id] = marginal_ll_history
            converged_iter_all_veh[veh_id] = converged_iter

    return mle_probs_final_all_veh, mle_probs_history_all_veh, marginal_ll_final_all_veh, marginal_ll_history_all_veh, converged_iter_all_veh





