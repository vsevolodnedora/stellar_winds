#====================================================#
#
# This is the main file, containg my project
# reading, interpolating, analysing OPAL tables
# reading, plotting properites of the star from
# sm.data files of BEC output and more)
#
#====================================================#

#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------MAIN-LIBRARIES-----------------------------------------------------
import sys
import pylab as pl
from matplotlib import cm
import numpy as np
from ply.ctokens import t_COMMENT
from scipy import interpolate

from sklearn.linear_model import LinearRegression
# import scipy.ndimage
# from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
# from scipy.interpolate import griddata
import os
#-----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------CLASSES-----------------------------------------------------
from PhysMath import Errors
from PhysMath import Math
from PhysMath import Physics
from PhysMath import Constants

# from OPAL import Read_Table
# from OPAL import Row_Analyze
from OPAL import Table_Analyze
from OPAL import OPAL_Interpol
# from OPAL import New_Table

from FilesWork import Read_Observables
from FilesWork import Read_Plot_file
from FilesWork import Read_SM_data_file

from PhysPlots import PhysPlots
#-----------------------------------------------------------------------------------------------------------------------

# class Treat_Observables:
#
#     clumping_used = 4
#     cluming_required = 4
#
#     def __init__(self, obs_files):
#
#         self.files = obs_files
#         self.n_of_fls = len(obs_files)
#
#
#
#         self.obs = []
#         for i in range(len(obs_files)):
#             self.obs.append( Read_Observables(obs_files[i], self.clumping_used, self.cluming_required))
#
#         if (len(obs_files)) > 1 :
#             for i in range(1,len(self.files)):
#                 if not np.array_equal(self.obs[i-1].names, self.obs[i].names):
#                     print('\t__Error. Files with observations contain different *names* row')
#                     print('\t  {} has: {} \n\t  {} has: {} '
#                           .format(obs_files[i-1], self.obs[i-1].names, obs_files[i], self.obs[i].names))
#                     raise NameError('Files with observations contain different *names* row')
#
#     def check_if_var_name_in_list(self, var_name):
#         if var_name == 'lm' or var_name == 'ts' or var_name == 'rs': # special case for L/M and sonic temperature
#             pass
#         else:
#             for i in range(self.n_of_fls):
#                 if var_name not in self.obs[i].names:
#                     print('\n\t__Error. Variable:  {} is not in the list of names: \n\t  {} \n\t  in file: {}'
#                           .format(var_name, self.obs[i].names, self.files[i]))
#                     raise  NameError('Only lm, l, and rs varnames are available. {} is not listed.'.format(var_name))
#
#     def get_x_y_of_all_observables(self, x_name, y_name, var_for_label,
#                                    ts_arr = np.empty(1,), l_lm_arr= np.empty(1,), m_dot= np.empty(1,),
#                                    lim_t1_obs = None, lim_t2_obs = None):
#         '''
#         RETURN:  np.array( [plotted_stars, plotted_labels] )  [0][:,0] - nums of all plotted stars
#                                                               [0][:,1] - x - coord.
#                                                               [0][:,2] - y - coord
#                                                               [0][:,3] - ints from 0 to 9, uniqe for uniqe 'var_for_label'
#                                                               [1][:,0] - nums of selected stars for labels
#                                                               [1][:,1] - x - coord
#                                                               [1][:,2] - y - coord
#                                                               [1][:,3] - ints from 0 to 9
#         To get index in the [0] arr of the element in [1] Use: int( np.where( res[0][:, 0]==res[1][j, 0] )[0] )
#
#         Warning! If there are more unique str(var_for_label), PROGRAM BRAKES
#         :param x_name:
#         :param y_name:
#         :param var_for_label:
#         :param ts_arr:
#         :param l_lm_arr:
#         :param m_dot:
#         :param lim_t1_obs:
#         :param lim_t2_obs:
#         :return:
#         '''
#         self.check_if_var_name_in_list(x_name)
#         self.check_if_var_name_in_list(y_name)
#         self.check_if_var_name_in_list(var_for_label)
#
#         s = 0
#
#         leble = []
#         plotted_stars = np.array([0., 0., 0., 0.])
#         plotted_labels= np.array([0., 0., 0., 0. ])
#
#         # if self.obs != None:  # plot observed stars
#         ''' Read the observables file and get the necessary values'''
#         ts_ = []
#         y_coord_ = []
#
#         import re  # for searching the number in 'WN7-e' string, to plot them different colour
#         for i in range(self.obs[s].num_stars):
#             star_x_coord = []
#             star_y_coord = []
#
#             # ---------------------------------------Y-------------------------
#             if y_name == 'lm':
#                 star_y_coord = [ Physics.loglm(self.obs[s].obs_par('l', float)[i],
#                                              self.obs[s].obs_par('m', float)[i]) ]
#             else:
#                 star_y_coord = [ self.obs[s].obs_par(y_name, float)[i] ]
#
#
#             # ---------------------------------------X-------------------------
#             if x_name == 'ts' or x_name == 'rs':
#                 if not ts_arr.any() or not l_lm_arr.any() or not m_dot.any():
#                     print('\t__Error. For ts to be evaluated for a star : *ts_arr, l_lm_arr, m_dot* to be provided')
#                     raise ValueError
#
#                 x_y_coord = Physics.lm_mdot_obs_to_ts_lm(ts_arr, l_lm_arr, m_dot, star_y_coord[0],
#                                                          self.obs[s].obs_par('mdot', float)[i],
#                                                          i, lim_t1_obs, lim_t2_obs)
#                 if x_y_coord.any():
#                     ts_ = np.append(ts_, x_y_coord[1, :])  # FOR linear fit
#                     y_coord_ = np.append(y_coord_, x_y_coord[0, :])
#                     star_x_coord =  x_y_coord[1, :]
#                     star_y_coord =  x_y_coord[0, :]  # If the X coord is Ts the Y coord is overritten.
#
#             else:
#                 star_x_coord = [ self.obs[s].obs_par(x_name, float)[i] ]
#
#             if x_name == 'lm':
#                 star_x_coord = [ Physics.loglm(self.obs[s].obs_par('l', float)[i],
#                                              self.obs[s].obs_par('m', float)[i]) ]
#
#
#
#
#
#             star_x_coord = np.array(star_x_coord)
#             star_y_coord = np.array(star_y_coord)
#             if len(star_x_coord) == len(star_y_coord) and star_x_coord.any() :
#
#                 se = re.search(r"\d+(\.\d+)?", self.obs[s].obs_par('type', str)[i])  # this is searching for the niumber
#                 #             color = 'C' + str(int(s.group(0)))  # making a colour our of C1 - C9 range
#
#                 for j in range(len(star_x_coord)):  # plot every solution in the degenerate set of solutions
#
#                     row = self.obs[s].table[i]  # to get the 0th element, which is alwas the star index
#
#                     cur_type = int(se.group(0))
#                     if cur_type not in leble:  # plotting the label for unique class of stars
#                         leble.append( cur_type )
#
#                         plotted_labels = np.vstack((plotted_labels, np.array((int(row[0:3]),
#                                                                               star_x_coord[j],
#                                                                               star_y_coord[j],
#                                                                               cur_type ))))
#
#                     plotted_stars = np.vstack((plotted_stars, np.array((int(row[0:3]),
#                                                                         star_x_coord[j],
#                                                                         star_y_coord[j],
#                                                                         cur_type ))))  # for further printing
#             # print(self.obs[s].table)
#
#         # -----------------------------------------------LINEAR FIT TO THE DATA-------------------------------------
#         # ts_grid_y_grid = Math.line_fit(ts_, y_coord_)
#         # plt.plot(ts_grid_y_grid[0, :], ts_grid_y_grid[1, :], '-.', color='blue')
#         # np.delete(plotted_stars,1,0)
#         plotted_stars  = np.delete(plotted_stars, 0, 0) # removing [0,0,0,] row
#         plotted_labels = np.delete(plotted_labels, 0, 0)
#
#         if plotted_stars.any():
#             print('\n| Plotted Stras from Observ |')
#             print('|  i  | {} | {}  | col |'.format(x_name, y_name))
#             print('|-----|---------|----------|')
#             for i in range(len(plotted_stars[:, 0])):
#                 print('| {} |  {} \t| {} | {} |'.format("%3.f" % plotted_stars[i, 0], "%.2f" % plotted_stars[i, 1],
#                                                  "%.2f" % plotted_stars[i, 2], plotted_stars[i, 3]))
#
#         if plotted_labels.any():
#             print('\n| Plotted Labels from Observ |')
#             print('|  i  | {} | {}  | col |'.format(x_name, y_name))
#             print('|-----|-----------|---------|')
#             for i in range(len(plotted_labels[:, 0])):
#                 print('| {} |  {} \t| {} | {} |'.format("%3.f" % plotted_labels[i, 0], "%.2f" % plotted_labels[i, 1],
#                                                  "%.2f" % plotted_labels[i, 2], plotted_labels[i, 3]))
#
#         return( np.array( [plotted_stars, plotted_labels] ) )
#
# class Treat_Numercials:
#
#     def __init__(self, files):
#         self.files = files
#         self.n_of_files = len(files)
#
#         self.mdl = []
#         for i in range(self.n_of_files):
#             self.mdl.append(Read_SM_data_file.from_sm_data_file(self.files[i]))
#             # print(self.mdl[i].mdot_[-1])
#
#         # self.nmdls = len(self.mdl)
#         print('\t__Note: {} sm.data files has been uploaded.'.format(self.n_of_files))
#
#     def ind_from_condition(self, cur_model, condition):
#         '''
#
#         :param cur_model: index of a model out of list of class instances that is now in the MAIN LOOP
#         :param condition: 'sp' - for sonic point, 'last' for -1, or like 't=5.2' for point where temp = 5.2
#         :return: index of that point
#         '''
#         if condition == 'last' or condition == '' :
#             return -1
#
#         if condition == 'sp': # Returns the i of the velocity that is >= sonic one. (INTERPOLATION would be better)
#             return self.mdl[cur_model].sp_i()
#
#         var_name = condition.split('=')[ 0] # for condition like 't = 5.2' separates t as a var in sm.file and
#         var_value= condition.split('=')[-1]
#
#         if var_name not in self.mdl[cur_model].var_names:                   # Checking if var_name is in list of names for SM files
#             raise NameError('Var_name: {} is not in var_name list: \n\t {}'
#                             .format(var_name, self.mdl[cur_model].var_names))
#
#         arr = np.array( self.mdl[cur_model].get_col(var_name) ) # checking if var_value is in the column of var_name
#         print(var_value, arr.min(), arr.max())
#         if var_value < arr.min() or var_value > arr.max() :
#             raise ValueError('Given var_value={} is beyond {} range: ({}, {})'
#                              .format(var_value,var_name,arr.min(),arr.max()))
#
#         ind = -1
#         for i in range(len(arr)): # searching for the next element, >= var_value. [INTERPOLATION would be better]
#             if var_value >= arr[i]:
#                 ind = i
#                 break
#         if ind == -1:
#             raise ValueError('ind = -1 -> var_value is not found in the arr. | var_value={}, array range: ({}, {})'
#                              .format(var_value, var_name,arr.min(), arr.max()))
#
#         return ind
#
#
#
#     def get_x_y_of_all_numericals(self, condition, x_name, y_name, var_for_label1, var_for_label2,
#                                   ts_arr = np.empty(0,), l_lm_arr = np.empty(0,), mdot2d_arr = np.empty(0,),
#                                   lim_t1_obs = None, lim_t2_obs = None):
#         '''
#
#         :param condition: 'sp' - for sonic point, 'last' for -1, or like 't=5.2' for point where temp = 5.2
#         :param x_name: can be the sm.file car name (eg from the bec output list)
#         :param y_name: can be the sm.file car name (eg from the bec output list)
#         :param var_for_label1:
#                var_for_label2: Same, but can me 'color' - ro return unique value from 1 to 9
#         :return: np.array([ i , x , y , var_lbl1 , var_lbl2 ]) - set of coluns for each sm.file - one row
#         '''
#         model_stars1 = np.array([0., 0., 0., 0., 0.]) # for output i
#
#         for i in range(self.n_of_files):
#             x_coord = None
#             y_coord = None
#
#             i_req = self.ind_from_condition(i, condition)
#
#             '''---------------------MAIN CYCLE-------------------'''
#             #-------------------SETTING-COORDINATES------------------------
#             if x_name in  self.mdl[i].var_names:
#                 x_coord = self.mdl[i].get_col(x_name)[i_req]
#             if y_name in  self.mdl[i].var_names:
#                 y_coord = self.mdl[i].get_col(y_name)[i_req]
#
#             if y_name == 'lm':
#                 y_coord = self.mdl[i].get_spec_val('lm', i_req)
#
#             if var_for_label1 == 'Y_c':
#                 add_data1 = self.mdl[i].get_col('He4')[0]
#             else:
#                 add_data1 = self.mdl[i].get_col(var_for_label1)[i_req]
#
#             if var_for_label2 == 'color':
#                 add_data2 = int(Math.get_0_to_max([i],9)[i]) # from 1 to 9 for plotting C+[1-9]
#             else:
#                 add_data2 = self.mdl[i].get_col(var_for_label2)[i_req]
#
#             #-------------------------_CASE TO INTERPOLATE MDOT -> ts InterpolateioN----------------
#             if x_name == 'ts':
#                 if not ts_arr.any() or not l_lm_arr.any() or not mdot2d_arr.any():
#                     raise ValueError('x_coord {} requires ts, l_lm_arr and mdot2arr to be interpolated'.format(x_name))
#
#
#             if x_name == 'ts' and (y_name == 'l' or y_name == 'lm') and ts_arr.any() and l_lm_arr.any() and mdot2d_arr.any():
#                 p_mdot = self.mdl[i].mdot_[i_req]
#                 x_y_coord = Physics.lm_mdot_obs_to_ts_lm(ts_arr, l_lm_arr, mdot2d_arr, y_coord, p_mdot, i, lim_t1_obs, lim_t2_obs)
#
#                 if x_y_coord.any():
#                     for j in range(len(x_y_coord[0, :])):
#
#                             # plt.plot(x_y_coord[1, j], x_y_coord[0, j], marker='.', markersize=9, color=color)
#                             # ax.annotate('m' + str(i), xy=(x_y_coord[1, j], x_y_coord[0, j]), textcoords='data')
#
#                         model_stars1 = np.vstack(( model_stars1, np.array(( i, x_y_coord[1, j],
#                                                                                x_y_coord[0, j],
#                                                                                add_data1,
#                                                                                add_data2  ))))
#             else:
#                 if x_coord == None or y_coord == None:
#                     raise ValueError('x_coord={} or y_coord={} is not obtained.'.format(x_coord,y_coord))
#
#                 model_stars1 = np.vstack((model_stars1, np.array((i, x_coord,
#                                                                      y_coord,
#                                                                      add_data1,
#                                                                      add_data2  ))))
#
#
#             # color = 'C' + str(int(i * 10 / self.n_of_files))
#             # plt.plot(x_coord, y_coord, marker='.', markersize=9, color=color)
#             # # label='Model {}: T_s {} , L/M {} , Mdot {}'.format(i, "%.2f" % p_t, "%.2f" % p_lm, "%.2f" % p_mdot))
#             # ax.annotate(str(i), xy=(x_coord, y_coord), textcoords='data')
#
#             # --------------------------SAME BUT USING Mdot TO GET SONIC TEMPERATURE (X-Coordinate)------------------------
#             # p_mdot = self.mdl[i].mdot_[i_req]
#             # x_y_coord = Physics.lm_mdot_obs_to_ts_lm(t, y_coord, m_dot, y_coord, p_mdot, i, lim_t1_obs, lim_t2_obs)
#             # if x_y_coord.any():
#             #     for j in range(len(x_y_coord[0, :])):
#             #         plt.plot(x_y_coord[1, j], x_y_coord[0, j], marker='.', markersize=9, color=color)
#             #         ax.annotate('m' + str(i), xy=(x_y_coord[1, j], x_y_coord[0, j]), textcoords='data')
#             #         model_stars1 = np.vstack(
#             #             (model_stars1, np.array((i, x_y_coord[1, j], x_y_coord[0, j], p_mdot, self.mdl[i].He4_[0]))))
#             #
#             #     model_stars2 = np.vstack((model_stars2, np.array(
#             #         (i, x_coord, y_coord, p_mdot, self.mdl[i].He4_[0]))))  # for further printing
#
#         # -------------------------PLOT FIT FOR THE NUMERICAL MODELS AND TABLES WITH DATA --------------------------------
#
#         model_stars1  = np.delete(model_stars1, 0, 0) # removing [0,0,0,] row
#
#         if model_stars1.any():
#             print('\n| Models plotted by ts & lm |')
#             print('\t| Conditon: {} |'.format(condition))
#             print('|  i  | {} | {} | {} | {}  |'.format(x_name, y_name, var_for_label1, var_for_label2))
#             print('|-----|------|------|-------|------|')
#             # print(model_stars1.shape)
#             for i in range(1, len(model_stars1[:, 0])):
#                 print('| {} | {} | {} | {} | {} |'.format("%3.f" % model_stars1[i, 0], "%.2f" % model_stars1[i, 1],
#                                                           "%.2f" % model_stars1[i, 2], "%.2f" % model_stars1[i, 3],
#                                                           "%.2f" % model_stars1[i, 4]))
#         else:
#             print('\t__Warning: No stars to Print. Coordinates are not obtained')
#
#
#         self.table(y_name, -1)
#
#         return model_stars1
#
#             # fit = np.polyfit(model_stars1[:, 1], model_stars1[:, 2], 3)  # fit = set of coeddicients (highest first)
#             # f = np.poly1d(fit)
#             # fit_x_coord = np.mgrid[(model_stars1[1:, 1].min() - 0.02):(model_stars1[1:, 1].max() + 0.02):100j]
#             # plt.plot(fit_x_coord, f(fit_x_coord), '--', color='black', label='Model_fit')
#
#
#
#         # if model_stars2.any():
#         #     print('\n| Models plotted: lm & mdot |')
#         #     print('|  i  | in_t |  {}  | m_dot | Y_c  |'.format(y_mode))
#         #     print('|-----|------|------|-------|------|')
#         #     for i in range(1, len(model_stars2[:, 0])):
#         #         print('| {} | {} | {} | {} | {} |'.format("%3.f" % model_stars2[i, 0], "%.2f" % model_stars2[i, 1],
#         #                                                   "%.2f" % model_stars2[i, 2], "%.2f" % model_stars2[i, 3],
#         #                                                   "%.2f" % model_stars2[i, 4]))
#         #
#         #     fit = np.polyfit(model_stars2[:, 1], model_stars2[:, 2], 3)  # fit = set of coeddicients (highest first)
#         #     f = np.poly1d(fit)
#         #     fit_x_coord = np.mgrid[(model_stars2[1:, 1].min() - 0.02):(model_stars2[1:, 1].max() + 0.02):100j]
#         #     plt.plot(fit_x_coord, f(fit_x_coord), '--', color='black', label='Model_fit')
#
#     def get_x_y_z_arrays(self, n_of_model, x_name, y_name, z_name, var_for_label1, var_for_label2, var_for_label3):
#         '''
#         Returns                     0: [ var_for_label1  , var_for_label2 , var_for_label3 ]
#         :param x_name:              1: [ x_coord[:]      , y_coord[:]       z_coord[:]     ]
#         :param y_name:              2  [         :       ,         :                :      ]
#         :param var_for_label1:                      and off it goes
#         :param var_for_label2:
#         :return:
#         '''
#         x_coord = None
#         y_coord = None
#         z_coord = None
#
#         i_req = -1  # for the add_data, as a point
#         '''---------------------MAIN CYCLE-------------------'''
#         # -------------------SETTING-COORDINATES------------------------
#         if x_name in self.mdl [n_of_model].var_names:
#             x_coord = self.mdl[n_of_model].get_col(x_name)
#         if y_name in self.mdl [n_of_model].var_names:
#             y_coord = self.mdl[n_of_model].get_col(y_name)
#         if z_name in self.mdl [n_of_model].var_names:
#             z_coord = self.mdl[n_of_model].get_col(z_name)
#
#         add_data1 = self.mdl[n_of_model].get_spec_val(var_for_label1) # if v_n = Y_c or lm
#         add_data2 = self.mdl[n_of_model].get_spec_val(var_for_label2)
#         add_data3 = self.mdl[n_of_model].get_spec_val(var_for_label3)
#
#         if add_data1 == None:
#             add_data1 = self.mdl[n_of_model].get_col(var_for_label1)[i_req] # noraml bec variables
#         if add_data2 == None:
#             add_data2 = self.mdl[n_of_model].get_col(var_for_label2)[i_req]
#         if add_data3 == None:
#             add_data3 = self.mdl[n_of_model].get_col(var_for_label3)[i_req]
#
#         if len(x_coord) != len(y_coord) or len(x_coord) != len(z_coord):
#             raise ValueError('x_coord and y_coord: \n\t {} \t\n {} have different shape.'.format(x_coord, y_coord))
#
#         return np.vstack(( np.insert(x_coord, 0, add_data1), np.insert(y_coord, 0, add_data2),
#                            np.insert(z_coord, 0, add_data3)  )).T
#
#
#     def get_set_of_cols(self, v_n_arr, n_of_model):
#         '''
#         Returns v_n_arr * length of each column array, [:,0] - first var, and so on.
#         :param v_n_arr:
#         :param n_of_model:
#         :return:
#         '''
#         return self.mdl[n_of_model].get_set_of_cols(v_n_arr)
#
#
#     def get_sonic_vel_array(self, n_of_model):
#         mu = self.mdl[n_of_model].get_col('mu')
#         t  = self.mdl[n_of_model].get_col('t')
#         return Physics.sound_speed(t, mu, True)
#
#     def table(self, y_name = 'l', i_req = -1):
#
#         if y_name == 'l':
#
#             print(
#                 '\n'
#                 ' i'
#                 ' |  Mdot '
#                 '| Mass'
#                 '|  R/Rs  '
#                 '| L/Ls  '
#                 '| kappa  '
#                 '| l(Rho) '
#                 '| Temp  '
#                 '| mfp   '
#                 '| vel   '
#                 '| gamma  '
#                 '| tpar  '
#                 '|  HP   '
#                 '| log(C)  ')
#         if y_name == 'lm':
#             print(
#                 '\n'
#                 ' i'
#                 ' |  Mdot '
#                 '| Mass'
#                 '|  R/Rs  '
#                 '| L/M   '
#                 '| kappa  '
#                 '| l(Rho) '
#                 '| Temp  '
#                 '| mfp   '
#                 '| vel   '
#                 '| gamma  '
#                 '| tpar  '
#                 '|  HP   '
#                 '| log(C)  ')
#
#         print('---|-------|-----|--------|-------|--------|--------|-------|------'
#               '-|-------|--------|-------|-------|-------')
#
#         for i in range(self.n_of_files):
#             self.mdl[i].get_par_table(i, y_name, i_req)
#
# class Tables:
#     def __init__(self, opal_name, t1, t2, n_interp = 1000, load_lim_cases = False, output_dir = '../data/output/', plot_dir = '../data/plots/'):
#         self.op_name = opal_name
#         self.t1 = t1
#         self.t2 = t2
#         self.n_inter = n_interp
#
#         self.out_dir = output_dir
#         self.plot_dir = plot_dir
#
#         self.opal = OPAL_Interpol(opal_name, n_interp)
#         self.tbl_anl = Table_Analyze(opal_name, n_interp, load_lim_cases, output_dir, plot_dir)
#
#     def save_t_rho_k(self, rho1 = None, rho2=None):
#         op_cl = OPAL_Interpol(self.op_name, self.n_inter)
#         t1, t2, rho1, rho2 = op_cl.check_t_rho_limits(self.t1, self.t2, rho1, rho2)
#         op_table = op_cl.interp_opal_table(self.t1, self.t2, rho1, rho2)
#
#         Tables.save_table(op_table,self.op_name,'t_rho_k','t','rho','k',self.out_dir)
#
#         # header = np.zeros(len(op_table[0, :]))
#         # header[0] = self.t1
#         # header[1] = self.t2
#         # header[2] = rho1
#         # header[3] = rho2
#         #
#         # op_and_head = np.vstack((header, op_table))
#         #
#         # part = self.op_name.split('/')[-1]
#         # name = self.out_dir + 'interp_' + part
#         #
#         # np.savetxt(name, op_and_head, '%.4f', '  ', '\n',
#         #            'INTERPOLATED {} TABLE'.format(part), '', '# Table:{}, t1:{}, t2:{}, rho1:{}, rho2:{}, grid:{}'
#         #            .format(self.op_name, self.t1, self.t2, rho1, rho2, self.n_inter))
#         #
#         # # def read_interp_opal(table_names):
#
#     def save_t_kappa_rho(self, k1 = None, k2 = None, n_out = 1000):
#
#         cl = Table_Analyze(self.op_name, self.n_inter, False, self.out_dir, self.plot_dir)
#         cl.treat_tasks_tlim(n_out, self.t1, self.t2, k1, k2, False)
#
#     @staticmethod
#     def get_k1_k2_from_llm1_llm2(t1, t2, l1, l2):
#         lm1 = None
#         if l1 != None:
#             lm1 = Physics.l_to_lm(l1)
#         lm2 = None
#         if l2 != None:
#             lm2 = Physics.l_to_lm(l2)
#
#         if lm1 != None:
#             k2 = Physics.loglm_logk(lm1)
#         else:
#             k2 = None
#         if lm2 != None:
#             k1 = Physics.loglm_logk(lm2)
#         else:
#             k1 = None
#
#         print('\t__Provided LM limits ({}, {}), translated to L limits: ({}, {})'.format(lm1, lm2, l1, l2))
#         print('\t__Provided T limits ({},{}), and kappa limits ({}, {})'.format(t1, t2, k1, k2))
#         return [k1, k2]
#
#     @staticmethod
#     def save_table(d2arr, opal_used, name, x_name, y_name, z_name, output_dir = '../data/output/'):
#
#         header = np.zeros(len(d2arr)) # will be first row with limtis and
#         # header[0] = x1
#         # header[1] = x2
#         # header[2] = y1
#         # header[3] = y2
#         # tbl_name = 't_k_rho'
#         # op_and_head = np.vstack((header, d2arr))  # arraching the line with limits to the array
#
#         part = opal_used.split('/')[-1]
#         full_name = output_dir + name + '_' + part  # dir/t_k_rho_table8.data
#
#         np.savetxt(full_name, d2arr, '%.4f', '  ', '\n',
#                    '\nINTERPOLATED OPAL {} TABLE for {} relation'.format(part, name), '',
#                    '# {} | {} | {} | {} |'
#                    .format(opal_used, x_name, y_name, z_name))
#
#         # np.savetxt(full_name, d2arr, '%.4f', '  ', '\n',
#         #            '\nINTERPOLATED OPAL {} TABLE for {} relation'.format(part, name), '',
#         #            '# {} | {} {} {} | {} {} {} | {} | {} | {}'
#         #            .format(opal_used, x_name, x1, x2, y_name, y1, y2, z_name, n_int, n_out))
#
#     @staticmethod
#     def read_table(name, x_name, y_name, z_name, opal_used, dir = '../data/output/'):
#         part = opal_used.split('/')[-1]
#         full_name = dir + name + '_' + part
#
#         f = open(full_name, 'r').readlines()
#
#         boxes = f[0].split('|')
#         f.clear()
#         # print(boxes)
#         # r_table = boxes[0].split()[-1]
#         # r_x_name = boxes[1].split()[0]
#         # x1 = boxes[1].split()[1]
#         # x2 = boxes[1].split()[2]
#         # r_y_name = boxes[2].split()[0]
#         # y1 = boxes[2].split()[1]
#         # y2 = boxes[2].split()[2]
#         # r_z_name = boxes[3].split()[0]
#         # n1 = boxes[4].split()[-1]
#         # n2 = boxes[5].split()[-1]
#
#         r_table  = boxes[0].split()[-1]
#         r_x_name = boxes[1].split()[-1]
#         r_y_name = boxes[2].split()[-1]
#         r_z_name = boxes[3].split()[-1]
#
#         if r_table != opal_used:
#             raise NameError('Read OPAL | {} | not the same is opal_used | {} |'.format(r_table, opal_used))
#
#         if x_name != r_x_name:
#             raise NameError('Provided x_name: {} not equal to table x_name: {}'.format(x_name, r_x_name))
#
#         if y_name != r_y_name:
#             raise NameError('Provided x_name: {} not equal to table x_name: {}'.format(y_name, r_y_name))
#
#         if z_name != r_z_name:
#             raise NameError('Provided x_name: {} not equal to table x_name: {}'.format(z_name, r_z_name))
#
#         # if x1 == 'None':
#         #     x1 = None
#         # else:
#         #     x1 = float(x1)
#         #
#         # if x2 == 'None:':
#         #     x2 = None
#         # else:
#         #     x2 = float(x2)
#         #
#         # if y1 == 'None':
#         #     y1 = None
#         # else:
#         #     y1 = float(y1)
#         #
#         # if y2 == 'None':
#         #     y2 = None
#         # else:
#         #     y2 = float(y2)
#         #
#         # n1 = int(n1)
#         # n2 = int(n2)
#
#         print('\t__OPAL_USED: {}, X is {} | Y is {} | Z is {} '
#               .format(r_table, r_x_name, r_y_name, r_z_name))
#
#         print('\t__File | {} | is loaded successfully'.format(full_name))
#
#         # global file_table
#         file_table = np.loadtxt(full_name, dtype=float)
#
#         return file_table #[x1, x2, y1, y2, n1, n2]
#
#     def save_t_k_rho(self, llm1=None, llm2=None, n_out = 1000):
#         k1, k2 = Tables.get_k1_k2_from_llm1_llm2(self.t1, self.t2, llm1, llm2) # assuming k = 4 pi c G (L/M)
#
#         global t_k_rho
#         k_t_rho = self.tbl_anl.treat_tasks_interp_for_t(self.t1, self.t2, n_out, self.n_inter, k1, k2)
#         t_k_rho = k_t_rho.T # as from table_analize you get t as an Y axis as in OPAL table
#
#
#         self.save_table(t_k_rho, self.op_name, 't_k_rho', 't', 'k', 'rho', self.out_dir)
#         print('\t__Note. Table | t_k_rho | has been saved in {}'.format(self.out_dir))
#         # self.read_table('t_k_rho', 't', 'k', 'rho', self.op_name)
#         # def save_t_llm_vrho(self, llm1=None, llm2=None, n_out = 1000):
#
#     def save_t_llm_vrho(self, l_or_lm_name):
#
#         # 1 load the t_k_rho
#         t_k_rho = self.read_table('t_k_rho', 't', 'k', 'rho', self.op_name)
#
#         k = t_k_rho[0, 1:]
#         t = t_k_rho[1:, 0]
#         rho2d = t_k_rho[1:, 1:]
#
#         vrho = Physics.get_vrho(t, rho2d.T, 2) # mu = 1.34 by default | rho2d.T is because in OPAL t is Y axis, not X.
#
#         # ----------------------------- SELECT THE Y AXIS -----------------
#         if l_or_lm_name == 'l':
#             l_lm_arr = Physics.lm_to_l(Physics.logk_loglm(k, True))  # Kappa -> L/M -> L
#         else:
#             l_lm_arr = Physics.logk_loglm(k, 1)
#
#
#         l_lm_arr = np.flip(l_lm_arr, 0)  # accounting for if k1 > k2 the l1 < l2 or lm1 < lm2
#         vrho     = np.flip(vrho, 0)
#
#         global t_llm_vrho
#         t_llm_vrho = Math.combine(t, l_lm_arr, vrho)
#         name = 't_'+ l_or_lm_name + '_vrho'
#
#         self.save_table(t_llm_vrho, self.op_name, name, 't', l_or_lm_name, '_vrho', self.out_dir)
#
#         return t_llm_vrho
#
#         # print(t_llm_vrho.shape)
#
#     def save_t_llm_mdot(self, r_s, l_or_lm, r_s_for_t_l_vrho, mu = 1.34): # mu assumed constant
#
#         # r_s_for_t_l_vrho = '', 't', 'l', 'lm', 'vrho'
#
#
#         fname = 't_' + l_or_lm + '_vrho'
#         t_llm_vrho = self.read_table(fname, 't', l_or_lm, '_vrho', self.op_name)
#         vrho = t_llm_vrho[1:,1:]
#
#         # -------------------- --------------------- ----------------------------
#         c = np.log10(4 * 3.14 * ((Constants.solar_r) ** 2) / Constants.smperyear)
#         mdot = np.zeros((vrho.shape))
#
#         if r_s_for_t_l_vrho == '': #------------------------REQUIRED r_s = float
#             c2 = c + np.log10(r_s ** 2)
#             mdot = vrho + c2
#
#         if r_s_for_t_l_vrho == 't' or r_s_for_t_l_vrho == 'ts': # ---r_s = 1darray
#             for i in range(vrho[:,0]):
#                 mdot[i,:] = vrho[i,:] + c + np.log10(r_s[i] ** 2)
#
#         if r_s_for_t_l_vrho == 'l' or r_s_for_t_l_vrho == 'lm': # ---r_s = 1darray
#             for i in range(vrho[0,:]):
#                 mdot[:,i] = vrho[:,i] + c + np.log10(r_s[i] ** 2)
#
#         if r_s_for_t_l_vrho == 'vrho': #---------------------REQUIRED r_s = 2darray
#             cols = len(vrho[0, :])
#             rows = len(vrho[:, 0])
#             m_dot = np.zeros((rows, cols))
#
#             for i in range(rows):
#                 for j in range(cols):
#                     m_dot[i, j] = vrho[i, j] + c + np.log10(r_s[i, j] ** 2)
#
#         global t_llm_mdot
#         t_llm_mdot = Math.combine(t_llm_vrho[0,1:], t_llm_vrho[1:,0], mdot)
#         self.save_table(t_llm_mdot, self.op_name, 't_'+l_or_lm+'_mdot','t', l_or_lm, 'mdot', self.out_dir)
#
# class ClassPlots:
#
#     def __init__(self, opal_file, smfls = list(), obs_fils = list(), plot_files = list(), n_anal = 1000, load_lim_cases = False,
#                  output_dir = '../data/output/', plot_dir = '../data/plots/'):
#         '''
#         :param path: './smdata/' filder with sm.datas
#         :param smfls: '5d-6' name, extension sm.data added automatically
#         :param opal_file: 'table1' in folder './opal' and extension '.data'
#         :param n_anal: interpolation depth
#         '''
#         self.opalfl = opal_file
#         self.output_dir = output_dir
#         self.plot_dir = plot_dir
#
#         self.obs_files = obs_fils
#         self.num_files = smfls
#
#         self.obs = Treat_Observables(self.obs_files)
#         self.nums= Treat_Numercials (self.num_files)
#
#         # --- INSTANCES
#         self.opal    = OPAL_Interpol(self.opalfl, n_anal)
#         self.tbl_anl = Table_Analyze(self.opalfl, n_anal, load_lim_cases, output_dir, plot_dir)
#
#         self.plotcl = []
#         self.plot_files = plot_files
#         for i in range(len(plot_files)):
#             self.plotcl.append(Read_Plot_file.from_file(plot_files[i]))
#
#         # self.obs = None
#         # if obs_file != None:
#         #     self.obs = Read_Observables(obs_file[0])
#         #
#         # self.mdl = []
#         # for i in range(len(smfls)):
#         #     self.mdl.append(Read_SM_data_File.from_sm_data_file(smfls[i]))
#         #     # print(self.mdl[i].mdot_[-1])
#         #
#         # self.nmdls = len(self.mdl)
#         # print('\t__Note: {} sm.data files has been uploaded.'.format(self.nmdls))
#         #
#         # self.plfl = []
#
#
#         self.mins = []
#
#         # self.tbl_anl.delete()
#
#     def xy_profile(self, v_n1, v_n2, var_for_label1, var_for_label2, sonic = -1):
#
#         fig = plt.figure()
#         ax1 = fig.add_subplot(111)
#         ax2 = ax1.twiny()
#
#         tlt = v_n2 + '(' + v_n1 + ') profile'
#         plt.title(tlt, loc='left')
#
#         for i in range(len(self.num_files)):
#
#             res = self.nums.get_x_y_z_arrays( i, v_n1, v_n2, '-', var_for_label1, var_for_label2, '-')
#             lbl = '{}:{} , {}:{}'.format(var_for_label1,'%.2f' % res[0,0],var_for_label2,'%.2f' % res[0,1])
#             ax1.plot(res[1:, 0], res[1:, 1], '-', color='C' + str(Math.get_0_to_max([i], 9)[i]), label=lbl)
#             ax1.plot(res[-1, 0], res[-1, 1], 'x', color='C' + str(Math.get_0_to_max([i], 9)[i]))
#             ax1.annotate(str('%.2f' % res[0,0]), xy=(res[-1, 0], res[-1, 1]), textcoords='data')
#             if sonic != -1 and sonic < len(self.num_files) and v_n2 == 'u':
#                 u_s = self.nums.get_sonic_vel_array(i)
#                 ax1.plot(res[1:, 0], u_s, '-', color='black')
#
#         last_model_t = self.nums.get_set_of_cols(['t'], sonic )[:,0]
#         n_tic_loc = []
#         n_tic_lbl = []
#         temps = [last_model_t[-1], 4.2, 4.6, 5.2, 6.2, last_model_t[0]]
#         for t in temps:
#             if t <= last_model_t[0] and t >= last_model_t[-1]:
#                 i = Math.find_nearest_index(last_model_t, t)
#                 n_tic_loc  = np.append(n_tic_loc, self.nums.get_set_of_cols([v_n1], sonic )[i,0])
#                 n_tic_lbl = np.append(n_tic_lbl, "%.1f" % t)
#
#         ax2.set_xlim(ax1.get_xlim())
#         ax2.set_xticks(n_tic_loc)
#         ax2.set_xticklabels(n_tic_lbl)
#         ax2.set_xlabel('log(T)')
#
#         ax1.set_xlabel(v_n1)
#         ax1.set_ylabel(v_n2)
#
#         ax1.grid(which='both')
#         ax1.grid(which='minor', alpha=0.2)
#         ax1.grid(which='major', alpha=0.2)
#
#         ax1.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
#         plot_name = self.plot_dir + v_n1 + '_vs_' + v_n2 + '_profile.pdf'
#         plt.savefig(plot_name)
#         plt.show()
#
#         # col = []
#         # for i in range(len(self.num_files)):
#         #     col.append(i)
#         #     res = self.nums.get_set_of_cols([v_n1,v_n2, 'xm','l', 'mdot'],i)
#         #     lbl = 'M:' + str('%.2f' % res[-1,2]) + ' L:' + \
#         #           str('%.2f' % res[-1,3]) + ' Mdot:' + \
#         #           str('%.2f' % res[-1,4])
#         #     ax1.plot(res[:,0],res[:,1], '-', color='C'+str(Math.get_0_to_max(col,9)[i]), label=lbl)
#         #     ax1.plot(res[-1,0], res[-1,1], 'x', color='C'+str(Math.get_0_to_max(col,9)[i]))
#         #     if sonic and v_n2 == 'u':
#         #         u_s = self.nums.get_sonic_vel_array(i)
#         #         ax1.plot(res[:,0], u_s, '-', color='black')
#
#
#         # res = self.nums.get_set_of_cols([v_n1,v_n2, 'xm','l', 'mdot'], -1 )
#         # n_tic_loc = []
#         # n_tic_lbl = []
#         # temps = [self.mdl[-1].t_[-1], 4.2, 4.6, 5.2, 6.2, self.mdl[-1].t_[0]]
#         # for t in temps:
#         #     if t <= self.mdl[-1].t_[0] and t >= self.mdl[-1].t_[-1]:
#         #         i = Math.find_nearest_index(self.mdl[-1].t_, t)
#         #         n_tic_loc  = np.append(n_tic_loc, self.mdl[-1].get_col(v_n1)[i])
#         #         n_tic_lbl = np.append(n_tic_lbl, "%.1f" % t)
#         #         # plt.axvline(x=self.mdl[-1].get_col(v_n1)[i], linestyle='dashed', color='black', label = 'log(T):{}'.format(t))
#         #
#         # ax2.set_xlim(ax1.get_xlim())
#         # # ax2.set_xticks(n_tic_loc)
#         # # ax2.set_xticklabels(n_tic_lbl)
#         # ax2.set_xlabel('log(T)')
#
#
#
#
#
#         # for i in range(len(self.num_files)):
#         #     x = self.nums[i].get_col(v_n1)
#         #     y = self.mdl[i].get_col(v_n2)
#         #     color = 'C' + str(i)
#         #
#         #     lbl = 'M:' + str('%.2f' % self.mdl[i].get_col('xm')[-1]) + ' L:' + \
#         #            str('%.2f' % self.mdl[i].get_col('l')[-1]) + ' Mdot:' + \
#         #            str('%.2f' % self.mdl[i].get_col('mdot')[-1])
#         #     ax1.plot(x, y, '-', color=color, label=lbl)
#         #     ax1.plot(x[-1], y[-1], 'x', color=color)
#         #
#         # r_arr = []
#         # for i in range(self.nmdls):
#         #     r_arr = np.append(r_arr, self.mdl[i].get_col(v_n1)[-1] )
#         # ind = np.where(r_arr == r_arr.max())
#         # # print(int(ind[-1]))
#
#         # if sonic:
#         #     for i in range(self.nmdls):
#         #         x = self.mdl[i].get_col(v_n1)
#         #         t = self.mdl[i].get_col('t')
#         #         mu = self.mdl[i].get_col('mu')
#         #         ax1.plot(x, Physics.sound_speed(t, mu, True), '-', color='black')
#
#         # ax1.set_xlabel(v_n1)
#         # ax1.set_ylabel(v_n2)
#
#         #---------------------------------------MINOR-TICKS-------------------------------
#         # if lx1 != None and lx2 != None:
#         #     plt.xlim(lx1, lx2)
#         #
#         # if ly1 != None and ly2 != None:
#         #     plt.ylim(ly1, ly2)
#
#         # ax1.grid(which='both')
#         # ax1.grid(which='minor', alpha=0.2)
#         # ax1.grid(which='major', alpha=0.2)
#
#         # n_tic_loc = []
#         # n_tic_lbl = []
#         # temps = [self.mdl[-1].t_[-1], 4.2, 4.6, 5.2, 6.2, self.mdl[-1].t_[0]]
#         # for t in temps:
#         #     if t <= self.mdl[-1].t_[0] and t >= self.mdl[-1].t_[-1]:
#         #         i = Math.find_nearest_index(self.mdl[-1].t_, t)
#         #         n_tic_loc  = np.append(n_tic_loc, self.mdl[-1].get_col(v_n1)[i])
#         #         n_tic_lbl = np.append(n_tic_lbl, "%.1f" % t)
#         #         # plt.axvline(x=self.mdl[-1].get_col(v_n1)[i], linestyle='dashed', color='black', label = 'log(T):{}'.format(t))
#
#         # ax2.set_xlim(ax1.get_xlim())
#         # # ax2.set_xticks(n_tic_loc)
#         # # ax2.set_xticklabels(n_tic_lbl)
#         # ax2.set_xlabel('log(T)')
#
#     def xyy_profile(self, v_n1, v_n2, v_n3, var_for_label1, var_for_label2, var_for_label3, edd_kappa = True, mdl_for_t_axis = 0):
#
#         # for i in range(self.nmdls):
#         #     x = self.mdl[i].get_col(v_n1)
#         #     y = self.mdl[i].get_col(v_n2)
#         #     color = 'C' + str(i)
#         #
#         #     lbl = 'M:' + str('%.2f' % self.mdl[i].get_col('xm')[-1]) + ' L:' + \
#         #            str('%.2f' % self.mdl[i].get_col('l')[-1]) + ' Mdot:' + \
#         #            str('%.2f' % self.mdl[i].get_col('mdot')[-1])
#         #     ax1.plot(x, y, '-', color=color, label=lbl)
#         #     ax1.plot(x[-1], y[-1], 'x', color=color)
#
#         fig, ax1 = plt.subplots()
#         tlt = v_n2 + ' , '+ v_n3 + ' = f(' + v_n1 + ') profile'
#         plt.title(tlt)
#
#         ax1.set_xlabel(v_n1)
#         # Make the y-axis label, ticks and tick labels match the line color.
#         ax1.set_ylabel(v_n2, color='b')
#         ax1.tick_params('y', colors='b')
#         ax1.grid()
#         ax2 = ax1.twinx()
#
#         for i in range(len(self.num_files)):
#             res = self.nums.get_x_y_z_arrays( i, v_n1, v_n2, v_n3, var_for_label1, var_for_label2, var_for_label3)
#
#             color = 'C' + str(Math.get_0_to_max([i], 9)[i])
#             lbl = '{}:{} , {}:{} , {}:{}'.format(var_for_label1,'%.2f' % res[0,0], var_for_label2,'%.2f' % res[0,1], var_for_label3,'%.2f' % res[0,2])
#             ax1.plot(res[1:, 0], res[1:, 1], '-', color = color, label=lbl)
#             ax1.plot(res[-1, 0], res[-1, 1], 'x', color = color )
#             ax1.annotate(str('%.2f' % res[0,0]), xy=(res[-1, 0], res[-1, 1]), textcoords='data')
#
#             if edd_kappa and v_n3 == 'kappa':
#                 k_edd = Physics.edd_opacity(self.nums.get_set_of_cols(['xm'], i)[-1], self.nums.get_set_of_cols(['l'], i)[-1])
#                 ax2.plot(ax1.get_xlim(), [k_edd, k_edd], color=color, label = 'Model: {}, k_edd: {}'.format(i, k_edd))
#
#             ax2.plot(res[1:, 0], res[1:, 2], '--', color = color)
#             ax2.plot(res[-1, 0], res[-1, 2], 'o',  color = color)
#
#         ax3 = ax2.twiny() # for temp
#         last_model_t = self.nums.get_set_of_cols(['t'], mdl_for_t_axis)[:, 0]
#         n_tic_loc = []
#         n_tic_lbl = []
#         temps = [last_model_t[-1], 4.2, 4.6, 5.2, 6.2, last_model_t[0]]
#         for t in temps:
#             if t <= last_model_t[0] and t >= last_model_t[-1]:
#                 i = Math.find_nearest_index(last_model_t, t)
#                 n_tic_loc  = np.append(n_tic_loc, self.nums.get_set_of_cols([v_n1], mdl_for_t_axis)[i, 0])
#                 n_tic_lbl = np.append(n_tic_lbl, "%.1f" % t)
#
#         ax3.set_xlim(ax1.get_xlim())
#         ax3.set_xticks(n_tic_loc)
#         ax3.set_xticklabels(n_tic_lbl)
#         ax3.set_xlabel('log(T)')
#
#         ax2.set_ylabel(v_n3, color='r')
#         ax2.tick_params('y', colors='r')
#
#         plt.title(tlt, loc='left')
#         fig.tight_layout()
#         ax1.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
#         plot_name = self.plot_dir + v_n1 + '_' + v_n2 + '_' + v_n3 + '_profile.pdf'
#         plt.savefig(plot_name)
#         plt.show()
#
#
#         # k_edd = Physics.edd_opacity(self.mdl[-1].xm_[-1], self.mdl[-1].l_[-1])
#         # ax2.plot(ax1.get_xlim(), [k_edd,k_edd], c='black')
#         # ----------------------------EDDINGTON OPACITY------------------------------------
#         # ax2.plot(np.mgrid[x1.min():x1.max():100j], np.mgrid[edd_k:edd_k:100j], c='black')
#
#         # for i in range(self.nmdls):
#         #     x = self.mdl[i].get_col(v_n1)
#         #     y = self.mdl[i].get_col(v_n3)
#         #     color = 'C' + str(i)
#         #
#         #     lbl = 'M:' + str('%.2f' % self.mdl[i].get_col('xm')[-1]) + ' L:' + \
#         #            str('%.2f' % self.mdl[i].get_col('l')[-1]) + ' Mdot:' + \
#         #            str('%.2f' % self.mdl[i].get_col('mdot')[-1])
#         #     ax2.plot(x, y, '--', color=color, label=lbl)
#         #     ax2.plot(x[-1], y[-1], 'o', color=color)
#
#
#         # ax2.set_ylabel(v_n3, color='r')
#         # ax2.tick_params('y', colors='r')
#
#         # plt.axvline(x=4.6, color='black', linestyle='solid', label='T = 4.6, He Op Bump')
#         # plt.axvline(x=5.2, color='black', linestyle='solid', label='T = 5.2, Fe Op Bump')
#         # plt.axvline(x=6.2, color='black', linestyle='solid', label='T = 6.2, Deep Fe Op Bump')
#
#         # ax3 = ax2.twiny()
#         # n_tic_loc = []
#         # n_tic_lbl = []
#         # temps = [self.mdl[-1].t_[-1], 4.2, 4.6, 5.2, 6.2, self.mdl[-1].t_[0]]
#         # for t in temps:
#         #     if t <= self.mdl[-1].t_[0] and t >= self.mdl[-1].t_[-1]:
#         #         i = Math.find_nearest_index(self.mdl[-1].t_, t)
#         #         n_tic_loc = np.append(n_tic_loc, self.mdl[-1].get_col(v_n1)[i])
#         #         n_tic_lbl = np.append(n_tic_lbl, "%.1f" % t)
#         #         # plt.axvline(x=self.mdl[-1].get_col(v_n1)[i], linestyle='dashed', color='black', label = 'log(T):{}'.format(t))
#
#         # ax3.set_xlim(ax1.get_xlim())
#         # ax3.set_xticks(n_tic_loc)
#         # ax3.set_xticklabels(n_tic_lbl)
#         # ax3.set_xlabel('log(T)')
#         #
#         # plt.title(tlt, loc='left')
#         # # plt.ylim(-8.5, -4)
#         # fig.tight_layout()
#         # plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
#         # plot_name = self.plot_dir + v_n1 + '_' + v_n2 + '_' + v_n3 + '_profile.pdf'
#         # plt.savefig(plot_name)
#         # plt.show()
#
#     def plot_t_rho_kappa(self, t1, t2, rho1 = None, rho2 = None, n_int = 1000, plot_edd = True):
#         # self.int_edd = self.tbl_anlom_OPAL_table(self.op_name, 1, n_int, load_lim_cases)
#
#
#         t_k_rho = self.opal.interp_opal_table(t1, t2, rho1, rho2)
#
#         t = t_k_rho[0, 1:]  # x
#         rho = t_k_rho[1:, 0]  # y
#         kappa = t_k_rho[1:, 1:]  # z
#
#         plt.figure()
#         levels = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
#         pl.xlim(t.min(), t.max())
#         pl.ylim(rho.min(), rho.max())
#         contour_filled = plt.contourf(t, rho, 10 ** (kappa), levels, cmap=plt.get_cmap('RdYlBu_r'))
#         plt.colorbar(contour_filled)
#         contour = plt.contour(t, rho, 10 ** (kappa), levels, colors='k')
#         plt.clabel(contour, colors='k', fmt='%2.1f', fontsize=12)
#         plt.title('OPACITY PLOT')
#         plt.xlabel('Log(T)')
#         plt.ylabel('log(rho)')
#
#         #------------------------EDDINGTON-----------------------------------
#         Table_Analyze.plot_k_vs_t = False # there is no need to plot just one kappa in the range of availability
#
#         if plot_edd: #n_model_for_edd_k.any():
#             for i in range(len(self.num_files)):  # self.nmdls
#                 res = self.nums.get_set_of_cols(['xm', 'l', 'xm'], i)
#                 k_edd = Physics.edd_opacity(res[-1, 0], res[-1, 1])
#                 # print(k_edd)
#
#                 n_model_for_edd_k = self.tbl_anl.interp_for_single_k(t1, t2, n_int, k_edd)
#                 x = n_model_for_edd_k[0, :]
#                 y = n_model_for_edd_k[1, :]
#                 color = 'black'
#                 lbl = 'Model:{}, k_edd:{}'.format(i,'%.2f' % 10**k_edd)
#                 plt.plot(x, y, '-.', color=color, label=lbl)
#                 plt.plot(x[-1], y[-1], 'x', color=color)
#
#         Table_Analyze.plot_k_vs_t = True
#         #----------------------DENSITY----------------------------------------
#
#         for i in range(len(self.num_files)):
#             res = self.nums.get_set_of_cols(['t', 'rho', 'He4', 'mdot'], i)
#             # res = self.nums.get_x_y_z_arrays( i, 't', 'rho', '', 'l', 'mdot', '-')
#             print(res.shape)
#             # lbl = 'Model:{} , Yc:{} , mdot:{}'.format(i, 't','%.2f' % res[0,0], 'mdot','%.2f' % res[0,1])
#             lbl = 'Model:{} , Yc:{} , mdot:{}'.format(i, '%.2f' % res[0,2], '%.2f' % res[0,3])
#             plt.plot(res[:, 0], res[:, 1], '-', color='C' + str(Math.get_0_to_max([i], 9)[i]), label=lbl)
#             plt.plot(res[-1, 0], res[-1, 1], 'x', color='C' + str(Math.get_0_to_max([i], 9)[i]))
#             plt.annotate(str('%.2f' % res[0,2]), xy=(res[-1, 0], res[-1, 1]), textcoords='data')
#
#
#
#         # for i in range(self.nmdls):
#         #
#         #     x = self.mdl[i].t_
#         #     y = self.mdl[i].rho_
#         #     color = color = 'C' + str(i)
#         #
#         #     lbl = 'M:' + str('%.2f' % self.mdl[i].get_col('xm')[-1]) + ' L:' + \
#         #           str('%.2f' % self.mdl[i].get_col('l')[-1]) + ' Mdot:' + \
#         #           str('%.2f' % self.mdl[i].get_col('mdot')[-1])
#         #     plt.plot(x, y, '-', color=color, label=lbl)
#         #     plt.plot(x[-1], y[-1], 'x', color=color)
#
#         plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
#         name = self.plot_dir + 't_rho_kappa.pdf'
#         plt.savefig(name)
#
#         plt.show()
#
#     def plot_t_mdot_lm(self, t1, t2, var_num='Y_c', mdot1 = None, mdot2 = None, r_s = 1.):
#         '''
#
#         :param t1:
#         :param t2:
#         :param mdot1:
#         :param mdot2:
#         :param r_s:
#         :return:
#         '''
#         # mdot1 = -6  to = -4
#
#         if mdot1 == None:
#             rho1 = None
#         else:
#             rho1 = Physics.mdot_rho(t1, mdot1, 0, r_s)
#
#         if mdot2 == None:
#             rho2 = None
#         else:
#             rho2 = Physics.mdot_rho(t2, mdot2, 0, r_s)
#
#         # -------------------------------
#
#         t_k_rho = self.opal.interp_opal_table(t1, t2, rho1, rho2)
#
#         t  = t_k_rho[0, 1:]  # x
#         rho= t_k_rho[1:, 0]  # y
#         k  = t_k_rho[1:, 1:]  # z
#
#         mdot = Physics.rho_mdot(t, rho, 1, r_s)
#         lm = Physics.logk_loglm(k, 2)
#
#         #-----------------------------------
#
#         fig = plt.figure()
#         ax = fig.add_subplot(1, 1, 1)
#         pl.xlim(t.min(), t.max())
#         pl.ylim(mdot.min(), mdot.max())
#         levels = [3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 5.0, 5.2]
#         contour_filled = plt.contourf(t, mdot, lm, levels, cmap=plt.get_cmap('RdYlBu_r'))
#         plt.colorbar(contour_filled)
#         contour = plt.contour(t, mdot, lm, levels, colors='k')
#         plt.clabel(contour, colors='k', fmt='%2.1f', fontsize=12)
#         plt.title('L/M PLOT')
#         plt.xlabel('Log(t_s)')
#         plt.ylabel('log(M_dot)')
#
#         nums = Treat_Numercials(self.num_files) # Surface Temp as a x coordinate
#         res = nums.get_x_y_of_all_numericals('sp', 't', 'mdot', var_num, 'color')
#         for i in range(len(res[:,0])):
#             lbl = 'model:{}'.format(i)
#             plt.plot(res[i, 1], res[i, 2], marker='x', color='C' + str(Math.get_0_to_max([i],9)[i]), ls='', label=lbl)  # plot color dots)))
#             ax.annotate(str("%.2f" % res[i, 3]), xy=(res[i, 1], res[i, 2]), textcoords='data')
#
#         plt.legend(bbox_to_anchor=(1, 0), loc='lower right', ncol=1)
#         name = self.plot_dir + 't_mdot_lm_plot.pdf'
#         plt.savefig(name)
#         plt.show()
#
#
#         # # nums = Treat_Numercials(self.num_files)   # Sonic Temp (interpol from Mdot, assuming the r_s) is x_coord
#         # res = nums.get_x_y_of_all_numericals('sp', 'ts', 'lm', 'Y_c', 'color' ,t, l_lm_arr,m_dot, lim_t1_obs, lim_t2_obs)
#         # for i in range(len(res[:,0])):
#         #     plt.plot(res[i, 1], res[i, 2], marker='.', color='C' + str(int(res[i, 4])), ls='')  # plot color dots)))
#         #     ax.annotate(str("%.2f" % res[i, 3]), xy=(res[i, 1], res[i, 2]), textcoords='data')
#
#         # for i in range(len(self.num_files)):
#         #     res = self.nums.get_set_of_cols(['t', 'rho', 'He4', 'mdot'], i)
#         #
#         #
#         #     p_lm = Physics.loglm(self.mdl[i].l_[-1], self.mdl[i].xm_[-1], False)
#         #     p_mdot = self.mdl[i].mdot_[-1]
#         #     p_t = self.mdl[i].t_[-1]
#         #
#         #     color = color = 'C' + str(i)
#         #     plt.plot(p_t, p_mdot, marker='x', markersize=9, color=color,
#         #              label='Model {}: T_s {} , mdot {} , L/M: {}'.format(i, "%.2f" % p_t, "%.2f" % p_mdot, "%.2f" % p_lm))
#
#
#         #
#         # t_c = t_coord[Math.find_nearest_index(t_coord, 5.2)]
#         #
#         #
#         #
#         # import re  # for searching the number in 'WN7-e' string, to plot them different colour
#         # obs = Read_Observables()
#         # for i in range(len(obs.numb)):
#         #     s = re.search(r"\d+(\.\d+)?", obs.type[i])  # this is searching for the niumber
#         #     color = 'C' + str(int(s.group(0)))  # making a colour our of C1 - C9 range
#         #
#         #     y_val = Physics.loglm(obs.l_obs[i], obs.m_obs[i])
#         #
#         #     if y_val > loglm[0] or y_val < loglm[-1]:
#         #         print(
#         #             '\t__Warning! Star: {} cannot be plotted: lm_star: {} not in lm_aval: ({}, {})'.format(obs.numb[i],
#         #                                                                                                    "%.2f" % y_val,
#         #                                                                                                    loglm[0],
#         #                                                                                                    loglm[-1]))
#         #     else:
#         #
#         #         i_lm = Math.find_nearest_index(loglm, y_val)
#         #         t_coord = Math.solv_inter_row(t, m_dot[i_lm, :], obs.mdot_obs[i])
#         #         Errors.is_arr_empty(t_coord, '|plot_observer|', True,
#         #                             'y_val: {}, lm:({}, {})'.format("%.2f" % obs.mdot_obs[i],
#         #                                                             "%.2f" % m_dot[i_lm, :].min(),
#         #                                                             "%.2f" % m_dot[i_lm, :].max()))
#         #
#         #         t_c = t_coord[Math.find_nearest_index(t_coord, 5.2)]
#         #         print("t_coord: {}, lm_c: {}, mdot_c: {}".format(t_coord, "%.2f" % y_val, obs.mdot_obs[i]))
#         #
#         #         plt.plot(t_c, y_val, marker='o', color=color, ls='')
#         #         # label = str(obs.numb[i]) + ' ' + obs.type[i])
#         #         ax.annotate(str(obs.numb[i]), xy=(t_c, y_val), textcoords='data')
#
#     # def plot_t_lm_mdot(self, t1, t2, lm1, lm2, r_s_, n_int = 100, n_out = 100,
#     #                    lim_t1_obs = None, lim_t2_obs = None):
#     #
#     #     # loglm1 = 3.8
#     #     # loglm2 = 4.4
#     #
#     #     r_s = r_s_[0]
#     #     if lm1 != None:
#     #         k2 = Physics.loglm_logk(lm1)
#     #     else:
#     #         k2 = None
#     #     if lm2 != None:
#     #         k1 = Physics.loglm_logk(lm2)
#     #     else:
#     #         k1 = None
#     #
#     #
#     #     # i_t1 = Math.find_nearest_index(t, t1)
#     #     # i_t2 = Math.find_nearest_index(t, t2)
#     #     print("Selected T range: ", t1, ' to ', t2)
#     #     print("Selected k range: ", k1, ' to ', k2)
#     #
#     #     # ta = Table_Analyze.from_OPAL_table(self.op_name, n_out, n_int, load_lim_cases)
#     #     res_ = self.tbl_anl.treat_tasks_interp_for_t(t1, t2, n_out, n_int, k1, k2)
#     #
#     #     kap = res_[0, 1:]
#     #     t =   res_[1:, 0]
#     #     rho2d = res_[1:, 1:]
#     #
#     #     # loglm = np.zeros((r_s_, ))
#     #     # for i in range(len(r_s_)):
#     #
#     #     loglm = Physics.logk_loglm(kap, True)
#     #     m_dot = Physics.rho_mdot(t, rho2d.T, 2, r_s)
#     #
#     #
#     #
#     #     print(t.shape, loglm.shape, m_dot.shape)
#     #     #-------------------------------------------POLT-Ts-LM-MODT-COUTUR------------------------------------
#     #     name = './results/t_LM_Mdot_plot.pdf'
#     #
#     #     fig = plt.figure()
#     #     ax = fig.add_subplot(1, 1, 1)
#     #
#     #     pl.xlim(t.min(), t.max())
#     #     pl.ylim(loglm.min(), loglm.max())
#     #     levels = [-7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2]
#     #     # levels = [-10, -9, -8, -7, -6, -5, -4]
#     #     contour_filled = plt.contourf(t, loglm, m_dot, levels, cmap=plt.get_cmap('RdYlBu_r'))
#     #     plt.colorbar(contour_filled)
#     #     contour = plt.contour(t, loglm, m_dot, levels, colors='k')
#     #     plt.clabel(contour, colors='k', fmt='%2.1f', fontsize=12)
#     #     plt.title('MASS LOSS PLOT')
#     #     plt.xlabel('Log(t)')
#     #     plt.ylabel('log(L/M)')
#     #
#     #     #--------------------------------------------------PLOT-MINS-------------------------------------------
#     #     self.mins = Math.get_mins_in_every_row(t, loglm, m_dot, 5000, 5.1, 5.3)
#     #     plt.plot(self.mins[0, :], self.mins[1, :], '-.', color='red', label='min_Mdot')
#     #
#     #     #-----------------------------------------------PLOT-OBSERVABLES-----------------------------------
#     #     types = []
#     #     if self.obs != None: # plot observed stars
#     #
#     #         ''' Read the observables file and get the necessary values'''
#     #         import re  # for searching the number in 'WN7-e' string, to plot them different colour
#     #         ts_ = []
#     #         lm_ = []
#     #         for i in range(self.obs.num_stars):
#     #             s = re.search(r"\d+(\.\d+)?", self.obs.obs_par('type', str)[i])  # this is searching for the niumber
#     #             color = 'C' + str(int(s.group(0)))  # making a colour our of C1 - C9 range
#     #
#     #             star_lm = Physics.loglm(self.obs.obs_par('log(L)', float)[i], self.obs.obs_par('M', float)[i])
#     #             # Effective T << T_s, that you have to get from mass loss!
#     #             ts_lm = Physics.lm_mdot_obs_to_ts_lm(t, loglm, m_dot, star_lm, self.obs.obs_par('log(Mdot)',float)[i],
#     #                                                  self.obs.obs_par('WR',int)[i], lim_t1_obs, lim_t2_obs)
#     #             # ts_lm = np.vstack((star_lm, np.log10(obs.get_parms('T_*', float)[i])))
#     #
#     #             if ts_lm.any():
#     #                 # print(ts_lm[1, :], ts_lm[0, :])
#     #                 ts_ = np.append(ts_, ts_lm[1, :]) # FOR linear fit
#     #                 lm_ = np.append(lm_, ts_lm[0, :])
#     #                 # print(len(ts_lm[0,:]))
#     #                 for j in range(len(ts_lm[0,:])): # plot every solution in the degenerate set of solutions
#     #                     plt.plot(ts_lm[1, j], ts_lm[0, j], marker='^', color=color, ls='')
#     #                     ax.annotate(self.obs.obs_par('WR',str)[i], xy=(ts_lm[1, j], ts_lm[0, j]), textcoords='data')
#     #
#     #                     if int(s.group(0)) not in types: # plotting the legent for unique class of stars
#     #                         plt.plot(ts_lm[1, j], ts_lm[0, j], marker='^', color=color, ls='',
#     #                                  label=self.obs.obs_par('type',str)[i])
#     #                     types.append(int(s.group(0)))
#     #
#     #     # -----------------------------------------------LINEAR FIT TO THE DATA-------------------------------------
#     #         ts_grid_lm_grid = Math.line_fit(ts_, lm_)
#     #         plt.plot(ts_grid_lm_grid[0,:],ts_grid_lm_grid[1,:], '-.', color='blue')
#     #
#     #     #----------------------------------------------PLOT-NUMERICAL-MODELS-----------------------------
#     #     # m_dots = ["%.2f" %  self.mdl[i].mdot_[-1] for i in range(self.nmdls)]
#     #     # colors = Math.get_list_uniq_ints(m_dots)
#     #     # print(m_dots)
#     #     # print(colors)
#     #
#     #     for i in range(self.nmdls):
#     #         p_lm = Physics.loglm(self.mdl[i].l_[-1], self.mdl[i].xm_[-1], False)
#     #         p_mdot = self.mdl[i].mdot_[-1]
#     #         p_t = self.mdl[i].t_[-1]
#     #
#     #
#     #         color = 'C' + str(int(i*10/self.nmdls))
#     #         plt.plot(p_t, p_lm, marker='.', markersize=9, color=color)
#     #                  # label='Model {}: T_s {} , L/M {} , Mdot {}'.format(i, "%.2f" % p_t, "%.2f" % p_lm, "%.2f" % p_mdot))
#     #         ax.annotate(str(i), xy=(p_t, p_lm), textcoords='data')
#     #
#     #
#     #     plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
#     #
#     #     plt.savefig(name)
#     #
#     #     plt.show()
#     #
#     #     #===============================================================================================================
#     #     #           Minimum Mass loss = f(L/M)
#     #     #===============================================================================================================
#     #     '''<<< Possible to treat multiple sonic radii >>>'''
#     #
#     #     plot_name = './results/Min_Mdot.pdf'
#     #
#     #
#     #     fig = plt.figure()
#     #     ax = fig.add_subplot(1, 1, 1)
#     #
#     #     plt.title('L/M = f(min M_dot)')
#     #
#     #     #--------------------------------------_PLOT MINS-------------------------------------------------
#     #     for i in range(len(r_s_)):
#     #         loglm_ = Physics.logk_loglm(kap, True)
#     #         m_dot_ = Physics.rho_mdot(t, rho2d.T, 2, r_s_[i])
#     #
#     #         self.mins_ = Math.get_mins_in_every_row(t, loglm_, m_dot_, 5000, 5.1, 5.3)
#     #
#     #         min_mdot_arr_ = np.array(self.mins_[2, :])
#     #         color = 'C' + str(i)
#     #         plt.plot(min_mdot_arr_, loglm_, '-', color=color, label='min_Mdot for r_s: {}'.format(r_s_[i]))
#     #
#     #     #---------------------------------------ADJUST MAXIMUM L/M FOR OBSERVATIONS------------------------
#     #
#     #     min_mdot_arr = np.array(self.mins[2, :])
#     #     plt.xlim(-6.0, min_mdot_arr.max())
#     #
#     #     if self.obs != None:
#     #         star_lm = np.zeros(self.obs.num_stars)
#     #         for i in range(self.obs.num_stars):
#     #             star_lm[i] = Physics.loglm(self.obs.obs_par('log(L)', float)[i], self.obs.obs_par('M', float)[i])
#     #
#     #         plt.ylim(loglm.min(), star_lm.max())
#     #
#     #     plt.xlabel('log(M_dot)')
#     #     plt.ylabel('log(L/M)')
#     #
#     #
#     #     # major_xticks = np.array([-6.5,-6,-5.5,-5,-4.5,-4,-3.5])
#     #     # minor_xticks = np.arange(-7.0,-3.5,0.1)
#     #     #
#     #     # major_yticks = np.array([3.8,3.9,4.0,4.1,4.2,4.3,4.4,4.5])
#     #     # minor_yticks = np.arange(3.8, 4.5, 0.05)
#     #
#     #     # major_xticks = np.arange(min_mdot_arr.min(), min_mdot_arr.max()+0.1, (min_mdot_arr.max() - min_mdot_arr.min())/4)
#     #     # minor_xticks = np.arange(min_mdot_arr.min(), min_mdot_arr.max(), (min_mdot_arr.max() - min_mdot_arr.min())/8)
#     #     #
#     #     # major_yticks = np.arange(lm_arr.min(), lm_arr.max() + 0.1, ((lm_arr.max() - lm_arr.min()) / 4))
#     #     # minor_yticks = np.arange(lm_arr.min(), lm_arr.max(), ((lm_arr.max() - lm_arr.min()) / 8))
#     #
#     #     ax.grid(which='major', alpha=0.2)
#     #
#     #     plt.legend(bbox_to_anchor=(1, 1), loc='upper right', ncol=1)
#     #
#     #     #--------------------------------------PLOT-OBSERVABLES----------------------------------------
#     #     x_data = []
#     #     y_data = []
#     #     types = []
#     #     if self.obs != None:  # plot array of observed stars from Read_Observ()
#     #         import re  # for searching the number in 'WN7-e' string, to plot them different colour
#     #
#     #         x_data = []
#     #         y_data = [] # for linear fit as well
#     #         for i in range(self.obs.num_stars):
#     #             s = re.search(r"\d+(\.\d+)?", self.obs.obs_par('type', str)[i])  # this is searching for the niumber
#     #             color = 'C' + str(int(s.group(0)))  # making a colour our of C1 - C9 range
#     #
#     #             x_data = np.append(x_data, self.obs.obs_par('log(Mdot)', float)[i])
#     #             y_data = np.append(y_data, Physics.loglm(self.obs.obs_par('log(L)', float)[i], self.obs.obs_par('M', float)[i]) )
#     #
#     #             plt.plot(x_data[i], y_data[i], marker='^', color=color, ls='')  # plot dots
#     #             # label = str(obs.numb[i]) + ' ' + obs.type[i])
#     #             ax.annotate(self.obs.obs_par('WR', str)[i], xy=(x_data[i], y_data[i]), textcoords='data')  # plot names next to dots
#     #
#     #             if int(s.group(0)) not in types:  # plotting the legent for unique class of stars
#     #                 plt.plot(x_data[i], y_data[i], marker='^', color=color, ls='',
#     #                          label=self.obs.obs_par('type', str)[i])
#     #             types.append(int(s.group(0)))
#     #
#     #     # -----------------------------------------------LINEAR FIT TO THE DATA-------------------------------------
#     #         xy_line = Math.line_fit(x_data, y_data)
#     #         plt.plot(xy_line[0,:], xy_line[1,:], '-.', color='blue')
#     #
#     #
#     #     # ---------------------------------------PLOT MODELS ------------------------------------------
#     #     for i in range(self.nmdls):
#     #         y = Physics.loglm(self.mdl[i].l_[-1],self.mdl[i].xm_[-1])
#     #         x = self.mdl[i].mdot_[-1]
#     #
#     #         color = 'C' + str(int(i * 10 / self.nmdls))
#     #         plt.plot(x, y, marker='.', markersize=9, color=color)
#     #             # label='Model {}: T_s {} , L/M {} , Mdot {}'.format(i, "%.2f" % p_t, "%.2f" % p_lm, "%.2f" % p_mdot))
#     #         ax.annotate(str(i), xy=(x, y), textcoords='data')
#     #
#     #
#     #     # ax.set_xticks(major_xticks)
#     #     # ax.set_xticks(minor_xticks, minor=True)
#     #     # ax.set_yticks(major_yticks)
#     #     # ax.set_yticks(minor_yticks, minor=True)
#     #
#     #
#     #     ax.grid(which='both')
#     #     ax.grid(which='minor', alpha=0.2)
#     #     ax.fill_between(min_mdot_arr, loglm, color="lightgray", label = 'Mdot < Minimun')
#     #     plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
#     #     plt.savefig(plot_name)
#     #     plt.show()
#
#     @staticmethod
#     def get_min_mdot_set(r_s_, y_label, t, kap, rho2d, ):
#         '''
#         Permorms interpolation over a set of soinic rs, returning: Math.combine(r_s_, y_coord_, min_mdot_arr_.T)
#         (Transposed, because initially in rho, the t - x.
#         :param r_s_:
#         :param y_label:
#         :param t:
#         :param kap:
#         :param rho2d:
#         :return:
#         '''
#
#         min_mdot_arr_ = np.zeros((len(r_s_), len(kap)))
#         r_s_ = np.array(r_s_)
#         y_coord_ = []
#
#         if y_label == 'l':
#             y_coord_ = Physics.lm_to_l(Physics.logk_loglm(kap, True))
#         if y_label == 'lm':
#             y_coord_ = Physics.logk_loglm(kap, True)
#         if y_label != 'l' and y_label !='lm':
#             raise NameError('Only l or lm are allowed as a *y* coordinate, provided: {}'.format(y_label))
#
#         for i in range(len(r_s_)):
#             m_dot_ = Physics.rho_mdot(t, rho2d.T, 2, r_s_[i])
#             mins_ = Math.get_mins_in_every_row(t, y_coord_, m_dot_, 5000, 5.1, 5.3)
#             min_mdot_arr_[i,:] = np.array(mins_[2, :])
#             # color = 'C' + str(i)
#
#         print('\t__Mdot_set_shape: -- {} r_s ; | {} l or lm ; |- {} mdot'.format(r_s_.shape, y_coord_.shape, min_mdot_arr_.T.shape))
#
#         mm = np.flip(min_mdot_arr_.T ,0)
#         ll = np.flip(y_coord_, 0)
#
#         return Math.combine(r_s_, ll, mm)
#         # return Math.combine(r_s_, y_coord_, min_mdot_arr_.T)
#
#
#             # plt.plot(min_mdot_arr_[i,:], y_coord_, '-', color=color, label='min_Mdot for r_s: {}'.format(r_s_[i]))
#
#     def plot_min_mdot(self, t, kap, rho2d, y_coord, r_s_, y_name, num_var_plot ='Y_c'):
#         # ===============================================================================================================
#         #           Minimum Mass loss = f(L/M)
#         # ===============================================================================================================
#         '''<<< Possible to treat multiple sonic radii >>>'''
#
#         y_coord = np.flip(y_coord, 0)  # as you flip the mdot 2d array.
#
#         plot_name = self.plot_dir+'minMdot_l.pdf'
#
#         fig = plt.figure()
#         ax = fig.add_subplot(1, 1, 1)
#
#         plt.title('L or L/M = f(min M_dot)')
#
#         # --------------------------------------_PLOT MINS-------------------------------------------------
#         ClassPlots.get_min_mdot_set(r_s_, y_name, t, kap, rho2d)
#
#         # min_mdot_arr_ = np.zeros((len(r_s_), len(y_coord)))  # Down are the r_s, -> are the min mass losses
#         # for i in range(len(r_s_)):
#         #
#         #     if y_label == 'log(L)':
#         #         y_coord_ = Physics.lm_to_l(Physics.logk_loglm(kap, True))
#         #     else:
#         #         y_coord_ = Physics.logk_loglm(kap, True)
#         #
#         #     m_dot_ = Physics.rho_mdot(t, rho2d.T, 2, r_s_[i])
#         #
#         #     mins_ = Math.get_mins_in_every_row(t, y_coord_, m_dot_, 5000, 5.1, 5.3)
#         #
#         #     min_mdot_arr_[i,:] = np.array(mins_[2, :])
#         #     color = 'C' + str(i)
#         #     plt.plot(min_mdot_arr_[i,:], y_coord_, '-', color=color, label='min_Mdot for r_s: {}'.format(r_s_[i]))
#
#         min_mdot_arr_ = ClassPlots.get_min_mdot_set(r_s_, y_name, t, kap, rho2d)
#         min_mdot_0 = min_mdot_arr_[1:, 1]
#
#         for i in range(len(r_s_)):
#             color = 'C' + str(i)
#             # print(min_mdot_arr_[1:, 1].shape, min_mdot_arr_[1:,0].shape)
#             plt.plot(min_mdot_arr_[1:, (1+i)], min_mdot_arr_[1:,0], '-', color=color, label='min_Mdot for r_s: {}'.format(r_s_[i]))
#         # ---------------------------------------ADJUST MAXIMUM L/M FOR OBSERVATIONS------------------------
#
#         plt.xlim(-6.0, min_mdot_0.max())
#
#         #----------------------------------------------PLOT-NUMERICALS--------------------------------------------------
#         nums = Treat_Numercials(self.num_files) # Surface Temp as a x coordinate
#         res = nums.get_x_y_of_all_numericals('sp', 'mdot', y_name, num_var_plot, 'color')
#         for i in range(len(res[:,0])):
#             plt.plot(res[i, 1], res[i, 2], marker='.', color='C' + str(int(res[i, 4])), ls='',
#                      label='{}:{}'.format(num_var_plot, "%.2f" % res[i, 3]))  # plot color dots)))
#             ax.annotate(str("%.2f" % res[i, 3]), xy=(res[i, 1], res[i, 2]), textcoords='data')
#
#         #-----------------------------------------------PLOT-OBSERVABLES------------------------------------------------
#         obs = Treat_Observables(self.obs_files)
#         res = obs.get_x_y_of_all_observables('mdot', y_name, 'type')
#         for i in range(len(res[0][:, 1])):
#             ax.annotate(int(res[0][i, 0]), xy=(res[0][i, 1], res[0][i, 2]), textcoords='data')  # plot numbers of stars
#             plt.plot(res[0][i, 1], res[0][i, 2], marker='^', color='C' + str(int(res[0][i, 3])),
#                      ls='')  # plot color dots)))
#         # marker='s', mec='w', mfc='g', mew='3', ms=8
#         for j in range(len(res[1][:, 0])):
#             plt.plot(res[1][j, 1], res[1][j, 2], marker='^', color='C' + str(int(res[1][j, 3])), ls='',
#                      label='WN'+str(int(res[1][j, 3])))
#
#
#         x_grid_y_grid = Math.line_fit(res[0][:, 1], res[0][:, 2])
#         plt.plot(x_grid_y_grid[0, :], x_grid_y_grid[1, :], '-.', color='blue')
#
#         #------------------------------------------PLOT-LUMINOCITY-RANGES-OF-MODELS-------------------------------------
#         for i in range(len(self.plot_files)):
#             min_l, max_l = self.plotcl[i].get_min_max_l_lm_val(y_name)
#             ax.fill_between(np.array([-6.0, min_mdot_0.max()]), np.array([min_l]), np.array([max_l]),  alpha=0.5,
#                             label='L range of ({}->{})sm star'.format(self.plotcl[i].m_[0], self.plotcl[i].m_[-1]))
#
#
#
#         plt.xlabel('log(M_dot)')
#         plt.ylabel(y_name)
#         ax.grid(which='major', alpha=0.2)
#         plt.legend(bbox_to_anchor=(1, 1), loc='upper right', ncol=1)
#
#         ax.grid(which='both')
#         ax.grid(which='minor', alpha=0.2)
#         ax.fill_between(min_mdot_0, min_mdot_arr_[1:,0], color="lightgray", label='Mdot < Minimun')
#         plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
#         plt.savefig(plot_name)
#         plt.show()
#
#
#         # if self.obs != None:
#         #     star_y_coord = np.zeros(self.obs.num_stars)
#         #     for i in range(self.obs.num_stars):
#         #         if y_label == 'log(L)':
#         #             star_y_coord[i] = self.obs.obs_par('log(L)', float)[i]
#         #         else:
#         #             star_y_coord[i] = Physics.loglm(self.obs.obs_par('log(L)', float)[i], self.obs.obs_par('M', float)[i])
#         #
#         #     plt.ylim(y_coord.min(), star_y_coord.max())
#
#         # plt.xlabel('log(M_dot)')
#         # plt.ylabel(y_label)
#         # ax.grid(which='major', alpha=0.2)
#         # plt.legend(bbox_to_anchor=(1, 1), loc='upper right', ncol=1)
#
#         # --------------------------------------PLOT-OBSERVABLES----------------------------------------
#         # obs = Treat_Observables(self.obs_files)
#         # res = obs.get_x_y_of_all_observables('mdot', 'l', 'type')
#         #
#         # for i in range(len(res[0][:, 1])):
#         #     ax.annotate(int(res[0][i, 0]), xy=(res[0][i, 1], res[0][i, 2]), textcoords='data')  # plot numbers of stars
#         #     plt.plot(res[0][i, 1], res[0][i, 2], marker='^', color='C' + str(int(res[0][i, 3])),
#         #              ls='')  # plot color dots)))
#         # # marker='s', mec='w', mfc='g', mew='3', ms=8
#         # for j in range(len(res[1][:, 0])):
#         #     plt.plot(res[1][j, 1], res[1][j, 2], marker='^', color='C' + str(int(res[1][j, 3])), ls='',
#         #              label='WN'+str(int(res[1][j, 3])))
#         #
#         # x_grid_y_grid = Math.line_fit(res[0][:, 1], res[0][:, 2])
#         # plt.plot(x_grid_y_grid[0, :], x_grid_y_grid[1, :], '-.', color='blue')
#
#
#
#         # types = []
#         # if self.obs != None:  # plot array of observed stars from Read_Observ()
#         #     import re  # for searching the number in 'WN7-e' string, to plot them different colour
#         #
#         #     x_data = []
#         #     y_data = []  # for linear fit as well
#         #     for i in range(self.obs.num_stars):
#         #         s = re.search(r"\d+(\.\d+)?", self.obs.obs_par('type', str)[i])  # this is searching for the niumber
#         #         color = 'C' + str(int(s.group(0)))  # making a colour our of C1 - C9 range
#         #
#         #         x_data = np.append(x_data, self.obs.obs_par('log(Mdot)', float)[i])
#         #         if y_label == 'log(L)':
#         #             y_data = np.append(y_data, self.obs.obs_par('log(L)', float)[i])
#         #         else:
#         #             y_data = np.append(y_data, Physics.loglm(self.obs.obs_par('log(L)', float)[i], self.obs.obs_par('M', float)[i]))
#         #
#         #         plt.plot(x_data[i], y_data[i], marker='^', color=color, ls='')  # plot dots
#         #         # label = str(obs.numb[i]) + ' ' + obs.type[i])
#         #         ax.annotate(self.obs.obs_par('WR', str)[i], xy=(x_data[i], y_data[i]),
#         #                     textcoords='data')  # plot names next to dots
#         #
#         #         if int(s.group(0)) not in types:  # plotting the legent for unique class of stars
#         #             plt.plot(x_data[i], y_data[i], marker='^', color=color, ls='',
#         #                      label=self.obs.obs_par('type', str)[i])
#         #         types.append(int(s.group(0)))
#         #
#         #     # -----------------------------------------------LINEAR FIT TO THE DATA-------------------------------------
#         #     xy_line = Math.line_fit(x_data, y_data)
#         #     plt.plot(xy_line[0, :], xy_line[1, :], '-.', color='blue')
#
#         # ---------------------------------------PLOT MODELS ------------------------------------------
#         # sp_i = -1
#         # for i in range(self.nmdls):
#         #
#         #     sp_v = Physics.sound_speed(self.mdl[i].t_, self.mdl[i].mu_)
#         #     for k in range(len(sp_v)):
#         #         if sp_v[k] <= self.mdl[i].u_[k]:
#         #             sp_i = k
#         #             # print('\t__Note: Last l: {} | sp_l {} '.format("%.3f" % self.mdl[i].l_[-1],
#         #             #                                                "%.3f" % self.mdl[i].l_[sp_i]))
#         #             # print('\t__Note: Last t: {} | sp_t {} '.format("%.3f" % self.mdl[i].t_[-1],
#         #             #                                                "%.3f" % self.mdl[i].t_[sp_i]))
#         #             break
#         #     if sp_i == -1:
#         #         print('Warning! Sonic Velocity is not resolved. Using -1 element f the u arrau.')
#         #
#         #     if y_label == 'log(L)':
#         #         y = self.mdl[i].l_[sp_i]
#         #     else:
#         #         y = Physics.loglm(self.mdl[i].l_[-1],self.mdl[i].xm_[-1])
#         #     x = self.mdl[i].mdot_[-1]
#         #
#         #     color = 'C' + str(int(i * 10 / self.nmdls))
#         #     plt.plot(x, y, marker='.', markersize=9, color=color)
#         #     # label='Model {}: T_s {} , L/M {} , Mdot {}'.format(i, "%.2f" % p_t, "%.2f" % p_lm, "%.2f" % p_mdot))
#         #     ax.annotate(str(i), xy=(x, y), textcoords='data')
#
#         # ax.grid(which='both')
#         # ax.grid(which='minor', alpha=0.2)
#         # ax.fill_between(min_mdot_0, y_coord, color="lightgray", label='Mdot < Minimun')
#         # plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
#         # plt.savefig(plot_name)
#         # plt.show()
#
#     @staticmethod
#     def get_k1_k2_from_llm1_llm2(t1, t2, l1, l2):
#         lm1 = None
#         if l1 != None:
#             lm1 = Physics.l_to_lm(l1)
#         lm2 = None
#         if l2 != None:
#             lm2 = Physics.l_to_lm(l2)
#
#         if lm1 != None:
#             k2 = Physics.loglm_logk(lm1)
#         else:
#             k2 = None
#         if lm2 != None:
#             k1 = Physics.loglm_logk(lm2)
#         else:
#             k1 = None
#
#         print('\t__Provided LM limits ({}, {}), translated to L limits: ({}, {})'.format(lm1, lm2, l1, l2))
#         print('\t__Provided T limits ({},{}), and kappa limits ({}, {})'.format(t1, t2, k1, k2))
#         return [k1, k2]
#
#
#
#     def plot_rs_l_mdot_min(self, y_name, t1, t2, l1, l2, r_s1, r_s2, n_int = 100, n_out = 100, n_r_s = 100, load = False):
#         # ---------------------SETTING LM1 LM2 K1 K2---------------------------
#
#         k1, k2 = Tables.get_k1_k2_from_llm1_llm2(t1, t2, l1, l2)
#
#         # ---------------------Getting KAPPA[], T[], RHO2D[]-------------------------
#         res_ = self.tbl_anl.treat_tasks_interp_for_t(t1, t2, n_out, n_int, k1, k2)
#         kap = res_[0, 1:]
#         t = res_[1:, 0]
#         rho2d = res_[1:, 1:]
#
#         r_s_ = np.mgrid[r_s1:r_s2:n_r_s*1j] # grid of sonic radii
#         name_out = self.output_dir + 'rs_l_mdot.data'
#         if load:
#             min_mdot = np.loadtxt(name_out, dtype=float, delimiter=' ').T
#         else:
#             min_mdot = ClassPlots.get_min_mdot_set(r_s_, y_name, t, kap, rho2d) # compute min_mdot for set of rs
#             np.savetxt(name_out, min_mdot.T, delimiter=' ', fmt='%.3f')
#
#         rs_arr = min_mdot[0,1:]
#         l_lm_arr=min_mdot[1:,0]
#         minmdot_arr=min_mdot[1:,1:]
#
#         fig = plt.figure()
#         ax = fig.add_subplot(1, 1, 1)
#
#         pl.xlim(rs_arr.min(), rs_arr.max())
#         pl.ylim(l_lm_arr.min(), l_lm_arr.max())
#         levels = [-7.5, -7.3, -7, -6.7, -6.5, -6.3, -6, -5.7, -5.5, -5.3, -5, -4.7, -4.5, -4.3, -4, -3.7, -3.5, -3.3, -3]
#         contour_filled = plt.contourf(rs_arr, l_lm_arr, minmdot_arr, levels, cmap=plt.get_cmap('RdYlBu_r'))
#         plt.colorbar(contour_filled)
#         contour = plt.contour(rs_arr, l_lm_arr, minmdot_arr, levels, colors='k')
#         plt.clabel(contour, colors='k', fmt='%2.1f', fontsize=12)
#         plt.title('MINIMUM MASS LOSS RATE PLOT')
#         plt.xlabel('r_s')
#
#         #------------------------------------------------------OBSERVABLES----------------------------------------------
#         # obs = Treat_Observables(self.obs_files)
#         # res = obs.get_x_y_of_all_observables('rs', 'l', 'type', rs_arr, l_lm_arr, minmdot_arr)
#         #
#         # for i in range(len(res[0][:, 1])):
#         #     ax.annotate(int(res[0][i, 0]), xy=(res[0][i, 1], res[0][i, 2]), textcoords='data')  # plot numbers of stars
#         #     plt.plot(res[0][i, 1], res[0][i, 2], marker='^', ms=8,  color='C' + str(int(res[0][i, 3])),
#         #              ls='')  # plot color dots)))
#         #
#         # for j in range(len(res[1][:, 0])):
#         #     plt.plot(res[1][j, 1], res[1][j, 2], marker='^', ms=8, color='C' + str(int(res[1][j, 3])), ls='',
#         #              label='WN'+str(int(res[1][j, 3])))
#         #
#         # x_grid_y_grid = Math.line_fit(res[0][:, 1], res[0][:, 2])
#         # plt.plot(x_grid_y_grid[0, :], x_grid_y_grid[1, :], '-.', color='blue')
#
#         #------------------------------------------------------ATTEMPT-TO-SET-THE-Rs-AND-GET-THE-MDOT-------------------
#         # new_rs = 1.
#         # obs = Treat_Observables(self.obs_files)
#         # # res = obs.get_x_y_of_all_observables('mdot', 'l', 'type', rs_arr, l_lm_arr, minmdot_arr)
#         # res = obs.get_x_y_of_all_observables('rs', 'l', 'type', rs_arr, l_lm_arr, minmdot_arr)
#         #
#         # f = interpolate.interp2d(rs_arr, l_lm_arr, minmdot_arr, kind='cubic')
#         #
#         #
#         # for i in range(len(res[0][:, 1])):
#         #     # mdot = f(res[0][i, 1], res[0][i, 2])
#         #     mdot = f(new_rs, res[0][i, 2])
#         #     print("Star: {} has mdot:{} at rs:{}, if rs:{}, mdot is {} "
#         #           .format(int(res[0][i, 0]),res[0][i, 1] , res[0][i, 1], new_rs, mdot))
#         #     print(mdot)
#         #     rs_lm = Physics.lm_mdot_obs_to_ts_lm(rs_arr, l_lm_arr, minmdot_arr, res[0][i, 2], mdot, i)
#         #     # print(rs_arr)
#         #     if rs_lm.any():
#         #         ax.annotate(int(res[0][i, 0]), xy=(rs_lm[1,:], rs_lm[0,:]), textcoords='data')  # plot numbers of stars
#         #         plt.plot(rs_lm[1,:], rs_lm[0,:], marker='*', ms=12,  color='C' + str(int(res[0][i, 3])),
#         #                  ls='')  # plot color dots)))
#         #
#
#         #-------------------------------------------------------NUMERICALS----------------------------------------------
#         nums = Treat_Numercials(self.num_files)  # Surface Temp as a x coordinate
#         res = nums.get_x_y_of_all_numericals('sp', 'r', y_name, 'xm', 'mdot')
#         for i in range(len(res[:, 0])):
#             plt.plot(res[i, 1], res[i, 2], marker='.', color='C' + str( Math.get_0_to_max([i],9)[i] ), ls='',
#                      label='{} {}:{}, {}:{}'.format(res[i, 0], 'Y_c', "%.2f" % res[i, 3], 'mdot', "%.2f" % res[i, 4] ))  # plot color dots)))
#             ax.annotate(str("%.2f" % res[i, 3]), xy=(res[i, 1], res[i, 2]), textcoords='data')
#
#
#         plt.ylabel(y_name)
#         plt.legend(bbox_to_anchor=(1, 0), loc='lower right', ncol=1)
#         name = self.plot_dir + 'rs_l_minMdot.pdf'
#         plt.savefig(name)
#         plt.show()
#
#     def plot_t_l_mdot(self, l_or_lm, t1, t2, llm1, llm2, r_s_, n_int = 100, n_out = 100,
#                       num_var_plot = 'xm', num_pol_fit = None, lim_t1_obs = None, lim_t2_obs = None):
#         # ---------------------SETTING LM1 LM2 K1 K2---------------------------
#
#         k1, k2 = ClassPlots.get_k1_k2_from_llm1_llm2(t1, t2, llm1, llm2)
#
#         #---------------------Getting KAPPA[], T[], RHO2D[]-------------------------
#         res_ = self.tbl_anl.treat_tasks_interp_for_t(t1, t2, n_out, n_int, k1, k2)
#         kap = res_[0, 1:]
#         t =   res_[1:, 0]
#         rho2d = res_[1:, 1:]
#
#         if l_or_lm == 'l':
#             l_lm_arr  = Physics.lm_to_l( Physics.logk_loglm(kap, True) ) # Kappa -> L/M -> L
#         else:
#             l_lm_arr = Physics.logk_loglm(kap, 1)
#
#
#         self.plot_min_mdot(t, kap, rho2d, l_lm_arr, r_s_, l_or_lm, num_var_plot)
#
#
#
#         m_dot = Physics.rho_mdot(t, rho2d.T, 2, r_s_[0]) # the
#
#         mins = Math.get_mins_in_every_row(t, l_lm_arr, m_dot, 5000, 5.1, 5.3)
#
#         print('\t__Note: PLOT: x: {}, y: {}, z: {} shapes.'.format(t.shape, l_lm_arr.shape, m_dot.shape))
#
#         #-------------------------------------------POLT-Ts-LM-MODT-COUTUR------------------------------------
#         t_l_mdot = Tables.read_table('t_l_mdot','t','l','mdot',self.opalfl, self.output_dir)
#         t = t_l_mdot[0,1:]
#         l_lm_arr = t_l_mdot[1:,0]
#         m_dot = t_l_mdot[1:,1:]
#
#         name = self.plot_dir + 'rs_lm_minMdot_plot.pdf'
#
#         fig = plt.figure()
#         ax = fig.add_subplot(1, 1, 1)
#         plt.xlim(t.min(), t.max())
#         plt.ylim(l_lm_arr.min(), l_lm_arr.max())
#         levels = [-7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2]
#         contour_filled = plt.contourf(t, l_lm_arr, m_dot, levels, cmap=plt.get_cmap('RdYlBu_r'))
#         plt.colorbar(contour_filled)
#         contour = plt.contour(t, l_lm_arr, m_dot, levels, colors='k')
#         plt.clabel(contour, colors='k', fmt='%2.1f', fontsize=12)
#         plt.title('MASS LOSS PLOT')
#         plt.xlabel('Log(t)')
#         plt.ylabel(l_or_lm)
#         plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
#         plt.savefig(name)
#
#         #--------------------------------------------------PLOT-MINS-------------------------------------------
#
#         # plt.plot(mins[0, :], mins[1, :], '-.', color='red', label='min_Mdot (rs: {} )'.format(r_s_[0]))
#
#         #-----------------------------------------------PLOT-OBSERVABLES-----------------------------------
#         obs = Treat_Observables(self.obs_files)
#         res = obs.get_x_y_of_all_observables('ts', l_or_lm, 'type', t, l_lm_arr, m_dot, lim_t1_obs, lim_t2_obs)
#
#         for i in range(len( res[0][:, 1] )):
#             ax.annotate(int(res[0][i, 0]), xy=(res[0][i, 1], res[0][i, 2]), textcoords='data') # plot numbers of stars
#             plt.plot(res[0][i, 1], res[0][i, 2], marker='^', color='C' + str(int(res[0][i, 3])), ls='') # plot color dots)))
#
#         for j in range(len(res[1][:, 0])):
#             plt.plot(res[1][j, 1], res[1][j, 2], marker='^', color='C' + str(int(res[1][j, 3])), ls='',
#                      label='WN'+str(int(res[1][j, 3])))
#
#         x_grid_y_grid = Math.line_fit(res[0][:, 1], res[0][:, 2])
#         plt.plot(x_grid_y_grid[0, :], x_grid_y_grid[1, :], '-.', color='blue')
#
#         # ------------------------------------------------PLOT-NUMERICALS-----------------------------------
#
#         nums = Treat_Numercials(self.num_files) # Surface Temp as a x coordinate
#         res = nums.get_x_y_of_all_numericals('sp', 't', l_or_lm, num_var_plot, 'color', t, l_lm_arr, m_dot, lim_t1_obs, lim_t2_obs)
#         for i in range(len(res[:,0])):
#             plt.plot(res[i, 1], res[i, 2], marker='.', color='C' + str(int(res[i, 4])), ls='')  # plot color dots)))
#             ax.annotate(str("%.2f" % res[i, 3]), xy=(res[i, 1], res[i, 2]), textcoords='data')
#
#         if num_pol_fit !=None:
#             fit = np.polyfit(res[:, 1], res[:, 2], 3)  # fit = set of coeddicients (highest first)
#             f = np.poly1d(fit)
#             fit_x_coord = np.mgrid[(res[1:, 1].min() - 0.02):(res[1:, 1].max() + 0.02):100j]
#             plt.plot(fit_x_coord, f(fit_x_coord), '--', color='black', label='Model_fit')
#
#
#         res = nums.get_x_y_of_all_numericals('sp', 'ts', l_or_lm, num_var_plot, 'color', t, l_lm_arr, m_dot, lim_t1_obs, lim_t2_obs)
#         for i in range(len(res[:,0])):
#             plt.plot(res[i, 1], res[i, 2], marker='.', color='C' + str(int(res[i, 4])), ls='')  # plot color dots)))
#             ax.annotate(str("%.2f" % res[i, 3]), xy=(res[i, 1], res[i, 2]), textcoords='data')
#
#         if num_pol_fit != None:
#             fit = np.polyfit(res[:, 1], res[:, 2], 3)  # fit = set of coeddicients (highest first)
#             f = np.poly1d(fit)
#             fit_x_coord = np.mgrid[(res[1:, 1].min() - 0.02):(res[1:, 1].max() + 0.02):100j]
#             plt.plot(fit_x_coord, f(fit_x_coord), '--', color='black', label='Model_fit')
#
#         #------------------------------------------PLOT-LUMINOCITY-RANGES-OF-MODELS-------------------------------------
#         for i in range(len(self.plot_files)):
#             min_l, max_l = self.plotcl[i].get_min_max_l_lm_val(l_or_lm)
#             ax.fill_between(np.array([t.min(), t.max()]), np.array([min_l]), np.array([max_l]),  alpha=0.5,
#                             label='L range of ({}->{})sm star'.format(self.plotcl[i].m_[0], self.plotcl[i].m_[-1]))
#
#
#
#         #
#         #
#         #
#         # # types = []
#         # # plotted_stars = np.array([0., 0., 0., 0.])
#         # # if self.obs != None: # plot observed stars
#         # #     ''' Read the observables file and get the necessary values'''
#         # #     import re  # for searching the number in 'WN7-e' string, to plot them different colour
#         # #     ts_ = []
#         # #     y_coord_ = []
#         # #     for i in range(self.obs.num_stars):
#         # #         if y_name == 'l':
#         # #             star_y_coord = self.obs.obs_par('log(L)', float)[i]
#         # #         else:
#         # #             star_y_coord = Physics.loglm(self.obs.obs_par('log(L)', float)[i], self.obs.obs_par('M', float)[i])
#         # #
#         # #         # Effective T << T_s, that you have to get from mass loss!
#         # #         ts_y_coord = Physics.lm_mdot_obs_to_ts_lm(t, l_lm_arr, m_dot, star_y_coord, self.obs.obs_par('log(Mdot)',float)[i],
#         # #                                              self.obs.obs_par('WR',int)[i], lim_t1_obs, lim_t2_obs)
#         # #
#         # #         if ts_y_coord.any():
#         # #             # print(ts_lm[1, :], ts_lm[0, :])
#         # #             ts_ = np.append(ts_, ts_y_coord[1, :]) # FOR linear fit
#         # #             y_coord_ = np.append(y_coord_, ts_y_coord[0, :])
#         # #
#         # #             # print(len(ts_lm[0,:]))
#         # #
#         # #             s = re.search(r"\d+(\.\d+)?", self.obs.obs_par('type', str)[i])  # this is searching for the niumber
#         # #             color = 'C' + str(int(s.group(0)))  # making a colour our of C1 - C9 range
#         # #
#         # #             for j in range(len(ts_y_coord[0,:])): # plot every solution in the degenerate set of solutions
#         # #                 plt.plot(ts_y_coord[1, j], ts_y_coord[0, j], marker='^', color=color, ls='')
#         # #                 ax.annotate(self.obs.obs_par('WR',str)[i], xy=(ts_y_coord[1, j], ts_y_coord[0, j]), textcoords='data')
#         # #
#         # #                 # print( np.array((i, ts_y_coord[1, j], ts_y_coord[0, j], self.obs.obs_par('log(Mdot)',float)[i] )) )
#         # #                 plotted_stars = np.vstack((plotted_stars, np.array((self.obs.obs_par('WR',int)[i], ts_y_coord[1, j], ts_y_coord[0, j], self.obs.obs_par('log(Mdot)',float)[i] )))) # for further printing
#         # #
#         # #                 if int(s.group(0)) not in types: # plotting the legent for unique class of stars
#         # #                     plt.plot(ts_y_coord[1, j], ts_y_coord[0, j], marker='^', color=color, ls='',
#         # #                              label=self.obs.obs_par('type',str)[i])
#         # #                 types.append(int(s.group(0)))
#         # #
#         # #     # -----------------------------------------------LINEAR FIT TO THE DATA-------------------------------------
#         # #     ts_grid_y_grid = Math.line_fit(ts_, y_coord_)
#         # #     plt.plot(ts_grid_y_grid[0,:],ts_grid_y_grid[1,:], '-.', color='blue')
#         # #
#         # #
#         # # print('\n| Plotted Stras from Observ |')
#         # # print(  '|  i  |  t   |  {}  | m_dot |'.format(y_mode))
#         # # print(  '|-----|------|------|-------|')
#         # # for i in range(1, len(plotted_stars[:,0])):
#         # #     print('| {} | {} | {} | {} |'.format("%3.f" % plotted_stars[i,0], "%.2f" % plotted_stars[i,1], "%.2f" %plotted_stars[i,2], "%.2f" %plotted_stars[i,3]))
#         #
#         # # ----------------------------------------------PLOT-NUMERICAL-MODELS-----------------------------
#         # m_dots = ["%.2f" %  self.mdl[i].mdot_[-1] for i in range(self.nmdls)]
#         # colors = Math.get_list_uniq_ints(m_dots)
#         # # print(m_dots)
#         # # print(colors)
#         #
#         # sp_i = -1
#         #
#         # model_stars1 = np.array([0., 0., 0., 0., 0.])
#         # model_stars2 = np.array([0., 0., 0., 0., 0.])
#         # for i in range(self.nmdls):
#         #     sp_v = Physics.sound_speed(self.mdl[i].t_, self.mdl[i].mu_)
#         #     for k in range(len(sp_v)):
#         #         if sp_v[k] <= self.mdl[i].u_[k]:
#         #             sp_i = k
#         #             break
#         #     if sp_i == -1:
#         #         print('Warning! Sonic Velocity is not resolved. Using -1 element f the u arrau.')
#         #         # print('\t__Note: Last l: {} | sp_l {} '.format("%.3f" % self.mdl[i].l_[-1], "%.3f" % self.mdl[i].l_[sp_i]))
#         #         # print('\t__Note: Last t: {} | sp_t {} '.format("%.3f" % self.mdl[i].t_[-1], "%.3f" % self.mdl[i].t_[sp_i]))
#         #
#         #     mod_x_coord = self.mdl[i].t_[sp_i]
#         #     if y_name == 'l':
#         #         mod_y_coord = self.mdl[i].l_[sp_i]
#         #     else:
#         #         mod_y_coord = Physics.loglm(self.mdl[i].l_[sp_i], self.mdl[i].xm_[sp_i])
#         #
#         #     color = 'C' + str(int(i*10/self.nmdls))
#         #     plt.plot(mod_x_coord, mod_y_coord, marker='.', markersize=9, color=color)
#         #              # label='Model {}: T_s {} , L/M {} , Mdot {}'.format(i, "%.2f" % p_t, "%.2f" % p_lm, "%.2f" % p_mdot))
#         #     ax.annotate(str(i), xy=(mod_x_coord, mod_y_coord), textcoords='data')
#         #
#         #
#         #     #--------------------------SAME BUT USING Mdot TO GET SONIC TEMPERATURE (X-Coordinate)------------------------
#         #     p_mdot = self.mdl[i].mdot_[sp_i]
#         #     ts_y_model = Physics.lm_mdot_obs_to_ts_lm(t, l_lm_arr, m_dot, mod_y_coord, p_mdot, i, lim_t1_obs, lim_t2_obs)
#         #     if ts_y_model.any():
#         #         for j in range(len(ts_y_model[0, :])):
#         #             plt.plot(ts_y_model[1, j], ts_y_model[0, j], marker='.', markersize=9, color=color)
#         #             ax.annotate('m'+str(i), xy=(ts_y_model[1, j], ts_y_model[0, j]), textcoords='data')
#         #             model_stars1 = np.vstack((model_stars1, np.array((i, ts_y_model[1, j], ts_y_model[0, j], p_mdot, self.mdl[i].He4_[0] ))))
#         #
#         #         model_stars2 = np.vstack((model_stars2, np.array((i, mod_x_coord, mod_y_coord, p_mdot, self.mdl[i].He4_[0]))))  # for further printing
#         #
#         #
#         # # -------------------------PLOT FIT FOR THE NUMERICAL MODELS AND TABLES WITH DATA --------------------------------
#         # if model_stars1.any():
#         #     print('\n| Models plotted by ts & lm |')
#         #     print(  '|  i  |  t   |  {}  | m_dot | Y_c  |'.format(y_mode))
#         #     print(  '|-----|------|------|-------|------|')
#         #     print(model_stars1.shape)
#         #     for i in range(1, len(model_stars1[:,0])):
#         #         print('| {} | {} | {} | {} | {} |'.format("%3.f" % model_stars1[i,0], "%.2f" % model_stars1[i,1], "%.2f" % model_stars1[i,2], "%.2f" %model_stars1[i,3], "%.2f" %model_stars1[i,4]))
#         #
#         #     fit = np.polyfit(model_stars1[:, 1], model_stars1[:, 2], 3)  # fit = set of coeddicients (highest first)
#         #     f = np.poly1d(fit)
#         #     fit_x_coord = np.mgrid[(model_stars1[1:, 1].min() - 0.02):(model_stars1[1:, 1].max() + 0.02):100j]
#         #     plt.plot(fit_x_coord, f(fit_x_coord), '--', color='black', label='Model_fit')
#         #
#         # if model_stars2.any():
#         #     print('\n| Models plotted: lm & mdot |')
#         #     print(  '|  i  | in_t |  {}  | m_dot | Y_c  |'.format(y_mode))
#         #     print(  '|-----|------|------|-------|------|')
#         #     for i in range(1, len(model_stars2[:,0])):
#         #         print('| {} | {} | {} | {} | {} |'.format("%3.f" % model_stars2[i,0], "%.2f" % model_stars2[i,1], "%.2f" % model_stars2[i,2], "%.2f" %model_stars2[i,3], "%.2f" %model_stars2[i,4]))
#         #
#         #     fit = np.polyfit(model_stars2[:,1], model_stars2[:,2], 3) # fit = set of coeddicients (highest first)
#         #     f = np.poly1d(fit)
#         #     fit_x_coord = np.mgrid[(model_stars2[1:,1].min()-0.02):(model_stars2[1:,1].max()+0.02):100j]
#         #     plt.plot(fit_x_coord, f(fit_x_coord), '--', color='black', label='Model_fit')
#
#         plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
#         plt.savefig(name)
#         plt.show()
#
#     def hrd(self, t1,t2, observ_table_name, plot_file_names):
#
#         fig, ax = plt.subplots(1, 1)
#
#         plt.title('HRD')
#         plt.xlabel('log(T_eff)')
#         plt.ylabel('log(L)')
#
#         plt.xlim(t1, t2)
#         ax.grid(which='major', alpha=0.2)
#         plt.legend(bbox_to_anchor=(1, 1), loc='upper right', ncol=1)
#
#         res = self.obs.get_x_y_of_all_observables('t', 'l', 'type')
#
#         for i in range(len(res[0][:, 1])):
#             ax.annotate(int(res[0][i, 0]), xy=(res[0][i, 1], res[0][i, 2]), textcoords='data')  # plot numbers of stars
#             plt.plot(res[0][i, 1], res[0][i, 2], marker='^', color='C' + str(int(res[0][i, 3])),
#                      ls='')  # plot color dots)))
#
#         for j in range(len(res[1][:, 0])):
#             plt.plot(res[1][j, 1], res[1][j, 2], marker='^', color='C' + str(int(res[1][j, 3])), ls='',
#                      label='WN' + str(int(res[1][j, 3])))
#
#         ind_arr = []
#         for j in range(len(plot_file_names)):
#             ind_arr.append(j)
#             col_num = Math.get_0_to_max(ind_arr, 9)
#             plfl = Read_Plot_file.from_file(plot_file_names[j])
#
#             mod_x = plfl.t_eff
#             mod_y = plfl.l_
#             color = 'C' + str(col_num[j])
#
#             fname = plot_file_names[j].split('/')[-2] + plot_file_names[j].split('/')[-1]# get the last folder in which the .plot1 is
#
#             plt.plot(mod_x, mod_y, '-', color=color,
#                      label='{}, m:({}->{})'.format(fname, "%.1f" % plfl.m_[0], "%.1f" % plfl.m_[-1]) )
#                      # str("%.2f" % plfl.m_[0]) + ' to ' + str("%.2f" % plfl.m_[-1]) + ' solar mass')
#
#
#
#             for i in range(10):
#                 ind = Math.find_nearest_index(plfl.y_c, (i / 10))
#                 # print(plfl.y_c[i], (i/10))
#                 x_p = mod_x[ind]
#                 y_p = mod_y[ind]
#                 plt.plot(x_p, y_p, '.', color='red')
#                 ax.annotate("%.2f" % plfl.y_c[ind], xy=(x_p, y_p), textcoords='data')
#
#         ax.grid(which='both')
#         ax.grid(which='minor', alpha=0.2)
#
#         plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
#         plot_name = self.output_dir + 'hrd.pdf'
#         plt.savefig(plot_name)
#
#         plt.show()
#
#         # from matplotlib.collections import LineCollection
#         # from matplotlib.colors import ListedColormap, BoundaryNorm
#
#         # for file_name in plot_file_names:
#         #     self.plfl.append(Read_Plot_file.from_file(file_name))
#         #
#         # x_coord = []
#         # y_coord = []
#         # for i in range(len(plot_file_names)):
#         #
#         #     x_coord.append(self.plfl[i].t_eff)
#         #     y_coord.append(self.plfl[i].l_)
#         #
#         #
#         #
#         # points = np.array([x_coord, y_coord]).T.reshape(-1, 1, 2)
#         #     # points = np.append(p,  np.array([x_coord, y_coord]).T.reshape(-1, 1, 2) )
#         # segments = np.concatenate([points[:-1], points[1:]], axis=1)
#         #
#         # # segments = np.append(s, np.concatenate([ points[:-1], points[1:] ], axis=1))
#         #
#         # z_coord = self.plfl[0].y_c
#         # # Create a continuous norm to map from data points to colors
#         # norm = plt.Normalize(z_coord.min(), z_coord.max())
#         # lc = LineCollection(segments, cmap='viridis', norm=norm)
#         # # Set the values used for colormapping
#         # lc.set_array(z_coord)
#         # lc.set_linewidth(2)
#         # line = axs.add_collection(lc)
#         # fig.colorbar(line, ax=axs)
#
#         # n_in_type = []
#         # x = []
#         # y = []
#         # if observ_table_name != None:  # plot array of observed stars from Read_Observ()
#         #     import re  # for searching the number in 'WN7-e' string, to plot them different colour
#         #     obs = Read_Observables(observ_table_name)
#         #
#         #     for i in range(obs.num_stars):
#         #         s = re.search(r"\d+(\.\d+)?", obs.obs_par('type', str)[i])  # this is searching for the niumber
#         #         # n_in_type.append(int(s.group(0)))
#         #         color = 'C' + str(int(s.group(0)))  # making a colour our of C1 - C9 range
#         #
#         #         x = np.append(x, np.log10(obs.obs_par('t', float)[i]*1000) )
#         #         y = np.append(y, obs.obs_par('l', float)[i] )
#         #         # y = Physics.loglm(obs.obs_par('log(L)', float)[i], obs.obs_par('M', float)[i])
#         #
#         #         plt.plot(x[i], y[i], marker='o', color=color, ls='')  # plot dots
#         #         # label = str(obs.numb[i]) + ' ' + obs.type[i])
#         #         ax.annotate(obs.obs_par('WR', str)[i], xy=(x[i], y[i]), textcoords='data')  # plot names next to dots
#         #
#         #         if int(s.group(0)) not in n_in_type:  # plotting the legent for unique class of stars
#         #             plt.plot(x[i], y[i], marker='o', color=color, ls='',
#         #                      label=obs.obs_par('type', str)[i])
#         #         n_in_type.append(int(s.group(0)))
#
#         # axs.set_ylim(  np.array(y_coord.min(), y.min()).min() , np.array(y_coord.max(), y.max()).max()  )
#
#         # for i in range(self.nmdls):
#         #     # p_y = Physics.loglm(self.mdl[i].l_[-1], self.mdl[i].xm_[-1], False)
#         #     p_y = self.mdl[i].l_[-1]
#         #     p_mdot = self.mdl[i].mdot_[-1]
#         #     p_x = self.mdl[i].t_[-1]
#         #
#         #     mdot_color = []
#         #     color = 'C0'
#         #     if p_mdot not in mdot_color:
#         #         color = 'C' + str(i)
#         #     plt.plot(p_mdot, p_y, marker='x', markersize=9, color=color,
#         #              label='Model {}: T_s {} , L/M {} , Mdot {}'.format(i, "%.2f" % p_x, "%.2f" % p_y,
#         #                                                                 "%.2f" % p_mdot))
#
#         # -------------------------------------------------------------------------Math.get_0_to_max()
#
#         # ind_arr = []
#         # for j in range(len(plot_file_names)):
#         #     ind_arr.append(j)
#         #     col_num = Math.get_0_to_max(ind_arr, 9)
#         #     plfl = Read_Plot_file.from_file(plot_file_names[j])
#         #
#         #     mod_x = plfl.t_eff
#         #     mod_y = plfl.l_
#         #     color = 'C' + str(col_num[j])
#         #
#         #     plt.plot(mod_x, mod_y, '-', color=color, label=str("%.2f" % plfl.m_[0])+' to ' + str("%.2f" % plfl.m_[-1]) +' solar mass')
#         #     for i in range(10):
#         #         ind = Math.find_nearest_index(plfl.y_c, (i/10) )
#         #         # print(plfl.y_c[i], (i/10))
#         #         x_p = mod_x[ind]
#         #         y_p = mod_y[ind]
#         #         plt.plot(x_p, y_p, '.', color='red')
#         #         ax.annotate("%.2f" % plfl.y_c[ind], xy=(x_p, y_p), textcoords='data')
#
#             # from matplotlib.collections import LineCollection
#             # from matplotlib.colors import ListedColormap, BoundaryNorm
#             # fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
#             #
#             # # Create a continuous norm to map from data points to colors
#             # norm = plt.Normalize(model.y_c.min(), model.y_c.max())
#             # lc = LineCollection(segments, cmap='viridis', norm=norm)
#             # # Set the values used for colormapping
#             # lc.set_array(dydx)
#             # lc.set_linewidth(2)
#             # line = axs[0].add_collection(lc)
#             # fig.colorbar(line, ax=axs[0])
#
#
#
#         # ax.set_xticks(major_xticks)
#         # ax.set_xticks(minor_xticks, minor=True)
#         # ax.set_yticks(major_yticks)
#         # ax.set_yticks(minor_yticks, minor=True)
#
#         # ax.fill_between(min_mdot_arr, loglm, color="lightgray", label='Mdot < Minimun')
#         # ax.grid(which='both')
#         # ax.grid(which='minor', alpha=0.2)
#         #
#         # plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
#         # plot_name = self.output_dir + 'hrd.pdf'
#         # plt.savefig(plot_name)
#         #
#         # plt.show()
#
#     def opacity_check(self, depth, t=None, rho=None, k = None, task = 'rho'):
#         if self.nmdls == 0 and t == None and rho == None and k == None:
#             sys.exit('\t__Error. No t and rho provided for the opacity to be calculated at. |opacity_check|')
#
#         if task == 'rho':
#             if t != None and k != None:
#                 Table_Analyze.plot_k_vs_t = False
#
#                 res = self.tbl_anl.interp_for_single_k(t, t, depth, k)
#
#                 print('For t: {} and k: {}, rho is {} and r is {}'
#                       .format("%.3f" % res[0, 0], "%.3f" % k, "%.3f" % res[1, 0],
#                               "%.3f" % Physics.get_r(res[0, 0], res[1, 0])))
#                 Table_Analyze.plot_k_vs_t = True
#             else:
#                 if self.nmdls != 0:
#                     t_arr = np.zeros(self.nmdls)
#                     k_arr = np.zeros(self.nmdls)
#                     n_rho_arr = np.zeros(self.nmdls)
#                     i_rho_arr = np.zeros(self.nmdls)
#
#                     Table_Analyze.plot_k_vs_t = False
#                     for i in range(self.nmdls):
#                         t_arr[i] = self.mdl[i].t_[-1]
#                         k_arr[i] = self.mdl[i].kappa_[-1]
#                         n_rho_arr[i] = self.mdl[i].rho_[-1]
#                         i_rho_arr[i] = self.tbl_anl.interp_for_single_k(t_arr[i],t_arr[i], depth, k_arr[i])[1,0]
#
#                     Table_Analyze.plot_k_vs_t = True
#
#                     print('\n')
#                     print('|   t   | kappa |  m_rho | in_rho | Error |')
#                     print('|-------|-------|--------|--------|-------|')
#                     for i in range(self.nmdls):
#
#                         print('| {} | {} | {} | {} | {} |'.
#                               format("%.3f" % t_arr[i], "%.3f" % 10**k_arr[i], "%.3f" % n_rho_arr[i], "%.3f" % i_rho_arr[i],
#                                      "%.3f" % np.abs(n_rho_arr[i] - i_rho_arr[i]) ) )
#
#
#         if task == 'kappa' or task == 'k':
#             if t != None and rho != None:
#                 pass # DOR FOR GIVEN T AND RHO (OPAL_interpolate)
#             else:
#                 if self.nmdls != 0:
#                     pass # DO FOR THE SET OF NUMERICAL MODELS
#
#
#
#
#         if t != None and rho == None and k != None:
#             Table_Analyze.plot_k_vs_t = False
#             res = self.tbl_anl.interp_for_single_k(t,t, depth, k)
#
#             print('For t: {} and k: {}, rho is {} and r is {}'
#                   .format("%.3f" % res[0,0], "%.3f" % k, "%.3f" % res[1,0], "%.3f" % Physics.get_r(res[0,0], res[1,0])))
#             Table_Analyze.plot_k_vs_t = True
#
#     def table_of_plot_files(self, flnames, descript):
#
#
#         discr = Read_Observables(descript, '', '')
#
#         #
#         # if len(flnames) != len(descript):
#         #     sys.exit('\t__Error. Number of: flnames {} , descript {} , must be the same |table_of_plot_files|'
#         #              .format(flnames,descript))
#
#         print(discr.obs_par_row(0, str))
#         print(discr.names)
#
#         plfl = []
#         for file in flnames:
#             plfl.append( Read_Plot_file.from_file(file, '') )
#
#
#         # print('| i | File | Var_name | Value | T[-1] | L[-1] | R[-1] | Y_c[-1] ')
#         # for i in range(len(plfl)):
#         #     print('| {} | {} | {} | {} | {} | {} |  |  |'.format(i, flnames[i], descript[i], plfl[i].t_eff[-1], plfl[i].l_[-1]))
#
#         print(discr.names)
#         for i in range(discr.num_stars):
#             print(discr.obs_par_row(i, str),
#                   '| {} | {} |'.format("%.3f" % plfl[i].t_eff[-1], "%.3f" % plfl[i].l_[-1]))
#
#     def rs_l_models(self, num_pol_fit = True):
#         fig = plt.figure()
#         ax = fig.add_subplot(1, 1, 1)
#
#         nums = Treat_Numercials(self.num_files)  # Surface Temp as a x coordinate
#         res = nums.get_x_y_of_all_numericals('sp', 'r', 'l', 'mdot', 'color')
#
#         for i in range(len(res[:, 0])):
#             plt.plot(res[i, 1], res[i, 2], marker='.', color='C' + str(int(res[i, 4])), ls='')  # plot color dots)))
#             ax.annotate(str("%.2f" % res[i, 3]), xy=(res[i, 1], res[i, 2]), textcoords='data')
#
#         if num_pol_fit:
#             fit = np.polyfit(res[:, 1], res[:, 2], 3)  # fit = set of coeddicients (highest first)
#             f = np.poly1d(fit)
#
#             print('Equation:', f.coefficients)
#             fit_x_coord = np.mgrid[(res[1:, 1].min()):(res[1:, 1].max()):100j]
#             plt.plot(fit_x_coord, f(fit_x_coord), '--', color='black', label='Model_fit')
#
#
#
#         name = self.output_dir+'rs_lum_models.pdf'
#         plt.title('MASS LOSS PLOT')
#         plt.xlabel('Sonic Radii')
#         plt.ylabel('Luminosity')
#         plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
#         plt.savefig(name)
#
#         plt.show()