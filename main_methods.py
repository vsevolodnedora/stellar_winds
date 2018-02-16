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
from Err_Const_Math_Phys import Errors
from Err_Const_Math_Phys import Math
from Err_Const_Math_Phys import Physics
from Err_Const_Math_Phys import Constants

from OPAL import Read_Table
from OPAL import Row_Analyze
from OPAL import Table_Analyze
from OPAL import OPAL_Interpol
from OPAL import New_Table

from Read_Obs_Numers import Read_Observables
from Read_Obs_Numers import Read_Plot_file
from Read_Obs_Numers import Read_SM_data_File

from PhysPlots import PhysPlots
#-----------------------------------------------------------------------------------------------------------------------

class Save_Load_tables:
    def __init__(self):
        pass

    @staticmethod
    def save_table(d2arr, opal_used, name, x_name, y_name, z_name, output_dir = '../data/output/'):

        header = np.zeros(len(d2arr)) # will be first row with limtis and
        # header[0] = x1
        # header[1] = x2
        # header[2] = y1
        # header[3] = y2
        # tbl_name = 't_k_rho'
        # op_and_head = np.vstack((header, d2arr))  # arraching the line with limits to the array

        part = opal_used.split('/')[-1]
        full_name = output_dir + name + '_' + part  # dir/t_k_rho_table8.data

        np.savetxt(full_name, d2arr, '%.4f', '  ', '\n',
                   '\nINTERPOLATED OPAL {} TABLE for {} relation'.format(part, name), '',
                   '# {} | {} | {} | {} |'
                   .format(opal_used, x_name, y_name, z_name))

        # np.savetxt(full_name, d2arr, '%.4f', '  ', '\n',
        #            '\nINTERPOLATED OPAL {} TABLE for {} relation'.format(part, name), '',
        #            '# {} | {} {} {} | {} {} {} | {} | {} | {}'
        #            .format(opal_used, x_name, x1, x2, y_name, y1, y2, z_name, n_int, n_out))

    @staticmethod
    def load_table(name, x_name, y_name, z_name, opal_used, dir = '../data/output/'):
        part = opal_used.split('/')[-1]
        full_name = dir + name + '_' + part

        f = open(full_name, 'r').readlines()

        boxes = f[0].split('|')
        f.clear()
        # print(boxes)
        # r_table = boxes[0].split()[-1]
        # r_x_name = boxes[1].split()[0]
        # x1 = boxes[1].split()[1]
        # x2 = boxes[1].split()[2]
        # r_y_name = boxes[2].split()[0]
        # y1 = boxes[2].split()[1]
        # y2 = boxes[2].split()[2]
        # r_z_name = boxes[3].split()[0]
        # n1 = boxes[4].split()[-1]
        # n2 = boxes[5].split()[-1]

        r_table  = boxes[0].split()[-1]
        r_x_name = boxes[1].split()[-1]
        r_y_name = boxes[2].split()[-1]
        r_z_name = boxes[3].split()[-1]

        if r_table != opal_used:
            raise NameError('Read OPAL | {} | not the same is opal_used | {} |'.format(r_table, opal_used))

        if x_name != r_x_name:
            raise NameError('Provided x_name: {} not equal to table x_name: {}'.format(x_name, r_x_name))

        if y_name != r_y_name:
            raise NameError('Provided x_name: {} not equal to table x_name: {}'.format(y_name, r_y_name))

        if z_name != r_z_name:
            raise NameError('Provided x_name: {} not equal to table x_name: {}'.format(z_name, r_z_name))

        # if x1 == 'None':
        #     x1 = None
        # else:
        #     x1 = float(x1)
        #
        # if x2 == 'None:':
        #     x2 = None
        # else:
        #     x2 = float(x2)
        #
        # if y1 == 'None':
        #     y1 = None
        # else:
        #     y1 = float(y1)
        #
        # if y2 == 'None':
        #     y2 = None
        # else:
        #     y2 = float(y2)
        #
        # n1 = int(n1)
        # n2 = int(n2)

        print('\t__OPAL_USED: {}, X is {} | Y is {} | Z is {} '
              .format(r_table, r_x_name, r_y_name, r_z_name))

        print('\t__File | {} | is loaded successfully'.format(full_name))

        # global file_table
        file_table = np.loadtxt(full_name, dtype=float)

        return file_table #[x1, x2, y1, y2, n1, n2]


class Creation:

    def __init__(self, opal_name, t1, t2, n_interp = 1000, load_lim_cases = False, output_dir = '../data/output/', plot_dir = '../data/plots/'):
        self.op_name = opal_name
        self.t1 = t1
        self.t2 = t2
        self.n_inter = n_interp

        self.out_dir = output_dir
        self.plot_dir = plot_dir

        self.opal = OPAL_Interpol(opal_name, n_interp)
        self.tbl_anl = Table_Analyze(opal_name, n_interp, load_lim_cases, output_dir, plot_dir)

    def save_t_rho_k(self, rho1 = None, rho2=None):
        op_cl = OPAL_Interpol(self.op_name, self.n_inter)
        t1, t2, rho1, rho2 = op_cl.check_t_rho_limits(self.t1, self.t2, rho1, rho2)
        op_table = op_cl.interp_opal_table(self.t1, self.t2, rho1, rho2)

        Save_Load_tables.save_table(op_table, self.op_name,'t_rho_k','t','rho','k',self.out_dir)

    def save_t_k_rho(self, llm1=None, llm2=None, n_out = 1000):

        k1, k2 = Physics.get_k1_k2_from_llm1_llm2(self.t1, self.t2, llm1, llm2) # assuming k = 4 pi c G (L/M)

        global t_k_rho
        t_k_rho = self.tbl_anl.treat_tasks_interp_for_t(self.t1, self.t2, n_out, self.n_inter, k1, k2).T

        Save_Load_tables.save_table(t_k_rho, self.op_name, 't_k_rho', 't', 'k', 'rho', self.out_dir)
        print('\t__Note. Table | t_k_rho | has been saved in {}'.format(self.out_dir))
        # self.read_table('t_k_rho', 't', 'k', 'rho', self.op_name)
        # def save_t_llm_vrho(self, llm1=None, llm2=None, n_out = 1000):

    def save_t_llm_vrho(self, l_or_lm_name):
        '''
        Table required: t_k_rho (otherwise - won't work) [Run save_t_k_rho() function ]
        :param l_or_lm_name:
        :return:
        '''

        # 1 load the t_k_rho
        t_k_rho = Save_Load_tables.load_table('t_k_rho', 't', 'k', 'rho', self.op_name)

        k = t_k_rho[0, 1:]
        t = t_k_rho[1:, 0]
        rho2d = t_k_rho[1:, 1:]

        vrho = Physics.get_vrho(t, rho2d.T, 2) # mu = 1.34 by default | rho2d.T is because in OPAL t is Y axis, not X.

        # ----------------------------- SELECT THE Y AXIS -----------------
        if l_or_lm_name == 'l':
            l_lm_arr = Physics.lm_to_l(Physics.logk_loglm(k, True))  # Kappa -> L/M -> L
        else:
            l_lm_arr = Physics.logk_loglm(k, 1)


        l_lm_arr = np.flip(l_lm_arr, 0)  # accounting for if k1 > k2 the l1 < l2 or lm1 < lm2
        vrho     = np.flip(vrho, 0)

        global t_llm_vrho
        t_llm_vrho = Math.combine(t, l_lm_arr, vrho)
        name = 't_'+ l_or_lm_name + '_vrho'

        Save_Load_tables.save_table(t_llm_vrho, self.op_name, name, 't', l_or_lm_name, '_vrho', self.out_dir)

        return t_llm_vrho

        # print(t_llm_vrho.shape)

    def save_t_llm_mdot(self, r_s, l_or_lm, r_s_for_t_l_vrho): # mu assumed constant
        '''
        Table required: l_or_lm (otherwise - won't work) [Run save_t_llm_vrho() function ]

        :param r_s: float, 1darray or 2darray
        :param l_or_lm:
        :param r_s_for_t_l_vrho: 't' - means change rows   of   vrho to get mdot (rho = f(ts))
                                 'l' - means change columns:    vrho to get mdot (rho = f(llm))
                                 vrho- means change rows + cols vrho to get mdot (rho = f(ts, llm))
        :param mu:
        :return:
        '''
        # r_s_for_t_l_vrho = '', 't', 'l', 'lm', 'vrho'


        fname = 't_' + l_or_lm + '_vrho'
        t_llm_vrho = Save_Load_tables.load_table(fname, 't', l_or_lm, '_vrho', self.op_name)
        vrho = t_llm_vrho[1:,1:]

        # -------------------- --------------------- ----------------------------
        c = np.log10(4 * 3.14 * ((Constants.solar_r) ** 2) / Constants.smperyear)
        mdot = np.zeros((vrho.shape))

        if r_s_for_t_l_vrho == '': #------------------------REQUIRED r_s = float
            c2 = c + np.log10(r_s ** 2)
            mdot = vrho + c2

        if r_s_for_t_l_vrho == 't' or r_s_for_t_l_vrho == 'ts': # ---r_s = 1darray
            for i in range(vrho[:,0]):
                mdot[i,:] = vrho[i,:] + c + np.log10(r_s[i] ** 2)

        if r_s_for_t_l_vrho == 'l' or r_s_for_t_l_vrho == 'lm': # ---r_s = 1darray
            for i in range(vrho[0,:]):
                mdot[:,i] = vrho[:,i] + c + np.log10(r_s[i] ** 2)

        if r_s_for_t_l_vrho == 'vrho': #---------------------REQUIRED r_s = 2darray
            cols = len(vrho[0, :])
            rows = len(vrho[:, 0])
            m_dot = np.zeros((rows, cols))

            for i in range(rows):
                for j in range(cols):
                    m_dot[i, j] = vrho[i, j] + c + np.log10(r_s[i, j] ** 2)

        global t_llm_mdot
        t_llm_mdot = Math.combine(t_llm_vrho[0,1:], t_llm_vrho[1:,0], mdot)
        Save_Load_tables.save_table(t_llm_mdot, self.op_name, 't_'+l_or_lm+'_mdot','t', l_or_lm, 'mdot', self.out_dir)



# class Num_Models:
#
#     def __init__(self, smfls = list(), plotfls = list()):
#         self.files = smfls
#         self.n_of_smfiles = len(smfls)
#         self.n_of_plotfls = len(plotfls)
#
#
#         self.mdl = []
#         for i in range(self.n_of_smfiles):
#             self.mdl.append( Read_SM_data_File.from_sm_data_file(self.files[i]) )
#             # print(self.mdl[i].mdot_[-1])
#
#         # self.nmdls = len(self.mdl)
#         print('\t__Note: {} sm.data files has been uploaded.'.format(self.n_of_smfiles))
#
#         self.plts = []
#         for i in range(self.n_of_plotfls):
#             self.plts.append( Read_Plot_file.from_file(plotfls[i]) )
#
#         print('\t__Note: {} .plot1 files has been uploaded.'.format(self.n_of_plotfls))
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
#         for i in range(self.n_of_smfiles):
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
#     def get_ts_llm_of_one_mdel(self, i, i_req, l_or_lm, ts_arr, l_lm_arr, mdot2d_arr, lim_t1_obs = None, lim_t2_obs = None):
#
#         return self.mdl[i].get_ts_llm_cols(self, i_req, l_or_lm, ts_arr, l_lm_arr, mdot2d_arr, lim_t1_obs, lim_t2_obs)
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
class Labels:
    def __init__(self):
        pass

    @staticmethod
    def lbls(v_n):
        #solar
        if v_n == 'l':
            return '$\log(L)$'#(L_{\odot})
        if v_n == 'r':
            return '$R(R_{\odot})$'

        #sonic and general
        if v_n == 'v' or v_n == 'u':
            return 'v (km/s)'
        if v_n == 'rho':
            return '$\log(\rho)$'
        if v_n == 'k' or v_n == 'kappa':
            return '$\kappa$'
        if v_n == 't':
            return 'log(T)'
        if v_n == 'ts':
            return '$\log(T_{s})$'
        if v_n == 'lm':
            return '$\log(L/M)$'
        if v_n == 'mdot':
            return '$\log(\dot{M}$)'



class Observables:

    clumping_used = 4
    cluming_required = 4

    def __init__(self, obs_files):

        self.files = [obs_files]
        self.n_of_fls = len(obs_files)


        self.obs = []
        for i in range(len(self.files)):
            self.obs.append( Read_Observables(self.files[i], self.clumping_used, self.cluming_required))

        if (len(self.files)) > 1 :
            for i in range(1,len(self.files)):
                if not np.array_equal(self.obs[i-1].names, self.obs[i].names):
                    print('\t__Error. Files with observations contain different *names* row')
                    print('\t  {} has: {} \n\t  {} has: {} '
                          .format(self.files[i-1], self.obs[i-1].names, self.files[i], self.obs[i].names))
                    raise NameError('Files with observations contain different *names* row')

    def check_if_var_name_in_list(self, var_name):
        if var_name == 'lm' or var_name == 'ts' or var_name == 'rs': # special case for L/M and sonic temperature
            pass
        else:
            for i in range(self.n_of_fls):
                if var_name not in self.obs:
                    print('\n\t__Error. Variable:  {} is not in the list of names: \n\t  {} \n\t  in file: {}'
                          .format(var_name, self.obs[i].names, self.files[i]))
                    raise  NameError('Only lm, l, and rs varnames are available. {} is not listed.'.format(var_name))

    def get_x_y_of_all_observables(self, x_name, y_name, var_for_label,
                                   ts_arr = np.empty(1,), l_lm_arr= np.empty(1,), m_dot= np.empty(1,),
                                   lim_t1_obs = None, lim_t2_obs = None):
        '''
        RETURN:  np.array( [plotted_stars, plotted_labels] )  [0][:,0] - nums of all plotted stars
                                                              [0][:,1] - x - coord.
                                                              [0][:,2] - y - coord
                                                              [0][:,3] - ints from 0 to 9, uniqe for uniqe 'var_for_label'
                                                              [1][:,0] - nums of selected stars for labels
                                                              [1][:,1] - x - coord
                                                              [1][:,2] - y - coord
                                                              [1][:,3] - ints from 0 to 9
        To get index in the [0] arr of the element in [1] Use: int( np.where( res[0][:, 0]==res[1][j, 0] )[0] )

        Warning! If there are more unique str(var_for_label), PROGRAM BRAKES
        :param x_name:
        :param y_name:
        :param var_for_label:
        :param ts_arr:
        :param l_lm_arr:
        :param m_dot:
        :param lim_t1_obs:
        :param lim_t2_obs:
        :return:
        '''
        self.check_if_var_name_in_list(x_name)
        self.check_if_var_name_in_list(y_name)
        self.check_if_var_name_in_list(var_for_label)

        s = 0

        leble = []
        plotted_stars = np.array([0., 0., 0., 0.])
        plotted_labels= np.array([0., 0., 0., 0. ])

        # if self.obs != None:  # plot observed stars
        ''' Read the observables file and get the necessary values'''
        ts_ = []
        y_coord_ = []

        import re  # for searching the number in 'WN7-e' string, to plot them different colour
        for i in range(self.obs[s].num_stars):
            star_x_coord = []
            star_y_coord = []

            # ---------------------------------------Y-------------------------
            if y_name == 'lm':
                star_y_coord = [ Physics.loglm(self.obs[s].obs_par('l', float)[i],
                                             self.obs[s].obs_par('m', float)[i]) ]
            else:
                star_y_coord = [ self.obs[s].obs_par(y_name, float)[i] ]


            # ---------------------------------------X-------------------------
            if x_name == 'ts' or x_name == 'rs':
                if not ts_arr.any() or not l_lm_arr.any() or not m_dot.any():
                    print('\t__Error. For ts to be evaluated for a star : *ts_arr, l_lm_arr, m_dot* to be provided')
                    raise ValueError

                x_y_coord = Physics.lm_mdot_obs_to_ts_lm(ts_arr, l_lm_arr, m_dot, star_y_coord[0],
                                                         self.obs[s].obs_par('mdot', float)[i],
                                                         i, lim_t1_obs, lim_t2_obs)
                if x_y_coord.any():
                    ts_ = np.append(ts_, x_y_coord[1, :])  # FOR linear fit
                    y_coord_ = np.append(y_coord_, x_y_coord[0, :])
                    star_x_coord =  x_y_coord[1, :]
                    star_y_coord =  x_y_coord[0, :]  # If the X coord is Ts the Y coord is overritten.

            else:
                star_x_coord = [ self.obs[s].obs_par(x_name, float)[i] ]

            if x_name == 'lm':
                star_x_coord = [ Physics.loglm(self.obs[s].obs_par('l', float)[i],
                                             self.obs[s].obs_par('m', float)[i]) ]





            star_x_coord = np.array(star_x_coord)
            star_y_coord = np.array(star_y_coord)
            if len(star_x_coord) == len(star_y_coord) and star_x_coord.any() :

                se = re.search(r"\d+(\.\d+)?", self.obs[s].obs_par('type', str)[i])  # this is searching for the niumber
                #             color = 'C' + str(int(s.group(0)))  # making a colour our of C1 - C9 range

                for j in range(len(star_x_coord)):  # plot every solution in the degenerate set of solutions

                    row = self.obs[s].table[i]  # to get the 0th element, which is alwas the star index

                    cur_type = int(se.group(0))
                    if cur_type not in leble:  # plotting the label for unique class of stars
                        leble.append( cur_type )

                        plotted_labels = np.vstack((plotted_labels, np.array((int(row[0:3]),
                                                                              star_x_coord[j],
                                                                              star_y_coord[j],
                                                                              cur_type ))))

                    plotted_stars = np.vstack((plotted_stars, np.array((int(row[0:3]),
                                                                        star_x_coord[j],
                                                                        star_y_coord[j],
                                                                        cur_type ))))  # for further printing
            # print(self.obs[s].table)

        # -----------------------------------------------LINEAR FIT TO THE DATA-------------------------------------
        # ts_grid_y_grid = Math.line_fit(ts_, y_coord_)
        # plt.plot(ts_grid_y_grid[0, :], ts_grid_y_grid[1, :], '-.', color='blue')
        # np.delete(plotted_stars,1,0)
        plotted_stars  = np.delete(plotted_stars, 0, 0) # removing [0,0,0,] row
        plotted_labels = np.delete(plotted_labels, 0, 0)

        if plotted_stars.any():
            print('\n| Plotted Stras from Observ |')
            print('|  i  | {} | {}  | col |'.format(x_name, y_name))
            print('|-----|---------|----------|')
            for i in range(len(plotted_stars[:, 0])):
                print('| {} |  {} \t| {} | {} |'.format("%3.f" % plotted_stars[i, 0], "%.2f" % plotted_stars[i, 1],
                                                 "%.2f" % plotted_stars[i, 2], plotted_stars[i, 3]))

        if plotted_labels.any():
            print('\n| Plotted Labels from Observ |')
            print('|  i  | {} | {}  | col |'.format(x_name, y_name))
            print('|-----|-----------|---------|')
            for i in range(len(plotted_labels[:, 0])):
                print('| {} |  {} \t| {} | {} |'.format("%3.f" % plotted_labels[i, 0], "%.2f" % plotted_labels[i, 1],
                                                 "%.2f" % plotted_labels[i, 2], plotted_labels[i, 3]))

        return( np.array( [plotted_stars, plotted_labels] ) )


class Combine:
    output_dir = '../data/output/'
    plot_dir = '../data/plots/'

    def __init__(self, smfls = list(), plotfls = list(), obs_files = list(), opal_used = None):
        self.num_files = smfls
        self.plt_pltfiles = plotfls
        self.obs_files = obs_files

        self.mdl = []
        for file in smfls:
            self.mdl.append(Read_SM_data_File.from_sm_data_file(file))


        # self.nums = Num_Models(smfls, plotfls)
        self.obs = Read_Observables(obs_files)


        self.opal_used = opal_used

    #--METHODS THAT DO NOT REQUIRE OPAL TABLES
    def xy_profile(self, v_n1, v_n2, var_for_label1, var_for_label2, sonic = True):

        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        tlt = v_n2 + '(' + v_n1 + ') profile'
        plt.title(tlt, loc='left')

        for i in range(len(self.num_files)):

            x =      self.mdl[i].get_col(v_n1)
            y      = self.mdl[i].get_col(v_n2)          # simpler syntaxis
            label1 = self.mdl[i].get_col(var_for_label1)[-1]
            label2 = self.mdl[i].get_col(var_for_label2)[-1]

            lbl = '{}:{} , {}:{}'.format(var_for_label1,'%.2f' % label1,var_for_label2,'%.2f' % label2)
            ax1.plot(x,  y,  '.',   color='C' + str(Math.get_0_to_max([i], 9)[i]), label=lbl)
            ax1.plot(x[-1], y[-1], 'x',   color='C' + str(Math.get_0_to_max([i], 9)[i]))

            ax1.annotate(str('%.2e' % 10**self.mdl[i].get_col('mdot')[-1]), xy=(x[-1], y[-1]), textcoords='data')


            if sonic and v_n2 == 'u':
                u_s = self.mdl[i].get_sonic_u()
                ax1.plot(x, u_s, '-', color='black')

                xc, yc = Math.interpolated_intercept(x,y, u_s)
                # print('Sonic r: {} | Sonic u: {} | {}'.format( np.float(xc),  np.float(yc), len(xc)))
                plt.plot(xc, yc, 'X', color='red', label='Intersection')

        ax1.set_xlabel(Labels.lbls(v_n1))
        ax1.set_ylabel(Labels.lbls(v_n2))

        ax1.grid(which='both')
        ax1.grid(which='minor', alpha=0.2)
        ax1.grid(which='major', alpha=0.2)

        ax1.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
        plot_name = self.plot_dir + v_n1 + '_vs_' + v_n2 + '_profile.pdf'
        plt.savefig(plot_name)
        plt.show()

    def xyy_profile(self, v_n1, v_n2, v_n3, var_for_label1, var_for_label2, var_for_label3, edd_kappa = True):

        # for i in range(self.nmdls):
        #     x = self.mdl[i].get_col(v_n1)
        #     y = self.mdl[i].get_col(v_n2)
        #     color = 'C' + str(i)
        #
        #     lbl = 'M:' + str('%.2f' % self.mdl[i].get_col('xm')[-1]) + ' L:' + \
        #            str('%.2f' % self.mdl[i].get_col('l')[-1]) + ' Mdot:' + \
        #            str('%.2f' % self.mdl[i].get_col('mdot')[-1])
        #     ax1.plot(x, y, '-', color=color, label=lbl)
        #     ax1.plot(x[-1], y[-1], 'x', color=color)

        fig, ax1 = plt.subplots()
        tlt = v_n2 + ' , '+ v_n3 + ' = f(' + v_n1 + ') profile'
        plt.title(tlt)

        ax1.set_xlabel(v_n1)
        # Make the y-axis label, ticks and tick labels match the line color.
        ax1.set_ylabel(v_n2, color='b')
        ax1.tick_params('y', colors='b')
        ax1.grid()
        ax2 = ax1.twinx()

        for i in range(len(self.num_files)):

            xyy2  = self.mdl[i].get_set_of_cols([v_n1, v_n2, v_n3])
            lbl1 =  self.mdl[i].get_col(var_for_label1)[-1]
            lbl2 =  self.mdl[i].get_col(var_for_label2)[-1]
            lbl3 =  self.mdl[i].get_col(var_for_label3)[-1]

            color = 'C' + str(Math.get_0_to_max([i], 9)[i])
            lbl = '{}:{} , {}:{} , {}:{}'.format(var_for_label1, '%.2f' % lbl1, var_for_label2, '%.2f' % lbl2,
                                                 var_for_label3, '%.2f' % lbl3)

            ax1.plot(xyy2[:, 0],  xyy2[:, 1],  '-', color=color, label=lbl)
            ax1.plot(xyy2[-1, 0], xyy2[-1, 1], 'x', color=color)
            ax1.annotate(str('%.2f' % lbl1), xy=(xyy2[-1, 0], xyy2[-1, 1]), textcoords='data')

            if edd_kappa and v_n3 == 'kappa':
                k_edd = Physics.edd_opacity(self.mdl[i].get_col('xm')[-1],
                                            self.mdl[i].get_col('l')[-1])
                ax2.plot(ax1.get_xlim(), [k_edd, k_edd], color='black', label='Model: {}, k_edd: {}'.format(i, k_edd))

            ax2.plot(xyy2[:, 0],  xyy2[:, 2], '--', color=color)
            ax2.plot(xyy2[-1, 0], xyy2[-1, 2], 'o', color=color)

        ax2.set_ylabel(v_n3, color='r')
        ax2.tick_params('y', colors='r')

        plt.title(tlt, loc='left')
        fig.tight_layout()
        ax1.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
        plot_name = self.plot_dir + v_n1 + '_' + v_n2 + '_' + v_n3 + '_profile.pdf'
        plt.savefig(plot_name)
        plt.show()

    def hrd(self, plot_file_names):

        fig, ax = plt.subplots(1, 1)

        plt.title('HRD')
        plt.xlabel('log(T_eff)')
        plt.ylabel('log(L)')

        # plt.xlim(t1, t2)
        ax.grid(which='major', alpha=0.2)
        plt.legend(bbox_to_anchor=(1, 1), loc='upper right', ncol=1)

        # res = self.obs.get_x_y_of_all_observables('t', 'l', 'type')
        #
        # for i in range(len(res[0][:, 1])):
        #     ax.annotate(int(res[0][i, 0]), xy=(res[0][i, 1], res[0][i, 2]), textcoords='data')  # plot numbers of stars
        #     plt.plot(res[0][i, 1], res[0][i, 2], marker='^', color='C' + str(int(res[0][i, 3])),
        #              ls='')  # plot color dots)))
        #
        # for j in range(len(res[1][:, 0])):
        #     plt.plot(res[1][j, 1], res[1][j, 2], marker='^', color='C' + str(int(res[1][j, 3])), ls='',
        #              label='WN' + str(int(res[1][j, 3])))

        ind_arr = []
        for j in range(len(plot_file_names)):
            ind_arr.append(j)
            col_num = Math.get_0_to_max(ind_arr, 9)
            plfl = Read_Plot_file.from_file(plot_file_names[j])

            mod_x = plfl.t_eff
            mod_y = plfl.l_
            color = 'C' + str(col_num[j])

            fname = plot_file_names[j].split('/')[-2] + plot_file_names[j].split('/')[-1]# get the last folder in which the .plot1 is

            plt.plot(mod_x, mod_y, '-', color=color,
                     label='{}, m:({}->{})'.format(fname, "%.1f" % plfl.m_[0], "%.1f" % plfl.m_[-1]) )
                     # str("%.2f" % plfl.m_[0]) + ' to ' + str("%.2f" % plfl.m_[-1]) + ' solar mass')

            for i in range(10):
                ind = Math.find_nearest_index(plfl.y_c, (i / 10))
                # print(plfl.y_c[i], (i/10))
                x_p = mod_x[ind]
                y_p = mod_y[ind]
                plt.plot(x_p, y_p, '.', color='red')
                ax.annotate("%.2f" % plfl.y_c[ind], xy=(x_p, y_p), textcoords='data')

        ax.grid(which='both')
        ax.grid(which='minor', alpha=0.2)

        plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
        plot_name = self.output_dir + 'hrd.pdf'
        plt.savefig(plot_name)

        plt.show()

    def xy_last_points(self, v_n1, v_n2, v_lbl1, num_pol_fit = True):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        # nums = Treat_Numercials(self.num_files)  # Surface Temp as a x coordinate
        # res = nums.get_x_y_of_all_numericals('sp', 'r', 'l', 'mdot', 'color')
        x = []
        y = []
        for i in range(len(self.num_files)):
            x = np.append(x, self.mdl[i].get_cond_value(v_n1, 'sp') )
            y = np.append(y, self.mdl[i].get_cond_value(v_n2, 'sp') )

            lbl1 = self.mdl[i].get_cond_value(v_lbl1, 'sp')
            # print(x, y, lbl1)

            plt.plot(x[i], y[i], marker='.', color='C' + str(Math.get_0_to_max([i],9)[i]), ls='', label='{}:{} , {}:{} , {}:{}'
                     .format(v_n1, "%.2f" % x[i], v_n2, "%.2f" % y[i], v_lbl1, "%.2f" % lbl1))  # plot color dots)))
            ax.annotate(str("%.2f" % lbl1), xy=(x[i], y[i]), textcoords='data')

        if num_pol_fit:
            fit = np.polyfit(x, y, 3)  # fit = set of coeddicients (highest first)
            f = np.poly1d(fit)

            # print('Equation:', f.coefficients)
            fit_x_coord = np.mgrid[(x.min()):(x.max()):100j]
            plt.plot(fit_x_coord, f(fit_x_coord), '--', color='black', label='Model_fit')


        name = self.output_dir+'{}_{}_dependance.pdf'.format(v_n2,v_n1)
        plt.title('{} = f({}) plot'.format(v_n2,v_n1))
        plt.xlabel(v_n1)
        plt.ylabel(v_n2)
        plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
        plt.savefig(name)

        plt.show()

    #--METHODS THAT DO REQUIRE OPAL TABLES
    def plot_t_rho_kappa(self, var_for_label1, var_for_label2,  n_int_edd = 1000, plot_edd = True):
        # self.int_edd = self.tbl_anlom_OPAL_table(self.op_name, 1, n_int, load_lim_cases)

        # t_k_rho = self.opal.interp_opal_table(t1, t2, rho1, rho2)

        t_rho_k = Save_Load_tables.load_table('t_rho_k','t','rho','k',self.opal_used, self.output_dir)


        t      = t_rho_k[0, 1:]  # x
        rho    = t_rho_k[1:, 0]  # y
        kappa  = t_rho_k[1:, 1:] # z

        plt.figure()
        levels = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        pl.xlim(t.min(), t.max())
        pl.ylim(rho.min(), rho.max())
        contour_filled = plt.contourf(t, rho, 10 ** (kappa), levels, cmap=plt.get_cmap('RdYlBu_r'))
        plt.colorbar(contour_filled)
        contour = plt.contour(t, rho, 10 ** (kappa), levels, colors='k')
        plt.clabel(contour, colors='k', fmt='%2.1f', fontsize=12)
        plt.title('OPACITY PLOT')
        plt.xlabel('Log(T)')
        plt.ylabel('log(rho)')

        # ------------------------EDDINGTON-----------------------------------
        Table_Analyze.plot_k_vs_t = False  # there is no need to plot just one kappa in the range of availability

        if plot_edd:  # n_model_for_edd_k.any():
            clas_table_anal = Table_Analyze(self.opal_used, 1000, False, self.output_dir, self.plot_dir)

            for i in range(len(self.num_files)):  # self.nmdls
                mdl_m = self.mdl[i].get_value('xm', 'sp')
                mdl_l = self.mdl[i].get_value('l',  'sp')

                k_edd = Physics.edd_opacity(mdl_m, mdl_l)

                n_model_for_edd_k = clas_table_anal.interp_for_single_k(t.min(), t.max(), n_int_edd, k_edd)
                x = n_model_for_edd_k[0, :]
                y = n_model_for_edd_k[1, :]
                color = 'black'
                lbl = 'Model:{}, k_edd:{}'.format(i, '%.2f' % 10 ** k_edd)
                plt.plot(x, y, '-.', color=color, label=lbl)
                plt.plot(x[-1], y[-1], 'x', color=color)

        Table_Analyze.plot_k_vs_t = True
        # ----------------------DENSITY----------------------------------------

        for i in range(len(self.num_files)):
            res = self.mdl[i].get_set_of_cols(['t', 'rho', var_for_label1, var_for_label2])

            lbl = '{} , {}:{} , {}:{}'.format(i, var_for_label1, '%.2f' % res[0, 2], var_for_label2, '%.2f' % res[0, 3])
            plt.plot(res[:, 0], res[:, 1], '-', color='C' + str(Math.get_0_to_max([i], 9)[i]), label=lbl)
            plt.plot(res[-1, 0], res[-1, 1], 'x', color='C' + str(Math.get_0_to_max([i], 9)[i]))
            plt.annotate(str('%.2f' % res[0, 2]), xy=(res[-1, 0], res[-1, 1]), textcoords='data')

        plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
        name = self.plot_dir + 't_rho_kappa.pdf'
        plt.savefig(name)
        plt.show()

    def plot_t_mdot_lm(self, v_lbl, r_s = 1., lim_t1_mdl = 5.2, lim_t2_mdl = None):

        t_rho_k = Save_Load_tables.load_table('t_rho_k', 't', 'rho', 'k', self.opal_used,self.output_dir)

        t_s= t_rho_k[0, 1:]  # x
        rho= t_rho_k[1:, 0]  # y
        k  = t_rho_k[1:, 1:]  # z

        vrho = Physics.get_vrho(t_s, rho, 1, 1.34)    # assuming mu = constant
        mdot = Physics.vrho_mdot(vrho, r_s, '')       # assuming rs = constant

        lm_arr = Physics.logk_loglm(k, 2)

        #-----------------------------------

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        pl.xlim(t_s.min(), t_s.max())
        pl.ylim(mdot.min(), mdot.max())
        levels = [3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 5.0, 5.2]
        contour_filled = plt.contourf(t_s, mdot, lm_arr, levels, cmap=plt.get_cmap('RdYlBu_r'))
        plt.colorbar(contour_filled)
        contour = plt.contour(t_s, mdot, lm_arr, levels, colors='k')
        plt.clabel(contour, colors='k', fmt='%2.1f', fontsize=12)
        plt.title('L/M PLOT')
        plt.xlabel('Log(t_s)')
        plt.ylabel('log(M_dot)')

        for i in range(len(self.num_files)):
            ts_llm_mdot = self.mdl[i].get_xyz_from_yz(i, 'sp', 'mdot', 'lm', t_s,mdot, lm_arr, lim_t1_mdl, lim_t2_mdl)
            lbl1 = self.mdl[i].get_cond_val(v_lbl, 'sp')

            if ts_llm_mdot.any():
                lbl = 'i:{}, lm:{}, {}:{}'.format(i, "%.2f" % ts_llm_mdot[2, -1], v_lbl, "%.2f" % lbl1)
                plt.plot(ts_llm_mdot[0, :], ts_llm_mdot[1,:], marker='x', color='C' + str(Math.get_0_to_max([i],9)[i]), ls='', label=lbl)  # plot color dots)))
                ax.annotate(str("%.2f" % ts_llm_mdot[2, -1]), xy=(ts_llm_mdot[0, -1], ts_llm_mdot[1,-1]), textcoords='data')

        plt.legend(bbox_to_anchor=(1, 0), loc='lower right', ncol=1)
        name = self.plot_dir + 't_mdot_lm_plot.pdf'
        plt.savefig(name)
        plt.show()

    @staticmethod
    def empirical_l_r(x):
        '''
        X:[5.141 , 5.722], Y:[0.923 , 1.906]
        :param x:
        :return:
        '''

        return -859.098 + 489.056*x - 92.827*x**2 + 5.882*x**3


    def plot_t_l_mdot(self, l_or_lm, r_s_, num_var_plot = 'xm', lim_t1 = None, lim_t2 = None):

        # ---------------------LOADING-INTERPOLATED-TABLE---------------------------

        t_k_rho = Save_Load_tables.load_table('t_k_rho', 't', 'k', 'rho', self.opal_used, self.output_dir)

        #---------------------Getting KAPPA[], T[], RHO2D[]-------------------------

        kap   = t_k_rho[1:, 0]
        t   =   t_k_rho[0, 1:]
        rho2d = t_k_rho[1:, 1:]

        if l_or_lm == 'l':
            l_lm_arr  = Physics.lm_to_l( Physics.logk_loglm(kap, True) ) # Kappa -> L/M -> L
        else:
            l_lm_arr = Physics.logk_loglm(kap, 1)

        l_limits = [5.141, 5.722]
        ind_1 = Math.find_nearest_index(l_lm_arr, l_limits[0]) + 1
        ind_2 = Math.find_nearest_index(l_lm_arr, l_limits[1]) - 1
        print(ind_2, ind_1, '->', l_lm_arr[ind_2], l_lm_arr[ind_1], '->', len(l_lm_arr[ind_2:ind_1]))

        l_lm_arr = l_lm_arr[ind_2:ind_1]
        rho2d = rho2d[ind_2:ind_1 , :]

        rs = Combine.empirical_l_r(l_lm_arr)
        print('rs(l=5.2):{} , rs(l=5.7):{}'.format(Combine.empirical_l_r(5.2), Combine.empirical_l_r(5.7)))
        # rs = l_lm_arr.fill(1.)


        vrho = Physics.get_vrho(t,rho2d,2, 1.34)          # mu = 1.34 everywhere
        m_dot = Physics.vrho_mdot(vrho, rs, 't')      # r_s = constant

        mins = Math.get_mins_in_every_row(t, l_lm_arr, m_dot, 5000, 5.1, 5.3)

        print('\t__Note: PLOT: x: {}, y: {}, z: {} shapes.'.format(t.shape, l_lm_arr.shape, m_dot.shape))

        #-------------------------------------------POLT-Ts-LM-MODT-COUTUR------------------------------------

        name = self.plot_dir + 'rs_lm_minMdot_plot.pdf'

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.xlim(t.min(), t.max())
        plt.ylim(l_lm_arr.min(), l_lm_arr.max())
        plt.ylabel(Labels.lbls(l_or_lm))
        plt.xlabel(Labels.lbls('ts'))
        levels = [-7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2]
        contour_filled = plt.contourf(t, l_lm_arr, m_dot, levels, cmap=plt.get_cmap('RdYlBu_r'))
        plt.colorbar(contour_filled, label=Labels.lbls('mdot'))
        contour = plt.contour(t, l_lm_arr, m_dot, levels, colors='k')
        plt.clabel(contour, colors='k', fmt='%2.1f', fontsize=12)
        plt.title('SONIC HR DIAGRAM')


        # plt.ylabel(l_or_lm)
        plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
        plt.savefig(name)

        #--------------------------------------------------PLOT-MINS----------------------------------------------------

        plt.plot(mins[0, :], mins[1, :], '-.', color='red', label='min_Mdot (rs: {} )'.format(r_s_))

        #-----------------------------------------------PLOT-OBSERVABLES------------------------------------------------
        classes = []
        classes.append('dum')
        x = []
        y = []
        # classes.append(self.obs.get_star_class(self.obs.stars_n[0]))
        for star_n in self.obs.stars_n:
            xyz = self.obs.get_xyz_from_yz(star_n, l_or_lm, 'mdot', t, l_lm_arr, m_dot, lim_t1, lim_t2)
            if xyz.any():
                x = np.append(x, xyz[0, 0])
                y = np.append(y, xyz[1, 0])
                for i in range(len(xyz[0,:])):
                    plt.plot(xyz[0, i], xyz[1, i], marker=self.obs.get_clss_marker(star_n), markersize='9', color=self.obs.get_class_color(star_n), ls='')  # plot color dots)))
                    ax.annotate(int(star_n), xy=(xyz[0,i], xyz[1,i]),
                                textcoords='data')  # plot numbers of stars
                    if self.obs.get_star_class(star_n) not in classes:
                        plt.plot(xyz[0, i], xyz[1, i], marker=self.obs.get_clss_marker(star_n), markersize='9', color=self.obs.get_class_color(star_n), ls='', label='{}'.format(self.obs.get_star_class(star_n)))  # plot color dots)))
                        classes.append(self.obs.get_star_class(star_n))

        fit = np.polyfit(x, y, 1)  # fit = set of coeddicients (highest first)
        f = np.poly1d(fit)
        fit_x_coord = np.mgrid[(x.min()-1):(x.max()+1):1000j]
        plt.plot(fit_x_coord, f(fit_x_coord), '-.', color='blue')

        # plt.ylim(y.min(),y.max())


        #--------------------------------------------------_NUMERICALS--------------------------------------------------
        for i in range(len(self.num_files)):
            ts_llm_mdot = self.mdl[i].get_xyz_from_yz(i, 'sp', l_or_lm, 'mdot', t , l_lm_arr, m_dot, lim_t1, lim_t2)
            lbl1 = self.mdl[i].get_cond_value(num_var_plot, 'sp')

            if ts_llm_mdot.any():
                lbl = 'i:{}, lm:{}, {}:{}'.format(i, "%.2f" % ts_llm_mdot[2, -1], num_var_plot, "%.2f" % lbl1)
                plt.plot(ts_llm_mdot[0, :], ts_llm_mdot[1,:], marker='x', color='C' + str(Math.get_0_to_max([i],9)[i]), ls='', label=lbl)  # plot color dots)))
                ax.annotate(str("%.2f" % ts_llm_mdot[2, -1]), xy=(ts_llm_mdot[0, -1], ts_llm_mdot[1,-1]), textcoords='data')

        for i in range(len(self.num_files)):
            x_coord = self.mdl[i].get_cond_value('t', 'sp')
            y_coord = self.mdl[i].get_cond_value(l_or_lm, 'sp')
            lbl1 = self.mdl[i].get_cond_value(num_var_plot, 'sp')
            lbl2 = self.mdl[i].get_cond_value('He4', 'core')

            lbl = 'i:{}, Yc:{}, {}:{}'.format(i, "%.2f" % lbl2, num_var_plot, "%.2f" % lbl1)
            plt.plot(x_coord, y_coord, marker='X', color='C' + str(Math.get_0_to_max([i], 9)[i]),
                     ls='', label=lbl)  # plot color dots)))
            ax.annotate(str(int(i)), xy=(x_coord, y_coord),
                        textcoords='data')


        plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
        plt.gca().invert_xaxis()
        plt.savefig(name)
        plt.show()


    def min_mdot(self, l_or_lm, r_s, num_var_plot = 'xm', lim_t1 = None, lim_t2 = None):
        # ---------------------LOADING-INTERPOLATED-TABLE---------------------------

        t_k_rho = Save_Load_tables.load_table('t_k_rho', 't', 'k', 'rho', self.opal_used, self.output_dir)

        # ---------------------Getting KAPPA[], T[], RHO2D[]-------------------------

        kap = t_k_rho[1:, 0]
        t = t_k_rho[0, 1:]
        rho2d = t_k_rho[1:, 1:]

        if l_or_lm == 'l':
            l_lm_arr = Physics.lm_to_l(Physics.logk_loglm(kap, True))  # Kappa -> L/M -> L
        else:
            l_lm_arr = Physics.logk_loglm(kap, 1)

        l_limits = [5.141, 5.722]
        ind_1 = Math.find_nearest_index(l_lm_arr, l_limits[0]) + 1
        ind_2 = Math.find_nearest_index(l_lm_arr, l_limits[1]) - 1
        print(ind_2, ind_1, '->', l_lm_arr[ind_2], l_lm_arr[ind_1], '->', len(l_lm_arr[ind_2:ind_1]))

        l_lm_arr = l_lm_arr[ind_2:ind_1]
        rho2d = rho2d[ind_2:ind_1, :]

        rs = Combine.empirical_l_r(l_lm_arr)
        print('rs(l=5.2):{} , rs(l=5.7):{}'.format(Combine.empirical_l_r(5.2), Combine.empirical_l_r(5.7)))
        # rs = l_lm_arr.fill(1.)

        vrho = Physics.get_vrho(t, rho2d, 2)  # mu = 1.34 everywhere
        m_dot = Physics.vrho_mdot(vrho, rs, 't')  # r_s = constant

        mins = Math.get_mins_in_every_row(t, l_lm_arr, m_dot, 5000, 5.0, None)

        print('\t__Note: PLOT: x: {}, y: {}, z: {} shapes.'.format(t.shape, l_lm_arr.shape, m_dot.shape))

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        #----------------------------------------------PLOT MIN MDOT----------------------------------------------------

        plt.plot(mins[2,:], mins[1,:], '-', color='black')

        # -----------------------------------------------PLOT-OBSERVABLES------------------------------------------------
        classes = []
        classes.append('dum')
        x = []
        y = []
        # classes.append(self.obs.get_star_class(self.obs.stars_n[0]))
        for star_n in self.obs.stars_n:
            i=-1
            x = np.append(x, self.obs.get_num_par('mdot',  star_n))
            y = np.append(y, self.obs.get_num_par(l_or_lm, star_n))

            plt.plot(x[i], y[i], marker=self.obs.get_clss_marker(star_n), markersize='9',
                     color=self.obs.get_class_color(star_n), ls='')  # plot color dots)))
            ax.annotate(int(star_n), xy=(x[i], y[i]),
                        textcoords='data')  # plot numbers of stars
            if self.obs.get_star_class(star_n) not in classes:
                plt.plot(x[i], y[i], marker=self.obs.get_clss_marker(star_n), markersize='9',
                         color=self.obs.get_class_color(star_n), ls='',
                         label='{}'.format(self.obs.get_star_class(star_n)))  # plot color dots)))
                classes.append(self.obs.get_star_class(star_n))

        fit = np.polyfit(x, y, 1)  # fit = set of coeddicients (highest first)
        f = np.poly1d(fit)
        fit_x_coord = np.mgrid[(x.min() - 1):(x.max() + 1):1000j]
        plt.plot(fit_x_coord, f(fit_x_coord), '-.', color='blue')

        # --------------------------------------------------_NUMERICALS--------------------------------------------------

        for i in range(len(self.num_files)):
            x_coord = self.mdl[i].get_cond_value('mdot', 'sp')
            y_coord = self.mdl[i].get_cond_value(l_or_lm, 'sp')
            lbl1 = self.mdl[i].get_cond_value(num_var_plot, 'sp')
            lbl2 = self.mdl[i].get_cond_value('He4', 'core')

            lbl = 'i:{}, Yc:{}, {}:{}'.format(i, "%.2f" % lbl2, num_var_plot, "%.2f" % lbl1)
            plt.plot(x_coord, y_coord, marker='X', color='C' + str(Math.get_0_to_max([i], 9)[i]),
                     ls='', label=lbl)  # plot color dots)))
            ax.annotate(str(int(i)), xy=(x_coord, y_coord),
                        textcoords='data')



        # plt.ylim(y.min(),y.max())

        # plt.xlim(-6.0, mins[2,:].max())

        plt.ylabel(Labels.lbls(l_or_lm))
        plt.xlabel(Labels.lbls('mdot'))
        ax.grid(which='major', alpha=0.2)
        plt.legend(bbox_to_anchor=(1, 1), loc='upper right', ncol=1)

        ax.grid(which='both')
        ax.grid(which='minor', alpha=0.2)
        ax.fill_between(mins[2,:], mins[1,:], color="lightgray")
        plt.legend(bbox_to_anchor=(1, 0), loc='lower right', ncol=1)
        plot_name = self.plot_dir + 'minMdot_l.pdf'
        plt.savefig(plot_name)
        plt.show()

#================================================3D=====================================================================
#
#
#================================================3D=====================================================================


from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation

class Annotation3D(Annotation):
    '''Annotate the point xyz with text s'''

    def __init__(self, s, xyz, *args, **kwargs):
        Annotation.__init__(self,s, xy=(0,0), *args, **kwargs)
        self._verts3d = xyz

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.xy=(xs,ys)
        Annotation.draw(self, renderer)



class TEST:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        pass

    def xy_last_points(self, v_n1, v_n2, v_lbl1, v_lbl_cond, list_of_list_of_smfiles = list(), num_pol_fit = True):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        for j in range(len(list_of_list_of_smfiles)):

            x = []
            y = []
            for i in range(len(list_of_list_of_smfiles[j])):
                sm1 = Read_SM_data_File.from_sm_data_file(list_of_list_of_smfiles[j][i])
                x = np.append(x, sm1.get_cond_value(v_n1, 'sp') )
                y = np.append(y, sm1.get_cond_value(v_n2, 'sp') )

                lbl1 = sm1.get_cond_value(v_lbl1, v_lbl_cond)
                # print(x, y, lbl1)
                #     color='C' + str(Math.get_0_to_max([i],9)[i])
                plt.plot(x[i], y[i], marker='.', color='C' + str(j), ls='', label='{}:{} , {}:{} , {}:{}'
                         .format(v_n1, "%.2f" % x[i], v_n2, "%.2f" % y[i], v_lbl1, "%.2f" % lbl1))  # plot color dots)))
                ax.annotate(str("%.2f" % lbl1), xy=(x[i], y[i]), textcoords='data')

            if num_pol_fit:
                def fitFunc(t, a, b, c, d, e):
                        # return c * np.exp(-b * t ** a) + d
                        return a + t**b + t**c + t**d + e ** t    #
                        # return a + b/t + c/t**2 + d/t**3

                def fitting():
                    from scipy.optimize import curve_fit

                    plt.plot(x, y, 'b.', label='data')
                    popt, pcov = curve_fit(fitFunc, x, y)
                    print(popt)

                    # plt.plot(x, fitFunc(x, *popt), 'r-', label = '' % tuple(popt))
                    x_new = np.mgrid[x[0]:x[-1]:100j]

                    plt.plot(x_new, fitFunc(x_new, popt[0], popt[1], popt[2], popt[3], popt[4]), 'r-')

                # fitting() # - Sophisticated fitting.


                fit = np.polyfit(x, y, 3)  # fit = set of coeddicients (highest first)
                f = np.poly1d(fit)

                # print('Equation:', f.coefficients)
                fit_x_coord = np.mgrid[(x.min()):(x.max()):100j]
                lbl = '{} + {}*x + {}*x**2 + {}*x**3'.format("%.3f" % f.coefficients[3], "%.3f" % f.coefficients[2], "%.3f" % f.coefficients[1], "%.3f" % f.coefficients[0])

                plt.plot(fit_x_coord, f(fit_x_coord), '--', color='black', label=lbl)
                print('smfls1:', lbl)
                print('X:[{} , {}], Y:[{} , {}]'.format("%.3f" % x.min(),  "%.3f" % x.max(), "%.3f" % y.min(), "%.3f" % y.max()))
                # plt.plot(x, f.coefficients[0]*x**3 + f.coefficients[1]*x**2 + f.coefficients[2]*x + f.coefficients[3], 'x', color = 'red')

        name = self.out_dir+'{}_{}_dependance.pdf'.format(v_n2,v_n1)
        plt.title('{} = f({}) plot'.format(v_n2,v_n1))
        plt.xlabel(v_n1)
        plt.ylabel(v_n2)
        plt.legend(bbox_to_anchor=(1, 0), loc='lower right', ncol=1)
        plt.savefig(name)


    def d3_plotting_x_y_z(self, v_n1, v_n2, v_n3, v_lbl1, v_lbl_cond, list_of_list_of_smfiles = list(), num_pol_fit = True):
        from mpl_toolkits.mplot3d import Axes3D


        fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        ax = fig.gca(projection='3d')


        for j in range(len(list_of_list_of_smfiles)):

            x = []
            y = []
            z = []
            for i in range(len(list_of_list_of_smfiles[j])):
                sm1 = Read_SM_data_File.from_sm_data_file(list_of_list_of_smfiles[j][i])
                x = np.append(x, sm1.get_cond_value(v_n1, 'sp') )
                y = np.append(y, sm1.get_cond_value(v_n2, 'sp') )
                z = np.append(z, sm1.get_cond_value(v_n3, 'sp') )

                lbl1 = sm1.get_cond_value(v_lbl1, v_lbl_cond)

            print(x.shape, y.shape, z.shape)
            # ax.plot_surface(x, y, x, rstride=4, cstride=4, alpha=0.25)

            ax.scatter(x, y, z, c='r', marker='o')
            ax.set_xlabel(Labels.lbls(v_n1))
            ax.set_ylabel(Labels.lbls(v_n2))
            ax.set_zlabel(Labels.lbls(v_n3))



        plt.show()




        # def fitFunc(t, a, b, c, d, e):
        #         # return c * np.exp(-b * t ** a) + d
        #         return a + t**b + t**c + t**d + e ** t    #
        #         # return a + b/t + c/t**2 + d/t**3
        #
        # def myfunc(x, a, b, c):
        #     return a * np.exp(b * x**4) + c*x
        #
        # def fitting():
        #     from scipy.optimize import curve_fit
        #
        #     plt.plot(x, y, 'b.', label='data')
        #     popt, pcov = curve_fit(fitFunc, x, y)
        #     print(popt)
        #
        #     # plt.plot(x, fitFunc(x, *popt), 'r-', label = '' % tuple(popt))
        #     x_new = np.mgrid[x[0]:x[-1]:100j]
        #
        #     plt.plot(x_new, fitFunc(x_new, popt[0], popt[1], popt[2], popt[3], popt[4]), 'r-')
        #
        #     # plt.plot(x, myfunc(x, 1, 1, y[0]))
        #
        #
        #     # t = x# np.linspace(0, 4, 50)
        #     # temp = y# fitFunc(t, 2.5, 1.3, 0.5)
        #     # noisy = temp + 0.05 * np.random.normal(size=len(temp))
        #     # fitParams, fitCovariances = curve_fit(fitFunc, t, noisy)
        #     # print(fitParams)
        #     # print(fitCovariances)
        #     #
        #     # plt.ylabel('Temperature (C)', fontsize=16)
        #     # plt.xlabel('time (s)', fontsize=16)
        #     # plt.xlim(0, 4.1)
        #     # # plot the data as red circles with errorbars in the vertical direction
        #     # plt.errorbar(t, noisy, fmt='ro', yerr=0.2)
        #     # # now plot the best fit curve and also +- 3 sigma curves
        #     # # the square root of the diagonal covariance matrix element
        #     # # is the uncertianty on the corresponding fit parameter.
        #     # sigma = [fitCovariances[0, 0], fitCovariances[1, 1], fitCovariances[2, 2]]
        #     # plt.plot(t, fitFunc(t, fitParams[0], fitParams[1], fitParams[2]),
        #     #          t, fitFunc(t, fitParams[0] + sigma[0], fitParams[1] - sigma[1], fitParams[2] + sigma[2]),
        #     #          t, fitFunc(t, fitParams[0] - sigma[0], fitParams[1] + sigma[1], fitParams[2] - sigma[2])
        #     #          )
        #     plt.show()
        #
        # fitting()
        # save plot to a fil    e
        # savefig('dataFitted.pdf', bbox_inches=0, dpi=600)


        # def fitting()


    def new_3d(self):

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d
        from mpl_toolkits.mplot3d.art3d import Line3DCollection


        def annotate3D(ax, s, *args, **kwargs):
            '''add anotation text s to to Axes3d ax'''

            tag = Annotation3D(s, *args, **kwargs)
            ax.add_artist(tag)



        # data: coordinates of nodes and links
        xn = [1.1, 1.9, 0.1, 0.3, 1.6, 0.8, 2.3, 1.2, 1.7, 1.0, -0.7, 0.1, 0.1, -0.9, 0.1, -0.1, 2.1, 2.7, 2.6, 2.0]
        yn = [-1.2, -2.0, -1.2, -0.7, -0.4, -2.2, -1.0, -1.3, -1.5, -2.1, -0.7, -0.3, 0.7, -0.0, -0.3, 0.7, 0.7, 0.3,
              0.8, 1.2]
        zn = [-1.6, -1.5, -1.3, -2.0, -2.4, -2.1, -1.8, -2.8, -0.5, -0.8, -0.4, -1.1, -1.8, -1.5, 0.1, -0.6, 0.2, -0.1,
              -0.8, -0.4]
        group = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 2, 2, 2, 3, 3, 3, 3]
        edges = [(1, 0), (2, 0), (3, 0), (3, 2), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (11, 10), (11, 3),
                 (11, 2), (11, 0), (12, 11), (13, 11), (14, 11), (15, 11), (17, 16), (18, 16), (18, 17), (19, 16),
                 (19, 17), (19, 18)]

        xyzn = zip(xn, yn, zn)
        segments = [(list(xyzn)[s], list(xyzn)[t]) for s, t in edges]

        # create figure
        fig = plt.figure(dpi=60)
        ax = fig.gca(projection='3d')
        ax.set_axis_off()

        # plot vertices
        ax.scatter(xn, yn, zn, marker='o', c=group, s=64)
        # plot edges
        edge_col = Line3DCollection(segments, lw=0.2)
        ax.add_collection3d(edge_col)
        # add vertices annotation.
        for j, xyz_ in enumerate(xyzn):
            annotate3D(ax, s=str(j), xyz=xyz_, fontsize=10, xytext=(-3, 3),
                       textcoords='offset points', ha='right', va='bottom')
        plt.show()