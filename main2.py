import sys
import pylab as pl
from matplotlib import cm
import numpy as np
from scipy import interpolate
import scipy.ndimage
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os
from Plots_Obs_treat import Errors
from Plots_Obs_treat import Read_Table
from Plots_Obs_treat import Row_Analyze
from Plots_Obs_treat import Table_Analyze
from Plots_Obs_treat import Math
from Plots_Obs_treat import Physics
from Plots_Obs_treat import PhysPlots
from Plots_Obs_treat import OPAL_Interpol
from Plots_Obs_treat import Constants
from Plots_Obs_treat import Read_SM_data_File
from Plots_Obs_treat import ClassPlots
from Plots_Obs_treat import Read_Observables
from Plots_Obs_treat import New_Table
from Plots_Obs_treat import Read_Plot_file
from Plots_Obs_treat import Treat_Observables
#--------------------------------------------
#
# Here, the actual work proceeds,
#
#--------------------------------------------
''' --------------------evolved models -------------------'''
from os import listdir
output_dir = '../data/output'
plot_dir = '../data/plots'

def get_files(compath, req_dirs, requir_files, extension):
    comb = []
    # extension = 'sm.data'
    for dir_ in req_dirs:
        for file in listdir(compath+dir_):
            ext = file.split('.')
            req_ext = extension.split('.')
            # print(ext[-1],req_ext[-1])
            if ext[-1] == req_ext[-1]:

            # print(file, requir)
                if requir_files == []:
                    print('\t__Note: Files to be plotted: ', dir_ + file)
                    comb.append(compath + dir_ + file)
                else:
                    for req_file in requir_files:
                        if req_file+extension == file:
                            print('\t__Note: Files to be plotted: ', dir_+file)
                            comb.append(compath+dir_+file)

    return comb

# smfiles = get_files('../../sse/sm_e10z002ml/', ['sm9/', 'sm8/', 'sm7/', 'sm6/', 'sm5/', 'sm4/', 'sm3/', 'sm2/', 'sm1/', 'sm0/'], ['2d-5'], 'sm.data')
smfiles = get_files('../../sse/sp_z002new/', ['13/'], [], 'sm.data')
# smfiles = get_files('../../sse/sp_z002/', ['16/','17/','18/'], [], 'sm.data')


# files = get_files('../../sse/', ['e10z002ml/7/test/', 'test_ml_sp/7/test/'], [], 'sm.data')
plotfls = get_files('../../sse/', ['plots_ml/'], [], '.plot1')
print(smfiles)
# files = get_files('../../sse/sm_e10z002ml/', ['sm0/', 'sm1/', 'sm2/', 'sm3/', 'sm4/', 'sm5/', 'sm6/', 'sm7/', 'sm8/', 'sm9/'], ['1d-5'])
# files = get_files('./../../sse/', ['sm10z002/'], [])
#'sm9/', 'sm8/', 'sm7/', 'sm6/', 'sm5/', 'sm4/', 'sm3/', 'sm2/', 'sm1/', 'sm0/'
# Read_SM_data_File.compart = ''
cl1 = ClassPlots('../data/opal/table8.data', smfiles, ['../data/obs/gal_wn.data'], [], 1000, True, '../data/output/', '../data/plots/')



t1 = 4.6
t2 = 5.5
r_s = 1.

Treat_Observables.cluming_required = 4

cl1.xy_profile('r','u', 'mdot', 'l', 0)
# cl1.xyy_profile('r','rho','kappa', 'mdot', 'l', 'r')

# cl1.opacity_check(1, None, None, None)
# cl1.opacity_check(1, 4.75, None, 0.068)
# cl1.plot_t_rho_kappa(4.5, 5.5, -9.5, -7.5, 1000, True)
# cl1.plot_t_mdot_lm(  t1, t2, 'Y_c',-5.5, -3, r_s)
# cl1.plot_t_l_mdot('l', t1, t2, 4.8, None, [0.8], 500, 500, 'xm','mdot', 5.2, None)

# cl1.plot_rs_l_mdot_min('l', t1, t2, 4.8, None, 0.1, 20.0, 200, 200, 50, True)

# # 'e10z002/', 'e15z002/','e20z002/' # 'e20z002ml/', 'e20z002ml_2/', 'test/', 'e20z002ml_test2/' # 'plots_ml/'
# files = get_files('../../sse/', ['plots_ml/'], [], '.plot1')
# # print(files)
# cl1.hrd(4.5, 5.2,'gal_wn', files)

# cl_plot= Read_Plot_file.from_file('../../sse/e20z002ml/ev.plot1')
# plt.plot(cl_plot.time, cl_plot.t_eff)
# plt.show()



# files = get_files('./../../sse/grid/', ['plots/'], [])
#
# cl1.table_of_plot_files(files, './../../sse/grid/test.txt')

# print(Physics.mass_lum_rel(Physics.loglm([5.4,5.5,5.6,5.7], [15,15,16,16], True)))
# print(Physics.mass_lum_rel(Physics.loglm(5.4, 15)


'''---------------------'''
# cl2 = New_Table('./opal/',['table1', 'table2', 'table3', 'table4', 'table5', 'table6', 'table7', 'table8', 'table9', 'table10', 'table11', 'table12', 'table13'],
#                           [0.0000,    0.0001,   0.0003,   0.0010,   0.0020,   0.0040,   0.0100,   0.0200,   0.0300,   0.0400,   0.0600,     0.0800,    0.1000])
#
# new_opal = cl2.get_new_opal(0.008)

# cl3 = OPAL_Interpol.from_OPAL_table('table_x.data', 1000)

# clx = Read_Observables('gal_wn')
#['num', 'type', 't_eff', 'r_t', 'v_inf', 'x_h', 'e_bv', 'law', 'dm', 'm_v','r_o','mdot','l','form','m']
# print(clx.get_parms('type',str))

# Read_Plot_file.path = '../../sse/e10z002/'
# cly = Read_Plot_file.from_file('ev')
#
# plt.plot(cly.t_eff, Physics.loglm(cly.l_, cly.m_))
# plt.show()



# a = [1,2,3,4,5,6,77,8,9,9]
# print(a[:len(a)])


    # comb.append([f for f in listdir(compath+dir_)])



# print(comb)

# cl1 = Table_Analyze('./opal/table8', 1000, False)
# cl1.table_plotting(4.6, 5.2)

# cl2 = OPAL_Interpol('./opal/table8',1000)
# res = cl2.interp_opal_table(4.5, 5.5,None,-7.5)
# PhysPlots.t_rho_kappa(res[0,1:],res[1:,0],res[1:,1:])


# clpl = Read_Plot_file.from_file('../../sse/e15z002ml/ev')


# name_out = 'rs_l_mdot.data'
# if True:
#     min_mdot = np.loadtxt(name_out, dtype=float, delimiter=' ')
# else:
#     min_mdot = ClassPlots.get_min_mdot_set(r_s_, y_label, t, kap, rho2d)
# np.savetxt(name_out, min_mdot, delimiter=' ', fmt='%.3f')
#
# # sys.exit('stop: {}'.format(min_mdot[0,1:]))
# cl2 = Table_Analyze(name_out,1000,False)
# cl2.treat_tasks_tlim(1000, 0.52, 1.5)

# a = np.vstack(([1,2,3],[4,5,6],[7,8,9]))
# print(np.flip(a,0))


# cl = Treat_Observables(['gal_wn'])
# cl.check_if_var_name_in_list('WR')
# cl.get_x_y_of_all_observables('log(Mdot)','log(L)','type')
#
#
# a = np.array(([1,2],[2,3]))
# b = np.array(([1,2,3], [4,5,6], [7,8,9]))
#
# c = np.array([a,b])
#
# print(c[0][0,0])
# print(c)


# a = np.array([1,2,3,4,5,6,7,8])
#
# print(int(np.where(a == 3)[0]))


var_names = ['nan', 'u', 'r', 'rho', 't', 'l', 'vu', 'vr',
             'vrho', 'vt', 'vsl','e', 'dm', 'xm', 'n', 'H',
             'D', 'He3', 'He4', 'Li6', 'Li7', 'Be7', 'Be9', 'B8', 'B10',
             'B11', 'C11', 'C12', 'C13', 'N12', 'Li7', 'N15', 'O16', 'O17', 'O18',
             'Ne20', 'Ne21', 'Ne22', 'Na23', 'Mg24', 'Mg25', 'Mg26', 'Al27',
             'Si28', 'Si29', 'Si30', 'Fe56', 'F19', 'Al26', 'w', 'j', 'diff',
             'dg', 'd1', 'd2', 'd3', 'd4', 'd5', 'bvis', 'bdiff', 'br', 'bphi',
             'bfq', 'bfq0', 'bfq1', 'ibflag', 'Pg', 'Pr', 'HP', 'Grav', 'kappa',
             'ediss', 'tau', 'nabla_rad', 'L/Ledd', 'nabla', 'P_total', 'mu',
             'psi', 'dPg_dPr|rho', 'Pturb', 'beta', 'vel_conv', 'mdot', 'tau_ph',
             ]

# print('T=5.2'.split('='))

# def get_0_to_max(arr, max):
#     j = 0
#     n = 1
#     res = []
#     for i in range(len(arr)):
#         res.append(j)
#         j = j + 1
#         if arr[i] == n*max:
#             j = 0
#             n = n + 1
#
#     return res

# print(Math.get_0_to_max([14], 9)[14])
#
# a = np.array([1,2,3,4,])
# print(np.insert(a,0,0))

# def a(list=list()):
#     print(list)
#
# a()

# c=2
# def b(v):
#     global c
#     c = 2*v
#
#
# b(c)
# print(c)
#
# a = np.array([2])
# def b(v):
#     global a
#     a = np.zeros(2)
#
# b(a)
# print(a, a.max())

from Plots_Obs_treat import Tables

clll = Tables('../data/opal/table8.data', t1, t2)
# clll.save_interp_opals()
# clll.save_t_k_rho(3.8,None)
file_table = np.zeros(1)
clll.read_table('t_k_rho', 't', 'k', 'rho', '../data/opal/table8.data')
print(file_table)