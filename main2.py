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
from PlottingClasses import Errors
from PlottingClasses import Read_Table
from PlottingClasses import Row_Analyze
from PlottingClasses import Table_Analyze
from PlottingClasses import Math
from PlottingClasses import Physics
from PlottingClasses import PhysPlots
from PlottingClasses import OPAL_Interpol
from PlottingClasses import Constants
from PlottingClasses import Read_SM_data_File
from PlottingClasses import ClassPlots
from PlottingClasses import Read_Observables
from PlottingClasses import New_Table
from PlottingClasses import Read_Plot_file
from PlottingClasses import Treat_Observables
from MainClasses import Combine
#--------------------------------------------
#
# Here, the actual work proceeds,
#
#--------------------------------------------
''' --------------------evolved models -------------------'''
from os import listdir
output_dir  = '../data/output'
plot_dir    = '../data/plots'
opal_fl     = '../data/opal/table8.data'
obs_fl      = '../data/obs/gal_wn.data'

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

smfiles  = get_files('../../sse/sm_e10z002ml/', ['sm9/', 'sm8/', 'sm7/', 'sm6/', 'sm5/', 'sm4/', 'sm3/', 'sm2/', 'sm1/', 'sm0/'], ['2d-5'], 'sm.data')
smfiles1 = get_files('../../sse/sp_z002new/', ['10/u/', '11/u/', '12/u/', '13/u/', '14/u/', '15/u/', '16/u/', '17/u/', '18/u/', '19/u/', '20/u/', '21/u/'], [], 'sm.data')
smfiles2 = get_files('../../sse/sp_z002new/', ['10e5/u/', '11e5/u/', '12e5/u/', '13e5/u/', '14e5/u/', '15e5/u/','16e5/u/','17e5/u/','18e5/u/','19e5/u/','20e5/u/'], [], 'sm.data')
smfiles3 = get_files('../../sse/sp_z002new/15_full/', ['9/u/', '8/u/', '7/u/', '6/u/', '5/u/', '4/u/','3_5/u/','3/u/','2_5/u/', '2/u/','1_5/u/','1/u/','0_5/u/'], [], 'sm.data')
smfiles4 = get_files('../../sse/sp_z002new/20_full/', ['9/u/', '8/u/', '7/u/', '6/u/', '5/u/', '4/u/','3_5/u/','3/u/','2_5/u/','2/u/','1_5/u/','1/u/','0_5/u/'], [], 'sm.data')

#,'2_5/u/', '2/u/','1_5/u/','1/u/','0_5/u/'

# from main_methods import TEST
# tst = TEST(output_dir)
# tst.xy_last_points('l','r','He4', 'core', [smfiles1,smfiles2,smfiles3,smfiles4])
# tst.d3_plotting_x_y_z('l','r','mdot','He4', 'core', [smfiles1,smfiles2,smfiles3,smfiles4])
# tst.new_3d()


# files = get_files('../../sse/', ['e10z002ml/7/test/', 'test_ml_sp/7/test/'], [], 'sm.data')
plotfls = get_files('../../sse/sp_z002new/', ['20_full/'], [], '.plot1')
# print(smfiles)
# smfiles = get_files('../../sse/sm_e10z002ml/', ['sm0/', 'sm1/', 'sm2/', 'sm3/', 'sm4/', 'sm5/', 'sm6/', 'sm7/', 'sm8/', 'sm9/'], ['1d-5'], 'sm.data')
# smfiles = get_files('./../../sse/comp15DYNFAK/', ['on/', 'off/'], [], 'sm.data')
# plotfls = get_files('./../../sse/comp15DYNFAK/', ['on/', 'off/'], [], '.plot1')
#'sm9/', 'sm8/', 'sm7/', 'sm6/', 'sm5/', 'sm4/', 'sm3/', 'sm2/', 'sm1/', 'sm0/'
# Read_SM_data_File.compart = ''

'''-------------------------------------------------------------'''
from MainClasses import Creation

# make = Creation(opal_fl, 4.9, 5.5, 1000)
# # make.save_t_rho_k()
# make.save_t_k_rho(3.8, None, 1000)
# # file_table = np.zeros(1)
# # make.read_table('t_k_rho', 't', 'k', 'rho', '../data/opal/table8.data')
# make.save_t_llm_vrho('l')
# # print(file_table)
# make.save_t_llm_mdot(1.,'l','',1.34)
#
# a = np.array(([1,1,1,1], [5,5,5,5], [7,7,7,7], [9,9,9,9]))
# a = np.flip(a, 0)
# print(a)

'''-------------------------------------------------------------'''

# comb = Combine(smfiles4, plotfls, obs_fl, opal_fl)
# comb.xy_profile('r','u','mdot','xm')
# comb.xyy_profile('r','rho','kappa','mdot','xm','t', False)
# comb.xy_last_points('r','l','mdot',True)
# comb.hrd(plotfls)
# comb.plot_t_rho_kappa('mdot','xm')
# comb.plot_t_mdot_lm()
# comb.plot_t_l_mdot('l',1.0,'xm',5.2)
# comb.min_mdot('l',None,'xm',5.2)

# print(np.arange(10**(-4),10**(-6),10))
















'''---------------------------------------------------------------'''



# cl1 = ClassPlots('../data/opal/table8.data', smfiles, ['../data/obs/gal_wn.data'], [], 1000, True, '../data/output/', '../data/plots/')



# t1 = 4.6
# t2 = 5.5
# r_s = 1.
#
# Treat_Observables.cluming_required = 4

# cl1.xy_profile('r','u', 'mdot', 'l', 0)
# cl1.xyy_profile('r','rho','kappa', 'mdot', 'l', 'r')

# cl1.opacity_check(1, None, None, None)
# cl1.opacity_check(1, 4.75, None, 0.068)
# cl1.plot_t_rho_kappa(5.25, 5.3, -9.5, -7.5, 1000, True)
# cl1.plot_t_mdot_lm( t1, t2, 'xm',-6, -4.5, r_s )
# cl1.plot_t_l_mdot('l', t1, t2, 4.8, None, [0.8], 500, 500, 'xm','mdot', 5.2, None)

# cl1.plot_rs_l_mdot_min('l', t1, t2, 4.1, 4.5, 0.5, 2.0, 200, 200, 50, False)

# # 'e10z002/', 'e15z002/','e20z002/' # 'e20z002ml/', 'e20z002ml_2/', 'test/', 'e20z002ml_test2/' # 'plots_ml/'
# files = get_files('../../sse/', ['plots_ml/'], [], '.plot1')
# # print(files)
# cl1.hrd(4.5, 5.2,'gal_wn', files)

# cl1.rs_l_models()



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


# var_names = ['nan', 'u', 'r', 'rho', 't', 'l', 'vu', 'vr',
#              'vrho', 'vt', 'vsl','e', 'dm', 'xm', 'n', 'H',
#              'D', 'He3', 'He4', 'Li6', 'Li7', 'Be7', 'Be9', 'B8', 'B10',
#              'B11', 'C11', 'C12', 'C13', 'N12', 'Li7', 'N15', 'O16', 'O17', 'O18',
#              'Ne20', 'Ne21', 'Ne22', 'Na23', 'Mg24', 'Mg25', 'Mg26', 'Al27',
#              'Si28', 'Si29', 'Si30', 'Fe56', 'F19', 'Al26', 'w', 'j', 'diff',
#              'dg', 'd1', 'd2', 'd3', 'd4', 'd5', 'bvis', 'bdiff', 'br', 'bphi',
#              'bfq', 'bfq0', 'bfq1', 'ibflag', 'Pg', 'Pr', 'HP', 'Grav', 'kappa',
#              'ediss', 'tau', 'nabla_rad', 'L/Ledd', 'nabla', 'P_total', 'mu',
#              'psi', 'dPg_dPr|rho', 'Pturb', 'beta', 'vel_conv', 'mdot', 'tau_ph',
#              ]

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


