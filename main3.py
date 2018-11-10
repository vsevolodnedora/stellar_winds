# from __future__ import division
import sys
import pylab as pl
from matplotlib import cm
import numpy as np
from scipy import interpolate
import scipy.ndimage
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from os import listdir
import matplotlib.patches as patches

import os
# from PlottingClasses import Read_Table
# from PlottingClasses import Row_Analyze
# from PlottingClasses import Table_Analyze
# from PlottingClasses import Math
# from PlottingClasses import Physics
# from PlottingClasses import PhysPlots
# from PlottingClasses import OPAL_Interpol
# from PlottingClasses import Constants
# from PlottingClasses import Read_SM_data_file
# from PlottingClasses import ClassPlots
# from PlottingClasses import Read_Observables
# from PlottingClasses import New_Table
# from PlottingClasses import Read_Plot_file
# from PlottingClasses import Treat_Observables
from MainClasses import Combine
from FilesWork import Files
# from PhysMath import Opt_Depth_Analythis
# from FilesWork import SP_file_work

def get_z_from_fh(fh, zgal = 0.02):
    return zgal * 10**(fh)

# sys.exit(get_z_from_fh(-0.49))

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
                    # print('\t__Note: Files to be plotted: ', dir_ + file)
                    comb.append(compath + dir_ + file)
                else:
                    for req_file in requir_files:
                        if req_file+extension == file:
                            # print('\t__Note: Files to be plotted: ', dir_+file)
                            comb.append(compath+dir_+file)

    return comb

# for i in [9,8,7,6,5]:
#     print('mkdir y{}/sp55_d; mkdir y{}/sp55_d/prec'.format(i,i))
#     print('cp y10/sp55_d/prec/*.dat y{}/sp55_d/prec/'.format(i))
#     print('cp y10/sp55_d/prec/*.sh y{}/sp55_d/prec/'.format(i))
#     print('cp y{}/sp/3.50.bin1 y{}/sp55_d/prec/'.format(i,i))

# for i in range(1,9):
#     print('mkdir y{}; '.format(i))
#
#     print('cp ../../ga_z002/10sm/y{}/fy{}.bin1 y{}/; '.format(i, i, i))
#     print('cp ../ev_m.dat y{}/; '.format(i))
#     print('cp ../fred.sh y{}; '.format(i))
#     print('cp ../run_mass_loss.py y{}'.format(i))

# for i in range(10,31):
#     print('cp ref_m.dat *.sh {}sm/y{}/sp55_d/prec/;'.format(i, 10))
    # print('cp {}sm/y{}/sp/3.50.bin1 {}sm/y{}/sp55_d/prec/;'.format(i, 10, i, 10))
    # print('cd {}sm/y{}/sp55_d/prec/; rm 3* 4* 5* 6*; sse; cd ga_z0008;'.format(i, 10))
    # print('mkdir {}z0004'.format(i))

# for i in range(10,31):
#     for j in range(1,11):
#         print('mkdir {}sm/; mkdir {}sm/y{}/; '.format(i, i, j))
              # 'mkdir {}sm/y{}/sp55_d/; mkdir {}sm/y{}/sp55_d/prec/; '
              # 'mkdir {}sm/y{}/sp55_d/prec/b09_v2200/;'.format(i,i,j, i,j, i,j, i,j))
#     print('rm {}sm/y10/sp55_d/prec/3.5*'.format(i))
# # # #     # print('rm -r {}sm/y10/sp55/; mkdir {}sm/y10/sp55_d'.format(i,i))
# #     print('cp {}sm/y10/sp/3.50.bin1 {}sm/y10/sp55_d/prec/'.format(i, i))
# #     print('cp ref_m.dat {}sm/y10/sp55/'.format(i))
# # #     # print('cp auto_*.sh {}sm/y10/sp55_d/'.format(i))
# # #
#     print('mkdir {}sm/y10/sp55_d/; mkdir {}sm/y10/sp55_d/prec/; '
#           'cp auto_ev2.sh auto_rename.sh ref_m.dat {}sm/y10/sp55_d/prec/; '
#           'cp {}sm/y10/sp/3.50.bin1 {}sm/y10/sp55_d/prec/'
#           .format(i,i,i,i,i,i))
#     print('cp {}sm/y10/sp/3.50.bin1 {}sm/y10/sp55_d/prec/'.format(i,i))
    # print('rm {}sm/y10/sp55_d/b1_v2200/3.*; rm {}sm/y10/sp55_d/b1_v2200/4.*; rm {}sm/y10/sp55_d/b1_v2200/5.*'.format(i,i,i))
    # print('cp {}sm/y10/sp55_d/5.50.bin1 {}sm/y10/sp55_u/b1_v2200/'.format(i,i))
    # print('cp ref_m.dat {}sm/y10/'.format(i))
    # print('cp auto_ev2.sh {}sm/y10/sp55_d/b1_13vesc/'.format(i))
# #     print('cp ga_z0008_old/{}sm/{}ev.plot1 ga_z0008/{}sm/'.format(i, i, i))
#     for j in range(1,10):
#         # print('cp *.sh *.dat {}sm/y{}/'.format(i, j))
# # #         # print('rm ga_z002_2/{}sm/y{}/fy{}.bin1'.format(i, j, j))
#         print('cp ga_z002_2tmp/{}sm/y{}/fy{}.bin1 ga_z0008_2/{}sm/y{}/'.format(i, j, j, i, j))
#         print('cp ga_z0004_tmp/{}sm/y{}/*.bin1 '
#               'ga_z0004_tmp/{}sm/y{}/*.plot1 '
#               'ga_z0004_tmp/{}sm/y{}/*sm.data '
#               'ga_z0004/{}sm/y{}/'
#               .format(i,j, i,j, i,j, i,j))
        # print('mv {}sm/y{}/sp_files {}sm/y{}/sp'.format(i, j, i, j))
        # print('cp auto_* ref_m.dat re_m.dat check.py {}sm/y{}/; '.format(i, j)) # EVE
        # print('cp re_m.dat {}sm/y{}/'.format(i, j))
        # print('rm -r {}sm/y{}/sp/*_WIND'.format(i, j))
        # print('cp ev_m.dat {}sm/y{}/'.format(i, j))
        # print('cp auto_rename.sh {}sm/y{}/'.format(i, j))
        # print('cp auto_ev2.sh {}sm/y{}/'.format(i, j))
        # print('cp ref_m.dat {}sm/y{}/'.format(i, j))
        # print('cp re_m.dat {}sm/y{}/'.format(i, j))
        # print('cp ref_w_m.dat ../{}sm/y{}/sp/'.format(i, j))
        # print('cp run_wind.py ../{}sm/y{}/sp/'.format(i, j))



output_dir  = '../data/output/'
plot_dir    = '../data/plots/'
sse_locaton = '/media/vsevolod/HDD/sse/'

# gal_plotfls = get_files(sse_locaton + 'ga_z002/', ['10sm/', '11sm/', '12sm/', '13sm/', '14sm/', '15sm/', '16sm/',
#                                                    '17sm/', '18sm/', '19sm/', '20sm/', '21sm/', '22sm/', '23sm/',
#                                                    '24sm/', '25sm/'], [], '.plot1')
# lmc_plotfls = get_files(sse_locaton +'ga_z0008/', ['10sm/', '11sm/', '12sm/', '13sm/', '14sm/', '15sm/', '16sm/',
#                                                    '17sm/', '18sm/', '19sm/', '20sm/', '21sm/', '22sm/', '23sm/',
#                                                    '24sm/', '25sm/', '26sm/', '27sm/', '28sm/', '29sm/', '30sm/'
#                                                    ], [], '.plot1')
# tst_plotfls = get_files(sse_locaton +'ga_z0008/', ['10sm/', '15sm/', '20sm/', '25sm/', '30sm/'], [], '.plot1')
#
# gal_spfiles = get_files('../data/sp_cr_files/', ['7z002/','8z002/','9z002/', '10z002/', '11z002/', '12z002/', '13z002/', '14z002/', '15z002/',
#                                                  '16z002/', '17z002/', '18z002/', '19z002/', '20z002/', '21z002/',
#                                                  '22z002/', '23z002/', '24z002/', '25z002/', '26z002/', '27z002/',
#                                                  '28z002/', '29z002/', '30z002/'
#                                                  ], [], '.data')
# lmc_spfiles = get_files('../data/sp3_files/', ['10z0008/', '11z0008/', '12z0008/', '13z0008/', '14z0008/',
#                                               '15z0008/', '16z0008/', '17z0008/', '18z0008/', '19z0008/',
#                                               # '20z0008/', '21z0008/', '22z0008/', '23z0008/', '24z0008/',
#                                               # '25z0008/', '26z0008/', '27z0008/', '28z0008/', '29z0008/',
#                                               # '30z0008/'
#                                               ], [], '.data')
# tst_spfiles = get_files('../data/sp3_files/', ['10z0008/'], [], '.data')
#
# lmc_obs_file = '../data/obs/lmc_wne.data'
# gal_obs_file = '../data/obs/gal_wne.data'
# tst_obs_file = ''
#
# lmc_opal_file = '../data/opal/table_x.data'
# gal_opal_file = '../data/opal/table8.data'
# tst_opal_file = ''
#
# gal_atm_file = '../data/atm/gal_atm_models.data'
# lmc_atm_file = '../data/atm/lmc_atm_models.data'
#
# smfiles = get_files(sse_locaton + 'ga_z0008/', ['13sm/y10/sp/'], [], 'sm.data')
# # smfiles = get_files(sse_locaton, ['zams_004/hecore/t1/'], [], 'sm.data')
#
# # lmc_ml_relation = '../data/output/l_yc_m_lmc_wne.data'
# # gal_ml_relation = '../data/output/l_yc_m_gal_wne.data'
#
# def select_sp_files(spfiles, req_z, req_m, req_yc):
#
#     res_spfiles = []
#
#     '''----------------------SELECT-req_z-FILES----------------------'''
#
#     z_files = []
#     for spfile in spfiles:
#
#         if len(req_z) == 0:
#             z_files.append(spfile)
#         else:
#             no_extens_sp_file = spfile.split('.')[-2]  # getting rid of '.data'
#
#             # print(spfile, '   ', no_extens_sp_file)
#
#             for req_part in req_z:
#                 if req_part in no_extens_sp_file.split('_')[1:]:
#                     if spfile not in z_files:
#                         req_z.append(spfile)
#                 else:
#                     pass
#
#     print('__For z:{} requirement, {} SP-files selected.'.format(req_z, len(z_files)))
#     '''----------------------SELECT-req_m-FILES----------------------'''
#
#     zm_files = []
#     for spfile in z_files:
#
#         if len(req_m) == 0:
#             zm_files.append(spfile)
#         else:
#             no_extens_sp_file = spfile.split('.')[-2]  # getting rid of '.data'
#
#             # print(spfile, '   ', no_extens_sp_file)
#
#             for req_part in req_m:
#                 if req_part in no_extens_sp_file.split('_')[1:]:
#                     if spfile not in zm_files:
#                         zm_files.append(spfile)
#                 else:
#                     pass
#
#
#     print('__For z:{} and m:{} requirement, {} SP-files selected.'.format(req_z, req_m, len(zm_files)))
#     '''----------------------SELECT-req_m-FILES----------------------'''
#
#     zmy_files = []
#     for spfile in zm_files:
#
#         if len(req_yc) == 0:
#             zmy_files.append(spfile)
#         else:
#             no_extens_sp_file = spfile.split('.')[-2]  # getting rid of '.data'
#
#             # print(spfile, '   ', no_extens_sp_file)
#
#             for req_part in req_yc:
#                 if req_part in no_extens_sp_file.split('_')[1:]:
#                     if spfile not in zmy_files:
#                         zmy_files.append(spfile)
#                 else:
#                     pass
#
#
#
#     print('\t__ With Conditions: z:{}, m:{}, yc:{}, the {} sp_files selected.'.format(req_z, req_m, req_yc, len(zmy_files)))
#     print('\n')
#     return zmy_files
def fe_h_to_metal(fe_h,z_gal = 0.02):
    return 10**(fe_h)*z_gal
# sys.exit(fe_h_to_metal(-1.50)) # ave
'''===============================================SETTING=FILES======================================================'''
# spfiles = []
# opalfile =[]
# obsfile = []
# plotfiles=[]
# atmfile=[]
# def set_sp_oopal_obs(gal_or_lmc):
#     global spfiles
#     global opalfile
#     global plotfiles
#     global obsfile
#     if gal_or_lmc == 'lmc':
#         spfiles = lmc_spfiles
#         opalfile = lmc_opal_file
#         plotfiles = lmc_plotfls
#         obsfile = lmc_obs_file
#         atmfile = lmc_atm_file
#     if gal_or_lmc == 'gal':
#         spfiles = gal_spfiles
#         opalfile = gal_opal_file
#         plotfiles = gal_plotfls
#         obsfile = gal_obs_file
#         atmfile = gal_atm_file
#     if gal_or_lmc == 'tst':
#         spfiles = tst_spfiles
#         opalfile = tst_opal_file
#         plotfiles = tst_plotfls
#         obsfile = tst_obs_file
#     # raise NameError('Wrong name: {}'.format(gal_or_lmc))
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# set_sp_oopal_obs('gal')
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''=================================================NEW=TABLE========================================================'''

# ntbl = New_Table('../data/opal/',['table1','table2','table3','table4','table5','table6','table7','table8', 'table9', 'table10'],
#                  [0.000, 0.0001, 0.0003, 0.0010, 0.0020, 0.0040, 0.0100, 0.0200, 0.0300, 0.0400], '../data/opal/')
#
# ntbl.check_if_opals_same_range()
# ntbl.get_new_opal(0.008)
from PhysMath import Physics
print('T_eff:', 10**Physics.steph_boltz_law_t_eff(5.139, 3.4))
print('v_esc:', Physics.get_v_esc(20,1.3))
'''===========================================GRAY=ATMPOSPHERE=ANALYSYS=============================================='''
# from FilesWork import Read_Atmosphere_File
# atm=Read_Atmosphere_File(opalfile)
# atm.plot_tstar_rt('t', 'r', 't_eff')

# def gray_analysis(z, m_set, y_set, plot):
#
#     from Temp import master_gray
#     for m in m_set:
#         for y in y_set:
#             root_name = 'ga_z' + z + '/'
#             folder_name = str(m)+'sm/y'+str(y)+'/'
#             out_name = str(m) + 'z' + z + '/'
#
#
#             print('COMPUTING: ({}) {} , to be saved in {}'.format(root_name, folder_name, out_name))
#             smfiles_ = get_files(sse_locaton + root_name, [folder_name], [], 'sm.data')
#
#             cr = master_gray(smfiles_, '../data/sp_files/' + out_name, plot_dir,
#                          ['sse', 'ga_z002','ga_z0008', 'vnedora', 'media', 'vnedora', 'HDD'])  # [] is a listof folders not to be put in output name
#             # cr.save(1000, ['kappa-sp', 'L/Ledd-sp', 'HP-sp', 'mfp-sp'], plot)
#
#             print('m:{}, y:{} DONE'.format(m,y))
#
# def gray_analysis2(z, m_set, y_set, plot):
#
#     from Sonic_Criticals import Criticals2 # CRITICALS2 also computes sonic-BEC sm and plot files to get tau.
#     for m in m_set:
#         for y in y_set:
#             root_name = 'ga_z' + z + '/'
#             folder_name = str(m)+'sm/y'+str(y)+'/'
#             out_name = str(m) + 'z' + z + '/'
#
#             print('COMPUTING: ({}) {} , to be saved in {}'.format(root_name, folder_name, out_name))
#
#             smfiles_ga = get_files(sse_locaton + root_name, [folder_name], [], 'sm.data')
#             smfiles_sp = get_files(sse_locaton + root_name, [folder_name+'sp/'], [], 'sm.data')
#             plotfls_sp = get_files(sse_locaton + root_name, [folder_name+'sp/'], [], '.plot1')
#
#             cr = Criticals2(smfiles_ga, smfiles_sp, plotfls_sp,
#                             ['sse', 'ga_z002','ga_z0008', 'vnedora', 'media', 'vnedora', 'HDD'], '../data/sp2_files/' + out_name)
#
#             cr.combine_ga_sp(1000, ['kappa-sp', 'L/Ledd-sp', 'HP-sp', 'mfp-sp', 'tpar-'], plot, 'IntUni')
#
#             print('m:{}, y:{} DONE'.format(m,y))
#
# def gray_analysis3(z, m_set, y_set, plot):
#
#     from Sonic_Criticals import Criticals3 # CRITICALS2 also computes sonic-BEC sm and plot files to get tau.
#     from Temp import master_sonic
#     for m in m_set:
#         for y in y_set:
#             root_name = 'ga_z' + z + '/'
#             folder_name = str(m)+'sm/y'+str(y)+'/'
#             out_name = str(m) + 'z' + z + '/'
#
#             print('COMPUTING: ({}) {} , to be saved in {}'.format(root_name, folder_name, out_name))
#
#             # smfiles_ga = get_files(sse_locaton + root_name, [folder_name], [], 'sm.data')
#             smfiles_sp = get_files(sse_locaton + root_name, [folder_name+'sp/'], [], 'sm.data')
#             sucafls    = get_files(sse_locaton + root_name, [folder_name+'sp/'], [], 'wind')
#
#             cr = master_sonic(smfiles_sp, sucafls, '../data/sp3_files/' + out_name, plot_dir,
#                             ['sse', 'ga_z002','ga_z0008', 'vnedora', 'media', 'vnedora', 'HDD'])
#
#             # cr.combine_save(1000, ['kappa-sp', 'L/Ledd-sp', 'HP-sp', 'mfp-sp', 'tpar-'], plot, wind)
#
#             print('m:{}, y:{} DONE'.format(m,y))

# gray_analysis3('0008', [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30], [10], True, False)
# gray_analysis('002', [26], [10, 9, 8, 7, 6, 5, 4, 3, 2], True)
# gray_analysis3('0008', [18], [9], True, False)
# gray_analysis3('002', [10], [10], True)
# 10,11,12,13,14,15,16,17,18,19,20,21,22,

'''====================================================CREATION======================================================'''
from FilesWork import Read_Observables
from FilesWork import Read_Atmosphere_File

# def save_atm_table(z_v_n, obs_file, atm_file, opal_used, bump):
#     obs = Read_Observables(obs_file, opal_used, False)
#     obs.set_use_atm_file = False
#     obs.set_use_gaia = False
#     obs.set_atm_file = atm_file
#
#     atm = Read_Atmosphere_File(atm_file, opal_used)
#     table = atm.get_table('t_*', 'rt', z_v_n)
#
#     out_table = np.zeros((2))
#     for star_n in obs.stars_n:
#         tstar = obs.get_num_par('t', star_n)
#         rt = obs.get_num_par('lRt', star_n)
#         t_eff = Math.interpolate_value_table(table, [tstar, rt])
#         out_table = np.vstack((out_table, [star_n, t_eff]))
#
#     from FilesWork import Save_Load_tables
#     Save_Load_tables.save_table(out_table, opal_used, bump, 'none_model_t_eff', 'none', 'model', 't_eff')

# save_atm_table('t_eff', obsfile, atmfile, opalfile, 'Fe')

from FilesWork import OPAL_work
# make = OPAL_work('lmc', 'Fe', 1000, False) # False to load the cases-limits
# make.set_plots_clean=True
#
# make.save_t_k_rho(3.2, None, 1000)
# make.from_t_k_rho__to__t_lm_rho(1.0)
# make.save_t_rho_k(None, None, 4.1, 5.6)
# make.plot_t_rho_kappa('lmc', 'Fe')


from FilesWork import SP_file_work
spcls = SP_file_work(0.1, 'lmc', 'Fe', 'wd', ['prec'])
spcls.set_clean_plots       = True
spcls.set_extrapol_pars     = [0, 0, 0, 0] # in %: v, ^, <-, ->         | Problem with 5.47 > 5.469 in x_y_z method...
spcls.set_int_or_pol        = 'pol' # to save
spcls.set_init_fit_method   = '1dLinear'
spcls.invert_xaxis          = False
spcls.set_do_tech_plots     = False
spcls.set_check_x_y_z_arrs  = True
# spcls.save_y_yc_z_relation('l', 'lm',  True)
# spcls.save_y_yc_z_relation('lm', 'r',  True)
# spcls.save_y_yc_z_relation('lm', 'l',  True)
# spcls.save_min_max_lm('lm')
# spcls.save_y_yc_z_relation('lm', 'r',  True)

# spcls.save_t_llm_mdot('lm', 1.0, 'Fe', 500, 'min', False) # for sHRD
# spcls.save_t_llm_mdot_const_r('lm', 0.9, 'HeII', 1.0, 500, 'min', True)

# spcls.test()
spcls.set_xy_int_method      = 'Uni'
spcls.set_xz_int_method      = 'poly4'
spcls.set_yz_int_method      = '1dLinear'
spcls.set_load_cond          = 'ts>5.0'     # reading only Fe bunp sols

# spcls.plot_2_x_y_z_for_yc('mdot', 'lm', 'a_p', 1.0, 500, 'max', False, True)
# spcls.save_beta_x_y_vinf('mdot', 'lm', 'grad_c_p', [1.15,1.0], 100, 'max', False)
# spcls.plot_beta_x_y_vinf_3d('mdot', 'lm', 'vinf', [0.10, 1.00, 1.10])

# spcls.wind()

# spcls.save_min_max_lm('l')
# # spcls.separate_sp_by_fname()
# spcls.save_y_yc_z_relation('lm', 'mdot', 'pol', True)
# spcls.save_y_yc_z_relation('lm', 'ys', 'int', True)           # ONLY for Ys

# spcls.save_x_y_yc_evol_relation('m', 'ys')

# spcls.plot_x_y_z_for_yc('mdot', 'lm', 'grad_c_p', 1.0, 500, 'max', False, True)
# spcls.save_x_y_z('mdot', 'lm', 'grad_c_p', 500, 'max', False, '1dLinear') #    Uni, IntUni, 1dLinear, 1dCubic methods availabel
# spcls.plot_x_y_z('t', 'lm', 'r', [1.0], 100, 'min', True)

# spcls.plot_t_llm_mdot_for_yc(1.0, 'lm', 1.0, 'Fe', 'min')
# spcls.save_t_llm_mdot('lm', 1.0, 'Fe', 500, 'max', True)
# spcls.plot_t_llm_mdot_for_yc_const_r(1.0, 1.0, 'lm', 1.0, 'Fe', 'min')
# spcls.save_y_yc_z_relation_sp('t', 'lm', 'r', 'pol', True)
# spcls.save_t_llm_mdot_const_r('lm', 1.0, 'Fe', 1.0, 500, 'min', True)
# spcls.separate_sp_by_crit_val('Yc', 0.1)


from MainClasses import Critical_Mdot
mdot = Critical_Mdot('lmc', 'Fe', 1.0, 'lmc')
# mdot.save_yc_llm_mdot_cr()      # yc_lm_l, yc_lm_r are used and should be for [10] Yc values
# mdot.save_yc_llm_mdot_cr_const_r(1.0)
# mdot.plot_test_min_mdot(1.0)

'''=====================================================ATMOSPHERE==================================================='''
from FilesWork import Read_Atmosphere_File
# atm = Read_Atmosphere_File(Files.get_atm_file('gal'), 'gal')
# atm.plot_tstar_rt('t_*','rt','t_eff', 'gal')
'''=======================================================TABLES====================================================='''
m  = '20'
yc = '10'
z  = '0008'
from MainClasses import Table

tbl = Table(

    # get_files(sse_locaton, ['ga_z002/16sm/y10/sp55_d/prec/',
    #                         'ga_z004/16sm/y10/sp55_d/prec/'],
    #           ['5.00'], 'sm.data'),
    #
    # get_files('../data/sp55_w_files/', ['16z002/', '16z004/'],
    #           ['SP_16sm_y10_sp55_d_prec','SP_ga_z004_16sm_y10_sp55_d_prec'], '.data'))

    get_files(sse_locaton, ['ga_z002/20sm/y10/sp55_d/prec/',
                            'ga_z002/20sm/y9/sp55_d/prec/',
                            'ga_z002/20sm/y8/sp55_d/prec/',
                            'ga_z002/20sm/y7/sp55_d/prec/',
                            'ga_z002/20sm/y6/sp55_d/prec/',
                            'ga_z002/20sm/y5/sp55_d/prec/' ],
              ['4.00'], 'sm.data'),
    get_files('../data/sp55_w_files/', ['20z002/'], ['SP_20sm_y10_sp55_d_prec', 'SP_20sm_y9_sp55_d_prec', 'SP_20sm_y8_sp55_d_prec',
                                                     'SP_20sm_y7_sp55_d_prec', 'SP_20sm_y6_sp55_d_prec', 'SP_20sm_y5_sp55_d_prec'], '.data'))


    # get_files(sse_locaton + 'ga_z{}/'.format(z), ['{}sm/y{}/sp55_d/'.format(m,yc)],
    #           ['3.50', '4.00', '4.50', '5.00','5.50', '5.90'], 'sm.data'),
    # get_files('../data/sp55_w_files/', ['{}z{}/'.format(m,z)], [], '.data')[0])

tbl.set_use_only_spfls = True


print('\n<<< TABLE FOR : z: {} M: {} Yc: {} >>>'.format(z, m, yc))
tbl.latex_table(['Yc-1', 'l-1', 'lm-1', 'kappa-sp', 't-sp', 'r-sp', 'L/Ledd-sp', 'mfp-sp', 'u-sp', 'a_p-sp', 'tau-sp', 't-ph'], #'tau-sp', 'r-ph', 't-ph'],
                [0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.2,0.2,0.2,0.3,0.1,0.2] # 0.2, 0.2, 0.2]
                )
#
# tbl.latex_table(['mdot-', 'teff/ts4-', 'u-', 'Pg/P_total-', 'grad_u-', 'mfp/c-', 'tpar-'],
#                 [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])

# sys.exit('FUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUCK')

from MainClasses import PrintTable
# pltbl = PrintTable(get_files(sse_locaton + 'ga_z0008/', ['13sm/'], [], 'plot1'), [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
# pltbl.latex_table(['mdot', 't', 'lm'], [0.2, 0.2, 0.2])
# sys.exit('FUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUCK') # 'env_ev_pj/18sm/sp_evol/'

'''====================================================MAIN=METHODS=================================================='''

from MainClasses import GenericMethods # 'ga_z0008/15sm/y10/', 'ga_z002/15sm/y10/', 'ga_z004/15sm/y10/',
# '10sm/y10/sp55/', '15sm/y10/sp55/', '20sm/y10/sp55/','25sm/y10/sp55/','30sm/y10/sp55/'
# sm = GenericMethods(get_files(sse_locaton + 'ga_z002/', ['10sm/y10/sp55/', '11sm/y10/sp55/', '12sm/y10/sp55/', '13sm/y10/sp55/', '14sm/y10/sp55/', '15sm/y10/sp55/',], ['4.50'], 'sm.data'))
# sm.inflection_point('r','u',None)
# sm.plot_multiple_inflect_point('r', 'u', True, None, 12)


comb = Combine()

# comb.set_metal = 'gal'
comb.set_sp_files = []#select_sp_files(spfiles, [], [], []) # '4.00','4.50','5.00','5.50'
comb.set_sm_files = get_files(sse_locaton + 'ga_z004/', ['16sm/y10/sp55_d/prec/'], ['4.00', '4.50', '5.00'], 'sm.data')
comb.set_sm_files2= get_files(sse_locaton + 'ga_z002/', ['16sm/y10/sp55_d/prec/'], ['4.00', '4.50', '5.00'], 'sm.data')
# # get_files(sse_locaton + 'ga_z002/', ['10sm/y10/sp55/'], ['3.50'], 'sm.data') # MANUAL SET FOR SM FILES
# comb.set_obs_file = None
# comb.set_plot_files = get_files(sse_locaton + 'ga_z0008/', [], [], '.plot1')
#
#
# comb.set_files()



# comb.sp_xy_last_points('mdot','L/Ledd','mdot', 3)

# comb.tst()
# comb.xy_profile('t','u','mdot','lm', True, True) # Pg/P_total
# comb.xy_profile('r','L/Ledd','mdot','lm', True, True) # Pg/P_total

# comb.dxdy_profile('r', 'kappa', 'mdot', 'lm', False, True)
# comb.xyy_profile('t','Pg/P_total', 'L/Ledd','mdot', 'HP', 'mfp', True, True)
# comb.xy_last_points('r','u','mdot',False)
# comb.hrd2('l', False)
# comb.hrd('t_eff', 'lm', True, True, False)
# comb.mdot_check()
# comb.evol_mdot()
# comb.time_analysis(50)
# comb.sp_get_r_lt_table2('rho', 'lm')
# comb.save_yc_llm_mdot_cr('l')

# comb.plot_t_rho_kappa('Fe', 'mdot','xm')
# comb.plot_t_mdot_lm()
# comb.plot_t_l_mdot('lm', 0, True, False, 5.22, None)
# comb.min_mdot_sp('lm', 1.)

from PhysMath import Constants, Math
from FilesWork import PlotBackground2
def tst():
    def beta_law(r, r0, v0, vinf, beta):
        return (v0 + (vinf - v0) * ((1 - (r0 / r)) ** beta))
        # return vinf*((1 - (r0 / r))**beta)

    def diff_beta_law(r, r0, vinf, beta):
        return (r0 * vinf * beta / (r**2)) * (1 - (r0 / r)) ** (beta - 1)



    rs = 1.0
    vs = 40

    betas = np.mgrid[0.5:1.50:100j]
    vinfs = np.mgrid[1400:3000:200j]

    grads = np.zeros((len(betas), len(vinfs)))
    ass = np.zeros((len(betas), len(vinfs)))

    radii = np.mgrid[rs:(rs+1):10j]

    for i in range(len(betas)):
        for j in range(len(vinfs)):

            vels = beta_law(radii, radii[0], vs, vinfs[j], betas[i])
            grad_ = np.gradient(vels, radii * Constants.solar_r / 10**5)

            grad = diff_beta_law(radii[1:] * Constants.solar_r / 10**5, radii[0] * Constants.solar_r / 10**5, vinfs[j], betas[i]) #* np.diff(radii)[0] * Constants.solar_r / 10**5
            grad0 = interpolate.UnivariateSpline(radii[1:], grad)(radii[0])
            grad = np.append(grad0, grad)

            grads[i, j] = grad_[0] * 10**5

            ass[i, j] = np.array(vels * grad_)[0]

    print(grads)

    res_grad = Math.combine(vinfs, betas, grads)
    res_a = Math.combine(vinfs, betas, ass)

    PlotBackground2.plot_color_table(res_grad, 'v_inf', 'beta', 'grad_c', 'gal', 'Fe')
    # PlotBackground2.plot_color_table(res_a, 'v_inf', 'beta', 'a', 'gal', 'Fe')


# tst()
'''-------------------------------------------------------CRITICAL MDOT---------------------------------------------'''
from MainClasses import HRD
hrd = HRD('gal', 'Fe')
hrd.set_obs_file = 'gal_wne'
# hrd.plot_hrd('t_eff', 'lm', True)
# hrd.plot_hrd_treks('t_eff','lm')



from MainClasses import Plot_Critical_Mdot
cr = Plot_Critical_Mdot('gal', 'Fe', 1.0)
# cr.plot_natives('mdot', 'lm')
# cr.plot_cr_mdot('lm',1.0,None,None,True)
# cr.plot_cr_mdot_obs('lm',1.0,None,None,True)
# cr.plot_cr_mdot_obs_trecks('lm',1.0,None,None,True)
# cr.plot_cr_mdot_obs_trecks_back('lm', 'vinf', 1.0,None,None,True)
# cr.save_stars_affiliation()
# cr.plot_cr_3d_betas('mdot','lm','vinf', [0.9,1.0,1.1])

cr.set_fill_gray = False
# cr.plot_mult_2d_betas('mdot','lm','vinf', [0.50, 1.00, 1.15]) # | For Wind Model Fitting (Beta-law)

# --- --- ---
from MainClasses import Plot_Sonic_HRD
shrd = Plot_Sonic_HRD('gal', 'Fe', 1.0)


# shrd.plot_sonic_hrd(1.0, 'lm')
# shrd.plot_sonic_hrd_set('lm', [1.0], 1.0, 0.1)
# shrd.plot_sonic_hrd_const_r('lm', 1., [1.0])

shrd.plot_ts_y('t_eff', 1.0, 'lm', None)


from MainClasses import Plot_Two_sHRDs
shrds = Plot_Two_sHRDs(['lm', 'lm'], ['lmc', 'lmc'], ['Fe', 'HeII'], [1.0, 0.9], [None, 1.0], [1.0, 1.0])

# shrds.plot_two_shrd()

'''=================================================MULTIPLE=BUMP=METHODS============================================'''

from MainClasses import Plot_Multiple_Crit_Mdots
# pmm = Plot_Multiple_Crit_Mdots('lm')
# pmm.set_coeff = [1.0]  #[1.0, 0.8]
# pmm.set_r_cr =  [None] #[None, 1.0]
# pmm.set_bump =  ['Fe'] #['Fe', 'HeII']
# pmm.set_yc   =  [1.0]  #[1.0, 1.0]
# pmm.set_metal = ['gal']#['gal', opalfile]
# pmm.set_y_coord=['lm'] #['lm', 'lm']
#
# pmm.plot_crit_mdots(False, None, plotfiles)

from MainClasses import Plot_Tow_Sonic_HRDs
# pts = Plot_Tow_Sonic_HRDs([obsfile], [opalfile])
# pts.coeff = [1.0, 0.8]
# pts.rs =  [None, 1.0]
# pts.bump =  ['Fe', 'HeII']
# pts.yc   =  [1.0, 1.0]
# pts.opal =  [opalfile, opalfile]
# pts.y_coord=['lm', 'lm']

# pts.plot_srhd(True)

'''===========================================================3D====================================================='''
#
# from main_methods import TEST
# tst = TEST(select_sp_files(lmc_spfiles, [], ['10sm', '15sm', '20sm', '25sm', '30sm'], []), output_dir, plot_dir)
# # '10sm', '15sm', '20sm', '25sm'
# # 'y1','y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9', 'y10'
# # # # tst.sp_3d_plotting_x_y_z('t','l','r','mdot')
# tst.sp_3d_and_multiplot('t','l','r','Yc')


'''==========================================================REST===================================================='''
# from FilesWork import Read_SP_data_file
# tst = Read_SP_data_file(tst_spfiles[0])
# print(tst.get_crit_value('mdot'))
# print(tst.get_sonic_cols('mdot'))
# from __future__ import division
# import numpy as np
# import matplotlib.pyplot as plt

# def interpolated_intercept(x, y1, y2):
#     """Find the intercept of two curves, given by the same x data"""
#
#     def intercept(point1, point2, point3, point4):
#         """find the intersection between two lines
#         the first line is defined by the line between point1 and point2
#         the first line is defined by the line between point3 and point4
#         each point is an (x,y) tuple.
#
#         So, for example, you can find the intersection between
#         intercept((0,0), (1,1), (0,1), (1,0)) = (0.5, 0.5)
#
#         Returns: the intercept, in (x,y) format
#         """
#
#         def line(p1, p2):
#             A = (p1[1] - p2[1])
#             B = (p2[0] - p1[0])
#             C = (p1[0]*p2[1] - p2[0]*p1[1])
#             return A, B, -C
#
#         def intersection(L1, L2):
#             D  = L1[0] * L2[1] - L1[1] * L2[0]
#             Dx = L1[2] * L2[1] - L1[1] * L2[2]
#             Dy = L1[0] * L2[2] - L1[2] * L2[0]
#
#             x = Dx / D
#             y = Dy / D
#             return x,y
#
#         L1 = line([point1[0],point1[1]], [point2[0],point2[1]])
#         L2 = line([point3[0],point3[1]], [point4[0],point4[1]])
#
#         R = intersection(L1, L2)
#
#         return R
#
#     idx = np.argwhere(np.diff(np.sign(y1 - y2)) != 0)
#     xc, yc = intercept((x[idx], y1[idx]),((x[idx+1], y1[idx+1])), ((x[idx], y2[idx])), ((x[idx+1], y2[idx+1])))
#     return xc,yc
#
# def main():
#     x  = np.linspace(1, 4, 20)
#     y1 = np.sin(x)
#     y2 = 0.05*x
#
#     plt.plot(x, y1, marker='o', mec='none', ms=4, lw=1, label='y1')
#     plt.plot(x, y2, marker='o', mec='none', ms=4, lw=1, label='y2')
#
#     idx = np.argwhere(np.diff(np.sign(y1 - y2)) != 0)
#
#     plt.plot(x[idx], y1[idx], 'ms', ms=7, label='Nearest data-point method')
#
#     # new method!
#     xc, yc = interpolated_intercept(x,y1,y2)
#     plt.plot(xc, yc, 'co', ms=5, label='Nearest data-point, with linear interpolation')
#
#
#     plt.legend(frameon=False, fontsize=10, numpoints=1, loc='lower left')
#
#     plt.savefig('curve crossing.png', dpi=200)
#     plt.show()
#
# if __name__ == '__main__':
#     main()


# # use splines to fit and interpolate data
# from scipy.interpolate import interp1d
# from scipy.optimize import fmin
#
#
# x = np.array([ 0.,      1.,      2.,      3.,      4.    ])
# y = np.array([ 0.,     0.308,  0.55,   0.546,  0.44 ])
#
# # create the interpolating function
# f = interp1d(x, y, kind='cubic', bounds_error=False)
#
# # to find the maximum, we minimize the negative of the function. We
# # cannot just multiply f by -1, so we create a new function here.
# f2 = interp1d(x, -y, kind='cubic')
# guess = x[np.where(y == y.max())]
# xmax = fmin(f2, guess)
#
# xfit = np.linspace(0,4)
#
# plt.plot(x,y,'bo')
# plt.plot(xfit, f(xfit),'r-')
# plt.plot(xmax, f(xmax),'g*')
# plt.legend(['data','fit','max'], loc='best', numpoints=1)
# plt.xlabel('x data')
# plt.ylabel('y data')
# plt.title('Max point = ({0:1.2f}, {1:1.2f})'.format(float(xmax),
#                                                     float(f(xmax))))
# plt.show()


# x = np.array([1,2,3,4,5])
# y = np.array([3,5,5,5,5])
# z= np.array([5,6,6,6,6])
# xy = np.vstack((x,y,z))
# print(xy)
# print(xy[::-1])
# print('--------------')
# print(x[1:])
# x = np.array([0,,4,4,4])
# y = np.array([3,5,5,5,5])


# fig, ax = plt.subplots()
#
# cax = ax.imshow(P_min)#, norm=matplotlib.colors.LogNorm())#, cmap=cm.coolwarm)
# #plt.matshow(bla, norm=matplotlib.colors.LogNorm())
# cbar=fig.colorbar(cax, label='P / days')
# #cbar.ax.set_yticklabels(['','1','2','','','5','','','','','10','20','','','50','','','','90','100'])
# ax.set_title("minimal Period")
# ax.set_xlabel("M$_\mathrm{He}$ / M$_{\odot}$")
# ax.set_ylabel("M$_\mathrm{MS}$ / M$_{\odot}$")
# plt.xticks(arange(8), (2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5), size=8)
# plt.yticks(arange(13), (1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,7.5,10.0,12.5,15.0), size=8)
# ax.invert_yaxis()
#
# for i in range(0,8):
#     for j in range(0,13):
#         c = round(P_min[j][i],2)
#         ax.text(i, j, str(c), va='center', ha='center', size=6)
#
# plt.show()
# #plt.savefig('min_P.png', dpi=1000)


# tau = Opt_Depth_Analythis(20, 2000, 1.,1.,-4.522,0.20)
# print('ref tau:', tau.anal_eq_b1(1.))
# tau = Opt_Depth_Analythis(30,1600,1.,1.,-5.46,0.20)
# print('86 tau:', tau.anal_eq_b1(1.))
#
# print(Physics.logk_loglm(-0.28, 0))
# tau.kappa_test()


# for i in range(1,11):
#     print('cp y{}/fy{}.bin1 y{}/sp/;'.format(i,i,i))




folder = '/media/vnedora/HDD/sse/ga_z0004/20sm/y10/sp55/'
mdot='5.50'

from FilesWork import Read_Wind_file
wind = Read_Wind_file.from_wind_dat_file(folder + '{}.wind'.format(mdot))

from FilesWork import Read_SM_data_file
smfl = Read_SM_data_file(folder + '{}sm.data'.format(mdot))
# smfl = Read_SM_data_file.from_sm_data_file(folder + '{}sm.data'.format(mdot))

#plfl = Read_Plot_file.from_file(sse_locaton +'ga_z002/' + folder + '{}.plot1'.format(mdot), True)
#print('PLOT: R_s: {}; Tau: {}'.format(plfl.r_n_rsun[-1], plfl.tauatR[-1]))
from FilesWork import Labels
from FilesWork import Get_Z

def esc_vel(m, r):
    '''
    Escape Velocity at radius 'r' in sol_r / yr
    :param m:
    :param r:
    :return:
    '''
    g = 1.90809 * 10 ** 5  # grav const in r_sol/m_sol
    return np.sqrt(2 * g * m / r)

# def eff_kappa(r, kappa_s, r_s, )

def plot_tau(x_v_n, y_v_n, smcls, wndcls, opal_used, log=True):
    x_wind = wndcls.get_col(x_v_n)

    ax = plt.subplot(111)
    # tau_sp = 10**pltcls.tauatR[-1]
    # x_wind_sp = pltcls.get_col(x_v_n)[-1]
    tau = wndcls.get_col(y_v_n)

    # t_eff_luca = pltcls.teffluca[-1]
    # ax.axvline(x=t_eff_luca, label='T_eff(Luca)', color='green')

    if log:
        # tau_sp =  tau_sp
        ax.set_yscale("log", nonposy='clip')
        # tau = tau

    # ax.plot(x_wind_sp, tau_sp, 'x', color='black', label='Tau(.plot1)')
    ax.plot(x_wind, tau, '.', color='blue', label='log({})(w)'.format(y_v_n))

    # if opal_used!=None:
    #     lm = Physics.loglm(smcls.get_col('l')[-1], smcls.get_col('xm')[-1])
    #     tlt = 'z{}_lm{}_log(mdot){}'.format(Get_Z.z(opal_used), '%.1f' % lm, '%.2f' % smcls.get_col('mdot')[-1])
    #     plt.title(tlt)

    atm_p = wndcls.get_col('gp')
    x_wind_atm = wndcls.get_col(x_v_n)[np.int(atm_p)]
    ax.axvline(x=x_wind_atm, label='Atmosphere')

    # x_sp = smcls.get_col(x_v_n)[-1]
    # y_sp = smcls.get_col(y_v_n)[-1]
    # ax.plot(x_sp, y_sp, 'X', color='red', label='SP (sm.file)')

    ax.grid()
    ax.set_xlabel(Labels.lbls(x_v_n))
    ax.set_ylabel(Labels.lbls(y_v_n))
    # ax.set_yscale("log", nonposy='clip')
    ax.legend()
    plt.show()

def plot_core_wind(x_v_n, y_v_n, smcls, wndcls, metal, log):
    x_core = smcls.get_col(x_v_n)
    y_core = smcls.get_col(y_v_n)
    #
    # if y_v_n=='13':
    #     x_core = smcls.get_col(x_v_n)
    #     y_core = smcls.get_col('t')
    # else:
    #     x_core = smcls.get_col(x_v_n)
    #     y_core = smcls.get_col(y_v_n)

    x_wind = wndcls.get_col(x_v_n)
    y_wind = wndcls.get_col(y_v_n)

    if y_v_n == 'kappa':
        y_core = 10**smcls.get_col(y_v_n)
        y_wind2 = wndcls.get_col('kappa_eff')
        plt.plot(x_wind, y_wind2, '.', color='orange', label='kappa_eff')

    if metal!=None:
        lm = Physics.loglm(smcls.get_col('l')[-1], smcls.get_col('xm')[-1])
        tlt = 'z{}_lm{}_log(mdot){}'.format(Get_Z.z(metal), '%.1f' % lm, '%.2f' % smcls.get_col('mdot')[-1])
        plt.title(tlt)

    ax = plt.subplot(111)
    atm_p = wndcls.get_col('gp')
    x_wind_atm = wndcls.get_col(x_v_n)[np.int(atm_p)]
    # ax.axvline(x=x_wind_atm, label='Atmosphere')


    # def plot_tau(log=True):
    #     ax = plt.subplot(111)
    #     tau_sp = pltcls.tauatR[-1]
    #     x_wind_sp = pltcls.get_col(x_v_n)[-1]
    #     tau = np.log10(wndcls.get_col(y_v_n))
    #
    #     if log:
    #         tau_sp = 10**tau_sp
    #         ax.set_yscale("log", nonposy='clip')
    #         tau = 10**tau
    #
    #     ax.plot(x_wind_sp, tau_sp, 'x', color='black', label='Tau(.plot1)')
    #     ax.plot(x_wind, tau, '.', color='blue', label='log({})(w)'.format(y_v_n))
    #
    #     ax.grid()
    #     ax.set_xlabel(Labels.lbls(x_v_n))
    #     ax.set_ylabel(Labels.lbls(y_v_n))
    #     ax.set_yscale("log", nonposy='clip')
    #     ax.legend()
    #     plt.show()
    #     sys.exit(0)
    #
    # if y_v_n == 'tau':
    #     plot_tau(False)
    # ax.set_xlim(3.0, 5.5)

    # if y_v_n == '13':
    #     x_sp = smcls.get_col(x_v_n)[-1]
    #     y_sp = smcls.get_col('t')[-1]
    # else:
    x_sp = smcls.get_col(x_v_n)[-1]
    y_sp = smcls.get_col(y_v_n)[-1]

    if y_v_n == 'kappa':
        y_sp = 10**y_sp
    ax.plot(x_sp, y_sp, 'X', color='red', label='SP (sm.file)')

    if log:
        # tau_sp =  tau_sp
        ax.set_yscale("log", nonposy='clip')
        # tau = tau

    ax.plot(x_core, y_core, '.', color='red', label='{}(c)'.format(y_v_n))
    ax.plot(x_wind, y_wind, '.', color='blue', label='{}(w)'.format(y_v_n))
    ax.grid()
    ax.set_xlabel(Labels.lbls(x_v_n))
    ax.set_ylabel(Labels.lbls(y_v_n))
    ax.legend()
    plt.show()

plot_core_wind('t', 'u', smfl, wind, 'gal', False)
# plot_tau('r', 'tau', smfl, wind, opalfile, True)
# tsm = smfl.get_col('t')
#
# r_wind = suca.r/Constants.solar_r
# t_wind = np.log10(suca.t)
# tau = suca.tau
# gp = np.int(suca.gp[0])
#
# rs =  plfl.r_n_rsun[-1]
# tau_s = plfl.tauatR[-1]
# ts = plfl.t_eff[-1]
#
# plt.plot(t_wind, np.log10(tau), '-', color='black')
# plt.plot(t_wind[gp], np.log10(tau[gp]), 'x', color='black')
#
# plt.grid()
# # plt.axvline(x=plfl.r_n_rsun[-1])
# plt.plot(ts, tau_s, 'x', color='red')
# plt.show()


# print('WIND: R_eff: {}; Tau: {}'.format(rs, tau_s))
