#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------MAIN-LIBRARIES-----------------------------------------------------
import sys
from locale import format

import pylab as pl
import matplotlib.patches as patches
from matplotlib import cm
import numpy as np
from ply.ctokens import t_COMMENT
from scipy import interpolate
import re

from sklearn.linear_model import LinearRegression
# import scipy.ndimage
# from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
# from scipy.interpolate import griddata
import os
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------CLASSES-----------------------------------------------------

from PhysMath import Math, Physics, Constants
# import PhysMath as pm
from OPAL import Read_Table, Row_Analyze, Table_Analyze, OPAL_Interpol, New_Table

# ====================================================| SETTINGS CLASSES |=============================================#

# pm.Math.ind



class Files:

    output_dir   = '../data/output/'
    plot_dir     = '../data/plots/'
    sse_locaton  = '/media/vsevolod/HDD/sse/'




    def __init__(self):
        pass

    @staticmethod
    def get_obs_err_tstar(met):
        if met == 'gal':
            return 0.1
        if met == 'lmc':
            return 0.1
        raise NameError('met {} is not recognised '.format(met))

    @staticmethod
    def get_obs_err_rt(met):
        if met == 'gal':
            return 0.1
        if met == 'lmc':
            return 0.1
        raise NameError('met {} is not recognised '.format(met))

    @staticmethod
    def get_obs_err_mdot(met):
        if met == 'gal':
            return 0.15
        if met == 'lmc':
            return 0.15
        raise NameError('met {} is not recognised '.format(met))

    @staticmethod
    def get_files(compath, req_dirs, requir_files, extension):
        comb = []
        from os import listdir
        # extension = 'sm.data'
        for dir_ in req_dirs:
            for file in listdir(compath + dir_):
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
                            if req_file + extension == file:
                                # print('\t__Note: Files to be plotted: ', dir_+file)
                                comb.append(compath + dir_ + file)

        return comb

    @staticmethod
    def get_atm_file(gal_or_lmc):
        if gal_or_lmc == 'gal':  return '../data/atm/gal_atm_models.data'
        if gal_or_lmc == 'lmc':  return '../data/atm/lmc_atm_models.data'
        if gal_or_lmc == '2gal': raise NameError('No Atm. file for z=0.04 metallicity')
        if gal_or_lmc == 'smc':  raise NameError('No Atm. file for z=0.004 metallicity')
        if gal_or_lmc not in ['gal', 'lmc']: raise NameError('Unknown metallicity : {}'.format(gal_or_lmc))

    @staticmethod
    def get_obs_file(gal_or_lmc):
        if gal_or_lmc == 'gal':  return '../data/obs/gal_wne.data'
        if gal_or_lmc == 'lmc':  return '../data/obs/lmc_wne.data'
        if gal_or_lmc == '2gal': raise NameError('No Obs. file for z=0.04 metallicity')
        if gal_or_lmc == 'smc':  raise NameError('No Obs. file for z=0.004 metallicity')
        if gal_or_lmc not in ['gal', 'lmc']: raise NameError('Unknown metallicity : {}'.format(gal_or_lmc))

    @staticmethod
    def get_sp_files(gal_or_lmc, cr_or_wd, marks=list()):
        '''
        O, hi mark
        :param gal_or_lmc:
        :param cr_or_wd:
        :param mark:
        :return:
        '''

        def select_sp_files(files, mark):

            if mark != '' and mark != None:
                res = []
                for file in files:
                    fname = file.split('/')[-1]
                    if len(fname.split(mark)) > 1:
                        res.append(file)

                if len(res):
                    return res
                else:
                    raise NameError('No SP files selected out of {} with the required mark: {}'.format(len(files), mark))
            else:
                return files

        def get_sp_files_(gal_or_lmc, sm, y, marks, main_fold = '../data/SP/'):
            main_fold = '../data/SP/'

            dirs = []

            if gal_or_lmc   == '2gal': z = ['004']
            elif gal_or_lmc == 'gal':  z = ['002']
            elif gal_or_lmc == '2lmc': z = ['001']
            elif gal_or_lmc == 'lmc':  z = ['0008']
            elif gal_or_lmc == 'smc':  z = ['0004']
            else: raise NameError('gal_or_lmc: {} is not recognised.'.format(gal_or_lmc))

            for z_val in z:
                for sm_val in sm:
                    for y_val in y:
                        dirs.append('z{}/{}sm/y{}/'.format(z_val, sm_val, y_val))

            files = Files.get_files(main_fold, dirs, [], '.data')

            req_files = []

            if len(marks) > 0:
                for file in files:
                    for mark in marks:
                        if mark in file:
                            req_files.append(file)
                return req_files

            else: return files

        # HERE you can adjust what files are to be loaded by selecting [sm_array] and [y_core_array] and [marks_array]

        if gal_or_lmc == '2gal':

            if cr_or_wd == 'wd' or cr_or_wd == 'sp':
                return get_sp_files_(gal_or_lmc,
                                     [10, 11, 12, 13, 14, 15, 16], [10], marks, '../data/SP/')  # y

            if cr_or_wd == 'cr' or cr_or_wd == 'ga':
                return get_sp_files_(gal_or_lmc,
                                     [10, 11, 12, 13, 14, 15, 16],  [10, 9, 8, 7, 6 ,5, 4, 3, 2, 1], [], '../data/GA/')  # y



        if cr_or_wd == 'wd' or cr_or_wd == 'sp':
            return get_sp_files_(gal_or_lmc,
                                 [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30], # sm
                                 [10], marks, '../data/SP/') # y

        if cr_or_wd == 'cr' or cr_or_wd == 'ga':
            return get_sp_files_(gal_or_lmc,
                                 [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],  # sm
                                 [10, 9, 8, 7, 6, 5, 4, 3, 2, 1], [], '../data/GA/')  # y


        #
        # folder = ''
        #
        # if gal_or_lmc == 'gal': z = '002'
        # else: z = '0.008'
        #
        #
        # if cr_or_wd == 'cr':
        #
        # elif cr_or_wd == 'wd': folder = '../data/SP/z{}/'.format(z)
        # else: raise NameError('cr_or_wd can ne only cr or wd, given: {}'.format(cr_or_wd))
        #
        #
        #
        #
        #
        #
        #
        # gal_cr_spfiles = Files.get_files(folder, [#'7z002/',  '8z002/',  '9z002/',
        #                         '10sm/', '11sm/', '12sm/', '13sm/', '14sm/', '15sm/', '16sm/', '17sm/', '18sm/',
        #     '19sm/', '20sm/', '21sm/', '22sm/', '23sm/', '24sm/', '25sm/', '26sm/', '27sm/', '28sm/', '29sm/', '30sm/'
        #                          ], [], '.data')
        # if gal_or_lmc == 'gal': return select_sp_files(gal_cr_spfiles, mark)
        #
        # lmc_cr_spfiles = Files.get_files(folder,
        #                          ['10z0008/', '11z0008/', '12z0008/', '13z0008/', '14z0008/', '15z0008/', '16z0008/',
        #                           '17z0008/', '18z0008/', '19z0008/', '20z0008/', '21z0008/', '22z0008/', '23z0008/',
        #                           '24z0008/', '25z0008/', '26z0008/', '27z0008/', '28z0008/', '29z0008/','30z0008/',
        #                           ], [], '.data')
        # if gal_or_lmc == 'lmc': return select_sp_files(lmc_cr_spfiles, mark)
        #
        # smc_cr_spfiles = Files.get_files(folder,
        #                          ['10z0004/', '11z0004/', '12z0004/', '13z0004/', '14z0004/', '15z0004/', '16z0004/',
        #                           '17z0004/', '18z0004/', '19z0004/', '20z0004/', '21z0004/', '22z0004/', '23z0004/',
        #                           '24z0004/', '25z0004/', '26z0004/', '27z0004/', '28z0004/', '29z0004/','30z0004/',
        #                           ], [], '.data')
        # if gal_or_lmc == 'smc': return select_sp_files(smc_cr_spfiles, mark)
        #
        # gal2_cr_spfiles = Files.get_files(folder,
        #                                  ['10z004/', '11z004/', '12z004/', '13z004/', '14z004/', '15z004/', '16z004/'
        #                                   ], [], '.data')
        # if gal_or_lmc == '2gal': return select_sp_files(gal2_cr_spfiles, mark)


        raise NameError('Only gal_or_lmc avaialbale for *cr* file set, given: metallicity: {}, cr_or_wd: {}'
                        .format(gal_or_lmc, cr_or_wd))



    @staticmethod
    def get_plot_files(gal_or_lmc):
        if gal_or_lmc == 'gal':
            return Files.get_files(Files.sse_locaton + 'ga_z002/',
                                   [ '7sm/', '8sm/', '9sm/', '10sm/', '11sm/', '12sm/', '13sm/',
                                     '14sm/', '15sm/', '16sm/', '17sm/', '18sm/', '19sm/', '20sm/',
                                     '21sm/', '22sm/', '23sm/', '24sm/', '25sm/', '26sm/', '27sm/',
                                     '28sm/', '29sm/', '30sm/'], [], '.plot1')

        if gal_or_lmc == 'lmc':
            return Files.get_files(Files.sse_locaton + 'ga_z0008/',
                                   ['10sm/', '11sm/', '12sm/', '13sm/',
                                    '14sm/', '15sm/', '16sm/', '17sm/', '18sm/', '19sm/', '20sm/',
                                    '21sm/', '22sm/', '23sm/', '24sm/', '25sm/', '26sm/', '27sm/',
                                    '28sm/', '29sm/', '30sm/'], [], '.plot1')

        if gal_or_lmc == 'smc':
            return Files.get_files(Files.sse_locaton + 'ga_z0004/',
                                   ['10sm/', '11sm/', '12sm/', '13sm/',
                                    '14sm/', '15sm/', '16sm/', '17sm/', '18sm/', '19sm/', '20sm/',
                                    '21sm/', '22sm/', '23sm/', '24sm/', '25sm/', '26sm/', '27sm/',
                                    '28sm/', '29sm/', '30sm/'], [], '.plot1')

    @staticmethod
    def get_opal(gal_or_lmc):
        if gal_or_lmc == 'gal': return '../data/opal/table8.data'
        if gal_or_lmc == 'lmc': return '../data/opal/table_x.data'
        if gal_or_lmc == '2gal':return '../data/opal/table10.data'
        if gal_or_lmc == 'smc': return '../data/opal/table6.data'

        raise NameError('Only gal_or_lmc avaialbale for OPAL files, given: {},'
                        .format(gal_or_lmc))

class Fits:

    def __init__(self):
        '''

            # ----------------------- GALACTIC WNE -----------------------

            ts-lm
             (-1.617) + (1.119*x) z = 0.02
             (-1.859) + (1.157*x) z = 0.04
             (-0.556) + (0.932*x) z = 0.008 Warning No WN4 or WN5 are displayed. (wrong slope by a lot)

             (-2.688) + (1.325*x) z = 0.02  D=10
             (-1.927) + (1.174*x) z = 0.04  D=10 Warning. WN8 could not be displayed.
             (-0.735) + (0.971*x) z = 0.008 D=10 Warning No WN4 or WN5 are displayed. (wrong slope by a lot)

             --- OLD (Mdot unchanged) ---
             (-5.966) + (1.938*x) GAIA z = 0.02
             (-2.695) + (1.313*x) GAIA z = 0.04
             (-3.453) + (1.482*x) GAIA z = 0.008

             --- NEW (Mdot changed as well) ---
             (-4.726) + (1.706*x) GAIA z = 0.02
             (-4.426) + (1.637*x) GaIA z = 0.04
             (-4.110) + (1.604*x) GAIA z = 0.008

            ts-teff
             (18.816) + (-2.652*x) z = 0.02
             (19.838) + (-2.829*x) z = 0.04
             (14.713) + (-1.902*x) z = 0.008

             (16.740) + (-2.271*x) z = 0.02  D=10
             (18.503) + (-2.589*x) z = 0.04  D=10 Warning. WN8 could not be displayed.
             (8.467) +  (-0.729*x)  z = 0.008 D=10 Warning No WN4 or WN5 are displayed. (wrong slope by a lot)

             --- OLD mdot ---
             (11.471) + (-1.271*x) GAIA z = 0.02
             (18.389) + (-2.559*x) GAIA z = 0.04
             (7.046) + (-0.446*x) GAIA z = 0.008

             --- NEW mdot ---
             (7.178) + (-0.463*x) GAIA z = 0.02

             teff-lm
             (4.878) + (-0.117*x)     z = 0.02
             (6.040) + (-0.366*x_arr) z = 0.04 Expected
             (6.122) + (-0.378*x_arr) z = 0.008 Expected


            # --------------------------- LMC WNE ----------------------------

            sHRD
             (0.469) + (0.747*x) z = 0.08 Fe bump (only Fe stars)
             (1.608) + (0.560*x) z = 0.08 He bump (Fe stars cutted)

            > z = 0.08
                > sHRD
                    - HeII | (1.607) + (0.560*x)
                    - Fe   | (0.489) + (0.744*x)
                TsTeff
                    - HeII | (6.594) + (-0.360*x)
                    - Fe   | (0.796) + (0.771*x)
                critMdot
                    (-31.789) + (-26.175*x) + (-7.200*x**2) + (-0.889*x**3) + (-0.041*x**4)

            > z = 0.1   < ONLY OPAL, NO R=f(T,L/M) >
                > sHRD
                    - HeII | (2.545) + (0.363*x)
                    - Fe   | (0.055) + (0.822*x)

                > TsTeff
                    - HeII | (6.952) + (-0.437*x)
                    - Fe   | (3.908) + (0.179*x)

                > critMdot
                     (-29.576) + (-23.927*x) + (-6.411*x**2) + (-0.771*x**3) + (-0.035*x**4)

            > z = 0.004

                > sHRD
                    - HeII | (1.457) + (0.594*x)
                    - Fe   |    1 star

                > TsTeff
                    - HeII | (5.971) + (-0.231*x)
                    - Fe   |    1 star

                > critMdot
                    (44.408) + (41.164*x) + (15.146*x**2) + (2.401*x**3) + (0.140*x**4)




        '''
        pass

    @staticmethod
    def equations():

        def eq_x(a, b, c, d, teff):
            a_ = a + (b/d)*(-c)
            b_ = (b/d)
            print('RESULTED EQUATION: ({}) + ({}*x_arr)'.format("%.3f"%a_,"%.3f"% b_))
            return a_ + b_ * teff
            # return a + (b/d)*(teff-c)

        eq_x(-4.726 , 1.706 , 13.255, -1.611, None)

        eq_x(-4.726, 1.706, 7.178, -0.463, None)
        eq_x(-1.617, 1.119, 18.847, -2.658, None) # GAL Hamman, z=0.02
        eq_x(-3.453, 1.482, 7.046, -0.446, None)
        # eq_x(-2.695, 1.313, 18.389, -2.559, None)
        # eq_x(-5.966, 1.938, 11.471, -1.271,  None)

        # eq_x(-1.617,1.119,18.816,-2.652, None)  # gal
        # eq_x(-1.859,1.157,19.838,-2.829, None)  # 2gal
        # eq_x(-0.556,0.932,14.713,-1.902, None) # lmc

        eq_x(-0.735, 0.971, 8.467, -0.729, None)  # lmc D=10
        eq_x(-1.927, 1.174, 18.503, -2.589, None) # 2gal D=10
        eq_x(-2.688, 1.325, 16.740, -2.271, None) # gal D=10


    # @staticmethod
    # def get_teff_lm(x_arr, metal, clump, gaia):
    #
    #
    #
    #     if metal == 'gal_th' and clump == 4 and gaia == False:
    #         return (6.322) + (-0.422*x_arr)
    #
    #     if metal == '2gal_th' and clump == 4 and gaia == False:
    #         return (6.254) + (-0.409*x_arr)
    #
    #     if metal == 'lmc_th' and clump == 4 and gaia == False:
    #         return (6.654) + (-0.490*x_arr)
    #
    #     # clumping D=10
    #     if metal == 'gal' and clump == 10 and gaia == False:
    #         return (4.878) + (-0.117*x_arr)
    #
    #     if metal == 'gal_th' and clump == 10 and gaia == False:
    #         return (7.079) + (-0.583*x_arr)
    #
    #     if metal == '2gal_th' and clump == 10 and gaia == False:
    #         return (6.463) + (-0.453*x_arr)
    #
    #     if metal == 'lmc_th' and clump == 10 and gaia == False:
    #         return (10.543) + (-1.332*x_arr)
    #
    #     # GAIA --------------------------------------------
    #
    #     # if metal == 'gal' and clump == 4 and gaia == True:
    #     #     return (11.525) + (-1.525*x_arr)
    #
    #     if metal == 'gal_th' and clump == 4 and gaia == True:
    #         return (11.525) + (-1.525*x_arr)
    #
    #     if metal == '2gal_th' and clump == 4 and gaia == True:
    #         return (6.740) + (-0.513*x_arr)
    #
    #     if metal == 'lmc_th' and clump == 4 and gaia == True:
    #         return (19.960) + (-3.323*x_arr)
    #
    #     else:
    #         raise NameError('Fit is abailable only for *gal* (no dependance on the analysis)')
    #
    # @staticmethod
    # def get_ts_teff(x, metal, clump, gaia):
    #
    #     if metal == 'gal' and clump == 4 and gaia == False:
    #         return (18.847) + (-2.658*x)
    #
    #     if metal == '2gal' and clump == 4 and gaia == False:
    #         return (19.838) + (-2.829*x)
    #
    #     if metal == 'lmc' and clump == 4 and gaia == False:
    #         return (14.713) + (-1.902*x)
    #
    #     # ------------ CLUMPING 10 ----------------
    #
    #     if metal == 'gal' and clump == 10 and gaia == False:
    #         return (16.740) + (-2.271*x)
    #
    #     if metal == '2gal' and clump == 10 and gaia == False:
    #         return (18.503) + (-2.589*x)
    #
    #     if metal == 'lmc' and clump == 10 and gaia == False:
    #         return (8.467) + (-0.729*x)
    #
    #     # ----------- GAIA mdot ------------------------
    #
    #     if metal == 'gal' and clump == 4 and gaia == True:
    #         return (7.178) + (-0.463*x)
    #
    #     if metal == '2gal' and clump == 4 and gaia == True:
    #         return (16.409) + (-2.194*x)
    #
    #     if metal == 'lmc' and clump == 4 and gaia == True:
    #         return (1.892) + (0.531*x)
    #
    # @staticmethod
    # def get_ts_lm(x, metal, clump, gaia):
    #
    #     if metal == 'gal' and clump == 4 and gaia == False:
    #         return (-1.617) + (1.119*x)
    #
    #     if metal == '2gal' and clump == 4 and gaia == False:
    #         return (-1.859) + (1.157*x)
    #
    #     if metal == 'lmc' and clump == 4 and gaia == False:
    #         return (-0.556) + (0.932*x)
    #
    #     # Changing clumping --------------------------------------
    #     if metal == 'gal' and clump == 10 and gaia == False:
    #         return (-2.688) + (1.325*x)
    #
    #     if metal == '2gal' and clump == 10 and gaia == False:
    #         return (-1.927) + (1.174*x)
    #
    #     if metal == 'lmc' and clump == 10 and gaia == False:
    #         return (-0.735) + (0.971*x)
    #
    #     # GAIA ---------------------------------------------------
    #
    #     if metal == 'gal' and clump == 4 and gaia == True:
    #         return (-4.726) + (1.706*x)
    #
    #     if metal == '2gal' and clump == 4 and gaia == True:
    #         return (-4.426) + (1.637*x)
    #
    #     if metal == 'lmc' and clump == 4 and gaia == True:
    #         return (-4.110) + (1.604*x)

    @staticmethod
    def get_gal_fit(v_n_x, v_n_y, x_arr, metal, clump, gaia):

        # ------------------------------------------------

        def get_teff_lm(x_arr, metal, clump, gaia):

            if metal == 'gal_th' and clump == 4 and gaia == False:
                return (6.322) + (-0.422 * x_arr)

            if metal == '2gal_th' and clump == 4 and gaia == False:
                return (6.254) + (-0.409 * x_arr)

            if metal == 'lmc_th' and clump == 4 and gaia == False:
                return (6.654) + (-0.490 * x_arr)

            # clumping D=10
            if metal == 'gal' and clump == 10 and gaia == False:
                return (4.878) + (-0.117 * x_arr)

            if metal == 'gal_th' and clump == 10 and gaia == False:
                return (7.079) + (-0.583 * x_arr)

            if metal == '2gal_th' and clump == 10 and gaia == False:
                return (6.463) + (-0.453 * x_arr)

            if metal == 'lmc_th' and clump == 10 and gaia == False:
                return (10.543) + (-1.332 * x_arr)

            # GAIA --------------------------------------------

            # if metal == 'gal' and clump == 4 and gaia == True:
            #     return (11.525) + (-1.525*x_arr)

            if metal == 'gal_th' and clump == 4 and gaia == True:
                return (11.525) + (-1.525 * x_arr)

            if metal == '2gal_th' and clump == 4 and gaia == True:
                return (6.740) + (-0.513 * x_arr)

            if metal == 'lmc_th' and clump == 4 and gaia == True:
                return (19.960) + (-3.323 * x_arr)

            else:
                raise NameError('Fit is abailable only for *gal* (no dependance on the analysis)')

        def get_ts_teff(x, metal, clump, gaia):

            if metal == 'gal' and clump == 4 and gaia == False:
                return (18.847) + (-2.658 * x)

            if metal == '2gal' and clump == 4 and gaia == False:
                return (19.838) + (-2.829 * x)

            if metal == 'lmc' and clump == 4 and gaia == False:
                return (14.713) + (-1.902 * x)

            # ------------ CLUMPING 10 ----------------

            if metal == 'gal' and clump == 10 and gaia == False:
                return (16.740) + (-2.271 * x)

            if metal == '2gal' and clump == 10 and gaia == False:
                return (18.503) + (-2.589 * x)

            if metal == 'lmc' and clump == 10 and gaia == False:
                return (8.467) + (-0.729 * x)

            # ----------- GAIA ------------------------

            if metal == 'gal' and clump == 4 and gaia == True:
                return (13.255) + (-1.611*x)

            if metal == '2gal' and clump == 4 and gaia == True:
                return (16.409) + (-2.194*x)

            if metal == 'lmc' and clump == 4 and gaia == True:
                return (1.892) + (0.531*x)

        def get_ts_lm(x, metal, clump, gaia):

            if metal == 'gal' and clump == 4 and gaia == False:
                return (-1.617) + (1.119 * x)

            if metal == '2gal' and clump == 4 and gaia == False:
                return (-1.859) + (1.157 * x)

            if metal == 'lmc' and clump == 4 and gaia == False:
                return (-0.556) + (0.932 * x)

            # Changing clumping --------------------------------------
            if metal == 'gal' and clump == 10 and gaia == False:
                return (-2.688) + (1.325 * x)

            if metal == '2gal' and clump == 10 and gaia == False:
                return (-1.927) + (1.174 * x)

            if metal == 'lmc' and clump == 10 and gaia == False:
                return (-0.735) + (0.971 * x)

            # GAIA ---------------------------------------------------

            if metal == 'gal' and clump == 4 and gaia == True:
                return (-4.726) + (1.706*x)

            if metal == '2gal' and clump == 4 and gaia == True:
                return (-4.256) + (1.606*x)

            if metal == 'lmc' and clump == 4 and gaia == True:
                return (-4.104) + (1.603*x)

        def get_teff_lm_theor_fit(x, metal, clump, gaia):

            if metal == 'gal' and clump == 4 and gaia == False:
                return (6.317) + (-0.421 * x)

            if metal == 'gal' and clump == 4 and gaia == True:
                return (9.311) + (-1.059*x)

        # ------------------------------------------------

        if v_n_x == 'ts' and v_n_y == 'lm':
            return get_ts_lm(x_arr, metal, clump, gaia)

        if v_n_x == 'ts' and v_n_y == 't_eff':
            return get_ts_teff(x_arr, metal, clump, gaia)

        # if v_n_x == 't_eff' and v_n_y == 'lm':
        #     return get_teff_lm(x_arr, metal, clump, gaia)

        if v_n_x == 't_eff' and v_n_y == 'lm':
            # theoretical fits based on the ts-lm and ts-teff relations
            return get_teff_lm_theor_fit(x_arr, metal, clump, gaia)

    @staticmethod
    def get_lmc_fit(v_n_x, v_n_y, x_arr, metal, bump):

        # ------------------------------------------------

        def get_teff_lm(x, metal, clump, bump, gaia):

            if bump == 'Fe':
                if metal == 'lmc_th' and clump == 10 and gaia == False:
                    pass

                if metal == '2lmc_th' and clump == 10 and gaia == False:
                    pass

                if metal == 'smc_th' and clump == 10 and gaia == False:
                    pass

            if bump == 'HeII':
                if metal == 'lmc_th' and clump == 10 and gaia == False:
                    pass

                if metal == '2lmc_th' and clump == 10 and gaia == False:
                    pass

                if metal == 'smc_th' and clump == 10 and gaia == False:
                    pass


        def get_ts_teff(x, metal, clump, bump, gaia):

            if bump == 'Fe':
                if metal == 'lmc' and clump == 10 and gaia == False:
                    return (0.796) + (0.771*x)

                if metal == '2lmc' and clump == 10 and gaia == False:
                    return (3.908) + (0.179*x)

                if metal == 'smc' and clump == 10 and gaia == False:
                    return np.empty(len(x))

            if bump == 'HeII':
                if metal == 'lmc' and clump == 10 and gaia == False:
                    return (6.594) + (-0.360*x)

                if metal == '2lmc' and clump == 10 and gaia == False:
                    return (6.952) + (-0.437*x)

                if metal == 'smc' and clump == 10 and gaia == False:
                    return (5.971) + (-0.231*x)


        def get_ts_lm(x, metal, clump, bump, gaia):

            if bump == 'Fe':
                if metal == 'lmc' and clump == 10 and gaia == False:
                    return (0.489) + (0.744*x)

                if metal == '2lmc' and clump == 10 and gaia == False:
                    return (0.055) + (0.822*x)

                if metal == 'smc' and clump == 10 and gaia == False:
                    return np.empty(len(x))

            if bump == 'HeII':
                if metal == 'lmc' and clump == 10 and gaia == False:
                    return (1.607) + (0.560*x)

                if metal == '2lmc' and clump == 10 and gaia == False:
                    return (2.545) + (0.363*x)

                if metal == 'smc' and clump == 10 and gaia == False:
                    return (1.457) + (0.594*x)


        def get_crit_mdot_lm(x, metal, bump):

            if bump == 'Fe':
                if metal == 'lmc':
                    return (-31.789) + (-26.175*x) + (-7.200*x**2) + (-0.889*x**3) + (-0.041*x**4)
                if metal == '2lmc':
                    return (-29.576) + (-23.927*x) + (-6.411*x**2) + (-0.771*x**3) + (-0.035*x**4)
                if metal == 'smc':
                    return (44.408) + (41.164*x) + (15.146*x**2) + (2.401*x**3) + (0.140*x**4)
        # ------------------------------------------------

        gaia = False
        clump = 10

        if v_n_x == 'ts' and v_n_y == 'lm':
            return get_ts_lm(x_arr, metal, clump, bump, gaia)

        if v_n_x == 'ts' and v_n_y == 't_eff':
            return get_ts_teff(x_arr, metal, clump, bump, gaia)

        if v_n_x == 't_eff' and v_n_y == 'lm':
            return get_teff_lm(x_arr, metal, clump, bump, gaia)

        if v_n_x == 'mdot_cr' and v_n_y == 'lm':
            return get_crit_mdot_lm(x_arr, metal, bump)


class Affiliation:
    '''
    returns
    '''
    def __init__(self):
        pass

    @staticmethod
    def get_list(metal, bump):
        '''
        Returns True in star is in [  7.  17.  24.  26.  65.  88.  94. 131. 132.] list
        for 'lmc' metal and 'Fe' bum
        :param star_n:
        :return:
        '''

        lmc2_fe_list = [7,  15,  17,  24,  26,  37,  41,  56,  65,  88,  94, 131, 132]
        lmc2_he_list = [1,   2,   3,  5,  23,  46,  48,  51,  57,  75,  86, 124, 128, 134]
        lmc_fe_list = [7,  17,  24,  26,  65,  88,  94, 131, 132]
        lmc_he_list = [1,   2,   3,   5,  15,  23,  37,  41,  46,  48,  51,  56,  57,  75, 86, 124, 128, 134]

        if metal == 'lmc' and bump == 'Fe':
            return lmc_fe_list # lmc_fe_list
        elif metal == 'lmc' and bump == 'HeII':
            return lmc_he_list # lmc_he_list

        else:
            raise ValueError('No affiliation list for metal: {} bumb: {}'.format(metal, bump))


class Levels:

    def __init__(self):
        pass

    @staticmethod
    def get_levels(v_n, opal_used, bump):

        if bump == 'gen' and opal_used == 'lmc':

            if v_n == 'grad_c' or v_n == 'grad_w' or v_n == 'grad_w_p' or v_n == 'grad_c_p':
                return [10, 50, 100, 200, 400, 800, 1000, 2000, 3000, 4000, 5000]

            if v_n == 'a_p' or v_n == 'a':
                return [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

            if v_n == 'vinf':
                # return [1.,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0]
                return [1.4, 1.8, 2.2, 2.6, 3.0]


        if opal_used.split('/')[-1] == 'gal':

            if v_n == 'r':
                return [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6]
                             # 1.5, 1.6, 1.7, 1.8, 1.9, 2., 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,4.0
            if v_n == 'm':
                levels = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
            if v_n == 'mdot':
                return [-5.5, -5.25, -5., -4.75, -4.5, -4.25, -4, -3.75, -3.5, -3.25, -3.]
                # levels = [-6.0, -5.9, -5.8, -5.7, -5.6, -5.5, -5.4, -5.3, -5.2, -5.1, -5., -4.9, -4.8, -4.7, -4.6, -4.5]
            if v_n == 'l':
                return [5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2]
            if v_n == 'lm':
                return [4.0, 4.05, 4.1, 4.15, 4.2, 4.25, 4.3, 4.35, 4.4, 4.45, 4.50, 4.55]
                          # 4.6, 4.65, 4.7, 4.75, 4.8, 4.85, 4.9, 4.95, 5.0]
            if v_n == 't':
                return [5.15, 5.16, 5.17, 5.18, 5.19, 5.20, 5.21, 5.22, 5.23, 5.24, 5.25, 5.26, 5.27, 5.28, 5.29, 5.30]

            if v_n == 'k':
                # return [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]  # FOR log Kappa
                return [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2]  # FOR log Kappa
            if v_n == 'rho':
                return [-10, -9.5, -9, -8.5, -8, -7.5, -7, -6.5, -6, -5.5] # , -5, -4.5, -4
            # if v_n_z == 'r':   levels = [0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6,
            #                            1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0, 2.05, 2.10, 2.15, 2.20]
            if v_n == 'tau': #
                return [0, 10, 20, 40, 80]
            # [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]

            if v_n == 'm':
                return [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

        if opal_used.split('/')[-1]== '2gal':

            if v_n == 'r':
                return [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8]
                             # 1.5, 1.6, 1.7, 1.8, 1.9, 2., 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,4.0
            if v_n == 'm':
                levels = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
            if v_n == 'mdot':
                return [-5.5, -5.25, -5., -4.75, -4.5, -4.25, -4, -3.75, -3.5, -3.25, -3.]
                # levels = [-6.0, -5.9, -5.8, -5.7, -5.6, -5.5, -5.4, -5.3, -5.2, -5.1, -5., -4.9, -4.8, -4.7, -4.6, -4.5]
            if v_n == 'l':
                return [5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2]
            if v_n == 'lm':
                return [4.0, 4.05, 4.1, 4.15, 4.2, 4.25, 4.3, 4.35, 4.4]
                          # 4.6, 4.65, 4.7, 4.75, 4.8, 4.85, 4.9, 4.95, 5.0]
            if v_n == 't':
                return [5.15, 5.16, 5.17, 5.18, 5.19, 5.20, 5.21, 5.22, 5.23, 5.24, 5.25, 5.26, 5.27, 5.28, 5.29, 5.30]

            if v_n == 'k':
                # return [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]  # FOR log Kappa
                return [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2]  # FOR log Kappa
            if v_n == 'rho':
                return [-10, -9.5, -9, -8.5, -8, -7.5, -7, -6.5, -6, -5.5] # , -5, -4.5, -4
            # if v_n_z == 'r':   levels = [0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6,
            #                            1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0, 2.05, 2.10, 2.15, 2.20]
            if v_n == 'tau':
                return [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
            if v_n == 'm':
                return [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

        if opal_used.split('/')[-1] == 'lmc' or opal_used.split('/')[-1] == 'smc':

            # if bump == 'HeII':
            #     if v_n == 'mdot':
            #         return [-6.0, -5.75, -5.5, -5.25, -5., -4.75, -4.5]

            if v_n == 'r':
                return [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2., 2.1]
            if v_n == 'm':
                levels = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
            if v_n == 'mdot':
                return [-6.0, -5.75, -5.5, -5.25, -5., -4.75, -4.5, -4.25, -4, -3.75, -3.5] # , -4.25, -4, -3.75, -3.5
                # levels = [-6.0, -5.9, -5.8, -5.7, -5.6, -5.5, -5.4, -5.3, -5.2, -5.1, -5., -4.9, -4.8, -4.7, -4.6, -4.5]
            if v_n == 'l':
                return [5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4]
            if v_n == 'lm':
                return [4.0, 4.05, 4.1, 4.15, 4.2, 4.25, 4.3, 4.35, 4.4, 4.45,
                          4.5, 4.55, 4.6, 4.65, 4.7, 4.75, 4.8, 4.85, 4.9, 4.95, 5.0]
            if v_n == 't':
                return [5.15, 5.16, 5.17, 5.18, 5.19, 5.20, 5.21, 5.22, 5.23, 5.24, 5.25, 5.26, 5.27, 5.28, 5.29, 5.30]

            if v_n == 'k':
                # return [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]  # FOR log Kappa
                return [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2]  # FOR log Kappa
            if v_n == 'rho':
                return [-10, -9.5, -9, -8.5, -8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4]
            # if v_n_z == 'r':   levels = [0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6,
            #                            1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0, 2.05, 2.10, 2.15, 2.20]

            if v_n == 't_eff':
                return [4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1]

            if v_n == 'tau':
                return [2, 4, 8, 16, 32]
                # return [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]

            if bump == 'HeII':
                if v_n == 'a_p' or v_n == 'a':
                    return [0.02, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]

                if v_n == 'grad_c' or v_n == 'grad_w' or v_n == 'grad_w_p' or v_n == 'grad_c_p':
                    return [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]

        if v_n == 'r':
            return [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2., 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,4.0]

        if v_n == 'm':
            return [5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

        if v_n == 'lm':
            return [4.0, 4.05, 4.1, 4.15, 4.2, 4.25, 4.3, 4.35, 4.4, 4.45,
                          4.5, 4.55, 4.6, 4.65, 4.7, 4.75, 4.8, 4.85, 4.9, 4.95, 5.0]

        if v_n == 'vrho': return [-4.4, -4.0, -3.6, -3.2, -2.8, -2.4, -2.0, -1.6, -1.2, -0.8, -0.4, 0.4]
            # return [0.4, 0.0, -0.4, -0.8, -1.2, -1.6, -2.0, -2.4, -2.8, -3.2, -3.6, -4.0, -4.4]

        if v_n == 'Ys' or v_n == 'ys':
            return [0.5, 0.55, 0.6, 0.65, 0.7,0.75,0.8,0.85,0.9,0.95,1.0]

        if v_n == 'm_env':
            return [-10.0,-9.9,-9.8,-9.7,-9.6,-9.5,-9.4,-9.3,-9.2,-9.1,-9.0]
        if v_n == 'r_env':
            return [0.0025,0.0050,0.0075,0.01,0.0125,0.0150,0.0175]

        if v_n == 't_eff':
            return [4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4]

        if v_n == 'r_eff':
            return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,11 ,12, 13, 14 , 16, 18, 20]

        if v_n == 'log_tau':
            return [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]

        if v_n == 'grad_c' or v_n == 'grad_w' or v_n == 'grad_w_p' or v_n == 'grad_c_p':
            return [10, 50, 100, 200, 400, 600, 800, 1000, 1200]

        if v_n == 'L/Ledd':
            return [0.8,0.9,0.95,0.98, 0.99, 1.00, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07]

        if v_n == 'vinf':
            # return [1.,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0]
            return [1.4, 1.8, 2.2, 2.6, 3.0]
            #return [1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000]

        if v_n == 'a_p' or v_n == 'a':
            return [0.02, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30]

        raise NameError('Levels are not found for <{}> Opal:{}'.format(v_n, opal_used))

class Labels:
    def __init__(self):
        pass

    @staticmethod
    def lbls(v_n):
        #solar
        if v_n == 'l':
            return r'$\log(L/L_{\odot})$'#(L_{\odot})

        if v_n == 'lgaia':
            return r'$\log(L_{GAIA}/L_{\odot})$'#(L_{\odot})

        if v_n == 'r':
            return r'$R(R_{\odot})$'

        if v_n == 'm' or v_n == 'xm':
            return r'$M(M_{\odot})$'

        #sonic and general
        if v_n == 'v' or v_n == 'u':
            return 'v (km/s)'

        if v_n == 'rho':
            return r'$\log(\rho)$'

        if v_n == 'k' or v_n == 'kappa':
            return r'$\kappa$'

        if v_n == 't':
            return r'log(T/K)'

        if v_n == 'ts':
            return r'$\log(T_{s}/K)$'

        if v_n == 'rs':
            return r'$\log(R_{s}/R_{\odot})$'

        if v_n == 'lm':
            return r'$\log(L/M)$'

        if v_n == 'mdot':
            return r'$\log(\dot{M}$)'

        if v_n == 'Yc' or v_n == 'yc':
            return r'$^{4}$He$_{core}$'

        if v_n == 'He4':
            return r'$^{4}$He$_{surf}$'

        if v_n == 'Ys' or v_n == 'ys':
            return r'$^{4}$He$_{surf}$'

        if v_n == 't_eff' or v_n == 'T_eff':
            return r'$\log($T$_{eff}/K)$'

        if v_n == 't_*' or v_n == 'T_*':
            return r'$\log($T$_{*}$/K$)$'

        if v_n == 'r_eff' or v_n == 'R_eff':
            return r'$\log($R$_{eff}/R_{\odot})$'

        if v_n == 'rho':
            return r'$\log(\rho)$'

        if v_n == 'tau':
            return r'$\tau$'


        if v_n == 'Pr':
            return r'$P_{rad}$'

        if v_n == 'Pg':
            return r'$P_{gas}$'

        if v_n == 'Pg/P_total':
            return r'$P_{gas}/P_{total}$'

        if v_n == 'Pr/P_total':
            return r'$P_{rad}/P_{total}$'


        if v_n == 'mfp':
            return r'$\log(\lambda)$'

        if v_n == 'HP' or v_n == 'Hp':
            return r'$H_{p}$'

        if v_n == 'L/Ledd':
            return r'$L/L_{Edd}$'

        if v_n == 'delta_t':
            return r'$t_i - ts_i$'

        if v_n == 'delta_u':
            return r'$u_i - us_i$'

        if v_n == 'delta_grad_u':
            return r'$\nabla_{r<r_s} - \nabla_{r>r_s}$'

        if v_n == 'rt':
            return r'$\log(R_t/R_{\odot})$'


        if v_n == 'r_infl':
            return r'$R_{inflect} (R_{\odot})$'

        if v_n == 'z':
            return r'$z$'

        if v_n == 'log_tau':
            return r'$\log(\tau)$'

        if v_n == 'vinf' or v_n == 'v_inf':
            return r'$v_{\inf}\cdot 10^3$ (km/s)'

        if v_n == 'a' or v_n == 'a_p':
            return  r'$a$ (km/s$^2$)'

        if v_n == 'beta' or v_n == 'b':
            return r'$\beta$'


        if v_n == 'grad_c' or v_n == 'grad_c_p':
            return r'$\nabla u \cdot 10^5$ $(c^{-1})$'

class Get_Z:
    def __init__(self):
        pass

    @staticmethod
    def z(metal):
        if metal.split('/')[-1] == 'gal':
            return 0.02
        if metal.split('/')[-1] == 'lmc':
            return 0.008
        if metal.split('/')[-1] == '2gal':
            return 0.04
        if metal.split('/')[-1] == 'smc':
            return 0.004
        raise NameError('Metallicity is not recognised: (given: {})'.format(metal))

class T_kappa_bump:


    def __init__(self):
        pass

    @staticmethod
    def t_for_bump(bump_name):
        '''
        Returns t1, t2 for the opacity bump, corresponding to a Fe or HeII increase
        :param bump_name:
        :return:
        '''

        t_fe_bump1  = 5.19# 5.21 # 5.18
        t_fe_bump2  = 5.45

        t_he2_bump1 = 4.65
        t_he2_bump2 = 5.00

        if bump_name == 'HeII':
            return t_he2_bump1, t_he2_bump2

        if bump_name == 'Fe':
            return t_fe_bump1, t_fe_bump2

        raise NameError('Incorrect bump_name. Opacity bumps availabel: {}'.format(['HeII', 'Fe']))

# ========================================================| GENERAL |==================================================#

class Save_Load_tables:
    def __init__(self):
        pass

    @staticmethod
    def save_table(d2arr, metal, bump, name, x_name, y_name, z_name, output_dir ='../data/output/'):

        header = np.zeros(len(d2arr)) # will be first row with limtis and
        # header[0] = x1
        # header[1] = x2
        # header[2] = y1
        # header[3] = y2
        # tbl_name = 't_k_rho'
        # op_and_head = np.vstack((header, d2arr))  # arraching the line with limits to the array

        part = '_' + bump + '_' + metal
        full_name = output_dir + name + '_' + part + '.data'  # dir/t_k_rho_table8.data

        np.savetxt(full_name, d2arr, '%.4f', '  ', '\n',
                   '\nINTERPOLATED OPAL {} TABLE for {} relation'.format(part, name), '',
                   '# {} | {} | {} | {} |'
                   .format('_' + bump + '_' + metal, x_name, y_name, z_name))

        # np.savetxt(full_name, d2arr, '%.4f', '  ', '\n',
        #            '\nINTERPOLATED OPAL {} TABLE for {} relation'.format(part, name), '',
        #            '# {} | {} {} {} | {} {} {} | {} | {} | {}'
        #            .format(opal_used, x_name, x1, x2, y_name, y1, y2, z_name, n_int, n_out))

    @staticmethod
    def load_table(name, x_name, y_name, z_name, metal, bump, dir ='../data/output/'):
        part =  '_' + bump + '_' + metal
        full_name = dir + name + '_' + part + '.data'

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

        if r_table != '_' + bump + '_' + metal:
            raise NameError('Read OPAL | {} | not the same is opal_used | {} |'.format(r_table,  '_' + bump + '_' + metal))

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

        return np.array(file_table, dtype='float64') #[x1, x2, y1, y2, n1, n2]

    @staticmethod
    def save_3d_table(d3array, metal, bump, name, t_name, x_name, y_name, z_name, output_dir ='../data/output/'):
        i = 0

        part =  '_' + bump + '_' + metal
        full_name = output_dir + name + '_' + part  + '.data'  # dir/t_k_rho_table8.data

        # np.savetxt(full_name, d2arr, '%.4f', '  ', '\n',
        #            '\nINTERPOLATED OPAL {} TABLE for {} relation'.format(part, name), '',
        #            '# {} | {} | {} | {} |'
        #            .format(opal_used, x_name, y_name, z_name))

        with open(full_name, 'w') as outfile:
            # I'm writing a header here just for the sake of readability
            # Any line starting with "#" will be ignored by numpy.loadtxt
            # outfile.write('# Array shape: {0}\n'.format(d3array.shape))
            outfile.write(
                '# {} | {} | {} | {} | {} | {} | \n'.format(d3array.shape, t_name, x_name, y_name, z_name,  '_' + bump + '_' + metal))

            # Iterating through a ndimensional array produces slices along
            # the last axis. This is equivalent to data[i,:,:] in this case
            for data_slice in d3array:
                # The formatting string indicates that I'm writing out
                # the values in left-justified columns 7 characters in width
                # with 2 decimal places.
                np.savetxt(outfile, data_slice, '%.4f', '  ', '\n',
                           '\n# {}:{} | {} | {} | {} | {}\n'.format(t_name, data_slice[0,0], x_name, y_name, z_name,  '_' + bump + '_' + metal), '',
                           '')
                # np.savetxt(outfile, data_slice, fmt='%-7.2f')

                # Writing out a break to indicate different slices...
                # outfile.write('# \n')
                i = i + 1

    @staticmethod
    def load_3d_table(metal, bump, name, t_name, x_name, y_name, z_name, output_dir='../data/output/'):
        '''
                            RETURS a 3d table
        :param metal:
        :param name:
        :param t_name:
        :param x_name:
        :param y_name:
        :param z_name:
        :param output_dir:
        :return:
        '''


        part = '_' + bump + '_' + metal
        full_name = output_dir + name + '_' + part + '.data'

        with open(full_name, 'r') as f: # reads only first line (to save time)
            first_line = f.readline()


        first_line = first_line.split('# ')[1] # get rid of '# '
        r_shape = first_line.split(' | ')[0]
        r_t_v_n = first_line.split(' | ')[1]
        r_x_v_n = first_line.split(' | ')[2]
        r_y_v_n = first_line.split(' | ')[3]
        r_z_v_n = first_line.split(' | ')[4]
        r_opal  = first_line.split(' | ')[5]

        # --- Checks for opal (metallicity) and x,y,z,t, var_names.

        if r_opal !=  '_' + bump + '_' + metal:
            raise NameError('Read OPAL <{}> not the same is opal_used <{}>'.format(r_opal,  '_' + bump + '_' + metal))
        if t_name != r_t_v_n:
            raise NameError('Provided x_name: {} not equal to table x_name: {}'.format(t_name, r_t_v_n))
        if x_name != r_x_v_n:
            raise NameError('Provided x_name: {} not equal to table x_name: {}'.format(x_name, r_x_v_n))
        if y_name != r_y_v_n:
            raise NameError('Provided x_name: {} not equal to table x_name: {}'.format(y_name, r_y_v_n))
        if z_name != r_z_v_n:
            raise NameError('Provided x_name: {} not equal to table x_name: {}'.format(z_name, r_z_v_n))

        # --- --- Actual load of a table (as a 2d array) --- --- ---

        d2_table = np.loadtxt(full_name)

        # --- --- Reshaping the 2d arrays into 3d Array --- --- ---

        from ast import literal_eval as make_tuple # to convert str(n, n, n) into a tuple(n, n, n)
        shape = make_tuple(r_shape)

        d3_table = d2_table.reshape(shape)

        print('\t__Table {} is read succesfully. Shape is {}'.format(full_name, d3_table.shape))

        return d3_table

    @staticmethod
    def read_genergic_table(file_name, str_col=None):
        '''
        Reads the the file table, returning the list with names and the table
        structure: First Row must be with '#' in the beginning and then, the var names.
        other Rows - table with the same number of elements as the row of var names
        :return:
        :str_col: Name of the volumn with string values
        '''
        table = []
        with open(file_name, 'r') as f:
            for line in f:
                if '#' not in line.split() and line.strip():  # if line is not empty and does not contain '#'
                    table.append(line)

        names = table[0].split()[:]  # getting rid of '#' which is the first element in a row
        num_colls = len(table) - 1  # as first row is of var names

        if len(names) != len(table[1].split()):
            print('\t___Error. Number of vars in list({}) != number of cols in observ.data file({}) '
                  '|Read_Observables, __init__|'.format(len(names), len(table[1].split())))
        print('\t__Note: Data include following paramters:\n\t | {} |'.format(names))

        table.remove(table[0])  # removing the var_names line from the array. (only actual values left)

        if str_col==None:
            tmp = np.zeros(len(names))
            for row in table:
                tmp = np.vstack((tmp, np.array(row.split(), dtype=np.float)))
            table = np.delete(tmp, 0, 0)

            return names, table

        else:
            # In case there is a column with strings in a table.
            # Remove it from the table. Vstack the rest of the table. Remove it name from the 'names'
            # Return all of it
            tmp = np.zeros(len(names) - 1)
            str_col_val = []
            for row in table:
                raw_row = row.split()
                str_col_val.append(raw_row[names.index(str_col)])
                raw_row.remove(raw_row[names.index(str_col)])
                tmp = np.vstack((tmp, np.array(raw_row, dtype=np.float)))

            table = np.delete(tmp, 0, 0)
            names.remove(str_col)

            return names, table, str_col_val


# =========================================================| OAPL |====================================================#

class OPAL_work:

    def __init__(self, metal, bump, n_interp = 1000, load_lim_cases = False,
                 output_dir = '../data/output/', plot_dir = '../data/plots/'):

        self.set_metal = metal
        self.op_name = Files.get_opal(metal)
        self.bump = bump
        self.t1, self.t2 = T_kappa_bump.t_for_bump(bump)
        self.n_inter = n_interp
        self.set_plots_clean = False

        self.out_dir = output_dir
        self.plot_dir = plot_dir

        self.opal = OPAL_Interpol(Files.get_opal(metal), n_interp)
        self.tbl_anl = Table_Analyze(Files.get_opal(metal), n_interp, load_lim_cases, output_dir, plot_dir)

    def save_t_rho_k(self, rho1 = None, rho2=None, t1=None, t2=None):
        if t1==None: t1 = self.t1
        if t2==None: t2 = self.t2

        op_cl = self.opal
        t1, t2, rho1, rho2 = op_cl.check_t_rho_limits(t1, t2, rho1, rho2)
        op_table = op_cl.interp_opal_table(t1, t2, rho1, rho2)

        Save_Load_tables.save_table(op_table, self.set_metal, self.bump, 't_rho_k', 't', 'rho', 'k', self.out_dir)

    def save_t_k_rho(self, llm1=None, llm2=None, n_out = 1000):

        k1, k2 = Physics.get_k1_k2_from_llm1_llm2(self.t1, self.t2, llm1, llm2) # assuming k = 4 pi c G (L/M)

        global t_k_rho
        t_k_rho = self.tbl_anl.treat_tasks_interp_for_t(self.t1, self.t2, n_out, self.n_inter, k1, k2).T

        t_k_rho__ = Math.combine(t_k_rho[0,1:], 10**t_k_rho[1:,0], t_k_rho[1:,1:])

        lbl = 'z:{}'.format(Get_Z.z(self.set_metal))
        if self.set_plots_clean: lbl = None

        PlotBackground.plot_color_table(t_k_rho__, 't', 'k', 'rho', self.set_metal, lbl)

        Save_Load_tables.save_table(t_k_rho, self.set_metal, self.bump, 't_k_rho', 't', 'k', 'rho', self.out_dir)
        print('\t__Note. Table | t_k_rho | has been saved in {}'.format(self.out_dir))
        # self.read_table('t_k_rho', 't', 'k', 'rho', self.op_name)
        # def save_t_llm_vrho(self, llm1=None, llm2=None, n_out = 1000):

    def from_t_k_rho__to__t_lm_rho(self, coeff = 1.0):
        t_k_rho = Save_Load_tables.load_table('t_k_rho', 't', 'k', 'rho', self.set_metal, self.bump)

        t = t_k_rho[0, 1:]
        k = t_k_rho[1:, 0]
        rho2d = t_k_rho[1:, 1:]

        lm = Physics.logk_loglm(k, 1, coeff)

        t_lm_rho = Math.invet_to_ascending_xy( Math.combine(t, lm, rho2d) )

        Save_Load_tables.save_table(t_lm_rho, self.set_metal, self.bump, 't_{}lm_rho'.format(coeff), 't', '{}lm'.format(coeff), 'rho', self.out_dir)
        print('\t__Note. Table | t_lm_rho | for | {} | has been saved in {}'.format(self.op_name, self.out_dir))

    def save_t_llm_vrho(self, l_or_lm_name):
        '''
        Table required: t_k_rho (otherwise - won't work) [Run save_t_k_rho() function ]
        :param l_or_lm_name:
        :return:
        '''

        # 1 load the t_k_rho
        t_k_rho = Save_Load_tables.load_table('t_k_rho', 't', 'k', 'rho', self.set_metal, self.bump)

        k = t_k_rho[0, 1:]
        t = t_k_rho[1:, 0]
        rho2d = t_k_rho[1:, 1:]

        vrho = Physics.get_vrho(t, rho2d.T, 2) # mu = 1.34 by default | rho2d.T is because in OPAL t is Y axis, not X.

        # ----------------------------- SELECT THE Y AXIS -----------------
        if l_or_lm_name == 'l':
            l_lm_arr = Physics.lm_to_l_langer(Physics.logk_loglm(k, True))  # Kappa -> L/M -> L
        else:
            l_lm_arr = Physics.logk_loglm(k, 1)


        l_lm_arr = np.flip(l_lm_arr, 0)  # accounting for if k1 > k2 the l1 < l2 or lm1 < lm2
        vrho     = np.flip(vrho, 0)

        global t_llm_vrho
        t_llm_vrho = Math.combine(t, l_lm_arr, vrho)
        name = 't_'+ l_or_lm_name + '_vrho'

        Save_Load_tables.save_table(t_llm_vrho, self.set_metal, self.bump, name, 't', l_or_lm_name, '_vrho', self.out_dir)

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
        t_llm_vrho = Save_Load_tables.load_table(fname, 't', l_or_lm, '_vrho', self.set_metal, self.bump)
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
        Save_Load_tables.save_table(t_llm_mdot, self.set_metal, 't_' + l_or_lm + '_mdot', 't', l_or_lm, 'mdot', self.out_dir)

    def plot_t_rho_kappa(self, metal, bump, ax=None, fsz=12):

        show_plot = False
        if ax == None:  # if the plotting class is not given:
            fig = plt.figure()
            # fig.subplots_adjust(hspace=0.2, wspace=0.3)
            ax = fig.add_subplot(1, 1, 1)
            show_plot = True

        t_rho_k = Save_Load_tables.load_table('t_rho_k','t','rho','k', metal, bump, self.out_dir)

        PlotBackground.plot_color_background(ax, t_rho_k, 't', 'rho', 'k', metal, 'z:{}'.format(Get_Z.z(metal)), 1.0, True, 12, 0)


        if show_plot:
            ax.text(0.95, 0.05, 'PRELIMINARY', fontsize=50, color='gray', ha='right', va='bottom', alpha=0.5)
            ax.legend(bbox_to_anchor=(1, 0), loc='lower right', ncol=1, fontsize=fsz)
            plot_name = Files.plot_dir + 't_rho_k__{}_{}.pdf'.format(self.bump, metal)
            ax.set_xlabel(Labels.lbls('t'), fontsize=fsz)
            ax.set_ylabel(Labels.lbls('rho'), fontsize=fsz)
            # plt.grid()
            plt.xticks(fontsize=fsz)
            plt.yticks(fontsize=fsz)
            plt.savefig(plot_name)
            plt.show()
        else:
            return ax


# =======================================================| SP FILES |==================================================#

class Read_SP_data_file:

    list_of_names = ['log(L)', 'M(Msun)', 'Yc', 'mdot', 'r-sp' ,'t-sp', 'Tau', 'R_wind', 'T_eff', 'kappa-sp', 'L/Ledd-sp', 'rho-sp']

    # log(L)  M(Msun)     Yc     Ys      l(Mdot)   Rs(Rsun) log(Ts) kappa-sp L/Ledd-sp rho-sp

    def __init__(self, sp_data_file, out_dir = '../data/output/', plot_dir = '../data/plots/'):

        self.files = sp_data_file
        self.out_dir = out_dir
        self.plot_dir = plot_dir

        self.names, self.table = self.read_genergic_table(sp_data_file) # now it suppose to read the table

        print('')

        # self.list_of_v_n = ['l', 'm', 't', 'mdot', 'tau', 'r', 'Yc', 'k']

        # self.table = []
        # with open(sp_data_file, 'r') as f:
        #     for line in f:
        #         # if '#' not in line.split() and line.strip():  # if line is not empty and does not contain '#'
        #         self.table.append(line)
        #
        # self.names = self.table[0].split()[1:] # getting rid of '#' which is the first element in a row
        # self.num_colls = len(self.table) - 1  # as first row is of var names
        #
        # if len(self.names) != len(self.table[1].split()):
        #     print('\t___Error. Number of vars in list({}) != number of cols in observ.data file({}) '
        #           '|Read_Observables, __init__|'.format(len(self.names), len(self.table[1].split())))
        # print('\t__Note: Data include following paramters:\n\t | {} |'.format(self.names))
        #
        # self.table.remove(self.table[0])  # removing the var_names line from the array. (only actual values left)
        #
        # tmp = np.zeros(len(self.names))
        # for row in self.table:
        #     tmp = np.vstack((tmp, np.array(row.split(), dtype=np.float)))
        # self.table = np.delete(tmp, 0, 0)
        #
        # print('File: {} has been loaded successfully.'.format(sp_data_file))

    @staticmethod
    def read_genergic_table(file_name):
        '''
        Reads the the file table, returning the list with names and the table
        structure: First Row must be with '#' in the beginning and then, the var names.
        other Rows - table with the same number of elements as the row of var names
        :return:
        '''
        table = []
        with open(file_name, 'r') as f:
            for line in f:
                # if '#' not in line.split() and line.strip():  # if line is not empty and does not contain '#'
                table.append(line)

        names = table[0].split()[1:]  # getting rid of '#' which is the first element in a row
        num_colls = len(table) - 1  # as first row is of var names

        if len(names) != len(table[1].split()):
            print('\t___Error. Number of vars in list({}) != number of cols in observ.data file({}) '
                  '|Read_Observables, __init__|'.format(len(names), len(table[1].split())))
        print('\t__Note: Data include following paramters:\n\t | {} |'.format(names))

        table.remove(table[0])  # removing the var_names line from the array. (only actual values left)

        tmp = np.zeros(len(names))
        for row in table:
            tmp = np.vstack((tmp, np.array(row.split(), dtype=np.float)))
        table = np.delete(tmp, 0, 0)

        return names, table


    def v_n_to_v_n(self, v_n):
        if v_n == 'm': return 'xm-1'
        if v_n == 'l': return 'l-1'
        if v_n == 'He' or v_n == 'Yc' or v_n == 'He4-0': return 'He4-0'
        if v_n == 'ys': return 'He4-1'
        if v_n == 'mdot' or v_n == 'mdot-1': return 'mdot-1'
        if v_n == 'u' or v_n == 'us': return 'u-sp'
        if v_n == 'r' or v_n == 'rs': return 'r-sp'
        if v_n == 't' or v_n=='ts': return 't-sp'
        if v_n == 'r_env': return 'r-env'
        if v_n == 'm_env': return 'm-env'
        if v_n == 'k' or v_n == 'kappa': return 'kappa-sp'
        if v_n == 'L/Ledd-sp' or v_n == 'L/Ledd': return 'L/Ledd-sp'
        if v_n == 'rho': return 'rho-sp'
        if v_n == 'tau' or v_n=='tau-sp': return 'tau-sp'
        if v_n == 'r_eff' or v_n == 'r-ph': return 'r-ph'
        if v_n == 't_eff'or v_n == 't-ph':  return 't-ph'
        if v_n == 'hp' or v_n == 'HP': return 'HP-sp' # log(Hp)
        if v_n == 'tpar': return 'tpar-'
        if v_n == 'mfp': return 'mfp-sp' # log(mfp)

        if v_n == 'grad_c': return 'grad_c-sp'
        if v_n == 'grad_w': return 'grad_w-sp'

        if v_n == 'grad_c_p' or v_n == 'grad_c_p-sp': return 'grad_c_p-sp'
        if v_n == 'grad_w_p': return 'grad_w_p-sp'

        # if v_n == 'lm': return 'lm'
        raise NameError('Translation for v_n({}) not provided'.format(v_n))

    def get_crit_value(self, v_n):

        if v_n == 't_eff':
            v_n_tr = self.v_n_to_v_n(v_n)
            if not v_n_tr in self.names:
                raise NameError('v_n_traslated({}) not in the list of names from file ({})'.format(v_n_tr, self.names))

            r = self.table[0, self.names.index(self.v_n_to_v_n('r_eff'))]
            l = self.table[0, self.names.index(self.v_n_to_v_n('l'))]
            t_eff = Physics.steph_boltz_law_t_eff(l,r)
            return t_eff

        if v_n.split('_')[0] == 'log':
            v_n = v_n.split('_')[-1]
            v_n_tr = self.v_n_to_v_n(v_n)
            if not v_n_tr in self.names:
                raise NameError('v_n_traslated({}) not in the list of names from file ({})'.format(v_n_tr, self.names))
            return np.log10(self.table[0, self.names.index(self.v_n_to_v_n(v_n))])

        if v_n == 'lm':
            l = self.table[0, self.names.index(self.v_n_to_v_n('l'))]
            m = self.table[0, self.names.index(self.v_n_to_v_n('m'))]
            return Physics.loglm(l, m)



        v_n_tr = self.v_n_to_v_n(v_n)
        if not v_n_tr in self.names:
            raise NameError('v_n_traslated({}) not in the list of names from file ({})'.format(v_n_tr, self.names))
        return self.table[0, self.names.index(self.v_n_to_v_n(v_n))]

    def get_uncond_col(self, v_n):

        if len(v_n.split('_')) >0:
            if v_n.split('_')[0] == 'log':
                v_n = v_n.split('_')[-1]
                return np.log10(self.table[1:, self.names.index(self.v_n_to_v_n(v_n))])

        if v_n == 'a_p' or v_n == 'a_p-sp':
            us = self.table[1:, self.names.index(self.v_n_to_v_n('us'))]
            grad_c_p = self.table[1:, self.names.index(self.v_n_to_v_n('grad_c_p'))] / 10**5 # as given grads are higher

            return us * grad_c_p



        if v_n == 'lm':
            l = self.table[1:, self.names.index(self.v_n_to_v_n('l'))]
            m = self.table[1:, self.names.index(self.v_n_to_v_n('m'))]
            return Physics.loglm(l, m, True)
        else:
            return self.table[1:, self.names.index(self.v_n_to_v_n(v_n))]

    def get_sonic_cols(self, v_n, cond=None, precision=2):
        '''
        Cond in a form of 'ts=5.0'
        return part of the col with that condtion
        :param v_n:
        :param cond:
        :return:
        '''
        if cond == None:
            return self.get_uncond_col(v_n)

        elif '=' in cond:
            v_n_cond = cond.split('=')[0]
            cond_val = np.float(cond.split('=')[-1])

            cond_col = self.get_uncond_col(v_n_cond)
            col = self.get_uncond_col(v_n)
            cropped_col = []
            for i in range(len(cond_col)):
                if np.round(cond_col[i], precision) == np.round(cond_val, precision):
                    cropped_col = np.append(cropped_col, col[i])

            if len(cropped_col) == 0: raise ValueError('Condition {} led to zero values appended'.format(cond))

            return cropped_col

        elif '>' in cond:
            v_n_cond = cond.split('>')[0]
            cond_val = np.float(cond.split('>')[-1])

            cond_col = self.get_uncond_col(v_n_cond)
            col      = self.get_uncond_col(v_n)
            cropped_col = []
            for i in range(len(cond_col)):
                if cond_col[i] > cond_val:
                    cropped_col = np.append(cropped_col, col[i])

            if len(cropped_col) == 0: raise ValueError('Condition {} led to zero values appended'.format(cond))

            return cropped_col

        elif '<' in cond:
            v_n_cond = cond.split('<')[0]
            cond_val = np.float(cond.split('<')[-1])

            cond_col = self.get_uncond_col(v_n_cond)
            col      = self.get_uncond_col(v_n)
            cropped_col = []
            for i in range(len(cond_col)):
                if cond_col[i] < cond_val:
                    cropped_col = np.append(cropped_col, col[i])

            # if len(cropped_col) == 0:
            #     raise ValueError('Condition {} led to zero values appended\n'
            #                      'M: {} Yc: {}'.format(cond, self.get_crit_value('m'), self.get_crit_value('Yc')))

            return cropped_col

        else:
            return self.get_uncond_col(v_n)


        # --- Critical values ---

        # self.l_cr = np.float(self.table[0, 0])  # mass array is 0 in the sp file
        # self.m_cr = np.float(self.table[0, 1])  # mass array is 1 in the sp file
        # self.yc_cr = np.float(self.table[0, 2])  # mass array is 2 in the sp file
        # self.ys_cr = np.float(self.table[0, 3])  # mass array is 2 in the sp file
        # self.lmdot_cr = np.float(self.table[0, 4])  # mass array is 3 in the sp file
        # self.r_cr = np.float(self.table[0, 5])  # mass array is 4 in the sp file
        # self.t_cr = np.float(self.table[0, 6])  # mass array is 4 in the sp file
        # self.tau_cr = np.float(self.table[0, 7])
        #
        # # --- Sonic Point Values ---
        #
        # self.l = np.array(self.table[1:, 0])
        # self.m = np.array(self.table[1:, 1])
        # self.yc = np.array(self.table[1:, 2])
        # self.ys = np.array(self.table[1:, 3])
        # self.lmdot = np.array(self.table[1:, 4])
        # self.rs = np.array(self.table[1:, 5])
        # self.ts = np.array(self.table[1:, 6])
        # self.tau = np.array(self.table[1:, 7])
        # self.k = np.array(self.table[1:, 8])
        # self.rho = np.array(self.table[1:, 9])

        # self.k = np.array(self.table[1:, 6])
        # self.rho = np.array(self.table[1:, 8])



    #
    #
    # def get_crit_value(self, v_n):
    #     if v_n == 'l':
    #         return np.float( self.l_cr )
    #
    #     if v_n =='m' or v_n == 'xm':
    #         return np.float( self.m_cr )
    #
    #     if v_n == 't':
    #         return np.float( self.t_cr )
    #
    #     if v_n == 'mdot':
    #         return np.float( self.lmdot_cr)
    #
    #     if v_n == 'r':
    #         return np.float(self.r_cr)
    #
    #     if v_n == 'Yc':
    #         return np.float(self.yc_cr)
    #
    #     if v_n == 'ys':
    #         return np.float(self.ys_cr)
    #
    #     if v_n == 'tau':
    #         return self.tau_cr
    #
    #     if v_n == 'lm':
    #         return np.float(Physics.loglm(self.l_cr, self.m_cr, False) )
    #
    #     raise NameError('v_n {} is not in the list: {} (for critical values)'.format(v_n, self.list_of_v_n))
    #
    # def get_sonic_cols(self, v_n):
    #     if v_n == 'l':
    #         return self.l
    #
    #     if v_n =='m' or v_n == 'xm':
    #         return self.m
    #
    #     if v_n == 't':
    #         return self.ts
    #
    #     if v_n == 'mdot':
    #         return self.lmdot
    #
    #     if v_n == 'r':
    #         return self.rs
    #
    #     if v_n == 'Yc':
    #         return self.yc
    #
    #     if v_n == 'ys':
    #         return self.ys
    #
    #     if v_n == 'k':
    #         return self.k
    #
    #     if v_n == 'rho':
    #         return self.rho
    #
    #     if v_n == 'tau':
    #         return self.tau
    #
    #     if v_n == 'lm':
    #         return Physics.loglm(self.l, self.m, True)
    #
    #     raise NameError('v_n {} is not in the list: {} (for critical values)'.format(v_n, self.list_of_v_n))

class SP_file_work:

    def __init__(self, yc_precision, gal_or_lmc, bump, cr_or_wd, marks=list()):
        self.sp_files = Files.get_sp_files(gal_or_lmc, cr_or_wd, marks)
        self.out_dir = Files.output_dir
        self.plot_dir = Files.plot_dir
        self.metal = gal_or_lmc
        self.yc_prec = yc_precision
        self.bump = bump
        self.cr_or_wd = cr_or_wd


        self.set_clean_plots        = False
        self.set_extrapol_pars      = []
        self.set_init_fit_method    ='IntUni'

        self.set_xy_int_method      = 'IntUni'
        self.set_xz_int_method      = 'IntUni'
        self.set_yz_int_method      = 'IntUni'
        self.set_load_cond          = 'ts>5.0'


        self.set_int_or_pol         = 'int'
        self.set_invert_x_ax        = False
        self.set_do_tech_plots      = False
        self.set_check_x_y_z_arrs   = False

        self.spmdl = []
        for file in self.sp_files:
            self.spmdl.append(Read_SP_data_file(file, self.out_dir, self.plot_dir))

        if len(self.spmdl) < 2: raise IOError('No SP files found: Metal: {} Cr/Wd: {}'.format(gal_or_lmc, cr_or_wd))

        pass

    def get_min_max(self, v_n):
        x = []
        for i in range(len(self.sp_files)):
            x =  np.append(x, self.spmdl[i].get_crit_value(v_n))

        return x.min(), x.max()

    def separate_sp_by_crit_val(self, v_n, yc_prec = .1):

        yc_arr = []
        for i in range(len(self.sp_files)):
            yc = self.spmdl[i].get_crit_value(v_n)
            yc_arr = np.append(yc_arr, np.float("%{}f".format(yc_prec) % yc) )

        print('\t__Total {} unique values found: {}'.format(v_n, len(set(np.sort(yc_arr)))))

        # if not yc_val in yc_arr:
        #     raise ValueError('{}={} is not found in {}'.format(v_n, yc_val, yc_arr))

        # print(np.array(set(yc_arr), dtype=float))
        yc_arr = np.sort( list(set(yc_arr)), axis=0 )

        set_of_files = []
        for j in range(len(yc_arr)):
            files = []
            for i in range(len(self.sp_files)):
                yc = self.spmdl[i].get_crit_value(v_n)
                if "%{}f".format(yc_prec) % yc == "%{}f".format(yc_prec) % yc_arr[j]:
                    files.append( self.spmdl[i] )
            set_of_files.append( files )

        # print( set_of_files[9] )

        return yc_arr, set_of_files

    def separate_sp_by_fname_mass(self):
        '''
        Separation is based on file name an _10sm_ part in that name, where 10 is the value.
        :return: m_arr[:], m_files[:,:] one row for one m
        '''

        # m_files = []
        m_arr = np.array([str(0)])
        fls = []
        for i in range(len(self.sp_files)):
            cut_dirs = self.sp_files[i].split('/')[-1]
            cut_extension = cut_dirs.split('.')[0]
            parts = cut_extension.split('_')

            for j in range(len(parts)):
                if len(parts[j].split('sm')) > 1:
                    m = parts[j].split('sm')[0]
                    # fls.append(self.sp_files[i])
                    if not m in m_arr:
                        m_arr = np.append(m_arr, m)
                        # m_files = np.append(m_files, fls)
                        # fls = []
                    # print(parts[j])

        print('__Initial Masses, read from file name are: {}'.format(m_arr))

        m_files = []
        for j in range(len(m_arr)):
            piece = str(m_arr[j])+'sm' # 10sm ...


            for i in range(len(self.sp_files)):
                cut_dirs = self.sp_files[i].split('/')[-1]
                cut_extension = cut_dirs.split('.')[0]
                parts = cut_extension.split('_')


                if piece in parts:
                    # files.append(self.sp_files[i])
                    m_files = np.append(m_files,  self.spmdl[i])

        m_arr = np.delete(m_arr, 0, 0)
        print(len(m_files), len(m_arr))

        m_files = np.reshape(m_files, (len(m_arr), int(len(m_files)/len(m_arr)) ))



        return m_arr, m_files

    def save_y_yc_z_relation(self, y_v_n, z_v_n, plot=False, depth=100, yc_prec=0.1):

        if self.cr_or_wd == 'wd': raise IOError('You do not have SP wind files for all the Yc don*t you?:)')

        yc, cls = self.separate_sp_by_crit_val('Yc', yc_prec)

        def interp(x, y, x_grid):
            f = interpolate.interp1d(x, y, kind='cubic', bounds_error=False)
            # f = interpolate.InterpolatedUnivariateSpline(x, y)
            # f = interpolate.UnivariateSpline(x, y)
            return x_grid, f(x_grid)



        y_ = []
        for i in range(len(self.sp_files)):
            y_ = np.append(y_, self.spmdl[i].get_crit_value(y_v_n))

        y_grid = np.mgrid[y_.min():y_.max():depth*1j]

        z2d_pol = np.zeros(len(y_grid))
        z2d_int = np.zeros(len(y_grid))


        fig = plt.figure(figsize=plt.figaspect(1.0))

        ax1 = fig.add_subplot(221)
        ax1.grid()
        ax1.set_ylabel(Labels.lbls(z_v_n))
        ax1.set_xlabel(Labels.lbls(y_v_n))
        ax1.set_title('INTERPOLATION')

        ax2 = fig.add_subplot(222)
        ax2.grid()
        ax2.set_ylabel(Labels.lbls(y_v_n))
        ax2.set_xlabel(Labels.lbls(z_v_n))
        ax2.set_title('EXTRAPOLATION')


        for i in range(len(yc)):
            y_z = []
            for cl in cls[i]:
                y_z = np.append(y_z, [cl.get_crit_value(y_v_n), cl.get_crit_value(z_v_n)])
            y_z_sort = np.sort(y_z.view('float64, float64'), order=['f0'], axis=0).view(np.float)
            y_z_shaped = np.reshape(y_z_sort, (int(len(y_z_sort) / 2), 2))

            '''----------------------------POLYNOMIAL EXTRAPOLATION------------------------------------'''
            print('\n\t Yc = {}'.format(yc[i]))
            y_pol, z_pol = Math.fit_polynomial(y_z_shaped[:, 0], y_z_shaped[:, 1], 3, depth, y_grid)
            z2d_pol = np.vstack((z2d_pol, z_pol))
            color = 'C' + str(int(yc[i] * 10)-1)
            ax2.plot(y_pol, z_pol, '--', color=color)
            ax2.plot(y_z_shaped[:, 0], y_z_shaped[:, 1], '.', color=color, label='yc:{}'.format("%.2f" % yc[i]))

            '''--------------------------------INTERPOLATION ONLY--------------------------------------'''
            z_int = Math.interpolate_arr(y_z_shaped[:, 0], y_z_shaped[:, 1], y_grid, self.set_init_fit_method)
            y_int = y_grid
            # y_int, z_int = interp(y_z_shaped[:, 0], y_z_shaped[:, 1], y_grid)
            z2d_int = np.vstack((z2d_int, z_int))
            ax1.plot(y_int, z_int, '--', color=color)
            ax1.plot(y_z_shaped[:, 0], y_z_shaped[:, 1], '.', color=color, label='yc:{}'.format("%.2f" % yc[i]))


        ax1.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
        ax2.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)

        z2d_int = np.delete(z2d_int, 0, 0)
        z2d_pol = np.delete(z2d_pol, 0, 0)

        yc_llm_m_pol = Math.combine(yc, y_grid, z2d_pol.T)  # changing the x/y
        yc_llm_m_int = Math.combine(yc, y_grid, z2d_int.T)  # changing the x/y


        # --- This part fills the nan values in interpolated array with values from polynomial
        tmp = z2d_int.T
        for k in range(len(yc)):
            for m in range(len(y_grid)):
                if np.isnan(tmp[m,k]):
                    tmp[m,k]=z2d_pol.T[m,k]


        yc_llm_m_pol = Math.combine(yc, y_grid, tmp)

        # unsuccesfull attempt to write an extraplation by Teylor
        def extrapolate_taylor(table):

            x = table[0, 1:]
            y = table[1:, 0]
            zz = table[1:,1:]

            # # adding the coulumn for extrapolated values
            # zz = np.vstack((zz.T, np.zeros(len(zz[:,0])))).T

            # extending the x_axis, by linear regression
            x_ = x[-1] + np.diff(x)[-1]
            x = np.append(x, x_)


            # extapolating the values for the new column
            gxx, gyy = np.gradient(zz[:, :-1])
            zz_ = zz[:, -2] + gxx[:, -1]*np.diff(x)[-1] + gyy[:,-1]*0

            # second order
            ggxx, _ = np.gradient(gxx)
            _, ggyy = np.gradient(gyy)
            # zz_ = zz[:, -2] + gxx[:, -1] + gyy[:, -1] + ggxx[:, -1] + ggyy[:, -1]

            # appending the column to the array
            zz = np.vstack((zz.T, zz_)).T

            res = Math.combine(x, y, zz)

            return res

        def extrapolate_taylor_y(table):

            x = table[0, 1:]
            y = table[1:, 0]
            zz = table[1:,1:]

            # # adding the coulumn for extrapolated values
            # zz = np.vstack((zz.T, np.zeros(len(zz[:,0])))).T

            # extending the x_axis, by linear regression
            y_ = x[-1] + np.diff(x)[-1]
            y = np.append(y, y_)


            # extapolating the values for the new column
            gxx, gyy = np.gradient(zz[:, :-1])
            zz_ = zz[:, -2] + gxx[:, -1]*0 + gyy[:,-1]*np.diff(y)[-1]

            # second order
            ggxx, _ = np.gradient(gxx)
            _, ggyy = np.gradient(gyy)
            # zz_ = zz[:, -2] + gxx[:, -1] + gyy[:, -1] + ggxx[:, -1] + ggyy[:, -1]

            # appending the column to the array
            zz = np.vstack((zz, zz_))

            res = Math.combine(x, y, zz)

            return res

        # yc_llm_m_pol = Math.interpolated_nan(yc_llm_m_int)
        # yc_llm_m_pol = extrapolate_taylor(yc_llm_m_pol)
        # yc_llm_m_pol = extrapolate_taylor_y(yc_llm_m_pol)

        if self.set_extrapol_pars != [0, 0, 0, 0]:
            left  = self.set_extrapol_pars[0]
            right = self.set_extrapol_pars[1]
            down  = self.set_extrapol_pars[2]
            up    = self.set_extrapol_pars[3]
            _, yc_llm_m_pol = Math.extrapolate2(yc_llm_m_pol, left, right, down, up, 100, 4, True) # Extrapolation



        table_name = '{}_{}_{}'.format('yc', y_v_n, z_v_n)
        if self.set_int_or_pol == 'int':
            Save_Load_tables.save_table(yc_llm_m_int, self.metal, '', table_name, 'yc', y_v_n, z_v_n)
        if self.set_int_or_pol == 'pol':
            Save_Load_tables.save_table(yc_llm_m_pol, self.metal, '', table_name, 'yc', y_v_n, z_v_n)

        # Save_Load_tables.save_table(yc_llm_m_pol, opal_used, table_name, 'yc', y_v_n, z_v_n)

        if plot:

            levels = []

            # from  import Levels
            levels = Levels.get_levels(z_v_n, self.metal)


            ax = fig.add_subplot(223)

            # ax = fig.add_subplot(1, 1, 1)
            ax.set_xlim(yc_llm_m_int[0,1:].min(), yc_llm_m_int[0,1:].max())
            ax.set_ylim(yc_llm_m_int[1:,0].min(), yc_llm_m_int[1:,0].max())
            ax.set_ylabel(Labels.lbls(y_v_n))
            ax.set_xlabel(Labels.lbls('Yc'))

            contour_filled = plt.contourf(yc_llm_m_int[0, 1:], yc_llm_m_int[1:, 0], yc_llm_m_int[1:,1:], levels, cmap=plt.get_cmap('RdYlBu_r'))
            # plt.colorbar(contour_filled, label=Labels.lbls('m'))
            contour = plt.contour(yc_llm_m_int[0, 1:], yc_llm_m_int[1:, 0], yc_llm_m_int[1:,1:], levels, colors='k')

            clb = plt.colorbar(contour_filled)
            clb.ax.set_title(Labels.lbls(z_v_n))

            plt.clabel(contour, colors='k', fmt='%2.2f', fontsize=12)
            #ax.set_title('MASS-LUMINOSITY RELATION')

            # plt.ylabel(l_or_lm)
            # ax.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
            # plt.savefig(name)




            ax = fig.add_subplot(224)

            # ax = fig.add_subplot(1, 1, 1)
            ax.set_xlim(yc_llm_m_pol[0, 1:].min(), yc_llm_m_pol[0, 1:].max())
            ax.set_ylim(yc_llm_m_pol[1:, 0].min(), yc_llm_m_pol[1:, 0].max())
            ax.set_ylabel(Labels.lbls(y_v_n))
            ax.set_xlabel(Labels.lbls('Yc'))


            contour_filled = plt.contourf(yc_llm_m_pol[0, 1:], yc_llm_m_pol[1:, 0], yc_llm_m_pol[1:, 1:], levels,
                                          cmap=plt.get_cmap('RdYlBu_r'))
            # plt.colorbar(contour_filled, label=Labels.lbls('m'))
            contour = plt.contour(yc_llm_m_pol[0, 1:], yc_llm_m_pol[1:, 0], yc_llm_m_pol[1:, 1:], levels, colors='k')

            clb = plt.colorbar(contour_filled)
            clb.ax.set_title(Labels.lbls(z_v_n))

            plt.clabel(contour, colors='k', fmt='%2.2f', fontsize=12)
            #ax.set_title('MASS-LUMINOSITY RELATION')

            # plt.ylabel(l_or_lm)
            # ax.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)


            plt.show()

        if plot:
            # from PlottingClasses import PlotBackground
            # from PhysMath import Get_Z

            plt.figure()
            ax = plt.subplot(111)

            if not self.set_clean_plots:
                lbl = 'z:{}'.format(Get_Z.z(self.metal))
            else:
                lbl = None

            if self.set_int_or_pol == 'int':
                PlotBackground.plot_color_background(ax, yc_llm_m_int, 'yc', y_v_n, z_v_n, self.metal, lbl, 1.0, False, 12, 0)

            if self.set_int_or_pol == 'pol':
                PlotBackground.plot_color_background(ax, yc_llm_m_pol, 'yc', y_v_n, z_v_n, self.metal, lbl, 1.0, False, 12, 0)

            plt.show()

        # yc_llm_m_pol

    def save_x_y_yc_evol_relation(self, y_v_tmp, z_v_n_tmp, wne=True):
        '''
        Retruns EVOLUTION: 'Yc:[0,1:]  |  y_v_n(ZAMS vales):[1:,0]  |  z_v_n(zams->last Yc)
        :param z_v_n_tmp:
        :param y_v_tmp:
        :return:
        '''

        m_arr, cls = self.separate_sp_by_fname_mass()

        # print(m_files2d.shape)

        x_all = np.zeros(len(cls[0, :]))
        y_all = np.zeros(len(cls[0, :]))
        yc_all= np.zeros(len(cls[0, :]))
        for i in range(len(m_arr)):

            z2d = []
            y = []
            yc = []

            for cl in cls[i,:]:
                z2d = np.append(z2d, cl.get_crit_value(z_v_n_tmp))
                y = np.append(y, cl.get_crit_value(y_v_tmp))
                yc = np.append(yc, np.float("%.1f" % cl.get_crit_value('Yc')) )
                yc, z2d, y = Math.x_y_z_sort(yc, z2d, y)

            x_all = np.vstack((x_all, z2d))
            y_all = np.vstack((y_all, y))
            yc_all = np.vstack((yc_all, yc))

            plt.plot(yc, z2d, '-', color = 'C'+str(Math.get_0_to_max([i],9)[i]))

        plt.grid()
        plt.xlabel(Labels.lbls('Yc'))
        plt.ylabel(Labels.lbls(z_v_n_tmp))
        plt.show()

        x_all = np.delete(x_all, 0, 0)
        y_all = np.delete(y_all, 0, 0)
        yc_all = np.delete(yc_all, 0, 0)


        y_arr = np.array(["%.2f" % val for val in y_all[:,-1]])
        res = Math.combine(yc_all[0,:], y_arr, x_all)

        Save_Load_tables.save_table(res, self.metal, '', 'evol_yc_{}_{}'.format(y_v_tmp, z_v_n_tmp),
                                    'evol_yc', y_v_tmp, z_v_n_tmp)

    @staticmethod
    def yc_x__to__y__sp(yc_value, y_v_n, z_v_n, y_inp, opal_used, dimension=0):
        '''
        LOADS the table with given v_ns and extract the row with given Yc and interpolateds the y_value, for x_val given
        :param yc_value:
        :param y_v_n:
        :param z_v_n:
        :param y_inp:
        :param opal_used:
        :param dimension:
        :return:
        '''

        name = '{}_{}_{}'.format('yc', y_v_n, z_v_n)
        yc_x_y = Save_Load_tables.load_table(name, 'yc', y_v_n, z_v_n, opal_used)
        x_arr = yc_x_y[1:, 0]
        yc_arr = yc_x_y[0, 1:]
        y2d = yc_x_y[1:, 1:]

        print()

        # yc_value = np.float("%.3f" % yc_value)

        if yc_value in yc_arr:
            ind_yc = Math.find_nearest_index(yc_arr, yc_value)
        else:
            raise ValueError('Table: {} Given yc_arr({}) is not in available yc_arr:({})'.format(name, yc_value, yc_arr))


        if dimension == 0:
            if y_inp >= x_arr.min() and y_inp <= x_arr.max():

                y_arr = y2d[:, ind_yc]
                # lm_arr = []
                # for i in range(len(y_arr)):
                #     lm_arr = np.append(lm_arr, [x_arr[i], y_arr[i]])
                #
                # lm_arr_sort = np.sort(lm_arr.view('float64, float64'), order=['f0'], axis=0).view(np.float)
                # lm_arr_shaped = np.reshape(lm_arr_sort, (len(y_arr), 2))

                f = interpolate.InterpolatedUnivariateSpline(x_arr, y_arr)
                y = f(y_inp)
                # print(log_l, y)
                print('Yc: {}, y_row({} - {}), y_res: {}'.format(yc_value, y_arr.min(), y_arr.max(), y))
                return y_inp, y
            else:
                raise ValueError('Given l({}) not in available range of l:({}, {})'.format(y_inp, x_arr.min(), x_arr.max()))

        if dimension == 1:
            x_arr_f = []
            y_arr_f = []
            for i in range(len(y_inp)):
                if y_inp[i] >= x_arr.min() and y_inp[i] <= x_arr.max():
                    f = interpolate.UnivariateSpline(x_arr, y2d[:, ind_yc])
                    y_arr_f = np.append(x_arr_f, f(y_inp[i]))
                    x_arr_f = np.append(x_arr_f, y_inp[i])
                # else:
                #     raise ValueError(
                #         'Given x({}, {}) not in available range of x:({}, {})'.format(x_inp[0], x_inp[-1], x_arr.min(), x_arr.max()))

            return x_arr_f, y_arr_f

    # @staticmethod
    # def l_y_to_m(log_l, yc_value, dimension=0):
    #     '''
    #     RETURNS l, m (if l dimension = 0) and l_arr, m_arr (if dimension = 1)
    #     :param log_l:
    #     :param yc_value:
    #     :param dimension:
    #     :return:
    #     '''
    #     # from FilesWork import Save_Load_tables
    #
    #     yc_l_m = Save_Load_tables.load_table('yc_l_m', 'yc', 'l', 'm', opal_used)
    #     l_arr = yc_l_m[1:, 0]
    #     yc_arr= yc_l_m[0, 1:]
    #     m2d = yc_l_m[1:, 1:]
    #
    #     # yc_value = np.float("%.3f" % yc_value)
    #
    #     if yc_value in yc_arr:
    #         ind_yc = Math.find_nearest_index(yc_arr, yc_value)
    #     else:
    #         raise ValueError('Given yc_arr({}) is not in available yc_arr:({})'.format(yc_value, yc_arr))
    #
    #     if dimension == 0:
    #         if log_l >= l_arr.min() and log_l <= l_arr.max():
    #             m_arr = m2d[:, ind_yc]
    #             # lm_arr = []
    #             # for i in range(len(m_arr)):
    #             #     lm_arr = np.append(lm_arr, [l_arr[i], m_arr[i]])
    #             #
    #             # lm_arr_sort = np.sort(lm_arr.view('float64, float64'), order=['f0'], axis=0).view(np.float)
    #             # lm_arr_shaped = np.reshape(lm_arr_sort, (len(m_arr), 2))
    #
    #             f = interpolate.UnivariateSpline(l_arr, m_arr)
    #             m = f(log_l)
    #             # print(log_l, m)
    #
    #             return log_l, m
    #         else:
    #             raise ValueError('Given l({}) not in available range of l:({}, {})'.format(log_l, l_arr.min, l_arr.max))
    #     if dimension == 1:
    #         m_arr_f = []
    #         l_arr_f = []
    #         for i in range(len(log_l)):
    #             if log_l[i] >= l_arr.min() and log_l[i] <= l_arr.max():
    #                 f = interpolate.UnivariateSpline(l_arr, m2d[ind_yc, :])
    #                 m_arr_f = np.append(m_arr_f, f(log_l[i]))
    #                 l_arr_f = np.append(l_arr_f, log_l[i])
    #             else:
    #                 raise ValueError('Given l({}) not in available range of l:({}, {})'.format(log_l, l_arr.min, l_arr.max))
    #
    #         return l_arr_f, m_arr_f

    # def sp_get_r_lt_table2(self, v_n, l_or_lm, plot=True, ref_t_llm_vrho=np.empty([])):
    #     '''
    #
    #     :param l_or_lm:
    #     :param depth:
    #     :param plot:
    #     :param t_llm_vrho:
    #     :return:
    #     '''
    #     if not ref_t_llm_vrho.any():
    #         print('\t__ No *ref_t_llm_vrho* is provided. Loading {} interp. opacity table.'.format(self.opal_used))
    #         t_k_rho = Save_Load_tables.load_table('t_k_rho', 't', 'k', 'rho', self.opal_used, self.out_dir)
    #         table = Physics.t_kap_rho_to_t_llm_rho(t_k_rho, l_or_lm)
    #     else:
    #         table = ref_t_llm_vrho
    #
    #     t_ref = table[0, 1:]
    #     llm_ref=table[1:, 0]
    #     # rho_ref=table[1:, 1:]
    #
    #
    #     '''=======================================ESTABLISHING=LIMITS================================================'''
    #
    #     t_mins = []
    #     t_maxs = []
    #
    #     for i in range(len(self.sp_files)): # as every sp.file has different number of t - do it one by one
    #
    #         t = self.spmdl[i].get_sonic_cols('t')                     # Sonic
    #         t = np.append(t, self.spmdl[i].get_crit_value('t'))       # critical
    #
    #         t_mins = np.append(t_mins, t.min())
    #         t_maxs = np.append(t_maxs, t.max())
    #
    #     t_min = t_mins.max()
    #     t_max = t_maxs.min()
    #
    #     print('\t__ SP files t limits: ({}, {})'.format(t_min, t_max))
    #     print('\t__REF table t limits: ({}, {})'.format(t_ref.min(), t_ref.max()))
    #
    #     it1 = Math.find_nearest_index(t_ref, t_min)       # Lower t limit index in t_ref
    #     it2 = Math.find_nearest_index(t_ref, t_max)       # Upper t limit index in t_ref
    #     t_grid = t_ref[it1:it2]
    #
    #     print('\t__     Final t limits: ({}, {}) with {} elements'.format(t_ref[it1], t_ref[it2], len(t_grid)))
    #
    #
    #     '''=========================INTERPOLATING=ALONG=T=ROW=TO=HAVE=EQUAL=N=OF=ENTRIES============================='''
    #
    #     llm_r_rows = np.empty(1 + len(t_ref[it1:it2]))
    #
    #     for i in range(len(self.sp_files)):
    #
    #         if l_or_lm == 'l':
    #             llm = self.spmdl[i].get_crit_value('l')
    #         else:
    #             llm = Physics.loglm(self.spmdl[i].get_crit_value('l'), self.spmdl[i].get_crit_value('m'), False)
    #
    #
    #         r = self.spmdl[i].get_sonic_cols(v_n)                     # get sonic
    #         if v_n == 'r': r = np.append(r, self.spmdl[i].get_crit_value('r'))       # get Critical
    #
    #         t = self.spmdl[i].get_sonic_cols('t')                                     # Sonic
    #         if v_n == 'r': t = np.append(t, self.spmdl[i].get_crit_value('t'))        # critical
    #
    #         r_t = []        # Dictionary for sorting
    #         for i in range(len(r)):
    #             r_t = np.append(r_t, [r[i], t[i]])
    #
    #         r_t_sort = np.sort(r_t.view('float64, float64'), order=['f1'], axis=0).view(np.float)
    #         r_t_reshaped = np.reshape(r_t_sort, (len(r), 2)) # insure that the t values are rising along the t_r arr.
    #
    #         r_sort = r_t_reshaped[:,0]
    #         t_sort = r_t_reshaped[:,1]
    #
    #         f = interpolate.InterpolatedUnivariateSpline(t_sort, r_sort)
    #
    #         l_r_row = np.array([llm])
    #         l_r_row = np.append(l_r_row, f(t_grid))
    #         llm_r_rows = np.vstack((llm_r_rows, l_r_row))
    #
    #     llm_r_rows = np.delete(llm_r_rows, 0, 0)
    #
    #     llm_r_rows_sort = llm_r_rows[llm_r_rows[:,0].argsort()] # UNTESTED sorting function
    #
    #     t_llm_r = Math.combine(t_grid, llm_r_rows_sort[:,0], llm_r_rows_sort[:,1:]) # intermediate result
    #
    #     '''======================================INTERPOLATING=EVERY=COLUMN=========================================='''
    #
    #
    #     l      = t_llm_r[1:, 0]
    #     t      = t_llm_r[0, 1:]
    #     r      = t_llm_r[1:, 1:]
    #     il1    = Math.find_nearest_index(llm_ref, l.min())
    #     il2    = Math.find_nearest_index(llm_ref, l.max())
    #
    #     print('\t__ SP files l limits: ({}, {})'.format(l.min(), l.max()))
    #     print('\t__REF table t limits: ({}, {})'.format(llm_ref.min(), llm_ref.max()))
    #
    #     l_grid = llm_ref[il1:il2]
    #
    #     print('\t__     Final l limits: ({}, {}) with {} elements'.format(llm_ref[il1], llm_ref[il2], len(l_grid)))
    #
    #     r_final = np.empty((len(l_grid),len(t)))
    #     for i in range(len(t)):
    #         f = interpolate.InterpolatedUnivariateSpline(l, r[:, i])
    #         r_final[:, i] = f(l_grid)
    #
    #     t_llm_r = Math.combine(t, l_grid, r_final)
    #
    #     if plot:
    #         plt.figure()
    #         # ax = fig.add_subplot(1, 1, 1)
    #         plt.xlim(t_llm_r[0,1:].min(), t_llm_r[0,1:].max())
    #         plt.ylim(t_llm_r[1:,0].min(), t_llm_r[1:,0].max())
    #         plt.ylabel(Labels.lbls(l_or_lm))
    #         plt.xlabel(Labels.lbls('ts'))
    #
    #         levels = []
    #         if v_n == 'k':   levels = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95] # FOR log Kappa
    #         if v_n == 'rho': levels = [-10, -9.5, -9, -8.5, -8, -7.5, -7, -6.5, -6, -5.5,  -5, -4.5, -4]
    #         if v_n == 'r':   levels = [0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7,
    #                                    1.75, 1.8, 1.85, 1.9, 1.95, 2.0, 2.05, 2.10, 2.15, 2.20]
    #
    #         contour_filled = plt.contourf(t_llm_r[0, 1:], t_llm_r[1:, 0], t_llm_r[1:,1:], levels, cmap=plt.get_cmap('RdYlBu_r'))
    #         plt.colorbar(contour_filled, label=Labels.lbls(v_n))
    #         contour = plt.contour(t_llm_r[0, 1:], t_llm_r[1:, 0], t_llm_r[1:,1:], levels, colors='k')
    #         plt.clabel(contour, colors='k', fmt='%2.2f', fontsize=12)
    #         plt.title('SONIC HR DIAGRAM')
    #
    #         # plt.ylabel(l_or_lm)
    #         plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
    #         # plt.savefig(name)
    #         plt.show()
    #
    #     return t_llm_r

    # --- --- --- | ---
    # --- --- --- | ---
    # --- --- --- | ---

    # def save_yc_m_l
    @staticmethod
    def x_y_z(cls, x_v_n, y_v_n, z_v_n, x_grid, y_grid, append_crit=True, int_method='IntUni', check_nans=False):
        '''
        cls = set of classes of sp. files with the same Yc.
        :param cls:
        :return:
        '''

        # x_grid, y_grid = np.round(x_grid, 3), np.round(y_grid, 3)

        y_zg = np.zeros(len(x_grid) + 1)  # +1 for y-value (l,lm,m,Yc)

        for cl in cls:  # INTERPOLATING EVERY ROW to achive 'depth' number of points
            x = cl.get_sonic_cols(x_v_n)
            y = cl.get_crit_value(y_v_n)  # Y should be unique value for a given Yc (like m, l/lm, or Yc)
            z = cl.get_sonic_cols(z_v_n)

            if append_crit:
                x = np.append(x, cl.get_crit_value(x_v_n))
                z = np.append(z, cl.get_crit_value(z_v_n))

            if len(x) != len(x[~np.isnan(x)]): raise ValueError('NaN in sonic/criticals')
            if len(z) != len(z[~np.isnan(z)]): raise ValueError('NaN in sonic/criticals')
            if np.isnan(y): raise ValueError('NaN in sonic/criticals')

            x, z = np.array(x, dtype=np.float), np.array(z, dtype=np.float)

            xi, zi = Math.x_y_z_sort(x, z)


            #z_grid = interpolate.InterpolatedUnivariateSpline(xi, zi)(x_grid)

            # if x_grid.min() < xi.min(): raise ValueError('x_grid.min(){} < xi.min(){}'.format(xi.min() , x_grid.min()))
            # if x_grid.max() > xi.max(): raise ValueError('x_grid.max(){} > xi.max(){}'.format(xi.max() , x_grid.max()))

            z_grid = Math.interpolate_arr(xi, zi, x_grid, 'Uni')

            # z_grid = interpolate.interp1d(xi, zi, kind='cubic', bounds_error=False)(x_grid)

            y_zg = np.vstack((y_zg, np.insert(z_grid, 0, y, 0)))

            plt.plot(xi, zi, '.', color='black')  # FOR interplation analysis (how good is the fit)
            plt.plot(x_grid, z_grid, '-', color='grady')

        y_zg = np.delete(y_zg, 0, 0)
        y = y_zg[:, 0]
        zi = y_zg[:, 1:]

        z_grid2 = np.zeros(len(y_grid))
        for i in range(len(x_grid)):  # INTERPOLATING EVERY COLUMN to achive 'depth' number of points
            #z_grid2 = np.vstack((z_grid2, interpolate.InterpolatedUnivariateSpline(y, zi[:, i])(y_grid)))
            # z_grid2 = np.vstack((z_grid2, interpolate.interp1d(y, zi[:, i], kind='cubic', bounds_error=False)(y_grid)))
            # y_, zi_ = Math.x_y_z_sort(y, zi[:, i])
            row = Math.interpolate_arr(y, zi[:, i], y_grid, int_method)
            z_grid2 = np.vstack((z_grid2, row))
        z_grid2 = np.delete(z_grid2, 0, 0)

        x_y_z_final = Math.combine(x_grid, y_grid, z_grid2.T)


        if check_nans:
            x_y_z_final_ = Math.remove_nan_col_raw_from_table(x_y_z_final)
        else:
            x_y_z_final_ = x_y_z_final
        return x_y_z_final_

    def x_y_z2d(self, cls, x_v_n, y_v_n, z_v_n, x_grid, y_grid, append_crit = True, z_fill=False):
        '''
        cls = set of classes of sp. files with the same Yc.
        :param cls:
        :return:
        '''

        # --- --- --- GET COLS INTERPOLATE TILL COMMON END --- --- ---



        z_cols_int_x = np.zeros(len(x_grid))
        z_cols_int_y = np.zeros(len(y_grid))

        def fill_out_of_data_with_nans(arr, new_arr):
            for i in range(len(new_arr)):
                if new_arr[i] > arr.max() or new_arr[i] < arr.min():
                    new_arr[i] = np.nan
            return new_arr
        for cl in cls:

            x_col = cl.get_sonic_cols(x_v_n, self.set_load_cond)
            if append_crit: x_col = np.append(x_col, cl.get_crit_value(x_v_n))
            z_col = cl.get_sonic_cols(z_v_n, self.set_load_cond)
            if append_crit: z_col = np.append(z_col, cl.get_crit_value(z_v_n))
            if z_fill:
                z_col.fill(cl.get_crit_value(z_v_n))

            if len(x_col) > 0:

                y_col = cl.get_sonic_cols(y_v_n, self.set_load_cond)
                if append_crit: y_col = np.append(y_col, cl.get_crit_value(y_v_n))

                if self.set_do_tech_plots: plt.plot(x_col, y_col, '.', color='red')

                if self.set_do_tech_plots: plt.plot(x_col, z_col, '.', color='blue')

                # --- XZ ---
                x_col, z_col = Math.x_y_z_sort(x_col, z_col) # in case crit is not the last
                z_col_gr_down = Math.interpolate_arr(x_col, z_col, x_grid, self.set_xz_int_method)
                if 'poly' in self.set_xz_int_method: z_col_gr_down = fill_out_of_data_with_nans(z_col, z_col_gr_down)
                z_cols_int_x = np.vstack((z_cols_int_x, z_col_gr_down))

                if self.set_do_tech_plots: plt.plot(x_grid, z_col_gr_down, '-', color='blue')

                # --- XY ---
                x_col, y_col = Math.x_y_z_sort(x_col, y_col)  # in case crit is not the last
                z_col_gr_down = Math.interpolate_arr(x_col, y_col, x_grid, self.set_xy_int_method)
                if 'poly' in self.set_xy_int_method: z_col_gr_down = fill_out_of_data_with_nans(y_col, z_col_gr_down)
                z_cols_int_y = np.vstack((z_cols_int_y, z_col_gr_down))

                if self.set_do_tech_plots: plt.plot(x_grid, z_col_gr_down, '-', color='red')

                # plt.plot(x_col, y_col, '.', color='blue')

                if self.set_do_tech_plots:
                    plt.annotate('m:{}'.format("%.1f"%cl.get_crit_value('m')), xy=(y_col[-1], z_col[-1]), textcoords = 'data')
                    plt.annotate('m:{}'.format("%.1f" % cl.get_crit_value('m')), xy=(y_col[0], z_col[0]), textcoords='data')

            # plt.show()

        z_cols_int_x = np.delete(z_cols_int_x, 0, 0)
        z_cols_int_y = np.delete(z_cols_int_y, 0, 0)

        # --- --- Now it can be interpolated row y row among all columns, or using Int2d

        z_rows_int = np.zeros(len(y_grid))
        for i in range(len(x_grid)):
            # print('i:{}'.format(i))

            y_col, z_col = Math.x_y_z_sort(z_cols_int_y[:, i], z_cols_int_x[:, i])

            z_row_gr = Math.interpolate_arr(y_col, z_col, y_grid, self.set_yz_int_method)

            z_rows_int = np.vstack((z_rows_int, z_row_gr))

            if self.set_do_tech_plots:
                plt.plot(y_grid, z_row_gr, '-', color='orange')

        z_rows_int = np.delete(z_rows_int, 0, 0)

        x_y_z_final = Math.combine(x_grid, y_grid, z_rows_int.T)

        # y_zg = np.zeros(len(x_grid) + 1)  # +1 for y-value (l,lm,m,Yc)
        #
        # def cut_inf_in_y(x_arr, y_arr):
        #     new_x = []
        #     new_y = []
        #     for i in range(len(y_arr)):
        #         if y_arr[i] != np.inf and y_arr[i] != -np.inf:
        #             new_x = np.append(new_x, x_arr[i])
        #             new_y = np.append(new_y, y_arr[i])
        #     return new_x, new_y
        #
        # for cl in cls:  # INTERPOLATING EVERY ROW to achive 'depth' number of points
        #
        #
        #     x = cl.get_sonic_cols(x_v_n)
        #     y = cl.get_crit_value(y_v_n)  # Y should be unique value for a given Yc (like m, l/lm, or Yc)
        #     z = cl.get_sonic_cols(z_v_n)
        #
        #     # x, y, z = Math.x_y_z_sort(x, y, z, 0)
        #
        #     if append_crit:
        #         x = np.append(x, cl.get_crit_value(x_v_n))
        #         z = np.append(z, cl.get_crit_value(z_v_n))
        #
        #     xi, zi = Math.x_y_z_sort(x, z)
        #     xi, zi = cut_inf_in_y(xi, zi)
        #
        #
        #     z_grid = Math.interpolate_arr(xi, zi, x_grid, interp_method)
        #
        #     # z_grid = []
        #     # if interp_method == 'IntUni':
        #     #     z_grid = interpolate.InterpolatedUnivariateSpline(xi, zi)(x_grid)
        #     # if interp_method == 'Uni':
        #     #     z_grid = interpolate.UnivariateSpline(xi, zi)(x_grid)
        #     # if interp_method == '1dCubic':
        #     #     z_grid = interpolate.interp1d(xi, zi, kind='cubic', bounds_error=False)(x_grid)
        #     # if interp_method == '1dLinear':
        #     #     z_grid = interpolate.interp1d(xi, zi, kind='linear', bounds_error=False)(x_grid)
        #     # if len(z_grid) == 0:
        #     #     raise NameError('IntMethod is not recognised (or interpolation is failed)')
        #     # z_grid = interpolate.interp1d(xi, zi, kind='cubic', bounds_error=False)(x_grid)
        #
        #     y_zg = np.vstack((y_zg, np.insert(z_grid, 0, y, 0)))
        #
        #     plt.plot(xi, zi, '.', color='red')                # FOR interplation analysis (how good is the fit)
        #     plt.plot(x_grid, z_grid, '-', color='red' )
        #
        #     plt.annotate('m:{}'.format("%.1f"%cl.get_crit_value('m')), xy=(x_grid[-1], z_grid[-1]), textcoords = 'data')
        #     plt.annotate('m:{}'.format("%.1f" % cl.get_crit_value('m')), xy=(x_grid[0], z_grid[0]), textcoords='data')
        #     # plt.legend()
        #     # plt.show()
        # y_zg = np.delete(y_zg, 0, 0)
        # y = y_zg[:, 0]
        # zi = y_zg[:, 1:]
        #
        # # f = interpolate.interp2d(x_grid, y, zi, kind='linear', bounds_error=False)
        # # z_grid2 = f(x_grid, y_grid)
        #
        # z_grid2 = np.zeros(len(y_grid))
        # for i in range(len(x_grid)):  # INTERPOLATING EVERY COLUMN to achive 'depth' number of points
        #     # z_grid2 = np.vstack((z_grid2, interpolate.InterpolatedUnivariateSpline(y, zi[:, i])(y_grid)))
        #     z_grid2 = np.vstack((z_grid2, Math.interpolate_arr(y, zi[:, i], y_grid, interp_method)))
        # z_grid2 = np.delete(z_grid2, 0, 0)
        #
        # x_y_z_final = Math.combine(x_grid, y_grid, z_grid2)

        return x_y_z_final

    @staticmethod
    def x_y_limits(cls, x_v_n, y_v_n, min_or_max = 'min', append_crit = True):
        x_mins = []
        y_mins = []
        x_maxs = []
        y_maxs = []
        for cl in cls:
            x = cl.get_sonic_cols(x_v_n)
            y = cl.get_sonic_cols(y_v_n)
            if append_crit:
                x = np.append(x, cl.get_crit_value(x_v_n))
                y = np.append(y, cl.get_crit_value(y_v_n))
            x_mins = np.append(x_mins, x.min())
            y_mins = np.append(y_mins, y.min())
            x_maxs = np.append(x_maxs, x.max())
            y_maxs = np.append(y_maxs, y.max())
        if min_or_max == 'min':
            return x_mins.max(), x_maxs.min(), y_mins.min(), y_maxs.max()
        if min_or_max == 'max':
            return x_mins.min(), x_maxs.max(), y_mins.min(), y_maxs.max()
        else:
            raise NameError('min_or_max can be only: [{}, or {}] given: {}'.format('min', 'max', min_or_max))

    @staticmethod
    def y_limits(cls, y_v_n, min_or_max='min', append_crit=True):
        y_mins = []
        y_maxs = []
        for cl in cls:
            y = cl.get_sonic_cols(y_v_n)
            if append_crit:
                y = np.append(y, cl.get_crit_value(y_v_n))
            y_mins = np.append(y_mins, y.min())
            y_maxs = np.append(y_maxs, y.max())
        if min_or_max == 'min':
            return  y_mins.min(), y_maxs.max()
        if min_or_max == 'max':
            return  y_mins.min(), y_maxs.max()
        else:
            raise NameError('min_or_max can be only: [{}, or {}] given: {}'.format('min', 'max', min_or_max))

    def plot_x_y_z_for_yc(self, x_v_n, y_v_n, z_v_n, yc_val, depth, min_or_max = 'min', append_crit = True, plot=True):

        yc, cls = self.separate_sp_by_crit_val('Yc', self.yc_prec)
        if yc_val in yc:
            yc_index = Math.find_nearest_index(yc, yc_val)
        else:
            raise ValueError('Value Yc:{} is not is available set {}'.format(yc_val, yc))

        x1, x2, y1, y2 = self.x_y_limits(cls[yc_index], x_v_n, y_v_n, min_or_max, append_crit)
        x_grid = np.mgrid[x1.min():x2.max():depth * 1j]
        y_grid = np.mgrid[y1.min():y2.max():depth * 1j]

        x_y_z = self.x_y_z2d(cls[yc_index],x_v_n, y_v_n, z_v_n, x_grid, y_grid, append_crit)

        if self.set_extrapol_pars != [0, 0, 0, 0]:
            left  = self.set_extrapol_pars[0]
            right = self.set_extrapol_pars[1]
            down  = self.set_extrapol_pars[2]
            up    = self.set_extrapol_pars[3]
            _, x_y_z = Math.extrapolate2(x_y_z, left, right, down, up, 100, 4, True) # Extrapolation

        if plot:

            if not self.set_clean_plots: lbl = 'Yc:{}'.format(yc_val)
            else: lbl = None

            fig = plt.figure()
            ax = fig.add_subplot(111)

            if x_v_n == 't': x_v_n = 'ts'
            if y_v_n == 't': y_v_n = 'ts'


            PlotBackground.plot_color_background(ax, x_y_z, x_v_n, y_v_n, z_v_n, self.metal, self.bump, lbl, 1.0, False, 12, 0)
            if self.set_invert_x_ax:
                ax.invert_xaxis()
            plt.show()

        return x_y_z

    def plot_2_x_y_z_for_yc(self, x_v_n, y_v_n, z_v_n, yc_val, depth, min_or_max = 'min', append_crit = True, plot=True):

        yc, cls = self.separate_sp_by_crit_val('Yc', self.yc_prec)
        if yc_val in yc:
            yc_index = Math.find_nearest_index(yc, yc_val)
        else:
            raise ValueError('Value Yc:{} is not is available set {}'.format(yc_val, yc))

        x1, x2, y1, y2 = self.x_y_limits(cls[yc_index], x_v_n, y_v_n, min_or_max, append_crit)
        x_grid = np.mgrid[x1.min():x2.max():depth * 1j]
        y_grid = np.mgrid[y1.min():y2.max():depth * 1j]

        self.bump          = 'Fe'
        self.set_load_cond = 'ts>5.1'
        x_y_z1 = self.x_y_z2d(cls[yc_index],x_v_n, y_v_n, z_v_n, x_grid, y_grid, append_crit)

        self.bump          = 'HeII'
        self.set_load_cond = 'ts<5.0'
        x_y_z2 = self.x_y_z2d(cls[yc_index], x_v_n, y_v_n, z_v_n, x_grid, y_grid, append_crit)

        from MainClasses import Plot_Critical_Mdot
        cr = Plot_Critical_Mdot('lmc', 'Fe', 1.0)
        cr.set_fill_gray = False

        def cut_table_by_curve(table, x_arr, y_arr):

            for i in range(len(table[1:, 0])):
                for j in range(len(y_arr)):
                    ind = Math.find_nearest_index(table[i,1:], y_arr[j])

        fig = plt.figure()
        ax = fig.add_subplot(111)

        mdot, llm = cr.plot_cr_mdot(y_v_n, yc_val, None, ax)

        plbg = PlotBackground2()
        plbg.set_rotate_labels = 0
        plbg.set_contour_fmt = 0.2
        # plbg.set_label_sise=18

        plbg.set_show_colorbar = False
        plbg.plot_color_background(ax, x_y_z2, x_v_n, y_v_n, z_v_n, self.metal, 'gen')

        arr = np.zeros(len(llm))
        arr.fill(llm.max())

        ax.fill_between(mdot, llm, arr, color="lightgray")

        plbg.set_show_colorbar = True
        plbg.plot_color_background(ax, x_y_z1, x_v_n, y_v_n, z_v_n, self.metal, 'gen')


        plt.show()


        # if self.set_extrapol_pars != [0, 0, 0, 0]:
        #     left  = self.set_extrapol_pars[0]
        #     right = self.set_extrapol_pars[1]
        #     down  = self.set_extrapol_pars[2]
        #     up    = self.set_extrapol_pars[3]
        #     _, x_y_z = Math.extrapolate2(x_y_z, left, right, down, up, 100, 4, True) # Extrapolation
        #
        # if plot:
        #
        #     if not self.set_clean_plots: lbl = 'Yc:{}'.format(yc_val)
        #     else: lbl = None
        #
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111)
        #
        #     if x_v_n == 't': x_v_n = 'ts'
        #     if y_v_n == 't': y_v_n = 'ts'
        #
        #
        #     PlotBackground.plot_color_background(ax, x_y_z, x_v_n, y_v_n, z_v_n, self.metal, self.bump, lbl, 1.0, False, 12, 0)
        #     if self.set_invert_x_ax:
        #         ax.invert_xaxis()
        #     plt.show()

        # return x_y_z


    # --- --- --- BETA LAW & GRADIENTS WORK --- --- ---

    def interp_vinf(self, r_arr, rs, us, vinf_grid, b, grad):

        grads_ = []

        # getting gradients for every vinf from the vinf_grad
        for vinf in vinf_grid:
            vels = Physics.beta_law(r_arr, rs, us, 1000*vinf, b)
            grad_ = np.gradient(vels, r_arr * Constants.solar_r / 10 ** 5)[0] * 10 ** 5
            grads_ = np.append(grads_, grad_)

        # interpolating the vinf for the given gradiend 'grad'
        vinf_ = interpolate.interp1d(grads_, vinf_grid, kind='linear', bounds_error=False)(grad)



        if np.isnan(vinf_):
            # print('\t\tWarning. Grad:{} not found in a grad:[{} {}]'
            #       .format("%.2f" % grad, "%.2f" % grads_.min(), "%.2f" % grads_.max()))

            # plt.plot(vinf_grid, grads_, '.', color='black')
            # plt.axhline(y=grad)
            # plt.show()

            return np.nan
        else:
            return vinf_

    def get_2d_vinfs(self, beta, x_y_r, x_y_u, x_y_grad, vinf_grid):

        len_x = len(x_y_r[0, 1:])
        len_y = len(x_y_r[1:, 0])

        v_infs = np.zeros((len_y, len_x))

        for i in range(len_x):
            print('\t Computing v_inf row #{} (out of {})'.format(i, len_x))
            for j in range(len_y):

                rs = x_y_r[j, i]
                us = x_y_u[j, i]
                grad = x_y_grad[j, i]

                # print('\t Computing v_inf for rs:{}, us:{}, grad:{} '.format(rs, us, grad))
                if not np.isnan(rs) and not np.isnan(us) and not np.isnan(grad):
                    r_arr = np.array([rs, rs + 0.01, rs + 0.02])

                    vinf = self.interp_vinf(r_arr, rs, us, vinf_grid, beta, grad)

                    # tmp = tst()

                    v_infs[j, i] = vinf
                else:
                    v_infs[j, i] = np.nan

        res = Math.combine(x_y_r[0, 1:], x_y_r[1:, 0], v_infs)

        return res

    def save_beta_x_y_vinf(self, x_v_n, y_v_n, v_n_grad, betas, depth, min_or_max = 'min', append_crit = True):

        yc, cls = self.separate_sp_by_crit_val('Yc', self.yc_prec)

        if len(yc) > 1 and yc[0] != 1. :
            raise IOError('This method so far is applicable only for Yc=1. SP files contain: {}'.format(yc))

        # betas = [0.80,0.85,0.90,0.95,1.00,1.05,1.10,1.15,1.20] # 0.80,0.85,0.90,0.95,1.00,1.05,1.10,1.15,1.20

        # betas = [1.0]

        depth_vinf = 20

        vinf_grid = np.mgrid[1.4:3.0:depth_vinf * 1j] # vinf = x*1000

        x_y_z3d = []

        cl = cls[0]

        for beta in betas:

            print('\n <<<< Computing beta = {} >>>>>>'.format(beta))

            x1, x2, y1, y2 = self.x_y_limits(cl, x_v_n, y_v_n, min_or_max, append_crit)
            x_grid = np.mgrid[x1.min():x2.max():depth * 1j]
            y_grid = np.mgrid[y1.min():y2.max():depth * 1j]

            x_y_r    = self.x_y_z2d(cl, x_v_n, y_v_n, 'r', x_grid, y_grid, append_crit)
            x_y_u    = self.x_y_z2d(cl, x_v_n, y_v_n, 'u', x_grid, y_grid, append_crit)
            x_y_grad = self.x_y_z2d(cl, x_v_n, y_v_n, v_n_grad, x_grid, y_grid, append_crit)



            if not np.array_equal(x_y_r[0, 1:], x_y_u[0, 1:]) or not np.array_equal(x_y_r[0, 1:], x_y_grad[0, 1:]):
                raise ValueError('X arrays are not the same')
            if not np.array_equal(x_y_r[1:, 0], x_y_u[1:, 0]) or not np.array_equal(x_y_r[1:, 0], x_y_grad[1:, 0]):
                raise ValueError('Y arrays are not the same')

            x_y_vinfs = self.get_2d_vinfs(beta, x_y_r[1:, 1:], x_y_u[1:,1:], x_y_grad[1:,1:], vinf_grid)

            x_y_vinfs_final = Math.combine(x_y_r[0, 1:], x_y_r[1:, 0], x_y_vinfs)

            x_y_vinfs_final[0, 0] = beta

            PlotBackground2.plot_color_table(x_y_vinfs_final, x_v_n, y_v_n, 'vinf', self.metal, self.bump, 'b: {}'.format(beta))

            x_y_z3d = np.append(x_y_z3d, x_y_vinfs_final)

        res = np.reshape(x_y_z3d, (len(betas), depth+1, depth+1))

        Save_Load_tables.save_3d_table(res, self.metal, self.bump,'{}_{}_{}_{}'.format('beta', x_v_n, y_v_n, 'vinf'), 'beta', x_v_n, y_v_n, 'vinf',
                                       Files.output_dir)



    def wind(self):

        # --- --- SP Boundary conditions --- ---
        mdot = -4.50
        rs   = 1.349
        ts   = 5.3692
        us   = 37.96
        kap_s= -0.25708
        rho_s=-7.826
        mas  = 20
        # --- --- WIND properies
        vinf = 1800
        beta = 1.0
        rinf = 1000
        # --- --- Functions ---

        def plotting(x, y, v_n_x, v_n_y):
            fig = plt.figure()
            ax = fig.add_subplot(111)

            ax.plot(x, y, '-', color='black')

            ax.set_xlabel(Labels.lbls(v_n_x))
            ax.set_ylabel(Labels.lbls(v_n_y))

            if v_n_y == 'tau' : ax.axhline(y=2/3, linestyle='dashed', color='gray')

            ax.minorticks_on()
            plt.show()

        def rho_cont_eq(r, u, rs, us, rho_s):
            return np.log10((us * (10 ** rho_s) * rs**2) / (np.multiply(u, r**2)))

        def integ_tau(r_gr, rho, kap):

            r_gr = r_gr[::-1] * Constants.solar_r
            rho  = rho[::-1]
            kap  = kap[::-1]

            kaprho = np.multiply(kap, rho)
            tau = np.zeros(1)

            for i in range(1, len(r_gr)):
                tau_cur = (kap[-1]*rho[i]) * (r_gr[i - 1] - r_gr[i])
                # tau_cur = (kaprho[i] + 0.5 * (kaprho[i-1] - kaprho[i])) * (r_gr[i-1] - r_gr[i])
                tau = np.append(tau, tau[i-1] + tau_cur)

            tau = np.delete(tau, 0, 0)
            return tau[::-1]

            # tau = np.zeros(1)
            # for i in range(len(r_gr)):
            #     tau_cur = rho[i] * kap[0] * r_gr[i] * Constants.solar_r
            #     tau = np.append(tau, tau[i-1] + tau_cur)
            #
            # tau = np.delete(tau, 0, 0)
            # return tau[::-1]


            # tau = np.zeros(1)
            # diff_r = np.diff(r_gr)
            #
            # kap = np.multiply( kap[:-1], np.abs(np.diff(kap)) )[::-1]
            # rho = np.multiply( rho[:-1], np.abs(np.diff(rho)) )[::-1]
            # diff_r=np.abs(diff_r[::-1])
            #
            # for i in range(1, len(diff_r)):
            #     # print(i)
            #     tau_cur = kap[i]*rho[i]*diff_r[i]*Constants.solar_r
            #     tau = np.append(tau, tau[i-1] + tau_cur)
            #
            # return np.log10(tau[::-1])

        # --- --- MAIN --- ---
        r_gr = np.mgrid[rs:rinf:1000j]

        u_w = Physics.beta_law(r_gr, rs, us, vinf, beta)
        # plotting(r_gr, u_w, 'r', 'u')

        rho_w = rho_cont_eq(r_gr, u_w, rs, us, rho_s)
        # plotting(r_gr, rho_w, 'r', 'rho')

        vesc = Physics.get_v_esc(mas, rs)
        k_eff= Physics.kap_eff(r_gr, kap_s, beta, vinf, vesc, rs)
        # plotting(r_gr, k_eff, 'r', 'k')
        k_eff = 10**k_eff
        rho_w = 10**rho_w

        tau = integ_tau(r_gr, rho_w, k_eff)
        plotting(r_gr[:-1], tau, 'r', 'tau')


    # --- --- ---

    def plot_beta_x_y_vinf(self,x_v_n, y_v_n, z_v_n, betas_to_plot):

        b_x_y_z = Save_Load_tables.load_3d_table(self.metal, '', 'beta_{}_{}_{}'
                                                       .format(x_v_n, y_v_n, z_v_n), 'beta', x_v_n,
                                                       y_v_n, z_v_n)
        betas = b_x_y_z[:, 0, 0]

        for i in range(len(betas_to_plot)):
            if not betas_to_plot[i] in betas:
                raise ValueError('Value betas_to_plot[{}] not in betas:\n\t {}'.format(betas_to_plot[i], betas))

        beta_vals = np.sort(betas_to_plot, axis=0)

        b_n = len(betas_to_plot)

        fig = plt.figure()
        fig.subplots_adjust(hspace=0.2, wspace=0.3)

        for i in range(1, b_n + 1):
            print(i)
            b_val = betas_to_plot[i - 1]

            ind = Math.find_nearest_index(beta_vals, b_val)
            x_y_z = b_x_y_z[ind, :, :]
            # x_y_z = Math.extrapolate(x_y_z, 10, None, None, 25, 500,
            #                               'unispline')  # 2 is better to linear part

            if b_n % 2 == 0:
                ax = fig.add_subplot(2, b_n / 2, i)
            else:
                ax = fig.add_subplot(1, b_n, i)

            # fig = plt.figure(figsize=plt.figaspect(0.8))
            # ax = fig.add_subplot(111)  # , projection='3d'
            PlotBackground.plot_color_background(ax, x_y_z, x_v_n, y_v_n, z_v_n, self.metal,
                                                 'b: {}'.format(beta_vals))
            # self.plot_obs_t_llm_mdot_int(ax, t_llm_mdot, y_v_n, self.lim_t1, self.lim_t2, True)

            # if self.set_coeff != 1.0:
            #     ax.text(0.5, 0.9, 'K:{}'.format(self.set_coeff), style='italic',
            #             bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center',
            #             verticalalignment='center', transform=ax.transAxes)

        plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
        plot_name = Files.plot_dir + 'sonic_HRD.pdf'
        plt.savefig(plot_name)
        # plt.gca().invert_xaxis()
        plt.show()



    @staticmethod
    def mosaic(vol, fig=None, title=None, size=[10, 10], vmin=None, vmax=None,
               return_mosaic=False, cbar=True, return_cbar=False, **kwargs):
        """
        Display a 3-d volume of data as a 2-d mosaic

        Parameters
        ----------
        vol: 3-d array
           The data

        fig: matplotlib figure, optional
            If this should appear in an already existing figure instance

        title: str, optional
            Title for the plot

        size: [width, height], optional

        vmin/vmax: upper and lower clip-limits on the color-map

        **kwargs: additional arguments to matplotlib.pyplot.matshow
           For example, the colormap to use, etc.

        Returns
        -------
        fig: The figure handle

        """
        if vmin is None:
            vmin = np.nanmin(vol)
        if vmax is None:
            vmax = np.nanmax(vol)

        sq = int(np.ceil(np.sqrt(len(vol))))

        # Take the first one, so that you can assess what shape the rest should be:
        im = np.hstack(vol[0:sq])
        height = im.shape[0]
        width = im.shape[1]

        # If this is a 4D thing and it has 3 as the last dimension
        if len(im.shape) > 2:
            if im.shape[2] == 3 or im.shape[2] == 4:
                mode = 'rgb'
            else:
                e_s = "This array has too many dimensions for this"
                raise ValueError(e_s)
        else:
            mode = 'standard'

        for i in range(1, sq):
            this_im = np.hstack(vol[(len(vol) / sq) * i:(len(vol) / sq) * (i + 1)])
            wid_margin = width - this_im.shape[1]
            if wid_margin:
                if mode == 'standard':
                    this_im = np.hstack([this_im,
                                         np.nan * np.ones((height, wid_margin))])
                else:
                    this_im = np.hstack([this_im,
                                         np.nan * np.ones((im.shape[2],
                                                           height,
                                                           wid_margin))])
            im = np.concatenate([im, this_im], 0)

        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, aspect='equal')
        else:
            # This assumes that the figure was originally created with this
            # function:
            ax = fig.axes[0]

        if mode == 'standard':
            imax = ax.matshow(im.T, vmin=vmin, vmax=vmax, **kwargs)
        else:
            imax = plt.imshow(np.rot90(im), interpolation='nearest')
            cbar = False
        ax.get_axes().get_xaxis().set_visible(False)
        ax.get_axes().get_yaxis().set_visible(False)
        returns = [fig]
        if cbar:
            # The colorbar will refer to the last thing plotted in this figure
            cbar = fig.colorbar(imax, ticks=[np.nanmin([0, vmin]),
                                             vmax - (vmax - vmin) / 2,
                                             np.nanmin([vmax, np.nanmax(im)])],
                                format='%1.2f')
            if return_cbar:
                returns.append(cbar)

        if title is not None:
            ax.set_title(title)
        if size is not None:
            fig.set_size_inches(size)

        if return_mosaic:
            returns.append(im)

        # If you are just returning the fig handle, unpack it:
        if len(returns) == 1:
            returns = returns[0]

        return returns




    def plot_beta_x_y_vinf_3d(self,x_v_n, y_v_n, z_v_n, betas_to_plot):

        fsz = 12
        alpha = 1.0


        b_x_y_z = Save_Load_tables.load_3d_table(self.metal, '', 'beta_{}_{}_{}'
                                                       .format(x_v_n, y_v_n, z_v_n), 'beta', x_v_n,
                                                       y_v_n, z_v_n)
        betas = b_x_y_z[:, 0, 0]

        for i in range(len(betas_to_plot)):
            if not betas_to_plot[i] in betas:
                raise ValueError('Value betas_to_plot[{}] not in betas:\n\t {}'.format(betas_to_plot[i], betas))

        fig = plt.figure() # figsize=(10, 5) changes the overall size of the popping up window
        ax = fig.gca(projection='3d')
        pg = PlotBackground2()
        # pg.plot_3d_curved_surf(x_arr, y_arr)
        pg.plot_3d_back(ax, b_x_y_z, x_v_n, y_v_n, z_v_n, self.metal,0,50)
        plt.tight_layout(0.05, 0.05, 0.05) # <0.1 lets you to malke it tighter
        plt.show()



        beta_vals = np.sort(betas_to_plot, axis=0)

        b_n = len(betas_to_plot)

        # self.mosaic(b_x_y_z)

        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')

        # ax.set_xlim(x_y_z[0, 1:].min(), x_y_z[0, 1:].max())
        # ax.set_ylim(x_y_z[1:, 0].min(), x_y_z[1:, 0].max())



        from mpl_toolkits.mplot3d import Axes3D
        # import numpy as np
        # import matplotlib.pyplot as plt

        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        #
        # PlotBackground2.plot_3d_back(b_x_y_z, ax)





        for i in range(1, b_n + 1):
            print(i)
            b_val = betas_to_plot[i - 1]

            ind = Math.find_nearest_index(beta_vals, b_val)
            x_y_z = b_x_y_z[ind, :, :]
            # x_y_z = Math.extrapolate(x_y_z, 10, None, None, 25, 500,
            #                               'unispline')  # 2 is better to linear part
            # if label != None:
            #     print('TEXT')

            # ax.text(table[0, 1:].min(), table[1:, 0].min(), s=label)
            # bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10}
            # plt.text(2, 0.65, r'$\cos(2 \pi t) \exp(-t)$')

            # ax = fig.add_subplot(1, 1, 1)

            ax.set_xlim(x_y_z[0, 1:].min(), x_y_z[0, 1:].max())
            ax.set_ylim(x_y_z[1:, 0].min(), x_y_z[1:, 0].max())
            ax.set_ylabel(Labels.lbls(x_y_z), fontsize=fsz)
            ax.set_xlabel(Labels.lbls(x_y_z), fontsize=fsz)

            levels = Levels.get_levels(z_v_n, self.metal, '')

            # 'RdYlBu_r'

            contour_filled = plt.contourf(table[0, 1:], table[1:, 0], table[1:, 1:], levels,
                                          cmap=plt.get_cmap('RdYlBu_r'), alpha=alpha)
            clb = plt.colorbar(contour_filled)
            clb.ax.set_title(Labels.lbls(v_n_z), fontsize=fsz)

            # ax.colorbar(contour_filled, label=Labels.lbls(v_n_z))

            contour = plt.contour(table[0, 1:], table[1:, 0], table[1:, 1:], levels, colors='k')

            if v_n_x == 'mdot' and v_n_y == 'lm' and v_n_z == 'tau':
                labs = ax.clabel(contour, colors='k', fmt='%2.0f', fontsize=fsz, manual=True)
            else:
                labs = ax.clabel(contour, colors='k', fmt='%2.2f', fontsize=fsz)

            if rotation != None:
                for lab in labs:
                    lab.set_rotation(rotation)  # ORIENTATION OF LABELS IN COUNTUR PLOTS
            # ax.set_title('SONIC HR DIAGRAM')

            # print('Yc:{}'.format(yc_val))
            if not clean and label != None and label != '':
                ax.text(0.9, 0.1, label, style='italic',
                        bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center',
                        verticalalignment='center', transform=ax.transAxes)

            ax.tick_params('y', labelsize=fsz)
            ax.tick_params('x', labelsize=fsz)
            # if b_n % 2 == 0:
            #     ax = fig.add_subplot(2, b_n / 2, i)
            # else:
            #     ax = fig.add_subplot(1, b_n, i)

            # fig = plt.figure(figsize=plt.figaspect(0.8))
            # ax = fig.add_subplot(111)  # , projection='3d'
            PlotBackground.plot_color_background(ax, x_y_z, x_v_n, y_v_n, z_v_n, self.metal,
                                                 'b: {}'.format(beta_vals))
            # self.plot_obs_t_llm_mdot_int(ax, t_llm_mdot, y_v_n, self.lim_t1, self.lim_t2, True)

            # if self.set_coeff != 1.0:
            #     ax.text(0.5, 0.9, 'K:{}'.format(self.set_coeff), style='italic',
            #             bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center',
            #             verticalalignment='center', transform=ax.transAxes)

        plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
        plot_name = Files.plot_dir + 'sonic_HRD.pdf'
        plt.savefig(plot_name)
        # plt.gca().invert_xaxis()
        plt.show()

    # --- --- --- --- --- --- ---


    def plot_x_y_z(self, x_v_n, y_v_n, z_v_n, yc_arr, depth, min_or_max = 'min', append_crit = True):

        yc, cls = self.separate_sp_by_crit_val('Yc', self.yc_prec)

        yc_arr = np.sort(yc_arr, axis=0)

        ind_arr = []
        for yc_val in yc_arr:
            ind_arr = np.append(ind_arr, np.int(Physics.ind_of_yc(yc, yc_val)))

        yc_n = np.int(len(yc_arr))

        fig = plt.figure()
        fig.subplots_adjust(hspace=0.2, wspace=0.3)

        for i in range(1, len(yc_arr)+1):

            if yc_n % 2 == 0: ax = fig.add_subplot(2, yc_n/2, i)
            else:             ax = fig.add_subplot(1, yc_n, i)

            yc_val = yc_arr[i-1]
            print(yc_val)

            x_y_z = self.plot_x_y_z_for_yc(x_v_n, y_v_n, z_v_n, yc_val, depth, min_or_max, append_crit, False)
            PlotBackground.plot_color_background(ax, x_y_z, x_v_n, y_v_n, z_v_n, self.metal, 'Yc:{}'.format(yc_val))

        # plt.legend(bbox_to_anchor=(1, 0), loc='lower right', ncol=1)
        plot_name = self.plot_dir + '{}_{}_{}.pdf'.format(x_v_n, y_v_n, z_v_n)
        plt.savefig(plot_name)
        plt.show()


    @staticmethod
    def lm_l(lm, yc_lm_l, yc_val):
        '''
        Give it lm (value of array) and yc_lm_l relation from the file, --> l (value or array)
        :param lm:
        :param yc_lm_l:
        :param yc_val:
        :return:
        '''

        def prec2(var):
            var = "%.2f" % var
            return np.float(var)

        yc = yc_lm_l[0, 1:]
        lm_ = yc_lm_l[1:, 0]
        l2d = yc_lm_l[1:, 1:]

        if not yc_val in yc:
            raise ValueError('Yc:{} not in Yc array from <yc_lm_l> relation ({})'.format(yc_val, yc))

        if prec2(lm.min()) < prec2(lm_.min()):
            raise ValueError('lm.min({}) < lm_.min({})'.format(lm.min(), lm_.min()))
        if prec2(lm.max()) > prec2(lm_.max()):
            raise ValueError('lm.max({}) > lm_.max({})'.format(lm.max(), lm_.max()))

        i_yc = Math.find_nearest_index(yc, yc_val)
        l    = interpolate.InterpolatedUnivariateSpline(lm_, l2d[:, i_yc])(lm)

        return l

    def get_t_llm_rho_crop_for_yc(self, cls, t_lm_rho, yc_val, y_v_n, min_max = 'min', opal_used = None, append_crit = True):

        # t_lm_rho = Save_Load_tables.load_table('t_lm_rho', 't', 'lm', 'rho', opal_used)
        yc_lm_l = Save_Load_tables.load_table('yc_lm_l', 'yc', 'lm', 'l', opal_used, '')
        t_lm_rho_com, yc_lm_l_com = Math.common_y(t_lm_rho, yc_lm_l)

        if y_v_n == 'l':
            l = self.lm_l(t_lm_rho_com[1:, 0], yc_lm_l, yc_val)

            t_l_rho = Math.combine(t_lm_rho_com[0, 1:], l, t_lm_rho_com[1:, 1:])

            x1, x2, y1, y2 = self.x_y_limits(cls, 't', y_v_n, min_max, append_crit)
            t_l_rho_crop = Math.crop_2d_table2(t_l_rho, x1, x2, y1, y2)  # taking absolut limits

            return t_l_rho_crop

        if y_v_n == 'lm':
            x1, x2, y1, y2 = self.x_y_limits(cls, 't', y_v_n, min_max, append_crit)
            t_lm_rho_crop = Math.crop_2d_table2(t_lm_rho_com, x1, x2, y1, y2)

            return t_lm_rho_crop

    def get_t_llm_mdot_for_yc(self, cls, t_lm_rho, yc_val, y_v_n, min_max = 'min', opal_used=None, bump=None, append_crit = True):

        t_llm_rho_ = self.get_t_llm_rho_crop_for_yc(cls, t_lm_rho, yc_val, y_v_n, min_max, opal_used, append_crit)

        t_llm_r_ = self.x_y_z(cls, 't', y_v_n, 'r', t_llm_rho_[0, 1:], t_llm_rho_[1:, 0],
                             append_crit, self.set_init_fit_method, self.set_check_x_y_z_arrs)

        # in case the nans are removed from t_llm_r, the t_llm_rho has to be cropped.
        # if len( )


        # t_llm_rho = Math.crop_2d_table2(t_llm_rho, t_llm_r[0,1],t_llm_r[0,-1], t_llm_r[1,0],t_llm_r[-1,0])
        t_llm_r, t_llm_rho = Math.common_y(t_llm_r_, t_llm_rho_)


        # t_llm_r = self.x_y_z2d(cls, 't', y_v_n, 'r', t_llm_rho[0, 1:], t_llm_rho[1:, 0], append_crit, self.set_init_fit_method)

        if self.set_do_tech_plots:
            PlotBackground2.plot_color_table(t_llm_r, 't', 'lm', 'r', self.metal, bump)

        rho2d= t_llm_rho[1:, 1:]
        r2d  = t_llm_r[1:, 1:]
        t    = t_llm_r[0, 1:]
        llm  = t_llm_r[1:, 0]

        vro = Physics.get_vrho(t, rho2d, 2, np.array([1.34]))
        mdot= Physics.vrho_mdot(vro, r2d,'tl')
        return Math.combine(t, llm, mdot, yc_val)
    def plot_t_llm_mdot_for_yc(self, yc_val, l_or_lm, coeff, bump, min_max = 'min', append_crit = True):

        yc, cls = self.separate_sp_by_crit_val('Yc', self.yc_prec)
        if yc_val in yc:
            yc_index = Math.find_nearest_index(yc, yc_val)
        else:
            raise ValueError('Value Yc:{} is not is available set {}'.format(yc_val, yc))

        t_lm_rho = Save_Load_tables.load_table('t_{}lm_rho'.format(coeff), 't', '{}lm'.format(coeff), 'rho', self.metal, bump)
        t_llm_mdot = self.get_t_llm_mdot_for_yc(cls[yc_index], t_lm_rho, yc[yc_index], l_or_lm, min_max, self.metal, append_crit)

        # from PhysMath import Get_Z
        z = Get_Z.z(self.metal)
        PlotBackground.plot_color_table(t_llm_mdot, 't', l_or_lm, 'mdot', self.metal, None) # 'z:{}({}) Yc:{} K:{}'.format(z, bump, yc_val, coeff)

    def get_t_llm_rho_crop_for_yc_const_r(self, cls, t_lm_rho, yc_val, y_v_n, min_max = 'min', opal_used = None, append_crit = True):

        # t_lm_rho = Save_Load_tables.load_table('t_lm_rho', 't', 'lm', 'rho', opal_used)
        yc_lm_l = Save_Load_tables.load_table('yc_lm_l', 'yc', 'lm', 'l', opal_used, '')
        t_lm_rho_com, yc_lm_l_com = Math.common_y(t_lm_rho, yc_lm_l)

        if y_v_n == 'l':
            l = self.lm_l(t_lm_rho_com[1:, 0], yc_lm_l, yc_val)

            t_l_rho = Math.combine(t_lm_rho_com[0, 1:], l, t_lm_rho_com[1:, 1:])

            y1, y2 = self.y_limits(cls, y_v_n, min_max, append_crit)
            t_l_rho_crop = Math.crop_2d_table2(t_l_rho, None, None, y1, y2)  # taking absolut limits

            return t_l_rho_crop

        if y_v_n == 'lm':
            y1, y2 = self.y_limits(cls, y_v_n, min_max, append_crit)
            t_lm_rho_crop = Math.crop_2d_table2(t_lm_rho_com, None, None, y1, y2)

            return t_lm_rho_crop
    def get_t_llm_mdot_for_yc_const_r(self, cls, t_lm_rho, yc_val, rs, y_v_n, min_max = 'min', opal_used = None, append_crit = True):

        t_llm_rho = self.get_t_llm_rho_crop_for_yc_const_r(cls, t_lm_rho, yc_val, y_v_n, min_max, opal_used, append_crit)

        # t_llm_r = self.x_y_z(cls, 't', y_v_n, 'r', t_llm_rho[0, 1:], t_llm_rho[1:, 0], append_crit)

        rho2d= t_llm_rho[1:, 1:]
        t = t_llm_rho[0, 1:]
        llm =t_llm_rho[1:, 0]
        # r2d  = t_llm_r[1:, 1:]
        # t    = t_llm_r[0, 1:]
        # llm  = t_llm_r[1:, 0]

        vro = Physics.get_vrho(t, rho2d, 2, np.array([1.34]))
        mdot= Physics.vrho_mdot(vro, rs, '')
        return Math.combine(t, llm, mdot, yc_val)
    def plot_t_llm_mdot_for_yc_const_r(self, yc_val, rs, l_or_lm, coeff, bump, min_max = 'min', append_crit = True, plot=True):

        yc, cls = self.separate_sp_by_crit_val('Yc', self.yc_prec)
        if yc_val in yc:
            yc_index = Math.find_nearest_index(yc, yc_val)
        else:
            raise ValueError('Value Yc:{} is not is available set {}'.format(yc_val, yc))

        t_lm_rho = Save_Load_tables.load_table('t_{}lm_rho'.format(coeff), 't', '{}lm'.format(coeff), 'rho', self.metal, bump)
        t_llm_mdot = self.get_t_llm_mdot_for_yc_const_r(cls[yc_index], t_lm_rho, yc[yc_index], rs, l_or_lm, min_max, self.metal, append_crit)

        if plot:
            lbl = 'Yc:{} R:{}'.format(yc_val, rs)
            if self.set_clean_plots: lbl = None

            PlotBackground.plot_color_table(t_llm_mdot, 'ts', l_or_lm, 'mdot', self.metal, lbl)

        # Save_Load_tables.save_3d_table(t_llm_mdot, self.opal_used, 'yc_t_{}_mdot_r_{}'.format(l_or_lm, rs), 'yc', 't',
        #                                l_or_lm, 'mdot_r_{}'.format(rs), self.out_dir)


    def save_min_max_lm(self, v_n):
        yc, cls = self.separate_sp_by_crit_val('Yc', self.yc_prec)

        ys_zams = cls[-1][0].get_crit_value('ys')

        fig = plt.figure()
        ax = fig.add_subplot(111)

        yc_lm_min_max = np.zeros(3)
        for i in range(len(yc)):
            lm_tmp = []
            ys_tmp = []
            for c in cls[i]:
                lm_tmp = np.append(lm_tmp, c.get_crit_value(v_n))
                ys_tmp = np.append(ys_tmp, c.get_crit_value('ys'))
            lm_tmp, ys_tmp = Math.x_y_z_sort(lm_tmp, ys_tmp)
            tmp = np.zeros(len(lm_tmp))
            tmp.fill(yc[i])
            ax.plot(tmp, lm_tmp, '.', color='gray')

            lm_tmp2 = []
            for j in range(len(cls[i])):
                if ys_tmp[j] < ys_zams: break
                else: lm_tmp2 = np.append(lm_tmp2, lm_tmp[j])

            if len(lm_tmp2) == 0:
                raise ValueError('No values with zams Ys  found for Yc: {}'.format(yc[i]))
            lm_max = lm_tmp2[-1]
            lm_min = lm_tmp2[0]
            # print(i)
            yc_lm_min_max = np.vstack((yc_lm_min_max, [yc[i], lm_min, lm_max]))

        # yc_lm_min_max = np.reshape(yc_lm_min_max, (len(yc), 3))

        # yc_lm_min_max = np.insert(yc_lm_min_max,0,  np.zeros(len(yc_lm_min_max[0, 1:])), 1)
        yc_lm_min_max = yc_lm_min_max.T

        ax.plot(yc_lm_min_max[0,1:], yc_lm_min_max[1,1:], '-', color='blue', label='WNE region')
        ax.plot(yc_lm_min_max[0, 1:], yc_lm_min_max[2, 1:], '-', color='blue')
        ax.fill_between(yc_lm_min_max[0,1:], yc_lm_min_max[1,1:], yc_lm_min_max[2, 1:], color='blue',
                         alpha='.3')
        ax.set_xlabel(Labels.lbls('yc'))
        ax.set_ylabel(Labels.lbls(v_n))
        ax.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
        ax.grid()
        plt.show()

        Save_Load_tables.save_table(yc_lm_min_max, self.metal, '',
                                    'yc_nan_{}lim'.format(v_n), 'yc', 'nan', '{}lim'.format(v_n))

            # lm_arr = np.append(lm_arr, cl.get_crit_value('lm'))
            # ys_arr = np.append(ys_arr, cl.get_crit_value('ys'))
        # print('a')
        # yc, lm_arr, ys_arr = Math.x_y_z_sort(yc, lm_arr, ys_arr)

    @staticmethod
    def get_square(table, depth):
        '''
        Performs row by row and column bu column interpolation of a table to get a square matrix with required depth
        :param table:
        :param depth:
        :return:
        '''
        x = table[0, 1:]
        y = table[1:, 0]
        z = table[1:,1:]

        z1 = np.zeros(depth)
        x_grid = np.mgrid[x.min():x.max():depth*1j]
        for i in range(len(y)):
            tmp = interpolate.InterpolatedUnivariateSpline(x, z[i, :])(x_grid)
            z1 = np.vstack((z1, tmp))

        z1 = np.delete(z1, 0, 0)

        z2 = np.zeros((depth))
        y_grid = np.mgrid[y.min():y.max():depth*1j]
        for i in range(depth):
            tmp = interpolate.InterpolatedUnivariateSpline(y, z1[:, i])(y_grid)
            z2 = np.vstack((z2, tmp))


        z2 = np.delete(z2, 0, 0)
        return Math.combine(x_grid, y_grid, z2.T)

    def save_t_llm_mdot(self, l_or_lm, coeff, bump, depth_square = 500, min_max = 'min', append_crit = True):

        yc, cls = self.separate_sp_by_crit_val('Yc', self.yc_prec)
        t_lm_rho = Save_Load_tables.load_table('t_{}lm_rho'.format(coeff), 't', '{}lm'.format(coeff), 'rho', self.metal, bump)

        print('\n <<< INITIALISATION *save_t_llm_mdot* >>>>\n')

        print('\t__Note: Limits from t_lm_rho: t[{}, {}], lm[{}, {}]'
              .format(t_lm_rho[0,1], t_lm_rho[0,-1], t_lm_rho[1,0], t_lm_rho[-1, 0]))

        # yc_t_llm_mdot = np.array([[[k*j*i for k in np.arange(0, 5, 1)] for j in np.arange(0, 5, 1)] for i in np.arange(0, 5, 1)])
        yc_t_llm_mdot = []
        for i in range(len(yc)):
            print('\n<--- --- --- ({}) Yc:{} K:{} --- --- --->\n'.format(bump, yc[i], coeff))
            tmp1 = self.get_t_llm_mdot_for_yc(cls[i], t_lm_rho, yc[i], l_or_lm, min_max, self.metal, bump, append_crit)
            tmp2 = self.get_square(tmp1, depth_square)
            tmp2[0, 0] = yc[i] # appending the yc value as a [0, 0] in the array.
            # print(tmp2.shape)
            yc_t_llm_mdot = np.append(yc_t_llm_mdot, tmp2)

            if self.set_do_tech_plots:
                PlotBackground2.plot_color_table(t_lm_rho, 't', 'lm', 'rho', self.metal, bump)
                PlotBackground2.plot_color_table(tmp1, 't', 'lm', 'mdot', self.metal, bump)

        yc_t_llm_mdot = np.reshape(yc_t_llm_mdot, (len(yc), depth_square+1, depth_square+1))
        print('\t__Saving the ({}) yc_t_{}{}_mdot table for. Shape:{}'.format(bump, coeff, l_or_lm, yc_t_llm_mdot.shape))



        Save_Load_tables.save_3d_table(yc_t_llm_mdot, self.metal, bump, 'yc_t_{}{}_mdot'.format(coeff, l_or_lm),
                                       'yc', 't', str(coeff) + l_or_lm, 'mdot', self.out_dir)

        # print(Save_Load_tables.load_3d_table(self.opal_used, 'yc_t_l_mdot', 'yc', 't', 'l', 'mdot', self.out_dir))

    def save_t_llm_mdot_const_r(self, l_or_lm, coeff, bump, rs, depth_square = 500, min_max = 'min', append_crit = True):

        yc, cls = self.separate_sp_by_crit_val('Yc', self.yc_prec)
        t_lm_rho = Save_Load_tables.load_table('t_{}lm_rho'.format(coeff), 't', '{}lm'.format(coeff), 'rho', self.metal, bump)

        # yc_t_llm_mdot_rs_const = np.array([[[k*j*i for k in np.arange(0, 5, 1)] for j in np.arange(0, 5, 1)] for i in np.arange(0, 5, 1)])
        yc_t_llm_mdot_rs_const = []
        for i in range(len(yc)):
            print('\n<--- --- --- ({}) Yc:{} K:{} Rs:{} (const) --- --- --->\n'.format(bump, yc[i], coeff, rs))
            tmp1 = self.get_t_llm_mdot_for_yc_const_r(cls[i], t_lm_rho, yc[i], rs, l_or_lm, min_max, self.metal, append_crit)
            tmp2 = self.get_square(tmp1, depth_square)
            tmp2[0, 0] = yc[i]  # appending the yc value as a [0, 0] in the array.
            # print(tmp2.shape)
            yc_t_llm_mdot_rs_const = np.append(yc_t_llm_mdot_rs_const, tmp2)

            print('\t__ L/M limits: OPAL:[{},{}] FINAL:[{},{}]'.format(t_lm_rho[1:,0].min(), t_lm_rho[1:,0].max(),
                                                                       tmp2[1:,0].min(), tmp2[1:,0].max()))

        yc_t_llm_mdot_rs_const = np.reshape(yc_t_llm_mdot_rs_const, (len(yc), depth_square + 1, depth_square + 1))
        print('\t__Saving the ({}) yc_t_{}{}_mdot table for Rs = {} (const). Shape:{}'.format(bump, coeff, l_or_lm, rs,
                                                                                              yc_t_llm_mdot_rs_const.shape))

        Save_Load_tables.save_3d_table(yc_t_llm_mdot_rs_const, self.metal, bump, 'yc_t_{}{}_mdot_rs_{}'.format(coeff, l_or_lm, rs), 'yc', 't',
                                       str(coeff) + l_or_lm, 'mdot_rs_{}'.format(rs), self.out_dir)

    # ------------------------------------------------

    # -------------------------------------------------
    def test(self, yc_val = 1.0):
        yc, cls = self.separate_sp_by_crit_val('Yc', self.yc_prec)
        if yc_val in yc:
            yc_index = Math.find_nearest_index(yc, yc_val)
        else:
            raise ValueError('Value Yc:{} is not is available set {}'.format(yc_val, yc))

        cl = cls[yc_index]

        arr = []

        for c in cl:
            mdot = c.get_crit_value('mdot')
            r = c.get_crit_value('r')
            m = c.get_crit_value('m')
            l = c.get_crit_value('l')


            arr = np.append(arr, [mdot, r, m , l])

            # print(cl)

        arr_sort = np.sort(arr.view('f8, f8, f8, f8'), order=['f3'], axis=0).view(np.float)
        arr_shaped = np.reshape(arr_sort, (np.int(len(arr_sort)/4), 4))

        var = (1/arr_shaped[:,1])*(arr_shaped[:,2]/arr_shaped[:,3])*arr_shaped[:,0]

        plt.plot(arr_shaped[:,0], var, '.', color='black')
        plt.grid()
        plt.xlabel('mdot')
        plt.ylabel('MY')
        plt.show()

        t_lm_rho = Save_Load_tables.load_table('t_1.0lm_rho','t','1.0lm','rho', self.metal, 'Fe')
        t = t_lm_rho[0, 1:]
        lm = t_lm_rho[1:, 0]
        k = Physics.loglm_logk(lm, True)
        rho = t_lm_rho[1:, 1:]
        vrho = Physics.get_vrho(t, rho, 2)

        t_k_vrho = Math.combine(t,k,vrho)
        mdot = Physics.vrho_mdot(vrho,1.0,'')
        mins = Math.get_maxs_in_every_row(t,k,vrho,5000)

        fig = plt.figure(figsize=plt.figaspect(0.8))
        ax = fig.add_subplot(111)  # , projection='3d'

        Plots.plot_color_background(ax, t_k_vrho,'t','k','vrho', self.metal)
        ax.plot(mins[0,:],mins[1,:])
        plt.show()

        var2 = []
        for i in range(len(mins[0,:])):
            var2 = np.append(var2, mins[1,i]*mins[2,i])




        print(var)
        print('haha: {}'.format(Constants.light_v*Constants.grav_const/np.sqrt(Constants.k_b/Constants.m_H)))

    # def save_y_yc_z_relation_sp(self, x_v_n, y_v_n, z_v_n, save, plot=False, depth=100):
    #     append_crit = True
    #     if not y_v_n in ['m', 'l', 'lm', 'Yc']:
    #         raise NameError('y_v_n must be one of [{}] , give:{}'.format(['m', 'l', 'lm', 'Yc'], y_v_n))
    #
    #     def x_y_z_sort(x_arr, y_arr, z_arr=None, sort_by_012=0):
    #         '''
    #         RETURNS x_arr, y_arr, (z_arr) sorted as a matrix by a row, given 'sort_by_012'
    #         :param x_arr:
    #         :param y_arr:
    #         :param z_arr:
    #         :param sort_by_012:
    #         :return:
    #         '''
    #
    #         if z_arr == None and sort_by_012 < 2:
    #             if len(x_arr) != len(y_arr):
    #                 raise ValueError('len(x)[{}]!= len(y)[{}]'.format(len(x_arr), len(y_arr)))
    #
    #             x_y_arr = []
    #             for i in range(len(x_arr)):
    #                 x_y_arr = np.append(x_y_arr, [x_arr[i], y_arr[i]])
    #
    #             x_y_sort = np.sort(x_y_arr.view('float64, float64'), order=['f{}'.format(sort_by_012)], axis=0).view(np.float)
    #             x_y_arr_shaped = np.reshape(x_y_sort, (int(len(x_y_sort) / 2), 2))
    #             return x_y_arr_shaped[:,0], x_y_arr_shaped[:,1]
    #
    #         if z_arr != None:
    #             if len(x_arr) != len(y_arr) or len(x_arr)!=len(z_arr):
    #                 raise ValueError('len(x)[{}]!= len(y)[{}]!=len(z_arr)[{}]'.format(len(x_arr), len(y_arr), len(z_arr)))
    #
    #             x_y_z_arr = []
    #             for i in range(len(x_arr)):
    #                 x_y_z_arr = np.append(x_y_z_arr, [x_arr[i], y_arr[i], z_arr[i]])
    #
    #             x_y_z_sort = np.sort(x_y_z_arr.view('float64, float64, float64'), order=['f{}'.format(sort_by_012)], axis=0).view(
    #                 np.float)
    #             x_y_z_arr_shaped = np.reshape(x_y_z_sort, (int(len(x_y_z_sort) / 3), 3))
    #             return x_y_z_arr_shaped[:, 0], x_y_z_arr_shaped[:, 1], x_y_z_arr_shaped[:, 2]
    #
    #     def x_y_limits(cls, min_or_max):
    #         x_mins = []
    #         y_mins = []
    #         x_maxs = []
    #         y_maxs = []
    #         for cl in cls:
    #             x = cl.get_sonic_cols(x_v_n)
    #             y = cl.get_sonic_cols(y_v_n)
    #             if append_crit:
    #                 x = np.append(x, cl.get_crit_value(x_v_n))
    #                 y = np.append(y, cl.get_crit_value(y_v_n))
    #             x_mins = np.append(x_mins, x.min())
    #             y_mins = np.append(y_mins, y.min())
    #             x_maxs = np.append(x_maxs, x.max())
    #             y_maxs = np.append(y_maxs, y.max())
    #         if min_or_max == 'min':
    #             return x_mins.max(), x_maxs.min(), y_mins.min(), y_maxs.max()
    #         if min_or_max == 'max':
    #             return x_mins.min(), x_maxs.max(), y_mins.min(), y_maxs.max()
    #         else:
    #             raise NameError('min_or_max can be only: [{}, or {}] given: {}'.format('min', 'max', min_or_max))
    #
    #     def common_y(arr1, arr2):
    #
    #         y1 = arr1[1:, 0]
    #         y2 = arr2[1:, 0]
    #
    #         y_min = np.array([y1.min(), y2.min()]).max()
    #         y_max = np.array([y1.max(), y2.max()]).min()
    #
    #         arr1_cropped = Math.crop_2d_table(arr1, None, None, y_min, y_max)
    #         arr2_cropped = Math.crop_2d_table(arr2, None, None, y_min, y_max)
    #
    #         return arr1_cropped, arr2_cropped
    #
    #     def lm_l(lm, yc_lm_l, yc_val):
    #
    #         yc = yc_lm_l[0,  1:]
    #         lm_= yc_lm_l[1:,  0]
    #         l2d= yc_lm_l[1:, 1:]
    #
    #         if not yc_val in yc:
    #             raise ValueError('Yc:{} not in Yc array from <yc_lm_l> relation ({})'.format(yc_val, yc))
    #
    #         if lm.min() < lm_.min():
    #             raise ValueError('lm.min({}) < lm_.min({})'.format(lm.min(), lm_.min()))
    #         if lm.max() > lm_.max():
    #             raise ValueError('lm.max({}) > lm_.max({})'.format(lm.max(), lm_.max()))
    #
    #         i_yc = Math.find_nearest_index(yc, yc_val)
    #         l = interpolate.InterpolatedUnivariateSpline(lm_, l2d[:, i_yc])(lm)
    #
    #         return l
    #
    #     def set_t_lm_grids_opal(yc_val, cls, min_max, opal_used = None):
    #
    #         t_lm_rho = Save_Load_tables.load_table('t_lm_rho', 't', 'lm', 'rho', opal_used)
    #         yc_lm_l  = Save_Load_tables.load_table('yc_lm_l', 'yc', 'lm', 'l',   opal_used)
    #         t_lm_rho_com, yc_lm_l_com = common_y(t_lm_rho, yc_lm_l)
    #
    #         if y_v_n == 'l':
    #
    #             l = lm_l(t_lm_rho_com[1:, 0], yc_lm_l, yc_val)
    #             t_l_rho = Math.combine(t_lm_rho_com[0, 1:], l, t_lm_rho_com[1:, 1:])
    #
    #             x1, x2, y1, y2 = x_y_limits(cls, min_max)
    #             t_l_rho_crop = Math.crop_2d_table2(t_l_rho, x1, x2, y1, y2) # taking absolut limits
    #
    #             return t_l_rho_crop
    #
    #         if y_v_n == 'lm':
    #             x1, x2, y1, y2 = x_y_limits(cls, min_max)
    #             t_lm_rho_crop = Math.crop_2d_table(t_lm_rho_com, x1, x2, y1, y2)
    #
    #             return t_lm_rho_crop
    #
    #
    #     def set_xgrid_ygrid(cls, min_max):
    #         x1, x2, y1, y2 = x_y_limits(cls, min_max)
    #         x_grid = np.mgrid[x1.min():x2.max():depth * 1j]
    #         y_grid = np.mgrid[y1.min():y2.max():depth * 1j]
    #         return x_grid, y_grid
    #
    #
    #
    #     yc, cls = self.separate_sp_by_crit_val('Yc', self.yc_prec)
    #
    #     t_llm_rho = set_t_lm_grids_opal(1, cls[0], 'min', self.opal_used)
    #     Plots.plot_color_table(t_llm_rho, x_v_n, y_v_n, 'rho')
    #     x_grid = t_llm_rho[0, 1:]
    #     y_grid = t_llm_rho[1:, 0]
    #
    #     def x_y_z(cls, x_grid, y_grid):
    #         '''
    #         cls = set of classes of sp. files with the same Yc.
    #         :param cls:
    #         :return:
    #         '''
    #
    #         y_zg = np.zeros(len(x_grid)+1)     # +1 for y-value (l,lm,m,Yc)
    #
    #         for cl in cls:                    # INTERPOLATING EVERY ROW to achive 'depth' number of points
    #             x =  cl.get_sonic_cols(x_v_n)
    #             y =  cl.get_crit_value(y_v_n) # Y should be unique value for a given Yc (like m, l/lm, or Yc)
    #             z =  cl.get_sonic_cols(z_v_n)
    #
    #             if append_crit:
    #                 x = np.append(x, cl.get_crit_value(x_v_n))
    #                 z = np.append(z, cl.get_crit_value(z_v_n))
    #             xi, zi = x_y_z_sort(x, z)
    #
    #             z_grid = interpolate.InterpolatedUnivariateSpline(xi, zi)(x_grid)
    #             y_zg = np.vstack((y_zg, np.insert(z_grid, 0, y, 0)))
    #
    #             plt.plot(xi, zi, '.', color='red')
    #             plt.plot(x_grid, z_grid, '-', color='red')
    #
    #         y_zg = np.delete(y_zg, 0, 0)
    #         y = y_zg[:,0]
    #         zi = y_zg[:, 1:]
    #
    #         z_grid2 = np.zeros(len(y_grid))
    #         for i in range(len(x_grid)):   # INTERPOLATING EVERY COLUMN to achive 'depth' number of points
    #             z_grid2 = np.vstack((z_grid2, interpolate.InterpolatedUnivariateSpline(y, zi[:,i])(y_grid) ))
    #         z_grid2 = np.delete(z_grid2, 0, 0)
    #
    #         x_y_z_final = Math.combine(x_grid, y_grid, z_grid2.T)
    #
    #         from Phys_Math_Labels import Plots
    #         Plots.plot_color_table(x_y_z_final, x_v_n, y_v_n, z_v_n)
    #
    #         plt.show()
    #         print('a')
    #         return x_y_z_final
    #
    #     x_y_z = x_y_z(cls[0], x_grid, y_grid)
    #
    #
    #
    #     print('a')
    #     # for i in range(len(yc)):
    #     #     x_y_z(cls[i])
    #
    #
    #
    #
    #             # x_y_z = []
    #             # for j in range(len(y)):
    #             #     x_y_z = np.append(x_y_z, [x[j], y[j], z[j]])
    #             #
    #             # x_y_z_sort = np.sort(x_y_z.view('float64, float64, float64'), order=['f0'], axis=0).view(np.float)
    #             # x_y_z_shaped = np.reshape(x_y_z_sort, (int(len(x_y_z_sort) / 3), 3))
    #             #
    #             # x_grid = np.mgrid[x_y_z_shaped[0, 0]:x_y_z_shaped[-1, 0]:depth*1j]
    #             # f = interpolate.InterpolatedUnivariateSpline(x_y_z_shaped[:,0],x_y_z_shaped[:,2]) # follows the data
    #             # z_grid = f(x_grid)
    #             #
    #             # y_xi_zi = []
    #             # for j in range(len(y)):
    #             #     y_xi_zi = np.append(y_xi_zi, [y[j], ])
    #             #
    #             #
    #             # plt.plot(x_y_z_shaped[:,0],x_y_z_shaped[:,1], '.', color='red')
    #             # # plt.plot(x_grid, z_grid, '-', color='red')
    #             # plt.show()
    #             # print('B')
    #
    #         # yc_x_y_z = np.append(yc_x_y_z, x_y_z_shaped)
    #
    #
    #     # yc_x_y_z = np.reshape(yc_x_y_z, (len(yc), int(len(x_y_z_sort) / 3), 3))
    #
    #
    #     # print('a')
    #
    #
    #
    #
    #
    #
    #
    #     # y_ = []
    #     # for i in range(len(self.sp_files)):
    #     #     y_ = np.append(y_, self.spmdl[i].get_crit_value(y_v_n))
    #     # y_grid = np.mgrid[y_.min():y_.max():depth*1j]
    #     #
    #     # z2d_pol = np.zeros(len(y_grid))
    #     # z2d_int = np.zeros(len(y_grid))
    #     #
    #     #
    #     # fig = plt.figure(figsize=plt.figaspect(1.0))
    #     #
    #     # ax1 = fig.add_subplot(221)
    #     # ax1.grid()
    #     # ax1.set_ylabel(Labels.lbls(z_v_n))
    #     # ax1.set_xlabel(Labels.lbls(y_v_n))
    #     # ax1.set_title('INTERPOLATION')
    #     #
    #     # ax2 = fig.add_subplot(222)
    #     # ax2.grid()
    #     # ax2.set_ylabel(Labels.lbls(y_v_n))
    #     # ax2.set_xlabel(Labels.lbls(z_v_n))
    #     # ax2.set_title('EXTRAPOLATION')
    #     #
    #     # for i in range(len(yc)):
    #     #     y_z = []
    #     #     for cl in cls[i]:
    #     #         y_z = np.append(y_z, [cl.get_crit_value(y_v_n), cl.get_crit_value(z_v_n)])
    #     #     y_z_sort = np.sort(y_z.view('float64, float64'), order=['f0'], axis=0).view(np.float)
    #     #     y_z_shaped = np.reshape(y_z_sort, (int(len(y_z_sort) / 2), 2))
    #     #
    #     #     '''----------------------------POLYNOMIAL EXTRAPOLATION------------------------------------'''
    #     #     print('\n\t Yc = {}'.format(yc[i]))
    #     #     y_pol, z_pol = Math.fit_plynomial(y_z_shaped[:, 0], y_z_shaped[:, 1], 3, depth, y_grid)
    #     #     z2d_pol = np.vstack((z2d_pol, z_pol))
    #     #     color = 'C' + str(int(yc[i] * 10)-1)
    #     #     ax2.plot(y_pol, z_pol, '--', color=color)
    #     #     ax2.plot(y_z_shaped[:, 0], y_z_shaped[:, 1], '.', color=color, label='yc:{}'.format("%.2f" % yc[i]))
    #     #
    #     #     '''------------------------------INTERPOLATION ONLY---------------------------------------'''
    #     #     y_int, z_int = interp(y_z_shaped[:, 0], y_z_shaped[:, 1], y_grid)
    #     #     z2d_int = np.vstack((z2d_int, z_int))
    #     #     ax1.plot(y_int, z_int, '--', color=color)
    #     #     ax1.plot(y_z_shaped[:, 0], y_z_shaped[:, 1], '.', color=color, label='yc:{}'.format("%.2f" % yc[i]))
    #     #
    #     # ax1.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
    #     # ax2.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
    #     #
    #     # z2d_int = np.delete(z2d_int, 0, 0)
    #     # z2d_pol = np.delete(z2d_pol, 0, 0)
    #     #
    #     # yc_llm_m_pol = Math.combine(yc, y_grid, z2d_pol.T)  # changing the x/y
    #     # yc_llm_m_int = Math.combine(yc, y_grid, z2d_int.T)  # changing the x/y
    #     #
    #     # table_name = '{}_{}_{}'.format('yc', y_v_n, z_v_n)
    #     # if save == 'int':
    #     #     Save_Load_tables.save_table(yc_llm_m_int, opal_used, table_name, 'yc', y_v_n, z_v_n)
    #     # if save == 'pol':
    #     #     Save_Load_tables.save_table(yc_llm_m_pol, opal_used, table_name, 'yc', y_v_n, z_v_n)
    #     #
    #     # # Save_Load_tables.save_table(yc_llm_m_pol, opal_used, table_name, 'yc', y_v_n, z_v_n)
    #     #
    #     # if plot:
    #     #
    #     #     levels = []
    #     #
    #     #     if z_v_n == 'r':
    #     #         levels = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2., 2.1, 2.2, 2.3, 2.4, 2.5]
    #     #     if z_v_n == 'm':
    #     #         levels = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    #     #     if z_v_n == 'mdot':
    #     #         levels = [-6.0, -5.9, -5.8, -5.7, -5.6, -5.5, -5.4, -5.3, -5.2, -5.1, -5., -4.9, -4.8, -4.7, -4.6, -4.5]
    #     #     if z_v_n == 'l':
    #     #         levels = [5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4]
    #     #     if z_v_n == 'lm':
    #     #         levels = [4.0, 4.05, 4.1, 4.15, 4.2, 4.25, 4.3, 4.35,  4.4, 4.45,
    #     #                   4.5, 4.55,  4.6, 4.65,  4.7, 4.75, 4.8, 4.85,  4.9, 4.95, 5.0]
    #     #     if z_v_n == 't':
    #     #         levels = [5.15, 5.16,5.17,5.18,5.19,5.20,5.21,5.22,5.23,5.24,5.25,5.26,5.27,5.28,5.29,5.30]
    #     #
    #     #
    #     #     ax = fig.add_subplot(223)
    #     #
    #     #     # ax = fig.add_subplot(1, 1, 1)
    #     #     ax.set_xlim(yc_llm_m_int[0,1:].min(), yc_llm_m_int[0,1:].max())
    #     #     ax.set_ylim(yc_llm_m_int[1:,0].min(), yc_llm_m_int[1:,0].max())
    #     #     ax.set_ylabel(Labels.lbls(y_v_n))
    #     #     ax.set_xlabel(Labels.lbls('Yc'))
    #     #
    #     #     contour_filled = plt.contourf(yc_llm_m_int[0, 1:], yc_llm_m_int[1:, 0], yc_llm_m_int[1:,1:], levels, cmap=plt.get_cmap('RdYlBu_r'))
    #     #     # plt.colorbar(contour_filled, label=Labels.lbls('m'))
    #     #     contour = plt.contour(yc_llm_m_int[0, 1:], yc_llm_m_int[1:, 0], yc_llm_m_int[1:,1:], levels, colors='k')
    #     #
    #     #     clb = plt.colorbar(contour_filled)
    #     #     clb.ax.set_title(Labels.lbls(z_v_n))
    #     #
    #     #     plt.clabel(contour, colors='k', fmt='%2.2f', fontsize=12)
    #     #     #ax.set_title('MASS-LUMINOSITY RELATION')
    #     #
    #     #     # plt.ylabel(l_or_lm)
    #     #     # ax.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
    #     #     # plt.savefig(name)
    #     #
    #     #
    #     #
    #     #
    #     #     ax = fig.add_subplot(224)
    #     #
    #     #     # ax = fig.add_subplot(1, 1, 1)
    #     #     ax.set_xlim(yc_llm_m_pol[0, 1:].min(), yc_llm_m_pol[0, 1:].max())
    #     #     ax.set_ylim(yc_llm_m_pol[1:, 0].min(), yc_llm_m_pol[1:, 0].max())
    #     #     ax.set_ylabel(Labels.lbls(y_v_n))
    #     #     ax.set_xlabel(Labels.lbls('Yc'))
    #     #
    #     #
    #     #     contour_filled = plt.contourf(yc_llm_m_pol[0, 1:], yc_llm_m_pol[1:, 0], yc_llm_m_pol[1:, 1:], levels,
    #     #                                   cmap=plt.get_cmap('RdYlBu_r'))
    #     #     # plt.colorbar(contour_filled, label=Labels.lbls('m'))
    #     #     contour = plt.contour(yc_llm_m_pol[0, 1:], yc_llm_m_pol[1:, 0], yc_llm_m_pol[1:, 1:], levels, colors='k')
    #     #
    #     #     clb = plt.colorbar(contour_filled)
    #     #     clb.ax.set_title(Labels.lbls(z_v_n))
    #     #
    #     #     plt.clabel(contour, colors='k', fmt='%2.2f', fontsize=12)
    #     #     #ax.set_title('MASS-LUMINOSITY RELATION')
    #     #
    #     #     # plt.ylabel(l_or_lm)
    #     #     # ax.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
    #     #
    #     #
    #     #     plt.show()
    #
    #     # yc_llm_m_pol



    # def save_ly_m_or_r_relation_old(self, obs_filename, y_coord, plot = False, yc_prec = 0.3, depth = 100):
    #
    #
    #     # --- --- --- READING SP FILES --- --- --- ---
    #     l_m_yc = []
    #     for i in range(len(self.sp_files)):
    #         l = self.spmdl[i].get_crit_value('l')
    #         m = self.spmdl[i].get_crit_value(y_coord)
    #         yc= self.spmdl[i].get_crit_value('Yc')
    #
    #         l_m_yc = np.append(l_m_yc, [l, m, yc])
    #
    #     l_m_yc_sort = np.sort(l_m_yc.view('float64, float64, float64'), order=['f2'], axis=0).view(np.float)
    #     l_m_yc_shaped = np.reshape(l_m_yc_sort, (len(self.sp_files), 3))
    #
    #     l =  l_m_yc_shaped[:,0]
    #     m =  l_m_yc_shaped[:,1]
    #     yc = l_m_yc_shaped[:,2]
    #
    #     l_min = l.min()
    #     l_max = l.max()
    #     l_grid = np.mgrid[l_min:l_max:depth*1j]
    #
    #
    #     yc = np.append(yc, -1)
    #     y_steps = np.array([yc[0]])
    #     m_l_row = []
    #
    #
    #     m2d_pol = np.zeros(len(l_grid))
    #     m2d_int = np.zeros(len(l_grid))
    #
    #     # --- FOR PLOTTING ----
    #     fig = plt.figure(figsize=plt.figaspect(1.0))
    #
    #     ax1 = fig.add_subplot(221)
    #     ax1.grid()
    #     ax1.set_ylabel(Labels.lbls(y_coord))
    #     ax1.set_xlabel(Labels.lbls('l'))
    #     ax1.set_title('INTERPOLATION')
    #
    #
    #     ax2 = fig.add_subplot(222)
    #     ax2.grid()
    #     ax2.set_ylabel(Labels.lbls(y_coord))
    #     ax2.set_xlabel(Labels.lbls('l'))
    #     ax2.set_title('EXTRAPOLATION')
    #
    #     yc_prec = np.float(yc_prec)
    #     for i in range(len(yc)):
    #         if "%{}f".format(yc_prec) % yc[i] == "%{}f".format(yc_prec) % y_steps[-1]:
    #             m_l_row = np.append(m_l_row, [m[i], l[i]])
    #         else:
    #             m_l_row_sort = np.sort(m_l_row.view('float64, float64'), order=['f1'], axis=0).view(np.float)
    #             m_l_row_shaped = np.reshape(m_l_row_sort, (int(len(m_l_row) / 2), 2))
    #
    #
    #             m_sort = m_l_row_shaped[:, 0]
    #             l_sort = m_l_row_shaped[:, 1]
    #
    #             '''----------------------------POLYNOMIAL EXTRAPOLATION------------------------------------'''
    #
    #             fit = np.polyfit(l_sort, m_sort, 3)  # fit = set of coeddicients (highest first)
    #             f = np.poly1d(fit)
    #             m2d_pol =  np.vstack(( m2d_pol, f(l_grid) ))
    #
    #             color = 'C' + str(int((y_steps[-1] * 10)))
    #             ax2.plot(l_grid, f(l_grid), '--', color=color)
    #             ax2.plot(l_sort, m_sort, '.', color=color, label='yc:{}'.format("%.2f" % y_steps[-1]))
    #
    #
    #             '''------------------------------INTERPOLATION ONLY---------------------------------------'''
    #
    #             f = interpolate.interp1d(l_sort, m_sort, kind='cubic', bounds_error=False) # False means that 'nan' result is allowed
    #             m2d_int = np.vstack(( m2d_int, f(l_grid)))
    #
    #             color = 'C' + str(int((y_steps[-1] * 10)))
    #             ax1.plot(l_grid, f(l_grid), '--', color=color)
    #             ax1.plot(l_sort, m_sort, '.', color=color, label='yc:{}'.format("%.2f" % y_steps[-1]))
    #
    #             yc_value = np.float("%{}f".format(yc_prec) % yc[i])
    #             print('min_m:{} max_m:{}'.format(m_sort.min(), m_sort.max()))
    #             print('min_l:{} max_l:{}'.format(l_sort.min(), l_sort.max()))
    #             print('Yc step: {} size: {}'.format("%.2f" % y_steps[-1], len(m_l_row_shaped)))
    #             print('\t')
    #             y_steps = np.append(y_steps, np.float("%{}f".format(yc_prec) % yc[i]))
    #             m_l_row = []
    #
    #     ax1.legend()
    #     ax2.legend()
    #
    #     y_steps = np.delete(y_steps, -1, 0)
    #     m2d_int     = np.delete(m2d_int, 0, 0)
    #     m2d_pol = np.delete(m2d_pol, 0, 0)
    #
    #     l_yc_m_pol = Math.combine(l_grid, y_steps, m2d_pol).T  # changing the x/y
    #     l_yc_m_int = Math.combine(l_grid, y_steps, m2d_int).T # changing the x/y
    #
    #
    #     # --- Saving only polynom. table. ---
    #
    #     table_name = 'l_yc_'+y_coord
    #     Save_Load_tables.save_table(l_yc_m_pol, obs_filename, table_name, 'l', 'yc', y_coord)
    #
    #
    #     if plot:
    #
    #         levels = []
    #
    #         if y_coord == 'r':
    #             levels = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2., 2.1, 2.2, 2.3, 2.4, 2.5]
    #         if y_coord == 'm':
    #             levels = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    #         if y_coord == 'mdot':
    #             levels = [-6.0, -5.9, -5.8, -5.7, -5.6, -5.5, -5.4, -5.3, -5.2, -5.1, -5., -4.9, -4.8, -4.7, -4.6, -4.5]
    #
    #
    #         ax = fig.add_subplot(223)
    #
    #         # ax = fig.add_subplot(1, 1, 1)
    #         ax.set_xlim(l_yc_m_int[0,1:].min(), l_yc_m_int[0,1:].max())
    #         ax.set_ylim(l_yc_m_int[1:,0].min(), l_yc_m_int[1:,0].max())
    #         ax.set_ylabel(Labels.lbls('l'))
    #         ax.set_xlabel(Labels.lbls('Yc'))
    #
    #         contour_filled = plt.contourf(l_yc_m_int[0, 1:], l_yc_m_int[1:, 0], l_yc_m_int[1:,1:], levels, cmap=plt.get_cmap('RdYlBu_r'))
    #         # plt.colorbar(contour_filled, label=Labels.lbls('m'))
    #         contour = plt.contour(l_yc_m_int[0, 1:], l_yc_m_int[1:, 0], l_yc_m_int[1:,1:], levels, colors='k')
    #
    #         clb = plt.colorbar(contour_filled)
    #         clb.ax.set_title(Labels.lbls(y_coord))
    #
    #         plt.clabel(contour, colors='k', fmt='%2.2f', fontsize=12)
    #         #ax.set_title('MASS-LUMINOSITY RELATION')
    #
    #         # plt.ylabel(l_or_lm)
    #         ax.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
    #         # plt.savefig(name)
    #
    #
    #
    #
    #         ax = fig.add_subplot(224)
    #
    #         # ax = fig.add_subplot(1, 1, 1)
    #         ax.set_xlim(l_yc_m_pol[0, 1:].min(), l_yc_m_pol[0, 1:].max())
    #         ax.set_ylim(l_yc_m_pol[1:, 0].min(), l_yc_m_pol[1:, 0].max())
    #         ax.set_ylabel(Labels.lbls('l'))
    #         ax.set_xlabel(Labels.lbls('Yc'))
    #
    #
    #         contour_filled = plt.contourf(l_yc_m_pol[0, 1:], l_yc_m_pol[1:, 0], l_yc_m_pol[1:, 1:], levels,
    #                                       cmap=plt.get_cmap('RdYlBu_r'))
    #         # plt.colorbar(contour_filled, label=Labels.lbls('m'))
    #         contour = plt.contour(l_yc_m_pol[0, 1:], l_yc_m_pol[1:, 0], l_yc_m_pol[1:, 1:], levels, colors='k')
    #
    #         clb = plt.colorbar(contour_filled)
    #         clb.ax.set_title(Labels.lbls(y_coord))
    #
    #         plt.clabel(contour, colors='k', fmt='%2.2f', fontsize=12)
    #         #ax.set_title('MASS-LUMINOSITY RELATION')
    #
    #         # plt.ylabel(l_or_lm)
    #         ax.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
    #
    #
    #         plt.show()

# class Read_Observables_old:
#
#     def __init__(self, observ_name, gal_or_lmc, clump_used=4, clump_req=4):
#         '''
#
#         :param observ_name:
#         :param yc: -1 (None, use m), 0.1 to 0.99 (My relation for diff yc), 10 (use Langer et al. 1989)
#         :param clump_used:
#         :param clump_req:
#         '''
#
#         self.set_use_gaia = True
#         self.set_use_atm_file = True
#         self.set_atm_file = None
#
#         self.set_load_yc_l_lm = True
#         self.set_load_yc_nan_lmlim = True
#         # --------------------------------------------------------------------------------------------------------------
#         self.file_name = observ_name
#
#         self.clump = clump_used
#         self.new_clump = clump_req
#
#         self.metal = gal_or_lmc
#
#
#
#         self.table = []
#         with open(observ_name, 'r') as f:
#             for line in f:
#                 if '#' not in line.split() and line.strip(): # if line is not empty and does not contain '#'
#                     self.table.append(line)
#
#         self.names = self.table[0].split()
#         self.num_stars = len(self.table)-1 # as first row is of var names
#
#
#         if len(self.names) != len(self.table[1].split()):
#             print('\t___Error. Number of vars in list({}) != number of cols in observ.data file({}) '
#                      '|Read_Observables, __init__|'.format(len(self.names), len(self.table[1].split())))
#         print('\t__Note: Data include following paramters:\n\t | {} |'.format(self.table[0].split()))
#
#         self.table.remove(self.table[0])  # removing the var_names line from the array. (only actual values left)
#
#
#         # -----------------DISTRIBUTING READ COLUMNS-------------
#         self.num_v_n = ['N', 't_*', 'lRt', 'm', 'l', 'mdot', 'v_inf', 'eta']  # list of used v_n's; can be extended upon need ! 'N' should be first
#         self.cls = 'class'                           # special variable
#
#         self.str_v_n = []
#
#         self.set_num_str_tables()
#
#         # --- --- GAIA LUMINOCITIES --- ---
#         if self.set_use_gaia and gal_or_lmc == 'gal':
#             self.gaia_names, self.gaia_table = self.read_genergic_table(observ_name.split('.data')[0] + '_gaia_lum.data')
#
#         # --- --- Potsdam Atmospheres --- ---
#         if self.set_use_atm_file:
#             self.atm = Read_Atmosphere_File(self.set_atm_file, self.metal)
#             self.atm_table = self.atm.get_table('t_*', 'rt', 't_eff')
#
#         if self.set_load_yc_l_lm:
#             self.yc_l_lm = Save_Load_tables.load_table('yc_l_lm', 'yc', 'l', 'lm', self.metal, '')
#
#         if self.set_load_yc_nan_lmlim:
#             self.yc_nan_lmlim = Save_Load_tables.load_table('yc_nan_lmlim', 'yc', 'nan', 'lmlim', self.metal, '')
#
#     # ------------------------------------------------
#     @staticmethod
#     def read_genergic_table(file_name):
#         '''
#         Reads the the file table, returning the list with names and the table
#         structure: First Row must be with '#' in the beginning and then, the var names.
#         other Rows - table with the same number of elements as the row of var names
#         :return:
#         '''
#         table = []
#         with open(file_name, 'r') as f:
#             for line in f:
#                 # if '#' not in line.split() and line.strip():  # if line is not empty and does not contain '#'
#                 table.append(line)
#
#         names = table[0].split()[1:]  # getting rid of '#' which is the first element in a row
#         num_colls = len(table) - 1  # as first row is of var names
#
#         if len(names) != len(table[1].split()):
#             print('\t___Error. Number of vars in list({}) != number of cols in observ.data file({}) '
#                   '|Read_Observables, __init__|'.format(len(names), len(table[1].split())))
#         print('\t__Note: Data include following paramters:\n\t | {} |'.format(names))
#
#         table.remove(table[0])  # removing the var_names line from the array. (only actual values left)
#
#         tmp = np.zeros(len(names))
#         for row in table:
#             tmp_ = np.array(row.split(), dtype=np.float)
#             if len(tmp_)!=len(names):
#                 print('row:', row, '\n', 'tmp:',tmp_,len(tmp_))
#                 raise NameError(len(tmp_),'\n',len(names))
#             tmp = np.vstack((tmp, tmp_))
#         table = np.delete(tmp, 0, 0)
#
#         return names, table
#     # ------------------------------------------------
#
#     def set_num_str_tables(self):
#
#         self.numers = np.zeros((self.num_stars, len(self.num_v_n)))
#         self.clses= []
#
#         #----------------------------------------SET THE LIST OF CLASSES------------------
#         for i in range(len(self.table)):
#             n = self.names.index(self.cls)
#             if self.cls in self.names:
#                 self.clses.append(self.table[i].split()[n])
#
#         #-----------------------------------------SET 2D ARRAY OF NUM PARAMETERS---------
#         for i in range(len(self.table)):
#             for j in range(len(self.num_v_n)):
#                 n = self.names.index(self.num_v_n[j])
#                 if self.num_v_n[j] in self.names:
#                     self.numers[i, j] =  np.float(self.table[i].split()[n])
#
#         #-----------------------------------------SETTING THE CATALOGUE NAMES------------
#         self.stars_n = self.numers[:,0]
#
#         if len(self.numers[:,0])!= len(self.clses):
#             raise ValueError('Size of str. vars ({}) != size of num.var ({})'.format( len(self.numers[:,0]) ,len(self.clses) ))
#
#         # print(self.numers, self.clses)
#         print('\n\t__Note. In file: {} total {} stars are loaded. \n\t  Available numerical parameters: {} '
#               .format(self.file_name, len(self.numers), self.num_v_n))
#
#     def modify_value(self, v_n, value):
#         if v_n == 't' or v_n == 't_*':
#             return np.log10(value*1000)
#
#         if v_n == 'mdot':
#             new_mdot = value
#             if self.clump != self.new_clump:
#
#                 f_WR = 10**(0.5 * np.log10(self.clump / self.new_clump))  # modify for a new clumping
#                 new_mdot = value +  np.log10(f_WR)
#                 print('\nClumping factor changed from {} to {}'.format(self.clump, self.new_clump))
#                 print('new_mdot = old_mdot + ({}) (f_WR: {} )'
#                       .format( "%.2f" % np.log10(f_WR), "%.2f" % f_WR ))
#                 # print('| i | mdot | n_mdot | f_WR |')
#                 # for i in range(len(res)):
#                 #     f_wr = 10**np.float(res2[i]) / 10**np.float(res[i])
#                 #     print('| {} | {} | {} | {} |'.format("%2.f" % i, "%.2f" % np.float(res[i]) ,"%.2f" % np.float(res2[i]), "%.2f" % f_wr ) )
#             return new_mdot
#
#         return value
#
#     def get_num_par_from_table(self, v_n, star_n, use_gaia=False):
#
#         if v_n not in self.num_v_n:
#             raise NameError('v_n: {} is not in set of num. pars: {}'.format(v_n, self.num_v_n))
#         if star_n not in self.numers[: , 0]:
#             raise NameError('star_n: {} is not in the list of star numbers: {}'.format(star_n, self.numers[:,0]))
#
#         # -- GAIA MISSION for GALAXY (table8) ---
#         if v_n == 'l' and self.metal.split('/')[-1] == 'gal' and use_gaia:
#             l = self.get_gaia_lum(star_n)[0]
#             return l
#
#         ind_star = np.where(self.numers[: , 0]==star_n)[0][0]
#         ind_v_n  = self.num_v_n.index(v_n)
#         value = self.numers[ind_star, ind_v_n]
#
#
#         value = self.modify_value(v_n, value)
#
#         return value
#
#     # ---
#     def get_gaia_lum(self, star_n):
#         if star_n not in self.gaia_table[:,0]:
#             raise ValueError('Star: {} is not in the GAIA list: {}'.format(star_n, self.gaia_table[:,0]))
#         ind = Math.find_nearest_index(self.gaia_table[:,0], star_n)
#         # print(ind)
#         return np.array([self.gaia_table[ind,2], self.gaia_table[ind,3], self.gaia_table[ind,4]])
#     #---------------------------------------------PUBLIC FUNCTIONS---------------------------------------
#     def l_lm_for_errs(self, l, star_n, yc_val):
#         '''
#         Changes the L of the star to the maximmum available for plotting.
#         :param v_n:
#         :param star_n:
#         :param yc_val:
#         :return:
#         '''
#
#         if l < self.yc_l_lm[1:, 0].min(): l = self.yc_l_lm[1:, 0].min()
#         if l > self.yc_l_lm[1:, 0].max(): l = self.yc_l_lm[1:, 0].max()
#         # print('\__ Yc: {}'.format(yc_val))
#
#         if yc_val >= 0 and yc_val <= 1.:
#             l, lm = Math.get_z_for_yc_and_y(yc_val, self.yc_l_lm, l, 0)
#             yc_lim_arr=self.yc_nan_lmlim[0,1:]
#             lm_low_lim_arr=self.yc_nan_lmlim[1,1:]
#             lm_up_lim_arr=self.yc_nan_lmlim[2,1:]
#
#             yc_ind = Math.find_nearest_index(yc_lim_arr, yc_val)
#             # lm_lim = self.yc_nan_lmlim[1:, yc_ind]
#             if yc_val != 1.0 and (lm < lm_low_lim_arr[yc_ind] or lm > lm_up_lim_arr[yc_ind]):
#                 raise ValueError('For Yc:{}, star:{} L/M limits are: {}, {}, given lm {}'
#                                  .format(yc_val, star_n, "%.2f" % lm_low_lim_arr[yc_ind], "%.2f" % lm_up_lim_arr[yc_ind], "%.2f" % lm))
#
#             # l, lm = SP_file_work.yc_x__to__y__sp(yc_val, 'l', 'lm', l, self.opal_used, 0)
#             return lm
#
#         if yc_val == -1 or yc_val == 'langer':
#             return np.log10(10 ** l / 10 ** Physics.l_to_m_langer(l))
#
#         if yc_val == None or yc_val == 'hamman':
#             m = self.get_num_par_from_table('m', star_n)
#             return np.log10(10 ** l / m)
#
#     def get_star_l_obs_err(self,star_n, yc_assumed, use_gaia=False):
#
#         l_err1 = l_err2 = None
#         l = self.get_llm('l', star_n, yc_assumed, use_gaia)
#
#         if self.metal.split('/')[-1] == 'gal':
#
#             l_err1 = 0.2 # --- OLD correction from Hamman2006
#             l_err2 = 0.2
#             if use_gaia:
#                 l_err1 = self.get_gaia_lum(star_n)[0]-self.get_gaia_lum(star_n)[1]
#                 l_err2 = self.get_gaia_lum(star_n)[2]-self.get_gaia_lum(star_n)[0]
#
#                 print('star: {}, l:{} , l_lower: {}, l_upper: {}'.format(star_n, l, l_err1, l_err2))
#
#         if self.metal.split('/')[-1] == 'lmc':
#
#             l_err1 = 0.1
#             l_err2 = 0.1
#
#         if l_err1 == None or l_err2 == None: raise ValueError('Error for l (star:{}) not found'.format(star_n))
#
#         return l-l_err1, l+l_err2
#
#     def get_star_lm_obs_err(self,star_n, yc_assumed, use_gaia=False):
#         # l_err1 = l_err2 = None
#         # l = None
#
#         l_err1, l_err2 = self.get_star_l_obs_err(star_n, yc_assumed, use_gaia)
#         lm1 = self.l_lm_for_errs(l_err1, star_n, yc_assumed)
#         lm2 = self.l_lm_for_errs(l_err2, star_n, yc_assumed)
#
#         return lm1, lm2
#
#         # if self.opal_used.split('/')[-1] == 'table8.data':
#         #
#         #     l_err1, l_err2 = self.get_star_l_obs_err(star_n, yc_assumed, use_gaia)
#         #
#         #     # l_err1 = 0.2 # --- OLD correction from Hamman2006
#         #     # l_err2 = 0.2
#         #     # l_err1 = self.get_gaia_lum(star_n)[0]-self.get_gaia_lum(star_n)[1]
#         #     # l_err2 = self.get_gaia_lum(star_n)[2]-self.get_gaia_lum(star_n)[0]
#         # if self.opal_used.split('/')[-1] == 'table_x.data':
#         #
#         #
#         #
#         #     l_err1 = 0.1
#         #     l_err2 = 0.1
#         #
#         #
#         # l = self.get_num_par('l', star_n, yc_assumed)
#         # lm1 = self.l_lm_for_errs(l - l_err1, star_n, yc_assumed)
#         # lm2 = self.l_lm_for_errs(l + l_err2, star_n, yc_assumed)
#         #
#         # return lm1, lm2
#
#
#     def get_star_lm_obs_err_old(self,star_n, yc_assumed, use_gaia=False):
#         l_err1 = l_err2 = None
#         l = None
#         if self.metal.split('/')[-1] == 'gal':
#
#             l_err1 = 0.2 # --- OLD correction from Hamman2006
#             l_err2 = 0.2
#             # l_err1 = self.get_gaia_lum(star_n)[0]-self.get_gaia_lum(star_n)[1]
#             # l_err2 = self.get_gaia_lum(star_n)[2]-self.get_gaia_lum(star_n)[0]
#         if self.metal.split('/')[-1] == 'lmc':
#
#             l_err1 = 0.1
#             l_err2 = 0.1
#
#
#         l = self.get_num_par('l', star_n, yc_assumed)
#         lm1 = self.l_lm_for_errs(l - l_err1, star_n, yc_assumed)
#         lm2 = self.l_lm_for_errs(l + l_err2, star_n, yc_assumed)
#
#         return lm1, lm2
#
#     def lm_mdot_to_ts_errs(self, lm, mdot, star_n, t_llm_mdot, yc_assumed, lim_t1, lim_t2):
#         mdot_arr = t_llm_mdot[1:, 1:]
#         i = Math.find_nearest_index(t_llm_mdot[1:, 0], lm)
#
#         if t_llm_mdot[i, 1:].min() > mdot:
#             print('\t__Star: {} Using the min. mdot in the row for the error. '.format(star_n))
#             mdot = mdot_arr[i,:].min()
#
#         xyz = Physics.model_yz_to_xyz(t_llm_mdot[0, 1:], t_llm_mdot[1:, 0], t_llm_mdot[1:, 1:], lm, mdot, star_n, lim_t1, lim_t2)
#
#         if xyz.any():
#             if len(xyz[0, :]) > 1:
#                 raise ValueError('Multiple coordinates for star: {} | Yc: {}'.format(star_n, yc_assumed))
#             else:
#                 ts = xyz[0, 0] # xyz[0, :] ALL X coordinates
#                 return ts
#
#         else:
#
#             print('\t__ Error. No (Error) solutions found star: {}, mdot: {}, mdot array:({}, {})'
#                              .format(star_n, mdot, mdot_arr[i, :].min(), mdot_arr[i, :].max()))
#             return lim_t2
#             raise ValueError('No Error solutions found star: {}, mdot: {}, mdot array:({}, {})'
#                              .format(star_n, mdot, mdot_arr[i, :].min(), mdot_arr[i, :].max()))
#     def get_star_mdot_obs_err(self, star_n, yc_assumed):
#         mdot_err = None
#         if self.metal.split('/')[-1] == 'gal':
#             mdot_err = 0.15
#         if self.metal.split('/')[-1] == 'lmc':
#             mdot_err = 0.15
#         if mdot_err == None: raise NameError('Opal_table is not recognised. '
#                                              'Expected: *gal* or *lmc*, Given: {}'.format(self.metal))
#         mdot = self.get_num_par('mdot', star_n, yc_assumed)
#         return mdot - mdot_err, mdot + mdot_err
#
#     def get_star_ts_obs_err(self, star_n, t_llm_mdot, yc_assumed, lim_t1, lim_t2):
#
#         mdot_err = None
#         if self.metal.split('/')[-1] == 'gal':
#             mdot_err = 0.15
#         if self.metal.split('/')[-1] == 'lmc':
#             mdot_err = 0.15
#         if mdot_err == None: raise NameError('Opal_table is not recognised. '
#                                              'Expected: *gal* or *lmc*, Given: {}'.format(self.metal))
#
#         lm1, lm2 = self.get_star_lm_obs_err(star_n, yc_assumed)
#         # lm = self.get_num_par('lm', star_n, yc_assumed)
#         # mdot = self.get_num_par('mdot', star_n, yc_assumed)
#
#         mdot1, mdot2 = self.get_star_mdot_obs_err(star_n, yc_assumed)
#
#         ts1_b = self.lm_mdot_to_ts_errs(lm1, mdot1, star_n, t_llm_mdot, yc_assumed, lim_t1, lim_t2)
#         ts2_b = self.lm_mdot_to_ts_errs(lm1, mdot2, star_n, t_llm_mdot, yc_assumed, lim_t1, lim_t2)
#         ts1_t = self.lm_mdot_to_ts_errs(lm2, mdot1, star_n, t_llm_mdot, yc_assumed, lim_t1, lim_t2)
#         ts2_t = self.lm_mdot_to_ts_errs(lm2, mdot2, star_n, t_llm_mdot, yc_assumed, lim_t1, lim_t2)
#
#         return np.float(ts1_b), np.float(ts2_b), np.float(ts1_t), np.float(ts2_t)
#     # --------------------------------------------
#
#     def get_llm(self, l_or_lm, star_n, yc_val=None, use_gaia = False, check_wne=False):
#
#         l = self.get_num_par_from_table('l', star_n, use_gaia)
#         if l_or_lm == 'l':
#             return self.get_num_par_from_table('l', star_n, use_gaia)
#
#         if yc_val >= 0 and yc_val <= 1.:
#             l, lm = Math.get_z_for_yc_and_y(yc_val, self.yc_l_lm, l, 0)
#
#             if check_wne:
#                 yc_ind = Math.find_nearest_index(self.yc_nan_lmlim[0, 1:], yc_val) + 1
#                 lm_lim = self.yc_nan_lmlim[1:, yc_ind]
#                 if lm < lm_lim[0] or lm > lm_lim[1]:
#                     raise ValueError('For Yc:{}, star:{} L/M limits are: {}, {}, given lm {}'
#                                      .format(yc_val, star_n, "%.2f" % lm_lim[0], "%.2f" % lm_lim[1], "%.2f" % lm))
#
#                 # l, lm = SP_file_work.yc_x__to__y__sp(yc_val, 'l', 'lm', l, self.opal_used, 0)
#                 return lm
#             else:
#                 return lm
#
#         if yc_val == -1 or yc_val == 'langer':
#             return np.log10(10 ** l / 10 ** Physics.l_to_m_langer(l))
#
#         if yc_val == None or yc_val == 'hamman':
#             m = self.get_num_par_from_table('m', star_n)
#             np.log10(10 ** l / m)
#
#         raise ValueError('Yc = {} is not recognised. '
#                          'Use: 0-1 for loading tables | -1 or langer for langer |  None or hamman for hamman')
#
#     def get_num_par(self, v_n, star_n, yc_val = None, use_gaia = False):
#         '''
#
#         :param v_n:
#         :param star_n:
#         :param yc_val: '-1' for Langer1989 | 'None' for use 'm' | 0.9-to-0 for use of l-m-yc relation
#         :return:
#         '''
#
#         if v_n == 't_eff':
#             return self.get_t_eff_from_atmosphere(star_n)
#
#         if v_n == 'lm' or v_n=='l':
#             raise NameError('Do not use this for lm')
#             return self.get_llm(star_n, yc_val, use_gaia, False)
#
#             # # print('\__ Yc: {}'.format(yc_val))
#             # l = self.get_num_par_from_table('l', star_n, use_gaia)
#             # if yc_val >= 0 and yc_val <= 1.:
#             #     l, lm = Math.get_z_for_yc_and_y(yc_val, self.yc_l_lm, l, 0)
#             #     yc_ind = Math.find_nearest_index(self.yc_nan_lmlim[0,1:], yc_val)+1
#             #     lm_lim = self.yc_nan_lmlim[1:, yc_ind]
#             #     if lm < lm_lim[0] or lm > lm_lim[1]:
#             #         raise ValueError('For Yc:{}, star:{} L/M limits are: {}, {}, given lm {}'
#             #                          .format(yc_val, star_n , "%.2f" % lm_lim[0], "%.2f" % lm_lim[1], "%.2f" % lm))
#             #
#             #     # l, lm = SP_file_work.yc_x__to__y__sp(yc_val, 'l', 'lm', l, self.opal_used, 0)
#             #     return lm
#
#             # if yc_val == -1 or yc_val == 'langer':
#             #     return np.log10(10 ** l / 10 ** Physics.l_to_m_langer(l))
#
#             # if yc_val == None or yc_val == 'hamman':
#             #     m = self.get_num_par_from_table('m', star_n)
#             #     np.log10(10 ** l / m)
#
#             # raise ValueError('Yc = {} is not recognised. '
#             #                  'Use: 0-1 for loading tables | -1 or langer for langer |  None or hamman for hamman')
#
#         return self.get_num_par_from_table(v_n, star_n, use_gaia)
#
#     def get_min_max(self, v_n, yc = None):
#         arr = []
#         for star_n in self.stars_n:
#             arr = np.append(arr, self.get_num_par(v_n, star_n, yc))
#         return arr.min(), arr.max()
#
#     def get_min_max_llm(self, v_n, yc = None, use_gaia=False, check_wne=False):
#         arr = []
#         for star_n in self.stars_n:
#             arr = np.append(arr, self.get_llm(v_n, star_n, yc, use_gaia, check_wne))
#         return arr.min(), arr.max()
#
#     def get_xyz_from_yz(self, yc_val, model_n, y_name, z_name, x_1d_arr, y_1d_arr, z_2d_arr, lx1 = None, lx2 = None):
#
#         if y_name == z_name:
#             raise NameError('y_name and z_name are the same : {}'.format(z_name))
#
#         star_y = self.get_num_par(y_name, model_n, yc_val)
#         star_z = self.get_num_par(z_name, model_n, yc_val)
#
#         if star_z == None or star_y == None:
#             raise ValueError('star_y:{} or star_z:{} not defined'.format(star_y,star_z))
#
#         xyz = Physics.model_yz_to_xyz(x_1d_arr, y_1d_arr, z_2d_arr,  star_y, star_z, model_n, lx1, lx2)
#
#         return xyz
#
#     def get_star_class(self, n):
#         for i in range(len(self.numers[:, 0])):
#             if n == self.numers[i, 0]:
#                 return self.clses[i]
#
#     def get_class_color(self, n):
#         # cls = self.get_star_class(n)
#         #
#         # if cls == 'WN2' or cls == 'WN3':
#         #     return 'v'
#         # if cls == 'WN4-s':
#         #     return 'o' # circle
#         # if cls == 'WN4-w':
#         #     return 's' # square
#         # if cls == 'WN5-w' or cls == 'WN6-w':
#         #     return '1' # tri down
#         # if cls == 'WN5-s' or cls == 'WN6-s':
#         #     return 'd' #diamond
#         # if cls == 'WN7':
#         #     return '^'
#         # if cls == 'WN8' or cls == 'WN9':
#         #     return 'P' #plus filled
#
#
#
#         # import re  # for searching the number in 'WN7-e' string, to plot them different colour
#         se = re.search(r"\d+(\.\d+)?", self.get_star_class(n))  # this is searching for the niumber
#         # cur_type = int(se.group(0))
#         color = 'C'+se.group(0)
#         return color
#
#     def get_clss_marker(self, n):
#         # se = re.search(r"\d+(\.\d+)?", self.get_star_class(n))
#         # n_class =  int(se.group(0))
#         # se = re.search(r"\d+(\.\d+)?", self.get_star_class(n))  # this is searching for the niumber
#         # cur_type = int(se.group(0))
#         cls = self.get_star_class(n)
#
#
#         # --- FOR galactic stars ---
#         if cls == 'WN2-w' or cls == 'WN3-w':
#             return 'v'
#         if cls == 'WN4-s':
#             return 'o' # circle
#         if cls == 'WN4-w':
#             return 's' # square
#         if cls == 'WN5-w' or cls == 'WN6-w':
#             return '1' # tri down
#         if cls == 'WN5-s' or cls == 'WN6-s':
#             return 'd' #diamond
#         if cls == 'WN7':
#             return '^'
#         if cls == 'WN8' or cls == 'WN9':
#             return 'P' #plus filled
#
#         # --- FOR LMC ---
#
#         if cls == 'WN3b' or cls == 'WN3o':
#             return 'v'
#         if cls == 'WN2b(h)' or cls == 'WN2b':
#             return 'o'  # circle
#         if cls == 'WN4b/WCE' or cls == 'WN4b(h)':
#             return 's'  # square
#         if cls == 'WN2':
#             return '.'
#         if cls == 'WN3' or cls == 'WN3(h)':
#             return '*'
#         if cls == 'WN4b' or cls == 'WN4o':
#             return '^'
#         if cls == 'WN4':
#             return 'd'  # diamond
#
#         raise NameError('Class {} is not defined'.format(cls))
#
#     # def get_star_lm_err(self, star_n, yc_assumed, yc1, yc2):
#     #
#     #     def check_if_ys(lm, yc_arr = np.array([1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])):
#     #         # ys_zams = yc_lm_ys[1:, 1:].max()
#     #
#     #         tmp, ys_zams = SP_file_work.yc_x__to__y__sp(1., 'lm', 'ys', lm, self.opal_used)
#     #
#     #         yc_avl = []
#     #         for i in range(len(yc_arr)):
#     #             tmp, ys = SP_file_work.yc_x__to__y__sp(yc_arr[i],'lm','ys',lm, self.opal_used)
#     #             if ys <=ys_zams:
#     #                 yc_avl = np.append(yc_avl, yc_arr[i])
#     #             else:
#     #                 pass
#     #
#     #         if len(yc_avl) == 0:
#     #             raise ValueError('Array yc_avl empty, - not yc are found for which ys = ys_zams ({})'.format(ys_zams))
#     #
#     #
#     #
#     #         return yc_avl.max(), yc_avl.min()
#     #
#     #     lm  = 4.29471667968977 # np.float(self.get_num_par('lm', star_n, yc_assumed))
#     #
#     #     yc1, yc2 = check_if_ys(lm)
#     #     lm1 = self.get_num_par('lm', star_n, yc1)  # Getting lm at different Yc (assiming the same l, but
#     #     lm2 = self.get_num_par('lm', star_n, yc2)
#     #
#     #     print('\t__STAR: {} | {} : {} (+{} -{})'.format(star_n, 'L/M', lm, "%.2f" % np.abs(lm - lm1),
#     #                                                     "%.2f" % np.abs(lm - lm2)))
#     #
#     #     return np.abs(lm - lm1), np.abs(lm - lm2)
#     # ------------------------------------------- EVOLUTION ERRORS
#     def get_min_yc_for_lm(self, star_n, use_gaia=False):
#         '''
#         Get min lm for which the surface comp. hasn't changed. (checling the (yc, lm) point in yc_nan_lmlim file
#         :param star_n:
#         :return:
#         '''
#         yc = self.yc_nan_lmlim[0, 1:]
#         i = 0
#         yc1 = yc[i]  # assyming the lowest yc (highly likely sorface change, and therefore change in yc1)
#
#         found = True
#         while found:
#
#             l = self.get_llm('l', star_n, yc1, use_gaia)
#             l, lm_tmp = Math.get_z_for_yc_and_y(yc[i], self.yc_l_lm, l, 0) # getting lm without chacking if lm is within limits, at it is for cycle.
#
#             lm_lim = self.yc_nan_lmlim[1:, 1:]
#             lm_lim1 = lm_lim[0, i]
#             lm_lim2 = lm_lim[1, i]
#             if lm_tmp < lm_lim1 or lm_tmp > lm_lim2:
#                 i = i + 1
#             else:
#                 return yc[i]
#
#     def get_star_lm_err(self, star_n, yc_assumed, use_gaia = False):
#         '''
#         Assuming the same l, but different L -> L/M relations for different Yc, retruns (lm-lm1), (lm-lm2)
#         :param star_n:
#         :param yc_assumed:
#         :param yc1: ZAMS ( or high Yc)
#         :param yc2: END  ( or Low  Yc)
#         :return:
#         '''
#         yc = self.yc_nan_lmlim[0, 1:]
#         yc2 = yc[-1]                # Max yc is always 1.0, - ZAMS (no surface change expected)
#         yc1 = self.get_min_yc_for_lm(star_n, use_gaia)
#
#         lm1 = self.get_llm('lm', star_n, yc1, use_gaia, True) # Getting lm at different Yc (assiming the same l, but
#         lm2 = self.get_llm('lm', star_n, yc2, use_gaia, True)
#         # lm  = self.get_llm('lm', star_n, yc_assumed, use_gaia)
#
#         print('\t__STAR: {} | Evol. Err. L/M for Yc ({} - {}) L/M: ({}, {})'.format(star_n, yc1, yc2, lm1, lm2))
#
#         if lm1 == lm2 :
#              raise  ValueError('Star: {} lm1==lm2'.format(star_n))
#
#         return lm1, lm2
#
#     def get_star_ts_err(self, star_n, t_llm_mdot, yc_assumed, lim_t1, lim_t2):
#
#         # if yc1 <= yc2:
#         #     raise ValueError('yc1({}) <= yc2({}) (should be > )'.format(yc1, yc2))
#         yc = self.yc_nan_lmlim[0, 1:]
#         # lm1 = self.get_num_par('lm', star_n, yc1)  # Getting lm at different Yc (assiming the same l, but
#         # lm2 = self.get_num_par('lm', star_n, yc2)
#         # lm  = self.get_num_par('lm', star_n, yc_assumed)
#         #
#         # mdot = self.get_num_par('mdot', star_n, yc_assumed)
#
#         ts = None
#
#         xyz = self.get_xyz_from_yz(yc_assumed, star_n, 'lm', 'mdot',
#                                       t_llm_mdot[0, 1:], t_llm_mdot[1:, 0], t_llm_mdot[1:, 1:], lim_t1, lim_t2)
#         if xyz.any():
#             if len(xyz[0, :]) > 1:
#                 raise ValueError('Multiple coordinates for star: {} | Yc: {}'.format(star_n, yc_assumed))
#             else:
#                 ts = xyz[0, 0] # xyz[0, :] ALL X coordinates
#
#         ts1 = 0
#         yc1 = self.get_min_yc_for_lm(star_n)
#         xyz1 = self.get_xyz_from_yz(yc1, star_n, 'lm', 'mdot',
#                                    t_llm_mdot[0, 1:], t_llm_mdot[1:, 0], t_llm_mdot[1:, 1:], lim_t1, lim_t2)
#         if xyz1.any():
#             if len(xyz1[0, :]) > 1:
#                 raise ValueError('Multiple coordinates for star: {} | Yc: {}'.format(star_n, yc_assumed))
#             else:
#                 ts1 = xyz1[0, 0]  # xyz[0, :] ALL X coordinates
#         else:
#             ts1 = t_llm_mdot[0, -1]
#
#         ts2 = 0
#         yc2 = yc[-1]
#         xyz2 = self.get_xyz_from_yz(yc2, star_n, 'lm', 'mdot',
#                                    t_llm_mdot[0, 1:], t_llm_mdot[1:, 0], t_llm_mdot[1:, 1:], lim_t1, lim_t2)
#         if xyz2.any():
#             if len(xyz2[0, :]) > 1:
#                 raise ValueError('Multiple coordinates for star: {} | Yc: {}'.format(star_n, yc_assumed))
#             else:
#                 ts2 = xyz2[0, 0]  # xyz[0, :] ALL X coordinates
#         else:
#             ts2 = t_llm_mdot[0, 0]
#
#         print('\t__STAR: {} | {} : {} (+{} -{})'.format(star_n, 'Ts', ts, "%.2f" % np.abs(ts - ts1),
#                                                         "%.2f" % np.abs(ts - ts2)))
#
#         return ts1, ts2
#
#     # def get_star_lm_evol_err(self, star_n, l_or_lm, yc_assumed, yc1, yc2):
#     #
#     #     if l_or_lm == 'l':
#     #         yc_m_llm = Save_Load_tables.load_table('evol_yc_m_l', 'evol_yc', 'm', 'l', self.opal_used)
#     #         star_llm = self.get_num_par('l', star_n, yc_assumed)
#     #     else:
#     #         yc_m_llm = Save_Load_tables.load_table('evol_yc_m_lm', 'evol_yc', 'm', 'lm', self.opal_used)
#     #         star_llm = self.get_num_par('lm', star_n, yc_assumed)
#     #
#     #     llm = yc_m_llm[1:, 1:]
#     #
#     #     yc_ind  = Physics.ind_of_yc(yc_m_llm[0, 1:], yc_assumed)
#     #     yc1_ind = Physics.ind_of_yc(yc_m_llm[0, 1:], yc1)
#     #     yc2_ind = Physics.ind_of_yc(yc_m_llm[0, 1:], yc2)
#     #
#     #     yc_assumed_llm_row = llm[:, yc_ind]
#     #     yc1_llm_row = llm[:, yc1_ind]
#     #     yc2_llm_row = llm[:, yc2_ind]
#     #
#     #     f = interpolate.InterpolatedUnivariateSpline(yc_assumed_llm_row, yc1_llm_row)
#     #     llm1 = np.float(f([star_llm]))
#     #     f = interpolate.InterpolatedUnivariateSpline(yc_assumed_llm_row, yc2_llm_row)
#     #     llm2 = np.float(f([star_llm]))
#     #
#     #     print('\t__STAR: {} | {} : {} (+{} -{})'.format(star_n, l_or_lm, star_llm, "%.2f"%np.abs(star_llm - llm1),
#     #                                                     "%.2f" %np.abs(star_llm - llm2)))
#     #
#     #     return np.abs(star_llm - llm2), np.abs(star_llm - llm1),
#
#     # def get_star_mdot_err(self, star_n, l_or_lm, yc_assumed, yc1, yc2, l_mdot_rescription = 'nugis'):
#     #
#     #     def get_z(opal_used):
#     #         z_lmc = 0.008
#     #         table_lmc = 'table_x.data'
#     #         z_gal = 0.02
#     #         table_gal = 'table8.data'
#     #
#     #         if opal_used.split('/')[-1] == table_lmc:
#     #             return z_lmc
#     #         if opal_used.split('/')[-1] == table_gal:
#     #             return z_gal
#     #
#     #     if yc1 <= yc2:
#     #         raise ValueError('yc1({}) <= yc2({}) (should be > )'.format(yc1, yc2))
#     #
#     #     if l_or_lm == 'l':
#     #         llm1_, llm2_ = self.get_star_llm_evol_err(star_n, l_or_lm, yc_assumed, yc1, yc2)
#     #         star_llm = self.get_num_par('l', star_n, yc_assumed)
#     #         llm1 = llm1_ + star_llm  # absolute value of llm at Yc1 and Yc2
#     #         llm2 = llm2_ + star_llm
#     #
#     #         mdot1 = Physics.l_mdot_prescriptions(llm1, 10 ** get_z(self.opal_used), l_mdot_rescription)
#     #         mdot2 = Physics.l_mdot_prescriptions(llm2, 10 ** get_z(self.opal_used), l_mdot_rescription)
#     #
#     #         star_mdot = self.get_num_par('mdot', star_n, yc_assumed)
#     #
#     #         print('\t__STAR: {} | Mdot : {} (+{} -{})'.format(star_n, star_mdot, "%.2f"%np.abs(star_mdot - mdot1),
#     #                                                           "%.2f" %np.abs(star_mdot - mdot2)))
#     #
#     #         return np.abs(star_mdot - mdot1), np.abs(star_mdot - mdot2)
#     #
#     #     # llm1_, llm2_ = self.get_star_llm_evol_err(star_n, l_or_lm, yc_assumed, yc1, yc2)
#     #     #
#     #     # if l_or_lm == 'l':
#     #     #     star_llm = self.get_num_par('l', star_n, yc_assumed)
#     #     # else:
#     #     #     star_llm = self.get_num_par('lm', star_n, yc_assumed)
#     #     #
#     #     # llm1 = llm1_ + star_llm # absolute value of llm at Yc1 and Yc2
#     #     # llm2 = llm2_ + star_llm
#     #     #
#     #     # mdot1 = Physics.l_mdot_prescriptions(llm1, 10 * get_z(self.opal_used), l_mdot_rescription)
#     #     # mdot2 = Physics.l_mdot_prescriptions(llm2, 10 * get_z(self.opal_used), l_mdot_rescription)
#
#     def get_t_eff_from_atmosphere(self, star_n):
#
#         tstar = self.get_num_par('t_*', star_n)
#         rt = self.get_num_par('lRt', star_n)
#
#
#         return Math.interpolate_value_table(self.atm_table, [tstar, rt])

# =====================================================| PLOT & SM.DATA |==============================================#

class Read_Plot_file:

    # path = './data/'
    # compart = '.plot1'

    def __init__(self, plot_table, sonic_bec = False):
        i_stop = len(plot_table[:,0])

        for i in range(len(plot_table[:,8])): # if T_eff == 0 stop using the data
            if plot_table[i,8]==0:
                i_stop = i
                print('\t__Warning! In plot_table the T_eff = 0 at {} step. The data is limited to that point! '.format(
                    i))
                break


        self.time = plot_table[:i_stop, 1]          # timeyr
        self.t_c = plot_table[:i_stop,2]            # t(1)/1.d8
        self.y_c = plot_table[:i_stop,3]            # yps(1,5)
        self.l_h = plot_table[:i_stop,4]            # eshp
        self.l_he= plot_table[:i_stop,5]            # eshep
        self.m_  = plot_table[:i_stop,6]            # gms
        self.unknown = plot_table[:i_stop,7]        # esdissp
        self.t_eff= np.log10(plot_table[:i_stop,8]) # t(n)
        self.l_     = plot_table[:i_stop,9]         # dlog10(sl(n-1)/3.83d33)
        self.lm_    = Physics.loglm(self.l_, self.m_,True)
        self.rho_c  = plot_table[:i_stop,10]        # dlog10(ro(1))
        self.l_carb = plot_table[:i_stop,11]        # escop
        self.l_nu  = plot_table[:i_stop,12]         # esnyp
        self.mdot_=np.log10(plot_table[:i_stop,13]) # windmd*sec/sun
        self.t_max = plot_table[:i_stop,14]         # vvcmax
        self.rho_at_t_max = plot_table[:i_stop,15]  # gammax
        self.m_at_t_max = plot_table[:i_stop,16]    # wcrit
        if sonic_bec:
            self.logg = plot_table[:i_stop,17]          # logg
            self.rwindluca = plot_table[:i_stop,18]     # rwindluca/rsun
            self.r_n_rsun = plot_table[:i_stop, 19]     # r(n)/rsun
            self.teffluca = np.log10(plot_table[:i_stop, 20])   # teffluca
            self.Tn = np.log10(plot_table[:i_stop, 21])         # T(N)
            self.tauatR = np.log10(plot_table[:i_stop, 22]   )  # tauatR
            self.windform_23 = plot_table[:i_stop, 23]          # Abs(windmd)/(4.d0*PI*rwindluca**2*(v0+(vinf-v0)*(1.d0-R(N)/rwindluca))),
            self.windform_24 = plot_table[:i_stop, 24]          # v0+(vinf-v0)*(1-R(N)/rwindluca)
    # def

    def get_col(self, v_n):
        if v_n == 'time':
            return self.time
        if v_n == 't_c':
            return self.t_c
        if v_n == 'yc':
            return self.y_c
        if v_n == 'l_h':
            return self.l_h
        if v_n == 'l_he':
            return self.l_he
        if v_n == 'm':
            return self.m_
        if v_n == 'gp':
            return np.int(self.gp[0])
        if v_n == 'mdot':
            return self.mdot_
        if v_n == 't':
            return self.t_eff
        if v_n == 'l':
            return self.l_
        if v_n == 'rwindluca':
            return self.rwindluca
        if v_n == 'teffluca':
            return self.teffluca
        if v_n == 'tauatR':
            return self.tauatR
        if v_n == 'lm':
            return self.lm_

        raise NameError('Name: {} is not defined here. Sorrry:('.format(v_n))

    @classmethod
    def from_file(cls, plot_file_name, sonic_bec = False):

        full_name =  plot_file_name

        print('\t__Note: Reading: * {} * file.'.format(full_name))
        f = open(full_name, 'r').readlines()
        elements = len(np.array(f[0].replace("D", "E").split()))
        raws = f.__len__()
        table = np.zeros((raws, elements))

        for i in range(raws):
            # print(i)
            line = f[i].replace("D", "E").split()
            try:
                table[i, :] = np.array(line)
            except Exception:
                print('\t__Error. At raw: {} \n\t Unexpected value: {} at'.format(i, line))



            # table[i, :] = np.array(f[i].replace("D", "E").split())
        f.clear()

        # print(table[4985,:])
        print('\t__Note: file *',full_name,'* has been loaded successfully {} out of {}.'.format(table.shape, raws))
        # attached an empty raw to match the index of array
        return cls((np.vstack((np.zeros(len(table[:, 0])), table.T))).T, sonic_bec)

    def get_min_max_l_lm_val(self, y_name):
        '''
        Returns max and min available l or lm in .plot1 file
        :param y_name:
        :return:
        '''
        if y_name == 'l':
            min_l = self.l_.min()
            max_l = self.l_.max()
        else:
            # print('---------------------------------------------------------------------------')
            lm = np.log10(
                np.array([10 ** self.l_[j] / self.m_[j] for j in range(len(self.l_))]))
            min_l = lm.min()
            max_l = lm.max()

        return (min_l, max_l)

    # def get_wne_val(self, v_n):
    #     he_surf0 = self.

class Read_SM_data_file:
    '''
    The following data are available in file sm.data
    u  r  ro t  sl vu vr vro vt vsl e  dm xm
    1  2  3  4  5  6  7  8   9  10  11 12 13

    n  H  D  3He   4He  6Li 7Li 7Be 9Be 8B 10B 11B 11C
    14 15 16   17   18   19  20  21  22 23  24  25  26

    12C 13C 12N 14N 15N 16O 17O 18O 20Ne 21Ne 22Ne 23Na
    27  28  29  30  31  32  33  34   35   36   37   38

    24Mg 25Mg 26Mg 27Al 28Si 29Si 30Si 56Fe 19F 26Al
    39   40   41   42   43   44   45   46  47   48

    w  j  diff dg  d1  d2  d3  d4  d5
    49 50  51  52  53  54  55  56  57

    bvis bdiff br  bphi bfq  bfq0  bfq1  ibflag
    58    59   60   61  62     63   64     65

    Pg   Pr   HP  Grav  kappa  ediss  tau
    66   67   68   69    70     71    72

    nabla_rad   L/Ledd   nabla  P_total  mean mol wt
    73          74      75       76       77

    psi    dPg_dPr|rho  Pturb   beta     vel_conv
    78          79      80       81       82

         mdot      tau_ph
         83        84
    '''

    # example ./smdata/MYsm.data

    def __init__(self, file):

        smdata_table = self.read_sm_file_to_table(file)



        self.table = smdata_table

        self.var_names = ['nan', 'u', 'r', 'rho', 't', 'l', 'vu', 'vr',
             'vrho', 'vt', 'vsl','e', 'dm', 'xm', 'n', 'H',
             'D', 'He3', 'He4', 'Li6', 'Li7', 'Be7', 'Be9', 'B8', 'B10',
             'B11', 'C11', 'C12', 'C13', 'N12', 'Li7', 'N15', 'O16', 'O17', 'O18',
             'Ne20', 'Ne21', 'Ne22', 'Na23', 'Mg24', 'Mg25', 'Mg26', 'Al27',
             'Si28', 'Si29', 'Si30', 'Fe56', 'F19', 'Al26', 'w', 'j', 'diff',
             'dg', 'd1', 'd2', 'd3', 'd4', 'd5', 'bvis', 'bdiff', 'br', 'bphi',
             'bfq', 'bfq0', 'bfq1', 'ibflag', 'Pg', 'Pr', 'HP', 'Grav', 'kappa',
             'ediss', 'tau', 'nabla_rad', 'L/Ledd', 'nabla', 'P_total', 'mu',
             'psi', 'dPg_dPr|rho', 'Pturb', 'beta', 'vel_conv', 'mdot', 'tau_ph', '-' # the last is an empty place
             ] # 84 names (0th is not physical)

        if len(self.var_names)-1 != len(self.table[0,:]): # had to modify the -1 - to accoint for '-' case :(
            raise ValueError('len(var_names={})!=len(table[0,:]={}) in sm.file'.
                             format(len(self.var_names), len(self.table[0,:]) ))

        print('\t__Note: sm.data table shape is {}'.format(smdata_table.shape))

        self.u_ = (self.table[:, 1] / 100000) # in km/s

        self.r_ = self.table[:, 2] / Constants.solar_r  # in solar radii

        self.rho_ = np.log10(self.table[:, 3])  # in log10(rho)

        self.t_ = np.log10(self.table[:, 4])  # log10(t)

        l = self.table[:, 5]
        l[0] = 100 # in the core it is 0 otherwise
        self.l_ = np.log10(l)

        self.vu_ = self.table[:, 6] / 100000  # in km/s

        self.vr_ = self.table[:, 7] / Constants.solar_r  # in solar radii

        self.vro_ = np.log10(self.table[:, 8])

        self.vt_ = self.table[:, 9]

        self.table[0, 10] = 1  # in the core it is 0 otherwise
        self.vsl_ = np.log10(self.table[:, 10])

        self.e_ = self.table[:, 11]

        self.dm_ = self.table[:, 12]

        self.xm_ = self.table[:, 13]

        self.n_ = self.table[:, 14]

        self.H_ = self.table[:, 15]

        self.D_ = self.table[:, 16]

        self.He3_ = self.table[:, 17]

        self.He4_ = self.table[:, 18]

        self.Li6_ = self.table[:, 19]

        self.Li7_ = self.table[:, 20]

        self.Be7_ = self.table[:, 21]

        self.Be9_ = self.table[:, 22]

        self.B8_ = self.table[:, 23]

        self.B10_ = self.table[:, 24]

        self.B11_ = self.table[:, 25]

        self.C11_ =  self.table[:, 26]

        self.C12_ = self.table[:, 27]

        self.C13_ = self.table[:, 28]

        self.N12_ =  self.table[:, 29]

        self.Li7_ = self.table[:, 30]

        self.N15_ = self.table[:, 31]

        self.O16_ = self.table[:, 32]

        self.O17_ = self.table[:, 33]

        self.O18_ = self.table[:, 34]

        self.Ne20_ = self.table[:, 35]

        self.Ne21_ = self.table[:, 36]

        self.Ne22_ = self.table[:, 37]

        self.Na23_ = self.table[:, 38]

        self.Mg24_ = self.table[:, 39]

        self.Mg25_ = self.table[:, 40]

        self.Mg26_ = self.table[:, 41]

        self.Al27_ = self.table[:, 42]

        self.Si28_ = self.table[:, 43]

        self.Si29_ = self.table[:, 44]

        self.Si30_ = self.table[:, 45]

        self.Fe56_ = self.table[:, 46]

        self.F19_ = self.table[:, 47]

        self.Al26_ = self.table[:, 48]

        self.w_ =  self.table[:, 49]

        self.j_ = self.table[:, 50]

        self.diff_ = self.table[:, 51]

        self.dg_ = self.table[:, 52]

        self.d1_ = self.table[:, 53]

        self.d2_ = self.table[:, 54]

        self.d3_ = self.table[:, 55]

        self.d4_ = self.table[:, 56]

        self.d5_ = self.table[:, 57]

        self.bvis_ = self.table[:, 58]

        self.bdiff_ =self.table[:, 59]

        self.br_ = self.table[:, 60]

        self.bphi_ =self.table[:, 61]

        self.bfq_ = self.table[:, 62]

        self.bfq0_ =  self.table[:, 63]

        self.bfq1_ = self.table[:, 64]

        self.ibflag_ = self.table[:, 65]

        self.Pg_ = self.table[:, 66]

        self.Pr_ = self.table[:, 67]

        self.HP_ = np.log10(self.table[:, 68])

        self.Grav_ = self.table[:, 69]

        self.kappa_ = np.log10(self.table[:, 70])  # log(kappa)

        self.ediss_ = self.table[:, 71]

        self.tau_ = self.table[:, 72]

        self.nabla_rad_ =  self.table[:, 73]

        self.LLedd_ = self.table[:, 74]

        self.nabla_ =  self.table[:, 75]

        self.P_total_ = self.table[:, 76]

        self.mu_ = self.table[:, 77]

        self.psi_ = self.table[:, 78]

        self.dPg_dPr_rho_ = self.table[:, 79]

        self.Pturb_ = self.table[:, 80]

        self.beta_ = self.table[:, 81]

        self.vel_conv_ = self.table[:, 82]

        self.mdot_ = np.log10(self.table[:, 83] / Constants.smperyear)  # log10(mdot (in sm/year))

        self.tau_ph_ = self.table[:, 84]

    @staticmethod
    def read_sm_file_to_table(name):
        '''
                0 col - Zeors, 1 col - u, 2 col r and so forth
                :param name: name of the sm.data file (without sm.data part!)
                :return: class
                '''
        full_name = name  # + Read_SM_data_File.compart

        f = open(full_name, 'r').readlines()
        elements = len(np.array(f[0].replace("D", "E").split()))
        raws = f.__len__()
        table = np.zeros((raws, elements))

        for i in range(raws):
            table[i, :] = np.array(f[i].replace("D", "E").split())
        f.clear()

        # print('\t__Note: file *',full_name,'* has been loaded successfully.')
        # attached an empty raw to match the index of array
        return (np.vstack((np.zeros(len(table[:, 0])), table.T))).T

    @classmethod
    def from_sm_data_file(cls, name):
        '''
        0 col - Zeors, 1 col - u, 2 col r and so forth
        :param name: name of the sm.data file (without sm.data part!)
        :return: class
        '''
        full_name = name # + Read_SM_data_File.compart

        f = open(full_name, 'r').readlines()
        elements = len(np.array(f[0].replace("D", "E").split()))
        raws = f.__len__()
        table = np.zeros((raws, elements))

        for i in range(raws):
            table[i, :] = np.array(f[i].replace("D", "E").split())
        f.clear()

        # print('\t__Note: file *',full_name,'* has been loaded successfully.')
        # attached an empty raw to match the index of array
        return cls((np.vstack((np.zeros(len(table[:, 0])), table.T))).T)

    def get_xyz_from_yz(self, model_i, condition, y_name, z_name, x_1d_arr, y_1d_arr, z_2d_arr, lx1 = None, lx2 = None):
        i_req = self.ind_from_condition(condition)

        star_y = None
        star_z = None


        if y_name == z_name:
            raise NameError('y_name and z_name are the same : {}'.format(z_name))

        if y_name == 'l':
            star_y = self.l_[i_req]
        if y_name == 'lm':
            star_y = Physics.loglm(self.l_[i_req], self.xm_[i_req])
        if y_name == 'mdot':
            star_y = self.mdot_[i_req]

        if z_name == 'mdot':
            star_z = self.mdot_[i_req]
        if z_name == 'lm':
            star_z = Physics.loglm(self.l_[i_req], self.xm_[i_req])

        if star_z == None or star_y == None:
            raise ValueError('star_y:{} or star_z:{} not defined'.format(star_y,star_z))

        xyz = Physics.model_yz_to_xyz(x_1d_arr, y_1d_arr, z_2d_arr,  star_y, star_z, model_i, lx1, lx2)

        return xyz

    def get_ts_llm_mdot(self, model_i, condition, l_or_lm,
                        ts_arr, l_lm_arr, mdot2d_arr, lim_t1_obs = None, lim_t2_obs = None):
        '''
        RETURN: [0,:] - ts , [1,:] - llm , [2,:] - mdot (if there are more than one coordinates for given ts)
        :param i_req:
        :param l_or_lm: 'l' or 'lm'
        :param ts_arr:     1d_array of ts       \
        :param l_lm_arr:   1d_array of l or lm   |- From interp. tables
        :param mdot2d_arr: 2d array of mdots    /
        :model_i: is for printing which stars can or cannot be interpolated
        :param lim_t1_obs:
        :param lim_t2_obs:
        :return:
        '''
        i_req = self.ind_from_condition(condition)
        p_mdot = self.mdot_[i_req]

        if l_or_lm == 'l':
            y_coord = self.l_[i_req]
        else:
            y_coord = Physics.loglm(self.l_[i_req],self.xm_[i_req])

        # print(y_coord)
        ts_llm_mdot_coord= Physics.lm_mdot_obs_to_ts_lm(ts_arr, l_lm_arr, mdot2d_arr,
                                                        y_coord, p_mdot, model_i, lim_t1_obs, lim_t2_obs)

        return ts_llm_mdot_coord

    def get_tpar(self):

        return Physics.opt_depth_par2(self.rho_, self.t_, self.r_, self.u_, self.kappa_, self.mu_)
                               # self.mdl[i].get_col('r'), self.mdl[i].get_col('u'), self.mdl[i].get_col('kappa'),
                               # self.mdl[i].get_col('mu'))

    def get_col(self, v_n):
        '''
            The following data are available in file sm.data
            u(km/s)  r(r_sol)  ro(log10) t(log10)  sl(log10,sol)
               1        2         3         4           5

            vu vr vro vt vsl e  dm xm
            6  7  8   9  10  11 12 13

            n  H  D  3He   4He  6Li 7Li 7Be 9Be 8B 10B 11B 11C
            14 15 16   17   18   19  20  21  22 23  24  25  26

            12C 13C 12N 14N 15N 16O 17O 18O 20Ne 21Ne 22Ne 23Na
            27  28  29  30  31  32  33  34   35   36   37   38

            24Mg 25Mg 26Mg 27Al 28Si 29Si 30Si 56Fe 19F 26Al
            39   40   41   42   43   44   45   46  47   48

            w  j  diff dg  d1  d2  d3  d4  d5
            49 50  51  52  53  54  55  56  57

            bvis bdiff br  bphi bfq  bfq0  bfq1  ibflag
            58    59   60   61  62     63   64     65

            Pg   Pr   HP  Grav  kappa  ediss  tau
            66   67   68   69    70     71    72

            nabla_rad   L/Ledd   nabla  P_total  mean mol wt
            73          74        75       76       77

            psi    dPg_dPr|rho  Pturb   beta     vel_conv
            78          79      80       81       82

                 mdot(log10(sm/year)      tau_ph
                         83                 84
            '''

        if v_n == 'lm':
            return Physics.loglm(self.l_, self.xm_)

        if v_n == 1 or v_n == self.var_names[1]: #'u' or v_n == 'v': #1
            return self.u_   # in km/s

        if v_n == 2 or v_n == self.var_names[2]: #2
            return self.r_ # in solar radii

        if v_n == 3 or v_n == self.var_names[3]: #'ro' or v_n == 'rho': # 3
            return self.rho_       # in log10(rho)

        if v_n == 4 or v_n == self.var_names[4]: # 3
            return self.t_     #log10(t)

        if v_n == 5 or v_n == self.var_names[5]: # 'sl' or v_n == 'l': # 5
            self.table[0, 5] = 1 # L in the core is Zero! (error in log10() )
            return self.l_                 # log(l/l_sun)

        if v_n == 6 or v_n == self.var_names[6]: #'vu': #6
            return self.vu_         # in km/s

        if v_n == 7 or v_n == self.var_names[7]: #'vr': #7
            return self.vr_ # in solar radii

        if v_n == 8 or v_n == self.var_names[8]: #'vrho': #8
            return self.vro_

        if v_n == 9 or v_n == self.var_names[9]: #'vt': #9
            return self.vt_

        if v_n == 10 or v_n == self.var_names[10]: #'vsl': #10
            return self.vsl_

        if v_n == 11 or v_n == self.var_names[11]: #'e': #11
            return self.e_

        if v_n == 12 or v_n == self.var_names[12]: #'dm': #12
            return self.dm_

        if v_n == 13 or v_n == self.var_names[13]: #'xm' or v_n == 'm': #13
            return self.xm_

        if v_n == 14 or v_n == self.var_names[14]: #'n': #14
            return self.n_

        if v_n == 15 or v_n == self.var_names[15]: #'H': #15
            return self.H_

        if v_n == 16 or v_n == self.var_names[16]: #'D': #16
            return self.D_

        if v_n == 17 or v_n == self.var_names[17]: #'3He': #17
            return self.He3_

        if v_n == 18 or v_n == self.var_names[18]: #'4He':
            return self.He4_

        if v_n == 19 or v_n == self.var_names[19]: #'6Li':
            return self.Li6_

        if v_n == 20 or v_n == self.var_names[20]: # '7Li':
            return self.Li7_

        if v_n == 21 or v_n == self.var_names[21]: #'7Be':
            return self.Be7_

        if v_n == 22 or v_n == self.var_names[22]: # '9Be':
            return self.Be9_

        if v_n == 23 or v_n == self.var_names[23]: # '8B':
            return self.B8_

        if v_n == 24 or v_n == self.var_names[24]: # '10B':
            return self.B10_

        if v_n == 25 or v_n == self.var_names[25]: #'11B':
            return self.B11_

        if v_n == 26 or v_n == self.var_names[26]: #'11C':
            return self.C11_

        if v_n == 27 or v_n == self.var_names[27]: # '12C':
            return self.C12_

        if v_n == 28 or v_n == self.var_names[28]: # '13C':
            return self.C13_

        if v_n == 29 or v_n == self.var_names[29]: # '12N':
            return self.N12_

        if v_n == 30 or v_n == self.var_names[30]: # '7Li':
            return self.Li7_

        if v_n == 31 or v_n == self.var_names[31]: # '15N':
            return self.N15_

        if v_n == 32 or v_n == self.var_names[32]: # '16O':
            return self.O16_

        if v_n == 33 or v_n == self.var_names[33]: # '17O':
            return self.O17_

        if v_n == 34 or v_n == self.var_names[34]: # '18O':
            return self.O18_

        if v_n == 35 or v_n == self.var_names[35]: # '20Ne':
            return self.Ne20_

        if v_n == 36 or v_n == self.var_names[36]: # '21Ne':
            return self.Ne21_

        if v_n == 37 or v_n == self.var_names[37]: # '22Ne':
            return self.Ne22_

        if v_n == 38 or v_n == self.var_names[38]: # '23Na':
            return self.Na23_

        if v_n == 39 or v_n == self.var_names[39]: # '24Mg':
            return self.Mg24_

        if v_n == 40 or v_n == self.var_names[40]: # '25Mg':
            return self.Mg25_

        if v_n == 41 or v_n == self.var_names[41]: # '26Mg':
            return self.Mg26_

        if v_n == 42 or v_n == self.var_names[42]: # '27Al':
            return self.Al27_

        if v_n == 43 or v_n == self.var_names[43]: # '28Si':
            return self.Si28_

        if v_n == 44 or v_n == self.var_names[44]: # '29Si':
            return self.Si29_

        if v_n == 45 or v_n == self.var_names[45]: # '30Si':
            return self.Si30_

        if v_n == 46 or v_n == self.var_names[46]: # '56Fe':
            return self.Fe56_

        if v_n == 47 or v_n == self.var_names[47]: # '19F':
            return self.F19_

        if v_n == 48 or v_n == self.var_names[48]: # '26Al':
            return self.Al26_

        if v_n == 49 or v_n == self.var_names[49]: # 'w':
            return self.w_

        if v_n == 50 or v_n == self.var_names[50]: # 'j':
            return self.j_

        if v_n == 51 or v_n == self.var_names[51]: #  'diff':
            return self.diff_

        if v_n == 52 or v_n == self.var_names[52]: # 'dg':
            return self.dg_

        if v_n == 53 or v_n == self.var_names[53]: # 'd1':
            return self.d1_

        if v_n == 54 or v_n == self.var_names[54]: # 'd2':
            return self.d2_

        if v_n == 55 or v_n == self.var_names[55]: # 'd3':
            return self.d3_

        if v_n == 56 or v_n == self.var_names[56]: # 'd4':
            return self.d4_

        if v_n == 57 or v_n == self.var_names[57]: # 'd5':
            return self.d5_

        if v_n == 58 or v_n == self.var_names[58]: # 'bvis':
            return self.bvis_

        if v_n == 59 or v_n == self.var_names[59]: # 'bdiff':
            return self.bdiff_

        if v_n == 60 or v_n == self.var_names[60]: # 'br':
            return self.br_

        if v_n == 61 or v_n == self.var_names[61]: # 'bphi':
            return self.bphi_

        if v_n == 62 or v_n == self.var_names[62]: # 'bfq':
            return self.bfq_

        if v_n == 63 or v_n == self.var_names[63]: # 'bfq0':
            return self.bfq0_

        if v_n == 64 or v_n == self.var_names[64]: # 'bfq1':
            return self.bfq1_

        if v_n == 65 or v_n == self.var_names[65]: # 'ibflag':
            return self.ibflag_

        if v_n == 66 or v_n == self.var_names[66]: # 'Pg':
            return self.Pg_

        if v_n == 67 or v_n == self.var_names[67]: # 'Pr':
            return self.Pr_

        if v_n == 68 or v_n == self.var_names[68]: # 'HP':
            return self.HP_

        if v_n == 69 or v_n == self.var_names[69]: # 'Grav':
            return self.Grav_

        if v_n == 70 or v_n == self.var_names[70]: # 'kappa' or v_n == 'k':
            return self.kappa_         # log(kappa)

        if v_n == 71 or v_n == self.var_names[71]: # 'ediss':
            return self.ediss_

        if v_n == 72 or v_n == self.var_names[72]: # 'tau':
            return self.tau_

        if v_n == 73 or v_n == self.var_names[73]: # 'nabla_rad':
            return self.nabla_rad_

        if v_n == 74 or v_n == self.var_names[74]: # 'L/Ledd':
            return self.LLedd_

        if v_n == 75 or v_n == self.var_names[75]: # 'nabla':
            return self.nabla_

        if v_n == 76 or v_n == self.var_names[76]: # 'P_total':
            return self.P_total_

        if v_n == 77 or v_n == self.var_names[77]: # 'mean mol wt' or v_n == 'mu':
            return self.mu_

        if v_n == 78 or v_n == self.var_names[78]: # 'psi':
            return self.psi_

        if v_n == 79 or v_n == self.var_names[79]: # 'dPg_dPr|rho':
            return self.dPg_dPr_rho_

        if v_n == 80 or v_n == self.var_names[80]: # 'Pturb':
            return self.Pturb_

        if v_n == 81 or v_n == self.var_names[81]: # 'beta':
            return self.beta_

        if v_n == 82 or v_n == self.var_names[82]: # 'vel_conv':
            return self.vel_conv_

        if v_n == 83 or v_n == self.var_names[83]: # 'mdot':
            return self.mdot_# log10(mdot (in sm/year))

        if v_n == 84 or v_n == self.var_names[84]: # 'tau_ph':
            return self.tau_ph_

        if v_n == 'Pg/P_total':
            return self.Pg_/self.P_total_

        if v_n == 'Pg/Pr':
            return self.Pg_/self.Pr_

        if v_n == 'Pr/P_total':
            return self.Pr_/self.P_total_

        if v_n == 'mfp':
            return -(self.rho_ + self.kappa_)

        if v_n == 'mfp/c':
            return 10**(-(self.rho_ + self.kappa_))/Constants.light_v

        if v_n == '-': # to fill the empty arrays, (mask arrays)
            return np.zeros(self.t_.shape)


        raise NameError('\t__Error. Variable < {} > is not found |get_col|. Available name list:\n\t {}'
                 .format(v_n,self.var_names))

    def sp_i(self):
        # u = self.get_col('u')
        # t = self.get_col('t')
        # mu= self.get_col('mu')
        i = -1
        for v_n in range(len(self.u_)):
            if self.u_[v_n] >= Physics.sound_speed(self.t_[v_n], self.mu_[v_n], False):
                i = v_n
                break
        if i == -1:
            pass
            # raise ValueError('\t__Error. Sound speed is not found in data. |get_sp|')

        return i

    def get_sonic_u(self):
        return Physics.sound_speed(self.t_, self.mu_, True)

    def ind_from_condition(self, condition):
        '''

        :param cur_model: index of a model out of list of class instances that is now in the MAIN LOOP
        :param condition: 'sp' - for sonic point, 'last' for -1, or like 't=5.2' for point where temp = 5.2
        :return: index of that point
        '''
        if condition == 'last' or condition == '':
            return -1

        if condition == 'core':
            return 0

        if condition == 'sp':  # Returns the i of the velocity that is >= sonic one. (INTERPOLATION would be better)
            return self.sp_i()

        var_name = condition.split('=')[0]  # for condition like 't = 5.2' separates t as a var in sm.file and
        var_value = condition.split('=')[-1]

        if var_name not in self.var_names:  # Checking if var_name is in list of names for SM files
            raise NameError('Var_name: {} is not in var_name list: \n\t {}'
                            .format(var_name, self.var_names))

        var_arr = np.array(self.get_col(var_name))  # checking if var_value is in the column of var_name
        # print(var_value, var_arr.min(), var_arr.max())

        if var_value < var_arr.min() or var_value > var_arr.max():
            raise ValueError('Given var_value={} is beyond {} range: ({}, {})'
                             .format(var_value, var_name, var_arr.min(), var_arr.max()))

        ind = -1
        for i in range(len(var_arr)):  # searching for the next element, >= var_value. [INTERPOLATION would be better]
            if var_value >= var_arr[i]:
                ind = i
                break
        if ind == -1:
            raise ValueError('ind = -1 -> var_value is not found in the var_arr. | var_value={}, array range: ({}, {})'
                             .format(var_value, var_name, var_arr.min(), var_arr.max()))

        return ind

    def get_lm_col(self):
        return Physics.loglm(self.l_, self.xm_, True)


    def get_cond_value(self, v_n, condition):
        '''
        CONDITIONS: 'sp'(v==v_s); 't=5.2' or any v_n=number
        :param v_n:
        :param condition:
        :return:
        '''

        def xy(x, y):
            res = []
            for i in range(1,len(x)):
                res = np.append(res, (y[i]-y[i-1])/(x[i]-y[i-1]))

            return res

        if v_n == 'grad_u':
            depth_r = 0.0000001

            ind = self.ind_from_condition(condition)
            u = self.get_col('u')
            r = self.get_col('r')
            diff_u = np.diff(u)
            diff_r = np.diff(r)

            # grid_r = np.linspace( r[0], r[-1], np.int((r[-1]-r[0])/depth) )
            # grid_r = np.mgrid[r[0] : r[-1] : np.int((r[-1]-r[0])/depth_r)*1j]
            # grid_u = interpolate.interp1d(r,u,kind='linear')(grid_r)



            # grad_u = xy(r,u)
            grad_u = np.gradient(u,r)
            # plt.plot(r, grad_u, '-', color='black')
            # plt.plot(r, u, '-', color='blue')
            # plt.plot(r[:-1], grad_u, '-', color='black')
            # plt.plot(r[:-1], u[:-1], '-', color='blue')
            # plt.show()
            return  grad_u[-1] # (u[-1]-u[-2])/(r[-1]-r[-2])#

        if v_n == 'teff'and condition == '':
            ind = self.ind_from_condition(condition)
            l = self.get_col('l')[ind]
            r = self.get_col('r')[ind]
            return Physics.steph_boltz_law_t_eff(l, r)

        if v_n == 'teff/ts4' and condition =='':
            ind = self.ind_from_condition(condition)
            l = self.get_col('l')[ind]
            r = self.get_col('r')[ind]
            ts = 10**self.get_col('t')[ind]
            teff = 10**Physics.steph_boltz_law_t_eff(l, r)


            return ((teff/ts)**4)*100 # in %

        if v_n == 'tpar' and condition == '': # optical depth parameter can be estimated only at a point, as it requires dr/du
            return Physics.opt_depth_par2(self.rho_,self.t_,self.r_,self.u_,self.kappa_,self.mu_)

        # if v_n == 'r_env' and condition == '':
        #     return self.get_m_r_envelope('r')
        #
        # if v_n == 'm_env' and condition == '':
        #     return self.get_m_r_envelope('xm')

        ind = self.ind_from_condition(condition)
        return np.float(self.get_col(v_n)[ind])

    def get_par_table(self, y_name = 'l', i = -1):


        # print(
        #     '\t| Mdot'
        #     '\t| Mass'
        #     '\t| R/Rs '
        #     '\t\t| L/Ls'
        #     '\t| kappa  '
        #     '\t| log(Rho)'
        #     '\t\t| Temp'
        #     '\t\t| mfp  '
        #     '\t\t| vel '
        #     '\t\t| gamma'
        #     '\t\t| tpar '
        #     '\t\t| HP '
        #     '\t\t| tau '
        #     '\t\t|')

        if y_name == 'l':
            print(
                # "%2.0f" % model,
                '|', "%.2f" % self.mdot_[i],
                '|', "%.1f" % self.xm_[i],
                '|', "%.4f" % self.r_[i],
                '|', "%.3f" % self.l_[i],
                '|', "%.4f" % 10**self.kappa_[i],
                '|', "%.3f" % self.rho_[i],
                '|', "%.3f" % self.t_[i],
                '|', "%.3f" % Physics.mean_free_path(self.rho_[i], self.kappa_[i]),
                '|', "%5.2f" % self.u_[i],
                '|', "%.4f" % self.LLedd_[i],
                '|', "%.3f" % Physics.opt_depth_par(i, self.rho_,self.kappa_,self.u_,self.r_, self.t_, self.mu_),
                '|', "%.3f" % self.HP_[i],
                '|', "%.3f" % np.log10(self.C12_[i]))
        if y_name == 'lm':
            print(
                # "%2.0f" % model,
                '|', "%.2f" % self.mdot_[i],
                '|', "%.1f" % self.xm_[i],
                '|', "%.4f" % self.r_[i],
                '|', "%.3f" % np.log10(10**self.l_[i]/self.xm_[i]),
                '|', "%.4f" % 10**self.kappa_[i],
                '|', "%.3f" % self.rho_[i],
                '|', "%.3f" % self.t_[i],
                '|', "%.3f" % Physics.mean_free_path(self.rho_[i], self.kappa_[i]),
                '|', "%5.2f" % self.u_[i],
                '|', "%.4f" % self.LLedd_[i],
                '|', "%.3f" % Physics.opt_depth_par(i, self.rho_,self.kappa_,self.u_,self.r_, self.t_, self.mu_),
                '|', "%.3f" % self.HP_[i],
                '|', "%.3f" % np.log10(self.C12_[i]))


        # return np.array([ self.mdot_[i],
        #                   self.xm_[i],
        #                   self.r_[i],
        #                   self.l_[i],
        #                   self.kappa_[i],
        #                   self.rho_[i],
        #                   self.t_[i],
        #                   Physics.mean_free_path(self.rho_[i], self.kappa_[i]),
        #                   self.u_[i],self.LLedd_[i],
        #                   Physics.opt_depth_par(i, self.rho_, self.kappa_, self.u_, self.r_, self.t_, self.mu_),
        #                   self.HP_[i], self.tau_[i], ])

    def get_set_of_cols(self, v_n_arr):
        res = np.zeros(( len(self.r_), len(v_n_arr) ))
        for i in range(len(v_n_arr)):
            res[:,i] = self.get_col(v_n_arr[i])

        print('\t__Note: vars:[', v_n_arr, '] returned arr:', res.shape)
        return res

# class Read_SP_data_file_OLD:
#
#     def __init__(self, sp_data_file, out_dir = '../data/output/', plot_dir = '../data/plots/'):
#
#         self.files = sp_data_file
#         self.out_dir = out_dir
#         self.plot_dir = plot_dir
#
#
#         self.list_of_v_n = ['l', 'm', 't', 'mdot', 'tau', 'r', 'Yc', 'k']
#
#         self.table = np.loadtxt(sp_data_file)
#
#         print('File: {} has been loaded successfully.'.format(sp_data_file))
#
#         # --- Critical values ---
#
#         self.l_cr = np.float(self.table[0, 0])  # mass array is 0 in the sp file
#         self.m_cr = np.float(self.table[0, 1])  # mass array is 1 in the sp file
#         self.yc_cr = np.float(self.table[0, 2])  # mass array is 2 in the sp file
#         self.ys_cr = np.float(self.table[0, 3])  # mass array is 2 in the sp file
#         self.lmdot_cr = np.float(self.table[0, 4])  # mass array is 3 in the sp file
#         self.r_cr = np.float(self.table[0, 5])  # mass array is 4 in the sp file
#         self.t_cr = np.float(self.table[0, 6])  # mass array is 4 in the sp file
#         self.tau_cr = np.float(self.table[0, 7])
#
#         # --- Sonic Point Values ---
#
#         self.l = np.array(self.table[1:, 0])
#         self.m = np.array(self.table[1:, 1])
#         self.yc = np.array(self.table[1:, 2])
#         self.ys = np.array(self.table[1:, 3])
#         self.lmdot = np.array(self.table[1:, 4])
#         self.rs = np.array(self.table[1:, 5])
#         self.ts = np.array(self.table[1:, 6])
#         self.tau = np.array(self.table[1:, 7])
#         self.k = np.array(self.table[1:, 8])
#         self.rho = np.array(self.table[1:, 9])
#
#         # self.k = np.array(self.table[1:, 6])
#         # self.rho = np.array(self.table[1:, 8])
#
#     def get_crit_value(self, v_n):
#         if v_n == 'l':
#             return np.float( self.l_cr )
#
#         if v_n =='m' or v_n == 'xm':
#             return np.float( self.m_cr )
#
#         if v_n == 't':
#             return np.float( self.t_cr )
#
#         if v_n == 'mdot':
#             return np.float( self.lmdot_cr)
#
#         if v_n == 'r':
#             return np.float(self.r_cr)
#
#         if v_n == 'Yc':
#             return np.float(self.yc_cr)
#
#         if v_n == 'ys':
#             return np.float(self.ys_cr)
#
#         if v_n == 'tau':
#             return self.tau_cr
#
#         if v_n == 'lm':
#             return np.float(Physics.loglm(self.l_cr, self.m_cr, False) )
#
#         raise NameError('v_n {} is not in the list: {} (for critical values)'.format(v_n, self.list_of_v_n))
#
#     def get_sonic_cols(self, v_n):
#         if v_n == 'l':
#             return self.l
#
#         if v_n =='m' or v_n == 'xm':
#             return self.m
#
#         if v_n == 't':
#             return self.ts
#
#         if v_n == 'mdot':
#             return self.lmdot
#
#         if v_n == 'r':
#             return self.rs
#
#         if v_n == 'Yc':
#             return self.yc
#
#         if v_n == 'ys':
#             return self.ys
#
#         if v_n == 'k':
#             return self.k
#
#         if v_n == 'rho':
#             return self.rho
#
#         if v_n == 'tau':
#             return self.tau
#
#         if v_n == 'lm':
#             return Physics.loglm(self.l, self.m, True)
#
#         raise NameError('v_n {} is not in the list: {} (for critical values)'.format(v_n, self.list_of_v_n))

# ====================================================| WIND & ATMOSPHERE |============================================#

class Read_Wind_file:
    def __init__(self, wind_table):

        self.table = wind_table
        self.v = wind_table[:, 1] / 100000
        self.r = wind_table[:, 2] / Constants.solar_r
        self.rho=np.log10(wind_table[:, 3])
        self.t = np.log10(wind_table[:, 4])
        self.kappa= wind_table[:, 5]
        self.tau = wind_table[:, 6]
        self.gp = wind_table[:, 7]
        self.mdot= wind_table[:, 8]
        self.nine= wind_table[:, 9]
        self.ten = wind_table[:, 10]
        self.eleven= wind_table[:, 11] # wind luminocity
        self.kappa_eff= wind_table[:, 12]
        self.thirteen=np.log10(wind_table[:, 13]) # eta
        self.last = wind_table[:, -1]

        self.var_names=['u', 'r', 'rho', 't', 'kappa', 'tau', 'gp', 'mdot', '9', '10', '11', 'kappa_eff', '13', 'last']

    def get_col(self, v_n):
        if v_n == 'u':
            return self.v
        if v_n == 'r':
            return self.r
        if v_n == 'rho':
            return self.rho
        if v_n == 't':
            return self.t
        if v_n == 'kappa':
            return self.kappa
        if v_n == 'tau':
            return self.tau
        if v_n == 'gp':
            return np.int(self.gp[0])
        if v_n == 'mdot':
            return np.log10(self.mdot/Constants.smperyear)
        if v_n == '9':
            return self.nine # diffusive approximation temperature
        if v_n == '10':
            return self.ten
        if v_n == '11':
            return self.eleven
        if v_n == 'kappa_eff':
            return self.kappa_eff
        if v_n == '13':
            return self.thirteen
        if v_n == 'last':
            return self.last

    def get_value(self, v_n, condition='ph'):
        '''
        Condtitions: 0, 1, ... Special: 'ph'
        :param v_n:
        :param condition:
        :return:
        '''
        if condition == 'ph':
            i = self.get_col('gp')
            return self.get_col(v_n)[i]
        else:
            return self.get_col(v_n)[condition]


    @classmethod
    def from_wind_dat_file(cls, name):
        '''
        0 col - Zeors, 1 col - u, 2 col r and so forth
        :param name: name of the sm.data file (without sm.data part!)
        :return: class
        '''
        full_name = name # + Read_SM_data_File.compart

        f = open(full_name, 'r').readlines()
        elements = len(np.array(f[0].replace("D", "E").split()))
        raws = f.__len__()
        table = np.zeros((raws, elements))

        i_inf = None
        if 'Infinity' in f[0].split(' '):
            # i_inf = f[0].split(' ').index('Infinity')
            parts = f[0].split('Infinity')
            res = parts[0] + '0.00000000000000D+00' + parts[-1]

        for i in range(raws): # WARNING ! RADING ROWS N-1! BECAUSE the last row can have 'infinity'

            if 'Infinity' in f[i].split(' '):
                parts = f[i].split('Infinity')
                #if len(parts) > 2: raise ValueError('More than 1 *Infinity* detected')

                res = parts[0]
                for i in range(1, len(parts)):

                    res = res + '0.00000000000000D+00' + parts[i]


                #res = parts[0] + '0.00000000000000D+00' + parts[-1]  # In case the vierd value is there :(

                print('\t__Replaced Row in WIND file [Infinity] Row: {} File: {}'.format(i, name))
                table[i, :] = np.array(res.replace("D", "E").split())

            else:
                if '0.14821969375237-322' in f[i].split(' '):
                    parts = f[i].split('0.14821969375237-32')
                    res = parts[0] + '0.00000000000000D+00' + parts[-1]         # In case the vierd value is there :(
                    print('\t__Replaced Row in WIND file [0.14821969375237-322] Row: {} File: {}'.format(i, name))
                    table[i, :] = np.array(res.replace("D", "E").split())
                else:
                    table[i, :] = np.array(f[i].replace("D", "E").split())
        f.clear()

        table = (np.vstack((np.zeros(len(table[:, 0])), table.T))).T

        # if i_inf !=None:
        #     x_row = table[1:, 0]
        #     y_row = table[]

        # print('\t__Note: file *',full_name,'* has been loaded successfully.')
        # attached an empty raw to match the index of array
        return cls(table)

class Read_Atmosphere_File:

    list_of_names = ['model', 't_*', 'rt',  'l', 'mdot', 'v_inf', 'r_*',  't_eff', 'r_eff', 'm','g','q','d_inf','eta']

    def __init__(self, atm_file, out_dir = '../data/output/', plot_dir = '../data/plots/'):

        self.atm_file = atm_file

        self.out_dir = out_dir
        self.plot_dir= plot_dir

        if atm_file == None: raise NameError('Atmosphere file is not given')
        self.names, self.table, self.models = Save_Load_tables.read_genergic_table(self.atm_file, 'model') # now it suppose to read the table

    def v_n_to_v_n(self, v_n):

        if not v_n in self.names:
            raise NameError('Translation is not provided for {} \n (not found in {})'.format(v_n, self.names))

        else: return v_n

    def get_col(self, v_n):
        col = self.table[:, self.names.index(self.v_n_to_v_n(v_n))]

        if v_n == 't_*' or v_n == 't_eff':
            return np.log10(col*1000)
        else:
            return col

    def get_value(self, v_n, str_model):
        value = self.table[self.models.index(str_model), self.names.index(self.v_n_to_v_n(v_n))]

        if v_n == 't_*' or v_n == 't_eff':
            return np.log10(value*1000)
        else:
            return value

    #
    #
    #
    #
    # def get_crit_value(self, v_n):
    #
    #
    #     if v_n == 'lm':
    #         l = self.table[0, self.names.index(self.v_n_to_v_n('l'))]
    #         m = self.table[0, self.names.index(self.v_n_to_v_n('m'))]
    #         return Physics.loglm(l, m)
    #
    #
    #     v_n_tr = self.v_n_to_v_n(v_n)
    #     if not v_n_tr in self.names:
    #         raise NameError('v_n_traslated({}) not in the list of names from file ({})'.format(v_n_tr, self.names))
    #     return self.table[0, self.names.index(self.v_n_to_v_n(v_n))]
    #
    # def get_sonic_cols(self, v_n):
    #
    #     if v_n == 'lm':
    #         l = self.table[1:, self.names.index(self.v_n_to_v_n('l'))]
    #         m = self.table[1:, self.names.index(self.v_n_to_v_n('m'))]
    #         return Physics.loglm(l, m, True)
    #     else:
    #         return self.table[1:, self.names.index(self.v_n_to_v_n(v_n))]
    #
    #     # --- Critical values ---
    #
    #     # self.l_cr = np.float(self.table[0, 0])  # mass array is 0 in the sp file
    #     # self.m_cr = np.float(self.table[0, 1])  # mass array is 1 in the sp file
    #     # self.yc_cr = np.float(self.table[0, 2])  # mass array is 2 in the sp file
    #     # self.ys_cr = np.float(self.table[0, 3])  # mass array is 2 in the sp file
    #     # self.lmdot_cr = np.float(self.table[0, 4])  # mass array is 3 in the sp file
    #     # self.r_cr = np.float(self.table[0, 5])  # mass array is 4 in the sp file
    #     # self.t_cr = np.float(self.table[0, 6])  # mass array is 4 in the sp file
    #     # self.tau_cr = np.float(self.table[0, 7])
    #     #
    #     # # --- Sonic Point Values ---
    #     #
    #     # self.l = np.array(self.table[1:, 0])
    #     # self.m = np.array(self.table[1:, 1])
    #     # self.yc = np.array(self.table[1:, 2])
    #     # self.ys = np.array(self.table[1:, 3])
    #     # self.lmdot = np.array(self.table[1:, 4])
    #     # self.rs = np.array(self.table[1:, 5])
    #     # self.ts = np.array(self.table[1:, 6])
    #     # self.tau = np.array(self.table[1:, 7])
    #     # self.k = np.array(self.table[1:, 8])
    #     # self.rho = np.array(self.table[1:, 9])
    #
    #     # self.k = np.array(self.table[1:, 6])
    #     # self.rho = np.array(self.table[1:, 8])

    def get_2d_arr(self, v_n, filling=-1, replace_to=-1):

        arr = np.zeros((100, 100))
        arr.fill(filling)

        models = np.zeros(2)
        for m in self.models:
            row = np.array([m.split('-')[0], m.split('-')[-1]], dtype=np.int)
            models = np.vstack((models, row))

        models = np.delete(models, 0, 0)

        for x in models[:, 0]:
            for y in models[:, 1]:
                str_model = "%0.2d" % x + '-' + "%0.2d" % y
                if str_model in self.models:
                    # print('Model {} found'.format(str_model))
                    arr[np.int(y), np.int(x)] = self.get_value(v_n, str_model)
                # else:
                # print('Model {} not found'.format(str_model))

        arr = arr[~np.all(arr == filling, axis=1)]
        arr = arr.T[~np.all(arr.T == filling, axis=1)].T

        arr[arr == filling] = replace_to

        # print(arr)

        return arr

    def get_table(self, x_v_n, y_v_n, z_v_n, filling=None):

        x_data = np.unique(self.get_2d_arr(x_v_n, -1))
        x_data = np.delete(x_data, 0, 0)
        y_data = np.unique(self.get_2d_arr(y_v_n, -1))
        y_data = np.delete(y_data, 0, 0)

        z_data = self.get_2d_arr(z_v_n, -1, filling)

        if len(x_data) != len(z_data[0, :]): raise ValueError('Wrong x_len: len(x_data){}!=len(z_data[0,:]){}'
                                                              .format(len(x_data), len(z_data[0, :])))
        if len(y_data) != len(z_data[:, 0]): raise ValueError('Wrong x_len: len(x_data){}!=len(z_data[0,:]){}'
                                                              .format(len(y_data), len(z_data[:, 0])))

        table = Math.combine(x_data, y_data, z_data[::-1, :])

        return table



    def plot_tstar_rt(self, x_v_n, y_v_n, z_v_n, metal):

        models = np.zeros(2)
        for m in self.models:
            row = np.array([m.split('-')[0], m.split('-')[-1]], dtype=np.int)
            models = np.vstack((models, row))

        models=np.delete(models, 0, 0)

        table = self.get_table('t_*', 'rt', 't_eff', None)


        plots=PlotBackground2()
        plots.set_clean=True
        plots.set_show_contours=True
        plots.set_rotate_labels=310
        plots.set_label_sise=12

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        plots.plot_color_background(ax, table, x_v_n, y_v_n, z_v_n, metal, '')
        plt.show()

        return 0

# =======================================================| OBSERVABLES |===============================================#

class Read_Observables:

    def __init__(self, observ_name, gal_or_lmc):
        '''

        :param observ_name:
        :param yc: -1 (None, use m), 0.1 to 0.99 (My relation for diff yc), 10 (use Langer et al. 1989)
        :param clump_used:
        :param clump_req:
        '''

        self.set_use_gaia        = True
        self.set_use_atm_file    = True
        self.set_check_lm_for_wne= True

        self.set_load_yc_l_lm       = True
        self.set_load_yc_nan_lmlim  = True

        self.set_if_evol_err_out = 't1' # what boundary to return if the ts error is out of boundary
        # --------------------------------------------------------------------------------------------------------------

        self.file_name = observ_name
        self.set_clump_used = 4
        self.set_clump_modified = 4
        self.metal = gal_or_lmc

        self.table = []
        with open(observ_name, 'r') as f:
            for line in f:
                if '#' not in line.split() and line.strip():  # if line is not empty and does not contain '#'
                    self.table.append(line)

        self.names = self.table[0].split()
        self.num_stars = len(self.table) - 1  # as first row is of var names

        if len(self.names) != len(self.table[1].split()):
            print('\t___Error. Number of vars in list({}) != number of cols in observ.data file({}) '
                  '|Read_Observables, __init__|'.format(len(self.names), len(self.table[1].split())))
        print('\t__Note: Data include following paramters:\n\t | {} |'.format(self.table[0].split()))

        self.table.remove(self.table[0])  # removing the var_names line from the array. (only actual values left)

        # -----------------DISTRIBUTING READ COLUMNS-------------
        #  'lgaia', 'llgaia', 'ulgaia'
        self.num_v_n = ['N', 't_*', 'lRt', 'm', 'l', 'mdot', 'v_inf', 'eta']

        # list of used v_n's; can be extended upon need ! 'N' should be first
        self.cls = 'class'  # special variable

        self.str_v_n = []

        self.set_num_str_tables()

        # --- --- GAIA LUMINOCITIES --- ---
        if self.set_use_gaia and gal_or_lmc == 'gal':
            self.gaia_names, self.gaia_table = Save_Load_tables.read_genergic_table(
                observ_name.split('.data')[0] + '_gaia_lum.data')


        # --- --- Potsdam Atmospheres --- ---
        if self.set_use_atm_file:
            self.atm = Read_Atmosphere_File(Files.get_atm_file(gal_or_lmc), self.metal)
            self.atm_table_teff = self.atm.get_table('t_*', 'rt', 't_eff')
            self.atm_table_reff = self.atm.get_table('t_*', 'rt', 'r_eff')

        # --- --- Mass-Luminocity Relations --- ---
        if self.set_load_yc_l_lm:
            self.yc_l_lm = Save_Load_tables.load_table('yc_l_lm', 'yc', 'l', 'lm', self.metal, '')

        if self.set_load_yc_nan_lmlim:
            self.yc_nan_lmlim = Save_Load_tables.load_table('yc_nan_lmlim', 'yc', 'nan', 'lmlim', self.metal, '')

    def set_num_str_tables(self):

        self.numers = np.zeros((self.num_stars, len(self.num_v_n)))
        self.clses = []

        # ----------------------------------------SET THE LIST OF CLASSES------------------
        for i in range(len(self.table)):
            n = self.names.index(self.cls)
            if self.cls in self.names:
                self.clses.append(self.table[i].split()[n])

        # -----------------------------------------SET 2D ARRAY OF NUM PARAMETERS---------
        for i in range(len(self.table)):
            for j in range(len(self.num_v_n)):
                n = self.names.index(self.num_v_n[j])
                if self.num_v_n[j] in self.names:
                    self.numers[i, j] = np.float(self.table[i].split()[n])

        # -----------------------------------------SETTING THE CATALOGUE NAMES------------
        self.stars_n = self.numers[:, 0]

        if len(self.numers[:, 0]) != len(self.clses):
            raise ValueError(
                'Size of str. vars ({}) != size of num.var ({})'.format(len(self.numers[:, 0]), len(self.clses)))

        # print(self.numers, self.clses)
        print('\n\t__Note. In file: {} total {} stars are loaded. \n\t  Available numerical parameters: {} '
              .format(self.file_name, len(self.numers), self.num_v_n))

    def modify_value(self, v_n, value):
        if v_n == 't' or v_n == 't_*':
            return np.log10(value * 1000)

        if v_n == 'mdot':
            new_mdot = value
            if self.set_clump_used != self.set_clump_modified:
                f_WR = 10 ** (0.5 * np.log10(self.set_clump_used / self.set_clump_modified))  # modify for a new clumping
                new_mdot = value + np.log10(f_WR)
                print('\nClumping factor changed from {} to {}'.format(self.set_clump_used, self.set_clump_modified))
                print('new_mdot = old_mdot + ({}) (f_WR: {} )'
                      .format("%.2f" % np.log10(f_WR), "%.2f" % f_WR))
                # print('| i | mdot | n_mdot | f_WR |')
                # for i in range(len(res)):
                #     f_wr = 10**np.float(res2[i]) / 10**np.float(res[i])
                #     print('| {} | {} | {} | {} |'.format("%2.f" % i, "%.2f" % np.float(res[i]) ,"%.2f" % np.float(res2[i]), "%.2f" % f_wr ) )
            return new_mdot

        return value

    def get_num_par_from_table(self, v_n, star_n):

        if v_n not in self.num_v_n:
            raise NameError('v_n: {} is not in set of num. pars: {}'.format(v_n, self.num_v_n))
        if star_n not in self.numers[:, 0]:
            raise NameError('star_n: {} is not in the list of star numbers: {}'.format(star_n, self.numers[:, 0]))

        # -- GAIA MISSION for GALAXY (table8) ---
        if v_n == 'l' and self.metal.split('/')[-1] == 'gal' and self.set_use_gaia:
            l = self.get_gaia_lum(star_n)[0]
            return l

        if v_n == 'mdot' and self.metal.split('/')[-1] == 'gal' and self.set_use_gaia:
            mdot = self.get_gaia_mdot(star_n)[0]
            return mdot

        ind_star = np.where(self.numers[:, 0] == star_n)[0][0]
        ind_v_n = self.num_v_n.index(v_n)
        value = self.numers[ind_star, ind_v_n]

        value = self.modify_value(v_n, value)

        # if v_n == 'mdot': print('\t\t mdot: {}'.format(value))

        return value

    def get_llm(self, l_or_lm, star_n, yc_val=None):

        l = self.get_num_par_from_table('l', star_n)
        if l_or_lm == 'l':
            return self.get_num_par_from_table('l', star_n)

        if l < self.yc_l_lm[1:, 0].min() or l > self.yc_l_lm[1:, 0].max():
            lm = np.log10(10 ** l / 10 ** Physics.l_to_m_langer(l))
            print('___Warning: Star: {} l:{} is beyong l->lm table [{}, {}]. Using Langer Prescription lm:{}'
                  .format(star_n, "%.2f" % l, "%.2f" %  self.yc_l_lm[1:, 0].min(), "%.2f" % self.yc_l_lm[1:, 0].max(), "%.2f" % lm ))
            return lm

        if yc_val >= 0 and yc_val <= 1.:

            if self.set_load_yc_l_lm:
                l, lm = Math.get_z_for_yc_and_y(yc_val, self.yc_l_lm, l, 0)
            else:
                lm = np.log10(10 ** l / 10 ** Physics.l_to_m_langer(l))
                print('\t__Star: {} Using Langer l->lm. {}->{}'.format(star_n, l, lm))


            if self.set_load_yc_nan_lmlim and self.set_check_lm_for_wne:
                yc_ind = Math.find_nearest_index(self.yc_nan_lmlim[0, 1:], yc_val) + 1
                lm_lim = self.yc_nan_lmlim[1:, yc_ind]

                if lm < lm_lim[0] or lm > lm_lim[1]:
                    print('For Yc:{}, star:{} has lm:{} (l:{}), available wne range: [{}, {}]'
                                     .format(yc_val, star_n, "%.2f" % lm, "%.2f" % l, "%.2f" % lm_lim[0], "%.2f" % lm_lim[1]))
                    # raise ValueError('For Yc:{}, star:{} has lm:{} (l:{}), available wne range: [{}, {}]'
                    #                  .format(yc_val, star_n, "%.2f" % lm, "%.2f" % l, "%.2f" % lm_lim[0], "%.2f" % lm_lim[1]))
                else:
                    print('\t__Star:{} l:{} lm: {} m:{} is in L/M WNE range [{}, {}]'
                          .format(star_n, "%.2f" % l, "%.2f" % lm,
                                  "%.1f" % Physics.lm_l__to_m(lm, l),
                                  "%.2f" % lm_lim[0], "%.2f" % lm_lim[1]))


                return lm
            else:
                return lm

        if yc_val == -1 or yc_val == 'langer':
            return np.log10(10 ** l / 10 ** Physics.l_to_m_langer(l))

        if yc_val == None or yc_val == 'hamman':
            m = self.get_num_par_from_table('m', star_n)
            np.log10(10 ** l / m)

        raise ValueError('Yc = {} is not recognised. '
                         'Use: 0-1 for loading tables | -1 or langer for langer |  None or hamman for hamman')

    # ------------------------GAIA & ATMOSPHERE ------------------------------------------------------------------------

    def get_gaia_lum(self, star_n):
        '''
        returns [l, ll, ul]
        :param star_n:
        :return:
        '''
        if star_n not in self.gaia_table[:, 0]:
            raise ValueError('Star: {} is not in the GAIA list: {}'.format(star_n, self.gaia_table[:, 0]))
        ind = Math.find_nearest_index(self.gaia_table[:, 0], star_n)
        # print(ind)
        return np.array([self.gaia_table[ind, 2], self.gaia_table[ind, 3], self.gaia_table[ind, 4]])

    def get_gaia_mdot(self, star_n):
        '''
        Returns [mdot, lmdot, umdot], assuming that in the input file [star_n, l, ll, ul, mdot, lmdot, umdot]
        :param star_n:
        :return:
        '''
        if star_n not in self.gaia_table[:, 0]:
            raise ValueError('Star: {} is not in the GAIA list: {}'.format(star_n, self.gaia_table[:, 0]))
        ind = Math.find_nearest_index(self.gaia_table[:, 0], star_n)
        # print(ind)
        return np.array([self.gaia_table[ind, 5], self.gaia_table[ind, 6], self.gaia_table[ind, 7]])

    def get_t_eff_from_atmosphere(self, star_n):

        tstar = self.get_num_par('t_*', star_n)
        rt = self.get_num_par('lRt', star_n)

        return Math.interpolate_value_table(self.atm_table_teff, [tstar, rt])











    def rescale_rt_for_gaia(self, star_n, rt_old):

        self.set_use_gaia = False
        mdot_old = self.get_num_par('mdot', star_n)

        self.set_use_gaia = True
        mdot_new = self.get_gaia_mdot(star_n)[0]

        rt_new = (2/3) * (mdot_old - mdot_new) + rt_old

        print('_____rt_old:{}, rt_new:{}, as mdot_old:{} and mdot_new:{}'.
              format(rt_old,rt_new,mdot_old,mdot_new))

        return rt_new


    def get_y_from_atmosphere(self, v_n, star_n):

        tstar = self.get_num_par('t_*', star_n)
        rt = self.get_num_par('lRt', star_n)

        if self.set_use_gaia:
            rt = self.rescale_rt_for_gaia(star_n, rt)


        if v_n == 'r_eff':
            return Math.interpolate_value_table(self.atm_table_reff, [tstar, rt])
        if v_n == 't_eff':
            t_eff = Math.interpolate_value_table(self.atm_table_teff, [tstar, rt])
            return t_eff

    def get_star_atm_obs_err(self, v_n, star_n):

        rt_err    = Files.get_obs_err_rt(self.metal)
        tstar_err = Files.get_obs_err_tstar(self.metal)

        tstar     = self.get_num_par('t_*', star_n)
        rt        = self.get_num_par('lRt', star_n)

        if self.set_use_gaia:
            rt = self.rescale_rt_for_gaia(star_n, rt)

        if v_n == 't_eff':
            t1 = Math.interpolate_value_table(self.atm_table_teff, [tstar + tstar_err, rt + rt_err])
            t2 = Math.interpolate_value_table(self.atm_table_teff, [tstar + tstar_err, rt - rt_err])
            t3 = Math.interpolate_value_table(self.atm_table_teff, [tstar - tstar_err, rt + rt_err])
            t4 = Math.interpolate_value_table(self.atm_table_teff, [tstar - tstar_err, rt - rt_err])

            return np.array([t1, t2, t3, t4]).min(), np.array([t1, t2, t3, t4]).max()

        if v_n == 'r_eff':
            r1 = Math.interpolate_value_table(self.atm_table_reff, [tstar + tstar_err, rt + rt_err])
            r2 = Math.interpolate_value_table(self.atm_table_reff, [tstar + tstar_err, rt - rt_err])
            r3 = Math.interpolate_value_table(self.atm_table_reff, [tstar - tstar_err, rt + rt_err])
            r4 = Math.interpolate_value_table(self.atm_table_reff, [tstar - tstar_err, rt - rt_err])

            return np.array([r1, r2, r3, r4]).min(), np.array([r1, r2, r3, r4]).max()




    # ---------------------------------------------PUBLIC FUNCTIONS---------------------------------------
    def l_lm_for_errs(self, l, star_n, yc_val):
        '''
        Changes the L of the star to the maximmum available for plotting.
        :param v_n:
        :param star_n:
        :param yc_val:
        :return:
        '''

        # if l < self.yc_l_lm[1:, 0].min(): l = self.yc_l_lm[1:, 0].min()
        # if l > self.yc_l_lm[1:, 0].max(): l = self.yc_l_lm[1:, 0].max()
        # print('\__ Yc: {}'.format(yc_val))

        if yc_val >= 0 and yc_val <= 1.:

            if l < self.yc_l_lm[1:, 0].min() or l > self.yc_l_lm[1:, 0].max(): # this allows to get ZAMS Yc for models outside the WNE range
                lm = np.log10(10 ** l / 10 ** Physics.l_to_m_langer(l))
                print('___Warning: Star: {} l:{} is beyong l->lm table [{}, {}]. Using Langer Prescription lm:{}'
                      .format(star_n, "%.2f" % l, "%.2f" % self.yc_l_lm[1:, 0].min(),
                              "%.2f" % self.yc_l_lm[1:, 0].max(), "%.2f" % lm))
                return lm

            l, lm = Math.get_z_for_yc_and_y(yc_val, self.yc_l_lm, l, 0)


            yc_lim_arr = self.yc_nan_lmlim[0, 1:]
            lm_low_lim_arr = self.yc_nan_lmlim[1, 1:]
            lm_up_lim_arr = self.yc_nan_lmlim[2, 1:]

            yc_ind = Math.find_nearest_index(yc_lim_arr, yc_val)
            # lm_lim = self.yc_nan_lmlim[1:, yc_ind]
            if yc_val != 1.0 and (lm < lm_low_lim_arr[yc_ind] or lm > lm_up_lim_arr[yc_ind]):
                raise ValueError('For Yc:{}, star:{} L/M limits are: {}, {}, given lm {}'
                                 .format(yc_val, star_n, "%.2f" % lm_low_lim_arr[yc_ind],
                                         "%.2f" % lm_up_lim_arr[yc_ind], "%.2f" % lm))

            # l, lm = SP_file_work.yc_x__to__y__sp(yc_val, 'l', 'lm', l, self.opal_used, 0)
            return lm

        if yc_val == -1 or yc_val == 'langer':
            return np.log10(10 ** l / 10 ** Physics.l_to_m_langer(l))

        if yc_val == None or yc_val == 'hamman':
            m = self.get_num_par_from_table('m', star_n)
            return np.log10(10 ** l / m)

    def get_star_l_obs_err(self, star_n, yc_assumed):
        '''
        Obsevational Error From Hamman (0.2 - gal, 0.1 - lmc, or from GaiA)
        :param star_n:
        :param yc_assumed:
        :return:
        '''

        l_err1 = l_err2 = None
        l = self.get_llm('l', star_n, yc_assumed)

        if self.metal == 'gal':

            l_err1 = 0.2  # --- OLD correction from Hamman2006
            l_err2 = 0.2
            if self.set_use_gaia:
                l_err1 = self.get_gaia_lum(star_n)[0] - self.get_gaia_lum(star_n)[1]
                l_err2 = self.get_gaia_lum(star_n)[2] - self.get_gaia_lum(star_n)[0]

                print('star: {}, l:{} , l_lower: {}, l_upper: {}'.format(star_n, l, l_err1, l_err2))

        if self.metal == 'lmc':
            l_err1 = 0.1
            l_err2 = 0.1

        if l_err1 == None or l_err2 == None: raise ValueError('Error for l (star:{}) not found'.format(star_n))

        return l - l_err1, l + l_err2

    def get_star_lm_obs_err(self, star_n, yc_assumed):
        # l_err1 = l_err2 = None
        # l = None

        l_err1, l_err2 = self.get_star_l_obs_err(star_n, yc_assumed)
        lm1 = self.l_lm_for_errs(l_err1, star_n, yc_assumed)
        lm2 = self.l_lm_for_errs(l_err2, star_n, yc_assumed)

        return lm1, lm2

        # if self.opal_used.split('/')[-1] == 'table8.data':
        #
        #     l_err1, l_err2 = self.get_star_l_obs_err(star_n, yc_assumed, use_gaia)
        #
        #     # l_err1 = 0.2 # --- OLD correction from Hamman2006
        #     # l_err2 = 0.2
        #     # l_err1 = self.get_gaia_lum(star_n)[0]-self.get_gaia_lum(star_n)[1]
        #     # l_err2 = self.get_gaia_lum(star_n)[2]-self.get_gaia_lum(star_n)[0]
        # if self.opal_used.split('/')[-1] == 'table_x.data':
        #
        #
        #
        #     l_err1 = 0.1
        #     l_err2 = 0.1
        #
        #
        # l = self.get_num_par('l', star_n, yc_assumed)
        # lm1 = self.l_lm_for_errs(l - l_err1, star_n, yc_assumed)
        # lm2 = self.l_lm_for_errs(l + l_err2, star_n, yc_assumed)
        #
        # return lm1, lm2

    def lm_mdot_to_ts_errs(self, lm, mdot, star_n, t_llm_mdot, yc_assumed, lim_t1, lim_t2):

        mdot_arr = t_llm_mdot[1:, 1:]
        i = Math.find_nearest_index(t_llm_mdot[1:, 0], lm)

        if t_llm_mdot[i, 1:].min() < mdot or t_llm_mdot[i, 1:].max() > mdot:
            print('\t__Error: Star:{} | mdot {} NOT in [{}, {}] mdot_row (for lm:{})'
                  .format(star_n, "%.2f" % mdot, "%.2f" % t_llm_mdot[i, 1:].min(), "%.2f" % t_llm_mdot[i, 1:].max(),
                          "%.2f" % lm))

        # if t_llm_mdot[i, 1:].min() > mdot:
        #     print('\t__Warning: Star: {} Using the min. mdot in the row for the error. '.format(star_n))
        #     mdot = mdot_arr[i, :].min()

        xyz = Physics.model_yz_to_xyz(t_llm_mdot[0, 1:], t_llm_mdot[1:, 0], t_llm_mdot[1:, 1:], lm, mdot, star_n,
                                      lim_t1, lim_t2)

        if xyz.any():
            if len(xyz[0, :]) > 1:
                raise ValueError('Multiple coordinates for star: {} | Yc: {}'.format(star_n, yc_assumed))
            else:
                ts = xyz[0, 0]  # xyz[0, :] ALL X coordinates

                # if ts < t_llm_mdot[0, 1:].min():
                #     print('\t\t ts {} < {} ts.min()  '.format(ts, t_llm_mdot[0, 1:].min()))
                #     return t_llm_mdot[0, 1:].min()
                # if ts > t_llm_mdot[0, 1:].max():
                #     print('\t\t ts {} > {} ts.min()  '.format(ts, t_llm_mdot[0, 1:].max()))
                #     return t_llm_mdot[0, 1:].max()
                return ts

        else:

            print('\t__ Error. No (Error) solutions found star: {}, mdot: {}, mdot array:({}, {})'
                  .format(star_n, mdot, mdot_arr[i, :].min(), mdot_arr[i, :].max()))

            if self.set_if_evol_err_out == 't1': return lim_t1
            else: return  lim_t2

            # raise ValueError('No Error solutions found star: {}, mdot: {}, mdot array:({}, {})'
                             # .format(star_n, mdot, mdot_arr[i, :].min(), mdot_arr[i, :].max()))

    def get_star_mdot_obs_err(self, star_n, yc_assumed):

        mdot_err = Files.get_obs_err_mdot(self.metal)

        mdot = self.get_num_par('mdot', star_n)

        if self.metal == 'gal':

            if self.set_use_gaia:
                mdot_err1 = self.get_gaia_mdot(star_n)[0] - self.get_gaia_mdot(star_n)[1]
                mdot_err2 = self.get_gaia_mdot(star_n)[2] - self.get_gaia_mdot(star_n)[0]

                print('star: {}, l:{} , l_lower: {}, l_upper: {}'.format(star_n, mdot, mdot_err1, mdot_err2))

                return self.get_gaia_mdot(star_n)[1], self.get_gaia_mdot(star_n)[2]

        return mdot - mdot_err, mdot + mdot_err

    def get_star_ts_obs_err(self, star_n, t_llm_mdot, yc_assumed, lim_t1, lim_t2):

        lm1, lm2 = self.get_star_lm_obs_err(star_n, yc_assumed)
        # lm = self.get_num_par('lm', star_n, yc_assumed)
        # mdot = self.get_num_par('mdot', star_n, yc_assumed)

        mdot1, mdot2 = self.get_star_mdot_obs_err(star_n, yc_assumed)

        ts1_b = self.lm_mdot_to_ts_errs(lm1, mdot1, star_n, t_llm_mdot, yc_assumed, lim_t1, lim_t2)
        ts2_b = self.lm_mdot_to_ts_errs(lm1, mdot2, star_n, t_llm_mdot, yc_assumed, lim_t1, lim_t2)
        ts1_t = self.lm_mdot_to_ts_errs(lm2, mdot1, star_n, t_llm_mdot, yc_assumed, lim_t1, lim_t2)
        ts2_t = self.lm_mdot_to_ts_errs(lm2, mdot2, star_n, t_llm_mdot, yc_assumed, lim_t1, lim_t2)

        return np.float(ts1_b), np.float(ts2_b), np.float(ts1_t), np.float(ts2_t)

    # --------------------------------------------

    def get_num_par(self, v_n, star_n):
        '''

        :param v_n:
        :param star_n:
        :param yc_val: '-1' for Langer1989 | 'None' for use 'm' | 0.9-to-0 for use of l-m-yc relation
        :return:
        '''

        if v_n == 't_eff' or v_n == 'r_eff':
            if self.set_use_atm_file:
                return self.get_y_from_atmosphere(v_n, star_n)
            else:
                raise IOError('Atmpsphere file is not loaded (optopns) for : {} '.format(self.metal))



        if v_n == 'lm' or v_n == 'l':
            raise NameError('Do not use this for lm or l')
            # return self.get_llm(star_n, yc_val, use_gaia, False)



        return self.get_num_par_from_table(v_n, star_n)

    def get_min_max(self, v_n, yc=None):
        arr = []
        for star_n in self.stars_n:
            arr = np.append(arr, self.get_num_par(v_n, star_n))
        return arr.min(), arr.max()

    def get_min_max_llm(self, v_n, yc=None):
        arr = []
        for star_n in self.stars_n:
            arr = np.append(arr, self.get_llm(v_n, star_n, yc))
        return arr.min(), arr.max()

    def get_xyz_from_yz(self, yc_val, model_n, y_name, z_name, x_1d_arr, y_1d_arr, z_2d_arr, lx1=None, lx2=None):

        if y_name == z_name:
            raise NameError('y_name and z_name are the same : {}'.format(z_name))


        if y_name == 'l' or y_name =='lm':
            star_y = self.get_llm(y_name, model_n, yc_val)
        else:
            star_y = self.get_num_par(y_name, model_n)

        if z_name == 'l' or z_name =='lm':
            star_z = self.get_llm(z_name, model_n, yc_val)
        else:
            star_z = self.get_num_par(z_name, model_n)



        if star_z == None or star_y == None:
            raise ValueError('star_y:{} or star_z:{} not defined'.format(star_y, star_z))

        xyz = Physics.model_yz_to_xyz(x_1d_arr, y_1d_arr, z_2d_arr, star_y, star_z, model_n, lx1, lx2)

        return xyz

    def get_star_class(self, n):
        for i in range(len(self.numers[:, 0])):
            if n == self.numers[i, 0]:
                return self.clses[i]

    def get_class_color(self, n):
        # cls = self.get_star_class(n)
        #
        # if cls == 'WN2' or cls == 'WN3':
        #     return 'v'
        # if cls == 'WN4-s':
        #     return 'o' # circle
        # if cls == 'WN4-w':
        #     return 's' # square
        # if cls == 'WN5-w' or cls == 'WN6-w':
        #     return '1' # tri down
        # if cls == 'WN5-s' or cls == 'WN6-s':
        #     return 'd' #diamond
        # if cls == 'WN7':
        #     return '^'
        # if cls == 'WN8' or cls == 'WN9':
        #     return 'P' #plus filled

        # import re  # for searching the number in 'WN7-e' string, to plot them different colour
        se = re.search(r"\d+(\.\d+)?", self.get_star_class(n))  # this is searching for the niumber
        # cur_type = int(se.group(0))
        color = 'C' + se.group(0)
        return color

    def get_clss_marker(self, n):
        # se = re.search(r"\d+(\.\d+)?", self.get_star_class(n))
        # n_class =  int(se.group(0))
        # se = re.search(r"\d+(\.\d+)?", self.get_star_class(n))  # this is searching for the niumber
        # cur_type = int(se.group(0))
        cls = self.get_star_class(n)

        # --- FOR galactic stars ---
        if cls == 'WN2-w' or cls == 'WN3-w':
            return 'v'
        if cls == 'WN4-s':
            return 'o'  # circle
        if cls == 'WN4-w':
            return 's'  # square
        if cls == 'WN5-w' or cls == 'WN6-w':
            return '1'  # tri down
        if cls == 'WN5-s' or cls == 'WN6-s':
            return 'd'  # diamond
        if cls == 'WN7':
            return '^'
        if cls == 'WN8' or cls == 'WN9':
            return 'P'  # plus filled

        # --- FOR LMC ---

        if cls == 'WN3b' or cls == 'WN3o':
            return 'v'
        if cls == 'WN2b(h)' or cls == 'WN2b':
            return 'o'  # circle
        if cls == 'WN4b/WCE' or cls == 'WN4b(h)':
            return 's'  # square
        if cls == 'WN2':
            return '.'
        if cls == 'WN3' or cls == 'WN3(h)':
            return '*'
        if cls == 'WN4b' or cls == 'WN4o':
            return '^'
        if cls == 'WN4':
            return 'd'  # diamond

        raise NameError('Class {} is not defined'.format(cls))

    def get_min_yc_for_lm(self, star_n):
        '''
        Get min lm for which the surface comp. hasn't changed. (checling the (yc, lm) point in yc_nan_lmlim file
        :param star_n:
        :return:
        '''
        yc = self.yc_nan_lmlim[0, 1:]
        i = 0
        yc1 = yc[i]  # assyming the lowest yc (highly likely sorface change, and therefore change in yc1)

        found = True
        while found:
            if i == len(yc): # this allows to get lm for stars outside WNE region but wwith ZERO evol. error bar
                return yc[-1]

            l = self.get_llm('l', star_n, yc1)

            if l < self.yc_l_lm[1:, 0].min() or l > self.yc_l_lm[1:, 0].max(): # this allows to get ZAMS Yc for models outside the WNE range
                lm = np.log10(10 ** l / 10 ** Physics.l_to_m_langer(l))
                print('___Warning: Star: {} l:{} is beyong l->lm table [{}, {}]. Using Langer Prescription lm:{}'
                      .format(star_n, "%.2f" % l, "%.2f" % self.yc_l_lm[1:, 0].min(),
                              "%.2f" % self.yc_l_lm[1:, 0].max(), "%.2f" % lm))
                return yc[-1]


            l, lm_tmp = Math.get_z_for_yc_and_y(yc[i], self.yc_l_lm, l, 0)
            # getting lm without chacking if lm is within limits, at it is for cycle.

            lm_lim = self.yc_nan_lmlim[1:, 1:]
            lm_lim1 = lm_lim[0, i]
            lm_lim2 = lm_lim[1, i]
            if lm_tmp < lm_lim1 or lm_tmp > lm_lim2:
                i = i + 1
            else:
                return yc[i]

    def get_star_lm_err(self, star_n, yc_assumed):
        '''
        Assuming the same l, but different L -> L/M relations for different Yc, retruns (lm-lm1), (lm-lm2)
        :param star_n:
        :return:
        '''
        yc = self.yc_nan_lmlim[0, 1:]
        yc2 = yc[-1]  # Max yc is always 1.0, - ZAMS (no surface change expected)
        yc1 = self.get_min_yc_for_lm(star_n)

        if yc_assumed < yc1 or yc_assumed > yc2: raise ValueError('Yc[{}, {}], given: {}'.format(yc1, yc2, yc_assumed))

        lm1 = self.get_llm('lm', star_n, yc1)  # Getting lm at different Yc (assiming the same l, but
        lm2 = self.get_llm('lm', star_n, yc2)
        # lm  = self.get_llm('lm', star_n, yc_assumed, use_gaia)

        print('\t__STAR: {} | Evol. Err. L/M for Yc ({} - {}) L/M: ({}, {})'.format(star_n, yc1, yc2, lm1, lm2))

        if lm1 == lm2:
            print('\t__ Star: {} lm1==lm2 - No Evol. Error Bar Found'.format(star_n))
            #raise ValueError('Star: {} lm1==lm2'.format(star_n))

        return lm1, lm2

    def get_star_ts_err(self, star_n, t_llm_mdot, yc_assumed, lim_t1, lim_t2):


        yc = self.yc_nan_lmlim[0, 1:]

        ts = None

        xyz = self.get_xyz_from_yz(yc_assumed, star_n, 'lm', 'mdot',
                                   t_llm_mdot[0, 1:], t_llm_mdot[1:, 0], t_llm_mdot[1:, 1:], lim_t1, lim_t2)
        if xyz.any():
            if len(xyz[0, :]) > 1:
                raise ValueError('Multiple coordinates Star: {} | Yc: {} | xyz: [{}] '.format(star_n, yc_assumed, xyz))
            else:
                ts = xyz[0, 0]  # xyz[0, :] ALL X coordinates

        ts1 = 0
        yc1 = self.get_min_yc_for_lm(star_n)
        xyz1 = self.get_xyz_from_yz(yc1, star_n, 'lm', 'mdot',
                                    t_llm_mdot[0, 1:], t_llm_mdot[1:, 0], t_llm_mdot[1:, 1:], lim_t1, lim_t2)
        if xyz1.any():
            if len(xyz1[0, :]) > 1:
                raise ValueError('Multiple coordinates for star: {} | Yc: {}'.format(star_n, yc_assumed))
            else:
                ts1 = xyz1[0, 0]  # xyz[0, :] ALL X coordinates
        else:
            ts1 = t_llm_mdot[0, -1]

        ts2 = 0
        yc2 = yc[-1]
        xyz2 = self.get_xyz_from_yz(yc2, star_n, 'lm', 'mdot',
                                    t_llm_mdot[0, 1:], t_llm_mdot[1:, 0], t_llm_mdot[1:, 1:], lim_t1, lim_t2)
        if xyz2.any():
            if len(xyz2[0, :]) > 1:
                raise ValueError('Multiple coordinates for star: {} | Yc: {}'.format(star_n, yc_assumed))
            else:
                ts2 = xyz2[0, 0]  # xyz[0, :] ALL X coordinates
        else:
            ts2 = t_llm_mdot[0, 0]

        print('\t__STAR: {} | {} : {} (+{} -{})'.format(star_n, 'Ts', ts, "%.2f" % np.abs(ts - ts1),
                                                        "%.2f" % np.abs(ts - ts2)))

        return ts1, ts2

class PlotObs(Read_Observables):

    def __init__(self, metal, bump, obs_f_name, atm_f_name=None):

        Read_Observables.__init__(self, obs_f_name, metal)

        self.set_clump_used = 4
        self.set_clump_modified = 4

        self.metal        = metal
        self.bump         = bump # for affiliation
        self.set_use_gaia = False
        self.set_clean    = False

        self.set_label_size = 12

        self.set_check_affiliation = False

        self.set_load_yc_l_lm      = True
        self.set_load_yc_nan_lmlim = True
        self.set_use_atm_file      = True
        self.set_atm_file          = atm_f_name
        self.set_check_lm_for_wne  = True

        self.set_patches_or_lines = 'patches'

        self.set_patches_or_lines_alpha = 1.0

        self.set_do_plot_obs_err  = True
        self.set_do_plot_evol_err = True
        self.set_do_plot_line_fit = True

        self.set_if_evol_err_out = 't1'

    # ----------------------------------------------------------------------------



    def plot_t_eff_err_x(self, ax, star_n, x_coord, y_coord, y_bottom, y_up):

        t_eff_err1, t_eff_err2 = self.get_star_atm_obs_err('t_eff', star_n)

        t_eff_coord = [t_eff_err1, t_eff_err1, t_eff_err2, t_eff_err2]
        y_all = [y_bottom, y_up, y_up, y_bottom]

        if self.set_patches_or_lines == 'patches':
            ax.add_patch(patches.Polygon(xy=list(zip(t_eff_coord, y_all)), fill=True,
                                         alpha=self.set_patches_or_lines_alpha,
                                         color=self.get_class_color(star_n)))

        if self.set_patches_or_lines == 'lines':
            ax.plot([t_eff_err1, t_eff_err2], [y_coord, y_coord], '-', color='gray',
                    alpha=self.set_patches_or_lines_alpha)
            ax.plot([x_coord, x_coord], [y_bottom, y_up], '-', color='gray',
                    alpha=self.set_patches_or_lines_alpha)

    def plot_obs_err(self, ax, star_n, v_n_y, v_n_x, yc_val):

        x_coord = self.get_num_par(v_n_x, star_n)
        y_coord = self.get_llm(v_n_y, star_n, yc_val)

        if v_n_y == 'lm':

            lm_err1, lm_err2 = self.get_star_lm_obs_err(star_n, yc_val)  # ERRORS Mdot


            if v_n_x == 'mdot':
                mdot1, mdot2 = self.get_star_mdot_obs_err(star_n, yc_val)
                mdot_coord = [mdot1, mdot2, mdot2, mdot1]
                lm_coord = [lm_err1, lm_err1, lm_err2, lm_err2]

                if self.set_patches_or_lines == 'patches':
                    ax.add_patch(patches.Polygon(xy=list(zip(mdot_coord, lm_coord)), fill=True,
                                                 alpha=self.set_patches_or_lines_alpha,
                                                 color=self.get_class_color(star_n)))

                if self.set_patches_or_lines == 'lines':
                    ax.plot([mdot1, mdot2], [y_coord, y_coord], '-', color='gray',
                            alpha=self.set_patches_or_lines_alpha)
                    ax.plot([x_coord, x_coord], [lm_err1, lm_err2], '-', color='gray',
                            alpha=self.set_patches_or_lines_alpha)


            if v_n_x == 't_eff':
                self.plot_t_eff_err_x(ax, star_n, x_coord, y_coord, lm_err1, lm_err2)

            if v_n_x != 'mdot' and v_n_x != 't_eff':
                ax.plot([x_coord, x_coord], [lm_err1, lm_err2], '-', color=self.get_class_color(star_n))




        if v_n_y == 'l':

            l_err1, l_err2 = self.get_star_l_obs_err(star_n, yc_val)

            if v_n_x == 'mdot':
                mdot1, mdot2 = self.get_star_mdot_obs_err(star_n, yc_val)
                mdot_coord = [mdot1, mdot2, mdot2, mdot1]
                l_coord = [l_err1, l_err1, l_err2, l_err2]
                ax.add_patch(patches.Polygon(xy=list(zip(mdot_coord, l_coord)), fill=True,
                                             alpha=self.set_patches_or_lines_alpha,
                                             color=self.get_class_color(star_n)))
            else:
                ax.plot([x_coord, x_coord], [l_err1, l_err2], '-', color='gray', alpha=self.set_patches_or_lines_alpha)

    def plot_stars(self, ax, star_n, l_or_lm, v_n_x, yc_val):

        x_coord = self.get_num_par(v_n_x, star_n)
        llm_obs = self.get_llm(l_or_lm, star_n, yc_val)

        ax.plot(x_coord, llm_obs, marker=self.get_clss_marker(star_n), markersize='9',
                color=self.get_class_color(star_n), ls='', mec='black')  # plot color dots)))

        if not self.set_clean:
            ax.annotate('{}'.format(int(star_n)), xy=(x_coord, llm_obs),
                        textcoords='data')  # plot numbers of stars

    def plot_3d_stars(self, ax, star_n, l_or_lm, v_n_x, yc_val, z_coord):

        x_coord = self.get_num_par(v_n_x, star_n)
        llm_obs = self.get_llm(l_or_lm, star_n, yc_val)

        # ax.plot(x_coord, z_coord, llm_obs)

        ax.scatter(x_coord, z_coord, llm_obs, marker=self.get_clss_marker(star_n), # markersize='9',
                color=self.get_class_color(star_n)) #  ls='', mec='black'  # plot color dots)))

    def plot_star_lines(self, ax, star_n, l_or_lm, v_n_x, yc_val, y_slices, separator):

        xs = []
        ys = []
        zs = []

        for s in range(len(y_slices)):

            x_coord = self.get_num_par(v_n_x, star_n)
            llm_obs = self.get_llm(l_or_lm, star_n, yc_val)
            z_coord = s * separator

            xs = np.append(xs, x_coord)
            ys = np.append(ys, llm_obs)
            zs = np.append(zs, z_coord)

        ax.plot(xs, zs, ys, '-', color=self.get_class_color(star_n))

    def plot_evol_err(self, ax, star_n, l_or_lm, v_n_x, yc_assumed, color='black'):
        if l_or_lm == 'lm':
            llm1, llm2 = self.get_star_lm_err(star_n, yc_assumed)
            # obs_cls.get_star_llm_evol_err(star_n, l_or_lm, yc_val, 1.0, 0.1)                  # ERRORS L/LM
            # mdot1, mdot2 = obs_cls.get_star_mdot_err(star_n, l_or_lm, yc_val, 1.0, 0.1, 'nugis')           # ERRORS Mdot

            x_coord = self.get_num_par(v_n_x, star_n)

            ax.plot([x_coord, x_coord], [llm1, llm2], '-', color=color)
            ax.plot([x_coord, x_coord], [llm1, llm2], '.', color=color)


    # def plot_one_for_all_x_llm(self, ax, star_n, l_or_lm, v_n_x, yc_val, obs_err=True, evol_err=True):
    def plot_obs_all_x_llm(self, ax, l_or_lm, v_n_x, yc_val, return_ax=False, collect_legend=True):

        classes = []
        classes.append('dum')
        x_coord = []
        llm_obs = []

        # from Phys_Math_Labels import Opt_Depth_Analythis
        # use_gaia = False

        # def plot_one_star()

        # self.set_check_affiliation

        for star_n in self.stars_n:

            i = -1
            x_coord = np.append(x_coord, self.get_num_par(v_n_x, star_n))
            llm_obs = np.append(llm_obs, self.get_llm(l_or_lm, star_n, yc_val))
            # llm_obs = np.append(llm_obs, obs_cls.get_num_par(l_or_lm, star_n, yc_val, use_gaia))

            if self.set_do_plot_obs_err: self.plot_obs_err(ax, star_n, l_or_lm, v_n_x, yc_val)

            if self.set_do_plot_evol_err: self.plot_evol_err(ax, star_n, l_or_lm, v_n_x, yc_val, self.get_class_color(star_n))

            self.plot_stars(ax, star_n, l_or_lm, v_n_x, yc_val)

            if self.get_star_class(star_n) not in classes:
                if collect_legend:
                    plt.plot(x_coord[i], llm_obs[i], marker=self.get_clss_marker(star_n), markersize='9',
                             color=self.get_class_color(star_n), ls='', mec='black',
                             label='{}'.format(self.get_star_class(star_n)))  # plot color dots)))
                    classes.append(self.get_star_class(star_n))
                else:
                    plt.plot(x_coord[i], llm_obs[i], marker=self.get_clss_marker(star_n), markersize='9',
                             color=self.get_class_color(star_n), ls='', mec='black',
                            )  # plot color dots)))
                    classes.append(self.get_star_class(star_n))


        print('\t__PLOT: total stars: {}'.format(len(self.stars_n)))
        print(len(x_coord), len(llm_obs))

        # fit = np.polyfit(mdot_obs, llm_obs, 1)  # fit = set of coeddicients (highest first)
        # f = np.poly1d(fit)
        # fit_x_coord = np.mgrid[(mdot_obs.min() - 1):(mdot_obs.max() + 1):1000j]

        if self.set_do_plot_line_fit:
            mdot_grid = np.mgrid[(x_coord.min() - 1):(x_coord.max() + 1):100j]
            print('OBSERVATIONS FIT')
            x_fit, y_fit = Math.fit_polynomial(x_coord, llm_obs, 1, 100, mdot_grid)
            ax.plot(x_fit, y_fit, '--', color='black')

        # mdot_grid = np.mgrid[(x_coord.min() - 1):(x_coord.max() + 1):100j]
        # x_coord__, y_coord__ = Math.fit_polynomial(x_coord, llm_obs, 1, 100, mdot_grid)
        # ax.plot(x_coord__, y_coord__, '-.', color='blue')

        # min_mdot, max_mdot = self.get_min_max('mdot')
        # min_llm, max_llm = self.get_min_max_llm(l_or_lm, yc_val)

        # ax.set_xlim(min_mdot - 0.2, max_mdot + 0.2)
        # ax.set_ylim(min_llm - 0.05, max_llm + 0.05)

        # ax.set_ylabel(Labels.lbls(l_or_lm))
        # ax.set_xlabel(Labels.lbls('mdot'))
        # ax.grid(which='major', alpha=0.2)
        # ax.legend(bbox_to_anchor=(1, 1), loc='upper right', ncol=1)

        print('Yc:{}'.format(yc_val))

        if not self.set_clean:
            ax.text(0.9, 0.9, 'Yc:{}'.format(yc_val), style='italic',
                    bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center',
                    verticalalignment='center', transform=ax.transAxes)
            if self.set_use_gaia:
                ax.text(0.9, 0.75, 'GAIA', style='italic',
                        bbox={'facecolor': 'blue', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center',
                        verticalalignment='center', transform=ax.transAxes)


        if len(self.stars_n) != len(x_coord): raise ValueError('not all stars plotted WHY?')
        if return_ax: return ax

        return self.stars_n, x_coord, llm_obs

    def plot_3d_obs_all_x_llm(self, ax, v_n_x, l_or_lm, yc_val, y_slices, separator=100):

        classes = []
        classes.append('dum')
        x_coord = []
        llm_obs = []

        # from Phys_Math_Labels import Opt_Depth_Analythis
        # use_gaia = False

        # def plot_one_star()

        # self.set_check_affiliation

        for s in range(len(y_slices)):

            if len(y_slices) == 1:
                z_coord = 1 * separator
            else:
                z_coord = s * separator

            for star_n in self.stars_n:

                i = -1
                x_coord = np.append(x_coord, self.get_num_par(v_n_x, star_n))
                llm_obs = np.append(llm_obs, self.get_llm(l_or_lm, star_n, yc_val))
                # llm_obs = np.append(llm_obs, obs_cls.get_num_par(l_or_lm, star_n, yc_val, use_gaia))


                self.plot_3d_stars(ax, star_n, l_or_lm, v_n_x, yc_val, z_coord)

                # self.plot_star_lines(ax, star_n, l_or_lm, v_n_x, yc_val, y_slices, separator)

                if self.get_star_class(star_n) not in classes:
                    ax.scatter(x_coord[i], z_coord, llm_obs[i], marker=self.get_clss_marker(star_n), #markersize='9',
                             color=self.get_class_color(star_n),
                               # ls='', mec='black',
                             label='{}'.format(self.get_star_class(star_n)))  # plot color dots)))
                    classes.append(self.get_star_class(star_n))




    def plot_obs_err_ts_y(self, ax, star_n, star_x_coord, star_y_coord, yc_val, v_n_y, t_llm_mdot, lim_t1=None, lim_t2=None):

        if v_n_y == 'lm':
            lm_err1, lm_err2 = self.get_star_lm_obs_err(star_n, yc_val)
            ts1_b, ts2_b, ts1_t, ts2_t = self.get_star_ts_obs_err(star_n, t_llm_mdot, yc_val, lim_t1, lim_t2)
            ts_coord = [ts1_b, ts2_b, ts2_t, ts1_t]
            lm_coord = [lm_err1, lm_err1, lm_err2, lm_err2]
            ax.add_patch(patches.Polygon(xy=list(zip(ts_coord, lm_coord)), fill=True, alpha=.7,
                                         color=self.get_class_color(star_n)))
        if v_n_y == 'l':
            ts1_b, ts2_b, ts1_t, ts2_t = self.get_star_ts_obs_err(star_n, t_llm_mdot, yc_val, lim_t1,
                                                                  lim_t2)

            ax.plot([np.array([ts1_b, ts2_b, ts1_t, ts2_t]).min(),
                     np.array([ts1_b, ts2_b, ts1_t, ts2_t]).max()],
                    [star_y_coord, star_y_coord], '-', color='gray')


        if v_n_y == 't_eff' or v_n_y == 'r_eff':

            t_eff_err1, t_eff_err2 = self.get_star_atm_obs_err(v_n_y, star_n)
            ts1_b, ts2_b, ts1_t, ts2_t = self.get_star_ts_obs_err(star_n, t_llm_mdot, yc_val, lim_t1, lim_t2)
            # ts_coord = [ts1_b, ts2_b, ts2_t, ts1_t]
            ts_minmax = [np.array([ts1_b, ts2_b, ts2_t, ts1_t]).min(),
                         np.array([ts1_b, ts2_b, ts2_t, ts1_t]).max(),]

            ts_coord =    [ts_minmax[0], ts_minmax[1], ts_minmax[1], ts_minmax[0]]
            t_eff_coord = [t_eff_err1, t_eff_err1, t_eff_err2, t_eff_err2]

            if self.set_patches_or_lines == 'patches':
                ax.add_patch(patches.Polygon(xy=list(zip(ts_coord, t_eff_coord)), fill=True,
                                             alpha=self.set_patches_or_lines_alpha,
                                             color=self.get_class_color(star_n)))
            if self.set_patches_or_lines == 'lines':

                ax.plot([ts_minmax[0], ts_minmax[1]], [star_y_coord, star_y_coord], '-', color='gray',
                        alpha = self.set_patches_or_lines_alpha)
                ax.plot([star_x_coord, star_x_coord], [t_eff_err1, t_eff_err2],     '-', color='gray',
                        alpha = self.set_patches_or_lines_alpha)

                # ax.plot([t_eff_err1[0], t_eff_err1[1]], [ts_minmax[0], ts_minmax[1]])
                # ax.plot([t_eff_err1[0], t_eff_err1[1]], [t_eff_err1, t_eff_err2])

                # ax.plot([np.array([ts1_b, ts2_b, ts1_t, ts2_t]).min(),
                #         np.array([ts1_b, ts2_b, ts1_t, ts2_t]).max()],
                #         [y_val, y_val], '-', color='gray')

    def plot_all_obs_ts_y_mdot(self, ax, v_n_y, t_llm_mdot, l_or_lm, lim_t1=None, lim_t2=None, show_legend=True):

        if lim_t1 == None: lim_t1 = t_llm_mdot[0, 1:].min()
        if lim_t2 == None: lim_t2 = t_llm_mdot[0, 1:].max()

        yc_val = t_llm_mdot[0, 0]  #

        classes = []
        classes.append('dum')
        x = []
        y = []

        if self.set_check_affiliation:
            list_of_stars = Affiliation.get_list(self.metal, self.bump)
        else:
            list_of_stars = self.stars_n

        for star_n in list_of_stars:
            xyz = self.get_xyz_from_yz(yc_val, star_n, l_or_lm, 'mdot',
                                          t_llm_mdot[0, 1:], t_llm_mdot[1:, 0], t_llm_mdot[1:, 1:], lim_t1, lim_t2)

            if xyz.any():

                # print('Star {}, {} range: ({}, {})'.format(star_n,l_or_lm, llm1, llm2))

                for i in range(len(xyz[0, :])):


                    if v_n_y != 'l' and v_n_y != 'lm':
                        val = self.get_num_par(v_n_y, star_n)
                        xyz[1, i] = val

                    x = np.append(x, xyz[0, i])
                    y = np.append(y, xyz[1, i])

                    # --- ----------------------------- ERRORS ---------------------
                    if self.set_do_plot_obs_err:

                        self.plot_obs_err_ts_y(ax, star_n, xyz[0,i],xyz[1,i],yc_val,v_n_y,t_llm_mdot,lim_t1,lim_t2)

                        # if v_n_y == 'lm':
                        #     lm_err1, lm_err2 = self.get_star_lm_obs_err(star_n, yc_val)
                        #     ts1_b, ts2_b, ts1_t, ts2_t = self.get_star_ts_obs_err(star_n, t_llm_mdot, yc_val, lim_t1, lim_t2)
                        #     ts_coord = [ts1_b, ts2_b, ts2_t, ts1_t]
                        #     lm_coord = [lm_err1, lm_err1, lm_err2, lm_err2]
                        #     ax.add_patch(patches.Polygon(xy=list(zip(ts_coord, lm_coord)), fill=True, alpha=.7,
                        #                                  color=self.get_class_color(star_n)))
                        # else:
                        #     ts1_b, ts2_b, ts1_t, ts2_t = self.get_star_ts_obs_err(star_n, t_llm_mdot, yc_val, lim_t1,
                        #                                                       lim_t2)
                        #
                        #     ax.plot([np.array([ts1_b, ts2_b, ts1_t, ts2_t]).min(),
                        #              np.array([ts1_b, ts2_b, ts1_t, ts2_t]).max()],
                        #              [xyz[1, i], xyz[1, i]], '-', color='gray')

                    if self.set_do_plot_evol_err:
                        if l_or_lm == 'lm':
                            lm1, lm2 = self.get_star_lm_err(star_n, yc_val)
                            ts1, ts2 = self.get_star_ts_err(star_n, t_llm_mdot, yc_val, lim_t1, lim_t2)
                            ax.plot([ts1, ts2], [lm1, lm2], '-', color='white')

                        else:
                            ts1, ts2 = self.get_star_ts_err(star_n, t_llm_mdot, yc_val, lim_t1, lim_t2)
                            ax.plot([ts1, ts2], [xyz[1, i], xyz[1, i]], '-', color='black')

                    # ----------------------------------DATA POINTS ----------------
                    ax.plot(xyz[0, i], xyz[1, i], marker=self.get_clss_marker(star_n), markersize='9',
                            color=self.get_class_color(star_n), ls='', mec='black')  # plot color dots)))
                    if not self.set_clean:
                        ax.annotate(int(star_n), xy=(xyz[0, i], xyz[1, i]),
                                    textcoords='data')  # plot numbers of stars

                    if self.get_star_class(star_n) not in classes:
                        ax.plot(xyz[0, i], xyz[1, i], marker=self.get_clss_marker(star_n), markersize='9',
                                color=self.get_class_color(star_n), mec='black', ls='',
                                label='{}'.format(self.get_star_class(star_n)))  # plot color dots)))
                        classes.append(self.get_star_class(star_n))




                        # color=obs_cls.get_class_color(star_n))

                        # ax.plot([xyz[0, i], xyz[0, i]], [lm1, lm2], '-',
                        #         color=obs_cls.get_class_color(star_n))
                        # ax.plot([ts1, ts2], [xyz[1, i], xyz[1, i]], '-',
                        #         color=obs_cls.get_class_color(star_n))

                        # ax.add_patch(patches.Rectangle((xyz[0, i] - ts1, xyz[1, i] - lm1), ts2 + ts1, lm2 + lm1,
                        #                                alpha=.3, color=obs_cls.get_class_color(star_n)))

                        # ax.add_patch(patches.Rectangle((xyz[0, i] - ts1, xyz[1, i] - lm1), ts2 + ts1, lm2 + lm1,
                        #                                alpha=.3, color=obs_cls.get_class_color(star_n)))

                        # ax.plot([xyz[0, i] - ts1, xyz[1, i] - lm1], [xyz[0, i]+ts2, xyz[1, i] + lm2], '-', color=obs_cls.get_class_color(star_n))
                        # ax.plot([xyz[0, i] - ts1, xyz[0, i]+ts2], [xyz[1, i] - lm1, xyz[1, i] + lm2], '-', color=obs_cls.get_class_color(star_n))

                        # ax.errorbar(xyz[0, i], xyz[1, i], yerr=[[lm1], [lm2]], fmt='--.', color = obs_cls.get_class_color(star_n))
                        # ax.errorbar(xyz[0, i], xyz[1, i], xerr=[[ts1], [ts2]], fmt='--.', color=obs_cls.get_class_color(star_n))



        if self.set_do_plot_line_fit:
            print('FIT TO OBSERVATIONS')

            x_fit, y_fit = Math.fit_polynomial(x, y, 1, 500)
            ax.plot(x_fit, y_fit, '-.', color='blue')

            # fit = np.polyfit(x, y, 1)  # fit = set of coeddicients (highest first)
            # f = np.poly1d(fit)
            # fit_x_coord = np.mgrid[(t_llm_mdot[0, 1:].min()):(t_llm_mdot[0, 1:].max()):1000j]
            # ax.plot(fit_x_coord, f(fit_x_coord), '-.', color='blue')

        ax.set_xlim(t_llm_mdot[0, 1:].min(), t_llm_mdot[0, 1:].max())
        ax.set_ylim(t_llm_mdot[1:, 0].min(), t_llm_mdot[1:, 0].max())
        # ax.text(0.9, 0.9,'Yc:{}'.format(yc_val), style='italic',
        #         bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)

        # ax.text(x.max(), y.max(), 'Yc:{}'.format(yc_val), style='italic',
        #         bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10})
        if show_legend:
            ax.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)

        return ax

    @staticmethod
    def plot_obs_mdot_llm(ax, obs_cls, l_or_lm, yc_val, clean=False):
        '''

        :param ax:
        :param obs_cls:
        :param l_or_lm:
        :param yc_val:
        :param yc1:
        :param yc2:
        :return:
        '''
        classes = []
        classes.append('dum')
        mdot_obs = []
        llm_obs = []

        # from Phys_Math_Labels import Opt_Depth_Analythis

        for star_n in obs_cls.stars_n:
            i = -1
            mdot_obs = np.append(mdot_obs, obs_cls.get_num_par('mdot', star_n))
            llm_obs = np.append(llm_obs, obs_cls.get_num_par(l_or_lm, star_n, yc_val))
            eta = obs_cls.get_num_par('eta', star_n)

            lm_err1, lm_err2 = obs_cls.get_star_lm_obs_err(star_n, yc_val)
            mdot1, mdot2 = obs_cls.get_star_mdot_obs_err(star_n, yc_val)

            mdot_coord = [mdot1, mdot2, mdot2, mdot1]
            lm_coord = [lm_err1, lm_err1, lm_err2, lm_err2]
            ax.add_patch(patches.Polygon(xy=list(zip(mdot_coord, lm_coord)), fill=True, alpha=.4,
                                         color=obs_cls.get_class_color(star_n)))

            if l_or_lm == 'lm':
                llm1, llm2 = obs_cls.get_star_lm_err(star_n, yc_val)
                # obs_cls.get_star_llm_evol_err(star_n, l_or_lm, yc_val, 1.0, 0.1)                  # ERRORS L/LM
                # mdot1, mdot2 = obs_cls.get_star_mdot_err(star_n, l_or_lm, yc_val, 1.0, 0.1, 'nugis')           # ERRORS Mdot
                mdot = obs_cls.get_num_par('mdot', star_n)
                ax.plot([mdot, mdot], [llm1, llm2], '-', color='white')
                # color=obs_cls.get_class_color(star_n)

                # ax.errorbar(mdot_obs[i], llm_obs[i], yerr=[[llm1],  [llm2]], fmt='--.', color=obs_cls.get_class_color(star_n))

            ax.plot(mdot_obs[i], llm_obs[i], marker=obs_cls.get_clss_marker(star_n), markersize='9',
                    color=obs_cls.get_class_color(star_n), ls='', mec='black')  # plot color dots)))
            if not clean:
                ax.annotate('{}'.format(int(star_n)), xy=(mdot_obs[i], llm_obs[i]),
                            textcoords='data')  # plot numbers of stars

            # t = obs_cls.get_num_par('t', star_n)
            # ax.annotate('{}'.format("%.2f" % t), xy=(mdot_obs[i], llm_obs[i]), textcoords='data')  # plot numbers of stars

            # v_inf = obs_cls.get_num_par('v_inf', star_n)
            # tau_cl = Opt_Depth_Analythis(30, v_inf, 1., 1., mdot_obs[i], 0.20)
            # tau = tau_cl.anal_eq_b1(1.)
            # # # # ax.annotate(str(int(tau)), xy=(mdot_obs[i], llm_obs[i]), textcoords='data')  # plo
            # ax.annotate('{} {}'.format(str(int(tau)), eta), xy=(mdot_obs[i], llm_obs[i]),
            #             textcoords='data')  # plot numbers of stars

            if obs_cls.get_star_class(star_n) not in classes:
                plt.plot(mdot_obs[i], llm_obs[i], marker=obs_cls.get_clss_marker(star_n), markersize='9',
                         color=obs_cls.get_class_color(star_n), ls='', mec='black',
                         label='{}'.format(obs_cls.get_star_class(star_n)))  # plot color dots)))
                classes.append(obs_cls.get_star_class(star_n))

        print('\t__PLOT: total stars: {}'.format(len(obs_cls.stars_n)))
        print(len(mdot_obs), len(llm_obs))

        # fit = np.polyfit(mdot_obs, llm_obs, 1)  # fit = set of coeddicients (highest first)
        # f = np.poly1d(fit)
        # fit_x_coord = np.mgrid[(mdot_obs.min() - 1):(mdot_obs.max() + 1):1000j]

        mdot_grid = np.mgrid[(mdot_obs.min() - 1):(mdot_obs.max() + 1):100j]
        x_coord, y_coord = Math.fit_polynomial(mdot_obs, llm_obs, 1, 100, mdot_grid)
        ax.plot(x_coord, y_coord, '-.', color='blue')

        min_mdot, max_mdot = obs_cls.get_min_max('mdot')
        min_llm, max_llm = obs_cls.get_min_max(l_or_lm, yc_val)

        ax.set_xlim(min_mdot - 0.2, max_mdot + 0.2)
        ax.set_ylim(min_llm - 0.05, max_llm + 0.05)

        ax.set_ylabel(Labels.lbls(l_or_lm))
        ax.set_xlabel(Labels.lbls('mdot'))
        ax.grid(which='major', alpha=0.2)
        ax.legend(bbox_to_anchor=(1, 1), loc='upper right', ncol=1)

        print('Yc:{}'.format(yc_val))

        if not clean:
            ax.text(0.9, 0.9, 'Yc:{}'.format(yc_val), style='italic',
                    bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center',
                    verticalalignment='center', transform=ax.transAxes)

        return ax
        # ax.text(min_mdot, max_llm, 'Yc:{}'.format(yc_val), style='italic',
        #         bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10})

        # l_grid = np.mgrid[5.2:6.:100j]
        # ax.plot(Physics.l_mdot_prescriptions(l_grid, ), l_grid, '-.', color='orange', label='Nugis & Lamers 2000')
        #
        # ax.plot(Physics.yoon(l_grid, 10 ** 0.02), l_grid, '-.', color='green', label='Yoon 2017')

    def plot_obs_t_llm_mdot_int(self, ax, t_llm_mdot, l_or_lm, lim_t1=None, lim_t2=None, show_legend=True):

        if lim_t1 == None: lim_t1 = t_llm_mdot[0, 1:].min()
        if lim_t2 == None: lim_t2 = t_llm_mdot[0, 1:].max()

        yc_val = t_llm_mdot[0, 0]  #

        classes = []
        classes.append('dum')
        x = []
        y = []

        # self.set_check_affiliation
        if self.set_check_affiliation:
            list_of_stars = Affiliation.get_list(self.metal, self.bump)
        else:
            list_of_stars = self.stars_n


        for star_n in list_of_stars:
            xyz = self.get_xyz_from_yz(yc_val, star_n, l_or_lm, 'mdot',
                                          t_llm_mdot[0, 1:], t_llm_mdot[1:, 0], t_llm_mdot[1:, 1:], lim_t1, lim_t2)

            if xyz.any():
                x = np.append(x, xyz[0, 0])
                y = np.append(y, xyz[1, 0])

                # print('Star {}, {} range: ({}, {})'.format(star_n,l_or_lm, llm1, llm2))

                for i in range(len(xyz[0, :])):

                    # -------------------------OBSERVABLE ERRORS FOR L and Mdot ----------------------------------------

                    if self.set_do_plot_obs_err:
                        lm_err1, lm_err2 = self.get_star_lm_obs_err(star_n, yc_val)
                        ts1_b, ts2_b, ts1_t, ts2_t = self.get_star_ts_obs_err(star_n, t_llm_mdot, yc_val, lim_t1,
                                                                              lim_t2)
                        ts_coord = [ts1_b, ts2_b, ts2_t, ts1_t]
                        lm_coord = [lm_err1, lm_err1, lm_err2, lm_err2]

                        if self.set_patches_or_lines == 'patches':
                            ax.add_patch(patches.Polygon(xy=list(zip(ts_coord, lm_coord)), fill=True,
                                                         alpha=self.set_patches_or_lines_alpha,
                                                         color=self.get_class_color(star_n)))

                        if self.set_patches_or_lines == 'lines':
                            ax.plot([ts1_b, ts2_t], [xyz[1, i], xyz[1, i]], '-', color='gray',
                                    alpha=self.set_patches_or_lines_alpha)
                            ax.plot([xyz[0, i], xyz[0, i]], [lm_err1, lm_err2], '-', color='gray',
                                    alpha=self.set_patches_or_lines_alpha)

                        if self.set_patches_or_lines == 'lines2':
                            ax.plot([ts1_b, ts2_t], [lm_err1, lm_err2], '-', color='gray',
                                    alpha=self.set_patches_or_lines_alpha)
                            ax.plot([ts2_b, ts1_t], [lm_err1, lm_err2], '-', color='gray',
                                    alpha=self.set_patches_or_lines_alpha)

                    if self.set_do_plot_evol_err:
                        if l_or_lm == 'lm':
                            lm1, lm2 = self.get_star_lm_err(star_n, yc_val)
                            ts1, ts2 = self.get_star_ts_err(star_n, t_llm_mdot, yc_val, lim_t1, lim_t2)
                            ax.plot([ts1, ts2], [lm1, lm2], '-', color='black')
                            ax.plot([ts1], [lm1], '.', color='black')
                        # color=obs_cls.get_class_color(star_n))

                        # ax.plot([xyz[0, i], xyz[0, i]], [lm1, lm2], '-',
                        #         color=obs_cls.get_class_color(star_n))
                        # ax.plot([ts1, ts2], [xyz[1, i], xyz[1, i]], '-',
                        #         color=obs_cls.get_class_color(star_n))

                        # ax.add_patch(patches.Rectangle((xyz[0, i] - ts1, xyz[1, i] - lm1), ts2 + ts1, lm2 + lm1,
                        #                                alpha=.3, color=obs_cls.get_class_color(star_n)))

                        # ax.add_patch(patches.Rectangle((xyz[0, i] - ts1, xyz[1, i] - lm1), ts2 + ts1, lm2 + lm1,
                        #                                alpha=.3, color=obs_cls.get_class_color(star_n)))

                        # ax.plot([xyz[0, i] - ts1, xyz[1, i] - lm1], [xyz[0, i]+ts2, xyz[1, i] + lm2], '-', color=obs_cls.get_class_color(star_n))
                        # ax.plot([xyz[0, i] - ts1, xyz[0, i]+ts2], [xyz[1, i] - lm1, xyz[1, i] + lm2], '-', color=obs_cls.get_class_color(star_n))

                        # ax.errorbar(xyz[0, i], xyz[1, i], yerr=[[lm1], [lm2]], fmt='--.', color = obs_cls.get_class_color(star_n))
                        # ax.errorbar(xyz[0, i], xyz[1, i], xerr=[[ts1], [ts2]], fmt='--.', color=obs_cls.get_class_color(star_n))

                    # -------------------------OBSERVABLE STARS --------------------------------------------------------

                    ax.plot(xyz[0, i], xyz[1, i], marker=self.get_clss_marker(star_n), markersize='9',
                            color=self.get_class_color(star_n), ls='', mec='black')  # plot color dots)))
                    if not self.set_clean:
                        ax.annotate(int(star_n), xy=(xyz[0, i], xyz[1, i]),
                                    textcoords='data', fontsize=self.set_label_size)  # plot numbers of stars

                    if self.get_star_class(star_n) not in classes:
                        ax.plot(xyz[0, i], xyz[1, i], marker=self.get_clss_marker(star_n), markersize='9',
                                color=self.get_class_color(star_n), mec='black', ls='',
                                label='{}'.format(self.get_star_class(star_n)))  # plot color dots)))
                        classes.append(self.get_star_class(star_n))




        if self.set_do_plot_line_fit:

            print('LINEAR FIT TO OBSERVATIONS')
            x_grid, y_grid = Math.fit_polynomial(x, y, 1, 100)
            ax.plot(x_grid, y_grid, '-.', color='blue')
            # fit = np.polyfit(x, y, 1)  # fit = set of coeddicients (highest first)
            # f = np.poly1d(fit)
            # fit_x_coord = np.mgrid[(t_llm_mdot[0, 1:].min()):(t_llm_mdot[0, 1:].max()):1000j]
            # ax.plot(fit_x_coord, f(fit_x_coord), '-.', color='blue')



        ax.set_xlim(t_llm_mdot[0, 1:].min(), t_llm_mdot[0, 1:].max())
        ax.set_ylim(t_llm_mdot[1:, 0].min(), t_llm_mdot[1:, 0].max())
        # ax.text(0.9, 0.9,'Yc:{}'.format(yc_val), style='italic',
        #         bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)

        # ax.text(x.max(), y.max(), 'Yc:{}'.format(yc_val), style='italic',
        #         bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10})
        if show_legend:
            ax.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1, fontsize=self.set_label_size)

        return ax

    def plot_obs_l_l_comparison(self, ax):

        classes = []
        classes.append('dum')
        x_coord = []
        y_coord = []

        l_x = 'lgaia'
        l_y = 'l'
        yc_val = 1.0

        ax.set_xticks(np.array([4.5, 5.0, 5.5, 6.0]))
        ax.set_yticks(np.array([4.5, 5.0, 5.5, 6.0]))

        ax.plot([0, 10], [0, 10], '--', color='black')

        # from Phys_Math_Labels import Opt_Depth_Analythis   lgaia llgaia ulgaia
        # use_gaia = False
        for star_n in self.stars_n:
            i = -1
            x_coord = np.append(x_coord, self.get_num_par(l_x, star_n))
            y_coord = np.append(y_coord, self.get_llm(l_y, star_n, yc_val))


            self.plot_obs_err(ax, star_n, l_y, l_x, yc_val)

            right_l = self.get_num_par('llgaia', star_n)
            left_l = self.get_num_par('ulgaia', star_n)

            ax.plot([right_l, left_l], [y_coord[i], y_coord[i]], '-',
                    color='gray', alpha=self.set_patches_or_lines_alpha)

            # print('{} {}'.format(right_l, left_l))

            self.plot_stars(ax, star_n, l_y, l_x, yc_val)


            if self.get_star_class(star_n) not in classes:
                plt.plot(x_coord[i], y_coord[i], marker=self.get_clss_marker(star_n), markersize='9',
                         color=self.get_class_color(star_n), ls='', mec='black',
                         label='{}'.format(self.get_star_class(star_n)))  # plot color dots)))
                classes.append(self.get_star_class(star_n))




        print('\t__PLOT: total stars: {}'.format(len(self.stars_n)))
        print(len(x_coord), len(y_coord))

        # fit = np.polyfit(mdot_obs, y_coord, 1)  # fit = set of coeddicients (highest first)
        # f = np.poly1d(fit)
        # fit_x_coord = np.mgrid[(mdot_obs.min() - 1):(mdot_obs.max() + 1):1000j]

        if self.set_do_plot_line_fit:
            mdot_grid = np.mgrid[(x_coord.min() - 1):(x_coord.max() + 1):100j]
            print('OBSERVATIONS FIT')
            x_fit, y_fit = Math.fit_polynomial(x_coord, y_coord, 1, 100, mdot_grid)
            ax.plot(x_fit, y_fit, '--', color='black')

        # mdot_grid = np.mgrid[(x_coord.min() - 1):(x_coord.max() + 1):100j]
        # x_coord__, y_coord__ = Math.fit_polynomial(x_coord, y_coord, 1, 100, mdot_grid)
        # ax.plot(x_coord__, y_coord__, '-.', color='blue')

        min_mdot, max_mdot = self.get_min_max('mdot')
        min_llm, max_llm = self.get_min_max_llm(l_y, yc_val)

        # ax.set_xlim(min_mdot - 0.2, max_mdot + 0.2)
        # ax.set_ylim(min_llm - 0.05, max_llm + 0.05)

        ax.set_ylabel(Labels.lbls(l_y))
        ax.set_xlabel(Labels.lbls('mdot'))

        ax.legend(bbox_to_anchor=(1, 1), loc='upper right', ncol=1)

        print('Yc:{}'.format(yc_val))

        if not self.set_clean:
            ax.text(0.9, 0.9, 'Yc:{}'.format(yc_val), style='italic',
                    bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center',
                    verticalalignment='center', transform=ax.transAxes)
            if self.set_use_gaia:
                ax.text(0.9, 0.75, 'GAIA', style='italic',
                        bbox={'facecolor': 'blue', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center',
                        verticalalignment='center', transform=ax.transAxes)

        return ax

# ========================================================| BACKGROUND |===============================================#

class PlotBackground2:

    def __init__(self):

        self.set_alpha = 1.0
        self.set_label_sise=12
        self.set_clean = False
        self.set_rotate_labels=295
        self.set_show_contours=True
        self.set_contour_fmt = 2.2
        self.set_show_colorbar= True

        self.set_auto_limits = True

    @staticmethod
    def plot_color_table(table, v_n_x, v_n_y, v_n_z, opal_used, bump, label=None, fsz=12,count_angle=0):

        plt.figure()
        ax = plt.subplot(111)

        if label != None:
            ax.text(0.8, 0.1, label, style='italic',
                    bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center',
                    verticalalignment='center', transform=ax.transAxes)

            # print('TEXT')
            # plt.text(table[0, 1:].min(), table[1:, 0].min(), label, style='italic')
            # bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10}
            # plt.text(2, 0.65, r'$\cos(2 \pi t) \exp(-t)$')

        # ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(table[0, 1:].min(), table[0, 1:].max())
        ax.set_ylim(table[1:, 0].min(), table[1:, 0].max())
        ax.set_ylabel(Labels.lbls(v_n_y), fontsize=fsz)
        ax.set_xlabel(Labels.lbls(v_n_x), fontsize=fsz)

        levels = Levels.get_levels(v_n_z, opal_used, bump)

        contour_filled = plt.contourf(table[0, 1:], table[1:, 0], table[1:, 1:], levels, cmap=plt.get_cmap('RdYlBu_r'),
                                      alpha=1.0, fontsize=fsz)


        clb = plt.colorbar(contour_filled)  # orientation='horizontal', :)
        clb.ax.set_title(Labels.lbls(v_n_z), fontsize=fsz)
        clb.ax.tick_params(labelsize=fsz)


        # ax.colorbar(contour_filled, label=Labels.lbls(v_n_z))
        contour = plt.contour(table[0, 1:], table[1:, 0], table[1:, 1:], levels, colors='k')
        labs = ax.clabel(contour, colors='k', fmt='%2.2f', inline=True, fontsize=fsz)
        for lab in labs:
            lab.set_rotation(count_angle)  # 295
        # contour_filled = plt.contourf(table[0, 1:], table[1:, 0], table[1:, 1:], levels, cmap=plt.get_cmap('RdYlBu_r'))
        # clb = plt.colorbar(contour_filled)
        # clb.ax.set_title(Labels.lbls(v_n_z), fontsize=fsz)
        # contour = plt.contour(table[0, 1:], table[1:, 0], table[1:, 1:], levels, colors='k')
        # # plt.clabel(contour, colors='k', fmt='%2.2f', fontsize=fsz)
        # # plt.title('SONIC HR DIAGRAM')

        ax.minorticks_on()

        # ax.invert_xaxis()

        plt.xticks(fontsize=fsz)
        plt.yticks(fontsize=fsz)

        # plt.ylabel(l_or_lm)
        # plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
        # plt.savefig(name)
        plt.show()


    def plot_color_background(self, ax, table, v_n_x, v_n_y, v_n_z, opal_used, bump, label=None):

        # if label != None:
        #     print('TEXT')

        # ax.text(table[0, 1:].min(), table[1:, 0].min(), s=label)
        # bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10}
        # plt.text(2, 0.65, r'$\cos(2 \pi t) \exp(-t)$')

        # ax = fig.add_subplot(1, 1, 1)
        if self.set_auto_limits:
            ax.set_xlim(table[0, 1:].min(), table[0, 1:].max())
            ax.set_ylim(table[1:, 0].min(), table[1:, 0].max())
        ax.set_ylabel(Labels.lbls(v_n_y), fontsize=self.set_label_sise)
        ax.set_xlabel(Labels.lbls(v_n_x), fontsize=self.set_label_sise)

        levels = Levels.get_levels(v_n_z, opal_used, bump)

        contour_filled = plt.contourf(table[0, 1:], table[1:, 0], table[1:, 1:], levels, cmap=plt.get_cmap('RdYlBu_r'),
                                      alpha=self.set_alpha)
        if self.set_show_colorbar:
            clb = plt.colorbar(contour_filled)
            clb.ax.tick_params(labelsize=self.set_label_sise)
            clb.ax.set_title(Labels.lbls(v_n_z), fontsize=self.set_label_sise)

        # ax.colorbar(contour_filled, label=Labels.lbls(v_n_z))

        if self.set_show_contours:
            contour = plt.contour(table[0, 1:], table[1:, 0], table[1:, 1:], levels, colors='k')

            labs = ax.clabel(contour, colors='k', fmt='%{}f'.format(self.set_contour_fmt), fontsize=self.set_label_sise)
            if self.set_rotate_labels != None:
                for lab in labs:
                    lab.set_rotation(self.set_rotate_labels)  # ORIENTATION OF LABELS IN COUNTUR PLOTS
        # ax.set_title('SONIC HR DIAGRAM')

        # print('Yc:{}'.format(yc_val))
        if not self.set_clean and label != None and label != '':
            ax.text(0.9, 0.1, label, style='italic',
                    bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center',
                    verticalalignment='center', transform=ax.transAxes)

        ax.tick_params('y', labelsize=self.set_label_sise)
        ax.tick_params('x', labelsize=self.set_label_sise)
        plt.minorticks_on()

        # plt.ylabel(l_or_lm)
        # plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
        # plt.savefig(name)
        # plt.show()
        return ax

    def plot_3d_curved_surf(self,ax, x_arr, z_arr, y_levels, elevation=100):

        def construct_2d_arrs(x_arr, z_arr, y_level):

            x2d = np.zeros(len(x_arr))
            for i in range(len(z_arr)):
                x2d = np.vstack((x2d, x_arr))
            x2d = np.delete(x2d, 0, 0)

            z2d = np.zeros(len(z_arr))
            for i in range(len(x_arr)):
                z2d = np.vstack((z2d, z_arr))
            z2d = np.delete(z2d, 0, 0).T

            y2d = np.zeros((len(x_arr), len(z_arr)))
            y2d.fill(y_level)


            return x2d, y2d, z2d

        # for i in range(len(y_levels)):
        #     x2, y2, z2 = construct_2d_arrs(x_arr, z_arr, y_levels[i])
        #
        #     ax.contourf(x2, z2, y2, zdir='y', levels=i * elevation + np.array(levels),
        #                 cmap=cmap, alpha=self.set_alpha)

        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from matplotlib.collections import PolyCollection
        from matplotlib import colors as mcolors

        z2d = np.zeros((len(x_arr), len(z_arr)))

        def cc(arg):
            '''
            Shorthand to convert 'named' colors to rgba format at 60% opacity.
            '''
            return mcolors.to_rgba(arg, alpha=0.6)

        def polygon_under_graph(xlist, ylist):
            '''
            Construct the vertex list which defines the polygon filling the space under
            the (xlist, ylist) line graph.  Assumes the xs are in ascending order.
            '''
            return [(xlist[0], 4.0)] + list(zip(xlist, ylist)) + [(xlist[-1], 4.)]

        verts = []
        slice_positions = []
        for i in range(len(y_levels)):
            verts.append(polygon_under_graph(x_arr, z_arr))
            slice_positions = np.append(slice_positions, i * elevation)

        poly = PolyCollection(verts, facecolors=[cc('gray')])
        ax.add_collection3d(poly, zs=slice_positions, zdir='y')


        # z_arr = []
        # for i in range(len(z_levels)):
        #     # z2d = np.zeros((len(x_arr), len(y_arr)))
        #     z2d.fill(elevation*i)
        #     z_arr = np.append(z_arr, z2d)
        #
        #     ax.plot(x_arr, y_arr, [elevation*i], color='black')
        #
        #     poly3dCollection = Poly3DCollection(v)

    def plot_3d_back(self, ax, d3table, x_v_n, y_v_n, color_v_n, metal, slize_for_counturs=0, elevation=100):

        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
        cmap = plt.get_cmap('Blues') #

        ax.set_xlabel(Labels.lbls(x_v_n), fontsize=self.set_label_sise)
        ax.set_ylabel(Labels.lbls(y_v_n), fontsize=self.set_label_sise)
        ax.w_zaxis.line.set_lw(0.)
        ax.set_zticks([])
        # ax.set_zlabel(Labels.lbls(z_v_n), fontsize=fsz)

        levels = Levels.get_levels(color_v_n, metal, '')

        x_arr = d3table[0, 0, 1:]
        y_arr = d3table[0, 1:, 0]

        # ax.set_zlim(x_y_z[0, 1:].min(), x_y_z[0, 1:].max())
        # ax.set_ylim(x_y_z[1:, 0].min(), x_y_z[1:, 0].max())

        # Plot Ground Slice and Set Countours
        if slize_for_counturs < len(d3table[:, 0, 0]):
            z2d_for_levels = d3table[slize_for_counturs, 1:, 1:]
        else:
            raise IOError('Error. slize_for_counturs({}) > max. n of slices if 3D array: {}'
                          .format(slize_for_counturs, len(d3table[:, 0, 0])))

        contour_filled = ax.contourf(x_arr, y_arr, z2d_for_levels, zdir='z', levels=np.array(levels),
                                     cmap=cmap, alpha=self.set_alpha)
        clb = plt.colorbar(contour_filled)
        clb.ax.set_title(Labels.lbls(color_v_n), fontsize=self.set_label_sise)

        # Plot Other slices
        for i in range(1, len(d3table[:, 0, 0])):
            d2arr = d3table[i, 1:, 1:]
            ax.contourf(x_arr, y_arr, i*elevation + d2arr, zdir='z', levels= i*elevation + np.array(levels),
                        cmap=cmap, alpha=self.set_alpha)

        ax.set_zlim3d( 0, (len(d3table[:, 0, 0])-1)*elevation )

    def plot_3d_back2(self, ax, d3table, x_v_n, z_v_n, color_v_n, metal, slize_for_counturs=0, elevation=100):

        def construct_2d_arrs(x_arr, y_arr, z2d_arr):

            x2d = np.zeros(len(z2d_arr[0, :]))
            for i in range(len(z2d_arr[:, 0])):
                x2d = np.vstack((x2d, x_arr))
            x2d = np.delete(x2d, 0, 0)

            y2d = np.zeros(len(z2d_arr[:, 0]))
            for i in range(len(z2d_arr[0, :])):
                y2d = np.vstack((y2d, y_arr))
            y2d = np.delete(y2d, 0, 0).T

            return x2d, y2d, z2d_arr


        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
        cmap = plt.get_cmap('RdYlBu_r') #

        ax.tick_params('y', labelsize=self.set_label_sise)
        ax.tick_params('x', labelsize=self.set_label_sise)
        ax.tick_params('z', labelsize=self.set_label_sise)


        ax.set_xlabel(Labels.lbls(x_v_n), fontsize=self.set_label_sise)
        # ax.set_ylabel(Labels.lbls(y_v_n), fontsize=self.set_label_sise)
        ax.set_zlabel(Labels.lbls(z_v_n), fontsize=self.set_label_sise)
        ax.w_yaxis.line.set_lw(0.)
        ax.set_yticks([])
        # ax.set_zlabel(Labels.lbls(z_v_n), fontsize=fsz)

        levels = Levels.get_levels(color_v_n, metal, '')

        x_arr = d3table[0, 0, 1:]
        y_arr = d3table[0, 1:, 0]

        # ax.set_zlim(x_y_z[0, 1:].min(), x_y_z[0, 1:].max())
        # ax.set_ylim(x_y_z[1:, 0].min(), x_y_z[1:, 0].max())

        # Plot Ground Slice and Set Countours
        if slize_for_counturs < len(d3table[:, 0, 0]):
            z2d_for_levels = d3table[slize_for_counturs, 1:, 1:]
        else:
            raise IOError('Error. slize_for_counturs({}) > max. n of slices if 3D array: {}'
                          .format(slize_for_counturs, len(d3table[:, 0, 0])))

        x2d, y2d, z2d = construct_2d_arrs(x_arr, y_arr, z2d_for_levels)

        contour_filled = ax.contourf(x2d, z2d, y2d, zdir='y', levels=np.array(levels),
                                     cmap=cmap, alpha=self.set_alpha)
        clb = plt.colorbar(contour_filled)
        clb.ax.set_title(Labels.lbls(color_v_n), fontsize=self.set_label_sise)
        clb.ax.tick_params(labelsize=self.set_label_sise)

        # Plot Other slices
        for i in range(1, len(d3table[:, 0, 0])):
            d2arr = d3table[i, 1:, 1:]
            x2d, y2d, z2d = construct_2d_arrs(x_arr, y_arr, d2arr)
            ax.contourf(x2d, i*elevation + z2d, y2d, zdir='y', levels= i*elevation + np.array(levels),
                        cmap=cmap, alpha=self.set_alpha)


    # @staticmethod
    # def plot_obs_x_llm(ax, obs_cls, l_or_lm, v_n_x, yc_val, use_gaia = False, clean=False, check_star_lm_wne=False):
    #
    #     def plot_obs_err(ax, obs_cls, star_n, l_or_lm, v_n_x, yc_val, use_gaia = False):
    #
    #         x_coord =  obs_cls.get_num_par(v_n_x, star_n)
    #
    #         if l_or_lm == 'lm':
    #
    #             lm_err1, lm_err2 = obs_cls.get_star_lm_obs_err(star_n, yc_val, use_gaia)          # ERRORS Mdot
    #
    #             if v_n_x == 'mdot':
    #                 mdot1, mdot2 = obs_cls.get_star_mdot_obs_err(star_n, yc_val)
    #                 mdot_coord = [mdot1, mdot2, mdot2, mdot1]
    #                 lm_coord = [lm_err1, lm_err1, lm_err2, lm_err2]
    #                 ax.add_patch(patches.Polygon(xy=list(zip(mdot_coord, lm_coord)), fill=True, alpha=.4,
    #                                              color=obs_cls.get_class_color(star_n)))
    #             else:
    #                 ax.plot([x_coord, x_coord], [lm_err1, lm_err2], '-', color=obs_cls.get_class_color(star_n))
    #
    #         if l_or_lm == 'l':
    #
    #             l_err1, l_err2 = obs_cls.get_star_l_obs_err(star_n, yc_val, use_gaia)
    #
    #             if v_n_x == 'mdot':
    #                 mdot1, mdot2 = obs_cls.get_star_mdot_obs_err(star_n, yc_val)
    #                 mdot_coord = [mdot1, mdot2, mdot2, mdot1]
    #                 l_coord = [l_err1, l_err1, l_err2, l_err2]
    #                 ax.add_patch(patches.Polygon(xy=list(zip(mdot_coord, l_coord)), fill=True, alpha=.4,
    #                                              color=obs_cls.get_class_color(star_n)))
    #             else:
    #                 ax.plot([x_coord, x_coord], [l_err1, l_err2], '-', color=obs_cls.get_class_color(star_n))
    #
    #     def plot_stars(ax, obs_cls, star_n, l_or_lm, v_n_x, yc_val, use_gaia = False, clean=False):
    #
    #         x_coord = obs_cls.get_num_par(v_n_x, star_n)
    #         llm_obs = obs_cls.get_llm(l_or_lm, star_n, yc_val, use_gaia, check_star_lm_wne)
    #
    #         ax.plot(x_coord, llm_obs, marker=obs_cls.get_clss_marker(star_n), markersize='9',
    #                 color=obs_cls.get_class_color(star_n), ls='', mec='black')  # plot color dots)))
    #
    #         if not clean:
    #             ax.annotate('{}'.format(int(star_n)), xy=(x_coord, llm_obs),
    #                         textcoords='data')  # plot numbers of stars
    #
    #     def plot_evol_err(ax, obs_cls, star_n, l_or_lm, v_n_x, yc_val):
    #         if l_or_lm == 'lm':
    #             llm1, llm2 = obs_cls.get_star_lm_err(star_n, yc_val)
    #             # obs_cls.get_star_llm_evol_err(star_n, l_or_lm, yc_val, 1.0, 0.1)                  # ERRORS L/LM
    #             # mdot1, mdot2 = obs_cls.get_star_mdot_err(star_n, l_or_lm, yc_val, 1.0, 0.1, 'nugis')           # ERRORS Mdot
    #
    #             x_coord = obs_cls.get_num_par(v_n_x, star_n)
    #
    #             ax.plot([x_coord, x_coord], [llm1, llm2], '-', color='gray')
    #             ax.plot([x_coord, x_coord], [llm1, llm2], '.', color='gray')
    #
    #
    #
    #     classes = []
    #     classes.append('dum')
    #     x_coord = []
    #     llm_obs = []
    #
    #     # from Phys_Math_Labels import Opt_Depth_Analythis
    #     # use_gaia = False
    #     for star_n in obs_cls.stars_n:
    #         i = -1
    #         x_coord = np.append(x_coord, obs_cls.get_num_par(v_n_x, star_n))
    #         llm_obs = np.append(llm_obs, obs_cls.get_llm(l_or_lm, star_n, yc_val, use_gaia, check_star_lm_wne))
    #         # llm_obs = np.append(llm_obs, obs_cls.get_num_par(l_or_lm, star_n, yc_val, use_gaia))
    #
    #         plot_obs_err(ax, obs_cls, star_n, l_or_lm, v_n_x, yc_val, use_gaia)
    #         plot_stars(ax, obs_cls, star_n, l_or_lm, v_n_x, yc_val, use_gaia, clean)
    #         plot_evol_err(ax,obs_cls, star_n, l_or_lm, v_n_x, yc_val)
    #
    #         if obs_cls.get_star_class(star_n) not in classes:
    #             plt.plot(x_coord[i], llm_obs[i], marker=obs_cls.get_clss_marker(star_n), markersize='9',
    #                      color=obs_cls.get_class_color(star_n), ls='', mec='black',
    #                      label='{}'.format(obs_cls.get_star_class(star_n)))  # plot color dots)))
    #             classes.append(obs_cls.get_star_class(star_n))
    #
    #         # # --- PLOT OBSERVABLE ERROR ---
    #         # if l_or_lm == 'lm':
    #         #     llm1, llm2 = obs_cls.get_star_lm_err(star_n, yc_val, use_gaia)          # ERRORS Mdot
    #         #     lm = obs_cls.get_num_par(v_n_x, star_n)
    #         #
    #         #     if v_n_x == 'mdot':
    #         #         mdot1, mdot2 = obs_cls.get_star_mdot_obs_err(star_n, yc_val)
    #         #         mdot_coord = [mdot1, mdot2, mdot2, mdot1]
    #         #         lm_coord = [lm_err1, lm_err1, lm_err2, lm_err2]
    #         #         ax.add_patch(patches.Polygon(xy=list(zip(mdot_coord, lm_coord)), fill=True, alpha=.4,
    #         #                                      color=obs_cls.get_class_color(star_n)))
    #         #     else:
    #         #         ax.plot([x_coord_, x_coord_], [llm1, llm2], '-', color='gray')
    #         #
    #         # else:
    #         #     l1, l2 = obs_cls.get_star_l_err(star_n, yc_val, use_gaia)
    #         #     x_coord_ = obs_cls.get_num_par(v_n_x, star_n)
    #         #     ax.plot([x_coord_, x_coord_], [l1, l2], '-', color='gray')
    #         #
    #         # if v_n_x == 'mdot':
    #         #     mdot1, mdot2 = obs_cls.get_star_mdot_obs_err(star_n, yc_val)
    #         #     ax.plot([x_coord_, x_coord_], [l1, l2], '-', color='gray')
    #
    #             # ax.errorbar(mdot_obs[i], llm_obs[i], yerr=[[llm1],  [llm2]], fmt='--.', color=obs_cls.get_class_color(star_n))
    #
    #         # ax.plot(x_coord[i], llm_obs[i], marker=obs_cls.get_clss_marker(star_n), markersize='9',
    #         #         color=obs_cls.get_class_color(star_n), ls='', mec='black')  # plot color dots)))
    #         # if not clean:
    #         #     ax.annotate('{}'.format(int(star_n)), xy=(x_coord[i], llm_obs[i]),
    #         #                 textcoords='data')  # plot numbers of stars
    #
    #         # t = obs_cls.get_num_par('t', star_n)
    #         # ax.annotate('{}'.format("%.2f" % t), xy=(mdot_obs[i], llm_obs[i]), textcoords='data')  # plot numbers of stars
    #
    #         # v_inf = obs_cls.get_num_par('v_inf', star_n)
    #         # tau_cl = Opt_Depth_Analythis(30, v_inf, 1., 1., mdot_obs[i], 0.20)
    #         # tau = tau_cl.anal_eq_b1(1.)
    #         # # # # ax.annotate(str(int(tau)), xy=(mdot_obs[i], llm_obs[i]), textcoords='data')  # plo
    #         # ax.annotate('{} {}'.format(str(int(tau)), eta), xy=(mdot_obs[i], llm_obs[i]),
    #         #             textcoords='data')  # plot numbers of stars
    #
    #         # if obs_cls.get_star_class(star_n) not in classes:
    #         #     plt.plot(x_coord[i], llm_obs[i], marker=obs_cls.get_clss_marker(star_n), markersize='9',
    #         #              color=obs_cls.get_class_color(star_n), ls='', mec='black',
    #         #              label='{}'.format(obs_cls.get_star_class(star_n)))  # plot color dots)))
    #         #     classes.append(obs_cls.get_star_class(star_n))
    #
    #
    #     print('\t__PLOT: total stars: {}'.format(len(obs_cls.stars_n)))
    #     print(len(x_coord), len(llm_obs))
    #
    #     # fit = np.polyfit(mdot_obs, llm_obs, 1)  # fit = set of coeddicients (highest first)
    #     # f = np.poly1d(fit)
    #     # fit_x_coord = np.mgrid[(mdot_obs.min() - 1):(mdot_obs.max() + 1):1000j]
    #
    #     mdot_grid = np.mgrid[(x_coord.min() - 1):(x_coord.max() + 1):100j]
    #     x_coord__, y_coord__ = Math.fit_plynomial(x_coord, llm_obs, 1, 100, mdot_grid)
    #     ax.plot(x_coord__, y_coord__, '-.', color='blue')
    #
    #     min_mdot, max_mdot = obs_cls.get_min_max('mdot')
    #     min_llm, max_llm = obs_cls.get_min_max_llm(l_or_lm, yc_val, use_gaia, check_star_lm_wne)
    #
    #     # ax.set_xlim(min_mdot - 0.2, max_mdot + 0.2)
    #     # ax.set_ylim(min_llm - 0.05, max_llm + 0.05)
    #
    #     ax.set_ylabel(Labels.lbls(l_or_lm))
    #     ax.set_xlabel(Labels.lbls('mdot'))
    #     ax.grid(which='major', alpha=0.2)
    #     ax.legend(bbox_to_anchor=(1, 1), loc='upper right', ncol=1)
    #
    #     print('Yc:{}'.format(yc_val))
    #
    #     if not clean:
    #         ax.text(0.9, 0.9, 'Yc:{}'.format(yc_val), style='italic',
    #                 bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center',
    #                 verticalalignment='center', transform=ax.transAxes)
    #         if use_gaia:
    #             ax.text(0.9, 0.75, 'GAIA', style='italic',
    #                     bbox={'facecolor': 'blue', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center',
    #                     verticalalignment='center', transform=ax.transAxes)
    #
    #     return ax
    #
    # @staticmethod
    # def plot_obs_mdot_llm(ax, obs_cls, l_or_lm, yc_val, clean = False):
    #     '''
    #
    #     :param ax:
    #     :param obs_cls:
    #     :param l_or_lm:
    #     :param yc_val:
    #     :param yc1:
    #     :param yc2:
    #     :return:
    #     '''
    #     classes = []
    #     classes.append('dum')
    #     mdot_obs = []
    #     llm_obs = []
    #
    #     # from Phys_Math_Labels import Opt_Depth_Analythis
    #
    #     for star_n in obs_cls.stars_n:
    #         i = -1
    #         mdot_obs = np.append(mdot_obs, obs_cls.get_num_par('mdot', star_n))
    #         llm_obs = np.append(llm_obs, obs_cls.get_num_par(l_or_lm, star_n, yc_val))
    #         eta = obs_cls.get_num_par('eta', star_n)
    #
    #         lm_err1, lm_err2 = obs_cls.get_star_lm_obs_err(star_n, yc_val)
    #         mdot1, mdot2 = obs_cls.get_star_mdot_obs_err(star_n, yc_val)
    #
    #         mdot_coord = [mdot1, mdot2, mdot2, mdot1]
    #         lm_coord = [lm_err1, lm_err1, lm_err2, lm_err2]
    #         ax.add_patch(patches.Polygon(xy=list(zip(mdot_coord, lm_coord)), fill=True, alpha=.4,
    #                                      color=obs_cls.get_class_color(star_n)))
    #
    #         if l_or_lm == 'lm':
    #
    #             llm1, llm2 = obs_cls.get_star_lm_err(star_n, yc_val)
    #                 # obs_cls.get_star_llm_evol_err(star_n, l_or_lm, yc_val, 1.0, 0.1)                  # ERRORS L/LM
    #             # mdot1, mdot2 = obs_cls.get_star_mdot_err(star_n, l_or_lm, yc_val, 1.0, 0.1, 'nugis')           # ERRORS Mdot
    #             mdot = obs_cls.get_num_par('mdot', star_n)
    #             ax.plot([mdot, mdot], [llm1, llm2], '-', color='white')
    #             #color=obs_cls.get_class_color(star_n)
    #
    #             # ax.errorbar(mdot_obs[i], llm_obs[i], yerr=[[llm1],  [llm2]], fmt='--.', color=obs_cls.get_class_color(star_n))
    #
    #
    #         ax.plot(mdot_obs[i], llm_obs[i], marker=obs_cls.get_clss_marker(star_n), markersize='9',
    #                  color=obs_cls.get_class_color(star_n), ls='', mec='black')  # plot color dots)))
    #         if not clean:
    #             ax.annotate('{}'.format(int(star_n)), xy=(mdot_obs[i], llm_obs[i]),textcoords='data')  # plot numbers of stars
    #
    #         # t = obs_cls.get_num_par('t', star_n)
    #         # ax.annotate('{}'.format("%.2f" % t), xy=(mdot_obs[i], llm_obs[i]), textcoords='data')  # plot numbers of stars
    #
    #
    #         # v_inf = obs_cls.get_num_par('v_inf', star_n)
    #         # tau_cl = Opt_Depth_Analythis(30, v_inf, 1., 1., mdot_obs[i], 0.20)
    #         # tau = tau_cl.anal_eq_b1(1.)
    #         # # # # ax.annotate(str(int(tau)), xy=(mdot_obs[i], llm_obs[i]), textcoords='data')  # plo
    #         # ax.annotate('{} {}'.format(str(int(tau)), eta), xy=(mdot_obs[i], llm_obs[i]),
    #         #             textcoords='data')  # plot numbers of stars
    #
    #         if obs_cls.get_star_class(star_n) not in classes:
    #             plt.plot(mdot_obs[i], llm_obs[i], marker=obs_cls.get_clss_marker(star_n), markersize='9',
    #                      color=obs_cls.get_class_color(star_n), ls='', mec='black',
    #                      label='{}'.format(obs_cls.get_star_class(star_n)))  # plot color dots)))
    #             classes.append(obs_cls.get_star_class(star_n))
    #
    #     print('\t__PLOT: total stars: {}'.format(len(obs_cls.stars_n)))
    #     print(len(mdot_obs), len(llm_obs))
    #
    #     # fit = np.polyfit(mdot_obs, llm_obs, 1)  # fit = set of coeddicients (highest first)
    #     # f = np.poly1d(fit)
    #     # fit_x_coord = np.mgrid[(mdot_obs.min() - 1):(mdot_obs.max() + 1):1000j]
    #
    #     mdot_grid = np.mgrid[(mdot_obs.min() - 1):(mdot_obs.max() + 1):100j]
    #     x_coord, y_coord = Math.fit_plynomial(mdot_obs, llm_obs, 1, 100, mdot_grid)
    #     ax.plot(x_coord, y_coord, '-.', color='blue')
    #
    #     min_mdot, max_mdot = obs_cls.get_min_max('mdot')
    #     min_llm, max_llm = obs_cls.get_min_max(l_or_lm, yc_val)
    #
    #     ax.set_xlim(min_mdot - 0.2, max_mdot + 0.2)
    #     ax.set_ylim(min_llm - 0.05, max_llm + 0.05)
    #
    #     ax.set_ylabel(Labels.lbls(l_or_lm))
    #     ax.set_xlabel(Labels.lbls('mdot'))
    #     ax.grid(which='major', alpha=0.2)
    #     ax.legend(bbox_to_anchor=(1, 1), loc='upper right', ncol=1)
    #
    #     print('Yc:{}'.format(yc_val))
    #
    #     if not clean:
    #         ax.text(0.9, 0.9, 'Yc:{}'.format(yc_val), style='italic',
    #                 bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center',
    #                 verticalalignment='center', transform=ax.transAxes)
    #
    #     return ax
    #     # ax.text(min_mdot, max_llm, 'Yc:{}'.format(yc_val), style='italic',
    #     #         bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10})
    #
    #     # l_grid = np.mgrid[5.2:6.:100j]
    #     # ax.plot(Physics.l_mdot_prescriptions(l_grid, ), l_grid, '-.', color='orange', label='Nugis & Lamers 2000')
    #     #
    #     # ax.plot(Physics.yoon(l_grid, 10 ** 0.02), l_grid, '-.', color='green', label='Yoon 2017')
    #
    # @staticmethod
    # def plot_obs_t_llm_mdot_int(ax, t_llm_mdot, obs_cls, l_or_lm, lim_t1 = None, lim_t2 = None, show_legend = True, clean = False):
    #
    #     if lim_t1 == None: lim_t1 = t_llm_mdot[0, 1:].min()
    #     if lim_t2 == None: lim_t2 = t_llm_mdot[0, 1:].max()
    #
    #     yc_val = t_llm_mdot[0, 0] #
    #
    #     classes = []
    #     classes.append('dum')
    #     x = []
    #     y = []
    #     for star_n in obs_cls.stars_n:
    #         xyz = obs_cls.get_xyz_from_yz(yc_val, star_n, l_or_lm, 'mdot',
    #                                       t_llm_mdot[0,1:], t_llm_mdot[1:,0], t_llm_mdot[1:,1:], lim_t1, lim_t2)
    #
    #         if xyz.any():
    #             x = np.append(x, xyz[0, 0])
    #             y = np.append(y, xyz[1, 0])
    #
    #             # print('Star {}, {} range: ({}, {})'.format(star_n,l_or_lm, llm1, llm2))
    #
    #             for i in range(len(xyz[0, :])):
    #
    #                 ax.plot(xyz[0, i], xyz[1, i], marker=obs_cls.get_clss_marker(star_n), markersize='9',
    #                          color=obs_cls.get_class_color(star_n), ls='', mec='black')  # plot color dots)))
    #                 if not clean:
    #                     ax.annotate(int(star_n), xy=(xyz[0, i], xyz[1, i]),
    #                                 textcoords='data')  # plot numbers of stars
    #
    #                 if obs_cls.get_star_class(star_n) not in classes:
    #                     ax.plot(xyz[0, i], xyz[1, i], marker=obs_cls.get_clss_marker(star_n), markersize='9',
    #                              color=obs_cls.get_class_color(star_n), mec='black', ls='',
    #                              label='{}'.format(obs_cls.get_star_class(star_n)))  # plot color dots)))
    #                     classes.append(obs_cls.get_star_class(star_n))
    #
    #                 # -------------------------OBSERVABLE ERRORS FOR L and Mdot ----------------------------------------
    #                 lm_err1, lm_err2 = obs_cls.get_star_lm_obs_err(star_n, yc_val)
    #                 ts1_b, ts2_b, ts1_t, ts2_t = obs_cls.get_star_ts_obs_err(star_n, t_llm_mdot, yc_val, lim_t1, lim_t2)
    #                 ts_coord = [ts1_b, ts2_b, ts2_t, ts1_t]
    #                 lm_coord = [lm_err1, lm_err1, lm_err2, lm_err2]
    #                 ax.add_patch(patches.Polygon(xy=list(zip(ts_coord, lm_coord)), fill=True, alpha=.7,
    #                                              color=obs_cls.get_class_color(star_n)))
    #
    #                 # ax.plot([xyz[0, i], xyz[0, i]], [lm_err1, lm_err2], '-',
    #                 #         color='gray')
    #
    #                 # ax.plot([ts_err1, ts_err2], [xyz[1, i], xyz[1, i]], '-',
    #                 #         color='gray')
    #                 # print('  :: Star {}. L/M:{}(+{}/-{}) ts:{}(+{}/-{})'
    #                 #       .format(star_n, "%.2f" % xyz[0, i], "%.2f" % np.abs(xyz[0, i]-lm_err2), "%.2f" % np.abs(xyz[0, i]-lm_err2),
    #                 #               "%.2f" % xyz[1, i],  "%.2f" % np.abs(xyz[1, i]-ts_err2), "%.2f" % np.abs(xyz[1, i]-ts_err1)))
    #                 # ax.plot([ts_err1, ts_err2], [lm_err1, lm_err2], '-', color='gray')
    #
    #                 #
    #
    #
    #
    #                 if l_or_lm == 'lm':
    #                     lm1, lm2 = obs_cls.get_star_lm_err(star_n, yc_val)
    #                     ts1, ts2 = obs_cls.get_star_ts_err(star_n, t_llm_mdot, yc_val, lim_t1, lim_t2)
    #                     ax.plot([ts1, ts2], [lm1, lm2], '-', color='white')
    #                             # color=obs_cls.get_class_color(star_n))
    #
    #                     # ax.plot([xyz[0, i], xyz[0, i]], [lm1, lm2], '-',
    #                     #         color=obs_cls.get_class_color(star_n))
    #                     # ax.plot([ts1, ts2], [xyz[1, i], xyz[1, i]], '-',
    #                     #         color=obs_cls.get_class_color(star_n))
    #
    #
    #                     # ax.add_patch(patches.Rectangle((xyz[0, i] - ts1, xyz[1, i] - lm1), ts2 + ts1, lm2 + lm1,
    #                     #                                alpha=.3, color=obs_cls.get_class_color(star_n)))
    #
    #                     # ax.add_patch(patches.Rectangle((xyz[0, i] - ts1, xyz[1, i] - lm1), ts2 + ts1, lm2 + lm1,
    #                     #                                alpha=.3, color=obs_cls.get_class_color(star_n)))
    #
    #                     # ax.plot([xyz[0, i] - ts1, xyz[1, i] - lm1], [xyz[0, i]+ts2, xyz[1, i] + lm2], '-', color=obs_cls.get_class_color(star_n))
    #                     # ax.plot([xyz[0, i] - ts1, xyz[0, i]+ts2], [xyz[1, i] - lm1, xyz[1, i] + lm2], '-', color=obs_cls.get_class_color(star_n))
    #
    #                     # ax.errorbar(xyz[0, i], xyz[1, i], yerr=[[lm1], [lm2]], fmt='--.', color = obs_cls.get_class_color(star_n))
    #                     # ax.errorbar(xyz[0, i], xyz[1, i], xerr=[[ts1], [ts2]], fmt='--.', color=obs_cls.get_class_color(star_n))
    #
    #
    #
    #     fit = np.polyfit(x, y, 1)  # fit = set of coeddicients (highest first)
    #     f = np.poly1d(fit)
    #     fit_x_coord = np.mgrid[(t_llm_mdot[0,1:].min()):(t_llm_mdot[0,1:].max()):1000j]
    #     ax.plot(fit_x_coord, f(fit_x_coord), '-.', color='blue')
    #
    #     ax.set_xlim(t_llm_mdot[0,1:].min(), t_llm_mdot[0,1:].max())
    #     ax.set_ylim(t_llm_mdot[1:,0].min(), t_llm_mdot[1:,0].max())
    #     # ax.text(0.9, 0.9,'Yc:{}'.format(yc_val), style='italic',
    #     #         bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
    #
    #     # ax.text(x.max(), y.max(), 'Yc:{}'.format(yc_val), style='italic',
    #     #         bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10})
    #     if show_legend:
    #         ax.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
    #
    #     return ax

    @staticmethod
    def plot_edd_kappa(ax, t_arr, sm_cls, opal_used, n_int_edd):
        from OPAL import Table_Analyze
        Table_Analyze.plot_k_vs_t = False  # there is no need to plot just one kappa in the range of availability
        clas_table_anal = Table_Analyze(opal_used, n_int_edd, False)

        for i in range(len(sm_cls)):  # self.nmdls
            mdl_m = sm_cls[i].get_cond_value('xm', 'sp')
            mdl_l = sm_cls[i].get_cond_value('l', 'sp')

            k_edd = Physics.edd_opacity(mdl_m, mdl_l)

            n_model_for_edd_k = clas_table_anal.interp_for_single_k(t_arr.min(), t_arr.max(), n_int_edd, k_edd)
            x = n_model_for_edd_k[0, :]
            y = n_model_for_edd_k[1, :]
            color = 'black'
            # lbl = 'Model:{}, k_edd:{}'.format(i, '%.2f' % 10 ** k_edd)
            ax.plot(x, y, '-.', color=color)  # , label=lbl)
            ax.plot(x[-1], y[-1], 'x', color=color)

        Table_Analyze.plot_k_vs_t = True
        return ax



class PlotBackground:


    def __init__(self):
        pass

    @staticmethod
    def plot_color_table(table, v_n_x, v_n_y, v_n_z, opal_used, label = None, fsz=12, lagel_angle=0):

        plt.figure()
        ax = plt.subplot(111)


        if label != None:
            ax.text(0.8, 0.1, label, style='italic',
                    bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center',
                    verticalalignment='center', transform=ax.transAxes)


            # print('TEXT')
            # plt.text(table[0, 1:].min(), table[1:, 0].min(), label, style='italic')
            # bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10}
            # plt.text(2, 0.65, r'$\cos(2 \pi t) \exp(-t)$')

        # ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(table[0, 1:].min(), table[0, 1:].max())
        ax.set_ylim(table[1:, 0].min(), table[1:, 0].max())
        ax.set_ylabel(Labels.lbls(v_n_y), fontsize=fsz)
        ax.set_xlabel(Labels.lbls(v_n_x), fontsize=fsz)

        levels = Levels.get_levels(v_n_z, opal_used)

        contour_filled = plt.contourf(table[0, 1:], table[1:, 0], table[1:, 1:], levels, cmap=plt.get_cmap('RdYlBu_r'),
                                      alpha=1.0)
        clb = plt.colorbar(contour_filled) # orientation='horizontal', :)
        clb.ax.set_title(Labels.lbls(v_n_z), fontsize=fsz)
        clb.ax.tick_params(labelsize=fsz)

        # ax.colorbar(contour_filled, label=Labels.lbls(v_n_z))
        contour = plt.contour(table[0, 1:], table[1:, 0], table[1:, 1:], levels, colors='k')
        labs = ax.clabel(contour, colors='k', fmt='%2.2f', inline=True, fontsize=fsz)
        for lab in labs:
            lab.set_rotation(lagel_angle)#295
        # contour_filled = plt.contourf(table[0, 1:], table[1:, 0], table[1:, 1:], levels, cmap=plt.get_cmap('RdYlBu_r'))
        # clb = plt.colorbar(contour_filled)
        # clb.ax.set_title(Labels.lbls(v_n_z), fontsize=fsz)
        # contour = plt.contour(table[0, 1:], table[1:, 0], table[1:, 1:], levels, colors='k')
        # # plt.clabel(contour, colors='k', fmt='%2.2f', fontsize=fsz)
        # # plt.title('SONIC HR DIAGRAM')

        ax.minorticks_on()

        ax.invert_xaxis()

        plt.xticks(fontsize=fsz)
        plt.yticks(fontsize=fsz)

        # plt.ylabel(l_or_lm)
        # plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
        # plt.savefig(name)
        plt.show()

    @staticmethod
    def plot_color_background(ax, table, v_n_x, v_n_y, v_n_z, opal_used, bump, label = None, alpha = 0.8, clean=False, fsz=12, rotation=0):



        # if label != None:
        #     print('TEXT')

            # ax.text(table[0, 1:].min(), table[1:, 0].min(), s=label)
            # bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10}
            # plt.text(2, 0.65, r'$\cos(2 \pi t) \exp(-t)$')

        # ax = fig.add_subplot(1, 1, 1)

        ax.set_xlim(table[0, 1:].min(), table[0, 1:].max())
        ax.set_ylim(table[1:, 0].min(), table[1:, 0].max())
        ax.set_ylabel(Labels.lbls(v_n_y), fontsize=fsz)
        ax.set_xlabel(Labels.lbls(v_n_x), fontsize=fsz)

        levels = Levels.get_levels(v_n_z, opal_used, bump)

        # 'RdYlBu_r'

        if v_n_x == 'mdot' and v_n_y == 'lm' and v_n_z == 'tau':
            pass
        else:
            contour_filled = plt.contourf(table[0, 1:], table[1:, 0], table[1:, 1:], levels, cmap=plt.get_cmap('RdYlBu_r'), alpha=alpha)
            clb = plt.colorbar(contour_filled)
            clb.ax.set_title(Labels.lbls(v_n_z), fontsize=fsz)
            clb.ax.tick_params(labelsize=fsz)

        # ax.colorbar(contour_filled, label=Labels.lbls(v_n_z))

        contour = plt.contour(table[0, 1:], table[1:, 0], table[1:, 1:], levels, colors='k')

        if v_n_x == 'mdot' and v_n_y == 'lm' and v_n_z == 'tau':
            labs=ax.clabel(contour, colors='k', fmt='%2.0f', fontsize=fsz, manual=True)
        else:
            labs = ax.clabel(contour, colors='k', fmt='%2.2f', fontsize=fsz)

        if rotation != None:
            for lab in labs:
                lab.set_rotation(rotation)       # ORIENTATION OF LABELS IN COUNTUR PLOTS
        # ax.set_title('SONIC HR DIAGRAM')

        # print('Yc:{}'.format(yc_val))
        if not clean and label != None and label != '':
            ax.text(0.9, 0.1, label, style='italic',
                    bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center',
                    verticalalignment='center', transform=ax.transAxes)

        ax.tick_params('y', labelsize=fsz)
        ax.tick_params('x', labelsize=fsz)

        # plt.ylabel(l_or_lm)
        # plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
        # plt.savefig(name)
        # plt.show()
        return ax

    # @staticmethod
    # def plot_obs_x_llm(ax, obs_cls, l_or_lm, v_n_x, yc_val, use_gaia = False, clean=False, check_star_lm_wne=False):
    #
    #     def plot_obs_err(ax, obs_cls, star_n, l_or_lm, v_n_x, yc_val, use_gaia = False):
    #
    #         x_coord =  obs_cls.get_num_par(v_n_x, star_n)
    #
    #         if l_or_lm == 'lm':
    #
    #             lm_err1, lm_err2 = obs_cls.get_star_lm_obs_err(star_n, yc_val, use_gaia)          # ERRORS Mdot
    #
    #             if v_n_x == 'mdot':
    #                 mdot1, mdot2 = obs_cls.get_star_mdot_obs_err(star_n, yc_val)
    #                 mdot_coord = [mdot1, mdot2, mdot2, mdot1]
    #                 lm_coord = [lm_err1, lm_err1, lm_err2, lm_err2]
    #                 ax.add_patch(patches.Polygon(xy=list(zip(mdot_coord, lm_coord)), fill=True, alpha=.4,
    #                                              color=obs_cls.get_class_color(star_n)))
    #             else:
    #                 ax.plot([x_coord, x_coord], [lm_err1, lm_err2], '-', color=obs_cls.get_class_color(star_n))
    #
    #         if l_or_lm == 'l':
    #
    #             l_err1, l_err2 = obs_cls.get_star_l_obs_err(star_n, yc_val, use_gaia)
    #
    #             if v_n_x == 'mdot':
    #                 mdot1, mdot2 = obs_cls.get_star_mdot_obs_err(star_n, yc_val)
    #                 mdot_coord = [mdot1, mdot2, mdot2, mdot1]
    #                 l_coord = [l_err1, l_err1, l_err2, l_err2]
    #                 ax.add_patch(patches.Polygon(xy=list(zip(mdot_coord, l_coord)), fill=True, alpha=.4,
    #                                              color=obs_cls.get_class_color(star_n)))
    #             else:
    #                 ax.plot([x_coord, x_coord], [l_err1, l_err2], '-', color=obs_cls.get_class_color(star_n))
    #
    #     def plot_stars(ax, obs_cls, star_n, l_or_lm, v_n_x, yc_val, use_gaia = False, clean=False):
    #
    #         x_coord = obs_cls.get_num_par(v_n_x, star_n)
    #         llm_obs = obs_cls.get_llm(l_or_lm, star_n, yc_val, use_gaia, check_star_lm_wne)
    #
    #         ax.plot(x_coord, llm_obs, marker=obs_cls.get_clss_marker(star_n), markersize='9',
    #                 color=obs_cls.get_class_color(star_n), ls='', mec='black')  # plot color dots)))
    #
    #         if not clean:
    #             ax.annotate('{}'.format(int(star_n)), xy=(x_coord, llm_obs),
    #                         textcoords='data')  # plot numbers of stars
    #
    #     def plot_evol_err(ax, obs_cls, star_n, l_or_lm, v_n_x, yc_val):
    #         if l_or_lm == 'lm':
    #             llm1, llm2 = obs_cls.get_star_lm_err(star_n, yc_val)
    #             # obs_cls.get_star_llm_evol_err(star_n, l_or_lm, yc_val, 1.0, 0.1)                  # ERRORS L/LM
    #             # mdot1, mdot2 = obs_cls.get_star_mdot_err(star_n, l_or_lm, yc_val, 1.0, 0.1, 'nugis')           # ERRORS Mdot
    #
    #             x_coord = obs_cls.get_num_par(v_n_x, star_n)
    #
    #             ax.plot([x_coord, x_coord], [llm1, llm2], '-', color='gray')
    #             ax.plot([x_coord, x_coord], [llm1, llm2], '.', color='gray')
    #
    #
    #
    #     classes = []
    #     classes.append('dum')
    #     x_coord = []
    #     llm_obs = []
    #
    #     # from Phys_Math_Labels import Opt_Depth_Analythis
    #     # use_gaia = False
    #     for star_n in obs_cls.stars_n:
    #         i = -1
    #         x_coord = np.append(x_coord, obs_cls.get_num_par(v_n_x, star_n))
    #         llm_obs = np.append(llm_obs, obs_cls.get_llm(l_or_lm, star_n, yc_val, use_gaia, check_star_lm_wne))
    #         # llm_obs = np.append(llm_obs, obs_cls.get_num_par(l_or_lm, star_n, yc_val, use_gaia))
    #
    #         plot_obs_err(ax, obs_cls, star_n, l_or_lm, v_n_x, yc_val, use_gaia)
    #         plot_stars(ax, obs_cls, star_n, l_or_lm, v_n_x, yc_val, use_gaia, clean)
    #         plot_evol_err(ax,obs_cls, star_n, l_or_lm, v_n_x, yc_val)
    #
    #         if obs_cls.get_star_class(star_n) not in classes:
    #             plt.plot(x_coord[i], llm_obs[i], marker=obs_cls.get_clss_marker(star_n), markersize='9',
    #                      color=obs_cls.get_class_color(star_n), ls='', mec='black',
    #                      label='{}'.format(obs_cls.get_star_class(star_n)))  # plot color dots)))
    #             classes.append(obs_cls.get_star_class(star_n))
    #
    #         # # --- PLOT OBSERVABLE ERROR ---
    #         # if l_or_lm == 'lm':
    #         #     llm1, llm2 = obs_cls.get_star_lm_err(star_n, yc_val, use_gaia)          # ERRORS Mdot
    #         #     lm = obs_cls.get_num_par(v_n_x, star_n)
    #         #
    #         #     if v_n_x == 'mdot':
    #         #         mdot1, mdot2 = obs_cls.get_star_mdot_obs_err(star_n, yc_val)
    #         #         mdot_coord = [mdot1, mdot2, mdot2, mdot1]
    #         #         lm_coord = [lm_err1, lm_err1, lm_err2, lm_err2]
    #         #         ax.add_patch(patches.Polygon(xy=list(zip(mdot_coord, lm_coord)), fill=True, alpha=.4,
    #         #                                      color=obs_cls.get_class_color(star_n)))
    #         #     else:
    #         #         ax.plot([x_coord_, x_coord_], [llm1, llm2], '-', color='gray')
    #         #
    #         # else:
    #         #     l1, l2 = obs_cls.get_star_l_err(star_n, yc_val, use_gaia)
    #         #     x_coord_ = obs_cls.get_num_par(v_n_x, star_n)
    #         #     ax.plot([x_coord_, x_coord_], [l1, l2], '-', color='gray')
    #         #
    #         # if v_n_x == 'mdot':
    #         #     mdot1, mdot2 = obs_cls.get_star_mdot_obs_err(star_n, yc_val)
    #         #     ax.plot([x_coord_, x_coord_], [l1, l2], '-', color='gray')
    #
    #             # ax.errorbar(mdot_obs[i], llm_obs[i], yerr=[[llm1],  [llm2]], fmt='--.', color=obs_cls.get_class_color(star_n))
    #
    #         # ax.plot(x_coord[i], llm_obs[i], marker=obs_cls.get_clss_marker(star_n), markersize='9',
    #         #         color=obs_cls.get_class_color(star_n), ls='', mec='black')  # plot color dots)))
    #         # if not clean:
    #         #     ax.annotate('{}'.format(int(star_n)), xy=(x_coord[i], llm_obs[i]),
    #         #                 textcoords='data')  # plot numbers of stars
    #
    #         # t = obs_cls.get_num_par('t', star_n)
    #         # ax.annotate('{}'.format("%.2f" % t), xy=(mdot_obs[i], llm_obs[i]), textcoords='data')  # plot numbers of stars
    #
    #         # v_inf = obs_cls.get_num_par('v_inf', star_n)
    #         # tau_cl = Opt_Depth_Analythis(30, v_inf, 1., 1., mdot_obs[i], 0.20)
    #         # tau = tau_cl.anal_eq_b1(1.)
    #         # # # # ax.annotate(str(int(tau)), xy=(mdot_obs[i], llm_obs[i]), textcoords='data')  # plo
    #         # ax.annotate('{} {}'.format(str(int(tau)), eta), xy=(mdot_obs[i], llm_obs[i]),
    #         #             textcoords='data')  # plot numbers of stars
    #
    #         # if obs_cls.get_star_class(star_n) not in classes:
    #         #     plt.plot(x_coord[i], llm_obs[i], marker=obs_cls.get_clss_marker(star_n), markersize='9',
    #         #              color=obs_cls.get_class_color(star_n), ls='', mec='black',
    #         #              label='{}'.format(obs_cls.get_star_class(star_n)))  # plot color dots)))
    #         #     classes.append(obs_cls.get_star_class(star_n))
    #
    #
    #     print('\t__PLOT: total stars: {}'.format(len(obs_cls.stars_n)))
    #     print(len(x_coord), len(llm_obs))
    #
    #     # fit = np.polyfit(mdot_obs, llm_obs, 1)  # fit = set of coeddicients (highest first)
    #     # f = np.poly1d(fit)
    #     # fit_x_coord = np.mgrid[(mdot_obs.min() - 1):(mdot_obs.max() + 1):1000j]
    #
    #     mdot_grid = np.mgrid[(x_coord.min() - 1):(x_coord.max() + 1):100j]
    #     x_coord__, y_coord__ = Math.fit_plynomial(x_coord, llm_obs, 1, 100, mdot_grid)
    #     ax.plot(x_coord__, y_coord__, '-.', color='blue')
    #
    #     min_mdot, max_mdot = obs_cls.get_min_max('mdot')
    #     min_llm, max_llm = obs_cls.get_min_max_llm(l_or_lm, yc_val, use_gaia, check_star_lm_wne)
    #
    #     # ax.set_xlim(min_mdot - 0.2, max_mdot + 0.2)
    #     # ax.set_ylim(min_llm - 0.05, max_llm + 0.05)
    #
    #     ax.set_ylabel(Labels.lbls(l_or_lm))
    #     ax.set_xlabel(Labels.lbls('mdot'))
    #     ax.grid(which='major', alpha=0.2)
    #     ax.legend(bbox_to_anchor=(1, 1), loc='upper right', ncol=1)
    #
    #     print('Yc:{}'.format(yc_val))
    #
    #     if not clean:
    #         ax.text(0.9, 0.9, 'Yc:{}'.format(yc_val), style='italic',
    #                 bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center',
    #                 verticalalignment='center', transform=ax.transAxes)
    #         if use_gaia:
    #             ax.text(0.9, 0.75, 'GAIA', style='italic',
    #                     bbox={'facecolor': 'blue', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center',
    #                     verticalalignment='center', transform=ax.transAxes)
    #
    #     return ax
    #
    # @staticmethod
    # def plot_obs_mdot_llm(ax, obs_cls, l_or_lm, yc_val, clean = False):
    #     '''
    #
    #     :param ax:
    #     :param obs_cls:
    #     :param l_or_lm:
    #     :param yc_val:
    #     :param yc1:
    #     :param yc2:
    #     :return:
    #     '''
    #     classes = []
    #     classes.append('dum')
    #     mdot_obs = []
    #     llm_obs = []
    #
    #     # from Phys_Math_Labels import Opt_Depth_Analythis
    #
    #     for star_n in obs_cls.stars_n:
    #         i = -1
    #         mdot_obs = np.append(mdot_obs, obs_cls.get_num_par('mdot', star_n))
    #         llm_obs = np.append(llm_obs, obs_cls.get_num_par(l_or_lm, star_n, yc_val))
    #         eta = obs_cls.get_num_par('eta', star_n)
    #
    #         lm_err1, lm_err2 = obs_cls.get_star_lm_obs_err(star_n, yc_val)
    #         mdot1, mdot2 = obs_cls.get_star_mdot_obs_err(star_n, yc_val)
    #
    #         mdot_coord = [mdot1, mdot2, mdot2, mdot1]
    #         lm_coord = [lm_err1, lm_err1, lm_err2, lm_err2]
    #         ax.add_patch(patches.Polygon(xy=list(zip(mdot_coord, lm_coord)), fill=True, alpha=.4,
    #                                      color=obs_cls.get_class_color(star_n)))
    #
    #         if l_or_lm == 'lm':
    #
    #             llm1, llm2 = obs_cls.get_star_lm_err(star_n, yc_val)
    #                 # obs_cls.get_star_llm_evol_err(star_n, l_or_lm, yc_val, 1.0, 0.1)                  # ERRORS L/LM
    #             # mdot1, mdot2 = obs_cls.get_star_mdot_err(star_n, l_or_lm, yc_val, 1.0, 0.1, 'nugis')           # ERRORS Mdot
    #             mdot = obs_cls.get_num_par('mdot', star_n)
    #             ax.plot([mdot, mdot], [llm1, llm2], '-', color='white')
    #             #color=obs_cls.get_class_color(star_n)
    #
    #             # ax.errorbar(mdot_obs[i], llm_obs[i], yerr=[[llm1],  [llm2]], fmt='--.', color=obs_cls.get_class_color(star_n))
    #
    #
    #         ax.plot(mdot_obs[i], llm_obs[i], marker=obs_cls.get_clss_marker(star_n), markersize='9',
    #                  color=obs_cls.get_class_color(star_n), ls='', mec='black')  # plot color dots)))
    #         if not clean:
    #             ax.annotate('{}'.format(int(star_n)), xy=(mdot_obs[i], llm_obs[i]),textcoords='data')  # plot numbers of stars
    #
    #         # t = obs_cls.get_num_par('t', star_n)
    #         # ax.annotate('{}'.format("%.2f" % t), xy=(mdot_obs[i], llm_obs[i]), textcoords='data')  # plot numbers of stars
    #
    #
    #         # v_inf = obs_cls.get_num_par('v_inf', star_n)
    #         # tau_cl = Opt_Depth_Analythis(30, v_inf, 1., 1., mdot_obs[i], 0.20)
    #         # tau = tau_cl.anal_eq_b1(1.)
    #         # # # # ax.annotate(str(int(tau)), xy=(mdot_obs[i], llm_obs[i]), textcoords='data')  # plo
    #         # ax.annotate('{} {}'.format(str(int(tau)), eta), xy=(mdot_obs[i], llm_obs[i]),
    #         #             textcoords='data')  # plot numbers of stars
    #
    #         if obs_cls.get_star_class(star_n) not in classes:
    #             plt.plot(mdot_obs[i], llm_obs[i], marker=obs_cls.get_clss_marker(star_n), markersize='9',
    #                      color=obs_cls.get_class_color(star_n), ls='', mec='black',
    #                      label='{}'.format(obs_cls.get_star_class(star_n)))  # plot color dots)))
    #             classes.append(obs_cls.get_star_class(star_n))
    #
    #     print('\t__PLOT: total stars: {}'.format(len(obs_cls.stars_n)))
    #     print(len(mdot_obs), len(llm_obs))
    #
    #     # fit = np.polyfit(mdot_obs, llm_obs, 1)  # fit = set of coeddicients (highest first)
    #     # f = np.poly1d(fit)
    #     # fit_x_coord = np.mgrid[(mdot_obs.min() - 1):(mdot_obs.max() + 1):1000j]
    #
    #     mdot_grid = np.mgrid[(mdot_obs.min() - 1):(mdot_obs.max() + 1):100j]
    #     x_coord, y_coord = Math.fit_plynomial(mdot_obs, llm_obs, 1, 100, mdot_grid)
    #     ax.plot(x_coord, y_coord, '-.', color='blue')
    #
    #     min_mdot, max_mdot = obs_cls.get_min_max('mdot')
    #     min_llm, max_llm = obs_cls.get_min_max(l_or_lm, yc_val)
    #
    #     ax.set_xlim(min_mdot - 0.2, max_mdot + 0.2)
    #     ax.set_ylim(min_llm - 0.05, max_llm + 0.05)
    #
    #     ax.set_ylabel(Labels.lbls(l_or_lm))
    #     ax.set_xlabel(Labels.lbls('mdot'))
    #     ax.grid(which='major', alpha=0.2)
    #     ax.legend(bbox_to_anchor=(1, 1), loc='upper right', ncol=1)
    #
    #     print('Yc:{}'.format(yc_val))
    #
    #     if not clean:
    #         ax.text(0.9, 0.9, 'Yc:{}'.format(yc_val), style='italic',
    #                 bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center',
    #                 verticalalignment='center', transform=ax.transAxes)
    #
    #     return ax
    #     # ax.text(min_mdot, max_llm, 'Yc:{}'.format(yc_val), style='italic',
    #     #         bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10})
    #
    #     # l_grid = np.mgrid[5.2:6.:100j]
    #     # ax.plot(Physics.l_mdot_prescriptions(l_grid, ), l_grid, '-.', color='orange', label='Nugis & Lamers 2000')
    #     #
    #     # ax.plot(Physics.yoon(l_grid, 10 ** 0.02), l_grid, '-.', color='green', label='Yoon 2017')
    #
    # @staticmethod
    # def plot_obs_t_llm_mdot_int(ax, t_llm_mdot, obs_cls, l_or_lm, lim_t1 = None, lim_t2 = None, show_legend = True, clean = False):
    #
    #     if lim_t1 == None: lim_t1 = t_llm_mdot[0, 1:].min()
    #     if lim_t2 == None: lim_t2 = t_llm_mdot[0, 1:].max()
    #
    #     yc_val = t_llm_mdot[0, 0] #
    #
    #     classes = []
    #     classes.append('dum')
    #     x = []
    #     y = []
    #     for star_n in obs_cls.stars_n:
    #         xyz = obs_cls.get_xyz_from_yz(yc_val, star_n, l_or_lm, 'mdot',
    #                                       t_llm_mdot[0,1:], t_llm_mdot[1:,0], t_llm_mdot[1:,1:], lim_t1, lim_t2)
    #
    #         if xyz.any():
    #             x = np.append(x, xyz[0, 0])
    #             y = np.append(y, xyz[1, 0])
    #
    #             # print('Star {}, {} range: ({}, {})'.format(star_n,l_or_lm, llm1, llm2))
    #
    #             for i in range(len(xyz[0, :])):
    #
    #                 ax.plot(xyz[0, i], xyz[1, i], marker=obs_cls.get_clss_marker(star_n), markersize='9',
    #                          color=obs_cls.get_class_color(star_n), ls='', mec='black')  # plot color dots)))
    #                 if not clean:
    #                     ax.annotate(int(star_n), xy=(xyz[0, i], xyz[1, i]),
    #                                 textcoords='data')  # plot numbers of stars
    #
    #                 if obs_cls.get_star_class(star_n) not in classes:
    #                     ax.plot(xyz[0, i], xyz[1, i], marker=obs_cls.get_clss_marker(star_n), markersize='9',
    #                              color=obs_cls.get_class_color(star_n), mec='black', ls='',
    #                              label='{}'.format(obs_cls.get_star_class(star_n)))  # plot color dots)))
    #                     classes.append(obs_cls.get_star_class(star_n))
    #
    #                 # -------------------------OBSERVABLE ERRORS FOR L and Mdot ----------------------------------------
    #                 lm_err1, lm_err2 = obs_cls.get_star_lm_obs_err(star_n, yc_val)
    #                 ts1_b, ts2_b, ts1_t, ts2_t = obs_cls.get_star_ts_obs_err(star_n, t_llm_mdot, yc_val, lim_t1, lim_t2)
    #                 ts_coord = [ts1_b, ts2_b, ts2_t, ts1_t]
    #                 lm_coord = [lm_err1, lm_err1, lm_err2, lm_err2]
    #                 ax.add_patch(patches.Polygon(xy=list(zip(ts_coord, lm_coord)), fill=True, alpha=.7,
    #                                              color=obs_cls.get_class_color(star_n)))
    #
    #                 # ax.plot([xyz[0, i], xyz[0, i]], [lm_err1, lm_err2], '-',
    #                 #         color='gray')
    #
    #                 # ax.plot([ts_err1, ts_err2], [xyz[1, i], xyz[1, i]], '-',
    #                 #         color='gray')
    #                 # print('  :: Star {}. L/M:{}(+{}/-{}) ts:{}(+{}/-{})'
    #                 #       .format(star_n, "%.2f" % xyz[0, i], "%.2f" % np.abs(xyz[0, i]-lm_err2), "%.2f" % np.abs(xyz[0, i]-lm_err2),
    #                 #               "%.2f" % xyz[1, i],  "%.2f" % np.abs(xyz[1, i]-ts_err2), "%.2f" % np.abs(xyz[1, i]-ts_err1)))
    #                 # ax.plot([ts_err1, ts_err2], [lm_err1, lm_err2], '-', color='gray')
    #
    #                 #
    #
    #
    #
    #                 if l_or_lm == 'lm':
    #                     lm1, lm2 = obs_cls.get_star_lm_err(star_n, yc_val)
    #                     ts1, ts2 = obs_cls.get_star_ts_err(star_n, t_llm_mdot, yc_val, lim_t1, lim_t2)
    #                     ax.plot([ts1, ts2], [lm1, lm2], '-', color='white')
    #                             # color=obs_cls.get_class_color(star_n))
    #
    #                     # ax.plot([xyz[0, i], xyz[0, i]], [lm1, lm2], '-',
    #                     #         color=obs_cls.get_class_color(star_n))
    #                     # ax.plot([ts1, ts2], [xyz[1, i], xyz[1, i]], '-',
    #                     #         color=obs_cls.get_class_color(star_n))
    #
    #
    #                     # ax.add_patch(patches.Rectangle((xyz[0, i] - ts1, xyz[1, i] - lm1), ts2 + ts1, lm2 + lm1,
    #                     #                                alpha=.3, color=obs_cls.get_class_color(star_n)))
    #
    #                     # ax.add_patch(patches.Rectangle((xyz[0, i] - ts1, xyz[1, i] - lm1), ts2 + ts1, lm2 + lm1,
    #                     #                                alpha=.3, color=obs_cls.get_class_color(star_n)))
    #
    #                     # ax.plot([xyz[0, i] - ts1, xyz[1, i] - lm1], [xyz[0, i]+ts2, xyz[1, i] + lm2], '-', color=obs_cls.get_class_color(star_n))
    #                     # ax.plot([xyz[0, i] - ts1, xyz[0, i]+ts2], [xyz[1, i] - lm1, xyz[1, i] + lm2], '-', color=obs_cls.get_class_color(star_n))
    #
    #                     # ax.errorbar(xyz[0, i], xyz[1, i], yerr=[[lm1], [lm2]], fmt='--.', color = obs_cls.get_class_color(star_n))
    #                     # ax.errorbar(xyz[0, i], xyz[1, i], xerr=[[ts1], [ts2]], fmt='--.', color=obs_cls.get_class_color(star_n))
    #
    #
    #
    #     fit = np.polyfit(x, y, 1)  # fit = set of coeddicients (highest first)
    #     f = np.poly1d(fit)
    #     fit_x_coord = np.mgrid[(t_llm_mdot[0,1:].min()):(t_llm_mdot[0,1:].max()):1000j]
    #     ax.plot(fit_x_coord, f(fit_x_coord), '-.', color='blue')
    #
    #     ax.set_xlim(t_llm_mdot[0,1:].min(), t_llm_mdot[0,1:].max())
    #     ax.set_ylim(t_llm_mdot[1:,0].min(), t_llm_mdot[1:,0].max())
    #     # ax.text(0.9, 0.9,'Yc:{}'.format(yc_val), style='italic',
    #     #         bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
    #
    #     # ax.text(x.max(), y.max(), 'Yc:{}'.format(yc_val), style='italic',
    #     #         bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10})
    #     if show_legend:
    #         ax.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
    #
    #     return ax

    @staticmethod
    def plot_edd_kappa(ax, t_arr, sm_cls, opal_used, n_int_edd):
        from OPAL import Table_Analyze
        Table_Analyze.plot_k_vs_t = False  # there is no need to plot just one kappa in the range of availability
        clas_table_anal = Table_Analyze(opal_used, n_int_edd, False)

        for i in range(len(sm_cls)):  # self.nmdls
            mdl_m = sm_cls[i].get_cond_value('xm', 'sp')
            mdl_l = sm_cls[i].get_cond_value('l', 'sp')

            k_edd = Physics.edd_opacity(mdl_m, mdl_l)

            n_model_for_edd_k = clas_table_anal.interp_for_single_k(t_arr.min(), t_arr.max(), n_int_edd, k_edd)
            x = n_model_for_edd_k[0, :]
            y = n_model_for_edd_k[1, :]
            color = 'black'
            # lbl = 'Model:{}, k_edd:{}'.format(i, '%.2f' % 10 ** k_edd)
            ax.plot(x, y, '-.', color=color)  # , label=lbl)
            ax.plot(x[-1], y[-1], 'x', color=color)

        Table_Analyze.plot_k_vs_t = True
        return ax