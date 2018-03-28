#====================================================#
#
#  Methods to analyze the set of 'sm.data' files
#  to find critical ridii, that separates the
#  compact, iron bump solution and extendnd.
#
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
from Phys_Math_Labels import Errors
from Phys_Math_Labels import Math
from Phys_Math_Labels import Physics
from Phys_Math_Labels import Constants
from Phys_Math_Labels import Labels

from OPAL import Read_Table
from OPAL import Row_Analyze
from OPAL import Table_Analyze
from OPAL import OPAL_Interpol
from OPAL import New_Table

from FilesWork import Read_Observables
from FilesWork import Read_Plot_file
from FilesWork import Read_SM_data_file

class Criticals:
    def __init__(self, smfiles, out_dir, plot_dir, dirs_not_to_be_included):

        self.input_dirs = smfiles[0].split('/')[:-1]
        # print(self.input_dirs)
        self.dirs_not_to_be_included = dirs_not_to_be_included # name of folders that not to be in the name of out. file.

        self.out_dir = out_dir
        self. plot_dir = plot_dir

        self.num_files = smfiles
        self.mdl = []
        self.smdl =[]

        for file in smfiles:
            self.mdl.append(Read_SM_data_file.from_sm_data_file(file))

        self.sort_smfiles('mdot', -1)

    def get_boundary(self, u_min):
        '''
        RETURNS ' bourders.min() ' - min radius among all the models, where u exceeds the 'u_min'
        :param u_min:
        :return:
        '''
        bourders = []

        for i in range(len(self.num_files)):
            u = self.smdl[i].get_col('u')
            r = self.smdl[i].get_col('r')
            for i in range(len(r)):
                if u[i] > u_min:
                    # ax1.axvline(x=r[i], color='red')
                    bourders = np.append(bourders, r[i])
                    break

        return bourders.min()

    def sort_smfiles(self, v_n, where = -1, descending=True):
        '''

        :param v_n: what value to use to sort sm.files
        :param where: where value si caken (surface -1 or core 0 )
        :param descending: if True, sm.files are sorted by descending order of the chosen parameter.
        :return: NOTHING (changes the smdl[])
        '''


        i_and_mdots = []
        for i in range(len(self.num_files)):
            i_and_mdots = np.append(i_and_mdots, [i, self.mdl[i].get_col(v_n)[where]])

        i_and_mdots_sorted = np.sort(i_and_mdots.view('f8, f8'), order=['f1'], axis=0).view(np.float)
        i_and_mdots_reshaped = np.reshape(i_and_mdots_sorted, (len(self.num_files), 2))

        if descending:
            i_and_mdots_reshaped_inversed = np.flip(i_and_mdots_reshaped, 0) # flip for ascending order
        else:
            i_and_mdots_reshaped_inversed = i_and_mdots_reshaped # no flipping


        sorted_by_mdot_files = []
        for i in range(len(self.num_files)):
            sorted_by_mdot_files.append(self.num_files[np.int(i_and_mdots_reshaped_inversed[i][0])])


        for file in sorted_by_mdot_files:
            self.smdl.append(Read_SM_data_file.from_sm_data_file(file))

        # for i in range(len(self.num_files)):
        #     print(self.smdl[i].get_col(v_n)[where])

    def sonic_criticals(self, depth, add_sonic_vals, show_plot):
        def get_boundary(u_min):
            '''
            RETURNS ' bourders.min() ' - min radius among all the models, where u exceeds the 'u_min'
            :param u_min:
            :return:
            '''
            bourders = []

            for i in range(len(self.num_files)):
                u = self.smdl[i].get_col('u')
                r = self.smdl[i].get_col('r')
                for i in range(len(r)):
                    if u[i] > u_min:
                        # ax1.axvline(x=r[i], color='red')
                        bourders = np.append(bourders, r[i])
                        break

            return bourders.min()
        def get_sonic_t(r, u, mu, t):
            '''
            RETURNS rs0, ts0 (if ts0 > 5.2, else - ERROR)
            :param r:
            :param u:
            :param mu:
            :param t:
            :return:
            '''
            if len(r) != len(u) != len(mu) != len(t):
                raise ValueError('Different length of arrays: len(r){} != len(u){} != len(mu){} != len(t){}'
                                 .format(len(r), len(u), len(mu), len(t)))

            ts_arr = np.log10( (mu * Constants.m_H * (u * 100000) ** 2) / Constants.k_b )
            rs, ts = Math.interpolated_intercept(r, ts_arr, t)

            if rs.any():
                rs0 = rs[0][0]
                ts0 = ts[0][0]

                if ts0 < 5.2:
                    raise ValueError('Ts({})<5.2'.format(ts0))

            else:
                rs0 = None
                ts0 = None

            return rs0, ts0, r, ts_arr

        def crop_ends(x, y):
            '''
            In case of 'wierd' vel/temp profile with rapidly rising end, first this rising part is to be cut of
            before maximum can be searched for.
            :param x:
            :param y:
            :return:
            '''
            x_mon = x
            y_mon = y

            non_monotonic = True

            while non_monotonic:

                if len(x_mon) <= 10:
                    return x, y
                    # raise ValueError('Whole array is removed in a searched for monotonic part.')

                if y_mon[-1] > y_mon[-2]:
                    y_mon = y_mon[:-1]
                    x_mon = x_mon[:-1]
                    # print(x_mon[-1], y_mon[-1])
                else:
                    non_monotonic = False

            return Math.find_nearest_index(x, x_mon[-1])

        def all_values_array(i_model, rs_p, ts_p, min_indx, add_sonic_vals):

            out_array = []

            # --- --- GET ARRAYS AND VALUES AND APPENDING TO OUTPUT ARRAY --- --- --- --- --- --- --- ---

            r = self.smdl[i_model].get_col('r')[min_indx:]
            t = self.smdl[i_model].get_col('t')[min_indx:]

            out_array = np.append(out_array, self.smdl[i_model].get_col('l')[-1])  # appending 'l'    __1__
            out_array = np.append(out_array, self.smdl[i_model].get_col('xm')[-1])  # appending 'xm'   __2__
            out_array = np.append(out_array, self.smdl[i_model].get_col('He4')[0])  # appending 'Yc'   __3__

            out_array = np.append(out_array, self.smdl[i_model].get_col('mdot')[-1])  # appending 'mdot' __4__
            out_array = np.append(out_array, rs_p)  # appending 'mdot' __5__
            out_array = np.append(out_array, ts_p)  # appending 'mdot' __6__

            # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

            val_array = []
            for v_n_cond in add_sonic_vals:

                if len(v_n_cond.split('-')) > 2:
                    raise NameError(
                        'For *add_sonic_vals* use format *var_name-location*, (given: {}) where var_name is '
                        'one of the BEC sm.data variables and location is *core = [-1] surface=[0]',
                        'sp = [sonic_point_interpolated]'.format(v_n_cond))

                v_n = v_n_cond.split('-')[0]
                cond = v_n_cond.split('-')[-1]

                if cond != 'sp':
                    var_val = self.smdl[i_model].get_cond_value(v_n,
                                                                cond)  # assuming that condition is not required interp

                else:
                    ''' Here The Interpolation of v_n is Done'''

                    v_n_val_arr = self.smdl[i_model].get_col(v_n)[min_indx:]
                    f = interpolate.InterpolatedUnivariateSpline(r, v_n_val_arr)
                    var_val = f(rs_p)

                    if len([var_val]) > 1:
                        raise ValueError('More than one solution found for *{}* sonic value: ({})'.format(v_n, var_val))

                val_array = np.append(val_array, var_val)

            if len(val_array) != len(add_sonic_vals):
                raise ValueError('len(val_array)[{}] != len(add_sonic_vals)[{}]'
                                 .format(len(val_array), len(add_sonic_vals)))

            for i in range(len(val_array)):  # 5 is a number of main values: [ l, m, Yc, ts, rs ]
                out_array = np.append(out_array, val_array[i])  # appending 'v_ns' __n__

            return out_array

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()

        tlt = 'VELOCITY PROFILE'
        plt.title(tlt, loc='left')

        mdots   = np.array([0.])
        # rs_ts   = np.array([0.,0.])
        r_u_mdot_max = np.array([0., 0., 0.])
        r_t_mdot_max = np.array([0., 0., 0.])
        # rs_ts_mdot   = np.array([0.,0.,0.])

        out_array = np.zeros(len(add_sonic_vals) + 6) # where 6 are: [l, m, Yc, mdot, rs, ts] # always include

        min_ind = 0
        max_ind = -1
        r_min = get_boundary(0.1) # setting the lower value of r that above which the analysis will take place
        for i in range(len(self.num_files)):
            r = self.smdl[i].get_col('r')
            u = self.smdl[i].get_col('u')
            t = self.smdl[i].get_col('t')
            min_ind = Math.find_nearest_index(r, r_min)
            max_ind = crop_ends(r, u)

            ax1.plot(r[min_ind:], u[min_ind:], '.', color='black')
            ax2.plot(r[min_ind:], t[min_ind:], '.', color='orange')


        '''------------------------------------------MAIN CYCLE------------------------------------------------------'''
        for i in range(len(self.num_files)):

            r  = self.smdl[i].get_col('r')
            u  = self.smdl[i].get_col('u')
            mu = self.smdl[i].get_col('mu')
            t  = self.smdl[i].get_col('t')
            u_s= self.smdl[i].get_sonic_u()
            mdot_u=self.smdl[i].get_col('mdot')[-1]
            mdots = np.append(mdots, mdot_u)
            print('\t__Initical Array Length: {}'.format(len(r)))


            min_ind = Math.find_nearest_index(r, r_min)


            # ----------------------- R U ----------------------------
            r  = r[min_ind:max_ind]
            u  = u[min_ind:max_ind]
            u_s= u_s[min_ind:max_ind]
            t = t[min_ind:max_ind]
            mu = mu[min_ind:max_ind]

            print('\t__Cropped Array Length: {}'.format(len(r)))

            int_r  = np.mgrid[r[0]:r[-1]:depth*1j]
            int_u  = Math.interp_row(r, u, int_r)

            # ax1.plot(r, u, '.', color='black')
            ax1.plot(r, u_s, '--', color='black')
            ax1.annotate(str('%.2f' % mdot_u), xy=(r[-1], u[-1]), textcoords='data')
            ax1.plot(int_r, int_u, '-', color='gray')

            # ------------------------R T --------------------------------

            ts_arr = np.log10((mu * Constants.m_H * (u * 100000) ** 2) / Constants.k_b)


            int_r = np.mgrid[r[0]:r[-1]:depth * 1j]
            int_t  = Math.interp_row(r, t, int_r)
            int_ts_arr=Math.interp_row(r, ts_arr, int_r)

            ax2.plot(r, ts_arr, '.', color='orange')
            ax2.plot(int_r, int_t,'-',color='orange')
            ax2.plot(int_r, int_ts_arr, '--', color='orange')


            r_u_max_p, u_max_p = Math.get_max_by_interpolating(int_r, int_u)      # MAXIMUM VALUES OF VELOCITY
            r_u_mdot_max = np.vstack((r_u_mdot_max, [r_u_max_p[0], u_max_p[0], mdot_u]))
            r_t_max_p, t_max_p = Math.get_max_by_interpolating(int_r, int_ts_arr) # MAXIMUM VALUES OF TEMPERATURE
            r_t_mdot_max = np.vstack((r_t_mdot_max, [r_t_max_p[0], t_max_p[0], mdot_u]))

            ax1.plot(r_u_max_p, u_max_p, 'x', color='red')
            ax2.plot(r_t_max_p, t_max_p, 'x', color='red')


            # --- --- ---| SONIC POINT PARAMTERS |--- --- ---
            rs_p, ts_p = Math.interpolated_intercept(int_r, int_ts_arr, int_t)     # SONIC TEPERATURE
            if rs_p.any():
                rs_p = rs_p[0][0]
                ts_p = ts_p[0][0]
                ax2.plot(rs_p, ts_p, 'X', color='red')
                ax2.annotate(str('%.2f' % ts_p), xy=(rs_p, ts_p), textcoords='data')

                row = all_values_array(i, rs_p, ts_p, min_ind, add_sonic_vals)
                out_array = np.vstack((out_array, row))



        r_u_mdot_max = np.delete(r_u_mdot_max, 0, 0) # removing the 0th row with zeros
        r_t_mdot_max = np.delete(r_t_mdot_max, 0, 0)
        mdots = np.delete(mdots, 0, 0)


        if len(mdots) != len(r_u_mdot_max[:,0]):
            raise ValueError('len(mdots){} != len(r_u_mdot_max[:,0]){}'.format(len(mdots), len(r_u_mdot_max[:,0])))

        if len(mdots) ==0:
            raise ValueError('len(mdots) = 0')

        ax1.plot(r_u_mdot_max[:,0],  r_u_mdot_max[:,1],  '-.', color='blue')

        # plt.show()
        # print(out_array)




        def cross(mdot, r1, u1, r2, u2, mdot_maxs, u_ot_t = 'u'):
            '''
            Finds the delta = (u_i_max - cs_i) where u_i is a maximum of u profile along the r, cs_i - point along sonic
            velocity profile, that lies on the line thac connects the u_i maximums.
            As close the mass loss to a critical one, the closer delta to 0 ( u_i_max = cs_i at the critical
            (deflection) point)

            :param mdot:
            :param r1: full length of a sonic profile (r)
            :param u1: values of sonic velocity along (r)
            :param r2: set of 'r' points for maximums of velocity profiles for every mass loss
            :param u2: values of velocity at 'r' points
            :param mdot_maxs: mdot values of every point 'r,u' above
            :return:
            '''



            r1_u1 = []
            for i in range(len(r1)):
                r1_u1 = np.append(r1_u1, [r1[i], u1[i]])

            r1_u1 = np.sort(r1_u1.view('f8, f8'), order=['f1'], axis=0).view(np.float)
            r1_u1_sort = np.reshape(r1_u1, (len(r1), 2))


            r2_u2_mdot = []
            for i in range(len(r2)):
                r2_u2_mdot = np.append(r2_u2_mdot, [r2[i], u2[i], mdot_maxs[i]])

            r2_u2_mdot = np.sort(r2_u2_mdot.view('f8, f8, f8'), order=['f1'], axis=0).view(np.float)
            r2_u2_mdot_sort = np.reshape(r2_u2_mdot, (len(r2), 3))

            r1 = r1_u1_sort[:,0]
            u1 = r1_u1_sort[:,1]

            r2    = r2_u2_mdot_sort[:,0]
            u2    = r2_u2_mdot_sort[:,1]
            mdots = r2_u2_mdot_sort[:,2]

            #
            # u_rmax1 = u1[np.where(r1 == r1.max())]
            # u_rmax2 = u2[np.where(r2 == r2.max())]
            #
            # r_umin1_i = r1[np.where(u1 == u1.min())]
            # r_umin2_i = r2[np.where(u2 == u2.min())]


            i_u2_where_u1_max = Math.find_nearest_index(u2, u1.max()) # needed to avoide violent behaviour in high mdot
            u2_crop = u2[:i_u2_where_u1_max]            # otherwise there was crossing with sonic profiles at temps 5.9,
            r2_crop = r2[:i_u2_where_u1_max]


            u_lim1 = np.array([u1.min(), u2_crop.min()]).max()
            u_lim2 = np.array([u1.max(), u2_crop.max()]).min()

            if u_lim2 < u_lim1:
                raise ValueError('u_lim1({}) < u_lim2({})'.format(u_lim1, u_lim2))

            u_grid = np.mgrid[u_lim2:u_lim1:1000*1j]

            if u_grid.max() > u2.max() or u_grid.max() > u1.max():
                raise ValueError('u_grid.max({}) > u2.max({}) or u_grid.max({}) > u1.max({})'
                                 .format(u_grid.max(), u2.max(), u_grid.max(), u1.max()))
            #
            # if u_grid.min() < u2.min() or u_grid.min() < u1.min():
            #     raise ValueError('u_grid.min({}) < u2.min({}) or u_grid.min({}) < u1.min({})'
            #                      .format(u_grid.min(), u2.min(), u_grid.min(), u1.min()))


            f1 = interpolate.InterpolatedUnivariateSpline(u1, r1)
            r1_grid = f1(u_grid)

            # if u_ot_t == 't':
                # print('\t')
                # print('u1:[{} {}] u_lim1:{} u_lim2:{}'.format(u1.min(), u1.max(), u_lim1, u_lim2))
                # print('u2:[{} {}] u_lim1:{} u_lim2:{}'.format(u2.min(), u2.max(), u_lim1, u_lim2))
                #


                # ax2.plot(r2_crop, u2_crop, 'o', color='magenta')

            f2 = interpolate.InterpolatedUnivariateSpline(u2_crop, r2_crop) # cropping is done to cut the high mdot prob
            r2_grid = f2(u_grid)

            # r1_grid = Math.interp_row(u1, r1, u_grid)
            # r2_grid = Math.interp_row(u2, r2, u_grid)

            if u_ot_t == 'u':
                ax1.plot(r1_grid, u_grid, '-.', color='green')
                ax1.plot(r2_grid, u_grid, '-.', color='green')
            else:
                ax2.plot(r1_grid, u_grid, '-.', color='green')
                ax2.plot(r2_grid, u_grid, '-.', color='green')

            uc, rc = Math.interpolated_intercept(u_grid, r1_grid, r2_grid)
            if uc.any():  # if there is an intersections between sonic vel. profile and max.r-u line
                uc0 = uc[0][0]
                rc0 = rc[0][0]

                if u_ot_t == 'u':
                    ax1.plot(rc0, uc0, 'X', color='green')
                    ax1.annotate(str('%.2f' % mdot), xy=(rc0, uc0), textcoords='data')
                if u_ot_t == 't':
                    ax2.plot(rc0, uc0, 'X', color='green')
                    ax2.annotate(str('%.2f' % mdot), xy=(rc0, uc0), textcoords='data')

                delta = u2[np.where(mdots == mdot)] - uc0
                # print(uc, rc, '\t', delta)
                # print('Delta: ' , delta_ut, )

                return delta


            # if u_min2 < u_min1:        # if there is a common area in terms of radii
            #     if r_umin1 > r_umin2:   # if there is a common area in case of velocity
            #         u_lim1 = u_min1
            #         u_lim2 = u_max1
            #
            #         u_grid = np.mgrid[u_lim2:u_lim1:1000*1j]
            #
            #         r1_grid = Math.interp_row(u1, r1, u_grid)
            #         r2_grid = Math.interp_row(u2, r2, u_grid)
            #
            #         ax1.plot(r1_grid, u_grid, '-', color='green')
            #         ax1.plot(r2_grid, u_grid, '-', color='green')
            #
            #         uc, rc = Math.interpolated_intercept(u_grid, r1_grid, r2_grid)
            #         if uc.any(): # if there is an intersections between sonic vel. profile and max.r-u line
            #
            #             uc0 = uc[0][0]
            #             rc0 = rc[0][0]
            #
            #             ax1.plot(rc0, uc0, 'X', color='green')
            #             ax1.annotate(str('%.2f' % mdot), xy=(rc0, uc0), textcoords='data')
            #
            #             delta = u2[np.where(mdots == mdot)] - uc0
            #
            #             # print('Delta: ' , delta_ut, )
            #
            #             return delta
            #         else:
            #             print('\t__Warning. No common area in velocity found: '
            #                   'r_at_u_min1:{} > r_at_u_min2{}'.format(r_umin1, r_umin2))
            #     else:
            #         print('\t__Warning. No common area in radii found: '
            #               'r_at_u_min2:{} < r_at_u_min2{}'.format(r_umin1, r_umin2))

        def get_mdot_delta(r_ut_mdot_max, u_or_t):

            mdot_delta_ut = []

            n = 0
            for i in range(len(self.num_files)):
                r = self.mdl[i].get_col('r')[min_ind:]
                # u = self.mdl[i].get_col('u')[min_ind:]
                # mu = self.mdl[i].get_col('mu')[min_ind:]
                t = self.mdl[i].get_col('t')[min_ind:]
                # ts_arr = np.log10((mu * Constants.m_H * (u * 100000) ** 2) / Constants.k_b)[min_ind:]
                u_s = self.mdl[i].get_sonic_u()[min_ind:]

                mdot = self.mdl[i].get_col('mdot')[-1]

                if u_or_t == 'u':
                    delta_ut = cross(mdot, r, u_s, r_ut_mdot_max[1:, 0], r_ut_mdot_max[1:, 1], r_ut_mdot_max[1:, 2], u_or_t)
                    # print(delta_ut, u_or_t)
                else:
                    delta_ut = cross(mdot, r, t,   r_ut_mdot_max[1:, 0], r_ut_mdot_max[1:, 1], r_ut_mdot_max[1:, 2], u_or_t)
                    # print(delta_ut, u_or_t)

                if delta_ut != None:
                    mdot_delta_ut = np.append(mdot_delta_ut, [mdot, delta_ut])
                    n = n + 1

            if len(mdot_delta_ut) == 0:
                raise ValueError('mdot_delta_ut is not found at all for <{}>'.format(u_or_t))

            mdot_delta_ut = np.sort(mdot_delta_ut.view('f8, f8'), order=['f0'], axis=0).view(np.float)
            mdot_delta_ut_shape = np.reshape(mdot_delta_ut, (n, 2))

            mdot     = mdot_delta_ut_shape[:, 0]
            delta_ut = mdot_delta_ut_shape[:, 1]

            crit_mdot_u = Math.solv_inter_row(mdot, delta_ut, 0.)  # Critical Mdot when the delta_ut == 0

            print('\t\t crit_mdot_u', crit_mdot_u)

            if not crit_mdot_u.any():
                raise ValueError('Critical Mdot is not found.')
            else:
                print('\t__Critical Mdot: {} (for: {})'.format(crit_mdot_u, u_or_t))

            return mdot, delta_ut


        mdot_arr, delta_arr = get_mdot_delta(r_u_mdot_max, 'u')

        if delta_arr.min() > 0. and delta_arr.max() > 0.:
            raise ValueError('if delta_arr.min({}) and delta_arr.max({}) > 0 : '
                             'peak of vel. profs do not cross sonic. vel.'.format(delta_arr.min(), delta_arr.max()))

        if delta_arr.min() < 0. and delta_arr.max() < 0.:
            raise ValueError('if delta_arr.min({}) and delta_arr.max({}) < 0 : '
                             'vel. profile does not crossing the sonic val.'.format(delta_arr.min(), delta_arr.max()))

        crit_mdot_u = Math.solv_inter_row(mdot_arr, delta_arr, 0.)
        if len(crit_mdot_u) > 1:
            raise ValueError('More than one mdot_cr. found. {}'.format(crit_mdot_u))

        ax3 = fig.add_axes([0.18, 0.18, 0.25, 0.25])
        ax3.set_xlabel(Labels.lbls('mdot'))
        ax3.set_ylabel('$u_i - cs_i$')
        ax3.grid()

        ax3.plot(mdot_arr, delta_arr, '-', color='black')
        ax3.plot(crit_mdot_u, 0., 'x', color='black')
        ax3.annotate('({}, {})'.format('%.3f' % crit_mdot_u, 0. ), xy=(crit_mdot_u, 0.),
                     textcoords='data')


        mdot_arr, delta_arr = get_mdot_delta(r_t_mdot_max, 't')
        crit_mdot_t = Math.solv_inter_row(mdot_arr, delta_arr, 0.)


        # --- --- FROM CRITICAL MDOT GET CRITICAL R, T, OTHERS --- --- ---

        r_u_mdot_max = np.sort(r_u_mdot_max.view('f8, f8, f8'), order=['f2'], axis=0).view(np.float)
        r_u_mdot_max = np.reshape(r_u_mdot_max, (len(self.num_files), 3))

        r_t_mdot_max = np.sort(r_t_mdot_max.view('f8, f8, f8'), order=['f2'], axis=0).view(np.float)
        r_t_mdot_max = np.reshape(r_t_mdot_max, (len(self.num_files), 3))


        f = interpolate.InterpolatedUnivariateSpline(r_t_mdot_max[:, 2], r_t_mdot_max[:, 1])  # t_crit
        crit_t = f(crit_mdot_t)

        f = interpolate.InterpolatedUnivariateSpline(r_u_mdot_max[:, 2], r_u_mdot_max[:, 0])  # r_crit
        crit_r = f(crit_mdot_u)

        ax2.plot(r_t_mdot_max[:, 0], r_t_mdot_max[:, 1], '-.', color='red')
        ax2.plot(crit_r, crit_t, 'X', color='red')
        ax1.annotate('({}, {})'.format('%.2f' % crit_r, '%.2f' % crit_t), xy=(crit_r, crit_t), textcoords='data')
        print('\t__Criticals: r={}, t={}, mdot={}'.format('%.4f' % crit_r, '%.4f' % crit_t, '%.4f' % crit_mdot_u))


        '''----------------------------------------------SUBPLOTS----------------------------------------------------'''

        ax4 = fig.add_axes([0.18, 0.50, 0.25, 0.25])
        ax4.set_xlabel(Labels.lbls('mdot'))
        ax4.set_ylabel('$u_i - cs_i$')
        ax4.grid()
        ax4.set_xlabel(Labels.lbls('mdot'))
        ax4.set_ylabel(Labels.lbls('t'), color='r')

        ax4.plot(r_t_mdot_max[:, 2], r_t_mdot_max[:, 1], '-', color='red')
        ax4.plot(crit_mdot_t, crit_t, 'x', color='red')
        ax4.annotate('({}, {})'.format('%.3f' % crit_mdot_t, '%.3f' % crit_t, ), xy=(crit_mdot_t, crit_t),
                     textcoords='data')
        ax4.tick_params('y', colors='r')

        ax5 = ax4.twinx()
        ax5.plot(r_u_mdot_max[:, 2], r_u_mdot_max[:, 0], '-', color='blue')
        ax5.plot(crit_mdot_u, crit_r, 'x', color='blue')
        ax5.annotate('({}, {})'.format('%.3f' % crit_mdot_u, '%.3f' % crit_r,), xy=(crit_mdot_u, crit_r),
                     textcoords='data')
        ax5.set_xlabel(Labels.lbls('mdot'))
        ax5.set_ylabel(Labels.lbls('r'), color='b')
        ax5.tick_params('y', colors='b')






        def subplot_mdot_t_max(mdot, t_max, depth):

            fig = plt.figure()

            int_mdot = np.mgrid[mdot.max():mdot.min():depth*1j]
            int_t_max = Math.interp_row(mdot, t_max, int_mdot)

            # ax3 = fig.add_axes([0.23, 0.5, 0.23, 0.23])
            ax3 = fig.add_subplot(111)
            ax3.plot(mdot, t_max, '.', color='black')
            ax3.plot(int_mdot, int_t_max, '-', color='black')
            ax3.set_xlabel(Labels.lbls('mdot'))
            ax3.set_ylabel(Labels.lbls('t'))
            ax3.grid()
            plt.show()
            # ax3.axhline(y=u_s_cr_c, color='gray', linestyle='--', label='Son_vel: {}'.format('%.3f' % u_s_cr_c))
        def subplot_mdot_r_max(mdot, r_max, depth):

            fig = plt.figure()

            int_mdot = np.mgrid[mdot.max():mdot.min():depth*1j]
            int_r_max = Math.interp_row(mdot, r_max, int_mdot)

            # ax3 = fig.add_axes([0.23, 0.5, 0.23, 0.23])
            ax3 = fig.add_subplot(111)
            ax3.plot(mdot, r_max, '.', color='black')
            ax3.plot(int_mdot, int_r_max, '-', color='black')
            ax3.set_xlabel(Labels.lbls('mdot'))
            ax3.set_ylabel(Labels.lbls('r'))
            ax3.grid()
            plt.show()
            # ax3.axhline(y=u_s_cr_c, color='gray', linestyle='--', label='Son_vel: {}'.format('%.3f' % u_s_cr_c))

        # subplot_mdot_t_max(mdots[1:], r_t_mdot_max[1:, 1], depth)
        # subplot_mdot_r_max(mdots[1:], r_t_mdot_max[1:, 0], depth)


        ax2.set_ylabel(Labels.lbls('ts'), color='r')
        ax2.tick_params('u', colors='r')
        ax2.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)

        ax1.tick_params('u', colors='b')
        ax1.set_xlabel(Labels.lbls('r'))
        ax1.set_ylabel(Labels.lbls('u'))
        ax1.grid(which='both')
        ax1.grid(which='minor', alpha=0.2)
        ax1.grid(which='major', alpha=0.2)
        ax1.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
        plot_name = self.plot_dir + 'critical_radius.pdf'
        plt.savefig(plot_name, dpi=240, format='pdf')

        if show_plot:
            plt.show()


        l = out_array[-1,0]  # choosing the last mpdel to get l, m, yc as low mdot would affect these them the least
        m = out_array[-1,1]
        yc= out_array[-1,2]

        out_array[0,0] = l   # inserting l, m, yc into the first row of the output array
        out_array[0,1] = m
        out_array[0,2] = yc

        out_array[0,3] = crit_mdot_u # inserting criticals in the first row
        out_array[0,4] = crit_r
        out_array[0,5] = crit_t


        print(out_array)

        print('\t__Note. Critical Values are found and written in the FIRST row (out of {}) in output file.'.
              format(len(self.num_files) + 1))

        tablehead = '{}  {}     {}     {}  {} {}'\
            .format('log(L)', 'M(Msun)', 'Yc', 'l(Mdot)', 'Rs(Rsun)', 'log(Ts)')


        tmp = ''
        for v_n in add_sonic_vals:
            tmp = tmp + ' {}'.format(v_n)

        extended_head = tablehead + tmp
        print(extended_head)

        # --- --- --- MAKING A OUTPUT FILE NAME OUT OF FOLDERS THE SM.DATA FILES CAME FROM --- --- ---
        out_name = 'SP'
        for i in range(len(self.input_dirs)):
            if self.input_dirs[i] not in self.dirs_not_to_be_included and self.input_dirs[i] != '..':
                out_name = out_name + self.input_dirs[i]
                if i < len(self.input_dirs) - 1:
                    out_name = out_name + '_'
        out_name = out_name + '.data'

        print('Results are saved in: {}'.format(self.out_dir + out_name))
        np.savetxt(self.out_dir + out_name, out_array, '%.5f', '  ', '\n', extended_head, '')

    # def velocity_profile(self):
    #     fig = plt.figure()
    #     ax1 = fig.add_subplot(111)
    #
    #     tlt = 'VELOCITY PROFILE'
    #     plt.title(tlt, loc='left')
    #     crit_sonic_values = []
    #
    #     r_u_mdot_max = []
    #     lmdot = []
    #     long_r = np.zeros(1)
    #     long_i = -1
    #
    #     # --- --- --- FUNCTIONS --- --- ---
    #     def get_sonic_t(r, u, mu, t):
    #         '''
    #         RETURNS rs0, ts0 (if ts0 > 5.2, else - ERROR)
    #         :param r:
    #         :param u:
    #         :param mu:
    #         :param t:
    #         :return:
    #         '''
    #         if len(r) != len(u) != len(mu) != len(t):
    #             raise ValueError('Different length of arrays: len(r){} != len(u){} != len(mu){} != len(t){}'
    #                              .format(len(r), len(u), len(mu), len(t)))
    #
    #         ts_arr = np.log10( (mu * Constants.m_H * (u * 100000) ** 2) / Constants.k_b )
    #         rs, ts = Math.interpolated_intercept(r, ts_arr, t)
    #
    #         if rs.any():
    #             rs0 = rs[0][0]
    #             ts0 = ts[0][0]
    #
    #             if ts0 < 5.2:
    #                 raise ValueError('Ts({})<5.2'.format(ts0))
    #
    #         else:
    #             rs0 = None
    #             ts0 = None
    #
    #         return rs0, ts0, r, ts_arr
    #
    #
    #
    #     # --- --- --- MAIN --- --- ---
    #     ax2 = ax1.twinx()
    #
    #
    #
    #
    #     for i in range(len(self.num_files)):
    #
    #         min_ind = Math.find_nearest_index(self.mdl[i].get_col('r'), 0.9)
    #
    #         r = self.mdl[i].get_col('r')[min_ind:]
    #         u = self.mdl[i].get_col('u')[min_ind:]
    #         mu= self.mdl[i].get_col('mu')[min_ind:]
    #         t = self.mdl[i].get_col('t')[min_ind:]
    #         u_s = self.mdl[i].get_sonic_u()[min_ind:]
    #
    #         lmdot =  np.append( lmdot,  self.mdl[i].get_col('mdot')[-1] )
    #
    #         # --- PLOTTING velocity and sonic velocity profiles ---
    #         ax1.plot(r, u, '-', color='black')
    #         ax1.plot(r, u_s, '-', color='gray')
    #         ax1.annotate(str('%.2f' % lmdot[i]), xy=(r[-1], u[-1]), textcoords='data')
    #
    #
    #         # --- get maximum of every velocity profile ---
    #         r_m, u_m =  Math.get_max_by_interpolating(r, u) # use of 1d array for further sorting
    #         r_u_mdot_max = np.append(r_u_mdot_max, [r_m, u_m, [lmdot[i]] ] ) # [] for lmdot bacause r_cr_c and u_s_cr_c are lists
    #         ax1.plot(r_m, u_m, '*', color='red')
    #
    #         # --- Getting sonic temperature ---
    #         rs, ts, r, ts_arr = get_sonic_t(r, u, mu, t)
    #         if rs != None:
    #             ax2.plot(r,t,'-',color='yellow')
    #             ax2.plot(r, ts_arr, '-.', color='gray')
    #             ax2.plot(rs, ts, 'x', color='black')
    #             ax2.annotate(str('%.2f' % ts), xy=(rs, ts), textcoords='data')
    #
    #         rts_m, ts_m = Math.get_max_by_interpolating(r, ts_arr)
    #
    #         print('_______', rts_m, ts_m)
    #         if rts_m != None:
    #             ax2.plot(rts_m, ts_m, 'x', color='red')
    #
    #
    #         # -- getting the longest (in r) model, so leter to get a critical radii from it
    #         if long_r.max() < r[-1]:
    #             long_r = np.append(long_r, r[-1])
    #             long_i = i
    #
    #         # --- filling the array for the output
    #         l = self.mdl[i].get_col('l')[-1]
    #         m = self.mdl[i].get_col('xm')[-1]
    #         yc= self.mdl[i].get_col('He4')[0]
    #
    #         if rs!=None:
    #             crit_sonic_values = np.append(crit_sonic_values, [l, m, yc, lmdot[i], rs, ts])
    #         else:
    #             crit_sonic_values = np.append(crit_sonic_values, [l, m, yc, lmdot[i], 0., 0.])
    #
    #     # as it r coord first increasing than decreasing - you cannot use it for interpolation,
    #     # u - coord, which is velocity, is decreasing if mdot is decreasing (or vise versa), but it behaves monotonicall
    #     # hence, it can be used as a 'r' coordinate, while radius is a 'u'.
    #     # print(r_u_mdot_max)
    #
    #     r_u_mdot_max_sorted = np.sort(r_u_mdot_max.view('f8, f8, f8'), order=['f1'], axis=0).view(np.float)
    #     r_u_mdot_reshaped = np.reshape(r_u_mdot_max_sorted, (len(self.num_files), 3))
    #
    #     new_r   = r_u_mdot_reshaped[:,0]
    #     new_u   = r_u_mdot_reshaped[:,1]
    #     new_mdot= r_u_mdot_reshaped[:,2]
    #
    #     def check_limits(old_r, new_r, old_u, new_u):
    #         if old_r.min() < new_r.min():
    #             print('\t__Error: old_r.min({}) < new_r.min({})'.format(old_r.min(), new_r.min()))
    #             return 'new_r_too_small'
    #
    #         if  new_r.max() > old_r.max()*1.2: # 1.2 is a criterium
    #             print('\t__Error: old_r.max()({}) < new_r.max({})'.format(old_r.max(), new_r.max()))
    #             return 'new_r_too_big'
    #
    #         if old_u.min() < new_u.min():
    #             print('\t__Error: old_u.min({}) < new_u.min({})'.format(old_u.min(), new_u.min()))
    #             return 'new_u_too_small'
    #
    #         # if new_u.max() > old_u.max():
    #         #     print('\t__Error: old_u.max()({}) < new_u.max({})'.format(old_u.max(), new_u.max()))
    #         #     return 'new_u_too_big'
    #
    #         return 'OK'
    #
    #     def try_interp_u_r(old_r, old_u, old_mdot):
    #
    #         u_max_grid = []
    #         r_max_grid = []
    #
    #         bad_fit = True
    #
    #         while bad_fit:
    #             ''' --- Do interpolation do determine if the fitted spline is well behavied --- '''
    #
    #             # print(len( old_u) )
    #
    #             u_max_grid = np.mgrid[old_u.min():old_u.max():1000j]
    #             r_max_grid = Math.interp_row(old_u, old_r, u_max_grid)
    #
    #             message = check_limits(old_r, r_max_grid, old_u, new_u)
    #
    #             if message == 'OK': # well behaved interpolation, - exit the loop
    #                 bad_fit = False
    #             else:
    #                 if message == 'new_r_too_big' or message == 'new_u_too_big':
    #                     print('\t__Error. Fitted spline exhibit violent behaviour. Removing the following data set:\n'
    #                           'r: {} u: {} mdot: {}'.format("%.2f" % old_r[-1], "%.2f" % old_u[-1], "%.2f" % old_mdot[-1]))
    #                     old_r = old_r[:-1]
    #                     old_u = old_u[:-1]
    #                     old_mdot = old_mdot[:-1]
    #
    #                     bad_fit = True
    #                 else:
    #                     raise ValueError('Unaccounted error: {}'.format(message))
    #
    #             if len(old_r) <= 3:
    #                 raise ValueError('Well Behaved spline not found after removing all but 2 (min) points.')
    #
    #         return r_max_grid, u_max_grid, old_r, old_u, old_mdot
    #
    #     # --- CHANGE all the arrays --- to account for possible violent behaviour of fitted spline
    #     r_max_grid, u_max_grid, new_r, new_u, new_mdot = try_interp_u_r(new_r, new_u, new_mdot)
    #
    #
    #     print('\t__Data R:({}, {}), U: ({}, {})'.format(new_r.min(),new_r.max(), new_u.min(), new_u.max()))
    #     print('\t__Intr R:({}, {}), U: ({}, {})'.format(r_max_grid.min(), r_max_grid.max(), u_max_grid.min(), u_max_grid.max()))
    #
    #
    #     ax1.plot(r_max_grid, u_max_grid, '-', color='blue') # line that connects maxs of vel. profs
    #
    #     # --- Find a cross between interplated curve and sonic velocity.
    #
    #     r = self.mdl[long_i].get_col('r')
    #     t = self.mdl[long_i].get_col('t')
    #     u_s = self.mdl[long_i].get_sonic_u()
    #
    #     ax1.plot(r, u_s, '-', color='blue') #
    #
    #     # ax2 = ax1.twinx()   # for plotting the temperature scale (to see the Temp at critical radius)
    #     ax2.plot(r, t, '-.', color='brown')
    #
    #     i1 = Math.find_nearest_index(r, r_max_grid.min()) # - 1
    #     i2 = Math.find_nearest_index(r, r_max_grid.max()) # + 1
    #
    #
    #
    #     print('\t__Cropped R : [{}, {}] -> ({}, {}) out of {}'.format(i1,i2, r[i1], r[i2], r.max()))
    #     r_cr = r[i1:i2]
    #     u_s_cr =u_s[i1:i2]
    #     t_cr = t[i1:i2]
    #
    #
    #     # x_for_us_grid = np.mgrid[r.min():r.max():1000j]
    #     int_r_for_us_grid = Math.interp_row(r_cr, u_s_cr, r_max_grid) # corpped radius range
    #     int_t_for_us_grid = Math.interp_row(r_cr, t_cr, r_max_grid)   # cropped temp range
    #
    #     ax1.plot(r_max_grid, int_r_for_us_grid, '-', color='green') # corpped radius range
    #     ax2.plot(r_max_grid, int_t_for_us_grid, '-', color='brown') # cropped temp range
    #
    #     # --- Critical Radius ---
    #
    #     r_cr_c, u_s_cr_c = Math.interpolated_intercept(r_max_grid, u_max_grid, int_r_for_us_grid)
    #
    #     mdot_cr = None # for the case if critical point is not found
    #     t_crit = None
    #
    #     if len(r_cr_c) == len(u_s_cr_c) == 0:
    #         pass
    #         #raise ValueError('Critical Radius is not found')
    #     else:
    #
    #         print('\t__Critical radii found: ({},{})'.format(len(r_cr_c), len(u_s_cr_c)) )
    #
    #         i_of_u_near_yc = Math.find_nearest_index(int_r_for_us_grid, u_s_cr_c) # for f
    #
    #         for i in range(len(r_cr_c)):
    #             ax1.plot(r_cr_c[i], u_s_cr_c[i], 'X', color='blue', label='Rc:{}'.format('%.3f' %  r_cr_c[i])) # critical Radius
    #             ax1.annotate('Rs:'+str('%.3f' % r_cr_c[i]), xy=(r_cr_c[i], u_s_cr_c[i]), textcoords='data')
    #
    #         # --- Critical Temperature ---
    #
    #         t_crit = int_t_for_us_grid[i_of_u_near_yc]  # temperature at a critical radius
    #         ax2.plot( r_cr_c, t_crit, 'X', color='red', label='Tc:{}'.format('%.3f' %  t_crit) )           # temp. at critical radius
    #         ax1.axvline(x=r_cr_c, color='gray', linestyle='solid')
    #
    #         # --- Critical Mass Loss --- [SUBPLOT]
    #
    #         ax3 = fig.add_axes([0.23, 0.5, 0.23, 0.23])
    #         ax3.plot(new_mdot, new_u, '-', color='black')
    #         ax3.plot(new_mdot, new_u, '.', color='black')
    #         ax3.set_xlabel(Labels.lbls('mdot'))
    #         ax3.set_ylabel(Labels.lbls('u'))
    #         ax3.grid()
    #         ax3.axhline(y=u_s_cr_c, color='gray', linestyle='--', label='Son_vel: {}'.format( '%.3f' % u_s_cr_c))
    #
    #
    #         if u_s_cr_c < new_u.min() or u_s_cr_c > new_u.max():
    #             raise ValueError('u_s_cr_c({}) is beyond available new_u range({} , {})'.format(u_s_cr_c, new_u.min(), new_u.max()))
    #         else:
    #
    #             # --- The 'new_mdot_new_u' and 'new_mdot_new_u_sort' are here to ensire assending order of x coord. for
    #             #     interpolation in 'solv_inter_row', otherwise, interpolation breakes.
    #
    #             new_mdot_new_u = np.array(  [ [new_mdot[i], new_u[i]] for i in range(len(new_mdot))] )
    #
    #             new_mdot_new_u_sort = np.sort(new_mdot_new_u.view('f8, f8'), order=['f0'], axis=0).view(np.float)
    #
    #             mdot_cr = Math.solv_inter_row(new_mdot_new_u_sort[:,0], new_mdot_new_u_sort[:,1], u_s_cr_c)
    #
    #             ax3.plot(mdot_cr, u_s_cr_c, 'X', color='black', label='Mdot_cr: {}'.format('%.3f' % np.float(mdot_cr[0])))
    #             ax3.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
    #             ax3.annotate('Mdot_cr:' + str('%.3f' % mdot_cr), xy=(mdot_cr, u_s_cr_c), textcoords='data')
    #
    #     ax2.set_ylabel(Labels.lbls('ts'), color='r')
    #     ax2.tick_params('u', colors='r')
    #     ax2.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
    #
    #     ax1.tick_params('u', colors='b')
    #     ax1.set_xlabel(Labels.lbls('r'))
    #     ax1.set_ylabel(Labels.lbls('u'))
    #     ax1.grid(which='both')
    #     ax1.grid(which='minor', alpha=0.2)
    #     ax1.grid(which='major', alpha=0.2)
    #     ax1.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
    #     plot_name = self.plot_dir + 'critical_radius.pdf'
    #     plt.savefig(plot_name)
    #     plt.show()
    #
    #     # --- SORTING the array according to the mass loss (max -> min)
    #     crit_sonic_values = np.sort(crit_sonic_values.view('f8, f8, f8, f8, f8, f8'), order=['f3'], axis=0).view(
    #         np.float)
    #
    #
    #     # head = '\t {} \t {} \t {} \t\t {} \t {} \t {}'\
    #     #     .format('log(L)', 'M(Msun)', 'Yc', 'log(Mdot)', 'Rs(Rsun)', 'log(Ts)')
    #     tablehead = '{}  {}  {}  {}  {}  {}'\
    #         .format('log(L)', 'M(Msun)', 'Yc', 'l(Mdot)', 'Rs(Rsun)', 'log(Ts)')
    #
    #     # --- Adding a first row of values for critical r, mdot, and t, and l, m, Yc for the model, whose sonic vel. was
    #     # used to interplate the temp amd mdot.
    #
    #     l = self.mdl[long_i].get_col('l')[-1] # for the first row
    #     m = self.mdl[long_i].get_col('xm')[-1]
    #     yc = self.mdl[long_i].get_col('He4')[0]
    #
    #     # --- Appending the ROW with critical values, if found
    #     if len(r_cr_c) == len(u_s_cr_c) == 0:
    #         crit_sonic_values = np.reshape(crit_sonic_values, (len(self.num_files), 6))
    #         print('\t__Warning! Critical Values are not found. Ouptut tale contain {} rows instead of {}'
    #               .format(len(self.num_files), len(self.num_files) + 1))
    #     else:
    #         crit_sonic_values = np.insert(crit_sonic_values, 0, [l, m, yc, mdot_cr, r_cr_c, t_crit])
    #         crit_sonic_values = np.reshape(crit_sonic_values, (len(self.num_files)+1, 6))
    #
    #         print('\t__Note. Critical Values are found and written in the FIRST row (out of {}) in output file.'.
    #               format(len(self.num_files)+1))
    #
    #     # print('Critical values:')
    #     # print(head)
    #     # print(crit_sonic_values[0,:])
    #     #
    #     #
    #     # print('\n')
    #     # print(head)
    #     # print(crit_sonic_values[1:,:])
    #     #
    #     # print('\t__Note: Value {} means that solution was not found (extended configutration)'.format(0.))
    #     # print('Critical Radii found: {}'.format(r_cr_c))
    #
    #     out_name = 'SP_'
    #     for i in range(len(self.input_dirs)):
    #         if self.input_dirs[i] not in self.dirs_not_to_be_included and self.input_dirs[i] != '..':
    #             out_name = out_name + self.input_dirs[i]
    #             if i < len(self.input_dirs) - 1:
    #                 out_name = out_name + '_'
    #     out_name = out_name + '.data'
    #
    #
    #
    #
    #     print('Results are saved in: {}'.format(self.out_dir + out_name))
    #     np.savetxt(self.out_dir + out_name, crit_sonic_values, '%.4f', '  ', '\n', tablehead, '')

class Criticals2:

    out_dir = '../data/output/'
    plot_dir = '../data/plots/'

    def __init__(self, ga_smfiles, sp_smfiles, sp_plotfiles, dirs_not_to_be_included, out_dir):

        self.input_dirs = ga_smfiles[0].split('/')[:-1]
        # print(self.input_dirs)
        self.dirs_not_to_be_included = dirs_not_to_be_included # name of folders that not to be in the name of out. file.

        self.out_dir = out_dir

        self.ga_smfls = ga_smfiles
        self.ga_mdl = []
        self.ga_smdl =[]

        self.sp_smfls = sp_smfiles
        self.sp_mdl = []
        self.sp_smdl =[]
        self.sp_plmdl = []

        for file in ga_smfiles:
            self.ga_mdl.append(Read_SM_data_file.from_sm_data_file(file))

        for file in sp_smfiles:
            self.sp_mdl.append(Read_SM_data_file.from_sm_data_file(file))

        for file in sp_plotfiles:
            self.sp_plmdl.append(Read_Plot_file.from_file(file))



        self.sort_ga_smfiles('mdot', -1)
        self.sort_sp_smfiles('mdot', -1)

    def get_boundary(self, u_min):
        '''
        RETURNS ' bourders.min() ' - min radius among all the models, where u exceeds the 'u_min'
        :param u_min:
        :return:
        '''
        bourders = []

        for i in range(len(self.ga_smfls)):
            u = self.ga_smdl[i].get_col('u')
            r = self.ga_smdl[i].get_col('r')
            for i in range(len(r)):
                if u[i] > u_min:
                    # ax1.axvline(x=r[i], color='red')
                    bourders = np.append(bourders, r[i])
                    break

        return bourders.min()

    def sort_ga_smfiles(self, v_n, where = -1, descending=True):
        '''

        :param v_n: what value to use to sort sm.files
        :param where: where value si caken (surface -1 or core 0 )
        :param descending: if True, sm.files are sorted by descending order of the chosen parameter.
        :return: NOTHING (changes the smdl[])
        '''


        i_and_mdots = []
        for i in range(len(self.ga_smfls)):
            i_and_mdots = np.append(i_and_mdots, [i, self.ga_mdl[i].get_col(v_n)[where]])

        i_and_mdots_sorted = np.sort(i_and_mdots.view('f8, f8'), order=['f1'], axis=0).view(np.float)
        i_and_mdots_reshaped = np.reshape(i_and_mdots_sorted, (len(self.ga_smfls), 2))

        if descending:
            i_and_mdots_reshaped_inversed = np.flip(i_and_mdots_reshaped, 0) # flip for ascending order
        else:
            i_and_mdots_reshaped_inversed = i_and_mdots_reshaped # no flipping


        sorted_by_mdot_files = []
        for i in range(len(self.ga_smfls)):
            sorted_by_mdot_files.append(self.ga_smfls[np.int(i_and_mdots_reshaped_inversed[i][0])])

        def get_i(file):
            for i in range(len(self.ga_smfls)):
                if file == self.ga_smfls[i]:
                    return i

        for file in sorted_by_mdot_files:

            self.ga_smdl.append(self.ga_mdl[get_i(file)])

        # for i in range(len(self.num_files)):
        #     print(self.smdl[i].get_col(v_n)[where])



    def sort_sp_smfiles(self, v_n, where = -1, descending=True):
        '''

        :param v_n: what value to use to sort sm.files
        :param where: where value si caken (surface -1 or core 0 )
        :param descending: if True, sm.files are sorted by descending order of the chosen parameter.
        :return: NOTHING (changes the smdl[])
        '''


        i_and_mdots = []
        for i in range(len(self.sp_smfls)):
            i_and_mdots = np.append(i_and_mdots, [i, self.sp_mdl[i].get_col(v_n)[where]])

        i_and_mdots_sorted = np.sort(i_and_mdots.view('f8, f8'), order=['f1'], axis=0).view(np.float)
        i_and_mdots_reshaped = np.reshape(i_and_mdots_sorted, (len(self.sp_smfls), 2))

        if descending:
            i_and_mdots_reshaped_inversed = np.flip(i_and_mdots_reshaped, 0) # flip for ascending order
        else:
            i_and_mdots_reshaped_inversed = i_and_mdots_reshaped # no flipping


        sorted_by_mdot_files = []
        for i in range(len(self.sp_smfls)):
            sorted_by_mdot_files.append(self.sp_smfls[np.int(i_and_mdots_reshaped_inversed[i][0])])

        def get_i(file):
            for i in range(len(self.sp_smfls)):
                if file == self.sp_smfls[i]:
                    return i

        for file in sorted_by_mdot_files:

            self.sp_smdl.append(self.sp_mdl[get_i(file)])

        # for i in range(len(self.num_files)):
        #     print(self.smdl[i].get_col(v_n)[where])

    def analyze_ga_sm_fls(self, depth, add_sonic_vals, show_plot):
        def get_boundary(u_min):
            '''
            RETURNS ' bourders.min() ' - min radius among all the models, where u exceeds the 'u_min'
            :param u_min:
            :return:
            '''
            bourders = []

            for i in range(len(self.ga_smfls)):
                u = self.ga_smdl[i].get_col('u')
                r = self.ga_smdl[i].get_col('r')
                for i in range(len(r)):
                    if u[i] > u_min:
                        # ax1.axvline(x=r[i], color='red')
                        bourders = np.append(bourders, r[i])
                        break

            return bourders.min()

        def crop_ends(x, y):
            '''
            In case of 'wierd' vel/temp profile with rapidly rising end, first this rising part is to be cut of
            before maximum can be searched for.
            :param x:
            :param y:
            :return:
            '''
            x_mon = x
            y_mon = y

            non_monotonic = True

            while non_monotonic:

                if len(x_mon) <= 10:
                    return x, y
                    # raise ValueError('Whole array is removed in a searched for monotonic part.')

                if y_mon[-1] > y_mon[-2]:
                    y_mon = y_mon[:-1]
                    x_mon = x_mon[:-1]
                    # print(x_mon[-1], y_mon[-1])
                else:
                    non_monotonic = False

            return Math.find_nearest_index(x, x_mon[-1])

        def all_values_array(i_model, rs_p, ts_p, min_indx, add_sonic_vals):

            out_array = []

            # --- --- GET ARRAYS AND VALUES AND APPENDING TO OUTPUT ARRAY --- --- --- --- --- --- --- ---

            r = self.ga_smdl[i_model].get_col('r')[min_indx:]
            t = self.ga_smdl[i_model].get_col('t')[min_indx:]

            out_array = np.append(out_array, self.ga_smdl[i_model].get_col('l')[-1])  # appending 'l'    __1__
            out_array = np.append(out_array, self.ga_smdl[i_model].get_col('xm')[-1])  # appending 'xm'   __2__
            out_array = np.append(out_array, self.ga_smdl[i_model].get_col('He4')[0])  # appending 'Yc'   __3__

            out_array = np.append(out_array, self.ga_smdl[i_model].get_col('mdot')[-1])  # appending 'mdot' __4__
            out_array = np.append(out_array, rs_p)  # appending 'mdot' __5__
            out_array = np.append(out_array, ts_p)  # appending 'mdot' __6__

            # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

            val_array = []
            for v_n_cond in add_sonic_vals:

                if len(v_n_cond.split('-')) > 2:
                    raise NameError(
                        'For *add_sonic_vals* use format *var_name-location*, (given: {}) where var_name is '
                        'one of the BEC sm.data variables and location is *core = [-1] surface=[0]',
                        'sp = [sonic_point_interpolated]'.format(v_n_cond))

                v_n = v_n_cond.split('-')[0]
                cond = v_n_cond.split('-')[-1]

                if cond != 'sp':
                    var_val = self.ga_smdl[i_model].get_cond_value(v_n,
                                                                   cond)  # assuming that condition is not required interp

                else:
                    ''' Here The Interpolation of v_n is Done'''

                    v_n_val_arr = self.ga_smdl[i_model].get_col(v_n)[min_indx:]
                    f = interpolate.InterpolatedUnivariateSpline(r, v_n_val_arr)
                    var_val = f(rs_p)

                    if len([var_val]) > 1:
                        raise ValueError('More than one solution found for *{}* sonic value: ({})'.format(v_n, var_val))

                val_array = np.append(val_array, var_val)

            if len(val_array) != len(add_sonic_vals):
                raise ValueError('len(val_array)[{}] != len(add_sonic_vals)[{}]'
                                 .format(len(val_array), len(add_sonic_vals)))

            for i in range(len(val_array)):  # 5 is a number of main values: [ l, m, Yc, ts, rs ]
                out_array = np.append(out_array, val_array[i])  # appending 'v_ns' __n__

            return out_array

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()

        tlt = 'VELOCITY PROFILE'
        plt.title(tlt, loc='left')

        mdots   = np.array([0.])
        # rs_ts   = np.array([0.,0.])
        r_u_mdot_max = np.array([0., 0., 0.])
        r_t_mdot_max = np.array([0., 0., 0.])
        # rs_ts_mdot   = np.array([0.,0.,0.])

        out_array = np.zeros(len(add_sonic_vals) + 6) # where 6 are: [l, m, Yc, mdot, rs, ts] # always include

        min_ind = 0
        max_ind = -1
        r_min = get_boundary(0.1) # setting the lower value of r that above which the analysis will take place
        for i in range(len(self.ga_smfls)):
            r = self.ga_smdl[i].get_col('r')
            u = self.ga_smdl[i].get_col('u')
            t = self.ga_smdl[i].get_col('t')
            min_ind = Math.find_nearest_index(r, r_min)
            max_ind = crop_ends(r, u)

            ax1.plot(r[min_ind:], u[min_ind:], '.', color='black')
            ax2.plot(r[min_ind:], t[min_ind:], '.', color='orange')


        '''------------------------------------------MAIN CYCLE------------------------------------------------------'''
        for i in range(len(self.ga_smfls)):

            r  = self.ga_smdl[i].get_col('r')
            u  = self.ga_smdl[i].get_col('u')
            mu = self.ga_smdl[i].get_col('mu')
            t  = self.ga_smdl[i].get_col('t')
            u_s= self.ga_smdl[i].get_sonic_u()
            mdot_u=self.ga_smdl[i].get_col('mdot')[-1]
            mdots = np.append(mdots, mdot_u)
            print('\t__Initical Array Length: {}'.format(len(r)))


            min_ind = Math.find_nearest_index(r, r_min)


            # ----------------------- R U ----------------------------
            r  = r[min_ind:max_ind]
            u  = u[min_ind:max_ind]
            u_s= u_s[min_ind:max_ind]
            t = t[min_ind:max_ind]
            mu = mu[min_ind:max_ind]

            print('\t__Cropped Array Length: {}'.format(len(r)))

            int_r  = np.mgrid[r[0]:r[-1]:depth*1j]
            int_u  = Math.interp_row(r, u, int_r)

            # ax1.plot(r, u, '.', color='black')
            ax1.plot(r, u_s, '--', color='black')
            ax1.annotate(str('%.2f' % mdot_u), xy=(r[-1], u[-1]), textcoords='data')
            ax1.plot(int_r, int_u, '-', color='gray')

            # ------------------------R T --------------------------------

            ts_arr = np.log10((mu * Constants.m_H * (u * 100000) ** 2) / Constants.k_b)


            int_r = np.mgrid[r[0]:r[-1]:depth * 1j]
            int_t  = Math.interp_row(r, t, int_r)
            int_ts_arr=Math.interp_row(r, ts_arr, int_r)

            ax2.plot(r, ts_arr, '.', color='orange')
            ax2.plot(int_r, int_t,'-',color='orange')
            ax2.plot(int_r, int_ts_arr, '--', color='orange')


            r_u_max_p, u_max_p = Math.get_max_by_interpolating(int_r, int_u)      # MAXIMUM VALUES OF VELOCITY
            r_u_mdot_max = np.vstack((r_u_mdot_max, [r_u_max_p[0], u_max_p[0], mdot_u]))
            r_t_max_p, t_max_p = Math.get_max_by_interpolating(int_r, int_ts_arr) # MAXIMUM VALUES OF TEMPERATURE
            r_t_mdot_max = np.vstack((r_t_mdot_max, [r_t_max_p[0], t_max_p[0], mdot_u]))

            ax1.plot(r_u_max_p, u_max_p, 'x', color='red')
            ax2.plot(r_t_max_p, t_max_p, 'x', color='red')


            # --- --- ---| SONIC POINT PARAMTERS |--- --- ---
            rs_p, ts_p = Math.interpolated_intercept(int_r, int_ts_arr, int_t)     # SONIC TEPERATURE
            if rs_p.any():
                rs_p = rs_p[0][0]
                ts_p = ts_p[0][0]
                ax2.plot(rs_p, ts_p, 'X', color='red')
                ax2.annotate(str('%.2f' % ts_p), xy=(rs_p, ts_p), textcoords='data')

                row = all_values_array(i, rs_p, ts_p, min_ind, add_sonic_vals)
                out_array = np.vstack((out_array, row))



        r_u_mdot_max = np.delete(r_u_mdot_max, 0, 0) # removing the 0th row with zeros
        r_t_mdot_max = np.delete(r_t_mdot_max, 0, 0)
        mdots = np.delete(mdots, 0, 0)


        if len(mdots) != len(r_u_mdot_max[:,0]):
            raise ValueError('len(mdots){} != len(r_u_mdot_max[:,0]){}'.format(len(mdots), len(r_u_mdot_max[:,0])))

        if len(mdots) ==0:
            raise ValueError('len(mdots) = 0')

        ax1.plot(r_u_mdot_max[:,0],  r_u_mdot_max[:,1],  '-.', color='blue')

        # plt.show()
        # print(out_array)




        def cross(mdot, r1, u1, r2, u2, mdot_maxs, u_ot_t = 'u'):
            '''
            Finds the delta = (u_i_max - cs_i) where u_i is a maximum of u profile along the r, cs_i - point along sonic
            velocity profile, that lies on the line thac connects the u_i maximums.
            As close the mass loss to a critical one, the closer delta to 0 ( u_i_max = cs_i at the critical
            (deflection) point)

            :param mdot:
            :param r1: full length of a sonic profile (r)
            :param u1: values of sonic velocity along (r)
            :param r2: set of 'r' points for maximums of velocity profiles for every mass loss
            :param u2: values of velocity at 'r' points
            :param mdot_maxs: mdot values of every point 'r,u' above
            :return:
            '''



            r1_u1 = []
            for i in range(len(r1)):
                r1_u1 = np.append(r1_u1, [r1[i], u1[i]])

            r1_u1 = np.sort(r1_u1.view('f8, f8'), order=['f1'], axis=0).view(np.float)
            r1_u1_sort = np.reshape(r1_u1, (len(r1), 2))


            r2_u2_mdot = []
            for i in range(len(r2)):
                r2_u2_mdot = np.append(r2_u2_mdot, [r2[i], u2[i], mdot_maxs[i]])

            r2_u2_mdot = np.sort(r2_u2_mdot.view('f8, f8, f8'), order=['f1'], axis=0).view(np.float)
            r2_u2_mdot_sort = np.reshape(r2_u2_mdot, (len(r2), 3))

            r1 = r1_u1_sort[:,0]
            u1 = r1_u1_sort[:,1]

            r2    = r2_u2_mdot_sort[:,0]
            u2    = r2_u2_mdot_sort[:,1]
            mdots = r2_u2_mdot_sort[:,2]

            #
            # u_rmax1 = u1[np.where(r1 == r1.max())]
            # u_rmax2 = u2[np.where(r2 == r2.max())]
            #
            # r_umin1_i = r1[np.where(u1 == u1.min())]
            # r_umin2_i = r2[np.where(u2 == u2.min())]


            i_u2_where_u1_max = Math.find_nearest_index(u2, u1.max()) # needed to avoide violent behaviour in high mdot
            u2_crop = u2[:i_u2_where_u1_max]            # otherwise there was crossing with sonic profiles at temps 5.9,
            r2_crop = r2[:i_u2_where_u1_max]


            u_lim1 = np.array([u1.min(), u2_crop.min()]).max()
            u_lim2 = np.array([u1.max(), u2_crop.max()]).min()

            if u_lim2 < u_lim1:
                raise ValueError('u_lim1({}) < u_lim2({})'.format(u_lim1, u_lim2))

            u_grid = np.mgrid[u_lim2:u_lim1:1000*1j]

            if u_grid.max() > u2.max() or u_grid.max() > u1.max():
                raise ValueError('u_grid.max({}) > u2.max({}) or u_grid.max({}) > u1.max({})'
                                 .format(u_grid.max(), u2.max(), u_grid.max(), u1.max()))
            #
            # if u_grid.min() < u2.min() or u_grid.min() < u1.min():
            #     raise ValueError('u_grid.min({}) < u2.min({}) or u_grid.min({}) < u1.min({})'
            #                      .format(u_grid.min(), u2.min(), u_grid.min(), u1.min()))


            f1 = interpolate.InterpolatedUnivariateSpline(u1, r1)
            r1_grid = f1(u_grid)

            # if u_ot_t == 't':
                # print('\t')
                # print('u1:[{} {}] u_lim1:{} u_lim2:{}'.format(u1.min(), u1.max(), u_lim1, u_lim2))
                # print('u2:[{} {}] u_lim1:{} u_lim2:{}'.format(u2.min(), u2.max(), u_lim1, u_lim2))
                #


                # ax2.plot(r2_crop, u2_crop, 'o', color='magenta')

            f2 = interpolate.InterpolatedUnivariateSpline(u2_crop, r2_crop) # cropping is done to cut the high mdot prob
            r2_grid = f2(u_grid)

            # r1_grid = Math.interp_row(u1, r1, u_grid)
            # r2_grid = Math.interp_row(u2, r2, u_grid)

            if u_ot_t == 'u':
                ax1.plot(r1_grid, u_grid, '-.', color='green')
                ax1.plot(r2_grid, u_grid, '-.', color='green')
            else:
                ax2.plot(r1_grid, u_grid, '-.', color='green')
                ax2.plot(r2_grid, u_grid, '-.', color='green')

            uc, rc = Math.interpolated_intercept(u_grid, r1_grid, r2_grid)
            if uc.any():  # if there is an intersections between sonic vel. profile and max.r-u line
                uc0 = uc[0][0]
                rc0 = rc[0][0]

                if u_ot_t == 'u':
                    ax1.plot(rc0, uc0, 'X', color='green')
                    ax1.annotate(str('%.2f' % mdot), xy=(rc0, uc0), textcoords='data')
                if u_ot_t == 't':
                    ax2.plot(rc0, uc0, 'X', color='green')
                    ax2.annotate(str('%.2f' % mdot), xy=(rc0, uc0), textcoords='data')

                delta = u2[np.where(mdots == mdot)] - uc0
                # print(uc, rc, '\t', delta)
                # print('Delta: ' , delta_ut, )

                return delta


            # if u_min2 < u_min1:        # if there is a common area in terms of radii
            #     if r_umin1 > r_umin2:   # if there is a common area in case of velocity
            #         u_lim1 = u_min1
            #         u_lim2 = u_max1
            #
            #         u_grid = np.mgrid[u_lim2:u_lim1:1000*1j]
            #
            #         r1_grid = Math.interp_row(u1, r1, u_grid)
            #         r2_grid = Math.interp_row(u2, r2, u_grid)
            #
            #         ax1.plot(r1_grid, u_grid, '-', color='green')
            #         ax1.plot(r2_grid, u_grid, '-', color='green')
            #
            #         uc, rc = Math.interpolated_intercept(u_grid, r1_grid, r2_grid)
            #         if uc.any(): # if there is an intersections between sonic vel. profile and max.r-u line
            #
            #             uc0 = uc[0][0]
            #             rc0 = rc[0][0]
            #
            #             ax1.plot(rc0, uc0, 'X', color='green')
            #             ax1.annotate(str('%.2f' % mdot), xy=(rc0, uc0), textcoords='data')
            #
            #             delta = u2[np.where(mdots == mdot)] - uc0
            #
            #             # print('Delta: ' , delta_ut, )
            #
            #             return delta
            #         else:
            #             print('\t__Warning. No common area in velocity found: '
            #                   'r_at_u_min1:{} > r_at_u_min2{}'.format(r_umin1, r_umin2))
            #     else:
            #         print('\t__Warning. No common area in radii found: '
            #               'r_at_u_min2:{} < r_at_u_min2{}'.format(r_umin1, r_umin2))

        def get_mdot_delta(r_ut_mdot_max, u_or_t):

            mdot_delta_ut = []

            n = 0
            for i in range(len(self.ga_smfls)):
                r = self.ga_mdl[i].get_col('r')[min_ind:]
                # u = self.mdl[i].get_col('u')[min_ind:]
                # mu = self.mdl[i].get_col('mu')[min_ind:]
                t = self.ga_mdl[i].get_col('t')[min_ind:]
                # ts_arr = np.log10((mu * Constants.m_H * (u * 100000) ** 2) / Constants.k_b)[min_ind:]
                u_s = self.ga_mdl[i].get_sonic_u()[min_ind:]

                mdot = self.ga_mdl[i].get_col('mdot')[-1]

                if u_or_t == 'u':
                    delta_ut = cross(mdot, r, u_s, r_ut_mdot_max[1:, 0], r_ut_mdot_max[1:, 1], r_ut_mdot_max[1:, 2], u_or_t)
                    # print(delta_ut, u_or_t)
                else:
                    delta_ut = cross(mdot, r, t,   r_ut_mdot_max[1:, 0], r_ut_mdot_max[1:, 1], r_ut_mdot_max[1:, 2], u_or_t)
                    # print(delta_ut, u_or_t)

                if delta_ut != None:
                    mdot_delta_ut = np.append(mdot_delta_ut, [mdot, delta_ut])
                    n = n + 1

            if len(mdot_delta_ut) == 0:
                raise ValueError('mdot_delta_ut is not found at all for <{}>'.format(u_or_t))

            mdot_delta_ut = np.sort(mdot_delta_ut.view('f8, f8'), order=['f0'], axis=0).view(np.float)
            mdot_delta_ut_shape = np.reshape(mdot_delta_ut, (n, 2))

            mdot     = mdot_delta_ut_shape[:, 0]
            delta_ut = mdot_delta_ut_shape[:, 1]

            crit_mdot_u = Math.solv_inter_row(mdot, delta_ut, 0.)  # Critical Mdot when the delta_ut == 0

            print('\t\t crit_mdot_u', crit_mdot_u)

            if not crit_mdot_u.any():
                raise ValueError('Critical Mdot is not found.')
            else:
                print('\t__Critical Mdot: {} (for: {})'.format(crit_mdot_u, u_or_t))

            return mdot, delta_ut


        mdot_arr, delta_arr = get_mdot_delta(r_u_mdot_max, 'u')

        if delta_arr.min() > 0. and delta_arr.max() > 0.:
            raise ValueError('if delta_arr.min({}) and delta_arr.max({}) > 0 : '
                             'peak of vel. profs do not cross sonic. vel.'.format(delta_arr.min(), delta_arr.max()))

        if delta_arr.min() < 0. and delta_arr.max() < 0.:
            raise ValueError('if delta_arr.min({}) and delta_arr.max({}) < 0 : '
                             'vel. profile does not crossing the sonic val.'.format(delta_arr.min(), delta_arr.max()))

        crit_mdot_u = Math.solv_inter_row(mdot_arr, delta_arr, 0.)
        if len(crit_mdot_u) > 1:
            raise ValueError('More than one mdot_cr. found. {}'.format(crit_mdot_u))

        ax3 = fig.add_axes([0.18, 0.18, 0.25, 0.25])
        ax3.set_xlabel(Labels.lbls('mdot'))
        ax3.set_ylabel('$u_i - cs_i$')
        ax3.grid()

        ax3.plot(mdot_arr, delta_arr, '-', color='black')
        ax3.plot(crit_mdot_u, 0., 'x', color='black')
        ax3.annotate('({}, {})'.format('%.3f' % crit_mdot_u, 0. ), xy=(crit_mdot_u, 0.),
                     textcoords='data')


        mdot_arr, delta_arr = get_mdot_delta(r_t_mdot_max, 't')
        crit_mdot_t = Math.solv_inter_row(mdot_arr, delta_arr, 0.)


        # --- --- FROM CRITICAL MDOT GET CRITICAL R, T, OTHERS --- --- ---

        r_u_mdot_max = np.sort(r_u_mdot_max.view('f8, f8, f8'), order=['f2'], axis=0).view(np.float)
        r_u_mdot_max = np.reshape(r_u_mdot_max, (len(self.ga_smfls), 3))

        r_t_mdot_max = np.sort(r_t_mdot_max.view('f8, f8, f8'), order=['f2'], axis=0).view(np.float)
        r_t_mdot_max = np.reshape(r_t_mdot_max, (len(self.ga_smfls), 3))


        f = interpolate.InterpolatedUnivariateSpline(r_t_mdot_max[:, 2], r_t_mdot_max[:, 1])  # t_crit
        crit_t = f(crit_mdot_t)

        f = interpolate.InterpolatedUnivariateSpline(r_u_mdot_max[:, 2], r_u_mdot_max[:, 0])  # r_crit
        crit_r = f(crit_mdot_u)

        ax2.plot(r_t_mdot_max[:, 0], r_t_mdot_max[:, 1], '-.', color='red')
        ax2.plot(crit_r, crit_t, 'X', color='red')
        ax1.annotate('({}, {})'.format('%.2f' % crit_r, '%.2f' % crit_t), xy=(crit_r, crit_t), textcoords='data')
        print('\t__Criticals: r={}, t={}, mdot={}'.format('%.4f' % crit_r, '%.4f' % crit_t, '%.4f' % crit_mdot_u))


        '''----------------------------------------------SUBPLOTS----------------------------------------------------'''

        ax4 = fig.add_axes([0.18, 0.50, 0.25, 0.25])
        ax4.set_xlabel(Labels.lbls('mdot'))
        ax4.set_ylabel('$u_i - cs_i$')
        ax4.grid()
        ax4.set_xlabel(Labels.lbls('mdot'))
        ax4.set_ylabel(Labels.lbls('t'), color='r')

        ax4.plot(r_t_mdot_max[:, 2], r_t_mdot_max[:, 1], '-', color='red')
        ax4.plot(crit_mdot_t, crit_t, 'x', color='red')
        ax4.annotate('({}, {})'.format('%.3f' % crit_mdot_t, '%.3f' % crit_t, ), xy=(crit_mdot_t, crit_t),
                     textcoords='data')
        ax4.tick_params('y', colors='r')

        ax5 = ax4.twinx()
        ax5.plot(r_u_mdot_max[:, 2], r_u_mdot_max[:, 0], '-', color='blue')
        ax5.plot(crit_mdot_u, crit_r, 'x', color='blue')
        ax5.annotate('({}, {})'.format('%.3f' % crit_mdot_u, '%.3f' % crit_r,), xy=(crit_mdot_u, crit_r),
                     textcoords='data')
        ax5.set_xlabel(Labels.lbls('mdot'))
        ax5.set_ylabel(Labels.lbls('r'), color='b')
        ax5.tick_params('y', colors='b')

        ax2.set_ylabel(Labels.lbls('ts'), color='r')
        ax2.tick_params('u', colors='r')
        ax2.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)

        ax1.tick_params('u', colors='b')
        ax1.set_xlabel(Labels.lbls('r'))
        ax1.set_ylabel(Labels.lbls('u'))
        ax1.grid(which='both')
        ax1.grid(which='minor', alpha=0.2)
        ax1.grid(which='major', alpha=0.2)
        ax1.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
        plot_name = self.plot_dir + 'critical_radius.pdf'
        plt.savefig(plot_name, dpi=240, format='pdf')

        if show_plot:
            plt.show()


        l = out_array[-1,0]  # choosing the last mpdel to get l, m, yc as low mdot would affect these them the least
        m = out_array[-1,1]
        yc= out_array[-1,2]

        out_array[0,0] = l   # inserting l, m, yc into the first row of the output array
        out_array[0,1] = m
        out_array[0,2] = yc

        out_array[0,3] = crit_mdot_u # inserting criticals in the first row
        out_array[0,4] = crit_r
        out_array[0,5] = crit_t


        print(out_array)

        print('\t__Note. Critical Values are found and written in the FIRST row (out of {}) in output file.'.
              format(len(self.ga_smfls) + 1))

        tablehead = '{} {} {} {} {} {}'\
            .format('log(L)', 'M(Msun)', 'Yc', 'mdot', 'r-sp', 't-sp')


        tmp = ''
        for v_n in add_sonic_vals:
            tmp = tmp + ' {}'.format(v_n)

        extended_head = tablehead + tmp
        print(extended_head)

        # --- --- --- MAKING A OUTPUT FILE NAME OUT OF FOLDERS THE SM.DATA FILES CAME FROM --- --- ---
        # out_name = 'SP'
        # for i in range(len(self.input_dirs)):
        #     if self.input_dirs[i] not in self.dirs_not_to_be_included and self.input_dirs[i] != '..':
        #         out_name = out_name + self.input_dirs[i]
        #         if i < len(self.input_dirs) - 1:
        #             out_name = out_name + '_'
        # out_name = out_name + '.data'

        # print('Results are saved in: {}'.format(self.out_dir + out_name))
        # np.savetxt(self.out_dir + out_name, out_array, '%.5f', '  ', '\n', extended_head, '')


        return extended_head.split(' '), out_array

    def analyze_sp_sm_and_plot_fls(self, depth, add_sonic_vals, show_plot):

        def crop_ends(x, y):
            '''
            In case of 'wierd' vel/temp profile with rapidly rising end, first this rising part is to be cut of
            before maximum can be searched for.
            :param x:
            :param y:
            :return:
            '''
            x_mon = x
            y_mon = y

            non_monotonic = True

            while non_monotonic:

                if len(x_mon) <= 10:
                    return x, y
                    # raise ValueError('Whole array is removed in a searched for monotonic part.')

                if y_mon[-1] > y_mon[-2]:
                    y_mon = y_mon[:-1]
                    x_mon = x_mon[:-1]
                    # print(x_mon[-1], y_mon[-1])
                else:
                    non_monotonic = False

            return Math.find_nearest_index(x, x_mon[-1])

        def get_boundary(u_min):
            '''
            RETURNS ' bourders.min() ' - min radius among all the models, where u exceeds the 'u_min'
            :param u_min:
            :return:
            '''
            bourders = []

            for i in range(len(self.sp_smfls)):
                u = self.sp_smdl[i].get_col('u')
                r = self.sp_smdl[i].get_col('r')
                for i in range(len(r)):
                    if u[i] > u_min:
                        # ax1.axvline(x=r[i], color='red')
                        bourders = np.append(bourders, r[i])
                        break

            return bourders.min()

        def all_values_array(cls, min_indx, rs_p, ts_p, arr, add_sonic_vals):

            out_array = []

            mdot = cls.get_col('mdot')[-1]

            # arr = [mdot, ts, r_n_rsun, teffluca, r_wind, tau]
            r_wind = -1
            t_eff  = -1
            tau    = -1
            for i in range(len(arr[:, 0])):
                print('{} =? {}'.format("%.2f"%arr[i, 0] , "%.2f"%mdot))
                if np.float("%.2f"%arr[i, 0]) == np.float("%.2f"%mdot):
                    print('     Found')
                    r_wind = arr[i, 4]
                    t_eff  = arr[i, 3]
                    tau    = arr[i, 5]
                    break
                if i == len(arr)-1:
                    raise ValueError('i not found')

            # --- --- GET ARRAYS AND VALUES AND APPENDING TO OUTPUT ARRAY --- --- --- --- --- --- --- ---

            r = cls.get_col('r')[min_indx:]
            t = cls.get_col('t')[min_indx:]

            out_array = np.append(out_array, cls.get_col('l')[-1])  # appending 'l'         __1__
            out_array = np.append(out_array, cls.get_col('xm')[-1])  # appending 'xm'       __2__
            out_array = np.append(out_array, cls.get_col('He4')[0])  # appending 'Yc'       __3__

            out_array = np.append(out_array, cls.get_col('mdot')[-1])  # appending 'mdot'   __4__
            out_array = np.append(out_array, rs_p)  # appending 'rs' __5__
            out_array = np.append(out_array, ts_p)  # appending 'ts' __6__
            out_array = np.append(out_array, r_wind)# appending 'ts' __7__
            out_array = np.append(out_array, t_eff) # appending 'ts' __8__
            out_array = np.append(out_array, tau)   # appending 'ts' __9__


            # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

            val_array = []
            for v_n_cond in add_sonic_vals:

                if len(v_n_cond.split('-')) > 2:
                    raise NameError(
                        'For *add_sonic_vals* use format *var_name-location*, (given: {}) where var_name is '
                        'one of the BEC sm.data variables and location is *core = [-1] surface=[0]',
                        'sp = [sonic_point_interpolated]'.format(v_n_cond))

                v_n = v_n_cond.split('-')[0]
                cond = v_n_cond.split('-')[-1]

                if cond != 'sp':
                    var_val = cls.get_cond_value(v_n, cond)  # assuming that condition is not required interp

                else:
                    ''' Here The Interpolation of v_n is Done'''

                    v_n_val_arr = cls.get_col(v_n)[min_indx:]
                    f = interpolate.InterpolatedUnivariateSpline(r, v_n_val_arr)
                    var_val = f(rs_p)

                    if len([var_val]) > 1:
                        raise ValueError('More than one solution found for *{}* sonic value: ({})'.format(v_n, var_val))

                val_array = np.append(val_array, var_val)

            if len(val_array) != len(add_sonic_vals):
                raise ValueError('len(val_array)[{}] != len(add_sonic_vals)[{}]'
                                 .format(len(val_array), len(add_sonic_vals)))

            for i in range(len(val_array)):  # 5 is a number of main values: [ l, m, Yc, ts, rs ]
                out_array = np.append(out_array, val_array[i])  # appending 'v_ns' __n__

            return out_array

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()

        tlt = 'VELOCITY PROFILE'
        plt.title(tlt, loc='left')

        mdots = np.array([0.])
        # rs_ts   = np.array([0.,0.])
        # r_u_mdot_max = np.array([0., 0., 0.])
        # r_t_mdot_max = np.array([0., 0., 0.])
        # rs_ts_mdot   = np.array([0.,0.,0.])



        # min_ind = 0
        # max_ind = -1
        r_min = get_boundary(0.1)  # setting the lower value of r that above which the analysis will take place

        '''---PLOTTING TH U(R) PROFILES---'''
        for i in range(len(self.sp_smfls)):
            r = self.sp_smdl[i].get_col('r')
            u = self.sp_smdl[i].get_col('u')
            t = self.sp_smdl[i].get_col('t')
            min_ind = Math.find_nearest_index(r, r_min)
            # max_ind = crop_ends(r, u)

            ax1.plot(r[min_ind:], u[min_ind:], '.', color='black')
            ax2.plot(r[min_ind:], t[min_ind:], '.', color='orange')

        # for i in range(len(self.ga_smfls)):
        #     r_ = self.ga_smdl[i].get_col('r')
        #     u_ = self.ga_smdl[i].get_col('u')
        #     t_ = self.ga_smdl[i].get_col('t')
        #
        #     ax1.plot(r_[min_ind:], u_[min_ind:], '-', color='gray')
        #     ax2.plot(r_[min_ind:], t_[min_ind:], '-', color='gray')
        arr = []
        '''---GETTING THE .PLOT (TAU+R_wind)---'''
        for i in range(len(self.sp_plmdl)):

            mdot = self.sp_plmdl[i].mdot_[-1]
            ts = self.sp_plmdl[i].t_eff[-1]
            # rs = self.sp_plmdl[i].r_
            r_n_rsun = self.sp_plmdl[i].r_n_rsun[-1]
            r_wind = self.sp_plmdl[i].rwindluca[-1]
            teffluca = self.sp_plmdl[i].teffluca[-1]
            # Tn  = self.sp_plmdl[i].Tn[-1]
            tau = self.sp_plmdl[i].tauatR[-1]

            arr = np.append(arr, [mdot, ts, r_n_rsun, teffluca, r_wind, tau])

            ax2.plot(r_n_rsun, ts, 'o', color='gray')

        arr_sort = np.sort(arr.view('f8, f8, f8, f8, f8, f8'), order=['f0'], axis=0).view(np.float)
        arr_reshaped = np.reshape(arr_sort, (len(self.sp_plmdl), 6))

        '''---INTERPOLATING THE SONIC VALUES---'''

        out_array = np.zeros(len(add_sonic_vals) + 9)  # where 6 are: [l, m, Yc, mdot, rs, ts] # always include
        for i in range(len(self.sp_smdl)):

            r  = self.sp_smdl[i].get_col('r')
            u  = self.sp_smdl[i].get_col('u')
            mu = self.sp_smdl[i].get_col('mu')
            t  = self.sp_smdl[i].get_col('t')
            u_s= self.sp_smdl[i].get_sonic_u()
            mdot_u=self.sp_smdl[i].get_col('mdot')[-1]
            mdots = np.append(mdots, mdot_u)
            print('\t__Initical Array Length: {}'.format(len(r)))


            min_ind = Math.find_nearest_index(r, r_min)


            # ----------------------- R U ----------------------------
            r  = r[min_ind:]
            u  = u[min_ind:]
            u_s= u_s[min_ind:]
            t = t[min_ind:]
            mu = mu[min_ind:]

            print('\t__Cropped Array Length: {}'.format(len(r)))

            int_r  = np.mgrid[r[0]:r[-1]:depth*1j]
            int_u  = Math.interp_row(r, u, int_r)

            # ax1.plot(r, u, '.', color='black')
            ax1.plot(r, u_s, '--', color='black')
            ax1.annotate(str('%.2f' % mdot_u), xy=(r[-1], u[-1]), textcoords='data')
            ax1.plot(int_r, int_u, '-', color='gray')

            # ------------------------R T --------------------------------

            ts_arr = np.log10((mu * Constants.m_H * (u * 100000) ** 2) / Constants.k_b)


            int_r = np.mgrid[r[0]:r[-1]:depth * 1j]
            int_t  = Math.interp_row(r, t, int_r)
            int_ts_arr=Math.interp_row(r, ts_arr, int_r)

            ax2.plot(r, ts_arr, '.', color='orange')
            ax2.plot(int_r, int_t,'-',color='orange')
            ax2.plot(int_r, int_ts_arr, '--', color='orange')


            # r_u_max_p, u_max_p = Math.get_max_by_interpolating(int_r, int_u)      # MAXIMUM VALUES OF VELOCITY
            # r_u_mdot_max = np.vstack((r_u_mdot_max, [r_u_max_p[0], u_max_p[0], mdot_u]))
            # r_t_max_p, t_max_p = Math.get_max_by_interpolating(int_r, int_ts_arr) # MAXIMUM VALUES OF TEMPERATURE
            # r_t_mdot_max = np.vstack((r_t_mdot_max, [r_t_max_p[0], t_max_p[0], mdot_u]))

            # ax1.plot(r_u_max_p, u_max_p, 'x', color='black')
            # ax2.plot(r_t_max_p, t_max_p, 'x', color='black')


            # --- --- ---| SONIC POINT PARAMTERS |--- --- ---
            rs_p, ts_p = Math.interpolated_intercept(int_r, int_ts_arr, int_t)     # SONIC TEPERATURE
            if rs_p.any():
                rs_p = rs_p[0][0]
                ts_p = ts_p[0][0]
                ax2.plot(rs_p, ts_p, 'X', color='red')
                ax2.annotate(str('%.2f' % ts_p), xy=(rs_p, ts_p), textcoords='data')

                row = all_values_array(self.sp_mdl[i], min_ind, rs_p, ts_p, arr_reshaped, add_sonic_vals)

                out_array = np.vstack((out_array, row))

        if show_plot:
            plt.show()

        tablehead = '{} {} {} {} {} {} {} {} {}' \
            .format('log(L)', 'M(Msun)', 'Yc', 'mdot', 'r-sp', 't-sp', 'R_wind', 'T_eff', 'Tau')

        tmp = ''
        for v_n in add_sonic_vals:
            tmp = tmp + ' {}'.format(v_n)

        extended_head = tablehead + tmp
        print(extended_head)
        out_array = np.delete(out_array, 0, 0)
        return extended_head.split(' '), out_array

    def combine_ga_sp(self, depth, add_sonic_vals, show_plot):
        ga_head, ga_table = self.analyze_ga_sm_fls(depth, add_sonic_vals, show_plot)
        sp_head, sp_table = self.analyze_sp_sm_and_plot_fls(depth, add_sonic_vals, show_plot)

        def compare_sonic_values():
            '''
            Combines the [-sp] values from ga and sp tables, estimating the error for each pair, returning the
            3d array [v_n:, mdot:, [mdot_val, ga_val, sp_val, err, err in %]]
            :return:
            '''
            ga_sp_v_n = []
            ga_i_val = []
            for i in range(len(ga_head)):
                print(ga_head[i].split('-'))
                if 'sp' in ga_head[i].split('-'):
                    ga_sp_v_n.append(ga_head[i].split('-')[0])
                    ga_i_val = np.append(ga_i_val, np.int(i))

            sp_sp_v_n = []
            sp_i_val = []
            for i in range(len(sp_head)):
                print(sp_head[i].split('-'))
                if 'sp' in sp_head[i].split('-'):
                    sp_sp_v_n.append(sp_head[i].split('-')[0])
                    sp_i_val = np.append(sp_i_val, np.int(i))

            if ga_sp_v_n != sp_sp_v_n:
                raise NameError('Sonic Point names (SP): {}\n Sonic Point names (GA): {}\n NOT THE SAME'
                                .format(sp_sp_v_n, ga_sp_v_n))
            print('\t__{} =? {}'.format(len(ga_i_val), len(sp_i_val)))


            print('Mdot | {}'.format(ga_sp_v_n))

            mdot_row = []
            for i in range(len(ga_table[:,3])): # mdot
                ga_mdot = ga_table[i,3]
                if ga_mdot in sp_table[:,3]: # if [ga_mdot] is in [sp_mdot]
                    mdot_row = np.append(mdot_row, ga_table[i,3])

            arr = []
            for i in range(len(ga_sp_v_n)):
                # print('\n')
                arr_mdots = np.zeros(5)
                for j in range(len(mdot_row)):
                    ga_mdot_i = Math.find_nearest_index(ga_table[:, 3], mdot_row[j])
                    sp_mdot_i = Math.find_nearest_index(sp_table[:, 3], mdot_row[j])
                    ga_val = ga_table[ga_mdot_i, np.int(ga_i_val[i])]
                    sp_val = sp_table[sp_mdot_i, np.int(sp_i_val[i])]

                    err = np.abs(ga_val - sp_val)
                    err2= (ga_val - err) / 100      # in % with respect to GA

                    arr_mdots = np.vstack((arr_mdots, [mdot_row[i], ga_val, sp_val, err, err2]))

                arr = np.append(arr, np.delete(arr_mdots, 0, 0))

            arr = np.reshape(arr, (len(ga_sp_v_n), len(mdot_row), 5))

            return arr

        def extrapolate_crit_val(v_n):

            grid_mdot = ga_table[1:, 3] # mdot
            mdot = sp_table[:, 3]
            tau = []

            if not v_n in sp_head:
                raise NameError('v_n: {} is not in sp_head: {}'.format(v_n, sp_head))

            for i in range(len(sp_head)):

                if sp_head[i] == v_n:
                    tau = sp_table[:, i]
                    break
            print(mdot, tau)

            md_tau = []
            for i in range(len(mdot)):
                md_tau = np.append(md_tau, [mdot[i], tau[i]])

            md_tau_sort = np.sort(md_tau.view('f8, f8'), order=['f0'], axis=0).view(np.float)
            md_tau_sh = np.reshape(md_tau_sort, (np.int(len(md_tau)/2), 2))




            tau_grid1 = interpolate.InterpolatedUnivariateSpline(md_tau_sh[:,0], md_tau_sh[:,1])(grid_mdot)
            tau_cr1 = interpolate.InterpolatedUnivariateSpline(md_tau_sh[:, 0], md_tau_sh[:, 1])(ga_table[0, 3])

            tau_grid2 = interpolate.UnivariateSpline(md_tau_sh[:, 0], md_tau_sh[:, 1])(grid_mdot)
            tau_cr2 = interpolate.UnivariateSpline(md_tau_sh[:, 0], md_tau_sh[:, 1])(ga_table[0, 3])

            tmp, tau_grid3 = Math.fit_plynomial(md_tau_sh[:,0], md_tau_sh[:,1], 4, 0, grid_mdot)
            tmp, tau_cr3 = Math.fit_plynomial(md_tau_sh[:, 0], md_tau_sh[:, 1], 4, 0, np.array([ga_table[0, 3]]))

            tmp, tau_grid4 = Math.fit_plynomial(md_tau_sh[:,0], md_tau_sh[:,1], 3, 0, grid_mdot)
            tmp, tau_cr4 = Math.fit_plynomial(md_tau_sh[:, 0], md_tau_sh[:, 1], 3, 0, np.array([ga_table[0, 3]]))

            tmp, tau_grid5 = Math.fit_plynomial(md_tau_sh[:,0], md_tau_sh[:,1], 2, 0, grid_mdot)
            tmp, tau_cr5 = Math.fit_plynomial(md_tau_sh[:, 0], md_tau_sh[:, 1], 2, 0, np.array([ga_table[0, 3]]))

            tmp, tau_grid6 = Math.fit_plynomial(md_tau_sh[:,0], md_tau_sh[:,1], 1, 0, grid_mdot)
            tmp, tau_cr6 = Math.fit_plynomial(md_tau_sh[:, 0], md_tau_sh[:, 1], 1, 0, np.array([ga_table[0, 3]]))




            plt.plot(md_tau_sh[:,0], md_tau_sh[:,1], '.', color = 'black')
            plt.plot(grid_mdot, tau_grid1, '--', color='gray', label = '{}_cr(IntUn)  = {}'.format(v_n, "%.2f" % ( tau_cr1)))
            plt.plot(grid_mdot, tau_grid2, '--', color='blue', label = '{}_cr(Univ) = {}'.format(v_n, "%.2f" % ( tau_cr2)))
            plt.plot(grid_mdot, tau_grid3, '--', color='orange', label='{}_cr(ploy4) = {}'.format(v_n, "%.2f" % ( tau_cr3)))
            plt.plot(grid_mdot, tau_grid4, '--', color='red', label='{}_cr(ploy3) = {}'.format(v_n, "%.2f" % ( tau_cr4)))
            plt.plot(grid_mdot, tau_grid5, '--', color='cyan', label='{}_cr(ploy2) = {}'.format(v_n, "%.2f" % ( tau_cr5)))
            plt.plot(grid_mdot, tau_grid6, '--', color='black', label='{}_cr(ploy1) = {}'.format(v_n, "%.2f" % ( tau_cr6)))


            plt.axvline(x=ga_table[0, 3], label='Critical point')
            plt.legend()
            plt.grid()
            plt.xlabel(Labels.lbls('mdot'))
            plt.ylabel(v_n)
            if show_plot:
                plt.show()

            return np.append(np.array([tau_cr2]), tau_grid1) # Returns the Univariant interpolation!




        arr = compare_sonic_values()

        # 'log(L)', 'M(Msun)', 'Yc', 'mdot', 'r-sp', 't-sp', 'R_wind', 'T_eff', 'Tau'
        tau = extrapolate_crit_val('Tau')

        ga_head.insert(6,'Tau')
        tmp = ''
        for i in range(len(ga_head)):
            tmp = tmp+ga_head[i]

        ga_head = tmp
        ga_table = np.insert(ga_table, 6, tau, 1)


        # --- --- --- MAKING A OUTPUT FILE NAME OUT OF FOLDERS THE SM.DATA FILES CAME FROM --- --- ---
        out_name = 'SP2'
        for i in range(len(self.input_dirs)):
            if self.input_dirs[i] not in self.dirs_not_to_be_included and self.input_dirs[i] != '..':
                out_name = out_name + self.input_dirs[i]
                if i < len(self.input_dirs) - 1:
                    out_name = out_name + '_'
        out_name = out_name + '.data'

        print('Results are saved in: {}'.format(self.out_dir + out_name))

        np.savetxt(self.out_dir + out_name, ga_table, '%.5f', '  ', '\n', ga_head, '')


        # print(arr)

        print('a')
class Read_Tau:

    def __init__(self, plotfiles, smfiles, out_dir, plot_dir, dirs_not_to_be_included):

        self.input_dirs = plotfiles[0].split('/')[:-1]
        # print(self.input_dirs)
        self.dirs_not_to_be_included = dirs_not_to_be_included # name of folders that not to be in the name of out. file.

        self.out_dir = out_dir
        self. plot_dir = plot_dir

        self.plt_files = plotfiles
        self.sm_files = smfiles
        self.plotmdl = []
        self.smmdl = []

        if len(smfiles) != len(plotfiles):
            raise IOError('len(smfiles)[{}] != len(plotfiles)[{}]'.format(len(smfiles), len(plotfiles)))

        for file in plotfiles:
            self.plotmdl.append(Read_Plot_file.from_file(file))

        for file in smfiles:
            self.smmdl.append(Read_SM_data_file.from_sm_data_file(file))

    def tau(self):

        mdot_tau = []
        for i in range(len(self.plt_files)):
            t = self.plotmdl[i].t_eff[-1]
            mdot =  self.plotmdl[i].mdot_[-1]
            tau =  self.plotmdl[i].tauatR[-1]
            windform_24= self.plotmdl[i].windform_24[-1]
            mdot_tau = np.append(mdot_tau, [mdot, tau, windform_24, t])

        mdot_tau_shape = np.sort(mdot_tau.view('f8, f8, f8, f8'), order=['f0'], axis=0).view(np.float)
        mdot_tau_sorted = np.reshape(mdot_tau_shape, (len(self.plt_files), 4))



        mdot_grid = np.mgrid[-6.:-5.:100j]
        mdot_grid, tau_grid = Math.fit_plynomial(mdot_tau_sorted[:,0], mdot_tau_sorted[:,1], 4, 100, mdot_grid)




        plt.plot(mdot_tau_sorted[:,0], mdot_tau_sorted[:,1], 'x', color = 'black')
        plt.plot(mdot_grid, tau_grid, '-', color = 'red')
        plt.axvline(x=-4.88096)
        plt.grid()
        plt.title('M:20sm Yc:0.993 SP_BEC .plot1')
        # plt.legend()
        plt.xlabel('mdot')
        plt.ylabel('tau')
        plt.show()

        mdot_t = []
        for i in range(len(self.sm_files)):
            mdot =  self.smmdl[i].get_col('mdot')[-1]
            r    =  self.smmdl[i].get_col('r')
            u    =  self.smmdl[i].get_col('u')

            plt.plot(r, u, '.', color='gray')
            mdot_t = np.append(mdot_t, [mdot, r[-1], u[-1]])

        mdot_t_shape  = np.sort(mdot_t.view('f8, f8, f8'), order=['f0'], axis=0).view(np.float)
        mdot_t_sorted = np.reshape(mdot_t_shape, (len(self.sm_files), 3))



        sp_file = np.loadtxt('../data/sp_files/20z0008/SP_ga_z0008_20sm_y10.data')


        plt.plot(mdot_t_sorted[:,1], mdot_t_sorted[:,2], 'x', color = 'black')
        plt.plot(sp_file[1:,3], sp_file[1:,4], '-', color = 'red')
        # plt.axvline(x=-4.88096)
        plt.grid()
        # plt.title('M:20sm Yc:0.993 SP_BEC sm.data')
        # plt.legend()
        plt.xlabel(Labels.lbls('mdot'))
        plt.ylabel(Labels.lbls('r'))
        plt.show()

        print('plot: {}, sm: {}'.format(len(self.plt_files), len(self.sm_files)))

    def tau_crit(self):
        mdot_tau = []
        for i in range(len(self.plt_files)):
            mdot = self.plotmdl[i].mdot_[-1]
            tau = self.plotmdl[i].tauatR[-1]
            windform_24 = self.plotmdl[i].windform_24[-1]
            mdot_tau = np.append(mdot_tau, [mdot, tau, windform_24])

