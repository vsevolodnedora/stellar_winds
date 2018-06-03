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
from Phys_Math_Labels import Math
from Phys_Math_Labels import Physics
from Phys_Math_Labels import Constants
from Phys_Math_Labels import Labels

from FilesWork import Read_SM_data_file
from FilesWork import Read_Wind_file

class SonicPointConditions:
    '''
    This class takes set of sm_cls (from reading the sm.data diles), plots their velocity [plot_cls[0]] and their
    temperature [plot_cls[1]] profiles, and interpolates the sonic point values and the max. values of vel. and temp.

        Finds and plots maxiumums of every profile.
        Finds and plots points, where the profile crosses the sonic profiles (us or ts)
        Finds the sonic point values of values of v_ns given in 'v_n_sonic'
        Finds the core (-0) and surface (-1) values of v_ns in 'v_n_core_surf'

    Main method is 'main_sonic_cycle' returns :
        r_u_mdot_max, r_t_mdot_max, out_names, out_arr
        where
            r_u_mdot_max - [r, u, mdot] 2d array of maximum 'u' in every profile
            r_t_mdot_max - [r, t, mdot] 2d array of maximum 't' in every profile
            out_names - list of [v_n_core_surf, r-sp, t-sp, u-sp, v_n_sonic] v_ns (names of vars, conditions [v_n-cond])
            out_arr - 2d array of values of v_ns in out_names for every model (if sonic point is resolved)
    '''
    def __init__(self, sm_cls, plot_cls, depth, v_n_sonic, v_n_core_surf):

        self.smdl = sm_cls
        self.ax = plot_cls
        self.depth = depth
        self.v_n_sonic = v_n_sonic
        self.v_n_core_surf = v_n_core_surf

        if not 'mdot-1' in v_n_core_surf: raise NameError('v_n = *mdot-1* is not in v_n_core_surf. Give: {}'.format(v_n_core_surf))

        self.ax[0].set_xlabel(Labels.lbls('r'))
        self.ax[0].set_ylabel(Labels.lbls('u'))

        lbl = ''
        for i in range(len(self.smdl)):
            lbl = lbl + str(i) + ' : ' + "%.2f" % self.smdl[i].get_col('He4')[0] + \
                  ' ' + "%.2f" % (-1 * self.smdl[i].get_col('mdot')[-1]) + '\n'
        self.ax[0].text(0.0, 0.7, lbl, style='italic',
                   bbox={'facecolor': 'cyan', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center',
                   verticalalignment='center', transform=self.ax[0].transAxes)

        self.max_ind_ = -10 # for cutting the end of the vel. prof.
        self.u_min = 0.1    #

        # --- OUTPUT --- PUBLC vars
        self.r_u_mdot_max = []
        self.r_t_mdot_max = []

        self.out_names = []
        self.out_arr = []

        self.plot_all_init_vel_profs()
        pass

    def plot_all_init_vel_profs(self):  # black, orange
        for cl in self.smdl:

            r = cl.get_col('r')
            u = cl.get_col('u')
            t = cl.get_col('t')
            r_min = self.get_boundary()
            min_ind = Math.find_nearest_index(r, r_min)
            # max_ind = crop_ends(r, u)
            # print('__ {}:{}'.format(min_ind, max_ind))

            self.ax[0].plot(r[min_ind:], u[min_ind:], '.', color='black')  # vel. prof
            self.ax[1].plot(r[min_ind:], t[min_ind:], '.', color='orange')  # temp. prof

    def get_boundary(self):
        '''
        RETURNS ' bourders.min() ' - min radius among all the models, where u exceeds the 'u_min'
        :param u_min:
        :return:
        '''
        bourders = []

        for i in range(len(self.smdl)):
            u = self.smdl[i].get_col('u')
            r = self.smdl[i].get_col('r')
            for i in range(len(r)):
                if u[i] > self.u_min:
                    # ax1.axvline(x=r[i], color='red')
                    bourders = np.append(bourders, r[i])
                    break

        return bourders.min()

    def plot_interpol_vel_prof(self, cl, min_ind, max_ind):
        r = cl.get_col('r')
        u = cl.get_col('u')
        u_s = cl.get_sonic_u()
        mdot = cl.get_col('mdot')[-1]

        self.ax[0].annotate(str('%.2f' % mdot), xy=(r[-1], u[-1]), textcoords='data')  # when the end is not yet subtracted

        r = r[min_ind:max_ind]
        u = u[min_ind:max_ind]
        u_s = u_s[min_ind:max_ind]

        int_r = np.mgrid[r[0]:r[-1]:self.depth * 1j]
        int_u = Math.interp_row(r, u, int_r)

        self.ax[0].plot(r, u_s, '--', color='black')  # Original
        self.ax[0].plot(int_r, int_u, '-', color='gray')  # Interpolated

        return int_r, int_u

    def plot_interpol_t_prof(self, cl, min_ind, max_ind):

        r = cl.get_col('r')
        u = cl.get_col('u')
        mu = cl.get_col('mu')
        t = cl.get_col('t')

        r = r[min_ind:max_ind]
        u = u[min_ind:max_ind]

        t = t[min_ind:max_ind]
        mu = mu[min_ind:max_ind]

        ts_arr = np.log10((mu * Constants.m_H * (u * 100000) ** 2) / Constants.k_b)

        int_r = np.mgrid[r[0]:r[-1]:self.depth * 1j]
        int_t = Math.interp_row(r, t, int_r)
        int_ts_arr = Math.interp_row(r, ts_arr, int_r)

        self.ax[1].plot(r, ts_arr, '.', color='gray')
        # ax[1].plot(int_r, int_t,'-',color='gray')
        self.ax[1].plot(int_r, int_ts_arr, '--', color='gray')

        return int_t, int_ts_arr

    def plot_max_vel_temp(self, cl, int_r, int_u, int_ts_arr):
        '''
        Uses 'get_max_by_interpolating' to get max in every vel/ts profile and plot with cross
        :param int_r:
        :param int_u:
        :return:
        '''
        mdot_u = cl.get_col('mdot')[-1]

        r_u_max_p, u_max_p = Math.get_max_by_interpolating(int_r, int_u)  # MAXIMUM VALUES OF VELOCITY
        r_u_mdot_max = np.array([r_u_max_p[0], u_max_p[0], mdot_u])
        r_t_max_p, t_max_p = Math.get_max_by_interpolating(int_r, int_ts_arr)  # MAXIMUM VALUES OF TEMPERATURE
        r_t_mdot_max = np.array([r_t_max_p[0], t_max_p[0], mdot_u])

        self.ax[0].plot(r_u_max_p, u_max_p, 'x', color='red')
        self.ax[1].plot(r_t_max_p, t_max_p, 'x', color='red')

        return r_u_mdot_max, r_t_mdot_max

    def plot_rs_ts(self, cl, int_r, int_t, int_ts_arr):

        rs_p, ts_p, us_p = None, None, None  # initialize the variables
        rs_p, ts_p = Math.interpolated_intercept(int_r, int_ts_arr, int_t)  # SONIC TEPERATURE

        if rs_p.any():
            rs_p = rs_p[0][0]
            ts_p = ts_p[0][0]

            us = cl.get_sonic_u()
            r = cl.get_col('r')

            us_p = interpolate.InterpolatedUnivariateSpline(r, us)(rs_p)

            self.ax[0].plot(rs_p, us_p, 'X', color='blue')
            self.ax[0].annotate(str('%.2f' % ts_p), xy=(rs_p, us_p), textcoords='data')

            self.ax[1].plot(rs_p, ts_p, 'X', color='blue')
            self.ax[1].annotate(str('%.2f' % ts_p), xy=(rs_p, ts_p), textcoords='data')

            # row = all_values_array(cl, rs_p, ts_p, min_ind, v_n_sonic)

        return rs_p, ts_p, us_p
        # out_array = np.vstack((out_array, row))

    def out_row_core_surf_vals(self, cl):
        '''
        returns row with values taken at the core (-0) or at the surface (-1).
        :param cl:
        :param v_n_arr: [...'l-1', 'He4-1', 'xm-1...]
        :return:
        '''
        row = []
        for v_n_cond in self.v_n_core_surf:
            if len(v_n_cond.split('-')) > 2: raise NameError(
                'Coditions should be [v_n-where], given: {}'.format(v_n_cond))
            v_n = v_n_cond.split('-')[0]
            cond = v_n_cond.split('-')[-1]
            if cond != '0' and cond != '1': raise NameError(
                'Condition should be 1 or 0 (surface or core) given: {}'.format(cond))

            if cond == '0': cond = 0
            if cond == '1': cond = -1

            value = cl.get_col(v_n)[cond]

            row = np.append(row, value)

        if len(row) != len(self.v_n_core_surf):
            raise ValueError('len(row){} != len(v_n_cond){}'.format(len(row), len(self.v_n_core_surf)))
        return row

    def append_sp_vals(self, cl, rs_p, min_ind):

        out_row = []
        for v_n_cond in self.v_n_sonic:

            if len(v_n_cond.split('-')) > 2:
                raise NameError(
                    'For *add_sonic_vals* use format *var_name-location*, (given: {}) where var_name is '
                    'one of the BEC sm.data variables and location is *core = [-1] surface=[0]',
                    'sp = [sonic_point_interpolated]'.format(v_n_cond))

            v_n = v_n_cond.split('-')[0]
            cond = v_n_cond.split('-')[-1]

            if cond != 'sp':
                var_val = cl.get_cond_value(v_n, cond)  # assuming that condition is not required interp

            else:
                ''' Here The Interpolation of v_n is Done '''

                r = cl.get_col('r')[min_ind:]
                v_n_val_arr = cl.get_col(v_n)[min_ind:]

                f = interpolate.InterpolatedUnivariateSpline(r, v_n_val_arr)
                var_val = f(rs_p)

                if len([var_val]) > 1:
                    raise ValueError('More than one solution found for *{}* sonic value: ({})'.format(v_n, var_val))

            out_row = np.append(out_row, var_val)

        # if len(out_row) != len(add_sonic_vals):
        #     raise ValueError('len(val_array)[{}] != len(add_sonic_vals)[{}]'
        #                      .format(len(out_row), len(add_sonic_vals)))

        return out_row

    def combine_core_surf_sonic(self, core_surf, rs_p, ts_p, us_p, sp_row):
        '''
        Returns the [v_n_core_surf, r-sp, t-sp, u-sp, v_n_arr] Array
        :return:
        '''

        out_row = []


        for value in core_surf:
            out_row = np.append(out_row, value)

        if rs_p != None:
            out_row = np.append(out_row, rs_p)
            out_row = np.append(out_row, ts_p)
            out_row = np.append(out_row, us_p)

        for value in sp_row:
            out_row = np.append(out_row, value)

        if rs_p != None:  # Return only if sonic point is found (subsonic is not considered)
            return out_row
        else:
            return np.empty(0, )

    # --- PUBLIC-----------------
    def main_sonic_cycle(self):
        '''
        Plots Vel and Temp Profs (origin + interpol), cutting the beginning (and the end).
        Finds max in every profile. Finds the sonic Point (r, t) if exists
        Constructs the out_arr, with [v_n_core_surf, rs, ts, us, v_n_sonic] values ( NOT critical )
        :param ax:
        :param sm_cls:
        :return:
        '''

        def combine_names(core_surf, middle, sonic):

            out_names = []
            for v_n in core_surf:
                out_names.append(v_n)
            for v_n in middle:
                out_names.append(v_n)
            for v_n in sonic:
                out_names.append(v_n)
            return out_names

        r_u_mdot_max = np.array([0., 0., 0.])
        r_t_mdot_max = np.array([0., 0., 0.])
        out_arr = np.zeros(len(self.v_n_core_surf) + 3 + len(self.v_n_sonic))  # .. + rs ts us + ...
        out_names = []

        for cl in self.smdl:

            r = cl.get_col('r')
            r_min = self.get_boundary()

            print('\t__Initical Array Length: {}'.format(len(r)))

            if self.max_ind_ != 0:
                max_ind = len(r) - self.max_ind_
            else:
                max_ind = len(r)  # as for sonic-BEC this is not necessary.

            min_ind = Math.find_nearest_index(r, r_min)
            if min_ind > max_ind: raise ValueError('Index Problem: min_ind({}) > max_ind({})'.format(min_ind, max_ind))

            r_int, u_int = self.plot_interpol_vel_prof(cl, min_ind, max_ind)
            t_int, ts_int = self.plot_interpol_t_prof(cl, min_ind, max_ind)

            r_u_mdot_max_, r_t_mdot_max_ = self.plot_max_vel_temp(cl, r_int, u_int, ts_int)
            r_u_mdot_max = np.vstack((r_u_mdot_max, r_u_mdot_max_))
            r_t_mdot_max = np.vstack((r_t_mdot_max, r_t_mdot_max_))

            rs_p, ts_p, us_p = self.plot_rs_ts(cl, r_int, t_int, ts_int)  # get rs ts us

            core_surf = self.out_row_core_surf_vals(cl)  # get arr of core_surf values

            sp_row = self.append_sp_vals(cl, rs_p, min_ind)  # get arr or sp values (use rs and interpolating)

            out_row = self.combine_core_surf_sonic(core_surf, rs_p, ts_p, us_p, sp_row,)  # combine all in out_row

            if out_row.any():  # stack out_row on top of each other, - 2d array
                out_arr = np.vstack((out_arr, out_row))

        r_u_mdot_max = np.delete(r_u_mdot_max, 0, 0)  # removing the 0th row with zeros
        r_t_mdot_max = np.delete(r_t_mdot_max, 0, 0)
        out_arr = np.delete(out_arr, 0, 0)
        out_names = combine_names(self.v_n_core_surf, ['r-sp', 't-sp', 'u-sp'], self.v_n_sonic)

        self.ax[0].plot(r_u_mdot_max[:, 0], r_u_mdot_max[:, 1], '-.', color='blue')
        self.ax[1].plot(r_t_mdot_max[:, 0], r_t_mdot_max[:, 1], '-.', color='blue')


        return r_u_mdot_max, r_t_mdot_max, out_names, out_arr

class CriticalConditions:
    def __init__(self, ax, sm_cls, r_u_mdot_max, r_t_mdot_max):

        self.smdl = sm_cls

        self.r_u_mdot_max = r_u_mdot_max
        self.r_t_mdot_max = r_t_mdot_max

        self.ax = ax

        self.u_min = 0.1
        self.depth = 1000

        pass

    def get_boundary(self):
        '''
        RETURNS ' bourders.min() ' - min radius among all the models, where u exceeds the 'u_min'
        :param u_min:
        :return:
        '''
        bourders = []

        for i in range(len(self.smdl)):
            u = self.smdl[i].get_col('u')
            r = self.smdl[i].get_col('r')
            for i in range(len(r)):
                if u[i] > self.u_min:
                    # ax1.axvline(x=r[i], color='red')
                    bourders = np.append(bourders, r[i])
                    break

        return bourders.min()

    def cross(self, mdot, r1, u_or_t1, r2, u_or_ts2, mdot_maxs, u_ot_t='u'):  # interpol. up up to max of u_s [green]
        '''
        Finds the delta = (u_i_max - cs_i) where u_i is a maximum of u profile along the r, cs_i - point along sonic
        velocity profile, that lies on the line thac connects the u_i maximums.
        As close the mass loss to a critical one, the closer delta to 0 ( u_i_max = cs_i at the critical
        (deflection) point)

        :param mdot:
        :param r1: full length of a sonic profile (r)
        :param u_or_t1: values of sonic velocity along (r)
        :param r2: set of 'r' points for maximums of velocity profiles for every mass loss
        :param u_or_ts2: values of velocity at 'r' points
        :param mdot_maxs: mdot values of every point 'r,u' above
        :return:
        '''

        r1_u1 = []
        for i in range(len(r1)):
            r1_u1 = np.append(r1_u1, [r1[i], u_or_t1[i]])

        r1_u1 = np.sort(r1_u1.view('f8, f8'), order=['f1'], axis=0).view(np.float)
        r1_u1_sort = np.reshape(r1_u1, (len(r1), 2))

        r2_u2_mdot = []
        for i in range(len(r2)):
            r2_u2_mdot = np.append(r2_u2_mdot, [r2[i], u_or_ts2[i], mdot_maxs[i]])

        r2_u2_mdot = np.sort(r2_u2_mdot.view('f8, f8, f8'), order=['f1'], axis=0).view(np.float)
        r2_u2_mdot_sort = np.reshape(r2_u2_mdot, (len(r2), 3))

        r1 = r1_u1_sort[:, 0]
        u_or_t1 = r1_u1_sort[:, 1]

        r2 = r2_u2_mdot_sort[:, 0]
        u_or_ts2 = r2_u2_mdot_sort[:, 1]
        mdots = r2_u2_mdot_sort[:, 2]

        #
        # u_rmax1 = u1[np.where(r1 == r1.max())]
        # u_rmax2 = u2[np.where(r2 == r2.max())]
        #
        # r_umin1_i = r1[np.where(u1 == u1.min())]
        # r_umin2_i = r2[np.where(u2 == u2.min())]

        '''Interpolating only up to the max of sonic velo (u_ot_t2) to avoid violent behaviour'''
        i_u2_where_u1_max = Math.find_nearest_index(u_or_ts2,
                                                    u_or_t1.max())  # needed to avoide violent behaviour in high mdot
        u2_crop = u_or_ts2[:i_u2_where_u1_max]  # otherwise there was crossing with sonic profiles at temps 5.9,
        r2_crop = r2[:i_u2_where_u1_max]

        u_lim1 = np.array([u_or_t1.min(), u2_crop.min()]).max()
        u_lim2 = np.array([u_or_t1.max(), u2_crop.max()]).min()

        if u_lim2 < u_lim1:
            raise ValueError('u_lim1({}) < u_lim2({})'.format(u_lim1, u_lim2))

        u_or_t_grid = np.mgrid[u_lim2:u_lim1:self.depth * 1j]

        if u_or_t_grid.max() > u_or_ts2.max() or u_or_t_grid.max() > u_or_t1.max():
            raise ValueError('u_or_t_grid.max({}) > u2.max({}) or u_or_t_grid.max({}) > u1.max({})'
                             .format(u_or_t_grid.max(), u_or_ts2.max(), u_or_t_grid.max(), u_or_t1.max()))
        #
        # if u_or_t_grid.min() < u2.min() or u_or_t_grid.min() < u1.min():
        #     raise ValueError('u_or_t_grid.min({}) < u2.min({}) or u_or_t_grid.min({}) < u1.min({})'
        #                      .format(u_or_t_grid.min(), u2.min(), u_or_t_grid.min(), u1.min()))

        f1 = interpolate.InterpolatedUnivariateSpline(u_or_t1, r1)
        r1_grid = f1(u_or_t_grid)

        # if u_ot_t == 't':
        # print('\t')
        # print('u1:[{} {}] u_lim1:{} u_lim2:{}'.format(u1.min(), u1.max(), u_lim1, u_lim2))
        # print('u2:[{} {}] u_lim1:{} u_lim2:{}'.format(u2.min(), u2.max(), u_lim1, u_lim2))
        #

        # ax2.plot(r2_crop, u2_crop, 'o', color='magenta')

        f2 = interpolate.InterpolatedUnivariateSpline(u2_crop,
                                                      r2_crop)  # cropping is done to cut the high mdot prob
        r2_grid = f2(u_or_t_grid)

        # r1_grid = Math.interp_row(u1, r1, u_or_t_grid)
        # r2_grid = Math.interp_row(u2, r2, u_or_t_grid)

        if u_ot_t == 'u':
            self.ax[0].plot(r1_grid, u_or_t_grid, '-.', color='green')
            self.ax[0].plot(r2_grid, u_or_t_grid, '-.', color='green')
        else:
            self.ax[1].plot(r1_grid, u_or_t_grid, '-.', color='green')
            self.ax[1].plot(r2_grid, u_or_t_grid, '-.', color='green')

        uc, rc = Math.interpolated_intercept(u_or_t_grid, r1_grid, r2_grid)
        if uc.any():  # if there is an intersections between sonic vel. profile and max.r-u line
            uc0 = uc[0][0]
            rc0 = rc[0][0]

            if u_ot_t == 'u':
                self.ax[0].plot(rc0, uc0, 'X', color='green')
                self.ax[0].annotate(str('%.2f' % mdot), xy=(rc0, uc0), textcoords='data')
            if u_ot_t == 't':
                self.ax[1].plot(rc0, uc0, 'X', color='green')
                self.ax[1].annotate(str('%.2f' % mdot), xy=(rc0, uc0), textcoords='data')

            delta = u_or_ts2[np.where(mdots == mdot)] - uc0
            # print(uc, rc, '\t', delta)
            # print('Delta: ' , delta_ut, )

            return delta

        # if u_min2 < u_min1:        # if there is a common area in terms of radii
        #     if r_umin1 > r_umin2:   # if there is a common area in case of velocity
        #         u_lim1 = u_min1
        #         u_lim2 = u_max1
        #
        #         u_or_t_grid = np.mgrid[u_lim2:u_lim1:1000*1j]
        #
        #         r1_grid = Math.interp_row(u1, r1, u_or_t_grid)
        #         r2_grid = Math.interp_row(u2, r2, u_or_t_grid)
        #
        #         ax1.plot(r1_grid, u_or_t_grid, '-', color='green')
        #         ax1.plot(r2_grid, u_or_t_grid, '-', color='green')
        #
        #         uc, rc = Math.interpolated_intercept(u_or_t_grid, r1_grid, r2_grid)
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

    def get_mdot_delta(self, r_ut_mdot_max, u_or_t):

        mdot_delta_ut = []

        n = 0
        for cl in self.smdl:

            r_min = self.get_boundary()
            r = cl.get_col('r')
            min_ind = Math.find_nearest_index(r, r_min)

            r = cl.get_col('r')[min_ind:]
            t = cl.get_col('t')[min_ind:]
            u_s = cl.get_sonic_u()[min_ind:]
            mdot = cl.get_col('mdot')[-1]

            if u_or_t == 'u':
                delta_ut = self.cross(mdot, r, u_s, r_ut_mdot_max[1:, 0], r_ut_mdot_max[1:, 1], r_ut_mdot_max[1:, 2],
                                 u_or_t)
                # print(delta_ut, u_or_t)
            else:
                delta_ut = self.cross(mdot, r, t, r_ut_mdot_max[1:, 0], r_ut_mdot_max[1:, 1], r_ut_mdot_max[1:, 2],
                                 u_or_t)
                # print(delta_ut, u_or_t)

            if delta_ut != None:
                mdot_delta_ut = np.append(mdot_delta_ut, [mdot, delta_ut])
                n = n + 1

        if len(mdot_delta_ut) == 0:
            raise ValueError('mdot_delta_ut is not found at all for <{}>'.format(u_or_t))

        mdot_delta_ut = np.sort(mdot_delta_ut.view('f8, f8'), order=['f0'], axis=0).view(np.float)
        mdot_delta_ut_shape = np.reshape(mdot_delta_ut, (n, 2))

        mdot = mdot_delta_ut_shape[:, 0]
        delta_ut = mdot_delta_ut_shape[:, 1]

        crit_mdot_u = Math.solv_inter_row(mdot, delta_ut, 0.)  # Critical Mdot when the delta_ut == 0

        print('\t\t crit_mdot', crit_mdot_u)

        if not crit_mdot_u.any():
            raise ValueError('Critical Mdot is not found.')
        else:
            print('\t__Critical Mdot: {} (for: {})'.format(crit_mdot_u, u_or_t))


        # Sorting for delta rising (for future interpolation)
        mdot_arr = mdot
        delta_arr= delta_ut

        arr = []
        for i in range(len(mdot)):
            arr = np.append(arr, [mdot[i], delta_ut[i]])

        arr_sort = np.sort(arr.view('f8, f8'), order=['f1'], axis=0).view(np.float)
        arr_shaped = np.reshape(arr_sort, (len(mdot), 2))

        mdot_arr = arr_shaped[:, 0]
        delta_arr = arr_shaped[:, 1]


        if delta_ut.min() > 0. and delta_ut.max() > 0.:
            raise ValueError('if delta_arr.min({}) and delta_arr.max({}) > 0 : \n'
                             'peak of vel. profs is not crossing sonic. vel.'.format(delta_arr.min(),
                                                                                     delta_arr.max()))

        if delta_ut.min() < 0. and delta_ut.max() < 0.:
            raise ValueError('if delta_arr.min({}) and delta_arr.max({}) < 0 : \n'
                             'vel. profile is not crossing the sonic val.'.format(delta_arr.min(), delta_arr.max()))

        return mdot_arr, delta_arr

    def plot_critical_mdot(self, r_ut_mdot_max, u_or_t):
        '''
        Takes the array of [r, u_max, mdot] and [r ts_max, mdot] and by construting the delta values
        between every u_max and u_sonic, (perpendicular to u_sonic), plots the u_i - u_s = f(mdot)
        Using this it interpolates the mdot at which where u_i - u_s = 0, and this is the critcial mdot.
        Returns the mdot_crit, r_crit, and t_crit.
        :param ax:
        :param sm_cls:
        :param r_u_mdot_max:
        :param r_t_mdot_max:
        :u_min: 0.1 for cutting the profiles to [min_ind:]
        :return: critical mdot
        '''

        # interpol. up up to max of u_s [green]

        mdot_arr, delta_arr = self.get_mdot_delta(r_ut_mdot_max, u_or_t)

        crit_mdot = interpolate.InterpolatedUnivariateSpline(delta_arr, mdot_arr)(0.)

        if u_or_t == 'u':
            n = 3
            self.ax[n].set_ylabel('$u_i - cs_i$')
        else:
            n = 4
            self.ax[n].set_ylabel('$t_i - ts_i$')

        self.ax[n].set_xlabel(Labels.lbls('mdot'))
        self.ax[n].grid()

        self.ax[n].plot(mdot_arr, delta_arr, '-', color='black')
        self.ax[n].plot(crit_mdot, 0., 'x', color='black')
        self.ax[n].annotate('({}, {})'.format('%.3f' % crit_mdot, 0.), xy=(crit_mdot, 0.),
                       textcoords='data')

        return crit_mdot  # returns one value!

    def plot_critical_r(self, r_u_mdot_max, crit_mdot):
        '''
                interpolates the position of the critial mdot in r_ut_mdot_max array of max. values.
                                                                    with respect to axis r or t (if 'u' or 't' are given)
                Plots vertical line in plot[0] or plot[1] and annotates.
                Returns critical r or t
                :param ax:
                :param r_u_mdot_max:
                :param crit_mdot:
                :param u_or_t:
                :return:
                '''
        r_u_mdot_max = np.sort(r_u_mdot_max.view('f8, f8, f8'), order=['f2'], axis=0).view(np.float)
        r_u_mdot_max = np.reshape(r_u_mdot_max, (len(self.smdl), 3))

        f = interpolate.InterpolatedUnivariateSpline(r_u_mdot_max[:, 2], r_u_mdot_max[:, 0])  # r_crit
        crit_r = f(crit_mdot)

        print('\t__Criticals: r={}, mdot={}'.format('%.4f' % crit_r, '%.4f' % crit_mdot))

        self.ax[0].axvline(x=crit_r, color='red')
        self.ax[0].annotate('({})'.format('%.2f' % crit_r), xy=(crit_r, 0), textcoords='data')

        return crit_r

    def plot_critical_t(self, r_t_mdot_max, crit_mdot):
        '''
                interpolates the position of the critial mdot in r_ut_mdot_max array of max. values.
                                                                    with respect to axis r or t (if 'u' or 't' are given)
                Plots vertical line in plot[0] or plot[1] and annotates.
                Returns critical r or t
                :param ax:
                :param r_t_mdot_max:
                :param crit_mdot:
                :param u_or_t:
                :return:
        '''

        r_t_mdot_max = np.sort(r_t_mdot_max.view('f8, f8, f8'), order=['f2'], axis=0).view(np.float)
        r_t_mdot_max = np.reshape(r_t_mdot_max, (len(self.smdl), 3))

        f = interpolate.InterpolatedUnivariateSpline(r_t_mdot_max[:, 2], r_t_mdot_max[:, 1])  # r_crit
        crit_t = f(crit_mdot)

        print('\t__Criticals: t={}, mdot={}'.format('%.4f' % crit_t, '%.4f' % crit_mdot))


        self.ax[1].axhline(y=crit_t, color='red')
        self.ax[1].annotate('({})'.format('%.2f' % crit_t), xy=(crit_t, 0), textcoords='data')

        return crit_t

    # --- PUBLIC ---
    def set_crit_mdot_r_t(self):

        self.mdot_cr_u = self.plot_critical_mdot(self.r_u_mdot_max, 'u')
        self.mdot_cr_t = self.plot_critical_mdot(self.r_t_mdot_max, 't')

        self.r_cr = self.plot_critical_r(self.r_u_mdot_max, self.mdot_cr_u)  # using mdot from r_u plane
        self.t_cr = self.plot_critical_t(self.r_t_mdot_max, self.mdot_cr_t)  # using mdot from t_rs plane

        return self.mdot_cr_u, self.mdot_cr_t, self.r_cr, self.t_cr

    def extrapolate_crit_value(self, v_n_cond, mdot_cr, names, sonic_array, mdot_name='mdot-1', method='IntUni'):

        if len(names)!=len(sonic_array[0, :]):
            raise ValueError('len(sonic_names{})!=len(sonic_array[0,:]{})'.format(len(names), len(sonic_array[0, :])))
        if not v_n_cond in names:
            raise NameError('v_n ({}) not in names ({})'.format(v_n_cond, names))


        if not mdot_name in names:
            raise NameError('mdot-1 not in names ({})'.format(names))

        mdot_col = sonic_array[:, names.index(mdot_name)]
        var_col = sonic_array[:, names.index(v_n_cond)]

        # self.ax[5].plot(mdot_col, var_col, '.', color='black')

        return Math.extrapolate_value(mdot_col, var_col, mdot_cr, method, self.ax[5])

class Master_Gray(SonicPointConditions, CriticalConditions):

    def __init__(self, smfiles, out_dir, plot_dir, dirs_not_to_be_included):
        self.input_dirs = smfiles[0].split('/')[:-1]
        # print(self.input_dirs)
        self.dirs_not_to_be_included = dirs_not_to_be_included  # name of folders that not to be in the name of out. file.

        self.out_dir = out_dir
        self.plot_dir = plot_dir

        self.num_files = smfiles
        self.mdl = []
        self.smdl = []

        for file in smfiles:
            self.mdl.append(Read_SM_data_file.from_sm_data_file(file))

        self.sort_smfiles('mdot', -1)

        fig = plt.figure()
        ax = []
        ax.append(fig.add_subplot(231))
        ax.append(fig.add_subplot(232))
        ax.append(fig.add_subplot(233))
        ax.append(fig.add_subplot(234))
        ax.append(fig.add_subplot(235))
        ax.append(fig.add_subplot(236))

        self.v_n_sonic = ['kappa-sp', 'L/Ledd-sp', 'HP-sp', 'mfp-sp']
        self.v_n_core_surf = ['l-1', 'xm-1', 'He4-0', 'He4-1', 'mdot-1']
        self.plot_cls = ax
        self.depth = 1000

        # Inheritance -----------------------
        SonicPointConditions.__init__(self, self.smdl, self.plot_cls, self.depth, self.v_n_sonic, self.v_n_core_surf)
        self.r_u_mdot_max, self.r_t_mdot_max, self.out_names, self.out_arr = self.main_sonic_cycle()

        CriticalConditions.__init__(self, self.plot_cls, self.smdl, self.r_u_mdot_max, self.r_t_mdot_max)
        self.mdot_cr_u, self.mdot_cr_t, self.r_cr, self.t_cr = self.set_crit_mdot_r_t()

        val_cr = self.extrapolate_crit_value('L/Ledd-sp', self.mdot_cr_u, self.out_names, self.out_arr)

        out_name, out_arr = self.combine_sp_crit(self.mdot_cr_u,
                                                 ['r-sp', 't-sp', 'L/Ledd-sp'],
                                                 [self.r_cr, self.t_cr, val_cr],
                                                 self.out_names, self.out_arr)

        out_name, out_arr = self.add_core_surf_to_crit_raw(out_name, out_arr)

        self.save_out_arr(out_name, out_arr)

        # fig.clear()
        # ax.clear()
        plt.tight_layout()
        fig = plt.gcf()
        plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
        plt.show()

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

    # def combine_sp_crit(out_arr, mdot_cr, r_cr, t_cr, v_n_core_surf, v_n_sonic, filling=0.):
    #     # structure assumed:
    #     # [v_n_core_surf, rs, ts, us, v_n_sonic]
    #     if len(out_arr[0, :]) != (len(v_n_core_surf) + len(v_n_sonic) + 3):  # [rs, ts, us]
    #         raise ValueError(
    #             'Out_arr != all names : {} != {}'.format(len(out_arr[0, :]), len(v_n_core_surf) + len(v_n_sonic) + 3))
    #
    #     zeros = np.zeros(len(out_arr[0, :]))
    #     i_mdot = v_n_core_surf.index('mdot-1')
    #     zeros[i_mdot] = mdot_cr
    #     zeros[len(v_n_core_surf) + 1] = r_cr

    @staticmethod
    def combine_sp_crit(mdot_cr, crit_names, crit_arr, out_names, out_arr):
        # structure assumed:
        # [v_n_core_surf, rs, ts, us, v_n_sonic]


        if len(crit_names) != len(crit_arr):
            raise NameError('len(crit_names){} != len(crit_arr){}'.format(len(crit_names), len(crit_arr)))

        zeros = np.zeros(len(out_arr[0, :]))
        i_mdot = out_names.index('mdot-1')
        zeros[i_mdot] = mdot_cr

        for ii in range(len(crit_names)):
            if not crit_names[ii] in crit_names:
                raise NameError('No sonic row for critical value: {} (only {})'.format(crit_names[ii], out_names))

            i_val = out_names.index(crit_names[ii])
            zeros[i_val] = crit_arr[ii]



        res = np.vstack((zeros, out_arr))

        return out_names, res

    @staticmethod
    def add_core_surf_to_crit_raw(out_names, out_arr, criteria=-1):

        if criteria == -1:
            # Using the last row to get core-surf values.

            for v_n_cond in out_names:
                if v_n_cond.split('-')[-1]=='1' or v_n_cond.split('-')[-1]=='0':
                    i_coll = out_names.index(v_n_cond)
                    value = out_arr[-1, i_coll]
                    out_arr[0, i_coll] = value

            return out_names, out_arr

    def create_file_name(self, first_part='SP3', extension='.data'):
        '''Creates name like:
        SP3_ga_z0008_10sm_y10
        using the initial folders, where the sm.data files are located
        '''
        out_name = first_part
        for i in range(len(self.input_dirs)):
            if self.input_dirs[i] not in self.dirs_not_to_be_included and self.input_dirs[i] != '..':
                out_name = out_name + self.input_dirs[i]
                if i < len(self.input_dirs) - 1:
                    out_name = out_name + '_'
        out_name = out_name + extension
        return out_name

    def save_out_arr(self, out_names, out_arr, sp_fold_head='SP'):

        head = out_names
        table = out_arr

        fname = self.create_file_name(sp_fold_head)

        tmp = ''
        for i in range(len(head)):
            tmp = tmp + head[i] + '    '
        head__ = tmp

        np.savetxt(self.out_dir + fname, table, '%.5f', '  ', '\n', head__, '')

        print(' \n********* TABLE: {} IS SAVED IN {} *********\n'.format(fname, sp_fold_head))






