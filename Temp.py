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
    def __init__(self, sm_cls, plot_cls, depth, v_n_sonic, v_n_core_surf, v_n_env):

        self.smdl = sm_cls
        self.ax = plot_cls
        self.depth = depth
        self.v_n_sonic = v_n_sonic
        self.v_n_core_surf = v_n_core_surf
        self.v_n_env = v_n_env

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

        self.delited_outer_points = -10 # for cutting the end of the vel. prof.
        self.if_not_found_use_last = False
        self.check_for_mult_sp = False
        self.compute_envelope=False
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

        us = cl.get_sonic_u()
        r = cl.get_col('r')
        t = cl.get_col('t')

        if len(rs_p) > 1 and self.check_for_mult_sp:
            plt.show()
            raise ValueError('Multiple Sonic Point Found. (if it is OK, turn of the *check_for_mult_sp*')

        if rs_p.any():              # if the sonic point has to be found by interpolating
            rs_p = rs_p[0][0]
            ts_p = ts_p[0][0]

        else:
            if self.if_not_found_use_last: # If the sonic point is supposed to be the last point
                rs_p = r[-1]
                ts_p = t[-1]

        if rs_p != None:
            us_p = interpolate.InterpolatedUnivariateSpline(r, us)(rs_p)

            self.ax[0].plot(rs_p, us_p, 'X', color='blue')
            self.ax[0].annotate(str('%.2f' % ts_p), xy=(rs_p, us_p), textcoords='data')
            self.ax[1].plot(rs_p, ts_p, 'X', color='blue')
            self.ax[1].annotate(str('%.2f' % ts_p), xy=(rs_p, ts_p), textcoords='data')

        return rs_p, ts_p, us_p
        # out_array = np.vstack((out_array, row))

    @staticmethod
    def plot_envelope_r_t(ax, cls, v_n_env,  guess=5.2):
        '''
        Looks for a loal extremum between t_lim1 and t_lim2, and, if the extremem != sonic point: returns
        length and mass of whatever is left
        :param cls:
        :param t_lim1:
        :param t_lim2:
        :return: tp, up, rp, xmp (coordinates, where envelope begins)
        '''

        def get_envelope_r_or_m(v_n, cls, t_start):
            t = cls.get_col('t')
            ind = Math.find_nearest_index(t, t_start) - 5  # just before the limit, so we don't need to

            # interpolate across the whole t range
            t = t[ind:]
            var = cls.get_col(v_n)
            var = var[ind:]

            value = interpolate.InterpolatedUnivariateSpline(t[::-1], var[::-1])(t_start)

            # print('-_-: {}'.format(var[-1]-value))

            return value


        for v_n in v_n_env:
            if v_n not in ['t-env', 'u-env', 'r-env', 'm-env']:
                raise NameError('Var: {} not in the list. Change the *plot_envelope_r_t* to account for that')

        t = cls.get_col('t')  # x - axis
        u = cls.get_col('u')  # y - axis

        # s = len(t)-1
        # while go_on:
        #     if u[-2] < u[-1]
        #
        # for i in range(len(t)):
        #     if u[s-1] < u[s]:
        #         s = s-1
        #     else:
        #         t = t[:s]
        #         u = u[:s]
        #         break

        # if t.min() > t_lim1 and t_lim2 > t.min():
        # i1 = Math.find_nearest_index(t, t_lim2)
        # i2 = Math.find_nearest_index(t, t_lim1)

        # t_cropped= t[Math.find_nearest_index(t, t_lim2):Math.find_nearest_index(t,
        #                                                                          t_lim1)][::-1]   # t_lim2 > t_lim1 and t is Declining
        # u_cropped = u[Math.find_nearest_index(t, t_lim2):Math.find_nearest_index(t, t_lim1)][::-1]

        # print('<<<<<<<<<<<SIZE: {} {} (i1:{}, i2:{}) >>>>>>>>>>>>>>>>'.format(len(t), len(u), i1, i2))

        tp, up = Math.get_max_by_interpolating(t, u, True, guess)  # WHERE the envelope starts (if any)
        # TRUE for cutting out the rising part in the end
        if Math.find_nearest_index(t, tp) < len(t) - 1:  # if the tp is not the last point of the t array

            print('<<<<<<<<<<<Coord: {} {} >>>>>>>>>>>>>>>>'.format("%.2f" % tp, "%.2f" % up))

            r_env = get_envelope_r_or_m('r', cls, tp)
            m_env = get_envelope_r_or_m('xm', cls, tp)
            m = cls.get_col('xm')[-1]
            r = cls.get_col('r')[-1]

            ax[0].plot(r_env, up, 'X', color='black')  # location of the onset of an envelope
            ax[1].plot(r_env, tp, 'X', color='black')

            return np.array([tp, up, r-r_env, np.log10(m-m_env)])
        else:
            return np.array([0., 0., 0, 0])
        #     # print('L_env: {}'.format(get_envelope_r_or_m('r', smcl, tp)))
        #     # print('M_env: {}'.format(get_envelope_r_or_m('xm', smcl, tp)))
        #
        #     # var = get_envelope_r_or_m('r', smcl, tp)
        #     if m_or_r == 'xm': return tp, up, np.log10(get_envelope_r_or_m('xm', self, tp))
        #     if m_or_r == 'r': return tp, up, get_envelope_r_or_m('r', self, tp)
        #     raise NameError('m_or_r = {}'.format(m_or_r))
        # else:
        #     return 0.  # default value if there is no envelope

        # else: return None, None

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

    def combine_core_surf_sonic(self, core_surf, rs_p, ts_p, us_p, sp_row, env_row=np.zeros(0,)):
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

        for value in env_row:
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

        def combine_names(core_surf, middle, sonic, envelope):

            out_names = []

            for v_n in core_surf:
                out_names.append(v_n)
            for v_n in middle:
                out_names.append(v_n)
            for v_n in sonic:
                out_names.append(v_n)

            if len(envelope)!=0:
                for v_n in envelope:
                    out_names.append(v_n)

            return out_names

        r_u_mdot_max = np.array([0., 0., 0.])
        r_t_mdot_max = np.array([0., 0., 0.])
        out_arr = np.zeros(len(self.v_n_core_surf) + 3 + len(self.v_n_sonic) + len(self.v_n_env)) # .. + rs ts us + ...
        out_names = []

        for cl in self.smdl:

            r = cl.get_col('r')
            r_min = self.get_boundary()

            print('\t__Initical Array Length: {}'.format(len(r)))

            if self.delited_outer_points != 0:
                max_ind = len(r) - self.delited_outer_points
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

            env_vals = self.plot_envelope_r_t(self.ax, cl, self.v_n_env, 5.2)

            out_row = self.combine_core_surf_sonic(core_surf, rs_p, ts_p, us_p, sp_row, env_vals)  # combine all in out_row

            if out_row.any():  # stack out_row on top of each other, - 2d array
                out_arr = np.vstack((out_arr, out_row))

        r_u_mdot_max = np.delete(r_u_mdot_max, 0, 0)  # removing the 0th row with zeros
        r_t_mdot_max = np.delete(r_t_mdot_max, 0, 0)
        out_arr = np.delete(out_arr, 0, 0)

        out_names = combine_names(self.v_n_core_surf, ['r-sp', 't-sp', 'u-sp'], self.v_n_sonic, self.v_n_env)

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

class Wind:

    def __init__(self, wnd_cls, plot_cls, depth, v_n_arr):


        self.swnd = wnd_cls
        self.ax = plot_cls
        self.depth = depth
        self.v_n_arr = v_n_arr

        pass

    @staticmethod
    def get_coord_of_tau_ph(cl, v_n, tau_ph=2/3, method='IntUni'):

        tau_arr = cl.get_col('tau')[::-1]
        arr = cl.get_col(v_n)[::-1]

        if tau_arr[-1] == 0.0:
            tau_arr = tau_arr[:-1]
            arr = arr[:-1]

        if method == 'IntUni':
            value = interpolate.InterpolatedUnivariateSpline(tau_arr, arr)(tau_ph)
            if value == None or value == np.inf or value == -np.inf or value > arr.max() or value < arr.min():
                r = cl.get_col('r')
                plt.plot(r, arr, '.', color='black')
                plt.show()
                raise ValueError('Error in interpolation Photospheric Value for {} value is {}'.format(v_n, value))
            else:
                return value


    def get_wind_out_arr(self):
        '''
        Returns a 2d array with mdot a first row, and rest - rows for v_n,
        (if append0 or append1 != False: it will append before photosph. value, also the 0 or/and 1 value.
        '''

        out_arr = np.zeros(2 + len(self.v_n_arr))

        for cl in self.swnd:
            arr = []

            mdot = cl.get_value('mdot', 10)     # Checking Mdot
            if mdot == 0 or mdot == np.inf or mdot == -np.inf or mdot < -10**10 or mdot > 0:
                raise ValueError('Unphysical value of mdot: {}'.format(mdot))
            arr = np.append(arr, mdot)

            tau = cl.get_col('tau')             # Checking Tau
            tau_val = tau[0]
            if tau[0]==0 and tau[1]>(2/3): tau_val = tau[1]
            if tau[0] < (2/3) and tau[1] < (2/3): raise ValueError('Tau[0, and 1] < 2/3 [{} {}]'.format(tau[0], tau[1]))
            if tau[0] > 10000: raise ValueError('Suspicious value of tau[0] {}'.format(tau[0]))
            arr = np.append(arr, tau_val)  # Optical depth at the sonic point


            for v_n_cond in self.v_n_arr:   # appending all photosphereic (-ph) or index given (-0, -1, ... -n) values
                v_n = v_n_cond.split('-')[0]
                cond= v_n_cond.split('-')[-1]

                if v_n == 'mdot':
                    raise NameError('Mdot is already appended')
                if v_n == 'tau':
                    raise NameError('Tau-sp is appended by default as a first value of array')


                if cond == 'ph':
                    value = self.get_coord_of_tau_ph(cl, v_n)
                else:
                    value = cl.get_value(v_n, cond)

                if value == None: raise ValueError('Wind parameter {} does not obtained (None)'.format(v_n_cond))

                if value == -np.inf or value == np.inf:
                    plt.plot(cl.get_col('r'), cl.get_col('tau'), '.', color='red')
                    # plt.plot(cl.get_col('r'), cl.get_col(v_n), '.', color='black')

                    plt.show()
                    raise ValueError('unphysical value for {}'.format(v_n_cond))


                arr = np.append(arr, value)


            out_arr = np.vstack((out_arr, arr))

        out_arr = np.delete(out_arr, 0, 0)

        # --- HEAD of the table (array)

        head = []
        head.append('mdot-1')
        head.append('tau-sp')

        for v_n in self.v_n_arr:
            head.append(v_n)

        if len(head) != len(out_arr[0,:]):raise ValueError('out_names{} != out_array{}'.format(len(head), len(out_arr[0,:])))

        return head, out_arr



class CommonMethods:

    def __init__(self):
        pass

    @staticmethod
    def add_critical_row(mdot_cr, crit_names, crit_arr, out_names, out_arr):
        # Adds critical array to the out array ( and critical names to the out names
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
    def add_first_row(out_names, out_arr, criteria=-1):

        if criteria == -1:
            # Using the last row to get core-surf values.

            for v_n_cond in out_names:
                if v_n_cond.split('-')[-1] == '1' or v_n_cond.split('-')[-1] == '0':
                    i_coll = out_names.index(v_n_cond)
                    value = out_arr[-1, i_coll]
                    out_arr[0, i_coll] = value

            return out_names, out_arr

        else:
            raise NameError('criteria {} is not recognized'.format(criteria))

    @staticmethod
    def create_file_name(input_dirs, dirs_not_to_be_included, sp_fold_head='SP3', extension='.data'):
        '''Creates name like:
        SP3_ga_z0008_10sm_y10
        using the initial folders, where the sm.data files are located
        '''
        out_name = sp_fold_head
        for i in range(len(input_dirs)):
            if input_dirs[i] not in dirs_not_to_be_included and input_dirs[i] != '..':
                out_name = out_name + input_dirs[i]
                if i < len(input_dirs) - 1:
                    out_name = out_name + '_'
        out_name = out_name + extension
        return out_name

    @staticmethod
    def save_out_arr(out_names, out_arr, input_dir, out_dir, dirs_not_to_be_included, sp_fold_head='SP'):

        head = out_names
        table = out_arr

        fname = CommonMethods.create_file_name(input_dir, dirs_not_to_be_included, sp_fold_head, '.data')

        tmp = ''
        for i in range(len(head)):
            tmp = tmp + head[i] + '    '
        head__ = tmp

        np.savetxt(out_dir + fname, table, '%.5f', '  ', '\n', head__, '')

        print(' \n********* TABLE: {} IS SAVED IN {} *********\n'.format(fname, sp_fold_head))

    @staticmethod
    def sort_smfiles(sm_cls, v_n, where=-1):

        v_n_cls = []
        for i in range(len(sm_cls)):
            var = sm_cls[i].get_col(v_n)[where]
            v_n_cls.append([var, sm_cls[i]])

        v_n_cls.sort()

        smdl = []
        for i in range(len(sm_cls)):
            smdl.append(v_n_cls[i][1])

        return smdl

    @staticmethod
    def where(arr, value):
        for i in range(len(arr)):
            if arr[i] == value:
                return i
        raise ValueError('arr[{}] == value[{}] | not found'.format(arr, value))

    @staticmethod
    def where_mdot(arr, value, precision=.2):
        for i in range(len(arr)):
            # print('======== {} != {} ======='.format(np.round(arr[i], 2),np.round(value, 2)))
            if np.round(arr[i], 2) == np.round(value, 2):
                return i
        return None

        #     if "%{}f".format(precision) % arr[i] == "%{}f".format(precision) % value:
        #         return i
        # return None
        # # raise ValueError('arr[{}] == value[{}] | not found'.format(arr, value))

    @staticmethod
    def combine_two_tables_by_v_n(head1, table1, head2, table2, v_n):

        i_mdot1_name = CommonMethods.where(head1, v_n)
        i_mdot2_name = CommonMethods.where(head2, v_n)
        mdot1_arr = table1[:, i_mdot1_name]
        mdot2_arr = table2[:, i_mdot2_name]

        # if len(mdot1_arr) != len(mdot2_arr):
        #     raise ValueError('len(mdot1_arr)[{}] != len(mdot2_arr)[{}]'.format(len(mdot1_arr), len(mdot2_arr)))

        out_arr = []

        n_mdot_avl = 0
        for i in range(len(mdot1_arr)):

            i_mdot1 = CommonMethods.where_mdot(mdot1_arr, mdot1_arr[i])
            i_mdot2 = CommonMethods.where_mdot(mdot2_arr, mdot1_arr[i])

            if i_mdot1 == None or i_mdot2 == None:
                pass  # Mdot is not found in one of two arrays - passing by.
                # print(mdot1_arr[i])
                # break
            else:
                n_mdot_avl = n_mdot_avl + 1

                two_tables = []
                two_tables = np.hstack((two_tables, table1[i_mdot1, :]))
                two_tables = np.hstack((two_tables, table2[i_mdot2, 1:]))

                out_arr = np.append(out_arr, two_tables)

        if n_mdot_avl == 0: raise ValueError('No common Mdot between sm.data and .wind data files fround: \n'
                                             'sm.data: {} \n'
                                             '.wind:   {} '.format(mdot1_arr, mdot2_arr))

        out_array = np.reshape(out_arr, (n_mdot_avl, len(table1[0, :]) + len(table2[0, 1:])))

        head = head1
        for i in range(1, len(head2)):  # starting from 1, as the table2 also has mdot, which is not needed twice
            head.append(head2[i])

        if len(head) != len(out_array[0, :]):
            raise ValueError('Something is wrong here...')
        return head, out_array

class D1Plots:
    def __init__(self):
        pass

    @staticmethod
    def plot_xy(ax, cl, start_ind, v_n1, v_n2, x_sp=None, x_env=None, i_file=0):

        if i_file == 0:
            ax.tick_params('y', colors='b')
            ax.set_xlabel(Labels.lbls(v_n1))
            ax.set_ylabel(Labels.lbls(v_n2), color='b')
        # --- ---.tick_params('y', colors='b')
        x = cl.get_col(v_n1)[start_ind:]
        y = cl.get_col(v_n2)[start_ind:]

        if v_n2 == 'kappa':
            y = 10 ** y

        # --- ---

        ax.plot(x, y, '-', color='blue')
        ax.annotate('{}'.format(int(i_file)), xy=(x[-1], y[-1]), textcoords='data')

        if x_sp != None and x_sp != 0:
            y_sp = interpolate.interp1d(x, y, kind='linear', bounds_error=False)(x_sp)
            ax.plot(x_sp, y_sp, 'X', color='blue')

        if x_env != None and x_env != 0:
            y_env = interpolate.interp1d(x, y, kind='linear')(x_env)
            ax.plot(x_env, y_env, 'X', color='cyan')

        # ax_ = None

        if v_n2 == 'kappa':
            k_edd = 10 ** Physics.edd_opacity(cl.get_col('xm')[-1], cl.get_col('l')[-1], )
            if v_n2 == 'kappa':
                ax.plot(ax.get_xlim(), [k_edd, k_edd], '--', color='black')

        if v_n2 == 'u':
            u_s = cl.get_sonic_u()[start_ind:]
            if v_n2 == 'u':
                ax.plot(x, u_s, '--', color='gray')

        if v_n2 == 'L/Ledd':
            ax.plot(ax.get_xlim(), [1.0, 1.0], '-.', color='black')

        if v_n2 == 'Pg/P_total':
            ax.plot(ax.get_xlim(), [0.15, 0.15], '-.', color='gray')

    @staticmethod
    def plot_xyy(ax, ax_, cl, start_ind, v_n1, v_n2, v_n3, x_sp=None, x_env=None, i_file=0):

        if i_file == 0:
            ax.tick_params('y', colors='b')
            ax.set_xlabel(Labels.lbls(v_n1))
            ax.set_ylabel(Labels.lbls(v_n2), color='b')
            ax_.tick_params('y', colors='k')
            ax_.set_ylabel(Labels.lbls(v_n3), color='k')
        # --- ---.tick_params('y', colors='b')
        x = cl.get_col(v_n1)[start_ind:]
        y = cl.get_col(v_n2)[start_ind:]

        if v_n2 == 'kappa':
            y = 10 ** y

        # --- ---

        ax.plot(x, y, '-', color='blue')
        ax.annotate('{}'.format(int(i_file)), xy=(x[-1], y[-1]), textcoords='data')

        if x_sp != None and x_sp != 0:
            y_sp = interpolate.interp1d(x, y, kind='linear', bounds_error=False)(x_sp)
            ax.plot(x_sp, y_sp, 'X', color='blue')

        if x_env != None and x_env != 0:
            y_env = interpolate.interp1d(x, y, kind='linear', bounds_error=False)(x_env)
            ax.plot(x_env, y_env, 'X', color='cyan')

        y2 = cl.get_col(v_n3)[start_ind:]
        if v_n3 == 'kappa':
            y2 = 10 ** y2

        ax_.plot(x, y2, '-', color='black')
        ax_.annotate('{}'.format(int(i_file)), xy=(x[-1], y2[-1]), textcoords='data')

        if x_sp != None and x_sp != 0:
            y2_sp = interpolate.interp1d(x, y2, kind='linear', bounds_error=False)(x_sp)
            ax_.plot(x_sp, y2_sp, 'X', color='black')

        if x_env != None and x_env != 0:
            y2_env = interpolate.interp1d(x, y2, kind='linear', bounds_error=False)(x_env)
            ax_.plot(x_env, y2_env, 'X', color='orange')

        if v_n2 == 'kappa' or v_n3 == 'kappa':
            k_edd = 10 ** Physics.edd_opacity(cl.get_col('xm')[-1], cl.get_col('l')[-1], )
            if v_n2 == 'kappa':
                ax.plot(ax.get_xlim(), [k_edd, k_edd], '--', color='black')
            else:
                ax_.plot(ax.get_xlim(), [k_edd, k_edd], '--', color='black')

        if v_n2 == 'u' or v_n3 == 'u':
            u_s = cl.get_sonic_u()[start_ind:]
            if v_n2 == 'u':
                ax.plot(x, u_s, '--', color='gray')
            else:
                ax_.plot(x, u_s, '--', color='gray')

        if v_n2 == 'L/Ledd':
            ax.plot(ax.get_xlim(), [1.0, 1.0], '-.', color='black')
        if v_n3 == 'L/Ledd':
            ax_.plot(ax_.get_xlim(), [1.0, 1.0], '-.', color='black')

        if v_n2 == 'Pg/P_total':
            ax.plot(ax.get_xlim(), [0.15, 0.15], '-.', color='gray')
        if v_n3 == 'Pg/P_total':
            ax_.plot(ax_.get_xlim(), [0.15, 0.15], '-.', color='gray')

        #
        #
        #
        # # ax_ = None
        # if v_n3 != None:
        #     ax_ = ax.twinx()
        #     if set_axis:
        #         ax_.tick_params('y', colors='r')
        #         ax_.set_ylabel(Labels.lbls(v_n3), color='r')
        #
        #     y2 = cl.get_col(v_n3)[start_ind:]
        #     if v_n3 == 'kappa':
        #         y2 = 10 ** y2
        #
        #
        #     ax_.plot(x, y2, '-', color='red')
        #
        #     if x_sp != None and x_sp != 0:
        #         y2_sp = interpolate.interp1d(x, y2, kind='linear')(x_sp)
        #         ax_.plot(x_sp, y2_sp, 'X', color='red')
        #
        #     if x_env != None and x_env != 0:
        #         y2_env = interpolate.interp1d(x, y2, kind='linear')(x_env)
        #         ax_.plot(x_env, y2_env, 'X', color='orange')
        #
        # if v_n2 == 'kappa' or v_n3 == 'kappa':
        #     k_edd = 10 ** Physics.edd_opacity(cl.get_col('xm')[-1], cl.get_col('l')[-1], )
        #     if v_n2 == 'kappa':
        #         ax.plot(ax.get_xlim(), [k_edd, k_edd], '--', color='black')
        #     else:
        #         ax_.plot(ax.get_xlim(), [k_edd, k_edd], '--', color='black')
        #
        # if v_n2 == 'u' or v_n3 == 'u':
        #     u_s = cl.get_sonic_u()[start_ind:]
        #     if v_n2 == 'u':
        #         ax.plot(x, u_s, '--', color='gray')
        #     else:
        #         ax_.plot(x, u_s, '--', color='gray')
        #
        # if v_n2 == 'L/Ledd':
        #     ax.plot(ax.get_xlim(), [1.0, 1.0], '-.', color='black')
        # if v_n3 == 'L/Ledd':
        #     ax_.plot(ax_.get_xlim(), [1.0, 1.0], '-.', color='black')
        #
        # if v_n2 == 'Pg/P_total':
        #     ax.plot(ax.get_xlim(), [0.15, 0.15], '-.', color='gray')
        # if v_n3 == 'Pg/P_total':
        #     ax_.plot(ax_.get_xlim(), [0.15, 0.15], '-.', color='gray')

    @staticmethod
    def plot_tau(ax, v_n1, wndcl, smcl, x_sp=None, x_env=None, logscale=False, i_file=0):

        '''

        :param ax:
        :param v_n1:
        :param wndcl:
        :param smcl:
        :param logscale:
        :param limit:       limit=(2 / 3)
        :param inner:       inner=20
        :return:
        '''

        if i_file == 0:
            ax.set_xlabel(Labels.lbls(v_n1))
            ax.set_ylabel(Labels.lbls('tau'))

        def max_ind(cl, tau_lim=(2 / 3)):
            tau = cl.get_col('tau')
            ind = Math.find_nearest_index(tau, tau_lim)
            if ind == 0:
                raise ValueError('tau=2/3 is not found in the')
            return ind

        def min_ind(cl, u_min):
            u = cl.get_col('u')
            return Math.find_nearest_index(u, u_min)

        if logscale:
            ax.set_yscale("log", nonposy='clip')
        end = max_ind(wndcl)
        start = min_ind(smcl, 0.1)  # 0.1 - vel in km/s starting from which it plots

        tau_inner = smcl.get_col('tau')[start:]
        tau_outer = wndcl.get_col('tau')[1:end]  # use 1

        if tau_outer[0] < (2 / 3):
            raise ValueError('Tau in the begnning of the wind is < 2/3.\n'
                             'Value: tau[1] = {}\n'
                             'Mdot:{} \n use tau_outer[1]{} as tau_offset'
                             .format(tau_outer[0],smcl.get_col('mdot')[-1], tau_outer[1]))

        # use tau_outer[1] as tau_offset

        tau_offset = tau_outer[0]
        tau_inner2 = tau_inner + tau_offset

        tau_full = []
        tau_full = np.append(tau_full, tau_inner2)
        tau_full = np.append(tau_full, tau_outer)

        x_full = []
        x_full = np.append(x_full, smcl.get_col(v_n1)[start:])
        x_full = np.append(x_full, wndcl.get_col(v_n1)[1:end])  # use 1:end as t

        # ax.plot(smcl.get_col(v_n1)[start:], tau_inner2+tau_offset, '.', color='blue')
        # ax.plot(wndcl.get_col(v_n1)[1:end], tau_outer, '.', color='red')

        ax.plot(x_full, tau_full, '-', color='gray')
        ax.annotate('{}'.format(int(i_file)), xy=(x_full[-1], tau_full[-1]), textcoords='data')

        ind_atm = Math.find_nearest_index(tau_full, (2 / 3))
        ax.plot(x_full[ind_atm], tau_full[ind_atm], 'X', color='gray')

        ind20 = Math.find_nearest_index(tau_full, 20)
        ax.plot(x_full[ind20], tau_full[ind20], '*', color='black')

        if x_sp != None and x_sp != 0:
            tau_sp = interpolate.interp1d(x_full, tau_full, kind='linear')(x_sp)
            ax.plot(x_sp, tau_sp, 'X', color='blue')

        if x_env != None and x_env != 0:
            tau_env = interpolate.interp1d(x_full, tau_full, kind='linear')(x_env)
            ax.plot(x_env, tau_env, 'X', color='cyan')

    @staticmethod
    def plot_wind(ax, cl, v_n1, v_n2, v_n3=None, tau_ph=(2 / 3)):

        def max_ind(cl, tau_lim=(2 / 3)):
            tau = cl.get_col('tau')
            ind = Math.find_nearest_index(tau, tau_lim)
            if ind == 0:
                raise ValueError('tau=2/3 is not found in the')
            return ind

        ind = max_ind(cl)

        x = cl.get_col(v_n1)[:ind]
        y = cl.get_col(v_n2)[:ind]
        ax.plot(x, y, '.', color='red')
        ax.plot(x, y, '-', color='red')

        # ax.plot(x[-1], y[-1], '-', color='black')
        # ax.plot(x[-1], y[-1], 'X', color='green')

        if v_n2 == 'kappa':
            # ax.plot(x[-1], 10 ** y[-1], 'X', color='blue')
            y2 = cl.get_col('kappa_eff')[:ind]
            ax.plot(x, y2, '.', color='green', label='kappa_eff')
            ax.plot(x, y2, '-', color='green')

    @staticmethod
    def plot_tph_teff(sp_mdl, wndcls):
        teff = []
        tph = []
        mdot = []
        ts = []
        ts_eff = []
        arr = []

        for i in range(len(sp_mdl)):
            mdot = np.append(mdot, sp_mdl[i].get_col('mdot')[-1])
            ts = np.append(ts, sp_mdl[i].get_col('t')[-1])
            ts_eff = np.append(ts_eff, Physics.steph_boltz_law_t_eff(sp_mdl[i].get_col('l')[-1],
                                                                     sp_mdl[i].get_col('r')[-1]))
            tph = np.append(tph, wndcls[i].get_value('t'))
            teff = np.append(teff, Physics.steph_boltz_law_t_eff(sp_mdl[i].get_col('l')[-1],
                                                                 wndcls[i].get_value('r')))

            arr = np.append(arr, [mdot[i], ts[i], ts_eff[i], tph[i], teff[i]])

        arr_sort = np.sort(arr.view('f8, f8, f8, f8, f8'), order=['f1'], axis=0).view(np.float)
        arr_shaped = np.reshape(arr_sort, (len(sp_mdl), 5))

        teff = arr_shaped[:, 4]
        tph = arr_shaped[:, 3]
        mdot = arr_shaped[:, 0]
        ts = arr_shaped[:, 1]
        ts_eff = arr_shaped[:, 2]

        plt.plot(mdot, tph, '.', color='red', label='tph (tau=2/3)')
        plt.plot(mdot, tph, '-', color='red')
        plt.plot(mdot, teff, '.', color='blue', label='teff (BoltzLaw, tau2/3)')
        plt.plot(mdot, teff, '-', color='blue')
        plt.plot(mdot, ts, '.', color='black', label='tsonic')
        plt.plot(mdot, ts, '-', color='black')
        plt.plot(mdot, ts_eff, '.', color='green', label='teff (BoltzLaw, Sonic)')
        plt.plot(mdot, ts_eff, '-', color='green')
        plt.xlabel(Labels.lbls('mdot'))
        plt.ylabel(Labels.lbls('t'))
        plt.legend()
        plt.show()



def master_sonic(smfiles, wndfiles, out_dir, plot_dir, dirs_not_to_be_included):
    input_dirs = smfiles[0].split('/')[:-1]
    # print(self.input_dirs)
    dirs_not_to_be_included = dirs_not_to_be_included  # name of folders that not to be in the name of out. file.

    out_dir = out_dir
    plot_dir = plot_dir

    num_files = smfiles
    mdl = []
    wnd = []

    def check_files(smfiles, wndfiles):

        if len(smfiles) != len(wndfiles): raise IOError(
            'len(smfiles)[{}] != len(wndfiles)[{}] '.format(len(smfiles), len(wndfiles)))

        smfnames = []
        wnfnames = []

        for i in range(len(smfiles)):
            smfnames.append(smfiles[i].split('/')[-1].split('sm.data')[0])
            wnfnames.append(wndfiles[i].split('/')[-1].split('.wind')[0])

        for smfname in smfnames:
            if smfname not in wnfnames:
                raise NameError('There is a missing sm.data file: {}'.format(smfname))

    check_files(smfiles, wndfiles)

    for file in smfiles:
        mdl.append(Read_SM_data_file.from_sm_data_file(file))
    smdl = CommonMethods.sort_smfiles(mdl, 'mdot', -1)

    for file in wndfiles:
        wnd.append(Read_Wind_file.from_wind_dat_file(file))
    swnd = CommonMethods.sort_smfiles(wnd, 'mdot', -1)

    fig = plt.figure()
    ax = []
    ax.append(fig.add_subplot(231))
    ax.append(fig.add_subplot(232))
    ax.append(fig.add_subplot(233))
    ax.append(fig.add_subplot(234))
    ax.append(fig.add_subplot(235))
    ax.append(fig.add_subplot(236))

    v_n_sonic =     ['kappa-sp', 'L/Ledd-sp', 'HP-sp', 'mfp-sp']
    v_n_core_surf = ['l-1', 'xm-1', 'He4-0', 'He4-1', 'mdot-1']
    v_n_env =       ['t-env', 'u-env', 'r-env', 'm-env']
    v_n_wind =      ['r-ph', 't-ph']
    plot_cls = ax
    depth = 1000


    sp = SonicPointConditions(smdl, plot_cls, depth, v_n_sonic, v_n_core_surf, v_n_env)
    sp.max_ind_ = 0
    sp.if_not_found_use_last = True  # in case the sp is not found by interpolation
    sp.check_for_mult_sp = True
    sp.compute_envelope=True

    r_u_mdot_max, r_t_mdot_max, sp_out_names, sp_out_arr = sp.main_sonic_cycle()


    wnd = Wind(swnd, plot_cls, depth, v_n_wind)
    w_names, w_out_arr = wnd.get_wind_out_arr()

    names, out_arr = CommonMethods.combine_two_tables_by_v_n(sp_out_names, sp_out_arr, w_names, w_out_arr, 'mdot-1')

    # fig.clear()
    # ax.clear()
    plt.tight_layout()
    fig = plt.gcf()
    plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
    plt.show()


def master_gray(smfiles, out_dir, plot_dir, dirs_not_to_be_included):
    input_dirs = smfiles[0].split('/')[:-1]
    # print(self.input_dirs)
    dirs_not_to_be_included = dirs_not_to_be_included  # name of folders that not to be in the name of out. file.

    out_dir = out_dir
    plot_dir = plot_dir

    num_files = smfiles
    mdl = []
    smdl = []

    for file in smfiles:
        mdl.append(Read_SM_data_file.from_sm_data_file(file))

    smdl = CommonMethods.sort_smfiles(mdl, 'mdot', -1)

    fig = plt.figure()
    ax = []
    ax.append(fig.add_subplot(231))
    ax.append(fig.add_subplot(232))
    ax.append(fig.add_subplot(233))
    ax.append(fig.add_subplot(234))
    ax.append(fig.add_subplot(235))
    ax.append(fig.add_subplot(236))

    v_n_sonic = ['kappa-sp', 'L/Ledd-sp', 'HP-sp', 'mfp-sp']
    v_n_core_surf = ['l-1', 'xm-1', 'He4-0', 'He4-1', 'mdot-1']
    plot_cls = ax
    depth = 1000

    # Inheritance -----------------------
    sp = SonicPointConditions(smdl, plot_cls, depth, v_n_sonic, v_n_core_surf)
    r_u_mdot_max, r_t_mdot_max, out_names, out_arr = sp.main_sonic_cycle()

    cr = CriticalConditions(plot_cls, smdl, r_u_mdot_max, r_t_mdot_max)
    mdot_cr_u, mdot_cr_t, r_cr, t_cr = cr.set_crit_mdot_r_t()

    val_cr = cr.extrapolate_crit_value('L/Ledd-sp', mdot_cr_u, out_names, out_arr)

    out_name, out_arr = CommonMethods.add_critical_row(mdot_cr_u,
                                                       ['r-sp', 't-sp', 'L/Ledd-sp'],
                                                       [r_cr, t_cr, val_cr],
                                                       out_names, out_arr)

    out_name, out_arr = CommonMethods.add_first_row(out_name, out_arr, -1)

    CommonMethods.save_out_arr(out_name, out_arr, input_dirs, out_dir, dirs_not_to_be_included, 'SP')

    # fig.clear()
    # ax.clear()
    plt.tight_layout()
    fig = plt.gcf()
    plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
    plt.show()