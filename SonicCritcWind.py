#-----------------------------------------------------------------------------------------------------------------------
#
# These are set of algorithms and methods to analyze the sm.data and .wind files to extract sonic point
# conditions, critical conditions adn photospheric conditions respectively.
#
#-----------------------------------------------------------------------------------------------------------------------

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

from PhysMath import Math
from PhysMath import Physics
from PhysMath import Constants
from PhysMath import Labels

from FilesWork import Read_SM_data_file
from FilesWork import Read_Wind_file

class SonicPointAlgorithm:
    '''
    This class takes set of sm_cls (from reading the sm.data diles), interpolates the sonic point values and the max.
    values of vel. and temp.

        Finds maxiumums of every profile.
        Finds points, where the profile crosses the sonic profiles (us or ts)
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
    def __init__(self, sm_cls, v_n_sonic, v_n_core_surf, v_n_env=list()):

        self.smdl = sm_cls
        self.v_n_sonic = v_n_sonic
        self.v_n_core_surf = v_n_core_surf
        self.v_n_env = v_n_env


        if not 'mdot-1' in v_n_core_surf:
            raise NameError('v_n = *mdot-1* is not in v_n_core_surf. Give: {}'.format(v_n_core_surf))

        # --- Adjustable paramters ---
        self.set_delited_outer_points = -10 # for cutting the end of the vel. prof.
        self.set_if_not_found_use_last = False
        self.set_check_for_mult_sp = False
        self.set_compute_envelope=False
        self.set_u_min = 0.1    #
        self.set_depth=1000
        self.set_interpol_method_vel_profs='IntUni' # 'IntUni' 'Uni' '1dCubic' '1dLinear'
        self.set_interpol_method_sonic_values='IntUni'
        self.set_interpol_method_envelop_vals = 'IntUni'

        # --- OUTPUT --- PUBLC vars
        self.r_u_mdot_max = []
        self.r_t_mdot_max = []

        self.out_names = []
        self.out_arr = []

        pass


    def get_inner_boundary(self):
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
                if u[i] > self.set_u_min:
                    # ax1.axvline(x=r[i], color='red')
                    bourders = np.append(bourders, r[i])
                    break

        return bourders.min()

    def interpol_vel_prof(self, cl, min_ind, max_ind):

        r = cl.get_col('r')
        u = cl.get_col('u')
        u_s = cl.get_sonic_u()

        r = r[min_ind:max_ind]
        u = u[min_ind:max_ind]

        int_r = np.mgrid[r[0]:r[-1]:self.set_depth * 1j]
        int_u = Math.interpolate_arr(r, u, int_r, self.set_interpol_method_vel_profs)

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

        int_r = np.mgrid[r[0]:r[-1]:self.set_depth * 1j]
        int_t = Math.interpolate_arr(r, t, int_r, self.set_interpol_method_vel_profs)
        int_ts_arr = Math.interpolate_arr(r, ts_arr, int_r, self.set_interpol_method_vel_profs)

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


        return r_u_mdot_max, r_t_mdot_max

    def plot_rs_ts(self, cl, int_r, int_t, int_ts_arr):

        rs_p, ts_p, us_p = None, None, None  # initialize the variables
        rs_p, ts_p = Math.interpolated_intercept(int_r, int_ts_arr, int_t)  # SONIC TEPERATURE

        us = cl.get_sonic_u()
        r = cl.get_col('r')
        t = cl.get_col('t')

        if len(rs_p) > 1 and self.set_check_for_mult_sp:
            plt.plot(cl.get_col('r'), cl.get_col('u'), '.', color='black')
            plt.plot(cl.get_col('r'), cl.get_col('u'), '-', color='gray')
            plt.plot(cl.get_col('r'), cl.get_sonic_u(), '-', color='blue')
            plt.show()
            raise ValueError('Multiple Sonic Point Found. (if it is OK, turn of the *check_for_mult_sp*')

        if rs_p.any():              # if the sonic point has to be found by interpolating
            rs_p = rs_p[0][0]
            ts_p = ts_p[0][0]

        else:
            if self.set_if_not_found_use_last: # If the sonic point is supposed to be the last point
                rs_p = r[-1]
                ts_p = t[-1]

        if rs_p != None:
            us_p = Math.interpolate_arr(r, us, np.array([rs_p]), self.set_interpol_method_sonic_values)[0]

        return rs_p, ts_p, us_p
        # out_array = np.vstack((out_array, row))

    def plot_envelope_r_t(self, cls, v_n_env,  guess=5.2):
        '''
        Looks for a loal extremum between t_lim1 and t_lim2, and, if the extremem != sonic point: returns
        length and mass of whatever is left
        :param cls:
        :param t_lim1:
        :param t_lim2:
        :return: tp, up, rp, xmp (coordinates, where envelope begins)
        '''

        def get_envelope_r_or_m(v_n, cls, t_start, method):
            t = cls.get_col('t')
            ind = Math.find_nearest_index(t, t_start) - 5  # just before the limit, so we don't need to

            # interpolate across the whole t range
            t = t[ind:]
            var = cls.get_col(v_n)
            var = var[ind:]

            value = Math.interpolate_arr(t[::-1], var[::-1], t_start, method)

            return value

        for v_n in v_n_env:
            if v_n not in ['t-env', 'u-env', 'r-env', 'm-env']:
                raise NameError('Var: {} not in the list. Change the *plot_envelope_r_t* to account for that')

        t = cls.get_col('t')  # x - axis
        u = cls.get_col('u')  # y - axis

        tp, up = Math.get_max_by_interpolating(t, u, True, guess)  # WHERE the envelope starts (if any)
        # TRUE for cutting out the rising part in the end
        if Math.find_nearest_index(t, tp) < len(t) - 1:  # if the tp is not the last point of the t array

            print('<<<<<<<<<<<Coord: {} {} >>>>>>>>>>>>>>>>'.format("%.2f" % tp, "%.2f" % up))

            r_env = get_envelope_r_or_m('r', cls, tp, self.set_interpol_method_envelop_vals)
            m_env = get_envelope_r_or_m('xm', cls, tp, self.set_interpol_method_envelop_vals)
            m = cls.get_col('xm')[-1]
            r = cls.get_col('r')[-1]

            return np.array([tp, up, r-r_env, np.log10(m-m_env)])
        else:
            return np.array([0., 0., 0, 0])

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
            '''Takes lists of v_n (names)'''

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

        for cl in self.smdl:

            r = cl.get_col('r')
            r_min = self.get_inner_boundary()

            print('\t__Initical Array Length: {}'.format(len(r)))

            if self.set_delited_outer_points != 0:
                max_ind = len(r) - self.set_delited_outer_points
            else:
                max_ind = len(r)  # as for sonic-BEC this is not necessary.

            min_ind = Math.find_nearest_index(r, r_min)
            if min_ind > max_ind: raise ValueError('Index Problem: min_ind({}) > max_ind({})'.format(min_ind, max_ind))

            r_int, u_int = self.interpol_vel_prof(cl, min_ind, max_ind)
            t_int, ts_int = self.plot_interpol_t_prof(cl, min_ind, max_ind)

            r_u_mdot_max_, r_t_mdot_max_ = self.plot_max_vel_temp(cl, r_int, u_int, ts_int)
            r_u_mdot_max = np.vstack((r_u_mdot_max, r_u_mdot_max_))
            r_t_mdot_max = np.vstack((r_t_mdot_max, r_t_mdot_max_))

            rs_p, ts_p, us_p = self.plot_rs_ts(cl, r_int, t_int, ts_int)  # get rs ts us

            core_surf = self.out_row_core_surf_vals(cl)  # get arr of core_surf values

            sp_row = self.append_sp_vals(cl, rs_p, min_ind)  # get arr or sp values (use rs and interpolating)

            env_vals = self.plot_envelope_r_t(cl, self.v_n_env, 5.2)

            out_row = self.combine_core_surf_sonic(core_surf, rs_p, ts_p, us_p, sp_row, env_vals)  # combine all in out_row

            if out_row.any():  # stack out_row on top of each other, - 2d array
                out_arr = np.vstack((out_arr, out_row))

        r_u_mdot_max = np.delete(r_u_mdot_max, 0, 0)  # removing the 0th row with zeros
        r_t_mdot_max = np.delete(r_t_mdot_max, 0, 0)
        out_arr = np.delete(out_arr, 0, 0)

        out_names = combine_names(self.v_n_core_surf, ['r-sp', 't-sp', 'u-sp'], self.v_n_sonic, self.v_n_env)

        self.r_t_mdot_max = r_t_mdot_max
        self.r_t_mdot_max = r_t_mdot_max
        self.out_names = out_names
        self.out_arr = out_arr


        return r_u_mdot_max, r_t_mdot_max, out_names, out_arr

class PlotProfiles:
    def __init__(self, sm_cls, wnd_cls, n_plots, out_names=list(), out_array=np.empty(0,)):


        # v_n_x_y1, v_n_x_y2, v_n_x_y3, v_n_x_y4, v_n_x_y5, v_n_x_y6
        self.n_plots = n_plots
        self.v_n_all = [['r', 'u'], ['r', 't'], ['r', 'kappa']]

        self.ax = []

        if len(self.v_n_all[:]) != n_plots:
            raise ValueError('Number of plots and size of variable array should match exactly. {} != {}'
                             .format(len(self.v_n_all[:]), len(n_plots)))

        fig = plt.figure()

        pass

    def set_plot_ax(self, fig):

        if self.n_plots == 0:
            raise ValueError('0 plots cannot be set')

        if self.n_plots == 1:
            self.ax.append(fig.add_subplot(111))
        if self.n_plots == 2:
            self.ax.append(fig.add_subplot(121))
            self.ax.append(fig.add_subplot(122))
        if self.n_plots == 3:
            self.ax.append(fig.add_subplot(131))
            self.ax.append(fig.add_subplot(132))
            self.ax.append(fig.add_subplot(133))
        if self.n_plots == 4:
            self.ax.append(fig.add_subplot(221))
            self.ax.append(fig.add_subplot(222))
            self.ax.append(fig.add_subplot(223))
            self.ax.append(fig.add_subplot(224))
        if self.n_plots == 5:
            self.ax.append(fig.add_subplot(231))
            self.ax.append(fig.add_subplot(232))
            self.ax.append(fig.add_subplot(233))
            self.ax.append(fig.add_subplot(234))
            self.ax.append(fig.add_subplot(235))
        if self.n_plots == 6:
            self.ax.append(fig.add_subplot(231))
            self.ax.append(fig.add_subplot(232))
            self.ax.append(fig.add_subplot(233))
            self.ax.append(fig.add_subplot(234))
            self.ax.append(fig.add_subplot(235))
            self.ax.append(fig.add_subplot(236))

        if self.n_plots > 6:
            raise ValueError('Only 1-6 plots are supported')

    # def set_labels_for_all(self):
    #     for i in range(len(self.n_plots)):
    #         self.ax[i].


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

def test(smfiles, wndfiles):

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

    PlotProfiles(smdl, swnd, 3)

from os import listdir
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

output_dir  = '../data/output/'
plot_dir    = '../data/plots/'
sse_locaton = '/media/vnedora/HDD/sse/'

def gray_analysis3(z, m_set, y_set, plot):

    from Sonic_Criticals import Criticals3 # CRITICALS2 also computes sonic-BEC sm and plot files to get tau.
    from Temp import master_sonic
    for m in m_set:
        for y in y_set:
            root_name = 'ga_z' + z + '/'
            folder_name = str(m)+'sm/y'+str(y)+'/'
            out_name = str(m) + 'z' + z + '/'

            print('COMPUTING: ({}) {} , to be saved in {}'.format(root_name, folder_name, out_name))

            # smfiles_ga = get_files(sse_locaton + root_name, [folder_name], [], 'sm.data')
            smfiles_sp  = get_files(sse_locaton + root_name, [folder_name+'sp/'], [], 'sm.data')
            wind_fls    = get_files(sse_locaton + root_name, [folder_name+'sp/'], [], 'wind')

            cr = test(smfiles_sp, wind_fls)

            # cr.combine_save(1000, ['kappa-sp', 'L/Ledd-sp', 'HP-sp', 'mfp-sp', 'tpar-'], plot, wind)

            print('m:{}, y:{} DONE'.format(m,y))


gray_analysis3('002', [10], [10], True)

