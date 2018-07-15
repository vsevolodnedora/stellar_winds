#-----------------------------------------------------------------------------------------------------------------------
#
# These are set of algorithms and methods to analyze the sm.data and .wind files to extract sonic point
# conditions, critical conditions adn photospheric conditions respectively and to extrpolate other critical
# values based of the set of sonic values.
#
#-----------------------------------------------------------------------------------------------------------------------

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

from PhysMath import Math, Physics, Constants

from FilesWork import Read_SM_data_file, Read_Wind_file, Labels



class PlotProfile:
    def __init__(self, low_bound_cond, upper_wind_bound):

        self.low_bound_cond = low_bound_cond # 'u=0.1'
        self.upper_wind_bound = upper_wind_bound # 'tau=2/3'

        pass

    def get_low_bound_index(self, cl):

        v_n = self.low_bound_cond.split('=')[0]
        val = np.float(self.low_bound_cond.split('=')[-1])

        return Math.find_nearest_index(cl.get_col(v_n), val)

    def get_upper_wind_bound_index(self, wcl):

        v_n = self.upper_wind_bound.split('=')[0]
        val =self.upper_wind_bound.split('=')[-1]

        if '%' in val:
            percent_val = np.float(val.split('%')[0])
            max_wind = wcl.get_col(v_n).max()
            val=max_wind*percent_val/100

        if v_n=='tau' and val=='2/3':
            val = 2/3

        return Math.find_nearest_index(wcl.get_col(v_n), np.float(val))

    def plot_smcl_xy(self, ax, cl, v_n1, v_n2, label='mdot', ls='-', color='blue'):

        # if i_file == 0:
        #     ax.tick_params('y', colors='b')
        #     ax.set_xlabel(Labels.lbls(v_n1))
        #     ax.set_ylabel(Labels.lbls(v_n2), color='b')
        # --- ---.tick_params('y', colors='b')

        start_ind = self.get_low_bound_index(cl)

        x = cl.get_col(v_n1)[start_ind:]
        y = cl.get_col(v_n2)[start_ind:]

        if v_n2 == 'kappa':
            y = 10 ** y

        # --- ---

        ax.plot(x, y, ls, color=color)
        if label!=None:
            lbl = cl.get_col(label)[-1]
            ax.annotate('{}'.format("%.2f"%lbl), xy=(x[-1], y[-1]), textcoords='data')



        # ax.annotate('{}'.format(int(i_file)), xy=(x[-1], y[-1]), textcoords='data')

        # if x_sp != None and x_sp != 0:
        #     y_sp = interpolate.interp1d(x, y, kind='linear', bounds_error=False)(x_sp)
        #     ax.plot(x_sp, y_sp, 'X', color='blue')
        #
        # if x_env != None and x_env != 0:
        #     y_env = interpolate.interp1d(x, y, kind='linear')(x_env)
        #     ax.plot(x_env, y_env, 'X', color='cyan')

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

    def plot_xyy(self, ax, ax_, cl, v_n1, v_n2, v_n3, x_sp=None, x_env=None, i_file=0):

        # if i_file == 0:
        #     ax.tick_params('y', colors='b')
        #     ax.set_xlabel(Labels.lbls(v_n1))
        #     ax.set_ylabel(Labels.lbls(v_n2), color='b')
        #     ax_.tick_params('y', colors='k')
        #     ax_.set_ylabel(Labels.lbls(v_n3), color='k')
        # --- ---.tick_params('y', colors='b')
        start_ind = self.get_low_bound_index(cl)

        x = cl.get_col(v_n1)[start_ind:]
        y = cl.get_col(v_n2)[start_ind:]

        if v_n2 == 'kappa':
            y = 10 ** y

        # --- ---

        ax.plot(x, y, '-', color='blue')
        # ax.annotate('{}'.format(int(i_file)), xy=(x[-1], y[-1]), textcoords='data')

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
        # ax_.annotate('{}'.format(int(i_file)), xy=(x[-1], y2[-1]), textcoords='data')

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

    def plot_smcl_wndcl_tau(self, ax, v_n1, wndcl, smcl, x_sp=None, x_env=None, logscale=False, i_file=0):

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

        # if i_file == 0:
        #     ax.set_xlabel(Labels.lbls(v_n1))
        #     ax.set_ylabel(Labels.lbls('tau'))

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

        end = self.get_upper_wind_bound_index(wndcl) # max_ind(wndcl)
        start = self.get_low_bound_index(smcl) # min_ind(smcl, 0.1)  # 0.1 - vel in km/s starting from which it plots

        tau_inner = smcl.get_col('tau')[start:]
        tau_outer = wndcl.get_col('tau')[:end]  # use 1

        # if tau_outer[0] < (2 / 3):
        #     raise ValueError('Tau in the begnning of the wind is < 2/3.\n'
        #                      'Value: tau[1] = {}\n'
        #                      'Mdot:{} \n use tau_outer[1]{} as tau_offset'
        #                      .format(tau_outer[0],smcl.get_col('mdot')[-1], tau_outer[1]))

        # use tau_outer[1] as tau_offset

        if len(tau_outer) > 0: tau_offset = tau_outer[0]
        else: tau_offset = 0.

        tau_inner2 = tau_inner + tau_offset

        tau_full = []
        tau_full = np.append(tau_full, tau_inner2)
        tau_full = np.append(tau_full, tau_outer)

        x_full = []
        x_full = np.append(x_full, smcl.get_col(v_n1)[start:])
        x_full = np.append(x_full, wndcl.get_col(v_n1)[:end])  # use 1:end as t

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

    def plot_wind(self, ax, wcl, v_n1, v_n2, label='mdot', ls='-', color='red', rotation=None):



        def max_ind(cl, tau_lim=(2 / 3)):
            tau = cl.get_col('tau')
            ind = Math.find_nearest_index(tau, tau_lim)
            if ind == 0:
                raise ValueError('tau=2/3 is not found in the')
            return ind

        ind = self.get_upper_wind_bound_index(wcl) # max_ind(wcl)

        x = wcl.get_col(v_n1)[:ind]
        y = wcl.get_col(v_n2)[:ind]

        if label!=None and x.any():
            lbl = wcl.get_col(label)[-1]
            lab = ax.annotate('{}'.format("%.2f"%lbl), xy=(x[-1], y[-1]), textcoords='data', horizontalalignment='center',
                    verticalalignment='up')

            if rotation!=None: lab.set_rotation(rotation)


        ax.plot(x, y, ls, color=color)

        # ax.plot(x, y, '-', color='red')

        # ax.plot(x[-1], y[-1], '-', color='black')
        # ax.plot(x[-1], y[-1], 'X', color='gray')

        if v_n2 == 'kappa':
            # ax.plot(x[-1], 10 ** y[-1], 'X', color='blue')
            y2 = wcl.get_col('kappa_eff')[:ind]
            ax.plot(x, y2, '.', color='green', label='kappa_eff')
            ax.plot(x, y2, ls, color='green')

    @staticmethod
    def self_plot_tph_teff(sp_mdl, wndcls):
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

    @staticmethod
    def plot_points(ax, x, y, v_n_cond_x, v_n_cond_y):
        v_n_x = v_n_cond_x.split('-')[0]
        v_n_y = v_n_cond_y.split('-')[0]
        cond_x = v_n_cond_x.split('-')[-1]
        cond_y = v_n_cond_y.split('-')[-1]

        if cond_x != cond_y: raise NameError('Conditions for x and y should be the same, given: {} {}'
                                             .format(v_n_cond_x, v_n_cond_y))

        if cond_x == 'sp':
            ax.plot(x, y, 'x', color='red')
        if cond_x == 'env':
            ax.plot(x, y, 'x', color='black')

class PlotProfiles(PlotProfile):
    def __init__(self, sm_cls, wnd_cls, out_names=list(), out_array=np.empty(0,)):


        # v_n_x_y1, v_n_x_y2, v_n_x_y3, v_n_x_y4, v_n_x_y5, v_n_x_y6
        # self.n_plots = n_plots
        self.set_v_ns = []      # [[v_nx, v_ny], [v_nx, v_ny], [v_nx, v_ny], [v_nx, v_ny]]


        self.low_bound='u=0.1'
        self.upper_wind_bound='tau=2/3' #'r=10%'

        self.sm_cls = sm_cls
        self.wnd_cls = wnd_cls

        self.mask=0.0

        self.sp_arr = out_array
        self.sp_names = out_names

        if len(out_names)>0 and out_array.any():
            if len(self.sp_names) != len(self.sp_arr[0, :]):
                raise ValueError('Given out_names{} != out_arr{}'.format(len(self.sp_names), len(self.sp_arr[0, :])))


        self.ax = []

        # if len(self.set_v_ns[:]) != n_plots:
        #     raise ValueError('Number of plots and size of variable array should match exactly. {} != {}'
        #                      .format(len(self.set_v_ns[:]), n_plots))

        PlotProfile.__init__(self, self.low_bound, self.upper_wind_bound)

        # fig = plt.figure()
        # self.set_plot_ax(fig)
        # self.set_plot_ax(fig)
        # self.set_labels_for_all()
        # self.plot_sm_all()
        # self.plot_wind_all()
        # self.plot_out_arr_points()
        # plt.legend()
        # plt.show()
        pass

    def set_plot_ax(self, fig):

        self.n_plots = len(self.set_v_ns[:])

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
        if self.n_plots == 7:
            self.ax.append(fig.add_subplot(331))
            self.ax.append(fig.add_subplot(332))
            self.ax.append(fig.add_subplot(333))
            self.ax.append(fig.add_subplot(334))
            self.ax.append(fig.add_subplot(335))
            self.ax.append(fig.add_subplot(336))
            self.ax.append(fig.add_subplot(337))
        if self.n_plots == 8:
            self.ax.append(fig.add_subplot(331))
            self.ax.append(fig.add_subplot(332))
            self.ax.append(fig.add_subplot(333))
            self.ax.append(fig.add_subplot(334))
            self.ax.append(fig.add_subplot(335))
            self.ax.append(fig.add_subplot(336))
            self.ax.append(fig.add_subplot(337))
            self.ax.append(fig.add_subplot(338))
        if self.n_plots == 9:
            self.ax.append(fig.add_subplot(331))
            self.ax.append(fig.add_subplot(332))
            self.ax.append(fig.add_subplot(333))
            self.ax.append(fig.add_subplot(334))
            self.ax.append(fig.add_subplot(335))
            self.ax.append(fig.add_subplot(336))
            self.ax.append(fig.add_subplot(337))
            self.ax.append(fig.add_subplot(338))
            self.ax.append(fig.add_subplot(339))

        if self.n_plots > 9:
            raise ValueError('Only 1-6 plots are supported, Given {}'.format(self.n_plots))

    def set_labels_for_all(self, fsz=12):

        for i in range(len(self.set_v_ns)):
            if len(self.set_v_ns[i]) != 2:
                raise NameError('v_n_all should be [[a, b], [c, d]], given: {}'.format(self.set_v_ns))

            v_n_x = self.set_v_ns[i][0]
            v_n_y = self.set_v_ns[i][1]

            self.ax[i].minorticks_on()
            # self.ax[i].set_yticks(np.array([0, 0.01, 0.02]))

            self.ax[i].tick_params('y', labelsize=fsz)
            self.ax[i].tick_params('x', labelsize=fsz)

            if len(v_n_x.split('-'))>0:
                self.ax[i].set_xlabel(Labels.lbls(v_n_x.split('-')[0]), fontsize=fsz)
                self.ax[i].set_ylabel(Labels.lbls(v_n_y.split('-')[0]), fontsize=fsz)


            else:

                self.ax[i].set_xlabel(Labels.lbls(v_n_x))
                self.ax[i].set_ylabel(Labels.lbls(v_n_y))

    def plot_additional(self, v_n_x, v_n_y, x_arr=np.empty(0,), y_arr=np.empty(0,), how='-', color='black'):

        for i in range(len(self.set_v_ns)):
            dum = self.set_v_ns[i]
            if len(dum) != 2:
                raise NameError('v_n_all should be [[a, b], [c, d]], given: {}'.format(self.set_v_ns))

            v_n_x_ = self.set_v_ns[i][0]
            v_n_y_ = self.set_v_ns[i][1]

            if v_n_x == v_n_x_ and v_n_y == v_n_y_:
                if v_n_x == 'kappa' or v_n_x == 'kappa-sp': x_arr = 10 ** x_arr
                if v_n_y == 'kappa' or v_n_y == 'kappa-sp': y_arr = 10 ** y_arr

                self.ax[i].plot(x_arr, y_arr, how, color=color)

    def plot_sm_all(self, label='mdot', ls='-', color='blue'):

        for i in range(len(self.set_v_ns)):
            if len(self.set_v_ns[i]) != 2:
                raise NameError('v_n_all should be [[a, b], [c, d]], given: {}'.format(self.set_v_ns))

            v_n_x = self.set_v_ns[i][0]
            v_n_y = self.set_v_ns[i][1]


            if v_n_x in self.sm_cls[0].var_names and v_n_y in self.sm_cls[0].var_names:
                for j in range(len(self.sm_cls)):
                    if v_n_x == 'tau' or v_n_y == 'tau' and len(self.wnd_cls) > 0:
                        self.plot_smcl_wndcl_tau(self.ax[i], v_n_x, self.wnd_cls[j], self.sm_cls[j], None, None, True)
                    else:
                        self.plot_smcl_xy(self.ax[i], self.sm_cls[j], v_n_x, v_n_y, label, ls, color)

    def plot_out_arr_points(self):

        if self.sp_arr.any():

            for i in range(self.n_plots):
                if len(self.set_v_ns[i]) != 2:
                    raise NameError('v_n_all should be [[a, b], [c, d]], given: {}'.format(self.set_v_ns))

                v_n_x = self.set_v_ns[i][0]
                v_n_y = self.set_v_ns[i][1]

                out_v_n = []
                out_cond = []

                for v_n_cond in self.sp_names:
                    out_v_n.append(v_n_cond.split('-')[0])
                    out_cond.append(v_n_cond.split('-')[-1])

                for cond in out_cond:

                    i_x = None
                    if v_n_x + '-' + cond in self.sp_names:
                        i_x = self.sp_names.index(v_n_x + '-' + cond)

                    i_y = None
                    if v_n_y + '-' + cond in self.sp_names:
                        i_y = self.sp_names.index(v_n_y + '-' + cond)

                    if i_x != None and i_y != None:
                        x_col = self.sp_arr[:, i_x]
                        y_col = self.sp_arr[:, i_y]

                        for l in range(len(x_col)):

                            x = x_col[l]
                            y = y_col[l]

                            if x!=self.mask and y!=self.mask:

                                if v_n_x == 'kappa': x = 10 ** x
                                if v_n_y == 'kappa': y = 10 ** y

                                self.plot_points(self.ax[i], x, y, self.sp_names[i_x], self.sp_names[i_y])

                                # self.ax[i].plot(x, y, 'x', color='blue')

    def plot_wind_all(self, label='mdot', ls='-', color='red'):

        for i in range(len(self.set_v_ns)):
            if len(self.set_v_ns[i]) != 2:
                raise NameError('v_n_all should be [[a, b], [c, d]], given: {}'.format(self.set_v_ns))

            v_n_x = self.set_v_ns[i][0]
            v_n_y = self.set_v_ns[i][1]

            if v_n_x in self.wnd_cls[0].var_names and v_n_y in self.wnd_cls[0].var_names:

                for j in range(len(self.wnd_cls)):

                    self.plot_wind(self.ax[i],self.wnd_cls[j],v_n_x,v_n_y,label, ls, color)

    def plot_r_ut_mdot_max(self, r_tu_mdot_max, t_or_u):

        if len(r_tu_mdot_max) == 0: raise ValueError('Empty r_tu_mdot_max for {}'.format(t_or_u))

        for i in range(self.n_plots):
            if len(self.set_v_ns[i]) != 2:
                raise NameError('v_n_all should be [[a, b], [c, d]], given: {}'.format(self.set_v_ns))

            v_n_x = self.set_v_ns[i][0]
            v_n_y = self.set_v_ns[i][1]

            if v_n_x in ['r', 't', 'u'] and v_n_y in ['r', 't', 'u']:
                if v_n_x == 'r':
                    if v_n_y == 't' and t_or_u == 't':
                        self.ax[i].plot(r_tu_mdot_max[:, 0], r_tu_mdot_max[:, 1], 'x', color='blue')
                        self.ax[i].plot(r_tu_mdot_max[:, 0], r_tu_mdot_max[:, 1], '-', color='gray')
                    if v_n_y == 'u' and t_or_u == 'u':
                        self.ax[i].plot(r_tu_mdot_max[:, 0], r_tu_mdot_max[:, 1], 'x', color='blue')
                        self.ax[i].plot(r_tu_mdot_max[:, 0], r_tu_mdot_max[:, 1], '-', color='gray')
                if v_n_x == 't' and t_or_u == 't':
                    if v_n_y == 'r':
                        self.ax[i].plot(r_tu_mdot_max[:, 1], r_tu_mdot_max[:, 0], 'x', color='blue')
                        self.ax[i].plot(r_tu_mdot_max[:, 1], r_tu_mdot_max[:, 0], '-', color='gray')
                if v_n_x == 'u' and t_or_u == 'u':
                    if v_n_y == 'r':
                        self.ax[i].plot(r_tu_mdot_max[:, 1], r_tu_mdot_max[:, 0], 'x', color='blue')
                        self.ax[i].plot(r_tu_mdot_max[:, 1], r_tu_mdot_max[:, 0], '-', color='gray')

    def plot_vertial(self, v_n_x_, v_n_y_, x_value, ls ='dashed', color='black'):

        for i in range(len(self.set_v_ns)):
            if len(self.set_v_ns[i]) != 2:
                raise NameError('v_n_all should be [[a, b], [c, d]], given: {}'.format(self.set_v_ns))

            v_n_x = self.set_v_ns[i][0]
            v_n_y = self.set_v_ns[i][1]

            if v_n_x == v_n_x_ and v_n_y_ == v_n_y:
                self.ax[i].axvline(x=x_value, ls=ls, color=color)

    def plot_horisontal(self, v_n_x_, v_n_y_, y_value, ls ='dashed', color='black'):

        for i in range(len(self.set_v_ns)):
            if len(self.set_v_ns[i]) != 2:
                raise NameError('v_n_all should be [[a, b], [c, d]], given: {}'.format(self.set_v_ns))

            v_n_x = self.set_v_ns[i][0]
            v_n_y = self.set_v_ns[i][1]

            if v_n_y == v_n_y_ and v_n_x == v_n_x_:
                self.ax[i].axhline(y=y_value, ls=ls, color=color)

    def get_ax(self, v_n_x_, v_n_y_):

        for i in range(self.n_plots):
            if len(self.set_v_ns[i]) != 2:
                raise NameError('v_n_all should be [[a, b], [c, d]], given: {}'.format(self.set_v_ns))

            v_n_x = self.set_v_ns[i][0]
            v_n_y = self.set_v_ns[i][1]

            if v_n_y == v_n_y_ and v_n_x == v_n_x_:
                return self.ax[i]

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
    def __init__(self, sm_cls):

        self.smdl = sm_cls
        self.v_n_sonic = []
        self.v_n_core_surf = []
        self.v_n_env = []


        # --- Adjustable paramters ---
        self.ts_lim = 5.17 # in case of finding a ts in the low temp. regime

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
            raise ValueError('Multiple Sonic Point Found. (if it is OK, turn of the *check_for_mult_sp* Mdot: {}'
                             .format(cl.get_col('mdot')[-1]))

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

            # return np.array([tp, up, r-r_env, np.log10(m-m_env)])
            return np.array([tp, up, r_env, np.log10(m_env)])
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

        if not 'mdot-1' in self.v_n_core_surf:
            raise NameError('v_n = *mdot-1* is not in v_n_core_surf. Give: {}'.format(self.v_n_core_surf))

        if len(self.v_n_env)==0 and self.set_compute_envelope==True: raise NameError('What to compute for envelope?')

        if len(self.v_n_sonic)==0: raise NameError('Nothing for Sonic Point to compute')

        if len(self.v_n_core_surf)==0: raise NameError('Nothing for core-surf to compute')

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

        env_vals = []

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

            if len(self.v_n_env)>0: env_vals = self.plot_envelope_r_t(cl, self.v_n_env, 5.2)

            out_row = self.combine_core_surf_sonic(core_surf, rs_p, ts_p, us_p, sp_row, env_vals)  # combine all in out_row

            if out_row.any() and ts_p > self.ts_lim:  # stack out_row on top of each other, - 2d array
                # if len(out_arr[])!=len(out_row): raise ValueError('out_arr{} != out_row{}'.format(len(out_arr), len(out_row)))
                out_arr = np.vstack((out_arr, out_row))

        r_u_mdot_max = np.delete(r_u_mdot_max, 0, 0)  # removing the 0th row with zeros
        r_t_mdot_max = np.delete(r_t_mdot_max, 0, 0)
        out_arr = np.delete(out_arr, 0, 0)

        out_names = combine_names(self.v_n_core_surf, ['r-sp', 't-sp', 'u-sp'], self.v_n_sonic, self.v_n_env)

        self.r_u_mdot_max = r_u_mdot_max
        self.r_t_mdot_max = r_t_mdot_max
        self.out_names = out_names
        self.out_arr = np.array(out_arr)


        return r_u_mdot_max, r_t_mdot_max, out_names, out_arr

class CriticalMdotRT(PlotProfiles):

    '''
    This class contains set of codependant methods to, using the provided
            [r_u_mdot_max]
            [r_t_mdot_max]
        arrays to interplate the intersections between these [r_u_max] and [r_t_max] arraus and
        sonic u_s and t_s profiles. This for every profile there is One intersection.
        Thus for every profile there is a perpendicular between u_max (or t_max) and u_s (t_s) profile,
        for which the length can be obtained.

        For every profile now there is a Delta=u_max - u_s. From the plor Delta = f(mdot) it
        interpolates the mdot value at which the Delta = 0. This is the critical mdot.

            Using the critical mdot and a set of [r_u_mdot], the r critical is inteprolated
            Using the critical mdot and a set of [t_t_mdot], the t critical is interpolated

        reuslt is : mdot_cr, r_cr, t_cr values for this set of sm.data files.
    '''

    def __init__(self, sm_cls, r_u_mdot_max, r_t_mdot_max):

        PlotProfiles.__init__(self, sm_cls, [])

        self.set_v_ns    = [['r', 'u'], ['r', 't']]
        # self.set_v_ns_cr = ['mdot', 'r', 't']
        self.set_do_plot_tech=True
        # self.set_extrap_add_crit_v_n_from_sonics=False



        self.smdl = sm_cls

        self.r_u_mdot_max = r_u_mdot_max
        self.r_t_mdot_max = r_t_mdot_max

        self.set_crit_mdot_interp_method    = 'IntUni'
        self.set_crit_r_t_interp_method     = 'IntUni'
        self.set_interp_cross_ut_usts_method= 'IntUni'
        # self.set_crit_extrapol_method       = 'IntUni'  # ALSO 'test' is avaliable to see the best

        self.u_min = 0.1
        self.depth = 1000

        pass

    def set_initialize_plots(self):
        if len(self.set_v_ns) > 0 and self.set_do_plot_tech:
            fig = plt.figure()
            self.set_plot_ax(fig)
            self.set_labels_for_all()
            self.plot_additional('r', 'u', self.r_u_mdot_max[:, 0], self.r_u_mdot_max[:, 1], '-', 'cyan')
            self.plot_additional('r', 't', self.r_t_mdot_max[:, 0], self.r_t_mdot_max[:, 1], '-', 'cyan')

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
        if u_or_t1 == 'u' and u_or_t1.max()*(2/3) > \
                u_or_ts2[Math.find_nearest_index(r2, r2[Math.find_nearest_index(u_or_ts2, u_or_t1.max()*(2/3))])]:
            i_u2_where_u1_max = Math.find_nearest_index(u_or_ts2, u_or_t1.max() * (2 / 3))
            # needed to avoide violent behaviour in high mdot
        else:
            i_u2_where_u1_max = Math.find_nearest_index(u_or_ts2, u_or_t1.max())
                                                                        # needed to avoide violent behaviour in high mdot

        u2_crop = u_or_ts2[:i_u2_where_u1_max]  # otherwise there was crossing with sonic profiles at temps 5.9,
        r2_crop = r2[:i_u2_where_u1_max]

        # print('__ len(u2_crop){} , len(r2_crop){}'.format(len(u2_crop),len(r2_crop) ))

        if u2_crop.any() and r2_crop.any():

            u_lim1 = np.array([u_or_t1.min(), u2_crop.min()]).max()
            u_lim2 = np.array([u_or_t1.max(), u2_crop.max()]).min()

            if u_lim2 < u_lim1:
                raise ValueError('u_lim1({}) < u_lim2({})'.format(u_lim1, u_lim2))

            u_or_t_grid = np.mgrid[u_lim2:u_lim1:self.depth * 1j]

            if u_or_t_grid.max() > u_or_ts2.max() or u_or_t_grid.max() > u_or_t1.max():
                raise ValueError('u_or_t_grid.max({}) > u2.max({}) or u_or_t_grid.max({}) > u1.max({})'
                                 .format(u_or_t_grid.max(), u_or_ts2.max(), u_or_t_grid.max(), u_or_t1.max()))

            # plt.show()
            r1_grid = Math.interpolate_arr(u_or_t1, r1,      u_or_t_grid, self.set_interp_cross_ut_usts_method)
            #r2_grid = Math.interpolate_arr(u2_crop, r2_crop, u_or_t_grid, self.set_interp_cross_ut_usts_method)


            if self.set_do_plot_tech:
                self.plot_additional('r', u_ot_t, r1_grid, u_or_t_grid, '-')
            # plt.show()
            r2_grid = Math.interpolate_arr(u2_crop, r2_crop, u_or_t_grid, self.set_interp_cross_ut_usts_method)
                # if u_ot_t == 'u':
                #     self.ax_tech[0].plot(r1_grid, u_or_t_grid, '-.', color='green')
                #     self.ax_tech[0].plot(r2_grid, u_or_t_grid, '-.', color='green')
                # else:
                #     self.ax_tech[1].plot(r1_grid, u_or_t_grid, '-.', color='green')
                #     self.ax_tech[1].plot(r2_grid, u_or_t_grid, '-.', color='green')


            uc, rc = Math.interpolated_intercept(u_or_t_grid, r1_grid, r2_grid)



            if uc.any():  # if there is an intersections between sonic vel. profile and max.r-u line
                uc0 = uc[0][0]
                rc0 = rc[0][0]

                if self.set_do_plot_tech:
                    self.plot_additional('r', u_ot_t, np.array([rc0]), np.array([uc0]), 'X', 'red')


                    # if u_ot_t == 'u':
                    #     self.ax_tech[0].plot(rc0, uc0, 'X', color='green')
                    #     self.ax_tech[0].annotate(str('%.2f' % mdot), xy=(rc0, uc0), textcoords='data')
                    # if u_ot_t == 't':
                    #     self.ax_tech[1].plot(rc0, uc0, 'X', color='green')
                    #     self.ax_tech[1].annotate(str('%.2f' % mdot), xy=(rc0, uc0), textcoords='data')

                delta = u_or_ts2[Math.find_nearest_index(mdots, mdot)] - uc0
                # print('__Delta: {} mdot {}, uc0: {} '.format(delta, mdot, uc0))
                # print(uc, rc, '\t', delta)
                # print('Delta: ' , delta_ut, )

                return delta
            # else:
            #     plt.plot(r1, u_or_t1, 'x', color='blue', label='r1_grid')
            #     plt.plot(r2, u_or_ts2, 'x', color='red',  label='r2_grid')
            #     plt.plot(r1_grid, u_or_t_grid, '.', color='blue', label='r1_grid')
            #     plt.plot(r2_grid, u_or_t_grid, '.', color='red',  label='r2_grid')
            #     plt.title('No Intersection')
            #     plt.show()
            #     plt.legend()
            #     plt.grid()
            #     raise ValueError('No Intersection')


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
            # else:
            #     raise ValueError('No crossing between u_s=f(r) and r_ut_mdot_max found for mdot: {}'.format(cl.get_col('mdot')[-1]))



        if len(mdot_delta_ut) == 0:
            raise ValueError('mdot_delta_ut is not found at all for <{}>'.format(u_or_t))

        mdot_delta_ut = np.sort(mdot_delta_ut.view('f8, f8'), order=['f0'], axis=0).view(np.float)
        mdot_delta_ut_shape = np.reshape(mdot_delta_ut, (n, 2))

        mdot = mdot_delta_ut_shape[:, 0]
        delta_ut = mdot_delta_ut_shape[:, 1]

        crit_mdot_u = Math.solv_inter_row(mdot, delta_ut, 0.)  # Critical Mdot when the delta_ut == 0

        # print('\t\t crit_mdot', crit_mdot_u)


        if not crit_mdot_u.any():
            plt.show()
            raise ValueError('Critical Mdot is not found for {}'.format(u_or_t))
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
    # def get_critical_mdot(self, mdot_arr, delta_arr, u_or_t):
    #     '''
    #     Takes the array of [r, u_max, mdot] and [r ts_max, mdot] and by construting the delta values
    #     between every u_max and u_sonic, (perpendicular to u_sonic), plots the u_i - u_s = f(mdot)
    #     Using this it interpolates the mdot at which where u_i - u_s = 0, and this is the critcial mdot.
    #     Returns the mdot_crit, r_crit, and t_crit.
    #     :param ax:
    #     :param sm_cls:
    #     :param r_u_mdot_max:
    #     :param r_t_mdot_max:
    #     :u_min: 0.1 for cutting the profiles to [min_ind:]
    #     :return: critical mdot
    #     '''
    #
    #     # interpol. up up to max of u_s [green]
    #
    #     crit_mdot = Math.interpolate_arr(delta_arr, mdot_arr, np.array([0.]), self.set_crit_mdot_interp_method)
    #
    #     # if u_or_t == 'u' and self.plot_mdot_delta_u:
    #     #     self.ax_mdot_u.plot(mdot_arr, delta_arr, '-', color='black')
    #     #     self.ax_mdot_u.plot(crit_mdot, 0., 'x', color='black')
    #     # if u_or_t == 't' and self.plot_mdot_delta_t:
    #     #     self.ax_mdot_t.plot(mdot_arr, delta_arr, '-', color='black')
    #     #     self.ax_mdot_t.plot(crit_mdot, 0., 'x', color='black')
    #
    #     return crit_mdot  # returns one value!
    def get_critical_r(self, r_u_mdot_max, crit_mdot):
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

        crit_r= Math.interpolate_arr(r_u_mdot_max[:, 2], r_u_mdot_max[:, 0], np.array([crit_mdot]),
                                     self.set_crit_r_t_interp_method)


        print('\t__Criticals: r={}, mdot={}'.format('%.4f' % crit_r, '%.4f' % crit_mdot))


        return crit_r

    def get_critical_t(self, r_t_mdot_max, crit_mdot):
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

        crit_t = Math.interpolate_arr(r_t_mdot_max[:, 2], r_t_mdot_max[:, 1], np.array([crit_mdot]), self.set_crit_r_t_interp_method)  # r_crit


        print('\t__Criticals: t={}, mdot={}'.format('%.4f' % crit_t, '%.4f' % crit_mdot))


        return crit_t

    def extrapolate_crit_value(self, v_n_cond, mdot_cr, names, sonic_array, mdot_name='mdot-1'):

        if len(names)!=len(sonic_array[0, :]):
            raise ValueError('len(sonic_names{})!=len(sonic_array[0,:]{})'.format(len(names), len(sonic_array[0, :])))
        if not v_n_cond in names:
            raise NameError('v_n ({}) not in names ({})'.format(v_n_cond, names))


        if not mdot_name in names:
            raise NameError('mdot-1 not in names ({})'.format(names))

        mdot_col = sonic_array[:, names.index(mdot_name)]
        var_col = sonic_array[:, names.index(v_n_cond)]

        # self.ax[5].plot(mdot_col, var_col, '.', color='black')

        return Math.extrapolate_value(mdot_col, var_col, mdot_cr, self.set_crit_extrapol_method, None)



    # --- PUBLIC ---

    def main_crit_method(self):

        # plt.show()
        mdot_arr_u, delta_arr_u = self.get_mdot_delta(self.r_u_mdot_max, 'u')
        mdot_arr_t, delta_arr_t = self.get_mdot_delta(self.r_t_mdot_max, 't')

        mdot_cr_u = Math.interpolate_arr(delta_arr_u, mdot_arr_u, np.array([0.]), self.set_crit_mdot_interp_method)
        mdot_cr_t = Math.interpolate_arr(delta_arr_t, mdot_arr_t, np.array([0.]), self.set_crit_mdot_interp_method)

        r_cr = self.get_critical_r(self.r_u_mdot_max, mdot_cr_u)
        t_cr = self.get_critical_t(self.r_t_mdot_max, mdot_cr_t)


        if self.set_do_plot_tech:

            self.plot_additional('mdot', 'delta_u', mdot_arr_u, delta_arr_u, '-', 'black')
            self.plot_additional('mdot', 'delta_t', mdot_arr_t, delta_arr_t, '-', 'black')

            self.plot_vertial('r', 'u', r_cr, 'dashed', 'black')
            self.plot_vertial('r', 'u', r_cr, 'dashed', 'black')
            self.plot_horisontal('r', 't', t_cr, 'dashed', 'black')

            self.plot_vertial('mdot', 'delta_u', mdot_cr_u, 'dashed', 'black')
            self.plot_horisontal('mdot', 'delta_u', 0., 'dashed', 'black')
            self.plot_vertial('mdot', 'delta_t', mdot_cr_t, 'dashed', 'black')
            self.plot_horisontal('mdot', 'delta_t', 0., 'dashed', 'black')

            print('\n< << <<< <<<<| CRITICALS: Mdot_t:{} Mdot_u:{} R:{} T:{} |>>>> >>> >> >\n'
                  .format("%.2f"%mdot_cr_t, "%.2f"%mdot_cr_u, "%.2f"%r_cr, "%.2f"%t_cr))

            plt.show()

        self.out_mdot_cr_t = mdot_cr_t
        self.out_mdot_cr_u = mdot_cr_u
        self.out_r_cr = r_cr
        self.out_t_cr = t_cr

class ExtrapolateCriticals(PlotProfiles):

    def __init__(self, sp_names, sp_arr, v_n_cr, value_cr, sm_cls=list()):

        # def set_v_ns_for_plotting():
        #     self.set_v_ns = [[self.v_n_cr, self.set_v_ns_cr[0]]]

        PlotProfiles.__init__(self, [], [])

        self.set_v_ns_cr = []

        # self.set_v_ns = [['r', 'u'], ['r', 't']]

        self.sp_names = sp_names
        self.sp_arr   = sp_arr

        self.v_n_cr = v_n_cr
        self.cr_val = value_cr
        self.extrapol_method = 'IntUni' # Use 'test' to check different methods
        self.set_extrapol_lim_t = [6.5, 3.18]

        self.set_do_plots = True

        self.smdl = sm_cls

    def extrapolate_critical(self, v_n):

        if not v_n in self.sp_names: raise NameError('v_n to extrapolate ({}) not found in sonic array ({} \n '
                                                     .format(v_n, self.sp_names))

        if not v_n in self.sp_names: raise NameError('v_n base for extrapolation ({}) not found in sonic array ({} \n '
                                                     .format(v_n, self.sp_names))

        # limit extrapolation:


        sp_ts_ind = self.sp_names.index('t-sp')
        sp_ts_row = self.sp_arr[:, sp_ts_ind]

        sp_y_row = []
        sp_x_row = []

        sp_y_ind = self.sp_names.index(v_n)
        sp_x_ind = self.sp_names.index(self.v_n_cr)

        for i in range(len(sp_ts_row)):
            if sp_ts_row[i] >= np.array(self.set_extrapol_lim_t).min() and \
                sp_ts_row[i] <= np.array(self.set_extrapol_lim_t).max():
                sp_x_row = np.append(sp_x_row, self.sp_arr[i, sp_x_ind])
                sp_y_row = np.append(sp_y_row, self.sp_arr[i, sp_y_ind])

        # sp_y_ind = self.sp_names.index(v_n)
        # sp_y_row = self.sp_arr[:, sp_y_ind]
        #
        # sp_x_ind = self.sp_names.index(self.v_n_cr)
        # sp_x_row = self.sp_arr[:, sp_x_ind]
        #
        cr_x_val = self.cr_val

        if self.extrapol_method!='test':
            cr_y_val = Math.extrapolate_value(sp_x_row, sp_y_row, cr_x_val, self.extrapol_method)

            if self.set_do_plots:
                self.plot_additional(self.v_n_cr, v_n, sp_x_row, sp_y_row, '.', 'gray')
                self.plot_additional(self.v_n_cr, v_n, np.array([cr_x_val]), np.array([cr_y_val]), '.', 'red')

            return cr_y_val
        else:

            cr_y_val = Math.extrapolate_value(sp_x_row, sp_y_row, cr_x_val, self.extrapol_method,
                                              self.get_ax(self.v_n_cr, v_n,))

            return cr_y_val

    def main_extrapol_cycle(self):

        def set_v_ns_for_plotting(set_v_ns_cr, v_n_cr):
            v_ns = []
            for v_n in set_v_ns_cr:
                v_ns.append([v_n_cr, v_n])
            return v_ns

        self.set_v_ns = set_v_ns_for_plotting(self.set_v_ns_cr, self.v_n_cr)

        if self.set_do_plots:
            fig = plt.figure()
            self.set_plot_ax(fig)
            self.set_labels_for_all()

        cr_row = []
        cr_names = []
        for v_n in self.set_v_ns_cr:
            value = self.extrapolate_critical(v_n)
            cr_row = np.append(cr_row, value)
            cr_names.append(v_n)

        if self.set_do_plots:
            plt.show()

        self.out_crit_row = cr_row
        self.out_crit_names = cr_names

class NativeMassLoss(PlotProfiles):
    def __init__(self, sm_cls, wnd_cls):

        self.set_do_plot_native_tech=True
        self.set_v_ns = []  # , ['mdot', 'delta_grad_u']


        PlotProfiles.__init__(self, sm_cls, wnd_cls)


        self.set_use_poly_fit_core=True
        self.sm_cls = sm_cls
        self.wnd_cls = wnd_cls
        self.set_u_min = 0.5


    @staticmethod
    def grid(x, y, method='IntUni', depth=1000):
        x_grid = np.mgrid[x.min():x.max():depth * 1j]
        y_grid = np.mgrid[y.min():y.max():depth * 1j]

        # y_grid = interpolate.interp1d(x, y, kind='nearest')(x_grid)
        y_grid = Math.interpolate_arr(x, y, x_grid, method)

        return x_grid, y_grid

    @staticmethod
    def get_rising_xy(x, y):

        x = x[::-1]
        y = y[::-1]

        yn = y[0]
        x_res = []
        y_res = []
        diff = np.diff(y)
        for i in range(len(diff)):
            if diff[i] > 0:
                break
            else:
                x_res = np.append(x_res, x[i])
                y_res = np.append(y_res, y[i])

        return np.array(x_res[::-1]), np.array(y_res[::-1])

    def get_inner_boundary(self):
        '''
        RETURNS ' bourders.min() ' - min radius among all the models, where u exceeds the 'u_min'
        :param u_min:
        :return:
        '''
        bourders = []

        for i in range(len(self.sm_cls)):
            u = self.sm_cls[i].get_col('u')
            r = self.sm_cls[i].get_col('r')
            for i in range(len(r)):
                if u[i] > self.set_u_min:
                    # ax1.axvline(x=r[i], color='red')
                    bourders = np.append(bourders, r[i])
                    break

        return bourders.min()

    def get_int_r_u_core(self, cl, method='Uni', depth=1000, select_rising=True):
        u_core = cl.get_col('u')
        ind = Math.find_nearest_index(u_core, self.set_u_min)

        # mdot = cl.get_col('mdot')[-1]

        u_core = cl.get_col('u')[ind:]
        r_core = cl.get_col('r')[ind:]


        if select_rising: r_core, u_core = self.get_rising_xy(r_core, u_core)

        if len(r_core) < 5 or len(u_core) < 5:
            raise ValueError('No core profiles: len(r_core): {} Mdot:{}'.format(len(r_core), cl.get_col('mdot')[-1]))

        if self.set_use_poly_fit_core:
            r_core, u_core = self.grid(r_core, u_core, method, depth)

        return r_core, u_core

    def get_r_u_wind(self, wndcl, method='', depth=0):

        r_wind, u_wind = wndcl.get_col('r'), wndcl.get_col('u')
        ind = self.get_upper_wind_bound_index(wndcl)

        r_wind, u_wind = r_wind[:ind], u_wind[:ind]

        if self.set_use_poly_fit_core:
            if method!='' and depth!=0: r_wind, u_wind = self.grid(r_wind, u_wind, method, depth)

        return r_wind, u_wind

    def get_mdot(self, mdot_arr, delta_grad):

        mdot, delta = Math.interpolated_intercept(mdot_arr, np.zeros(len(delta_grad)), delta_grad)

        if len(mdot) > 2:
            plt.plot(mdot_arr, delta_grad, '.', color='black')
            plt.plot( mdot, delta, 'X', color='red')
            plt.axhline(y=0, ls='dashed', color='gray')
            plt.show()
            print('ERROR! More than two solutions found (while 1 is actually needed :( )')
            return mdot[-1]

        if len(mdot) < 1:
            return np.nan

        return mdot[-1]

    def main_cycle(self):

        if self.set_do_plot_native_tech:
            # self.set_v_ns = self.set_v_ns
            fig = plt.figure()
            self.set_plot_ax(fig)
            self.set_labels_for_all()


        arr = np.zeros(2)
        self.set_u_min = self.get_inner_boundary()
        if self.set_do_plot_native_tech:
            self.plot_sm_all(None, '.', 'blue')
            self.plot_wind_all('mdot', '.', 'red')
        # self.plot_wind_all('mdot', '-', 'gray')

        for i in range(len(self.sm_cls)):

            mdot = self.sm_cls[i].get_col('mdot')[-1]


            r_core, u_core = self.get_int_r_u_core(self.sm_cls[i], 'Uni', 1000, True) # select rising part = True
            if self.set_do_plot_native_tech: self.plot_additional('r', 'u', r_core, u_core, '-', 'gray')
            grad_core = np.gradient(r_core, u_core)


            r_wind, u_wind = self.get_r_u_wind(self.wnd_cls[i], 'Uni', 1000) # '' and 0 if no interpolation is needed.
            if self.set_do_plot_native_tech: self.plot_additional('r', 'u', r_wind, u_wind, '-', 'gray')
            grad_wind = np.gradient(r_wind, u_wind)

            arr = np.vstack((arr, [mdot, (grad_core[-1]-grad_wind[0]) ]))

        if self.set_do_plot_native_tech:
            self.plot_additional(None, 'delta_grad_u', arr[1:,0], arr[1:,1],'.','black')
            self.plot_additional('mdot', 'delta_grad_u', arr[1:, 0], arr[1:, 1], '-', 'black')


        mdot_nat = self.get_mdot(arr[1:, 0], arr[1:, 1])

        if self.set_do_plot_native_tech:
            self.plot_vertial('mdot', 'delta_grad_u', mdot_nat, 'dashed', 'black')
            self.plot_horisontal('mdot', 'delta_grad_u', 0., 'dashed', 'black')
            plt.show()

        self.mdot_naitive = np.float(mdot_nat)
        return np.float(mdot_nat)

class Wind:

    def __init__(self, wnd_cls):

        # self.swnd = sm_cls      # needed if tau < 2/3, if the wind is thin.
        self.swnd = wnd_cls
        self.set_depth = 1000
        self.v_n_arr = []

        self.set_interpol_method_ph_values = 'IntUni'

        # output

        self.out_names = []



    def get_coord_of_tau_ph(self, cl, v_n, tau_ph=2/3, method='IntUni'):

        tau_arr = cl.get_col('tau')[::-1]
        arr = cl.get_col(v_n)[::-1]

        if tau_arr[-1] == 0.0:
            tau_arr = tau_arr[:-1]
            arr = arr[:-1]

        if method == 'IntUni':

            value = Math.interpolate_arr(tau_arr, arr, np.array([tau_ph]), self.set_interpol_method_ph_values)

            if value == None or value == np.inf or value == -np.inf or value > arr.max() or value < arr.min():
                print('Error wrong value of {} at tau={}, where tau[{}, {}]'.format(v_n, tau_ph, tau_arr.min(), tau_arr.max()))
                r = cl.get_col('r')
                plt.plot(arr, tau_arr, '.', color='black')
                plt.axvline(x=value, ls='solid', color='blue')
                plt.axvline(x=arr.max(), ls='dashed', color='red')
                plt.axvline(x=arr.min(), ls='dashed', color='red')
                plt.axhline(y=tau_ph, ls='solid', color='blue')
                plt.xlabel(v_n)
                plt.ylabel('tau')

                mdot = cl.get_col('mdot')[-3]
                plt.title('ERROR Out of TAU at Mdot: {}'.format(mdot))
                plt.show()

                # plt.plot(r, arr, '.', color='black')
                # plt.axvline(x=value, ls='solid', color='blue')
                # plt.axvline(x=arr.max(), ls='dashed', color='red')
                # plt.axvline(x=arr.min(), ls='dashed', color='red')
                # plt.xlabel('r')
                # plt.ylabel(v_n)
                # plt.show()
                return np.nan
                raise ValueError('Error in interpolation Photospheric Value for {} value is {}'.format(v_n, value))
            else:
                return value

    def main_wind_cycle(self):
        '''
        Returns a 2d array with mdot a first row, and rest - rows for v_n,
        (if append0 or append1 != False: it will append before photosph. value, also the 0 or/and 1 value.
        '''

        # self.v_n_arr = self.v_n_arr
        out_arr = np.zeros(2 + len(self.v_n_arr))



        for cl in self.swnd:
            arr = []
            tau_ph = 2/3 # Main definition. (May changed because of otpically thin winds)

            mdot = cl.get_value('mdot', 10)     # Checking Mdot # ------------------------------------------------------
            if mdot == 0 or mdot == np.inf or mdot == -np.inf or mdot < -10**10 or mdot > 0:
                raise ValueError('Unphysical value of mdot: {}'.format(mdot))
            arr = np.append(arr, mdot)


            tau = cl.get_col('tau')             # Checking Tau #--------------------------------------------------------
            tau_val = tau[0]

            if tau[0]==0 and tau[1]>(2/3): tau_val = tau[1]
            if tau[0] < (2/3) and tau[1] < (2/3):
                tau_val = tau[0]
                tau_ph = tau[0]
                print('\t__Warning! Tau[0] < 2/3 ({}) For mdot: {}'.format(tau_val, mdot))

                # raise ValueError('Tau[0, and 1] < 2/3 [{} {}] (mdot:{})'.format(tau[0], tau[1], mdot))
            if tau[0] > 10000: raise ValueError('Suspicious value of tau[0] {}'.format(tau[0]))

            arr = np.append(arr, tau_val)   # Optical depth at the sonic point


            for v_n_cond in self.v_n_arr:   # appending all photosphereic (-ph) or index given (-0, -1, ... -n) values
                v_n = v_n_cond.split('-')[0]
                cond= v_n_cond.split('-')[-1]

                if v_n == 'mdot':
                    raise NameError('Mdot is already appended')
                if v_n == 'tau':
                    raise NameError('Tau-sp is appended by default as a first value of array')


                if cond == 'ph':
                    value = self.get_coord_of_tau_ph(cl, v_n, tau_ph, 'IntUni')
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

        self.out_names = head
        self.out_arr = out_arr

        # if len(head) != len(out_arr): raise ValueError('Arrays not equal: {} != {}'.format(len(head), len(out_arr)))

        return head, out_arr

class SavingOutput:
    def __init__(self):

        self.set_input_dirs = []
        self.set_output_dir = []
        self.set_dirs_not_to_be_included = []
        self.set_sp_fold_head = None
        self.set_extension='.data'

        self.set_v_n_delimeter = '    '

        pass

    def create_file_name(self):
        '''Creates name like:
        SP3_ga_z0008_10sm_y10
        using the initial folders, where the sm.data files are located
        '''
        out_name = self.set_sp_fold_head
        for i in range(len(self.set_input_dirs)):
            if self.set_input_dirs[i] not in self.set_dirs_not_to_be_included and self.set_input_dirs[i] != '..':
                out_name = out_name + self.set_input_dirs[i]
                if i < len(self.set_input_dirs) - 1:
                    out_name = out_name + '_'
        out_name = out_name + self.set_extension
        return out_name

    def save_out_arr(self, out_names, out_arr):

        head = out_names
        table = out_arr

        fname = self.create_file_name()

        tmp = ''
        for i in range(len(head)):
            tmp = tmp + head[i] + self.set_v_n_delimeter
        head__ = tmp

        np.savetxt(self.set_output_dir + fname, table, '%.5f', '  ', '\n', head__, '')

        print(' \n********* TABLE: {} IS SAVED IN {} *********\n'.format(fname, self.set_sp_fold_head))

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
            print('SORTING: {}'.format(v_n_cls[i][0]))

        v_n_cls.sort(key=lambda x: x[0])

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

    @staticmethod
    def check_sm_wind_file_names(smfiles, wndfiles):

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


    # --------------------APPENDING CRITICAL VALUES-------------------------

    @staticmethod
    def cr_fill_mask(out_names, out_arr, v_ns, n_of_col, mask=0.):

        for v_n in v_ns:
            if not v_n in out_names: raise NameError(' v_n: {} not in out_names {}'.format(v_n, out_names))

            value = out_arr[n_of_col, out_names.index(v_n)]

            if out_arr[0, out_names.index(v_n)] != mask:
                raise ValueError('Not a mask value detected: v_n {}, value: {}'.format(v_n, value))

            out_arr[0, out_names.index(v_n)] = value

        return out_names, out_arr

    @staticmethod
    def cr_append_critical_row(sp_names, sp_array, cr_names, cr_row, mask=0.):

        if len(sp_names) != len(sp_array[0,:]):
            raise ValueError('len(sp_names) {} != {} len(sp_array[0,:])'.format(len(sp_names), len(sp_array[0,:])))
        if len(cr_names) != len(cr_row):
            raise ValueError('len(cr_names) {} != {} len(cr_row)'.format(len(cr_names), len(cr_row)))

        out_arr = []
        for v_n in cr_names:
            if not v_n in sp_names: raise NameError('Critical v_n: {} not in sp_names {}'.format(v_n, sp_names))

        for v_n in sp_names:
            if v_n in cr_names:
                ind = cr_names.index(v_n)
                out_arr = np.append(out_arr, cr_row[ind])
            else:
                out_arr = np.append(out_arr, mask)

        if len(out_arr) != len(sp_names):
            raise ValueError('Something went wrong in making out_row with criticals')

        out_arr = np.vstack((out_arr, sp_array))

        return sp_names, out_arr

    @staticmethod
    def cr_combine_sp_criticals(sp_names, sp_array, cr_names, cr_row, mdot_cr=None, r_cr=None, t_cr=None):

        if mdot_cr != None:
            if not 'mdot-1' in sp_names:
                raise NameError('mdot-sp is not in sp_names ({}), Cannot append cricital'.format(sp_names))
            if not 'mdot-1' in cr_names:
                cr_names.append('mdot-1')
                cr_row = np.append(cr_row, mdot_cr)

        if r_cr != None:
            if not 'r-sp' in sp_names:
                raise NameError('r-sp is not in sp_names ({}), Cannot append cricital'.format(sp_names))
            if not 'r-sp' in cr_names:
                cr_names.append('r-sp')
                cr_row = np.append(cr_row, r_cr)

        if t_cr != None:
            if not 't-sp' in sp_names:
                raise NameError('t-sp is not in sp_names ({}), Cannot append cricital'.format(sp_names))
            if not 't-sp' in cr_names:
                cr_names.append('t-sp')
                cr_row = np.append(cr_row, t_cr)

        if len(cr_names) != len(cr_row):
            raise ValueError('len(cr_names){} != len(cr_row){}'.format(len(cr_names), len(cr_row)))

        return CommonMethods.cr_append_critical_row(sp_names, sp_array, cr_names, cr_row, 0.)



'''======================================================MAIN========================================================'''

def sp_wind_main(smfiles, wndfiles, input_dir, out_fname_z_m_y, new_fold='/'):

    mdl = []
    wnd = []

    CommonMethods.check_sm_wind_file_names(smfiles, wndfiles)

    for file in smfiles:
        mdl.append(Read_SM_data_file(file))
    smdl = CommonMethods.sort_smfiles(mdl, 'mdot', -1)

    for file in wndfiles:
        wnd.append(Read_Wind_file.from_wind_dat_file(file))
    swnd = CommonMethods.sort_smfiles(wnd, 'mdot', -1)


    # --------------------------------------------------| SETTINGS for sm.data |----------------------------------------
    sp = SonicPointAlgorithm(smdl)
    sp.v_n_sonic =     ['kappa-sp', 'L/Ledd-sp', 'HP-sp', 'mfp-sp'] # sp - means 'at sonic point'
    sp.v_n_core_surf = ['l-1', 'xm-1', 'He4-0', 'He4-1', 'mdot-1']  # 1 means surface, 0 means core
    sp.v_n_env =       ['t-env', 'u-env', 'r-env', 'm-env']         # env - menas envelope coordinate

    sp.ts_lim = 3.18 # set 5.18 if only iron bump needed

    sp.set_check_for_mult_sp=True
    sp.set_compute_envelope=True
    sp.set_delited_outer_points=False
    sp.set_if_not_found_use_last=True

    sp.main_sonic_cycle()

    sp_out_names = sp.out_names
    sp_out_arr = sp.out_arr

    # --------------------------------------------------| SETTINGS for .wind |------------------------------------------
    wd = Wind(swnd)

    wd.v_n_arr = ['r-ph', 't-ph'] # in addition to tau # ph - means photosphere (tau=2/3)

    wd.main_wind_cycle()

    wd_out_names = wd.out_names
    wd_out_arr = wd.out_arr

    out_names, out_arr = CommonMethods.combine_two_tables_by_v_n(sp_out_names, sp_out_arr, wd_out_names, wd_out_arr, 'mdot-1')

    # -----------------------------------------------| SETTINGS for Native Mdot |---------------------------------------

    n_mdot = NativeMassLoss(smdl, swnd)

    n_mdot.set_v_ns = [['mdot', 'delta_grad_u'], ['r', 'u']]  # [['mdot', 'delta_grad_u']]#
    n_mdot.upper_wind_bound = 'u=300'
    n_mdot.set_use_poly_fit_core = True
    n_mdot.set_use_poly_fit_wind = True

    n_mdot.set_do_plot_native_tech=True

    n_mdot.main_cycle()

    mdot_nat = n_mdot.mdot_naitive

    # PlotProfile.self_plot_tph_teff(smdl, swnd)

    # ------------------------------------------

    cr2 = ExtrapolateCriticals(out_names, out_arr, 'mdot-1', mdot_nat)

    cr2.set_v_ns_cr = ['kappa-sp', 'L/Ledd-sp', 'HP-sp', 'mfp-sp', 'tau-sp', 'r-sp', 't-sp', 'r-ph', 't-ph']  # for extrapolating (must also ne in sonic v_ns)
    cr2.set_do_plots = True
    cr2.set_extrapol_lim_t = [6.5, 5.18] # if some sonic models should be not used for extra/interpol

    cr2.main_extrapol_cycle()

    cr_names = cr2.out_crit_names
    cr_row = cr2.out_crit_row

    out_names, out_arr = CommonMethods.cr_combine_sp_criticals(out_names, out_arr, cr_names, cr_row, mdot_nat,
                                                               None, None)
    out_names, out_arr = CommonMethods.cr_fill_mask(out_names, out_arr, ['l-1', 'xm-1', 'He4-0', 'He4-1'], 1, 0.)

    # --------------------------------------------------| SETTINGS for PLOTTING |---------------------------------------
    pp = PlotProfiles(smdl, swnd, out_names, out_arr)

    pp.set_v_ns = [['r', 'u'], ['r', 't'], ['r', 'kappa'], ['t', 'L/Ledd'], ['t', 'tau']]

    fig = plt.figure()
    pp.set_plot_ax(fig)
    pp.set_labels_for_all()
    pp.plot_sm_all()
    pp.plot_wind_all()
    pp.plot_out_arr_points()
    plt.legend()
    plt.show()

    # --------------------------------------------------| SETTINGS for SAVING |-----------------------------------------

    save = SavingOutput()
    save.set_dirs_not_to_be_included = ['sse', 'ga_z002','ga_z0008', 'vnedora', 'media', 'HDD']
    save.set_input_dirs = input_dir
    save.set_output_dir = '../data/sp_w_files{}'.format(new_fold) + out_fname_z_m_y
    save.set_sp_fold_head = 'SP'

    save.save_out_arr(out_names, out_arr)

def ga_crit_main(smfiles, input_dir, out_fname_z_m_y):

    mdl = []

    for file in smfiles:
        mdl.append(Read_SM_data_file(file))
    smdl = CommonMethods.sort_smfiles(mdl, 'mdot', -1)

    # --------------------------------------------------| SETTINGS for sm.data |----------------------------------------
    sp = SonicPointAlgorithm(smdl)
    sp.v_n_sonic     = ['kappa-sp', 'L/Ledd-sp', 'HP-sp', 'mfp-sp']  # sp - means 'at sonic point'
    sp.v_n_core_surf = ['l-1', 'xm-1', 'He4-0', 'He4-1',  'mdot-1']  # 1 means surface, 0 means core

    sp.set_check_for_mult_sp = False
    sp.set_compute_envelope = False
    sp.set_delited_outer_points = True
    sp.set_if_not_found_use_last = False

    sp.main_sonic_cycle()

    sp_out_names = sp.out_names
    sp_out_arr = sp.out_arr

    r_t_mdot_max = sp.r_t_mdot_max
    r_u_mdot_max = sp.r_u_mdot_max

    # --------------------------------------------------| SETTINGS for CRITICALS |--------------------------------------
    cr = CriticalMdotRT(smdl, r_u_mdot_max, r_t_mdot_max)
    # ['r', 'u'], ['r', 't'], ['mdot', 'delta_u'], ['mdot', 'delta_t']
    cr.set_v_ns =  [['r', 'u'], ['r', 't'], ['mdot', 'delta_u'], ['mdot', 'delta_t']]   # PLOTS
    cr.set_do_plot_tech=False       # switch for tech plots

    # cr.set_initialize_plots()
    # cr.plot_sm_all()

    cr.main_crit_method()

    cr_mdot = cr.out_mdot_cr_u
    cr_t = cr.out_t_cr
    cr_r = cr.out_r_cr


    cr2 = ExtrapolateCriticals(sp_out_names, sp_out_arr, 'mdot-1', cr_mdot)

    cr2.set_v_ns_cr = ['kappa-sp', 'L/Ledd-sp', 'HP-sp', 'mfp-sp'] # for extrapolating (must also ne in sonic v_ns)
    cr2.set_do_plots = False

    cr2.main_extrapol_cycle()

    cr_names = cr2.out_crit_names
    cr_row   = cr2.out_crit_row

    out_names, out_arr = CommonMethods.cr_combine_sp_criticals(sp_out_names, sp_out_arr, cr_names, cr_row, cr_mdot, cr_r, cr_t)
    out_names, out_arr = CommonMethods.cr_fill_mask(out_names, out_arr, ['l-1', 'xm-1', 'He4-0', 'He4-1'], 1, 0.)


    # --------------------------------------------------| SETTINGS for PLOTTING |---------------------------------------
    pp = PlotProfiles(smdl, [], sp_out_names, sp_out_arr)

    pp.set_v_ns = [['r', 'u'], ['r', 't'], ['r', 'kappa'], ['t', 'L/Ledd']]

    # fig = plt.figure()
    # pp.set_plot_ax(fig)
    # pp.set_labels_for_all()
    # pp.plot_sm_all()
    # pp.plot_r_ut_mdot_max(r_t_mdot_max, 't')
    # pp.plot_r_ut_mdot_max(r_u_mdot_max, 'u')
    # pp.plot_vertial('r','u',cr_r,'dashed','red')
    # pp.plot_out_arr_points()
    #
    #
    # plt.legend()
    # plt.show()

    # --------------------------------------------------| SETTINGS for SAVING |-----------------------------------------
    save = SavingOutput()
    save.set_dirs_not_to_be_included = ['sse', 'ga_z002','ga_z002_2', 'ga_z0008','ga_z0008_2', 'vnedora', 'media', 'HDD']
    save.set_input_dirs = input_dir
    save.set_output_dir = '../data/sp_cr_files/' + out_fname_z_m_y
    save.set_sp_fold_head = 'SP'

    save.save_out_arr(out_names, out_arr)

'''=====================================================FILES========================================================'''

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

def load_files(z, m_set, y_set, sp=False, deep_fold=''):

    for m in m_set:
        for y in y_set:
            root_name = 'ga_z' + z + '/'
            folder_name = str(m)+'sm/y'+str(y)+'/'
            out_name = str(m) + 'z' + z + '/'

            print('COMPUTING: ({}) {} , to be saved in {}'.format(root_name, folder_name, out_name))

            # smfiles_ga = get_files(sse_locaton + root_name, [folder_name], [], 'sm.data')

            if sp: sp='sp55/' + deep_fold
            else: sp=''

            smfiles_sp  = get_files(sse_locaton + root_name, [folder_name+sp], [], 'sm.data')

            if deep_fold!='' and deep_fold!=None: deep_fold = '_' + deep_fold + '/'
            else: deep_fold = '/'
            if sp:
                wind_fls    = get_files(sse_locaton + root_name, [folder_name+sp], [], 'wind')

                cr = sp_wind_main(smfiles_sp, wind_fls, smfiles_sp[0].split('/')[:-1], out_name, deep_fold)
            else:
                ga = ga_crit_main(smfiles_sp, smfiles_sp[0].split('/')[:-1], out_name)



            print('m:{}, y:{} DONE'.format(m,y))


# 11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30
load_files('002', [14], [10], True, 'b075/')

def manual(full_path, sp=True):
    sm_files = get_files('', [full_path], [], 'sm.data')

    out_name='test.data'
    if sp:
        wind_fls = get_files('', [full_path], [], 'wind')
        cr = sp_wind_main(sm_files, wind_fls, sm_files[0].split('/')[:-1], out_name)
    else:
        ga = ga_crit_main(sm_files, sm_files[0].split('/')[:-1], out_name)

# manual('/media/vnedora/HDD/sse/ga_z002_2/20sm/y10/test/')