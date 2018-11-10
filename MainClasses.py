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

# from sklearn.linear_model import LinearRegression
# import scipy.ndimage
# from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
# from scipy.interpolate import griddata
import os
#-----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------CLASSES-----------------------------------------------------
from PhysMath import Math, Physics, Constants

from OPAL import Table_Analyze

from FilesWork import Read_Observables, Read_Plot_file, Read_SM_data_file, Read_SP_data_file, Save_Load_tables
from FilesWork import Save_Load_tables, T_kappa_bump, Files, Labels, Get_Z

from FilesWork import PlotObs, PlotBackground

# from PhysPlots import PhysPlots
#-----------------------------------------------------------------------------------------------------------------------

class GenericMethods():
    def __init__(self, sm_files):

        # if len(file) > 1: raise IOError('More than 1 sm.data file given {}'.format(len(file)))

        self.sm_cls = []
        for file in sm_files:
            self.sm_cls.append(Read_SM_data_file(file))
        if len(self.sm_cls) == 0: print('Warning! No sm_files froided')



    @staticmethod
    def formula(t, kappa, gamma):
        '''
        left2 = d(ln(kappa) / d(ln(t))
        right2 = 1 - (1/gamma)
        :param t:
        :param kappa:
        :param gamma:
        :return:
        '''
        t = 10 ** t
        kappa = 10 ** kappa

        left2 = np.gradient(np.log(kappa), np.log(t))
        right2 = 1 - (1 / gamma)
        return left2, right2

    def inflection_point(self, v_n_x=None, v_n_add=None, ax=None, clean=False):



        left, right = self.formula(self.get_col('t'), self.get_col('kappa'), self.get_col('L/Ledd'))

        if ax == None:
            plot_show = True
        else:
            plot_show = False

        if v_n_x != None:
            if ax == None:
                plt.figure()
                ax = plt.subplot(111)
            ax.set_xlabel(Labels.lbls(v_n_x), color='k')
            ax.tick_params('y', colors='k')
            ax.grid()

            x_coord = self.get_col(v_n_x)
            ax.plot(x_coord, left, '-', color='red', label='Left')
            ax.plot(x_coord, right, '-', color='blue', label='Right')

            xp, yp = Math.interpolated_intercept(x_coord, left, right)
            ax.plot(xp, yp, 'X', color='black')

            if v_n_add != None:
                y2 = self.get_col(v_n_add)
                if v_n_add == 'kappa': y2 = 10**y2

                y2 = y2/y2.max()

                ax2 = ax.twinx()
                ax2.plot(x_coord, y2, '--', color='gray')
                ax2.set_ylabel(Labels.lbls(v_n_add), color='gray')
        if not clean:
            ax.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
        if plot_show:
            plt.legend()
            plt.show()

    def plot_multiple_inflect_point(self, v_n_x, v_n_y, plot_inflect_eq=True, clean=False, fsz=12):

        if len(self.sm_cls) == 0: raise IOError('No sm.data classes found')

        # if ax == None:
        #     plot_show = True
        # else:
        #     plot_show = False

        # if ax == None:
        plt.figure()
        ax = plt.subplot(111)
        ax.set_xlabel(Labels.lbls(v_n_x), color='k', fontsize=fsz)
        ax.set_ylabel(Labels.lbls(v_n_y), color='k', fontsize=fsz)
        ax.minorticks_on()
        # ax.tick_params('y', colors='k')
        # ax.tick_params('x', colors='k')

        if plot_inflect_eq:
            ax2 = ax.twinx()
            ax2.minorticks_on()
            # ax2.set_ylabel(None, color='gray', fontsize=fsz)
        else: ax2 = None


        x_y_infl = np.zeros(2)
        i = 0
        for cl in self.sm_cls:

            mdot_str = "%.2f" % cl.get_col('mdot')[-1]

            x_coord = cl.get_col(v_n_x)
            y2 = cl.get_col(v_n_y)
            if v_n_y == 'kappa': y2 = 10 ** y2
            # y2 = y2 / y2.max()
            ax.plot(x_coord, y2, '-', color='C' + str(Math.get_0_to_max([i], 9)[i]))
            ax.plot(x_coord[-1], y2[-1], 'X', color='C' + str(Math.get_0_to_max([i], 9)[i]))
            ax.annotate('{}'.format(mdot_str), xy=(x_coord[-1], y2[-1]), textcoords='data',
                        horizontalalignment='left')


            if v_n_y == 'u':
                u_s = cl.get_sonic_u()
                ax.plot(x_coord, u_s, '--', color='gray')



            if plot_inflect_eq:
                left, right = self.formula(cl.get_col('t'), cl.get_col('kappa'), cl.get_col('L/Ledd'))

                xp, yp = Math.interpolated_intercept(x_coord, left, right)
                if v_n_x == 'r': x_y_infl = np.vstack((x_y_infl, [xp[0][0], yp[0][0]]))
                # if v_n_x == 't': x_y_infl = np.vstack((x_y_infl, [xp[-1], yp[-1]]))

                ind = Math.find_nearest_index(x_coord, xp[0][0])+100

                #ax2.plot(x_coord[:ind], left[:ind], '--', color='C' + str(Math.get_0_to_max([i], 9)[i]),
                #         label='{}:{} (L)'.format(Labels.lbls('mdot'), mdot_str))
                #ax2.plot(x_coord[:ind], right[:ind], '-.', color='C' + str(Math.get_0_to_max([i], 9)[i] + 1),
                #         label='{}:{} (R)'.format(Labels.lbls('mdot'), mdot_str))

                ax.plot(xp[0][0], y2[Math.find_nearest_index(x_coord,xp[0][0])], 'X', color='black')

                # ax2.plot(xp, yp, 'X', color='black')
            i = i + 1



        if not clean:
            plt.legend(bbox_to_anchor=(0, 0), loc='lower left', fontsize=fsz, ncol=1)

        if plot_inflect_eq: plt.xlim(0, x_y_infl[1:, 0].max())
        if v_n_y == 'Pg/P_total': ax.axhline(y=0.15, ls='dashed', color='gray')

        ax.grid()
        plt.xticks(fontsize=fsz)
        plt.yticks(fontsize=fsz)
        ax.tick_params('y', labelsize=fsz)
        ax.tick_params('x', labelsize=fsz)

        plt.minorticks_on()
        plt.show()

        # -----------------------------------__METALLICITY__
        plt.figure()
        ax = plt.subplot(111)
        ax.set_xlabel(Labels.lbls('r_infl'), color='k', fontsize=fsz)
        ax.set_ylabel(Labels.lbls('z'), color='k', fontsize=fsz)
        ax.minorticks_on()

        plt.xticks(fontsize=fsz)
        plt.yticks(fontsize=fsz)
        ax.tick_params('y', labelsize=fsz)
        ax.tick_params('x', labelsize=fsz)
        plt.plot(x_y_infl[1:, 0], [0.008, 0.02, 0.04], '-', color='black')
        plt.plot(x_y_infl[1:, 0], [0.008, 0.02, 0.04], 'x', color='black')
        plt.show()


class Combine:
    output_dir = '../data/output/'
    plot_dir = '../data/plots/'

    set_metal = ''
    set_sm_files = []
    set_sp_files = []
    set_sm_files2 = []

    set_obs_file = ''
    set_plot_files = []
    m_l_relation = None


    def __init__(self):
        pass



    def set_files(self):
        self.mdl = []
        for file in self.set_sm_files:
            self.mdl.append( Read_SM_data_file(file) )

        self.mdl2 = []
        for file in self.set_sm_files2:
            self.mdl2.append( Read_SM_data_file(file) )

        self.spmdl=[]
        for file in self.set_sp_files:
            self.spmdl.append( Read_SP_data_file(file, self.output_dir, self.plot_dir) )


        # self.nums = Num_Models(smfls, plotfls)
        # self.obs = Read_Observables(self.set_obs_file, self.set_metal)



    # --- METHODS THAT DO NOT REQUIRE OPAL TABLES ---



    def tst(self):

        def beta_law(r, r0, v0, vinf, beta):
            return (v0  + (vinf - v0) * ((1 - (r0 / r))**beta))
            # return vinf*((1 - (r0 / r))**beta)


        fig = plt.figure()
        ax1 = fig.add_subplot(111)


        # plt.title(tlt)

        x_tot = []
        y_tot = []
        y_tot2 = []
        y_tot3 = []

        for i in range(len(self.set_sm_files)):

            # inflection_point(self.mdl[i])

            x = self.mdl[i].get_col('r')
            y = self.mdl[i].get_col('u')  # simpler syntaxis
            label1 = self.mdl[i].get_col('mdot')[-1]

            # x = np.log(x)
            # y = np.log(y)

            # y = np.gradient(y, x)
            #
            # x = np.log(x)
            # y = np.log(y)

            # ax1.set_xlim(x[-100], x[-1])
            # ax1.set_ylim(x[-100], x[-1])


            lbl = '{}:{}'.format('Mdot', '%.2f' % label1)

            # ax1.plot(x,  y,  '-',   color='C' + str(Math.get_0_to_max([i], 9)[i]), label=lbl)
            # ax1.plot(x, y, 'x', color='C' + str(Math.get_0_to_max([i], 9)[i]))
            # ax1.plot(x, y, '.', color='C' + str(Math.get_0_to_max([i], 9)[i]))
            # ax1.plot(x[-1], y[-1], '.', color='C' + str(Math.get_0_to_max([i], 9)[i]))

            xinf = x[-1]+1.0
            x_w = np.mgrid[x[-1]:xinf:100j]

            y_w = []

            for j in range(len(x_w)):
                y_w = np.append(y_w, beta_law(x_w[j], x[-1], y[-1], 1800, 1.0))

            # ax1.plot(x_w, y_w, '-', color='gray')
            #
            # ax1.annotate(str('%.1f' % self.mdl[i].get_col('mdot')[-1]), xy=(x[-1], y[-1]), textcoords='data')
            # ax1.plot(x[-1], y[-1], '.', markersize='5', color='C' + str(Math.get_0_to_max([i], 9)[i]))

            # x, y = np.log10(x), np.log10(y)
            # x_w, y_w = np.log10(x_w), np.log10(y_w)

            x = x*Constants.solar_r / 10**5
            x_w = x_w * Constants.solar_r / 10**5

            y_grad = np.gradient(y, x)
            y_w_grad = np.gradient(y_w, x_w)

            # grad = y_grad - y_w_grad

            # y_grad = np.log(y_grad)
            # y_w_grad =  np.log(y_w_grad)

            # ax1.annotate(str('%.1f' % self.mdl[i].get_col('mdot')[-1]), xy=(x[-1], y[-1]), textcoords='data')
            # ax1.plot(x[-1], y_w_grad[-1], '.', markersize='5', color='C' + str(Math.get_0_to_max([i], 9)[i]))
            # ax1.plot(x[-1], y_grad[-1], '.', markersize='5', color='C' + str(Math.get_0_to_max([i], 9)[i]))
            # ax1.plot(x[-1], grad[-1], '.', markersize='5', color='C' + str(Math.get_0_to_max([i], 9)[i]))

            x_tot = np.append(x_tot, self.mdl[i].get_col('mdot')[-1])
            y_tot = np.append(y_tot, y_grad[-1])
            y_tot2 = np.append(y_tot2, y_w_grad[-1])
            y_tot3 = np.append(y_tot3, (y_grad[-1]-y_w_grad[-1]))
            # if sonic and v_n2 == 'u':
            #     u_s = self.mdl[i].get_sonic_u()
            #     ax1.plot(x, u_s, '-', color='black')
            #
            #     xc, yc = Math.interpolated_intercept(x, y, u_s)
            #     # print('Sonic r: {} | Sonic u: {} | {}'.format( np.float(xc),  np.float(yc), len(xc)))
            #     plt.plot(xc, yc, 'X', color='red')
            #
            # if v_n2 == 'kappa':
            #     k_edd = 10 ** Physics.edd_opacity(self.mdl[i].get_col('xm')[-1],
            #                                       self.mdl[i].get_col('l')[-1])
            #     ax1.plot(ax1.get_xlim(), [k_edd, k_edd], color='black', label='Model: {}, k_edd: {}'.format(i, k_edd))
            #
            # if v_n2 == 'Pg/P_total':
            #     plt.axhline(y=0.15, color='black')

            # tp, up = get_m_r_envelope(self.mdl[i])
            # if tp != None:
            #     ax1.plot(tp, up, 'X', color='black')

        x_tot, y_tot3 = Math.x_y_z_sort(x_tot, y_tot3)

        # ax1.plot(x_tot, y_tot, '-',  color='red' )
        ax1.plot(x_tot, y_tot3, '-',  color='black' )
        ax1.plot(x_tot, y_tot3, 'x', color='black')

        plt.axhline(y=0)

        y_tot3, x_tot = Math.x_y_z_sort(y_tot3, x_tot)
        xp = interpolate.InterpolatedUnivariateSpline(y_tot3, x_tot)(0.)
        ax1.plot(xp,0, 'X', markersize='5', color='black')
        print('Coord: {} '.format("%.5f"% xp))

        ax1.set_xlabel(Labels.lbls('r'), fontsize=12)
        ax1.set_ylabel(Labels.lbls('u'), fontsize=12)

        ax1.grid(which='both')
        ax1.grid(which='minor', alpha=0.2)
        ax1.grid(which='major', alpha=0.2)

        # ax1.set_xlim(0.78, 0.92)
        # ax1.set_xlim(4.0, 6.2)
        # if not clean:
        ax1.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
        # plot_name = self.plot_dir + v_n1 + '_vs_' + v_n2 + '_profile.pdf'
        # plt.savefig(plot_name)
        plt.show()


    def xy_profile(self, v_n1, v_n2, var_for_label1, var_for_label2, sonic = True, clean = False, fsz=12):

        # def get_m_r_envelope(smcl, t_lim1=5.1, t_lim2=5.3):
        #     '''
        #     Looks for a loal extremum between t_lim1 and t_lim2, and, if the extremem != sonic point: returns
        #     length and mass of whatever is left
        #     :param smcl:
        #     :param t_lim1:
        #     :param t_lim2:
        #     :return:
        #     '''
        #
        #     def get_envelope_l_or_m(v_n, cls, t_start, depth = 1000):
        #         t = cls.get_col('t')
        #         ind = Math.find_nearest_index(t,t_start) - 1 # just before the limit, so we don't need to interpolate across the whole t range
        #         t = t[ind:]
        #         var = cls.get_col(v_n)
        #         var = var[ind:]
        #
        #
        #         value = interpolate.InterpolatedUnivariateSpline(t[::-1], var[::-1])(t_start)
        #
        #         # print('-_-: {}'.format(var[-1]-value))
        #
        #         return (var[-1]-value)
        #
        #     t = smcl.get_col('t')  # x - axis
        #     u = smcl.get_col('u')  # y - axis
        #
        #     # if t.min() > t_lim1 and t_lim2 > t.min():
        #     # i1 = Math.find_nearest_index(t, t_lim2)
        #     # i2 = Math.find_nearest_index(t, t_lim1)
        #
        #     # t_cropped= t[Math.find_nearest_index(t, t_lim2):Math.find_nearest_index(t,
        #     #                                                                          t_lim1)][::-1]   # t_lim2 > t_lim1 and t is Declining
        #     # u_cropped = u[Math.find_nearest_index(t, t_lim2):Math.find_nearest_index(t, t_lim1)][::-1]
        #
        #     # print('<<<<<<<<<<<SIZE: {} {} (i1:{}, i2:{}) >>>>>>>>>>>>>>>>'.format(len(t), len(u), i1, i2))
        #
        #     tp, up = Math.get_max_by_interpolating(t, u, False, 5.2)
        #     if Math.find_nearest_index(t,tp) < len(t)-1: # if the tp is not the last point of the t array ( not th sonic point)
        #
        #         print('<<<<<<<<<<<Coord: {} {} >>>>>>>>>>>>>>>>'.format("%.2f" % tp, "%.2f" % up))
        #
        #         print('L_env: {}'.format(get_envelope_l_or_m('r', smcl, tp)))
        #         print('M_env: {}'.format(np.log10(get_envelope_l_or_m('xm', smcl, tp))))
        #
        #         # var = get_envelope_l_or_m('r', smcl, tp)
        #         return tp, up
        #     else:
        #         return None, None
        #
        #
        #
        #     # else: return None, None
        #
        # def inflection_point(sm_cl, v_n_x=None, v_n_add=None, ax=None):
        #
        #     def formula(t, kappa, gamma):
        #         '''
        #         left2 = d(ln(kappa) / d(ln(t))
        #         right2 = 1 - (1/gamma)
        #         :param t:
        #         :param kappa:
        #         :param gamma:
        #         :return:
        #         '''
        #         right2 = np.gradient(np.log(kappa), np.log(t))
        #         left2 = 1 - (1 / gamma)
        #         return left2, right2
        #
        #     left, right = formula(sm_cl.get_col('t'),sm_cl.get_col('kappa'),sm_cl.get_col('L/Ledd'))
        #
        #     if ax==None: plot_show=False
        #     else: plot_show=True
        #
        #     if v_n_x!=None:
        #         if ax==None:
        #             plt.figure()
        #             ax = plt.subplot(111)
        #         ax.set_xlabel(Labels.lbls(v_n_x),  color='k')
        #         ax.tick_params('y', colors='k')
        #         ax.grid()
        #
        #         x_coord = sm_cl.get_col(v_n_x)
        #         ax.plot(x_coord, left, '-', color='red', label='Left')
        #         ax.plot(x_coord, right, '-', color='blue', label='Right')
        #         if v_n_add != None:
        #             y2 = sm_cl.get_col(v_n_add)
        #             ax2 = ax1.twinx()
        #             ax2.plot(x_coord, y2, '--', color='gray')
        #             ax2.set_ylabel(Labels.lbls(v_n_add), color='gray')
        #     if plot_show:
        #         plt.show()
        #
        #
        # def inflection_point(cl):
        #     kappa = 10**cl.get_col('kappa')
        #     t = 10**cl.get_col('t')
        #     r = cl.get_col('r')
        #     gamma = cl.get_col('L/Ledd')
        #
        #     u = cl.get_col('u')
        #
        #     dkappa = []
        #     dt = []
        #     t_=[]
        #     kappa_ = []
        #     gamma_ = []
        #
        #     left = []
        #     right = []
        #
        #     u_ = []
        #     r_ = []
        #
        #     for i in range(1, len(kappa)):
        #         dkappa = np.append(dkappa, kappa[i]-kappa[i-1])
        #         dt = np.append(dt, t[i] - t[i-1])
        #         t_ = np.append(t_, t[i-1] + (t[i] - t[i-1])/2)
        #         r_ = np.append(r_, r[i-1] + (r[i] - r[i-1])/2)
        #         kappa_ = np.append(kappa_, kappa[i-1] + (kappa[i]-kappa[i-1])/2)
        #         gamma_ = np.append(gamma_, gamma[i-1] + (gamma[i]-gamma[i-1])/2)
        #         u_ = np.append(u_, u[i-1] + (u[i]-u[i-1])/2)
        #     for i in range(len(kappa_)):
        #
        #         right = np.append(right, (t_[i]/kappa_[i])*(dkappa[i]/dt[i]))
        #         left = np.append(left, (1-1/gamma_[i]))
        #
        #     right2 = np.gradient(np.log(kappa), np.log(t))
        #     left2 = 1 - (1/gamma)
        #
        #     u_norm = u_#/u_.max()
        #
        #     r_ = np.log10(t_)
        #
        #     # plt.plot(r, u_norm, '-.', color='gray')
        #     plt.plot(r, left2, '.', color='black')
        #     plt.plot(r, right2, '.', color='red')
        #     plt.xlabel(Labels.lbls('r'))
        #     plt.show()


        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        tlt = v_n2 + '(' + v_n1 + ') profile'
        # plt.title(tlt)

        for i in range(len(self.set_sm_files)):

            # inflection_point(self.mdl[i])


            x =      self.mdl[i].get_col(v_n1)
            y      = self.mdl[i].get_col(v_n2)          # simpler syntaxis
            label1 = self.mdl[i].get_col(var_for_label1)[-1]
            label2 = self.mdl[i].get_col(var_for_label2)[-1]

            # x = np.log(x)
            # y = np.log(y)

            # y = np.gradient(y,x)

            # x = np.log(x)
            # y = np.log(y)

            # ax1.set_xlim(x[-100], x[-1])
            # ax1.set_ylim(x[-100], x[-1])

            if v_n2 == 'kappa':
                y = 10**y

            print('\t __Core H: {} , core He: {} File: {}'.
                  format(self.mdl[i].get_col('H')[0], self.mdl[i].get_col('He4')[0], self.set_sm_files[i]))

            lbl = '{}:{} , {}:{}'.format(var_for_label1,'%.2f' % label1,var_for_label2,'%.2f' % label2)

            #ax1.plot(x,  y,  '-',   color='C' + str(Math.get_0_to_max([i], 9)[i]), label=lbl)
            #ax1.plot(x, y, 'x', color='C' + str(Math.get_0_to_max([i], 9)[i]))
            # ax1.plot(x, y, '.', color='C' + str(Math.get_0_to_max([i], 9)[i]))
            ax1.plot(x[-1], y[0], 'x',   color='C' + str(Math.get_0_to_max([i], 9)[i]))
            ax1.plot(x, y, '-', color='black')
            ax1.annotate(str('%.1f' % self.mdl[i].get_col('mdot')[-1]), xy=(x[-1], y[-1]), textcoords='data',
                         horizontalalignment='left',
                         verticalalignment='bottom'
                         )

            if not clean:
                ax1.text(0.9, 0.1, 'blablabla', style='italic',
                        bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center',
                        verticalalignment='center', transform=ax1.transAxes)

            if sonic and v_n2 == 'u':
                u_s = self.mdl[i].get_sonic_u()
                ax1.plot(x, u_s, '-', color='gray')

                xc, yc = Math.interpolated_intercept(x,y, u_s)
                # print('Sonic r: {} | Sonic u: {} | {}'.format( np.float(xc),  np.float(yc), len(xc)))
                plt.plot(xc, yc, 'X', color='red')

            if v_n2 == 'kappa':
                k_edd = 10**Physics.edd_opacity(self.mdl[i].get_col('xm')[-1],
                                            self.mdl[i].get_col('l')[-1])
                ax1.plot(ax1.get_xlim(), [k_edd, k_edd], color='black', label='Model: {}, k_edd: {}'.format(i, k_edd))

            if v_n2 == 'Pg/P_total':
                plt.axhline(y=0.15, color='black')


            # tp, up = get_m_r_envelope(self.mdl[i])
            # if tp != None:
            #     ax1.plot(tp, up, 'X', color='black')

        for i in range(len(self.set_sm_files2)):

            # inflection_point(self.mdl[i])


            x =      self.mdl2[i].get_col(v_n1)
            y      = self.mdl2[i].get_col(v_n2)          # simpler syntaxis
            label1 = self.mdl2[i].get_col(var_for_label1)[-1]
            label2 = self.mdl2[i].get_col(var_for_label2)[-1]

            # x = np.log(x)
            # y = np.log(y)

            # y = np.gradient(y,x)

            # x = np.log(x)
            # y = np.log(y)

            # ax1.set_xlim(x[-100], x[-1])
            # ax1.set_ylim(x[-100], x[-1])

            if v_n2 == 'kappa':
                y = 10**y

            print('\t __Core H: {} , core He: {} File: {}'.
                  format(self.mdl[i].get_col('H')[0], self.mdl[i].get_col('He4')[0], self.set_sm_files[i]))

            lbl = '{}:{} , {}:{}'.format(var_for_label1,'%.2f' % label1,var_for_label2,'%.2f' % label2)

            #ax1.plot(x,  y,  '-',   color='C' + str(Math.get_0_to_max([i], 9)[i]), label=lbl)
            #ax1.plot(x, y, 'x', color='C' + str(Math.get_0_to_max([i], 9)[i]))
            #ax1.plot(x, y, '.', color='C' + str(Math.get_0_to_max([i], 9)[i]))
            #ax1.plot(x[-1], y[-1], 'x',   color='C' + str(Math.get_0_to_max([i], 9)[i]))
            ax1.plot(x, y, '--', color='black')
            ax1.annotate(str('%.1f' % self.mdl2[i].get_col('mdot')[-1]), xy=(x[-1], y[-1]), textcoords='data',
                         horizontalalignment='left',
                         verticalalignment='bottom'
                         )

            if not clean:
                ax1.text(0.9, 0.1, 'blablabla', style='italic',
                        bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center',
                        verticalalignment='center', transform=ax1.transAxes)

            if sonic and v_n2 == 'u':
                u_s = self.mdl2[i].get_sonic_u()
                ax1.plot(x, u_s, '-', color='gray')

                xc, yc = Math.interpolated_intercept(x,y, u_s)
                # print('Sonic r: {} | Sonic u: {} | {}'.format( np.float(xc),  np.float(yc), len(xc)))
                plt.plot(xc, yc, 'X', color='red')

            if v_n2 == 'kappa':
                k_edd = 10**Physics.edd_opacity(self.mdl2[i].get_col('xm')[-1],
                                            self.mdl2[i].get_col('l')[-1])
                ax1.plot(ax1.get_xlim(), [k_edd, k_edd], color='black', label='Model: {}, k_edd: {}'.format(i, k_edd))

            if v_n2 == 'Pg/P_total':
                plt.axhline(y=0.15, color='black')


            # tp, up = get_m_r_envelope(self.mdl[i])
            # if tp != None:
            #     ax1.plot(tp, up, 'X', color='black')

        ax1.set_xlabel(Labels.lbls(v_n1), fontsize=fsz)
        ax1.set_ylabel(Labels.lbls(v_n2), fontsize=fsz)

        ax1.tick_params(labelsize=fsz)

        ax1.minorticks_on()

        # ax1.grid(which='both')
        # ax1.grid(which='minor', alpha=0.2)
        # ax1.grid(which='major', alpha=0.2)

        # ax1.set_xlim(0.78, 0.92)
        # ax1.set_xlim(4.0, 6.2)
        if not clean:
            ax1.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
        plot_name = self.plot_dir + v_n1 + '_vs_' + v_n2 + '_profile.pdf'
        # plt.savefig(plot_name)
        plt.show()

    def dxdy_profile(self, dv_n1, v_n2, var_for_label1, var_for_label2, sonic = False, clean=False):

        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        tlt = v_n2 + '(' + dv_n1 + ') profile'
        # plt.title(tlt)

        # dx/dydx

        def diff(x, y):
            
            if len(x) != len(y): raise ValueError('ERROR')
            dx = []
            for i in range(len(x)-1):
                dx = np.append(dx, (x[i+1]-x[i]))
            
            res = []
            for i in range(len(y)-1):
                dy = np.append(y, (y[i+1]-y[i]))
            
                res = np.append(res, dx[i]/dy[i])
            
            return res


        for i in range(len(self.set_sm_files)):


            x = self.mdl[i].get_col(dv_n1)
            y = self.mdl[i].get_col(v_n2)  # simpler syntaxis

            dydx = diff(y,x)

            # for velocity: dydx = y[:-1]*diff(y,x)



            # dydx = diff(y, y)

            dx = x[:-1]

            label1 = self.mdl[i].get_col(var_for_label1)[-1]
            label2 = self.mdl[i].get_col(var_for_label2)[-1]

            if v_n2 == 'kappa':
                dydx = 10 ** dydx

            print('\t __Core H: {} , core He: {} File: {}'.
                  format(self.mdl[i].get_col('H')[0], self.mdl[i].get_col('He4')[0], self.set_sm_files[i]))

            lbl = '{}:{} , {}:{}'.format(var_for_label1, '%.2f' % label1, var_for_label2, '%.2f' % label2)

            ax1.plot(dx, dydx, '-', color='C' + str(Math.get_0_to_max([i], 9)[i]), label=lbl)
            ax1.plot(dx, dydx, '.', color='C' + str(Math.get_0_to_max([i], 9)[i]))
            ax1.plot(dx[-1], dydx[-1], 'x', color='C' + str(Math.get_0_to_max([i], 9)[i]))

            ax1.annotate(str('%.1f' % self.mdl[i].get_col('mdot')[-1]), xy=(dx[-1], dydx[-1]), textcoords='data')

            if sonic and v_n2 == 'u':
                u_s = self.mdl[i].get_sonic_u()
                ax1.plot(dx, u_s, '-', color='black')

                xc, yc = Math.interpolated_intercept(dx, dydx, u_s)
                # print('Sonic r: {} | Sonic u: {} | {}'.format( np.float(xc),  np.float(yc), len(xc)))
                plt.plot(xc, yc, 'X', color='red')

            if v_n2 == 'kappa':
                k_edd = 10 ** Physics.edd_opacity(self.mdl[i].get_col('xm')[-1],
                                                  self.mdl[i].get_col('l')[-1])
                ax1.plot(ax1.get_xlim(), [k_edd, k_edd], color='black', label='Model: {}, k_edd: {}'.format(i, k_edd))

            if v_n2 == 'Pg/P_total':
                plt.axhline(y=0.15, color='black')

        ax1.set_xlabel(Labels.lbls(dv_n1))
        ax1.set_ylabel(Labels.lbls(v_n2))

        ax1.grid(which='both')
        ax1.grid(which='minor', alpha=0.2)
        ax1.grid(which='major', alpha=0.2)

        # ax1.set_xlim(0.78, 0.92)
        # ax1.set_xlim(4.0, 6.2)
        if not clean:
            ax1.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
        plot_name = self.plot_dir + dv_n1 + '_vs_' + v_n2 + '_profile.pdf'
        # plt.savefig(plot_name)
        plt.show()

    def xyy_profile(self, v_n1, v_n2, v_n3, var_for_label1, var_for_label2, var_for_label3, edd_kappa = True, clean = False):

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

        tlt ='M:10, Z:0.008'
        plt.title(tlt)

        ax1.set_xlabel(Labels.lbls(v_n1))
        # Make the y-axis label, ticks and tick labels match the line color.
        ax1.set_ylabel(Labels.lbls(v_n2), color='b')
        ax1.tick_params('y', colors='b')
        ax1.grid()
        ax2 = ax1.twinx()

        for i in range(len(self.set_sm_files)):

            xyy2  = self.mdl[i].get_set_of_cols([v_n1, v_n2, v_n3])
            lbl1 =  self.mdl[i].get_col(var_for_label1)[-1]
            lbl2 =  self.mdl[i].get_col(var_for_label2)[-1]
            lbl3 =  self.mdl[i].get_col(var_for_label3)[-1]

            color = 'C' + str(Math.get_0_to_max([i], 9)[i])


            t_par = Physics.opt_depth_par2(self.mdl[i].get_col('rho'), self.mdl[i].get_col('t'),
                                           self.mdl[i].get_col('r'), self.mdl[i].get_col('u'), self.mdl[i].get_col('kappa'), self.mdl[i].get_col('mu'))

            prameter = 10**(2*(self.mdl[i].get_col('mfp')[-1] - self.mdl[i].get_col('HP')[-1]))

            lbl = '{}:{} , {}:{} , {}:{}, {}:{}, {}:{}'.format(var_for_label1, '%.2f' % lbl1, var_for_label2, '%.2f' % lbl2,
                                                               var_for_label3, '%.2f' % lbl3,'AnisPar', '%.2f' % (prameter),
                                                 't_par', '%.2f' % t_par)



            ax1.plot(xyy2[:, 0],  xyy2[:, 1],  '-', color='blue', label=lbl)
            ax1.plot(xyy2[-1, 0], xyy2[-1, 1], 'x', color=color)
            ax1.annotate(str('%.2f' % lbl1), xy=(xyy2[-1, 0], xyy2[-1, 1]), textcoords='data')
            # ax2.annotate(str('%.2f' % lbl1), xy=(xyy2[-1, 0], xyy2[-1, 1]), textcoords='data')

            if edd_kappa and v_n3 == 'kappa':
                k_edd = Physics.edd_opacity(self.mdl[i].get_col('xm')[-1],
                                            self.mdl[i].get_col('l')[-1], )
                ax2.plot(ax1.get_xlim(), [k_edd, k_edd], color='black', label='Model: {}, k_edd: {}'.format(i, k_edd))

            ax2.plot(xyy2[:, 0],  xyy2[:, 2], '--', color='red')
            ax2.plot(xyy2[-1, 0], xyy2[-1, 2], 'o', color=color)

            if v_n2 == 'Pg/P_total':
                ax1.plot(ax1.get_xlim(), [0.15, 0.15], color='orange')
            if v_n3 == 'Pg/P_total':
                ax2.plot(ax1.get_xlim(), [0.15, 0.15], color='orange')


            if v_n2 == 'u':
                u_s = self.mdl[i].get_sonic_u()
                ax1.plot(xyy2[:, 0], u_s, color = 'gray')
            if v_n3 == 'u':
                u_s = self.mdl[i].get_sonic_u()
                ax2.plot(xyy2[:, 0], u_s, color = 'gray')

            if v_n2 == 'L/Ledd':
                ax1.plot(ax1.get_xlim(), [1.0, 1.0], color='black')
            if v_n3 == 'L/Ledd':
                ax2.plot(ax1.get_xlim(), [1.0, 1.0], color='black')


        # ax1.text(0.3, 0.9, lbl[i], style='italic',
        #          bbox={'facecolor': 'C' + str(i + 1), 'alpha': 0.5, 'pad': 5}, horizontalalignment='center',
        #          verticalalignment='center', transform=ax.transAxes)

        ax2.set_ylabel(Labels.lbls(v_n3), color='r')
        ax2.tick_params('y', colors='r')



        plt.title(tlt, loc='left')
        fig.tight_layout()
        if not clean:
            ax1.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
        plot_name = self.plot_dir  + 'xyy_profile.pdf'
        plt.savefig(plot_name)



        plt.show()

    def mdot_check(self):
        array = []
        for j in range(len(self.set_plot_files)):
            plfl = Read_Plot_file.from_file(self.set_plot_files[j])

            imx = Math.find_nearest_index(plfl.y_c, plfl.y_c.max())

            for i in range(10):
                ind = Math.find_nearest_index(plfl.y_c, (i / 10))
                yc = plfl.y_c[ind]
                mdot = plfl.mdot_[ind]
                l = plfl.l_[ind]

                mdot_prescr = Physics.l_mdot_prescriptions(l, 10**0.02, 'yoon')

                in_mass = self.set_plot_files[j].split('/')[-1].split('ev')[0]

                array = np.append(array, [in_mass, yc, l, mdot, mdot_prescr, np.abs(mdot-mdot_prescr)])
                print('\t__Mdots: Model: {} Mdot {}, presc: {}, diff: {}'
                      .format(self.set_plot_files[j].split('/')[-1], mdot, mdot_prescr, np.abs(mdot - mdot_prescr)))

                print('a')

        # array_sort = np.sort(array.view('i8, f8, f8, f8, f8, f8'), order=['f0'], axis=0).view(np.float)
        array_shaped = np.reshape(array, (np.int(len(array)/6), 6))
        print(array_shaped)

    def time_analysis(self, percent_of_lifetime, yc_steps = 10):


        for j in range(len(self.set_plot_files)):

            plfl = Read_Plot_file.from_file(self.set_plot_files[j])


            imx = Math.find_nearest_index(plfl.y_c, plfl.y_c.max())
            time = plfl.time - plfl.time[0]
            time_max = time.max()/1000000

            yc_vals = []
            time_yc_vals = []
            for i in range(yc_steps):
                ind = Math.find_nearest_index(plfl.y_c, (i / yc_steps))
                yc_vals = np.append(yc_vals, plfl.y_c[ind])
                time_yc_vals = np.append(time_yc_vals, time[ind]/1000000)

            time_yc_vals, yc_vals = Math.x_y_z_sort(time_yc_vals, yc_vals)

            yc_t_req = interpolate.InterpolatedUnivariateSpline(time_yc_vals, yc_vals)(percent_of_lifetime*time_max/100)

            plt.plot(time_yc_vals, yc_vals, '-', color='black')
            plt.plot(percent_of_lifetime*time_max/100, yc_t_req, 'x', color='red', label='Yc:{} Time:{} Myr'
                     .format("%.2f"%yc_t_req, "%.2f" % (percent_of_lifetime*time_max/100)))
            plt.ylabel(Labels.lbls('Yc'))
            plt.xlabel(Labels.lbls('time'))
            plt.grid()
            plt.legend()
        plt.show()

    def xy_last_points(self, v_n1, v_n2, v_lbl1, num_pol_fit = True):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        # nums = Treat_Numercials(self.num_files)  # Surface Temp as a x coordinate
        # res = nums.get_x_y_of_all_numericals('sp', 'r', 'l', 'mdot', 'color')
        x = []
        y = []
        for i in range(len(self.set_sm_files)):
            x = np.append(x, self.mdl[i].get_cond_value(v_n1, 'sp') )
            y = np.append(y, self.mdl[i].get_cond_value(v_n2, 'sp') )

            lbl1 = self.mdl[i].get_cond_value(v_lbl1, 'sp')
            # print(x, y, lbl1)

            plt.plot(x[i], y[i], marker='.', color='C' + str(Math.get_0_to_max([i],9)[i]), ls='', label='{}:{} , {}:{} , {}:{}'
                     .format(v_n1, "%.2f" % x[i], v_n2, "%.2f" % y[i], v_lbl1, "%.2f" % lbl1))  # plot color dots)))
            ax.annotate(str("%.2f" % lbl1), xy=(x[i], y[i]), textcoords='data')

        plt.plot(x,y,'--', color='black')

        if num_pol_fit:
            fit = np.polyfit(x, y, 4)  # fit = set of coeddicients (highest first)
            f = np.poly1d(fit)

            # print('Equation:', f.coefficients)
            fit_x_coord = np.mgrid[(x.min()):(x.max()):100j]
            plt.plot(fit_x_coord, f(fit_x_coord), '--', color='black', label='Model_fit')


        name = self.output_dir+'{}_{}_dependance.pdf'.format(v_n2,v_n1)
        plt.title('{} = f({}) plot'.format(v_n2,v_n1))
        plt.xlabel(Labels.lbls(v_n1))
        plt.ylabel(Labels.lbls(v_n2))
        plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
        plt.grid()
        plt.savefig(name)

        plt.show()

    def sp_xy_last_points(self, v_n1, v_n2, v_lbl1, num_pol_fit = 0):



        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        # nums = Treat_Numercials(self.num_files)  # Surface Temp as a x coordinate
        # res = nums.get_x_y_of_all_numericals('sp', 'r', 'l', 'mdot', 'color')
        x = []
        y = []
        yc =[]
        xm = []
        for i in range(len(self.set_sp_files)):

            x = np.append(x, self.spmdl[i].get_crit_value(v_n1) )
            y = np.append(y, self.spmdl[i].get_crit_value(v_n2) )
            yc = np.append(yc, self.spmdl[i].get_crit_value('Yc'))
            xm = np.append(xm, self.spmdl[i].get_crit_value('m'))

            lbl1 = self.spmdl[i].get_crit_value(v_lbl1)

            # print(x, y, lbl1) label='{} | {}:{} , {}:{} , {}:{}, Yc: {}'
            #          .format(i, v_n1, "%.2f" % x[i], v_n2, "%.2f" % y[i], v_lbl1, "%.2f" % lbl1, yc[i])

            plt.plot(x[i], y[i], marker='.', color='black', ls='', label=lbl1)  # plot color dots)))
            ax.annotate(str("%.2f" % xm[i]), xy=(x[i], y[i]), textcoords='data')

            # "%.2f" % yc[i]

        # x_pol, y_pol = Math.fit_polynomial(x, y, num_pol_fit, 500)
        # plt.plot(x_pol, y_pol, '--', color='black')

        if v_n1 == 'm' and v_n2 == 'l':
            plt.plot(x, Physics.m_to_l_langer(np.log10(x)), '-.', color='gray', label='Langer, 1987')


        # plt.plot(x, y, '-', color='gray')


        name = self.output_dir+'{}_{}_dependance.pdf'.format(v_n2,v_n1)
        plt.title('{} = f({}) plot'.format(v_n2, v_n1))
        plt.xlabel(Labels.lbls(v_n1))
        plt.ylabel(Labels.lbls(v_n2))
        plt.legend(bbox_to_anchor=(1, 0), loc='lower right', ncol=1)
        plt.savefig(name)
        plt.grid()

        plt.show()

    def table(self):
        print(
            '| Mdot'
            '\t| Mass'
            '\t| R/Rs '
            '\t| L/Ls'
            '\t| kappa'
            '\t| l(Rho)'
            '\t| Temp'
            '\t| mfp  '
            '\t| vel '
            '\t| gamma'
            '\t| tpar '
            '\t| HP '
            '\t| tau '
            '\t|')
        for mdl in self.mdl:

            mdl.get_par_table('l')


        # print(
        #     '| Mdot'
        #     '\t| Mass'
        #     '\t| R/Rs '
        #     '\t| L/Ls'
        #     '\t| kappa'
        #     '\t| l(Rho)'
        #     '\t| Temp'
        #     '\t| mfp  '
        #     '\t| vel '
        #     '\t| gamma'
        #     '\t| tpar '
        #     '\t| HP '
        #     '\t| tau '
        #     '\t|')
        # for mdl in self.mdl:
        #
        #     mdl.get_par_table('l')

    # --- METHODS THAT DO REQUIRE OPAL TABLES ---
    def sp_get_r_lt_table2(self, v_n, l_or_lm, plot=True, ref_t_llm_vrho=np.empty([])):
        '''

        :param l_or_lm:
        :param depth:
        :param plot:
        :param t_llm_vrho:
        :return:
        '''
        if not ref_t_llm_vrho.any():
            print('\t__ No *ref_t_llm_vrho* is provided. Loading {} interp. opacity table.'.format(self.set_metal))
            t_k_rho = Save_Load_tables.load_table('t_k_rho', 't', 'k', 'rho', self.set_metal, self.output_dir)
            table = Physics.t_kap_rho_to_t_llm_rho(t_k_rho, l_or_lm)
        else:
            table = ref_t_llm_vrho

        t_ref = table[0, 1:]
        llm_ref=table[1:, 0]


        '''=======================================ESTABLISHING=LIMITS================================================'''

        t_mins = []
        t_maxs = []

        for i in range(len(self.set_sp_files)): # as every sp.file has different number of t - do it one by one

            t = self.spmdl[i].get_sonic_cols('t')                     # Sonic
            t = np.append(t, self.spmdl[i].get_crit_value('t'))       # critical

            t_mins = np.append(t_mins, t.min())
            t_maxs = np.append(t_maxs, t.max())

        t_min = t_mins.max()
        t_max = t_maxs.min()

        print('\t__ SP files t limits: ({}, {})'.format(t_min, t_max))
        print('\t__REF table t limits: ({}, {})'.format(t_ref.min(), t_ref.max()))

        it1 = Math.find_nearest_index(t_ref, t_min)       # Lower t limit index in t_ref
        it2 = Math.find_nearest_index(t_ref, t_max)       # Upper t limit index in t_ref
        t_grid = t_ref[it1:it2]

        print('\t__     Final t limits: ({}, {}) with {} elements'.format(t_ref[it1], t_ref[it2], len(t_grid)))


        '''=========================INTERPOLATING=ALONG=T=ROW=TO=HAVE=EQUAL=N=OF=ENTRIES============================='''

        llm_r_rows = np.empty(1 + len(t_ref[it1:it2]))

        for i in range(len(self.set_sp_files)):

            if l_or_lm == 'l':
                llm = self.spmdl[i].get_crit_value('l')
            else:
                llm = Physics.loglm(self.spmdl[i].get_crit_value('l'), self.spmdl[i].get_crit_value('m'), False)


            r = self.spmdl[i].get_sonic_cols(v_n)                     # get sonic
            if v_n == 'r': r = np.append(r, self.spmdl[i].get_crit_value('r'))       # get Critical

            t = self.spmdl[i].get_sonic_cols('t')                                     # Sonic
            if v_n == 'r': t = np.append(t, self.spmdl[i].get_crit_value('t'))        # critical

            r_t = []        # Dictionary for sorting
            for i in range(len(r)):
                r_t = np.append(r_t, [r[i], t[i]])

            r_t_sort = np.sort(r_t.view('float64, float64'), order=['f1'], axis=0).view(np.float)
            r_t_reshaped = np.reshape(r_t_sort, (len(r), 2)) # insure that the t values are rising along the t_r arr.

            r_sort = r_t_reshaped[:,0]
            t_sort = r_t_reshaped[:,1]

            f = interpolate.InterpolatedUnivariateSpline(t_sort, r_sort)

            l_r_row = np.array([llm])
            l_r_row = np.append(l_r_row, f(t_grid))
            llm_r_rows = np.vstack((llm_r_rows, l_r_row))

        llm_r_rows = np.delete(llm_r_rows, 0, 0)

        llm_r_rows_sort = llm_r_rows[llm_r_rows[:,0].argsort()] # UNTESTED sorting function

        t_llm_r = Math.combine(t_grid, llm_r_rows_sort[:,0], llm_r_rows_sort[:,1:]) # intermediate result

        '''======================================INTERPOLATING=EVERY=COLUMN=========================================='''


        l      = t_llm_r[1:, 0]
        t      = t_llm_r[0, 1:]
        r      = t_llm_r[1:, 1:]
        il1    = Math.find_nearest_index(llm_ref, l.min())
        il2    = Math.find_nearest_index(llm_ref, l.max())

        print('\t__ SP files l limits: ({}, {})'.format(l.min(), l.max()))
        print('\t__REF table t limits: ({}, {})'.format(llm_ref.min(), llm_ref.max()))

        l_grid = llm_ref[il1:il2]

        print('\t__     Final l limits: ({}, {}) with {} elements'.format(llm_ref[il1], llm_ref[il2], len(l_grid)))

        r_final = np.empty((len(l_grid),len(t)))
        for i in range(len(t)):
            f = interpolate.InterpolatedUnivariateSpline(l, r[:, i])
            r_final[:, i] = f(l_grid)

        t_llm_r = Math.combine(t, l_grid, r_final)

        if plot:
            plt.figure()
            # ax = fig.add_subplot(1, 1, 1)
            plt.xlim(t_llm_r[0,1:].min(), t_llm_r[0,1:].max())
            plt.ylim(t_llm_r[1:,0].min(), t_llm_r[1:,0].max())
            plt.ylabel(Labels.lbls(l_or_lm))
            plt.xlabel(Labels.lbls('ts'))

            levels = []
            if v_n == 'k':   levels = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95] # FOR log Kappa
            if v_n == 'rho': levels = [-10, -9.5, -9, -8.5, -8, -7.5, -7, -6.5, -6, -5.5,  -5, -4.5, -4]
            if v_n == 'r':   levels = [0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7,
                                       1.75, 1.8, 1.85, 1.9, 1.95, 2.0, 2.05, 2.10, 2.15, 2.20]

            contour_filled = plt.contourf(t_llm_r[0, 1:], t_llm_r[1:, 0], t_llm_r[1:,1:], levels, cmap=plt.get_cmap('RdYlBu_r'))
            plt.colorbar(contour_filled, label=Labels.lbls(v_n))
            contour = plt.contour(t_llm_r[0, 1:], t_llm_r[1:, 0], t_llm_r[1:,1:], levels, colors='k')
            plt.clabel(contour, colors='k', fmt='%2.2f', fontsize=12)
            plt.title('SONIC HR DIAGRAM')

            # plt.ylabel(l_or_lm)
            plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
            # plt.savefig(name)
            plt.show()

        return t_llm_r

    def sp_get_r_lt_table(self, l_or_lm, depth = 1000, plot = False, t_llm_vrho = np.empty([])):
        '''

        :param l_or_lm:
        :param t_grid:
        :param l_lm:
        :t_llm_vrho:
        :return:
        '''
        '''===========================INTERPOLATING=EVERY=ROW=TO=HAVE=EQUAL=N=OF=ENTRIES============================='''

        r = np.empty((len(self.set_sp_files), 100))
        t = np.empty((len(self.set_sp_files), 100))
        l_lm = np.empty(len(self.set_sp_files))

        for i in range(len(self.set_sp_files)):
            r_i =  self.spmdl[i].get_sonic_cols('r')
            r_i = np.append(r_i, self.spmdl[i].get_crit_value('r'))

            t_i = self.spmdl[i].get_sonic_cols('t')
            t_i = np.append(t_i, self.spmdl[i].get_crit_value('t'))

            r_i_grid = np.mgrid[r_i.min():r_i.max():100j]
            f = interpolate.InterpolatedUnivariateSpline(r_i, t_i)
            t_i_grid = f(r_i_grid)

            r[i,:] = r_i_grid
            t[i,:] = t_i_grid

            if l_or_lm == 'l':
                l_lm[i] = self.spmdl[i].get_crit_value('l')
            else:
                l_lm[i] = Physics.loglm(self.spmdl[i].get_crit_value('l'), self.spmdl[i].get_crit_value('m'), False)

        '''====================================CREATING=OR=USING=L/T=GRID============================================'''

        if t_llm_vrho.any():

            t__ =  t_llm_vrho[0,1:]
            l_lm_grid_ = t_llm_vrho[1:, 0]
            vrho = t_llm_vrho[1:,1:]

            if l_lm_grid_[0] > l_lm_grid_[-1]:
                raise ValueError('Array l_lm_grid_ must be increasing, now it is from {} to {}'
                                 .format(l_lm_grid_[0], l_lm_grid_[-1]))


            l_lm_1 = l_lm.min()
            i1 = Math.find_nearest_index(l_lm_grid_, l_lm_1)
            l_lm_2 = l_lm.max()
            i2 = Math.find_nearest_index(l_lm_grid_, l_lm_2)

            l_lm_grid = l_lm_grid_[i1:i2]
            vrho = vrho[i1:i2,:]

            t_llm_vrho = Math.combine(t__, l_lm_grid, vrho)

            print('\t__Note: provided l_lm_grid_{} is cropped to {}, with limits: ({}, {})'
                  .format(l_lm_grid_.shape, l_lm_grid.shape, l_lm_grid.min(), l_lm_grid.max()))

        else:
            l_lm_grid = np.mgrid[l_lm.min():l_lm.max():depth*1j]

        '''=======================================INTERPOLATE=2D=T=AND=R============================================='''

        r2 = np.empty(( len(l_lm_grid), len(r[0,:]) ))
        t2 = np.empty(( len(l_lm_grid), len(r[0,:]) ))

        for i in range( len(r[0, :]) ):

            r_l = [] # DICTIONARY
            for j in range(len(r[:,i])): # this wierd thing allows you to create a dictionary that you can sort
               r_l = np.append(r_l, [ r[j,i], l_lm[j] ])

            r_l_  = np.sort(r_l.view('f8, f8'), order=['f1'], axis=0).view(np.float) # sorting dictionary according to l_lm
            r_l__ = np.reshape(r_l_, (len(l_lm), 2))


            r_ = r_l__[:,0] # i-th column, sorted by l_lm
            l_lm_ = r_l__[:,1] # l_lm sorted

            f2 = interpolate.InterpolatedUnivariateSpline(l_lm_, r_) # column by column it goes
            r2[:,i] = f2(l_lm_grid)


            # --- --- --- T --- --- ---

        for i in range( len(t[0, :]) ):
            t_l_lm = [] # DICTIONARY
            for j in range(len(t[:,i])): # this wierd thing allows you to create a dictionary that you can sort
               t_l_lm = np.append(t_l_lm, [ t[j,i], l_lm[j] ])

            t_l_lm_  = np.sort(t_l_lm.view('f8, f8'), order=['f1'], axis=0).view(np.float) # sorting dictionary according to l_lm
            t_l_lm__ = np.reshape(t_l_lm_, (len(l_lm), 2))


            t_ = t_l_lm__[:,0] # i-th column, sorted by l_lm
            l_lm_ = t_l_lm__[:,1] # l_lm sorted

            f2 = interpolate.InterpolatedUnivariateSpline(l_lm_, t_) # column by column it goes

            # if l_lm_grid.min() < l_lm_.min() or l_lm_grid.max() > l_lm_.max():
            #     raise ValueError('l_lm_grid.min({}) < l_lm_.min({}) or l_lm_grid.max({}) > l_lm_.max({})'
            #                      .format(l_lm_grid.min() , l_lm_.min() , l_lm_grid.max() , l_lm_.max()))

            t2[:,i] = f2(l_lm_grid)


        '''=======================================INTERPOLATE=R=f(L, T)=============================================='''

        # If in every row of const. l_lm, the temp 't' is decreasing monotonically:
        t2_ = t2[:, ::-1]    # so t is from min to max increasing
        r2_ = r2[:, ::-1]    # necessary of Univariate spline

        if not t2_.any() or not r2_.any():
            raise ValueError('Array t2_{} or r2_{} is empty'.format(t2_.shape, r2_.shape))

        def interp_t_l_r(l_1d_arr, t_2d_arr, r_2d_arr, depth = 1000, t_llm_vrho = np.empty([])):

            t1 = t_2d_arr[:, 0].max()
            t2 = t_2d_arr[:,-1].min()



            '''------------------------SETTING-T-GRID-OR-CROPPING-THE-GIVEN-ONE--------------------------------------'''
            if t_llm_vrho.any():

                t_grid_ = t_llm_vrho[0, 1:]
                l_lm_grid_ = t_llm_vrho[1:, 0]
                vrho = t_llm_vrho[1:, 1:]

                if t_grid_[0] > t_grid_[-1]:
                    raise ValueError('Array t_grid_ must be increasing, now it is from {} to {}'
                                     .format(t_grid_[0], t_grid_[-1]))

                i1 = Math.find_nearest_index(t_grid_, t1)
                i2 = Math.find_nearest_index(t_grid_, t2)

                print('\t  t1:{}[i:{}] t2:{}[i:{}] '.format(t1,i1,t2,i2))

                crop_t_grid = t_grid_[i1:i2]
                vrho = vrho[:,i1:i2]

                t_llm_vrho = Math.combine(crop_t_grid, l_lm_grid_, vrho)


                print('\t__Note: provided t_grid{} is cropped to {}, with limits: ({}, {})'
                      .format(t_grid_.shape, crop_t_grid.shape, crop_t_grid.min(), crop_t_grid.max()))
            else:
                crop_t_grid = np.mgrid[t1:t2:depth * 1j]
                t_llm_vrho = np.empty([])

            '''---------------------USING-2*1D-INTERPOLATIONS-TO-GO-FROM-2D_T->1D_T----------------------------------'''


            crop_r = np.empty(( len(l_1d_arr), len(crop_t_grid) ))
            crop_l_lm = []

            for si in range(len(l_1d_arr)):
                # if t1 <= t_2d_arr[si, :].max() and t2 >= t_2d_arr[si, :].min():

                t_row = t_2d_arr[si, :]
                r_row = r_2d_arr[si, :]

                print(t_row.shape, r_row.shape, crop_t_grid.T.shape)

                crop_r[si, :] = Math.interp_row(t_row, r_row, crop_t_grid.T)
                crop_l_lm = np.append(crop_l_lm, l_1d_arr[si])


            extend_crop_l = l_1d_arr # np.mgrid[l_1d_arr.min():l_1d_arr.max():depth*1j]

            extend_r = np.zeros((len(extend_crop_l), len(crop_t_grid)))

            print(crop_l_lm.shape, crop_r.shape, extend_crop_l.shape)

            for si in range(len(extend_r[0,:])):
                extend_r[:,si] = Math.interp_row( crop_l_lm, crop_r[:, si], extend_crop_l )

            return Math.combine(crop_t_grid, extend_crop_l, extend_r), t_llm_vrho

        t_l_or_lm_r, t_llm_vrho = interp_t_l_r(l_lm_grid, t2_, r2_, 1000, t_llm_vrho)

        if plot:
            plt.figure()
            # ax = fig.add_subplot(1, 1, 1)
            plt.xlim(t_l_or_lm_r[0,1:].min(), t_l_or_lm_r[0,1:].max())
            plt.ylim(t_l_or_lm_r[1:,0].min(), t_l_or_lm_r[1:,0].max())
            plt.ylabel(Labels.lbls(l_or_lm))
            plt.xlabel(Labels.lbls('ts'))
            levels = [0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8]
            contour_filled = plt.contourf(t_l_or_lm_r[0, 1:], t_l_or_lm_r[1:, 0], t_l_or_lm_r[1:,1:], levels, cmap=plt.get_cmap('RdYlBu_r'))
            plt.colorbar(contour_filled, label=Labels.lbls('r'))
            contour = plt.contour(t_l_or_lm_r[0, 1:], t_l_or_lm_r[1:, 0], t_l_or_lm_r[1:,1:], levels, colors='k')
            plt.clabel(contour, colors='k', fmt='%2.2f', fontsize=12)
            plt.title('SONIC HR DIAGRAM')

            # plt.ylabel(l_or_lm)
            plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
            # plt.savefig(name)
            plt.show()

        return t_l_or_lm_r, t_llm_vrho

    def plot_t_rho_kappa(self, bump, var_for_label1, var_for_label2,  n_int_edd = 1000, plot_edd = False):
        # self.int_edd = self.tbl_anlom_OPAL_table(self.op_name, 1, n_int, load_lim_cases)

        # t_k_rho = self.opal.interp_opal_table(t1, t2, rho1, rho2)

        t_rho_k = Save_Load_tables.load_table('t_rho_k','t','rho','k', self.set_metal, bump, self.output_dir)

        # t_rho_k = Math.extrapolate(t_rho_k,None,None,10,None,500,2)

        t      = t_rho_k[0, 1:]  # x
        rho    = t_rho_k[1:, 0]  # y
        kappa  = 10 ** t_rho_k[1:, 1:] # z   # as initial t_rho_k has log(k)
        t_rho_k = Math.combine(t, rho, kappa)

        fig = plt.figure(figsize=plt.figaspect(0.8))
        ax = fig.add_subplot(111)  # , projection='3d'
        # z = Get_Z.z(self.set_opal_used.split('/')[-1])
        PlotBackground.plot_color_background(ax, t_rho_k, 't', 'rho', 'k', self.set_metal, 'z:{}'.format(Get_Z.z(self.set_metal)))
        # Plot.plot_edd_kappa(ax, t, self.mdl, self.set_metal, 1000)

        # plt.figure()
        # levels = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        # pl.xlim(t.min(), t.max())
        # pl.ylim(rho.min(), rho.max())
        # contour_filled = plt.contourf(t, rho, 10 ** (kappa), levels, cmap=plt.get_cmap('RdYlBu_r'))
        # plt.colorbar(contour_filled)
        # contour = plt.contour(t, rho, 10 ** (kappa), levels, colors='k')
        # plt.clabel(contour, colors='k', fmt='%2.1f', fontsize=12)
        # plt.title('OPACITY PLOT')
        # plt.xlabel(Labels.lbls('t'))
        # plt.ylabel(Labels.lbls('rho'))

        # ------------------------EDDINGTON-----------------------------------
        Table_Analyze.plot_k_vs_t = False  # there is no need to plot just one kappa in the range of availability

        if plot_edd:  # n_model_for_edd_k.any():
            clas_table_anal = Table_Analyze(Files.get_opal(self.set_metal), 1000, False, self.output_dir, self.plot_dir)

            for i in range(len(self.set_sm_files)):  # self.nmdls
                mdl_m = self.mdl[i].get_cond_value('xm', 'sp')
                mdl_l = self.mdl[i].get_cond_value('l',  'sp')

                k_edd = Physics.edd_opacity(mdl_m, mdl_l)

                n_model_for_edd_k = clas_table_anal.interp_for_single_k(t.min(), t.max(), n_int_edd, k_edd)
                x = n_model_for_edd_k[0, :]
                y = n_model_for_edd_k[1, :]
                color = 'black'
                # lbl = 'Model:{}, k_edd:{}'.format(i, '%.2f' % 10 ** k_edd)
                plt.plot(x, y, '-.', color=color) #, label=lbl)
                plt.plot(x[-1], y[-1], 'x', color=color)

        Table_Analyze.plot_k_vs_t = True
        # ----------------------DENSITY----------------------------------------

        for i in range(len(self.set_sm_files)):
            res  = self.mdl[i].get_set_of_cols(['t', 'rho', var_for_label1, var_for_label2])
            xm   = self.mdl[i].get_cond_value('xm', 'sp')
            mdot = self.mdl[i].get_cond_value('mdot', 'sp')

            rho_sp = self.mdl[i].get_cond_value('rho', 'sp')
            t_sp = self.mdl[i].get_cond_value('t', 'sp')

            plt.plot(t_sp, rho_sp, 'x', color='C' + str(Math.get_0_to_max([i], 9)[i]))

            lbl = '{} , {}:{} , {}:{}'.format(i, 'M', '%.2f' % xm, 'Mdot', '%.2f' % mdot)
            plt.plot(res[:, 0], res[:, 1], '-', color='C' + str(Math.get_0_to_max([i], 9)[i]), label=lbl)
            plt.plot(res[-1, 0], res[-1, 1], 'x', color='C' + str(Math.get_0_to_max([i], 9)[i]))
            plt.annotate(str('%.2f' % res[0, 2]), xy=(res[-1, 0], res[-1, 1]), textcoords='data')

        plt.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
        name = self.plot_dir + 't_rho_kappa.pdf'
        plt.savefig(name)
        plt.show()

    def plot_t_mdot_lm(self, v_lbl, r_s = 1., lim_t1_mdl = 5.2, lim_t2_mdl = None):

        t_rho_k = Save_Load_tables.load_table('t_rho_k', 't', 'rho', 'k', self.set_metal, self.output_dir)

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

        for i in range(len(self.set_sm_files)):
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

    # @staticmethod
    # def llm_r_formulas(x, yc, z, l_or_lm):
    #
    #     def crop_x(x, lim_arr):
    #         if len(lim_arr) > 1:
    #             raise ValueError('Size of a lim_arr must be 2 [min, max], but provided {}'.format(lim_arr))
    #         return x[Math.find_nearest_index(x, lim_arr[0]) : Math.find_nearest_index(x, lim_arr[-1])]
    #
    #     #=======================================EMPIRICAL=FUNCTIONS=AND=LIMITS==========================================
    #     if z == 0.02 or z == 'gal':
    #         if yc == 1. or yc == 10 or yc == 'zams':
    #             if l_or_lm == 'l':
    #                 # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    #                 l_lim = []    # put here the l limits for which the polynomial fit is accurate
    #                 x = crop_x(x, l_lim)
    #                 return None # Put here the polinomoal (x-l, y-r)
    #                 # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    #             else:
    #                 # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    #                 lm_lim= []
    #                 x = crop_x(x, lm_lim)
    #                 return None
    #                 # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    #
    #     if z == 0.008 or z == 'lmc':
    #         if yc == 1. or yc == 10 or yc == 'zams':
    #             if l_or_lm == 'l':
    #                 # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    #                 l_lim = []    # put here the l limits for which the polynomial fit is accurate
    #                 x = crop_x(x, l_lim)
    #                 return None # Put here the polinomoal (x-l, y-r)
    #                 # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    #             else:
    #                 # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    #                 lm_lim= []
    #                 x = crop_x(x, lm_lim)
    #                 return None
    #                 # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    #
    #     #====================================================END========================================================

    # def t_llm_cr_sp(self, t_k_rho, yc_val, opal_used):
    #     kap = t_k_rho[1:, 0]
    #     t = t_k_rho[0, 1:]
    #     rho2d = t_k_rho[1:, 1:]
    #     lm_op = Physics.logk_loglm(kap, 1)
    #
    #     yc_lm_l = Save_Load_tables.load_table('yc_lm_l', 'yc', 'lm', 'l', opal_used)
    #     yc = yc_lm_l[0, 1:]
    #     lm_sp = yc_lm_l[1:, 0]
    #     l2d_sp = yc_lm_l[1:, 1:]
    #
    #     yc_lm_r = Save_Load_tables.load_table('yc_lm_r', 'yc', 'lm', 'r', opal_used)
    #     lm_sp = yc_lm_r[1:, 0]
    #     r2d_sp = yc_lm_r[1:, 1:]
    #
    #     lm1 = np.array([lm_op.min(), lm_sp.min()]).max()
    #     lm2 = np.array([lm_op.max(), lm_sp.max()]).min()
    #
    #     yc_lm_l = Math.crop_2d_table(Math.combine(yc, lm_sp, l2d_sp), None, None, lm1, lm2)
    #     yc_lm_r = Math.crop_2d_table(Math.combine(yc, lm_sp, r2d_sp), None, None, lm1, lm2)
    #     t_lm_rho = Math.crop_2d_table(Math.invet_to_ascending_xy(Math.combine(t, lm_op, rho2d)), None, None, lm1, lm2)
    #
    #     if not yc_val in yc:
    #         yc_ind = None
    #         raise ValueError('Yc = {} | not in yc from | {} | for {}'.format(yc_val, 'yc_lm_l', opal_used))
    #     else:
    #         yc_ind = Math.find_nearest_index(yc, yc_val) + 1  # as it starts with zero, which is non physical
    #
    #     t_op = t_lm_rho[0, 1:]
    #     lm_op = t_lm_rho[1:, 0]
    #     rho2d = t_lm_rho[1:, 1:]
    #     lm_sp = yc_lm_l[1:, 0]
    #     l2d = yc_lm_l[1:, yc_ind]
    #     r2d = yc_lm_r[1:, yc_ind]
    #
    #     f = interpolate.UnivariateSpline(lm_sp, l2d)
    #     l_ = f(lm_op)
    #
    #     f = interpolate.UnivariateSpline(lm_sp, r2d)
    #     r_ = f(lm_op)
    #
    #     vrho = Physics.get_vrho(t, rho2d, 2, np.array([1.34]))
    #     m_dot = Physics.vrho_mdot(vrho, r_, 'l')
    #
    #     return Math.combine(t, l_, m_dot),  Math.combine(t, lm_op, m_dot)
    #
    # @staticmethod
    # def t_l_mdot_cr_sp(t_k_rho, yc_val, opal_used):
    #     kap = t_k_rho[1:, 0]
    #     t   = t_k_rho[0, 1:]
    #     rho2d = t_k_rho[1:, 1:]
    #     lm_op = Physics.logk_loglm(kap, 1)
    #
    #     yc_lm_l = Save_Load_tables.load_table('yc_lm_l', 'yc', 'lm', 'l', opal_used)
    #     yc = yc_lm_l[0, 1:]
    #     lm_sp = yc_lm_l[1:, 0]
    #     l2d_sp= yc_lm_l[1:, 1:]
    #
    #     yc_lm_r = Save_Load_tables.load_table('yc_lm_r', 'yc', 'lm', 'r', opal_used)
    #     lm_sp  = yc_lm_r[1:, 0]
    #     r2d_sp = yc_lm_r[1:, 1:]
    #
    #     lm1 = np.array([lm_op.min(), lm_sp.min()]).max()
    #     lm2 = np.array([lm_op.max(), lm_sp.max()]).min()
    #
    #     yc_lm_l = Math.crop_2d_table(Math.combine(yc, lm_sp, l2d_sp), None, None, lm1, lm2)
    #     yc_lm_r = Math.crop_2d_table(Math.combine(yc, lm_sp, r2d_sp), None, None, lm1, lm2)
    #     t_lm_rho= Math.crop_2d_table( Math.invet_to_ascending_xy(Math.combine(t, lm_op, rho2d)), None, None, lm1, lm2)
    #
    #     if not yc_val in yc:
    #         yc_ind = None
    #         raise ValueError('Yc = {} | not in yc from | {} | for {}'.format(yc_val, 'yc_lm_l', opal_used))
    #     else:
    #         yc_ind = Math.find_nearest_index(yc, yc_val) + 1 # as it starts with zero, which is non physical
    #
    #
    #
    #
    #     t_op = t_lm_rho[0, 1:]
    #     lm_op = t_lm_rho[1:, 0]
    #     rho2d = t_lm_rho[1:,1:]
    #     lm_sp = yc_lm_l[1:,0]
    #     l2d = yc_lm_l[1:, yc_ind]
    #     r2d = yc_lm_r[1:, yc_ind]
    #
    #     f = interpolate.UnivariateSpline(lm_sp, l2d)
    #     l_ = f(lm_op)
    #
    #     f = interpolate.UnivariateSpline(lm_sp, r2d)
    #     r_ = f(lm_op)
    #
    #     vrho  = Physics.get_vrho(t, rho2d, 2, np.array([1.34]))
    #     m_dot = Physics.vrho_mdot(vrho, r_, 'l')
    #
    #     return Math.combine(t, l_, m_dot)
    #
    # @staticmethod
    # def t_lm_mdot_cr_sp(t_k_rho, yc_val, opal_used):
    #     kap = t_k_rho[1:, 0]
    #     t = t_k_rho[0, 1:]
    #     rho2d = t_k_rho[1:, 1:]
    #     lm_op = Physics.logk_loglm(kap, 1)
    #
    #     # yc_lm_l = Save_Load_tables.load_table('yc_lm_l', 'yc', 'lm', 'l', opal_used)
    #     # yc = yc_lm_l[0, 1:]
    #     # lm_sp = yc_lm_l[1:, 0]
    #     # l2d_sp = yc_lm_l[1:, 1:]
    #
    #     yc_lm_r = Save_Load_tables.load_table('yc_lm_r', 'yc', 'lm', 'r', opal_used)
    #     yc = yc_lm_r[0, 1:]
    #     lm_sp = yc_lm_r[1:, 0]
    #     r2d_sp = yc_lm_r[1:, 1:]
    #
    #     lm1 = np.array([lm_op.min(), lm_sp.min()]).max()
    #     lm2 = np.array([lm_op.max(), lm_sp.max()]).min()
    #
    #     # yc_lm_l = Math.crop_2d_table(Math.combine(yc, lm_sp, l2d_sp), None, None, lm1, lm2)
    #     yc_lm_r  = Math.crop_2d_table(Math.combine(yc, lm_sp, r2d_sp), None, None, lm1, lm2)
    #     t_lm_rho = Math.crop_2d_table(Math.invet_to_ascending_xy(Math.combine(t, lm_op, rho2d)), None, None, lm1, lm2)
    #
    #     if not yc_val in yc:
    #         yc_ind = None
    #         raise ValueError('Yc = {} | not in yc from | {} | for {}'.format(yc_val, 'yc_lm_l', opal_used))
    #     else:
    #         yc_ind = Math.find_nearest_index(yc, yc_val) + 1  # as it starts with zero, which is non physical
    #
    #
    #
    #
    #     t_op = t_lm_rho[0, 1:]
    #     lm_op = t_lm_rho[1:, 0]
    #     rho2d = t_lm_rho[1:, 1:]
    #     # lm_sp = yc_lm_l[1:, 0]
    #     # l2d = yc_lm_l[1:, yc_ind]
    #     r2d = yc_lm_r[1:, yc_ind]
    #
    #     # f = interpolate.UnivariateSpline(lm_sp, l2d)
    #     # l_ = f(lm_op)
    #
    #     f2 = interpolate.UnivariateSpline(lm_sp, r2d)
    #     r2_ = f2(lm_op)
    #
    #     vrho2 = Physics.get_vrho(t, rho2d, 2, np.array([1.34]))
    #     m_dot2 = Physics.vrho_mdot(vrho2, r2_, 'l')
    #
    #     return Math.combine(t, lm_op, m_dot2)

    # @staticmethod
    # def t_llm_mdot_crit_sp(sp_clss, t_k_rho, l_or_lm, yc, opal_used):
    #     '''
    #     READS the SP files for r_crit and l/lm. Than depending on limits of l/lm in SP and limits in OPAL
    #     selects the availabel range of l/am and uses OPAL values of l/lm as a grid, to interpolate critical R values
    #     :param t_k_rho:
    #     :param l_or_lm:
    #     :param yc:
    #     :param z:
    #     :return:
    #     '''
    #
    #     kap = t_k_rho[1:, 0]
    #     t   = t_k_rho[0, 1:]
    #     rho2d = t_k_rho[1:, 1:]
    #
    #     l_lm_r = []
    #
    #     for i in range(len(sp_clss)):
    #         l    = sp_clss[i].get_crit_value('l')
    #         m    = sp_clss[i].get_crit_value('m')
    #         lm   = Physics.loglm(l, m, False)
    #         r_cr = sp_clss[i].get_crit_value('r')
    #         # t_cr = self.spmdl[i].get_crit_value('t')
    #
    #         l_lm_r = np.append(l_lm_r, [l, lm, r_cr])
    #
    #     if l_or_lm == 'l':
    #         l_lm_r_sorted = np.sort(l_lm_r.view('f8, f8, f8'), order=['f0'], axis=0).view(np.float)
    #         l_lm_r_shaped = np.reshape(l_lm_r_sorted, (len(sp_clss), 3))
    #         llm_emp  = l_lm_r_shaped[:,0]
    #         r_cr_emp = l_lm_r_shaped[:,2]
    #
    #         llm_opal = []
    #         lm_aval = []
    #         if yc >= 0 and yc <= 1.:
    #             lm_aval, llm_opal = SP_file_work.yc_x__to__y__sp(yc, 'lm', 'l', Physics.logk_loglm(kap, True), opal_used, 1)
    #         if yc == 'langer':
    #             llm_opal = Physics.lm_to_l_langer(Physics.logk_loglm(kap, True))
    #         if len(llm_opal) == 0: raise ValueError('Yc = {} is not recognised.'.format(yc))
    #
    #
    #         old_table = Math.combine(t, Physics.logk_loglm(kap, True), rho2d)
    #         sorted_table = Math.invet_to_ascending_xy(old_table)
    #
    #         table_cropped = Math.crop_2d_table(sorted_table, None, None, lm_aval.min(), lm_aval.max())
    #
    #
    #
    #         t_llm_rhp_inv = Math.invet_to_ascending_xy(Math.combine(table_cropped[0,1:], llm_opal, table_cropped[1:,1:]))
    #     else:
    #         l_lm_r_sorted = np.sort(l_lm_r.view('f8, f8, f8'), order=['f1'], axis=0).view(np.float)
    #         l_lm_r_shaped = np.reshape(l_lm_r_sorted, (len(sp_clss), 3))
    #         llm_emp  = l_lm_r_shaped[:,1]
    #         r_cr_emp = l_lm_r_shaped[:,2]
    #
    #         llm_opal = Physics.logk_loglm(kap, True)
    #         t_llm_rhp_inv = Math.invet_to_ascending_xy(Math.combine(t, llm_opal, rho2d))
    #
    #
    #     l_lim1 = np.array([llm_emp.min(), llm_opal.min()]).max()
    #     l_lim2 = np.array([llm_emp.max(), llm_opal.max()]).min()
    #
    #     print('\t__Note: Opal <{}> limits: ({}, {}) | SP files <{}> limits: ({}, {}) \n\t  Selected: ({}, {})'
    #           .format(l_or_lm, "%.2f" % llm_opal.min(), "%.2f" %  llm_opal.max(),
    #                   l_or_lm, "%.2f" %  llm_emp.min(), "%.2f" %  llm_emp.max(),
    #                   "%.2f" %  l_lim1, "%.2f" % l_lim2))
    #
    #
    #
    #     cropped = Math.crop_2d_table(t_llm_rhp_inv, None, None, l_lim1, l_lim2)
    #     l_lm  = cropped[1:, 0]
    #     t     = cropped[0, 1:]
    #     rho2d = cropped[1:,1:]
    #
    #     f = interpolate.UnivariateSpline(llm_emp, r_cr_emp)
    #     r_arr = f(l_lm)
    #
    #     vrho = Physics.get_vrho(t, rho2d, 2, np.array([1.34]))
    #     m_dot = Physics.vrho_mdot(vrho, r_arr, 'l')
    #
    #     return Math.combine(t, l_lm, m_dot)
    #
    #     # else:
    #     #     l_lm_table = Physics.lm_to_l(Physics.logk_loglm(kap, True))
    #     #     l_lm_r_sorted = np.sort(l_lm_r.view('f8, f8'), order=['f1'], axis=0).view(np.float)
    #     #     l_lm_r_shaped = np.reshape(l_lm_r_sorted, (len(self.sp_files), 3))
    #     #
    #     #
    #     #
    #     #
    #     #
    #     #
    #     # if l_or_lm == 'l':  # to account for different limits in mass and luminocity
    #     #     l = Physics.lm_to_l( Physics.logk_loglm(kap, True) )
    #     #     cropped = Math.crop_2d_table(Math.invet_to_ascending_xy(Math.combine(t, l, rho2d)),
    #     #                                  None, None, l_lim[0], l_lim[-1])
    #     #
    #     #     l_lm = cropped[1:, 0]
    #     #     t = cropped[0, 1:]
    #     #     rho2d = cropped[1:, 1:]
    #     #
    #     #     vrho = Physics.get_vrho(t, rho2d, 2)
    #     #     m_dot = Physics.vrho_mdot(vrho, Combine.llm_r_formulas(l_lm, yc, z, l_or_lm), 'l')  # r_s = given by the func
    #     #
    #     # else:
    #     #     lm = Physics.logk_loglm(kap, True)
    #     #     cropped = Math.crop_2d_table(Math.invet_to_ascending_xy(Math.combine(t, lm, rho2d)),
    #     #                                  None, None, lm_lim[0], lm_lim[-1])
    #     #
    #     #
    #     #     l_lm = cropped[1:, 0]
    #     #     t    = cropped[0, 1:]
    #     #     rho2d = cropped[1:, 1:]
    #     #
    #     #     vrho = Physics.get_vrho(t, rho2d, 2)
    #     #     m_dot = Physics.vrho_mdot(vrho, r_lm(l_lm), 'l')  # r_s = given by the func
    #     #
    #     #
    #     # return Math.combine(t, l_lm, m_dot)
    #
    #     # return (40.843) + (-15.943*x) + (1.591*x**2)                    # FROM GREY ATMOSPHERE ESTIMATES
    #
    #     # return -859.098 + 489.056*x - 92.827*x**2 + 5.882*x**3        # FROM SONIC POINT ESTIMATES

    # def plot_t_l_mdot(self, l_or_lm, rs, plot_obs, plot_nums, lim_t1 = None, lim_t2 = None):
    #
    #     # ---------------------LOADING-INTERPOLATED-TABLE---------------------------
    #
    #     t_k_rho = Save_Load_tables.load_table('t_k_rho', 't', 'k', 'rho', self.opal_used, self.output_dir)
    #     t_llm_rho = Physics.t_kap_rho_to_t_llm_rho(t_k_rho, l_or_lm)
    #
    #
    #
    #     #---------------------Getting KAPPA[], T[], RHO2D[]-------------------------
    #
    #     if rs == 0: # here in *t_llm_r* and in *t_llm_rho*  t, and llm are equivalent
    #         t_llm_r = self.sp_get_r_lt_table2('r', l_or_lm, True, t_llm_rho)
    #         t   = t_llm_r[0, 1:]
    #         llm = t_llm_r[1:,0]
    #         rs  = t_llm_r[1:,1:]
    #
    #         t_llm_rho = Math.crop_2d_table(t_llm_rho, t.min(), t.max(), llm.min(), llm.max())
    #         rho = t_llm_rho[1:, 1:]
    #
    #         vrho  = Physics.get_vrho(t, rho, 2) # MU assumed constant!
    #         m_dot = Physics.vrho_mdot(vrho, rs, 'tl')
    #
    #     else:
    #         t = t_llm_rho[0, 1:]
    #         llm = t_llm_rho[1:,0]
    #         rho = t_llm_rho[1:, 1:]
    #
    #         vrho = Physics.get_vrho(t, rho, 2)
    #         m_dot = Physics.vrho_mdot(vrho, rs, '')
    #
    #
    #     #-------------------------------------------POLT-Ts-LM-MODT-COUTUR------------------------------------
    #
    #     name = self.plot_dir + 'rs_lm_minMdot_plot.pdf'
    #
    #     fig = plt.figure()
    #     ax = fig.add_subplot(1, 1, 1)
    #     plt.xlim(t.min(), t.max())
    #     plt.ylim(llm.min(), llm.max())
    #     plt.ylabel(Labels.lbls(l_or_lm))
    #     plt.xlabel(Labels.lbls('ts'))
    #     levels = [-7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2]
    #     contour_filled = plt.contourf(t, llm, m_dot, levels, cmap=plt.get_cmap('RdYlBu_r'))
    #     plt.colorbar(contour_filled, label=Labels.lbls('mdot'))
    #     contour = plt.contour(t, llm, m_dot, levels, colors='k')
    #     plt.clabel(contour, colors='k', fmt='%2.1f', fontsize=12)
    #     plt.title('SONIC HR DIAGRAM')
    #
    #
    #     # plt.ylabel(l_or_lm)
    #     plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
    #     plt.savefig(name)
    #
    #     #--------------------------------------------------PLOT-MINS----------------------------------------------------
    #
    #     # plt.plot(mins[0, :], mins[1, :], '-.', color='red', label='min_Mdot (rs: {} )'.format(r_s_))
    #
    #     #-----------------------------------------------PLOT-OBSERVABLES------------------------------------------------
    #     if plot_obs:
    #         classes = []
    #         classes.append('dum')
    #         x = []
    #         y = []
    #         for star_n in self.obs.stars_n:
    #             xyz = self.obs.get_xyz_from_yz(star_n, l_or_lm, 'mdot', t, llm, m_dot, lim_t1, lim_t2)
    #             if xyz.any():
    #                 x = np.append(x, xyz[0, 0])
    #                 y = np.append(y, xyz[1, 0])
    #                 for i in range(len(xyz[0,:])):
    #                     plt.plot(xyz[0, i], xyz[1, i], marker=self.obs.get_clss_marker(star_n), markersize='9', color=self.obs.get_class_color(star_n), ls='')  # plot color dots)))
    #                     ax.annotate(int(star_n), xy=(xyz[0,i], xyz[1,i]),
    #                                 textcoords='data')  # plot numbers of stars
    #                     if self.obs.get_star_class(star_n) not in classes:
    #                         plt.plot(xyz[0, i], xyz[1, i], marker=self.obs.get_clss_marker(star_n), markersize='9', color=self.obs.get_class_color(star_n), ls='', label='{}'.format(self.obs.get_star_class(star_n)))  # plot color dots)))
    #                         classes.append(self.obs.get_star_class(star_n))
    #
    #         fit = np.polyfit(x, y, 1)  # fit = set of coeddicients (highest first)
    #         f = np.poly1d(fit)
    #         fit_x_coord = np.mgrid[(x.min()-1):(x.max()+1):1000j]
    #         plt.plot(fit_x_coord, f(fit_x_coord), '-.', color='blue')
    #
    #     #--------------------------------------------------_NUMERICALS--------------------------------------------------
    #     if plot_nums:
    #         for i in range(len(self.sm_files)):
    #             ts_llm_mdot = self.mdl[i].get_xyz_from_yz(i, 'sp', l_or_lm, 'mdot', t , llm, m_dot, lim_t1, lim_t2)
    #             # lbl1 = self.mdl[i].get_cond_value(num_var_plot, 'sp')
    #
    #             if ts_llm_mdot.any():
    #                 # lbl = 'i:{}, lm:{}, {}:{}'.format(i, "%.2f" % ts_llm_mdot[2, -1], num_var_plot, "%.2f" % lbl1)
    #                 plt.plot(ts_llm_mdot[0, :], ts_llm_mdot[1,:], marker='x', color='C' + str(Math.get_0_to_max([i],9)[i]), ls='')  # plot color dots)))
    #                 ax.annotate(str("%.2f" % ts_llm_mdot[2, -1]), xy=(ts_llm_mdot[0, -1], ts_llm_mdot[1,-1]), textcoords='data')
    #
    #         for i in range(len(self.sm_files)):
    #             x_coord = self.mdl[i].get_cond_value('t', 'sp')
    #             y_coord = self.mdl[i].get_cond_value(l_or_lm, 'sp')
    #             # lbl1 = self.mdl[i].get_cond_value(num_var_plot, 'sp')
    #             # lbl2 = self.mdl[i].get_cond_value('He4', 'core')
    #
    #             # lbl = 'i:{}, Yc:{}, {}:{}'.format(i, "%.2f" % lbl2, num_var_plot, "%.2f" % lbl1)
    #             plt.plot(x_coord, y_coord, marker='X', color='C' + str(Math.get_0_to_max([i], 9)[i]),
    #                      ls='')  # plot color dots)))
    #             ax.annotate(str(int(i)), xy=(x_coord, y_coord), textcoords='data')
    #
    #     plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
    #     plt.gca().invert_xaxis()
    #     plt.savefig(name)
    #     plt.show()



    # def min_mdot(self, l_or_lm, rs, plot_obs, plot_nums, plot_sp_crits, lim_t1=5.18, lim_t2=None):
    #     # ---------------------LOADING-INTERPOLATED-TABLE---------------------------
    #
    #     t_k_rho = Save_Load_tables.load_table('t_k_rho', 't', 'k', 'rho', self.opal_used, self.output_dir)
    #
    #
    #     l_lim1, l_lim2 = None, None
    #     fig = plt.figure()
    #     ax = fig.add_subplot(1, 1, 1)
    #
    #     if rs == 0:
    #         t_llm_mdot = self.empirical_l_r_crit(t_k_rho, l_or_lm)
    #         t = t_llm_mdot[0, 1:]
    #         l_lm_arr = t_llm_mdot[1:, 0]
    #         m_dot = t_llm_mdot[1:, 1:]
    #
    #         mins = Math.get_mins_in_every_row(t, l_lm_arr, m_dot, 5000, lim_t1, lim_t2)
    #
    #         plt.plot(mins[2, :], mins[1, :], '-', color='black')
    #         ax.fill_between(mins[2, :], mins[1, :], color="lightgray")
    #
    #     else:
    #         t_llm_rho = Physics.t_kap_rho_to_t_llm_rho(t_k_rho, l_or_lm)
    #
    #         t   = t_llm_rho[0,1:]
    #         llm = t_llm_rho[1:,0]
    #         rho = t_llm_rho[1:,1:]
    #
    #         vrho  = Physics.get_vrho(t, rho, 2)       # mu = 1.34 everywhere
    #         m_dot = Physics.vrho_mdot(vrho, rs, '')   # r_s = constant
    #         mins  = Math.get_mins_in_every_row(t, llm, m_dot, 5000, lim_t1, lim_t2)
    #
    #         plt.plot(mins[2, :], mins[1, :], '-', color='black')
    #         ax.fill_between(mins[2, :], mins[1, :], color="lightgray")
    #
    #     if plot_sp_crits:
    #
    #         l = []
    #         m = []
    #         mdot = []
    #
    #         for i in range(len(self.sp_files)):
    #             l   = np.append(l,  self.spmdl[i].get_crit_value('l') )
    #             m   = np.append(m, self.spmdl[i].get_crit_value('m') )
    #             mdot= np.append(mdot, self.spmdl[i].get_crit_value('mdot') )
    #
    #         if l_or_lm == 'l':
    #             llm_sp = l
    #         else:
    #             llm_sp = Physics.loglm(l, m, True)
    #
    #         plt.plot(mdot, llm_sp, '-', color='red')
    #
    #     #         sp_file_1 = self.sp_files[0].split('/')[-1]
    #
    #     if plot_obs:
    #
    #         classes = []
    #         classes.append('dum')
    #         x = []
    #         y = []
    #
    #         from Phys_Math_Labels import Opt_Depth_Analythis
    #
    #         for star_n in self.obs.stars_n:
    #             i=-1
    #             x = np.append(x, self.obs.get_num_par('mdot',  star_n))
    #             y = np.append(y, self.obs.get_num_par(l_or_lm, star_n))
    #             # print(self.obs.get_num_par('mdot',  star_n), self.obs.get_num_par(l_or_lm, star_n))
    #
    #             plt.plot(x[i], y[i], marker=self.obs.get_clss_marker(star_n), markersize='9',
    #                      color=self.obs.get_class_color(star_n), ls='')  # plot color dots)))
    #             # ax.annotate(int(star_n), xy=(x[i], y[i]),
    #             #             textcoords='data')  # plot numbers of stars
    #
    #             v_inf = self.obs.get_num_par('v_inf',  star_n)
    #             tau_cl = Opt_Depth_Analythis(30, v_inf, 1., 1., x[i], 0.20)
    #             tau = tau_cl.anal_eq_b1(1.)
    #
    #             ax.annotate(str(int(tau)), xy=(x[i], y[i]),
    #                         textcoords='data')  # plo
    #
    #             if self.obs.get_star_class(star_n) not in classes:
    #                 plt.plot(x[i], y[i], marker=self.obs.get_clss_marker(star_n), markersize='9',
    #                          color=self.obs.get_class_color(star_n), ls='',
    #                          label='{}'.format(self.obs.get_star_class(star_n)))  # plot color dots)))
    #                 classes.append(self.obs.get_star_class(star_n))
    #
    #
    #         print('\t__PLOT: total stars: {}'.format(len(self.obs.stars_n)))
    #         print(len(x), len(y))
    #
    #         # fit = np.polyfit(x, y, 1)  # fit = set of coeddicients (highest first)
    #         # f = np.poly1d(fit)
    #         # fit_x_coord = np.mgrid[(x.min() - 1):(x.max() + 1):1000j]
    #         # plt.plot(fit_x_coord, f(fit_x_coord), '-.', color='blue')
    #
    #     # --------------------------------------------------NUMERICALS--------------------------------------------------
    #
    #     if plot_nums:
    #         for i in range(len(self.sm_files)):
    #             x_coord = self.mdl[i].get_cond_value('mdot', 'sp')
    #             y_coord = self.mdl[i].get_cond_value(l_or_lm, 'sp')
    #             # lbl1 = self.mdl[i].get_cond_value(i, 'sp')
    #             # lbl2 = self.mdl[i].get_cond_value('He4', 'core')
    #
    #             # lbl = 'i:{}, Yc:{}, {}:{}'.format(i, "%.2f" % lbl2, num_var_plot, "%.2f" % lbl1)
    #             plt.plot(x_coord, y_coord, marker='x', color='C' + str(Math.get_0_to_max([i], 9)[i]),
    #                      ls='')  # plot color dots)))
    #             ax.annotate(str(int(i)), xy=(x_coord, y_coord),
    #                         textcoords='data')
    #
    #
    #     # plt.ylim(y.min(),y.max())
    #
    #     # plt.xlim(-6.0, mins[2,:].max())
    #
    #     plt.ylabel(Labels.lbls(l_or_lm))
    #     plt.xlabel(Labels.lbls('mdot'))
    #     ax.grid(which='major', alpha=0.2)
    #     plt.legend(bbox_to_anchor=(1, 1), loc='upper right', ncol=1)
    #
    #     ax.grid(which='both')
    #     ax.grid(which='minor', alpha=0.2)
    #     plt.legend(bbox_to_anchor=(1, 0), loc='lower right', ncol=1)
    #     plot_name = self.plot_dir + 'minMdot_l.pdf'
    #     plt.savefig(plot_name)
    #     plt.show()

    # def min_mdot_sp(self, l_or_lm, yc_val):
    #
    #     name = '{}_{}_{}'.format('yc', l_or_lm, 'mdot_crit')
    #     yc_llm_mdot_cr = Save_Load_tables.load_table(name, 'yc', l_or_lm, 'mdot_crit', self.opal_used)
    #     yc  = yc_llm_mdot_cr[0, 1:]
    #     llm = yc_llm_mdot_cr[1:, 0]
    #     mdot2d= yc_llm_mdot_cr[1:, 1:]
    #
    #
    #
    #     if yc_val in yc:
    #         ind = Math.find_nearest_index(yc, yc_val)
    #         mdot = mdot2d[:, ind]
    #     else:
    #         raise ValueError('\tYc = {} not in the list: \n\t{}'.format(yc_val, yc))
    #
    #     fig = plt.figure()
    #     ax = fig.add_subplot(1, 1, 1)
    #     ax.plot(mdot, llm, '-',    color='black')
    #     ax.fill_between(mdot, llm, color="lightgray")
    #
    #     classes  = []
    #     classes.append('dum')
    #     mdot_obs = []
    #     llm_obs  = []
    #
    #
    #     '''=============================OBSERVABELS==============================='''
    #
    #     # from Phys_Math_Labels import Opt_Depth_Analythis
    #
    #     for star_n in self.obs.stars_n:
    #         i = -1
    #         mdot_obs = np.append(mdot_obs, self.obs.get_num_par('mdot', star_n))
    #         llm_obs = np.append(llm_obs, self.obs.get_num_par(l_or_lm, star_n, yc_val, self.opal_used))
    #         eta = self.obs.get_num_par('eta', star_n)
    #
    #         # print(self.obs.get_num_par('mdot',  star_n), self.obs.get_num_par(l_or_lm, star_n))
    #
    #         plt.plot(mdot_obs[i], llm_obs[i], marker=self.obs.get_clss_marker(star_n), markersize='9',
    #                  color=self.obs.get_class_color(star_n), ls='')  # plot color dots)))
    #         ax.annotate('{} {}'.format(int(star_n), eta), xy=(mdot_obs[i], llm_obs[i]), textcoords='data')  # plot numbers of stars
    #
    #
    #         # v_inf = self.obs.get_num_par('v_inf', star_n)
    #         # tau_cl = Opt_Depth_Analythis(30, v_inf, 1., 1., mdot_obs[i], 0.20)
    #         # tau = tau_cl.anal_eq_b1(1.)
    #         # ax.annotate(str(int(tau)), xy=(mdot_obs[i], llm_obs[i]), textcoords='data')  # plo
    #
    #
    #         if self.obs.get_star_class(star_n) not in classes:
    #             plt.plot(mdot_obs[i], llm_obs[i], marker=self.obs.get_clss_marker(star_n), markersize='9',
    #                      color=self.obs.get_class_color(star_n), ls = '',
    #                      label='{}'.format(self.obs.get_star_class(star_n)))  # plot color dots)))
    #             classes.append(self.obs.get_star_class(star_n))
    #
    #     print('\t__PLOT: total stars: {}'.format(len(self.obs.stars_n)))
    #     print(len(mdot_obs), len(llm_obs))
    #
    #     fit = np.polyfit(mdot_obs, llm_obs, 1)  # fit = set of coeddicients (highest first)
    #     f = np.poly1d(fit)
    #     fit_x_coord = np.mgrid[(mdot_obs.min() - 1):(mdot_obs.max() + 1):1000j]
    #     plt.plot(fit_x_coord, f(fit_x_coord), '-.', color='blue')
    #
    #     min_mdot, max_mdot = self.obs.get_min_max('mdot')
    #     min_llm, max_llm = self.obs.get_min_max(l_or_lm, yc_val, self.opal_used)
    #
    #     ax.set_xlim(min_mdot, max_mdot)
    #     ax.set_ylim(min_llm, max_llm)
    #
    #     ax.set_ylabel(Labels.lbls(l_or_lm))
    #     ax.set_xlabel(Labels.lbls('mdot'))
    #     ax.grid(which='major', alpha=0.2)
    #     ax.legend(bbox_to_anchor=(1, 1), loc='upper right', ncol=1)
    #
    #     ax.grid(which='both')
    #     ax.grid(which='minor', alpha=0.2)
    #     plt.legend(bbox_to_anchor=(1, 0), loc='lower right', ncol=1)
    #     plot_name = self.plot_dir + 'minMdot_l.pdf'
    #     plt.savefig(plot_name)
    #     plt.show()


class HRD(PlotObs):
    def __init__(self, gal_or_lmc, bump):

        self.set_obs_file = Files.get_obs_file(gal_or_lmc)
        self.set_atm_file = Files.get_atm_file(gal_or_lmc)
        self.set_plot_files = Files.get_plot_files(gal_or_lmc)
        self.set_metal = gal_or_lmc

        PlotObs.__init__(self, gal_or_lmc,bump, self.set_obs_file, self.set_atm_file)

        self.set_label_size = 12
        self.set_inverse_x = True

        self.set_clean            = True
        self.set_use_gaia         = True
        self.set_use_atm_file     = True
        self.set_load_yc_l_lm     = True
        self.set_load_yc_nan_lmlim= True
        self.set_check_lm_for_wne = True
        self.set_check_affiliation= False


        self.set_do_plot_evol_err = True
        self.set_do_plot_obs_err =  True
        self.set_do_plot_line_fit = True

        self.set_patches_or_lines = 'lines'
        self.set_patches_or_lines_alpha = 0.5
        self.set_patches_or_lines_alpha_fits = 0.8
        self.set_patches_or_lines_color_fits = 'blue'

        self.set_list_metals_fit = [self.metal]
        self.set_list_bumps_fit  = [self.bump]

        self.set_clump_used = 4
        self.set_clump_modified = 4



    def get_yc_min_for_m_init(self, m_in, m_yc_ys_lims):
        m_vals = m_yc_ys_lims[1:, 0]
        yc_vals = m_yc_ys_lims[0, 1:]
        ys_arr = m_yc_ys_lims[1:, 1:]

        m_ind = Math.find_nearest_index(m_vals, m_in)
        # if np.abs(m_in - m_vals[m_ind]) > 0.1: raise ValueError('Correct m_in is not found')

        ys_row = ys_arr[m_ind, ::-1]
        yc_row = yc_vals[::-1]
        ys_zams = ys_arr[m_ind, -1]

        for i in range(len(ys_row)):
            if ys_row[i] < ys_zams:
                print('----------------------- {} {} -------------------------'.format(m_in, yc_vals[i]))
                return yc_row[i - 1]
        print('----------------------- {} -------------------------'.format(yc_row[-1]))
        return yc_row[-1]

        #
        #
        # if not np.round(m_in, 1) in m_vals: raise ValueError('m_init({}) not in m_vals({}) from table file [evol_yc_m_ys]'.format(m_in, m_vals))
        #
        # ind = Math.find_nearest_index(m_vals, m_in)
        # for i in range(len(yc_vals)):
        #     if ys_arr[ind, i] < ys_arr[ind, -1]: # if the surface compostion has changed
        #         print('----------------------- {} {} -------------------------'.format( m_in, yc_vals[i]))
        #         return yc_vals[i-1] # return the yc for which ys has not yet changed
        # print('----------------------- {} -------------------------'.format(yc_vals[0]))
        # return yc_vals[0]
        # # for i in range(len(m_vals)):

    def plot_treks(self, ax, x_v_n, l_or_lm):

        x_y_z = np.zeros(3)
        if len(self.set_plot_files) > 0:

            m_yc_ys_lims = Save_Load_tables.load_table('evol_yc_m_ys', 'evol_yc', 'm', 'ys', self.set_metal, '')
            plcls = []
            for i in range(len(self.set_plot_files)):
                plcls.append(Read_Plot_file.from_file(self.set_plot_files[i]))

                if l_or_lm == 'l':
                    llm_plot = plcls[i].l_
                else:
                    llm_plot = plcls[i].lm_

                m_init = plcls[i].m_[0]
                y_plot = plcls[i].get_col(x_v_n)
                yc_plot = plcls[i].y_c
                # ys_plot =

                # ================================ PLOTTING THE NORMAN FULL TRACKS =====================================

                if not self.set_clean:

                    ax.plot(y_plot, llm_plot, '-', color='gray')

                    for j in range(10):
                        ind = Math.find_nearest_index(plcls[i].y_c, (j / 10))
                        # print(plfl.y_c[i], (i/10))
                        x_p = y_plot[ind]
                        y_p = llm_plot[ind]
                        plt.plot(x_p, y_p, '.', color='red')
                        if not self.set_clean:
                            ax.annotate('{} {}'.format("%.2f" % plcls[i].y_c[ind], "%.2f" % plcls[i].mdot_[ind]),
                                        xy=(x_p, y_p),
                                        textcoords='data')

                # ================================== PLOTTING ONLY THE WNE PHASE =======================================

                yc_min = self.get_yc_min_for_m_init(m_init, m_yc_ys_lims)

                teff_plot2 = []
                llm_plot2 = []
                yc_plot2 = []
                for k in range(len(yc_plot)):
                    if yc_plot[k] > yc_min:
                        teff_plot2 = np.append(teff_plot2, y_plot[k])
                        llm_plot2 = np.append(llm_plot2, llm_plot[k])
                        yc_plot2 = np.append(yc_plot2, yc_plot[k])
                        x_y_z = np.vstack((x_y_z, [y_plot[k], llm_plot[k], yc_plot[k]]))

                ax.plot(teff_plot2, llm_plot2, '-.', color='black')
                ax.annotate('{}'.format("%0.f" % m_init), xy=(teff_plot2[0], llm_plot2[0]), textcoords='data',
                            horizontalalignment='right')
                ax.annotate('{}'.format("%.1f" % yc_min), xy=(teff_plot2[-1], llm_plot2[-1]), textcoords='data',
                            horizontalalignment='left')

        sc = ax.scatter(x_y_z[1:, 0], x_y_z[1:, 1], c=x_y_z[1:, 2], marker='.', cmap=plt.get_cmap('RdYlBu_r'))

        clb = plt.colorbar(sc)
        clb.ax.set_title(Labels.lbls('Yc'), fontsize=self.set_label_size)

        return ax

    def plot_fits(self, ax, obs_metal, v_n_x, v_n_y, x_arr, metals, bumps, hatch='/'):

        def get_linestyle(metal):
            if metal == 'gal':
                return '--'

            if metal == '2gal' or metal == '2lmc':
                return '-.'

            if metal == 'lmc':
                return '-'

            if metal == 'smc':
                return ':'

        if len(metals) != len(bumps): raise IOError('Metals != bumps')

        y_arrs = np.zeros(len(x_arr))
        for metal in metals:
            if obs_metal == 'gal':
                y_arr = Fits.get_gal_fit(v_n_x, v_n_y, x_arr, metal, self.set_clump_modified, self.set_use_gaia)
            elif obs_metal == 'lmc':
                y_arr = Fits.get_lmc_fit(v_n_x, v_n_y, x_arr, metal, self.bump)
            else:
                raise IOError('FITS: Obs_metal gal and lmc are only supproted so far')


            ax.plot(x_arr, y_arr, get_linestyle(metal), color=self.set_patches_or_lines_color_fits,
                    alpha=self.set_patches_or_lines_alpha_fits)

            y_arrs = np.vstack((y_arrs, y_arr))

        y_arrs = np.delete(y_arrs, 0, 0)

        if len(metals) > 0:
            ax.fill_between(x_arr, y_arrs[0, :], y_arrs[-1, :], hatch=hatch, alpha=self.set_patches_or_lines_alpha_fits)

    def plot_hrd_treks(self, x_v_n, l_or_lm):

        fig, ax = plt.subplots(1, 1)
        # ax.set_title('HRD')
        ax.set_xlabel(Labels.lbls(x_v_n), fontsize=self.set_label_size)
        ax.set_ylabel(Labels.lbls(l_or_lm), fontsize=self.set_label_size)


        self.plot_treks(ax, x_v_n, l_or_lm)


        ax.minorticks_on()
        plt.xticks(fontsize=self.set_label_size)
        plt.yticks(fontsize=self.set_label_size)

        if self.set_inverse_x: plt.gca().invert_xaxis()  # inverse x axis
        if not self.set_clean:
            plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
        plot_name = Files.output_dir + 'hrd.pdf'
        plt.savefig(plot_name)
        plt.show()

    def plot_evol_mdot(self):

        fig, ax = plt.subplots(1, 1)
        ax.set_title('HRD')
        ax.set_xlabel(Labels.lbls('yc'))
        ax.set_ylabel(Labels.lbls('mdot'))

        def get_yc_min_for_m_init(m_in, m_yc_ys_lims):
            m_vals = m_yc_ys_lims[1:, 0]
            yc_vals = m_yc_ys_lims[0, 1:]
            ys_arr = m_yc_ys_lims[1:, 1:]

            m_ind = Math.find_nearest_index(m_vals, m_in)
            if np.abs(m_in - m_vals[m_ind]) > 0.1: raise ValueError('Correct m_in is not found')

            ys_row = ys_arr[m_ind, ::-1]
            yc_row = yc_vals[::-1]
            ys_zams = ys_arr[m_ind, -1]

            for i in range(len(ys_row)):
                if ys_row[i] < ys_zams:
                    print('----------------------- {} {} -------------------------'.format(m_in, yc_vals[i]))
                    return yc_row[i - 1]
            print('----------------------- {} -------------------------'.format(yc_row[-1]))
            return yc_row[-1]

            #
            #
            # if not np.round(m_in, 1) in m_vals: raise ValueError('m_init({}) not in m_vals({}) from table file [evol_yc_m_ys]'.format(m_in, m_vals))
            #
            # ind = Math.find_nearest_index(m_vals, m_in)
            # for i in range(len(yc_vals)):
            #     if ys_arr[ind, i] < ys_arr[ind, -1]: # if the surface compostion has changed
            #         print('----------------------- {} {} -------------------------'.format( m_in, yc_vals[i]))
            #         return yc_vals[i-1] # return the yc for which ys has not yet changed
            # print('----------------------- {} -------------------------'.format(yc_vals[0]))
            # return yc_vals[0]
            # # for i in range(len(m_vals)):

        # yc_vals = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, ]
        yc_vals = [1.0, 0.9,  0.8,  0.7,  0.6,  0.5,  0.4,  0.3, 0.2 ]
        v_n_conds = ['yc', 'mdot', 'lm']

        if len(self.set_obs_file) > 0:
            m_yc_ys_lims = Save_Load_tables.load_table('evol_yc_m_ys', 'evol_yc', 'm', 'ys', self.set_metal, '')
            plcls = []
            for i in range(len(self.set_obs_file)):
                plcls.append(Read_Plot_file.from_file(self.set_obs_file[i]))

                # size = '{'
                # head = ''
                # for i in range(len(v_n_conds)):
                #     size = size + 'c'
                #     head = head + '{}'.format(v_n_conds[i])
                #     if i != len(v_n_conds) - 1: size = size + ' '
                #     if i != len(v_n_conds) - 1: head = head + ' & '
                #     # if i % 2 == 0: size = size + ' '
                # head = head + ' \\\\'  # = \\
                #
                # size = size + '}'
                #
                # print('\\begin{table}[h!]')
                # print('\\begin{center}')
                # print('\\begin{tabular}' + '{}'.format(size))
                # print('\\hline')
                # print(head)
                # print('\\hline\\hline')
                #
                # # for i in range(len(self.smfiles)):
                # # 1 & 6 & 87837 & 787 \\
                # row = ''
                # for j in range(len(v_n_conds)):
                #     val = "%{}f".format(0.2) % self.mdl[i].get_col(v_n_conds[j])
                #     row = row + val
                #     if j != len(v_n_conds) - 1: row = row + ' & '
                # row = row + ' \\\\'  # = \\
                # print(row)
                #
                # print('\\hline')
                # print('\\end{tabular}')
                # print('\\end{center}')
                # print('\\caption{NAME_ME}')
                # print('\\label{tbl:1}')
                # print('\\end{table}')



                yc_plot = plcls[i].y_c
                mdot_plot = plcls[i].mdot_

                mdot_res = []

                for j in range(len(yc_vals)):
                    ind = Math.find_nearest_index(yc_plot, yc_vals[j])
                    mdot_res = np.append(mdot_res, plcls[i].mdot_[ind])
                    print('{} & {} & {}'.format("%.2f"%plcls[i].y_c[ind],"%.2f"% plcls[i].mdot_[ind],"%.2f"% plcls[i].lm_[ind]))
                    # print(plcls[i].y_c[ind], ' & ', plcls[i].mdot_[ind] )

                    # for k in range(len(yc_plot)):
                    #     ind = Math.find_nearest_index()
                    #     if np.round(yc_plot[k], 2) == np.round(yc_vals[j], 2):
                    #         mdot_res = np.append(mdot_res, plcls[i].mdot_[k])

                print('a')

    def plot_hrd(self, v_n_x, l_or_lm, obs=True):

        fig = plt.figure()
        ax = fig.add_subplot(111)

        if obs:
            # self.plot_obs_l_l_comparison(ax)
            stars_n, x_coords, y_coords = self.plot_obs_all_x_llm(ax, l_or_lm, v_n_x, 1.0, False, True) # return ax = false

            self.plot_fits(ax, self.set_metal, v_n_x, l_or_lm, x_coords,
                           self.set_list_metals_fit,  # ['lmc', 'gal', '2gal']
                           self.set_list_bumps_fit)

        # ax.set_title('HRD')
        ax.set_xlabel(Labels.lbls(v_n_x), fontsize=self.set_label_size)
        ax.set_ylabel(Labels.lbls(l_or_lm), fontsize=self.set_label_size)


        # x_min, x_max = ax.get_xlim()
        # y_min, y_max = ax.get_ylim()



        if self.set_inverse_x:
            plt.gca().invert_xaxis() # inverse x axis

        ax.tick_params('y', labelsize=self.set_label_size)
        ax.tick_params('x', labelsize=self.set_label_size)
        plt.minorticks_on()

        ax.set_ylabel(Labels.lbls(l_or_lm))
        ax.set_xlabel(Labels.lbls(v_n_x))

        # plt.grid()
        # if not self.set_clean:
        plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=3, fontsize=self.set_label_size)
        plot_name = Files.plot_dir + 'hrd.pdf'
        plt.savefig(plot_name)

        plt.show()

    # def gaia_lum_exerciese(self):


class PrintTable:
    def __init__(self, plotfiles, yc_arr):
        self.plotfiles = plotfiles
        self.yc_arr = yc_arr
        self.plcls = []
        for i in range(len(self.plotfiles)):
            self.plcls.append(Read_Plot_file.from_file(self.plotfiles[i]))

        if len(self.plcls)==0:
            raise NameError('No .plot1 files has been passed to |PlotTable|')

    def latex_table(self, v_ns, precis):
        ''' print a latex compatible table'''

        def get_str_of_elements(n, filling = 'l', borders='|'):
            row =''
            row = row + borders
            for i in range(n):
                row = row + filling + borders

            return row

        def print_multirow_nams(v_ns):
            '''
            \multicolumn{2}{c}{1.plot1} &
            \multicolumn{2}{c}{2.plot1} &
            \multicolumn{2}{c|}{3.plot1} \\
            '''
            # n = len(v_ns)
            for i in range(len(self.plotfiles)-1):
                # name = self.plotfiles[i].split('/')[-1]

                print('\\multicolumn{'+str(len(v_ns))+'}{c}{ f:'+ self.plotfiles[i].split('/')[-2]+ ' } &')
            print('\\multicolumn{' + str(len(v_ns)) + '}{c|}{ f:' + self.plotfiles[-1].split('/')[-2] + ' } \\\\')

        def print_column_names(v_ns):
            '''
            & O.B.R & A.R & O.B.R & A.R & O.B.R & A.R \\
            '''
            row = '& '
            for file in self.plotfiles:
                for v_n in v_ns:
                    row = row + v_n + ' & '
            row = row[:-2]
            row = row + ' \\\\'
            print(row)

        def get_str_col(plcl, v_ns, prec):

            col = []
            for i in range(len(self.yc_arr)):
                piece = ' '
                ind = Math.find_nearest_index(plcl.y_c, self.yc_arr[i])
                for i in range(len(v_ns)):
                    piece = piece + "%{}f".format(prec[i]) % plcl.get_col(v_ns[i])[ind]
                    if i < len(v_ns)-1:  piece = piece + ' & '
                col.append(piece)
            return col

        def print_table(v_ns, precs, row_separateor):
            cols = []
            for plcl in self.plcls:
                cols.append(get_str_col(plcl, v_ns, precs))

            # print(cols)
            # print(len(cols))
            rows = []
            for j in range(len(self.yc_arr)):
                row = "%.2f"%self.yc_arr[j] + ' &'
                for i in range(len(self.plcls)):
                    row = row + cols[i][j]
                    if i < len(self.plcls)-1:
                        row = row + ' &'

                print(row + ' \\\\')
                print(row_separateor)
                # print(i+len(v_ns))
                # row = cols[i+len(v_ns)]
                # print(row)


        # def print_row(v_ns):
        #     '''
        #     val1 & val1 & val1 & val1 & val1 & val1 & val1 \\
        #     '''
        #     row = '& '
        #     for file in self.plotfiles:
        #         for v_n in v_ns:
        #             row = row + v_n + ' & '
        #     row = row[:-1]
        #     row = row + ' \\\\'
        #     print(row)

        # v_ns = v_ns[::-1]
        # v_ns.append('yc') # addong yc as a first v_n
        # v_ns = v_ns[::-1]

        print('\n')
        print('\\begin{table}')
        print('\\begin{center}')
        print('\\begin{tabular}{'+get_str_of_elements(len(v_ns)*len(self.plotfiles))+'}')
        print('\\hline')
        print('\\multirow{2}{*}{$He_{core}$} &')
        print_multirow_nams(v_ns)
        print_column_names(v_ns)
        print('\\hline')
        print_table(v_ns, precis, '\\hline')
        print('\\end{tabular}')
        print('\\end{center}')
        print('\\caption{FILL_ME}')
        print('\\end{table}')

class Table:

    def __init__(self, smfiles, spfile):
        self.smfiles = smfiles
        self.spfiles = spfile

        self.set_use_only_spfls = False

        self.mdl = []
        for file in smfiles:
            self.mdl.append(Read_SM_data_file(file))

        self.spmdl = []
        if len(spfile) > 0:
            for file in spfile:
                self.spmdl.append(Read_SP_data_file(file))
            # if len(self.spmdl) != len(self.mdl):
                # raise NameError('sm.files: {} != {} sp.files'.format( len(self.mdl), len(self.spmdl)))

    def latex_table(self, v_n_conds, precis):
        '''

        :param v_ns:
        :return:

        \begin{center}
        \begin{tabular}{||c c c c||}
        \hline
        Col1 & Col2 & Col2 & Col3 \\ [0.5ex]
        \hline\hline
        1 & 6 & 87837 & 787 \\
        \hline
        2 & 7 & 78 & 5415 \\
        \hline
        3 & 545 & 778 & 7507 \\
        \hline
        4 & 545 & 18744 & 7560 \\
        \hline
        5 & 88 & 788 & 6344 \\ [1ex]
        hline
        \end{tabular}
        \end{center}

        '''
        if len(v_n_conds) != len(precis): raise NameError('len(v_ns) != len(precis)')
        # if len(self.spmdl)!=0 and len(self.spmdl) != len(self.mdl): raise IOError('SM files: {} SP files: {}'
        #                                                                           .format(len(self.mdl), len(self.spmdl)))
        #

        size='{'
        head = ''
        for i in range(len(v_n_conds)):
            size = size + 'c'
            head = head + '{}'.format(v_n_conds[i])
            if i != len(v_n_conds) - 1: size = size + ' '
            if i != len(v_n_conds) - 1: head = head + ' & '
            # if i % 2 == 0: size = size + ' '
        head = head + ' \\\\' # = \\

        size=size+'}'

        print('\\begin{table}[h!]')
        print('\\begin{center}')
        print('\\begin{tabular}'+'{}'.format(size))
        print('\\hline')
        print(head)
        print('\\hline\\hline')


        for i in range(len(self.mdl)):
            mdot = self.mdl[i].get_col('mdot')[-1]
            # 1 & 6 & 87837 & 787 \\
            row = ''
            for j in range(len(v_n_conds)):
                if v_n_conds[j].split('-')[-1] == '' and not self.set_use_only_spfls:
                    val = "%{}f".format(precis[j]) % self.mdl[i].get_cond_value(v_n_conds[j].split('-')[0],
                                                                                v_n_conds[j].split('-')[-1])

                elif v_n_conds[j].split('-')[-1] == 'sp':
                    val = "%{}f".format(precis[j]) % self.spmdl[i].get_sonic_cols(v_n_conds[j].split('-')[0], 'mdot={}'.format("%.2f"%mdot))

                elif v_n_conds[j].split('-')[-1] == 'ph' and self.spmdl[i]!=None:
                    val = "%{}f".format(precis[j]) % self.spmdl[i].get_sonic_cols(v_n_conds[j],
                                                                                  'mdot={}'.format("%.2f" % mdot))

                elif v_n_conds[j].split('-')[-1] == '0' or v_n_conds[j].split('-')[-1] == '1' and self.spmdl[i]!=None:
                    val = "%{}f".format(precis[j]) % self.spmdl[i].get_sonic_cols(v_n_conds[j].split('-')[0],
                                                                                  'mdot={}'.format("%.2f" % mdot))

                else: raise NameError('name {} is not recognised'.format(v_n_conds[j]))

                row = row + val
                if j != len(v_n_conds) - 1: row = row + ' & '
            row = row + ' \\\\'  # = \\
            print(row)

        print('\\hline')
        print('\\end{tabular}')
        print('\\end{center}')
        print('\\caption{NAME_ME}')
        print('\\label{tbl:1}')
        print('\\end{table}')

class Critical_Mdot:

    def __init__(self, metal, bump, coeff, sp_files_metal):

        self.set_metal = metal
        self.set_sp_files=Files.get_sp_files(sp_files_metal, 'cr')
        self.set_output_dir = Files.output_dir
        self.set_plot_dir = Files.plot_dir

        self.set_bump = bump
        self.set_coeff = coeff

        self.lim_t1, self.lim_t2 = T_kappa_bump.t_for_bump(bump)
        self.spmdl = []


        for file in self.set_sp_files:
            self.spmdl.append(Read_SP_data_file(file, self.set_output_dir, self.set_plot_dir))

    # def set_files(self, bump, coeff):
    #
    #     self.bump = bump
    #     self.coeff = coeff
    #     self.lim_t1, self.lim_t2 = T_kappa_bump.t_for_bump(bump)
    #     self.spmdl=[]
    #     for file in self.set_sp_files:
    #         self.spmdl.append(Read_SP_data_file(file, self.set_output_dir, self.set_plot_dir))

        # self.nums = Num_Models(smfls, plotfls)
        # self.obs = Read_Observables(self.obs_files, self.set_metal)

    @staticmethod
    def interp(x, y, x_grid):
        f = interpolate.interp1d(x, y, kind='cubic', bounds_error=False)
        return f(x_grid)

    @staticmethod
    def univar_interp(x, y, new_x):
        # if new_x.min() < x.min():
        #     raise ValueError('new_x.min({}) < x.min({})'.format(new_x.min(), x.min()))
        # if new_x.max() > x.max():
        #     raise ValueError('new_x.min({}) > x.min({})'.format(new_x.max(), x.max()))
        if len(x) != len(y):
            raise ValueError('len(x)[{}] != len(y)[{}]'.format(len(x), len(y)))
        for x_i in x:
            if x_i < x[0]:
                raise ValueError('Array x is not monotonically increasing. x_i({}) < x[0]({})'.format(x_i, x[0]))

        f = interpolate.UnivariateSpline(x, y)

        return f(new_x)

    @staticmethod
    def fit_plynomial(x, y, order, depth, new_x=np.empty(0, )):
        '''
        RETURNS f(new_x)
        :param x:
        :param y:
        :param order: 1-4 are supported
        :return:
        '''
        f = None
        lbl = None

        if new_x == np.empty(0, ):
            new_x = np.mgrid[(x.min()):(x.max()):depth * 1j]

        if order == 1:
            fit = np.polyfit(x, y, order)  # fit = set of coeddicients (highest first)
            f = np.poly1d(fit)
            lbl = '({}) + ({}*x)'.format(
                "%.3f" % f.coefficients[1],
                "%.3f" % f.coefficients[0]
            )
            # fit_x_coord = np.mgrid[(x.min()):(x.max()):depth*1j]
            # plt.plot(fit_x_coord, f(fit_x_coord), '--', color='black')

        if order == 2:
            fit = np.polyfit(x, y, order)  # fit = set of coeddicients (highest first)
            f = np.poly1d(fit)
            lbl = '({}) + ({}*x) + ({}*x**2)'.format(
                "%.3f" % f.coefficients[2],
                "%.3f" % f.coefficients[1],
                "%.3f" % f.coefficients[0]
            )
            # fit_x_coord = np.mgrid[(x.min()):(x.max()):depth*1j]
            # plt.plot(fit_x_coord, f(fit_x_coord), '--', color='black')
        if order == 3:
            fit = np.polyfit(x, y, order)  # fit = set of coeddicients (highest first)
            f = np.poly1d(fit)
            lbl = '({}) + ({}*x) + ({}*x**2) + ({}*x**3)'.format(
                "%.3f" % f.coefficients[3],
                "%.3f" % f.coefficients[2],
                "%.3f" % f.coefficients[1],
                "%.3f" % f.coefficients[0]
            )
            # fit_x_coord = np.mgrid[(x.min()):(x.max()):depth*1j]
            # plt.plot(fit_x_coord, f(fit_x_coord), '--', color='black')
        if order == 4:
            fit = np.polyfit(x, y, order)  # fit = set of coeddicients (highest first)
            f = np.poly1d(fit)
            lbl = '({}) + ({}*x) + ({}*x**2) + ({}*x**3) + ({}*x**4)'.format(
                "%.3f" % f.coefficients[4],
                "%.3f" % f.coefficients[3],
                "%.3f" % f.coefficients[2],
                "%.3f" % f.coefficients[1],
                "%.3f" % f.coefficients[0]
            )
            # fit_x_coord = np.mgrid[(x.min()):(x.max()):depth*1j]
            # plt.plot(fit_x_coord, f(fit_x_coord), '--', color='black')

        print(lbl)

        return f(new_x)

    @staticmethod
    def common_y3(arr1, arr2, arr3):

        y1 = arr1[1:, 0]
        y2 = arr2[1:, 0]
        y3 = arr3[1:, 0]

        y_min = np.array([y1.min(), y2.min(), y3.min()]).max()
        y_max = np.array([y1.max(), y2.max(), y3.max()]).min()

        arr1_cropped = Math.crop_2d_table(arr1, None, None, y_min, y_max)
        arr2_cropped = Math.crop_2d_table(arr2, None, None, y_min, y_max)
        arr3_cropped = Math.crop_2d_table(arr3, None, None, y_min, y_max)

        return arr1_cropped, arr2_cropped, arr3_cropped

    @staticmethod
    def common_y2(arr1, arr2):

        y1 = arr1[1:, 0]
        y2 = arr2[1:, 0]

        y_min = np.array([y1.min(), y2.min()]).max()
        y_max = np.array([y1.max(), y2.max()]).min()

        arr1_cropped = Math.crop_2d_table(arr1, None, None, y_min, y_max)
        arr2_cropped = Math.crop_2d_table(arr2, None, None, y_min, y_max)

        return arr1_cropped, arr2_cropped

    @staticmethod
    def x_y_grids(x_arr, y_arr, depth):
        x_grid = np.mgrid[x_arr.min():x_arr.max(): depth * 1j]
        y_grid = np.mgrid[y_arr.min():y_arr.max(): depth * 1j]
        return x_grid, y_grid

    @staticmethod
    def plot_4arrays(yc_l_mdot_int, yc_l_mdot_pol, yc_lm_mdot_int, yc_lm_mdot_pol):
        levels = [-7.0, -6.9, -6.8, -6.7, -6.6, -6.5, -6.4, -6.3, -6.2, -6.1, -6.0, -5.9, -5.8,
                  -5.7, -5.6, -5.5, -5.4, -5.3, -5.2, -5.1, -5., -4.9, -4.8, -4.7, -4.6, -4.5, -4.4, -4.3,
                  -4.2, -4.1, -4.]

        fig = plt.figure(figsize=plt.figaspect(1.0))

        ax = fig.add_subplot(221)

        ax.set_title('INTERPOLATION')
        ax.set_xlim(yc_l_mdot_int[0, 1:].min(), yc_l_mdot_int[0, 1:].max())
        ax.set_ylim(yc_l_mdot_int[1:, 0].min(), yc_l_mdot_int[1:, 0].max())
        ax.set_ylabel(Labels.lbls('l'))
        ax.set_xlabel(Labels.lbls('Yc'))
        contour_filled = plt.contourf(yc_l_mdot_int[0, 1:], yc_l_mdot_int[1:, 0], yc_l_mdot_int[1:, 1:], levels,
                                      cmap=plt.get_cmap('RdYlBu_r'))
        contour = plt.contour(yc_l_mdot_int[0, 1:], yc_l_mdot_int[1:, 0], yc_l_mdot_int[1:, 1:], levels, colors='k')
        clb = plt.colorbar(contour_filled)
        clb.ax.set_title(Labels.lbls('mdot'))
        plt.clabel(contour, colors='k', fmt='%2.2f', fontsize=12)

        # ----------------------------------------------------------------------------------------------------------

        ax = fig.add_subplot(222)

        ax.set_title('EXTRAPOLATION')
        ax.set_xlim(yc_l_mdot_pol[0, 1:].min(), yc_l_mdot_pol[0, 1:].max())
        ax.set_ylim(yc_l_mdot_pol[1:, 0].min(), yc_l_mdot_pol[1:, 0].max())
        ax.set_ylabel(Labels.lbls('l'))
        ax.set_xlabel(Labels.lbls('Yc'))
        contour_filled = plt.contourf(yc_l_mdot_pol[0, 1:], yc_l_mdot_pol[1:, 0], yc_l_mdot_pol[1:, 1:], levels,
                                      cmap=plt.get_cmap('RdYlBu_r'))
        contour = plt.contour(yc_l_mdot_pol[0, 1:], yc_l_mdot_pol[1:, 0], yc_l_mdot_pol[1:, 1:], levels, colors='k')
        clb = plt.colorbar(contour_filled)
        clb.ax.set_title(Labels.lbls('mdot'))
        plt.clabel(contour, colors='k', fmt='%2.2f', fontsize=12)

        # ----------------------------------------------------------------------------------------------------------

        ax = fig.add_subplot(223)

        ax.set_xlim(yc_lm_mdot_int[0, 1:].min(), yc_lm_mdot_int[0, 1:].max())
        ax.set_ylim(yc_lm_mdot_int[1:, 0].min(), yc_lm_mdot_int[1:, 0].max())
        ax.set_ylabel(Labels.lbls('lm'))
        ax.set_xlabel(Labels.lbls('Yc'))
        contour_filled = plt.contourf(yc_lm_mdot_int[0, 1:], yc_lm_mdot_int[1:, 0], yc_lm_mdot_int[1:, 1:], levels,
                                      cmap=plt.get_cmap('RdYlBu_r'))
        contour = plt.contour(yc_lm_mdot_int[0, 1:], yc_lm_mdot_int[1:, 0], yc_lm_mdot_int[1:, 1:], levels, colors='k')
        clb = plt.colorbar(contour_filled)
        clb.ax.set_title(Labels.lbls('mdot'))
        plt.clabel(contour, colors='k', fmt='%2.2f', fontsize=12)

        # ----------------------------------------------------------------------------------------------------------

        ax = fig.add_subplot(224)

        ax.set_xlim(yc_lm_mdot_pol[0, 1:].min(), yc_lm_mdot_pol[0, 1:].max())
        ax.set_ylim(yc_lm_mdot_pol[1:, 0].min(), yc_lm_mdot_pol[1:, 0].max())
        ax.set_ylabel(Labels.lbls('lm'))
        ax.set_xlabel(Labels.lbls('Yc'))
        contour_filled = plt.contourf(yc_lm_mdot_pol[0, 1:], yc_lm_mdot_pol[1:, 0], yc_lm_mdot_pol[1:, 1:], levels,
                                      cmap=plt.get_cmap('RdYlBu_r'))
        contour = plt.contour(yc_lm_mdot_pol[0, 1:], yc_lm_mdot_pol[1:, 0], yc_lm_mdot_pol[1:, 1:], levels, colors='k')
        clb = plt.colorbar(contour_filled)
        clb.ax.set_title(Labels.lbls('mdot'))
        plt.clabel(contour, colors='k', fmt='%2.2f', fontsize=12)

        plt.show()

    # --- --- --- PUBLIC --- --- ---

    def save_yc_llm_mdot_cr(self, depth = 100, plot = True):

        def l_lm_mdot_rows(t_lm_rho_opal, lm_l_sp, lm_r_sp, lim_t1, lim_t2):
            lm_op = t_lm_rho_opal[1:, 0]
            t = t_lm_rho_opal[0, 1:]
            rho2d = t_lm_rho_opal[1:, 1:]
            new_l = self.univar_interp(lm_l_sp[0, :], lm_l_sp[1, :], lm_op)
            new_r = self.univar_interp(lm_r_sp[0, :], lm_r_sp[1, :], lm_op)

            if len(lm_op) != len(new_l) or len(lm_op) != len(new_r):
                raise ValueError('len(lm_op)[{}] != len(new_l)[{}] or len(lm_op)[{}] != len(new_r)[{}]'
                                 .format(len(lm_op), len(new_l), len(lm_op), len(new_r)))


            vrho = Physics.get_vrho(t, rho2d, 2, np.array([1.34]))
            m_dot = Physics.vrho_mdot(vrho, new_r, 'l')
            mins = Math.get_mins_in_every_row(t, new_l, m_dot, 5000, lim_t1, lim_t2)
            mdot_cr = mins[2, :]

            return new_l, lm_op, mdot_cr

        if self.set_bump != 'Fe':
            raise NameError('This function is not available for bumps other than Fe.')

        t_lm_rho = Save_Load_tables.load_table('t_{}lm_rho'.format(self.set_coeff), 't', '{}lm'.format(self.set_coeff),
                                               'rho', self.set_metal, 'Fe', self.set_output_dir)
        yc_lm_l  = Save_Load_tables.load_table('yc_lm_l',  'yc', 'lm', 'l', self.set_metal, '', self.set_output_dir)
        yc_lm_r  = Save_Load_tables.load_table('yc_lm_r',  'yc', 'lm', 'r', self.set_metal, '', self.set_output_dir)

        t_lm_rho, yc_lm_l, yc_lm_r = self.common_y3(t_lm_rho, yc_lm_l, yc_lm_r)

        yc  = yc_lm_l[0, 1:]
        lm_sp  = yc_lm_l[1:, 0]

        l2d = yc_lm_l[1:, 1:]
        r2d = yc_lm_r[1:, 1:]

        lm_grid, l_grid = self.x_y_grids(lm_sp, l2d, depth)

        mdot_poly_l = np.zeros(len(l_grid))
        mdot_poly_lm= np.zeros(len(lm_grid))

        mdot_int_l = np.zeros(len(l_grid))
        mdot_int_lm= np.zeros(len(lm_grid))

        for i in range(len(yc)):
            print('\n\t__Yc={}__'.format(yc[i]))
            yc_i = Math.find_nearest_index(yc, yc[i])
            lm_l = np.vstack((yc_lm_l[1:, 0] , l2d[:, yc_i]))
            lm_r = np.vstack((yc_lm_r[1:, 0] , r2d[:, yc_i]))

            l, lm, mdot = l_lm_mdot_rows(t_lm_rho, lm_l, lm_r, self.lim_t1, self.lim_t2)

            mdot_poly_l  = np.vstack((mdot_poly_l,  self.fit_plynomial(l,  mdot, 3, depth, l_grid)))
            mdot_poly_lm = np.vstack((mdot_poly_lm, self.fit_plynomial(lm, mdot, 3, depth, lm_grid)))

            mdot_int_l  = np.vstack((mdot_int_l,  self.interp(l,  mdot, l_grid)))
            mdot_int_lm = np.vstack((mdot_int_lm, self.interp(lm, mdot, lm_grid)))

        mdot_poly_l  = np.delete(mdot_poly_l,  0, 0)
        mdot_poly_lm = np.delete(mdot_poly_lm, 0, 0)
        mdot_int_l   = np.delete(mdot_int_l,   0, 0)
        mdot_int_lm  = np.delete(mdot_int_lm,  0, 0)

        yc_l_mdot_pol  = Math.combine(yc, l_grid,  mdot_poly_l.T)
        yc_lm_mdot_pol = Math.combine(yc, lm_grid, mdot_poly_lm.T)
        yc_l_mdot_int  = Math.combine(yc, l_grid,  mdot_int_l.T)
        yc_lm_mdot_int = Math.combine(yc, lm_grid, mdot_int_lm.T)

        table_name = '{}_{}{}_{}'.format('yc', self.set_coeff, 'l', 'mdot_cr')
        Save_Load_tables.save_table(yc_l_mdot_pol, self.set_metal, self.set_bump, table_name, 'yc', '{}l'.format(self.set_coeff), 'mdot_cr')

        table_name = '{}_{}{}_{}'.format('yc', self.set_coeff, 'lm', 'mdot_cr')
        Save_Load_tables.save_table(yc_lm_mdot_pol, self.set_metal, self.set_bump, table_name, 'yc', '{}lm'.format(self.set_coeff), 'mdot_cr')

        if plot:
            self.plot_4arrays(yc_l_mdot_int, yc_l_mdot_pol, yc_lm_mdot_int, yc_lm_mdot_pol)

    def save_yc_llm_mdot_cr_const_r(self, r_cr, depth = 100, plot = True):

        def l_lm_mdot_rows_r_const(t_lm_rho_opal, lm_l_sp, r_cr, lim_t1, lim_t2):
            lm_op = t_lm_rho_opal[1:, 0]
            t = t_lm_rho_opal[0, 1:]
            rho2d = t_lm_rho_opal[1:, 1:]
            new_l = self.univar_interp(lm_l_sp[0, :], lm_l_sp[1, :], lm_op)

            if len(lm_op) != len(new_l):
                raise ValueError('len(lm_op)[{}] != len(new_l)[{}] or len(lm_op)[{}]'
                                 .format(len(lm_op), len(new_l), len(lm_op)))


            vrho = Physics.get_vrho(t, rho2d, 2, np.array([1.34]))
            m_dot = Physics.vrho_mdot(vrho, r_cr, '')                  # '' for constant r
            mins = Math.get_mins_in_every_row(t, new_l, m_dot, 5000, lim_t1, lim_t2)
            mdot_cr = mins[2, :]

            return new_l, lm_op, mdot_cr

        t_lm_rho = Save_Load_tables.load_table('t_{}lm_rho'.format(self.set_coeff), 't', '{}lm'.format(self.set_coeff),
                                               'rho', self.set_metal, self.set_bump, self.set_output_dir)
        yc_lm_l = Save_Load_tables.load_table('yc_lm_l', 'yc', 'lm', 'l', self.set_metal, '', self.set_output_dir)

        t_lm_rho, yc_lm_l = self.common_y2(t_lm_rho, yc_lm_l)

        yc    = yc_lm_l[0, 1:]
        lm_sp = yc_lm_l[1:, 0]
        l2d   = yc_lm_l[1:, 1:]

        lm_grid, l_grid = self.x_y_grids(lm_sp, l2d, depth)

        mdot_poly_l = np.zeros(len(l_grid))
        mdot_poly_lm = np.zeros(len(lm_grid))

        mdot_int_l = np.zeros(len(l_grid))
        mdot_int_lm = np.zeros(len(lm_grid))

        for i in range(len(yc)):
            print('\n\t__Yc={}__'.format(yc[i]))
            yc_i = Math.find_nearest_index(yc, yc[i])
            lm_l = np.vstack((yc_lm_l[1:, 0] , l2d[:, yc_i]))

            l, lm, mdot = l_lm_mdot_rows_r_const(t_lm_rho, lm_l, r_cr, self.lim_t1, self.lim_t2)

            mdot_poly_l  = np.vstack((mdot_poly_l,  self.fit_plynomial(l,  mdot, 3, depth, l_grid)))
            mdot_poly_lm = np.vstack((mdot_poly_lm, self.fit_plynomial(lm, mdot, 3, depth, lm_grid)))

            mdot_int_l  = np.vstack((mdot_int_l,  self.interp(l,  mdot, l_grid)))
            mdot_int_lm = np.vstack((mdot_int_lm, self.interp(lm, mdot, lm_grid)))

        mdot_poly_l = np.delete(mdot_poly_l, 0, 0)
        mdot_poly_lm = np.delete(mdot_poly_lm, 0, 0)
        mdot_int_l = np.delete(mdot_int_l, 0, 0)
        mdot_int_lm = np.delete(mdot_int_lm, 0, 0)

        yc_l_mdot_pol = Math.combine(yc, l_grid, mdot_poly_l.T)
        yc_lm_mdot_pol = Math.combine(yc, lm_grid, mdot_poly_lm.T)
        yc_l_mdot_int = Math.combine(yc, l_grid, mdot_int_l.T)
        yc_lm_mdot_int = Math.combine(yc, lm_grid, mdot_int_lm.T)

        table_name = '{}_{}{}_{}'.format('yc', self.set_coeff, 'l', 'mdot_cr_r_{}'.format(r_cr))
        Save_Load_tables.save_table(yc_l_mdot_pol, self.set_metal, self.set_bump, table_name, 'yc',
                                    '{}l'.format(self.set_coeff), 'mdot_cr_r_{}'.format(r_cr))

        table_name = '{}_{}{}_{}'.format('yc', self.set_coeff, 'lm', 'mdot_cr_r_{}'.format(r_cr))
        Save_Load_tables.save_table(yc_lm_mdot_pol, self.set_metal, self.set_bump, table_name, 'yc',
                                    '{}lm'.format(self.set_coeff), 'mdot_cr_r_{}'.format(r_cr))

        if plot:
            self.plot_4arrays(yc_l_mdot_int, yc_l_mdot_pol, yc_lm_mdot_int, yc_lm_mdot_pol)


    def plot_test_min_mdot(self, r_cr):

        def l_lm_mdot_rows_r_const(t_lm_rho_opal, r_cr, lim_t1, lim_t2):
            lm_op = t_lm_rho_opal[1:, 0]
            t = t_lm_rho_opal[0, 1:]
            rho2d = t_lm_rho_opal[1:, 1:]


            vrho = Physics.get_vrho(t, rho2d, 2, np.array([1.34]))
            m_dot = Physics.vrho_mdot(vrho, r_cr, '')                  # '' for constant r
            mins = Math.get_mins_in_every_row(t, lm_op, m_dot, 5000, lim_t1, lim_t2)
            mdot_cr = mins[2, :]
            lm = mins[1, :]
            ts = mins[0, :]

            return Math.combine(t, lm_op, m_dot), ts, lm, mdot_cr

        t_lm_rho = Save_Load_tables.load_table('t_{}lm_rho'.format(self.set_coeff), 't', '{}lm'.format(self.set_coeff),
                                               'rho', self.set_metal, self.set_bump, self.set_output_dir)

        t_lm_mdot, ts, lm, mdot = l_lm_mdot_rows_r_const(t_lm_rho, r_cr, self.lim_t1, self.lim_t2)

        fig = plt.figure(figsize=plt.figaspect(0.8))
        ax = fig.add_subplot(111)  # , projection='3d'
        PlotBackground.plot_color_background(ax, t_lm_mdot, 't', 'lm', 'mdot', self.set_metal, 'Test')
        ax.plot(ts, lm, '.-', color = 'red')
        plt.show()

    def plot_cr_mdot_sp_set(self, l_or_lm, yc_vals, r_cr = None, v_n_background = None):
        '''
        PLOTS the set of llm(mdot), with l->m relation assumed from yc_vals.
        :param l_or_lm:
        :param yc_vals:
        :return:
        '''



        if r_cr == None:
            name = '{}_{}{}_{}'.format('yc', self.set_coeff, l_or_lm, 'mdot_cr')
            yc_llm_mdot_cr = Save_Load_tables.load_table(name, 'yc', str(self.set_coeff) + l_or_lm,
                                                         'mdot_cr', self.set_metal, self.set_bump)
        else:
            name = '{}_{}{}_{}_r_{}'.format('yc', self.set_coeff, l_or_lm, 'mdot_cr', r_cr)
            yc_llm_mdot_cr = Save_Load_tables.load_table(name, 'yc', str(self.set_coeff) + l_or_lm,
                                                         'mdot_cr_r_{}'.format(r_cr), self.set_metal, self.set_bump)

        yc    = yc_llm_mdot_cr[0, 1:]
        llm   = yc_llm_mdot_cr[1:, 0]
        mdot2d= yc_llm_mdot_cr[1:, 1:]

        for i in range(len(yc_vals)):
            if not yc_vals[i] in yc:
                raise ValueError('Value yc_vals[{}] not in yc:\n\t {}'.format(yc_vals[i], yc))

        yc_vals = np.sort(yc_vals, axis=0)

        yc_n = len(yc_vals)

        fig = plt.figure()
        fig.subplots_adjust(hspace=0.2, wspace=0.3)

        for i in range(1, yc_n+1):
            print(i)
            yc_val = yc_vals[i-1]

            ind = Math.find_nearest_index(yc, yc_val)
            mdot = mdot2d[:, ind]

            if yc_n % 2 == 0: ax = fig.add_subplot(2, yc_n/2, i)
            else:             ax = fig.add_subplot(1, yc_n, i)

            ax.plot(mdot, llm, '-',    color='black')
            ax.fill_between(mdot, llm, color="lightgray")

            if r_cr != None:
                ax.text(0.1, 0.9, 'R:{}'.format(r_cr), style='italic',
                        bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center',
                        verticalalignment='center', transform=ax.transAxes)
            if self.set_coeff != 1.0:
                ax.text(0.5, 0.9, 'K:{}'.format(self.set_coeff), style='italic',
                        bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center',
                        verticalalignment='center', transform=ax.transAxes)

            '''=============================OBSERVABELS==============================='''



            # PlotObs.plot_obs_mdot_llm(ax, self.obs, l_or_lm, yc_val)

            '''=============================BACKGROUND================================'''
            if v_n_background != None:
                yc_mdot_llm_z = Save_Load_tables.load_3d_table(self.set_metal,
                                                               'yc_mdot_{}_{}'.format(l_or_lm, v_n_background),
                                                               'yc', 'mdot', l_or_lm, v_n_background, self.set_output_dir)

                yc_ind = Physics.ind_of_yc(yc_mdot_llm_z[:, 0, 0], yc_val)
                mdot_llm_z = yc_mdot_llm_z[yc_ind, :, :]

                PlotBackground.plot_color_background(ax, mdot_llm_z, 'mdot', l_or_lm, v_n_background, 'Yc:{}'.format(yc_val))

        # --- --- WATER MARK --- --- ---
        fig.text(0.95, 0.05, 'PRELIMINARY', fontsize=50, color='gray', ha='right', va='bottom', alpha=0.5)
        plt.legend(bbox_to_anchor=(1, 0), loc='lower right', ncol=1)
        plot_name = self.set_plot_dir + 'minMdot_l.pdf'
        plt.savefig(plot_name)
        plt.show()


    # def save_yc_llm_mdot_cr(self, l_or_lm='l',yc_prec=0.1, depth=100, plot=True):
    #
    #     def interp(x, y, x_grid):
    #         f = interpolate.interp1d(x, y, kind='cubic', bounds_error=False)
    #         return x_grid, f(x_grid)
    #
    #     def univar_interp(x, y, new_x):
    #         # if new_x.min() < x.min():
    #         #     raise ValueError('new_x.min({}) < x.min({})'.format(new_x.min(), x.min()))
    #         # if new_x.max() > x.max():
    #         #     raise ValueError('new_x.min({}) > x.min({})'.format(new_x.max(), x.max()))
    #         if len(x) != len(y):
    #             raise ValueError('len(x)[{}] != len(y)[{}]'.format(len(x), len(y)))
    #         for x_i in x:
    #             if x_i < x[0]:
    #                 raise ValueError('Array x is not monotonically increasing. x_i({}) < x[0]({})'.format(x_i, x[0]))
    #
    #         f = interpolate.UnivariateSpline(x, y)
    #
    #         return f(new_x)
    #
    #     def fit_plynomial(x, y, order, depth, new_x=np.empty(0, )):
    #         '''
    #         RETURNS f(new_x)
    #         :param x:
    #         :param y:
    #         :param order: 1-4 are supported
    #         :return:
    #         '''
    #         f = None
    #         lbl = None
    #
    #         if new_x == np.empty(0, ):
    #             new_x = np.mgrid[(x.min()):(x.max()):depth * 1j]
    #
    #         if order == 1:
    #             fit = np.polyfit(x, y, order)  # fit = set of coeddicients (highest first)
    #             f = np.poly1d(fit)
    #             lbl = '({}) + ({}*x)'.format(
    #                 "%.3f" % f.coefficients[1],
    #                 "%.3f" % f.coefficients[0]
    #             )
    #             # fit_x_coord = np.mgrid[(x.min()):(x.max()):depth*1j]
    #             # plt.plot(fit_x_coord, f(fit_x_coord), '--', color='black')
    #
    #         if order == 2:
    #             fit = np.polyfit(x, y, order)  # fit = set of coeddicients (highest first)
    #             f = np.poly1d(fit)
    #             lbl = '({}) + ({}*x) + ({}*x**2)'.format(
    #                 "%.3f" % f.coefficients[2],
    #                 "%.3f" % f.coefficients[1],
    #                 "%.3f" % f.coefficients[0]
    #             )
    #             # fit_x_coord = np.mgrid[(x.min()):(x.max()):depth*1j]
    #             # plt.plot(fit_x_coord, f(fit_x_coord), '--', color='black')
    #         if order == 3:
    #             fit = np.polyfit(x, y, order)  # fit = set of coeddicients (highest first)
    #             f = np.poly1d(fit)
    #             lbl = '({}) + ({}*x) + ({}*x**2) + ({}*x**3)'.format(
    #                 "%.3f" % f.coefficients[3],
    #                 "%.3f" % f.coefficients[2],
    #                 "%.3f" % f.coefficients[1],
    #                 "%.3f" % f.coefficients[0]
    #             )
    #             # fit_x_coord = np.mgrid[(x.min()):(x.max()):depth*1j]
    #             # plt.plot(fit_x_coord, f(fit_x_coord), '--', color='black')
    #         if order == 4:
    #             fit = np.polyfit(x, y, order)  # fit = set of coeddicients (highest first)
    #             f = np.poly1d(fit)
    #             lbl = '({}) + ({}*x) + ({}*x**2) + ({}*x**3) + ({}*x**4)'.format(
    #                 "%.3f" % f.coefficients[4],
    #                 "%.3f" % f.coefficients[3],
    #                 "%.3f" % f.coefficients[2],
    #                 "%.3f" % f.coefficients[1],
    #                 "%.3f" % f.coefficients[0]
    #             )
    #             # fit_x_coord = np.mgrid[(x.min()):(x.max()):depth*1j]
    #             # plt.plot(fit_x_coord, f(fit_x_coord), '--', color='black')
    #
    #         print(lbl)
    #
    #         return f(new_x)
    #
    #     def common_y(arr1, arr2, arr3):
    #
    #         y1 = arr1[1:, 0]
    #         y2 = arr2[1:, 0]
    #         y3 = arr3[1:, 0]
    #
    #         y_min = np.array([y1.min(), y2.min(), y3.min()]).max()
    #         y_max = np.array([y1.max(), y2.max(), y3.max()]).min()
    #
    #         arr1_cropped = Math.crop_2d_table(arr1, None, None, y_min, y_max)
    #         arr2_cropped = Math.crop_2d_table(arr2, None, None, y_min, y_max)
    #         arr3_cropped = Math.crop_2d_table(arr3, None, None, y_min, y_max)
    #
    #         return arr1_cropped, arr2_cropped, arr3_cropped
    #
    #     def l_lm_mdot_rows(t_lm_rho_opal, lm_l_sp, lm_r_sp, lim_t1, lim_t2):
    #         lm_op = t_lm_rho_opal[1:, 0]
    #         t = t_lm_rho_opal[0, 1:]
    #         rho2d = t_lm_rho_opal[1:, 1:]
    #         new_l = univar_interp(lm_l_sp[0, :], lm_l_sp[1, :], lm_op)
    #         new_r = univar_interp(lm_r_sp[0, :], lm_r_sp[1, :], lm_op)
    #
    #         if len(lm_op) != len(new_l) or len(lm_op) != len(new_r):
    #             raise ValueError('len(lm_op)[{}] != len(new_l)[{}] or len(lm_op)[{}] != len(new_r)[{}]'
    #                              .format(len(lm_op), len(new_l), len(lm_op), len(new_r)))
    #
    #         vrho = Physics.get_vrho(t, rho2d, 2, np.array([1.34]))
    #         m_dot = Physics.vrho_mdot(vrho, new_r, 'l')
    #         mins = Math.get_mins_in_every_row(t, new_l, m_dot, 5000, lim_t1, lim_t2)
    #         mdot_cr = mins[2, :]
    #
    #         return new_l, lm_op, mdot_cr
    #
    #     def x_y_grids(x_arr, y_arr, depth):
    #         x_grid = np.mgrid[x_arr.min():x_arr.max(): depth * 1j]
    #         y_grid = np.mgrid[y_arr.min():y_arr.max(): depth * 1j]
    #         return x_grid, y_grid
    #
    #     t_k_rho = Save_Load_tables.load_table('t_k_rho', 't', 'k', 'rho', self.opal_used, self.output_dir)
    #
    #     spfiles_ = SP_file_work(self.sp_files, self.output_dir, self.plot_dir) # FILES OUGHT TO BE LOADED
    #     min_llm, max_llm = spfiles_.get_min_max(l_or_lm)
    #     yc, spcls = spfiles_.separate_sp_by_crit_val('Yc', yc_prec)
    #
    #     y_grid = np.mgrid[min_llm:max_llm:depth * 1j]
    #
    #     mdot2d_pol = np.zeros(len(y_grid))
    #     mdot2d_int = np.zeros(len(y_grid))
    #
    #     fig = plt.figure(figsize=plt.figaspect(1.0))
    #
    #     ax1 = fig.add_subplot(221)
    #     ax1.grid()
    #     ax1.set_ylabel(Labels.lbls(l_or_lm))
    #     ax1.set_xlabel(Labels.lbls('mdot'))
    #     ax1.set_title('INTERPOLATION')
    #
    #     ax2 = fig.add_subplot(222)
    #     ax2.grid()
    #     ax2.set_ylabel(Labels.lbls(l_or_lm))
    #     ax2.set_xlabel(Labels.lbls('mdot'))
    #     ax2.set_title('EXTRAPOLATION')
    #
    #     for i in range(len(yc)):
    #
    #         # t_l_mdot, t_lm_mdot = self.t_llm_cr_sp(t_k_rho, yc[i], self.opal_used)
    #
    #         if l_or_lm == 'l':
    #             t_llm_mdot = self.t_l_mdot_cr_sp(t_k_rho, yc[i], self.opal_used)
    #         else:
    #             t_llm_mdot = self.t_lm_mdot_cr_sp(t_k_rho, yc[i], self.opal_used)
    #
    #         # t_llm_mdot = self.t_llm_mdot_crit_sp(spcls[i], t_k_rho, l_or_lm, yc[i], self.opal_used)
    #         t = t_llm_mdot[0, 1:]
    #         llm = t_llm_mdot[1:, 0]
    #         m_dot = t_llm_mdot[1:, 1:]
    #
    #         mins = Math.get_mins_in_every_row(t, llm, m_dot, 5000, self.lim_t1, self.lim_t2)
    #
    #         # llm = mins[1, :]
    #         mdot = mins[2, :]
    #
    #         # plt.plot(mins[2, :], mins[1, :], '-', color='black')
    #         # plt.annotate(str("%.2f" % yc[i]), xy=(mins[2, 0], mins[1, 0]), textcoords='data')
    #
    #         '''----------------------------POLYNOMIAL EXTRAPOLATION------------------------------------'''
    #         print('\n\t Yc = {}'.format(yc[i]))
    #
    #         llm_pol, mdot_pol = Math.fit_plynomial(llm, mdot, 3, depth, y_grid)
    #         mdot2d_pol = np.vstack((mdot2d_pol, mdot_pol))
    #         color = 'C' + str(int(yc[i] * 10) - 1)
    #         ax2.plot(mdot_pol, llm_pol, '--', color=color)
    #         ax2.plot(mdot, llm, '.', color=color, label='yc:{}'.format("%.2f" % yc[i]))
    #
    #         # ax2.annotate(str("%.2f" % yc[i]), xy=(mdot_pol[0], llm_pol[0]), textcoords='data')
    #
    #         '''------------------------------INTERPOLATION ONLY---------------------------------------'''
    #         llm_int, mdot_int = interp(llm, mdot, y_grid)
    #         mdot2d_int = np.vstack((mdot2d_int, mdot_int))
    #         ax1.plot(mdot_int, llm_int, '--', color=color)
    #         ax1.plot(mdot, llm, '-', color=color, label='yc:{}'.format("%.2f" % yc[i]))
    #         # ax1.annotate(str("%.2f" % yc[i]), xy=(mdot_int[0], llm_int[0]), textcoords='data')
    #
    #     ax1.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
    #     ax2.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
    #
    #     mdot2d_int = np.delete(mdot2d_int, 0, 0)
    #     mdot2d_pol = np.delete(mdot2d_pol, 0, 0)
    #
    #     yc_llm_mmdot_cr_pol = Math.combine(yc, y_grid, mdot2d_pol.T)  # changing the x/y
    #     yc_llm_mmdot_cr_int = Math.combine(yc, y_grid, mdot2d_int.T)  # changing the x/y
    #
    #     table_name = '{}_{}_{}'.format('yc', l_or_lm, 'mdot_crit')
    #     Save_Load_tables.save_table(yc_llm_mmdot_cr_pol, self.opal_used, table_name, 'yc', l_or_lm, 'mdot_crit')
    #
    #     if plot:
    #         levels = [-7.0, -6.9, -6.8, -6.7, -6.6, -6.5, -6.4, -6.3, -6.2, -6.1, -6.0, -5.9, -5.8,
    #                   -5.7, -5.6, -5.5, -5.4, -5.3, -5.2, -5.1, -5., -4.9, -4.8, -4.7, -4.6, -4.5, -4.4, -4.3,
    #                   -4.2, -4.1, -4.]
    #         ax = fig.add_subplot(223)
    #
    #         # ax = fig.add_subplot(1, 1, 1)
    #         ax.set_xlim(yc_llm_mmdot_cr_int[0, 1:].min(), yc_llm_mmdot_cr_int[0, 1:].max())
    #         ax.set_ylim(yc_llm_mmdot_cr_int[1:, 0].min(), yc_llm_mmdot_cr_int[1:, 0].max())
    #         ax.set_ylabel(Labels.lbls(l_or_lm))
    #         ax.set_xlabel(Labels.lbls('Yc'))
    #
    #         contour_filled = plt.contourf(yc_llm_mmdot_cr_int[0, 1:], yc_llm_mmdot_cr_int[1:, 0],
    #                                       yc_llm_mmdot_cr_int[1:, 1:], levels, cmap=plt.get_cmap('RdYlBu_r'))
    #         # plt.colorbar(contour_filled, label=Labels.lbls('m'))
    #         contour = plt.contour(yc_llm_mmdot_cr_int[0, 1:], yc_llm_mmdot_cr_int[1:, 0], yc_llm_mmdot_cr_int[1:, 1:],
    #                               levels, colors='k')
    #
    #         clb = plt.colorbar(contour_filled)
    #         clb.ax.set_title(Labels.lbls('mdot'))
    #
    #         plt.clabel(contour, colors='k', fmt='%2.2f', fontsize=12)
    #         # ax.set_title('MASS-LUMINOSITY RELATION')
    #
    #         # plt.ylabel(l_or_lm)
    #         # ax.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
    #         # plt.savefig(name)
    #
    #         ax = fig.add_subplot(224)
    #
    #         # ax = fig.add_subplot(1, 1, 1)
    #         ax.set_xlim(yc_llm_mmdot_cr_pol[0, 1:].min(), yc_llm_mmdot_cr_pol[0, 1:].max())
    #         ax.set_ylim(yc_llm_mmdot_cr_pol[1:, 0].min(), yc_llm_mmdot_cr_pol[1:, 0].max())
    #         ax.set_ylabel(Labels.lbls(l_or_lm))
    #         ax.set_xlabel(Labels.lbls('Yc'))
    #
    #         contour_filled = plt.contourf(yc_llm_mmdot_cr_pol[0, 1:], yc_llm_mmdot_cr_pol[1:, 0],
    #                                       yc_llm_mmdot_cr_pol[1:, 1:], levels,
    #                                       cmap=plt.get_cmap('RdYlBu_r'))
    #         # plt.colorbar(contour_filled, label=Labels.lbls('m'))
    #         contour = plt.contour(yc_llm_mmdot_cr_pol[0, 1:], yc_llm_mmdot_cr_pol[1:, 0], yc_llm_mmdot_cr_pol[1:, 1:],
    #                               levels, colors='k')
    #
    #         clb = plt.colorbar(contour_filled)
    #         clb.ax.set_title(Labels.lbls('mdot'))
    #
    #         plt.clabel(contour, colors='k', fmt='%2.2f', fontsize=12)
    #         # ax.set_title('MASS-LUMINOSITY RELATION')
    #
    #         # plt.ylabel(l_or_lm)
    #         # ax.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
    #
    #         plt.show()

class Plot_Critical_Mdot(PlotObs):

    def __init__(self, metal, bump, coef):

        self.set_metal =  metal
        obs_metal = metal

        self.set_coeff = coef
        self.set_bump = bump

        PlotObs.__init__(self, obs_metal, self.set_bump,
                         Files.get_obs_file(obs_metal), Files.get_atm_file(obs_metal))

        self.set_inverse_x = False

        self.set_clump_used = 4
        self.set_clump_modified = 4

        self.set_patches_or_lines = 'lines'
        self.set_patches_or_lines_alpha = 0.5

        self.set_clean              = False
        self.set_use_gaia           = True
        self.set_use_atm_file       = True
        self.set_load_yc_l_lm       = True
        self.set_load_yc_nan_lmlim  = True
        self.set_check_lm_for_wne   = True

        self.set_do_plot_obs_err    = True
        self.set_do_plot_evol_err   = True
        self.set_do_plot_line_fit   = False

        self.set_fill_gray          = False

        self.set_label_sise = 12


    def plot_multiple_natives(self, ax, v_n_x, v_n_y, metals, cr_or_wds, adds):

        if len(metals)!=len(cr_or_wds) or len(metals)!=len(adds):
            raise NameError('len(metals)[{}]!=len(cr_or_wds)[{}]'.format(metals, cr_or_wds))

        x_coords_bounds = np.zeros(2)
        y_coords_bounds = np.zeros(2)
        labels_bounds = np.zeros(2)

        colors = ['blue', 'red', 'green']
        names =  ['Set 1', 'Set 2', 'Set 3']


        for i in range(len(metals)):

            x_coord, y_coord, label = self.plot_native_models(ax, v_n_x, v_n_y,
                                                              metals[i], cr_or_wds[i], adds[i], colors[i], names[i])

            x_coords_bounds = np.vstack((x_coords_bounds, [x_coord[0], x_coord[-1]]))
            y_coords_bounds = np.vstack((y_coords_bounds, [y_coord[0], y_coord[-1]]))
            labels_bounds   = np.vstack((labels_bounds,   [label[0],   label[-1]]))

        x_coords_bounds = np.delete(x_coords_bounds, 0, 0)
        y_coords_bounds = np.delete(y_coords_bounds, 0, 0)
        labels_bounds   = np.delete(labels_bounds,   0, 0)

        # ax.plot(x_coords_bounds[:, 0], y_coords_bounds[:, 0], '--', lw=2, color='black')
        # ax.plot(x_coords_bounds[:, 1], y_coords_bounds[:, 1], '--', lw=2, color='black')
        #
        # ax.plot(x_coords_bounds[0, :], y_coords_bounds[0, :], '--', lw=2, color='black')
        # ax.plot(x_coords_bounds[1, :], y_coords_bounds[1, :], '--', lw=2, color='black')

        # ax.plot(x_coords_bounds[0, 0], y_coords_bounds[0, 0], 'X', lw=2, color='black', label='SET 1')
        # ax.plot(x_coords_bounds[1, 0], y_coords_bounds[1, 0], 'X', lw=2, color='blue',  label='SET 2')
        # ax.plot(x_coords_bounds[2, 0], y_coords_bounds[2, 0], 'X', lw=2, color='green', label='SET 3')
    def plot_native_models(self, ax, v_n_x, v_n_y, metal, cr_or_wd, add='/', color='black', label=None):

        set_sp_files = Files.get_sp_files(metal, cr_or_wd, add) # b1_v2200/

        # spmdl = [] folder = '../data/sp_w_files'

        x_coord = []
        y_coord = []
        labels = []

        for file in set_sp_files:
            # spmdl.append(Read_SP_data_file(file, Files.output_dir, Files.plot_dir))

            cl = Read_SP_data_file(file, Files.output_dir, Files.plot_dir)
            x = cl.get_crit_value(v_n_x)
            y = cl.get_crit_value(v_n_y)
            lbl = cl.get_crit_value('m')

            x_coord = np.append(x_coord, x)
            y_coord = np.append(y_coord, y)
            labels = np.append(labels, lbl)


            ax.plot(x, y, 'X', color=color)
            if not self.set_clean:
                ax.annotate('{}'.format("%.2f" % cl.get_crit_value('m')),
                        xy=(x, y), textcoords='data')

        x_coord, y_coord, labels = Math.x_y_z_sort(x_coord, y_coord, labels)

        ax.plot(x_coord[0], y_coord[0], 'X', color=color, label=label)
        ax.annotate('{}'.format("%.0f" % labels[0]),  xy=(x_coord[0],  y_coord[0]),
                    textcoords='data',
                    horizontalalignment='center',
                    verticalalignment='top',
                    fontsize=self.set_label_size)
        ax.annotate('{}'.format("%.0f" % labels[-1]), xy=(x_coord[-1], y_coord[-1]),
                    textcoords='data',
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    fontsize=self.set_label_size)


        print('FIT TO THE MODELS WITH {} {} {}'.format(metal, cr_or_wd, add))
        x_grid, y_grid = Math.fit_polynomial(x_coord, y_coord, 1, 500)

        # ax.plot(x_grid, y_grid, '-', linewidth=2, color='blue')

        return x_coord, y_coord, labels


    # def plot_natives(self, x_v_n, y_v_n):
    #
    #     fig = plt.figure()
    #     fig.subplots_adjust(hspace=0.2, wspace=0.3)
    #     ax = fig.add_subplot(1, 1, 1)
    #     show_plot = True
    #
    #     self.plot_native_models(ax, x_v_n, y_v_n, self.metal, 'wd')
    #
    #     ax.minorticks_on()
    #     ax.set_xlabel(Labels.lbls(x_v_n), fontsize=self.set_label_sise)
    #     ax.set_ylabel(Labels.lbls(y_v_n), fontsize=self.set_label_sise)
    #
    #     if self.set_inverse_x: ax.invert_xaxis()
    #     plt.xticks(fontsize=self.set_label_sise)
    #     plt.yticks(fontsize=self.set_label_sise)
    #     plt.savefig('tst')
    #     plt.show()

    def get_cr_mdot(self, l_or_lm, yc_val, r_cr = None):
        '''
        Returns mdot, llm (arrs)
        :param l_or_lm:
        :param yc_val:
        :param r_cr:
        :return:
        '''

        if r_cr == None:
            name = '{}_{}{}_{}'.format('yc', self.set_coeff, l_or_lm, 'mdot_cr')
            yc_llm_mdot_cr = Save_Load_tables.load_table(name, 'yc', str(self.set_coeff) + l_or_lm,
                                                         'mdot_cr', self.set_metal, self.set_bump)
        else:
            name = '{}_{}{}_{}_r_{}'.format('yc', self.set_coeff, l_or_lm, 'mdot_cr', r_cr)
            yc_llm_mdot_cr = Save_Load_tables.load_table(name, 'yc', str(self.set_coeff) + l_or_lm,
                                                         'mdot_cr_r_{}'.format(r_cr), self.set_metal, self.set_bump)

        yc = yc_llm_mdot_cr[0, 1:]
        llm = yc_llm_mdot_cr[1:, 0]
        mdot2d = yc_llm_mdot_cr[1:, 1:]

        if not yc_val in yc:
            raise ValueError('Value yc_val[{}] not in yc:\n\t {}'.format(yc_val, yc))


        ind = Math.find_nearest_index(yc, yc_val)
        mdot = mdot2d[:, ind]

        return mdot, llm



    def plot_cr_mdot(self, l_or_lm, yc_val, r_cr = None, ax = None):
        '''
        RETURNS mdot, llm
        :param l_or_lm:
        :param yc_val:
        :param r_cr:
        :param ax:
        :param fill_gray:
        :return:
        '''


        mdot, llm = self.get_cr_mdot(l_or_lm, yc_val, r_cr)

        show_plot = False
        if ax == None: # if the plotting class is not given:
            fig = plt.figure()
            fig.subplots_adjust(hspace=0.2, wspace=0.3)
            ax = fig.add_subplot(1,1,1)
            show_plot = True

        ax.plot(mdot, llm, '-', linewidth=5, color='black')

        if self.set_fill_gray: ax.fill_between(mdot, llm, color="lightgray")

        if r_cr != None and not self.set_clean:
            ax.text(0.1, 0.9, 'R:{}'.format(r_cr), style='italic',
                    bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center',
                    verticalalignment='center', transform=ax.transAxes)

        if self.set_coeff != 1.0 and not self.set_clean:
            ax.text(0.5, 0.9, 'K:{}'.format(self.set_coeff), style='italic',
                    bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center',
                    verticalalignment='center', transform=ax.transAxes)

        if show_plot:
            ax.text(0.95, 0.05, 'PRELIMINARY', fontsize=50, color='gray', ha='right', va='bottom', alpha=0.5)
            ax.legend(bbox_to_anchor=(1, 0), loc='lower right', ncol=1, fontsize=self.set_label_sise)
            plot_name = Files.plot_dir + 'cr_mdot_{}_{}_{}{}.pdf'.format(self.set_metal, self.set_bump, self.set_coeff,
                                                                         l_or_lm)
            ax.set_xlabel(Labels.lbls('mdot'), fontsize=self.set_label_sise)
            ax.set_ylabel(Labels.lbls(l_or_lm), fontsize=self.set_label_sise)
            # plt.grid()
            plt.xticks(fontsize=self.set_label_sise)
            plt.yticks(fontsize=self.set_label_sise)
            plt.savefig(plot_name)
            plt.show()

        return mdot, llm


    def plot_cr_mdot_obs(self, l_or_lm, yc_val, r_cr = None, ax = None, obs_err=True, evol_err=True):

        show_plot = False
        if ax == None: # if the plotting class is not given:
            fig = plt.figure()
            fig.subplots_adjust(hspace=0.2, wspace=0.3)
            ax = fig.add_subplot(1,1,1)
            show_plot = True

            mdot_grid, lm_grid = self.plot_cr_mdot(l_or_lm, yc_val, r_cr, ax)

            # self.plot_fits(ax, self.set_metal, 'mdot_cr', l_or_lm, mdot_grid, ['2lmc', 'smc'],
            #                [self.set_bump, self.set_bump])

        self.plot_obs_all_x_llm(ax, l_or_lm, 'mdot', yc_val, obs_err, evol_err)



        if show_plot:
            ax.text(0.05, 0.05, 'PRELIMINARY', fontsize=50, color='gray', ha='right', va='bottom', alpha=0.5)
            ax.legend(bbox_to_anchor=(1, 0), loc='lower right', ncol=2, fontsize=self.set_label_sise)
            plot_name = Files.plot_dir + 'cr_mdot_{}_{}_{}{}.pdf'.format(self.set_metal, self.set_bump, self.set_coeff,
                                                                         l_or_lm)
            ax.set_xlabel(Labels.lbls('mdot'), fontsize=self.set_label_sise)
            ax.set_ylabel(Labels.lbls(l_or_lm), fontsize=self.set_label_sise)

            # self.plot_native_models(ax, 'mdot', l_or_lm, self.metal, 'wd')

            plt.xticks(fontsize=self.set_label_sise)
            plt.yticks(fontsize=self.set_label_sise)
            # plt.grid()
            plt.savefig(plot_name)
            plt.show()
        else:
            return ax

    def get_yc_min_for_m_init(self, m_in, m_yc_ys_lims):
        m_vals = m_yc_ys_lims[1:, 0]
        yc_vals = m_yc_ys_lims[0, 1:]
        ys_arr = m_yc_ys_lims[1:, 1:]

        m_ind = Math.find_nearest_index(m_vals, m_in)
        # if np.abs(m_in - m_vals[m_ind]) > 0.1: raise ValueError('Correct m_in is not found')

        ys_row = ys_arr[m_ind, ::-1]
        yc_row = yc_vals[::-1]
        ys_zams = ys_arr[m_ind, -1]

        for i in range(len(ys_row)):
            if ys_row[i] < ys_zams:
                print('----------------------- {} {} -------------------------'.format(m_in, yc_vals[i]))
                return yc_row[i - 1]
        print('----------------------- {} -------------------------'.format(yc_row[-1]))
        return yc_row[-1]

        #
        #
        # if not np.round(m_in, 1) in m_vals: raise ValueError('m_init({}) not in m_vals({}) from table file [evol_yc_m_ys]'.format(m_in, m_vals))
        #
        # ind = Math.find_nearest_index(m_vals, m_in)
        # for i in range(len(yc_vals)):
        #     if ys_arr[ind, i] < ys_arr[ind, -1]: # if the surface compostion has changed
        #         print('----------------------- {} {} -------------------------'.format( m_in, yc_vals[i]))
        #         return yc_vals[i-1] # return the yc for which ys has not yet changed
        # print('----------------------- {} -------------------------'.format(yc_vals[0]))
        # return yc_vals[0]
        # # for i in range(len(m_vals)):
    def plot_treks(self, ax, x_v_n, l_or_lm, colorcode=True):

        x_y_z = np.zeros(3)
        if len(self.set_plot_files) > 0:

            m_yc_ys_lims = Save_Load_tables.load_table('evol_yc_m_ys', 'evol_yc', 'm', 'ys', self.set_metal, '')
            plcls = []
            for i in range(len(self.set_plot_files)):
                plcls.append(Read_Plot_file.from_file(self.set_plot_files[i]))

                if l_or_lm == 'l':
                    llm_plot = plcls[i].l_
                else:
                    llm_plot = plcls[i].lm_

                m_init = plcls[i].m_[0]
                y_plot = plcls[i].get_col(x_v_n)
                yc_plot = plcls[i].y_c
                # ys_plot =

                # ================================ PLOTTING THE NORMAN FULL TRACKS =====================================

                # if not self.set_clean:
                #
                #     ax.plot(y_plot, llm_plot, '-', color='gray')
                #
                #     for j in range(10):
                #         ind = Math.find_nearest_index(plcls[i].y_c, (j / 10))
                #         # print(plfl.y_c[i], (i/10))
                #         x_p = y_plot[ind]
                #         y_p = llm_plot[ind]
                #         plt.plot(x_p, y_p, '.', color='red')
                #         if not self.set_clean:
                #             ax.annotate('{} {}'.format("%.2f" % plcls[i].y_c[ind], "%.2f" % plcls[i].mdot_[ind]),
                #                         xy=(x_p, y_p),
                #                         textcoords='data')

                # ================================== PLOTTING ONLY THE WNE PHASE =======================================

                yc_min = self.get_yc_min_for_m_init(m_init, m_yc_ys_lims)

                teff_plot2 = []
                llm_plot2 = []
                yc_plot2 = []
                for k in range(len(yc_plot)):
                    if yc_plot[k] > yc_min:
                        teff_plot2 = np.append(teff_plot2, y_plot[k])
                        llm_plot2 = np.append(llm_plot2, llm_plot[k])
                        yc_plot2 = np.append(yc_plot2, yc_plot[k])
                        x_y_z = np.vstack((x_y_z, [y_plot[k], llm_plot[k], yc_plot[k]]))

                ax.plot(teff_plot2, llm_plot2, '-.', color='black')

                if not self.set_clean:
                    ax.annotate('{}'.format("%0.f" % m_init), xy=(teff_plot2[0], llm_plot2[0]), textcoords='data',
                                horizontalalignment='right')
                    ax.annotate('{}'.format("%.1f" % yc_min), xy=(teff_plot2[-1], llm_plot2[-1]), textcoords='data',
                                horizontalalignment='left')

        if colorcode:
            sc = ax.scatter(x_y_z[1:, 0], x_y_z[1:, 1], c=x_y_z[1:, 2], marker='.', cmap=plt.get_cmap('RdYlBu_r'))

            clb = plt.colorbar(sc)
            clb.ax.set_title(Labels.lbls('Yc'), fontsize=self.set_label_size)

        return ax

    def plot_cr_mdot_obs_trecks(self, l_or_lm, yc_val, r_cr = None, ax = None, fill_gray=True, obs_err=True, evol_err=True):

        self.set_plot_files = Files.get_plot_files(self.metal)

        show_plot = False
        if ax == None: # if the plotting class is not given:
            fig = plt.figure()
            fig.subplots_adjust(hspace=0.2, wspace=0.3)
            ax = fig.add_subplot(1,1,1)
            show_plot = True

        self.plot_cr_mdot(l_or_lm, yc_val, r_cr, ax)

        self.plot_treks(ax, 'mdot', l_or_lm)

        self.plot_obs_all_x_llm(ax, l_or_lm, 'mdot', yc_val, obs_err, evol_err)


        if show_plot:
            # ax.text(0.05, 0.05, 'PRELIMINARY', fontsize=50, color='gray', ha='right', va='bottom', alpha=0.5)
            ax.legend(bbox_to_anchor=(1, 0), loc='lower right', ncol=2, fontsize=self.set_label_sise)
            plot_name = Files.plot_dir + 'cr_mdot_{}_{}_{}{}.pdf'.format(self.set_metal, self.set_bump, self.set_coeff,
                                                                         l_or_lm)
            ax.set_xlabel(Labels.lbls('mdot'), fontsize=self.set_label_sise)
            ax.set_ylabel(Labels.lbls(l_or_lm), fontsize=self.set_label_sise)

            self.plot_native_models(ax, 'mdot', l_or_lm, self.metal, 'wd')

            plt.xticks(fontsize=self.set_label_sise)
            plt.yticks(fontsize=self.set_label_sise)
            # plt.grid()
            plt.savefig(plot_name)
            plt.show()
        else:
            return ax

    def plot_background(self, ax, x_v_n, y_v_n, z_v_n, koef='', yc_val=1.0):
        if z_v_n != None and z_v_n != '':

            name = '{}_{}{}_{}'.format(x_v_n, koef, y_v_n, z_v_n)

            # yc__ts_t_eff_tau_gal

            yc_table = Save_Load_tables.load_3d_table(self.metal, name, 'beta',
                                                      x_v_n, '{}{}'.format(koef, y_v_n), z_v_n, Files.output_dir)

            yc_ind = Physics.ind_of_yc(yc_table[:, 0, 0], yc_val)
            table = yc_table[yc_ind, :, :]

            PlotBackground.plot_color_background(ax, table, x_v_n, y_v_n, z_v_n,
                                                 self.metal, self.bump)


    def plot_fits(self, ax, obs_metal, v_n_x, v_n_y, x_arr, metals, bumps, hatch='///'):

        def get_linestyle(metal):
            if metal == 'gal':
                return '--'

            if metal == '2gal' or metal == '2lmc':
                return '-.'

            if metal == 'lmc':
                return '-'

            if metal == 'smc':
                return ':'

        if len(metals) != len(bumps): raise IOError('Metals != bumps')

        y_arrs = np.zeros(len(x_arr))
        for metal in metals:
            if obs_metal == 'gal':
                y_arr = Fits.get_gal_fit(v_n_x, v_n_y, x_arr, metal, self.set_clump_modified, self.set_use_gaia)
            elif obs_metal == 'lmc':
                y_arr = Fits.get_lmc_fit(v_n_x, v_n_y, x_arr, metal, self.bump)
            else:
                raise IOError('Obs_metal gal and lmc are only supproted so far')

            ax.plot(x_arr, y_arr, get_linestyle(metal), color='black', alpha=self.set_patches_or_lines_alpha)

            y_arrs = np.vstack((y_arrs, y_arr))

        y_arrs = np.delete(y_arrs, 0, 0)

        # ax.fill_between(x_arr, y_arrs[0, :], y_arrs[-1, :], hatch='/', alpha=0.0)
        ax.fill_between(x_arr, y_arrs[0, :], y_arrs[-1, :], color='gray', alpha=0.2)

    def plot_cr_mdot_obs_trecks_back(self, l_or_lm, back_v_n, yc_val, r_cr = None, ax = None, fill_gray=True, obs_err=True, evol_err=True):

        self.set_plot_files = Files.get_plot_files(self.metal)

        show_plot = False
        if ax == None: # if the plotting class is not given:
            fig = plt.figure()
            fig.subplots_adjust(hspace=0.2, wspace=0.3)
            ax = fig.add_subplot(1,1,1)
            show_plot = True

        # self.plot_multiple_natives(ax, 'mdot', 'lm', ['2gal', 'gal', 'gal'], ['wd', 'wd', 'wd'], ['/', '/', 'b1_v2200/'])

        self.plot_background(ax, 'mdot', l_or_lm, back_v_n, '', 1.0)

        mdot_arr, lm_arr = self.plot_cr_mdot(l_or_lm, yc_val, r_cr, ax)

        mdot_grid = np.mgrid[ax.get_xlim()[0]:ax.get_xlim()[-1]:100j]
        # self.plot_fits(ax, self.set_metal, 'mdot_cr', l_or_lm, mdot_grid, ['2lmc', 'smc'],
        #                [self.set_bump, self.set_bump])

        self.plot_treks(ax, 'mdot', l_or_lm, False)

        self.plot_obs_all_x_llm(ax, l_or_lm, 'mdot', yc_val, obs_err, evol_err)

        # self.plot_native_models(ax, 'mdot', l_or_lm, self.metal, 'wd')

        print('\n CRITICAL MDOT FIT: ')
        mdot_arr, lm_arr = Math.fit_polynomial(mdot_arr, lm_arr, 4, 500)

        if show_plot:
            # ax.text(0.05, 0.05, 'PRELIMINARY', fontsize=50, color='gray', ha='right', va='bottom', alpha=0.5)
            ax.legend(bbox_to_anchor=(1, 0), loc='lower right', ncol=3, fontsize=self.set_label_sise)
            plot_name = Files.plot_dir + 'cr_mdot_{}_{}_{}{}.pdf'.format(self.set_metal, self.set_bump, self.set_coeff,
                                                                         l_or_lm)
            ax.set_xlabel(Labels.lbls('mdot'), fontsize=self.set_label_sise)
            ax.set_ylabel(Labels.lbls(l_or_lm), fontsize=self.set_label_sise)



            plt.xticks(fontsize=self.set_label_sise)
            plt.yticks(fontsize=self.set_label_sise)
            # plt.grid()
            plt.savefig(plot_name)
            plt.show()
        else:
            return ax


    def plot_cr_mdot_sp_set(self, l_or_lm, yc_vals, r_cr = None, ax=None):
        '''
        PLOTS the set of llm(mdot), with l->m relation assumed from yc_vals.
        :param l_or_lm:
        :param yc_vals:
        :return:
        '''



        if r_cr == None:
            name = '{}_{}{}_{}'.format('yc', self.set_coeff, l_or_lm, 'mdot_cr')
            yc_llm_mdot_cr = Save_Load_tables.load_table(name, 'yc', str(self.set_coeff) + l_or_lm,
                                                         'mdot_cr', self.set_metal, self.set_bump)
        else:
            name = '{}_{}{}_{}_r_{}'.format('yc', self.set_coeff, l_or_lm, 'mdot_cr', r_cr)
            yc_llm_mdot_cr = Save_Load_tables.load_table(name, 'yc', str(self.set_coeff) + l_or_lm,
                                                         'mdot_cr_r_{}'.format(r_cr), self.set_metal, self.set_bump)

        yc    = yc_llm_mdot_cr[0, 1:]
        llm   = yc_llm_mdot_cr[1:, 0]
        mdot2d= yc_llm_mdot_cr[1:, 1:]

        for i in range(len(yc_vals)):
            if not yc_vals[i] in yc:
                raise ValueError('Value yc_vals[{}] not in yc:\n\t {}'.format(yc_vals[i], yc))

        yc_vals = np.sort(yc_vals, axis=0)

        yc_n = len(yc_vals)

        fig = plt.figure()
        fig.subplots_adjust(hspace=0.2, wspace=0.3)

        for i in range(1, yc_n+1):
            print(i)
            yc_val = yc_vals[i-1]

            ind = Math.find_nearest_index(yc, yc_val)
            mdot = mdot2d[:, ind]

            if yc_n % 2 == 0: ax = fig.add_subplot(2, yc_n/2, i)
            else:             ax = fig.add_subplot(1, yc_n, i)

            ax.plot(mdot, llm, '-',    color='black')
            ax.fill_between(mdot, llm, color="lightgray")

            if r_cr != None:
                ax.text(0.1, 0.9, 'R:{}'.format(r_cr), style='italic',
                        bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center',
                        verticalalignment='center', transform=ax.transAxes)
            if self.set_coeff != 1.0:
                ax.text(0.5, 0.9, 'K:{}'.format(self.set_coeff), style='italic',
                        bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center',
                        verticalalignment='center', transform=ax.transAxes)

            '''=============================OBSERVABELS==============================='''



            # PlotObs.plot_obs_mdot_llm(ax, self.obs, l_or_lm, yc_val)

            '''=============================BACKGROUND================================'''
            if False:
                yc_mdot_llm_z = Save_Load_tables.load_3d_table(self.set_metal,
                                                               'yc_mdot_{}_{}'.format(l_or_lm, v_n_background),
                                                               'yc', 'mdot', l_or_lm, v_n_background, self.set_output_dir)

                yc_ind = Physics.ind_of_yc(yc_mdot_llm_z[:, 0, 0], yc_val)
                mdot_llm_z = yc_mdot_llm_z[yc_ind, :, :]

                PlotBackground.plot_color_background(ax, mdot_llm_z, 'mdot', l_or_lm, v_n_background, 'Yc:{}'.format(yc_val))

        # --- --- WATER MARK --- --- ---
        fig.text(0.95, 0.05, 'PRELIMINARY', fontsize=50, color='gray', ha='right', va='bottom', alpha=0.5)
        plt.legend(bbox_to_anchor=(1, 0), loc='lower right', ncol=1)
        plot_name = Files.plot_dir + 'cr_mdot_{}_{}_{}{}.pdf'.format(self.set_metal, self.set_bump, self.set_coeff,
                                                                          l_or_lm)
        plt.savefig(plot_name)
        plt.show()

    def save_stars_affiliation(self, l_or_lm='lm', yc_val=1.0, r_cr=None):


        ax = None

        show_plot = False
        if ax == None:  # if the plotting class is not given:
            fig = plt.figure()
            fig.subplots_adjust(hspace=0.2, wspace=0.3)
            ax = fig.add_subplot(1, 1, 1)
            show_plot = True



        mdot_cr, lm_cr = self.plot_cr_mdot(l_or_lm, yc_val, r_cr, ax, True)

        stars_n, mdot_star, lm_star = self.plot_obs_all_x_llm(ax, l_or_lm, 'mdot', yc_val,
                                                              self.set_do_plot_obs_err, self.set_do_plot_evol_err)
        res = []
        res2 = []
        # for i in range(len(lm_cr)):
        #     star_ind = Math.find_nearest_index(lm_star, lm_cr[i])
        #     print('star{}: lm:{} >< {} :cr | mdot: {} >< {} :cr '
        #           .format(stars_n[star_ind], lm_star[star_ind], lm_cr[i], mdot_star[star_ind], mdot_cr[i]))
        #     if mdot_star[star_ind] > mdot_cr[i]:
        #         res = np.append(res, np.int(stars_n[star_ind]))

        mdot_star_int = interpolate.interp1d(lm_cr, mdot_cr, kind='linear', bounds_error=True)(lm_star)

        for i in range(len(stars_n)):
            # print('star{}: lm:{} >< {} :cr | mdot: {} >< {} :cr '
            #           .format(stars_n[star_ind], lm_star[star_ind], lm_cr[i], mdot_star[star_ind], mdot_cr[i]))
            if mdot_star[i] >= mdot_star_int[i]:# and lm_star[i] >= lm_cr[i]:
                res = np.append(res, np.int(stars_n[i]))
            else:
                res2 = np.append(res2, np.int(stars_n[i]))

        # for i in range(len(stars_n)):



        # for i in range(len(stars_n)):
        #     lm_coord = Math.find_nearest_index(lm_cr, lm_star[i])
        #     mdot_coord = Math.find_nearest_index(mdot_cr, mdot_star[i])
        #
        #     if lm_star[i] >= lm_coord and mdot_star[i] >= mdot_coord:
        #         res = np.append(res, stars_n[i])

        # sys.exit('ABOVE crit: {} \n BELOW crit {}'.format(np.unique(res), np.unique(res2)))

        if show_plot:
            ax.text(0.05, 0.05, 'PRELIMINARY', fontsize=50, color='gray', ha='right', va='bottom', alpha=0.5)
            ax.legend(bbox_to_anchor=(1, 0), loc='lower right', ncol=2, fontsize=self.set_label_sise)
            plot_name = Files.plot_dir + 'cr_mdot_{}_{}_{}{}.pdf'.format(self.set_metal, self.set_bump, self.set_coeff,
                                                                         l_or_lm)
            ax.set_xlabel(Labels.lbls('mdot'), fontsize=self.set_label_sise)
            ax.set_ylabel(Labels.lbls(l_or_lm), fontsize=self.set_label_sise)

            self.plot_native_models(ax, 'mdot', l_or_lm, self.metal, 'wd')

            plt.xticks(fontsize=self.set_label_sise)
            plt.yticks(fontsize=self.set_label_sise)
            # plt.grid()
            plt.savefig(plot_name)
            plt.show()
        else:
            return ax

    def plot_mult_2d_betas(self, x_v_n, y_v_n, z_v_n, betas_to_plot, yc_val = 1.0, r_cr = None):

        b_x_y_z = Save_Load_tables.load_3d_table(self.set_metal, 'Fe', 'beta_{}_{}_{}'
                                                 .format(x_v_n, y_v_n, z_v_n), 'beta', x_v_n,
                                                 y_v_n, z_v_n)

        b_x_y_z2 = Save_Load_tables.load_3d_table(self.set_metal, 'HeII', 'beta_{}_{}_{}'
                                                 .format(x_v_n, y_v_n, z_v_n), 'beta', x_v_n,
                                                 y_v_n, z_v_n)
        depth = len(b_x_y_z[0,1:,1:])
        b_x_y_z = np.append( b_x_y_z , b_x_y_z2)
        b_x_y_z = np.reshape(b_x_y_z, (3, depth + 1, depth + 1))
        # b_x_y_z[0, :, :] = b_x_y_z2[0, :, :]

        betas = b_x_y_z[:, 0, 0]
        for i in range(len(betas_to_plot)):
            if not betas_to_plot[i] in betas:
                raise ValueError('Value betas_to_plot[{}] not in betas:\n\t {}'.format(betas_to_plot[i], betas))

        # ------------------------

        def plot_background(ax, table, metal, z_v_n):
            # from PhysMath import Levels
            levels = Levels.get_levels(z_v_n, metal, '')

            contour_filled = ax.contourf(table[0, 1:], table[1:, 0], table[1:, 1:], levels, cmap=plt.get_cmap('RdYlBu_r'))
            # clb = plt.colorbar(contour_filled)
            # clb.ax.set_title(Labels.lbls('mdot'))

            contour = ax.contour(table[0, 1:], table[1:, 0], table[1:, 1:], levels, colors='k')
            # ax.clabel(contour, colors='k', fmt='%2.2f', fontsize=12)
            # ax.set_title('SONIC HR DIAGRAM')
            return ax, contour_filled


        from mpl_toolkits.axes_grid1 import make_axes_locatable

        fig = plt.figure()
        # fig.set_size_inches(18.5, 10.5)
        fig.set_size_inches(3*4.79, 4.79)

        ax1 = fig.add_subplot(1, 3, 1 ) # aspect='equal'
        ax2 = fig.add_subplot(1, 3, 2, sharey=ax1, sharex=ax1)  # Share y-axes with subplot 1
        ax3 = fig.add_subplot(1, 3, 3, sharey=ax1, sharex=ax1)

        # Set y-ticks of subplot 2 invisible
        plt.setp(ax2.get_yticklabels(), visible=False)
        plt.setp(ax3.get_yticklabels(), visible=False)

        i1 = Math.find_nearest_index(b_x_y_z[:, 0, 0], betas_to_plot[0])
        i2 = Math.find_nearest_index(b_x_y_z[:, 0, 0], betas_to_plot[1])
        i3 = Math.find_nearest_index(b_x_y_z[:, 0, 0], betas_to_plot[2])

        # --- Plot BackGrounds
        im1, cf1 = plot_background(ax1, b_x_y_z[i1,:,:], self.metal, z_v_n)
        ax1.text(0.9, 0.1, r'$\beta: $'+'{}'.format(b_x_y_z[i1,0,0]), style='italic',
                    bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center',
                    verticalalignment='center', transform=ax1.transAxes)
        im2, cf2 = plot_background(ax2, b_x_y_z[i2,:,:], self.metal, z_v_n)
        ax2.text(0.9, 0.1, r'$\beta: $'+'{}'.format(b_x_y_z[i2,0,0]), style='italic',
                    bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center',
                    verticalalignment='center', transform=ax2.transAxes)
        im3, cf3 = plot_background(ax3, b_x_y_z[i3,:,:], self.metal, z_v_n)
        ax3.text(0.9, 0.1, r'$\beta: $'+'{}'.format(b_x_y_z[i3,0,0]), style='italic',
                    bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center',
                    verticalalignment='center', transform=ax3.transAxes)

        # --- Plot Obs
        ax1_stars = self.plot_obs_all_x_llm(ax1, y_v_n, x_v_n, yc_val, True, False)
        # ax1.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=2)

        ax2_stars = self.plot_obs_all_x_llm(ax2, y_v_n, x_v_n, yc_val, True, False)
        # ax2_stars.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=2)

        ax3_stars = self.plot_obs_all_x_llm(ax3, y_v_n, x_v_n, yc_val, True, True)
        ax3.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=3)


        # Define locations of colorbars for both subplot 1 and 2
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)

        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)

        divider3 = make_axes_locatable(ax3)
        cax3 = divider3.append_axes("right", size="5%", pad=0.05)


        mdot_arr, lm_arr = self.plot_cr_mdot(y_v_n, yc_val, r_cr, ax1)
        mdot_arr, lm_arr = self.plot_cr_mdot(y_v_n, yc_val, r_cr, ax2)
        mdot_arr, lm_arr = self.plot_cr_mdot(y_v_n, yc_val, r_cr, ax3)


        # Create and remove the colorbar for the first subplot

        cbar1 = fig.colorbar(cf1, cax=cax1)
        # fig.delaxes(fig.axes[2])
        cbar1.remove()
        #
        # # Create second colorbar
        cbar2 = fig.colorbar(cf2, cax=cax2)
        # fig.delaxes(fig.axes[2])
        cbar2.remove()

        cbar3 = fig.colorbar(cf3, cax=cax3)
        cbar3.ax.set_title(Labels.lbls(z_v_n))
        cbar3.ax.tick_params(labelsize=self.set_label_sise)

        # --- --- WATER MARK -- -- --
        # fig.text(0.95, 0.05, 'PRELIMINARY', fontsize=50, color='gray', ha='right', va='bottom', alpha=0.5)

        # Adjust the widths between the subplots
        # plt.title('SONIC HR DIAGRAM', loc='center')

        ax1.set_ylabel(Labels.lbls(y_v_n), fontsize=self.set_label_sise)

        ax1.tick_params('y', labelsize=self.set_label_sise)
        ax1.tick_params('x', labelsize=self.set_label_sise)
        ax1.set_xlabel(Labels.lbls(x_v_n), fontsize=self.set_label_sise)

        ax2.tick_params('y', labelsize=self.set_label_sise)
        ax2.tick_params('x', labelsize=self.set_label_sise)
        ax2.set_xlabel(Labels.lbls(x_v_n), fontsize=self.set_label_sise)

        ax3.tick_params('y', labelsize=self.set_label_sise)
        ax3.tick_params('x', labelsize=self.set_label_sise)
        ax3.set_xlabel(Labels.lbls(x_v_n), fontsize=self.set_label_sise)

        plt.minorticks_on()
        ax1.minorticks_on()
        ax2.minorticks_on()
        ax3.minorticks_on()

        # ax1.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=2)
        # plt.figlegend(bbox_to_anchor=(0, 0), loc='lower left', ncol=2)
        plt.subplots_adjust(wspace=-1.1)
        plt.tight_layout()
        # ax1.invert_xaxis()
        # ax2.invert_xaxis()
        plt.show()


        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        # bg = PlotBackground2()
        # bg.set_auto_limits = False
        #
        #
        # b_x_y_z = Save_Load_tables.load_3d_table(self.metal, '', 'beta_{}_{}_{}'
        #                                          .format(x_v_n, y_v_n, z_v_n), 'beta', x_v_n,
        #                                          y_v_n, z_v_n)
        # betas = b_x_y_z[:, 0, 0]
        # for i in range(len(betas_to_plot)):
        #     if not betas_to_plot[i] in betas:
        #         raise ValueError('Value betas_to_plot[{}] not in betas:\n\t {}'.format(betas_to_plot[i], betas))
        #
        # def plot_all(ax, x_y_z, x_v_n, y_v_n, z_v_n, metal):
        #
        #     # if label != None:
        #     #     print('TEXT')
        #
        #     # ax.text(table[0, 1:].min(), table[1:, 0].min(), s=label)
        #     # bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10}
        #     # plt.text(2, 0.65, r'$\cos(2 \pi t) \exp(-t)$')
        #
        #     # ax = fig.add_subplot(1, 1, 1)
        #     # if self.set_auto_limits:
        #     #     ax.set_xlim(table[0, 1:].min(), table[0, 1:].max())
        #     #     ax.set_ylim(table[1:, 0].min(), table[1:, 0].max())
        #     # ax.set_ylabel(Labels.lbls(v_n_y), fontsize=self.set_label_sise)
        #     # ax.set_xlabel(Labels.lbls(v_n_x), fontsize=self.set_label_sise)
        #
        #     levels = Levels.get_levels(z_v_n, metal, '')
        #
        #     contour_filled = plt.contourf(x_y_z[0, 1:], x_y_z[1:, 0], x_y_z[1:, 1:], levels,
        #                                   cmap=plt.get_cmap('RdYlBu_r'),
        #                                   alpha=1.0)
        #     # clb = plt.colorbar(contour_filled)
        #     # clb.ax.tick_params(labelsize=self.set_label_sise)
        #     # clb.ax.set_title(Labels.lbls(v_n_z), fontsize=self.set_label_sise)
        #
        #     # ax.colorbar(contour_filled, label=Labels.lbls(v_n_z))
        #
        #     # if self.set_show_contours:
        #     #     contour = plt.contour(table[0, 1:], table[1:, 0], table[1:, 1:], levels, colors='k')
        #     #
        #     #     labs = ax.clabel(contour, colors='k', fmt='%2.2f', fontsize=self.set_label_sise)
        #     #     if self.set_rotate_labels != None:
        #     #         for lab in labs:
        #     #             lab.set_rotation(self.set_rotate_labels)  # ORIENTATION OF LABELS IN COUNTUR PLOTS
        #     # ax.set_title('SONIC HR DIAGRAM')
        #
        #     # print('Yc:{}'.format(yc_val))
        #     # if not self.set_clean and label != None and label != '':
        #     #     ax.text(0.9, 0.1, label, style='italic',
        #     #             bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center',
        #     #             verticalalignment='center', transform=ax.transAxes)
        #
        #     ax.tick_params('y', labelsize=self.set_label_sise)
        #     ax.tick_params('x', labelsize=self.set_label_sise)
        #
        #     # plt.minorticks_on()
        #
        #     # plt.ylabel(l_or_lm)
        #     # plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
        #     # plt.savefig(name)
        #     # plt.show()
        #     return ax
        #
        #
        #     # ax = bg.plot_color_background(ax, x_y_z, x_v_n, y_v_n, z_v_n, metal, '', None)
        #
        #     # return ax
        #
        # from mpl_toolkits.axes_grid1 import ImageGrid
        #
        # # Set up figure and image grid
        # fig = plt.figure(figsize=(9.75, 3))
        #
        # grid = ImageGrid(fig, 111,  # as in plt.subplot(111)
        #                  nrows_ncols=(1, len(betas)),
        #                  axes_pad=0.15,
        #                  share_all=True,
        #                  cbar_location="right",
        #                  cbar_mode="single",
        #                  cbar_size="7%",
        #                  cbar_pad=0.15,
        #                  )
        #
        # # Add data to image grid
        # i = 0
        # for ax in grid:
        #
        #     x_y_z = b_x_y_z[i, :, :]
        #     im = plot_all(ax, x_y_z, x_v_n, y_v_n, z_v_n, self.metal)
        #     i = i + 1
        #     # im = ax.imshow(np.random.random((10, 10)), vmin=0, vmax=1)
        #
        # # Colorbar
        # ax.cax.colorbar(im)
        # ax.cax.toggle_label(True)
        #
        # # plt.tight_layout()    # Works, but may still require rect paramater to keep colorbar labels visible
        # plt.show()

    def plot_cr_3d_betas(self, x_v_n, y_v_n, z_v_n, betas_to_plot, yc_val = 1.0, r_cr = None):

        fsz = 12
        alpha = 1.0


        b_x_y_z = Save_Load_tables.load_3d_table(self.metal, '', 'beta_{}_{}_{}'
                                                       .format(x_v_n, y_v_n, z_v_n), 'beta', x_v_n,
                                                       y_v_n, z_v_n)
        betas = b_x_y_z[:, 0, 0]

        for i in range(len(betas_to_plot)):
            if not betas_to_plot[i] in betas:
                raise ValueError('Value betas_to_plot[{}] not in betas:\n\t {}'.format(betas_to_plot[i], betas))

        mdot, llm = self.get_cr_mdot(y_v_n, yc_val, r_cr)

        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure() # figsize=(10, 5) changes the overall size of the popping up window
        ax = fig.gca(projection='3d')
        # ax = fig.add_subplot(111, projection='3d')
        pg = PlotBackground2()
        # pg.plot_3d_curved_surf(ax, mdot, llm, betas)
        pg.plot_3d_back2(ax, b_x_y_z, x_v_n, y_v_n, z_v_n, self.metal)
        #self.plot_3d_obs_all_x_llm(ax, x_v_n, y_v_n, yc_val, b_x_y_z[:,0,0])# np.array([b_x_y_z[0,0,0]])


        # ax.set_zlim(llm.min(), llm.max())
        # ax.set_xlim(-3.5,-6.0)

        plt.tight_layout(0.01, 0.01, 0.01) # <0.1 lets you to malke it tighter
        plt.minorticks_on()
        plt.show()



        beta_vals = np.sort(betas_to_plot, axis=0)

        b_n = len(betas_to_plot)


class Plot_Multiple_Crit_Mdots(PlotObs):

    output_dir = Files.output_dir
    plot_dir = Files.plot_dir

    def __init__(self,  l_or_lm):
        '''
        Table name expected: ' yc_0.8lm_mdot_cr_r_1.0__HeII_table_x '
        :param tables:
        :param yc:
        :param obs_files:
        '''
        self.set_y_coord = l_or_lm
        self.set_metal = []
        self.set_coeff = []
        self.set_bump = []
        self.set_r_cr = []
        self.set_yc = []

        # zeroes file goes for observables
        PlotObs.__init__(self, self.set_metal[0],
                         Files.get_obs_file(self.set_metal[0]), Files.get_atm_file(self.set_metal[0]))


        self.set_clean              = False
        self.set_use_gaia           = True
        self.set_use_atm_file       = True
        self.set_load_yc_l_lm       = True
        self.set_load_yc_nan_lmlim  = True
        self.set_check_lm_for_wne   = True

        self.set_do_plot_obs_err    = True
        self.set_do_plot_evol_err   = True
        self.set_do_plot_line_fit   = True


    def plot_crit_mdots(self, clean = False, v_n_background = None, plfls=list()):

        n = len(self.set_y_coord)

        if len(self.set_coeff) != n or len(self.set_r_cr) != n or len(self.set_metal) != n or len(self.set_bump) != n or len(self.set_yc) != n:
            raise ValueError('Length of all input array must me the same. Given: {}')

        fig = plt.figure(figsize=plt.figaspect(0.8))
        ax = fig.add_subplot(111)  # , projection='3d'

        lbl = []
        min_lm = []
        max_lm = []
        min_mdot = []
        max_mdot = []
        for i in range(n):

            l_or_lm = self.set_y_coord[i]
            coeff   = self.set_coeff[i]
            r_cr    = self.set_r_cr[i]
            metal    = self.set_metal[i]
            bump    = self.set_bump[i]
            yc      = self.set_yc[i]
            z = Get_Z.z(metal)

            if r_cr == None:
                name = '{}_{}{}_{}'.format('yc', coeff, l_or_lm, 'mdot_cr')
                lbl.append( 'z:{}({}) K:{}'.format(z, bump, coeff) )
                yc_llm_mdot_cr = Save_Load_tables.load_table(name, 'yc', str(coeff) + l_or_lm,
                                                             'mdot_cr', metal, bump)
            else:
                name = '{}_{}{}_{}_r_{}'.format('yc', coeff, l_or_lm, 'mdot_cr', r_cr)
                lbl.append( 'z:{}({}) K:{} R:{}'.format(z, bump, coeff, r_cr) )
                yc_llm_mdot_cr = Save_Load_tables.load_table(name, 'yc', str(coeff) + l_or_lm,
                                                             'mdot_cr_r_{}'.format(r_cr), metal, bump)

            yc_arr = yc_llm_mdot_cr[0,  1:]
            llm    = yc_llm_mdot_cr[1:, 0]
            mdot2d = yc_llm_mdot_cr[1:, 1:]

            for j in range(len(yc_arr)):
                if not yc in yc_arr:
                    raise ValueError('Value yc_vals[{}] not in yc:\n\t {}'.format(yc_arr[j], yc))

            ind = Math.find_nearest_index(yc_arr, yc)
            mdot = mdot2d[:, ind]
            min_mdot = np.append(min_mdot, mdot.min())
            max_mdot = np.append(max_mdot, mdot.max())
            min_lm = np.append(min_lm, llm.min())
            max_lm = np.append(max_lm, llm.max())

            ax.plot(mdot, llm, '-', color='C'+str(i+1))
            ax.fill_between(mdot, llm, color="lightgray")

        if not clean:
            for i in range(n):
                ax.text(0.3, 0.9-(i/10), lbl[i], style='italic',
                        bbox={'facecolor': 'C'+str(i+1), 'alpha': 0.5, 'pad': 5}, horizontalalignment='center',
                        verticalalignment='center', transform=ax.transAxes)

        self.plot_obs_all_x_llm(ax, self.set_y_coord, 'mdot', self.set_yc[0], True, True)

        # for i in range(len(self.obs_files)):
        #     l_or_lm = self.set_y_coord[i]
        #     yc = self.set_yc[i]
        #
        #     PlotBackground.plot_obs_mdot_llm(ax, self.obs[i], l_or_lm, yc, clean)

        # --- --- BACKGROUND --- --- ---
        if v_n_background != None:
            yc_mdot_llm_z = Save_Load_tables.load_3d_table(self.set_metal[0],
                                                           'yc_mdot_{}_{}'.format(self.set_y_coord[0], v_n_background),
                                                           'yc', 'mdot', self.set_y_coord[0], v_n_background, self.output_dir)

            yc_ind = Physics.ind_of_yc(yc_mdot_llm_z[:, 0, 0], self.set_yc[0])
            mdot_llm_z = yc_mdot_llm_z[yc_ind, :, :]

            # mdot_llm_z = Math.extrapolate(mdot_llm_z, 10, 0, 0, 0, 500)

            PlotBackground.plot_color_background(ax, mdot_llm_z, 'mdot', self.set_y_coord[0], v_n_background, self.set_metal[0], '', 0.6, clean)


        # --- --- WATER MARK -- -- --
        fig.text(0.95, 0.05, 'PRELIMINARY', fontsize=50, color='gray', ha='right', va='bottom', alpha=0.5)

        # --- --- TESTINGS --- --- ---

        def get_yc_min_for_m_init(m_in, m_yc_ys_lims):
            m_vals = m_yc_ys_lims[1:, 0]
            yc_vals = m_yc_ys_lims[0, 1:]
            ys_arr = m_yc_ys_lims[1:, 1:]


            m_ind = Math.find_nearest_index(m_vals, m_in)
            if np.abs(m_in - m_vals[m_ind]) > 0.1: raise ValueError('Correct m_in is not found')

            ys_row = ys_arr[m_ind, ::-1]
            yc_row = yc_vals[::-1]
            ys_zams= ys_arr[m_ind, -1]



            for i in range(len(ys_row)):
                if ys_row[i] < ys_zams:
                    print('----------------------- {} {} -------------------------'.format(m_in, yc_vals[i]))
                    return yc_row[i-1]
            print('----------------------- {} -------------------------'.format(yc_row[-1]))
            return yc_row[-1]


            #
            #
            # if not np.round(m_in, 1) in m_vals: raise ValueError('m_init({}) not in m_vals({}) from table file [evol_yc_m_ys]'.format(m_in, m_vals))
            #
            # ind = Math.find_nearest_index(m_vals, m_in)
            # for i in range(len(yc_vals)):
            #     if ys_arr[ind, i] < ys_arr[ind, -1]: # if the surface compostion has changed
            #         print('----------------------- {} {} -------------------------'.format( m_in, yc_vals[i]))
            #         return yc_vals[i-1] # return the yc for which ys has not yet changed
            # print('----------------------- {} -------------------------'.format(yc_vals[0]))
            # return yc_vals[0]
            # # for i in range(len(m_vals)):


        if len(plfls) > 1:
            m_yc_ys_lims = Save_Load_tables.load_table('evol_yc_m_ys', 'evol_yc', 'm', 'ys', self.opal_def[0], '')
            plcls = []
            for i in range(len(plfls)):
                plcls.append(Read_Plot_file.from_file(plfls[i]))
                yc_plot = plcls[i].y_c
                lm_plot = plcls[i].lm_
                m_init = plcls[i].m_[0]
                mdot_plot = plcls[i].mdot_

                yc_min = get_yc_min_for_m_init(m_init, m_yc_ys_lims)

                mdot_plot2 = []
                lm_plot2 = []
                for i in range(len(yc_plot)):
                    if yc_plot[i] > yc_min:
                        mdot_plot2 = np.append(mdot_plot2, mdot_plot[i])
                        lm_plot2 = np.append(lm_plot2, lm_plot[i])

                ax.plot(mdot_plot2, lm_plot2, '-.', color='black')
                ax.annotate('M:{}'.format("%.1f"%m_init), xy=(mdot_plot2[0], lm_plot2[0]), textcoords='data')

        plt.xlim(min_mdot.min(), max_mdot.max())
        plt.ylim(min_lm.min(), max_lm.max())
        plt.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1, framealpha=0.1)
        plt.legend()
        plt.show()


from FilesWork import PlotBackground2, Fits

class Plot_Sonic_HRD(PlotObs, PlotBackground2):

    def __init__(self, metal, bump, coeff):

        self.lim_t1, self.lim_t2 =   T_kappa_bump.t_for_bump(bump) # 5.25, 5.45 #

        self.set_exrtrapolation = [0, 0, 0, 30] # <-, ->, v, ^
        self.set_show_extrap_borders = True

        obs_metal = metal
        PlotObs.__init__(self, obs_metal, bump, Files.get_obs_file(obs_metal), Files.get_atm_file(obs_metal))
        PlotBackground2.__init__(self)

        self.set_metal  = metal
        self.set_coeff  = coeff
        self.set_bump   = bump

        self.set_clump_used = 4
        self.set_clump_modified = 4

        self.set_clean              = False
        self.set_use_gaia           = True
        self.set_use_atm_file       = True
        self.set_load_yc_l_lm       = True
        self.set_load_yc_nan_lmlim  = True
        self.set_check_lm_for_wne   = True
        self.set_check_affiliation  = False

        self.set_patches_or_lines   = 'lines' # lines2 are X shaped
        self.set_patches_or_lines_alpha = 0.5
        self.set_patches_or_lines_alpha_fits = 0.3

        self.set_do_plot_obs_err    = True
        self.set_do_plot_evol_err   = True
        self.set_do_plot_line_fit   = True

        self.set_if_evol_err_out = 't1' # t1 for Fe bump; t2 for HeII bump (inverse)

        self.set_ncol_legend=3
        self.set_label_sise=12
        self.set_rotate_labels=295
        self.set_alpha=1.0
        self.set_show_contours=False

        self.set_list_metals_fit = []#['lmc', 'gal', '2gal']   # 'smc', ['lmc', 'gal', '2gal'] # ['lmc', 'gal', '2gal']
        self.set_list_bumps_fit  =  []#[self.bump,self.bump,self.bump]

    # INTERNAL METHODS
    def plot_multiple_natives(self, ax, v_n_x, v_n_y, metals, cr_or_wds, adds):

        if len(metals)!=len(cr_or_wds) or len(metals)!=len(adds):
            raise NameError('len(metals)[{}]!=len(cr_or_wds)[{}]'.format(metals, cr_or_wds))

        x_coords_bounds = np.zeros(2)
        y_coords_bounds = np.zeros(2)
        labels_bounds = np.zeros(2)

        colors = ['blue', 'red', 'green']
        names =  ['Set 1', 'Set 2', 'Set 3']


        for i in range(len(metals)):

            x_coord, y_coord, label = self.plot_native_models(ax, v_n_x, v_n_y,
                                                              metals[i], cr_or_wds[i], adds[i], colors[i], names[i])

            x_coords_bounds = np.vstack((x_coords_bounds, [x_coord[0], x_coord[-1]]))
            y_coords_bounds = np.vstack((y_coords_bounds, [y_coord[0], y_coord[-1]]))
            labels_bounds   = np.vstack((labels_bounds,   [label[0],   label[-1]]))

        x_coords_bounds = np.delete(x_coords_bounds, 0, 0)
        y_coords_bounds = np.delete(y_coords_bounds, 0, 0)
        labels_bounds   = np.delete(labels_bounds,   0, 0)

        # ax.plot(x_coords_bounds[:, 0], y_coords_bounds[:, 0], '--', lw=2, color='black')
        # ax.plot(x_coords_bounds[:, 1], y_coords_bounds[:, 1], '--', lw=2, color='black')
        #
        # ax.plot(x_coords_bounds[0, :], y_coords_bounds[0, :], '--', lw=2, color='black')
        # ax.plot(x_coords_bounds[1, :], y_coords_bounds[1, :], '--', lw=2, color='black')

        # ax.plot(x_coords_bounds[0, 0], y_coords_bounds[0, 0], 'X', lw=2, color='black', label='SET 1')
        # ax.plot(x_coords_bounds[1, 0], y_coords_bounds[1, 0], 'X', lw=2, color='blue',  label='SET 2')
        # ax.plot(x_coords_bounds[2, 0], y_coords_bounds[2, 0], 'X', lw=2, color='green', label='SET 3')
    def plot_native_models(self, ax, v_n_x, v_n_y, metal, cr_or_wd, add='/', color='black', label=None):

        set_sp_files = Files.get_sp_files(metal, cr_or_wd, add) # b1_v2200/

        # spmdl = [] folder = '../data/sp_w_files'

        x_coord = []
        y_coord = []
        labels = []

        for file in set_sp_files:
            # spmdl.append(Read_SP_data_file(file, Files.output_dir, Files.plot_dir))

            cl = Read_SP_data_file(file, Files.output_dir, Files.plot_dir)
            x = cl.get_crit_value(v_n_x)
            y = cl.get_crit_value(v_n_y)
            lbl = cl.get_crit_value('m')

            x_coord = np.append(x_coord, x)
            y_coord = np.append(y_coord, y)
            labels = np.append(labels, lbl)


            ax.plot(x, y, 'X', color=color)
            if not self.set_clean:
                ax.annotate('{}'.format("%.2f" % cl.get_crit_value('m')),
                        xy=(x, y), textcoords='data')

        x_coord, y_coord, labels = Math.x_y_z_sort(x_coord, y_coord, labels)

        ax.plot(x_coord[0], y_coord[0], 'X', color=color, label=label)
        ax.annotate('{}'.format("%.0f" % labels[0]),  xy=(x_coord[0],  y_coord[0]),
                    textcoords='data',
                    horizontalalignment='center',
                    verticalalignment='top',
                    fontsize=self.set_label_size)
        ax.annotate('{}'.format("%.0f" % labels[-1]), xy=(x_coord[-1], y_coord[-1]),
                    textcoords='data',
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    fontsize=self.set_label_size)


        print('FIT TO THE MODELS WITH {} {} {}'.format(metal, cr_or_wd, add))
        x_grid, y_grid = Math.fit_polynomial(x_coord, y_coord, 1, 500)

        # ax.plot(x_grid, y_grid, '-', linewidth=2, color='blue')

        return x_coord, y_coord, labels

    def plot_fits(self, ax, obs_metal, v_n_x, v_n_y, x_arr, metals, bumps, hatch='/'):

        def get_linestyle(metal):
            if metal == 'gal':
                return '--'

            if metal == '2gal' or metal == '2lmc':
                return '-.'

            if metal == 'lmc':
                return '-'

            if metal == 'smc':
                return ':'

        if len(metals) != len(bumps): raise IOError('Metals != bumps')

        y_arrs = np.zeros(len(x_arr))
        for metal in metals:
            if obs_metal == 'gal':
                y_arr = Fits.get_gal_fit(v_n_x, v_n_y, x_arr, metal, self.set_clump_modified, self.set_use_gaia)
            elif obs_metal == 'lmc':
                y_arr = Fits.get_lmc_fit(v_n_x, v_n_y, x_arr, metal, self.bump)
            else:
                raise IOError('FITS: Obs_metal gal and lmc are only supproted so far')


            ax.plot(x_arr, y_arr, get_linestyle(metal), color='black', alpha=self.set_patches_or_lines_alpha_fits)

            y_arrs = np.vstack((y_arrs, y_arr))

        y_arrs = np.delete(y_arrs, 0, 0)

        if len(metals) > 0:
            ax.fill_between(x_arr, y_arrs[0, :], y_arrs[-1, :], hatch=hatch, alpha=self.set_patches_or_lines_alpha_fits)

    # EXTERNAL METHODS
    def plot_sonic_hrd(self, yc_val, l_or_lm, ax=None):

        if ax==None:
            do_plot = True
            fig = plt.figure() # figsize=plt.figaspect(0.8)
            ax = fig.add_subplot(111) # , projection='3d'
        else:
            do_plot = False


        yc_t_llm_mdot = Save_Load_tables.load_3d_table(self.set_metal, self.set_bump,
                                                       'yc_t_{}{}_mdot'.format(self.set_coeff, l_or_lm),
                                                       'yc', 't', str(self.set_coeff) + l_or_lm, 'mdot')

        yc_ind = Physics.ind_of_yc(yc_t_llm_mdot[:, 0, 0], yc_val)
        t_llm_mdot = yc_t_llm_mdot[yc_ind, :, :]

        # EXTRAPOLATION
        if sum(self.set_exrtrapolation) > 0:
            left = self.set_exrtrapolation[0]
            right = self.set_exrtrapolation[1]
            down = self.set_exrtrapolation[2]
            up = self.set_exrtrapolation[3]

            if self.set_show_extrap_borders:
                # if left != 0:
                #     ax.axvline(x=t_llm_mdot[0, 1:].min(), ls='dashed', lw=3, color='black')
                # if right != 0:
                #     ax.axvline(x=t_llm_mdot[0, 1:].max(), ls='dashed', lw=3, color='black')
                if down != 0:
                    ax.axhline(y=t_llm_mdot[1:, 0].min(), ls='dashed', lw=3, color='black')
                if up != 0:
                    ax.axhline(y=t_llm_mdot[1:, 0].max(), ls='dashed', lw=3, color='black')

            _, t_llm_mdot = Math.extrapolate2(t_llm_mdot, left, right, down, up, 500, 4, True)

        # self.plot_multiple_natives(ax, 'ts', 'lm', ['2gal', 'gal', 'gal'], ['wd', 'wd', 'wd'], ['/', '/', 'b1_v2200/'])
        # self.plot_multiple_natives(ax, 'ts', 'lm', ['gal'], ['wd'], ['/'])

        self.plot_color_background(ax, t_llm_mdot, 'ts', l_or_lm, 'mdot', self.set_metal, self.set_bump, 'Yc:{}'.format(yc_val))

        self.plot_obs_t_llm_mdot_int(ax, t_llm_mdot, l_or_lm, self.lim_t1, self.lim_t2, True)


        self.plot_fits(ax, self.set_metal, 'ts', l_or_lm, t_llm_mdot[0, 1:],
                       self.set_list_metals_fit,  # ['lmc', 'gal', '2gal']
                       self.set_list_bumps_fit)

        if not self.set_clean:
            if self.set_coeff != 1.0:
                ax.text(0.5, 0.9, 'K:{}'.format(self.set_coeff), style='italic',
                        bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center',
                        verticalalignment='center', transform=ax.transAxes)

        # --- --- WATER MARK --- --- ---
        if not self.set_clean:
            fig.text(0.95, 0.05, 'PRELIMINARY', fontsize=50, color='gray', ha='right', va='bottom', alpha=0.5)

        if do_plot:
            plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=self.set_ncol_legend, fontsize=self.set_label_size)
            plt.gca().invert_xaxis()
            plt.show()

    def plot_sonic_hrd_const_r(self,  l_or_lm, rs, yc_val, ax=None):


        if ax==None:
            do_plot = True
            fig = plt.figure() # figsize=plt.figaspect(0.8)
            ax = fig.add_subplot(111) # , projection='3d'
        else:
            do_plot = False

        yc_t_llm_mdot = Save_Load_tables.load_3d_table(self.set_metal, self.set_bump,
                                                       'yc_t_{}{}_mdot_rs_{}'.format(self.set_coeff, l_or_lm, rs),
                                                       'yc', 't', str(self.set_coeff) + l_or_lm,
                                                       'mdot_rs_{}'.format(rs))

        yc_ind = Physics.ind_of_yc(yc_t_llm_mdot[:, 0, 0], yc_val)
        t_llm_mdot = yc_t_llm_mdot[yc_ind, :, :]

        # EXTRAPOLATION
        if sum(self.set_exrtrapolation) > 0:
            left = self.set_exrtrapolation[0]
            right = self.set_exrtrapolation[1]
            down = self.set_exrtrapolation[2]
            up = self.set_exrtrapolation[3]

            if self.set_show_extrap_borders:
                # if left != 0:
                #     ax.axvline(x=t_llm_mdot[0, 1:].min(), ls='dashed', lw=3, color='black')
                # if right != 0:
                #     ax.axvline(x=t_llm_mdot[0, 1:].max(), ls='dashed', lw=3, color='black')
                if down != 0:
                    ax.axhline(y=t_llm_mdot[1:, 0].min(), ls='dashed', lw=3, color='black')
                if up != 0:
                    ax.axhline(y=t_llm_mdot[1:, 0].max(), ls='dashed', lw=3, color='black')

            _, t_llm_mdot = Math.extrapolate2(t_llm_mdot, left, right, down, up, 500, 4, True)


        self.plot_color_background(ax, t_llm_mdot, 'ts', l_or_lm, 'mdot', self.set_metal,
                                       'Yc:{} Rs:{}'.format(yc_val, rs))
        self.plot_obs_t_llm_mdot_int(ax, t_llm_mdot, l_or_lm, self.lim_t1, self.lim_t2, True)

        if self.set_coeff != 1.0 and not self.set_clean:
            ax.text(0.5, 0.9, 'K:{}'.format(self.set_coeff), style='italic',
                    bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center',
                    verticalalignment='center', transform=ax.transAxes)

        if do_plot:
            plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=self.set_ncol_legend, fontsize=self.set_label_size)
            plot_name = Files.plot_dir + 'sonic_HRD_const_rs.pdf'
            plt.savefig(plot_name)
            plt.gca().invert_xaxis()
            plt.show()
        else: return ax

    def plot_ts_y(self, v_n_y, yc_val, l_or_lm, v_n_background=None, ax=None):

        yc_t_llm_mdot = Save_Load_tables.load_3d_table(self.set_metal, self.set_bump,
                                                       'yc_t_{}{}_mdot'.format(self.set_coeff, l_or_lm),
                                                       'yc', 't', str(self.set_coeff) + l_or_lm, 'mdot')

        yc_ind = Physics.ind_of_yc(yc_t_llm_mdot[:, 0, 0], yc_val)
        t_llm_mdot = yc_t_llm_mdot[yc_ind, :, :]
        # lm_max = t_llm_mdot[1:, 0].max()
        # t_min = t_llm_mdot[0, 1:].min()


        if ax==None:
            do_plot = True
            fig = plt.figure() # figsize=plt.figaspect(0.8)
            ax = fig.add_subplot(111) # , projection='3d'
        else:
            do_plot = False

        if sum(self.set_exrtrapolation) > 0:
            left = self.set_exrtrapolation[0]
            right = self.set_exrtrapolation[1]
            down = self.set_exrtrapolation[2]
            up = self.set_exrtrapolation[3]

            self.set_show_extrap_borders = False
            if self.set_show_extrap_borders:
                # if left != 0:
                #     ax.axvline(x=t_llm_mdot[0, 1:].min(), ls='dashed', lw=3, color='black')
                # if right != 0:
                #     ax.axvline(x=t_llm_mdot[0, 1:].max(), ls='dashed', lw=3, color='black')
                if down != 0:
                    ax.axhline(y=t_llm_mdot[1:, 0].min(), ls='dashed', lw=3, color='black')
                if up != 0:
                    ax.axhline(y=t_llm_mdot[1:, 0].max(), ls='dashed', lw=3, color='black')

            _, t_llm_mdot = Math.extrapolate2(t_llm_mdot, left, right, down, up, 500, 4, True)

        #PlotBackground.plot_color_background(ax, t_llm_mdot, 't', l_or_lm, 'mdot', self.set_metal,
                                             #'Yc:{}'.format(yc_val), alpha, self.set_clean)

        # self.plot_native_models(ax, 'ts', v_n_y, self.metal, 'wd')
        # self.plot_native_models(ax, 'ts', v_n_y, '2gal', 'wd')



        self.plot_all_obs_ts_y_mdot(ax, v_n_y, t_llm_mdot, l_or_lm, self.lim_t1, self.lim_t2, True)

        self.plot_fits(ax, self.set_metal, 'ts', v_n_y, t_llm_mdot[0, 1:], self.set_list_metals_fit, self.set_list_bumps_fit)

        x = np.mgrid[ax.get_xlim()[0]:ax.get_xlim()[-1]:100j]
        y = x
        y2 = np.mgrid[100:100:100j]
        ax.plot(x, y, '-.', color='black')
        ax.fill_between(x, y, y2, color='gray', alpha=0.2)

        if v_n_background != None and v_n_background != '':

            # yc__ts_t_eff_tau_gal

            yc_table = Save_Load_tables.load_3d_table(self.metal,'ts_t_eff_{}'.format(v_n_background),
                                                           'yc', 'ts', 't_eff', 'tau', Files.output_dir)

            yc_ind = Physics.ind_of_yc(yc_table[:, 0, 0], yc_val)
            table = yc_table[yc_ind, :, :]

            PlotBackground.plot_color_background(ax, table, 'ts', 't_eff', v_n_background,
                                                 self.metal)


        # ax.axhline(y=lm_max, ls='dashed', lw=3, color='black')
        # ax.axvline(x=t_min, ls='dashed', lw=3, color='black')

        if not self.set_clean:
            if self.set_coeff != 1.0:
                ax.text(0.5, 0.9, 'K:{}'.format(self.set_coeff), style='italic',
                        bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center',
                        verticalalignment='center', transform=ax.transAxes)

        if not self.set_clean:
            fig.text(0.95, 0.05, 'PRELIMINARY', fontsize=50, color='gray', ha='right', va='bottom', alpha=0.5)

        if do_plot:

            ax.minorticks_on()
            plt.xticks(fontsize=self.set_label_sise)
            plt.yticks(fontsize=self.set_label_sise)
            ax.set_xlabel(Labels.lbls('ts'), fontsize=self.set_label_size)
            ax.set_ylabel(Labels.lbls(v_n_y), fontsize=self.set_label_size)
            ax.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=self.set_ncol_legend, fontsize=self.set_label_sise)

            plt.show()
        else:
            return ax

    # FUCKING METHODS
    def plot_sonic_hrd_set(self, l_or_lm, yc_arr):

        yc_t_llm_mdot = Save_Load_tables.load_3d_table(self.set_metal, self.set_bump, 'yc_t_{}{}_mdot'
                                                       .format(self.set_coeff, l_or_lm),'yc', 't',
                                                       str(self.set_coeff) + l_or_lm, 'mdot')
        yc = yc_t_llm_mdot[:, 0, 0]

        for i in range(len(yc_arr)):
            if not yc_arr[i] in yc:
                raise ValueError('Value yc_vals[{}] not in yc:\n\t {}'.format(yc_arr[i], yc))

        yc_vals = np.sort(yc_arr, axis=0)

        yc_n = len(yc_vals)

        fig = plt.figure()
        fig.subplots_adjust(hspace=0.2, wspace=0.3)

        for i in range(1, yc_n+1):
            print(i)
            yc_val = yc_vals[i-1]

            ind = Math.find_nearest_index(yc, yc_val)
            t_llm_mdot = yc_t_llm_mdot[ind, :, :]
            t_llm_mdot = Math.extrapolate(t_llm_mdot, 10, None, None, 25, 500, 'unispline') # 2 is better to linear part

            if yc_n % 2 == 0: ax = fig.add_subplot(2, yc_n/2, i)
            else:             ax = fig.add_subplot(1, yc_n, i)

            # fig = plt.figure(figsize=plt.figaspect(0.8))
            # ax = fig.add_subplot(111)  # , projection='3d'
            PlotBackground.plot_color_background(ax, t_llm_mdot, 't', l_or_lm, 'mdot', self.set_metal, 'Yc:{}'.format(yc_val))
            self.plot_obs_t_llm_mdot_int(ax, t_llm_mdot, l_or_lm, self.lim_t1, self.lim_t2, True)

            if self.set_coeff != 1.0:
                ax.text(0.5, 0.9, 'K:{}'.format(self.set_coeff), style='italic',
                        bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center',
                        verticalalignment='center', transform=ax.transAxes)

        plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
        plot_name = Files.plot_dir + 'sonic_HRD.pdf'
        plt.savefig(plot_name)
        plt.gca().invert_xaxis()
        plt.show()
    def plot_sonic_hrd_const_r_set(self, l_or_lm, rs, yc_arr):

        yc_t_llm_mdot = Save_Load_tables.load_3d_table(self.set_metal, self.set_bump, 'yc_t_{}{}_mdot_rs_{}'.format(self.set_coeff, l_or_lm, rs),
                                                       'yc', 't', str(self.set_coeff) + l_or_lm, 'mdot_rs_{}'.format(rs))
        yc = yc_t_llm_mdot[:, 0, 0]

        for i in range(len(yc_arr)):
            if not yc_arr[i] in yc:
                raise ValueError('Value yc_vals[{}] not in yc:\n\t {}'.format(yc_arr[i], yc))

        yc_vals = np.sort(yc_arr, axis=0)

        yc_n = len(yc_vals)

        fig = plt.figure()
        fig.subplots_adjust(hspace=0.2, wspace=0.3)

        for i in range(1, yc_n+1):
            print(i)
            yc_val = yc_vals[i-1]

            ind = Math.find_nearest_index(yc, yc_val)
            t_llm_mdot = yc_t_llm_mdot[ind, :, :]


            # limits, t_llm_mdot = Math.extrapolate2(t_llm_mdot, None, None, None, 5, 500, 'IntUni', True) # 2 is better to linear part


            if yc_n % 2 == 0: ax = fig.add_subplot(2, yc_n/2, i)
            else:             ax = fig.add_subplot(1, yc_n, i)

            # EXTRAPOLATION
            if sum(self.set_exrtrapolation) > 0:
                left = self.set_exrtrapolation[0]
                right = self.set_exrtrapolation[1]
                down = self.set_exrtrapolation[2]
                up = self.set_exrtrapolation[3]

                if self.set_show_extrap_borders:
                    # if left != 0:
                    #     ax.axvline(x=t_llm_mdot[0, 1:].min(), ls='dashed', lw=3, color='black')
                    # if right != 0:
                    #     ax.axvline(x=t_llm_mdot[0, 1:].max(), ls='dashed', lw=3, color='black')
                    if down !=0:
                        ax.axhline(y=t_llm_mdot[1:,0].min(), ls='dashed', lw=3, color='black')
                    if up != 0:
                        ax.axhline(y=t_llm_mdot[1:,0].max(), ls='dashed', lw=3, color='black')

                _, t_llm_mdot = Math.extrapolate2(t_llm_mdot, left, right, down, up, 500, 4, True)



            # fig = plt.figure(figsize=plt.figaspect(0.8))
            # ax = fig.add_subplot(111)  # , projection='3d'
            self.plot_color_background(ax, t_llm_mdot, 'ts', l_or_lm, 'mdot', self.set_metal, 'Yc:{} Rs:{}'.format(yc_val, rs))
            self.plot_obs_t_llm_mdot_int(ax, t_llm_mdot, l_or_lm, self.lim_t1, self.lim_t2, True)



            if self.set_coeff != 1.0:
                ax.text(0.5, 0.9, 'K:{}'.format(self.set_coeff), style='italic',
                        bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center',
                        verticalalignment='center', transform=ax.transAxes)


        plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=2, fontsize=self.set_label_size)
        plot_name = Files.plot_dir + 'sonic_HRD_const_rs.pdf'
        plt.savefig(plot_name)
        plt.gca().invert_xaxis()
        plt.show()



    # def t_llm_cr_sp(self, t_k_rho, yc_val, opal_used):
    #     kap = t_k_rho[1:, 0]
    #     t = t_k_rho[0, 1:]
    #     rho2d = t_k_rho[1:, 1:]
    #     lm_op = Physics.logk_loglm(kap, 1)
    #
    #     yc_lm_l = Save_Load_tables.load_table('yc_lm_l', 'yc', 'lm', 'l', opal_used)
    #     yc = yc_lm_l[0, 1:]
    #     lm_sp = yc_lm_l[1:, 0]
    #     l2d_sp = yc_lm_l[1:, 1:]
    #
    #     yc_lm_r = Save_Load_tables.load_table('yc_lm_r', 'yc', 'lm', 'r', opal_used)
    #     lm_sp = yc_lm_r[1:, 0]
    #     r2d_sp = yc_lm_r[1:, 1:]
    #
    #     lm1 = np.array([lm_op.min(), lm_sp.min()]).max()
    #     lm2 = np.array([lm_op.max(), lm_sp.max()]).min()
    #
    #     yc_lm_l = Math.crop_2d_table(Math.combine(yc, lm_sp, l2d_sp), None, None, lm1, lm2)
    #     yc_lm_r = Math.crop_2d_table(Math.combine(yc, lm_sp, r2d_sp), None, None, lm1, lm2)
    #     t_lm_rho = Math.crop_2d_table(Math.invet_to_ascending_xy(Math.combine(t, lm_op, rho2d)), None, None, lm1, lm2)
    #
    #     if not yc_val in yc:
    #         yc_ind = None
    #         raise ValueError('Yc = {} | not in yc from | {} | for {}'.format(yc_val, 'yc_lm_l', opal_used))
    #     else:
    #         yc_ind = Math.find_nearest_index(yc, yc_val) + 1  # as it starts with zero, which is non physical
    #
    #     t_op = t_lm_rho[0, 1:]
    #     lm_op = t_lm_rho[1:, 0]
    #     rho2d = t_lm_rho[1:, 1:]
    #     lm_sp = yc_lm_l[1:, 0]
    #     l2d = yc_lm_l[1:, yc_ind]
    #     r2d = yc_lm_r[1:, yc_ind]
    #
    #     f = interpolate.UnivariateSpline(lm_sp, l2d)
    #     l_ = f(lm_op)
    #
    #     f = interpolate.UnivariateSpline(lm_sp, r2d)
    #     r_ = f(lm_op)
    #
    #     vrho = Physics.get_vrho(t, rho2d, 2, np.array([1.34]))
    #     m_dot = Physics.vrho_mdot(vrho, r_, 'l')
    #
    #     return Math.combine(t, l_, m_dot),  Math.combine(t, lm_op, m_dot)

    # @staticmethod
    # def t_l_mdot_cr_sp(t_k_rho, yc_val, opal_used):
    #     kap = t_k_rho[1:, 0]
    #     t   = t_k_rho[0, 1:]
    #     rho2d = t_k_rho[1:, 1:]
    #     lm_op = Physics.logk_loglm(kap, 1)
    #
    #     yc_lm_l = Save_Load_tables.load_table('yc_lm_l', 'yc', 'lm', 'l', opal_used)
    #     yc = yc_lm_l[0, 1:]
    #     lm_sp = yc_lm_l[1:, 0]
    #     l2d_sp= yc_lm_l[1:, 1:]
    #
    #     yc_lm_r = Save_Load_tables.load_table('yc_lm_r', 'yc', 'lm', 'r', opal_used)
    #     lm_sp  = yc_lm_r[1:, 0]
    #     r2d_sp = yc_lm_r[1:, 1:]
    #
    #     lm1 = np.array([lm_op.min(), lm_sp.min()]).max()
    #     lm2 = np.array([lm_op.max(), lm_sp.max()]).min()
    #
    #     yc_lm_l = Math.crop_2d_table(Math.combine(yc, lm_sp, l2d_sp), None, None, lm1, lm2)
    #     yc_lm_r = Math.crop_2d_table(Math.combine(yc, lm_sp, r2d_sp), None, None, lm1, lm2)
    #     t_lm_rho= Math.crop_2d_table( Math.invet_to_ascending_xy(Math.combine(t, lm_op, rho2d)), None, None, lm1, lm2)
    #
    #     if not yc_val in yc:
    #         yc_ind = None
    #         raise ValueError('Yc = {} | not in yc from | {} | for {}'.format(yc_val, 'yc_lm_l', opal_used))
    #     else:
    #         yc_ind = Math.find_nearest_index(yc, yc_val) + 1 # as it starts with zero, which is non physical
    #
    #
    #
    #
    #     t_op = t_lm_rho[0, 1:]
    #     lm_op = t_lm_rho[1:, 0]
    #     rho2d = t_lm_rho[1:,1:]
    #     lm_sp = yc_lm_l[1:,0]
    #     l2d = yc_lm_l[1:, yc_ind]
    #     r2d = yc_lm_r[1:, yc_ind]
    #
    #     f = interpolate.UnivariateSpline(lm_sp, l2d)
    #     l_ = f(lm_op)
    #
    #     f = interpolate.UnivariateSpline(lm_sp, r2d)
    #     r_ = f(lm_op)
    #
    #     vrho  = Physics.get_vrho(t, rho2d, 2, np.array([1.34]))
    #     m_dot = Physics.vrho_mdot(vrho, r_, 'l')
    #
    #     return Math.combine(t, l_, m_dot)

    # @staticmethod
    # def t_lm_mdot_cr_sp(t_k_rho, yc_val, opal_used):
    #     kap = t_k_rho[1:, 0]
    #     t = t_k_rho[0, 1:]
    #     rho2d = t_k_rho[1:, 1:]
    #     lm_op = Physics.logk_loglm(kap, 1)
    #
    #     # yc_lm_l = Save_Load_tables.load_table('yc_lm_l', 'yc', 'lm', 'l', opal_used)
    #     # yc = yc_lm_l[0, 1:]
    #     # lm_sp = yc_lm_l[1:, 0]
    #     # l2d_sp = yc_lm_l[1:, 1:]
    #
    #     yc_lm_r = Save_Load_tables.load_table('yc_lm_r', 'yc', 'lm', 'r', opal_used)
    #     yc = yc_lm_r[0, 1:]
    #     lm_sp = yc_lm_r[1:, 0]
    #     r2d_sp = yc_lm_r[1:, 1:]
    #
    #     lm1 = np.array([lm_op.min(), lm_sp.min()]).max()
    #     lm2 = np.array([lm_op.max(), lm_sp.max()]).min()
    #
    #     # yc_lm_l = Math.crop_2d_table(Math.combine(yc, lm_sp, l2d_sp), None, None, lm1, lm2)
    #     yc_lm_r  = Math.crop_2d_table(Math.combine(yc, lm_sp, r2d_sp), None, None, lm1, lm2)
    #     t_lm_rho = Math.crop_2d_table(Math.invet_to_ascending_xy(Math.combine(t, lm_op, rho2d)), None, None, lm1, lm2)
    #
    #     if not yc_val in yc:
    #         yc_ind = None
    #         raise ValueError('Yc = {} | not in yc from | {} | for {}'.format(yc_val, 'yc_lm_l', opal_used))
    #     else:
    #         yc_ind = Math.find_nearest_index(yc, yc_val) + 1  # as it starts with zero, which is non physical
    #
    #
    #
    #
    #     t_op = t_lm_rho[0, 1:]
    #     lm_op = t_lm_rho[1:, 0]
    #     rho2d = t_lm_rho[1:, 1:]
    #     # lm_sp = yc_lm_l[1:, 0]
    #     # l2d = yc_lm_l[1:, yc_ind]
    #     r2d = yc_lm_r[1:, yc_ind]
    #
    #     # f = interpolate.UnivariateSpline(lm_sp, l2d)
    #     # l_ = f(lm_op)
    #
    #     f2 = interpolate.UnivariateSpline(lm_sp, r2d)
    #     r2_ = f2(lm_op)
    #
    #     vrho2 = Physics.get_vrho(t, rho2d, 2, np.array([1.34]))
    #     m_dot2 = Physics.vrho_mdot(vrho2, r2_, 'l')
    #
    #     return Math.combine(t, lm_op, m_dot2)

    # def save_yc_llm_mdot_cr(self, l_or_lm,
    #                         rs=None, yc_prec = 0.1, depth = 100, plot = True, lim_t1=5.18, lim_t2=None):
    #
    #     t_k_rho = Save_Load_tables.load_table('t_k_rho', 't', 'k', 'rho', self.opal_used, self.output_dir)
    #
    #     def interp(x, y, x_grid):
    #         f = interpolate.interp1d(x, y, kind='cubic', bounds_error=False)
    #         return x_grid, f(x_grid)
    #
    #     if rs == None:
    #
    #         spfiles_ = SP_file_work(self.sp_files, self.output_dir, self.plot_dir)
    #         min_llm, max_llm = spfiles_.get_min_max(l_or_lm)
    #         yc, spcls = spfiles_.separate_sp_by_crit_val('Yc', yc_prec)
    #
    #
    #         y_grid = np.mgrid[min_llm:max_llm:depth * 1j]
    #
    #         mdot2d_pol = np.zeros(len(y_grid))
    #         mdot2d_int = np.zeros(len(y_grid))
    #
    #         fig = plt.figure(figsize=plt.figaspect(1.0))
    #
    #         ax1 = fig.add_subplot(221)
    #         ax1.grid()
    #         ax1.set_ylabel(Labels.lbls(l_or_lm))
    #         ax1.set_xlabel(Labels.lbls('mdot'))
    #         ax1.set_title('INTERPOLATION')
    #
    #         ax2 = fig.add_subplot(222)
    #         ax2.grid()
    #         ax2.set_ylabel(Labels.lbls(l_or_lm))
    #         ax2.set_xlabel(Labels.lbls('mdot'))
    #         ax2.set_title('EXTRAPOLATION')
    #
    #         for i in range(len(yc)):
    #
    #             t_l_mdot, t_lm_mdot = self.t_llm_cr_sp(t_k_rho, yc[i], self.opal_used)
    #
    #             if l_or_lm == 'l':
    #                 t_llm_mdot = self.t_l_mdot_cr_sp(t_k_rho, yc[i], self.opal_used)
    #             else:
    #                 t_llm_mdot = self.t_lm_mdot_cr_sp(t_k_rho, yc[i], self.opal_used)
    #
    #
    #             # t_llm_mdot = self.t_llm_mdot_crit_sp(spcls[i], t_k_rho, l_or_lm, yc[i], self.opal_used)
    #             t = t_l_mdot[0, 1:]
    #             l = t_l_mdot[1:, 0]
    #             lm = t_lm_mdot[1:, 0]
    #             m_dot = t_l_mdot[1:, 1:]
    #
    #             mins = Math.get_mins_in_every_row(t, l, m_dot, 5000, lim_t1, lim_t2)
    #
    #
    #
    #             # llm = mins[1, :]
    #             mdot= mins[2, :]
    #
    #             # plt.plot(mins[2, :], mins[1, :], '-', color='black')
    #             # plt.annotate(str("%.2f" % yc[i]), xy=(mins[2, 0], mins[1, 0]), textcoords='data')
    #
    #             '''----------------------------POLYNOMIAL EXTRAPOLATION------------------------------------'''
    #             print('\n\t Yc = {}'.format(yc[i]))
    #
    #
    #             llm_pol, mdot_pol = Math.fit_plynomial(l, mdot, 3, depth, y_grid)
    #             mdot2d_pol = np.vstack((mdot2d_pol, mdot_pol))
    #             color = 'C' + str(int(yc[i] * 10) - 1)
    #             ax2.plot(mdot_pol, llm_pol, '--', color=color)
    #             ax2.plot(mdot, l, '.', color=color, label='yc:{}'.format("%.2f" % yc[i]))
    #
    #
    #             # ax2.annotate(str("%.2f" % yc[i]), xy=(mdot_pol[0], llm_pol[0]), textcoords='data')
    #
    #             '''------------------------------INTERPOLATION ONLY---------------------------------------'''
    #             llm_int, mdot_int = interp(llm, mdot, y_grid)
    #             mdot2d_int = np.vstack((mdot2d_int, mdot_int))
    #             ax1.plot(mdot_int, llm_int, '--', color=color)
    #             ax1.plot(mdot, llm, '-', color=color, label='yc:{}'.format("%.2f" % yc[i]))
    #             # ax1.annotate(str("%.2f" % yc[i]), xy=(mdot_int[0], llm_int[0]), textcoords='data')
    #
    #         ax1.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
    #         ax2.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
    #
    #         mdot2d_int = np.delete(mdot2d_int, 0, 0)
    #         mdot2d_pol = np.delete(mdot2d_pol, 0, 0)
    #
    #         yc_llm_mmdot_cr_pol = Math.combine(yc, y_grid, mdot2d_pol.T)  # changing the x/y
    #         yc_llm_mmdot_cr_int = Math.combine(yc, y_grid, mdot2d_int.T)  # changing the x/y
    #
    #
    #         table_name = '{}_{}_{}'.format('yc', l_or_lm, 'mdot_crit')
    #         Save_Load_tables.save_table(yc_llm_mmdot_cr_pol, self.opal_used, table_name, 'yc', l_or_lm, 'mdot_crit')
    #
    #
    #         if plot:
    #
    #             levels = [-7.0,-6.9,-6.8,-6.7,-6.6,-6.5, -6.4, -6.3, -6.2, -6.1, -6.0, -5.9, -5.8,
    #                       -5.7, -5.6, -5.5, -5.4, -5.3, -5.2, -5.1, -5., -4.9, -4.8, -4.7, -4.6, -4.5, -4.4, -4.3,
    #                       -4.2, -4.1, -4.]
    #             ax = fig.add_subplot(223)
    #
    #             # ax = fig.add_subplot(1, 1, 1)
    #             ax.set_xlim(yc_llm_mmdot_cr_int[0,1:].min(), yc_llm_mmdot_cr_int[0,1:].max())
    #             ax.set_ylim(yc_llm_mmdot_cr_int[1:,0].min(), yc_llm_mmdot_cr_int[1:,0].max())
    #             ax.set_ylabel(Labels.lbls(l_or_lm))
    #             ax.set_xlabel(Labels.lbls('Yc'))
    #
    #             contour_filled = plt.contourf(yc_llm_mmdot_cr_int[0, 1:], yc_llm_mmdot_cr_int[1:, 0], yc_llm_mmdot_cr_int[1:,1:], levels, cmap=plt.get_cmap('RdYlBu_r'))
    #             # plt.colorbar(contour_filled, label=Labels.lbls('m'))
    #             contour = plt.contour(yc_llm_mmdot_cr_int[0, 1:], yc_llm_mmdot_cr_int[1:, 0], yc_llm_mmdot_cr_int[1:,1:], levels, colors='k')
    #
    #             clb = plt.colorbar(contour_filled)
    #             clb.ax.set_title(Labels.lbls('mdot'))
    #
    #             plt.clabel(contour, colors='k', fmt='%2.2f', fontsize=12)
    #             #ax.set_title('MASS-LUMINOSITY RELATION')
    #
    #             # plt.ylabel(l_or_lm)
    #             # ax.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
    #             # plt.savefig(name)
    #
    #
    #
    #
    #             ax = fig.add_subplot(224)
    #
    #             # ax = fig.add_subplot(1, 1, 1)
    #             ax.set_xlim(yc_llm_mmdot_cr_pol[0, 1:].min(), yc_llm_mmdot_cr_pol[0, 1:].max())
    #             ax.set_ylim(yc_llm_mmdot_cr_pol[1:, 0].min(), yc_llm_mmdot_cr_pol[1:, 0].max())
    #             ax.set_ylabel(Labels.lbls(l_or_lm))
    #             ax.set_xlabel(Labels.lbls('Yc'))
    #
    #
    #             contour_filled = plt.contourf(yc_llm_mmdot_cr_pol[0, 1:], yc_llm_mmdot_cr_pol[1:, 0], yc_llm_mmdot_cr_pol[1:, 1:], levels,
    #                                           cmap=plt.get_cmap('RdYlBu_r'))
    #             # plt.colorbar(contour_filled, label=Labels.lbls('m'))
    #             contour = plt.contour(yc_llm_mmdot_cr_pol[0, 1:], yc_llm_mmdot_cr_pol[1:, 0], yc_llm_mmdot_cr_pol[1:, 1:], levels, colors='k')
    #
    #             clb = plt.colorbar(contour_filled)
    #             clb.ax.set_title(Labels.lbls('mdot'))
    #
    #             plt.clabel(contour, colors='k', fmt='%2.2f', fontsize=12)
    #             #ax.set_title('MASS-LUMINOSITY RELATION')
    #
    #             # plt.ylabel(l_or_lm)
    #             # ax.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
    #
    #
    #             plt.show()


    # def min_mdot_sp(self, l_or_lm, yc_val):
    #
    #     name = '{}_{}_{}'.format('yc', l_or_lm, 'mdot_crit')
    #     yc_llm_mdot_cr = Save_Load_tables.load_table(name, 'yc', l_or_lm, 'mdot_crit', self.opal_used)
    #     yc  = yc_llm_mdot_cr[0, 1:]
    #     llm = yc_llm_mdot_cr[1:, 0]
    #     mdot2d= yc_llm_mdot_cr[1:, 1:]
    #
    #
    #
    #     if yc_val in yc:
    #         ind = Math.find_nearest_index(yc, yc_val)
    #         mdot = mdot2d[:, ind]
    #     else:
    #         raise ValueError('\tYc = {} not in the list: \n\t{}'.format(yc_val, yc))
    #
    #     fig = plt.figure()
    #     ax = fig.add_subplot(1, 1, 1)
    #     ax.plot(mdot, llm, '-',    color='black')
    #     ax.fill_between(mdot, llm, color="lightgray")
    #
    #     classes  = []
    #     classes.append('dum')
    #     mdot_obs = []
    #     llm_obs  = []
    #
    #
    #     '''=============================OBSERVABELS==============================='''
    #
    #     # from Phys_Math_Labels import Opt_Depth_Analythis
    #
    #     for star_n in self.obs.stars_n:
    #         i = -1
    #         mdot_obs = np.append(mdot_obs, self.obs.get_num_par('mdot', star_n))
    #         llm_obs = np.append(llm_obs, self.obs.get_num_par(l_or_lm, star_n, yc_val, self.opal_used))
    #         eta = self.obs.get_num_par('eta', star_n)
    #
    #         # print(self.obs.get_num_par('mdot',  star_n), self.obs.get_num_par(l_or_lm, star_n))
    #
    #         plt.plot(mdot_obs[i], llm_obs[i], marker=self.obs.get_clss_marker(star_n), markersize='9',
    #                  color=self.obs.get_class_color(star_n), ls='')  # plot color dots)))
    #         ax.annotate('{} {}'.format(int(star_n), eta), xy=(mdot_obs[i], llm_obs[i]), textcoords='data')  # plot numbers of stars
    #
    #
    #         # v_inf = self.obs.get_num_par('v_inf', star_n)
    #         # tau_cl = Opt_Depth_Analythis(30, v_inf, 1., 1., mdot_obs[i], 0.20)
    #         # tau = tau_cl.anal_eq_b1(1.)
    #         # ax.annotate(str(int(tau)), xy=(mdot_obs[i], llm_obs[i]), textcoords='data')  # plo
    #
    #
    #         if self.obs.get_star_class(star_n) not in classes:
    #             plt.plot(mdot_obs[i], llm_obs[i], marker=self.obs.get_clss_marker(star_n), markersize='9',
    #                      color=self.obs.get_class_color(star_n), ls = '',
    #                      label='{}'.format(self.obs.get_star_class(star_n)))  # plot color dots)))
    #             classes.append(self.obs.get_star_class(star_n))
    #
    #     print('\t__PLOT: total stars: {}'.format(len(self.obs.stars_n)))
    #     print(len(mdot_obs), len(llm_obs))
    #
    #     fit = np.polyfit(mdot_obs, llm_obs, 1)  # fit = set of coeddicients (highest first)
    #     f = np.poly1d(fit)
    #     fit_x_coord = np.mgrid[(mdot_obs.min() - 1):(mdot_obs.max() + 1):1000j]
    #     plt.plot(fit_x_coord, f(fit_x_coord), '-.', color='blue')
    #
    #     min_mdot, max_mdot = self.obs.get_min_max('mdot')
    #     min_llm, max_llm = self.obs.get_min_max(l_or_lm, yc_val, self.opal_used)
    #
    #     ax.set_xlim(min_mdot, max_mdot)
    #     ax.set_ylim(min_llm, max_llm)
    #
    #     ax.set_ylabel(Labels.lbls(l_or_lm))
    #     ax.set_xlabel(Labels.lbls('mdot'))
    #     ax.grid(which='major', alpha=0.2)
    #     ax.legend(bbox_to_anchor=(1, 1), loc='upper right', ncol=1)
    #
    #     ax.grid(which='both')
    #     ax.grid(which='minor', alpha=0.2)
    #     plt.legend(bbox_to_anchor=(1, 0), loc='lower right', ncol=1)
    #     plot_name = self.plot_dir + 'minMdot_l.pdf'
    #     plt.savefig(plot_name)
    #     plt.show()

from FilesWork import Levels
class Plot_Two_sHRDs():
    def __init__(self, l_or_lms, metals, bumps, coefs, rss, yc_vals):

        self.set_obs_metals = metals
        self.set_back_metals = metals
        self.set_back_bumps = bumps
        self.set_back_coeffs = coefs
        self.set_back_rss = rss
        self.set_back_ycs = yc_vals
        self.set_l_or_lms = l_or_lms

        self.obs1 = PlotObs(self.set_obs_metals[0], self.set_back_bumps[0],
                            Files.get_obs_file(self.set_obs_metals[0]), Files.get_atm_file(self.set_obs_metals[0]))
        self.obs2 = PlotObs(self.set_obs_metals[1], self.set_back_bumps[1],
                            Files.get_obs_file(self.set_obs_metals[1]), Files.get_atm_file(self.set_obs_metals[1]))

        self.back = PlotBackground2()

    def add_plot_background(self, ax, table, opal):

        levels = Levels.get_levels('mdot', opal)

        contour_filled = ax.contourf(table[0, 1:], table[1:, 0], table[1:, 1:], levels, cmap=plt.get_cmap('RdYlBu_r'))
        # clb = plt.colorbar(contour_filled)
        # clb.ax.set_title(Labels.lbls('mdot'))

        contour = ax.contour(table[0, 1:], table[1:, 0], table[1:, 1:], levels, colors='k')
        # ax.clabel(contour, colors='k', fmt='%2.2f', fontsize=12)
        # ax.set_title('SONIC HR DIAGRAM')
        return ax, contour_filled

    def load_t_llm_mdot(self, l_or_lm, coeff, rs, opal, bump, yc):
        lbl = []
        z = Get_Z.z(opal)

        if rs == None:

            yc_t_llm_mdot = Save_Load_tables.load_3d_table(opal, bump, 'yc_t_{}{}_mdot'.format(coeff, l_or_lm),
                                                           'yc', 't', str(coeff) + l_or_lm, 'mdot')

            lbl.append('z:{}({}) K:{}'.format(z, bump, coeff))
        else:

            yc_t_llm_mdot = Save_Load_tables.load_3d_table(opal, bump,
                                                           'yc_t_{}{}_mdot_rs_{}'.format(coeff, l_or_lm, rs),
                                                           'yc', 't', str(coeff) + l_or_lm,
                                                           'mdot_rs_{}'.format(rs))

            lbl.append('z:{}({}) K:{} Rs:{}'.format(z, bump, coeff, rs))

        yc_arr = yc_t_llm_mdot[:, 0, 0]

        for i in range(len(yc_arr)):
            if not yc in yc_arr:
                raise ValueError('Value yc_vals[{}] not in yc:\n\t {}'.format(yc, yc_arr))

        # selecting one table with required Yc
        yc_ind = Physics.ind_of_yc(yc_t_llm_mdot[:, 0, 0], yc)
        t_llm_mdot = yc_t_llm_mdot[yc_ind, :, :]

        # Extrapolatin
        t_llm_mdot = Math.extrapolate(t_llm_mdot, None, None, 10, 5, 500, 4)
        return t_llm_mdot, lbl

    def plot_two_shrd(self):

        t_llm_mdot1, lbl1 = self.load_t_llm_mdot(self.set_l_or_lms[0], self.set_back_coeffs[0], self.set_back_rss[0],
                                                 self.set_back_metals[0], self.set_back_bumps[0], self.set_back_ycs[0])

        t_llm_mdot2, lbl2 = self.load_t_llm_mdot(self.set_l_or_lms[1], self.set_back_coeffs[1], self.set_back_rss[1],
                                                 self.set_back_metals[1], self.set_back_bumps[1], self.set_back_ycs[1])

        from mpl_toolkits.axes_grid1 import make_axes_locatable


        fig = plt.figure()
        # fig.set_size_inches(18.5, 10.5)
        # fig.set_size_inches(10, 4.5)

        ax1 = fig.add_subplot(1, 2, 1)  # aspect='equal'
        ax2 = fig.add_subplot(1, 2, 2, sharey=ax1)  # Share y-axes with subplot 1

        ax1.minorticks_on()
        ax2.minorticks_on()

        ax1.tick_params()

        # Set y-ticks of subplot 2 invisible
        plt.setp(ax2.get_yticklabels(), visible=False)

        # Plot data
        im1, cf1 = self.add_plot_background(ax1, t_llm_mdot1, self.set_back_metals[0])
        im2, cf2 = self.add_plot_background(ax2, t_llm_mdot2, self.set_back_metals[1])

        ax1_stars = self.obs1.plot_obs_t_llm_mdot_int(ax2, t_llm_mdot2, self.set_l_or_lms[0])
        ax1_stars.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=2)

        ax2_stars = self.obs2.plot_obs_t_llm_mdot_int(ax1, t_llm_mdot1, self.set_l_or_lms[0])
        ax2_stars.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)

        # Define locations of colorbars for both subplot 1 and 2
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)

        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)

        # Create and remove the colorbar for the first subplot

        cbar1 = fig.colorbar(cf1, cax=cax1)
        fig.delaxes(fig.axes[2])

        # Create second colorbar
        cbar2 = fig.colorbar(cf2, cax=cax2)
        cbar2.ax.set_title(Labels.lbls('mdot'))

        # --- --- WATER MARK -- -- --
        fig.text(0.95, 0.05, 'PRELIMINARY', fontsize=50, color='gray', ha='right', va='bottom', alpha=0.5)

        # Adjust the widths between the subplots
        # plt.title('SONIC HR DIAGRAM', loc='center')
        ax1.set_xlabel(Labels.lbls('t'))
        ax1.set_ylabel(Labels.lbls('lm'))
        plt.subplots_adjust(wspace=-0.0)
        ax1.invert_xaxis()
        ax2.invert_xaxis()
        plt.show()

class Plot_Tow_Sonic_HRDs:

    output_dir = '../data/output/'
    plot_dir = '../data/plots/'

    y_coord = []
    coeff = []
    rs = []
    metals = []
    bump = []
    yc = []

    def __init__(self, obs_files, opal_for_obs):
        '''
        Table name expected: ' yc_0.8lm_mdot_cr_r_1.0__HeII_table_x '
        :param tables:
        :param yc:
        :param obs_files:
        '''
        self.n = len(self.y_coord)

        self.obs = []
        self.obs_files = obs_files
        for i in range(len(obs_files)):
            self.obs.append(Read_Observables(obs_files[i], opal_for_obs[i]))

    def plot_srhd(self, clean):

        n = len(self.y_coord)
        if n > 2: raise ValueError('Only 2 sHRDs available now')

        if len(self.coeff) != n or len(self.rs) != n or len(self.metals) != n or len(self.bump) != n or len(self.yc) != n:
            raise ValueError('Length of all input array must me the same. Given: {}, {}, {}, {}, {}'
                             .format(len(self.coeff), len(self.rs), len(self.metals), len(self.bump), len(self.yc) != n))

        def load_t_llm_mdot(l_or_lm, coeff, rs, opal, bump, yc):
            lbl = []
            z = Get_Z.z(opal)

            if rs == None:

                yc_t_llm_mdot = Save_Load_tables.load_3d_table(opal, bump, 'yc_t_{}{}_mdot'.format(coeff, l_or_lm),
                                                               'yc', 't', str(coeff) + l_or_lm, 'mdot')

                lbl.append('z:{}({}) K:{}'.format(z, bump, coeff))
            else:

                yc_t_llm_mdot = Save_Load_tables.load_3d_table(opal, bump,
                                                               'yc_t_{}{}_mdot_rs_{}'.format(coeff, l_or_lm, rs),
                                                               'yc', 't', str(coeff) + l_or_lm,
                                                               'mdot_rs_{}'.format(rs))

                lbl.append('z:{}({}) K:{} Rs:{}'.format(z, bump, coeff, rs))

            yc_arr = yc_t_llm_mdot[:, 0, 0]

            for i in range(len(yc_arr)):
                if not yc in yc_arr:
                    raise ValueError('Value yc_vals[{}] not in yc:\n\t {}'.format(yc, yc_arr))

            # selecting one table with required Yc
            yc_ind = Physics.ind_of_yc(yc_t_llm_mdot[:, 0, 0], yc)
            t_llm_mdot = yc_t_llm_mdot[yc_ind, :, :]

            # Extrapolatin
            t_llm_mdot = Math.extrapolate(t_llm_mdot, None, None, 10, 5, 500, 4)
            return t_llm_mdot, lbl

        t_llm_mdot1, lbl1 = load_t_llm_mdot(self.y_coord[0], self.coeff[0], self.rs[0], self.metals[0], self.bump[0], self.yc[0])
        t_llm_mdot2, lbl2 = load_t_llm_mdot(self.y_coord[1], self.coeff[1], self.rs[1], self.metals[1], self.bump[1], self.yc[1])


        def plot_background(ax, table, opal):
            from PhysMath import Levels
            levels = Levels.get_levels('mdot', opal)

            contour_filled = ax.contourf(table[0, 1:], table[1:, 0], table[1:, 1:], levels, cmap=plt.get_cmap('RdYlBu_r'))
            # clb = plt.colorbar(contour_filled)
            # clb.ax.set_title(Labels.lbls('mdot'))

            contour = ax.contour(table[0, 1:], table[1:, 0], table[1:, 1:], levels, colors='k')
            # ax.clabel(contour, colors='k', fmt='%2.2f', fontsize=12)
            # ax.set_title('SONIC HR DIAGRAM')
            return ax, contour_filled

        from mpl_toolkits.axes_grid1 import make_axes_locatable

        fig = plt.figure()
        # fig.set_size_inches(18.5, 10.5)
        # fig.set_size_inches(10, 4.5)

        ax1 = fig.add_subplot(1, 2, 1 ) # aspect='equal'
        ax2 = fig.add_subplot(1, 2, 2, sharey=ax1)  # Share y-axes with subplot 1

        # Set y-ticks of subplot 2 invisible
        plt.setp(ax2.get_yticklabels(), visible=False)

        # Plot data
        im1, cf1 = plot_background(ax1, t_llm_mdot1, self.metals[0])
        im2, cf2 = plot_background(ax2, t_llm_mdot2, self.metals[1])


        ax1_stars = Plots.plot_obs_t_llm_mdot_int(ax2, t_llm_mdot2, self.obs[0], self.y_coord[1], None, None, False, clean)
        ax1_stars.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=2)

        ax2_stars = Plots.plot_obs_t_llm_mdot_int(ax1, t_llm_mdot1, self.obs[0], self.y_coord[0], None, None, False, clean)
        ax2_stars.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)


        # Define locations of colorbars for both subplot 1 and 2
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)

        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)

        # Create and remove the colorbar for the first subplot

        cbar1 = fig.colorbar(cf1, cax=cax1)
        fig.delaxes(fig.axes[2])

        # Create second colorbar
        cbar2 = fig.colorbar(cf2, cax=cax2)
        cbar2.ax.set_title(Labels.lbls('mdot'))

        # --- --- WATER MARK -- -- --
        fig.text(0.95, 0.05, 'PRELIMINARY', fontsize=50, color='gray', ha='right', va='bottom', alpha=0.5)

        # Adjust the widths between the subplots
        # plt.title('SONIC HR DIAGRAM', loc='center')
        ax1.set_xlabel(Labels.lbls('t'))
        ax1.set_ylabel(Labels.lbls('lm'))
        plt.subplots_adjust(wspace=-0.0)
        ax1.invert_xaxis()
        ax2.invert_xaxis()
        plt.show()




        # lbl = []
        # min_lm = []
        # max_lm = []
        # min_mdot = []
        # max_mdot = []
        #
        # for i in range(n):
        #
        #     l_or_lm = self.y_coord[i]
        #     coeff = self.coeff[i]
        #     r_cr = self.r_cr[i]
        #     opal = self.opal[i]
        #     bump = self.bump[i]
        #     yc = self.yc[i]
        #     z = Get_Z.z(opal)
        #
        #     if r_cr == None:
        #         name = '{}_{}{}_{}'.format('yc', coeff, l_or_lm, 'mdot_cr')
        #         lbl.append('z:{}({}) K:{}'.format(z, bump, coeff))
        #         yc_llm_mdot_cr = Save_Load_tables.load_table(name, 'yc', str(coeff) + l_or_lm,
        #                                                      'mdot_cr', opal, bump)
        #     else:
        #         name = '{}_{}{}_{}_r_{}'.format('yc', coeff, l_or_lm, 'mdot_cr', r_cr)
        #         lbl.append('z:{}({}) K:{} R:{}'.format(z, bump, coeff, r_cr))
        #         yc_llm_mdot_cr = Save_Load_tables.load_table(name, 'yc', str(coeff) + l_or_lm,
        #                                                      'mdot_cr_r_{}'.format(r_cr), opal, bump)


class GradAnalysis:

    def __init__(self, metal, bump):
        self.set_metal = metal
        self.set_bump = bump

    from PhysMath import Constants, Math
    from FilesWork import PlotBackground2


    def tst(self):

        rs = 2.0
        vs = 50

        betas = np.mgrid[0.75:1.50:100j]
        vinfs = np.mgrid[1600:2400:200j]

        grads = np.zeros((len(betas), len(vinfs)))

        radii = np.mgrid[rs:(rs + 1):100j]

        for i in range(len(betas)):
            for j in range(len(vinfs)):
                vels = Physics.beta_law(radii, radii[0], vs, vinfs[j], betas[i])
                grad = np.gradient(vels, radii * Constants.solar_r / 10 ** 5)

                # grad = diff_beta_law(radii * Constants.solar_r / 10**5, radii[0] * Constants.solar_r / 10**5, vinfs[j], betas[i])
                grads[i, j] = grad[0] * 10 ** 5

        print(grads)

        res = Math.combine(vinfs, betas, grads)

        PlotBackground2.plot_color_table(res, 'v_inf', 'beta', 'grad_w', 'gal', 'Fe')

    # tst()

    def main_cycle(self):

        # loading files
        pass


    def plot_master(self, v_n_x, v_n_y, v_n_col, beta_vals):
        pass
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
    def __init__(self, spfiles, out_dir = '../data/output/', plot_dir = '../data/plots/'):

        self.spfiles = spfiles

        # self.req_name_parts = req_name_parts


        # def select_sp_files(spfile, req_name_parts):
        #
        #     if len(req_name_parts) == 0:
        #         return spfile
        #
        #
        #     no_extens_sp_file = spfile.split('.')[-2]  # getting rid of '.data'
        #
        #     print(spfile, '   ', no_extens_sp_file)
        #
        #     for req_part in req_name_parts:
        #         if req_part in no_extens_sp_file.split('_')[1:]:
        #
        #             return spfile
        #         else:
        #             return None


        self.spmdl = []
        for file in spfiles:
            self.spmdl.append(Read_SP_data_file(file, out_dir, plot_dir))
            # if select_sp_files(file, req_name_parts):
            #     self.spmdl.append( Read_SP_data_file(file, out_dir, plot_dir) )

        print('\t__TEST: total {} files uploaded in total.'.format(len(self.spmdl)))

        self.plot_dit = plot_dir
        self.out_dir = out_dir


    def xy_last_points(self, v_n1, v_n2, v_lbl1, v_lbl_cond, list_of_list_of_smfiles = list(), num_pol_fit = True):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        for j in range(len(list_of_list_of_smfiles)):

            x = []
            y = []
            for i in range(len(list_of_list_of_smfiles[j])):
                sm1 = Read_SM_data_file.from_sm_data_file(list_of_list_of_smfiles[j][i])
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


    def sm_3d_plotting_x_y_z(self, v_n1, v_n2, v_n3, v_lbl1, v_lbl_cond, list_of_list_of_smfiles = list(), num_pol_fit = True):
        from mpl_toolkits.mplot3d import Axes3D


        fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        ax = fig.gca(projection='3d')


        for j in range(len(list_of_list_of_smfiles)):

            x = []
            y = []
            z = []
            for i in range(len(list_of_list_of_smfiles[j])):
                sm1 = Read_SM_data_file.from_sm_data_file(list_of_list_of_smfiles[j][i])
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

    def sp_3d_plotting_x_y_z(self, v_n1, v_n2, v_n3, v_n_col):
        from mpl_toolkits.mplot3d import Axes3D

        # fig = plt.subplot(2, 1, 1)


        # fig = plt.subplot(2, 1, 1,)
        # ax1 = plt.subplot(211)
        # ax = fig.add_subplot(111, projection='3d')
        ax = plt.gca(projection='3d')  # fig.gca(projection='3d')
        # ax1 = fig.add_subplot(2, 1, 2, projection='3d')

        all_x = []
        all_y = []
        all_z = []
        all_t = []

        all_x_cr = []
        all_y_cr = []
        all_z_cr = []
        all_t_cr = []


        for i in range(len(self.spfiles)):

            xc = self.spmdl[i].get_crit_value(v_n1)
            yc = self.spmdl[i].get_crit_value(v_n2)
            zc = self.spmdl[i].get_crit_value(v_n3)
            col_c = self.spmdl[i].get_crit_value(v_n_col)

            ax.scatter(xc, yc, zc, color='black', marker='x', linewidths='')


            n_of_rows = len( self.spmdl[i].table[:, 0] ) - 1
            x = []
            y = []
            z = []
            t = []

            for j in range(n_of_rows):
                if self.spmdl[i].get_sonic_cols('r')[j] > 0.:              # selecting only the solutions with found rs
                    x = np.append( x, self.spmdl[i].get_sonic_cols(v_n1)[j] )
                    y = np.append( y, self.spmdl[i].get_sonic_cols(v_n2)[j] )
                    z = np.append( z, self.spmdl[i].get_sonic_cols(v_n3)[j] )
                    t = np.append( t, self.spmdl[i].get_sonic_cols(v_n_col)[j] )

            all_x_cr = np.append(all_x_cr, self.spmdl[i].get_crit_value(v_n1))
            all_y_cr = np.append(all_y_cr, self.spmdl[i].get_crit_value(v_n2))
            all_z_cr = np.append(all_z_cr, self.spmdl[i].get_crit_value(v_n3))
            all_t_cr = np.append(all_t_cr, self.spmdl[i].get_crit_value(v_n_col))

            x = np.append(x, all_x_cr) # adding critical values
            y = np.append(y, all_y_cr)
            z = np.append(z, all_z_cr)
            t = np.append(t, all_t_cr)

            all_x = np.append(all_x, x)
            all_y = np.append(all_y, y)
            all_z = np.append(all_z, z)
            all_t = np.append(all_t, t)

            # ax.scatter(x, y, z, c=t, marker='o', cmap=plt.get_cmap('RdYlBu_r'))



        # ---------------------------------------------------------------------------------------

        print(len(all_x), len(all_y), len(all_z), len(all_t))

        sc = ax.scatter(all_x, all_y, all_z, c=all_t, marker='o', cmap=plt.get_cmap('Purples_r'))

        plt.colorbar(sc, label=Labels.lbls(v_n_col))

        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot(all_x, all_z, 'r.', zdir='y', zs = all_y.max() - all_y.max()/2)
        # ax.plot(all_y, all_z, 'g.', zdir='x', zs = all_x.max() - all_x.max()/20)
        # ax.plot(all_x, all_y, 'k.', zdir='z', zs = all_z.max() - all_z.max())

        # ax.plot(all_x, all_z, 'r.', zdir='y', zs = all_y.min())
        # ax.plot(all_y, all_z, 'g.', zdir='x', zs = all_x.min())
        # ax.plot(all_x, all_y, 'k.', zdir='z', zs = all_z.min())


        # --- --- --- SUBPLOTS --- --- ---

        # ax1 = fig.add_subplot(1, 1, 1)
        # plt.plot(all_x, all_y, '.')


        ax.w_xaxis.set_pane_color((0.4, 0.4, 0.6, 0.3))
        ax.w_yaxis.set_pane_color((0.4, 0.4, 0.6, 0.3))
        ax.w_zaxis.set_pane_color((0.4, 0.4, 0.6, 0.3))

        ax.set_xlabel(Labels.lbls(v_n1))
        ax.set_ylabel(Labels.lbls(v_n2))
        ax.set_zlabel(Labels.lbls(v_n3))

        plt.show()
        # fig.canvas.show()

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

    def sp_3d_and_multiplot(self, v_n1, v_n2, v_n3, v_n_col, sp_or_crit = 'both'):

        all_x = []
        all_y = []
        all_z = []
        all_t = []

        all_x_cr = []
        all_y_cr = []
        all_z_cr = []
        all_t_cr = []

        # set up a figure twice as wide as it is tall
        fig = plt.figure(figsize=plt.figaspect(1.0))

        # ===============
        # First subplot
        # ===============

        ax = fig.add_subplot(221, projection='3d')  # ax = fig.add_subplot(1, 2, 1, projection='3d') for 2 plots

        for i in range(len(self.spmdl)):

            xc = self.spmdl[i].get_crit_value(v_n1)
            yc = self.spmdl[i].get_crit_value(v_n2)
            zc = self.spmdl[i].get_crit_value(v_n3)
            col_c = self.spmdl[i].get_crit_value(v_n_col)

            ax.scatter(xc, yc, zc, color='black', marker='x', linewidths='')

            n_of_rows = len(self.spmdl[i].table[:, 0]) - 1
            x = []
            y = []
            z = []
            t = []


            if sp_or_crit == 'sp' or sp_or_crit == 'both':
                for j in range(n_of_rows):
                    if self.spmdl[i].get_sonic_cols('r')[j] > 0.:  # selecting only the solutions with found rs
                        x = np.append(x, self.spmdl[i].get_sonic_cols(v_n1)[j])
                        y = np.append(y, self.spmdl[i].get_sonic_cols(v_n2)[j])
                        z = np.append(z, self.spmdl[i].get_sonic_cols(v_n3)[j])
                        t = np.append(t, self.spmdl[i].get_sonic_cols(v_n_col)[j])

            if sp_or_crit == 'crit' or sp_or_crit == 'both':
                all_x_cr = np.append(all_x_cr, self.spmdl[i].get_crit_value(v_n1))
                all_y_cr = np.append(all_y_cr, self.spmdl[i].get_crit_value(v_n2))
                all_z_cr = np.append(all_z_cr, self.spmdl[i].get_crit_value(v_n3))
                all_t_cr = np.append(all_t_cr, self.spmdl[i].get_crit_value(v_n_col))

            x = np.append(x, all_x_cr)  # adding critical values
            y = np.append(y, all_y_cr)
            z = np.append(z, all_z_cr)
            t = np.append(t, all_t_cr)

            all_x = np.append(all_x, x)
            all_y = np.append(all_y, y)
            all_z = np.append(all_z, z)
            all_t = np.append(all_t, t)

        sc = ax.scatter(all_x, all_y, all_z, c=all_t, marker='o', cmap=plt.get_cmap('Spectral'))

        clb = plt.colorbar(sc)
        clb.ax.set_title(Labels.lbls(v_n_col))


        ax.w_xaxis.set_pane_color((0.4, 0.4, 0.6, 0.3))
        ax.w_yaxis.set_pane_color((0.4, 0.4, 0.6, 0.3))
        ax.w_zaxis.set_pane_color((0.4, 0.4, 0.6, 0.3))

        ax.set_xlabel(Labels.lbls(v_n1))
        ax.set_ylabel(Labels.lbls(v_n2))
        ax.set_zlabel(Labels.lbls(v_n3))

        # ===============
        # Second subplots
        # ===============

        ax = fig.add_subplot(222)
        ax.grid()
        sc = ax.scatter(all_x_cr, all_y_cr, c=all_t_cr, marker='o', cmap=plt.get_cmap('Spectral'))
        ax.set_xlabel(Labels.lbls(v_n1))
        ax.set_ylabel(Labels.lbls(v_n2))
        for i in range(len(all_x_cr)):
            ax.annotate("%.2f" % all_t_cr[i], xy=(all_x_cr[i], all_y_cr[i]), textcoords='data')  # plot numbers of stars
        clb = plt.colorbar(sc)
        clb.ax.set_title(Labels.lbls(v_n_col))


        ax = fig.add_subplot(223)
        ax.grid()
        sc = ax.scatter(all_y_cr, all_z_cr, c=all_t_cr, marker='o', cmap=plt.get_cmap('Spectral'))
        ax.set_xlabel(Labels.lbls(v_n2))
        ax.set_ylabel(Labels.lbls(v_n3))
        for i in range(len(all_x_cr)):
            ax.annotate("%.2f" % all_t_cr[i], xy=(all_y_cr[i], all_z_cr[i]), textcoords='data')  # plot numbers of stars
        clb = plt.colorbar(sc)
        clb.ax.set_title(Labels.lbls(v_n_col))


        ax = fig.add_subplot(224)
        ax.grid()
        sc = ax.scatter(all_x_cr, all_z_cr, c=all_t_cr, marker='o', cmap=plt.get_cmap('Spectral'))
        ax.set_xlabel(Labels.lbls(v_n1))
        ax.set_ylabel(Labels.lbls(v_n3))
        for i in range(len(all_x_cr)):
            ax.annotate("%.2f" % all_t_cr[i], xy=(all_x_cr[i], all_z_cr[i]), textcoords='data')  # plot numbers of stars
        clb = plt.colorbar(sc)
        clb.ax.set_title(Labels.lbls(v_n_col))



        plt.show()


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