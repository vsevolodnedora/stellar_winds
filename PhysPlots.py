#-----------------------------------------------------------------------------------------------------------------------
# Set of classes including:
#   PhysPlots   Contains old plotting teqnics. Used only by Table_Analyze for plotting avilabe kappa region.
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
#-----------------------------------------------------------CLASSES-----------------------------------------------------
from Err_Const_Math_Phys import Errors
from Err_Const_Math_Phys import Math
from Err_Const_Math_Phys import Physics
from Err_Const_Math_Phys import Constants

class PhysPlots:
    def __init__(self):
        pass

    @staticmethod
    def xy_profile(nm_x, nm_y, x1, y1, lx = np.zeros(1,), ly = np.zeros(1,),
                    x2=np.zeros(1,),y2=np.zeros(1,),x3=np.zeros(1,),y3=np.zeros(1,),x4=np.zeros(1,),
                    y4=np.zeros(1,),x5=np.zeros(1,),y5=np.zeros(1,),x6=np.zeros(1,),y6=np.zeros(1,),
                    x7=np.zeros(1, ), y7=np.zeros(1, )):

        plot_name = './results/' + nm_x + '_' + nm_y + 'profile.pdf'

        # plot_name = 'Vel_profile.pdf'

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        plt.title('Velocity Profile')

        plt.plot(x1, y1, '-', color='blue', label='model_1')

        if x2.shape!=(1,) and y2.shape!=(1,):
            plt.plot(x2, y2, '-', color='cyan', label='model_2')

        if x3.shape!=(1,) and y3.shape!=(1,):
            plt.plot(x3, y3, '-', color='green', label='model_3')

        if x4.shape!=(1,) and y4.shape!=(1,):
            plt.plot(x4, y4, '-', color='yellow', label='model_4')

        if x5.shape!=(1,) and y5.shape!=(1,):
            plt.plot(x5, y5, '-', color='orange', label='model_5')

        if x6.shape!=(1,) and y6.shape!=(1,):
            plt.plot(x6, y6, '-', color='red', label='model_6')

        if x7.shape!=(1,) and y7.shape!=(1,):
            plt.plot(x7, y7, '-', color='purple', label='model_6')

        plt.xlabel(nm_x)
        plt.ylabel(nm_y)

        #---------------------------------------MINOR-TICKS-------------------------------
        if lx.shape != (1,):
            major_xticks = np.arange(lx[0], lx[-1] + 1, (lx[-1] -lx[0]) / 5)
            minor_xticks = np.arange(lx[0], lx[-1], (lx[-1] -lx[0]) / 10)
            ax.set_xticks(major_xticks)
            ax.set_xticks(minor_xticks, minor=True)
        # else:
        #     major_xticks = np.arange(x1[0], x1[-1] , (x1[-1] - x1[0]) / 5)
        #     minor_xticks = np.arange(x1[0], x1[-1], (x1[-1] - x1[0]) / 10)

        if ly.shape != (1,):
            major_yticks = np.arange(ly[0], ly[-1] + 1, (ly[-1] -ly[0]) / 5)
            minor_yticks = np.arange(ly[0], ly[-1], (ly[-1] -ly[0]) / 10)
            ax.set_yticks(major_yticks)
            ax.set_yticks(minor_yticks, minor=True)
        # else:
        #     major_yticks = np.arange(y1.min(), y1.max() + 1, (y1.max() - y1.min()) / 5)
        #     minor_yticks = np.arange(y1.min(), y1.min(), (y1.max() - y1.min()) / 10)

        # ax.set_xticks(major_xticks)
        # ax.set_xticks(minor_xticks, minor=True)
        # ax.set_yticks(major_yticks)
        # ax.set_yticks(minor_yticks, minor=True)


        #-------------------------------------VERT/HORISONTAL LINES------------------------------
        # if lim_k1 != None:
        #     lbl = 'k1: ' + str("%.2f" % lim_k1)
        #     plt.axhline(y=lim_k1, color='r', linestyle='dashed', label=lbl)
        #
        # if lim_k2 != None:
        #     lbl = 'k1: ' + str("%.2f" % lim_k2)
        #     plt.axhline(y=lim_k2, color='r', linestyle='dashed', label=lbl)
        #
        # if lim_t1 != None:
        #     lbl = 't1: ' + str("%.2f" % lim_t1)
        #     plt.axvline(x=lim_t1, color='r', linestyle='dashed', label=lbl)
        #
        # if lim_t2 != None:
        #     lbl = 't2: ' + str("%.2f" % lim_t2)
        #     plt.axvline(x=lim_t2, color='r', linestyle='dashed', label=lbl)
        #
        # if it1 != None:
        #     lbl = 'int t1: ' + str("%.2f" % it1)
        #     plt.axvline(x=it1, color='orange', linestyle='dashed', label=lbl)
        #
        # if it2 != None:
        #     lbl = 'int t2: ' + str("%.2f" % it2)
        #     plt.axvline(x=it2, color='orange', linestyle='dashed', label=lbl)


        #----------------------------BOXES------------------------------
        # if any(y2_arr):
        #     ax.fill_between(x_arr, y_arr, y2_arr, label ='Available Region')
        #
        # if it1 != None and it2 != None and lim_k1 != None and lim_k2 != None:
        #     ax.fill_between(np.array([it1, it2]), np.array([lim_k1]), np.array([lim_k2]), label='Interpolation Region')
        #
        # if lim_t1 != None and lim_t2 != None and lim_k1 != None and lim_k2 != None:
        #     ax.fill_between(np.array([lim_t1, lim_t2]), np.array([lim_k1]), np.array([lim_k2]), label='Selected Region')


        ax.grid(which='both')
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.2)

        plt.legend(bbox_to_anchor=(1, 1), loc='upper right', ncol=1)


        plt.savefig(plot_name)


        plt.show()

    @staticmethod
    def xy_2y_profile(nm_x, nm_y, nm_yy, x1, y1, y11,
                    x2=np.zeros(1,), y2=np.zeros(1,), y22=np.zeros(1,),
                    x3=np.zeros(1,), y3=np.zeros(1,), y33=np.zeros(1,),
                    x4=np.zeros(1,), y4=np.zeros(1,), y44=np.zeros(1,),
                    x5=np.zeros(1,), y5=np.zeros(1,), y55=np.zeros(1,),
                    x6=np.zeros(1,), y6=np.zeros(1,), y66=np.zeros(1,),):
        '''***************************WITH-T-as-X-AXIS-------------------------------'''
        fig, ax1 = plt.subplots()

        ax1.plot(x1, y1, '-', color='blue', label='model_1')

        if x2.shape!=(1,) and y2.shape!=(1,):
            ax1.plot(x2, y2, '-', color='cyan', label='model_2')

        if x3.shape!=(1,) and y3.shape!=(1,):
            ax1.plot(x3, y3, '-', color='green', label='model_3')

        if x4.shape!=(1,) and y4.shape!=(1,):
            ax1.plot(x4, y4, '-', color='yellow', label='model_4')

        if x5.shape!=(1,) and y5.shape!=(1,):
            ax1.plot(x5, y5, '-', color='orange', label='model_5')

        if x6.shape!=(1,) and y6.shape!=(1,):
            ax1.plot(x6, y6, '-', color='red', label='model_6')


        # ax1.plot(t2ph, ro2ph, 'gray')
        # ax1.plot(t3ph, ro3ph, 'gray')
        # ax1.plot(t4ph, ro4ph, 'gray')
        # ax1.plot(t5ph, ro5ph, 'gray')
        # ax1.plot(t6ph, ro6ph, 'gray')
        #
        # ax1.plot(t1, ro1, 'b-')
        # ax1.plot(last_elmt(t1), last_elmt(ro1), 'bo')
        # ax1.plot(t2, ro2, 'b-')
        # ax1.plot(last_elmt(t2), last_elmt(ro2), 'bo')
        # ax1.plot(t3, ro3, 'b-')
        # ax1.plot(last_elmt(t3), last_elmt(ro3), 'bo')
        # ax1.plot(t4, ro4, 'b-')
        # ax1.plot(last_elmt(t4), last_elmt(ro4), 'bo')
        # ax1.plot(t5, ro5, 'b-')
        # ax1.plot(last_elmt(t5), last_elmt(ro5), 'bo')
        # ax1.plot(t6, ro6, 'b-')
        # ax1.plot(last_elmt(t6), last_elmt(ro6), 'bo')

        ax1.set_xlabel(nm_x)
        # Make the y-axis label, ticks and tick labels match the line color.
        ax1.set_ylabel(nm_y, color='b')
        ax1.tick_params('y', colors='b')
        ax1.set_xlim(6.2, 4.6)
        plt.grid()

        ax2 = ax1.twinx()

        # ----------------------------EDDINGTON OPACITY------------------------------------
        # ax2.plot(np.mgrid[x1.min():x1.max():100j], np.mgrid[edd_k:edd_k:100j], c='black')


        ax2.plot(x1, y11, '--', color='blue', label='model_1')

        if x2.shape!=(1,) and y22.shape!=(1,):
            ax2.plot(x2, y22, '--', color='cyan', label='model_2')

        if x3.shape!=(1,) and y33.shape!=(1,):
            ax2.plot(x3, y33, '--', color='green', label='model_3')

        if x4.shape!=(1,) and y44.shape!=(1,):
            ax2.plot(x4, y44, '--', color='yellow', label='model_4')

        if x5.shape!=(1,) and y55.shape!=(1,):
            ax2.plot(x5, y55, '--', color='orange', label='model_5')

        if x6.shape!=(1,) and y6.shape!=(1,):
            ax2.plot(x6, y66, '--', color='red', label='model_6')


        # ax2.plot(t1ph, k1ph, 'gray')
        # ax2.plot(t2ph, k2ph, 'gray')
        # ax2.plot(t3ph, k3ph, 'gray')
        # ax2.plot(t4ph, k4ph, 'gray')
        # ax2.plot(t5ph, k5ph, 'gray')
        # ax2.plot(t6ph, k6ph, 'gray')
        #
        # ax2.plot(t1, k1, 'r-')
        # ax2.plot(t2, k2, 'r-')
        # ax2.plot(t3, k3, 'r-')
        # ax2.plot(t4, k4, 'r-')
        # ax2.plot(t5, k5, 'r-')
        # ax2.plot(t6, k6, 'r-')

        ax2.set_ylabel(nm_yy, color='r')
        ax2.tick_params('y', colors='r')

        plt.axvline(x=4.6, color='black', linestyle='solid', label='T = 4.6, He Op Bump')
        plt.axvline(x=5.2, color='black', linestyle='solid', label='T = 5.2, Fe Op Bump')
        plt.axvline(x=6.2, color='black', linestyle='solid', label='T = 6.2, Deep Fe Op Bump')

        # plt.ylim(-8.5, -4)
        fig.tight_layout()
        plt.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
        plt.show()

    @staticmethod
    def Rho_k_plot(f_rho, f_kappa, rho_row = None, kappa_row = None,
                     lim_k1=None, lim_k2=None, case=None, temp=None, step=None, plot_dir = '../data/plot/'):

        plot_name = 'Rho_k_plot.pdf'
        # path = '/media/vnedora/HDD/opal_anal4/'
        path = plot_dir

        if (temp == None and step == None):
            plot_name = ''.join([path, 'plot_one_raw.pdf'])
        if (temp != None and step == None):
            plot_name = ''.join([path, 'T=', str("%.2f" % temp), '.pdf'])
        if (temp != None and step != None):
            plot_name = ''.join([path, str(step), '_T=', str("%.2f" % temp), '.pdf'])

        # Title of the file Cases
        plot_title = 'Rho_k_plot.pdf'
        if (temp == None and step == None):
            plot_title = ''.join(['T = const'])
        if (temp != None and step == None):
            plot_title = ''.join(['T = ', str(temp)])
        if (temp != None and step != None):
            plot_title = ''.join(['T(', str(step), ') = ', str(temp)])

        # case lalbe
        label_case = ''
        if case != None:
            label_case = ''.join(['Case: ', str(case)])

        # x coordinates of the selected region:
        rho1 = f_rho[Math.find_nearest_index(f_kappa, lim_k1)]
        rho2 = f_rho[Math.find_nearest_index(f_kappa, lim_k2)]

        # labels for vertical an horisontal lines
        lbl_rho_lim = ''
        lbl_op_lim = ''
        if lim_k1 != None and lim_k2 != None:
            lbl_rho_lim = ''.join(['Selected dencity(', str("%.2f" % rho1), ' ', str("%.2f" % rho2), ')'])
            lbl_op_lim = ''.join(['Selected opacity(', str("%.2f" % lim_k1), ' ', str("%.2f" % lim_k2), ')'])

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<PLOT>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        plt.title(plot_title)
        # pl.ylim(-4, 4)
        # pl.xlim(-10, 10)

        plt.plot(f_rho, f_kappa, '.',color='blue', label='(T, kap=[])->rho[]')
        if any(rho_row) and any(kappa_row):
            plt.plot(rho_row, kappa_row, 'x', color='black', label='table')

        if lim_k1 != None and lim_k2 != None:
            plt.axvspan(rho1, rho2, color='lightblue', linestyle='dotted', label=lbl_rho_lim)
            plt.axhspan(lim_k1, lim_k2, color='lightblue', linestyle='dotted', label=lbl_op_lim)

            plt.axvline(x=rho1, color='grey', linestyle='dotted')
            plt.axvline(x=rho2, color='grey', linestyle='dotted')

            plt.axhline(y=lim_k1, color='grey', linestyle='dotted')
            plt.axhline(y=lim_k2, color='grey', linestyle='dotted')

        plt.xlabel('log(rho)')
        plt.ylabel('opacity')

        if case != None:
            ax.text(f_rho.min(), f_kappa.mean(), label_case, style='italic',
                bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10})
            # box with data about case and limits of kappa

        if any(rho_row) and any(kappa_row):
            major_xticks = np.arange(rho_row.min(), rho_row.max() + 1, 1)
            minor_xticks = np.arange(rho_row.min(), rho_row.max(), 0.5)
            major_yticks = np.arange(kappa_row.min(), kappa_row.max() + 1, ((kappa_row.max() - kappa_row.min()) / 4))
            minor_yticks = np.arange(kappa_row.min(), kappa_row.max(), ((kappa_row.max() - kappa_row.min()) / 8))
        else:
            major_xticks = np.arange(f_rho.min(), f_rho.max() + 1, 1)
            minor_xticks = np.arange(f_rho.min(), f_rho.max(), 0.5)
            major_yticks = np.arange(f_kappa.min(), f_kappa.max() + 1, ((f_kappa.max() - f_kappa.min()) / 4))
            minor_yticks = np.arange(f_kappa.min(), f_kappa.max(), ((f_kappa.max() - f_kappa.min()) / 8))


        ax.set_xticks(major_xticks)
        ax.set_xticks(minor_xticks, minor=True)
        ax.set_yticks(major_yticks)
        ax.set_yticks(minor_yticks, minor=True)

        ax.grid(which='both')
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.2)

        plt.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)


        plt.savefig(plot_name)

        plt.show()

    @staticmethod
    def k_vs_t(t_arr, y_arr, y2_arr, show = False, save = False,
               lim_k1 = None, lim_k2 = None, lim_t1 = None, lim_t2 = None, it1 = None, it2 = None, plot_dir = '../data/plots/'):

        plot_name = './results/Kappa_Limits.pdf'

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        print(t_arr, '\n', y_arr)
        if len(t_arr) != len(y_arr):
            print('\t Error. len(t_arr {}) != len(y_arr {})'.format(len(t_arr), len(y_arr)))
            print('\t t_arr: {}'.format(t_arr))
            print('\t y_arr: {}'.format(y_arr))
            raise ValueError

        if y2_arr.any() and len(t_arr) != len(y2_arr):
            print('\t Error. len(t_arr {}) != len(y2_arr {})'.format(len(t_arr), len(y2_arr)))
            print('\t t_arr {}:'.format(t_arr))
            print('\t y_arr: {}'.format(y2_arr))
            raise ValueError

        plt.title('Limit Kappa = f(Temperature)')
        plt.plot(t_arr, y_arr, '-', color='blue', label='min k')
        if any(y2_arr):
            plt.plot(t_arr, y2_arr, '-', color='red', label='max k')

        plt.xlabel('t')
        plt.ylabel('kappa')

        #---------------------------------------MINOR-TICKS-------------------------------
        if it1 != None and it2 != None and lim_t1 != None and lim_t2 != None:
            major_xticks = np.array([t_arr.min(), lim_t1, it1, it2, lim_t2, t_arr.max()])
        else:
            major_xticks = np.array([t_arr.min(), t_arr.max()])
        minor_xticks = np.arange(t_arr.min(), t_arr.max(), 0.2)

        #---------------------------------------MAJOR TICKS-------------------------------
        major_yticks = np.arange(y_arr.min(), y_arr.max() + 1, ((y_arr.max() - y_arr.min()) / 4))
        minor_yticks = np.arange(y_arr.min(), y_arr.max(), ((y_arr.max() - y_arr.min()) / 8))

        if any(y2_arr):
            major_yticks = np.arange(y_arr.min(), y2_arr.max() + 1, ((y2_arr.max() - y_arr.min()) / 4))
            minor_yticks = np.arange(y_arr.min(), y2_arr.max(), ((y2_arr.max() - y_arr.min()) / 8))

        if any(y2_arr) and lim_k1 !=None and  lim_k2 != None:
            major_yticks = np.array([y_arr.min(), lim_k1, lim_k2, y2_arr.max()])
            minor_yticks = np.arange(y_arr.min(), y2_arr.max(), ((y2_arr.max() - y_arr.min()) / 10))


        ax.set_xticks(major_xticks)
        ax.set_xticks(minor_xticks, minor=True)
        ax.set_yticks(major_yticks)
        ax.set_yticks(minor_yticks, minor=True)

        #-------------------------------------VERT/HORISONTAL LINES------------------------------
        if lim_k1 != None:
            lbl = 'k1: ' + str("%.2f" % lim_k1)
            plt.axhline(y=lim_k1, color='r', linestyle='dashed', label=lbl)

        if lim_k2 != None:
            lbl = 'k1: ' + str("%.2f" % lim_k2)
            plt.axhline(y=lim_k2, color='r', linestyle='dashed', label=lbl)

        if lim_t1 != None:
            lbl = 't1: ' + str("%.2f" % lim_t1)
            plt.axvline(x=lim_t1, color='r', linestyle='dashed', label=lbl)

        if lim_t2 != None:
            lbl = 't2: ' + str("%.2f" % lim_t2)
            plt.axvline(x=lim_t2, color='r', linestyle='dashed', label=lbl)

        if it1 != None:
            lbl = 'int t1: ' + str("%.2f" % it1)
            plt.axvline(x=it1, color='orange', linestyle='dashed', label=lbl)

        if it2 != None:
            lbl = 'int t2: ' + str("%.2f" % it2)
            plt.axvline(x=it2, color='orange', linestyle='dashed', label=lbl)


        #----------------------------BOXES------------------------------
        if any(y2_arr):
            ax.fill_between(t_arr, y_arr, y2_arr, label = 'Available Region')

        if it1 != None and it2 != None and lim_k1 != None and lim_k2 != None:
            ax.fill_between(np.array([it1, it2]), np.array([lim_k1]), np.array([lim_k2]), label='Interpolation Region')

        if lim_t1 != None and lim_t2 != None and lim_k1 != None and lim_k2 != None:
            ax.fill_between(np.array([lim_t1, lim_t2]), np.array([lim_k1]), np.array([lim_k2]), label='Selected Region')


        ax.grid(which='both')
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.2)

        plt.legend(bbox_to_anchor=(1, 1), loc='upper right', ncol=1)

        if save:
            plot_name = plot_dir+'Kappa_Limits.pdf'
            plt.savefig(plot_name)

        if show:
            plt.show()

    # @staticmethod
    # def rho_vs_t(t_arr, y_arr):
    #
    #     plot_name = 'Rho_t_for_a_kappa.pdf'
    #
    #     fig = plt.figure()
    #     ax = fig.add_subplot(1, 1, 1)
    #
    #     plt.title('Rho = f(Temperature) for one kappa')
    #     plt.plot(t_arr, y_arr, '-', color='blue', label='k1')
    #
    #     plt.ylim(y_arr.min(), y_arr.max())
    #     plt.xlim(t_arr.min(), t_arr.max())
    #
    #     plt.xlabel('t')
    #     plt.ylabel('rho')
    #
    #     major_xticks = np.arange(t_arr.min(), t_arr.max()+0.1, (t_arr.max() - t_arr.min())/4)
    #     minor_xticks = np.arange(t_arr.min(), t_arr.max(), (t_arr.max() - t_arr.min())/8)
    #
    #     major_yticks = np.arange(y_arr.min(), y_arr.max() + 0.1, ((y_arr.max() - y_arr.min()) / 4))
    #     minor_yticks = np.arange(y_arr.min(), y_arr.max(), ((y_arr.max() - y_arr.min()) / 8))
    #
    #
    #     ax.set_xticks(major_xticks)
    #     ax.set_xticks(minor_xticks, minor=True)
    #     ax.set_yticks(major_yticks)
    #     ax.set_yticks(minor_yticks, minor=True)
    #
    #     ax.grid(which='both')
    #     ax.grid(which='minor', alpha=0.2)
    #     ax.grid(which='major', alpha=0.2)
    #
    #     plt.legend(bbox_to_anchor=(1, 1), loc='upper right', ncol=1)
    #
    #     plt.savefig(plot_name)
    #
    #
    #     plt.show()

    @staticmethod
    def t_rho_kappa(t, rho, kappa, edd_1 = np.zeros((1,)),
                    m_t = np.zeros((1,)), m_rho =  np.zeros((1,))):

        name = './results/t_rho_kappa.pdf'
        plt.figure()


        levels = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

        pl.xlim(t.min(), t.max())
        pl.ylim(rho.min(), rho.max())
        contour_filled = plt.contourf(t, rho, 10 ** (kappa), levels)
        plt.colorbar(contour_filled)
        contour = plt.contour(t, rho, 10 ** (kappa), levels, colors='k')
        plt.clabel(contour, colors='k', fmt='%2.1f', fontsize=12)
        plt.title('OPACITY PLOT')
        plt.xlabel('Log(T)')
        plt.ylabel('log(rho)')
        plt.axvline(x=4.2, color='r', linestyle='dashed', label='HeI Bump')
        plt.axvline(x=4.6, color='r', linestyle='dashed', label='HeII Fe Bump')
        plt.axvline(x=5.2, color='r', linestyle='dashed', label='Fe Bump')
        plt.axvline(x=6.2, color='r', linestyle='dashed', label='Deep Fe Bump')
        # plt.axhline(y = vrho, color='r', linestyle='dashed', label = lbl2)
        # pl.plot(t_edd, rho_edd, marker='o', color = 'r')
        if edd_1.any():
            pl.plot(edd_1[0, :], edd_1[1, :], '-', color='w')
        # if edd_2.any():
        #     pl.plot(edd_2[0, :], edd_2[1, :], '-', color='w')
        # if edd_3.any():
        #     pl.plot(edd_3[0, :], edd_3[1, :], '-', color='w')

        if m_rho.any() and m_t.any():
            pl.plot(m_t, m_rho, '-', color='maroon')
            pl.plot(m_t[-1], m_rho[-1], 'o', color='maroon')

        # proc=subprocess.Popen(shlex.split('lpr {f}'.format(f=fname)))
        # plt.legend()


        plt.savefig(name)

        plt.show()

    @staticmethod
    def t_kappa_rho(t, kappa, rho2d, mins=None, p1_t = None, p1_lm = None, val1_mdot = None,
                    p2_t = None, p2_lm = None, val2_mdot = None,
                    p3_t=None, p3_lm=None, val3_mdot=None,
                    p4_t=None, p4_lm=None, val4_mdot=None):

        name = './results/t_LM_Mdot_plot.pdf'

        plt.figure()

        # if new_levels != None:
        #     levels = new_levels
        # else:
        #     levels = [-8, -7, -6, -5, -4, -3, -2]

        pl.xlim(t.min(), t.max())
        pl.ylim(kappa.min(), kappa.max())
        levels = [-7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2]
        #levels = [-10, -9, -8, -7, -6, -5, -4]
        contour_filled = plt.contourf(t, kappa, rho2d.T, levels)
        plt.colorbar(contour_filled)
        contour = plt.contour(t, kappa, rho2d.T, levels, colors='k')
        plt.clabel(contour, colors='k', fmt='%2.1f', fontsize=12)
        plt.title('MASS LOSS PLOT')
        plt.xlabel('Log(t)')
        plt.ylabel('log(L/M)')
        # plt.axvline(x=4.2, color='r', linestyle='dashed', label='HeI Bump')
        # plt.axvline(x=4.6, color='r', linestyle='dashed', label='HeII Fe Bump')
        # plt.axvline(x=5.2, color='r', linestyle='dashed', label='Fe Bump')
        # plt.axvline(x=6.2, color='r', linestyle='dashed', label='Deep Fe Bump')
        # if p1_t != None and p1_lm != None:
        #     plt.axvline(x=p1_t, color='w', linestyle='dashed', label='p_t: {}, p_L/M: {}'.format("%.2f" % p1_t, "%.2f" % p1_lm))
        #     plt.axhline(y=p1_lm, color='w', linestyle='dashed', label='Expected M_dot: {}'.format("%.2f" % val1_mdot))

        plt.plot(mins[0,:], mins[1,:], '-', color='blue', label='min Mdot')

        if p1_t != None and p1_lm != None and val1_mdot != None:
            plt.plot([p1_t], [p1_lm], marker='x', markersize=9, color="blue",
                     label='Model 1: T_s {} , L/M {} , Mdot {}'.format(p1_t, "%.2f" % p1_lm, "%.2f" % val1_mdot))

        if p2_t != None and p2_lm != None and val2_mdot != None:
            plt.plot([p2_t], [p2_lm], marker='x', markersize=9, color="cyan",
                     label='Model 1: T_s {} , L/M {} , Mdot {}'.format(p2_t, "%.2f" % p2_lm, "%.2f" % val2_mdot))

        if p3_t != None and p3_lm != None and val3_mdot != None:
            plt.plot([p3_t], [p3_lm], marker='x', markersize=9, color="magenta",
                     label='Model 1: T_s {} , L/M {} , Mdot {}'.format(p3_t, "%.2f" % p3_lm, "%.2f" % val3_mdot))

        if p4_t != None and p4_lm != None and val4_mdot != None:
            plt.plot([p4_t], [p4_lm], marker='x', markersize=9, color="red",
                     label='Model 2: T_s {} , L/M {} , Mdot {}'.format(p4_t, "%.2f" % p4_lm, "%.2f" % val4_mdot))

        # plt.axhline(y = vrho, color='r', linestyle='dashed', label = lbl2)
        # pl.plot(t_edd, rho_edd, marker='o', color = 'r')
        # if edd_1.any():
        #     pl.plot(edd_1[0, :], edd_1[1, :], '-', color='w')
        # if edd_2.any():
        #     pl.plot(edd_2[0, :], edd_2[1, :], '-', color='w')
        # if edd_3.any():
        #     pl.plot(edd_3[0, :], edd_3[1, :], '-', color='w')


        # proc=subprocess.Popen(shlex.split('lpr {f}'.format(f=fname)))
        plt.legend()


        plt.savefig(name)

        plt.show()


    # @staticmethod
    # def t_kappa_rho(t, kappa, rho2d, new_levels = None, save = True):
    #
    #     plt.figure()
    #
    #     # if new_levels != None:
    #     #     levels = new_levels
    #     # else:
    #     #     levels = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    #
    #     pl.xlim(t.min(), t.max())
    #     pl.ylim(kappa.min(), kappa.max())
    #     contour_filled = plt.contourf(t, kappa, rho2d.T)
    #     plt.colorbar(contour_filled)
    #     contour = plt.contour(t, kappa, rho2d.T, colors='k')
    #     plt.clabel(contour, colors='k', fmt='%2.1f', fontsize=12)
    #     plt.title('DENSITY PLOT')
    #     plt.xlabel('Log(t)')
    #     plt.ylabel('log(kappa)')
    #     plt.axvline(x=4.2, color='r', linestyle='dashed', label='HeI Bump')
    #     plt.axvline(x=4.6, color='r', linestyle='dashed', label='HeII Fe Bump')
    #     plt.axvline(x=5.2, color='r', linestyle='dashed', label='Fe Bump')
    #     plt.axvline(x=6.2, color='r', linestyle='dashed', label='Deep Fe Bump')
    #     # plt.axhline(y = vrho, color='r', linestyle='dashed', label = lbl2)
    #     # pl.plot(t_edd, rho_edd, marker='o', color = 'r')
    #     # if edd_1.any():
    #     #     pl.plot(edd_1[0, :], edd_1[1, :], '-', color='w')
    #     # if edd_2.any():
    #     #     pl.plot(edd_2[0, :], edd_2[1, :], '-', color='w')
    #     # if edd_3.any():
    #     #     pl.plot(edd_3[0, :], edd_3[1, :], '-', color='w')
    #
    #
    #     # proc=subprocess.Popen(shlex.split('lpr {f}'.format(f=fname)))
    #     # plt.legend()
    #     fname = 'k_t_rho_plot.pdf'
    #
    #     plt.savefig(fname)
    #
    #     plt.show()

    @staticmethod
    def t_mdot_lm(t, mdot, lm, p1_t = None, p1_mdot = None, p1_lm = None,
                  p2_t = None, p2_mdot = None, p2_lm = None,
                  p3_t = None, p3_mdot = None, p3_lm = None,
                  p4_t = None, p4_mdot = None, p4_lm = None):
        name = './results/t_mdot_lm_plot.pdf'

        plt.figure()

        # if new_levels != None:
        #     levels = new_levels
        # else:
        #     levels = [-8, -7, -6, -5, -4, -3, -2]

        pl.xlim(t.min(), t.max())
        pl.ylim(mdot.min(), mdot.max())
        levels = [3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.1, 4.2, 4.3, 4.6, 4.8, 5.0, 5.2]
        contour_filled = plt.contourf(t, mdot, lm, levels)
        plt.colorbar(contour_filled)
        contour = plt.contour(t, mdot, lm, levels, colors='k')
        plt.clabel(contour, colors='k', fmt='%2.1f', fontsize=12)
        plt.title('L/M PLOT')
        plt.xlabel('Log(t_s)')
        plt.ylabel('log(M_dot)')
        # plt.axvline(x=4.2, color='r', linestyle='dashed', label='HeI Bump')
        # plt.axvline(x=4.6, color='r', linestyle='dashed', label='HeII Fe Bump')
        # plt.axvline(x=5.2, color='r', linestyle='dashed', label='Fe Bump')
        # plt.axvline(x=6.2, color='r', linestyle='dashed', label='Deep Fe Bump')

        # if test_t1 != None and test_mdot1 != None and test_lm1 != None:
        #     plt.axvline(x=test_t1, color='c', linestyle='dashed', label='T_s: {} , mdot: {}'.format(test_t1, "%.2f" % test_mdot1))
        #     plt.axhline(y=test_mdot1, color='c', linestyle='dashed',
        #                 label='Star L/M: {}'.format("%.2f" % test_lm1))
        #
        # if test_t2 != None and test_mdot2 != None and test_lm2 != None:
        #     plt.axvline(x=test_t2, color='m', linestyle='dashed', label='T_s: {} , mdot: {}'.format(test_t2, "%.2f" % test_mdot2))
        #     plt.axhline(y=test_mdot2, color='m', linestyle='dashed',
        #                 label='Star L/M: {}'.format("%.2f" % test_lm2))

        if p1_t != None and p1_mdot != None and p1_lm != None:
            plt.plot([p1_t], [p1_mdot], marker='x', markersize=9, color="blue",
                     label='Model 1: T_s {} , mdot {} , L/M: {}'.format(p1_t, "%.2f" % p1_mdot, "%.2f" % p1_lm))

        if p2_t != None and p2_mdot != None and p2_lm != None:
            plt.plot([p2_t], [p2_mdot], marker='x', markersize=9, color="cyan",
                     label='Model 2: T_s {} , mdot {} , L/M {}'.format(p2_t, "%.2f" % p2_mdot, "%.2f" % p2_lm))

        if p3_t != None and p3_mdot != None and p3_lm != None:
            plt.plot([p3_t], [p3_mdot], marker='x', markersize=9, color="magenta",
                     label='Model 1: T_s {} , mdot {} , L/M: {}'.format(p3_t, "%.2f" % p3_mdot, "%.2f" % p3_lm))

        if p4_t != None and p4_mdot != None and p4_lm != None:
            plt.plot([p4_t], [p4_mdot], marker='x', markersize=9, color="red",
                     label='Model 2: T_s {} , mdot {} , L/M {}'.format(p4_t, "%.2f" % p4_mdot, "%.2f" % p4_lm))

        # proc=subprocess.Popen(shlex.split('lpr {f}'.format(f=fname)))
        plt.legend()
        fname = 't_mdot_lm_plot.pdf'

        plt.savefig(name)

        plt.show()

    @staticmethod
    def lm_min_mdot(min_mdot_arr, lm_arr, x1 = None, y1 = None,
                                                            x2 = None, y2 = None,
                                                            x3 = None, y3 = None,
                                                            x4 = None, y4 = None):

        plot_name = './results/Min_Mdot.pdf'

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        plt.title('L/M = f(min M_dot)')
        plt.plot(min_mdot_arr, lm_arr, '-', color='blue', label='min k')


#-------------

        plt.ylim(lm_arr.min(), lm_arr.max())
        plt.xlim(min_mdot_arr.min(), min_mdot_arr.max())

        plt.xlabel('log(M_dot)')
        plt.ylabel('log(L/M)')


        major_xticks = np.array([-6.5,-6,-5.5,-5,-4.5,-4,-3.5])
        minor_xticks = np.arange(-7.0,-3.5,0.1)

        major_yticks = np.array([3.8,3.9,4.0,4.1,4.2,4.3,4.4,4.5])
        minor_yticks = np.arange(3.8, 4.5, 0.05)

        # major_xticks = np.arange(min_mdot_arr.min(), min_mdot_arr.max()+0.1, (min_mdot_arr.max() - min_mdot_arr.min())/4)
        # minor_xticks = np.arange(min_mdot_arr.min(), min_mdot_arr.max(), (min_mdot_arr.max() - min_mdot_arr.min())/8)
        #
        # major_yticks = np.arange(lm_arr.min(), lm_arr.max() + 0.1, ((lm_arr.max() - lm_arr.min()) / 4))
        # minor_yticks = np.arange(lm_arr.min(), lm_arr.max(), ((lm_arr.max() - lm_arr.min()) / 8))



        ax.grid(which='major', alpha=0.2)

        plt.legend(bbox_to_anchor=(1, 1), loc='upper right', ncol=1)

        if x1 != None and y1 != None:
            plt.plot([x1], [y1],  marker='x', markersize=9, color="blue",
                     label='Model 1: Mdot {} , L/M {}'.format("%.2f" % x1, "%.2f" % y1))
        if x2 != None and y2 != None:
            plt.plot([x2], [y2],  marker='x', markersize=9, color="cyan",
                     label='Model 2: Mdot {} , L/M {}'.format("%.2f" % x2, "%.2f" % y2))
        if x3 != None and y3 != None:
            plt.plot([x3], [y3],  marker='x', markersize=9, color="magenta",
                     label='Model 3: Mdot {} , L/M {}'.format("%.2f" % x3, "%.2f" % y3))
        if x4 != None and y4 != None:
            plt.plot([x4], [y4],  marker='x', markersize=9, color="red",
                     label='Model 4: Mdot {} , L/M {}'.format("%.2f" % x4, "%.2f" % y4))


        ax.set_xticks(major_xticks)
        ax.set_xticks(minor_xticks, minor=True)
        ax.set_yticks(major_yticks)
        ax.set_yticks(minor_yticks, minor=True)

        ax.grid(which='both')
        ax.grid(which='minor', alpha=0.2)


        ax.fill_between(min_mdot_arr, lm_arr, color="orange", label = 'Mdot < Minimun')


        # if x1 != None and y1 != None:
        #     plt.axvline(x=x1, color='g', linestyle='dashed', label='Model1 10sm')
        #     plt.axhline(y=y1, color='g', linestyle='dashed', label=' ')
        #
        # if x2 != None and y2 != None:
        #     plt.axvline(x=x2, color='g', linestyle='dashed', label='Model2 ')
        #     plt.axhline(y=y2, color='g', linestyle='dashed', label=' ')
        #
        # if x3 != None and y3 != None:
        #     plt.axvline(x=x3, color='g', linestyle='dashed', label='Model3 ')
        #     plt.axhline(y=y3, color='g', linestyle='dashed', label=' ')
        # if x4 != None and y4 != None:
        #     plt.axvline(x=x4, color='r', linestyle='dashed', label='Model3 ')
        #     plt.axhline(y=y4, color='r', linestyle='dashed', label=' ')

        plt.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
        plt.savefig(plot_name)


        plt.show()