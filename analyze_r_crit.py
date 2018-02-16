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
from Err_Const_Math_Phys import Errors
from Err_Const_Math_Phys import Math
from Err_Const_Math_Phys import Physics
from Err_Const_Math_Phys import Constants
from Err_Const_Math_Phys import Labels

from OPAL import Read_Table
from OPAL import Row_Analyze
from OPAL import Table_Analyze
from OPAL import OPAL_Interpol
from OPAL import New_Table

from Read_Obs_Numers import Read_Observables
from Read_Obs_Numers import Read_Plot_file
from Read_Obs_Numers import Read_SM_data_File


class Critical_R:
    def __init__(self, smfiles, out_dir, plot_dir):
        self.out_dir = out_dir
        self. plot_dir = plot_dir

        self.num_files = smfiles
        self.mdl = []
        for file in smfiles:
            self.mdl.append(Read_SM_data_File.from_sm_data_file(file))


    def velocity_profile(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        tlt = 'VELOCITY PROFILE'
        plt.title(tlt, loc='left')

        # x_max = np.zeros(len(self.num_files))
        # y_max = np.zeros(len(self.num_files))
        # yx_max = np.zeros((2, len(self.num_files)))
        # xy_max = np.zeros((2, len(self.num_files)))
        xy_max = []
        long_r = np.zeros(1)
        long_i = -1

        mdot_ts_rs = []
        for i in range(len(self.num_files)):

            x = self.mdl[i].get_col('r')
            y = self.mdl[i].get_col('u')
            u_s = self.mdl[i].get_sonic_u()
            lmdot = self.mdl[i].get_col('mdot')[-1]

            # lbl = 'l(Mdot):{}'.format('%.2f' % lmdot)
            ax1.plot(x, y, '-', color='C' + str(Math.get_0_to_max([i], 9)[i]))
            ax1.plot(x[-1], y[-1], '.', color='C' + str(Math.get_0_to_max([i], 9)[i]))
            ax1.plot(x, u_s, '-', color='black')
            ax1.annotate(str('%.2f' % lmdot), xy=(x[-1], y[-1]), textcoords='data')

            # --- get intercetion between velocity and sonic velocity profile (plot sonic t)
            xc, yc = Math.interpolated_intercept(x, y, u_s)
            plt.plot(xc, yc, 'x', color='orange')
            ts = -1
            if xc.any():
                ts = self.mdl[i].get_col('t')[Math.find_nearest_index(y,u_s)]
                ax1.annotate(str('%.2f' % ts), xy=(xc[0], yc[0]), textcoords='data')

            # --- get maximum of every velocity profile
            x_m, y_m =  Math.get_max_by_interpolating(x, y) # use of 1d array for further sorting
            xy_max = np.append(xy_max, [x_m, y_m] )

            plt.plot(x_m, y_m, '*', color='red')

            # -- getting the longest (in r) model, so leter to get a critical radii from it
            if long_r.max() < x[-1]:
                long_r = np.append(long_r, x[-1])
                long_i = i


            # --- filling the array for the output
            l = self.mdl[i].get_col('l')[-1]
            m = self.mdl[i].get_col('xm')[-1]
            yc= self.mdl[i].get_col('He4')[0]

            if xc.any():
                mdot_ts_rs = np.append(mdot_ts_rs, [l, m, yc, lmdot, np.float(xc[0]), ts])
            else:
                mdot_ts_rs = np.append(mdot_ts_rs, [l, m, yc, lmdot, 0., 0.])

        # as it x coord first increasing than decreasing - you cannot use it for interpolation,
        # y - coord, which is velocity, is decreasing if mdot is decreasing (or vise versa), but it behaves monotonicall
        # hence, it can be used as a 'x' coordinate, while radius is a 'y'.
        xy_max_sorted = np.sort(xy_max.view('i8, i8'), order=['f1'], axis=0).view(np.float)
        xy = np.reshape(xy_max_sorted, (len(self.num_files), 2))

        new_x = xy[:,0]
        new_y = xy[:,1]

        ymax_grid = np.mgrid[new_y.min():new_y.max():1000j]
        xmax_grid = Math.interp_row(new_y, new_x, ymax_grid)
        # print(xmax_grid, ymax_grid)

        ax1.plot(xmax_grid, ymax_grid, '-', color='gray')

        # --- Find a cross between interplated curve and sonic velocity.

        x = self.mdl[long_i].get_col('r')
        # y = self.mdl[long_i].get_col('u')
        u_s = self.mdl[long_i].get_sonic_u()

        # ax1.plot(x, u_s, '-', color='blue')


        i1 = Math.find_nearest_index(x, xmax_grid.min()) - 1
        i2 = Math.find_nearest_index(x, xmax_grid.max()) + 1
        print(i1,i2)
        x = x[i1:i2]
        u_s =u_s[i1:i2]


        # x_for_us_grid = np.mgrid[x.min():x.max():1000j]
        int_y_for_us_grid = Math.interp_row(x, u_s, xmax_grid)

        ax1.plot(xmax_grid, int_y_for_us_grid, '-', color='green')

        xc, yc = Math.interpolated_intercept(xmax_grid, ymax_grid, int_y_for_us_grid)

        ax1.plot(xc, yc, 'X', color='red')
        ax1.annotate('Rs:'+str('%.3f' % xc), xy=(xc[0], yc[0]), textcoords='data')


        ax1.set_xlabel(Labels.lbls('r'))
        ax1.set_ylabel(Labels.lbls('u'))

        ax1.grid(which='both')
        ax1.grid(which='minor', alpha=0.2)
        ax1.grid(which='major', alpha=0.2)

        ax1.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
        plot_name = self.plot_dir + 'critical_radius.pdf'
        plt.savefig(plot_name)
        plt.show()

        head = '\t {} \t {} \t {} \t\t {} \t {} \t {}'\
            .format('log(L)', 'M(Msun)', 'Yc', 'log(Mdot)', 'Rs(Rsun)', 'log(Ts)')

        print(head)
        mdot_ts_rs = np.reshape(mdot_ts_rs, (len(self.num_files), 6))

        print(mdot_ts_rs)

        print('\t__Note: Value {} means that solution was not found (extended configutration)'.format(0.))


        np.savetxt('crit_rad.data',mdot_ts_rs, '%.3f','  ','\n','','',head)