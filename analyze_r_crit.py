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
    def __init__(self, smfiles, out_dir, plot_dir, dirs_not_to_be_included):

        self.input_dirs = smfiles[0].split('/')[:-1]
        # print(self.input_dirs)
        self.dirs_not_to_be_included = dirs_not_to_be_included # name of folders that not to be in the name of out. file.

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
        # r_u_mdot_max = np.zeros((2, len(self.num_files)))
        r_u_mdot_max = []
        lmdot = []
        long_r = np.zeros(1)
        long_i = -1

        mdot_ts_rs = []
        for i in range(len(self.num_files)):

            r = self.mdl[i].get_col('r')
            u = self.mdl[i].get_col('u')
            u_s = self.mdl[i].get_sonic_u()
            lmdot =  np.append( lmdot,  self.mdl[i].get_col('mdot')[-1] )

            # lbl = 'l(Mdot):{}'.format('%.2f' % lmdot)
            # ax1.plot(r, u, '-', color='C' + str(Math.get_0_to_max([i], 9)[i]))
            ax1.plot(r, u, '-', color='black')
            # ax1.plot(r[-1], u[-1], '.', color='C' + str(Math.get_0_to_max([i], 9)[i]))
            ax1.plot(r, u_s, '-', color='gray')
            ax1.annotate(str('%.2f' % lmdot[i]), xy=(r[-1], u[-1]), textcoords='data')

            # --- get intercetion between velocity and sonic velocity profile (plot sonic t)
            r_cr_int, u_s_cr_int = Math.interpolated_intercept(r, u, u_s)
            plt.plot(r_cr_int, u_s_cr_int, 'r', color='orange')
            ts = -1
            if r_cr_int.any():
                ts = self.mdl[i].get_col('t')[Math.find_nearest_index(u,u_s)]
                ax1.annotate(str('%.2f' % ts), xy=(r_cr_int[0], u_s_cr_int[0]), textcoords='data')

            # --- get maximum of every velocity profile
            r_m, u_m =  Math.get_max_by_interpolating(r, u) # use of 1d array for further sorting
            r_u_mdot_max = np.append(r_u_mdot_max, [r_m, u_m, [ lmdot[i] ] ] ) # [] for lmdot bacause r_cr_int and u_s_cr_int are lists

            plt.plot(r_m, u_m, '*', color='red')

            # -- getting the longest (in r) model, so leter to get a critical radii from it
            if long_r.max() < r[-1]:
                long_r = np.append(long_r, r[-1])
                long_i = i


            # --- filling the array for the output
            l = self.mdl[i].get_col('l')[-1]
            m = self.mdl[i].get_col('xm')[-1]
            u_s_cr_int= self.mdl[i].get_col('He4')[0]

            if r_cr_int.any():
                mdot_ts_rs = np.append(mdot_ts_rs, [l, m, u_s_cr_int, lmdot[i], np.float(r_cr_int[0]), ts])
            else:
                mdot_ts_rs = np.append(mdot_ts_rs, [l, m, u_s_cr_int, lmdot[i], 0., 0.])

        # as it r coord first increasing than decreasing - you cannot use it for interpolation,
        # u - coord, which is velocity, is decreasing if mdot is decreasing (or vise versa), but it behaves monotonicall
        # hence, it can be used as a 'r' coordinate, while radius is a 'u'.
        # print(r_u_mdot_max)

        xy_max_sorted = np.sort(r_u_mdot_max.view('i8, i8, i8'), order=['f1'], axis=0).view(np.float)
        r_u_mdot_resh = np.reshape(xy_max_sorted, (len(self.num_files), 3))

        new_r   = r_u_mdot_resh[:,0]
        new_u   = r_u_mdot_resh[:,1]
        new_mdot= r_u_mdot_resh[:,2]

        u_max_grid = np.mgrid[new_u.min():new_u.max():1000j]
        r_max_grid = Math.interp_row(new_u, new_r, u_max_grid)
        # print(r_max_grid, u_max_grid)

        ax1.plot(r_max_grid, u_max_grid, '-', color='gray')

        # --- Find a cross between interplated curve and sonic velocity.

        r = self.mdl[long_i].get_col('r')
        t = self.mdl[long_i].get_col('t')
        u_s = self.mdl[long_i].get_sonic_u()

        # ax1.plot(r, u_s, '-', color='blue')

        ax2 = ax1.twinx()   # for plotting the temperature scale (to see the Temp at critical radius)
        ax2.plot(r, t, '-.', color='brown')

        i1 = Math.find_nearest_index(r, r_max_grid.min()) - 1
        i2 = Math.find_nearest_index(r, r_max_grid.max()) + 1
        print(i1,i2)
        r_cr = r[i1:i2]
        u_s_cr =u_s[i1:i2]
        t_cr = t[i1:i2]


        # x_for_us_grid = np.mgrid[r.min():r.max():1000j]
        int_y_for_us_grid = Math.interp_row(r_cr, u_s_cr, r_max_grid) # corpped radius range
        int_t_for_us_grid = Math.interp_row(r_cr, t_cr, r_max_grid)   # cropped temp range

        ax1.plot(r_max_grid, int_y_for_us_grid, '-', color='green') # corpped radius range
        ax2.plot(r_max_grid, int_t_for_us_grid, '-', color='brown') # cropped temp range

        # --- Critical Radius ---

        r_cr_int, u_s_cr_int = Math.interpolated_intercept(r_max_grid, u_max_grid, int_y_for_us_grid)
        i_of_u_near_yc       = Math.find_nearest_index(int_y_for_us_grid, u_s_cr_int) # for f

        ax1.plot(r_cr_int, u_s_cr_int, 'X', color='blue', label='Rc:{}'.format('%.3f' %  r_cr_int)) # critical Radius
        ax1.annotate('Rs:'+str('%.3f' % r_cr_int), xy=(r_cr_int[0], u_s_cr_int[0]), textcoords='data')

        # --- Critical Temperature ---

        t_crit = int_t_for_us_grid[i_of_u_near_yc]  # temperature at a critical radius
        ax2.plot( r_cr_int, t_crit, 'X', color='red', label='Tc:{}'.format('%.3f' %  t_crit) )           # temp. at critical radius
        ax1.axvline(x=r_cr_int, color='gray', linestyle='solid')

        # --- Critical Mass Loss --- [SUBPLOT]

        ax3 = fig.add_axes([0.65, 0.65, 0.23, 0.23])
        ax3.plot(new_mdot, new_u, '-', color='black')
        ax3.plot(new_mdot, new_u, '.', color='black')
        ax3.set_xlabel(Labels.lbls('mdot'))
        ax3.set_ylabel(Labels.lbls('u'))
        ax3.grid()
        ax3.axhline(y=u_s_cr_int, color='gray', linestyle='--', label='Son_vel: {}'.format( '%.3f' % u_s_cr_int))

        mdot_cr = Math.solv_inter_row(new_mdot, new_u, u_s_cr_int)

        ax3.plot(mdot_cr, u_s_cr_int, 'X', color='black', label='Mdot_cr: {}'.format('%.3f' % np.float(mdot_cr[0])))
        ax3.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
        # ax3.annotate('Mdot_cr:' + str('%.3f' % mdot_cr), xy=(mdot_cr, u_s_cr_int), textcoords='data')





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
        plt.savefig(plot_name)
        plt.show()

        head = '\t {} \t {} \t {} \t\t {} \t {} \t {}'\
            .format('log(L)', 'M(Msun)', 'Yc', 'log(Mdot)', 'Rs(Rsun)', 'log(Ts)')
        tablehead = '{}  {}  {}  {}  {}  {}'\
            .format('log(L)', 'M(Msun)', 'Yc', 'l(Mdot)', 'Rs(Rsun)', 'log(Ts)')

        # --- Adding a first row of values for critical r, mdot, and t, and l, m, Yc for the model, whose sonic vel. was
        # used to interplate the temp amd mdot.

        l = self.mdl[long_i].get_col('l')[-1]
        m = self.mdl[long_i].get_col('xm')[-1]
        yc = self.mdl[long_i].get_col('He4')[0]                            # adding the first row of critical parameters
        mdot_ts_rs = np.insert(mdot_ts_rs, 0, [l, m, yc, mdot_cr, r_cr_int, t_crit])
        mdot_ts_rs = np.reshape(mdot_ts_rs, (len(self.num_files)+1, 6))


        print('Critical values:')
        print(head)
        print(mdot_ts_rs[0,:])


        print('\n')
        print(head)
        print(mdot_ts_rs[1:,:])

        print('\t__Note: Value {} means that solution was not found (extended configutration)'.format(0.))
        print('Critical Radii found: {}'.format(r_cr_int))

        out_name = 'SP_'
        for i in range(len(self.input_dirs)):
            if self.input_dirs[i] not in self.dirs_not_to_be_included and self.input_dirs[i] != '..':
                out_name = out_name + self.input_dirs[i]
                if i < len(self.input_dirs) - 1:
                    out_name = out_name + '_'
        out_name = out_name + '.data'

        print('Results are saved in: {}'.format(self.out_dir + out_name))
        np.savetxt(self.out_dir + out_name,mdot_ts_rs, '%.4f','  ','\n',tablehead,'')

        # 0.93050756
        # 0.93046579

