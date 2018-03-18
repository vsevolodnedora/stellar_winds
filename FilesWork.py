#-----------------------------------------------------------------------------------------------------------------------
# Set of classes including:
#   Read_Observables   Reads the provided .data file, selecting the var_names from the first row and values from others
#   Read_Plot_file     Reads the .plot1 file
#   Read_SM_data_File  Reads sm.data file
#-----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------MAIN-LIBRARIES-----------------------------------------------------
import sys
import pylab as pl
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
#-----------------------------------------------------------------------------------------------------------------------

class Save_Load_tables:
    def __init__(self):
        pass

    @staticmethod
    def save_table(d2arr, opal_used, name, x_name, y_name, z_name, output_dir = '../data/output/'):

        header = np.zeros(len(d2arr)) # will be first row with limtis and
        # header[0] = x1
        # header[1] = x2
        # header[2] = y1
        # header[3] = y2
        # tbl_name = 't_k_rho'
        # op_and_head = np.vstack((header, d2arr))  # arraching the line with limits to the array

        part = opal_used.split('/')[-1]
        full_name = output_dir + name + '_' + part  # dir/t_k_rho_table8.data

        np.savetxt(full_name, d2arr, '%.4f', '  ', '\n',
                   '\nINTERPOLATED OPAL {} TABLE for {} relation'.format(part, name), '',
                   '# {} | {} | {} | {} |'
                   .format(opal_used, x_name, y_name, z_name))

        # np.savetxt(full_name, d2arr, '%.4f', '  ', '\n',
        #            '\nINTERPOLATED OPAL {} TABLE for {} relation'.format(part, name), '',
        #            '# {} | {} {} {} | {} {} {} | {} | {} | {}'
        #            .format(opal_used, x_name, x1, x2, y_name, y1, y2, z_name, n_int, n_out))

    @staticmethod
    def load_table(name, x_name, y_name, z_name, opal_used, dir = '../data/output/'):
        part = opal_used.split('/')[-1]
        full_name = dir + name + '_' + part

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

        if r_table != opal_used:
            raise NameError('Read OPAL | {} | not the same is opal_used | {} |'.format(r_table, opal_used))

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

class Creation:

    def __init__(self, opal_name, t1, t2, n_interp = 1000, load_lim_cases = False,
                 output_dir = '../data/output/', plot_dir = '../data/plots/'):
        self.op_name = opal_name
        self.t1 = t1
        self.t2 = t2
        self.n_inter = n_interp

        self.out_dir = output_dir
        self.plot_dir = plot_dir

        self.opal = OPAL_Interpol(opal_name, n_interp)
        self.tbl_anl = Table_Analyze(opal_name, n_interp, load_lim_cases, output_dir, plot_dir)

    def save_t_rho_k(self, rho1 = None, rho2=None):
        op_cl = OPAL_Interpol(self.op_name, self.n_inter)
        t1, t2, rho1, rho2 = op_cl.check_t_rho_limits(self.t1, self.t2, rho1, rho2)
        op_table = op_cl.interp_opal_table(self.t1, self.t2, rho1, rho2)

        Save_Load_tables.save_table(op_table, self.op_name,'t_rho_k','t','rho','k',self.out_dir)

    def save_t_k_rho(self, llm1=None, llm2=None, n_out = 1000):

        k1, k2 = Physics.get_k1_k2_from_llm1_llm2(self.t1, self.t2, llm1, llm2) # assuming k = 4 pi c G (L/M)

        global t_k_rho
        t_k_rho = self.tbl_anl.treat_tasks_interp_for_t(self.t1, self.t2, n_out, self.n_inter, k1, k2).T

        Save_Load_tables.save_table(t_k_rho, self.op_name, 't_k_rho', 't', 'k', 'rho', self.out_dir)
        print('\t__Note. Table | t_k_rho | has been saved in {}'.format(self.out_dir))
        # self.read_table('t_k_rho', 't', 'k', 'rho', self.op_name)
        # def save_t_llm_vrho(self, llm1=None, llm2=None, n_out = 1000):

    def from_t_k_rho__to__t_lm_rho(self):
        t_k_rho = Save_Load_tables.load_table('t_k_rho', 't', 'k', 'rho', self.op_name)

        t = t_k_rho[0, 1:]
        k = t_k_rho[1:, 0]
        rho2d = t_k_rho[1:, 1:]

        lm = Physics.logk_loglm(k, 1)

        t_lm_rho = Math.invet_to_ascending_xy( Math.combine(t, lm, rho2d) )

        Save_Load_tables.save_table(t_lm_rho, self.op_name, 't_lm_rho', 't', 'lm', 'rho', self.out_dir)
        print('\t__Note. Table | t_lm_rho | for | {} | has been saved in {}'.format(self.op_name, self.out_dir))

    def save_t_llm_vrho(self, l_or_lm_name):
        '''
        Table required: t_k_rho (otherwise - won't work) [Run save_t_k_rho() function ]
        :param l_or_lm_name:
        :return:
        '''

        # 1 load the t_k_rho
        t_k_rho = Save_Load_tables.load_table('t_k_rho', 't', 'k', 'rho', self.op_name)

        k = t_k_rho[0, 1:]
        t = t_k_rho[1:, 0]
        rho2d = t_k_rho[1:, 1:]

        vrho = Physics.get_vrho(t, rho2d.T, 2) # mu = 1.34 by default | rho2d.T is because in OPAL t is Y axis, not X.

        # ----------------------------- SELECT THE Y AXIS -----------------
        if l_or_lm_name == 'l':
            l_lm_arr = Physics.lm_to_l(Physics.logk_loglm(k, True))  # Kappa -> L/M -> L
        else:
            l_lm_arr = Physics.logk_loglm(k, 1)


        l_lm_arr = np.flip(l_lm_arr, 0)  # accounting for if k1 > k2 the l1 < l2 or lm1 < lm2
        vrho     = np.flip(vrho, 0)

        global t_llm_vrho
        t_llm_vrho = Math.combine(t, l_lm_arr, vrho)
        name = 't_'+ l_or_lm_name + '_vrho'

        Save_Load_tables.save_table(t_llm_vrho, self.op_name, name, 't', l_or_lm_name, '_vrho', self.out_dir)

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
        t_llm_vrho = Save_Load_tables.load_table(fname, 't', l_or_lm, '_vrho', self.op_name)
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
        Save_Load_tables.save_table(t_llm_mdot, self.op_name, 't_'+l_or_lm+'_mdot','t', l_or_lm, 'mdot', self.out_dir)

class SP_file_work():

    def __init__(self, sp_files, out_dir, plot_dir):
        self.sp_files = sp_files
        self.out_dir = out_dir
        self.plot_dir = plot_dir

        self.spmdl = []
        for file in self.sp_files:
            self.spmdl.append(Read_SP_data_file(file, self.out_dir, self.plot_dir))


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

    def save_y_yc_z_relation(self, y_v_n, z_v_n, opal_used, save, plot=False, yc_prec=0.1, depth=100):

        yc, cls = self.separate_sp_by_crit_val('Yc', yc_prec)

        def interp(x, y, x_grid):
            f = interpolate.interp1d(x, y, kind='cubic', bounds_error=False)
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
            y_pol, z_pol = Math.fit_plynomial(y_z_shaped[:, 0], y_z_shaped[:, 1], 3, depth, y_grid)
            z2d_pol = np.vstack((z2d_pol, z_pol))
            color = 'C' + str(int(yc[i] * 10)-1)
            ax2.plot(y_pol, z_pol, '--', color=color)
            ax2.plot(y_z_shaped[:, 0], y_z_shaped[:, 1], '.', color=color, label='yc:{}'.format("%.2f" % yc[i]))

            '''------------------------------INTERPOLATION ONLY---------------------------------------'''
            y_int, z_int = interp(y_z_shaped[:, 0], y_z_shaped[:, 1], y_grid)
            z2d_int = np.vstack((z2d_int, z_int))
            ax1.plot(y_int, z_int, '--', color=color)
            ax1.plot(y_z_shaped[:, 0], y_z_shaped[:, 1], '.', color=color, label='yc:{}'.format("%.2f" % yc[i]))

        ax1.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
        ax2.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)

        z2d_int = np.delete(z2d_int, 0, 0)
        z2d_pol = np.delete(z2d_pol, 0, 0)

        yc_llm_m_pol = Math.combine(yc, y_grid, z2d_pol.T)  # changing the x/y
        yc_llm_m_int = Math.combine(yc, y_grid, z2d_int.T)  # changing the x/y

        table_name = '{}_{}_{}'.format('yc', y_v_n, z_v_n)
        if save == 'int':
            Save_Load_tables.save_table(yc_llm_m_int, opal_used, table_name, 'yc', y_v_n, z_v_n)
        if save == 'pol':
            Save_Load_tables.save_table(yc_llm_m_pol, opal_used, table_name, 'yc', y_v_n, z_v_n)

        # Save_Load_tables.save_table(yc_llm_m_pol, opal_used, table_name, 'yc', y_v_n, z_v_n)

        if plot:

            levels = []

            if z_v_n == 'r':
                levels = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2., 2.1, 2.2, 2.3, 2.4, 2.5]
            if z_v_n == 'm':
                levels = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
            if z_v_n == 'mdot':
                levels = [-6.0, -5.9, -5.8, -5.7, -5.6, -5.5, -5.4, -5.3, -5.2, -5.1, -5., -4.9, -4.8, -4.7, -4.6, -4.5]
            if z_v_n == 'l':
                levels = [5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4]
            if z_v_n == 'lm':
                levels = [4.0, 4.05, 4.1, 4.15, 4.2, 4.25, 4.3, 4.35,  4.4, 4.45,
                          4.5, 4.55,  4.6, 4.65,  4.7, 4.75, 4.8, 4.85,  4.9, 4.95, 5.0]
            if z_v_n == 't':
                levels = [5.15, 5.16,5.17,5.18,5.19,5.20,5.21,5.22,5.23,5.24,5.25,5.26,5.27,5.28,5.29,5.30]


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

        # yc_llm_m_pol

    @staticmethod
    def yc_x__to__y__sp(yc_value, x_v_n, y_v_n, x_inp, opal_used, dimension=0):
        '''
        LOADS the table with given v_ns and extract the row with given Yc and interpolateds the y_value, for x_val given
        :param yc_value:
        :param x_v_n:
        :param y_v_n:
        :param x_inp:
        :param opal_used:
        :param dimension:
        :return:
        '''

        name = '{}_{}_{}'.format('yc', x_v_n, y_v_n)
        yc_x_y = Save_Load_tables.load_table(name, 'yc', x_v_n, y_v_n, opal_used)
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
            if x_inp >= x_arr.min() and x_inp <= x_arr.max():
                y_arr = y2d[:, ind_yc]
                # lm_arr = []
                # for i in range(len(y_arr)):
                #     lm_arr = np.append(lm_arr, [x_arr[i], y_arr[i]])
                #
                # lm_arr_sort = np.sort(lm_arr.view('float64, float64'), order=['f0'], axis=0).view(np.float)
                # lm_arr_shaped = np.reshape(lm_arr_sort, (len(y_arr), 2))

                f = interpolate.UnivariateSpline(x_arr, y_arr)
                y = f(x_inp)
                # print(log_l, y)

                return x_inp, y
            else:
                raise ValueError('Given l({}) not in available range of l:({}, {})'.format(x_inp, x_arr.min(), x_arr.max()))

        if dimension == 1:
            x_arr_f = []
            y_arr_f = []
            for i in range(len(x_inp)):
                if x_inp[i] >= x_arr.min() and x_inp[i] <= x_arr.max():
                    f = interpolate.UnivariateSpline(x_arr, y2d[:, ind_yc])
                    y_arr_f = np.append(x_arr_f, f(x_inp[i]))
                    x_arr_f = np.append(x_arr_f, x_inp[i])
                # else:
                #     raise ValueError(
                #         'Given x({}, {}) not in available range of x:({}, {})'.format(x_inp[0], x_inp[-1], x_arr.min(), x_arr.max()))

            return x_arr_f, y_arr_f


    @staticmethod
    def l_y_to_m(log_l, yc_value, opal_used, dimension=0):
        '''
        RETURNS l, m (if l dimension = 0) and l_arr, m_arr (if dimension = 1)
        :param log_l:
        :param yc_value:
        :param dimension:
        :return:
        '''
        # from FilesWork import Save_Load_tables

        yc_l_m = Save_Load_tables.load_table('yc_l_m', 'yc', 'l', 'm', opal_used)
        l_arr = yc_l_m[1:, 0]
        yc_arr= yc_l_m[0, 1:]
        m2d = yc_l_m[1:, 1:]

        # yc_value = np.float("%.3f" % yc_value)

        if yc_value in yc_arr:
            ind_yc = Math.find_nearest_index(yc_arr, yc_value)
        else:
            raise ValueError('Given yc_arr({}) is not in available yc_arr:({})'.format(yc_value, yc_arr))

        if dimension == 0:
            if log_l >= l_arr.min() and log_l <= l_arr.max():
                m_arr = m2d[:, ind_yc]
                # lm_arr = []
                # for i in range(len(m_arr)):
                #     lm_arr = np.append(lm_arr, [l_arr[i], m_arr[i]])
                #
                # lm_arr_sort = np.sort(lm_arr.view('float64, float64'), order=['f0'], axis=0).view(np.float)
                # lm_arr_shaped = np.reshape(lm_arr_sort, (len(m_arr), 2))

                f = interpolate.UnivariateSpline(l_arr, m_arr)
                m = f(log_l)
                # print(log_l, m)

                return log_l, m
            else:
                raise ValueError('Given l({}) not in available range of l:({}, {})'.format(log_l, l_arr.min, l_arr.max))
        if dimension == 1:
            m_arr_f = []
            l_arr_f = []
            for i in range(len(log_l)):
                if log_l[i] >= l_arr.min() and log_l[i] <= l_arr.max():
                    f = interpolate.UnivariateSpline(l_arr, m2d[ind_yc, :])
                    m_arr_f = np.append(m_arr_f, f(log_l[i]))
                    l_arr_f = np.append(l_arr_f, log_l[i])
                else:
                    raise ValueError('Given l({}) not in available range of l:({}, {})'.format(log_l, l_arr.min, l_arr.max))

            return l_arr_f, m_arr_f

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

class Read_Observables:

    def __init__(self, observ_name, m_l_rel_yc = None, clump_used=4, clump_req=4):
        '''

        :param observ_name:
        :param yc: -1 (None, use m), 0.1 to 0.99 (My relation for diff yc), 10 (use Langer et al. 1989)
        :param clump_used:
        :param clump_req:
        '''
        self.file_name = observ_name

        self.clump = clump_used
        self.new_clump = clump_req

        self.m_l_rel_yc = m_l_rel_yc

        self.table = []
        with open(observ_name, 'r') as f:
            for line in f:
                if '#' not in line.split() and line.strip(): # if line is not empty and does not contain '#'
                    self.table.append(line)


        self.names = self.table[0].split()
        self.num_stars = len(self.table)-1 # as first row is of var names


        if len(self.names) != len(self.table[1].split()):
            print('\t___Error. Number of vars in list({}) != number of cols in observ.data file({}) '
                     '|Read_Observables, __init__|'.format(len(self.names), len(self.table[1].split())))
        print('\t__Note: Data include following paramters:\n\t | {} |'.format(self.table[0].split()))

        self.table.remove(self.table[0])  # removing the var_names line from the array. (only actual values left)


        # -----------------DISTRIBUTING READ COLUMNS-------------
        self.num_v_n = ['N', 't', 'm', 'l', 'mdot', 'v_inf', 'eta']  # list of used v_n's; can be extended upon need ! 'N' should be first
        self.cls = 'class'                           # special variable

        self.str_v_n = []

        # print(self.get_star_class(75.))
        self.set_num_str_tables()
        # print(self.get_table_num_par('l', 110))
        # print(self.get_star_class(110))

    def set_num_str_tables(self):

        self.numers = np.zeros((self.num_stars, len(self.num_v_n)))
        self.clses= []

        #----------------------------------------SET THE LIST OF CLASSES------------------
        for i in range(len(self.table)):
            n = self.names.index(self.cls)
            if self.cls in self.names:
                self.clses.append(self.table[i].split()[n])

        #-----------------------------------------SET 2D ARRAY OF NUM PARAMETERS---------
        for i in range(len(self.table)):
            for j in range(len(self.num_v_n)):
                n = self.names.index(self.num_v_n[j])
                if self.num_v_n[j] in self.names:
                    self.numers[i, j] =  np.float(self.table[i].split()[n])

        #-----------------------------------------SETTING THE CATALOGUE NAMES------------
        self.stars_n = self.numers[:,0]

        if len(self.numers[:,0])!= len(self.clses):
            raise ValueError('Size of str. vars ({}) != size of num.var ({})'.format( len(self.numers[:,0]) ,len(self.clses) ))

        # print(self.numers, self.clses)
        print('\n\t__Note. In file: {} total {} stars are loaded. \n\t  Available numerical parameters: {} '
              .format(self.file_name, len(self.numers), self.num_v_n))

    def modify_value(self, v_n, value):
        if v_n == 't':
            return np.log10(value*1000)

        if v_n == 'mdot':
            new_mdot = value
            if self.clump != self.new_clump:

                f_WR = 10**(0.5 * np.log10(self.clump / self.new_clump))  # modify for a new clumping
                new_mdot = value +  np.log10(f_WR)
                print('\nClumping factor changed from {} to {}'.format(self.clump, self.new_clump))
                print('new_mdot = old_mdot + ({}) (f_WR: {} )'
                      .format( "%.2f" % np.log10(f_WR), "%.2f" % f_WR ))
                # print('| i | mdot | n_mdot | f_WR |')
                # for i in range(len(res)):
                #     f_wr = 10**np.float(res2[i]) / 10**np.float(res[i])
                #     print('| {} | {} | {} | {} |'.format("%2.f" % i, "%.2f" % np.float(res[i]) ,"%.2f" % np.float(res2[i]), "%.2f" % f_wr ) )
            return new_mdot

        return value

    def get_num_par_from_table(self, v_n, star_n):

        if v_n not in self.num_v_n:
            raise NameError('v_n: {} is not in set of num. pars: {}'.format(v_n, self.num_v_n))
        if star_n not in self.numers[: , 0]:
            raise NameError('star_n: {} is not in the list of star numbers: {}'.format(star_n, self.numers[:,0]))

        ind_star = np.where(self.numers[: , 0]==star_n)[0][0]
        ind_v_n  = self.num_v_n.index(v_n)
        value = self.numers[ind_star, ind_v_n]


        value = self.modify_value(v_n, value)

        return value

    #---------------------------------------------PUBLIC FUNCTIONS---------------------------------------
    def get_num_par(self, v_n, star_n, yc_val = None, opal_used = None):
        '''

        :param v_n:
        :param star_n:
        :param yc_val: '-1' for Langer1989 | 'None' for use 'm' | 0.9-to-0 for use of l-m-yc relation
        :return:
        '''

        if v_n == 'lm':
            # print('\__ Yc: {}'.format(yc_val))
            l = self.get_num_par_from_table('l', star_n)
            if yc_val >= 0 and yc_val <= 1.:
                l, lm = SP_file_work.yc_x__to__y__sp(yc_val, 'l', 'lm', l, opal_used, 0)
                return lm

            if yc_val == -1 or yc_val == 'langer':
                return np.log10(10 ** l / 10 ** Physics.l_to_m_langer(l))

            if yc_val == None or yc_val == 'hamman':
                m = self.get_num_par_from_table('m', star_n)
                np.log10(10 ** l / m)

            raise ValueError('Yc = {} is not recognised. '
                             'Use: 0-1 for loading tables | -1 or langer for langer |  None or hamman for hamman')

        return self.get_num_par_from_table(v_n, star_n)

    def get_min_max(self, v_n, yc = None, opal_used = None):
        arr = []
        for star_n in self.stars_n:
            arr = np.append(arr, self.get_num_par(v_n, star_n, yc, opal_used))
        return arr.min(), arr.max()

    def get_xyz_from_yz(self,model_n, y_name, z_name, x_1d_arr, y_1d_arr, z_2d_arr, lx1 = None, lx2 = None):

        if y_name == z_name:
            raise NameError('y_name and z_name are the same : {}'.format(z_name))

        star_y = self.get_num_par(y_name, model_n)
        star_z = self.get_num_par(z_name, model_n)

        if star_z == None or star_y == None:
            raise ValueError('star_y:{} or star_z:{} not defined'.format(star_y,star_z))

        xyz = Physics.model_yz_to_xyz(x_1d_arr, y_1d_arr, z_2d_arr,  star_y, star_z, model_n, lx1, lx2)

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
        color = 'C'+se.group(0)
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
            return 'o' # circle
        if cls == 'WN4-w':
            return 's' # square
        if cls == 'WN5-w' or cls == 'WN6-w':
            return '1' # tri down
        if cls == 'WN5-s' or cls == 'WN6-s':
            return 'd' #diamond
        if cls == 'WN7':
            return '^'
        if cls == 'WN8' or cls == 'WN9':
            return 'P' #plus filled

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

class Read_Plot_file:

    # path = './data/'
    # compart = '.plot1'

    def __init__(self, plot_table):
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
        self.rho_c  = plot_table[:i_stop,10]        # dlog10(ro(1))
        self.l_carb = plot_table[:i_stop,11]        # escop
        self.l_nu  = plot_table[:i_stop,12]         # esnyp
        self.mdot_=np.log10(plot_table[:i_stop,13]) # windmd*sec/sun
        self.t_max = plot_table[:i_stop,14]         # vvcmax
        self.rho_at_t_max = plot_table[:i_stop,15]  # gammax
        self.m_at_t_max = plot_table[:i_stop,16]    # wcrit
        self.logg = plot_table[:i_stop,17]          # logg
        self.rwindluca = plot_table[:i_stop,18]     # rwindluca/rsun
        self.r_n_rsun = plot_table[:i_stop, 19]     # r(n)/rsun
        self.teffluca = plot_table[:i_stop, 20]     # teffluca
        self.Tn = plot_table[:i_stop, 21]           # T(N)
        self.tauatR = plot_table[:i_stop, 22]       # tauatR
        self.windform_23 = plot_table[:i_stop, 23]     # Abs(windmd)/(4.d0*PI*rwindluca**2*(v0+(vinf-v0)*(1.d0-R(N)/rwindluca))),
        self.windform_24 = plot_table[:i_stop, 24]     # v0+(vinf-v0)*(1-R(N)/rwindluca)
    # def

    @classmethod
    def from_file(cls, plot_file_name):

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
        return cls((np.vstack((np.zeros(len(table[:, 0])), table.T))).T)

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

    def __init__(self, smdata_table):

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


    def get_xyz_from_yz(self,model_i, condition, y_name, z_name, x_1d_arr, y_1d_arr, z_2d_arr, lx1 = None, lx2 = None):
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
            return self.l_/self.xm_

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
            raise ValueError('\t__Error. Sound speed is not found in data. |get_sp|')

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
        ind = self.ind_from_condition(condition)
        return np.float(self.get_col(v_n)[ind])

    def get_par_table(self, model, y_name = 'l', i = -1):


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
                "%2.0f" % model,
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
                "%2.0f" % model,
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

class Read_SP_data_file:

    def __init__(self, sp_data_file, out_dir = '../data/output/', plot_dir = '../data/plots/'):

        self.files = sp_data_file
        self.out_dir = out_dir
        self.plot_dir = plot_dir


        self.list_of_v_n = ['l', 'm', 't', 'mdot', 'r', 'Yc', 'k']

        self.table = np.loadtxt(sp_data_file)

        print('File: {} has been loaded successfully.'.format(sp_data_file))

        # --- Critical values ---

        self.l_cr = np.float(self.table[0, 0])  # mass array is 0 in the sp file
        self.m_cr = np.float(self.table[0, 1])  # mass array is 1 in the sp file
        self.yc_cr = np.float(self.table[0, 2])  # mass array is 2 in the sp file
        self.lmdot_cr = np.float(self.table[0, 3])  # mass array is 3 in the sp file
        self.r_cr = np.float(self.table[0, 4])  # mass array is 4 in the sp file
        self.t_cr = np.float(self.table[0, 5])  # mass array is 4 in the sp file

        # --- Sonic Point Values ---

        self.l = np.array(self.table[1:, 0])
        self.m = np.array(self.table[1:, 1])
        self.yc = np.array(self.table[1:, 2])
        self.lmdot = np.array(self.table[1:, 3])
        self.rs = np.array(self.table[1:, 4])
        self.ts = np.array(self.table[1:, 5])
        self.k = np.array(self.table[1:, 6])
        self.rho = np.array(self.table[1:, 8])

    def get_crit_value(self, v_n):
        if v_n == 'l':
            return np.float( self.l_cr )

        if v_n =='m' or v_n == 'xm':
            return np.float( self.m_cr )

        if v_n == 't':
            return np.float( self.t_cr )

        if v_n == 'mdot':
            return np.float( self.lmdot_cr)

        if v_n == 'r':
            return np.float(self.r_cr)

        if v_n == 'Yc':
            return np.float(self.yc_cr)

        if v_n == 'lm':
            return np.float(Physics.loglm(self.l_cr, self.m_cr, False) )

        raise NameError('v_n {} is not in the list: {} (for critical values)'.format(v_n, self.list_of_v_n))

    def get_sonic_cols(self, v_n):
        if v_n == 'l':
            return self.l

        if v_n =='m' or v_n == 'xm':
            return self.m

        if v_n == 't':
            return self.ts

        if v_n == 'mdot':
            return self.lmdot

        if v_n == 'r':
            return self.rs

        if v_n == 'Yc':
            return self.yc

        if v_n == 'k':
            return self.k

        if v_n == 'rho':
            return self.rho

        if v_n == 'lm':
            return Physics.loglm(self.l, self.m, True)

        raise NameError('v_n {} is not in the list: {} (for critical values)'.format(v_n, self.list_of_v_n))