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
from Phys_Math_Labels import Plots

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
    def save_table(d2arr, opal_used, bump, name, x_name, y_name, z_name, output_dir = '../data/output/'):

        header = np.zeros(len(d2arr)) # will be first row with limtis and
        # header[0] = x1
        # header[1] = x2
        # header[2] = y1
        # header[3] = y2
        # tbl_name = 't_k_rho'
        # op_and_head = np.vstack((header, d2arr))  # arraching the line with limits to the array

        part = '_' + bump + '_' + opal_used.split('/')[-1]
        full_name = output_dir + name + '_' + part  # dir/t_k_rho_table8.data

        np.savetxt(full_name, d2arr, '%.4f', '  ', '\n',
                   '\nINTERPOLATED OPAL {} TABLE for {} relation'.format(part, name), '',
                   '# {} | {} | {} | {} |'
                   .format('_' + bump + '_' + opal_used, x_name, y_name, z_name))

        # np.savetxt(full_name, d2arr, '%.4f', '  ', '\n',
        #            '\nINTERPOLATED OPAL {} TABLE for {} relation'.format(part, name), '',
        #            '# {} | {} {} {} | {} {} {} | {} | {} | {}'
        #            .format(opal_used, x_name, x1, x2, y_name, y1, y2, z_name, n_int, n_out))

    @staticmethod
    def load_table(name, x_name, y_name, z_name, opal_used, bump, dir = '../data/output/'):
        part =  '_' + bump + '_' + opal_used.split('/')[-1]
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

        if r_table != '_' + bump + '_' + opal_used:
            raise NameError('Read OPAL | {} | not the same is opal_used | {} |'.format(r_table,  '_' + bump + '_' + opal_used))

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

    @staticmethod
    def save_3d_table(d3array, opal_used, bump, name, t_name, x_name, y_name, z_name, output_dir = '../data/output/'):
        i = 0

        part =  '_' + bump + '_' + opal_used.split('/')[-1]
        full_name = output_dir + name + '_' + part  # dir/t_k_rho_table8.data

        # np.savetxt(full_name, d2arr, '%.4f', '  ', '\n',
        #            '\nINTERPOLATED OPAL {} TABLE for {} relation'.format(part, name), '',
        #            '# {} | {} | {} | {} |'
        #            .format(opal_used, x_name, y_name, z_name))

        with open(full_name, 'w') as outfile:
            # I'm writing a header here just for the sake of readability
            # Any line starting with "#" will be ignored by numpy.loadtxt
            # outfile.write('# Array shape: {0}\n'.format(d3array.shape))
            outfile.write(
                '# {} | {} | {} | {} | {} | {} | \n'.format(d3array.shape, t_name, x_name, y_name, z_name,  '_' + bump + '_' + opal_used))

            # Iterating through a ndimensional array produces slices along
            # the last axis. This is equivalent to data[i,:,:] in this case
            for data_slice in d3array:
                # The formatting string indicates that I'm writing out
                # the values in left-justified columns 7 characters in width
                # with 2 decimal places.
                np.savetxt(outfile, data_slice, '%.4f', '  ', '\n',
                           '\n# {}:{} | {} | {} | {} | {}\n'.format(t_name, data_slice[0,0], x_name, y_name, z_name,  '_' + bump + '_' + opal_used), '',
                           '')
                # np.savetxt(outfile, data_slice, fmt='%-7.2f')

                # Writing out a break to indicate different slices...
                # outfile.write('# \n')
                i = i + 1

    @staticmethod
    def load_3d_table(opal_used, bump, name, t_name, x_name, y_name, z_name, output_dir='../data/output/'):
        '''
                            RETURS a 3d table
        :param opal_used:
        :param name:
        :param t_name:
        :param x_name:
        :param y_name:
        :param z_name:
        :param output_dir:
        :return:
        '''


        part = '_' + bump + '_' + opal_used.split('/')[-1]
        full_name = output_dir + name + '_' + part

        with open(full_name, 'r') as f: # reads only first line (to save time)
            first_line = f.readline()


        first_line = first_line.split('# ')[1] # get rid of '# '
        r_shape = first_line.split(' | ')[0]
        r_t_v_n = first_line.split(' | ')[1]
        r_x_v_n = first_line.split(' | ')[2]
        r_y_v_n = first_line.split(' | ')[3]
        r_z_v_n = first_line.split(' | ')[4]
        r_opal  = first_line.split(' | ')[5]

        # --- Checks for opal (metallicity) and x,y,z,t, var_names.

        if r_opal !=  '_' + bump + '_' + opal_used:
            raise NameError('Read OPAL <{}> not the same is opal_used <{}>'.format(r_opal,  '_' + bump + '_' + opal_used))
        if t_name != r_t_v_n:
            raise NameError('Provided x_name: {} not equal to table x_name: {}'.format(t_name, r_t_v_n))
        if x_name != r_x_v_n:
            raise NameError('Provided x_name: {} not equal to table x_name: {}'.format(x_name, r_x_v_n))
        if y_name != r_y_v_n:
            raise NameError('Provided x_name: {} not equal to table x_name: {}'.format(y_name, r_y_v_n))
        if z_name != r_z_v_n:
            raise NameError('Provided x_name: {} not equal to table x_name: {}'.format(z_name, r_z_v_n))

        # --- --- Actual load of a table (as a 2d array) --- --- ---

        d2_table = np.loadtxt(full_name)

        # --- --- Reshaping the 2d arrays into 3d Array --- --- ---

        from ast import literal_eval as make_tuple # to convert str(n, n, n) into a tuple(n, n, n)
        shape = make_tuple(r_shape)

        d3_table = d2_table.reshape(shape)

        print('\t__Table {} is read succesfully. Shape is {}'.format(full_name, d3_table.shape))

        return d3_table
        # print(d2_table.reshape(shape))


        # with open(full_name, 'r') as fullfile:
        #     for line in fullfile:
        #         if line.split(' ')[0] == '#':
        #             check_table_name(line)
        #             n_of_tables = n_of_tables + 1




        # f = open(full_name, 'r').readlines()
        # print(f[2])
        # boxes = f[0].split('|')
        #
        # # print(boxes)
        # # r_table = boxes[0].split()[-1]
        # # r_x_name = boxes[1].split()[0]
        # # x1 = boxes[1].split()[1]
        # # x2 = boxes[1].split()[2]
        # # r_y_name = boxes[2].split()[0]
        # # y1 = boxes[2].split()[1]
        # # y2 = boxes[2].split()[2]
        # # r_z_name = boxes[3].split()[0]
        # # n1 = boxes[4].split()[-1]
        # # n2 = boxes[5].split()[-1]
        #
        # r_table  = boxes[0].split()[-1]
        # r_x_name = boxes[1].split()[-1]
        # r_y_name = boxes[2].split()[-1]
        # r_z_name = boxes[3].split()[-1]
        #
        # f.clear()

    @staticmethod
    def read_genergic_table(file_name, str_col=None):
        '''
        Reads the the file table, returning the list with names and the table
        structure: First Row must be with '#' in the beginning and then, the var names.
        other Rows - table with the same number of elements as the row of var names
        :return:
        :str_col: Name of the volumn with string values
        '''
        table = []
        with open(file_name, 'r') as f:
            for line in f:
                # if '#' not in line.split() and line.strip():  # if line is not empty and does not contain '#'
                table.append(line)

        names = table[0].split()[1:]  # getting rid of '#' which is the first element in a row
        num_colls = len(table) - 1  # as first row is of var names

        if len(names) != len(table[1].split()):
            print('\t___Error. Number of vars in list({}) != number of cols in observ.data file({}) '
                  '|Read_Observables, __init__|'.format(len(names), len(table[1].split())))
        print('\t__Note: Data include following paramters:\n\t | {} |'.format(names))

        table.remove(table[0])  # removing the var_names line from the array. (only actual values left)

        if str_col==None:
            tmp = np.zeros(len(names))
            for row in table:
                tmp = np.vstack((tmp, np.array(row.split(), dtype=np.float)))
            table = np.delete(tmp, 0, 0)

            return names, table

        else:
            # In case there is a column with strings in a table.
            # Remove it from the table. Vstack the rest of the table. Remove it name from the 'names'
            # Return all of it
            tmp = np.zeros(len(names) - 1)
            str_col_val = []
            for row in table:
                raw_row = row.split()
                str_col_val.append(raw_row[names.index(str_col)])
                raw_row.remove(raw_row[names.index(str_col)])
                tmp = np.vstack((tmp, np.array(raw_row, dtype=np.float)))

            table = np.delete(tmp, 0, 0)
            names.remove(str_col)

            return names, table, str_col_val



class Opacity_Bump:


    def __init__(self):
        pass

    @staticmethod
    def t_for_bump(bump_name):
        '''
        Returns t1, t2 for the opacity bump, corresponding to a Fe or HeII increase
        :param bump_name:
        :return:
        '''

        t_fe_bump1  = 5.18
        t_fe_bump2  = 5.4

        t_he2_bump1 = 4.65
        t_he2_bump2 = 5.00

        if bump_name == 'HeII':
            return t_he2_bump1, t_he2_bump2

        if bump_name == 'Fe':
            return t_fe_bump1, t_fe_bump2

        raise NameError('Incorrect bump_name. Opacity bumps availabel: {}'.format(['HeII', 'Fe']))

class Creation:

    def __init__(self, opal_name, bump, n_interp = 1000, load_lim_cases = False,
                 output_dir = '../data/output/', plot_dir = '../data/plots/'):

        self.op_name = opal_name
        self.bump = bump
        self.t1, self.t2 = Opacity_Bump.t_for_bump(bump)
        self.n_inter = n_interp

        self.out_dir = output_dir
        self.plot_dir = plot_dir

        self.opal = OPAL_Interpol(opal_name, n_interp)
        self.tbl_anl = Table_Analyze(opal_name, n_interp, load_lim_cases, output_dir, plot_dir)

    def save_t_rho_k(self, rho1 = None, rho2=None):
        op_cl = OPAL_Interpol(self.op_name, self.n_inter)
        t1, t2, rho1, rho2 = op_cl.check_t_rho_limits(self.t1, self.t2, rho1, rho2)
        op_table = op_cl.interp_opal_table(self.t1, self.t2, rho1, rho2)

        Save_Load_tables.save_table(op_table, self.op_name, self.bump, 't_rho_k','t', 'rho','k', self.out_dir)

    def save_t_k_rho(self, llm1=None, llm2=None, n_out = 1000):

        k1, k2 = Physics.get_k1_k2_from_llm1_llm2(self.t1, self.t2, llm1, llm2) # assuming k = 4 pi c G (L/M)

        global t_k_rho
        t_k_rho = self.tbl_anl.treat_tasks_interp_for_t(self.t1, self.t2, n_out, self.n_inter, k1, k2).T

        Save_Load_tables.save_table(t_k_rho, self.op_name, self.bump, 't_k_rho', 't', 'k', 'rho', self.out_dir)
        print('\t__Note. Table | t_k_rho | has been saved in {}'.format(self.out_dir))
        # self.read_table('t_k_rho', 't', 'k', 'rho', self.op_name)
        # def save_t_llm_vrho(self, llm1=None, llm2=None, n_out = 1000):

    def from_t_k_rho__to__t_lm_rho(self, coeff = 1.0):
        t_k_rho = Save_Load_tables.load_table('t_k_rho', 't', 'k', 'rho', self.op_name, self.bump)

        t = t_k_rho[0, 1:]
        k = t_k_rho[1:, 0]
        rho2d = t_k_rho[1:, 1:]

        lm = Physics.logk_loglm(k, 1, coeff)

        t_lm_rho = Math.invet_to_ascending_xy( Math.combine(t, lm, rho2d) )

        Save_Load_tables.save_table(t_lm_rho, self.op_name, self.bump, 't_{}lm_rho'.format(coeff), 't', '{}lm'.format(coeff), 'rho', self.out_dir)
        print('\t__Note. Table | t_lm_rho | for | {} | has been saved in {}'.format(self.op_name, self.out_dir))

    def save_t_llm_vrho(self, l_or_lm_name):
        '''
        Table required: t_k_rho (otherwise - won't work) [Run save_t_k_rho() function ]
        :param l_or_lm_name:
        :return:
        '''

        # 1 load the t_k_rho
        t_k_rho = Save_Load_tables.load_table('t_k_rho', 't', 'k', 'rho', self.op_name, self.bump)

        k = t_k_rho[0, 1:]
        t = t_k_rho[1:, 0]
        rho2d = t_k_rho[1:, 1:]

        vrho = Physics.get_vrho(t, rho2d.T, 2) # mu = 1.34 by default | rho2d.T is because in OPAL t is Y axis, not X.

        # ----------------------------- SELECT THE Y AXIS -----------------
        if l_or_lm_name == 'l':
            l_lm_arr = Physics.lm_to_l_langer(Physics.logk_loglm(k, True))  # Kappa -> L/M -> L
        else:
            l_lm_arr = Physics.logk_loglm(k, 1)


        l_lm_arr = np.flip(l_lm_arr, 0)  # accounting for if k1 > k2 the l1 < l2 or lm1 < lm2
        vrho     = np.flip(vrho, 0)

        global t_llm_vrho
        t_llm_vrho = Math.combine(t, l_lm_arr, vrho)
        name = 't_'+ l_or_lm_name + '_vrho'

        Save_Load_tables.save_table(t_llm_vrho, self.op_name, self.bump, name, 't', l_or_lm_name, '_vrho', self.out_dir)

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

class SP_file_work:

    def __init__(self, sp_files, yc_precision, opal_used, out_dir, plot_dir):
        self.sp_files = sp_files
        self.out_dir = out_dir
        self.plot_dir = plot_dir
        self.opal_used = opal_used
        self.yc_prec = yc_precision

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

    def separate_sp_by_fname_mass(self):
        '''
        Separation is based on file name an _10sm_ part in that name, where 10 is the value.
        :return: m_arr[:], m_files[:,:] one row for one m
        '''

        # m_files = []
        m_arr = np.array([str(0)])
        fls = []
        for i in range(len(self.sp_files)):
            cut_dirs = self.sp_files[i].split('/')[-1]
            cut_extension = cut_dirs.split('.')[0]
            parts = cut_extension.split('_')

            for j in range(len(parts)):
                if len(parts[j].split('sm')) > 1:
                    m = parts[j].split('sm')[0]
                    # fls.append(self.sp_files[i])
                    if not m in m_arr:
                        m_arr = np.append(m_arr, m)
                        # m_files = np.append(m_files, fls)
                        # fls = []
                    # print(parts[j])

        print('__Initial Masses, read from file name are: {}'.format(m_arr))

        m_files = []
        for j in range(len(m_arr)):
            piece = str(m_arr[j])+'sm' # 10sm ...


            for i in range(len(self.sp_files)):
                cut_dirs = self.sp_files[i].split('/')[-1]
                cut_extension = cut_dirs.split('.')[0]
                parts = cut_extension.split('_')


                if piece in parts:
                    # files.append(self.sp_files[i])
                    m_files = np.append(m_files,  self.spmdl[i])

        m_arr = np.delete(m_arr, 0, 0)
        print(len(m_files), len(m_arr))

        m_files = np.reshape(m_files, (len(m_arr), int(len(m_files)/len(m_arr)) ))



        return m_arr, m_files

    def save_y_yc_z_relation(self, y_v_n, z_v_n, save, plot=False, depth=100, yc_prec=0.1):

        yc, cls = self.separate_sp_by_crit_val('Yc', yc_prec)

        def interp(x, y, x_grid):
            f = interpolate.interp1d(x, y, kind='linear', bounds_error=False)
            # f = interpolate.InterpolatedUnivariateSpline(x, y)
            # f = interpolate.UnivariateSpline(x, y)
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



        # yc_llm_m_pol = Math.extrapolate(yc_llm_m_pol,0,0,0,35,1000,4) # Extrapolation


        table_name = '{}_{}_{}'.format('yc', y_v_n, z_v_n)
        if save == 'int':
            Save_Load_tables.save_table(yc_llm_m_int, self.opal_used, '', table_name, 'yc', y_v_n, z_v_n)
        if save == 'pol':
            Save_Load_tables.save_table(yc_llm_m_pol, self.opal_used, '', table_name, 'yc', y_v_n, z_v_n)

        # Save_Load_tables.save_table(yc_llm_m_pol, opal_used, table_name, 'yc', y_v_n, z_v_n)

        if plot:

            levels = []

            from Phys_Math_Labels import Levels
            levels = Levels.get_levels(z_v_n, self.opal_used)


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

        if plot:
            from Phys_Math_Labels import Plots
            from Phys_Math_Labels import Get_Z
            if save == 'int':
                Plots.plot_color_table(yc_llm_m_int, 'yc', y_v_n, z_v_n, self.opal_used, 'z:{}'.format(Get_Z.z(self.opal_used)))
            if save == 'pol':
                Plots.plot_color_table(yc_llm_m_pol, 'yc', y_v_n, z_v_n, self.opal_used, 'z:{}'.format(Get_Z.z(self.opal_used)))

        # yc_llm_m_pol

    def save_x_y_yc_evol_relation(self, y_v_tmp, z_v_n_tmp, wne=True):
        '''
        Retruns EVOLUTION: 'Yc:[0,1:]  |  y_v_n(ZAMS vales):[1:,0]  |  z_v_n(zams->last Yc)
        :param z_v_n_tmp:
        :param y_v_tmp:
        :return:
        '''

        m_arr, cls = self.separate_sp_by_fname_mass()

        # print(m_files2d.shape)

        x_all = np.zeros(len(cls[0, :]))
        y_all = np.zeros(len(cls[0, :]))
        yc_all= np.zeros(len(cls[0, :]))
        for i in range(len(m_arr)):

            z2d = []
            y = []
            yc = []

            for cl in cls[i,:]:
                z2d = np.append(z2d, cl.get_crit_value(z_v_n_tmp))
                y = np.append(y, cl.get_crit_value(y_v_tmp))
                yc = np.append(yc, np.float("%.1f" % cl.get_crit_value('Yc')) )
                yc, z2d, y = Math.x_y_z_sort(yc, z2d, y)

            x_all = np.vstack((x_all, z2d))
            y_all = np.vstack((y_all, y))
            yc_all = np.vstack((yc_all, yc))

            plt.plot(yc, z2d, '-', color = 'C'+str(Math.get_0_to_max([i],9)[i]))

        plt.grid()
        plt.xlabel(Labels.lbls('Yc'))
        plt.ylabel(Labels.lbls(z_v_n_tmp))
        plt.show()

        x_all = np.delete(x_all, 0, 0)
        y_all = np.delete(y_all, 0, 0)
        yc_all = np.delete(yc_all, 0, 0)


        y_arr = np.array(["%.2f" % val for val in y_all[:,-1]])
        res = Math.combine(yc_all[0,:], y_arr, x_all)

        Save_Load_tables.save_table(res, self.opal_used,'', 'evol_yc_{}_{}'.format(y_v_tmp, z_v_n_tmp),
                                    'evol_yc', y_v_tmp, z_v_n_tmp)

    @staticmethod
    def yc_x__to__y__sp(yc_value, y_v_n, z_v_n, y_inp, opal_used, dimension=0):
        '''
        LOADS the table with given v_ns and extract the row with given Yc and interpolateds the y_value, for x_val given
        :param yc_value:
        :param y_v_n:
        :param z_v_n:
        :param y_inp:
        :param opal_used:
        :param dimension:
        :return:
        '''

        name = '{}_{}_{}'.format('yc', y_v_n, z_v_n)
        yc_x_y = Save_Load_tables.load_table(name, 'yc', y_v_n, z_v_n, opal_used)
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
            if y_inp >= x_arr.min() and y_inp <= x_arr.max():

                y_arr = y2d[:, ind_yc]
                # lm_arr = []
                # for i in range(len(y_arr)):
                #     lm_arr = np.append(lm_arr, [x_arr[i], y_arr[i]])
                #
                # lm_arr_sort = np.sort(lm_arr.view('float64, float64'), order=['f0'], axis=0).view(np.float)
                # lm_arr_shaped = np.reshape(lm_arr_sort, (len(y_arr), 2))

                f = interpolate.InterpolatedUnivariateSpline(x_arr, y_arr)
                y = f(y_inp)
                # print(log_l, y)
                print('Yc: {}, y_row({} - {}), y_res: {}'.format(yc_value, y_arr.min(), y_arr.max(), y))
                return y_inp, y
            else:
                raise ValueError('Given l({}) not in available range of l:({}, {})'.format(y_inp, x_arr.min(), x_arr.max()))

        if dimension == 1:
            x_arr_f = []
            y_arr_f = []
            for i in range(len(y_inp)):
                if y_inp[i] >= x_arr.min() and y_inp[i] <= x_arr.max():
                    f = interpolate.UnivariateSpline(x_arr, y2d[:, ind_yc])
                    y_arr_f = np.append(x_arr_f, f(y_inp[i]))
                    x_arr_f = np.append(x_arr_f, y_inp[i])
                # else:
                #     raise ValueError(
                #         'Given x({}, {}) not in available range of x:({}, {})'.format(x_inp[0], x_inp[-1], x_arr.min(), x_arr.max()))

            return x_arr_f, y_arr_f

    # @staticmethod
    # def l_y_to_m(log_l, yc_value, dimension=0):
    #     '''
    #     RETURNS l, m (if l dimension = 0) and l_arr, m_arr (if dimension = 1)
    #     :param log_l:
    #     :param yc_value:
    #     :param dimension:
    #     :return:
    #     '''
    #     # from FilesWork import Save_Load_tables
    #
    #     yc_l_m = Save_Load_tables.load_table('yc_l_m', 'yc', 'l', 'm', opal_used)
    #     l_arr = yc_l_m[1:, 0]
    #     yc_arr= yc_l_m[0, 1:]
    #     m2d = yc_l_m[1:, 1:]
    #
    #     # yc_value = np.float("%.3f" % yc_value)
    #
    #     if yc_value in yc_arr:
    #         ind_yc = Math.find_nearest_index(yc_arr, yc_value)
    #     else:
    #         raise ValueError('Given yc_arr({}) is not in available yc_arr:({})'.format(yc_value, yc_arr))
    #
    #     if dimension == 0:
    #         if log_l >= l_arr.min() and log_l <= l_arr.max():
    #             m_arr = m2d[:, ind_yc]
    #             # lm_arr = []
    #             # for i in range(len(m_arr)):
    #             #     lm_arr = np.append(lm_arr, [l_arr[i], m_arr[i]])
    #             #
    #             # lm_arr_sort = np.sort(lm_arr.view('float64, float64'), order=['f0'], axis=0).view(np.float)
    #             # lm_arr_shaped = np.reshape(lm_arr_sort, (len(m_arr), 2))
    #
    #             f = interpolate.UnivariateSpline(l_arr, m_arr)
    #             m = f(log_l)
    #             # print(log_l, m)
    #
    #             return log_l, m
    #         else:
    #             raise ValueError('Given l({}) not in available range of l:({}, {})'.format(log_l, l_arr.min, l_arr.max))
    #     if dimension == 1:
    #         m_arr_f = []
    #         l_arr_f = []
    #         for i in range(len(log_l)):
    #             if log_l[i] >= l_arr.min() and log_l[i] <= l_arr.max():
    #                 f = interpolate.UnivariateSpline(l_arr, m2d[ind_yc, :])
    #                 m_arr_f = np.append(m_arr_f, f(log_l[i]))
    #                 l_arr_f = np.append(l_arr_f, log_l[i])
    #             else:
    #                 raise ValueError('Given l({}) not in available range of l:({}, {})'.format(log_l, l_arr.min, l_arr.max))
    #
    #         return l_arr_f, m_arr_f

    # def sp_get_r_lt_table2(self, v_n, l_or_lm, plot=True, ref_t_llm_vrho=np.empty([])):
    #     '''
    #
    #     :param l_or_lm:
    #     :param depth:
    #     :param plot:
    #     :param t_llm_vrho:
    #     :return:
    #     '''
    #     if not ref_t_llm_vrho.any():
    #         print('\t__ No *ref_t_llm_vrho* is provided. Loading {} interp. opacity table.'.format(self.opal_used))
    #         t_k_rho = Save_Load_tables.load_table('t_k_rho', 't', 'k', 'rho', self.opal_used, self.out_dir)
    #         table = Physics.t_kap_rho_to_t_llm_rho(t_k_rho, l_or_lm)
    #     else:
    #         table = ref_t_llm_vrho
    #
    #     t_ref = table[0, 1:]
    #     llm_ref=table[1:, 0]
    #     # rho_ref=table[1:, 1:]
    #
    #
    #     '''=======================================ESTABLISHING=LIMITS================================================'''
    #
    #     t_mins = []
    #     t_maxs = []
    #
    #     for i in range(len(self.sp_files)): # as every sp.file has different number of t - do it one by one
    #
    #         t = self.spmdl[i].get_sonic_cols('t')                     # Sonic
    #         t = np.append(t, self.spmdl[i].get_crit_value('t'))       # critical
    #
    #         t_mins = np.append(t_mins, t.min())
    #         t_maxs = np.append(t_maxs, t.max())
    #
    #     t_min = t_mins.max()
    #     t_max = t_maxs.min()
    #
    #     print('\t__ SP files t limits: ({}, {})'.format(t_min, t_max))
    #     print('\t__REF table t limits: ({}, {})'.format(t_ref.min(), t_ref.max()))
    #
    #     it1 = Math.find_nearest_index(t_ref, t_min)       # Lower t limit index in t_ref
    #     it2 = Math.find_nearest_index(t_ref, t_max)       # Upper t limit index in t_ref
    #     t_grid = t_ref[it1:it2]
    #
    #     print('\t__     Final t limits: ({}, {}) with {} elements'.format(t_ref[it1], t_ref[it2], len(t_grid)))
    #
    #
    #     '''=========================INTERPOLATING=ALONG=T=ROW=TO=HAVE=EQUAL=N=OF=ENTRIES============================='''
    #
    #     llm_r_rows = np.empty(1 + len(t_ref[it1:it2]))
    #
    #     for i in range(len(self.sp_files)):
    #
    #         if l_or_lm == 'l':
    #             llm = self.spmdl[i].get_crit_value('l')
    #         else:
    #             llm = Physics.loglm(self.spmdl[i].get_crit_value('l'), self.spmdl[i].get_crit_value('m'), False)
    #
    #
    #         r = self.spmdl[i].get_sonic_cols(v_n)                     # get sonic
    #         if v_n == 'r': r = np.append(r, self.spmdl[i].get_crit_value('r'))       # get Critical
    #
    #         t = self.spmdl[i].get_sonic_cols('t')                                     # Sonic
    #         if v_n == 'r': t = np.append(t, self.spmdl[i].get_crit_value('t'))        # critical
    #
    #         r_t = []        # Dictionary for sorting
    #         for i in range(len(r)):
    #             r_t = np.append(r_t, [r[i], t[i]])
    #
    #         r_t_sort = np.sort(r_t.view('float64, float64'), order=['f1'], axis=0).view(np.float)
    #         r_t_reshaped = np.reshape(r_t_sort, (len(r), 2)) # insure that the t values are rising along the t_r arr.
    #
    #         r_sort = r_t_reshaped[:,0]
    #         t_sort = r_t_reshaped[:,1]
    #
    #         f = interpolate.InterpolatedUnivariateSpline(t_sort, r_sort)
    #
    #         l_r_row = np.array([llm])
    #         l_r_row = np.append(l_r_row, f(t_grid))
    #         llm_r_rows = np.vstack((llm_r_rows, l_r_row))
    #
    #     llm_r_rows = np.delete(llm_r_rows, 0, 0)
    #
    #     llm_r_rows_sort = llm_r_rows[llm_r_rows[:,0].argsort()] # UNTESTED sorting function
    #
    #     t_llm_r = Math.combine(t_grid, llm_r_rows_sort[:,0], llm_r_rows_sort[:,1:]) # intermediate result
    #
    #     '''======================================INTERPOLATING=EVERY=COLUMN=========================================='''
    #
    #
    #     l      = t_llm_r[1:, 0]
    #     t      = t_llm_r[0, 1:]
    #     r      = t_llm_r[1:, 1:]
    #     il1    = Math.find_nearest_index(llm_ref, l.min())
    #     il2    = Math.find_nearest_index(llm_ref, l.max())
    #
    #     print('\t__ SP files l limits: ({}, {})'.format(l.min(), l.max()))
    #     print('\t__REF table t limits: ({}, {})'.format(llm_ref.min(), llm_ref.max()))
    #
    #     l_grid = llm_ref[il1:il2]
    #
    #     print('\t__     Final l limits: ({}, {}) with {} elements'.format(llm_ref[il1], llm_ref[il2], len(l_grid)))
    #
    #     r_final = np.empty((len(l_grid),len(t)))
    #     for i in range(len(t)):
    #         f = interpolate.InterpolatedUnivariateSpline(l, r[:, i])
    #         r_final[:, i] = f(l_grid)
    #
    #     t_llm_r = Math.combine(t, l_grid, r_final)
    #
    #     if plot:
    #         plt.figure()
    #         # ax = fig.add_subplot(1, 1, 1)
    #         plt.xlim(t_llm_r[0,1:].min(), t_llm_r[0,1:].max())
    #         plt.ylim(t_llm_r[1:,0].min(), t_llm_r[1:,0].max())
    #         plt.ylabel(Labels.lbls(l_or_lm))
    #         plt.xlabel(Labels.lbls('ts'))
    #
    #         levels = []
    #         if v_n == 'k':   levels = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95] # FOR log Kappa
    #         if v_n == 'rho': levels = [-10, -9.5, -9, -8.5, -8, -7.5, -7, -6.5, -6, -5.5,  -5, -4.5, -4]
    #         if v_n == 'r':   levels = [0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7,
    #                                    1.75, 1.8, 1.85, 1.9, 1.95, 2.0, 2.05, 2.10, 2.15, 2.20]
    #
    #         contour_filled = plt.contourf(t_llm_r[0, 1:], t_llm_r[1:, 0], t_llm_r[1:,1:], levels, cmap=plt.get_cmap('RdYlBu_r'))
    #         plt.colorbar(contour_filled, label=Labels.lbls(v_n))
    #         contour = plt.contour(t_llm_r[0, 1:], t_llm_r[1:, 0], t_llm_r[1:,1:], levels, colors='k')
    #         plt.clabel(contour, colors='k', fmt='%2.2f', fontsize=12)
    #         plt.title('SONIC HR DIAGRAM')
    #
    #         # plt.ylabel(l_or_lm)
    #         plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
    #         # plt.savefig(name)
    #         plt.show()
    #
    #     return t_llm_r

    # --- --- --- | ---
    # --- --- --- | ---
    # --- --- --- | ---

    # def save_yc_m_l
    @staticmethod
    def x_y_z(cls, x_v_n, y_v_n, z_v_n, x_grid, y_grid, append_crit=True):
        '''
        cls = set of classes of sp. files with the same Yc.
        :param cls:
        :return:
        '''

        y_zg = np.zeros(len(x_grid) + 1)  # +1 for y-value (l,lm,m,Yc)

        for cl in cls:  # INTERPOLATING EVERY ROW to achive 'depth' number of points
            x = cl.get_sonic_cols(x_v_n)
            y = cl.get_crit_value(y_v_n)  # Y should be unique value for a given Yc (like m, l/lm, or Yc)
            z = cl.get_sonic_cols(z_v_n)

            if append_crit:
                x = np.append(x, cl.get_crit_value(x_v_n))
                z = np.append(z, cl.get_crit_value(z_v_n))
            xi, zi = Math.x_y_z_sort(x, z)

            z_grid = interpolate.InterpolatedUnivariateSpline(xi, zi)(x_grid)
            # z_grid = interpolate.interp1d(xi, zi, kind='cubic', bounds_error=False)(x_grid)

            y_zg = np.vstack((y_zg, np.insert(z_grid, 0, y, 0)))

            # plt.plot(xi, zi, '.', color='red')  # FOR interplation analysis (how good is the fit)
            # plt.plot(x_grid, z_grid, '-', color='red')

        y_zg = np.delete(y_zg, 0, 0)
        y = y_zg[:, 0]
        zi = y_zg[:, 1:]

        z_grid2 = np.zeros(len(y_grid))
        for i in range(len(x_grid)):  # INTERPOLATING EVERY COLUMN to achive 'depth' number of points
            z_grid2 = np.vstack((z_grid2, interpolate.InterpolatedUnivariateSpline(y, zi[:, i])(y_grid)))
            # z_grid2 = np.vstack((z_grid2, interpolate.interp1d(y, zi[:, i], kind='cubic', bounds_error=False)(y_grid)))
        z_grid2 = np.delete(z_grid2, 0, 0)

        x_y_z_final = Math.combine(x_grid, y_grid, z_grid2.T)

        return x_y_z_final

    @staticmethod
    def x_y_z2d(cls, x_v_n, y_v_n, z_v_n, x_grid, y_grid, append_crit = True, interp_method = 'IntUni'):
        '''
        cls = set of classes of sp. files with the same Yc.
        :param cls:
        :return:
        '''

        y_zg = np.zeros(len(x_grid) + 1)  # +1 for y-value (l,lm,m,Yc)

        for cl in cls:  # INTERPOLATING EVERY ROW to achive 'depth' number of points

            def cut_inf_in_y(x_arr, y_arr):
                new_x = []
                new_y = []
                for i in range(len(y_arr)):
                    if y_arr[i] != np.inf and y_arr[i] != -np.inf:
                        new_x = np.append(new_x, x_arr[i])
                        new_y = np.append(new_y, y_arr[i])
                return new_x, new_y

            x = cl.get_sonic_cols(x_v_n)
            y = cl.get_crit_value(y_v_n)  # Y should be unique value for a given Yc (like m, l/lm, or Yc)
            z = cl.get_sonic_cols(z_v_n)

            if append_crit:
                x = np.append(x, cl.get_crit_value(x_v_n))
                z = np.append(z, cl.get_crit_value(z_v_n))

            xi, zi = Math.x_y_z_sort(x, z)
            xi, zi = cut_inf_in_y(xi, zi)

            z_grid = []
            if interp_method == 'IntUni':
                z_grid = interpolate.InterpolatedUnivariateSpline(xi, zi)(x_grid)
            if interp_method == 'Uni':
                z_grid = interpolate.UnivariateSpline(xi, zi)(x_grid)
            if interp_method == '1dCubic':
                z_grid = interpolate.interp1d(xi, zi, kind='cubic')(x_grid)
            if interp_method == '1dLinear':
                z_grid = interpolate.interp1d(xi, zi, kind='linear')(x_grid)
            if len(z_grid) == 0:
                raise NameError('IntMethod is not recognised (or interpolation is failed)')
            # z_grid = interpolate.interp1d(xi, zi, kind='cubic', bounds_error=False)(x_grid)

            y_zg = np.vstack((y_zg, np.insert(z_grid, 0, y, 0)))

            plt.plot(xi, zi, '.', color='red')                # FOR interplation analysis (how good is the fit)
            plt.plot(x_grid, z_grid, '-', color='red')
            # plt.show()
        y_zg = np.delete(y_zg, 0, 0)
        y = y_zg[:, 0]
        zi = y_zg[:, 1:]

        f = interpolate.interp2d(x_grid, y, zi, kind='linear', bounds_error=False)
        z_grid2 = f(x_grid, y_grid)

        # z_grid2 = np.zeros(len(y_grid))
        # for i in range(len(x_grid)):  # INTERPOLATING EVERY COLUMN to achive 'depth' number of points
        #     # z_grid2 = np.vstack((z_grid2, interpolate.InterpolatedUnivariateSpline(y, zi[:, i])(y_grid)))
        #     z_grid2 = np.vstack((z_grid2, interpolate.interp1d(y, zi[:, i], kind='cubic', bounds_error=False)(y_grid)))
        # z_grid2 = np.delete(z_grid2, 0, 0)

        x_y_z_final = Math.combine(x_grid, y_grid, z_grid2)

        return x_y_z_final

    @staticmethod
    def x_y_limits(cls, x_v_n, y_v_n, min_or_max = 'min', append_crit = True):
        x_mins = []
        y_mins = []
        x_maxs = []
        y_maxs = []
        for cl in cls:
            x = cl.get_sonic_cols(x_v_n)
            y = cl.get_sonic_cols(y_v_n)
            if append_crit:
                x = np.append(x, cl.get_crit_value(x_v_n))
                y = np.append(y, cl.get_crit_value(y_v_n))
            x_mins = np.append(x_mins, x.min())
            y_mins = np.append(y_mins, y.min())
            x_maxs = np.append(x_maxs, x.max())
            y_maxs = np.append(y_maxs, y.max())
        if min_or_max == 'min':
            return x_mins.max(), x_maxs.min(), y_mins.min(), y_maxs.max()
        if min_or_max == 'max':
            return x_mins.min(), x_maxs.max(), y_mins.min(), y_maxs.max()
        else:
            raise NameError('min_or_max can be only: [{}, or {}] given: {}'.format('min', 'max', min_or_max))

    @staticmethod
    def y_limits(cls, y_v_n, min_or_max='min', append_crit=True):
        y_mins = []
        y_maxs = []
        for cl in cls:
            y = cl.get_sonic_cols(y_v_n)
            if append_crit:
                y = np.append(y, cl.get_crit_value(y_v_n))
            y_mins = np.append(y_mins, y.min())
            y_maxs = np.append(y_maxs, y.max())
        if min_or_max == 'min':
            return  y_mins.min(), y_maxs.max()
        if min_or_max == 'max':
            return  y_mins.min(), y_maxs.max()
        else:
            raise NameError('min_or_max can be only: [{}, or {}] given: {}'.format('min', 'max', min_or_max))

    def plot_x_y_z_for_yc(self, x_v_n, y_v_n, z_v_n, yc_val, depth, min_or_max = 'min', append_crit = True, plot=True):

        yc, cls = self.separate_sp_by_crit_val('Yc', self.yc_prec)
        if yc_val in yc:
            yc_index = Math.find_nearest_index(yc, yc_val)
        else:
            raise ValueError('Value Yc:{} is not is available set {}'.format(yc_val, yc))


        x1, x2, y1, y2 = self.x_y_limits(cls[yc_index], x_v_n, y_v_n, min_or_max, append_crit)
        x_grid = np.mgrid[x1.min():x2.max():depth * 1j]
        y_grid = np.mgrid[y1.min():y2.max():depth * 1j]


        x_y_z = self.x_y_z2d(cls[yc_index],x_v_n, y_v_n, z_v_n, x_grid, y_grid, append_crit)

        if plot:
            Plots.plot_color_table(x_y_z, x_v_n, y_v_n, z_v_n, self.opal_used, 'Yc:{}'.format(yc_val))

        return x_y_z

    def save_x_y_z(self, x_v_n, y_v_n, z_v_n, depth, min_or_max = 'min', append_crit = True, interp_method = 'IntUni'):
        yc, cls = self.separate_sp_by_crit_val('Yc', self.yc_prec)

        x_y_z3d = []

        for yc_val in yc:

            if yc_val in yc:
                yc_index = Math.find_nearest_index(yc, yc_val)
            else:
                raise ValueError('Value Yc:{} is not is available set {}'.format(yc_val, yc))

            x1, x2, y1, y2 = self.x_y_limits(cls[yc_index], x_v_n, y_v_n, min_or_max, append_crit)
            x_grid = np.mgrid[x1.min():x2.max():depth * 1j]
            y_grid = np.mgrid[y1.min():y2.max():depth * 1j]

            x_y_z = self.x_y_z2d(cls[yc_index], x_v_n, y_v_n, z_v_n, x_grid, y_grid, append_crit, interp_method)
            x_y_z[0, 0] = yc_val
            x_y_z3d = np.append(x_y_z3d, x_y_z)

        res = np.reshape(x_y_z3d, (len(yc), depth+1, depth+1))

        Save_Load_tables.save_3d_table(res,self.opal_used,'yc_{}_{}_{}'.format(x_v_n, y_v_n, z_v_n), 'yc', x_v_n, y_v_n, z_v_n,
                                       self.out_dir)

    # --- --- ---


    def plot_x_y_z(self, x_v_n, y_v_n, z_v_n, yc_arr, depth, min_or_max = 'min', append_crit = True):

        yc, cls = self.separate_sp_by_crit_val('Yc', self.yc_prec)

        yc_arr = np.sort(yc_arr, axis=0)

        ind_arr = []
        for yc_val in yc_arr:
            ind_arr = np.append(ind_arr, np.int(Physics.ind_of_yc(yc, yc_val)))

        yc_n = np.int(len(yc_arr))

        fig = plt.figure()
        fig.subplots_adjust(hspace=0.2, wspace=0.3)

        for i in range(1, len(yc_arr)+1):

            if yc_n % 2 == 0: ax = fig.add_subplot(2, yc_n/2, i)
            else:             ax = fig.add_subplot(1, yc_n, i)

            yc_val = yc_arr[i-1]
            print(yc_val)

            x_y_z = self.plot_x_y_z_for_yc(x_v_n, y_v_n, z_v_n, yc_val, depth, min_or_max, append_crit, False)
            Plots.plot_color_background(ax, x_y_z, x_v_n, y_v_n, z_v_n, self.opal_used, 'Yc:{}'.format(yc_val))

        # plt.legend(bbox_to_anchor=(1, 0), loc='lower right', ncol=1)
        plot_name = self.plot_dir + '{}_{}_{}.pdf'.format(x_v_n, y_v_n, z_v_n)
        plt.savefig(plot_name)
        plt.show()


    @staticmethod
    def lm_l(lm, yc_lm_l, yc_val):
        '''
        Give it lm (value of array) and yc_lm_l relation from the file, --> l (value or array)
        :param lm:
        :param yc_lm_l:
        :param yc_val:
        :return:
        '''

        def prec2(var):
            var = "%.2f" % var
            return np.float(var)

        yc = yc_lm_l[0, 1:]
        lm_ = yc_lm_l[1:, 0]
        l2d = yc_lm_l[1:, 1:]

        if not yc_val in yc:
            raise ValueError('Yc:{} not in Yc array from <yc_lm_l> relation ({})'.format(yc_val, yc))

        if prec2(lm.min()) < prec2(lm_.min()):
            raise ValueError('lm.min({}) < lm_.min({})'.format(lm.min(), lm_.min()))
        if prec2(lm.max()) > prec2(lm_.max()):
            raise ValueError('lm.max({}) > lm_.max({})'.format(lm.max(), lm_.max()))

        i_yc = Math.find_nearest_index(yc, yc_val)
        l    = interpolate.InterpolatedUnivariateSpline(lm_, l2d[:, i_yc])(lm)

        return l

    def get_t_llm_rho_crop_for_yc(self, cls, t_lm_rho, yc_val, y_v_n, min_max = 'min', opal_used = None, append_crit = True):

        # t_lm_rho = Save_Load_tables.load_table('t_lm_rho', 't', 'lm', 'rho', opal_used)
        yc_lm_l = Save_Load_tables.load_table('yc_lm_l', 'yc', 'lm', 'l', opal_used, '')
        t_lm_rho_com, yc_lm_l_com = Math.common_y(t_lm_rho, yc_lm_l)

        if y_v_n == 'l':
            l = self.lm_l(t_lm_rho_com[1:, 0], yc_lm_l, yc_val)

            t_l_rho = Math.combine(t_lm_rho_com[0, 1:], l, t_lm_rho_com[1:, 1:])

            x1, x2, y1, y2 = self.x_y_limits(cls, 't', y_v_n, min_max, append_crit)
            t_l_rho_crop = Math.crop_2d_table2(t_l_rho, x1, x2, y1, y2)  # taking absolut limits

            return t_l_rho_crop

        if y_v_n == 'lm':
            x1, x2, y1, y2 = self.x_y_limits(cls, 't', y_v_n, min_max, append_crit)
            t_lm_rho_crop = Math.crop_2d_table2(t_lm_rho_com, x1, x2, y1, y2)

            return t_lm_rho_crop

    def get_t_llm_mdot_for_yc(self, cls, t_lm_rho, yc_val, y_v_n, min_max = 'min', opal_used = None, append_crit = True):

        t_llm_rho = self.get_t_llm_rho_crop_for_yc(cls, t_lm_rho, yc_val, y_v_n, min_max, opal_used, append_crit)

        t_llm_r = self.x_y_z(cls, 't', y_v_n, 'r', t_llm_rho[0, 1:], t_llm_rho[1:, 0], append_crit)

        rho2d= t_llm_rho[1:, 1:]
        r2d  = t_llm_r[1:, 1:]
        t    = t_llm_r[0, 1:]
        llm  = t_llm_r[1:, 0]

        vro = Physics.get_vrho(t, rho2d, 2, np.array([1.34]))
        mdot= Physics.vrho_mdot(vro, r2d,'tl')
        return Math.combine(t, llm, mdot, yc_val)
    def plot_t_llm_mdot_for_yc(self, yc_val, l_or_lm, coeff, bump, min_max = 'min', append_crit = True):

        yc, cls = self.separate_sp_by_crit_val('Yc', self.yc_prec)
        if yc_val in yc:
            yc_index = Math.find_nearest_index(yc, yc_val)
        else:
            raise ValueError('Value Yc:{} is not is available set {}'.format(yc_val, yc))

        t_lm_rho = Save_Load_tables.load_table('t_{}lm_rho'.format(coeff), 't', '{}lm'.format(coeff), 'rho', self.opal_used, bump)
        t_llm_mdot = self.get_t_llm_mdot_for_yc(cls[yc_index], t_lm_rho, yc[yc_index], l_or_lm, min_max, self.opal_used, append_crit)

        from Phys_Math_Labels import Get_Z
        z = Get_Z.z(self.opal_used)
        Plots.plot_color_table(t_llm_mdot, 't', l_or_lm, 'mdot', self.opal_used, 'z:{}({}) Yc:{} K:{}'.format(z, bump, yc_val, coeff))

    def get_t_llm_rho_crop_for_yc_const_r(self, cls, t_lm_rho, yc_val, y_v_n, min_max = 'min', opal_used = None, append_crit = True):

        # t_lm_rho = Save_Load_tables.load_table('t_lm_rho', 't', 'lm', 'rho', opal_used)
        yc_lm_l = Save_Load_tables.load_table('yc_lm_l', 'yc', 'lm', 'l', opal_used, '')
        t_lm_rho_com, yc_lm_l_com = Math.common_y(t_lm_rho, yc_lm_l)

        if y_v_n == 'l':
            l = self.lm_l(t_lm_rho_com[1:, 0], yc_lm_l, yc_val)

            t_l_rho = Math.combine(t_lm_rho_com[0, 1:], l, t_lm_rho_com[1:, 1:])

            y1, y2 = self.y_limits(cls, y_v_n, min_max, append_crit)
            t_l_rho_crop = Math.crop_2d_table2(t_l_rho, None, None, y1, y2)  # taking absolut limits

            return t_l_rho_crop

        if y_v_n == 'lm':
            y1, y2 = self.y_limits(cls, y_v_n, min_max, append_crit)
            t_lm_rho_crop = Math.crop_2d_table2(t_lm_rho_com, None, None, y1, y2)

            return t_lm_rho_crop
    def get_t_llm_mdot_for_yc_const_r(self, cls, t_lm_rho, yc_val, rs, y_v_n, min_max = 'min', opal_used = None, append_crit = True):

        t_llm_rho = self.get_t_llm_rho_crop_for_yc_const_r(cls, t_lm_rho, yc_val, y_v_n, min_max, opal_used, append_crit)

        # t_llm_r = self.x_y_z(cls, 't', y_v_n, 'r', t_llm_rho[0, 1:], t_llm_rho[1:, 0], append_crit)

        rho2d= t_llm_rho[1:, 1:]
        t = t_llm_rho[0, 1:]
        llm =t_llm_rho[1:, 0]
        # r2d  = t_llm_r[1:, 1:]
        # t    = t_llm_r[0, 1:]
        # llm  = t_llm_r[1:, 0]

        vro = Physics.get_vrho(t, rho2d, 2, np.array([1.34]))
        mdot= Physics.vrho_mdot(vro, rs, '')
        return Math.combine(t, llm, mdot, yc_val)
    def plot_t_llm_mdot_for_yc_const_r(self, yc_val, rs, l_or_lm, coeff, bump, min_max = 'min', append_crit = True, plot=True):

        yc, cls = self.separate_sp_by_crit_val('Yc', self.yc_prec)
        if yc_val in yc:
            yc_index = Math.find_nearest_index(yc, yc_val)
        else:
            raise ValueError('Value Yc:{} is not is available set {}'.format(yc_val, yc))

        t_lm_rho = Save_Load_tables.load_table('t_{}lm_rho'.format(coeff), 't', '{}lm'.format(coeff), 'rho', self.opal_used, bump)
        t_llm_mdot = self.get_t_llm_mdot_for_yc_const_r(cls[yc_index], t_lm_rho, yc[yc_index], rs, l_or_lm, min_max, self.opal_used, append_crit)

        if plot:
            Plots.plot_color_table(t_llm_mdot, 't', l_or_lm, 'mdot',self.opal_used, 'Yc:{} R:{}'.format(yc_val, rs))

        # Save_Load_tables.save_3d_table(t_llm_mdot, self.opal_used, 'yc_t_{}_mdot_r_{}'.format(l_or_lm, rs), 'yc', 't',
        #                                l_or_lm, 'mdot_r_{}'.format(rs), self.out_dir)


    def save_min_max_lm(self, v_n):
        yc, cls = self.separate_sp_by_crit_val('Yc', self.yc_prec)

        ys_zams = cls[-1][0].get_crit_value('ys')

        fig = plt.figure()
        ax = fig.add_subplot(111)

        yc_lm_min_max = np.zeros(3)
        for i in range(len(yc)):
            lm_tmp = []
            ys_tmp = []
            for c in cls[i]:
                lm_tmp = np.append(lm_tmp, c.get_crit_value(v_n))
                ys_tmp = np.append(ys_tmp, c.get_crit_value('ys'))
            lm_tmp, ys_tmp = Math.x_y_z_sort(lm_tmp, ys_tmp)
            tmp = np.zeros(len(lm_tmp))
            tmp.fill(yc[i])
            ax.plot(tmp, lm_tmp, '.', color='gray')

            lm_tmp2 = []
            for j in range(len(cls[i])):
                if ys_tmp[j] < ys_zams: break
                else: lm_tmp2 = np.append(lm_tmp2, lm_tmp[j])

            if len(lm_tmp2) == 0:
                raise ValueError('No values with zams Ys  found for Yc: {}'.format(yc[i]))
            lm_max = lm_tmp2[-1]
            lm_min = lm_tmp2[0]
            # print(i)
            yc_lm_min_max = np.vstack((yc_lm_min_max, [yc[i], lm_min, lm_max]))

        # yc_lm_min_max = np.reshape(yc_lm_min_max, (len(yc), 3))

        # yc_lm_min_max = np.insert(yc_lm_min_max,0,  np.zeros(len(yc_lm_min_max[0, 1:])), 1)
        yc_lm_min_max = yc_lm_min_max.T

        ax.plot(yc_lm_min_max[0,1:], yc_lm_min_max[1,1:], '-', color='blue', label='WNE region')
        ax.plot(yc_lm_min_max[0, 1:], yc_lm_min_max[2, 1:], '-', color='blue')
        ax.fill_between(yc_lm_min_max[0,1:], yc_lm_min_max[1,1:], yc_lm_min_max[2, 1:], color='blue',
                         alpha='.3')
        ax.set_xlabel(Labels.lbls('yc'))
        ax.set_ylabel(Labels.lbls(v_n))
        ax.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
        ax.grid()
        plt.show()

        Save_Load_tables.save_table(yc_lm_min_max, self.opal_used, '',
                                    'yc_nan_{}lim'.format(v_n), 'yc', 'nan', '{}lim'.format(v_n))

            # lm_arr = np.append(lm_arr, cl.get_crit_value('lm'))
            # ys_arr = np.append(ys_arr, cl.get_crit_value('ys'))
        # print('a')
        # yc, lm_arr, ys_arr = Math.x_y_z_sort(yc, lm_arr, ys_arr)

    @staticmethod
    def get_square(table, depth):
        '''
        Performs row by row and column bu column interpolation of a table to get a square matrix with required depth
        :param table:
        :param depth:
        :return:
        '''
        x = table[0, 1:]
        y = table[1:, 0]
        z = table[1:,1:]

        z1 = np.zeros(depth)
        x_grid = np.mgrid[x.min():x.max():depth*1j]
        for i in range(len(y)):
            tmp = interpolate.InterpolatedUnivariateSpline(x, z[i, :])(x_grid)
            z1 = np.vstack((z1, tmp))

        z1 = np.delete(z1, 0, 0)

        z2 = np.zeros((depth))
        y_grid = np.mgrid[y.min():y.max():depth*1j]
        for i in range(depth):
            tmp = interpolate.InterpolatedUnivariateSpline(y, z1[:, i])(y_grid)
            z2 = np.vstack((z2, tmp))


        z2 = np.delete(z2, 0, 0)
        return Math.combine(x_grid, y_grid, z2.T)

    def save_t_llm_mdot(self, l_or_lm, coeff, bump, depth_square = 500, min_max = 'min', append_crit = True):

        yc, cls = self.separate_sp_by_crit_val('Yc', self.yc_prec)
        t_lm_rho = Save_Load_tables.load_table('t_{}lm_rho'.format(coeff), 't', '{}lm'.format(coeff), 'rho', self.opal_used, bump)


        # yc_t_llm_mdot = np.array([[[k*j*i for k in np.arange(0, 5, 1)] for j in np.arange(0, 5, 1)] for i in np.arange(0, 5, 1)])
        yc_t_llm_mdot = []
        for i in range(len(yc)):
            print('\n<--- --- --- ({}) Yc:{} K:{} --- --- --->\n'.format(bump, yc[i], coeff))
            tmp1 = self.get_t_llm_mdot_for_yc(cls[i], t_lm_rho, yc[i], l_or_lm, min_max, self.opal_used, append_crit)
            tmp2 = self.get_square(tmp1, depth_square)
            tmp2[0, 0] = yc[i] # appending the yc value as a [0, 0] in the array.
            # print(tmp2.shape)
            yc_t_llm_mdot = np.append(yc_t_llm_mdot, tmp2)

        yc_t_llm_mdot = np.reshape(yc_t_llm_mdot, (len(yc), depth_square+1, depth_square+1))
        print('\t__Saving the ({}) yc_t_{}{}_mdot table for. Shape:{}'.format(bump, coeff, l_or_lm, yc_t_llm_mdot.shape))



        Save_Load_tables.save_3d_table(yc_t_llm_mdot, self.opal_used, bump, 'yc_t_{}{}_mdot'.format(coeff, l_or_lm),
                                       'yc', 't', str(coeff) + l_or_lm, 'mdot', self.out_dir)

        # print(Save_Load_tables.load_3d_table(self.opal_used, 'yc_t_l_mdot', 'yc', 't', 'l', 'mdot', self.out_dir))

    def save_t_llm_mdot_const_r(self, l_or_lm, coeff, bump, rs, depth_square = 500, min_max = 'min', append_crit = True):

        yc, cls = self.separate_sp_by_crit_val('Yc', self.yc_prec)
        t_lm_rho = Save_Load_tables.load_table('t_{}lm_rho'.format(coeff), 't', '{}lm'.format(coeff), 'rho', self.opal_used, bump)

        # yc_t_llm_mdot_rs_const = np.array([[[k*j*i for k in np.arange(0, 5, 1)] for j in np.arange(0, 5, 1)] for i in np.arange(0, 5, 1)])
        yc_t_llm_mdot_rs_const = []
        for i in range(len(yc)):
            print('\n<--- --- --- ({}) Yc:{} K:{} Rs:{} (const) --- --- --->\n'.format(bump, yc[i], coeff, rs))
            tmp1 = self.get_t_llm_mdot_for_yc_const_r(cls[i], t_lm_rho, yc[i], rs, l_or_lm, min_max, self.opal_used, append_crit)
            tmp2 = self.get_square(tmp1, depth_square)
            tmp2[0, 0] = yc[i]  # appending the yc value as a [0, 0] in the array.
            # print(tmp2.shape)
            yc_t_llm_mdot_rs_const = np.append(yc_t_llm_mdot_rs_const, tmp2)

        yc_t_llm_mdot_rs_const = np.reshape(yc_t_llm_mdot_rs_const, (len(yc), depth_square + 1, depth_square + 1))
        print('\t__Saving the ({}) yc_t_{}{}_mdot table for Rs = {} (const). Shape:{}'.format(bump, coeff, l_or_lm, rs,
                                                                                              yc_t_llm_mdot_rs_const.shape))

        Save_Load_tables.save_3d_table(yc_t_llm_mdot_rs_const, self.opal_used, bump, 'yc_t_{}{}_mdot_rs_{}'.format(coeff, l_or_lm, rs), 'yc', 't',
                                       str(coeff) + l_or_lm, 'mdot_rs_{}'.format(rs), self.out_dir)


    # -------------------------------------------------
    def test(self, yc_val = 1.0):
        yc, cls = self.separate_sp_by_crit_val('Yc', self.yc_prec)
        if yc_val in yc:
            yc_index = Math.find_nearest_index(yc, yc_val)
        else:
            raise ValueError('Value Yc:{} is not is available set {}'.format(yc_val, yc))

        cl = cls[yc_index]

        arr = []

        for c in cl:
            mdot = c.get_crit_value('mdot')
            r = c.get_crit_value('r')
            m = c.get_crit_value('m')
            l = c.get_crit_value('l')


            arr = np.append(arr, [mdot, r, m , l])

            # print(cl)

        arr_sort = np.sort(arr.view('f8, f8, f8, f8'), order=['f3'], axis=0).view(np.float)
        arr_shaped = np.reshape(arr_sort, (np.int(len(arr_sort)/4), 4))

        var = (1/arr_shaped[:,1])*(arr_shaped[:,2]/arr_shaped[:,3])*arr_shaped[:,0]

        plt.plot(arr_shaped[:,0], var, '.', color='black')
        plt.grid()
        plt.xlabel('mdot')
        plt.ylabel('MY')
        plt.show()

        t_lm_rho = Save_Load_tables.load_table('t_1.0lm_rho','t','1.0lm','rho',self.opal_used,'Fe')
        t = t_lm_rho[0, 1:]
        lm = t_lm_rho[1:, 0]
        k = Physics.loglm_logk(lm, True)
        rho = t_lm_rho[1:, 1:]
        vrho = Physics.get_vrho(t, rho, 2)

        t_k_vrho = Math.combine(t,k,vrho)
        mdot = Physics.vrho_mdot(vrho,1.0,'')
        mins = Math.get_maxs_in_every_row(t,k,vrho,5000)

        fig = plt.figure(figsize=plt.figaspect(0.8))
        ax = fig.add_subplot(111)  # , projection='3d'

        Plots.plot_color_background(ax, t_k_vrho,'t','k','vrho',self.opal_used)
        ax.plot(mins[0,:],mins[1,:])
        plt.show()

        var2 = []
        for i in range(len(mins[0,:])):
            var2 = np.append(var2, mins[1,i]*mins[2,i])




        print(var)
        print('haha: {}'.format(Constants.light_v*Constants.grav_const/np.sqrt(Constants.k_b/Constants.m_H)))

    # def save_y_yc_z_relation_sp(self, x_v_n, y_v_n, z_v_n, save, plot=False, depth=100):
    #     append_crit = True
    #     if not y_v_n in ['m', 'l', 'lm', 'Yc']:
    #         raise NameError('y_v_n must be one of [{}] , give:{}'.format(['m', 'l', 'lm', 'Yc'], y_v_n))
    #
    #     def x_y_z_sort(x_arr, y_arr, z_arr=None, sort_by_012=0):
    #         '''
    #         RETURNS x_arr, y_arr, (z_arr) sorted as a matrix by a row, given 'sort_by_012'
    #         :param x_arr:
    #         :param y_arr:
    #         :param z_arr:
    #         :param sort_by_012:
    #         :return:
    #         '''
    #
    #         if z_arr == None and sort_by_012 < 2:
    #             if len(x_arr) != len(y_arr):
    #                 raise ValueError('len(x)[{}]!= len(y)[{}]'.format(len(x_arr), len(y_arr)))
    #
    #             x_y_arr = []
    #             for i in range(len(x_arr)):
    #                 x_y_arr = np.append(x_y_arr, [x_arr[i], y_arr[i]])
    #
    #             x_y_sort = np.sort(x_y_arr.view('float64, float64'), order=['f{}'.format(sort_by_012)], axis=0).view(np.float)
    #             x_y_arr_shaped = np.reshape(x_y_sort, (int(len(x_y_sort) / 2), 2))
    #             return x_y_arr_shaped[:,0], x_y_arr_shaped[:,1]
    #
    #         if z_arr != None:
    #             if len(x_arr) != len(y_arr) or len(x_arr)!=len(z_arr):
    #                 raise ValueError('len(x)[{}]!= len(y)[{}]!=len(z_arr)[{}]'.format(len(x_arr), len(y_arr), len(z_arr)))
    #
    #             x_y_z_arr = []
    #             for i in range(len(x_arr)):
    #                 x_y_z_arr = np.append(x_y_z_arr, [x_arr[i], y_arr[i], z_arr[i]])
    #
    #             x_y_z_sort = np.sort(x_y_z_arr.view('float64, float64, float64'), order=['f{}'.format(sort_by_012)], axis=0).view(
    #                 np.float)
    #             x_y_z_arr_shaped = np.reshape(x_y_z_sort, (int(len(x_y_z_sort) / 3), 3))
    #             return x_y_z_arr_shaped[:, 0], x_y_z_arr_shaped[:, 1], x_y_z_arr_shaped[:, 2]
    #
    #     def x_y_limits(cls, min_or_max):
    #         x_mins = []
    #         y_mins = []
    #         x_maxs = []
    #         y_maxs = []
    #         for cl in cls:
    #             x = cl.get_sonic_cols(x_v_n)
    #             y = cl.get_sonic_cols(y_v_n)
    #             if append_crit:
    #                 x = np.append(x, cl.get_crit_value(x_v_n))
    #                 y = np.append(y, cl.get_crit_value(y_v_n))
    #             x_mins = np.append(x_mins, x.min())
    #             y_mins = np.append(y_mins, y.min())
    #             x_maxs = np.append(x_maxs, x.max())
    #             y_maxs = np.append(y_maxs, y.max())
    #         if min_or_max == 'min':
    #             return x_mins.max(), x_maxs.min(), y_mins.min(), y_maxs.max()
    #         if min_or_max == 'max':
    #             return x_mins.min(), x_maxs.max(), y_mins.min(), y_maxs.max()
    #         else:
    #             raise NameError('min_or_max can be only: [{}, or {}] given: {}'.format('min', 'max', min_or_max))
    #
    #     def common_y(arr1, arr2):
    #
    #         y1 = arr1[1:, 0]
    #         y2 = arr2[1:, 0]
    #
    #         y_min = np.array([y1.min(), y2.min()]).max()
    #         y_max = np.array([y1.max(), y2.max()]).min()
    #
    #         arr1_cropped = Math.crop_2d_table(arr1, None, None, y_min, y_max)
    #         arr2_cropped = Math.crop_2d_table(arr2, None, None, y_min, y_max)
    #
    #         return arr1_cropped, arr2_cropped
    #
    #     def lm_l(lm, yc_lm_l, yc_val):
    #
    #         yc = yc_lm_l[0,  1:]
    #         lm_= yc_lm_l[1:,  0]
    #         l2d= yc_lm_l[1:, 1:]
    #
    #         if not yc_val in yc:
    #             raise ValueError('Yc:{} not in Yc array from <yc_lm_l> relation ({})'.format(yc_val, yc))
    #
    #         if lm.min() < lm_.min():
    #             raise ValueError('lm.min({}) < lm_.min({})'.format(lm.min(), lm_.min()))
    #         if lm.max() > lm_.max():
    #             raise ValueError('lm.max({}) > lm_.max({})'.format(lm.max(), lm_.max()))
    #
    #         i_yc = Math.find_nearest_index(yc, yc_val)
    #         l = interpolate.InterpolatedUnivariateSpline(lm_, l2d[:, i_yc])(lm)
    #
    #         return l
    #
    #     def set_t_lm_grids_opal(yc_val, cls, min_max, opal_used = None):
    #
    #         t_lm_rho = Save_Load_tables.load_table('t_lm_rho', 't', 'lm', 'rho', opal_used)
    #         yc_lm_l  = Save_Load_tables.load_table('yc_lm_l', 'yc', 'lm', 'l',   opal_used)
    #         t_lm_rho_com, yc_lm_l_com = common_y(t_lm_rho, yc_lm_l)
    #
    #         if y_v_n == 'l':
    #
    #             l = lm_l(t_lm_rho_com[1:, 0], yc_lm_l, yc_val)
    #             t_l_rho = Math.combine(t_lm_rho_com[0, 1:], l, t_lm_rho_com[1:, 1:])
    #
    #             x1, x2, y1, y2 = x_y_limits(cls, min_max)
    #             t_l_rho_crop = Math.crop_2d_table2(t_l_rho, x1, x2, y1, y2) # taking absolut limits
    #
    #             return t_l_rho_crop
    #
    #         if y_v_n == 'lm':
    #             x1, x2, y1, y2 = x_y_limits(cls, min_max)
    #             t_lm_rho_crop = Math.crop_2d_table(t_lm_rho_com, x1, x2, y1, y2)
    #
    #             return t_lm_rho_crop
    #
    #
    #     def set_xgrid_ygrid(cls, min_max):
    #         x1, x2, y1, y2 = x_y_limits(cls, min_max)
    #         x_grid = np.mgrid[x1.min():x2.max():depth * 1j]
    #         y_grid = np.mgrid[y1.min():y2.max():depth * 1j]
    #         return x_grid, y_grid
    #
    #
    #
    #     yc, cls = self.separate_sp_by_crit_val('Yc', self.yc_prec)
    #
    #     t_llm_rho = set_t_lm_grids_opal(1, cls[0], 'min', self.opal_used)
    #     Plots.plot_color_table(t_llm_rho, x_v_n, y_v_n, 'rho')
    #     x_grid = t_llm_rho[0, 1:]
    #     y_grid = t_llm_rho[1:, 0]
    #
    #     def x_y_z(cls, x_grid, y_grid):
    #         '''
    #         cls = set of classes of sp. files with the same Yc.
    #         :param cls:
    #         :return:
    #         '''
    #
    #         y_zg = np.zeros(len(x_grid)+1)     # +1 for y-value (l,lm,m,Yc)
    #
    #         for cl in cls:                    # INTERPOLATING EVERY ROW to achive 'depth' number of points
    #             x =  cl.get_sonic_cols(x_v_n)
    #             y =  cl.get_crit_value(y_v_n) # Y should be unique value for a given Yc (like m, l/lm, or Yc)
    #             z =  cl.get_sonic_cols(z_v_n)
    #
    #             if append_crit:
    #                 x = np.append(x, cl.get_crit_value(x_v_n))
    #                 z = np.append(z, cl.get_crit_value(z_v_n))
    #             xi, zi = x_y_z_sort(x, z)
    #
    #             z_grid = interpolate.InterpolatedUnivariateSpline(xi, zi)(x_grid)
    #             y_zg = np.vstack((y_zg, np.insert(z_grid, 0, y, 0)))
    #
    #             plt.plot(xi, zi, '.', color='red')
    #             plt.plot(x_grid, z_grid, '-', color='red')
    #
    #         y_zg = np.delete(y_zg, 0, 0)
    #         y = y_zg[:,0]
    #         zi = y_zg[:, 1:]
    #
    #         z_grid2 = np.zeros(len(y_grid))
    #         for i in range(len(x_grid)):   # INTERPOLATING EVERY COLUMN to achive 'depth' number of points
    #             z_grid2 = np.vstack((z_grid2, interpolate.InterpolatedUnivariateSpline(y, zi[:,i])(y_grid) ))
    #         z_grid2 = np.delete(z_grid2, 0, 0)
    #
    #         x_y_z_final = Math.combine(x_grid, y_grid, z_grid2.T)
    #
    #         from Phys_Math_Labels import Plots
    #         Plots.plot_color_table(x_y_z_final, x_v_n, y_v_n, z_v_n)
    #
    #         plt.show()
    #         print('a')
    #         return x_y_z_final
    #
    #     x_y_z = x_y_z(cls[0], x_grid, y_grid)
    #
    #
    #
    #     print('a')
    #     # for i in range(len(yc)):
    #     #     x_y_z(cls[i])
    #
    #
    #
    #
    #             # x_y_z = []
    #             # for j in range(len(y)):
    #             #     x_y_z = np.append(x_y_z, [x[j], y[j], z[j]])
    #             #
    #             # x_y_z_sort = np.sort(x_y_z.view('float64, float64, float64'), order=['f0'], axis=0).view(np.float)
    #             # x_y_z_shaped = np.reshape(x_y_z_sort, (int(len(x_y_z_sort) / 3), 3))
    #             #
    #             # x_grid = np.mgrid[x_y_z_shaped[0, 0]:x_y_z_shaped[-1, 0]:depth*1j]
    #             # f = interpolate.InterpolatedUnivariateSpline(x_y_z_shaped[:,0],x_y_z_shaped[:,2]) # follows the data
    #             # z_grid = f(x_grid)
    #             #
    #             # y_xi_zi = []
    #             # for j in range(len(y)):
    #             #     y_xi_zi = np.append(y_xi_zi, [y[j], ])
    #             #
    #             #
    #             # plt.plot(x_y_z_shaped[:,0],x_y_z_shaped[:,1], '.', color='red')
    #             # # plt.plot(x_grid, z_grid, '-', color='red')
    #             # plt.show()
    #             # print('B')
    #
    #         # yc_x_y_z = np.append(yc_x_y_z, x_y_z_shaped)
    #
    #
    #     # yc_x_y_z = np.reshape(yc_x_y_z, (len(yc), int(len(x_y_z_sort) / 3), 3))
    #
    #
    #     # print('a')
    #
    #
    #
    #
    #
    #
    #
    #     # y_ = []
    #     # for i in range(len(self.sp_files)):
    #     #     y_ = np.append(y_, self.spmdl[i].get_crit_value(y_v_n))
    #     # y_grid = np.mgrid[y_.min():y_.max():depth*1j]
    #     #
    #     # z2d_pol = np.zeros(len(y_grid))
    #     # z2d_int = np.zeros(len(y_grid))
    #     #
    #     #
    #     # fig = plt.figure(figsize=plt.figaspect(1.0))
    #     #
    #     # ax1 = fig.add_subplot(221)
    #     # ax1.grid()
    #     # ax1.set_ylabel(Labels.lbls(z_v_n))
    #     # ax1.set_xlabel(Labels.lbls(y_v_n))
    #     # ax1.set_title('INTERPOLATION')
    #     #
    #     # ax2 = fig.add_subplot(222)
    #     # ax2.grid()
    #     # ax2.set_ylabel(Labels.lbls(y_v_n))
    #     # ax2.set_xlabel(Labels.lbls(z_v_n))
    #     # ax2.set_title('EXTRAPOLATION')
    #     #
    #     # for i in range(len(yc)):
    #     #     y_z = []
    #     #     for cl in cls[i]:
    #     #         y_z = np.append(y_z, [cl.get_crit_value(y_v_n), cl.get_crit_value(z_v_n)])
    #     #     y_z_sort = np.sort(y_z.view('float64, float64'), order=['f0'], axis=0).view(np.float)
    #     #     y_z_shaped = np.reshape(y_z_sort, (int(len(y_z_sort) / 2), 2))
    #     #
    #     #     '''----------------------------POLYNOMIAL EXTRAPOLATION------------------------------------'''
    #     #     print('\n\t Yc = {}'.format(yc[i]))
    #     #     y_pol, z_pol = Math.fit_plynomial(y_z_shaped[:, 0], y_z_shaped[:, 1], 3, depth, y_grid)
    #     #     z2d_pol = np.vstack((z2d_pol, z_pol))
    #     #     color = 'C' + str(int(yc[i] * 10)-1)
    #     #     ax2.plot(y_pol, z_pol, '--', color=color)
    #     #     ax2.plot(y_z_shaped[:, 0], y_z_shaped[:, 1], '.', color=color, label='yc:{}'.format("%.2f" % yc[i]))
    #     #
    #     #     '''------------------------------INTERPOLATION ONLY---------------------------------------'''
    #     #     y_int, z_int = interp(y_z_shaped[:, 0], y_z_shaped[:, 1], y_grid)
    #     #     z2d_int = np.vstack((z2d_int, z_int))
    #     #     ax1.plot(y_int, z_int, '--', color=color)
    #     #     ax1.plot(y_z_shaped[:, 0], y_z_shaped[:, 1], '.', color=color, label='yc:{}'.format("%.2f" % yc[i]))
    #     #
    #     # ax1.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
    #     # ax2.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
    #     #
    #     # z2d_int = np.delete(z2d_int, 0, 0)
    #     # z2d_pol = np.delete(z2d_pol, 0, 0)
    #     #
    #     # yc_llm_m_pol = Math.combine(yc, y_grid, z2d_pol.T)  # changing the x/y
    #     # yc_llm_m_int = Math.combine(yc, y_grid, z2d_int.T)  # changing the x/y
    #     #
    #     # table_name = '{}_{}_{}'.format('yc', y_v_n, z_v_n)
    #     # if save == 'int':
    #     #     Save_Load_tables.save_table(yc_llm_m_int, opal_used, table_name, 'yc', y_v_n, z_v_n)
    #     # if save == 'pol':
    #     #     Save_Load_tables.save_table(yc_llm_m_pol, opal_used, table_name, 'yc', y_v_n, z_v_n)
    #     #
    #     # # Save_Load_tables.save_table(yc_llm_m_pol, opal_used, table_name, 'yc', y_v_n, z_v_n)
    #     #
    #     # if plot:
    #     #
    #     #     levels = []
    #     #
    #     #     if z_v_n == 'r':
    #     #         levels = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2., 2.1, 2.2, 2.3, 2.4, 2.5]
    #     #     if z_v_n == 'm':
    #     #         levels = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    #     #     if z_v_n == 'mdot':
    #     #         levels = [-6.0, -5.9, -5.8, -5.7, -5.6, -5.5, -5.4, -5.3, -5.2, -5.1, -5., -4.9, -4.8, -4.7, -4.6, -4.5]
    #     #     if z_v_n == 'l':
    #     #         levels = [5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4]
    #     #     if z_v_n == 'lm':
    #     #         levels = [4.0, 4.05, 4.1, 4.15, 4.2, 4.25, 4.3, 4.35,  4.4, 4.45,
    #     #                   4.5, 4.55,  4.6, 4.65,  4.7, 4.75, 4.8, 4.85,  4.9, 4.95, 5.0]
    #     #     if z_v_n == 't':
    #     #         levels = [5.15, 5.16,5.17,5.18,5.19,5.20,5.21,5.22,5.23,5.24,5.25,5.26,5.27,5.28,5.29,5.30]
    #     #
    #     #
    #     #     ax = fig.add_subplot(223)
    #     #
    #     #     # ax = fig.add_subplot(1, 1, 1)
    #     #     ax.set_xlim(yc_llm_m_int[0,1:].min(), yc_llm_m_int[0,1:].max())
    #     #     ax.set_ylim(yc_llm_m_int[1:,0].min(), yc_llm_m_int[1:,0].max())
    #     #     ax.set_ylabel(Labels.lbls(y_v_n))
    #     #     ax.set_xlabel(Labels.lbls('Yc'))
    #     #
    #     #     contour_filled = plt.contourf(yc_llm_m_int[0, 1:], yc_llm_m_int[1:, 0], yc_llm_m_int[1:,1:], levels, cmap=plt.get_cmap('RdYlBu_r'))
    #     #     # plt.colorbar(contour_filled, label=Labels.lbls('m'))
    #     #     contour = plt.contour(yc_llm_m_int[0, 1:], yc_llm_m_int[1:, 0], yc_llm_m_int[1:,1:], levels, colors='k')
    #     #
    #     #     clb = plt.colorbar(contour_filled)
    #     #     clb.ax.set_title(Labels.lbls(z_v_n))
    #     #
    #     #     plt.clabel(contour, colors='k', fmt='%2.2f', fontsize=12)
    #     #     #ax.set_title('MASS-LUMINOSITY RELATION')
    #     #
    #     #     # plt.ylabel(l_or_lm)
    #     #     # ax.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
    #     #     # plt.savefig(name)
    #     #
    #     #
    #     #
    #     #
    #     #     ax = fig.add_subplot(224)
    #     #
    #     #     # ax = fig.add_subplot(1, 1, 1)
    #     #     ax.set_xlim(yc_llm_m_pol[0, 1:].min(), yc_llm_m_pol[0, 1:].max())
    #     #     ax.set_ylim(yc_llm_m_pol[1:, 0].min(), yc_llm_m_pol[1:, 0].max())
    #     #     ax.set_ylabel(Labels.lbls(y_v_n))
    #     #     ax.set_xlabel(Labels.lbls('Yc'))
    #     #
    #     #
    #     #     contour_filled = plt.contourf(yc_llm_m_pol[0, 1:], yc_llm_m_pol[1:, 0], yc_llm_m_pol[1:, 1:], levels,
    #     #                                   cmap=plt.get_cmap('RdYlBu_r'))
    #     #     # plt.colorbar(contour_filled, label=Labels.lbls('m'))
    #     #     contour = plt.contour(yc_llm_m_pol[0, 1:], yc_llm_m_pol[1:, 0], yc_llm_m_pol[1:, 1:], levels, colors='k')
    #     #
    #     #     clb = plt.colorbar(contour_filled)
    #     #     clb.ax.set_title(Labels.lbls(z_v_n))
    #     #
    #     #     plt.clabel(contour, colors='k', fmt='%2.2f', fontsize=12)
    #     #     #ax.set_title('MASS-LUMINOSITY RELATION')
    #     #
    #     #     # plt.ylabel(l_or_lm)
    #     #     # ax.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
    #     #
    #     #
    #     #     plt.show()
    #
    #     # yc_llm_m_pol



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

    def __init__(self, observ_name, opal_used, load_relations = True, clump_used=4, clump_req=4):
        '''

        :param observ_name:
        :param yc: -1 (None, use m), 0.1 to 0.99 (My relation for diff yc), 10 (use Langer et al. 1989)
        :param clump_used:
        :param clump_req:
        '''

        self.set_use_gaia = True
        self.set_use_atm_file = True
        self.set_atm_file = None

        # --------------------------------------------------------------------------------------------------------------
        self.file_name = observ_name

        self.clump = clump_used
        self.new_clump = clump_req

        self.opal_used = opal_used



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
        self.num_v_n = ['N', 't', 'lRt', 'm', 'l', 'mdot', 'v_inf', 'eta']  # list of used v_n's; can be extended upon need ! 'N' should be first
        self.cls = 'class'                           # special variable

        self.str_v_n = []

        # print(self.get_star_class(75.))
        self.set_num_str_tables()
        # print(self.get_table_num_par('l', 110))
        # print(self.get_star_class(110))

        # --- --- GAIA LUMINOCITIES --- ---
        if self.set_use_gaia and opal_used.split('/')[-1] == 'table8.data':
            self.gaia_names, self.gaia_table = self.read_genergic_table(observ_name.split('.data')[0] + '_gaia_lum.data')

        # --- --- Potsdam Atmospheres --- ---
        if self.set_use_atm_file:
            self.atm = Read_Atmosphere_File(self.set_atm_file, self.opal_used)
            self.atm_table = self.atm.get_table('t_*', 'rt', 't_eff')

        # if opal_used.split('/')[-1] == 'table8.data':  # if Galactic Mode is set:
        #     gaia_lum_fname = observ_name.split('.data')[0] + '_gaia_lum.data'
        #     self.table_lum = []
        #
        #
        #
        #
        #
        #     with open(gaia_lum_fname, 'r') as f:
        #         for line in f:
        #             if '#' not in line.split() and line.strip():  # if line is not empty and does not contain '#'
        #                 # self.table_lum.append(line)
        #                 self.table_lum = np.append(self.table_lum, np.array(line.split()))
        #
        #     raw_names = ['WR', 'type', 'l', 'll', 'ul']
        #     self.num_stars2 = len(self.table_lum[:,0])
        #     if self.num_stars!=self.num_stars2:
        #         raise IOError('Number of stars in OBS file and luminocity from Gaia file not equal ({} != {})'
        #                       .format(self.num_stars, self.num_stars2  ))

            # gaia_star_n = np.array(self.table_lum[:,0])

        # --- ---
        # self.get_gaia_lum(110)

        if load_relations:
            self.yc_l_lm = Save_Load_tables.load_table('yc_l_lm', 'yc', 'l', 'lm', self.opal_used, '')
            self.yc_nan_lmlim = Save_Load_tables.load_table('yc_nan_lmlim', 'yc', 'nan', 'lmlim', self.opal_used, '')
    # ------------------------------------------------
    @staticmethod
    def read_genergic_table(file_name):
        '''
        Reads the the file table, returning the list with names and the table
        structure: First Row must be with '#' in the beginning and then, the var names.
        other Rows - table with the same number of elements as the row of var names
        :return:
        '''
        table = []
        with open(file_name, 'r') as f:
            for line in f:
                # if '#' not in line.split() and line.strip():  # if line is not empty and does not contain '#'
                table.append(line)

        names = table[0].split()[1:]  # getting rid of '#' which is the first element in a row
        num_colls = len(table) - 1  # as first row is of var names

        if len(names) != len(table[1].split()):
            print('\t___Error. Number of vars in list({}) != number of cols in observ.data file({}) '
                  '|Read_Observables, __init__|'.format(len(names), len(table[1].split())))
        print('\t__Note: Data include following paramters:\n\t | {} |'.format(names))

        table.remove(table[0])  # removing the var_names line from the array. (only actual values left)

        tmp = np.zeros(len(names))
        for row in table:
            tmp_ = np.array(row.split(), dtype=np.float)
            if len(tmp_)!=len(names):
                print('row:', row, '\n', 'tmp:',tmp_,len(tmp_))
                raise NameError(len(tmp_),'\n',len(names))
            tmp = np.vstack((tmp, tmp_))
        table = np.delete(tmp, 0, 0)

        return names, table
    # ------------------------------------------------

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

    def get_num_par_from_table(self, v_n, star_n, use_gaia=False):

        if v_n not in self.num_v_n:
            raise NameError('v_n: {} is not in set of num. pars: {}'.format(v_n, self.num_v_n))
        if star_n not in self.numers[: , 0]:
            raise NameError('star_n: {} is not in the list of star numbers: {}'.format(star_n, self.numers[:,0]))

        # -- GAIA MISSION for GALAXY (table8) ---
        if v_n == 'l' and self.opal_used.split('/')[-1] == 'table8.data' and use_gaia:
            l = self.get_gaia_lum(star_n)[0]
            return l

        ind_star = np.where(self.numers[: , 0]==star_n)[0][0]
        ind_v_n  = self.num_v_n.index(v_n)
        value = self.numers[ind_star, ind_v_n]


        value = self.modify_value(v_n, value)

        return value

    # ---
    def get_gaia_lum(self, star_n):
        if star_n not in self.gaia_table[:,0]:
            raise ValueError('Star: {} is not in the GAIA list: {}'.format(star_n, self.gaia_table[:,0]))
        ind = Math.find_nearest_index(self.gaia_table[:,0], star_n)
        # print(ind)
        return np.array([self.gaia_table[ind,2], self.gaia_table[ind,3], self.gaia_table[ind,4]])
    #---------------------------------------------PUBLIC FUNCTIONS---------------------------------------
    def l_lm_for_errs(self, l, star_n, yc_val):
        '''
        Changes the L of the star to the maximmum available for plotting.
        :param v_n:
        :param star_n:
        :param yc_val:
        :return:
        '''

        if l < self.yc_l_lm[1:, 0].min(): l = self.yc_l_lm[1:, 0].min()
        if l > self.yc_l_lm[1:, 0].max(): l = self.yc_l_lm[1:, 0].max()
        # print('\__ Yc: {}'.format(yc_val))

        if yc_val >= 0 and yc_val <= 1.:
            l, lm = Math.get_z_for_yc_and_y(yc_val, self.yc_l_lm, l, 0)
            yc_ind = Math.find_nearest_index(self.yc_l_lm, yc_val)
            lm_lim = self.yc_nan_lmlim[1:, yc_ind]
            if yc_val != 1.0 and (lm < lm_lim[0] or lm > lm_lim[1]):
                raise ValueError('For Yc:{}, star:{} L/M limits are: {}, {}, given lm {}'
                                 .format(yc_val, star_n, "%.2f" % lm_lim[0], "%.2f" % lm_lim[1], "%.2f" % lm))

            # l, lm = SP_file_work.yc_x__to__y__sp(yc_val, 'l', 'lm', l, self.opal_used, 0)
            return lm

        if yc_val == -1 or yc_val == 'langer':
            return np.log10(10 ** l / 10 ** Physics.l_to_m_langer(l))

        if yc_val == None or yc_val == 'hamman':
            m = self.get_num_par_from_table('m', star_n)
            return np.log10(10 ** l / m)
    def get_star_l_obs_err(self,star_n, yc_assumed, use_gaia=False):

        l_err1 = l_err2 = None
        l = self.get_num_par('l', star_n, yc_assumed, use_gaia)

        if self.opal_used.split('/')[-1] == 'table8.data':

            l_err1 = 0.2 # --- OLD correction from Hamman2006
            l_err2 = 0.2
            if use_gaia:
                l_err1 = self.get_gaia_lum(star_n)[0]-self.get_gaia_lum(star_n)[1]
                l_err2 = self.get_gaia_lum(star_n)[2]-self.get_gaia_lum(star_n)[0]

        if self.opal_used.split('/')[-1] == 'table_x.data':

            l_err1 = 0.1
            l_err2 = 0.1

        if l_err1 == None or l_err2 == None: raise ValueError('Error for l (star:{}) not found'.format(star_n))
        return l-l_err1, l+l_err2

    def get_star_lm_obs_err(self,star_n, yc_assumed, use_gaia=False):
        # l_err1 = l_err2 = None
        # l = None

        l_err1, l_err2 = self.get_star_l_obs_err(star_n, yc_assumed, use_gaia)
        lm1 = self.l_lm_for_errs(l_err1, star_n, yc_assumed)
        lm2 = self.l_lm_for_errs(l_err2, star_n, yc_assumed)

        return lm1, lm2

        # if self.opal_used.split('/')[-1] == 'table8.data':
        #
        #     l_err1, l_err2 = self.get_star_l_obs_err(star_n, yc_assumed, use_gaia)
        #
        #     # l_err1 = 0.2 # --- OLD correction from Hamman2006
        #     # l_err2 = 0.2
        #     # l_err1 = self.get_gaia_lum(star_n)[0]-self.get_gaia_lum(star_n)[1]
        #     # l_err2 = self.get_gaia_lum(star_n)[2]-self.get_gaia_lum(star_n)[0]
        # if self.opal_used.split('/')[-1] == 'table_x.data':
        #
        #
        #
        #     l_err1 = 0.1
        #     l_err2 = 0.1
        #
        #
        # l = self.get_num_par('l', star_n, yc_assumed)
        # lm1 = self.l_lm_for_errs(l - l_err1, star_n, yc_assumed)
        # lm2 = self.l_lm_for_errs(l + l_err2, star_n, yc_assumed)
        #
        # return lm1, lm2


    def get_star_lm_obs_err_old(self,star_n, yc_assumed, use_gaia=False):
        l_err1 = l_err2 = None
        l = None
        if self.opal_used.split('/')[-1] == 'table8.data':

            l_err1 = 0.2 # --- OLD correction from Hamman2006
            l_err2 = 0.2
            # l_err1 = self.get_gaia_lum(star_n)[0]-self.get_gaia_lum(star_n)[1]
            # l_err2 = self.get_gaia_lum(star_n)[2]-self.get_gaia_lum(star_n)[0]
        if self.opal_used.split('/')[-1] == 'table_x.data':

            l_err1 = 0.1
            l_err2 = 0.1


        l = self.get_num_par('l', star_n, yc_assumed)
        lm1 = self.l_lm_for_errs(l - l_err1, star_n, yc_assumed)
        lm2 = self.l_lm_for_errs(l + l_err2, star_n, yc_assumed)

        return lm1, lm2

    def lm_mdot_to_ts_errs(self, lm, mdot, star_n, t_llm_mdot, yc_assumed, lim_t1, lim_t2):
        mdot_arr = t_llm_mdot[1:, 1:]
        i = Math.find_nearest_index(t_llm_mdot[1:, 0], lm)

        if t_llm_mdot[i, 1:].min() > mdot:
            print('\t__Star: {} Using the min. mdot in the row for the error. '.format(star_n))
            mdot = mdot_arr[i,:].min()

        xyz = Physics.model_yz_to_xyz(t_llm_mdot[0, 1:], t_llm_mdot[1:, 0], t_llm_mdot[1:, 1:], lm, mdot, star_n, lim_t1, lim_t2)

        if xyz.any():
            if len(xyz[0, :]) > 1:
                raise ValueError('Multiple coordinates for star: {} | Yc: {}'.format(star_n, yc_assumed))
            else:
                ts = xyz[0, 0] # xyz[0, :] ALL X coordinates
                return ts

        else:

            print('\t__ Error. No (Error) solutions found star: {}, mdot: {}, mdot array:({}, {})'
                             .format(star_n, mdot, mdot_arr[i, :].min(), mdot_arr[i, :].max()))
            return lim_t2
            raise ValueError('No Error solutions found star: {}, mdot: {}, mdot array:({}, {})'
                             .format(star_n, mdot, mdot_arr[i, :].min(), mdot_arr[i, :].max()))
    def get_star_mdot_obs_err(self, star_n, yc_assumed):
        mdot_err = None
        if self.opal_used.split('/')[-1] == 'table8.data':
            mdot_err = 0.15
        if self.opal_used.split('/')[-1] == 'table_x.data':
            mdot_err = 0.15
        if mdot_err == None: raise NameError('Opal_table is not recognised. '
                                             'Expected: table8.data or table_x.data, Given: {}'.format(self.opal_used))
        mdot = self.get_num_par('mdot', star_n, yc_assumed)
        return mdot - mdot_err, mdot + mdot_err

    def get_star_ts_obs_err(self, star_n, t_llm_mdot, yc_assumed, lim_t1, lim_t2):

        mdot_err = None
        if self.opal_used.split('/')[-1] == 'table8.data':
            mdot_err = 0.15
        if self.opal_used.split('/')[-1] == 'table_x.data':
            mdot_err = 0.15
        if mdot_err == None: raise NameError('Opal_table is not recognised. '
                                             'Expected: table8.data or table_x.data, Given: {}'.format(self.opal_used))

        lm1, lm2 = self.get_star_lm_obs_err(star_n, yc_assumed)
        # lm = self.get_num_par('lm', star_n, yc_assumed)
        # mdot = self.get_num_par('mdot', star_n, yc_assumed)

        mdot1, mdot2 = self.get_star_mdot_obs_err(star_n, yc_assumed)

        ts1_b = self.lm_mdot_to_ts_errs(lm1, mdot1, star_n, t_llm_mdot, yc_assumed, lim_t1, lim_t2)
        ts2_b = self.lm_mdot_to_ts_errs(lm1, mdot2, star_n, t_llm_mdot, yc_assumed, lim_t1, lim_t2)
        ts1_t = self.lm_mdot_to_ts_errs(lm2, mdot1, star_n, t_llm_mdot, yc_assumed, lim_t1, lim_t2)
        ts2_t = self.lm_mdot_to_ts_errs(lm2, mdot2, star_n, t_llm_mdot, yc_assumed, lim_t1, lim_t2)

        return np.float(ts1_b), np.float(ts2_b), np.float(ts1_t), np.float(ts2_t)
    # --------------------------------------------

    def get_num_par(self, v_n, star_n, yc_val = None, use_gaia = False):
        '''

        :param v_n:
        :param star_n:
        :param yc_val: '-1' for Langer1989 | 'None' for use 'm' | 0.9-to-0 for use of l-m-yc relation
        :return:
        '''

        if v_n == 't_eff':
            return self.get_t_eff_from_atmosphere(star_n)

        if v_n == 'lm':
            # print('\__ Yc: {}'.format(yc_val))
            l = self.get_num_par_from_table('l', star_n, use_gaia)
            if yc_val >= 0 and yc_val <= 1.:
                l, lm = Math.get_z_for_yc_and_y(yc_val, self.yc_l_lm, l, 0)
                yc_ind = Math.find_nearest_index(self.yc_nan_lmlim[0,1:], yc_val)
                lm_lim = self.yc_nan_lmlim[1:, yc_ind]
                if lm < lm_lim[0] or lm > lm_lim[1]:
                    raise ValueError('For Yc:{}, star:{} L/M limits are: {}, {}, given lm {}'
                                     .format(yc_val, star_n , "%.2f" % lm_lim[0], "%.2f" % lm_lim[1], "%.2f" % lm))

                # l, lm = SP_file_work.yc_x__to__y__sp(yc_val, 'l', 'lm', l, self.opal_used, 0)
                return lm

            if yc_val == -1 or yc_val == 'langer':
                return np.log10(10 ** l / 10 ** Physics.l_to_m_langer(l))

            if yc_val == None or yc_val == 'hamman':
                m = self.get_num_par_from_table('m', star_n)
                np.log10(10 ** l / m)

            raise ValueError('Yc = {} is not recognised. '
                             'Use: 0-1 for loading tables | -1 or langer for langer |  None or hamman for hamman')

        return self.get_num_par_from_table(v_n, star_n, use_gaia)

    def get_min_max(self, v_n, yc = None):
        arr = []
        for star_n in self.stars_n:
            arr = np.append(arr, self.get_num_par(v_n, star_n, yc))
        return arr.min(), arr.max()

    def get_xyz_from_yz(self, yc_val, model_n, y_name, z_name, x_1d_arr, y_1d_arr, z_2d_arr, lx1 = None, lx2 = None):

        if y_name == z_name:
            raise NameError('y_name and z_name are the same : {}'.format(z_name))

        star_y = self.get_num_par(y_name, model_n, yc_val)
        star_z = self.get_num_par(z_name, model_n, yc_val)

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

    # def get_star_lm_err(self, star_n, yc_assumed, yc1, yc2):
    #
    #     def check_if_ys(lm, yc_arr = np.array([1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])):
    #         # ys_zams = yc_lm_ys[1:, 1:].max()
    #
    #         tmp, ys_zams = SP_file_work.yc_x__to__y__sp(1., 'lm', 'ys', lm, self.opal_used)
    #
    #         yc_avl = []
    #         for i in range(len(yc_arr)):
    #             tmp, ys = SP_file_work.yc_x__to__y__sp(yc_arr[i],'lm','ys',lm, self.opal_used)
    #             if ys <=ys_zams:
    #                 yc_avl = np.append(yc_avl, yc_arr[i])
    #             else:
    #                 pass
    #
    #         if len(yc_avl) == 0:
    #             raise ValueError('Array yc_avl empty, - not yc are found for which ys = ys_zams ({})'.format(ys_zams))
    #
    #
    #
    #         return yc_avl.max(), yc_avl.min()
    #
    #     lm  = 4.29471667968977 # np.float(self.get_num_par('lm', star_n, yc_assumed))
    #
    #     yc1, yc2 = check_if_ys(lm)
    #     lm1 = self.get_num_par('lm', star_n, yc1)  # Getting lm at different Yc (assiming the same l, but
    #     lm2 = self.get_num_par('lm', star_n, yc2)
    #
    #     print('\t__STAR: {} | {} : {} (+{} -{})'.format(star_n, 'L/M', lm, "%.2f" % np.abs(lm - lm1),
    #                                                     "%.2f" % np.abs(lm - lm2)))
    #
    #     return np.abs(lm - lm1), np.abs(lm - lm2)
    # ------------------------------------------- EVOLUTION ERRORS
    def get_min_yc_for_lm(self, star_n):
        '''
        Get min lm for which the surface comp. hasn't changed. (checling the (yc, lm) point in yc_nan_lmlim file
        :param star_n:
        :return:
        '''
        yc = self.yc_nan_lmlim[0, 1:]
        i = 0
        yc1 = yc[i]  # assyming the lowest yc (highly likely sorface change, and therefore change in yc1)

        found = True
        while found:

            l = self.get_num_par('l', star_n, yc1)
            l, lm_tmp = Math.get_z_for_yc_and_y(yc[i], self.yc_l_lm, l, 0) # getting lm without chacking if lm is within limits, at it is for cycle.

            lm_lim = self.yc_nan_lmlim[1:, 1:]
            lm_lim1 = lm_lim[0, i]
            lm_lim2 = lm_lim[1, i]
            if lm_tmp < lm_lim1 or lm_tmp > lm_lim2:
                i = i + 1
            else:
                return yc[i]

    def get_star_lm_err(self, star_n, yc_assumed, use_gaia = False):
        '''
        Assuming the same l, but different L -> L/M relations for different Yc, retruns (lm-lm1), (lm-lm2)
        :param star_n:
        :param yc_assumed:
        :param yc1: ZAMS ( or high Yc)
        :param yc2: END  ( or Low  Yc)
        :return:
        '''
        yc = self.yc_nan_lmlim[0, 1:]
        yc2 = yc[-1]                # Max yc is always 1.0, - ZAMS (no surface change expected)
        yc1 = self.get_min_yc_for_lm(star_n)

        lm1 = self.get_num_par('lm', star_n, yc1, use_gaia) # Getting lm at different Yc (assiming the same l, but
        lm2 = self.get_num_par('lm', star_n, yc2, use_gaia)
        lm  = self.get_num_par('lm', star_n, yc_assumed, use_gaia)

        print('\t__STAR: {} | Evol. Err. L/M for Yc ({} - {}) L/M: ({}, {})'.format(star_n, yc1, yc2, lm1, lm2))

        if lm1 == lm2 :
             raise  ValueError('Star: {} lm1==lm2'.format(star_n))

        return lm1, lm2

    def get_star_ts_err(self, star_n, t_llm_mdot, yc_assumed, lim_t1, lim_t2):

        # if yc1 <= yc2:
        #     raise ValueError('yc1({}) <= yc2({}) (should be > )'.format(yc1, yc2))
        yc = self.yc_nan_lmlim[0, 1:]
        # lm1 = self.get_num_par('lm', star_n, yc1)  # Getting lm at different Yc (assiming the same l, but
        # lm2 = self.get_num_par('lm', star_n, yc2)
        # lm  = self.get_num_par('lm', star_n, yc_assumed)
        #
        # mdot = self.get_num_par('mdot', star_n, yc_assumed)

        ts = None

        xyz = self.get_xyz_from_yz(yc_assumed, star_n, 'lm', 'mdot',
                                      t_llm_mdot[0, 1:], t_llm_mdot[1:, 0], t_llm_mdot[1:, 1:], lim_t1, lim_t2)
        if xyz.any():
            if len(xyz[0, :]) > 1:
                raise ValueError('Multiple coordinates for star: {} | Yc: {}'.format(star_n, yc_assumed))
            else:
                ts = xyz[0, 0] # xyz[0, :] ALL X coordinates

        ts1 = 0
        yc1 = self.get_min_yc_for_lm(star_n)
        xyz1 = self.get_xyz_from_yz(yc1, star_n, 'lm', 'mdot',
                                   t_llm_mdot[0, 1:], t_llm_mdot[1:, 0], t_llm_mdot[1:, 1:], lim_t1, lim_t2)
        if xyz1.any():
            if len(xyz1[0, :]) > 1:
                raise ValueError('Multiple coordinates for star: {} | Yc: {}'.format(star_n, yc_assumed))
            else:
                ts1 = xyz1[0, 0]  # xyz[0, :] ALL X coordinates
        else:
            ts1 = t_llm_mdot[0, -1]

        ts2 = 0
        yc2 = yc[-1]
        xyz2 = self.get_xyz_from_yz(yc2, star_n, 'lm', 'mdot',
                                   t_llm_mdot[0, 1:], t_llm_mdot[1:, 0], t_llm_mdot[1:, 1:], lim_t1, lim_t2)
        if xyz2.any():
            if len(xyz2[0, :]) > 1:
                raise ValueError('Multiple coordinates for star: {} | Yc: {}'.format(star_n, yc_assumed))
            else:
                ts2 = xyz2[0, 0]  # xyz[0, :] ALL X coordinates
        else:
            ts2 = t_llm_mdot[0, 0]

        print('\t__STAR: {} | {} : {} (+{} -{})'.format(star_n, 'Ts', ts, "%.2f" % np.abs(ts - ts1),
                                                        "%.2f" % np.abs(ts - ts2)))

        return ts1, ts2

    # def get_star_lm_evol_err(self, star_n, l_or_lm, yc_assumed, yc1, yc2):
    #
    #     if l_or_lm == 'l':
    #         yc_m_llm = Save_Load_tables.load_table('evol_yc_m_l', 'evol_yc', 'm', 'l', self.opal_used)
    #         star_llm = self.get_num_par('l', star_n, yc_assumed)
    #     else:
    #         yc_m_llm = Save_Load_tables.load_table('evol_yc_m_lm', 'evol_yc', 'm', 'lm', self.opal_used)
    #         star_llm = self.get_num_par('lm', star_n, yc_assumed)
    #
    #     llm = yc_m_llm[1:, 1:]
    #
    #     yc_ind  = Physics.ind_of_yc(yc_m_llm[0, 1:], yc_assumed)
    #     yc1_ind = Physics.ind_of_yc(yc_m_llm[0, 1:], yc1)
    #     yc2_ind = Physics.ind_of_yc(yc_m_llm[0, 1:], yc2)
    #
    #     yc_assumed_llm_row = llm[:, yc_ind]
    #     yc1_llm_row = llm[:, yc1_ind]
    #     yc2_llm_row = llm[:, yc2_ind]
    #
    #     f = interpolate.InterpolatedUnivariateSpline(yc_assumed_llm_row, yc1_llm_row)
    #     llm1 = np.float(f([star_llm]))
    #     f = interpolate.InterpolatedUnivariateSpline(yc_assumed_llm_row, yc2_llm_row)
    #     llm2 = np.float(f([star_llm]))
    #
    #     print('\t__STAR: {} | {} : {} (+{} -{})'.format(star_n, l_or_lm, star_llm, "%.2f"%np.abs(star_llm - llm1),
    #                                                     "%.2f" %np.abs(star_llm - llm2)))
    #
    #     return np.abs(star_llm - llm2), np.abs(star_llm - llm1),

    # def get_star_mdot_err(self, star_n, l_or_lm, yc_assumed, yc1, yc2, l_mdot_rescription = 'nugis'):
    #
    #     def get_z(opal_used):
    #         z_lmc = 0.008
    #         table_lmc = 'table_x.data'
    #         z_gal = 0.02
    #         table_gal = 'table8.data'
    #
    #         if opal_used.split('/')[-1] == table_lmc:
    #             return z_lmc
    #         if opal_used.split('/')[-1] == table_gal:
    #             return z_gal
    #
    #     if yc1 <= yc2:
    #         raise ValueError('yc1({}) <= yc2({}) (should be > )'.format(yc1, yc2))
    #
    #     if l_or_lm == 'l':
    #         llm1_, llm2_ = self.get_star_llm_evol_err(star_n, l_or_lm, yc_assumed, yc1, yc2)
    #         star_llm = self.get_num_par('l', star_n, yc_assumed)
    #         llm1 = llm1_ + star_llm  # absolute value of llm at Yc1 and Yc2
    #         llm2 = llm2_ + star_llm
    #
    #         mdot1 = Physics.l_mdot_prescriptions(llm1, 10 ** get_z(self.opal_used), l_mdot_rescription)
    #         mdot2 = Physics.l_mdot_prescriptions(llm2, 10 ** get_z(self.opal_used), l_mdot_rescription)
    #
    #         star_mdot = self.get_num_par('mdot', star_n, yc_assumed)
    #
    #         print('\t__STAR: {} | Mdot : {} (+{} -{})'.format(star_n, star_mdot, "%.2f"%np.abs(star_mdot - mdot1),
    #                                                           "%.2f" %np.abs(star_mdot - mdot2)))
    #
    #         return np.abs(star_mdot - mdot1), np.abs(star_mdot - mdot2)
    #
    #     # llm1_, llm2_ = self.get_star_llm_evol_err(star_n, l_or_lm, yc_assumed, yc1, yc2)
    #     #
    #     # if l_or_lm == 'l':
    #     #     star_llm = self.get_num_par('l', star_n, yc_assumed)
    #     # else:
    #     #     star_llm = self.get_num_par('lm', star_n, yc_assumed)
    #     #
    #     # llm1 = llm1_ + star_llm # absolute value of llm at Yc1 and Yc2
    #     # llm2 = llm2_ + star_llm
    #     #
    #     # mdot1 = Physics.l_mdot_prescriptions(llm1, 10 * get_z(self.opal_used), l_mdot_rescription)
    #     # mdot2 = Physics.l_mdot_prescriptions(llm2, 10 * get_z(self.opal_used), l_mdot_rescription)

    def get_t_eff_from_atmosphere(self, star_n):

        tstar = self.get_num_par('t', star_n)
        rt = self.get_num_par('lRt', star_n)


        return Math.interpolate_value_table(self.atm_table, [tstar, rt])

class Read_Plot_file:

    # path = './data/'
    # compart = '.plot1'

    def __init__(self, plot_table, sonic_bec = False):
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
        self.lm_    = Physics.loglm(self.l_, self.m_,True)
        self.rho_c  = plot_table[:i_stop,10]        # dlog10(ro(1))
        self.l_carb = plot_table[:i_stop,11]        # escop
        self.l_nu  = plot_table[:i_stop,12]         # esnyp
        self.mdot_=np.log10(plot_table[:i_stop,13]) # windmd*sec/sun
        self.t_max = plot_table[:i_stop,14]         # vvcmax
        self.rho_at_t_max = plot_table[:i_stop,15]  # gammax
        self.m_at_t_max = plot_table[:i_stop,16]    # wcrit
        if sonic_bec:
            self.logg = plot_table[:i_stop,17]          # logg
            self.rwindluca = plot_table[:i_stop,18]     # rwindluca/rsun
            self.r_n_rsun = plot_table[:i_stop, 19]     # r(n)/rsun
            self.teffluca = np.log10(plot_table[:i_stop, 20])   # teffluca
            self.Tn = np.log10(plot_table[:i_stop, 21])         # T(N)
            self.tauatR = np.log10(plot_table[:i_stop, 22]   )  # tauatR
            self.windform_23 = plot_table[:i_stop, 23]          # Abs(windmd)/(4.d0*PI*rwindluca**2*(v0+(vinf-v0)*(1.d0-R(N)/rwindluca))),
            self.windform_24 = plot_table[:i_stop, 24]          # v0+(vinf-v0)*(1-R(N)/rwindluca)
    # def

    def get_col(self, v_n):
        if v_n == 'time':
            return self.time
        if v_n == 't_c':
            return self.t_c
        if v_n == 'yc':
            return self.y_c
        if v_n == 'l_h':
            return self.l_h
        if v_n == 'l_he':
            return self.l_he
        if v_n == 'm':
            return self.m_
        if v_n == 'gp':
            return np.int(self.gp[0])
        if v_n == 'mdot':
            return self.mdot_
        if v_n == 't':
            return self.t_eff
        if v_n == 'l':
            return self.l_
        if v_n == 'rwindluca':
            return self.rwindluca
        if v_n == 'teffluca':
            return self.teffluca
        if v_n == 'tauatR':
            return self.tauatR
        if v_n == 'lm':
            return self.lm_

        raise NameError('Name: {} is not defined here. Sorrry:('.format(v_n))

    @classmethod
    def from_file(cls, plot_file_name, sonic_bec = False):

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
        return cls((np.vstack((np.zeros(len(table[:, 0])), table.T))).T, sonic_bec)

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

    # def get_wne_val(self, v_n):
    #     he_surf0 = self.

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

    def get_xyz_from_yz(self, model_i, condition, y_name, z_name, x_1d_arr, y_1d_arr, z_2d_arr, lx1 = None, lx2 = None):
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

    def get_tpar(self):

        return Physics.opt_depth_par2(self.rho_, self.t_, self.r_, self.u_, self.kappa_, self.mu_)
                               # self.mdl[i].get_col('r'), self.mdl[i].get_col('u'), self.mdl[i].get_col('kappa'),
                               # self.mdl[i].get_col('mu'))

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
            return Physics.loglm(self.l_, self.xm_)

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

        if v_n == 'Pg/P_total':
            return self.Pg_/self.P_total_

        if v_n == 'Pr/P_total':
            return self.Pr_/self.P_total_

        if v_n == 'mfp':
            return -(self.rho_ + self.kappa_)

        if v_n == 'mfp/c':
            return 10**(-(self.rho_ + self.kappa_))/Constants.light_v

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
            pass
            # raise ValueError('\t__Error. Sound speed is not found in data. |get_sp|')

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

        if v_n == 'teff'and condition == '':
            ind = self.ind_from_condition(condition)
            l = self.get_col('l')[ind]
            r = self.get_col('r')[ind]
            return Physics.steph_boltz_law_t_eff(l, r)

        if v_n == 'teff/ts4' and condition =='':
            ind = self.ind_from_condition(condition)
            l = self.get_col('l')[ind]
            r = self.get_col('r')[ind]
            ts = 10**self.get_col('t')[ind]
            teff = 10**Physics.steph_boltz_law_t_eff(l, r)


            return ((teff/ts)**4)*100 # in %

        if v_n == 'tpar' and condition == '': # optical depth parameter can be estimated only at a point, as it requires dr/du
            return Physics.opt_depth_par2(self.rho_,self.t_,self.r_,self.u_,self.kappa_,self.mu_)

        # if v_n == 'r_env' and condition == '':
        #     return self.get_m_r_envelope('r')
        #
        # if v_n == 'm_env' and condition == '':
        #     return self.get_m_r_envelope('xm')

        ind = self.ind_from_condition(condition)
        return np.float(self.get_col(v_n)[ind])

    def get_par_table(self, y_name = 'l', i = -1):


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
                # "%2.0f" % model,
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
                # "%2.0f" % model,
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

class Read_SP_data_file_OLD:

    def __init__(self, sp_data_file, out_dir = '../data/output/', plot_dir = '../data/plots/'):

        self.files = sp_data_file
        self.out_dir = out_dir
        self.plot_dir = plot_dir


        self.list_of_v_n = ['l', 'm', 't', 'mdot', 'tau', 'r', 'Yc', 'k']

        self.table = np.loadtxt(sp_data_file)

        print('File: {} has been loaded successfully.'.format(sp_data_file))

        # --- Critical values ---

        self.l_cr = np.float(self.table[0, 0])  # mass array is 0 in the sp file
        self.m_cr = np.float(self.table[0, 1])  # mass array is 1 in the sp file
        self.yc_cr = np.float(self.table[0, 2])  # mass array is 2 in the sp file
        self.ys_cr = np.float(self.table[0, 3])  # mass array is 2 in the sp file
        self.lmdot_cr = np.float(self.table[0, 4])  # mass array is 3 in the sp file
        self.r_cr = np.float(self.table[0, 5])  # mass array is 4 in the sp file
        self.t_cr = np.float(self.table[0, 6])  # mass array is 4 in the sp file
        self.tau_cr = np.float(self.table[0, 7])

        # --- Sonic Point Values ---

        self.l = np.array(self.table[1:, 0])
        self.m = np.array(self.table[1:, 1])
        self.yc = np.array(self.table[1:, 2])
        self.ys = np.array(self.table[1:, 3])
        self.lmdot = np.array(self.table[1:, 4])
        self.rs = np.array(self.table[1:, 5])
        self.ts = np.array(self.table[1:, 6])
        self.tau = np.array(self.table[1:, 7])
        self.k = np.array(self.table[1:, 8])
        self.rho = np.array(self.table[1:, 9])

        # self.k = np.array(self.table[1:, 6])
        # self.rho = np.array(self.table[1:, 8])

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

        if v_n == 'ys':
            return np.float(self.ys_cr)

        if v_n == 'tau':
            return self.tau_cr

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

        if v_n == 'ys':
            return self.ys

        if v_n == 'k':
            return self.k

        if v_n == 'rho':
            return self.rho

        if v_n == 'tau':
            return self.tau

        if v_n == 'lm':
            return Physics.loglm(self.l, self.m, True)

        raise NameError('v_n {} is not in the list: {} (for critical values)'.format(v_n, self.list_of_v_n))

class Read_SP_data_file:

    list_of_names = ['log(L)', 'M(Msun)', 'Yc', 'mdot', 'r-sp' ,'t-sp', 'Tau', 'R_wind', 'T_eff', 'kappa-sp', 'L/Ledd-sp', 'rho-sp']

    # log(L)  M(Msun)     Yc     Ys      l(Mdot)   Rs(Rsun) log(Ts) kappa-sp L/Ledd-sp rho-sp

    def __init__(self, sp_data_file, out_dir = '../data/output/', plot_dir = '../data/plots/'):

        self.files = sp_data_file
        self.out_dir = out_dir
        self.plot_dir = plot_dir

        self.names, self.table = self.read_genergic_table(sp_data_file) # now it suppose to read the table

        # self.list_of_v_n = ['l', 'm', 't', 'mdot', 'tau', 'r', 'Yc', 'k']

        # self.table = []
        # with open(sp_data_file, 'r') as f:
        #     for line in f:
        #         # if '#' not in line.split() and line.strip():  # if line is not empty and does not contain '#'
        #         self.table.append(line)
        #
        # self.names = self.table[0].split()[1:] # getting rid of '#' which is the first element in a row
        # self.num_colls = len(self.table) - 1  # as first row is of var names
        #
        # if len(self.names) != len(self.table[1].split()):
        #     print('\t___Error. Number of vars in list({}) != number of cols in observ.data file({}) '
        #           '|Read_Observables, __init__|'.format(len(self.names), len(self.table[1].split())))
        # print('\t__Note: Data include following paramters:\n\t | {} |'.format(self.names))
        #
        # self.table.remove(self.table[0])  # removing the var_names line from the array. (only actual values left)
        #
        # tmp = np.zeros(len(self.names))
        # for row in self.table:
        #     tmp = np.vstack((tmp, np.array(row.split(), dtype=np.float)))
        # self.table = np.delete(tmp, 0, 0)
        #
        # print('File: {} has been loaded successfully.'.format(sp_data_file))

    @staticmethod
    def read_genergic_table(file_name):
        '''
        Reads the the file table, returning the list with names and the table
        structure: First Row must be with '#' in the beginning and then, the var names.
        other Rows - table with the same number of elements as the row of var names
        :return:
        '''
        table = []
        with open(file_name, 'r') as f:
            for line in f:
                # if '#' not in line.split() and line.strip():  # if line is not empty and does not contain '#'
                table.append(line)

        names = table[0].split()[1:]  # getting rid of '#' which is the first element in a row
        num_colls = len(table) - 1  # as first row is of var names

        if len(names) != len(table[1].split()):
            print('\t___Error. Number of vars in list({}) != number of cols in observ.data file({}) '
                  '|Read_Observables, __init__|'.format(len(names), len(table[1].split())))
        print('\t__Note: Data include following paramters:\n\t | {} |'.format(names))

        table.remove(table[0])  # removing the var_names line from the array. (only actual values left)

        tmp = np.zeros(len(names))
        for row in table:
            tmp = np.vstack((tmp, np.array(row.split(), dtype=np.float)))
        table = np.delete(tmp, 0, 0)

        return names, table


    def v_n_to_v_n(self, v_n):
        if v_n == 'm': return 'M(Msun)'
        if v_n == 'l': return 'log(L)'
        if v_n == 'yc' or v_n == 'Yc': return 'Yc'
        if v_n == 'ys': return 'Ys'
        if v_n == 'mdot': return 'mdot'
        if v_n == 'r': return 'r-sp'
        if v_n == 't': return 't-sp'
        if v_n == 'r_env': return 'r_env'
        if v_n == 'm_env': return 'm_env'
        if v_n == 'k' or v_n == 'kappa': return 'kappa-sp'
        if v_n == 'gamma' or v_n == 'L/Ledd': return 'L/Ledd-sp'
        if v_n == 'rho': return 'rho-sp'
        if v_n == 'tau': return 'tau-sp'
        if v_n == 'R_wind': return 'R_wind'
        if v_n == 't_eff': return 't-ph'
        if v_n == 'ys'or v_n == 'Ys': return 'Ys'
        if v_n == 'hp' or v_n == 'HP': return 'HP-sp' # log(Hp)
        if v_n == 'tpar': return 'tpar-'
        if v_n == 'mfp': return 'mfp-sp' # log(mfp)

        # if v_n == 'lm': return 'lm'
        raise NameError('Translation for v_n({}) not provided'.format(v_n))

    def get_crit_value(self, v_n):

        if v_n == 'lm':
            l = self.table[0, self.names.index(self.v_n_to_v_n('l'))]
            m = self.table[0, self.names.index(self.v_n_to_v_n('m'))]
            return Physics.loglm(l, m)


        v_n_tr = self.v_n_to_v_n(v_n)
        if not v_n_tr in self.names:
            raise NameError('v_n_traslated({}) not in the list of names from file ({})'.format(v_n_tr, self.names))
        return self.table[0, self.names.index(self.v_n_to_v_n(v_n))]

    def get_sonic_cols(self, v_n):

        if v_n == 'lm':
            l = self.table[1:, self.names.index(self.v_n_to_v_n('l'))]
            m = self.table[1:, self.names.index(self.v_n_to_v_n('m'))]
            return Physics.loglm(l, m, True)
        else:
            return self.table[1:, self.names.index(self.v_n_to_v_n(v_n))]

        # --- Critical values ---

        # self.l_cr = np.float(self.table[0, 0])  # mass array is 0 in the sp file
        # self.m_cr = np.float(self.table[0, 1])  # mass array is 1 in the sp file
        # self.yc_cr = np.float(self.table[0, 2])  # mass array is 2 in the sp file
        # self.ys_cr = np.float(self.table[0, 3])  # mass array is 2 in the sp file
        # self.lmdot_cr = np.float(self.table[0, 4])  # mass array is 3 in the sp file
        # self.r_cr = np.float(self.table[0, 5])  # mass array is 4 in the sp file
        # self.t_cr = np.float(self.table[0, 6])  # mass array is 4 in the sp file
        # self.tau_cr = np.float(self.table[0, 7])
        #
        # # --- Sonic Point Values ---
        #
        # self.l = np.array(self.table[1:, 0])
        # self.m = np.array(self.table[1:, 1])
        # self.yc = np.array(self.table[1:, 2])
        # self.ys = np.array(self.table[1:, 3])
        # self.lmdot = np.array(self.table[1:, 4])
        # self.rs = np.array(self.table[1:, 5])
        # self.ts = np.array(self.table[1:, 6])
        # self.tau = np.array(self.table[1:, 7])
        # self.k = np.array(self.table[1:, 8])
        # self.rho = np.array(self.table[1:, 9])

        # self.k = np.array(self.table[1:, 6])
        # self.rho = np.array(self.table[1:, 8])



    #
    #
    # def get_crit_value(self, v_n):
    #     if v_n == 'l':
    #         return np.float( self.l_cr )
    #
    #     if v_n =='m' or v_n == 'xm':
    #         return np.float( self.m_cr )
    #
    #     if v_n == 't':
    #         return np.float( self.t_cr )
    #
    #     if v_n == 'mdot':
    #         return np.float( self.lmdot_cr)
    #
    #     if v_n == 'r':
    #         return np.float(self.r_cr)
    #
    #     if v_n == 'Yc':
    #         return np.float(self.yc_cr)
    #
    #     if v_n == 'ys':
    #         return np.float(self.ys_cr)
    #
    #     if v_n == 'tau':
    #         return self.tau_cr
    #
    #     if v_n == 'lm':
    #         return np.float(Physics.loglm(self.l_cr, self.m_cr, False) )
    #
    #     raise NameError('v_n {} is not in the list: {} (for critical values)'.format(v_n, self.list_of_v_n))
    #
    # def get_sonic_cols(self, v_n):
    #     if v_n == 'l':
    #         return self.l
    #
    #     if v_n =='m' or v_n == 'xm':
    #         return self.m
    #
    #     if v_n == 't':
    #         return self.ts
    #
    #     if v_n == 'mdot':
    #         return self.lmdot
    #
    #     if v_n == 'r':
    #         return self.rs
    #
    #     if v_n == 'Yc':
    #         return self.yc
    #
    #     if v_n == 'ys':
    #         return self.ys
    #
    #     if v_n == 'k':
    #         return self.k
    #
    #     if v_n == 'rho':
    #         return self.rho
    #
    #     if v_n == 'tau':
    #         return self.tau
    #
    #     if v_n == 'lm':
    #         return Physics.loglm(self.l, self.m, True)
    #
    #     raise NameError('v_n {} is not in the list: {} (for critical values)'.format(v_n, self.list_of_v_n))

class Read_Wind_file:
    def __init__(self, suca_table):

        self.table = suca_table
        self.v = suca_table[:, 1]/100000
        self.r = suca_table[:, 2]/Constants.solar_r
        self.rho=np.log10(suca_table[:, 3])
        self.t = np.log10(suca_table[:, 4])
        self.kappa=suca_table[:, 5]
        self.tau = suca_table[:, 6]
        self.gp = suca_table[:, 7]
        self.mdot=suca_table[:, 8]
        self.nine=suca_table[:, 9]
        self.ten = suca_table[:,10]
        self.eleven=suca_table[:, 11] # wind luminocity
        self.kappa_eff=suca_table[:,12]
        self.thirteen=np.log10(suca_table[:, 13]) # eta
        self.last = suca_table[:, -1]

        self.var_names=['u', 'r', 'rho', 't', 'kappa', 'tau', 'gp', 'mdot', '9', '10', '11', 'kappa_eff', '13', 'last']

    def get_col(self, v_n):
        if v_n == 'u':
            return self.v
        if v_n == 'r':
            return self.r
        if v_n == 'rho':
            return self.rho
        if v_n == 't':
            return self.t
        if v_n == 'kappa':
            return self.kappa
        if v_n == 'tau':
            return self.tau
        if v_n == 'gp':
            return np.int(self.gp[0])
        if v_n == 'mdot':
            return np.log10(self.mdot/Constants.smperyear)
        if v_n == '9':
            return self.nine # diffusive approximation temperature
        if v_n == '10':
            return self.ten
        if v_n == '11':
            return self.eleven
        if v_n == 'kappa_eff':
            return self.kappa_eff
        if v_n == '13':
            return self.thirteen
        if v_n == 'last':
            return self.last

    def get_value(self, v_n, condition='ph'):
        '''
        Condtitions: 0, 1, ... Special: 'ph'
        :param v_n:
        :param condition:
        :return:
        '''
        if condition == 'ph':
            i = self.get_col('gp')
            return self.get_col(v_n)[i]
        else:
            return self.get_col(v_n)[condition]


    @classmethod
    def from_wind_dat_file(cls, name):
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

        for i in range(raws): # WARNING ! RADING ROWS N-1! BECAUSE the last row can have 'infinity'
            if not 'Infinity' in f[i].split(' '):
                if '0.14821969375237-322' in f[i].split(' '):
                    parts = f[i].split('0.14821969375237-32')
                    res = parts[0] + '0.00000000000000D+00' + parts[-1]         # In case the vierd value is there :(
                    print('\t__Replaced Row in WIND file')
                    table[i, :] = np.array(res.replace("D", "E").split())
                else:
                    table[i, :] = np.array(f[i].replace("D", "E").split())
        f.clear()

        # print('\t__Note: file *',full_name,'* has been loaded successfully.')
        # attached an empty raw to match the index of array
        return cls((np.vstack((np.zeros(len(table[:, 0])), table.T))).T)

class Read_Atmosphere_File:

    list_of_names = ['model', 't_*', 'rt',  'l', 'mdot', 'v_inf', 'r_*',  't_eff', 'r_eff', 'm','g','q','d_inf','eta']

    def __init__(self, atm_file, opal_used, out_dir = '../data/output/', plot_dir = '../data/plots/'):

        gal_atm_file = '../data/atm/gal_atm_models.data'
        lmc_atm_file = '../data/atm/lmc_atm_models.data'


        if opal_used.split('/')[-1] == 'table8.data':
            self.atm_file = gal_atm_file
        if opal_used.split('/')[-1] == 'table_x.data':
            self.atm_file = lmc_atm_file


        self.out_dir = out_dir
        self.plot_dir= plot_dir

        self.names, self.table, self.models = Save_Load_tables.read_genergic_table(self.atm_file, 'model') # now it suppose to read the table

    def v_n_to_v_n(self, v_n):

        if not v_n in self.names:
            raise NameError('Translation is not provided for {} \n (not found in {})'.format(v_n, self.names))

        else: return v_n

    def get_col(self, v_n):
        col = self.table[:, self.names.index(self.v_n_to_v_n(v_n))]

        if v_n == 't_*' or v_n == 't_eff':
            return np.log10(col*1000)
        else:
            return col

    def get_value(self, v_n, str_model):
        value = self.table[self.models.index(str_model), self.names.index(self.v_n_to_v_n(v_n))]

        if v_n == 't_*' or v_n == 't_eff':
            return np.log10(value*1000)
        else:
            return value

    #
    #
    #
    #
    # def get_crit_value(self, v_n):
    #
    #
    #     if v_n == 'lm':
    #         l = self.table[0, self.names.index(self.v_n_to_v_n('l'))]
    #         m = self.table[0, self.names.index(self.v_n_to_v_n('m'))]
    #         return Physics.loglm(l, m)
    #
    #
    #     v_n_tr = self.v_n_to_v_n(v_n)
    #     if not v_n_tr in self.names:
    #         raise NameError('v_n_traslated({}) not in the list of names from file ({})'.format(v_n_tr, self.names))
    #     return self.table[0, self.names.index(self.v_n_to_v_n(v_n))]
    #
    # def get_sonic_cols(self, v_n):
    #
    #     if v_n == 'lm':
    #         l = self.table[1:, self.names.index(self.v_n_to_v_n('l'))]
    #         m = self.table[1:, self.names.index(self.v_n_to_v_n('m'))]
    #         return Physics.loglm(l, m, True)
    #     else:
    #         return self.table[1:, self.names.index(self.v_n_to_v_n(v_n))]
    #
    #     # --- Critical values ---
    #
    #     # self.l_cr = np.float(self.table[0, 0])  # mass array is 0 in the sp file
    #     # self.m_cr = np.float(self.table[0, 1])  # mass array is 1 in the sp file
    #     # self.yc_cr = np.float(self.table[0, 2])  # mass array is 2 in the sp file
    #     # self.ys_cr = np.float(self.table[0, 3])  # mass array is 2 in the sp file
    #     # self.lmdot_cr = np.float(self.table[0, 4])  # mass array is 3 in the sp file
    #     # self.r_cr = np.float(self.table[0, 5])  # mass array is 4 in the sp file
    #     # self.t_cr = np.float(self.table[0, 6])  # mass array is 4 in the sp file
    #     # self.tau_cr = np.float(self.table[0, 7])
    #     #
    #     # # --- Sonic Point Values ---
    #     #
    #     # self.l = np.array(self.table[1:, 0])
    #     # self.m = np.array(self.table[1:, 1])
    #     # self.yc = np.array(self.table[1:, 2])
    #     # self.ys = np.array(self.table[1:, 3])
    #     # self.lmdot = np.array(self.table[1:, 4])
    #     # self.rs = np.array(self.table[1:, 5])
    #     # self.ts = np.array(self.table[1:, 6])
    #     # self.tau = np.array(self.table[1:, 7])
    #     # self.k = np.array(self.table[1:, 8])
    #     # self.rho = np.array(self.table[1:, 9])
    #
    #     # self.k = np.array(self.table[1:, 6])
    #     # self.rho = np.array(self.table[1:, 8])

    def get_2d_arr(self, v_n, filling=-1, replace_to=-1):

        arr = np.zeros((100, 100))
        arr.fill(filling)

        models = np.zeros(2)
        for m in self.models:
            row = np.array([m.split('-')[0], m.split('-')[-1]], dtype=np.int)
            models = np.vstack((models, row))

        models = np.delete(models, 0, 0)

        for x in models[:, 0]:
            for y in models[:, 1]:
                str_model = "%0.2d" % x + '-' + "%0.2d" % y
                if str_model in self.models:
                    # print('Model {} found'.format(str_model))
                    arr[np.int(y), np.int(x)] = self.get_value(v_n, str_model)
                # else:
                # print('Model {} not found'.format(str_model))

        arr = arr[~np.all(arr == filling, axis=1)]
        arr = arr.T[~np.all(arr.T == filling, axis=1)].T

        arr[arr == filling] = replace_to

        # print(arr)

        return arr

    def get_table(self, x_v_n, y_v_n, z_v_n, filling=None):

        x_data = np.unique(self.get_2d_arr(x_v_n, -1))
        x_data = np.delete(x_data, 0, 0)
        y_data = np.unique(self.get_2d_arr(y_v_n, -1))
        y_data = np.delete(y_data, 0, 0)

        z_data = self.get_2d_arr(z_v_n, -1, filling)

        if len(x_data) != len(z_data[0, :]): raise ValueError('Wrong x_len: len(x_data){}!=len(z_data[0,:]){}'
                                                              .format(len(x_data), len(z_data[0, :])))
        if len(y_data) != len(z_data[:, 0]): raise ValueError('Wrong x_len: len(x_data){}!=len(z_data[0,:]){}'
                                                              .format(len(y_data), len(z_data[:, 0])))

        table = Math.combine(x_data, y_data, z_data[::-1, :])

        return table



    def plot_tstar_rt(self, x_v_n, y_v_n, z_v_n):

        models = np.zeros(2)
        for m in self.models:
            row = np.array([m.split('-')[0], m.split('-')[-1]], dtype=np.int)
            models = np.vstack((models, row))

        models=np.delete(models, 0, 0)

        table = self.get_table('t_*', 'rt', 't_eff', None)

        from Phys_Math_Labels import Plots
        plots=Plots()
        plots.plot_color_table(table, x_v_n, y_v_n, z_v_n, '../data/opal/table8.data')


        # z_cood = Math.interpolate_value_table(table, [4.85, 0.9], '1dCubic')
        # plots.plot_color_table(table, 't', 'r', 't_eff', '../data/opal/table8.data', 'x:{} y{} z:{}'
        #                        .format("%.2f"%4.85, "%.2f"%0.9, "%.2f"%z_cood))





        return 0

        from Phys_Math_Labels import Plots

