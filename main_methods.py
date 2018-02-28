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
from Read_Obs_Numers import Read_SM_data_file
from Read_Obs_Numers import Read_SP_data_file

from PhysPlots import PhysPlots
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

        return file_table #[x1, x2, y1, y2, n1, n2]

class Creation:

    def __init__(self, opal_name, t1, t2, n_interp = 1000, load_lim_cases = False, output_dir = '../data/output/', plot_dir = '../data/plots/'):
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

class Observables:

    clumping_used = 4
    cluming_required = 4

    def __init__(self, obs_files):

        self.files = [obs_files]
        self.n_of_fls = len(obs_files)


        self.obs = []
        for i in range(len(self.files)):
            self.obs.append( Read_Observables(self.files[i], self.clumping_used, self.cluming_required))

        if (len(self.files)) > 1 :
            for i in range(1,len(self.files)):
                if not np.array_equal(self.obs[i-1].names, self.obs[i].names):
                    print('\t__Error. Files with observations contain different *names* row')
                    print('\t  {} has: {} \n\t  {} has: {} '
                          .format(self.files[i-1], self.obs[i-1].names, self.files[i], self.obs[i].names))
                    raise NameError('Files with observations contain different *names* row')

    def check_if_var_name_in_list(self, var_name):
        if var_name == 'lm' or var_name == 'ts' or var_name == 'rs': # special case for L/M and sonic temperature
            pass
        else:
            for i in range(self.n_of_fls):
                if var_name not in self.obs:
                    print('\n\t__Error. Variable:  {} is not in the list of names: \n\t  {} \n\t  in file: {}'
                          .format(var_name, self.obs[i].names, self.files[i]))
                    raise  NameError('Only lm, l, and rs varnames are available. {} is not listed.'.format(var_name))

    def get_x_y_of_all_observables(self, x_name, y_name, var_for_label,
                                   ts_arr = np.empty(1,), l_lm_arr= np.empty(1,), m_dot= np.empty(1,),
                                   lim_t1_obs = None, lim_t2_obs = None):
        '''
        RETURN:  np.array( [plotted_stars, plotted_labels] )  [0][:,0] - nums of all plotted stars
                                                              [0][:,1] - x - coord.
                                                              [0][:,2] - y - coord
                                                              [0][:,3] - ints from 0 to 9, uniqe for uniqe 'var_for_label'
                                                              [1][:,0] - nums of selected stars for labels
                                                              [1][:,1] - x - coord
                                                              [1][:,2] - y - coord
                                                              [1][:,3] - ints from 0 to 9
        To get index in the [0] arr of the element in [1] Use: int( np.where( res[0][:, 0]==res[1][j, 0] )[0] )

        Warning! If there are more unique str(var_for_label), PROGRAM BRAKES
        :param x_name:
        :param y_name:
        :param var_for_label:
        :param ts_arr:
        :param l_lm_arr:
        :param m_dot:
        :param lim_t1_obs:
        :param lim_t2_obs:
        :return:
        '''
        self.check_if_var_name_in_list(x_name)
        self.check_if_var_name_in_list(y_name)
        self.check_if_var_name_in_list(var_for_label)

        s = 0

        leble = []
        plotted_stars = np.array([0., 0., 0., 0.])
        plotted_labels= np.array([0., 0., 0., 0. ])

        # if self.obs != None:  # plot observed stars
        ''' Read the observables file and get the necessary values'''
        ts_ = []
        y_coord_ = []

        import re  # for searching the number in 'WN7-e' string, to plot them different colour
        for i in range(self.obs[s].num_stars):
            star_x_coord = []
            star_y_coord = []

            # ---------------------------------------Y-------------------------
            if y_name == 'lm':
                star_y_coord = [ Physics.loglm(self.obs[s].obs_par('l', float)[i],
                                             self.obs[s].obs_par('m', float)[i]) ]
            else:
                star_y_coord = [ self.obs[s].obs_par(y_name, float)[i] ]


            # ---------------------------------------X-------------------------
            if x_name == 'ts' or x_name == 'rs':
                if not ts_arr.any() or not l_lm_arr.any() or not m_dot.any():
                    print('\t__Error. For ts to be evaluated for a star : *ts_arr, l_lm_arr, m_dot* to be provided')
                    raise ValueError

                x_y_coord = Physics.lm_mdot_obs_to_ts_lm(ts_arr, l_lm_arr, m_dot, star_y_coord[0],
                                                         self.obs[s].obs_par('mdot', float)[i],
                                                         i, lim_t1_obs, lim_t2_obs)
                if x_y_coord.any():
                    ts_ = np.append(ts_, x_y_coord[1, :])  # FOR linear fit
                    y_coord_ = np.append(y_coord_, x_y_coord[0, :])
                    star_x_coord =  x_y_coord[1, :]
                    star_y_coord =  x_y_coord[0, :]  # If the X coord is Ts the Y coord is overritten.

            else:
                star_x_coord = [ self.obs[s].obs_par(x_name, float)[i] ]

            if x_name == 'lm':
                star_x_coord = [ Physics.loglm(self.obs[s].obs_par('l', float)[i],
                                             self.obs[s].obs_par('m', float)[i]) ]





            star_x_coord = np.array(star_x_coord)
            star_y_coord = np.array(star_y_coord)
            if len(star_x_coord) == len(star_y_coord) and star_x_coord.any() :

                se = re.search(r"\d+(\.\d+)?", self.obs[s].obs_par('type', str)[i])  # this is searching for the niumber
                #             color = 'C' + str(int(s.group(0)))  # making a colour our of C1 - C9 range

                for j in range(len(star_x_coord)):  # plot every solution in the degenerate set of solutions

                    row = self.obs[s].table[i]  # to get the 0th element, which is alwas the star index

                    cur_type = int(se.group(0))
                    if cur_type not in leble:  # plotting the label for unique class of stars
                        leble.append( cur_type )

                        plotted_labels = np.vstack((plotted_labels, np.array((int(row[0:3]),
                                                                              star_x_coord[j],
                                                                              star_y_coord[j],
                                                                              cur_type ))))

                    plotted_stars = np.vstack((plotted_stars, np.array((int(row[0:3]),
                                                                        star_x_coord[j],
                                                                        star_y_coord[j],
                                                                        cur_type ))))  # for further printing
            # print(self.obs[s].table)

        # -----------------------------------------------LINEAR FIT TO THE DATA-------------------------------------
        # ts_grid_y_grid = Math.line_fit(ts_, y_coord_)
        # plt.plot(ts_grid_y_grid[0, :], ts_grid_y_grid[1, :], '-.', color='blue')
        # np.delete(plotted_stars,1,0)
        plotted_stars  = np.delete(plotted_stars, 0, 0) # removing [0,0,0,] row
        plotted_labels = np.delete(plotted_labels, 0, 0)

        if plotted_stars.any():
            print('\n| Plotted Stras from Observ |')
            print('|  i  | {} | {}  | col |'.format(x_name, y_name))
            print('|-----|---------|----------|')
            for i in range(len(plotted_stars[:, 0])):
                print('| {} |  {} \t| {} | {} |'.format("%3.f" % plotted_stars[i, 0], "%.2f" % plotted_stars[i, 1],
                                                 "%.2f" % plotted_stars[i, 2], plotted_stars[i, 3]))

        if plotted_labels.any():
            print('\n| Plotted Labels from Observ |')
            print('|  i  | {} | {}  | col |'.format(x_name, y_name))
            print('|-----|-----------|---------|')
            for i in range(len(plotted_labels[:, 0])):
                print('| {} |  {} \t| {} | {} |'.format("%3.f" % plotted_labels[i, 0], "%.2f" % plotted_labels[i, 1],
                                                 "%.2f" % plotted_labels[i, 2], plotted_labels[i, 3]))

        return( np.array( [plotted_stars, plotted_labels] ) )

class Combine:
    output_dir = '../data/output/'
    plot_dir = '../data/plots/'

    def __init__(self, smfls = list(), spfls = list(), plotfls = list(), obs_files = list(), opal_used = None):
        self.num_files = smfls
        self.plt_pltfiles = plotfls
        self.obs_files = obs_files
        self.sp_files = spfls


        self.mdl = []
        for file in smfls:
            self.mdl.append( Read_SM_data_file.from_sm_data_file(file) )

        self.spmdl=[]
        for file in spfls:
            # print(spfls)
            # print(spfls[i])
            self.spmdl.append( Read_SP_data_file(file, self.output_dir, self.plot_dir) )

        # self.nums = Num_Models(smfls, plotfls)
        self.obs = Read_Observables(obs_files)

        self.opal_used = opal_used

    #--METHODS THAT DO NOT REQUIRE OPAL TABLES
    def xy_profile(self, v_n1, v_n2, var_for_label1, var_for_label2, sonic = True):

        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        tlt = v_n2 + '(' + v_n1 + ') profile'
        plt.title(tlt)

        for i in range(len(self.num_files)):

            x =      self.mdl[i].get_col(v_n1)
            y      = self.mdl[i].get_col(v_n2)          # simpler syntaxis
            label1 = self.mdl[i].get_col(var_for_label1)[-1]
            label2 = self.mdl[i].get_col(var_for_label2)[-1]

            print('\t __Core H: {} , core He: {} File: {}'.
                  format(self.mdl[i].get_col('H')[0], self.mdl[i].get_col('He4')[0], self.num_files[i]))

            lbl = '{}:{} , {}:{}'.format(var_for_label1,'%.2f' % label1,var_for_label2,'%.2f' % label2)
            ax1.plot(x,  y,  '-',   color='C' + str(Math.get_0_to_max([i], 9)[i]), label=lbl)
            ax1.plot(x, y, '.', color='C' + str(Math.get_0_to_max([i], 9)[i]), label=lbl)
            ax1.plot(x[-1], y[-1], 'x',   color='C' + str(Math.get_0_to_max([i], 9)[i]))

            ax1.annotate(str('%.2e' % 10**self.mdl[i].get_col('mdot')[-1]), xy=(x[-1], y[-1]), textcoords='data')


            if sonic and v_n2 == 'u':
                u_s = self.mdl[i].get_sonic_u()
                ax1.plot(x, u_s, '-', color='black')

                xc, yc = Math.interpolated_intercept(x,y, u_s)
                # print('Sonic r: {} | Sonic u: {} | {}'.format( np.float(xc),  np.float(yc), len(xc)))
                plt.plot(xc, yc, 'X', color='red', label='Intersection')

        ax1.set_xlabel(Labels.lbls(v_n1))
        ax1.set_ylabel(Labels.lbls(v_n2))

        ax1.grid(which='both')
        ax1.grid(which='minor', alpha=0.2)
        ax1.grid(which='major', alpha=0.2)

        ax1.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
        plot_name = self.plot_dir + v_n1 + '_vs_' + v_n2 + '_profile.pdf'
        plt.savefig(plot_name)
        plt.show()

    def xyy_profile(self, v_n1, v_n2, v_n3, var_for_label1, var_for_label2, var_for_label3, edd_kappa = True):

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
        tlt = v_n2 + ' , '+ v_n3 + ' = f(' + v_n1 + ') profile'
        plt.title(tlt)

        ax1.set_xlabel(v_n1)
        # Make the y-axis label, ticks and tick labels match the line color.
        ax1.set_ylabel(v_n2, color='b')
        ax1.tick_params('y', colors='b')
        ax1.grid()
        ax2 = ax1.twinx()

        for i in range(len(self.num_files)):

            xyy2  = self.mdl[i].get_set_of_cols([v_n1, v_n2, v_n3])
            lbl1 =  self.mdl[i].get_col(var_for_label1)[-1]
            lbl2 =  self.mdl[i].get_col(var_for_label2)[-1]
            lbl3 =  self.mdl[i].get_col(var_for_label3)[-1]

            color = 'C' + str(Math.get_0_to_max([i], 9)[i])
            lbl = '{}:{} , {}:{} , {}:{}'.format(var_for_label1, '%.2f' % lbl1, var_for_label2, '%.2f' % lbl2,
                                                 var_for_label3, '%.2f' % lbl3)

            ax1.plot(xyy2[:, 0],  xyy2[:, 1],  '-', color=color, label=lbl)
            ax1.plot(xyy2[-1, 0], xyy2[-1, 1], 'x', color=color)
            ax1.annotate(str('%.2f' % lbl1), xy=(xyy2[-1, 0], xyy2[-1, 1]), textcoords='data')

            if edd_kappa and v_n3 == 'kappa':
                k_edd = Physics.edd_opacity(self.mdl[i].get_col('xm')[-1],
                                            self.mdl[i].get_col('l')[-1])
                ax2.plot(ax1.get_xlim(), [k_edd, k_edd], color='black', label='Model: {}, k_edd: {}'.format(i, k_edd))

            ax2.plot(xyy2[:, 0],  xyy2[:, 2], '--', color=color)
            ax2.plot(xyy2[-1, 0], xyy2[-1, 2], 'o', color=color)

        ax2.set_ylabel(v_n3, color='r')
        ax2.tick_params('y', colors='r')

        plt.title(tlt, loc='left')
        fig.tight_layout()
        ax1.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
        plot_name = self.plot_dir + v_n1 + '_' + v_n2 + '_' + v_n3 + '_profile.pdf'
        plt.savefig(plot_name)
        plt.show()

    def hrd(self, plot_file_names):

        fig, ax = plt.subplots(1, 1)

        plt.title('HRD')
        plt.xlabel('log(T_eff)')
        plt.ylabel('log(L)')

        # plt.xlim(t1, t2)
        ax.grid(which='major', alpha=0.2)
        plt.legend(bbox_to_anchor=(1, 1), loc='upper right', ncol=1)

        # res = self.obs.get_x_y_of_all_observables('t', 'l', 'type')
        #
        # for i in range(len(res[0][:, 1])):
        #     ax.annotate(int(res[0][i, 0]), xy=(res[0][i, 1], res[0][i, 2]), textcoords='data')  # plot numbers of stars
        #     plt.plot(res[0][i, 1], res[0][i, 2], marker='^', color='C' + str(int(res[0][i, 3])),
        #              ls='')  # plot color dots)))
        #
        # for j in range(len(res[1][:, 0])):
        #     plt.plot(res[1][j, 1], res[1][j, 2], marker='^', color='C' + str(int(res[1][j, 3])), ls='',
        #              label='WN' + str(int(res[1][j, 3])))

        ind_arr = []
        for j in range(len(plot_file_names)):
            ind_arr.append(j)
            col_num = Math.get_0_to_max(ind_arr, 9)
            plfl = Read_Plot_file.from_file(plot_file_names[j])

            mod_x = plfl.t_eff
            mod_y = plfl.l_
            color = 'C' + str(col_num[j])

            fname = plot_file_names[j].split('/')[-2] + plot_file_names[j].split('/')[-1]# get the last folder in which the .plot1 is

            plt.plot(mod_x, mod_y, '-', color=color,
                     label='{}, m:({}->{})'.format(fname, "%.1f" % plfl.m_[0], "%.1f" % plfl.m_[-1]) )
                     # str("%.2f" % plfl.m_[0]) + ' to ' + str("%.2f" % plfl.m_[-1]) + ' solar mass')


            imx = Math.find_nearest_index( plfl.y_c, plfl.y_c.max() )
            plt.plot(mod_x[imx], mod_y[imx], 'x')
            ax.annotate("%.4f" % plfl.y_c.max(), xy=(mod_x[imx], mod_y[imx]), textcoords='data')

            plt.plot()

            for i in range(10):
                ind = Math.find_nearest_index(plfl.y_c, (i / 10))
                # print(plfl.y_c[i], (i/10))
                x_p = mod_x[ind]
                y_p = mod_y[ind]
                plt.plot(x_p, y_p, '.', color='red')
                ax.annotate("%.2f" % plfl.y_c[ind], xy=(x_p, y_p), textcoords='data')

        ax.grid(which='both')
        ax.grid(which='minor', alpha=0.2)

        plt.gca().invert_xaxis() # inverse x axis

        plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
        plot_name = self.output_dir + 'hrd.pdf'
        plt.savefig(plot_name)

        plt.show()

    def xy_last_points(self, v_n1, v_n2, v_lbl1, num_pol_fit = True):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        # nums = Treat_Numercials(self.num_files)  # Surface Temp as a x coordinate
        # res = nums.get_x_y_of_all_numericals('sp', 'r', 'l', 'mdot', 'color')
        x = []
        y = []
        for i in range(len(self.num_files)):
            x = np.append(x, self.mdl[i].get_cond_value(v_n1, 'sp') )
            y = np.append(y, self.mdl[i].get_cond_value(v_n2, 'sp') )

            lbl1 = self.mdl[i].get_cond_value(v_lbl1, 'sp')
            # print(x, y, lbl1)

            plt.plot(x[i], y[i], marker='.', color='C' + str(Math.get_0_to_max([i],9)[i]), ls='', label='{}:{} , {}:{} , {}:{}'
                     .format(v_n1, "%.2f" % x[i], v_n2, "%.2f" % y[i], v_lbl1, "%.2f" % lbl1))  # plot color dots)))
            ax.annotate(str("%.2f" % lbl1), xy=(x[i], y[i]), textcoords='data')

        if num_pol_fit:
            fit = np.polyfit(x, y, 3)  # fit = set of coeddicients (highest first)
            f = np.poly1d(fit)

            # print('Equation:', f.coefficients)
            fit_x_coord = np.mgrid[(x.min()):(x.max()):100j]
            plt.plot(fit_x_coord, f(fit_x_coord), '--', color='black', label='Model_fit')


        name = self.output_dir+'{}_{}_dependance.pdf'.format(v_n2,v_n1)
        plt.title('{} = f({}) plot'.format(v_n2,v_n1))
        plt.xlabel(Labels.lbls(v_n1))
        plt.ylabel(Labels.lbls(v_n2))
        plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
        plt.savefig(name)

        plt.show()

    def sp_xy_last_points(self, v_n1, v_n2, v_lbl1, num_pol_fit = True):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        # nums = Treat_Numercials(self.num_files)  # Surface Temp as a x coordinate
        # res = nums.get_x_y_of_all_numericals('sp', 'r', 'l', 'mdot', 'color')
        x = []
        y = []
        yc =[]
        xm = []
        for i in range(len(self.sp_files)):

            x = np.append(x, self.spmdl[i].get_crit_value(v_n1) )
            y = np.append(y, self.spmdl[i].get_crit_value(v_n2) )
            yc = np.append(yc, self.spmdl[i].get_crit_value('Yc'))
            xm = np.append(xm, self.spmdl[i].get_crit_value('m'))

            lbl1 = self.spmdl[i].get_crit_value(v_lbl1)


            # print(x, y, lbl1)

            plt.plot(x[i], y[i], marker='.', color='black', ls='', label='{} | {}:{} , {}:{} , {}:{}, Yc: {}'
                     .format(i, v_n1, "%.2f" % x[i], v_n2, "%.2f" % y[i], v_lbl1, "%.2f" % lbl1, yc[i]))  # plot color dots)))
            ax.annotate(str("%.2f" % xm[i]), xy=(x[i], y[i]), textcoords='data')

            # "%.2f" % yc[i]

        if num_pol_fit:
            fit = np.polyfit(x, y, 2)  # fit = set of coeddicients (highest first)
            f = np.poly1d(fit)

            # lbl = '({}) + ({}*x) + ({}*x**2) + ({}*x**3)'.format(
            #                                                     "%.3f" % f.coefficients[3],
            #                                                     "%.3f" % f.coefficients[2],
            #                                                     "%.3f" % f.coefficients[1],
            #                                                     "%.3f" % f.coefficients[0]
            #                                                     )
            # print(lbl)

            lbl = '({}) + ({}*x) + ({}*x**2)'.format(
                                                                # "%.3f" % f.coefficients[3],
                                                                "%.3f" % f.coefficients[2],
                                                                "%.3f" % f.coefficients[1],
                                                                "%.3f" % f.coefficients[0]
                                                                )
            print(lbl)
            print('{} Limits: [{}, {}]'.format(v_n1, x.min(),x.max()))
            print('{} Limits: [{}, {}]'.format(v_n2, y.min(), y.max()))

            fit_x_coord = np.mgrid[(x.min()):(x.max()):100j]
            plt.plot(fit_x_coord, f(fit_x_coord), '--', color='black', label=lbl)



        # plt.plot(x, y, '-', color='gray')


        name = self.output_dir+'{}_{}_dependance.pdf'.format(v_n2,v_n1)
        plt.title('{} = f({}) plot'.format(v_n2, v_n1))
        plt.xlabel(Labels.lbls(v_n1))
        plt.ylabel(Labels.lbls(v_n2))
        plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
        plt.savefig(name)
        plt.grid()

        plt.show()

    def sp_get_r_lt_table(self, l_or_lm, depth = 1000, plot = False, t_llm_vrho = np.empty([])):
        '''

        :param l_or_lm:
        :param t_grid:
        :param l_lm:
        :t_llm_vrho:
        :return:
        '''
        '''===========================INTERPOLATING=EVERY=ROW=TO=HAVE=EQUAL=N=OF=ENTRIES============================='''

        r = np.empty((len(self.sp_files), 100))
        t = np.empty((len(self.sp_files), 100))
        l_lm = np.empty(len(self.sp_files))

        for i in range(len(self.sp_files)):
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
            t2[:,i] = f2(l_lm_grid)

        '''=======================================INTERPOLATE=R=f(L, T)=============================================='''

        # If in every row of const. l_lm, the temp 't' is decreasing monotonically:
        t2_ = t2[:, ::-1]    # so t is from min to max increasing
        r2_ = r2[:, ::-1]    # necessary of Univariate spline

        if not t2_.any() or not r2_.any():
            raise ValueError('Array t2_{} or r2_{} is empty'.format(t2_.shape, r2_.shape))

        def interp_t_l_r(l_1d_arr, t_2d_arr, r_2d_arr, depth = 1000, t_llm_vrho = np.empty([])):

            t1 = t_2d_arr[:, 0].max()
            t2 = t_2d_arr[:, -1].min()

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
            levels = [0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6]
            contour_filled = plt.contourf(t_l_or_lm_r[0, 1:], t_l_or_lm_r[1:, 0], t_l_or_lm_r[1:,1:], levels, cmap=plt.get_cmap('RdYlBu_r'))
            plt.colorbar(contour_filled, label=Labels.lbls('r'))
            contour = plt.contour(t_l_or_lm_r[0, 1:], t_l_or_lm_r[1:, 0], t_l_or_lm_r[1:,1:], levels, colors='k')
            plt.clabel(contour, colors='k', fmt='%2.1f', fontsize=12)
            plt.title('SONIC HR DIAGRAM')

            # plt.ylabel(l_or_lm)
            plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
            # plt.savefig(name)
            plt.show()

        return t_l_or_lm_r, t_llm_vrho


    #--METHODS THAT DO REQUIRE OPAL TABLES
    def plot_t_rho_kappa(self, var_for_label1, var_for_label2,  n_int_edd = 1000, plot_edd = True):
        # self.int_edd = self.tbl_anlom_OPAL_table(self.op_name, 1, n_int, load_lim_cases)

        # t_k_rho = self.opal.interp_opal_table(t1, t2, rho1, rho2)

        t_rho_k = Save_Load_tables.load_table('t_rho_k','t','rho','k',self.opal_used, self.output_dir)


        t      = t_rho_k[0, 1:]  # x
        rho    = t_rho_k[1:, 0]  # y
        kappa  = t_rho_k[1:, 1:] # z

        plt.figure()
        levels = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        pl.xlim(t.min(), t.max())
        pl.ylim(rho.min(), rho.max())
        contour_filled = plt.contourf(t, rho, 10 ** (kappa), levels, cmap=plt.get_cmap('RdYlBu_r'))
        plt.colorbar(contour_filled)
        contour = plt.contour(t, rho, 10 ** (kappa), levels, colors='k')
        plt.clabel(contour, colors='k', fmt='%2.1f', fontsize=12)
        plt.title('OPACITY PLOT')
        plt.xlabel('Log(T)')
        plt.ylabel('log(rho)')

        # ------------------------EDDINGTON-----------------------------------
        Table_Analyze.plot_k_vs_t = False  # there is no need to plot just one kappa in the range of availability

        if plot_edd:  # n_model_for_edd_k.any():
            clas_table_anal = Table_Analyze(self.opal_used, 1000, False, self.output_dir, self.plot_dir)

            for i in range(len(self.num_files)):  # self.nmdls
                mdl_m = self.mdl[i].get_value('xm', 'sp')
                mdl_l = self.mdl[i].get_value('l',  'sp')

                k_edd = Physics.edd_opacity(mdl_m, mdl_l)

                n_model_for_edd_k = clas_table_anal.interp_for_single_k(t.min(), t.max(), n_int_edd, k_edd)
                x = n_model_for_edd_k[0, :]
                y = n_model_for_edd_k[1, :]
                color = 'black'
                lbl = 'Model:{}, k_edd:{}'.format(i, '%.2f' % 10 ** k_edd)
                plt.plot(x, y, '-.', color=color, label=lbl)
                plt.plot(x[-1], y[-1], 'x', color=color)

        Table_Analyze.plot_k_vs_t = True
        # ----------------------DENSITY----------------------------------------

        for i in range(len(self.num_files)):
            res = self.mdl[i].get_set_of_cols(['t', 'rho', var_for_label1, var_for_label2])

            lbl = '{} , {}:{} , {}:{}'.format(i, var_for_label1, '%.2f' % res[0, 2], var_for_label2, '%.2f' % res[0, 3])
            plt.plot(res[:, 0], res[:, 1], '-', color='C' + str(Math.get_0_to_max([i], 9)[i]), label=lbl)
            plt.plot(res[-1, 0], res[-1, 1], 'x', color='C' + str(Math.get_0_to_max([i], 9)[i]))
            plt.annotate(str('%.2f' % res[0, 2]), xy=(res[-1, 0], res[-1, 1]), textcoords='data')

        plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
        name = self.plot_dir + 't_rho_kappa.pdf'
        plt.savefig(name)
        plt.show()

    def plot_t_mdot_lm(self, v_lbl, r_s = 1., lim_t1_mdl = 5.2, lim_t2_mdl = None):

        t_rho_k = Save_Load_tables.load_table('t_rho_k', 't', 'rho', 'k', self.opal_used,self.output_dir)

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

        for i in range(len(self.num_files)):
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

    @staticmethod
    def empirical_l_r_crit(t_k_rho, l_or_lm):

        #=======================================EMPIRICAL=FUNCTIONS=AND=LIMITS==========================================
        l_lim = [5.15, 5.7]
        def r_l(x):
            return (13.490) + (-5.437 * x) + (0.581 * x ** 2) # lmc

        # l_lim = [5.15, 5.7]
        # def r_l(x):
        #     return (38.976) + (-15.221*x) + (1.521*x**2) # gal

        # lm_lim = [4.14, 4.37]
        # def r_lm(x):
        #     return (157.341) + (-76.119*x) + (9.260*x**2) # gal
        #====================================================END========================================================

        kap = t_k_rho[1:, 0]
        t   = t_k_rho[0, 1:]
        rho2d = t_k_rho[1:, 1:]

        if l_or_lm == 'l':  # to account for different limits in mass and luminocity
            l = Physics.lm_to_l( Physics.logk_loglm(kap, True) )
            cropped = Math.crop_2d_table(Math.invet_to_ascending_xy(Math.combine(t, l, rho2d)),
                                         None, None, l_lim[0], l_lim[-1])

            l_lm = cropped[1:, 0]
            t = cropped[0, 1:]
            rho2d = cropped[1:, 1:]

            vrho = Physics.get_vrho(t, rho2d, 2)
            m_dot = Physics.vrho_mdot(vrho, r_l(l_lm), 'l')  # r_s = given by the func

        else:
            lm = Physics.logk_loglm(kap, True)
            cropped = Math.crop_2d_table(Math.invet_to_ascending_xy(Math.combine(t, lm, rho2d)),
                                         None, None, lm_lim[0], lm_lim[-1])


            l_lm = cropped[1:, 0]
            t    = cropped[0, 1:]
            rho2d = cropped[1:, 1:]

            vrho = Physics.get_vrho(t, rho2d, 2)
            m_dot = Physics.vrho_mdot(vrho, r_lm(l_lm), 'l')  # r_s = given by the func


        return Math.combine(t, l_lm, m_dot)

        # return (40.843) + (-15.943*x) + (1.591*x**2)                    # FROM GREY ATMOSPHERE ESTIMATES

        # return -859.098 + 489.056*x - 92.827*x**2 + 5.882*x**3        # FROM SONIC POINT ESTIMATES

    def plot_t_l_mdot(self, l_or_lm, rs, plot_obs, plot_nums, use_r_tl = False, lim_t1 = None, lim_t2 = None):

        # ---------------------LOADING-INTERPOLATED-TABLE---------------------------

        t_k_rho = Save_Load_tables.load_table('t_k_rho', 't', 'k', 'rho', self.opal_used, self.output_dir)

        #---------------------Getting KAPPA[], T[], RHO2D[]-------------------------

        kap   = t_k_rho[1:, 0]
        t   =   t_k_rho[0, 1:]
        rho2d = t_k_rho[1:, 1:]



        if l_or_lm == 'l':
            l_lm_arr  = Physics.lm_to_l( Physics.logk_loglm(kap, True) ) # Kappa -> L/M -> L

            l_lm_arr = l_lm_arr[::-1]
            rho2d = rho2d[::-1, :]

        else:
            l_lm_arr = Physics.logk_loglm(kap, 1)

        vrho = Physics.get_vrho(t,rho2d, 2)          # mu = 1.34 everywhere

        if use_r_tl:
            t_l_or_lm_r, t_llm_vrho = self.sp_get_r_lt_table(l_or_lm, 1000, True, Math.combine(t, l_lm_arr, vrho))

            rs = t_l_or_lm_r[1:,1:]
            t = t_l_or_lm_r[0,1:]
            l_lm_arr = t_l_or_lm_r[1:,0]

            vrho = t_llm_vrho[1:,1:]

            m_dot = Physics.vrho_mdot(vrho, rs, 'tl')

        else:
            m_dot = Physics.vrho_mdot(vrho, rs, '')      # r_s = constant

        print('\t__Note: PLOT: x: {}, y: {}, z: {} shapes.'.format(t.shape, l_lm_arr.shape, m_dot.shape))

        #-------------------------------------------POLT-Ts-LM-MODT-COUTUR------------------------------------

        name = self.plot_dir + 'rs_lm_minMdot_plot.pdf'

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.xlim(t.min(), t.max())
        plt.ylim(l_lm_arr.min(), l_lm_arr.max())
        plt.ylabel(Labels.lbls(l_or_lm))
        plt.xlabel(Labels.lbls('ts'))
        levels = [-7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2]
        contour_filled = plt.contourf(t, l_lm_arr, m_dot, levels, cmap=plt.get_cmap('RdYlBu_r'))
        plt.colorbar(contour_filled, label=Labels.lbls('mdot'))
        contour = plt.contour(t, l_lm_arr, m_dot, levels, colors='k')
        plt.clabel(contour, colors='k', fmt='%2.1f', fontsize=12)
        plt.title('SONIC HR DIAGRAM')


        # plt.ylabel(l_or_lm)
        plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
        plt.savefig(name)

        #--------------------------------------------------PLOT-MINS----------------------------------------------------

        # plt.plot(mins[0, :], mins[1, :], '-.', color='red', label='min_Mdot (rs: {} )'.format(r_s_))

        #-----------------------------------------------PLOT-OBSERVABLES------------------------------------------------
        if plot_obs:
            classes = []
            classes.append('dum')
            x = []
            y = []
            for star_n in self.obs.stars_n:
                xyz = self.obs.get_xyz_from_yz(star_n, l_or_lm, 'mdot', t, l_lm_arr, m_dot, lim_t1, lim_t2)
                if xyz.any():
                    x = np.append(x, xyz[0, 0])
                    y = np.append(y, xyz[1, 0])
                    for i in range(len(xyz[0,:])):
                        plt.plot(xyz[0, i], xyz[1, i], marker=self.obs.get_clss_marker(star_n), markersize='9', color=self.obs.get_class_color(star_n), ls='')  # plot color dots)))
                        ax.annotate(int(star_n), xy=(xyz[0,i], xyz[1,i]),
                                    textcoords='data')  # plot numbers of stars
                        if self.obs.get_star_class(star_n) not in classes:
                            plt.plot(xyz[0, i], xyz[1, i], marker=self.obs.get_clss_marker(star_n), markersize='9', color=self.obs.get_class_color(star_n), ls='', label='{}'.format(self.obs.get_star_class(star_n)))  # plot color dots)))
                            classes.append(self.obs.get_star_class(star_n))

            # fit = np.polyfit(x, y, 1)  # fit = set of coeddicients (highest first)
            # f = np.poly1d(fit)
            # fit_x_coord = np.mgrid[(x.min()-1):(x.max()+1):1000j]
            # plt.plot(fit_x_coord, f(fit_x_coord), '-.', color='blue')

        #--------------------------------------------------_NUMERICALS--------------------------------------------------
        if plot_nums:
            for i in range(len(self.num_files)):
                ts_llm_mdot = self.mdl[i].get_xyz_from_yz(i, 'sp', l_or_lm, 'mdot', t , l_lm_arr, m_dot, lim_t1, lim_t2)
                # lbl1 = self.mdl[i].get_cond_value(num_var_plot, 'sp')

                if ts_llm_mdot.any():
                    # lbl = 'i:{}, lm:{}, {}:{}'.format(i, "%.2f" % ts_llm_mdot[2, -1], num_var_plot, "%.2f" % lbl1)
                    plt.plot(ts_llm_mdot[0, :], ts_llm_mdot[1,:], marker='x', color='C' + str(Math.get_0_to_max([i],9)[i]), ls='')  # plot color dots)))
                    ax.annotate(str("%.2f" % ts_llm_mdot[2, -1]), xy=(ts_llm_mdot[0, -1], ts_llm_mdot[1,-1]), textcoords='data')

            for i in range(len(self.num_files)):
                x_coord = self.mdl[i].get_cond_value('t', 'sp')
                y_coord = self.mdl[i].get_cond_value(l_or_lm, 'sp')
                # lbl1 = self.mdl[i].get_cond_value(num_var_plot, 'sp')
                # lbl2 = self.mdl[i].get_cond_value('He4', 'core')

                # lbl = 'i:{}, Yc:{}, {}:{}'.format(i, "%.2f" % lbl2, num_var_plot, "%.2f" % lbl1)
                plt.plot(x_coord, y_coord, marker='X', color='C' + str(Math.get_0_to_max([i], 9)[i]),
                         ls='')  # plot color dots)))
                ax.annotate(str(int(i)), xy=(x_coord, y_coord), textcoords='data')

        plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
        plt.gca().invert_xaxis()
        plt.savefig(name)
        plt.show()

    def min_mdot(self, l_or_lm, plot_obs, plot_nums, lim_t1, lim_t2):
        # ---------------------LOADING-INTERPOLATED-TABLE---------------------------

        t_k_rho = Save_Load_tables.load_table('t_k_rho', 't', 'k', 'rho', self.opal_used, self.output_dir)
        l_lim1, l_lim2 = None, None
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        def plot_emp_min_mdot(t_k_rho, l_or_lm):

            t_llm_mdot = Combine.empirical_l_r_crit(t_k_rho, l_or_lm)
            t = t_llm_mdot[0, 1:]
            l_lm_arr = t_llm_mdot[1:, 0]
            m_dot = t_llm_mdot[1:, 1:]

            mins = Math.get_mins_in_every_row(t, l_lm_arr, m_dot, 5000, lim_t1, lim_t2)

            plt.plot(mins[2, :], mins[1, :], '-', color='black')

            ax.fill_between(mins[2, :], mins[1, :], color="lightgray")

            l_lm_lim1 = l_lm_arr[0]
            l_lm_lim2 = l_lm_arr[-1]

            return l_lm_lim1, l_lm_lim2

        def plot_min_mdot_with_const_r(t_k_rho, l_or_lm, rs, l_lim1, l_lim2):

            kap = t_k_rho[1:, 0]
            t = t_k_rho[0, 1:]
            rho2d = t_k_rho[1:, 1:]

            if l_or_lm == 'l':
                l_lm = Physics.lm_to_l(Physics.logk_loglm(kap, True))
            else:
                l_lm = Physics.logk_loglm(kap, True)



            cropped = Math.crop_2d_table( Math.invet_to_ascending_xy( Math.combine(t, l_lm, rho2d) ),  None, None, l_lim1, l_lim2)

            print(l_lim1, l_lim2, cropped.shape)

            l_lm = cropped[1:,0]
            t = cropped[0, 1:]
            rho2d = cropped[1:, 1:]

            vrho_ = Physics.get_vrho(t, rho2d, 2)  # mu = 1.34 everywhere
            m_dot_ = Physics.vrho_mdot(vrho_, rs, '')  # r_s = constant

            print('x:', t.shape, l_lm.shape, m_dot_.shape)
            mins_ = Math.get_mins_in_every_row(t, l_lm, m_dot_, 5000, lim_t1, lim_t2)

            plt.plot(mins_[2, :], mins_[1, :], '-', color='C'+str(int(rs)), label='rs:{}'.format(rs))

        #----------------------------------------------PLOT MIN MDOT----------------------------------------------------

        l_lim1, l_lim2 = plot_emp_min_mdot(t_k_rho, l_or_lm)

        plot_min_mdot_with_const_r(t_k_rho, l_or_lm, 1.0, l_lim1, l_lim2)

        plot_min_mdot_with_const_r(t_k_rho, l_or_lm, 2.0, l_lim1, l_lim2)

        # -----------------------------------------------PLOT-OBSERVABLES-----------------------------------------------

        if plot_obs:

            classes = []
            classes.append('dum')
            x = []
            y = []

            for star_n in self.obs.stars_n:
                i=-1
                x = np.append(x, self.obs.get_num_par('mdot',  star_n))
                y = np.append(y, self.obs.get_num_par(l_or_lm, star_n))

                plt.plot(x[i], y[i], marker=self.obs.get_clss_marker(star_n), markersize='9',
                         color=self.obs.get_class_color(star_n), ls='')  # plot color dots)))
                ax.annotate(int(star_n), xy=(x[i], y[i]),
                            textcoords='data')  # plot numbers of stars
                if self.obs.get_star_class(star_n) not in classes:
                    plt.plot(x[i], y[i], marker=self.obs.get_clss_marker(star_n), markersize='9',
                             color=self.obs.get_class_color(star_n), ls='',
                             label='{}'.format(self.obs.get_star_class(star_n)))  # plot color dots)))
                    classes.append(self.obs.get_star_class(star_n))

            fit = np.polyfit(x, y, 1)  # fit = set of coeddicients (highest first)
            f = np.poly1d(fit)
            fit_x_coord = np.mgrid[(x.min() - 1):(x.max() + 1):1000j]
            plt.plot(fit_x_coord, f(fit_x_coord), '-.', color='blue')

        # --------------------------------------------------NUMERICALS--------------------------------------------------

        if plot_nums:
            for i in range(len(self.num_files)):
                x_coord = self.mdl[i].get_cond_value('mdot', 'sp')
                y_coord = self.mdl[i].get_cond_value(l_or_lm, 'sp')
                # lbl1 = self.mdl[i].get_cond_value(i, 'sp')
                # lbl2 = self.mdl[i].get_cond_value('He4', 'core')

                # lbl = 'i:{}, Yc:{}, {}:{}'.format(i, "%.2f" % lbl2, num_var_plot, "%.2f" % lbl1)
                plt.plot(x_coord, y_coord, marker='x', color='C' + str(Math.get_0_to_max([i], 9)[i]),
                         ls='')  # plot color dots)))
                ax.annotate(str(int(i)), xy=(x_coord, y_coord),
                            textcoords='data')



        # plt.ylim(y.min(),y.max())

        # plt.xlim(-6.0, mins[2,:].max())

        plt.ylabel(Labels.lbls(l_or_lm))
        plt.xlabel(Labels.lbls('mdot'))
        ax.grid(which='major', alpha=0.2)
        plt.legend(bbox_to_anchor=(1, 1), loc='upper right', ncol=1)

        ax.grid(which='both')
        ax.grid(which='minor', alpha=0.2)
        plt.legend(bbox_to_anchor=(1, 0), loc='lower right', ncol=1)
        plot_name = self.plot_dir + 'minMdot_l.pdf'
        plt.savefig(plot_name)
        plt.show()

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
    def __init__(self, spfiles, out_dir, plot_dir):

        self.spfiles = spfiles

        self.spmdl = []
        for file in spfiles:
            self.spmdl.append( Read_SP_data_file(file, out_dir, plot_dir) )

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

    def sp_3d_and_multiplot(self, v_n1, v_n2, v_n3, v_n_col):

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

        for i in range(len(self.spfiles)):

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

            for j in range(n_of_rows):
                if self.spmdl[i].get_sonic_cols('r')[j] > 0.:  # selecting only the solutions with found rs
                    x = np.append(x, self.spmdl[i].get_sonic_cols(v_n1)[j])
                    y = np.append(y, self.spmdl[i].get_sonic_cols(v_n2)[j])
                    z = np.append(z, self.spmdl[i].get_sonic_cols(v_n3)[j])
                    t = np.append(t, self.spmdl[i].get_sonic_cols(v_n_col)[j])

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