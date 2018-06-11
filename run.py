# 1


# import sys
# import pylab as pl
# from matplotlib import cm
import numpy as np
# from scipy import interpolate
# import scipy.ndimage
# from scipy.interpolate import interp1d
# import matplotlib.pyplot as plt
# from scipy.interpolate import griddata
# from os import listdir
import subprocess
import sys
import pylab as pl
from matplotlib import cm
# import numpy as np
from ply.ctokens import t_COMMENT
from scipy import interpolate
from os import listdir
from scipy import optimize
from sklearn.linear_model import LinearRegression
# import scipy.ndimage
# from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
# from scipy.interpolate import griddata
import os

'''===================================================m.dat=change==================================================='''
#
#
#
'''======================================================METHODS====================================================='''
class Constants:

    light_v = np.float( 2.99792458 * (10 ** 10) )      # cm/s
    solar_m = np.float ( 1.99 * (10 ** 33)  )          # g
    solar_l = np.float ( 3.9 * (10 ** 33)  )           # erg s^-1
    solar_r = np.float ( 6.96 * (10 ** 10) )           #cm
    grav_const = np.float ( 6.67259 * (10 ** (-8) )  ) # cm3 g^-1 s^-2
    k_b     =  np.float ( 1.380658 * (10 ** (-16) ) )  # erg k^-1
    m_H     =  np.float ( 1.6733 * (10 ** (-24) ) )    # g
    m_He    =  np.float ( 6.6464764 * (10 ** (-24) ) ) # g
    c_k_edd =  np.float ( 4 * light_v * np.pi * grav_const * ( solar_m / solar_l ) )# k = c_k_edd*(M/L) (if M and L in solar units)

    yr      = np.float( 31557600. )
    smperyear = np.float(solar_m / yr)

    steph_boltz = np.float(5.6704*10**(-5)) # erg cm−2 s−1 K−4.

    def __init__(self):
        pass

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

def change_value(rows, v_n, value):

    def change_val_in_row(row, v_n, value):

        # all_set = row.split()
        # if np.where(all_set==v_n)[0][0] == len(all_set)-1 or np.where(all_set==v_n)[0][0] == len(all_set)-2:


        # print(row.split('='))
        # if len(row.split(v_n)) > 2: raise NameError('There should be ony two parts... Found: {}'.format(row.split(' '+v_n)))

        # all_set = row.split()
        # position = -1


        position = -1
        if v_n in row:
            position = row.index(v_n)
        else:
            if ' '+v_n in row:
                position = row.index(' '+v_n)
                # position = np.int(np.where(row == ' '+v_n)[0])
            else:
                raise NameError('v_n {} not found in the row...'.format(v_n))



        if position == 0:
            if len(row.split(v_n)) > 2: raise NameError(
                'There should be ony two parts... Found: {}'.format(row.split(v_n)))
            beginning = row.split(v_n)[0]
            important = row.split(v_n)[-1]

        else:
            if len(row.split(' ' + v_n)) > 2: raise NameError(
                'There should be ony two parts... Found: {}'.format(row.split(' ' + v_n)))
            beginning = row.split(' ' + v_n)[0]
            important = row.split(' ' + v_n)[-1]
            v_n = ' '+v_n


        # print('beginning:', beginning)
        # 'important' always contain the value to be cahnged as ' = oldval   'NEXT_V_N = ...'

        set = important.split()
        # print('important:', important)
        # print('set:', set)
        if set[0]!='=': raise NameError('There should be < = > at the beginning, instead of : {}'.format(set[0]))

        # print(set)
        old_value = set[1]

        if len(set) == 2:
            print('v_n {} is in the last column'.format(v_n))
            raise IOError('No Treatment written for last row (it is not difficult, I was just lazy')


        if len(set) > 2:
            print('v_n : {} not in the last column'.format(v_n))

            next_v_n = set[2]
            end_min_next_v_n = row.split(next_v_n)[-1]
            end_with_next_v_n = next_v_n+end_min_next_v_n
            print(end_with_next_v_n)

            stuff = important.split(old_value)[0]
            # print('stuff:', stuff)

            construct = beginning+v_n + stuff + value + end_with_next_v_n

            old_len = len(row)
            new_len = len(construct)
            # print('construct:', construct)

            if new_len > old_len: raise ValueError('You got too big row Old:{} New:{}'.format(old_len, new_len))
            blanks = ' '*(old_len - new_len)

            construct2 = beginning + v_n + stuff + value + blanks + end_with_next_v_n
            # print('construct2:', construct2)

            new_len2 = len(construct2)

            if new_len2 != old_len: raise ValueError('Rows must presurve length, instead you have: Old: {} New: {} Or\n '
                                                     '{}\n{}'.format(old_len,new_len2,row,construct2))

            return construct2

    new_lines = []
    for i in range(len(rows)):
        row = rows[i]
        if v_n in row.split() or v_n+'=' in row.split():

            # print(row)
            # raise IOError('blabla')
            new_lines.append( change_val_in_row(row, v_n, value) )

        else:
            new_lines.append(row)

    return new_lines

def modify_mdat_file(ref_mdat, v_n_string_array):
    f = open(ref_mdat, 'r').readlines()
    all_lines = []

    for line in f:
        all_lines.append(line)

    new_lines = all_lines

    if len(v_n_string_array) == 0: raise ValueError('WRONG')

    for v_n_val in v_n_string_array:
        if len(v_n_val.split('=')) > 2: raise NameError('Format is: NAME = value. Instead: {}'.format(v_n_val))
        new_lines = change_value(new_lines, v_n_val.split('=')[0], v_n_val.split('=')[-1])

    new_file_name = 'm.dat'
    g = open(new_file_name, 'w').writelines(new_lines)  # writing down the new file from modified rows

'''====================================================__SETTINGS__=================================================='''
# --- AVAILABLE MODES --- Chose when Initialize the Program ---
allowed_modes=['list', 'listlist', 'prescr', 'steps', 'allsteps', 'all']
print('Choose a Mode out of: {}'.format(allowed_modes))

    # Get Mdot from losts below
    # gen Mdot from prescription (N&L)
    # iterate, using previouse model
    # iterate over folders and do 'do_manual' in all of them
# -------------------------------------------------------------

all_mdot_files=['3.00', '3.10', '3.20', '3.30', '3.40', '3.50', '3.60', '3.70', '3.80', '3.90',
                '4.00', '4.10', '4.20', '4.30', '4.40', '4.50', '4.60', '4.70', '4.80', '4.90',
                '5.00', '5.10', '5.20', '5.30', '5.40', '5.50', '5.60', '5.70', '5.80', '5.90', '6.00']

#all_mdot_files=['3.00']
                # '3.05', '3.10', '3.15', '3.20', '3.25', '3.30', '3.35', '3.40', '3.45', '3.50', '3.55', '3.60', '3.65', '3.70', '3.75', '3.80', '3.85', '3.90', '3.95',
                # '4.00', '4.05', '4.10', '4.15', '4.20', '4.25', '4.30', '4.35', '4.40', '4.45', '4.50', '4.55', '4.60', '4.65', '4.70', '4.75', '4.80', '4.85', '4.90', '4.95',
                # '5.00', '5.05', '5.10', '5.15', '5.20', '5.25', '5.30', '5.35', '5.40', '5.45', '5.50', '5.55', '5.60', '5.65', '5.70', '5.75', '5.80', '5.85', '5.90', '5.95',
                # '6.00', '6.05', '6.10', '6.15', '6.20', '6.25', '6.30', '6.35', '6.40', '6.45', '6.50', '6.55', '6.60', '6.65', '6.70', '6.75', '6.80', '6.85', '6.90', '6.95'
                # ]

all_mdot_values =[3.00, 3.10, 3.20, 3.30, 3.40, 3.50, 3.60, 3.70, 3.80, 3.90,
               4.00, 4.10, 4.20, 4.30, 4.40, 4.50, 4.60, 4.70, 4.80, 4.90,
               5.00, 5.10, 5.20, 5.30, 5.40, 5.50, 5.60, 5.70, 5.80, 5.90, 6.00]
all_mdot_files = ['../'+mdot_file for mdot_file in all_mdot_files]
# all_mdot_values =[3.00]
                  # 3.05, 3.10, 3.15, 3.20, 3.25, 3.30, 3.35, 3.40, 3.45, 3.50, 3.55, 3.60, 3.65, 3.70, 3.75, 3.80, 3.85, 3.90, 3.95,
                  # 4.00, 4.05, 4.10, 4.15, 4.20, 4.25, 4.30, 4.35, 4.40, 4.45, 4.50, 4.55, 4.60, 4.65, 4.70, 4.75, 4.80, 4.85, 4.90, 4.95,
                  # 5.00, 5.05, 5.10, 5.15, 5.20, 5.25, 5.30, 5.35, 5.40, 5.45, 5.50, 5.55, 5.60, 5.65, 5.70, 5.75, 5.80, 5.85, 5.90, 5.95,
                  # 6.00, 6.05, 6.10, 6.15, 6.20, 6.25, 6.30, 6.35, 6.40, 6.45, 6.50, 6.55, 6.60, 6.65, 6.70, 6.75, 6.80, 6.85, 6.90, 6.95
                  # ]

            #  ['10sm/', '11sm/']
sm_dirs = ['12sm/', '13sm/', '14sm/', '15sm/', '16sm/', '17sm/', '18sm/',
           '19sm/', '20sm/', '21sm/', '22sm/', '23sm/', '24sm/', '25sm/', '26sm/', '27sm/'] #, '28sm/', '29sm/', '30sm/']

y_dirs = ['y10/sp/']
          # 'y9/', 'y8/', 'y7/', 'y6/', 'y5/', 'y4/', 'y3/', 'y2/', 'y1/']



main_dir = '/media/vnedora/HDD/sse/ga_z0008_2/'#dir before the '10sm/y10/'

tmp_sign = '_'




'''======================================================__INPUT__==================================================='''

mode =   input("Mode: ")
in_file= input(".bin1: ") # INPUT name of the file to be prcesses
mdot1  = input("mdot1: ")
mdot2  = input("mdot2: ")
step   = input("step: ")
maxzal_= input("maxzal: ")

ref_mdat = 'ref_m.dat'
var_name1 = 'FNAME'

# ------------------------------------------------
mdot_array = []

if mdot1 != '' and mdot2== '' and step == '':
    mdot_array = np.array([float(mdot1)])

if mdot1 != '' and mdot2 != '' and step == '':
    mdot_array = np.array([float(mdot1), float(mdot2)])

if mdot1 != '' and mdot2 != '' and step != '':
    mdot_array = np.arange(float(mdot1), float(mdot2), float(step))

if maxzal_ == '':
    maxzal = 1
else:
    maxzal = int(maxzal_)

# maxzal = 50

'''======================================================__MAIN__===================================================='''

def compute_one(mdot_val, file, ref_mdat, maxzal):

    mdot_str = "%.2f" % mdot_val

    ev_name = 'FNAME=_{}'.format(mdot_str)  # sae name as in passing to auto_ev2
    maxzal_mdat = 'MAXZAL={}'.format(maxzal)
    mdot_mdat = 'DMDT=' + '-' + str("%.3e" % (10 ** ((-1.) * mdot_val))).replace('e', 'd')

    modify_mdat_file(ref_mdat, [ev_name, mdot_mdat, 'MTU=00', maxzal_mdat])

    subprocess.call(['./auto_ev2.sh', file, '_{}'.format(mdot_str), str(maxzal)])

    smfile = get_files('', ['./'], ['_' + '_{}'.format(mdot_str)], 'sm.data')
    new_name = get_new_name_from_smfile(smfile)

    subprocess.call(['./auto_rename.sh', '_' + '_{}'.format(mdot_str), new_name])

    # import check
    print('<================== COMPUTED: File:{} Mdot:{} ============================>'.format(file, mdot_val))

def get_new_name_from_smfile(smfile):
    if len(smfile) > 1:  raise IOError('Extract_name_from_sm: More than 1 sm.data file found <{}>'.format(smfile))
    if len(smfile) == 0: return '_no_sm_data' #raise IOError('No sm.data file found <{}>'.format(smfile))
    smcls = Read_SM_data_file.from_sm_data_file(smfile[0])

    mdot = smcls.mdot_[-1]
    yc = smcls.He4_[0]

    # new_name = '{}_{}'.format("%.2f" % (-1*mdot), "%.2f" % yc)
    new_name = '{}'.format("%.2f" % (-1 * mdot))
    return new_name

def compute_for_list(list_mdots, in_files, ref_mdat, maxzal):
    '''
    Takes set of Mdots (to be used) ne per computed model, an initial file, to be used in EVERY
    computation as starting model and a reference m.dat file that will be modified in every iteration with new
    Mdot and new number of models

    :param list_files:
    :param list_mdots:
    :param in_file:
    :param ref_mdat:
    :return:
    '''
    if len(list_mdots) < 1: raise NameError(
        'Less then one value is passed {}.'.format(list_mdots))

    if len(in_files) == 1:
        in_file = in_files[0]
    else:
        if len(in_files)<len(list_mdots):raise ValueError('Size of list_mdots and in_files should be equial.')



    for i in range(len(list_mdots)):

        compute_one(mdot, file, ref_mdat, maxzal)

def compute_for_dirs_lists(sm_dirs, y_dirs, main_dir, all_mdot_values, ref_mdat, maxzal):
    '''
    Interates between sm_dirs and in every sm_dir, iterates between y_dirs. In every sm_dir/y_dir ir serts the
    os.chdir(this_dir) and computes the aut0(), which is automatically computes set of models for every Mdot value
    in, saving as 'all_m
    For the model name it usus the y_dir, which is 'y10 ro y1' and turns it into 'fy10.bin1 ...'

    :param sm_dirs:
    :param y_dirs:
    :param main_dir:
    :param manual_files:
    :param manual_mdats:
    :param ref_mdat:
    :return:
    '''

    for sm_folder in sm_dirs:
        for y_folder in y_dirs:

            print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<| DOING {} {} |>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
                  .format(sm_folder, y_folder))

            os.chdir(main_dir+sm_folder+y_folder)

            in_file = 'f' + y_folder.split('/')[0]  # fy10.bin1

            compute_for_list(all_mdot_values, in_file, ref_mdat, maxzal)

            print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<| DONE  {} {} |>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
                  .format(sm_folder, y_folder))

def compute_for_steps(in_file, mdot_array, ref_mdat, maxzal):
    last_computed = in_file

    for i in range(len(mdot_array)):
        mdot_str = "%.2f" % mdot_array[i]

        ev_name = 'FNAME=_{}'.format(mdot_str)  # sae name as in passing to auto_ev2
        maxzal_mdat = 'MAXZAL={}'.format(maxzal)
        mdot_mdat = 'DMDT=' + '-' + str("%.3e" % (10 ** ((-1.) * mdot_array[i]))).replace('e', 'd')

        modify_mdat_file(ref_mdat, [ev_name, mdot_mdat, 'MTU=00', maxzal_mdat])

        subprocess.call(['./auto_ev2.sh', last_computed, '_{}'.format(mdot_str), str(maxzal)])

        smfile = get_files('', ['./'], ['_' + '_{}'.format(mdot_str)], 'sm.data')
        new_name = get_new_name_from_smfile(smfile)

        subprocess.call(['./auto_rename.sh', '_' + '_{}'.format(mdot_str), new_name])

        last_computed = new_name

        # import check
        print('<================== COMPUTED: File:{} Mdot:{} ============================>'
              .format(last_computed, mdot_str))

if mode=='listlist':

    if len(all_mdot_values) != len(all_mdot_files):
        raise NameError('len(all_mdot_values){} != len(all_mdot_files){}'.format(len(all_mdot_values), len(all_mdot_files)))

    for i in range(len(all_mdot_files)):
        file = all_mdot_files[i]
        mdot = all_mdot_values[i]

        compute_one(mdot, file, ref_mdat, maxzal)


    exit('DONE')

if mode=='prescr':

    raise NameError('I AM NOT READY!')

    fname = 'FNAME=re_{}'.format(in_file)
    maxzal_mdat = 'MAXZAL={}'.format(maxzal)
    modify_mdat_file(ref_mdat, [fname, 'DMDT=0.0d-4', 'MTU=53', maxzal_mdat, 'DTIN=5.001d04'])

    subprocess.call(['./rd_rm.sh', in_file, str(1)])  # "bin_file_name: $1", "number_of_model: $2"
    smfile = get_files('', ['./'], ['re_'+in_file], 'sm.data')
    if len(smfile) > 1:  raise IOError('More than 1 sm.data file found for <{}> name'.format('re_' + in_file))
    if len(smfile) == 0: raise IOError('No sm.data file found for <{}> name'.format('re_' + in_file))

    smcls = Read_SM_data_file.from_sm_data_file(smfile[0])
    mdot = (-1.) * smcls.mdot_[-1]
    if mdot == np.inf: raise ValueError('NO mass loss rate in the {} model. USE only dynamic models computed with mass loss rate'.format(in_file))
    yc = smcls.He4_[0]
    new_name = '{}_{}'.format("%.2f" % mdot, "%.2f" % yc)

    subprocess.call(['./auto_wind.sh', in_file, new_name])  # "bin_file_name: $1" , "out_name: $2"

    import check
    print('<================== COMPUTED: Mdot:{} Yc:{} ============================>'.format(mdot, yc))
    print('DONE!')

if mode=='steps':

    compute_for_steps(in_file, mdot_array, ref_mdat, maxzal)
    print('DONE!')

if mode=='allsteps':

    for sm_folder in sm_dirs:
        for y_folder in y_dirs:

            print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<| DOING {} {} |>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
                  .format(sm_folder, y_folder))

            os.chdir(main_dir+sm_folder+y_folder)

            # in_file = 'f' + y_folder.split('/')[0]  # fy10.bin1

            compute_for_steps(in_file, mdot_array, ref_mdat, maxzal)

            print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<| DONE  {} {} |>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
                  .format(sm_folder, y_folder))

if mode=='all':

    compute_for_dirs_lists(sm_dirs, y_dirs, main_dir, all_mdot_values, ref_mdat, maxzal)

if mode=='':



    if  len(mdot_array) < 1:
        raise NameError('len(mdot_array){} < 1'.format(len(mdot_array)))

    file = in_file

    for i in range(len(mdot_array)):
        mdot = mdot_array[i]

        compute_one(mdot, file, ref_mdat, maxzal)


print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< MODE {} DONE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
          .format(mode))





# -- Reading the m.dat file --- (reference one)
# f = open(ref_mdat, 'r').readlines()
# all_lines = []
#
# for line in f:
#     all_lines.append(line)
#
# new_lines = create_mdat_file(all_lines, 'FNAME', 're_{}'.format(in_file)) # new file with changed parameters
# new_lines = create_mdat_file(new_lines, 'DMDT', '0.0d-4')       # new file with changed parameters
# new_lines = create_mdat_file(new_lines, 'MTU', '53')            # new file with changed parameters
# new_lines = create_mdat_file(new_lines, 'MAXZAL', '01')         # new file with changed parameters
# new_lines = create_mdat_file(new_lines, 'DTIN', '5.001d06')     # new file with changed parameters
#
# new_file_name = 'm.dat'
# g = open(new_file_name, 'w').writelines(new_lines) # writing down the new file from modified rows






