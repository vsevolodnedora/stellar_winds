# from __future__ import division
import sys
import pylab as pl
from matplotlib import cm
import numpy as np
from scipy import interpolate
import scipy.ndimage
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from os import listdir

import os
from Plots_Obs_treat import Errors
from Plots_Obs_treat import Read_Table
from Plots_Obs_treat import Row_Analyze
from Plots_Obs_treat import Table_Analyze
from Plots_Obs_treat import Math
from Plots_Obs_treat import Physics
from Plots_Obs_treat import PhysPlots
from Plots_Obs_treat import OPAL_Interpol
from Plots_Obs_treat import Constants
from Plots_Obs_treat import Read_SM_data_file
from Plots_Obs_treat import ClassPlots
from Plots_Obs_treat import Read_Observables
from Plots_Obs_treat import New_Table
from Plots_Obs_treat import Read_Plot_file
from Plots_Obs_treat import Treat_Observables
from main_methods import Combine

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
                    print('\t__Note: Files to be plotted: ', dir_ + file)
                    comb.append(compath + dir_ + file)
                else:
                    for req_file in requir_files:
                        if req_file+extension == file:
                            print('\t__Note: Files to be plotted: ', dir_+file)
                            comb.append(compath+dir_+file)

    return comb

output_dir  = '../data/output/'
plot_dir    = '../data/plots/'
opal_fl     = '../data/opal/table8.data'
obs_fl      = '../data/obs/gal_wn.data'

smfiles = get_files('../../sse/ga_z002/', ['15sm/y1/'], [], 'sm.data')
#'20sm/y10/'

plotfls = get_files('../../sse/ga_z002/', [], [], '.plot1')
#'10sm/','11sm/','12sm/','13sm/','14sm/','15sm/', '16sm/', '17sm/', '18sm/', '19sm/', '20sm/'

spfiles = get_files('../data/output/',    ['criticals/ZAMSz002/'], [], '.data')

'''===========================================GRAY=ATMPOSPHERE=ANALYSYS=============================================='''
#
from analyze_r_crit import Critical_R
cr = Critical_R(smfiles, output_dir+'criticals/15z002/', plot_dir, ['sse', 'ga_z002']) # [] is a listof folders not to be put in output name
cr.sonic_criticals(1000, ['kappa-sp', 'L/Ledd-sp', 'rho-sp'])
# # # # # # # cr.velocity_profile()

# from Read_Obs_Numers import Read_SP_data_file
# x = Read_SP_data_file(spfiles, output_dir, plot_dir)

'''======================================================TEST========================================================'''

# from main_methods import TEST
# tst = TEST(output_dir)
# tst.xy_last_points('l','r','He4', 'core', [smfiles1,smfiles2,smfiles3,smfiles4])
# tst.d3_plotting_x_y_z('l','r','mdot','He4', 'core', [smfiles1,smfiles2,smfiles3,smfiles4])
# tst.new_3d()

'''=====================================================CREATION====================================================='''

# from main_methods import Creation
# make = Creation(opal_fl, 4.9, 5.5, 1000)
# # make.save_t_rho_k()
# make.save_t_k_rho(3.8, None, 1000)
# # file_table = np.zeros(1)
# # make.read_table('t_k_rho', 't', 'k', 'rho', '../data/opal/table8.data')
# make.save_t_llm_vrho('l')
# # print(file_table)
# make.save_t_llm_mdot(1.,'l','',1.34)
#
# a = np.array(([1,1,1,1], [5,5,5,5], [7,7,7,7], [9,9,9,9]))
# a = np.flip(a, 0)
# print(a)

'''====================================================MAIN=METHODS=================================================='''

# comb = Combine(smfiles, spfiles, plotfls, obs_fl, opal_fl)

# comb.sp_xy_last_points('l','r','mdot',True)
# comb.sp_get_r_lt_table('lm')
# comb.xy_profile('xm','H','mdot','xm')
# comb.xyy_profile('r','rho','kappa','mdot','xm','t', False)
# comb.xy_last_points('r','l','mdot',True)
# comb.hrd(['../../sse/ev30z0008_H/ev.plot1'])

# comb.plot_t_rho_kappa('mdot','xm')
# comb.plot_t_mdot_lm()
# comb.plot_t_l_mdot('l', 1.0, True, False, True, 5.22, None)
# comb.min_mdot('lm', True, False, 5.2, None)

'''===========================================================3D====================================================='''
#
from main_methods import TEST
tst = TEST(spfiles, output_dir, plot_dir)
# tst.sp_3d_plotting_x_y_z('t','l','r','mdot')
# tst.sp_3d_and_multiplot('t','l','r','mdot')

'''==========================================================REST===================================================='''

# from __future__ import division
# import numpy as np
# import matplotlib.pyplot as plt

# def interpolated_intercept(x, y1, y2):
#     """Find the intercept of two curves, given by the same x data"""
#
#     def intercept(point1, point2, point3, point4):
#         """find the intersection between two lines
#         the first line is defined by the line between point1 and point2
#         the first line is defined by the line between point3 and point4
#         each point is an (x,y) tuple.
#
#         So, for example, you can find the intersection between
#         intercept((0,0), (1,1), (0,1), (1,0)) = (0.5, 0.5)
#
#         Returns: the intercept, in (x,y) format
#         """
#
#         def line(p1, p2):
#             A = (p1[1] - p2[1])
#             B = (p2[0] - p1[0])
#             C = (p1[0]*p2[1] - p2[0]*p1[1])
#             return A, B, -C
#
#         def intersection(L1, L2):
#             D  = L1[0] * L2[1] - L1[1] * L2[0]
#             Dx = L1[2] * L2[1] - L1[1] * L2[2]
#             Dy = L1[0] * L2[2] - L1[2] * L2[0]
#
#             x = Dx / D
#             y = Dy / D
#             return x,y
#
#         L1 = line([point1[0],point1[1]], [point2[0],point2[1]])
#         L2 = line([point3[0],point3[1]], [point4[0],point4[1]])
#
#         R = intersection(L1, L2)
#
#         return R
#
#     idx = np.argwhere(np.diff(np.sign(y1 - y2)) != 0)
#     xc, yc = intercept((x[idx], y1[idx]),((x[idx+1], y1[idx+1])), ((x[idx], y2[idx])), ((x[idx+1], y2[idx+1])))
#     return xc,yc
#
# def main():
#     x  = np.linspace(1, 4, 20)
#     y1 = np.sin(x)
#     y2 = 0.05*x
#
#     plt.plot(x, y1, marker='o', mec='none', ms=4, lw=1, label='y1')
#     plt.plot(x, y2, marker='o', mec='none', ms=4, lw=1, label='y2')
#
#     idx = np.argwhere(np.diff(np.sign(y1 - y2)) != 0)
#
#     plt.plot(x[idx], y1[idx], 'ms', ms=7, label='Nearest data-point method')
#
#     # new method!
#     xc, yc = interpolated_intercept(x,y1,y2)
#     plt.plot(xc, yc, 'co', ms=5, label='Nearest data-point, with linear interpolation')
#
#
#     plt.legend(frameon=False, fontsize=10, numpoints=1, loc='lower left')
#
#     plt.savefig('curve crossing.png', dpi=200)
#     plt.show()
#
# if __name__ == '__main__':
#     main()


# # use splines to fit and interpolate data
# from scipy.interpolate import interp1d
# from scipy.optimize import fmin
#
#
# x = np.array([ 0.,      1.,      2.,      3.,      4.    ])
# y = np.array([ 0.,     0.308,  0.55,   0.546,  0.44 ])
#
# # create the interpolating function
# f = interp1d(x, y, kind='cubic', bounds_error=False)
#
# # to find the maximum, we minimize the negative of the function. We
# # cannot just multiply f by -1, so we create a new function here.
# f2 = interp1d(x, -y, kind='cubic')
# guess = x[np.where(y == y.max())]
# xmax = fmin(f2, guess)
#
# xfit = np.linspace(0,4)
#
# plt.plot(x,y,'bo')
# plt.plot(xfit, f(xfit),'r-')
# plt.plot(xmax, f(xmax),'g*')
# plt.legend(['data','fit','max'], loc='best', numpoints=1)
# plt.xlabel('x data')
# plt.ylabel('y data')
# plt.title('Max point = ({0:1.2f}, {1:1.2f})'.format(float(xmax),
#                                                     float(f(xmax))))
# plt.show()


# x = np.array([1,2,3,4,5])
# y = np.array([3,5,5,5,5])
# z= np.array([5,6,6,6,6])
# xy = np.vstack((x,y,z))
# print(xy)
# print(xy[::-1])
# print('--------------')
# print(x[1:])
# x = np.array([0,,4,4,4])
# y = np.array([3,5,5,5,5])


# fig, ax = plt.subplots()
#
# cax = ax.imshow(P_min)#, norm=matplotlib.colors.LogNorm())#, cmap=cm.coolwarm)
# #plt.matshow(bla, norm=matplotlib.colors.LogNorm())
# cbar=fig.colorbar(cax, label='P / days')
# #cbar.ax.set_yticklabels(['','1','2','','','5','','','','','10','20','','','50','','','','90','100'])
# ax.set_title("minimal Period")
# ax.set_xlabel("M$_\mathrm{He}$ / M$_{\odot}$")
# ax.set_ylabel("M$_\mathrm{MS}$ / M$_{\odot}$")
# plt.xticks(arange(8), (2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5), size=8)
# plt.yticks(arange(13), (1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,7.5,10.0,12.5,15.0), size=8)
# ax.invert_yaxis()
#
# for i in range(0,8):
#     for j in range(0,13):
#         c = round(P_min[j][i],2)
#         ax.text(i, j, str(c), va='center', ha='center', size=6)
#
# plt.show()
# #plt.savefig('min_P.png', dpi=1000)