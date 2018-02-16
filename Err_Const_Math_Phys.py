#-----------------------------------------------------------------------------------------------------------------------
# Set of classes including:
#   Constants   Set of constants for inverting cgs units into solar mass/lum/radii or back.
#   Errors      (contain possible errors that can be raised, - useless after discovery of raise ValError
#   Math        Contains the important mathematical formulas and tools, including interpolation routines
#   Physics     Includes technics to work with phys. quantites
#-----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------MAIN-LIBRARIES-----------------------------------------------------

import sys
# import pylab as pl
# from matplotlib import cm
import numpy as np
# from ply.ctokens import t_COMMENT
from scipy import interpolate
from sklearn.linear_model import LinearRegression
from scipy.optimize import fmin

# import scipy.ndimage
# from scipy.interpolate import interp1d
# import matplotlib.pyplot as plt
# from scipy.interpolate import griddata
# import os
#-----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------CLASSES-----------------------------------------------------
class Errors:

    '''
        Class contains set of static methods, that print error
        messeges or/and terminates the program.

    '''

    begin = '\t___Error! '

    def __init__(self):
        pass

    @staticmethod
    def is_arr_empty(arr, place, stop=False, message = None, values = None):
        if not any(arr):
            print(Errors.begin, 'In (', place, ') empty array', message, values)
            if stop:
                sys.exit('|\tError! Empty Array.')


    @staticmethod
    def is_arr_eq(arr1, arr2, place, stop=False, message = None, values = None):
        res = False
        if arr1.shape != arr2.shape:
            res = True
            print(Errors.begin, 'In (', place, ') shape of arrays are different', message, values)
            if stop:
                sys.exit('|\tError! Different shape of arrays.')
        return res

    @staticmethod
    def is_arr_shape(arr, shape, place, stop=False, message = None, values = None):
        res = False
        arr = np.array(arr)
        if arr.shape!=shape :
            res = True
            print(Errors.begin, 'In (', place, ') empty array', message, values)
            if stop:
                sys.exit('|\tError! Empty Array.')
        return res

    @staticmethod
    def is_more_1(arr, place, stop=False, message = None, values = None):
        res = False
        if len(arr) > 1:
            res = True
            print(Errors.begin, 'In (', place, ') >1 solution', message, values)
            if stop:
                sys.exit('|\tError! More than one solution found.')
        return res

    @staticmethod
    def is_more_2(arr, place, stop=False, message = None, values = None):
        res = False
        if len(arr) > 2:
            res = True
            print(Errors.begin, 'In (', place, ') >2 solution', message, values)
            if stop:
                sys.exit('|\tError! More than two solution found.')
        return res

    @staticmethod
    def is_arrsize_eq_m(arr, val, place, stop=False, message = None, values = None):
        res = False
        if len(arr) != val:
            res = True
            print(Errors.begin, 'In (', place, ') size of array != value', message, values)
            if stop:
                sys.exit('|\tError! Size of the array != value.')
        return res

    @staticmethod
    def is_a_bigger_b(a, b, place, stop=False, message = None, values = None):
        res = False
        if(a >= b):
            res = True
            print(Errors.begin, 'In (', place, ') lower_lim > upper_lim', message, values)
            if stop:
                sys.exit('|\tError! lower_lim > upper_lim')
        return res

    @staticmethod
    def custom(place, txt1, val1=None, txt2=None, val2=None, txt3=None, val3=None, stop=False):
        print(Errors.begin, 'In (', place, ')', txt1, val1, txt2, val2, txt3, val3)
        if stop:
            sys.exit('|\tError! Custom Abort')

class Constants:

    light_v = float( 2.99792458 * (10 ** 10) )      # cm/s
    solar_m = float ( 1.99 * (10 ** 33)  )         # g
    solar_l = float ( 3.9 * (10 ** 33)  )         # erg s^-1
    solar_r = float ( 6.96 * (10 ** 10) )          #cm
    grav_const = float ( 6.67259 * (10 ** (-8) )  ) # cm3 g^-1 s^-2
    k_b     =  float ( 1.380658 * (10 ** (-16) ) )  # erg k^-1
    m_H     =  float ( 1.6733 * (10 ** (-24) ) )    # g
    c_k_edd =  float ( 4 * light_v * np.pi * grav_const * ( solar_m / solar_l ) )# k = c_k_edd*(M/L) (if M and L in solar units)

    yr      = float( 31557600. )
    smperyear = float(solar_m / yr)

    def __init__(self):
        pass

class Math:
    def __init__(self):
        pass

    @staticmethod
    def find_nearest_index(array, value):
        ''' Finds index of the value in the array that is the closest to the provided one '''
        idx = (np.abs(array - value)).argmin()
        return idx

    @staticmethod
    def clrs(arr):
        colors = []
        for i in range(len(arr)):
            colors.append('C'+str(i))
        return colors

    @staticmethod
    def combine(x, y, xy):
        '''creates a 2d array  1st raw    [0, 1:]-- x -- density      (log)
                               1st column [1:, 0] -- y -- lemperature (log)
                               Matrix     [1:,1:] -- xy --Opacity     (log)
           0th element in 1st raw (column) - meaningless - to be cut (as above)
        '''
        x = np.array(x)
        y = np.array(y)
        xy = np.array((xy))

        Errors.is_arr_eq(x, y, 'Math | combine', False, 'x!=y')
        Errors.is_arr_eq(x, xy[0, :], 'Math | combine', True, 'x!=xy[0, :]')
        Errors.is_arr_eq(y, xy[:, 0], 'Math | combine', True, 'y!=xy[:, 0]')

        res = np.insert(xy, 0, x, axis=0)
        new_y = np.insert(y, 0, 0, axis=0)  # inserting a 0 to a first column of a
        res = np.insert(res, 0, new_y, axis=1)

        return res

    @staticmethod
    def minmax(first_arr, second_arr):
        ''' Returns the upper limit from the first array and the lower from the second'''
        return np.array([first_arr.max(), second_arr.min()])

    @staticmethod
    def solv_inter_row(arr_x, arr_y, val):
        '''
        FUNNY! but if val == arr_y[i] exactly, the f.roots() return no solution for some reaon :)
        :param arr_x:
        :param arr_y:
        :param val:
        :return:
        '''
        if arr_x.shape != arr_y.shape:
            print("y_arr:({} to {}), x_arr: ({} to {}) find: y_val {} ."
                  .format("%.2f"%arr_y[0],"%.2f"%arr_y[-1], "%.2f"%arr_x[0], "%.2f"%arr_x[-1],"%.2f"%val))
            raise ValueError

        if val in arr_y:
            return np.array(arr_x[np.where(arr_y == val)])
        else:
            # Errors.is_a_bigger_b(val,arr_y[-1],"|solv_inter_row|", True, "y_arr:({} to {}), can't find find: y_val {} .".format("%.2f"%arr_y[0],"%.2f"%arr_y[-1],"%.2f"%val))
            red_arr_y = arr_y - val

            # new_x = np.mgrid[arr_x[0]:arr_x[-1]:1000j]
            # f1 = interpolate.UnivariateSpline(arr_x, red_arr_y, s=0)
            # new_y = f1(new_x)

            # f = interpolate.InterpolatedUnivariateSpline(new_x, new_y)

            # new_y = f1(new_x)
            # f = interpolate.UnivariateSpline(new_x, new_y, s=0)
            # print("y_arr:({} to {}), x_arr: ({} to {}) find: y_val {} ."
            #       .format("%.2f"%arr_y[0],"%.2f"%arr_y[-1], "%.2f"%arr_x[0], "%.2f"%arr_x[-1],"%.2f"%val))

            f = interpolate.InterpolatedUnivariateSpline(arr_x, red_arr_y)
            # print("y_arr:({} to {}), can't find find: y_val {} .".format("%.2f"%arr_y[0],"%.2f"%arr_y[-1],"%.2f"%val))
            # f = interpolate.UnivariateSpline(arr_x, red_arr_y, s = 0)
            return f.roots()

    @staticmethod
    def interp_row(x_arr, y_arr, new_x_arr):
        '''
            Uses 1d spline interpolation to give set of values new_y for provided
            cooednates x and y and new coordinates x_new (s - to be 0)
        '''
        f = interpolate.InterpolatedUnivariateSpline(x_arr, y_arr)

        return f(new_x_arr)

    @staticmethod
    def get_largest_rising_by_1_area(ints_arr):
        #
        df = np.diff(ints_arr)
        #
        # c = [i for i in range(len(b)) if b[i] == 1]
        # d = np.diff(c)
        # print(b)
        # print(c)
        # print(d)

        n = []
        for i in range(len(df)):
            if df[i] != 1:
                n = np.append(n, int(i))

        n = np.append(n, len(df))
        d = np.diff(n)
        if d.any():
            return ints_arr[int(n[d.argmax()]) + 1: int(n[d.argmax() + 1]) + 1]
        else:
            return ints_arr

    @staticmethod
    def get_mins_in_every_row(x_arr, y_arr, z2d_arr, depth, from_ = None, to_ = None):
        '''
        Finds a min. value in every row of z2darr
        :param x_arr: usually temp
        :param y_arr: usually L/M
        :param z2d_arr: usually Mdot 2d array (watch for .T)
        :param depth: usually 2000-5000
        :param from_: beginnignof a dip you are studying
        :param to_: end of a dip you are studying
        :return: x_points, y_arr, values
        '''
        x_points = np.zeros(len(z2d_arr[:, 0]))
        values = np.zeros(len(z2d_arr[:, 0]))


        x1 = x_arr[ 0] # default values (all the x range)
        x2 = x_arr[-1]

        if from_ != None:
            x1 = from_

        if to_ != None:
            x2 = to_

        if x1 < x_arr[0]:
            sys.exit('\t__Error. *from_* is too small. {} < {} '.format(x1, x_arr[0]))

        if x2 > x_arr[-1]:
            sys.exit('\t__Error. *from_* is too big. {} > {} '.format(x2, x_arr[-1]))

        new_x = np.mgrid[x1:x2:depth*1j]

        for i in range(len(z2d_arr[:, 0])):

            new_y = Math.interp_row(x_arr, z2d_arr[i, :], new_x)
            x_points[i] = new_x[ new_y.argmin() ]
            values[i] = new_y.min()

        return np.vstack((x_points, y_arr, values))

    @staticmethod
    def line_fit(x_cords, y_cords):

        model = LinearRegression(fit_intercept=True)

        model.fit(x_cords[:, np.newaxis], y_cords)

        xfit = np.linspace(-10, 10, 1000)
        yfit = model.predict(xfit[:, np.newaxis])

        # plt.scatter(x_cords, y_cords)
        # plt.plot(xfit, yfit)
        # plt.show()

        return np.vstack((xfit, yfit))

    @staticmethod
    def get_0_to_max(arr, max):
        if len(arr) == 1: # if ou supply only one element, it builds the array of ints up to that point
            arr = np.linspace(0, int(arr[0]), num=int(arr[0])+1, dtype=int) # +1 to get step of 1
            # print(arr)

        j = 0
        n = 1
        res = []
        for i in range(len(arr)):
            res.append(j)
            j = j + 1
            if arr[i] == n * max:
                j = 0
                n = n + 1

        return res

    @staticmethod
    def interpolated_intercept(x, y1, y2):
        """Find the intercept of two curves, given by the same x data"""

        def intercept(point1, point2, point3, point4):
            """find the intersection between two lines
            the first line is defined by the line between point1 and point2
            the first line is defined by the line between point3 and point4
            each point is an (x,y) tuple.

            So, for example, you can find the intersection between
            intercept((0,0), (1,1), (0,1), (1,0)) = (0.5, 0.5)

            Returns: the intercept, in (x,y) format
            """

            def line(p1, p2):
                A = (p1[1] - p2[1])
                B = (p2[0] - p1[0])
                C = (p1[0] * p2[1] - p2[0] * p1[1])
                return A, B, -C

            def intersection(L1, L2):
                D = L1[0] * L2[1] - L1[1] * L2[0]
                Dx = L1[2] * L2[1] - L1[1] * L2[2]
                Dy = L1[0] * L2[2] - L1[2] * L2[0]

                x = Dx / D
                y = Dy / D
                return x, y

            L1 = line([point1[0], point1[1]], [point2[0], point2[1]])
            L2 = line([point3[0], point3[1]], [point4[0], point4[1]])

            R = intersection(L1, L2)

            return R

        idx = np.argwhere(np.diff(np.sign(y1 - y2)) != 0)
        xc, yc = intercept((x[idx], y1[idx]), ((x[idx + 1], y1[idx + 1])), ((x[idx], y2[idx])),
                           ((x[idx + 1], y2[idx + 1])))
        return xc, yc

    @staticmethod
    def get_max_by_interpolating(x,y):

        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)

        # create the interpolating function
        f = interpolate.interp1d(x, y, kind='cubic', bounds_error=False)

        # to find the maximum, we minimize the negative of the function. We
        # cannot just multiply f by -1, so we create a new function here.
        f2 = interpolate.interp1d(x, -1.*y, kind='cubic', bounds_error=False)

        # x_new = np.mgrid[x[0]:x[-1]:1000j]
        # y_new = f2(x_new)
        # print(y_new)

        # print(x.min(), x.max())
        guess = x[np.where(y == y.max())]
        # print(guess)

        x_max = fmin(f2, guess)

        return x_max, f(x_max)

        # xfit = np.linspace(0, 4)



class Physics:
    def __init__(self):
        pass

    @staticmethod
    def get_rho(r_arr, t_arr):

        Errors.is_arr_empty(r_arr, 'Physics | get_rho', True, 'r array')
        Errors.is_arr_empty(t_arr, 'Physics | get_rho', True, 't array')

        cols = len(r_arr)  # 28
        raws = len(t_arr)  # 76

        rho = np.zeros((raws, cols))

        for i in range(raws):
            for j in range(cols):
                rho[i, j] = r_arr[j] + 3 * t_arr[i] - 18

        return rho

    @staticmethod
    def get_r(t,rho):
        # rho = r + 3 * t - 18
        # r = rho - 3 * t + 18
        return (rho - 3 * t + 18)

    @staticmethod
    def edd_opacity(m, l, arr = False):
        '''
            :m: - in solar masses,
            :l: is in log(l/l_solar).
            calculated by k_edd = 4.pi.c.G.(m/l) in cgs units
            :return: the value of Eddington opacity (log10)

            Test:   e1 = Physics.edd_opacity(10, 5.141)
                    print('e1:',e1, '10**e1:', 10**e1)
                    Result e1: -0.02625
        '''

        l = 10 ** (l) # inverse the logarithm
        if arr :
            res = np.zeros(len(m))
            for i in range(len(m)):
                res[i] = np.log10((Constants.c_k_edd * m[i] / l[i]))
            return res
        else:
            return np.log10((Constants.c_k_edd * m / l)) # returns log10 of opacity

    @staticmethod
    def loglm(log_l, m, array = False):
        '''
        For log_l = 5.141 and m = 10 loglm = 4.141
        :param log_l: log10(l/l_solar)
        :param m: m im M solar
        :return: log10(l/m) in l/m solar
        '''
        if array:
            res = np.zeros(len(log_l))
            for i in range(len(log_l)):
                res[i] = (np.log10((10**log_l[i])/m[i]))
            return res
        else:
            return(np.log10((10**log_l)/m))

    @staticmethod
    def loglm_logk(loglm, array = False):
        '''
        For log(l/m) = 4.141 -> log(k) = -0.026
        :param loglm:
        :return:
        '''
        if array:
            res = np.zeros(len(loglm))
            for i in range(len(loglm)):
                res[i] = np.log10(Constants.c_k_edd) + np.log10(1 / (10 ** loglm[i]))
            return res
        else:
            return np.log10(Constants.c_k_edd) + np.log10(1 / (10 ** loglm))

    @staticmethod
    def logk_loglm(logk, dimensions = 0):
        '''
        For logk = -0.026 -> log(l/m) = 4.141
        :param logk:
        :return:
        '''
        if dimensions == 1:
            res = np.zeros(len(logk))
            for i in range(len(logk)):
                res[i] = np.log10(1 / (10 ** logk[i])) + np.log10(Constants.c_k_edd)
            return res
        if dimensions == 0:
            return np.log10(1 / (10 ** logk)) + np.log10(Constants.c_k_edd)

        if dimensions == 2:
            res = np.zeros(( len(logk[:,0]), len(logk[0,:] )))
            for i in range(len(logk[:,0])):
                for j in range(len(logk[0,:])):
                    res[i,j] = np.log10(1 / (10 ** logk[i,j])) + np.log10(Constants.c_k_edd)
            return res

        else:
            sys.exit('\t__Error. Wrong number of dimensions. Use 0,1,2. Given: {}. | logk_loglm |'.format(dimensions))

    @staticmethod
    def sound_speed(t, mu, array = False):
        '''

        :param t_arr: log(t) array
        :param mu: mean molecular weight, by default 1.34 for pure He ionised matter
        :return: array of c_s (sonic vel) in cgs

        Test: print(Physics.sound_speed(5.2) / 100000) should be around 31
        '''

        if array:
            if len(mu)!= len(t):
                sys.exit('\t__Error. Mu and t must be arrays of the same size: (mu: {}; t: {})'.format(mu, t) )
            res = np.zeros(len(t))
            for i in range(len(t)):
                res[i] = (np.sqrt(Constants.k_b*(10**t[i]) / (mu[i] * Constants.m_H))) / 100000
            return res
        else:
            return (np.sqrt(Constants.k_b*(10**t) / (mu * Constants.m_H))) / 100000# t is given in log

    @staticmethod
    def get_k1_k2_from_llm1_llm2(t1, t2, l1, l2):
        lm1 = None
        if l1 != None:
            lm1 = Physics.l_to_lm(l1)
        lm2 = None
        if l2 != None:
            lm2 = Physics.l_to_lm(l2)

        if lm1 != None:
            k2 = Physics.loglm_logk(lm1)
        else:
            k2 = None
        if lm2 != None:
            k1 = Physics.loglm_logk(lm2)
        else:
            k1 = None

        print('\t__Provided LM limits ({}, {}), translated to L limits: ({}, {})'.format(lm1, lm2, l1, l2))
        print('\t__Provided T limits ({},{}), and kappa limits ({}, {})'.format(t1, t2, k1, k2))
        return [k1, k2]

    @staticmethod
    def vrho_formula(t,rho,mu):
        # assuming that mu is a constant!

        # c = np.log10(4 * 3.14 * ((Constants.solar_r) ** 2) / Constants.smperyear)
        # c2 = c + np.log10(r_s ** 2)
        c2 = 0
        return (rho + c2 + np.log10(Physics.sound_speed(t, mu, False) * 100000))

    @staticmethod
    def get_vrho(t, rho, dimensions  = 1, mu = np.array([1.34])):
        '''
        :param t:
        :param rho:
        :param dimensions:
        :param r_s:
        :param mu:
        :return:             THIS FUNCTION

                     |           rho*v                          |           Mdot
               L/M   |                                     L/M  |
             .*      |                              ->          |
        kappa        |                              ->      or  |
             *-.     |                                          |
                L    |                                      L   |
                     |____________________________              |________________________
                                                ts                                      ts
        '''

        if int(dimensions) == 0:
            return Physics.vrho_formula(t,rho,mu)

        if int(dimensions) == 1:
            res = np.zeros(len(t))
            for i in range(len(t)):
                res[i] = Physics.vrho_formula(t[i],rho[i],mu) # pissibility to add mu[i] if needed
            return res


        if int(dimensions) == 2:

            cols = len(rho[0, :])
            rows = len(rho[:, 0])
            m_dot = np.zeros((rows, cols))

            for i in range(rows):
                for j in range(cols):
                    m_dot[i,j] = Physics.vrho_formula(t[j], rho[i, j], mu)# + c + np.log10(Physics.sound_speed(t[j], mu, False)*100000))
            return m_dot
        else:
            raise ValueError(' Wrong number of dimensions. Use 0,1,2. Given: {} '.format(dimensions))

    @staticmethod
    def vrho_mdot(vrho, r_s, r_s_for_t_l_vrho):
        # if vrho is a float and r_s is a float - r_s_for_t_l_vrho = ''
        # if vrho is 1darray and r_s is a float - r_s_for_t_l_vrho = ''
        # if vrho is 2darray and r_s is a float - r_s_for_t_l_vrho = ''

        # if vrho is 1darray and r_s is a 1d array - r_s_for_t_l_vrho = '-'

        # if vrho is 2darray and r_s is a 1d array - r_s_for_t_l_vrho = 't' or 'l' to change columns or rows of vrho

        # if vrho is 2darray and r_s is a 2d array - r_s_for_t_l_vrho = 'vrho' to change columns and rows of vrho

        # r_s_for_t_l_vrho = '', 't', 'l', 'lm', 'vrho'

        # -------------------- --------------------- ----------------------------
        c = np.log10(4 * 3.14 * ((Constants.solar_r) ** 2) / Constants.smperyear)

        if r_s_for_t_l_vrho == '':            # vrho is a constant
            mdot = None
        else:
            if r_s_for_t_l_vrho == '-':       # vrho is a 1d array
                mdot = np.zeros(vrho.shape)
            else:
                mdot = np.zeros((vrho.shape)) #vrho is a 2d array


        if r_s_for_t_l_vrho == '':  # ------------------------REQUIRED r_s = float
            c2 = c + np.log10(r_s ** 2)
            mdot = vrho + c2

        if r_s_for_t_l_vrho == '-':
            if len(r_s)!=len(vrho): raise ValueError('len(r_s)={}!=len(vrho)={}'.format(len(r_s), len(vrho)))
            for i in range(len(vrho)):
                mdot[i] = vrho[i] + c + np.log10(r_s[i] ** 2)

        if r_s_for_t_l_vrho == 't' or r_s_for_t_l_vrho == 'ts':  # ---r_s = 1darray
            if len(r_s) != len(vrho[:, 0]): raise ValueError('len(r_s)={}!=len(vrho[:, 0])={}'.format(len(r_s), len(vrho[:, 0])))
            for i in range(len(vrho[:, 0])):
                mdot[i, :] = vrho[i, :] + c + np.log10(r_s[i] ** 2)

        if r_s_for_t_l_vrho == 'l' or r_s_for_t_l_vrho == 'lm':  # ---r_s = 1darray
            if len(r_s) != len(vrho[0, :]): raise ValueError('len(r_s)={}!=len(vrho[0, :])={}'.format(len(r_s), len(vrho[0, :])))
            for i in range(len(vrho[0, :])):
                mdot[:, i] = vrho[:, i] + c + np.log10(r_s[i] ** 2)

        if r_s_for_t_l_vrho == 'vrho':  # ---------------------REQUIRED r_s = 2darray
            if r_s.shape != vrho.shape: raise ValueError('r_s.shape {} != vrho.shape {}'.format(r_s.shape, vrho.shape))
            cols = len(vrho[0, :])
            rows = len(vrho[:, 0])
            m_dot = np.zeros((rows, cols))

            for i in range(rows):
                for j in range(cols):
                    m_dot[i, j] = vrho[i, j] + c + np.log10(r_s[i, j] ** 2)

        return mdot

    @staticmethod
    def rho_mdot(t, rho, dimensions = 1, r_s = 1., mu = 1.34):
        '''
        NOTE! Rho2d should be .T as in all outouts it is not .T in Table Analyze
        :param t: log10(t[:])
        :param rho: log10(rho[:,:])
        :param r_s:
        :param mu:
        :return:
        '''

        # c = np.log10(4*3.14*((r_s * Constants.solar_r)**2) / Constants.smperyear)
        c = np.log10(4 * 3.14 * ((Constants.solar_r) ** 2) / Constants.smperyear) + np.log10(r_s ** 2)

        if int(dimensions) == 0:
            return (rho + c + np.log10(Physics.sound_speed(t, mu, False)*100000))

        if int(dimensions) == 1:
            m_dot = np.zeros(len(t))
            for i in range(len(t)):
                m_dot[i] = ( rho[i] + c + np.log10(Physics.sound_speed(t[i], mu, False)*100000))
            return m_dot

        if int(dimensions) == 2:
            cols = len(rho[0, :])
            rows = len(rho[:, 0])
            m_dot = np.zeros((rows, cols))

            for i in range(rows):
                for j in range(cols):
                    m_dot[i,j] = (rho[i, j] + c + np.log10(Physics.sound_speed(t[j], mu, False)*100000))
            return m_dot
        else:
            sys.exit('\t__Error. Wrong number of dimensions. Use 0,1,2. Given: {} | m_dot |'.format(dimensions))

    @staticmethod
    def mdot_rho(t, mdot, dimensions = 1, r_s = 1., mu = 1.34):
        smperyear = Constants.solar_m / Constants.yr

        c = np.log10(4*3.14*((r_s * Constants.solar_r)**2) / smperyear)

        if int(dimensions) == 0:
            return (mdot - c - np.log10(Physics.sound_speed(t, mu, False)*100000))

        if int(dimensions) == 1:
            m_dot = np.zeros(len(t))
            for i in range(len(t)):
                m_dot[i] = ( mdot[i] - c - np.log10(Physics.sound_speed(t[i], mu, False)*100000))
            return m_dot

        if int(dimensions) == 2:
            cols = len(mdot[0, :])
            rows = len(mdot[:, 0])
            m_dot = np.zeros((rows, cols))

            for i in range(rows):
                for j in range(cols):
                    m_dot[i,j] = (mdot[i, j] - c - np.log10(Physics.sound_speed(t[j], mu, False)*100000))
            return m_dot
        else:
            sys.exit('\t__Error. Wrong number of dimensions. Use 0,1,2. Given: {} | mdot_rho |'.format(dimensions))

    @staticmethod
    def mean_free_path(rho, kap):
        '''
        :param rho: log(rho)
        :param kap: log(kap)
        :return: log(lambda)
        test: print(Physics.mean_free_path(-8.23, np.log10(0.83))) = 8.31
        '''

        return (-rho - kap) # = log(1/rho*kap)

    @staticmethod
    def sp_temp(rho, r, mdot, mu = 1.34):
        '''
        :param rho: in log(rho)
        :param r:  in solar radii
        :param mdot: in log(m in sm/year)
        :param mu: 1.34 for He matter
        :return: log(t_s)
        check: print(Physics.sp_temp(-7.971, 1.152, -4.3)) = 5.33...
        '''
        mdot_ = 10**(mdot) * Constants.smperyear # is cgs

        rho_ = 10**rho

        r_ = r * Constants.solar_r

        c = (mu * Constants.m_H / Constants.k_b)

        o = (mdot_ / (4 * np.pi * (r_ ** 2) * rho_)) ** 2

        return np.log10(c * o)

    @staticmethod
    def opt_depth_par(i, rho_arr, kap_arr, u_arr, r_arr, t_arr, mu_arr):

        i = i - 1
        dvdr = (u_arr[i] - u_arr[i-1]) / (r_arr[i] - r_arr[i-1])          # in km/s / r_sol

        # i = i - 1
        t = np.log10( np.abs((10**t_arr[i+1]  - 10**t_arr[i-1])  / 2) )


        mu= np.abs( (mu_arr[i+1] - mu_arr[i-1]) / 2 )


        rho=np.log10( np.abs((10**rho_arr[i+1] - 10**rho_arr[i-1]) /2) )


        kap=np.log10( np.abs((10**kap_arr[i+1] - 10**kap_arr[i-1]) /2) )


        t = t_arr[i]
        mu = mu_arr[i]
        rho = rho_arr[i]
        kap = kap_arr[i]

        v_s = Physics.sound_speed(t, mu, False) # in km/s
        mfp = 10**Physics.mean_free_path(rho, kap) / Constants.solar_r # in solar radii as the r and dr are
        # print(u, vu, r, vr, v_s, mfp)
        # print(dvdr)
        # print('t:', t_arr[i + 1], t_arr[i - 1], 10 ** t_arr[i + 1], 10 ** t_arr[i - 1])
        # print('mu:', mu_arr[i+1], mu_arr[i-1])
        # print('rho:', rho_arr[i+1], rho_arr[i-1])
        # print('kap:', 10**kap_arr[i+1], kap_arr[i-1])

        return (v_s / mfp) / dvdr #( km/s / r_sol) / (km/s / r_sol) - No units.

    @staticmethod
    def model_yz_to_xyz(x_1d_arr, y_1d_arr, z_2d_arr, star_y_coord, star_z_coord, num_of_model, lim_x1 = None, lim_x2 = None):
        '''
        Return: np.vstack(( int_star_x_coord, y_fill, mdot)) | [0,:] - ts
                                                   | [1,:] - llm | [2,:] - mdot  CONSTANTS FOR ALL ts

        :param x_1d_arr:
        :param y_1d_arr:
        :param z_2d_arr:
        :param star_y_coord:
        :param star_z_coord:
        :param num_of_model:
        :param lim_x1:
        :param lim_x2:
        :return:

        Uses interpolation, for find the ts coordinate of a star, if the mdot is provided (inverting Mdot = 4pi rs vs
            formula.
            In the Row of Mdot for a given L/M it finds the Mdot of the star and returns the ts of this point

            Given y_coord
            y   |      |    This is degeneracy, -> you have to restrict the ts - That are why lim_x1 = None, lim_x2 = None
                |      .                        Should be speciefied. Otherwise all possible solutions will be returned.
                |   .   .
                |.       .
                |         .
        y_star -|----------.------    -->  Finds a ts at which y == req.y for every
                |           .    x
                |            .
        '''

        #--------------------------------------------------CHECKING IF L or LM of the STAR is WITHIN L or LM limit------
        if y_1d_arr[0] < y_1d_arr[-1] and (star_y_coord < y_1d_arr[0] or star_y_coord > y_1d_arr[-1]):
            print('\t__Warning! Star: {} (lm: {}) '
                  'is beyond the lm range ({}, {})'.format(num_of_model, "%.2f" % star_y_coord,
                                                           "%.2f" % y_1d_arr[0], "%.2f" % y_1d_arr[-1]))
            return np.empty(0, )

        if y_1d_arr[0] > y_1d_arr[-1] and (star_y_coord > y_1d_arr[0] or star_y_coord < y_1d_arr[-1]):
            print('\t__Warning! Star: {} (lm: {}) '
                  'is beyond the lm range ({}, {})'.format(num_of_model, "%.2f" % star_y_coord,
                                                           "%.2f" % y_1d_arr[0], "%.2f" % y_1d_arr[-1]))
            return np.empty(0, )

        i_star_y_coord = Math.find_nearest_index(y_1d_arr, star_y_coord)
        z_row_for_star_z = np.array(z_2d_arr[i_star_y_coord, :])   # 1d Array of Mdot at a constant LM (this is y, while ts array is x)


        #--------------------------------------------------CHECKING IF Mdot of the STAR is WITHIN Mdot limit------------
        if star_z_coord > z_row_for_star_z.max() or star_z_coord < z_row_for_star_z.min(): # if true, you cannot solve the eq. for req. Mdot
            print('\t__Warning! Star: {} (lm: {}, mdot: {}) '
                  'is beyond the mdot range ({}, {})'.format(num_of_model, "%.2f" % star_y_coord, "%.2f" % star_z_coord,
                                                             "%.3f" % z_row_for_star_z.max(), "%.2f" % z_row_for_star_z.min()))
            return np.empty(0, )  # returns empty - no sloution possoble for that star withing given mdot array.


        # --------------------------------------------------SOLVING for REQ.Mdot. & GETTING THE Ts COORDINATE-----------
        int_star_x_coord = Math.solv_inter_row(x_1d_arr, z_row_for_star_z, star_z_coord)

        # print(int_star_x_coord)
        if not int_star_x_coord.any():
            print('\t__Warning. Star {} (l:{} , mdot:{}) no sol. found in region (l:[{}], mdot=[{}, {}])'
                  .format(num_of_model, star_y_coord, star_z_coord, y_1d_arr[star_y_coord], z_row_for_star_z.min(), z_row_for_star_z.max()))

        # z_row_for_star_z = z_row_for_star_z.fill(star_mdot)
        # print('m_dot: {} in ({}), t sols: {}'.format("%.3f" % star_mdot, z_row_for_star_z, int_star_x_coord))
        if not int_star_x_coord.any():
            raise ValueError('No solutions in |lm_mdot_obs_to_ts_lm| Given mdot: {} is in mdot range ({}, {})'.format(
                    "%.2f" % star_z_coord, "%.2f" % z_row_for_star_z.max(), "%.2f" % z_row_for_star_z.min()))


        # --------------------------------------------------CHECKING IF LIMITS FOR T ARE WITHING Ts ARRAY---------------

        if lim_x1 != None and lim_x1 < x_1d_arr[0] and lim_x2 == None:
            raise ValueError('lim_ts1({}) < t_s_arr[0]({}) '.format(lim_x1, x_1d_arr[0]))

        if lim_x2 != None and lim_x2 > x_1d_arr[-1] and lim_x1 == None:
            raise ValueError('lim_ts2({}) > t_s_arr[-1]({}) '.format(lim_x2, x_1d_arr[-1]))

        if lim_x1 != None and lim_x2 != None and lim_x1 > lim_x2:
            raise ValueError('lim_t1({}) > lim_t2({}) '.format(lim_x1, lim_x2))

        #-----------------------------------------------------CROPPING THE Ts SOLUTIONS TO ONLY THOSE WITHIN LIMITS-----
        if lim_x1 != None and lim_x2 == None:
            x_sol_crop = []
            for i in range(len(int_star_x_coord)):
                if int_star_x_coord[i] >= lim_x1:
                    x_sol_crop = np.append(x_sol_crop, int_star_x_coord[i]) # Contatins X  That satisfies the lim_t1 and lim_t2

            z_fill = np.zeros(len(x_sol_crop))
            z_fill.fill(star_z_coord)
            y_fill = np.zeros(len(x_sol_crop))
            y_fill.fill(star_y_coord)       # !! FIls the array with same L/M values (as L or LM is UNIQUE for a Given Star)
            return np.vstack(( np.array(x_sol_crop), y_fill,  z_fill))

        if lim_x1 == None and lim_x2 != None:
            x_sol_crop = []
            for i in range(len(int_star_x_coord)):
                if int_star_x_coord[i] <= lim_x2:
                    x_sol_crop = np.append(x_sol_crop, int_star_x_coord[i])

            z_fill = np.zeros(len(x_sol_crop))
            z_fill.fill(star_z_coord)
            y_fill = np.zeros(len(x_sol_crop))
            y_fill.fill(star_y_coord)       # !! FIls the array with same L/M values (as L or LM is UNIQUE for a Given Star)
            # y_fill.fill(star_l_lm)
            return np.vstack(( np.array(x_sol_crop), y_fill , z_fill))

        if lim_x1 != None and lim_x2 != None:
            x_sol_crop = []
            for i in range(len(int_star_x_coord)):
                if int_star_x_coord[i] >= lim_x1 and int_star_x_coord[i] <= lim_x2:
                    x_sol_crop = np.append(x_sol_crop, int_star_x_coord[i])

            z_fill = np.zeros(len(x_sol_crop))
            z_fill.fill(star_z_coord)
            y_fill = np.zeros(len(x_sol_crop))
            y_fill.fill(star_y_coord)       # !! FIls the array with same L/M values (as L or LM is UNIQUE for a Given Star)

            return np.vstack(( np.array(x_sol_crop), y_fill, z_fill))

        z_fill = np.zeros(len(int_star_x_coord))
        z_fill.fill(star_z_coord)
        y_fill = np.zeros(len(int_star_x_coord))
        y_fill.fill(star_y_coord)

        # print(np.array(int_star_x_coord).shape, y_fill.shape, z_fill.shape)

        return np.vstack(( np.array(int_star_x_coord), y_fill, z_fill))

    @staticmethod
    def lm_to_l(log_lm):
        '''
        From Langer 1987 paper Mass Lum relation for WNE stars
        :param log_lm:
        :return:
        '''
        a1 = 2.357485
        b1 = 3.407930
        c1 = -0.654431
        a2 = -0.158206
        b2 = -0.053868
        c2 = 0.055467
        # f1 = a1 + b1*lm + c1*(lm**2)
        # f2 = a2 + b2*ll + c2*(ll**2)

        d = log_lm + a2
        print((1-b2))
        disc = ((b2 - 1)**2 - 4*c2*d)
        #
        res = ( ( - (b2-1) - np.sqrt( disc ) ) / (2*c2) )

        return res

    @staticmethod
    def l_to_m(log_l):
        '''
        From Langer 1987 paper Mass Lum relation for WNE stars
        :param log_l:
        :return:
        '''
        a2 = -0.158206
        b2 = -0.053868
        c2 = 0.055467
        return ( a2 + b2 * log_l + c2*(log_l**2) )

    @staticmethod
    def m_to_l(log_m):
        '''
        From Langer 1987 paper Mass Lum relation for WNE stars
        :param log_m:
        :return:
        '''
        a1 = 2.357485
        b1 = 3.407930
        c1 = -0.654431
        return ( a1 + b1*log_m + (c1*log_m**2) )

    @staticmethod
    def l_to_lm(log_l):
        '''
        From Langer 1987 paper Mass Lum relation for WNE stars
        :param log_l:
        :return:
        '''
        a1 = 2.357485
        b1 = 3.407930
        c1 = -0.654431
        a2 = -0.158206
        b2 = -0.053868
        c2 = 0.055467
        return (-a2 -(b2 -1)*log_l - c2*(log_l**2) )

class Labels:
    def __init__(self):
        pass

    @staticmethod
    def lbls(v_n):
        #solar
        if v_n == 'l':
            return '$\log(L)$'#(L_{\odot})
        if v_n == 'r':
            return '$R(R_{\odot})$'

        #sonic and general
        if v_n == 'v' or v_n == 'u':
            return 'v (km/s)'
        if v_n == 'rho':
            return '$\log(\rho)$'
        if v_n == 'k' or v_n == 'kappa':
            return '$\kappa$'
        if v_n == 't':
            return 'log(T)'
        if v_n == 'ts':
            return '$\log(T_{s})$'
        if v_n == 'lm':
            return '$\log(L/M)$'
        if v_n == 'mdot':
            return '$\log(\dot{M}$)'

