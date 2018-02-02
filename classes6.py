#====================================================#
#
# This is the main file, containg my project
# reading, interpolating, analysing OPAL tables
# reading, plotting properites of the star from
# sm.data files of BEC output
#
#====================================================#


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
            print(arr)

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
    def get_list_uniq_ints(arr):
        ints = []
        counts = []
        ints.append(0)
        counts.append(arr[0])
        for i in range(1, len(arr)):
            if arr[i] not in counts:
                ints.append(i)
            else:
                ints.append(arr.index(arr[i]))
            counts.append(arr[i])
        return ints

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
    def rho_mdot(t, rho, dimensions = 1, r_s = 1., mu = 1.34):
        '''
        NOTE! Rho2d should be .T as in all outouts it is not .T in Table Analyze
        :param t: log10(t[:])
        :param rho: log10(rho[:,:])
        :param r_s:
        :param mu:
        :return:
        '''

        c = np.log10(4*3.14*((r_s * Constants.solar_r)**2) / Constants.smperyear)

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
    def lm_mdot_obs_to_ts_lm(t_s_arr, l_lm_arr, mdot_2darr, star_l_lm, star_mdot, number, lim_t1 = None, lim_t2 = None):
        '''
        Return: np.vstack(( lm_fill, t_sol ))

        :param t_s_arr:
        :param l_lm_arr:
        :param mdot_2darr:
        :param star_l_lm:
        :param star_mdot:
        :param number:
        :param lim_t1:
        :param lim_t2:
        :return:

        Uses interpolation, for find the ts coordinate of a star, if the mdot is provided (inverting Mdot = 4pi rs vs
            formula.
            In the Row of Mdot for a given L/M it finds the Mdot of the star and returns the ts of this point

            Given L/M
            Mdot|      |This is degeneracy, -> you have to restrict the ts - That are why lim_t1 = None, lim_t2 = None
                |      .                        Should be speciefied. Otherwise all possible solutions will be returned.
                |   .   .
                |.     | .
                |         .
        Req.Mdot-----------.------    -->  Finds a ts at which Mdot == req.Mdot for every
                |           .    ts
                |      |     .
        '''

        #--------------------------------------------------CHECKING IF L or LM of the STAR is WITHIN L or LM limit------
        if l_lm_arr[0] < l_lm_arr[-1] and (star_l_lm < l_lm_arr[0] or star_l_lm > l_lm_arr[-1]):
            print('\t__Warning! Star: {} (lm: {}) '
                  'is beyond the lm range ({}, {})'.format(number, "%.2f" % star_l_lm,
                                                           "%.2f" % l_lm_arr[0], "%.2f" % l_lm_arr[-1]))
            return np.empty(0, )

        if l_lm_arr[0] > l_lm_arr[-1] and (star_l_lm > l_lm_arr[0] or star_l_lm < l_lm_arr[-1]):
            print('\t__Warning! Star: {} (lm: {}) '
                  'is beyond the lm range ({}, {})'.format(number, "%.2f" % star_l_lm,
                                                           "%.2f" % l_lm_arr[0], "%.2f" % l_lm_arr[-1]))
            return np.empty(0, )

        i_lm = Math.find_nearest_index(l_lm_arr, star_l_lm)
        mdot_arr = np.array(mdot_2darr[i_lm, :])   # 1d Array of Mdot at a constant LM (this is y, while ts array is x)

        #--------------------------------------------------CHECKING IF Mdot of the STAR is WITHIN Mdot limit------------
        if star_mdot > mdot_arr.max() or star_mdot < mdot_arr.min(): # if true, you cannot solve the eq. for req. Mdot
            print('\t__Warning! Star: {} (lm: {}, mdot: {}) '
                  'is beyond the mdot range ({}, {})'.format(number, "%.2f" % star_l_lm, "%.2f" % star_mdot,
                                                             "%.3f" % mdot_arr.max(), "%.2f" % mdot_arr.min()))
            return np.empty(0, )  # returns empty - no sloution possoble for that star withing given mdot array.

        # --------------------------------------------------SOLVING for REQ.Mdot. & GETTING THE Ts COORDINATE-----------
        t_sol = Math.solv_inter_row(t_s_arr, mdot_arr, star_mdot)
        # print('m_dot: {} in ({}), t sols: {}'.format("%.3f" % star_mdot, mdot_arr, t_sol))
        if not t_sol.any():
            sys.exit(
                '\t__Error: No solutions in |lm_mdot_obs_to_ts_lm| Given mdot: {} is in mdot range ({}, {})'.format(
                    "%.2f" % star_mdot, "%.2f" % mdot_arr.max(), "%.2f" % mdot_arr.min()))

        # --------------------------------------------------CHECKING IF LIMITS FOR T ARE WITHING Ts ARRAY---------------

        if lim_t1 != None and lim_t1 < t_s_arr[0] and lim_t2 == None:
            print('\t__Error. lim_ts1({}) < t_s_arr[0]({}) '.format(lim_t1, t_s_arr[0]))
            raise ValueError

        if lim_t2 != None and lim_t2 > t_s_arr[-1] and lim_t1 == None:
            print('\t__Error. lim_ts2({}) > t_s_arr[-1]({}) '.format(lim_t2, t_s_arr[-1]))
            raise ValueError

        if lim_t1 != None and lim_t2 != None and lim_t1 > lim_t2:
            print('\t__Error. lim_t1({}) > lim_t2({}) '.format(lim_t1, lim_t2))
            raise ValueError

        #-----------------------------------------------------CROPPING THE Ts SOLUTIONS TO ONLY THOSE WITHIN LIMITS-----
        if lim_t1 != None and lim_t2 == None:
            t_sol_crop = []
            for i in range(len(t_sol)):
                if t_sol[i] >= lim_t1:
                    t_sol_crop = np.append(t_sol_crop, t_sol[i]) # Contatins X  That satisfies the lim_t1 and lim_t2

            lm_fill = np.zeros(len(t_sol_crop))
            lm_fill.fill(star_l_lm)     # !! FIls the array with same L/M values (as L or LM is UNIQUE for a Given Star)
            return np.vstack(( lm_fill, np.array(t_sol_crop) ))

        if lim_t1 == None and lim_t2 != None:
            t_sol_crop = []
            for i in range(len(t_sol)):
                if t_sol[i] <= lim_t2:
                    t_sol_crop = np.append(t_sol_crop, t_sol[i])

            lm_fill = np.zeros(len(t_sol_crop))
            lm_fill.fill(star_l_lm)
            return np.vstack(( lm_fill, np.array(t_sol_crop) ))

        if lim_t1 != None and lim_t2 != None:
            t_sol_crop = []
            for i in range(len(t_sol)):
                if t_sol[i] >= lim_t1 and t_sol[i] <= lim_t2:
                    t_sol_crop = np.append(t_sol_crop, t_sol[i])

            lm_fill = np.zeros(len(t_sol_crop))
            lm_fill.fill(star_l_lm)
            return np.vstack(( lm_fill, np.array(t_sol_crop) ))

        lm_fill = np.zeros(len(t_sol))
        lm_fill.fill(star_l_lm)

        return np.vstack((lm_fill, np.array(t_sol)))

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

class Read_Table:
    '''
        This class reads the 2D OPAL table, where
        0th raw is values of R
        0th column are temperatures
        else - values of opacity
        Everything in log10()
    '''

    def __init__(self, table_name):
        '''
        :param table_name: example ./opal/table1 extensiton is .data by default
        '''

        self.table_name = table_name


        try:
            f = open(self.table_name, 'r').readlines()
        except FileNotFoundError as fnf:
            print(fnf)
            sys.exit('|\tError! File with table is not found')
        except Exception as unk:
            print(unk)
        else:
            len1d = f.__len__()
            if len1d == 0: Errors.is_arr_empty(len1d, 'Readtable | __init__', True)
            len2d = f[0].split().__len__()

            self.table = np.zeros((len1d, len2d))
            for i in range(len1d):
                self.table[i, :] = np.array(f[i].split(), dtype=float)

            f.clear()

            # all parameters that can be taken directly from the table
            self.r = self.table[0, 1:]
            self.t = self.table[1:, 0]
            self.kappas = self.table[1:, 1:]
            self.rho = Physics.get_rho(self.r, self.t)

class Row_Analyze:

    mask = 9.999 # mask vale can be adjusted!
    crit_for_smooth = 2 # by  what factor two consequent values can be diff
                        # to still be smooth

    def __init__(self):
        pass

    #---------------------STEP_1-----------------------
    # Cut the 'mask' values
    @staticmethod
    def cut_mask_val(x_row, y_row):
        '''
            Removes mask values from opal_raw array and
            corresponding elements from rho_raw array
        '''

        Errors.is_arr_empty(x_row, 'from2darray | cut_mask_val', True)
        Errors.is_arr_eq(x_row, y_row, 'from2darray | cut_mask_val', True)

        x_row = np.array(x_row)
        y_row = np.array(y_row)

        arr_mask = []

        for i in range(len(y_row)):  # might be a problem with not enough elements. Put -1
            if y_row[i] == Row_Analyze.mask: # take val from class
                arr_mask = np.append(arr_mask, i)  # elements to be deleted from an array

        if any(arr_mask):
            print('\t___Note: Mask Values = ',Row_Analyze.mask,' removed at: ', arr_mask)

        y_row = np.delete(y_row, arr_mask)  # removing mask elements
        x_row = np.delete(x_row, arr_mask)

        return np.vstack((x_row, y_row))

    #---------------------STEP_2-----------------------
    # Cut the largest smooth area (never actually tested!!!)
    # Inapropriate way of doing it anyway...
    @staticmethod
    def get_smooth(x_row, y_row):

        # 2. Check for not smoothness
        arr_excess = []
        arr_excess = np.append(arr_excess, 0)
        delta = np.diff(y_row)
        delta_cr = Row_Analyze.crit_for_smooth * np.sum(delta) / (len(delta))  # mean value of all elements

        for i in range(len(delta)):  # delta -2 as delta has 1 less element than opal
            if (delta[i] == delta_cr):
                arr_excess = np.append(arr_excess, i)

        arr_excess = np.append(arr_excess, (len(y_row)))

        Errors.is_arr_shape(arr_excess,(2, ), 'AnalyzeTable | get_smooth',
                            False, 'Array is not smooth. Elements:', arr_excess[1:1])


        # 3. Selecting the biggest smooth region: (not checked!)
        if (len(arr_excess) > 2):
            print('\t___Warning! Values are no smooth. Taking the largest smooth area, | get_smooth |')
            diff2 = np.diff(arr_excess)
            ind_max = np.argmax(diff2)
            ind_begin = arr_excess[ind_max] + 1  # start of the biggest smooth region
            ind_end = arr_excess[ind_max + 1]  # end of the biggest smooth region

            if (ind_begin > (len(y_row) - 1)):   sys.exit('\t___Error in fingin the start of the smooth area. | get_smooth |')
            if (ind_end > (len(y_row) - 1)):   sys.exit('\t___Error in fingin the end of the smooth area. | get_smooth |')
            print('\t___Warning! only elements:', ind_begin, '->', ind_end, '(', ind_begin - ind_end, 'out of ',
                  len(y_row), 'are taken, |get_smooth|)')

            # print(ind_begin, ind_end)
            y_row = y_row[int(ind_begin):int(ind_end)]  # selecting the smooth region
            x_row = x_row[int(ind_begin):int(ind_end)]


        return np.vstack((x_row, y_row))

    # ---------------------STEP_3-----------------------
    # Cut the repeating values in the beginning of the raw
    # repetitions in the middel are not addressed!
    @staticmethod
    def cut_rep_val(x_row, y_row):
        '''
            Warning! Use with caution.
            Repetition in the middle of the data is not treated.
        :param x_row:
        :param y_row:
        :return:
        '''
        # 4. Selecting and removing the repeating regions (same value)
        delta = np.diff(y_row)  # redefine as length of an array has changed
        arr_zero = []
        if (delta[0] == 0):
            for i in range(len(delta)):
                if (delta[i] == 0):
                    arr_zero = np.append(arr_zero, i)  # collecting how many elements in the beginning are the same
                else:
                    break

        if (len(arr_zero) != 0): print('\t___Warning! Repetitions', arr_zero, ' in the beginning -> removed | cut_rep_val |')

        y_row = np.delete(y_row, arr_zero)  # removing the repetition in the beginning
        x_row = np.delete(x_row, arr_zero)

        # checking if there is repetition inside
        delta = np.diff(y_row)  # again - redefenition
        arr_zero2 = []
        for i in range(len(delta)):
            if (delta[i] == 0):
                arr_zero2 = np.append(arr_zero2, i)

        if (len(arr_zero2) > 0): print('\t___Warning! repeating values: ', arr_zero2,
                                       'inside an array -> NOT REMOVED!| cut_rep_val |')

        return np.vstack((x_row, y_row))

    # All abve methods together, performed if conditions are True
    @staticmethod
    def clear_row(x_row, y_row, cut_mask=True, cut_rep=True, cut_smooth=True):

        Errors.is_arr_eq(x_row, y_row, 'Row_Analyze | clear_row', True)
        Errors.is_arr_empty(x_row, 'Row_Analyze | clear_row', True)

        x_tmp = x_row
        y_tmp = y_row

        if cut_mask:
            no_masks = Row_Analyze.cut_mask_val(x_tmp, y_tmp)
            x_tmp = no_masks[0, :]
            y_tmp = no_masks[1, :]  # it will update them, if this option is chosen

        if cut_smooth:
            smooth = Row_Analyze.get_smooth(x_tmp, y_tmp)
            x_tmp = smooth[0, :]
            y_tmp = smooth[1, :]  # it will update them, if this option is chosen

        if cut_rep:
            no_rep = Row_Analyze.cut_rep_val(x_tmp, y_tmp)
            x_tmp = no_rep[0, :]
            y_tmp = no_rep[1, :]  # it will update them, if this option is chosen

        return np.vstack((x_tmp, y_tmp))

    # ---------------------STEP_4-----------------------
    # Identefy the cases and solve for the case
    # repetitions in the middel are not addressed!
    @staticmethod
    def get_case_limits(x_row, y_row, n_anal):
        '''
        :param x_row:
        :param y_row:
        :param n_anal: Here to be 1000 as it is only searching for limits. More points - more precise y1,y2
        :return: array[case, lim_op1, lim_y2]
        '''
        case = -1  # -1 stands for unknown
        x_row = np.array(x_row)
        y_row = np.array(y_row)
        y1 = y_row.min()
        y2 = y_row.max()
        lim_y1 = 0  # new lemeits from interploation
        lim_y2 = 0

        # print('\t Opal region: ', y1, ' ->', y2)

        new_y_grid = np.mgrid[y1:y2:(n_anal) * 1j]

        singl_i = []
        singl_sol = []
        singl_y = []
        db_i = []
        db_y = []
        db_sol_1 = []
        db_sol_2 = []
        exc_y_occur = []

        for i in range(1, n_anal - 1):  # Must be +1 -> -1 or there is no solution for the last and first point(

            sol = Math.solv_inter_row(x_row, y_row, new_y_grid[i])

            if (len(sol) == 0):
                print('\t___ERROR! At step {}/{} No solutions found | get_case_limits | \n '
                      ' k_row:({}, {}), k_grid_point: ({})'.format(i, n_anal - 1, y_row[0], y_row[-1], new_y_grid[i]))

                # sys.exit('\t___Error: No solutions Found in | Row_Analyze, get_case_limits |')

            if (len(sol) == 1):
                singl_i = np.append(singl_i, i)  # list of indexis of grid elements
                singl_sol = np.append(singl_sol, sol)  # list of kappa values
                singl_y = np.append(singl_y, new_y_grid[i])

            if (len(sol) == 2):
                db_i = np.append(db_i, int(i), )
                db_sol_1 = np.append(db_sol_1, sol[0])
                db_sol_2 = np.append(db_sol_2, sol[1])
                db_y = np.append(db_y, new_y_grid[i])

            if (len(sol) > 2):  # WARNING ! I Removed the stop signal
                exc_y_occur = np.append(exc_y_occur, new_y_grid[i])
                print('\t___Warning! At step', i, 'More than 2 solutions found | get_case_limits |', sol)
        #            sys.exit('___Error: more than 2 solution found for a given kappa.')



        print('\t__Note: single solutions for:', len(singl_y), ' out of ', n_anal - 2, ' elements | get_case_limits |')
        print('\t__Note: Double solutions for:', len(db_y), ' out of ', n_anal - 2 - len(singl_y), ' expected')

        # Check if there are several regions of degeneracy:
        delta = np.diff(db_i)
        for i in range(len(delta)):
            if (delta[i] > 1):
                sys.exit('\t___Error! Found more than 1 degenerate region. Treatment is not prescribed | get_case_limits |')

        # Defining the cases - M, A, B, C, D and limits of the opal in each case.
        if (len(db_i) == 0 and len(exc_y_occur) == 0):
            case = 0  # Monotonic CASE DETERMINED
            lim_y1 = singl_y[0]
            lim_y2 = singl_y[-1]
            print('\n\t<<<<< Case 0 (/) >>>>>\n')

        # If there is a degenerate region - determin what case it is:
        # and Remove the part of the kappa (initial) that we don't need:
        if (len(db_i) != 0):
            mid = db_sol_1[len(db_sol_1) - 1] + (db_sol_2[len(db_sol_2) - 1] - db_sol_1[len(db_sol_1) - 1]) / 2

            if ((db_sol_2[-1] - db_sol_1[-1]) > (db_sol_2[0] - db_sol_1[0])):
                if (singl_sol[len(singl_sol) - 1] > mid):
                    case = 1
                    lim_y1 = np.array([singl_y[0], db_y[0]]).min()
                    lim_y2 = singl_y[-1]

                    print('\n\t<<<<< Case 1 (-./) >>>>>\n')
                else:
                    case = 2
                    # print(db_i)
                    lim_y1 = np.array([new_y_grid[int(db_i[0])], singl_y[0]]).min()
                    lim_y2 = new_y_grid[int(db_i.max())]

                    print('\n\t <<<<< Case 2 (-*\) >>>>>\n')
                    print('\t___Warning! Case (-*\) reduces the k region to: ', "%.2f" % lim_y1, ', ', "%.2f" % lim_y2)
            else:
                if (singl_sol[0] > mid):
                    case = 3

                    lim_y1 = new_y_grid[int(db_i.min())]
                    lim_y2 = np.array([new_y_grid[int(db_i.max())], singl_y[len(singl_y) - 1]]).max()

                    print('\n\t <<<<< Case 3 (\.-) >>>>>\n')
                    print('\t___Warning! Case (\.-) limits the range of kappa to:', "%.2f" % lim_y1, ', ', "%.2f" % lim_y2)
                else:
                    lim_y1 = np.array(singl_y[0], db_y[0]).min()
                    lim_y2 = np.array([singl_y[-1], db_y[-1]]).max()
                    case = 4

                    print('\n\t <<<<< Case 4 (/*-) >>>>>\n')

        if (len(db_i) == 0 and len(exc_y_occur) != 0):
            lim_y1 = singl_y[0]
            lim_y2 = singl_y[-1]
            case = 5
            print('\n\t<<<<< Warning! Case unidentified! (Case 5) >>>>>\n')

        # db_sol = np.hstack(([db_i[0], db_i[-1]], [db_sol_1[0], db_sol_1[-1]], [db_sol_2[0], db_sol_2[-1]], [db_y[0], db_y[-1]]))
        # sl_sol = np.hstack(([singl_i[0], singl_i[-1]], [singl_sol[0], singl_sol[-1]], [singl_y[0], singl_y[-1]]))
        # exc = np.array([exc_y_occur[0], exc_y_occur[-1]])
        # db_sol =[db_i: (1st, last), db_sol_1: (1st, last), db_sol_2: (1st, last), db_y: (1st last)]
        # sl_sol = [singl_i: (1st last), sing_sol: (first, last), sing_op: (first, last)]
        # exc_global = [1st elemnt, last element]

        Errors.is_more_1([lim_y1],'| get_case |,', True, ' kim_k1 has more than 1 value')
        Errors.is_more_1([lim_y2], '| get_case |,', True, ' kim_k2 has more than 1 value')
        return np.array([case, lim_y1, lim_y2])

    # ---------------------STEP_5-----------------------
    # Solve for a given case
    # Warning! Case 5 is unidentified
    @staticmethod
    def case_0(x_row, y_row, y_grid, depth):
        f_opal = []
        f_rho = []
        for i in range(len(y_grid)):

            # kap = kappa_grid[i]+0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000001

            # f = interpolate.UnivariateSpline(rho_row, np.array(kappa_row - kap), s=0)
            sol = Math.solv_inter_row(x_row, y_row, y_grid[i])
            # sol = f.roots()
            if len(sol) == 0:
                print('\t__Error. No solutions in |case_0|. kappa_row - kappa_grid[{}] is {}, rho is {}'
                      .format(i, np.array(y_row - y_grid[i]), x_row))

            Errors.is_more_1(sol, 'case_0', True, ' at i:',i)

            f_opal = np.append(f_opal, y_grid[i])
            f_rho = np.append(f_rho, sol)

        Errors.is_arrsize_eq_m(f_rho, depth, 'case_0', True, 'Less elements in final arrray than in grid')

        return np.vstack((f_rho, f_opal))

    @staticmethod
    def case_1(x_row, y_row, y_grid, n_interp):
        f_opal = []
        f_rho = []
        for i in range(len(y_grid)):

            sol = Math.solv_inter_row(x_row, y_row, y_grid[i])

            Errors.is_arr_empty(sol, 'case_1', True, 'No solutions at i:', i)

            if (len(sol) == 1):
                f_opal = np.append(f_opal, y_grid[i])
                f_rho = np.append(f_rho, sol)

            if (len(sol) == 2):
                f_opal = np.append(f_opal, y_grid[i])
                f_rho = np.append(f_rho, sol[1])  # attach second element

            Errors.is_more_2(sol, 'case_1', False, ' at i:', i)# WARNING! I removed the stop signal! False!

            if (len(sol) > 2):
                f_opal = np.append(f_opal, y_grid[i])
                f_rho = np.append(f_rho, sol[len(sol) - 1])  # the last element

        Errors.is_arrsize_eq_m(f_rho, n_interp, 'case_1', True, 'Less elements in final arrray than in grid')

        return np.vstack((f_rho, f_opal))

    @staticmethod
    def case_2(x_row, y_row, y_grid, n_interp):
        f_opal = []
        f_rho = []
        for i in range(len(y_grid)):

            sol = Math.solv_inter_row(x_row, y_row, y_grid[i])

            Errors.is_arr_empty(sol, 'case_2', True, 'No solutions at i:', i)

            if (len(sol) == 2):
                f_opal = np.append(f_opal, y_grid[i])
                f_rho = np.append(f_rho, sol[0])  # attach first(!) element

            if (len(sol) == 1):  # should be just one element.
                f_opal = np.append(f_opal, y_grid[i])
                f_rho = np.append(f_rho, sol)

            Errors.is_more_2(sol, 'case_2', True, ' at i:', i)

        Errors.is_arrsize_eq_m(f_rho, n_interp, 'case_2', True, 'Less elements in final arrray than in grid')

        return np.vstack((f_rho, f_opal))

    @staticmethod
    def case_3(x_row, y_row, y_grid, n_interp):
        f_opal = []
        f_rho = []

        for i in range(len(y_grid)):
            sol = Math.solv_inter_row(x_row, y_row, y_grid[i])

            Errors.is_arr_empty(sol, 'case_3', True, 'No solutions at i:', i)

            if (len(sol) == 1):  # should be just one element.
                f_opal = np.append(f_opal, y_grid[i])
                f_rho = np.append(f_rho, sol)

            if (len(sol) == 2):
                f_opal = np.append(f_opal, y_grid[i])
                f_rho = np.append(f_rho, sol[1])  # attach second(!) element

                Errors.is_more_2(sol, 'case_3', True, ' at i:', i)

        Errors.is_arrsize_eq_m(f_rho, n_interp, 'case_3', True, 'Less elements in final arrray than in grid', [len(f_rho), n_interp])

        return np.vstack((f_rho, f_opal))

    @staticmethod
    def case_4(x_row, y_row, y_grid, n_interp):
        f_opal = []
        f_rho = []
        for i in range(len(y_grid)):
            sol = Math.solv_inter_row(x_row, y_row, y_grid[i])

            Errors.is_arr_empty(sol, 'case_4', True, 'No solutions at i:', i)

            if (len(sol) == 1):
                f_opal = np.append(f_opal, y_grid[i])
                f_rho = np.append(f_rho, sol)

            if (len(sol) == 2):
                f_opal = np.append(f_opal, y_grid[i])
                f_rho = np.append(f_rho, sol[0])  # attach first element

            Errors.is_more_2(sol, 'case_4', True, ' at i:', i)

        Errors.is_arrsize_eq_m(f_rho, n_interp, 'case_4', True, 'Less elements in final arrray than in grid')

        return np.vstack((f_rho, f_opal))

    @staticmethod
    def case_5(x_row, y_row, y_grid, n_interp):
        f_opal = []
        f_rho = []
        for i in range(len(y_grid)):

            sol = Math.solv_inter_row(x_row, y_row, y_grid[i])

            Errors.is_arr_empty(sol, 'case_5', True, 'No solutions at i:', i)

            Errors.is_more_1(sol, 'case_4', False, ' at i:', i)

            if (len(sol) > 1):
                f_opal = np.append(f_opal, y_grid[i])
                f_rho = np.append(f_rho, sol[-1])

            if (len(sol) == 1):
                f_opal = np.append(f_opal, y_grid[i])
                f_rho = np.append(f_rho, sol)

        Errors.is_arrsize_eq_m(f_rho, n_interp, 'case_5', True, 'Less elements in final arrray than in grid')
        print('\t__Note: In case_5 the 3rd out of 3 solutions is used. UNSAFE!')
        return np.vstack((f_rho, f_opal))

    @staticmethod
    def solve_for_row(lim_y1, lim_y2, case, n_interp, x_row, y_row):
        # After the Case has been identified, the actual (T=const, kappa[])-> rho[] can be done
        # given:
        # lim_op1 = op1#np.log10(0.54) #for 20sm
        # lim_op2 = op2#np.log10(0.94) #for 10sm model
        # depth = 1000

        kappa_grid = np.mgrid[lim_y1:lim_y2:n_interp * 1j]

        # treat cases:
        if (case == 0):  # monotonically increasing array - only one solution
            return Row_Analyze.case_0(x_row, y_row, kappa_grid, n_interp)

        if (case == 1):  # there is a decreasing part in the beginning, small.
            return Row_Analyze.case_1(x_row, y_row, kappa_grid, n_interp)

        if (case == 2):
            return Row_Analyze.case_2(x_row, y_row, kappa_grid, n_interp)

        if (case == 3):
            return Row_Analyze.case_3(x_row, y_row, kappa_grid, n_interp)

        if (case == 4):  # there is a decreasing part in the end, small.
            return Row_Analyze.case_4(x_row, y_row, kappa_grid, n_interp)

        if (case == 5):  # there are only single solutions and one triple! (!)
            return Row_Analyze.case_5(x_row, y_row, kappa_grid, n_interp)

        print('\t___Error! Case unspecified! | solve_for_row | case:', case)

class Table_Analyze(Read_Table):

    # o stands for options -------------

    o_cut_mask =   True # by default the full analythisis is performed
    o_cut_rep =    True
    o_cut_smooth = True

    plot_k_vs_t = True  # plots the available kappa region, the selected and the interpolated areas as well.

    def __init__(self, table_name, n_anal_, load_lim_cases, output_dir = '../data/output/', plot_dir = '../data/plots/'):

        # super().__init__(table_name) # using inheritance instead of calling for an instance, to get the rho, kappa and t
        Read_Table.__init__(self, table_name) # better way of inheritting it... The previous one required me to override something
        # cl1 = Read_Table(table_name)

        self.output_dir = output_dir
        self.plot_dir = plot_dir


        self.rho2d = Physics.get_rho(self.r, self.t)
        self.kappa2d = self.kappas


        # self.n_out = n_out_
        self.n_anal = n_anal_

        self.min_k = []
        self.max_k = []
        self.cases = []
        self.t_avl = []


        self.get_case_limits_in_table(load_lim_cases)

    def case_limits_in_table(self):
        ''' Finds an array of min and max kappas in the kappa=f(rho), - for each raw in
            The number of raws if given my t1 - t2 range. If t1 = t2 - only one t is considered
            and this function becomes equivalent to 'solve_for_row' from 'Raw_Analyze
        '''

        # if t1 == None:
        #     t1 = self.t.min()
        # if t2 == None:
        #     t2 = self.t.max()
        #
        # start = Math.find_nearest_index(self.t, t1)
        # stop  = Math.find_nearest_index(self.t, t2) + 1 # as the last element is exclusive in [i:j] operation

        print('=============================SEARCHING=FOR=LIMITS=&=CASES============================')
        print('\t__None: Peforming search withing temp range: ', self.t.min(), ' ', self.t.max())

        min_k = []
        max_k = []
        cases = []
        t_avl = []

        for i in range(len(self.t)):
            print('\t <---------------------| t = ',self.t[i],' |--------------------->\n')


            tmp1 = Row_Analyze.clear_row(self.rho2d[i, :], self.kappa2d[i, :], self.o_cut_mask, self.o_cut_rep, self.o_cut_smooth)
            tmp2 = Row_Analyze.get_case_limits(tmp1[0], tmp1[1], self.n_anal)

            min_k = np.append(min_k, tmp2[1])
            max_k = np.append(max_k, tmp2[2])
            cases = np.append(cases, tmp2[0])
            t_avl = np.append(t_avl, self.t[i])

            print('\t <-------------------| END of ', self.t[i], ' |-------------------->\n')


        print('===================================END=OF=SEARCHING==================================')

        # ----------------------------Saving the Table --------------------------
        out = np.vstack((cases, min_k, max_k, t_avl))

        table_name_extension = self.table_name.split('/')[-1]
        table_name = table_name_extension.split('.')[0]
        out_name = self.output_dir + table_name + '.caslim'
        np.savetxt(out_name, out, delimiter='  ', fmt='%1.3e')

        return out

    def get_case_limits_in_table(self, load_lim_cases):
        '''
        Updates the "self.cases self.min_k self.max_k self.t_avl" in the class either via direct computation
        or by loading the special file with 'out_name'
        :return:
        '''
        table_name_extension = self.table_name.split('/')[-1]
        table_name = table_name_extension.split('.')[0]
        out_name = self.output_dir +  table_name + '.caslim' # like ../data/table8.data.caslim

        if load_lim_cases :
            load = np.loadtxt(out_name, dtype=float, delimiter='  ')
            print('\t__Note. Table with cases and limits for opal table: < {} > has been loaded succesfully.'
                  .format(self.table_name))

        else:
            load = self.case_limits_in_table()

        self.cases = np.append(self.cases, load[0, :])
        self.min_k = np.append(self.min_k, load[1, :])
        self.max_k = np.append(self.max_k, load[2, :])
        self.t_avl = np.append(self.t_avl, load[3, :])


        print('\t t_aval: {}'.format(self.t_avl))
        print('\t__N: t({} , {})[{}] and t_aval({}, {})[{}]'
              .format(self.t[0], self.t[-1], len(self.t), self.t_avl[0], self.t_avl[-1], len(self.t_avl)))

        if len(self.t_avl) < len(self.t):
            print('Analyzed t_avl {} < {} t given.'.format(len(self.t_avl), len(self.t)))
            #
            # Possible solution is to cut the t_avl from t, together with rho2d and kappa 2d
            #
            sys.exit('\t__Error. Analyzed t region (t_avl) < than given t | get_case_limits_in_table |')

        if len(self.min_k) == len(self.max_k) == len(self.cases) == len(self.t_avl) ==  len(self.t):
            print('\t__Note: k limits, cases and available t region are set | get_case_limits_in_table | ')

        print('\t__Note: *case_limits* output: (0--) cases, (1--) min_k , (2--) max_k, (3--) t_avl')
        return load

    # @staticmethod
    def check_t_lim(self, t1, t2):
        if t1 > t2:
            sys.exit('\t__Error. t1 ({}) > t2 ({}) in |check_t_lim| in |Table_Analyze|')
        # a = Errors.is_a_bigger_b(t1, t2,    '|check_t_lim|', True, ' wrong temp. limits')
        if t2 > self.t[-1]:
            sys.exit('\t__Error. |check_t_lim|, t2 {} > t[-1] {} '.format(t2, self.t[-1]))
        if t1 < self.t[0]:
            print('\t: t_array is: ({} , {}) consisting of {} elements' .format(self.t[0], self.t[-1], len(self.t)))
            sys.exit('t__Error. |check_t_lim| t1 {} < t[0] {}'.format(t1, self.t[0]))

    def get_it_lim(self, t1, t2, k1, k2):
        indx_1 =  [i for i in range(len(self.t)) if self.t[i] == t1][0]  #GGGGGGenerator
        indx_2 =  [i for i in range(len(self.t)) if self.t[i] == t2][0]  # taking [0] element as it is a list :(


        new_t = []
        new_t = np.append(new_t, self.t[indx_1+1 : indx_2]) # adding the t1-t2 part
        s = indx_1
        for i in range (indx_1): # goes below t1 untill k1, k2 go outside the min/max k range
            if (k1 >= self.min_k[s] and k1 <= self.max_k[s] and k2 >= self.min_k[s] and k2 <= self.max_k[s]):
                new_t = np.append(new_t, self.t[s])
                s = s - 1
            else:
                break

        s = indx_2
        for i in range (len(self.t) - indx_2):# goes up from t1 untill k1, k2 go outside the min/max k range
            if k1 >= self.min_k[s] and k1 <= self.max_k[s] and k2 >= self.min_k[s] and k2 <= self.max_k[s] :
                new_t = np.append(new_t, self.t[s])
                s = s + 1
            else:
                break

        new_t = np.sort(new_t)

        return new_t

    def check_lim_task(self, t1_, t2_, k1, k2):
        '''

        # :param t1: user's t1
        # :param t2: user's t2
        # :param t_arr: from Table_Analyze
        # :param k1: user's k1
        # :param k2: user's k2
        # :param min_k_arr: from Table_Analyze
        # :param max_k_arr: from Table_Analyze
        # :return: [t1, t2, k1, k2, it1, it2], where it1 and it2 are the t limits for interpolation
        '''

        if t2_ > self.t.max(): sys.exit('t1 > t_lim.max() |check_lim_task|')
        if t2_ < self.t.min(): sys.exit('t1 < t_lim.min() |check_lim_task|')

        # indx_1 =  [i for i in range(len(self.t)) if self.t[i] == t1][0]  #GGGGGGenerator
        # indx_2 =  [i for i in range(len(self.t)) if self.t[i] == t2][0] # taking [0] element as it is a list :(

        indx_1 = Math.find_nearest_index(self.t, t1_)
        indx_2 = Math.find_nearest_index(self.t, t2_)

        t1 = self.t[indx_1] # Redefining the t range, in case the given t1 t2 do not equal to one of the value
        t2 = self.t[indx_2] #   in the self.t array

        print('\t__Note! Selected t range is from t:[{},{}] to t:[{},{}].'.format(t1_,t2_,t1,t2))

        lim_k1 = np.array( self.min_k[indx_1:indx_2+1] ).max() # looking for lim_k WITHIN the t1, t2 limit from user
        lim_k2 = np.array( self.max_k[indx_1:indx_2+1] ).min()

        print('\t__Note: lim_k1:',lim_k1, 'lim_k2', lim_k2)

        # Errors.is_a_bigger_b(lim_k1, lim_k2, '|check_k_lim|', True, ' lim_k1 > lim_k2')

        if lim_k1 > lim_k2:
            print('\t t_avl: {}'.format(self.t_avl))
            print('\t min_k: {}'.format(self.min_k))
            print('\t max_k: {}'.format(self.max_k))
            print('\t__Error: lim_k1 {} > {} lim_k2 |check_lim_task| '.format(lim_k1, lim_k2))
            print('!!!! DECREASE THE Y2 LIMIT or INCREASE Y1. THE SELECTED REGION IS BIGGER THAN ABAILABEL !!!!')
            raise ValueError

        #if k1 != None and k2 != None:

        if k1 != None and k1 < lim_k1 :
            sys.exit('\t__Error: k1 < lim_k1 in the region: {} < {}'.format(k1, lim_k1))

        if k2 != None and k2 > lim_k2 :
            sys.exit('\t__Error: k2 > lim_k2 in the region: {} > {}'.format(k2, lim_k2))

        if k2 != None and k2 < lim_k1 :
            sys.exit('\t__Error: k2 < lim_k1 in the region: {} < {}'.format(k2, lim_k1))

        if k1 != None and k1 > lim_k2 :
            sys.exit('\t__Error: k1 > lim_k2 in the region: {} < {}'.format(k1, lim_k2))


        if k1 == None and k2 == None:
            k1 = lim_k1
            k2 = lim_k2
            it1 = t1
            it2 = t2
            print('\t__Note: k1 and k2 not given. Setting k1, k2 : [', "%.2f" % k1,'', "%.2f" % k2,']')
            print('\t__Note: Interpolation is limited to: t:[',it1,'',it2,'] k:[', "%.2f" % k1,'', "%.2f" % k2,']')
            # it1, it2,  - don't have a meaning!
            return [t1, t2, k1, k2, it1, it2]


        if k1 == None and k2 != None and k2 <= lim_k2:
            #get k1 from 'get_case_limits_in_table'

            k1 = lim_k1  # changing k1
            i_t = self.get_it_lim(t1, t2, k1, k2)
            it1 = i_t.min()
            it2 = i_t.max()

            print('\t__Note: k1 is not given. Setting k1: ', "%.2f" % k1)
            print('\t__Note: Interpolation is extended to: t:[', it1,'',it2,'] k:[', "%.2f" % k1,'', "%.2f" % k2,']')

            return [t1, t2, k1, k2, it1, it2]


        if k1 != None and k2 == None and k1 >= lim_k1 :
            print('\t__Note: k1 and k2 not given. Using the available limits in the region')
            #get k2 from 'get_case_limits_in_table'
            k2 = lim_k2

            i_t = self.get_it_lim(t1, t2, k1, k2)
            it1 = i_t.min()
            it2 = i_t.max()

            print('\t__Note: k2 is not given. Setting k2: ', "%.2f" % k2)
            print('\t__Note: Interpolation is extended to: t:[',it1,'',it2,'] k:[', "%.2f" % k1,'', "%.2f" % k2,']')

            return [t1, t2, k1, k2, it1, it2]


        if k1 >= lim_k1 and k2 <= lim_k2 and k1 == k2 :

            i_t = self.get_it_lim(t1, t2, k1, k2)
            it1 = i_t.min()
            it2 = i_t.max()

            print('\t__Note: k1 = k2, Solving for unique k:', "%.2f" % k1)
            print('\t__Note: Interpolation is extended to: t:[',it1,'',it2,'] k:[', "%.2f" % k1,']')

            return [t1, t2, k1, k2, it1, it2]


        if k1 >= lim_k1 and k2 <= lim_k2 : # k1 and k2 != None

            i_t = self.get_it_lim(t1, t2, k1, k2)
            it1 = i_t.min()
            it2 = i_t.max()

            print('\t__Note: Interpolation is extended to: t:[',it1,'',it2,'] k:[', "%.2f" % k1,'', "%.2f" % k2,']')

            return [t1, t2, k1, k2, it1, it2]



        # return np.array([1, k1, k2]) # last value stands for using a band of common kappas for all rows

    def table_plotting(self, t1 = None, t2 = None, n_out = 1000):
        '''
        No universal kappa limits. Use unique limits for every t.
        :return: set of plots
        n_out: is 1000 by default
        '''
        if t1 == None: t1 = self.t.min()
        if t2 == None: t2 = self.t.max()

        self.check_t_lim(t1,t2)

        # i_1 = [i for i in range(len(self.t)) if self.t[i] == t1][0] # for self.t as rho2d and kappa2d have indexes as t
        # i_2 = [i for i in range(len(self.t)) if self.t[i] == t2][0] + 1

        i_1 = Math.find_nearest_index(self.t, t1)
        i_2 = Math.find_nearest_index(self.t, t2) + 1


        f_kappa = np.zeros((i_2 - i_1, n_out))  # all rows showld be the same!
        f_rho =   np.zeros((i_2 - i_1, n_out))
        t_f = []
        print('====================================INTERPOLATING====================================')

        s = 0
        for i in range(i_1, i_2):
            print('\t <---------------------------| t[',i,'] = ',self.t[i],' |---------------------------->\n')

            tmp1 = Row_Analyze.clear_row(self.rho2d[i, :], self.kappa2d[i, :],
                                         self.o_cut_mask,
                                         self.o_cut_rep,
                                         self.o_cut_smooth)
            tmp2 = Row_Analyze.solve_for_row(self.min_k[i], self.max_k[i], self.cases[i], n_out, tmp1[0], tmp1[1])

            f_rho[s, :] = np.array(tmp2[0])
            f_kappa[s, :] = np.array(tmp2[1])
            t_f = np.append(t_f, self.t[s])

            tmp3 = Row_Analyze.cut_mask_val(self.rho2d[i, :],   # ONLY for plotting
                                            self.kappa2d[i, :])  # I have to cut masked values, or they

            PhysPlots.Rho_k_plot(tmp2[0], tmp2[1], tmp3[0], tmp3[1],  # screw up the scale of the plot :(
                                 self.min_k[i], self.max_k[i], self.cases[i], self.t[i], i,
                                 self.plot_dir + 'opal_plots/')
            s = s + 1
            print('\t <----------------------------------------------------------------------------------->\n')

        print(f_rho.shape, f_kappa.shape, t_f.shape)

    def treat_tasks_tlim(self, n_out, t1, t2, k1 = None, k2 = None, plot = False):

        self.check_t_lim(t1,t2)

        t1, t2, k1, k2, it1, it2 = self.check_lim_task(t1, t2, k1, k2)

        self.min_k = np.array(self.min_k) # for some reason the Error was, that min_k is a list not a np.array
        self.max_k = np.array(self.max_k)

        if Table_Analyze.plot_k_vs_t:
            PhysPlots.k_vs_t(self.t, self.min_k, self.max_k, True, True, k1, k2, t1, t2, it1, it2)  # save but not show

        i_1 = [i for i in range(len(self.t)) if self.t[i] == it1][0]        # for self.t as rho2d and kappa2d have indexes as t
        i_2 = [i for i in range(len(self.t)) if self.t[i] == it2][0] + 1    # + 1 added so if t2 = 5.5 it goes up to 5.5.

        print('====================================INTERPOLATING====================================')
        print('\t__Note: Limits for kappa are: ',
              "%.2f" % k1, ' ', "%.2f" % k2, '\n\t  t range is: ', t1, ' ', t2)

        # c = 0  # for cases and appending arrays

        # ii_1 = [i for i in range(len(self.t)) if self.t[i] ==  self.t[0]][0]    # for self.t as rho2d and kappa2d have indexes as t
        # ii_2 = [i for i in range(len(self.t)) if self.t[i] == self.t[-1]][0]

        f_kappa = np.zeros((i_2 - i_1, n_out))  # all rows showld be the same!
        f_rho = np.zeros((i_2 - i_1,   n_out))
        f_t = []

        s = 0
        for i in range(i_1, i_2):
            print('\t <---------------------------| t[', i, '] = ', self.t[i], ' |---------------------------->\n')
            tmp1 = Row_Analyze.clear_row(self.rho2d[i, :], self.kappa2d[i, :],
                                         self.o_cut_mask,
                                         self.o_cut_rep,
                                         self.o_cut_smooth)
            tmp2 = Row_Analyze.solve_for_row(k1, k2, self.cases[i], n_out, tmp1[0], tmp1[1])
            f_rho[s, :]   = np.array(tmp2[0])
            f_kappa[s, :] = np.array(tmp2[1])
            f_t = np.append(f_t, self.t[i])

            if plot:
                tmp3 = Row_Analyze.cut_mask_val(self.rho2d[i, :],
                                                self.kappa2d[i, :])  # I have to cut masked values, or they

                PhysPlots.Rho_k_plot(tmp2[0], tmp2[1], tmp3[0], tmp3[1],  # screw up the scale of the plot :(
                                     k1, k2, self.cases[i], self.t[i], i, self.plot_dir)
            s = s + 1
            print('\t <----------------------------------------------------------------------------------->\n')
        print('\t__Note: t_limited output: (--) k: {}  (|) t: {}  (|-) rho: {}'.format(f_kappa[0,:].shape, f_t.shape, f_rho.shape))


        return Math.combine(f_kappa[0,:], f_t, f_rho)

    def treat_tasks_interp_for_t(self, t1, t2, n_out, n_interp, k1 = None, k2 = None):

        # self.check_t_lim(t1, t2)

        res = self.treat_tasks_tlim(n_out, t1, t2, k1, k2)
        kap = res[0,1:]
        t   = res[1:,0]
        rho = res[1:,1:]

        print('\t__Note: Performing interpolation from t:', len(t), ' points to', n_interp)


        # print(n_interp, len(kap))
        new_rho = np.zeros((n_interp, len(kap)))
        new_t = np.mgrid[t1: t2: n_interp * 1j]

        # PhysPlots.rho_vs_t(t,rho[:,1])

        for i in range(len(kap)):
            new_rho[:, i] = Math.interp_row(t, rho[:,i], new_t)

        print('\t__Note: t_interp output: (--) k: {}  (|) t: {}  (|-) rho: {}'.format(kap.shape, new_t.shape, new_rho.shape))

        # PhysPlots.rho_vs_t(new_t, new_rho[:, 1]) # one line of rho

        return Math.combine(kap, new_t, new_rho)

    def interp_for_single_k(self, t1, t2, n_interp, k):
        '''
        returns (0--) t , (1--) rho
        :param t1:
        :param t2:
        :param n_interp:
        :param k:
        :return:
        '''
        # if n_out != 1:
        #     sys.exit('\t___Error: Only n_out = 1 can be used for single k (n_out given is {} ).'.format(n_out))
        # n out already set as 1

        res = self.treat_tasks_interp_for_t(t1, t2, 1, n_interp, k, k) # for 1 k interpolation
        t = res[1:,0]
        rho = res[1:,1]

        print('\t__Note: Single t output: (1--) t: {}  (2--) rho: {}'.format(t.shape, rho.shape))
        return np.vstack((t, rho))

class OPAL_Interpol(Read_Table):

    def __init__(self, table_name, n_anal_):

        # super().__init__(table_name) # inheritance, to avoid calling the instance
        # super(table_name, self).__init__()
        Read_Table.__init__(self, table_name)

        self.rho2d = Physics.get_rho(self.r, self.t)
        self.kappa2d = self.kappas
        self.t = self.t
        self.depth = n_anal_



        '''
            Checks if t1,t2 belogs to t, and rho1, rho2 belogs to corresponding
            raws at given t1 and t2, and if length of tables are right
        '''

        # table = np.array((table))

    # @classmethod
    # def from_OPAL_table(cls, table_name, n):
    #     '''
    #
    #     :param table_name: NAME of the OPAL table to read (with root and extension)
    #     :param n_out_: n of kappas in a grid (1 for single kappa output, for eddington opacity)
    #     :param n_anal_: n of points in interpolation the limits of kappa in all temperatures. Around 1000
    #     :return: a class
    #     '''
    #
    #     cl1 = Read_Table(table_name)
    #
    #     r = cl1.r
    #     t = cl1.t
    #     kap = cl1.kappas
    #     rho = Physics.get_rho(r, t)
    #
    #     return cls(rho, kap, t, n)

    def check_t_rho_limits(self, t1, t2,rho1, rho2):

        # Errors.is_a_bigger_b(t1, t2,    '|CheckInputData|', True, 't1 > t2 - must be t1 < t2')
        if t1 > t2:
            sys.exit('\t__Error! t1({})>t2({}) |OPAL_Interpol|check_t_rho_limits|'.format(t1,t2))

        Errors.is_a_bigger_b(self.t[0], t1,  '|CheckInputData|', True, 't1 < t_min in the table')
        Errors.is_a_bigger_b(t2, self.t[-1], '|CheckInputData|', True, 'ERROR: t2 > t_max in the table')

        i = Math.find_nearest_index(self.t, t1)
        j = Math.find_nearest_index(self.t, t2)

        if (rho1 == None):
            rho1 = self.rho2d[j, 1]; print('\t__Note: Smallest rho in the given T range: ', rho1)
        if (rho2 == None):
            rho2 = self.rho2d[i, len(self.rho2d[i, :]) - 1]; print('\t__Note: Largest rho in the given T range ', rho2)

        # Errors.is_a_bigger_b(rho1, rho2,         '|CheckInputData|', True, 'rho1 > rho2, must be rho1 < rho2')
        if rho1 > rho2:
            sys.exit('\t___Error. rho1({}) > rho2({}) in | OPAL_Interpol | check_t_rho_limits|'.format(rho1,rho2))

        if self.rho2d[j, 0] > rho1:
            sys.exit('\t__Error:  rho1 ({}) < rho[0] ({}) |check_t_rho_limits|'.format(rho1, self.rho2d[j, 0]))

        if self.rho2d[i, -1] < rho2:
            sys.exit('\t__Error:  rho2 ({}) > rho[-1] ({}) |check_t_rho_limits|'.format(rho2, self.rho2d[i, -1]))

        Errors.is_arr_shape(self.rho2d, self.kappa2d.shape,    '|CheckInputData|', True, 'shapes of rho and opal are different')
        Errors.is_arrsize_eq_m(self.t, len(self.kappa2d[:, ]), '|CheckInputData|', True, 'length of t and n of raws in Opal are different')

        print('\t__Note: Overall: min_ro:', self.rho2d.min(), ' max_rho: ', self.rho2d.max())
        print('\t__Note: Min_ro in T area:', self.rho2d[j, 0], ' max_rho in T area: ', self.rho2d[i, len(self.rho2d[i, :]) - 1])

        return np.array([t1, t2, rho1, rho2])

    def interp_opal_table(self, t1, t2, rho1 = None, rho2 = None):
        '''
            Conducts 2 consequent interplations.
            1st along each raw at const temperature, establishing a grid of new rho
            2nd anlog each line of const rho (from grid) interpolates columns
        '''
        # print('ro1: ',rho1, ' ro2: ',rho2)

        t1, t2, rho1, rho2 = self.check_t_rho_limits(t1,t2,rho1,rho2)


        crop_rho = np.mgrid[rho1:rho2:self.depth * 1j]  # print(crop_rho)
        crop_t = []
        crop_k = np.zeros((len(self.t), self.depth))

        for si in range(len(self.t)):
            if (rho1 > self.rho2d[si, 0] and rho2 < self.rho2d[si, len(self.rho2d[si, :]) - 1]):

                clean_arrays = Row_Analyze.clear_row(self.rho2d[si, :], self.kappa2d[si, :], True, False, False)
                crop_k[si, :] = Math.interp_row(clean_arrays[0, :], clean_arrays[1, :], crop_rho)
                crop_t = np.append(crop_t, self.t[si])

        crop_k = crop_k[~(crop_k == 0).all(1)]  # Removing ALL=0 raws, to leave only filled onese


        Errors.is_arr_eq(crop_k[:, 0], crop_t, '||interp_opal_table|', True, 'crop_k row != crop_t')

        extend_k = np.zeros((self.depth, self.depth))
        extend_crop_t = np.mgrid[t1:t2: self.depth * 1j]

        for si in range(self.depth):
            extend_k[:, si] = Math.interp_row(crop_t, crop_k[:, si], extend_crop_t)

        if (len(extend_k[0, :]) != len(crop_rho)): sys.exit("N of columns in new table not equal to length of rho")
        if (len(extend_k[:, 0]) != len(extend_crop_t)): sys.exit("N of raws in new table not equal to length of t")

        #    print extend_crop_t.shape, crop_rho.shape, extend_k.shape

        print('\t__Note interp_opal_table out: (--) t: {}  (|) rho: {}  (|-) k: {}'.format(extend_crop_t.shape, crop_rho.shape,
                                                                                      extend_k.T.shape))
        return Math.combine(extend_crop_t, crop_rho, extend_k.T)

class Read_Observables:

    def __init__(self, observ_name, path = '../data/obs/', exten = '.data'):

        self.table = []
        with open(path + observ_name + exten, 'r') as f:
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

    def obs_par(self, v_n, dtype):
        if v_n not in self.names:
            sys.exit('\t__Error. Name: {} is not is a list of par.names: \n\t{}'.format(v_n, self.names))

        n = self.names.index(v_n)

        res = []
        for i in range(len(self.table)):
            res.append(self.table[i].split()[n])

        if v_n == 't':
            res = np.log10(res) # accounting that effective temp


        return np.array(res, dtype=dtype)

    def obs_par_row(self, i, dtype):
        return np.array(self.table[i].split(), dtype=dtype)

    # def __init__(self, observ_name = './data/gal_wn.data'):
    #
    #     self.numb = []
    #     self.type = []
    #     self.t_ef = []
    #     self.r_t  = []
    #     self. v_t = []
    #     self.x_h  = []
    #     self.e_bv = []
    #     self.law_a= []
    #     self.dmag = []
    #     self.mag_v= []
    #     self.r_ef = []
    #     self.mdot_obs = []
    #     self.l_obs = []
    #     self.mdotv_inf_lc = []
    #     self.m_obs = []
    #
    #
    #
    #     f = open(observ_name, 'r').readlines()
    #     elements = len(np.array(f[3].split()))
    #     raws = f.__len__()
    #     table = []
    #
    #     # print(len(f[3].split()))
    #     for i in range(3, raws):
    #
    #         row = f[i].split()
    #         table.
    #
    #         self.numb.append(int(row[0]))
    #         self.type.append(row[1])
    #         self.t_ef.append(float(row[2]))
    #         self.r_t.append(row[3])
    #         self.v_t.append(float(row[4]))
    #         self.x_h.append(float(row[5]))
    #         self.e_bv.append(float(row[6]))
    #         self.law_a.append(row[7])
    #         self.dmag.append(float(row[8]))
    #         self.mag_v.append(float(row[9]))
    #         self.r_ef.append(float(row[10]))
    #         self.mdot_obs.append(float(row[11]))
    #         self.l_obs.append(float(row[12]))
    #         self.mdotv_inf_lc.append(float(row[13]))
    #         self.m_obs.append(float(row[14]))
    #
    #     f.clear()
    #
    #     # print('nums: ', self.numb)
    #     # print('typs: ', self.type)
    #     # print('t_ef: ', self.t_ef)
    #     # print('r_t:  ', self.r_t)
    #     # print('v_t:  ', self.v_t)
    #     # print('x_h:  ', self.x_h)
    #     # print('e_bv: ', self.e_bv)
    #     # print('law_a:', self.law_a)
    #     # print('dmag: ', self.dmag)
    #     # print('mag_v:', self.mag_v)
    #     # print('r_ef: ', self.r_ef)
    #     # print('mdot: ', self.mdot_obs)
    #     # print('l_obs:', self.l_obs)
    #     # print('form: ', self.mdotv_inf_lc)
    #     # print('m_obs:', self.m_obs)

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


        self.time = plot_table[:i_stop, 1]
        self.t_c = plot_table[:i_stop,2]
        self.y_c = plot_table[:i_stop,3]
        self.l_h = plot_table[:i_stop,4]
        self.l_he= plot_table[:i_stop,5]
        self.m_  = plot_table[:i_stop,6]
        self.unknown = plot_table[:i_stop,7]
        self.t_eff  = np.log10(plot_table[:i_stop,8])
        # [print(i) for i in range(len(plot_table[:,8])) if plot_table[i,8]==0]
        self.l_     = plot_table[:i_stop,9]
        self.rho_c  = plot_table[:i_stop,10]
        self.l_carb = plot_table[:i_stop,11]
        self.l_nu  = plot_table[:i_stop,12]
        self.mdot_ = plot_table[:i_stop,13]
        self.t_max = plot_table[:i_stop,14]
        self.rho_at_t_max = plot_table[:i_stop,15]
        self.m_at_t_max = plot_table[:i_stop,16]


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

class Read_SM_data_File:
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
        full_name = name# + Read_SM_data_File.compart

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

    # def get_val(self, v_n, i = -1):
    #     if i > len(self.get_col(v_n)):
    #         sys.exit('\t__Error i {} > len(arr of v_n) {}'.format(i, v_n))
    #     return self.get_col(v_n)[i]


    # def get_lval_arr(self, i = -1):
    #     values = np.zeros((84,), dtype=float)
    #     for v_n in range(1, 84):
    #         values[v_n] = self.get_val(v_n, i)
    #
    #     return values
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

    def get_par_table(self, model, i = -1):


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


        return np.array([ self.mdot_[i],
                          self.xm_[i],
                          self.r_[i],
                          self.l_[i],
                          self.kappa_[i],
                          self.rho_[i],
                          self.t_[i],
                          Physics.mean_free_path(self.rho_[i], self.kappa_[i]),
                          self.u_[i],self.LLedd_[i],
                          Physics.opt_depth_par(i, self.rho_, self.kappa_, self.u_, self.r_, self.t_, self.mu_),
                          self.HP_[i], self.tau_[i], ])

    def get_set_of_cols(self, v_n_arr):
        res = np.zeros(( len(self.r_), len(v_n_arr) ))
        for i in range(len(v_n_arr)):
            res[:,i] = self.get_col(v_n_arr[i])

        print('\t__Note: vars:[', v_n_arr, '] returned arr:', res.shape)
        return res

    def get_spec_val(self, v_n):
        '''
        Available specific v_n:  'lm', 'Y_c'
        :param v_n:
        :return:
        '''
        if v_n == 'lm':
            return Physics.loglm(self.l_[-1], self.xm_[-1], False)

        if v_n == 'Y_c':
            return self.He4_[0]

        if v_n == '-': # mask value, if not nedeeded
            return 0

        return None

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
               lim_k1 = None, lim_k2 = None, lim_t1 = None, lim_t2 = None, it1 = None, it2 = None):

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

class Treat_Observables:
    def __init__(self, obs_files):

        self.files = obs_files
        self.n_of_fls = len(obs_files)

        self.obs = []
        for i in range(len(obs_files)):
            self.obs.append( Read_Observables(obs_files[i]) )

        if (len(obs_files)) > 1 :
            for i in range(1,len(self.files)):
                if not np.array_equal(self.obs[i-1].names, self.obs[i].names):
                    print('\t__Error. Files with observations contain different *names* row')
                    print('\t  {} has: {} \n\t  {} has: {} '
                          .format(obs_files[i-1], self.obs[i-1].names, obs_files[i], self.obs[i].names))
                    raise NameError

    def check_if_var_name_in_list(self, var_name):
        if var_name == 'lm' or var_name == 'ts': # special case for L/M and sonic temperature
            pass
        else:
            for i in range(self.n_of_fls):
                if var_name not in self.obs[i].names:
                    print('\n\t__Error. Variable:  {} is not in the list of names: \n\t  {} \n\t  in file: {}'
                          .format(var_name, self.obs[i].names, self.files[i]))
                    raise  NameError

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
                star_y_coord = [ Physics.loglm(self.obs[s].obs_par('log(L)', float)[i],
                                             self.obs[s].obs_par('M', float)[i]) ]
            else:
                star_y_coord = [ self.obs[s].obs_par(y_name, float)[i] ]


            # ---------------------------------------X-------------------------
            if x_name == 'ts':
                if not ts_arr.any() or not l_lm_arr.any() or not m_dot.any():
                    print('\t__Error. For ts to be evaluated for a star : *ts_arr, l_lm_arr, m_dot* to be provided')
                    raise ValueError

                x_y_coord = Physics.lm_mdot_obs_to_ts_lm(ts_arr, l_lm_arr, m_dot, star_y_coord[0],
                                                         self.obs[s].obs_par('log(Mdot)', float)[i],
                                                         i, lim_t1_obs, lim_t2_obs)
                if x_y_coord.any():
                    ts_ = np.append(ts_, x_y_coord[1, :])  # FOR linear fit
                    y_coord_ = np.append(y_coord_, x_y_coord[0, :])
                    star_x_coord =  x_y_coord[1, :]
                    star_y_coord =  x_y_coord[0, :]  # If the X coord is Ts the Y coord is overritten.

            else:
                star_x_coord = [ self.obs[s].obs_par(x_name, float)[i] ]

            if x_name == 'lm':
                star_x_coord = [ Physics.loglm(self.obs[s].obs_par('log(L)', float)[i],
                                             self.obs[s].obs_par('M', float)[i]) ]





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


        # -----------------------------------------------LINEAR FIT TO THE DATA-------------------------------------
        # ts_grid_y_grid = Math.line_fit(ts_, y_coord_)
        # plt.plot(ts_grid_y_grid[0, :], ts_grid_y_grid[1, :], '-.', color='blue')
        # np.delete(plotted_stars,1,0)
        plotted_stars  = np.delete(plotted_stars, 0, 0) # removing [0,0,0,] row
        plotted_labels = np.delete(plotted_labels, 0, 0)

        if plotted_stars.any():
            print('\n| Plotted Stras from Observ |')
            print('|  i  | {} | {}  | col |'.format(x_name, y_name))
            print('|-----|-----------|---------|')
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

class Treat_Numercials:

    def __init__(self, files):
        self.files = files
        self.n_of_files = len(files)

        self.mdl = []
        for i in range(self.n_of_files):
            self.mdl.append(Read_SM_data_File.from_sm_data_file(self.files[i]))
            # print(self.mdl[i].mdot_[-1])

        # self.nmdls = len(self.mdl)
        print('\t__Note: {} sm.data files has been uploaded.'.format(self.n_of_files))

    def ind_from_condition(self, cur_model, condition):
        '''

        :param cur_model: index of a model out of list of class instances that is now in the MAIN LOOP
        :param condition: 'sp' - for sonic point, 'last' for -1, or like 't=5.2' for point where temp = 5.2
        :return: index of that point
        '''
        if condition == 'last' or condition == '' :
            return -1

        if condition == 'sp': # Returns the i of the velocity that is >= sonic one. (INTERPOLATION would be better)
            return self.mdl[cur_model].sp_i()

        var_name = condition.split('=')[ 0] # for condition like 't = 5.2' separates t as a var in sm.file and
        var_value= condition.split('=')[-1]

        if var_name not in self.mdl[cur_model].var_names:                   # Checking if var_name is in list of names for SM files
            raise NameError('Var_name: {} is not in var_name list: \n\t {}'
                            .format(var_name, self.mdl[cur_model].var_names))

        arr = np.array( self.mdl[cur_model].get_col(var_name) ) # checking if var_value is in the column of var_name
        if var_value < arr.min() or var_value > arr.max() :
            raise ValueError('Given var_value={} is beyond {} range: ({}, {})'
                             .format(var_value,var_name,arr.min(),arr.max()))

        ind = -1
        for i in range(len(arr)): # searching for the next element, >= var_value. [INTERPOLATION would be better]
            if var_value >= arr[i]:
                ind = i
                break
        if ind == -1:
            raise ValueError('ind = -1 -> var_value is not found in the arr. | var_value={}, array range: ({}, {})'
                             .format(var_value, var_name,arr.min(), arr.max()))

        return ind



    def get_x_y_of_all_numericals(self, condition, x_name, y_name, var_for_label1, var_for_label2,
                                  ts_arr = np.empty(0,), l_lm_arr = np.empty(0,), mdot2d_arr = np.empty(0,),
                                  lim_t1_obs = None, lim_t2_obs = None):
        '''

        :param condition: 'sp' - for sonic point, 'last' for -1, or like 't=5.2' for point where temp = 5.2
        :param x_name: can be the sm.file car name (eg from the bec output list)
        :param y_name: can be the sm.file car name (eg from the bec output list)
        :param var_for_label1:
               var_for_label2: Same, but can me 'color' - ro return unique value from 1 to 9
        :return: np.array([ i , x , y , var_lbl1 , var_lbl2 ]) - set of coluns for each sm.file - one row
        '''
        model_stars1 = np.array([0., 0., 0., 0., 0.]) # for output i
        model_stars2 = np.array([0., 0., 0., 0., 0.])

        for i in range(self.n_of_files):
            x_coord = None
            y_coord = None

            i_req = self.ind_from_condition(i, condition)

            '''---------------------MAIN CYCLE-------------------'''
            #-------------------SETTING-COORDINATES------------------------
            if x_name in  self.mdl[i].var_names:
                x_coord = self.mdl[i].get_col(x_name)[i_req]
            if y_name in  self.mdl[i].var_names:
                y_coord = self.mdl[i].get_col(y_name)[i_req]

            if y_name == 'lm':
                y_coord = self.mdl[i].get_lm[i_req]

            if var_for_label1 == 'Y_c':
                add_data1 = self.mdl[i].get_col('He4')[0]
            else:
                add_data1 = self.mdl[i].get_col(var_for_label1)[i_req]

            if var_for_label2 == 'color':
                add_data2 = int(Math.get_0_to_max([i],9)[i]) # from 1 to 9 for plotting C+[1-9]
            else:
                add_data2 = self.mdl[i].get_col(var_for_label2)[i_req]

            #-------------------------_CASE TO INTERPOLATE MDOT -> ts InterpolateioN----------------
            if x_name == 'ts':
                if not ts_arr.any() or not l_lm_arr.any() or not mdot2d_arr.any():
                    raise ValueError('x_coord {} requires ts, l_lm_arr and mdot2arr to be interpolated'.format(x_name))


            if x_name == 'ts' and (y_name == 'l' or y_name == 'lm') and ts_arr.any() and l_lm_arr.any() and mdot2d_arr.any():
                p_mdot = self.mdl[i].mdot_[i_req]
                x_y_coord = Physics.lm_mdot_obs_to_ts_lm(ts_arr, l_lm_arr, mdot2d_arr, y_coord, p_mdot, i, lim_t1_obs, lim_t2_obs)

                if x_y_coord.any():
                    for j in range(len(x_y_coord[0, :])):

                            # plt.plot(x_y_coord[1, j], x_y_coord[0, j], marker='.', markersize=9, color=color)
                            # ax.annotate('m' + str(i), xy=(x_y_coord[1, j], x_y_coord[0, j]), textcoords='data')

                        model_stars1 = np.vstack(( model_stars1, np.array(( i, x_y_coord[1, j],
                                                                               x_y_coord[0, j],
                                                                               add_data1,
                                                                               add_data2  ))))
            else:
                if x_coord == None or y_coord == None:
                    raise ValueError('x_coord={} or y_coord={} is not obtained.'.format(x_coord,y_coord))

                model_stars1 = np.vstack((model_stars1, np.array((i, x_coord,
                                                                     y_coord,
                                                                     add_data1,
                                                                     add_data2  ))))


            # color = 'C' + str(int(i * 10 / self.n_of_files))
            # plt.plot(x_coord, y_coord, marker='.', markersize=9, color=color)
            # # label='Model {}: T_s {} , L/M {} , Mdot {}'.format(i, "%.2f" % p_t, "%.2f" % p_lm, "%.2f" % p_mdot))
            # ax.annotate(str(i), xy=(x_coord, y_coord), textcoords='data')

            # --------------------------SAME BUT USING Mdot TO GET SONIC TEMPERATURE (X-Coordinate)------------------------
            # p_mdot = self.mdl[i].mdot_[i_req]
            # x_y_coord = Physics.lm_mdot_obs_to_ts_lm(t, y_coord, m_dot, y_coord, p_mdot, i, lim_t1_obs, lim_t2_obs)
            # if x_y_coord.any():
            #     for j in range(len(x_y_coord[0, :])):
            #         plt.plot(x_y_coord[1, j], x_y_coord[0, j], marker='.', markersize=9, color=color)
            #         ax.annotate('m' + str(i), xy=(x_y_coord[1, j], x_y_coord[0, j]), textcoords='data')
            #         model_stars1 = np.vstack(
            #             (model_stars1, np.array((i, x_y_coord[1, j], x_y_coord[0, j], p_mdot, self.mdl[i].He4_[0]))))
            #
            #     model_stars2 = np.vstack((model_stars2, np.array(
            #         (i, x_coord, y_coord, p_mdot, self.mdl[i].He4_[0]))))  # for further printing

        # -------------------------PLOT FIT FOR THE NUMERICAL MODELS AND TABLES WITH DATA --------------------------------
        if model_stars1.any():
            print('\n| Models plotted by ts & lm |')
            print('\t| Conditon: {} |'.format(condition))
            print('|  i  | {} | {} | {} | {}  |'.format(x_name, y_name, var_for_label1, var_for_label2))
            print('|-----|------|------|-------|------|')
            # print(model_stars1.shape)
            for i in range(1, len(model_stars1[:, 0])):
                print('| {} | {} | {} | {} | {} |'.format("%3.f" % model_stars1[i, 0], "%.2f" % model_stars1[i, 1],
                                                          "%.2f" % model_stars1[i, 2], "%.2f" % model_stars1[i, 3],
                                                          "%.2f" % model_stars1[i, 4]))
        else:
            print('\t__Warning: No stars to Print. Coordinates are not obtained')


        return model_stars1

            # fit = np.polyfit(model_stars1[:, 1], model_stars1[:, 2], 3)  # fit = set of coeddicients (highest first)
            # f = np.poly1d(fit)
            # fit_x_coord = np.mgrid[(model_stars1[1:, 1].min() - 0.02):(model_stars1[1:, 1].max() + 0.02):100j]
            # plt.plot(fit_x_coord, f(fit_x_coord), '--', color='black', label='Model_fit')



        # if model_stars2.any():
        #     print('\n| Models plotted: lm & mdot |')
        #     print('|  i  | in_t |  {}  | m_dot | Y_c  |'.format(y_mode))
        #     print('|-----|------|------|-------|------|')
        #     for i in range(1, len(model_stars2[:, 0])):
        #         print('| {} | {} | {} | {} | {} |'.format("%3.f" % model_stars2[i, 0], "%.2f" % model_stars2[i, 1],
        #                                                   "%.2f" % model_stars2[i, 2], "%.2f" % model_stars2[i, 3],
        #                                                   "%.2f" % model_stars2[i, 4]))
        #
        #     fit = np.polyfit(model_stars2[:, 1], model_stars2[:, 2], 3)  # fit = set of coeddicients (highest first)
        #     f = np.poly1d(fit)
        #     fit_x_coord = np.mgrid[(model_stars2[1:, 1].min() - 0.02):(model_stars2[1:, 1].max() + 0.02):100j]
        #     plt.plot(fit_x_coord, f(fit_x_coord), '--', color='black', label='Model_fit')


    def get_x_y_z_arrays(self, n_of_model, x_name, y_name, z_name, var_for_label1, var_for_label2, var_for_label3):
        '''
        Returns                     0: [ var_for_label1  , var_for_label2 , var_for_label3 ]
        :param x_name:              1: [ x_coord[:]      , y_coord[:]       z_coord[:]     ]
        :param y_name:              2  [         :       ,         :                :      ]
        :param var_for_label1:                      and off it goes
        :param var_for_label2:
        :return:
        '''
        x_coord = None
        y_coord = None
        z_coord = None

        i_req = -1  # for the add_data, as a point
        '''---------------------MAIN CYCLE-------------------'''
        # -------------------SETTING-COORDINATES------------------------
        if x_name in self.mdl [n_of_model].var_names:
            x_coord = self.mdl[n_of_model].get_col(x_name)
        if y_name in self.mdl [n_of_model].var_names:
            y_coord = self.mdl[n_of_model].get_col(y_name)
        if z_name in self.mdl [n_of_model].var_names:
            z_coord = self.mdl[n_of_model].get_col(z_name)

        add_data1 = self.mdl[n_of_model].get_spec_val(var_for_label1) # if v_n = Y_c or lm
        add_data2 = self.mdl[n_of_model].get_spec_val(var_for_label2)
        add_data3 = self.mdl[n_of_model].get_spec_val(var_for_label3)

        if add_data1 == None:
            add_data1 = self.mdl[n_of_model].get_col(var_for_label1)[i_req] # noraml bec variables
        if add_data2 == None:
            add_data2 = self.mdl[n_of_model].get_col(var_for_label2)[i_req]
        if add_data3 == None:
            add_data3 = self.mdl[n_of_model].get_col(var_for_label3)[i_req]

        #
        # if var_for_label1 == 'Y_c':
        #     add_data1 = self.mdl[n_of_model].get_col('He4')[0]
        # if var_for_label1 == 'lm':
        #     add_data1 = self.mdl[n_of_model].get_lm_last()
        #
        # if add_data1 == None:
        #     add_data1 = self.mdl[n_of_model].get_col(var_for_label1)[i_req]
        #
        # if var_for_label2 == 'color':
        #     add_data2 = Math.get_0_to_max( [n_of_model], 9 )[n_of_model]  # from 1 to 9 for plotting C+[1-9]
        # if var_for_label2 == 'lm':
        #     add_data2 = self.mdl[n_of_model].get_lm_last()
        # if var_for_label2 == 'Y_c':
        #     add_data2 = self.mdl[n_of_model].get_col('He4')[0]
        #
        # if add_data2 == None:
        #     add_data2 = self.mdl[n_of_model].get_col(var_for_label2)[i_req]

        # if x_coord == None or y_coord == None:
        #     raise ValueError('x_coord={} or y_coord={} is not obtained.'.format(x_coord, y_coord))

        print(var_for_label3, add_data3)
        if len(x_coord) != len(y_coord) or len(x_coord) != len(z_coord):
            raise ValueError('x_coord and y_coord: \n\t {} \t\n {} have different shape.'.format(x_coord, y_coord))

        return np.vstack(( np.insert(x_coord, 0, add_data1), np.insert(y_coord, 0, add_data2),
                           np.insert(z_coord, 0, add_data3)  )).T


    def get_set_of_cols(self, v_n_arr, n_of_model):
        '''
        Returns v_n_arr * length of each column array, [:,0] - first var, and so on.
        :param v_n_arr:
        :param n_of_model:
        :return:
        '''
        return self.mdl[n_of_model].get_set_of_cols(v_n_arr)


    def get_sonic_vel_array(self, n_of_model):
        mu = self.mdl[n_of_model].get_col('mu')
        t  = self.mdl[n_of_model].get_col('t')
        return Physics.sound_speed(t, mu, True)

    def table(self):
        print(
            '\n'
            ' i'
            ' |  Mdot '
            '| Mass'
            '|  R/Rs  '
            '| L/Ls  '
            '| kappa  '
            '| l(Rho) '
            '| Temp  '
            '| mfp   '
            '| vel   '
            '| gamma  '
            '| tpar  '
            '|  HP   '
            '| log(C)  ')
        print('---|-------|-----|--------|-------|--------|--------|-------|------'
              '-|-------|--------|-------|-------|-------')
        for i in range(self.n_of_files):
            self.mdl[i].get_par_table(i)

class ClassPlots:


    def __init__(self, opal_file, smfls, obs_file, n_anal = 1000, load_lim_cases = False,
                 output_dir = '../data/output/', plot_dir = '../data/plots/'):
        '''
        :param path: './smdata/' filder with sm.datas
        :param smfls: '5d-6' name, extension sm.data added automatically
        :param opal_file: 'table1' in folder './opal' and extension '.data'
        :param n_anal: interpolation depth
        '''

        self.output_dir = output_dir
        self.plot_dir = plot_dir

        self.obs_files = obs_file
        self.num_files = smfls

        self.obs = Treat_Observables(self.obs_files)
        self.nums= Treat_Numercials(self.num_files)

        # --- INSTANCES
        self.opal    = OPAL_Interpol(opal_file, n_anal)
        self.tbl_anl = Table_Analyze(opal_file, n_anal, load_lim_cases, output_dir, plot_dir)

        # self.obs = None
        # if obs_file != None:
        #     self.obs = Read_Observables(obs_file[0])
        #
        # self.mdl = []
        # for i in range(len(smfls)):
        #     self.mdl.append(Read_SM_data_File.from_sm_data_file(smfls[i]))
        #     # print(self.mdl[i].mdot_[-1])
        #
        # self.nmdls = len(self.mdl)
        # print('\t__Note: {} sm.data files has been uploaded.'.format(self.nmdls))
        #
        # self.plfl = []


        self.mins = []

        # self.tbl_anl.delete()

    def xy_profile(self, v_n1, v_n2, var_for_label1, var_for_label2, sonic = -1):

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twiny()

        tlt = v_n2 + '(' + v_n1 + ') profile'
        plt.title(tlt, loc='left')

        for i in range(len(self.num_files)):
            res = self.nums.get_x_y_z_arrays( i, v_n1, v_n2, '-', var_for_label1, var_for_label2, '-')
            lbl = '{}:{} , {}:{}'.format(var_for_label1,'%.2f' % res[0,0],var_for_label2,'%.2f' % res[0,1])
            ax1.plot(res[1:, 0], res[1:, 1], '-', color='C' + str(Math.get_0_to_max([i], 9)[i]), label=lbl)
            ax1.plot(res[-1, 0], res[-1, 1], 'x', color='C' + str(Math.get_0_to_max([i], 9)[i]))
            ax1.annotate(str('%.2f' % res[0,0]), xy=(res[-1, 0], res[-1, 1]), textcoords='data')
            if sonic != -1 and sonic < len(self.num_files) and v_n2 == 'u':
                u_s = self.nums.get_sonic_vel_array(i)
                ax1.plot(res[1:, 0], u_s, '-', color='black')

        last_model_t = self.nums.get_set_of_cols(['t'], sonic )[:,0]
        n_tic_loc = []
        n_tic_lbl = []
        temps = [last_model_t[-1], 4.2, 4.6, 5.2, 6.2, last_model_t[0]]
        for t in temps:
            if t <= last_model_t[0] and t >= last_model_t[-1]:
                i = Math.find_nearest_index(last_model_t, t)
                n_tic_loc  = np.append(n_tic_loc, self.nums.get_set_of_cols([v_n1], sonic )[i,0])
                n_tic_lbl = np.append(n_tic_lbl, "%.1f" % t)

        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xticks(n_tic_loc)
        ax2.set_xticklabels(n_tic_lbl)
        ax2.set_xlabel('log(T)')

        ax1.set_xlabel(v_n1)
        ax1.set_ylabel(v_n2)

        ax1.grid(which='both')
        ax1.grid(which='minor', alpha=0.2)
        ax1.grid(which='major', alpha=0.2)

        ax1.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
        plot_name = self.plot_dir + v_n1 + '_vs_' + v_n2 + '_profile.pdf'
        plt.savefig(plot_name)
        plt.show()

        # col = []
        # for i in range(len(self.num_files)):
        #     col.append(i)
        #     res = self.nums.get_set_of_cols([v_n1,v_n2, 'xm','l', 'mdot'],i)
        #     lbl = 'M:' + str('%.2f' % res[-1,2]) + ' L:' + \
        #           str('%.2f' % res[-1,3]) + ' Mdot:' + \
        #           str('%.2f' % res[-1,4])
        #     ax1.plot(res[:,0],res[:,1], '-', color='C'+str(Math.get_0_to_max(col,9)[i]), label=lbl)
        #     ax1.plot(res[-1,0], res[-1,1], 'x', color='C'+str(Math.get_0_to_max(col,9)[i]))
        #     if sonic and v_n2 == 'u':
        #         u_s = self.nums.get_sonic_vel_array(i)
        #         ax1.plot(res[:,0], u_s, '-', color='black')


        # res = self.nums.get_set_of_cols([v_n1,v_n2, 'xm','l', 'mdot'], -1 )
        # n_tic_loc = []
        # n_tic_lbl = []
        # temps = [self.mdl[-1].t_[-1], 4.2, 4.6, 5.2, 6.2, self.mdl[-1].t_[0]]
        # for t in temps:
        #     if t <= self.mdl[-1].t_[0] and t >= self.mdl[-1].t_[-1]:
        #         i = Math.find_nearest_index(self.mdl[-1].t_, t)
        #         n_tic_loc  = np.append(n_tic_loc, self.mdl[-1].get_col(v_n1)[i])
        #         n_tic_lbl = np.append(n_tic_lbl, "%.1f" % t)
        #         # plt.axvline(x=self.mdl[-1].get_col(v_n1)[i], linestyle='dashed', color='black', label = 'log(T):{}'.format(t))
        #
        # ax2.set_xlim(ax1.get_xlim())
        # # ax2.set_xticks(n_tic_loc)
        # # ax2.set_xticklabels(n_tic_lbl)
        # ax2.set_xlabel('log(T)')





        # for i in range(len(self.num_files)):
        #     x = self.nums[i].get_col(v_n1)
        #     y = self.mdl[i].get_col(v_n2)
        #     color = 'C' + str(i)
        #
        #     lbl = 'M:' + str('%.2f' % self.mdl[i].get_col('xm')[-1]) + ' L:' + \
        #            str('%.2f' % self.mdl[i].get_col('l')[-1]) + ' Mdot:' + \
        #            str('%.2f' % self.mdl[i].get_col('mdot')[-1])
        #     ax1.plot(x, y, '-', color=color, label=lbl)
        #     ax1.plot(x[-1], y[-1], 'x', color=color)
        #
        # r_arr = []
        # for i in range(self.nmdls):
        #     r_arr = np.append(r_arr, self.mdl[i].get_col(v_n1)[-1] )
        # ind = np.where(r_arr == r_arr.max())
        # # print(int(ind[-1]))

        # if sonic:
        #     for i in range(self.nmdls):
        #         x = self.mdl[i].get_col(v_n1)
        #         t = self.mdl[i].get_col('t')
        #         mu = self.mdl[i].get_col('mu')
        #         ax1.plot(x, Physics.sound_speed(t, mu, True), '-', color='black')

        # ax1.set_xlabel(v_n1)
        # ax1.set_ylabel(v_n2)

        #---------------------------------------MINOR-TICKS-------------------------------
        # if lx1 != None and lx2 != None:
        #     plt.xlim(lx1, lx2)
        #
        # if ly1 != None and ly2 != None:
        #     plt.ylim(ly1, ly2)

        # ax1.grid(which='both')
        # ax1.grid(which='minor', alpha=0.2)
        # ax1.grid(which='major', alpha=0.2)

        # n_tic_loc = []
        # n_tic_lbl = []
        # temps = [self.mdl[-1].t_[-1], 4.2, 4.6, 5.2, 6.2, self.mdl[-1].t_[0]]
        # for t in temps:
        #     if t <= self.mdl[-1].t_[0] and t >= self.mdl[-1].t_[-1]:
        #         i = Math.find_nearest_index(self.mdl[-1].t_, t)
        #         n_tic_loc  = np.append(n_tic_loc, self.mdl[-1].get_col(v_n1)[i])
        #         n_tic_lbl = np.append(n_tic_lbl, "%.1f" % t)
        #         # plt.axvline(x=self.mdl[-1].get_col(v_n1)[i], linestyle='dashed', color='black', label = 'log(T):{}'.format(t))

        # ax2.set_xlim(ax1.get_xlim())
        # # ax2.set_xticks(n_tic_loc)
        # # ax2.set_xticklabels(n_tic_lbl)
        # ax2.set_xlabel('log(T)')

    def xyy_profile(self, v_n1, v_n2, v_n3, var_for_label1, var_for_label2, var_for_label3, edd_kappa = True, mdl_for_t_axis = 0):

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
            res = self.nums.get_x_y_z_arrays( i, v_n1, v_n2, v_n3, var_for_label1, var_for_label2, var_for_label3)

            color = 'C' + str(Math.get_0_to_max([i], 9)[i])
            lbl = '{}:{} , {}:{} , {}:{}'.format(var_for_label1,'%.2f' % res[0,0], var_for_label2,'%.2f' % res[0,1], var_for_label3,'%.2f' % res[0,2])
            ax1.plot(res[1:, 0], res[1:, 1], '-', color = color, label=lbl)
            ax1.plot(res[-1, 0], res[-1, 1], 'x', color = color )
            ax1.annotate(str('%.2f' % res[0,0]), xy=(res[-1, 0], res[-1, 1]), textcoords='data')

            if edd_kappa and v_n3 == 'kappa':
                k_edd = Physics.edd_opacity(self.nums.get_set_of_cols(['xm'], i)[-1], self.nums.get_set_of_cols(['l'], i)[-1])
                ax2.plot(ax1.get_xlim(), [k_edd, k_edd], color=color, label = 'Model: {}, k_edd: {}'.format(i, k_edd))

            ax2.plot(res[1:, 0], res[1:, 2], '--', color = color)
            ax2.plot(res[-1, 0], res[-1, 2], 'o',  color = color)

        ax3 = ax2.twiny() # for temp
        last_model_t = self.nums.get_set_of_cols(['t'], mdl_for_t_axis)[:, 0]
        n_tic_loc = []
        n_tic_lbl = []
        temps = [last_model_t[-1], 4.2, 4.6, 5.2, 6.2, last_model_t[0]]
        for t in temps:
            if t <= last_model_t[0] and t >= last_model_t[-1]:
                i = Math.find_nearest_index(last_model_t, t)
                n_tic_loc  = np.append(n_tic_loc, self.nums.get_set_of_cols([v_n1], mdl_for_t_axis)[i, 0])
                n_tic_lbl = np.append(n_tic_lbl, "%.1f" % t)

        ax3.set_xlim(ax1.get_xlim())
        ax3.set_xticks(n_tic_loc)
        ax3.set_xticklabels(n_tic_lbl)
        ax3.set_xlabel('log(T)')

        ax2.set_ylabel(v_n3, color='r')
        ax2.tick_params('y', colors='r')

        plt.title(tlt, loc='left')
        fig.tight_layout()
        ax1.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
        plot_name = self.plot_dir + v_n1 + '_' + v_n2 + '_' + v_n3 + '_profile.pdf'
        plt.savefig(plot_name)
        plt.show()


        # k_edd = Physics.edd_opacity(self.mdl[-1].xm_[-1], self.mdl[-1].l_[-1])
        # ax2.plot(ax1.get_xlim(), [k_edd,k_edd], c='black')
        # ----------------------------EDDINGTON OPACITY------------------------------------
        # ax2.plot(np.mgrid[x1.min():x1.max():100j], np.mgrid[edd_k:edd_k:100j], c='black')

        # for i in range(self.nmdls):
        #     x = self.mdl[i].get_col(v_n1)
        #     y = self.mdl[i].get_col(v_n3)
        #     color = 'C' + str(i)
        #
        #     lbl = 'M:' + str('%.2f' % self.mdl[i].get_col('xm')[-1]) + ' L:' + \
        #            str('%.2f' % self.mdl[i].get_col('l')[-1]) + ' Mdot:' + \
        #            str('%.2f' % self.mdl[i].get_col('mdot')[-1])
        #     ax2.plot(x, y, '--', color=color, label=lbl)
        #     ax2.plot(x[-1], y[-1], 'o', color=color)


        # ax2.set_ylabel(v_n3, color='r')
        # ax2.tick_params('y', colors='r')

        # plt.axvline(x=4.6, color='black', linestyle='solid', label='T = 4.6, He Op Bump')
        # plt.axvline(x=5.2, color='black', linestyle='solid', label='T = 5.2, Fe Op Bump')
        # plt.axvline(x=6.2, color='black', linestyle='solid', label='T = 6.2, Deep Fe Op Bump')

        # ax3 = ax2.twiny()
        # n_tic_loc = []
        # n_tic_lbl = []
        # temps = [self.mdl[-1].t_[-1], 4.2, 4.6, 5.2, 6.2, self.mdl[-1].t_[0]]
        # for t in temps:
        #     if t <= self.mdl[-1].t_[0] and t >= self.mdl[-1].t_[-1]:
        #         i = Math.find_nearest_index(self.mdl[-1].t_, t)
        #         n_tic_loc = np.append(n_tic_loc, self.mdl[-1].get_col(v_n1)[i])
        #         n_tic_lbl = np.append(n_tic_lbl, "%.1f" % t)
        #         # plt.axvline(x=self.mdl[-1].get_col(v_n1)[i], linestyle='dashed', color='black', label = 'log(T):{}'.format(t))

        # ax3.set_xlim(ax1.get_xlim())
        # ax3.set_xticks(n_tic_loc)
        # ax3.set_xticklabels(n_tic_lbl)
        # ax3.set_xlabel('log(T)')
        #
        # plt.title(tlt, loc='left')
        # # plt.ylim(-8.5, -4)
        # fig.tight_layout()
        # plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
        # plot_name = self.plot_dir + v_n1 + '_' + v_n2 + '_' + v_n3 + '_profile.pdf'
        # plt.savefig(plot_name)
        # plt.show()

    def plot_t_rho_kappa(self, t1, t2, rho1 = None, rho2 = None, n_int = 1000, plot_edd = True):
        # self.int_edd = self.tbl_anlom_OPAL_table(self.op_name, 1, n_int, load_lim_cases)


        t_k_rho = self.opal.interp_opal_table(t1, t2, rho1, rho2)

        t = t_k_rho[0, 1:]  # x
        rho = t_k_rho[1:, 0]  # y
        kappa = t_k_rho[1:, 1:]  # z

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

        #------------------------EDDINGTON-----------------------------------
        Table_Analyze.plot_k_vs_t = False # there is no need to plot just one kappa in the range of availability

        if plot_edd: #n_model_for_edd_k.any():
            for i in range(len(self.num_files)):  # self.nmdls
                res = self.nums.get_set_of_cols(['xm', 'l', 'He4'], i)
                k_edd = Physics.edd_opacity(res[-1, 0], res[-1, 1])
                # print(k_edd)

                n_model_for_edd_k = self.tbl_anl.interp_for_single_k(t1, t2, n_int, k_edd)
                x = n_model_for_edd_k[0, :]
                y = n_model_for_edd_k[1, :]
                color = 'black'
                lbl = 'Model:{}, k_edd:{}'.format(i,'%.2f' % 10**k_edd)
                plt.plot(x, y, '-.', color=color, label=lbl)
                plt.plot(x[-1], y[-1], 'x', color=color)

        Table_Analyze.plot_k_vs_t = True
        #----------------------DENSITY----------------------------------------

        for i in range(len(self.num_files)):
            res = self.nums.get_set_of_cols(['t', 'rho', 'He4', 'mdot'], i)
            # res = self.nums.get_x_y_z_arrays( i, 't', 'rho', '', 'l', 'mdot', '-')
            print(res.shape)
            # lbl = 'Model:{} , Yc:{} , mdot:{}'.format(i, 't','%.2f' % res[0,0], 'mdot','%.2f' % res[0,1])
            lbl = 'Model:{} , Yc:{} , mdot:{}'.format(i, '%.2f' % res[0,2], '%.2f' % res[0,3])
            plt.plot(res[:, 0], res[:, 1], '-', color='C' + str(Math.get_0_to_max([i], 9)[i]), label=lbl)
            plt.plot(res[-1, 0], res[-1, 1], 'x', color='C' + str(Math.get_0_to_max([i], 9)[i]))
            plt.annotate(str('%.2f' % res[0,2]), xy=(res[-1, 0], res[-1, 1]), textcoords='data')



        # for i in range(self.nmdls):
        #
        #     x = self.mdl[i].t_
        #     y = self.mdl[i].rho_
        #     color = color = 'C' + str(i)
        #
        #     lbl = 'M:' + str('%.2f' % self.mdl[i].get_col('xm')[-1]) + ' L:' + \
        #           str('%.2f' % self.mdl[i].get_col('l')[-1]) + ' Mdot:' + \
        #           str('%.2f' % self.mdl[i].get_col('mdot')[-1])
        #     plt.plot(x, y, '-', color=color, label=lbl)
        #     plt.plot(x[-1], y[-1], 'x', color=color)

        plt.legend(bbox_to_anchor=(1, 0), loc='lower right', ncol=1)
        name = self.plot_dir + 't_rho_kappa.pdf'
        plt.savefig(name)

        plt.show()

    def plot_t_mdot_lm(self, t1, t2, mdot1 = None, mdot2 = None, r_s = 1.):
        name = self.plot_dir + 't_mdot_lm_plot.pdf'

        # mdot1 = -6  # new way of setting limits to the plot!!!
        # mdot2 = -4

        if mdot1 == None:
            rho1 = None
        else:
            rho1 = Physics.mdot_rho(t1, mdot1, 0, r_s)

        if mdot2 == None:
            rho2 = None
        else:
            rho2 = Physics.mdot_rho(t2, mdot2, 0, r_s)

        # -------------------------------

        # op_int = opal.from_OPAL_table(self.op_name, 1000)
        t_k_rho = self.opal.interp_opal_table(t1, t2, rho1, rho2)



        t = t_k_rho[0, 1:]  # x
        rho = t_k_rho[1:, 0]  # y
        k = t_k_rho[1:, 1:]  # z

        t_s = t
        mdot = Physics.rho_mdot(t_s, rho, 1, r_s)
        lm = Physics.logk_loglm(k, 2)


        #-----------------------------------

        plt.figure()

        pl.xlim(t.min(), t.max())
        pl.ylim(mdot.min(), mdot.max())
        levels = [3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 5.0, 5.2]
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

        for i in range(self.nmdls):
            p_lm = Physics.loglm(self.mdl[i].l_[-1], self.mdl[i].xm_[-1], False)
            p_mdot = self.mdl[i].mdot_[-1]
            p_t = self.mdl[i].t_[-1]

            color = color = 'C' + str(i)
            plt.plot(p_t, p_mdot, marker='x', markersize=9, color=color,
                     label='Model {}: T_s {} , mdot {} , L/M: {}'.format(i, "%.2f" % p_t, "%.2f" % p_mdot, "%.2f" % p_lm))

        plt.legend(bbox_to_anchor=(1, 0), loc='lower right', ncol=1)

        plt.savefig(name)

        plt.show()


        #
        # t_c = t_coord[Math.find_nearest_index(t_coord, 5.2)]
        #
        #
        #
        # import re  # for searching the number in 'WN7-e' string, to plot them different colour
        # obs = Read_Observables()
        # for i in range(len(obs.numb)):
        #     s = re.search(r"\d+(\.\d+)?", obs.type[i])  # this is searching for the niumber
        #     color = 'C' + str(int(s.group(0)))  # making a colour our of C1 - C9 range
        #
        #     y_val = Physics.loglm(obs.l_obs[i], obs.m_obs[i])
        #
        #     if y_val > loglm[0] or y_val < loglm[-1]:
        #         print(
        #             '\t__Warning! Star: {} cannot be plotted: lm_star: {} not in lm_aval: ({}, {})'.format(obs.numb[i],
        #                                                                                                    "%.2f" % y_val,
        #                                                                                                    loglm[0],
        #                                                                                                    loglm[-1]))
        #     else:
        #
        #         i_lm = Math.find_nearest_index(loglm, y_val)
        #         t_coord = Math.solv_inter_row(t, m_dot[i_lm, :], obs.mdot_obs[i])
        #         Errors.is_arr_empty(t_coord, '|plot_observer|', True,
        #                             'y_val: {}, lm:({}, {})'.format("%.2f" % obs.mdot_obs[i],
        #                                                             "%.2f" % m_dot[i_lm, :].min(),
        #                                                             "%.2f" % m_dot[i_lm, :].max()))
        #
        #         t_c = t_coord[Math.find_nearest_index(t_coord, 5.2)]
        #         print("t_coord: {}, lm_c: {}, mdot_c: {}".format(t_coord, "%.2f" % y_val, obs.mdot_obs[i]))
        #
        #         plt.plot(t_c, y_val, marker='o', color=color, ls='')
        #         # label = str(obs.numb[i]) + ' ' + obs.type[i])
        #         ax.annotate(str(obs.numb[i]), xy=(t_c, y_val), textcoords='data')

    # def plot_t_lm_mdot(self, t1, t2, lm1, lm2, r_s_, n_int = 100, n_out = 100,
    #                    lim_t1_obs = None, lim_t2_obs = None):
    #
    #     # loglm1 = 3.8
    #     # loglm2 = 4.4
    #
    #     r_s = r_s_[0]
    #     if lm1 != None:
    #         k2 = Physics.loglm_logk(lm1)
    #     else:
    #         k2 = None
    #     if lm2 != None:
    #         k1 = Physics.loglm_logk(lm2)
    #     else:
    #         k1 = None
    #
    #
    #     # i_t1 = Math.find_nearest_index(t, t1)
    #     # i_t2 = Math.find_nearest_index(t, t2)
    #     print("Selected T range: ", t1, ' to ', t2)
    #     print("Selected k range: ", k1, ' to ', k2)
    #
    #     # ta = Table_Analyze.from_OPAL_table(self.op_name, n_out, n_int, load_lim_cases)
    #     res_ = self.tbl_anl.treat_tasks_interp_for_t(t1, t2, n_out, n_int, k1, k2)
    #
    #     kap = res_[0, 1:]
    #     t =   res_[1:, 0]
    #     rho2d = res_[1:, 1:]
    #
    #     # loglm = np.zeros((r_s_, ))
    #     # for i in range(len(r_s_)):
    #
    #     loglm = Physics.logk_loglm(kap, True)
    #     m_dot = Physics.rho_mdot(t, rho2d.T, 2, r_s)
    #
    #
    #
    #     print(t.shape, loglm.shape, m_dot.shape)
    #     #-------------------------------------------POLT-Ts-LM-MODT-COUTUR------------------------------------
    #     name = './results/t_LM_Mdot_plot.pdf'
    #
    #     fig = plt.figure()
    #     ax = fig.add_subplot(1, 1, 1)
    #
    #     pl.xlim(t.min(), t.max())
    #     pl.ylim(loglm.min(), loglm.max())
    #     levels = [-7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2]
    #     # levels = [-10, -9, -8, -7, -6, -5, -4]
    #     contour_filled = plt.contourf(t, loglm, m_dot, levels, cmap=plt.get_cmap('RdYlBu_r'))
    #     plt.colorbar(contour_filled)
    #     contour = plt.contour(t, loglm, m_dot, levels, colors='k')
    #     plt.clabel(contour, colors='k', fmt='%2.1f', fontsize=12)
    #     plt.title('MASS LOSS PLOT')
    #     plt.xlabel('Log(t)')
    #     plt.ylabel('log(L/M)')
    #
    #     #--------------------------------------------------PLOT-MINS-------------------------------------------
    #     self.mins = Math.get_mins_in_every_row(t, loglm, m_dot, 5000, 5.1, 5.3)
    #     plt.plot(self.mins[0, :], self.mins[1, :], '-.', color='red', label='min_Mdot')
    #
    #     #-----------------------------------------------PLOT-OBSERVABLES-----------------------------------
    #     types = []
    #     if self.obs != None: # plot observed stars
    #
    #         ''' Read the observables file and get the necessary values'''
    #         import re  # for searching the number in 'WN7-e' string, to plot them different colour
    #         ts_ = []
    #         lm_ = []
    #         for i in range(self.obs.num_stars):
    #             s = re.search(r"\d+(\.\d+)?", self.obs.obs_par('type', str)[i])  # this is searching for the niumber
    #             color = 'C' + str(int(s.group(0)))  # making a colour our of C1 - C9 range
    #
    #             star_lm = Physics.loglm(self.obs.obs_par('log(L)', float)[i], self.obs.obs_par('M', float)[i])
    #             # Effective T << T_s, that you have to get from mass loss!
    #             ts_lm = Physics.lm_mdot_obs_to_ts_lm(t, loglm, m_dot, star_lm, self.obs.obs_par('log(Mdot)',float)[i],
    #                                                  self.obs.obs_par('WR',int)[i], lim_t1_obs, lim_t2_obs)
    #             # ts_lm = np.vstack((star_lm, np.log10(obs.get_parms('T_*', float)[i])))
    #
    #             if ts_lm.any():
    #                 # print(ts_lm[1, :], ts_lm[0, :])
    #                 ts_ = np.append(ts_, ts_lm[1, :]) # FOR linear fit
    #                 lm_ = np.append(lm_, ts_lm[0, :])
    #                 # print(len(ts_lm[0,:]))
    #                 for j in range(len(ts_lm[0,:])): # plot every solution in the degenerate set of solutions
    #                     plt.plot(ts_lm[1, j], ts_lm[0, j], marker='^', color=color, ls='')
    #                     ax.annotate(self.obs.obs_par('WR',str)[i], xy=(ts_lm[1, j], ts_lm[0, j]), textcoords='data')
    #
    #                     if int(s.group(0)) not in types: # plotting the legent for unique class of stars
    #                         plt.plot(ts_lm[1, j], ts_lm[0, j], marker='^', color=color, ls='',
    #                                  label=self.obs.obs_par('type',str)[i])
    #                     types.append(int(s.group(0)))
    #
    #     # -----------------------------------------------LINEAR FIT TO THE DATA-------------------------------------
    #         ts_grid_lm_grid = Math.line_fit(ts_, lm_)
    #         plt.plot(ts_grid_lm_grid[0,:],ts_grid_lm_grid[1,:], '-.', color='blue')
    #
    #     #----------------------------------------------PLOT-NUMERICAL-MODELS-----------------------------
    #     # m_dots = ["%.2f" %  self.mdl[i].mdot_[-1] for i in range(self.nmdls)]
    #     # colors = Math.get_list_uniq_ints(m_dots)
    #     # print(m_dots)
    #     # print(colors)
    #
    #     for i in range(self.nmdls):
    #         p_lm = Physics.loglm(self.mdl[i].l_[-1], self.mdl[i].xm_[-1], False)
    #         p_mdot = self.mdl[i].mdot_[-1]
    #         p_t = self.mdl[i].t_[-1]
    #
    #
    #         color = 'C' + str(int(i*10/self.nmdls))
    #         plt.plot(p_t, p_lm, marker='.', markersize=9, color=color)
    #                  # label='Model {}: T_s {} , L/M {} , Mdot {}'.format(i, "%.2f" % p_t, "%.2f" % p_lm, "%.2f" % p_mdot))
    #         ax.annotate(str(i), xy=(p_t, p_lm), textcoords='data')
    #
    #
    #     plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
    #
    #     plt.savefig(name)
    #
    #     plt.show()
    #
    #     #===============================================================================================================
    #     #           Minimum Mass loss = f(L/M)
    #     #===============================================================================================================
    #     '''<<< Possible to treat multiple sonic radii >>>'''
    #
    #     plot_name = './results/Min_Mdot.pdf'
    #
    #
    #     fig = plt.figure()
    #     ax = fig.add_subplot(1, 1, 1)
    #
    #     plt.title('L/M = f(min M_dot)')
    #
    #     #--------------------------------------_PLOT MINS-------------------------------------------------
    #     for i in range(len(r_s_)):
    #         loglm_ = Physics.logk_loglm(kap, True)
    #         m_dot_ = Physics.rho_mdot(t, rho2d.T, 2, r_s_[i])
    #
    #         self.mins_ = Math.get_mins_in_every_row(t, loglm_, m_dot_, 5000, 5.1, 5.3)
    #
    #         min_mdot_arr_ = np.array(self.mins_[2, :])
    #         color = 'C' + str(i)
    #         plt.plot(min_mdot_arr_, loglm_, '-', color=color, label='min_Mdot for r_s: {}'.format(r_s_[i]))
    #
    #     #---------------------------------------ADJUST MAXIMUM L/M FOR OBSERVATIONS------------------------
    #
    #     min_mdot_arr = np.array(self.mins[2, :])
    #     plt.xlim(-6.0, min_mdot_arr.max())
    #
    #     if self.obs != None:
    #         star_lm = np.zeros(self.obs.num_stars)
    #         for i in range(self.obs.num_stars):
    #             star_lm[i] = Physics.loglm(self.obs.obs_par('log(L)', float)[i], self.obs.obs_par('M', float)[i])
    #
    #         plt.ylim(loglm.min(), star_lm.max())
    #
    #     plt.xlabel('log(M_dot)')
    #     plt.ylabel('log(L/M)')
    #
    #
    #     # major_xticks = np.array([-6.5,-6,-5.5,-5,-4.5,-4,-3.5])
    #     # minor_xticks = np.arange(-7.0,-3.5,0.1)
    #     #
    #     # major_yticks = np.array([3.8,3.9,4.0,4.1,4.2,4.3,4.4,4.5])
    #     # minor_yticks = np.arange(3.8, 4.5, 0.05)
    #
    #     # major_xticks = np.arange(min_mdot_arr.min(), min_mdot_arr.max()+0.1, (min_mdot_arr.max() - min_mdot_arr.min())/4)
    #     # minor_xticks = np.arange(min_mdot_arr.min(), min_mdot_arr.max(), (min_mdot_arr.max() - min_mdot_arr.min())/8)
    #     #
    #     # major_yticks = np.arange(lm_arr.min(), lm_arr.max() + 0.1, ((lm_arr.max() - lm_arr.min()) / 4))
    #     # minor_yticks = np.arange(lm_arr.min(), lm_arr.max(), ((lm_arr.max() - lm_arr.min()) / 8))
    #
    #     ax.grid(which='major', alpha=0.2)
    #
    #     plt.legend(bbox_to_anchor=(1, 1), loc='upper right', ncol=1)
    #
    #     #--------------------------------------PLOT-OBSERVABLES----------------------------------------
    #     x_data = []
    #     y_data = []
    #     types = []
    #     if self.obs != None:  # plot array of observed stars from Read_Observ()
    #         import re  # for searching the number in 'WN7-e' string, to plot them different colour
    #
    #         x_data = []
    #         y_data = [] # for linear fit as well
    #         for i in range(self.obs.num_stars):
    #             s = re.search(r"\d+(\.\d+)?", self.obs.obs_par('type', str)[i])  # this is searching for the niumber
    #             color = 'C' + str(int(s.group(0)))  # making a colour our of C1 - C9 range
    #
    #             x_data = np.append(x_data, self.obs.obs_par('log(Mdot)', float)[i])
    #             y_data = np.append(y_data, Physics.loglm(self.obs.obs_par('log(L)', float)[i], self.obs.obs_par('M', float)[i]) )
    #
    #             plt.plot(x_data[i], y_data[i], marker='^', color=color, ls='')  # plot dots
    #             # label = str(obs.numb[i]) + ' ' + obs.type[i])
    #             ax.annotate(self.obs.obs_par('WR', str)[i], xy=(x_data[i], y_data[i]), textcoords='data')  # plot names next to dots
    #
    #             if int(s.group(0)) not in types:  # plotting the legent for unique class of stars
    #                 plt.plot(x_data[i], y_data[i], marker='^', color=color, ls='',
    #                          label=self.obs.obs_par('type', str)[i])
    #             types.append(int(s.group(0)))
    #
    #     # -----------------------------------------------LINEAR FIT TO THE DATA-------------------------------------
    #         xy_line = Math.line_fit(x_data, y_data)
    #         plt.plot(xy_line[0,:], xy_line[1,:], '-.', color='blue')
    #
    #
    #     # ---------------------------------------PLOT MODELS ------------------------------------------
    #     for i in range(self.nmdls):
    #         y = Physics.loglm(self.mdl[i].l_[-1],self.mdl[i].xm_[-1])
    #         x = self.mdl[i].mdot_[-1]
    #
    #         color = 'C' + str(int(i * 10 / self.nmdls))
    #         plt.plot(x, y, marker='.', markersize=9, color=color)
    #             # label='Model {}: T_s {} , L/M {} , Mdot {}'.format(i, "%.2f" % p_t, "%.2f" % p_lm, "%.2f" % p_mdot))
    #         ax.annotate(str(i), xy=(x, y), textcoords='data')
    #
    #
    #     # ax.set_xticks(major_xticks)
    #     # ax.set_xticks(minor_xticks, minor=True)
    #     # ax.set_yticks(major_yticks)
    #     # ax.set_yticks(minor_yticks, minor=True)
    #
    #
    #     ax.grid(which='both')
    #     ax.grid(which='minor', alpha=0.2)
    #     ax.fill_between(min_mdot_arr, loglm, color="lightgray", label = 'Mdot < Minimun')
    #     plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
    #     plt.savefig(plot_name)
    #     plt.show()

    @staticmethod
    def get_min_mdot_set(r_s_, y_label, t, kap, rho2d, ):
        '''
        Permorms interpolation over a set of soinic rs, returning: Math.combine(r_s_, y_coord_, min_mdot_arr_.T)
        (Transposed, because initially in rho, the t - x.
        :param r_s_:
        :param y_label:
        :param t:
        :param kap:
        :param rho2d:
        :return:
        '''

        min_mdot_arr_ = np.zeros((len(r_s_), len(kap)))
        r_s_ = np.array(r_s_)

        if y_label == 'log(L)':
            y_coord_ = Physics.lm_to_l(Physics.logk_loglm(kap, True))
        else:
            y_coord_ = Physics.logk_loglm(kap, True)

        for i in range(len(r_s_)):
            m_dot_ = Physics.rho_mdot(t, rho2d.T, 2, r_s_[i])
            mins_ = Math.get_mins_in_every_row(t, y_coord_, m_dot_, 5000, 5.1, 5.3)
            min_mdot_arr_[i,:] = np.array(mins_[2, :])
            # color = 'C' + str(i)

        print('\t__Mdot_set_shape: -- {} r_s ; | {} l or lm ; |- {} mdot'.format(r_s_.shape, y_coord_.shape, min_mdot_arr_.T.shape))

        mm = np.flip(min_mdot_arr_.T ,0)
        ll = np.flip(y_coord_, 0)

        return Math.combine(r_s_, ll, mm)
        # return Math.combine(r_s_, y_coord_, min_mdot_arr_.T)


            # plt.plot(min_mdot_arr_[i,:], y_coord_, '-', color=color, label='min_Mdot for r_s: {}'.format(r_s_[i]))

    def plot_min_mdot(self, t, kap, rho2d, y_coord, y_label, r_s_):
        # ===============================================================================================================
        #           Minimum Mass loss = f(L/M)
        # ===============================================================================================================
        '''<<< Possible to treat multiple sonic radii >>>'''

        y_coord = np.flip(y_coord, 0)  # as you flip the mdot 2d array.

        plot_name = self.plot_dir+'minMdot_l.pdf'

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        plt.title('L or L/M = f(min M_dot)')

        # --------------------------------------_PLOT MINS-------------------------------------------------
        ClassPlots.get_min_mdot_set(r_s_,y_label,t,kap,rho2d)

        # min_mdot_arr_ = np.zeros((len(r_s_), len(y_coord)))  # Down are the r_s, -> are the min mass losses
        # for i in range(len(r_s_)):
        #
        #     if y_label == 'log(L)':
        #         y_coord_ = Physics.lm_to_l(Physics.logk_loglm(kap, True))
        #     else:
        #         y_coord_ = Physics.logk_loglm(kap, True)
        #
        #     m_dot_ = Physics.rho_mdot(t, rho2d.T, 2, r_s_[i])
        #
        #     mins_ = Math.get_mins_in_every_row(t, y_coord_, m_dot_, 5000, 5.1, 5.3)
        #
        #     min_mdot_arr_[i,:] = np.array(mins_[2, :])
        #     color = 'C' + str(i)
        #     plt.plot(min_mdot_arr_[i,:], y_coord_, '-', color=color, label='min_Mdot for r_s: {}'.format(r_s_[i]))

        min_mdot_0 = []
        for i in range(len(r_s_)):
            color = 'C' + str(i)
            min_mdot_arr_ = ClassPlots.get_min_mdot_set(r_s_,y_label,t,kap,rho2d)
            min_mdot_0 = min_mdot_arr_[1:,1]
            print(min_mdot_arr_[1:, 1].shape, min_mdot_arr_[1:,0].shape)

            plt.plot(min_mdot_arr_[1:, (1+i)], min_mdot_arr_[1:,0], '-', color=color, label='min_Mdot for r_s: {}'.format(r_s_[i]))
        # ---------------------------------------ADJUST MAXIMUM L/M FOR OBSERVATIONS------------------------

        plt.xlim(-6.0, min_mdot_0.max())

        if self.obs != None:
            star_y_coord = np.zeros(self.obs.num_stars)
            for i in range(self.obs.num_stars):
                if y_label == 'log(L)':
                    star_y_coord[i] = self.obs.obs_par('log(L)', float)[i]
                else:
                    star_y_coord[i] = Physics.loglm(self.obs.obs_par('log(L)', float)[i], self.obs.obs_par('M', float)[i])

            plt.ylim(y_coord.min(), star_y_coord.max())

        plt.xlabel('log(M_dot)')
        plt.ylabel(y_label)
        ax.grid(which='major', alpha=0.2)
        plt.legend(bbox_to_anchor=(1, 1), loc='upper right', ncol=1)

        # --------------------------------------PLOT-OBSERVABLES----------------------------------------
        obs = Treat_Observables(self.obs_files)
        res = obs.get_x_y_of_all_observables('mdot', 'l', 'type')

        for i in range(len(res[0][:, 1])):
            ax.annotate(int(res[0][i, 0]), xy=(res[0][i, 1], res[0][i, 2]), textcoords='data')  # plot numbers of stars
            plt.plot(res[0][i, 1], res[0][i, 2], marker='^', color='C' + str(int(res[0][i, 3])),
                     ls='')  # plot color dots)))
        # marker='s', mec='w', mfc='g', mew='3', ms=8
        for j in range(len(res[1][:, 0])):
            plt.plot(res[1][j, 1], res[1][j, 2], marker='^', color='C' + str(int(res[1][j, 3])), ls='',
                     label='WN'+str(int(res[1][j, 3])))

        x_grid_y_grid = Math.line_fit(res[0][:, 1], res[0][:, 2])
        plt.plot(x_grid_y_grid[0, :], x_grid_y_grid[1, :], '-.', color='blue')



        # types = []
        # if self.obs != None:  # plot array of observed stars from Read_Observ()
        #     import re  # for searching the number in 'WN7-e' string, to plot them different colour
        #
        #     x_data = []
        #     y_data = []  # for linear fit as well
        #     for i in range(self.obs.num_stars):
        #         s = re.search(r"\d+(\.\d+)?", self.obs.obs_par('type', str)[i])  # this is searching for the niumber
        #         color = 'C' + str(int(s.group(0)))  # making a colour our of C1 - C9 range
        #
        #         x_data = np.append(x_data, self.obs.obs_par('log(Mdot)', float)[i])
        #         if y_label == 'log(L)':
        #             y_data = np.append(y_data, self.obs.obs_par('log(L)', float)[i])
        #         else:
        #             y_data = np.append(y_data, Physics.loglm(self.obs.obs_par('log(L)', float)[i], self.obs.obs_par('M', float)[i]))
        #
        #         plt.plot(x_data[i], y_data[i], marker='^', color=color, ls='')  # plot dots
        #         # label = str(obs.numb[i]) + ' ' + obs.type[i])
        #         ax.annotate(self.obs.obs_par('WR', str)[i], xy=(x_data[i], y_data[i]),
        #                     textcoords='data')  # plot names next to dots
        #
        #         if int(s.group(0)) not in types:  # plotting the legent for unique class of stars
        #             plt.plot(x_data[i], y_data[i], marker='^', color=color, ls='',
        #                      label=self.obs.obs_par('type', str)[i])
        #         types.append(int(s.group(0)))
        #
        #     # -----------------------------------------------LINEAR FIT TO THE DATA-------------------------------------
        #     xy_line = Math.line_fit(x_data, y_data)
        #     plt.plot(xy_line[0, :], xy_line[1, :], '-.', color='blue')

        # ---------------------------------------PLOT MODELS ------------------------------------------
        sp_i = -1
        for i in range(self.nmdls):

            sp_v = Physics.sound_speed(self.mdl[i].t_, self.mdl[i].mu_)
            for k in range(len(sp_v)):
                if sp_v[k] <= self.mdl[i].u_[k]:
                    sp_i = k
                    # print('\t__Note: Last l: {} | sp_l {} '.format("%.3f" % self.mdl[i].l_[-1],
                    #                                                "%.3f" % self.mdl[i].l_[sp_i]))
                    # print('\t__Note: Last t: {} | sp_t {} '.format("%.3f" % self.mdl[i].t_[-1],
                    #                                                "%.3f" % self.mdl[i].t_[sp_i]))
                    break
            if sp_i == -1:
                print('Warning! Sonic Velocity is not resolved. Using -1 element f the u arrau.')

            if y_label == 'log(L)':
                y = self.mdl[i].l_[sp_i]
            else:
                y = Physics.loglm(self.mdl[i].l_[-1],self.mdl[i].xm_[-1])
            x = self.mdl[i].mdot_[-1]

            color = 'C' + str(int(i * 10 / self.nmdls))
            plt.plot(x, y, marker='.', markersize=9, color=color)
            # label='Model {}: T_s {} , L/M {} , Mdot {}'.format(i, "%.2f" % p_t, "%.2f" % p_lm, "%.2f" % p_mdot))
            ax.annotate(str(i), xy=(x, y), textcoords='data')

        ax.grid(which='both')
        ax.grid(which='minor', alpha=0.2)
        ax.fill_between(min_mdot_0, y_coord, color="lightgray", label='Mdot < Minimun')
        plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
        plt.savefig(plot_name)
        plt.show()

    @staticmethod
    def get_k1_k2_from_l1_l2(t1, t2, l1, l2):
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

    def plot_rs_l_mdot_min(self, y_name, t1, t2, l1, l2, r_s1, r_s2, n_int = 100, n_out = 100, n_r_s = 100, load = False):
        # ---------------------SETTING LM1 LM2 K1 K2---------------------------

        k1, k2 = ClassPlots.get_k1_k2_from_l1_l2(t1, t2, l1, l2)

        # ---------------------Getting KAPPA[], T[], RHO2D[]-------------------------
        res_ = self.tbl_anl.treat_tasks_interp_for_t(t1, t2, n_out, n_int, k1, k2)
        kap = res_[0, 1:]
        t = res_[1:, 0]
        rho2d = res_[1:, 1:]

        if y_name == 'l':
            y_mode = 'l '
            y_coord = Physics.lm_to_l(Physics.logk_loglm(kap, True))  # Kappa -> L/M -> L
            y_label = 'log(L)'
        else:
            y_mode = 'lm'
            y_coord = Physics.logk_loglm(kap, 1)
            y_label = 'log(L/M)'

        r_s_ = np.mgrid[r_s1:r_s2:n_r_s*1j]



        name_out = self.output_dir + 'rs_l_mdot.data'
        if load:
            min_mdot = np.loadtxt(name_out, dtype=float, delimiter=' ').T
        else:
            min_mdot = ClassPlots.get_min_mdot_set(r_s_, y_label, t, kap, rho2d)
            np.savetxt(name_out, min_mdot.T, delimiter=' ', fmt='%.3f')

        # sys.exit('stop: {}'.format(min_mdot[0,1:]))

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        #
        pl.xlim(min_mdot[0,1:].min(), min_mdot[0,1:].max())
        pl.ylim(min_mdot[1:,0].min(), min_mdot[1:,0].max())
        levels = [-7.5, -7.3, -7, -6.7, -6.5, -6.3, -6, -5.7, -5.5, -5.3, -5, -4.7, -4.5, -4.3, -4, -3.7, -3.5, -3.3, -3]
        # levels = [-10, -9, -8, -7, -6, -5, -4]
        contour_filled = plt.contourf(min_mdot[0,1:], min_mdot[1:,0], min_mdot[1:,1:], levels, cmap=plt.get_cmap('RdYlBu_r'))
        plt.colorbar(contour_filled)
        contour = plt.contour(min_mdot[0,1:], min_mdot[1:,0], min_mdot[1:,1:], levels, colors='k')
        plt.clabel(contour, colors='k', fmt='%2.1f', fontsize=12)
        plt.title('MINIMUM MASS LOSS RATE PLOT')
        plt.xlabel('r_s')


        obs = Treat_Observables(self.obs_files)
        res = obs.get_x_y_of_all_observables('ts', 'l', 'type', min_mdot[0,1:], min_mdot[1:,0], min_mdot[1:,1:])

        for i in range(len(res[0][:, 1])):
            ax.annotate(int(res[0][i, 0]), xy=(res[0][i, 1], res[0][i, 2]), textcoords='data')  # plot numbers of stars
            plt.plot(res[0][i, 1], res[0][i, 2], marker='^', ms=8,  color='C' + str(int(res[0][i, 3])),
                     ls='')  # plot color dots)))

        for j in range(len(res[1][:, 0])):
            plt.plot(res[1][j, 1], res[1][j, 2], marker='^', ms=8, color='C' + str(int(res[1][j, 3])), ls='',
                     label='WN'+str(int(res[1][j, 3])))


        x_grid_y_grid = Math.line_fit(res[0][:, 1], res[0][:, 2])
        plt.plot(x_grid_y_grid[0, :], x_grid_y_grid[1, :], '-.', color='blue')



        for i in range(self.nmdls):
            sp_v = Physics.sound_speed(self.mdl[i].t_, self.mdl[i].mu_)
            for k in range(len(sp_v)):
                if sp_v[k] <= self.mdl[i].u_[k]:
                    sp_i = k
                    break
            if sp_i == -1:
                print('Warning! Sonic Velocity is not resolved. Using -1 element f the u arrau.')
                # print('\t__Note: Last l: {} | sp_l {} '.format("%.3f" % self.mdl[i].l_[-1], "%.3f" % self.mdl[i].l_[sp_i]))
                # print('\t__Note: Last t: {} | sp_t {} '.format("%.3f" % self.mdl[i].t_[-1], "%.3f" % self.mdl[i].t_[sp_i]))

            mod_x_coord = self.mdl[i].r_[sp_i]
            if y_name == 'l':
                mod_y_coord = self.mdl[i].l_[sp_i]
            else:
                mod_y_coord = Physics.loglm(self.mdl[i].l_[sp_i], self.mdl[i].xm_[sp_i])

            color = 'C' + str(int(i*10/self.nmdls))
            plt.plot(mod_x_coord, mod_y_coord, marker='.', markersize=9, color=color)
                     # label='Model {}: T_s {} , L/M {} , Mdot {}'.format(i, "%.2f" % p_t, "%.2f" % p_lm, "%.2f" % p_mdot))
            ax.annotate(str(i), xy=(mod_x_coord, mod_y_coord), textcoords='data')


        name = self.plot_dir + 'rs_l_minMdot.pdf'
        plt.ylabel(y_label)
        plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
        plt.savefig(name)
        plt.show()

        #======================================================================

        # cl2 = Table_Analyze(name_out,1000,True)
        # # cl2.table_plotting(4.8,5.8)
        # rs_l_mimmdot = cl2.treat_tasks_tlim(1000, 0.52,2.0) # x = l
        #
        # print(rs_l_mimmdot)
        #
        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1)
        # #
        # # pl.xlim(rs_l_mimmdot[1:, 0].min(), rs_l_mimmdot[1:, 0].max())
        # # pl.ylim(rs_l_mimmdot[0, 1:].min(), rs_l_mimmdot[0, 1:].max())
        # # levels = [-7.5, -7.3, -7, -6.7, -6.5, -6.3, -6, -5.7, -5.5, -5.3, -5, -4.7, -4.5, -4.3, -4, -3.7, -3.5, -3.3,
        # #           -3]
        # # levels = [-10, -9, -8, -7, -6, -5, -4]
        # contour_filled = plt.contourf(rs_l_mimmdot[1:, 0], rs_l_mimmdot[0, 1:], rs_l_mimmdot[1:, 1:].T,
        #                               cmap=plt.get_cmap('RdYlBu_r'))
        # plt.colorbar(contour_filled)
        # contour = plt.contour(rs_l_mimmdot[1:, 0], rs_l_mimmdot[0, 1:], rs_l_mimmdot[1:, 1:].T, colors='k')
        # plt.clabel(contour, colors='k', fmt='%2.1f', fontsize=12)
        # plt.title('MINIMUM MASS LOSS RATE PLOT')
        # plt.xlabel('Log(r_s)')
        # plt.ylabel(y_label)
        #
        # # for i in range(self.nmdls):
        # #     sp_v = Physics.sound_speed(self.mdl[i].t_, self.mdl[i].mu_)
        # #     for k in range(len(sp_v)):
        # #         if sp_v[k] <= self.mdl[i].u_[k]:
        # #             sp_i = k
        # #             break
        # #     if sp_i == -1:
        # #         print('Warning! Sonic Velocity is not resolved. Using -1 element f the u arrau.')
        # #         # print('\t__Note: Last l: {} | sp_l {} '.format("%.3f" % self.mdl[i].l_[-1], "%.3f" % self.mdl[i].l_[sp_i]))
        # #         # print('\t__Note: Last t: {} | sp_t {} '.format("%.3f" % self.mdl[i].t_[-1], "%.3f" % self.mdl[i].t_[sp_i]))
        # #
        # #     mod_x_coord = self.mdl[i].r_[sp_i]
        # #     if y_name == 'l':
        # #         mod_y_coord = self.mdl[i].l_[sp_i]
        # #     else:
        # #         mod_y_coord = Physics.loglm(self.mdl[i].l_[sp_i], self.mdl[i].xm_[sp_i])
        # #
        # #     color = 'C' + str(int(i * 10 / self.nmdls))
        # #     plt.plot(mod_x_coord, mod_y_coord, marker='.', markersize=9, color=color)
        # #     # label='Model {}: T_s {} , L/M {} , Mdot {}'.format(i, "%.2f" % p_t, "%.2f" % p_lm, "%.2f" % p_mdot))
        # #     ax.annotate(str(i), xy=(mod_x_coord, mod_y_coord), textcoords='data')
        #
        # plt.show()

    def plot_t_l_mdot(self, y_name, t1, t2, y1, y2, r_s_, n_int = 100, n_out = 100,
                      lim_t1_obs = None, lim_t2_obs = None):
        # ---------------------SETTING LM1 LM2 K1 K2---------------------------

        k1, k2 = ClassPlots.get_k1_k2_from_l1_l2(t1, t2, y1, y2)

        #---------------------Getting KAPPA[], T[], RHO2D[]-------------------------
        res_ = self.tbl_anl.treat_tasks_interp_for_t(t1, t2, n_out, n_int, k1, k2)
        kap = res_[0, 1:]
        t =   res_[1:, 0]
        rho2d = res_[1:, 1:]

        if y_name == 'l':
            y_mode = 'l '
            l_lm_arr  = Physics.lm_to_l( Physics.logk_loglm(kap, True) ) # Kappa -> L/M -> L
            # y_label = 'log(L)'
        else:
            y_mode = 'lm'
            l_lm_arr = Physics.logk_loglm(kap, 1)
            y_label = 'log(L/M)'

        self.plot_min_mdot(t, kap, rho2d, l_lm_arr, y_name, r_s_)

        m_dot = Physics.rho_mdot(t, rho2d.T, 2, r_s_[0])

        mins = Math.get_mins_in_every_row(t, l_lm_arr, m_dot, 5000, 5.1, 5.3)

        print('\t__Note: PLOT: x: {}, y: {}, z: {} shapes.'.format(t.shape, l_lm_arr.shape, m_dot.shape))

        #-------------------------------------------POLT-Ts-LM-MODT-COUTUR------------------------------------
        name = self.plot_dir + 'rs_lm_minMdot_plot.pdf'

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        pl.xlim(t.min(), t.max())
        pl.ylim(l_lm_arr.min(), l_lm_arr.max())
        levels = [-7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2]
        contour_filled = plt.contourf(t, l_lm_arr, m_dot, levels, cmap=plt.get_cmap('RdYlBu_r'))
        plt.colorbar(contour_filled)
        contour = plt.contour(t, l_lm_arr, m_dot, levels, colors='k')
        plt.clabel(contour, colors='k', fmt='%2.1f', fontsize=12)
        plt.title('MASS LOSS PLOT')
        plt.xlabel('Log(t)')
        plt.ylabel(y_name)
        plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
        plt.savefig(name)

        #--------------------------------------------------PLOT-MINS-------------------------------------------

        plt.plot(mins[0, :], mins[1, :], '-.', color='red', label='min_Mdot (rs: {} )'.format(r_s_[0]))

        #-----------------------------------------------PLOT-OBSERVABLES-----------------------------------
        obs = Treat_Observables(self.obs_files)
        res = obs.get_x_y_of_all_observables('ts',y_name,'type',t,l_lm_arr,m_dot, lim_t1_obs, lim_t2_obs)

        for i in range(len( res[0][:, 1] )):
            ax.annotate(int(res[0][i, 0]), xy=(res[0][i, 1], res[0][i, 2]), textcoords='data') # plot numbers of stars
            plt.plot(res[0][i, 1], res[0][i, 2], marker='^', color='C' + str(int(res[0][i, 3])), ls='') # plot color dots)))

        for j in range(len(res[1][:, 0])):
            plt.plot(res[1][j, 1], res[1][j, 2], marker='^', color='C' + str(int(res[1][j, 3])), ls='',
                     label='WN'+str(int(res[1][j, 3])))

        x_grid_y_grid = Math.line_fit(res[0][:, 1], res[0][:, 2])
        plt.plot(x_grid_y_grid[0, :], x_grid_y_grid[1, :], '-.', color='blue')

        # ------------------------------------------------PLOT-NUMERICALS-----------------------------------

        nums = Treat_Numercials(self.num_files) # Surface Temp as a x coordinate
        res = nums.get_x_y_of_all_numericals('sp', 't', y_name, 'Y_c', 'color' ,t, l_lm_arr,m_dot, lim_t1_obs, lim_t2_obs)
        for i in range(len(res[:,0])):
            plt.plot(res[i, 1], res[i, 2], marker='.', color='C' + str(int(res[i, 4])), ls='')  # plot color dots)))
            ax.annotate(str("%.2f" % res[i, 3]), xy=(res[i, 1], res[i, 2]), textcoords='data')

        # nums = Treat_Numercials(self.num_files)   # Sonic Temp (interpol from Mdot, assuming the r_s) is x_coord
        res = nums.get_x_y_of_all_numericals('sp', 'ts', y_name, 'Y_c', 'color' ,t, l_lm_arr,m_dot, lim_t1_obs, lim_t2_obs)
        for i in range(len(res[:,0])):
            plt.plot(res[i, 1], res[i, 2], marker='.', color='C' + str(int(res[i, 4])), ls='')  # plot color dots)))
            ax.annotate(str("%.2f" % res[i, 3]), xy=(res[i, 1], res[i, 2]), textcoords='data')


        fit = np.polyfit(res[:, 1], res[:, 2], 3)  # fit = set of coeddicients (highest first)
        f = np.poly1d(fit)
        fit_x_coord = np.mgrid[(res[1:, 1].min() - 0.02):(res[1:, 1].max() + 0.02):100j]
        plt.plot(fit_x_coord, f(fit_x_coord), '--', color='black', label='Model_fit')


        #
        #
        #
        # # types = []
        # # plotted_stars = np.array([0., 0., 0., 0.])
        # # if self.obs != None: # plot observed stars
        # #     ''' Read the observables file and get the necessary values'''
        # #     import re  # for searching the number in 'WN7-e' string, to plot them different colour
        # #     ts_ = []
        # #     y_coord_ = []
        # #     for i in range(self.obs.num_stars):
        # #         if y_name == 'l':
        # #             star_y_coord = self.obs.obs_par('log(L)', float)[i]
        # #         else:
        # #             star_y_coord = Physics.loglm(self.obs.obs_par('log(L)', float)[i], self.obs.obs_par('M', float)[i])
        # #
        # #         # Effective T << T_s, that you have to get from mass loss!
        # #         ts_y_coord = Physics.lm_mdot_obs_to_ts_lm(t, l_lm_arr, m_dot, star_y_coord, self.obs.obs_par('log(Mdot)',float)[i],
        # #                                              self.obs.obs_par('WR',int)[i], lim_t1_obs, lim_t2_obs)
        # #
        # #         if ts_y_coord.any():
        # #             # print(ts_lm[1, :], ts_lm[0, :])
        # #             ts_ = np.append(ts_, ts_y_coord[1, :]) # FOR linear fit
        # #             y_coord_ = np.append(y_coord_, ts_y_coord[0, :])
        # #
        # #             # print(len(ts_lm[0,:]))
        # #
        # #             s = re.search(r"\d+(\.\d+)?", self.obs.obs_par('type', str)[i])  # this is searching for the niumber
        # #             color = 'C' + str(int(s.group(0)))  # making a colour our of C1 - C9 range
        # #
        # #             for j in range(len(ts_y_coord[0,:])): # plot every solution in the degenerate set of solutions
        # #                 plt.plot(ts_y_coord[1, j], ts_y_coord[0, j], marker='^', color=color, ls='')
        # #                 ax.annotate(self.obs.obs_par('WR',str)[i], xy=(ts_y_coord[1, j], ts_y_coord[0, j]), textcoords='data')
        # #
        # #                 # print( np.array((i, ts_y_coord[1, j], ts_y_coord[0, j], self.obs.obs_par('log(Mdot)',float)[i] )) )
        # #                 plotted_stars = np.vstack((plotted_stars, np.array((self.obs.obs_par('WR',int)[i], ts_y_coord[1, j], ts_y_coord[0, j], self.obs.obs_par('log(Mdot)',float)[i] )))) # for further printing
        # #
        # #                 if int(s.group(0)) not in types: # plotting the legent for unique class of stars
        # #                     plt.plot(ts_y_coord[1, j], ts_y_coord[0, j], marker='^', color=color, ls='',
        # #                              label=self.obs.obs_par('type',str)[i])
        # #                 types.append(int(s.group(0)))
        # #
        # #     # -----------------------------------------------LINEAR FIT TO THE DATA-------------------------------------
        # #     ts_grid_y_grid = Math.line_fit(ts_, y_coord_)
        # #     plt.plot(ts_grid_y_grid[0,:],ts_grid_y_grid[1,:], '-.', color='blue')
        # #
        # #
        # # print('\n| Plotted Stras from Observ |')
        # # print(  '|  i  |  t   |  {}  | m_dot |'.format(y_mode))
        # # print(  '|-----|------|------|-------|')
        # # for i in range(1, len(plotted_stars[:,0])):
        # #     print('| {} | {} | {} | {} |'.format("%3.f" % plotted_stars[i,0], "%.2f" % plotted_stars[i,1], "%.2f" %plotted_stars[i,2], "%.2f" %plotted_stars[i,3]))
        #
        # # ----------------------------------------------PLOT-NUMERICAL-MODELS-----------------------------
        # m_dots = ["%.2f" %  self.mdl[i].mdot_[-1] for i in range(self.nmdls)]
        # colors = Math.get_list_uniq_ints(m_dots)
        # # print(m_dots)
        # # print(colors)
        #
        # sp_i = -1
        #
        # model_stars1 = np.array([0., 0., 0., 0., 0.])
        # model_stars2 = np.array([0., 0., 0., 0., 0.])
        # for i in range(self.nmdls):
        #     sp_v = Physics.sound_speed(self.mdl[i].t_, self.mdl[i].mu_)
        #     for k in range(len(sp_v)):
        #         if sp_v[k] <= self.mdl[i].u_[k]:
        #             sp_i = k
        #             break
        #     if sp_i == -1:
        #         print('Warning! Sonic Velocity is not resolved. Using -1 element f the u arrau.')
        #         # print('\t__Note: Last l: {} | sp_l {} '.format("%.3f" % self.mdl[i].l_[-1], "%.3f" % self.mdl[i].l_[sp_i]))
        #         # print('\t__Note: Last t: {} | sp_t {} '.format("%.3f" % self.mdl[i].t_[-1], "%.3f" % self.mdl[i].t_[sp_i]))
        #
        #     mod_x_coord = self.mdl[i].t_[sp_i]
        #     if y_name == 'l':
        #         mod_y_coord = self.mdl[i].l_[sp_i]
        #     else:
        #         mod_y_coord = Physics.loglm(self.mdl[i].l_[sp_i], self.mdl[i].xm_[sp_i])
        #
        #     color = 'C' + str(int(i*10/self.nmdls))
        #     plt.plot(mod_x_coord, mod_y_coord, marker='.', markersize=9, color=color)
        #              # label='Model {}: T_s {} , L/M {} , Mdot {}'.format(i, "%.2f" % p_t, "%.2f" % p_lm, "%.2f" % p_mdot))
        #     ax.annotate(str(i), xy=(mod_x_coord, mod_y_coord), textcoords='data')
        #
        #
        #     #--------------------------SAME BUT USING Mdot TO GET SONIC TEMPERATURE (X-Coordinate)------------------------
        #     p_mdot = self.mdl[i].mdot_[sp_i]
        #     ts_y_model = Physics.lm_mdot_obs_to_ts_lm(t, l_lm_arr, m_dot, mod_y_coord, p_mdot, i, lim_t1_obs, lim_t2_obs)
        #     if ts_y_model.any():
        #         for j in range(len(ts_y_model[0, :])):
        #             plt.plot(ts_y_model[1, j], ts_y_model[0, j], marker='.', markersize=9, color=color)
        #             ax.annotate('m'+str(i), xy=(ts_y_model[1, j], ts_y_model[0, j]), textcoords='data')
        #             model_stars1 = np.vstack((model_stars1, np.array((i, ts_y_model[1, j], ts_y_model[0, j], p_mdot, self.mdl[i].He4_[0] ))))
        #
        #         model_stars2 = np.vstack((model_stars2, np.array((i, mod_x_coord, mod_y_coord, p_mdot, self.mdl[i].He4_[0]))))  # for further printing
        #
        #
        # # -------------------------PLOT FIT FOR THE NUMERICAL MODELS AND TABLES WITH DATA --------------------------------
        # if model_stars1.any():
        #     print('\n| Models plotted by ts & lm |')
        #     print(  '|  i  |  t   |  {}  | m_dot | Y_c  |'.format(y_mode))
        #     print(  '|-----|------|------|-------|------|')
        #     print(model_stars1.shape)
        #     for i in range(1, len(model_stars1[:,0])):
        #         print('| {} | {} | {} | {} | {} |'.format("%3.f" % model_stars1[i,0], "%.2f" % model_stars1[i,1], "%.2f" % model_stars1[i,2], "%.2f" %model_stars1[i,3], "%.2f" %model_stars1[i,4]))
        #
        #     fit = np.polyfit(model_stars1[:, 1], model_stars1[:, 2], 3)  # fit = set of coeddicients (highest first)
        #     f = np.poly1d(fit)
        #     fit_x_coord = np.mgrid[(model_stars1[1:, 1].min() - 0.02):(model_stars1[1:, 1].max() + 0.02):100j]
        #     plt.plot(fit_x_coord, f(fit_x_coord), '--', color='black', label='Model_fit')
        #
        # if model_stars2.any():
        #     print('\n| Models plotted: lm & mdot |')
        #     print(  '|  i  | in_t |  {}  | m_dot | Y_c  |'.format(y_mode))
        #     print(  '|-----|------|------|-------|------|')
        #     for i in range(1, len(model_stars2[:,0])):
        #         print('| {} | {} | {} | {} | {} |'.format("%3.f" % model_stars2[i,0], "%.2f" % model_stars2[i,1], "%.2f" % model_stars2[i,2], "%.2f" %model_stars2[i,3], "%.2f" %model_stars2[i,4]))
        #
        #     fit = np.polyfit(model_stars2[:,1], model_stars2[:,2], 3) # fit = set of coeddicients (highest first)
        #     f = np.poly1d(fit)
        #     fit_x_coord = np.mgrid[(model_stars2[1:,1].min()-0.02):(model_stars2[1:,1].max()+0.02):100j]
        #     plt.plot(fit_x_coord, f(fit_x_coord), '--', color='black', label='Model_fit')

        plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
        plt.savefig(name)
        plt.show()

    def hrd(self, t1,t2, observ_table_name, plot_file_names):
        plot_name = self.output_dir+'hrd.pdf'
        fig, ax = plt.subplots(1, 1)
        # from matplotlib.collections import LineCollection
        # from matplotlib.colors import ListedColormap, BoundaryNorm

        # for file_name in plot_file_names:
        #     self.plfl.append(Read_Plot_file.from_file(file_name))
        #
        # x_coord = []
        # y_coord = []
        # for i in range(len(plot_file_names)):
        #
        #     x_coord.append(self.plfl[i].t_eff)
        #     y_coord.append(self.plfl[i].l_)
        #
        #
        #
        # points = np.array([x_coord, y_coord]).T.reshape(-1, 1, 2)
        #     # points = np.append(p,  np.array([x_coord, y_coord]).T.reshape(-1, 1, 2) )
        # segments = np.concatenate([points[:-1], points[1:]], axis=1)
        #
        # # segments = np.append(s, np.concatenate([ points[:-1], points[1:] ], axis=1))
        #
        # z_coord = self.plfl[0].y_c
        # # Create a continuous norm to map from data points to colors
        # norm = plt.Normalize(z_coord.min(), z_coord.max())
        # lc = LineCollection(segments, cmap='viridis', norm=norm)
        # # Set the values used for colormapping
        # lc.set_array(z_coord)
        # lc.set_linewidth(2)
        # line = axs.add_collection(lc)
        # fig.colorbar(line, ax=axs)


        plt.title('HRD')
        plt.xlabel('log(T_eff)')
        plt.ylabel('log(L)')

        plt.xlim(t1, t2)
        ax.grid(which='major', alpha=0.2)
        plt.legend(bbox_to_anchor=(1, 1), loc='upper right', ncol=1)

        # n_in_type = []
        # x = []
        # y = []

        res = self.obs.get_x_y_of_all_observables('t', 'l', 'type')

        for i in range(len(res[0][:, 1])):
            ax.annotate(int(res[0][i, 0]), xy=(res[0][i, 1], res[0][i, 2]), textcoords='data')  # plot numbers of stars
            plt.plot(res[0][i, 1], res[0][i, 2], marker='^', color='C' + str(int(res[0][i, 3])),
                     ls='')  # plot color dots)))

        for j in range(len(res[1][:, 0])):
            plt.plot(res[1][j, 1], res[1][j, 2], marker='^', color='C' + str(int(res[1][j, 3])), ls='',
                     label='WN' + str(int(res[1][j, 3])))

        if observ_table_name != None:  # plot array of observed stars from Read_Observ()
            import re  # for searching the number in 'WN7-e' string, to plot them different colour
            obs = Read_Observables(observ_table_name)

            for i in range(obs.num_stars):
                s = re.search(r"\d+(\.\d+)?", obs.obs_par('type', str)[i])  # this is searching for the niumber
                # n_in_type.append(int(s.group(0)))
                color = 'C' + str(int(s.group(0)))  # making a colour our of C1 - C9 range

                x = np.append(x, np.log10(obs.obs_par('T_*', float)[i]*1000) )
                y = np.append(y, obs.obs_par('log(L)', float)[i] )
                # y = Physics.loglm(obs.obs_par('log(L)', float)[i], obs.obs_par('M', float)[i])

                plt.plot(x[i], y[i], marker='o', color=color, ls='')  # plot dots
                # label = str(obs.numb[i]) + ' ' + obs.type[i])
                axs.annotate(obs.obs_par('WR', str)[i], xy=(x[i], y[i]), textcoords='data')  # plot names next to dots

                if int(s.group(0)) not in n_in_type:  # plotting the legent for unique class of stars
                    plt.plot(x[i], y[i], marker='o', color=color, ls='',
                             label=obs.obs_par('type', str)[i])
                n_in_type.append(int(s.group(0)))

        # axs.set_ylim(  np.array(y_coord.min(), y.min()).min() , np.array(y_coord.max(), y.max()).max()  )

        # for i in range(self.nmdls):
        #     # p_y = Physics.loglm(self.mdl[i].l_[-1], self.mdl[i].xm_[-1], False)
        #     p_y = self.mdl[i].l_[-1]
        #     p_mdot = self.mdl[i].mdot_[-1]
        #     p_x = self.mdl[i].t_[-1]
        #
        #     mdot_color = []
        #     color = 'C0'
        #     if p_mdot not in mdot_color:
        #         color = 'C' + str(i)
        #     plt.plot(p_mdot, p_y, marker='x', markersize=9, color=color,
        #              label='Model {}: T_s {} , L/M {} , Mdot {}'.format(i, "%.2f" % p_x, "%.2f" % p_y,
        #                                                                 "%.2f" % p_mdot))

        # -------------------------------------------------------------------------Math.get_0_to_max()

        ind_arr = []
        for j in range(len(plot_file_names)):
            ind_arr.append(j)
            col_num = Math.get_0_to_max(ind_arr, 9)
            plfl = Read_Plot_file.from_file(plot_file_names[j])

            mod_x = plfl.t_eff
            mod_y = plfl.l_
            color = 'C' + str(col_num[j])

            plt.plot(mod_x, mod_y, '-', color=color, label=str("%.2f" % plfl.m_[0])+' to ' + str("%.2f" % plfl.m_[-1]) +' solar mass')
            for i in range(10):
                ind = Math.find_nearest_index(plfl.y_c, (i/10) )
                # print(plfl.y_c[i], (i/10))
                x_p = mod_x[ind]
                y_p = mod_y[ind]
                plt.plot(x_p, y_p, '.', color='red')
                ax.annotate("%.2f" % plfl.y_c[ind], xy=(x_p, y_p), textcoords='data')

            # from matplotlib.collections import LineCollection
            # from matplotlib.colors import ListedColormap, BoundaryNorm
            # fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
            #
            # # Create a continuous norm to map from data points to colors
            # norm = plt.Normalize(model.y_c.min(), model.y_c.max())
            # lc = LineCollection(segments, cmap='viridis', norm=norm)
            # # Set the values used for colormapping
            # lc.set_array(dydx)
            # lc.set_linewidth(2)
            # line = axs[0].add_collection(lc)
            # fig.colorbar(line, ax=axs[0])



        # ax.set_xticks(major_xticks)
        # ax.set_xticks(minor_xticks, minor=True)
        # ax.set_yticks(major_yticks)
        # ax.set_yticks(minor_yticks, minor=True)

        ax.grid(which='both')
        ax.grid(which='minor', alpha=0.2)

        # ax.fill_between(min_mdot_arr, loglm, color="lightgray", label='Mdot < Minimun')

        plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
        plt.savefig(plot_name)

        plt.show()

    def opacity_check(self, depth, t=None, rho=None, k = None, task = 'rho'):
        if self.nmdls == 0 and t == None and rho == None and k == None:
            sys.exit('\t__Error. No t and rho provided for the opacity to be calculated at. |opacity_check|')

        if task == 'rho':
            if t != None and k != None:
                Table_Analyze.plot_k_vs_t = False

                res = self.tbl_anl.interp_for_single_k(t, t, depth, k)

                print('For t: {} and k: {}, rho is {} and r is {}'
                      .format("%.3f" % res[0, 0], "%.3f" % k, "%.3f" % res[1, 0],
                              "%.3f" % Physics.get_r(res[0, 0], res[1, 0])))
                Table_Analyze.plot_k_vs_t = True
            else:
                if self.nmdls != 0:
                    t_arr = np.zeros(self.nmdls)
                    k_arr = np.zeros(self.nmdls)
                    n_rho_arr = np.zeros(self.nmdls)
                    i_rho_arr = np.zeros(self.nmdls)

                    Table_Analyze.plot_k_vs_t = False
                    for i in range(self.nmdls):
                        t_arr[i] = self.mdl[i].t_[-1]
                        k_arr[i] = self.mdl[i].kappa_[-1]
                        n_rho_arr[i] = self.mdl[i].rho_[-1]
                        i_rho_arr[i] = self.tbl_anl.interp_for_single_k(t_arr[i],t_arr[i], depth, k_arr[i])[1,0]

                    Table_Analyze.plot_k_vs_t = True

                    print('\n')
                    print('|   t   | kappa |  m_rho | in_rho | Error |')
                    print('|-------|-------|--------|--------|-------|')
                    for i in range(self.nmdls):

                        print('| {} | {} | {} | {} | {} |'.
                              format("%.3f" % t_arr[i], "%.3f" % 10**k_arr[i], "%.3f" % n_rho_arr[i], "%.3f" % i_rho_arr[i],
                                     "%.3f" % np.abs(n_rho_arr[i] - i_rho_arr[i]) ) )


        if task == 'kappa' or task == 'k':
            if t != None and rho != None:
                pass # DOR FOR GIVEN T AND RHO (OPAL_interpolate)
            else:
                if self.nmdls != 0:
                    pass # DO FOR THE SET OF NUMERICAL MODELS




        if t != None and rho == None and k != None:
            Table_Analyze.plot_k_vs_t = False
            res = self.tbl_anl.interp_for_single_k(t,t, depth, k)

            print('For t: {} and k: {}, rho is {} and r is {}'
                  .format("%.3f" % res[0,0], "%.3f" % k, "%.3f" % res[1,0], "%.3f" % Physics.get_r(res[0,0], res[1,0])))
            Table_Analyze.plot_k_vs_t = True

    def table_of_plot_files(self, flnames, descript):


        discr = Read_Observables(descript, '', '')

        #
        # if len(flnames) != len(descript):
        #     sys.exit('\t__Error. Number of: flnames {} , descript {} , must be the same |table_of_plot_files|'
        #              .format(flnames,descript))

        print(discr.obs_par_row(0, str))
        print(discr.names)

        plfl = []
        for file in flnames:
            plfl.append( Read_Plot_file.from_file(file, '') )


        # print('| i | File | Var_name | Value | T[-1] | L[-1] | R[-1] | Y_c[-1] ')
        # for i in range(len(plfl)):
        #     print('| {} | {} | {} | {} | {} | {} |  |  |'.format(i, flnames[i], descript[i], plfl[i].t_eff[-1], plfl[i].l_[-1]))

        print(discr.names)
        for i in range(discr.num_stars):
            print(discr.obs_par_row(i, str),
                  '| {} | {} |'.format("%.3f" % plfl[i].t_eff[-1], "%.3f" % plfl[i].l_[-1]))

class New_Table:
    def __init__(self, path, tables, values, out_dir_name):
        self.plot_dir_name = out_dir_name
        self.tbls = []
        self.val = values
        if len(values)!=len(tables):
            sys.exit('\t___Error. |New_table, init| n of tables and values is different: {} != {}'.format(len(tables), len(values)))
        self.ntbls = 0
        for i in range(len(tables)):
            self.tbls.append(Read_Table(path + tables[i]))
            self.ntbls = self.ntbls + 1


        print('\t__Note: {} opal tables files has been uploaded.'.format(self.ntbls))

    def check_if_opals_same_range(self):
        for i in range(self.ntbls-1):

            if not np.array_equal(self.tbls[i].r, self.tbls[i+1].r):
                sys.exit('\t___Error. Arrays *r* are not equal '
                         '|check_if_opals_same_range| \n r[{}]: {} \n r[{}]: {}'.format(i, self.tbls[i].r, i+1, self.tbls[i+1].r))

            if not np.array_equal(self.tbls[i].t, self.tbls[i+1].t):
                sys.exit('\t___Error. Arrays *t* are not equal '
                         '|check_if_opals_same_range| \n t[{}]: {} \n t[{}]: {}'.format(i, self.tbls[i].r, i+1, self.tbls[i+1].r))


        # for i in range(len(self.tbls[0].r)):
        #     pass

    def get_new_opal(self, value, mask = 9.999):
        if value > self.val[-1] or value < self.val[0]:
            sys.exit('\t___Error. |get_new_opal| value {} is not is range of tables: ({}, {})'.format(value, self.val[0],self.val[-1]))

        self.check_if_opals_same_range()

        rows = len(self.tbls[0].t)
        cols = len(self.tbls[0].r)

        new_kappa = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):

                val_aval = []
                k_row = []
                for k in range(self.ntbls):
                    k_val = self.tbls[k].kappas[i,j]
                    if k_val != mask:
                        k_row = np.append(k_row, self.tbls[k].kappas[i,j])
                        val_aval = np.append(val_aval, self.val[k])

                if len(val_aval) == 0:
                    new_kappa[i, j] = mask
                else:
                    if value >= val_aval[0] and value <= val_aval[-1]:
                        new_kappa[i, j] = Math.interp_row(val_aval, k_row, value)
                    else:
                        new_kappa[i, j] = mask

        # "%.2f" %
        fname = self.plot_dir_name
        res = Math.combine(self.tbls[0].r,self.tbls[0].t, new_kappa)
        np.savetxt(fname,res,'%.3f','\t')
        return res

