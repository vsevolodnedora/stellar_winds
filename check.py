
import sys
import pylab as pl
from matplotlib import cm
import numpy as np
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

# ======================================================================================================================
class Labels:
    def __init__(self):
        pass

    @staticmethod
    def lbls(v_n):
        #solar
        if v_n == 'l':
            return r'$\log(L)$'#(L_{\odot})

        if v_n == 'r':
            return r'$R(R_{\odot})$'

        if v_n == 'm' or v_n == 'xm':
            return r'$M(M_{\odot})$'

        #sonic and general
        if v_n == 'v' or v_n == 'u':
            return 'v (km/s)'

        if v_n == 'rho':
            return r'$\log(\rho)$'

        if v_n == 'k' or v_n == 'kappa':
            return r'$\kappa$'

        if v_n == 't':
            return r'log(T)'

        if v_n == 'ts':
            return r'$\log(T_{s})$'

        if v_n == 'lm':
            return r'$\log(L/M)$'

        if v_n == 'mdot':
            return r'$\log(\dot{M}$)'

        if v_n == 'Yc' or v_n == 'yc':
            return r'$^{4}$He$_{core}$'

        if v_n == 'He4':
            return r'$^{4}$He$_{surf}$'

        if v_n == 'Ys' or v_n == 'ys':
            return r'$^{4}$He$_{surf}$'

        if v_n == 't_eff' or v_n == 'T_eff':
            return r'$\log($T$_{eff})$'

        if v_n == 'rho':
            return r'$\log(\rho)$'

        if v_n == 'tau':
            return r'$\tau$'


        if v_n == 'Pr':
            return r'$P_{rad}$'

        if v_n == 'Pg':
            return r'$P_{gas}$'

        if v_n == 'Pg/P_total':
            return r'$P_{gas}/P_{total}$'

        if v_n == 'Pr/P_total':
            return r'$P_{rad}/P_{total}$'


        if v_n == 'mfp':
            return r'$\log(\lambda)$'

        if v_n == 'HP' or v_n == 'Hp':
            return r'$H_{p}$'

        if v_n == 'L/Ledd':
            return r'$L/L_{Edd}$'

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
    def combine(x, y, xy, corner_val = None):
        '''creates a 2d array  1st raw    [0, 1:]-- x -- density      (log)
                               1st column [1:, 0] -- y -- lemperature (log)
                               Matrix     [1:,1:] -- xy --Opacity     (log)
           0th element in 1st raw (column) - can be used a corner value

        '''
        x = np.array(x)
        y = np.array(y)
        xy = np.array((xy))

        if len(x) != len(y):
            print('\t__Warning. x({}) != y({}) (combine)'.format(len(x), len(y)))
        if len(x) != len(xy[0, :]):
            raise ValueError('\t__Warning. x({}) != xy[0, :]({}) (combine)'.format(len(x), len(xy[0, :])))
        if len(y) != len(xy[:, 0]):
            raise ValueError('\t__Warning. y({}) != xy[:, 0]({}) (combine)'.format(len(y), len(xy[:, 0])))

        res = np.insert(xy, 0, x, axis=0)
        new_y = np.insert(y, 0, 0, axis=0)  # inserting a 0 to a first column of a
        res = np.insert(res, 0, new_y, axis=1)

        if corner_val != None:
            res[0, 0] = corner_val

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

            red_arr_y = np.array(red_arr_y, dtype=np.float)
            arr_x = np.array(arr_x, dtype=np.float)



            f = interpolate.InterpolatedUnivariateSpline(arr_x, red_arr_y)
            # print("y_arr:({} to {}), can't find find: y_val {} .".format("%.2f"%arr_y[0],"%.2f"%arr_y[-1],"%.2f"%val))
            # f = interpolate.UnivariateSpline(arr_x, red_arr_y, s = 0)
            return f.roots() # x must be ascending to get roots()!

    @staticmethod
    def interp_row(x_arr, y_arr, new_x_arr):
        '''
            Uses 1d spline interpolation to give set of values new_y for provided
            cooednates x and y and new coordinates x_new (s - to be 0)
        '''
        # print(x_arr)
        f = interpolate.InterpolatedUnivariateSpline(x_arr, y_arr)
        # f = interpolate.interp1d(x_arr, y_arr, kind='cubic')


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
        Return np.vstack((x_points, y_arr, values))
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
            raise ValueError('*from_* is too small. {} < {} '.format(x1, x_arr[0]))

        if x2 > x_arr[-1]:
            raise ValueError('*from_* is too big. {} > {} '.format(x2, x_arr[-1]))

        new_x = np.mgrid[x1:x2:depth*1j]

        for i in range(len(z2d_arr[:, 0])):

            new_y = Math.interp_row(x_arr, z2d_arr[i, :], new_x)
            x_points[i] = new_x[ new_y.argmin() ]
            values[i] = new_y.min()

        return np.vstack((x_points, y_arr, values))

    @staticmethod
    def get_maxs_in_every_row(x_arr, y_arr, z2d_arr, depth, from_=None, to_=None):
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

        x1 = x_arr[0]  # default values (all the x range)
        x2 = x_arr[-1]

        if from_ != None:
            x1 = from_

        if to_ != None:
            x2 = to_

        if x1 < x_arr[0]:
            sys.exit('\t__Error. *from_* is too small. {} < {} '.format(x1, x_arr[0]))

        if x2 > x_arr[-1]:
            sys.exit('\t__Error. *from_* is too big. {} > {} '.format(x2, x_arr[-1]))

        new_x = np.mgrid[x1:x2:depth * 1j]

        for i in range(len(z2d_arr[:, 0])):
            new_y = Math.interp_row(x_arr, z2d_arr[i, :], new_x)
            x_points[i] = new_x[new_y.argmin()]
            values[i] = new_y.max()

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
    def get_max_by_interpolating(x, y, crop_rising_end=True, guess=None):

        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)

        def crop_ends(x, y):
            '''
            In case of 'wierd' vel/temp profile with rapidly rising end, first this rising part is to be cut of
            before maximum can be searched for.
            :param x:
            :param y:
            :return:
            '''

            x_mon = x
            y_mon = y

            non_monotonic = True

            while non_monotonic:

                if len(x_mon) <= len(x)*0.9: # criteria for rising ends
                    return x, y
                    # raise ValueError('Whole array is removed in a searched for monotonic part.')

                if y_mon[-1] > y_mon[-2]:
                    y_mon = y_mon[:-1]
                    x_mon = x_mon[:-1]
                    # print(x_mon[-1], y_mon[-1])
                else:
                    non_monotonic = False

            return x_mon, y_mon

        if crop_rising_end:
            x, y = crop_ends(x, y)

        # create the interpolating function
        print(len(x), len(y))
        f = interpolate.interp1d(x, y, kind='cubic', bounds_error=False)

        # to find the maximum, we minimize the negative of the function. We
        # cannot just multiply f by -1, so we create a new function here.
        f2 = interpolate.interp1d(x, -1.*y, kind='cubic', bounds_error=False)

        # x_new = np.mgrid[x[0]:x[-1]:1000j]
        # y_new = f2(x_new)
        # print(y_new)

        # print(x.min(), x.max())
        if guess == None:
            guess = x[np.where(y == y.max())]
        # guess = x[np.where(y == y.max())]
        # print(guess)
        print('Gues:', guess)

        x_max = optimize.fmin(f2, guess)


        return x_max, f(x_max)

        # xfit = np.linspace(0, 4)

    @staticmethod
    def invet_to_ascending_xy(d2array):
        x = np.array(d2array[0, 1:])
        y = np.array(d2array[1:, 0])
        z = np.array(d2array[1:, 1:])

        if x[0] > x[-1]:
            print('\t__Note: Inverting along X axis')
            x = x[::-1]
            z = z.T
            z = z[::1]
            z = z.T

        if y[0] > y[-1]:
            print('\t__Note: Inverting along Y axis')
            y = y[::-1]
            z = z[::-1]

        print(x.shape, y.shape, z.shape)
        return Math.combine(x, y, z)

    @staticmethod
    def crop_2d_table(table, x1 ,x2 ,y1, y2):
        x = table[0,1:]
        y = table[1:,0]
        z = table[1:,1:]

        if x1 != None:
            if x[0] > x[-1]:
                raise ValueError('x[0]({}) > x[-1]({}) Consider inverting the axis'.format(x[0],x[-1]))
            if x1 > x[-1]:
                raise ValueError('x1({}) > x[-1]({})'.format(x1, x[-1]))
            if x1 < x[0]:
                raise ValueError('x1({}) < x[0]({})'.format(x1, x[0]))

            ix1 = Math.find_nearest_index(x, x1)
            x = x[ix1:]
            z = z[:, ix1:]

        if x2 != None:
            if x[0] > x[-1]:
                raise ValueError('x[0]({}) > x[-1]({}) Consider inverting the axis'.format(x[0],x[-1]))
            if x2 > x[-1]:
                raise ValueError('x2({}) > x[-1]({})'.format(x2, x[-1]))
            if x2 < x[0]:
                raise ValueError('x2({}) < x[0]({})'.format(x2, x[0]))

            ix2 = Math.find_nearest_index(x,x2)
            x = x[:ix2 + 1]
            z = z[:, :ix2 + 1]

        if y1 != None:
            if y[0] > y[-1]:
                raise ValueError('x[0]({}) > x[-1]({}) Consider inverting the axis'.format(y[0],y[-1]))
            if y1 > y[-1]:
                raise ValueError('y1({}) > y[-1]({})'.format(y1, y[-1]))
            if y1 < y[0]:
                raise ValueError('y1({}) < y[0]({})'.format(y1, y[0]))

            iy1 = Math.find_nearest_index(y, y1)
            y = y[iy1: ]
            z = z[iy1:, :]

        if y2 != None:
            if y2 > y[-1]:
                raise ValueError('y2({}) > y[-1]({})'.format(y2, y[-1]))
            if y2 < y[0]:
                raise ValueError('y1({}) < y[0]({})'.format(y2, y[0]))
            if y[0] > y[-1]:
                raise ValueError('x[0]({}) > x[-1]({}) Consider inverting the axis'.format(y[0],y[-1]))

            iy2 = Math.find_nearest_index(y, y2)
            y = y[:iy2+1]
            z = z[:iy2+1, : ]



        print(x.shape, y.shape, z.shape)
        return Math.combine(x,y,z)

    @staticmethod
    def crop_2d_table2(table, x1, x2, y1, y2):
        x = table[0, 1:]
        y = table[1:, 0]
        z = table[1:, 1:]

        if x1 != None:
            if x[0] > x[-1]:
                raise ValueError('x[0]({}) > x[-1]({}) Consider inverting the axis'.format(x[0], x[-1]))
            if x1 > x[-1]:
                raise ValueError('x1({}) > x[-1]({})'.format(x1, x[-1]))


            ix1 = Math.find_nearest_index(x, x1)
            x = x[ix1:]
            z = z[:, ix1:]

        if x2 != None:
            if x[0] > x[-1]:
                raise ValueError('x[0]({}) > x[-1]({}) Consider inverting the axis'.format(x[0], x[-1]))
            if x2 < x[0]:
                raise ValueError('x2({}) < x[0]({})'.format(x2, x[0]))

            ix2 = Math.find_nearest_index(x, x2)
            x = x[:ix2 + 1]
            z = z[:, :ix2 + 1]

        if y1 != None:
            if y[0] > y[-1]:
                raise ValueError('x[0]({}) > x[-1]({}) Consider inverting the axis'.format(y[0], y[-1]))
            if y1 > y[-1]:
                raise ValueError('y1({}) > y[-1]({})'.format(y1, y[-1]))


            iy1 = Math.find_nearest_index(y, y1)
            y = y[iy1:]
            z = z[iy1:, :]

        if y2 != None:

            if y2 < y[0]:
                raise ValueError('y1({}) < y[0]({})'.format(y2, y[0]))
            if y[0] > y[-1]:
                raise ValueError('x[0]({}) > x[-1]({}) Consider inverting the axis'.format(y[0], y[-1]))

            iy2 = Math.find_nearest_index(y, y2)
            y = y[:iy2 + 1]
            z = z[:iy2 + 1, :]

        print(x.shape, y.shape, z.shape)
        return Math.combine(x, y, z)

    @staticmethod
    def fit_plynomial(x, y, order, depth, new_x = np.empty(0, )):
        '''
        RETURNS new_x, f(new_x)
        :param x:
        :param y:
        :param order: 1-4 are supported
        :return:
        '''
        f = None
        lbl = None

        if not new_x.any():
            new_x = np.mgrid[(x.min()):(x.max()):depth * 1j]


        if order == 1:
            fit = np.polyfit(x, y, order)  # fit = set of coeddicients (highest first)
            f = np.poly1d(fit)
            lbl = '({}) + ({}*x)'.format(
                "%.3f" % f.coefficients[1],
                "%.3f" % f.coefficients[0]
            )
            # fit_x_coord = np.mgrid[(x.min()):(x.max()):depth*1j]
            # plt.plot(fit_x_coord, f(fit_x_coord), '--', color='black')

        if order == 2:
            fit = np.polyfit(x, y, order)  # fit = set of coeddicients (highest first)
            f = np.poly1d(fit)
            lbl = '({}) + ({}*x) + ({}*x**2)'.format(
                "%.3f" % f.coefficients[2],
                "%.3f" % f.coefficients[1],
                "%.3f" % f.coefficients[0]
            )
            # fit_x_coord = np.mgrid[(x.min()):(x.max()):depth*1j]
            # plt.plot(fit_x_coord, f(fit_x_coord), '--', color='black')
        if order == 3:
            fit = np.polyfit(x, y, order)  # fit = set of coeddicients (highest first)
            f = np.poly1d(fit)
            lbl = '({}) + ({}*x) + ({}*x**2) + ({}*x**3)'.format(
                "%.3f" % f.coefficients[3],
                "%.3f" % f.coefficients[2],
                "%.3f" % f.coefficients[1],
                "%.3f" % f.coefficients[0]
            )
            # fit_x_coord = np.mgrid[(x.min()):(x.max()):depth*1j]
            # plt.plot(fit_x_coord, f(fit_x_coord), '--', color='black')
        if order == 4:
            fit = np.polyfit(x, y, order)  # fit = set of coeddicients (highest first)
            f = np.poly1d(fit)
            lbl = '({}) + ({}*x) + ({}*x**2) + ({}*x**3) + ({}*x**4)'.format(
                "%.3f" % f.coefficients[4],
                "%.3f" % f.coefficients[3],
                "%.3f" % f.coefficients[2],
                "%.3f" % f.coefficients[1],
                "%.3f" % f.coefficients[0]
            )
            # fit_x_coord = np.mgrid[(x.min()):(x.max()):depth*1j]
            # plt.plot(fit_x_coord, f(fit_x_coord), '--', color='black')

        if not order in [1,2,3,4]:
            fit = np.polyfit(x, y, order)  # fit = set of coeddicients (highest first)
            f = np.poly1d(fit)
            # raise ValueError('Supported orders: 1,2,3,4 only')

        print(lbl)

        return new_x, f(new_x)

    @staticmethod
    def x_y_z_sort(x_arr, y_arr, z_arr=np.empty(0,), sort_by_012=0):
        '''
        RETURNS x_arr, y_arr, (z_arr) sorted as a matrix by a row, given 'sort_by_012'
        :param x_arr:
        :param y_arr:
        :param z_arr:
        :param sort_by_012:
        :return:
        '''

        if not z_arr.any() and sort_by_012 < 2:
            if len(x_arr) != len(y_arr):
                raise ValueError('len(x)[{}]!= len(y)[{}]'.format(len(x_arr), len(y_arr)))

            x_y_arr = []
            for i in range(len(x_arr)):
                x_y_arr = np.append(x_y_arr, [x_arr[i], y_arr[i]])

            x_y_sort = np.sort(x_y_arr.view('float64, float64'), order=['f{}'.format(sort_by_012)], axis=0).view(
                np.float)
            x_y_arr_shaped = np.reshape(x_y_sort, (int(len(x_y_sort) / 2), 2))
            return x_y_arr_shaped[:, 0], x_y_arr_shaped[:, 1]

        if z_arr.any():
            if len(x_arr) != len(y_arr) or len(x_arr) != len(z_arr):
                raise ValueError('len(x)[{}]!= len(y)[{}]!=len(z_arr)[{}]'.format(len(x_arr), len(y_arr), len(z_arr)))

            x_y_z_arr = []
            for i in range(len(x_arr)):
                x_y_z_arr = np.append(x_y_z_arr, [x_arr[i], y_arr[i], z_arr[i]])

            x_y_z_sort = np.sort(x_y_z_arr.view('float64, float64, float64'), order=['f{}'.format(sort_by_012)],
                                 axis=0).view(
                np.float)
            x_y_z_arr_shaped = np.reshape(x_y_z_sort, (int(len(x_y_z_sort) / 3), 3))
            return x_y_z_arr_shaped[:, 0], x_y_z_arr_shaped[:, 1], x_y_z_arr_shaped[:, 2]

    @staticmethod
    def common_y(arr1, arr2):

        y1 = arr1[1:, 0]
        y2 = arr2[1:, 0]

        y_min = np.array([y1.min(), y2.min()]).max()
        y_max = np.array([y1.max(), y2.max()]).min()

        arr1_cropped = Math.crop_2d_table(arr1, None, None, y_min, y_max)
        arr2_cropped = Math.crop_2d_table(arr2, None, None, y_min, y_max)

        return arr1_cropped, arr2_cropped

    @staticmethod
    def extrapolate(table, x_left, x_right, y_down, y_up, depth, pol_order):
        '''
        Performs row by row and column by column interpolation, using polynomial of order 'pol_order'
        x_left-y_up are in percentage to extrapolate in a direction left-up. (if None or 0 returns same array)
        :param table:
        :param x_left:  in % to extend left
        :param x_right: in % to right
        :param y_down:  in % to down
        :param y_up:    in % to up
        :return:
        '''

        if x_left == None and x_right == None and y_down == None and y_up == None:
            return table

        x = table[0, 1:]
        y = table[1:, 0]
        z = table[1:,1:]

        if x_left != None:
            x1 = x.min() - (x_left * (x.max() - x.min()) / 100)     #
        else:
            x1 = x.min()

        if x_right != None:
            x2 = x.max() + (x_right * (x.max() - x.min()) / 100)    #
        else:
            x2 = x.max()

        if y_down != None:
            y1 = y.min() - (y_down * (y.max() - y.min()) / 100)     #
        else:
            y1 = y.min()

        if y_up != None:
            y2 = y.max() + (y_up * (y.max() - y.min()) / 100)       #
        else:
            y2 = y.max()


        def int_pol(x_arr, y_arr, x_grid):

            if x_grid.min() == x_arr.min() and x_arr.max() == x_grid.max():
                return interpolate.InterpolatedUnivariateSpline(x_arr, y_arr)(x_grid)
            else:
                if pol_order in [1,2,3,4]:
                    new_x, new_y = Math.fit_plynomial(x_arr, y_arr, pol_order, depth, x_grid)
                    return new_y
                if pol_order == 'unispline':
                    new_y = interpolate.UnivariateSpline(x_arr, y_arr)(x_grid)
                    return new_y

        x_grid = np.mgrid[x1:x2:depth*1j]
        z_y = np.zeros(len(x_grid))
        for i in range(len(y)):
            z_y = np.vstack((z_y, int_pol(x, z[i, :], x_grid)))

        z_y = np.delete(z_y, 0, 0)

        y_grid = np.mgrid[y1:y2:depth*1j]
        z_x = np.zeros(len(y_grid))
        for i in range(len(x_grid)):
            z_x = np.vstack((z_x, int_pol(y, z_y[:, i], y_grid)))

        z_x = np.delete(z_x, 0, 0).T

        res = Math.combine(x_grid, y_grid, z_x)
        res[0, 0] = table[0, 0]

        return res

    @staticmethod
    def get_z_for_yc_and_y(yc_value, yc_y_z, y_inp, dimension=0):
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

        # name = '{}_{}_{}'.format('yc', y_v_n, z_v_n)
        # yc_x_y = Save_Load_tables.load_table(name, 'yc', y_v_n, z_v_n, opal_used)
        x_arr = yc_y_z[1:, 0]
        yc_arr = yc_y_z[0, 1:]
        z2d = yc_y_z[1:, 1:]

        # print()

        # yc_value = np.float("%.3f" % yc_value)

        if yc_value in yc_arr:
            ind_yc = Math.find_nearest_index(yc_arr, yc_value)
        else:
            raise ValueError(
                'Table: {} Given yc_arr({}) is not in available yc_arr:({})'.format(name, yc_value, yc_arr))

        if dimension == 0:
            if y_inp >= x_arr.min() and y_inp <= x_arr.max():

                y_arr = z2d[:, ind_yc]
                # lm_arr = []
                # for i in range(len(y_arr)):
                #     lm_arr = np.append(lm_arr, [x_arr[i], y_arr[i]])
                #
                # lm_arr_sort = np.sort(lm_arr.view('float64, float64'), order=['f0'], axis=0).view(np.float)
                # lm_arr_shaped = np.reshape(lm_arr_sort, (len(y_arr), 2))

                f = interpolate.InterpolatedUnivariateSpline(x_arr, y_arr)
                y = f(y_inp)
                # print(log_l, y)
                # print('Yc: {}, y_row({} - {}), y_res: {}'.format(yc_value, y_arr.min(), y_arr.max(), y))
                return y_inp, y
            else:
                raise ValueError(
                    'Given l({}) not in available range of l:({}, {})'.format(y_inp, x_arr.min(), x_arr.max()))

        if dimension == 1:
            x_arr_f = []
            y_arr_f = []
            for i in range(len(y_inp)):
                if y_inp[i] >= x_arr.min() and y_inp[i] <= x_arr.max():
                    f = interpolate.UnivariateSpline(x_arr, z2d[:, ind_yc])
                    y_arr_f = np.append(x_arr_f, f(y_inp[i]))
                    x_arr_f = np.append(x_arr_f, y_inp[i])
                # else:
                #     raise ValueError(
                #         'Given x({}, {}) not in available range of x:({}, {})'.format(x_inp[0], x_inp[-1], x_arr.min(), x_arr.max()))

            return x_arr_f, y_arr_f

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
    def logk_loglm(logk, dimensions = 0, coeff = 1.0):
        '''
        k_opal = coeff * k_edd; k_edd = 4*pi*c*G*M / L
        For logk = -0.026 -> log(l/m) = 4.141
        :param logk:
        :return:
        '''
        if dimensions == 1:
            res = np.zeros(len(logk))
            for i in range(len(logk)):
                res[i] = np.log10(1 / (10 ** logk[i])) + np.log10(coeff * Constants.c_k_edd)
            return res
        if dimensions == 0:
            return np.log10(1 / (10 ** logk)) + np.log10(coeff * Constants.c_k_edd)

        if dimensions == 2:
            res = np.zeros(( len(logk[:,0]), len(logk[0,:] )))
            for i in range(len(logk[:,0])):
                for j in range(len(logk[0,:])):
                    res[i, j] = np.log10(1 / (10 ** logk[i,j])) + np.log10(coeff * Constants.c_k_edd)
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
            lm1 = Physics.l_to_lm_langer(l1)
        lm2 = None
        if l2 != None:
            lm2 = Physics.l_to_lm_langer(l2)

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


    # --- --- --- --- --- --- --- MDOT --- --- --- --- --- --- ---
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

        # if vrho is 2darray and r_s is a 2d array - r_s_for_t_l_vrho = 'tl' to change columns and rows of vrho

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

        if r_s_for_t_l_vrho == 'l' or r_s_for_t_l_vrho == 'lm':  # ---r_s = 1darray
            if len(r_s) != len(vrho[:, 0]): raise ValueError('len(r_s)={}!=len(vrho[:, 0])={}'.format(len(r_s), len(vrho[:, 0])))
            for i in range(len(vrho[:, 0])):
                mdot[i, :] = vrho[i, :] + c + np.log10(r_s[i] ** 2)

        if r_s_for_t_l_vrho == 't' or r_s_for_t_l_vrho == 'ts':  # ---r_s = 1darray
            if len(r_s) != len(vrho[0, :]): raise ValueError('len(r_s)={}!=len(vrho[0, :])={}'.format(len(r_s), len(vrho[0, :])))
            for i in range(len(vrho[0, :])):
                mdot[:, i] = vrho[:, i] + c + np.log10(r_s[i] ** 2)

        if r_s_for_t_l_vrho == 'tl':  # ---------------------REQUIRED r_s = 2darray
            if r_s.shape != vrho.shape: raise ValueError('r_s.shape {} != vrho.shape {}'.format(r_s.shape, vrho.shape))
            cols = len(vrho[0, :])
            rows = len(vrho[:, 0])
            mdot = np.zeros((rows, cols))

            for i in range(rows):
                for j in range(cols):
                    mdot[i, j] = vrho[i, j] + c + np.log10(r_s[i, j] ** 2)

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
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---


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
    def opt_depth_par2(rho_, t_, r_, u_, kap_, mu_):

        u_ = u_ * 100000    # return to cm/s (cgs)
        r_ = r_ * Constants.solar_r # return to cm
        # t = 10**t_          # return to kelvin
        # rho=10**rho_        # return to cgs

        t_mean =   10**t_[-1]   #+ np.abs((10**t_[-2] - 10**t_[-1]) / 2)
        rho_mean = 10**rho_[-1] #+ np.abs((10**rho_[-2] - 10**rho_[-1]) / 2)
        kap_mean = 10**kap_[-1] #+ np.abs((10**kap_[-2] - 10**kap_[-1]) / 2)
        mu_mean = mu_[-1] #+ np.abs((mu_[-2] - mu_[-1]) / 2)

        def therm_vel(t):
            '''
            Computes classical thermal velocity, where
            :param t:
            :return: v_th in km/s (!)
            '''

            return (np.sqrt( 2 * Constants.k_b*(t) / (mu_mean * Constants.m_H)))

        du = u_[-1] - u_[-2]
        dr = r_[-1] - r_[-2]
        drdu = dr/du

        xh = 0. # hydrogen fraction in the envelope
        kap_es = 0.2 * (1 + xh) # cm^2 / g
        v_th = therm_vel(t_mean)

        return kap_mean * v_th * rho_mean * drdu

        pass

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
    def steph_boltz_law_t_eff(logl, r):
        '''
        The law: L = 4 pi R^2 sigma T^4 => (L/4 \pi R^2)^1/4
        :return:
        '''
        l = (10**logl)*Constants.solar_l # in cgs
        r = r*Constants.solar_r # in cgs

        return 0.25*np.log10( l/(4*np.pi * Constants.steph_boltz * r**2))

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

    # --- --- --- LANGER --- --- ---
    @staticmethod
    def lm_to_l_langer(log_lm):
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
    def l_to_m_langer(log_l):
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
    def m_to_l_langer(log_m):
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
    def l_to_lm_langer(log_l):
        '''
        From Langer 1987 paper Mass Lum relation for WNE stars
        :param log_l:
        :return: log_lm
        '''
        a1 = 2.357485
        b1 = 3.407930
        c1 = -0.654431
        a2 = -0.158206
        b2 = -0.053868
        c2 = 0.055467
        return (-a2 -(b2 -1)*log_l - c2*(log_l**2) )
    # --- --- --- ---- ---- --- ---

    @staticmethod
    def ind_of_yc(yc_arr, yc_val):

        if not yc_val in yc_arr:
            raise ValueError('Value yc_vals[{}] not in yc:\n\t {}'.format(yc_val, yc_arr))
        else:
            yc_ind = Math.find_nearest_index(yc_arr, yc_val)

        return yc_ind

    # --- --- --- --- MASS LOSS PRESCRIPTIONS --- --- ---

    @staticmethod
    def l_mdot_prescriptions(log_l, log_z, author):

        def nugis_lamers(log_l, log_z):
            '''
            L -> mdot, evolution, Nugis & Lamers 2000 with (z-dependance from Vink & Koter 2005)
            :param z:
            :return:
            '''
            return 1.63 * log_l + 0.86 * np.log10(log_z) - 13.6

        def yoon(log_l, log_z):
            '''
            L -> Mdot Yoon 2017
            :param log_l:
            :param z:
            :return:
            '''
            return 1.18 * log_l + 0.60 * np.log10(log_z) - 11.32

        def my(log_l, log_z):
            if log_z == 10 ** 0.02:
                return 7.87*log_l - 48.18

        # --- --- --- --- --- --- ---



        if author in ['nugis', 'lamers', 'nugis and lamers']:
            return nugis_lamers(log_l, log_z)
        if author == 'yoon':
            return yoon(log_l, log_z)
        if author == 'my':
            return my(log_l, log_z)



    # --- --- --- --- --- --- --- --- --- -- --- --- ----

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
        self.eleven=suca_table[:, 11]
        self.kappa_eff=suca_table[:,12]
        self.thirteen=suca_table[:, 13]

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
            return self.nine
        if v_n == '10':
            return self.ten
        if v_n == '11':
            return self.eleven
        if v_n == 'kappa_eff':
            return self.kappa_eff
        if v_n == '13':
            return self.thirteen

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

class Criticals3:

    out_dir = '../data/output/'
    plot_dir = '../data/plots/'

    def __init__(self, root, sp_smfiles, wind_files, dirs_not_to_be_included, out_dir):

        self.input_dirs = sp_smfiles[0].split('/')[:-1]
        # print(self.input_dirs)
        self.dirs_not_to_be_included = dirs_not_to_be_included # name of folders that not to be in the name of out. file.

        self.out_dir = out_dir

        # self.ga_smfls = ga_smfiles
        # self.ga_mdl = []
        # self.ga_smdl =[]
        self.wmdfils = wind_files
        self.sp_smfls = sp_smfiles
        self.sp_mdl = []
        # self.sp_smdl =[]
        self.wndcls = []

        # for file in ga_smfiles:
        #     self.ga_mdl.append(Read_SM_data_file.from_sm_data_file(file))

        if len(self.sp_mdl) != len(self.wndcls): raise IOError('len(self.sp_mdl)[{}] != len(self.wndcls)[{}] '.format(len(self.sp_mdl), len(self.wndcls)))
        smfnames = []
        wnfnames = []
        fnames = []

        for i in range(len(sp_smfiles)):
            smfnames.append(sp_smfiles[i].split('/')[-1].split('sm.data')[0])
            wnfnames.append(wind_files[i].split('/')[-1].split('.wind')[0])


        for smfname in smfnames:
            if smfname not in wnfnames:
                raise NameError('There is a missing sm.data file: {}'.format(smfname))

        for fname in smfnames:
            self.sp_mdl.append(Read_SM_data_file.from_sm_data_file(root+fname+'sm.data'))
            self.wndcls.append(Read_Wind_file.from_wind_dat_file(root+fname+'.wind'))

        # for file in sp_smfiles:
        #     self.sp_mdl.append(Read_SM_data_file.from_sm_data_file(file))
        #
        # for file in wind_files:
        #     self.wndcls.append(Read_Wind_file.from_wind_dat_file(file))

        # self.sort_ga_smfiles('mdot', -1)
        # self.sort_sp_smfiles('mdot', -1)

    def get_boundary(self, u_min):
        '''
        RETURNS ' bourders.min() ' - min radius among all the models, where u exceeds the 'u_min'
        :param u_min:
        :return:
        '''
        bourders = []

        for i in range(len(self.ga_smfls)):
            u = self.ga_smdl[i].get_col('u')
            r = self.ga_smdl[i].get_col('r')
            for i in range(len(r)):
                if u[i] > u_min:
                    # ax1.axvline(x=r[i], color='red')
                    bourders = np.append(bourders, r[i])
                    break

        return bourders.min()

    def sort_sp_smfiles(self, v_n, where = -1, descending=True):
        '''

        :param v_n: what value to use to sort sm.files
        :param where: where value si caken (surface -1 or core 0 )
        :param descending: if True, sm.files are sorted by descending order of the chosen parameter.
        :return: NOTHING (changes the smdl[])
        '''


        i_and_mdots = []
        for i in range(len(self.sp_smfls)):
            i_and_mdots = np.append(i_and_mdots, [i, self.sp_mdl[i].get_col(v_n)[where]])

        i_and_mdots_sorted = np.sort(i_and_mdots.view('f8, f8'), order=['f1'], axis=0).view(np.float)
        i_and_mdots_reshaped = np.reshape(i_and_mdots_sorted, (len(self.sp_smfls), 2))

        if descending:
            i_and_mdots_reshaped_inversed = np.flip(i_and_mdots_reshaped, 0) # flip for ascending order
        else:
            i_and_mdots_reshaped_inversed = i_and_mdots_reshaped # no flipping


        sorted_by_mdot_files = []
        for i in range(len(self.sp_smfls)):
            sorted_by_mdot_files.append(self.sp_smfls[np.int(i_and_mdots_reshaped_inversed[i][0])])

        def get_i(file):
            for i in range(len(self.sp_smfls)):
                if file == self.sp_smfls[i]:
                    return i

        for file in sorted_by_mdot_files:

            self.sp_smdl.append(self.sp_mdl[get_i(file)])

        # for i in range(len(self.num_files)):
        #     print(self.smdl[i].get_col(v_n)[where])

    @staticmethod
    def plot_xy(ax, cl, start_ind, v_n1, v_n2, x_sp=None, x_env=None, i_file=0):

        if i_file == 0:
            ax.tick_params('y', colors='b')
            ax.set_xlabel(Labels.lbls(v_n1))
            ax.set_ylabel(Labels.lbls(v_n2), color='b')
        # --- ---.tick_params('y', colors='b')
        x = cl.get_col(v_n1)[start_ind:]
        y = cl.get_col(v_n2)[start_ind:]

        if v_n2 == 'kappa':
            y = 10 ** y

        # --- ---

        ax.plot(x, y, '-', color='blue')
        ax.annotate('{}'.format(int(i_file)), xy=(x[-1], y[-1]), textcoords='data')

        if x_sp != None and x_sp != 0:
            y_sp = interpolate.interp1d(x, y, kind='linear', bounds_error=False)(x_sp)
            ax.plot(x_sp, y_sp, 'X', color='blue')

        if x_env != None and x_env != 0:
            y_env = interpolate.interp1d(x, y, kind='linear')(x_env)
            ax.plot(x_env, y_env, 'X', color='cyan')

        # ax_ = None


        if v_n2 == 'kappa' :
            k_edd = 10 ** Physics.edd_opacity(cl.get_col('xm')[-1], cl.get_col('l')[-1], )
            if v_n2 == 'kappa':
                ax.plot(ax.get_xlim(), [k_edd, k_edd], '--', color='black')

        if v_n2 == 'u':
            u_s = cl.get_sonic_u()[start_ind:]
            if v_n2 == 'u':
                ax.plot(x, u_s, '--', color='gray')


        if v_n2 == 'L/Ledd':
            ax.plot(ax.get_xlim(), [1.0, 1.0], '-.', color='black')


        if v_n2 == 'Pg/P_total':
            ax.plot(ax.get_xlim(), [0.15, 0.15], '-.', color='gray')

    @staticmethod
    def plot_xyy(ax, ax_, cl, start_ind, v_n1, v_n2, v_n3, x_sp=None, x_env=None, i_file=0):

        if i_file == 0:
            ax.tick_params('y', colors='b')
            ax.set_xlabel(Labels.lbls(v_n1))
            ax.set_ylabel(Labels.lbls(v_n2), color='b')
            ax_.tick_params('y', colors='k')
            ax_.set_ylabel(Labels.lbls(v_n3), color='k')
        # --- ---.tick_params('y', colors='b')
        x = cl.get_col(v_n1)[start_ind:]
        y = cl.get_col(v_n2)[start_ind:]

        if v_n2 == 'kappa':
            y = 10 ** y

        # --- ---

        ax.plot(x, y, '-', color='blue')
        ax.annotate('{}'.format(int(i_file)), xy=(x[-1], y[-1]), textcoords='data')

        if x_sp != None and x_sp != 0:
            y_sp = interpolate.interp1d(x, y, kind='linear', bounds_error=False)(x_sp)
            ax.plot(x_sp, y_sp, 'X', color='blue')

        if x_env != None and x_env != 0:
            y_env = interpolate.interp1d(x, y, kind='linear', bounds_error=False)(x_env)
            ax.plot(x_env, y_env, 'X', color='cyan')

        y2 = cl.get_col(v_n3)[start_ind:]
        if v_n3 == 'kappa':
            y2 = 10 ** y2

        ax_.plot(x, y2, '-', color='black')
        ax_.annotate('{}'.format(int(i_file)), xy=(x[-1], y2[-1]), textcoords='data')

        if x_sp != None and x_sp != 0:
            y2_sp = interpolate.interp1d(x, y2, kind='linear', bounds_error=False)(x_sp)
            ax_.plot(x_sp, y2_sp, 'X', color='black')

        if x_env != None and x_env != 0:
            y2_env = interpolate.interp1d(x, y2, kind='linear', bounds_error=False)(x_env)
            ax_.plot(x_env, y2_env, 'X', color='orange')

        if v_n2 == 'kappa' or v_n3 == 'kappa':
            k_edd = 10 ** Physics.edd_opacity(cl.get_col('xm')[-1], cl.get_col('l')[-1], )
            if v_n2 == 'kappa':
                ax.plot(ax.get_xlim(), [k_edd, k_edd], '--', color='black')
            else:
                ax_.plot(ax.get_xlim(), [k_edd, k_edd], '--', color='black')

        if v_n2 == 'u' or v_n3 == 'u':
            u_s = cl.get_sonic_u()[start_ind:]
            if v_n2 == 'u':
                ax.plot(x, u_s, '--', color='gray')
            else:
                ax_.plot(x, u_s, '--', color='gray')

        if v_n2 == 'L/Ledd':
            ax.plot(ax.get_xlim(), [1.0, 1.0], '-.', color='black')
        if v_n3 == 'L/Ledd':
            ax_.plot(ax_.get_xlim(), [1.0, 1.0], '-.', color='black')

        if v_n2 == 'Pg/P_total':
            ax.plot(ax.get_xlim(), [0.15, 0.15], '-.', color='gray')
        if v_n3 == 'Pg/P_total':
            ax_.plot(ax_.get_xlim(), [0.15, 0.15], '-.', color='gray')

        #
        #
        #
        # # ax_ = None
        # if v_n3 != None:
        #     ax_ = ax.twinx()
        #     if set_axis:
        #         ax_.tick_params('y', colors='r')
        #         ax_.set_ylabel(Labels.lbls(v_n3), color='r')
        #
        #     y2 = cl.get_col(v_n3)[start_ind:]
        #     if v_n3 == 'kappa':
        #         y2 = 10 ** y2
        #
        #
        #     ax_.plot(x, y2, '-', color='red')
        #
        #     if x_sp != None and x_sp != 0:
        #         y2_sp = interpolate.interp1d(x, y2, kind='linear')(x_sp)
        #         ax_.plot(x_sp, y2_sp, 'X', color='red')
        #
        #     if x_env != None and x_env != 0:
        #         y2_env = interpolate.interp1d(x, y2, kind='linear')(x_env)
        #         ax_.plot(x_env, y2_env, 'X', color='orange')
        #
        # if v_n2 == 'kappa' or v_n3 == 'kappa':
        #     k_edd = 10 ** Physics.edd_opacity(cl.get_col('xm')[-1], cl.get_col('l')[-1], )
        #     if v_n2 == 'kappa':
        #         ax.plot(ax.get_xlim(), [k_edd, k_edd], '--', color='black')
        #     else:
        #         ax_.plot(ax.get_xlim(), [k_edd, k_edd], '--', color='black')
        #
        # if v_n2 == 'u' or v_n3 == 'u':
        #     u_s = cl.get_sonic_u()[start_ind:]
        #     if v_n2 == 'u':
        #         ax.plot(x, u_s, '--', color='gray')
        #     else:
        #         ax_.plot(x, u_s, '--', color='gray')
        #
        # if v_n2 == 'L/Ledd':
        #     ax.plot(ax.get_xlim(), [1.0, 1.0], '-.', color='black')
        # if v_n3 == 'L/Ledd':
        #     ax_.plot(ax_.get_xlim(), [1.0, 1.0], '-.', color='black')
        #
        # if v_n2 == 'Pg/P_total':
        #     ax.plot(ax.get_xlim(), [0.15, 0.15], '-.', color='gray')
        # if v_n3 == 'Pg/P_total':
        #     ax_.plot(ax_.get_xlim(), [0.15, 0.15], '-.', color='gray')

    # @staticmethod
    def plot_tau(self, ax, v_n1, wndcl, smcl, x_sp=None, x_env=None, logscale=False, i_file=0):

        '''

        :param ax:
        :param v_n1:
        :param wndcl:
        :param smcl:
        :param logscale:
        :param limit:       limit=(2 / 3)
        :param inner:       inner=20
        :return:
        '''

        if i_file==0:
            ax.set_xlabel(Labels.lbls(v_n1))
            ax.set_ylabel(Labels.lbls('tau'))

        def max_ind(cl, tau_lim=(2 / 3)):
            tau = cl.get_col('tau')
            ind = Math.find_nearest_index(tau, tau_lim)
            if ind == 0:
                raise ValueError('tau=2/3 is not found in the')
            return ind

        def min_ind(cl, u_min):
            u = cl.get_col('u')
            return Math.find_nearest_index(u, u_min)

        if logscale:
            ax.set_yscale("log", nonposy='clip')
        end = max_ind(wndcl)
        start = min_ind(smcl, 0.1)  # 0.1 - vel in km/s starting from which it plots

        tau_inner = smcl.get_col('tau')[start:]
        tau_outer = wndcl.get_col('tau')[1:end] # use 1

        if tau_outer[0] < (2/3):
            raise ValueError('Tau in the begnning of the wind is < 2/3.\n'
                             'Value: tau[1] = {}\n'
                             'File: {}'.format(tau_outer[1], self.sp_smfls[i_file]))

        tau_offset = tau_outer[1]
        tau_inner2 = tau_inner + tau_offset

        tau_full = []
        tau_full = np.append(tau_full, tau_inner2)
        tau_full = np.append(tau_full, tau_outer)

        x_full = []
        x_full = np.append(x_full, smcl.get_col(v_n1)[start:])
        x_full = np.append(x_full, wndcl.get_col(v_n1)[1:end]) # use 1:end as t

        # ax.plot(smcl.get_col(v_n1)[start:], tau_inner2+tau_offset, '.', color='blue')
        # ax.plot(wndcl.get_col(v_n1)[1:end], tau_outer, '.', color='red')

        ax.plot(x_full, tau_full, '-', color='gray')
        ax.annotate('{}'.format(int(i_file)), xy=(x_full[-1], tau_full[-1]), textcoords='data')


        ind_atm = Math.find_nearest_index(tau_full, (2 / 3))
        ax.plot(x_full[ind_atm], tau_full[ind_atm], 'X', color='gray')

        ind20 = Math.find_nearest_index(tau_full, 20)
        ax.plot(x_full[ind20], tau_full[ind20], '*', color='black')

        if x_sp != None and x_sp != 0:
            tau_sp = interpolate.interp1d(x_full, tau_full, kind='linear')(x_sp)
            ax.plot(x_sp, tau_sp, 'X', color='blue')

        if x_env != None and x_env != 0:
            tau_env = interpolate.interp1d(x_full, tau_full, kind='linear')(x_env)
            ax.plot(x_env, tau_env, 'X', color='cyan')

    @staticmethod
    def plot_wind(ax, cl, v_n1, v_n2, v_n3=None, tau_ph=(2 / 3)):

        def max_ind(cl, tau_lim=(2 / 3)):
            tau = cl.get_col('tau')
            ind = Math.find_nearest_index(tau, tau_lim)
            if ind == 0:
                raise ValueError('tau=2/3 is not found in the')
            return ind

        ind = max_ind(cl)

        x = cl.get_col(v_n1)[:ind]
        y = cl.get_col(v_n2)[:ind]
        ax.plot(x, y, '.', color='red')
        ax.plot(x, y, '-', color='red')

        # ax.plot(x[-1], y[-1], '-', color='black')
        # ax.plot(x[-1], y[-1], 'X', color='green')

        if v_n2 == 'kappa':
            # ax.plot(x[-1], 10 ** y[-1], 'X', color='blue')
            y2 = cl.get_col('kappa_eff')[:ind]
            ax.plot(x, y2, '.', color='green', label='kappa_eff')
            ax.plot(x, y2, '-', color='green')

    def plot_tph_teff(self):
        teff = []
        tph  = []
        mdot = []
        ts   = []
        ts_eff=[]
        arr = []

        for i in range(len(self.sp_mdl)):
            mdot = np.append(mdot, self.sp_mdl[i].get_col('mdot')[-1])
            ts = np.append(ts, self.sp_mdl[i].get_col('t')[-1])
            ts_eff=np.append(ts_eff, Physics.steph_boltz_law_t_eff(self.sp_mdl[i].get_col('l')[-1], self.sp_mdl[i].get_col('r')[-1]))
            tph = np.append(tph, self.wndcls[i].get_value('t'))
            teff = np.append( teff, Physics.steph_boltz_law_t_eff(self.sp_mdl[i].get_col('l')[-1], self.wndcls[i].get_value('r')) )

            arr = np.append(arr, [mdot[i], ts[i], ts_eff[i], tph[i], teff[i]])

        arr_sort = np.sort(arr.view('f8, f8, f8, f8, f8'), order=['f1'], axis=0).view(np.float)
        arr_shaped = np.reshape(arr_sort, (len(self.sp_mdl), 5))

        teff =  arr_shaped[:,4]
        tph =   arr_shaped[:,3]
        mdot =  arr_shaped[:,0]
        ts =    arr_shaped[:,1]
        ts_eff =arr_shaped[:,2]


        plt.plot(mdot, tph, '.', color='red',label='tph (tau=2/3)')
        plt.plot(mdot, tph, '-', color='red')
        plt.plot(mdot, teff, '.', color='blue', label='teff (BoltzLaw, tau2/3)')
        plt.plot(mdot, teff, '-', color='blue')
        plt.plot(mdot, ts, '.', color='black', label='tsonic')
        plt.plot(mdot, ts, '-', color='black')
        plt.plot(mdot, ts_eff, '.', color='green', label='teff (BoltzLaw, Sonic)')
        plt.plot(mdot, ts_eff, '-', color='green')
        plt.xlabel(Labels.lbls('mdot'))
        plt.ylabel(Labels.lbls('t'))
        plt.legend()
        plt.show()

    def analyze_sp_sm_fls(self, ax, depth, add_sonic_vals, show_plot, v_n_arr):
        '''

        :param depth:
        :param add_sonic_vals:
        :param show_plot:
        :return:
        '''

        def crop_ends(x, y):
            '''
            In case of 'wierd' vel/temp profile with rapidly rising end, first this rising part is to be cut of
            before maximum can be searched for.
            :param x:
            :param y:
            :return:
            '''
            x_mon = x
            y_mon = y

            non_monotonic = True

            while non_monotonic:

                if len(x_mon) <= 10:
                    return x, y
                    # raise ValueError('Whole array is removed in a searched for monotonic part.')

                if y_mon[-1] > y_mon[-2]:
                    y_mon = y_mon[:-1]
                    x_mon = x_mon[:-1]
                    # print(x_mon[-1], y_mon[-1])
                else:
                    non_monotonic = False

            return Math.find_nearest_index(x, x_mon[-1])

        def get_boundary(u_min):
            '''
            RETURNS ' bourders.min() ' - min radius among all the models, where u exceeds the 'u_min'
            :param u_min:
            :return:
            '''
            bourders = []

            for i in range(len(self.sp_smfls)):
                u = self.sp_mdl[i].get_col('u')
                r = self.sp_mdl[i].get_col('r')
                for i in range(len(r)):
                    if u[i] > u_min:
                        # ax1.axvline(x=r[i], color='red')
                        bourders = np.append(bourders, r[i])
                        break

            return bourders.min()

        def all_values_array(cls, min_indx, rs_p, ts_p, rp_env, xmp_env, add_sonic_vals):

            out_array = []

            mdot = cls.get_col('mdot')[-1]

            # --- --- GET ARRAYS AND VALUES AND APPENDING TO OUTPUT ARRAY --- --- --- --- --- --- --- ---

            r = cls.get_col('r')[min_indx:]
            t = cls.get_col('t')[min_indx:]
            xm = cls.get_col('xm')[-1]

            out_array = np.append(out_array, cls.get_col('l')[-1])   # appending 'l'        __1__
            out_array = np.append(out_array, cls.get_col('xm')[-1])  # appending 'xm'       __2__
            # out_array = np.append(out_array, cls.get_col('xm')[-1])
            out_array = np.append(out_array, cls.get_col('He4')[0])  # appending 'Yc'       __3__

            out_array = np.append(out_array, cls.get_col('mdot')[-1])  # appending 'mdot'   __4__
            out_array = np.append(out_array, rs_p)  # appending 'rs' __5__
            out_array = np.append(out_array, ts_p)  # appending 'ts' __6__
            if rp_env != 0.:
                out_array = np.append(out_array, r[-1] - rp_env)         # appending 'renv' __7__ (length of the envelope)
                out_array = np.append(out_array, np.log10(xm-xmp_env))   # appending 'renv' __8__ (length of the envelope)
                # raise ValueError('-------{}-{}--------'.format(r[-1] - rp_env, np.log10(xm-xmp_env)))
            else:
                out_array = np.append(out_array, 0.)  # appending 'renv' __7__ (length of the envelope)
                out_array = np.append(out_array, 0.)  # appending 'renv' __8__ (length of the envelope)
            # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

            val_array = []
            for v_n_cond in add_sonic_vals:

                if len(v_n_cond.split('-')) > 2:
                    raise NameError(
                        'For *add_sonic_vals* use format *var_name-location*, (given: {}) where var_name is '
                        'one of the BEC sm.data variables and location is *core = [-1] surface=[0]',
                        'sp = [sonic_point_interpolated]'.format(v_n_cond))

                v_n = v_n_cond.split('-')[0]
                cond = v_n_cond.split('-')[-1]

                if cond != 'sp':
                    var_val = cls.get_cond_value(v_n, cond)  # assuming that condition is not required interp

                else:
                    ''' Here The Interpolation of v_n is Done'''

                    v_n_val_arr = cls.get_col(v_n)[min_indx:]
                    if rs_p < r.min() or rs_p > r.max():
                        raise ValueError('rs ({}) outside of r region ({},{})'.format(rs_p, r.min(), r.max()))

                    f = interpolate.InterpolatedUnivariateSpline(r, v_n_val_arr)
                    var_val = f(rs_p)

                    if len([var_val]) > 1:
                        raise ValueError('More than one solution found for *{}* sonic value: ({})'.format(v_n, var_val))

                val_array = np.append(val_array, var_val)

            if len(val_array) != len(add_sonic_vals):
                raise ValueError('len(val_array)[{}] != len(add_sonic_vals)[{}]'
                                 .format(len(val_array), len(add_sonic_vals)))

            for i in range(len(val_array)):  # 7 is a number of main values: [ l, m, Yc, ts, rs, renv, menv]
                out_array = np.append(out_array, val_array[i])  # appending 'v_ns' __n__ SONIC VALUES

            return out_array

        def get_tp_up_r_xm__env(cls, guess=5.2):
            '''
            Looks for a loal extremum between t_lim1 and t_lim2, and, if the extremem != sonic point: returns
            length and mass of whatever is left
            :param cls:
            :param t_lim1:
            :param t_lim2:
            :return: tp, up, rp, xmp (coordinates, where envelope begins)
            '''

            def get_envelope_r_or_m(v_n, cls, t_start):
                t = cls.get_col('t')
                ind = Math.find_nearest_index(t, t_start) - 5  # just before the limit, so we don't need to

                # interpolate across the whole t range
                t = t[ind:]
                var = cls.get_col(v_n)
                var = var[ind:]

                value = interpolate.InterpolatedUnivariateSpline(t[::-1], var[::-1])(t_start)

                # print('-_-: {}'.format(var[-1]-value))

                return value

            t = cls.get_col('t')  # x - axis
            u = cls.get_col('u')  # y - axis

            # s = len(t)-1
            # while go_on:
            #     if u[-2] < u[-1]
            #
            # for i in range(len(t)):
            #     if u[s-1] < u[s]:
            #         s = s-1
            #     else:
            #         t = t[:s]
            #         u = u[:s]
            #         break

            # if t.min() > t_lim1 and t_lim2 > t.min():
            # i1 = Math.find_nearest_index(t, t_lim2)
            # i2 = Math.find_nearest_index(t, t_lim1)

            # t_cropped= t[Math.find_nearest_index(t, t_lim2):Math.find_nearest_index(t,
            #                                                                          t_lim1)][::-1]   # t_lim2 > t_lim1 and t is Declining
            # u_cropped = u[Math.find_nearest_index(t, t_lim2):Math.find_nearest_index(t, t_lim1)][::-1]

            # print('<<<<<<<<<<<SIZE: {} {} (i1:{}, i2:{}) >>>>>>>>>>>>>>>>'.format(len(t), len(u), i1, i2))

            tp, up = Math.get_max_by_interpolating(t, u, True, guess) # WHERE the envelope starts (if any)
                                                                      # TRUE for cutting out the rising part in the end
            if Math.find_nearest_index(t, tp) < len(t) - 1:  # if the tp is not the last point of the t array

                print('<<<<<<<<<<<Coord: {} {} >>>>>>>>>>>>>>>>'.format("%.2f" % tp, "%.2f" % up))

                return tp, up, get_envelope_r_or_m('r', cls, tp), get_envelope_r_or_m('xm', cls, tp)
            else:
                return 0., 0., 0, 0
            #     # print('L_env: {}'.format(get_envelope_r_or_m('r', smcl, tp)))
            #     # print('M_env: {}'.format(get_envelope_r_or_m('xm', smcl, tp)))
            #
            #     # var = get_envelope_r_or_m('r', smcl, tp)
            #     if m_or_r == 'xm': return tp, up, np.log10(get_envelope_r_or_m('xm', self, tp))
            #     if m_or_r == 'r': return tp, up, get_envelope_r_or_m('r', self, tp)
            #     raise NameError('m_or_r = {}'.format(m_or_r))
            # else:
            #     return 0.  # default value if there is no envelope

            # else: return None, None



        # def xy_profile(ax, mdl, v_n1, v_n2, var_for_label1, var_for_label2):
        #
        #     def get_m_r_envelope(smcl):
        #         '''
        #         Looks for a loal extremum between t_lim1 and t_lim2, and, if the extremem != sonic point: returns
        #         length and mass of whatever is left
        #         :param smcl:
        #         :param t_lim1:
        #         :param t_lim2:
        #         :return:
        #         '''
        #
        #         def get_envelope_l_or_m(v_n, cls, t_start, depth=1000):
        #             '''
        #             ESTIMATE the length or mass of the envelope
        #             :param v_n:
        #             :param cls:
        #             :param t_start:
        #             :param depth:
        #             :return:
        #             '''
        #
        #             t = cls.get_col('t')
        #             ind = Math.find_nearest_index(t,
        #                                           t_start) - 1  # just before the limit, so we don't need to interpolate across the whole t range
        #             t = t[ind:]
        #             var = cls.get_col(v_n)
        #             var = var[ind:]
        #
        #             value = interpolate.InterpolatedUnivariateSpline(t[::-1], var[::-1])(t_start)
        #
        #             # print('-_-: {}'.format(var[-1]-value))
        #
        #             return (var[-1] - value)
        #
        #         t = smcl.get_col('t')  # x - axis
        #         u = smcl.get_col('u')  # y - axis
        #
        #         # if t.min() > t_lim1 and t_lim2 > t.min():
        #         # i1 = Math.find_nearest_index(t, t_lim2)
        #         # i2 = Math.find_nearest_index(t, t_lim1)
        #
        #         # t_cropped= t[Math.find_nearest_index(t, t_lim2):Math.find_nearest_index(t,
        #         #                                                                          t_lim1)][::-1]   # t_lim2 > t_lim1 and t is Declining
        #         # u_cropped = u[Math.find_nearest_index(t, t_lim2):Math.find_nearest_index(t, t_lim1)][::-1]
        #
        #         # print('<<<<<<<<<<<SIZE: {} {} (i1:{}, i2:{}) >>>>>>>>>>>>>>>>'.format(len(t), len(u), i1, i2))
        #
        #         tp, up = Math.get_max_by_interpolating(t, u, False, 5.2)
        #         if Math.find_nearest_index(t, tp) < len(
        #                 t) - 1:  # if the tp is not the last point of the t array ( not th sonic point)
        #
        #             print('<<<<<<<<<<<Coord: {} {} >>>>>>>>>>>>>>>>'.format("%.2f" % tp, "%.2f" % up))
        #
        #             print('L_env: {}'.format(get_envelope_l_or_m('r', smcl, tp)))
        #             print('M_env: {}'.format(np.log10(get_envelope_l_or_m('xm', smcl, tp))))
        #
        #             # var = get_envelope_l_or_m('r', smcl, tp)
        #             return tp, up
        #         else:
        #             return None, None
        #
        #         # else: return None, None
        #
        #     fig = plt.figure()
        #     ax1 = fig.add_subplot(111)
        #
        #     tlt = v_n2 + '(' + v_n1 + ') profile'
        #     # plt.title(tlt)
        #
        #     for i in range(len(self.sm_files)):
        #
        #         x = self.mdl[i].get_col(v_n1)
        #         y = self.mdl[i].get_col(v_n2)  # simpler syntaxis
        #         label1 = self.mdl[i].get_col(var_for_label1)[-1]
        #         label2 = self.mdl[i].get_col(var_for_label2)[-1]
        #
        #         if v_n2 == 'kappa':
        #             y = 10 ** y
        #
        #         print('\t __Core H: {} , core He: {} File: {}'.
        #               format(self.mdl[i].get_col('H')[0], self.mdl[i].get_col('He4')[0], self.sm_files[i]))
        #
        #         lbl = '{}:{} , {}:{}'.format(var_for_label1, '%.2f' % label1, var_for_label2, '%.2f' % label2)
        #
        #         ax1.plot(x, y, '-', color='C' + str(Math.get_0_to_max([i], 9)[i]), label=lbl)
        #         # ax1.plot(x, y, '.', color='C' + str(Math.get_0_to_max([i], 9)[i]))
        #         ax1.plot(x[-1], y[-1], 'x', color='C' + str(Math.get_0_to_max([i], 9)[i]))
        #
        #         ax1.annotate(str('%.1f' % self.mdl[i].get_col('mdot')[-1]), xy=(x[-1], y[-1]), textcoords='data')
        #
        #         if sonic and v_n2 == 'u':
        #             u_s = self.mdl[i].get_sonic_u()
        #             ax1.plot(x, u_s, '-', color='black')
        #
        #             xc, yc = Math.interpolated_intercept(x, y, u_s)
        #             # print('Sonic r: {} | Sonic u: {} | {}'.format( np.float(xc),  np.float(yc), len(xc)))
        #             plt.plot(xc, yc, 'X', color='red')
        #
        #         if v_n2 == 'kappa':
        #             k_edd = 10 ** Physics.edd_opacity(self.mdl[i].get_col('xm')[-1],
        #                                               self.mdl[i].get_col('l')[-1])
        #             ax1.plot(ax1.get_xlim(), [k_edd, k_edd], color='black',
        #                      label='Model: {}, k_edd: {}'.format(i, k_edd))
        #
        #         if v_n2 == 'Pg/P_total':
        #             plt.axhline(y=0.15, color='black')
        #
        #         tp, up = get_m_r_envelope(self.mdl[i])
        #         if tp != None:
        #             ax1.plot(tp, up, 'X', color='black')
        #
        #     ax1.set_xlabel(Labels.lbls(v_n1))
        #     ax1.set_ylabel(Labels.lbls(v_n2))
        #
        #     ax1.grid(which='both')
        #     ax1.grid(which='minor', alpha=0.2)
        #     ax1.grid(which='major', alpha=0.2)
        #
        #     # ax1.set_xlim(0.78, 0.92)
        #     # ax1.set_xlim(4.0, 6.2)
        #     if not clean:
        #         ax1.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
        #     plot_name = self.plot_dir + v_n1 + '_vs_' + v_n2 + '_profile.pdf'
        #     # plt.savefig(plot_name)
        #     plt.show()



        # ==============================================================================================================
        # fig = plt.figure()
        # ax1 = fig.add_subplot(111)
        # ax2 = ax1.twinx()

        # tlt = 'VELOCITY PROFILE'
        # plt.title(tlt, loc='left')


        # ==============================================================================================================

        mdots = np.array([0.])

        r_min = get_boundary(0.1)  # setting the lower value of r that above which the analysis will take place

        ax[0].set_xlabel(Labels.lbls('r'))
        ax[0].set_ylabel(Labels.lbls('u'))

        if len(v_n_arr[0]) > 2: ax1 = ax[1].twinx()
        if len(v_n_arr[1]) > 2: ax2 = ax[2].twinx()
        if len(v_n_arr[2]) > 2: ax3 = ax[3].twinx()
        if len(v_n_arr[3]) > 2: ax4 = ax[4].twinx()
        if len(v_n_arr[4]) > 2: ax5 = ax[5].twinx()

        lbl = ''
        for i in range(len(self.sp_mdl)):
            lbl = lbl + str(i) + ' : ' + "%.2f"%self.sp_mdl[i].get_col('He4')[0] + \
                            ' ' + "%.2f"%(-1*self.sp_mdl[i].get_col('mdot')[-1]) + '\n'


        ax[0].text(0.7, 0.7, lbl, style='italic',
                bbox={'facecolor': 'cyan', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center',
                verticalalignment='center', transform=ax[0].transAxes)

        for cl in self.sp_mdl:
            r = cl.get_col('r')
            u = cl.get_col('u')
            min_ind = Math.find_nearest_index(r, r_min)

            ax[0].plot(r[min_ind:], u[min_ind:], '.', color='blue' )

        '''---INTERPOLATING THE SONIC VALUES---'''

        i_sm = 0
        out_array = np.zeros(len(add_sonic_vals) + 8)  # where 8 are: [l, m, Yc, mdot, rs, ts, r_env, m_emv] # always include
        for cl in self.sp_mdl:

            r  = cl.get_col('r')
            u  = cl.get_col('u')
            mu = cl.get_col('mu')
            t  = cl.get_col('t')
            u_s= cl.get_sonic_u()
            mdot_u=cl.get_col('mdot')[-1]
            mdots = np.append(mdots, mdot_u)
            print('\t__Initical Array Length: {}'.format(len(r)))

            min_ind = Math.find_nearest_index(r, r_min)

            # ----------------------- R U ----------------------------
            r  = r[min_ind:]
            u  = u[min_ind:]
            u_s= u_s[min_ind:]
            t = t[min_ind:]
            mu = mu[min_ind:]

            print('\t__Cropped Array Length: {}'.format(len(r)))

            int_r  = np.mgrid[r[0]:r[-1]:depth*1j]
            int_u  = Math.interp_row(r, u, int_r)

            ax[0].plot(r, u_s, '--', color='black')
            ax[0].annotate(str('%.2f' % mdot_u), xy=(r[-1], u[-1]), textcoords='data')
            ax[0].plot(int_r, int_u, '-', color='gray')

            # Check if the vel. profile is bad':)
            subson_prof = True
            check = 0
            for i in range(len(r)):
                if u[i] > u_s[i]:
                    check = check + 1
                    # print('                  u[{}] > u_s[{}]'.format( u[i] ,))
            if check > 3:
                print('                 Check: {}'.format(check))
                subson_prof = False

            # ------------------------R T --------------------------------

            ts_arr = np.log10((mu * Constants.m_H * (u * 100000) ** 2) / Constants.k_b)


            int_r = np.mgrid[r[0]:r[-1]:depth * 1j]
            int_t  = Math.interp_row(r, t, int_r)
            int_ts_arr=Math.interp_row(r, ts_arr, int_r)

            # ax2.plot(r, ts_arr, '.', color='orange')
            # ax2.plot(int_r, int_t,'-',color='orange')
            # ax2.plot(int_r, int_ts_arr, '--', color='orange')

            # --- --- ---| SONIC POINT PARAMTERS |--- --- ---
            if subson_prof: # if the cel profile stays subsonic

                rs_p, ts_p = Math.interpolated_intercept(int_r, int_ts_arr, int_t)     # SONIC TEPERATURE
                if rs_p.any():
                    # if len(rs_p)>1:
                    #     raise ValueError('Multiple sonic points in SP_BEC file(?) : Rs:{}'.format(rs_p))
                    # if rs_p > int_r[-1] or rs_p < int_r[-1]/2:
                    #     rs_p = int_r[-1]
                    #     ts_p = int_t[-1]
                    # else:
                    rs_p = rs_p[0][0]
                    ts_p = ts_p[0][0]
                    # pass
                else:
                    rs_p = int_r[-1]
                    ts_p = int_t[-1]

                print('\t__         Rs ({}) AND   R region ({},{})'.format(rs_p, r.min(), r.max()))
                if rs_p < r.min() or rs_p > r.max():
                    raise ValueError('rs ({}) outside of r region ({},{})'.format(rs_p, r.min(), r.max()))

                # ax2.plot(rs_p, ts_p, 'X', color='red')
                # ax2.annotate(str('%.2f' % ts_p), xy=(rs_p, ts_p), textcoords='data')
                # print('--Rs{} Ts{}'.format(rs_p, ts_p))

                u_s_p = interpolate.InterpolatedUnivariateSpline(r,u_s)(rs_p)
                ax[0].plot(rs_p, u_s_p, 'X', color='blue')

                tp_env, up_env, rp_env, xmp_env = get_tp_up_r_xm__env(cl, 5.2) # 5.2 = initial guess
                if rp_env != 0.:
                    ax[0].plot(rp_env, up_env, 'X', color='cyan') # location of the onset of an envelope
                    # ax2.plot(rp_env, tp_env, 'X', color='black')

                row = all_values_array(cl, min_ind, rs_p, ts_p, rp_env, xmp_env, add_sonic_vals)
                out_array = np.vstack((out_array, row))

                

                # --- ADDITIONAL PLOTTING --- --- ---
                # plot_xy(ax[0], cl, min_ind, 'r', 'u', 'kappa', rs_p, rp_env)
                # v_n_arr = []

                # self.plot_xy(  ax[1], cl, min_ind, 'r', 'rho', 'kappa', rs_p, rp_env)
                # self.plot_xy(  ax[2], cl, min_ind, 'r', 'Pg/P_total', 'L/Ledd', rs_p, rp_env)
                #
                #
                # self.plot_xy(  ax[3], cl, min_ind, 't', 'u', None, ts_p, tp_env)
                # self.plot_wind(ax[3], self.wndcls[i_sm], 't', 'u', None)
                #
                # self.plot_xy(  ax[4], cl, min_ind, 't', 'kappa', None, ts_p, tp_env)
                # self.plot_wind(ax[4], self.wndcls[i_sm], 't', 'kappa')
                #
                # self.plot_tau( ax[5], 't', self.wndcls[i_sm], cl, ts_p, tp_env, True)

                # ---



                if len(v_n_arr[0]) == 2:
                    self.plot_xy(ax[1], self.sp_mdl[i_sm], min_ind, v_n_arr[0][0], v_n_arr[0][1], rs_p, rp_env, i_sm)
                else:
                    self.plot_xyy(ax[1], ax1, self.sp_mdl[i_sm], min_ind, v_n_arr[0][0], v_n_arr[0][1], v_n_arr[0][2], rs_p, rp_env, i_sm)


                if len(v_n_arr[1]) == 2:
                    self.plot_xy(ax[2], self.sp_mdl[i_sm], min_ind, v_n_arr[1][0], v_n_arr[1][1], rs_p, rp_env, i_sm)
                else:
                    self.plot_xyy(ax[2], ax2, self.sp_mdl[i_sm], min_ind, v_n_arr[1][0], v_n_arr[1][1], v_n_arr[1][2], rs_p, rp_env, i_sm)



                self.plot_xy(ax[3], self.sp_mdl[i_sm], min_ind, v_n_arr[2][0], v_n_arr[2][1], ts_p, tp_env, i_sm)
                self.plot_wind(ax[3], self.wndcls[i_sm], v_n_arr[2][0], v_n_arr[2][1])

                self.plot_xy(ax[4], self.sp_mdl[i_sm], min_ind, v_n_arr[3][0], v_n_arr[3][1], ts_p, tp_env, i_sm)
                self.plot_wind(ax[4], self.wndcls[i_sm], v_n_arr[3][0], v_n_arr[3][1])

                self.plot_tau(ax[5], v_n_arr[4][0], self.wndcls[i_sm], self.sp_mdl[i_sm], ts_p, tp_env, True, i_sm)



                # self.plot_xy(ax[1], cl, min_ind, v_n_arr[0], v_n_arr[1], v_n_arr[2], rs_p, rp_env, plot_axis)
                # self.plot_xy(ax[2], cl, min_ind, v_n_arr[3], v_n_arr[4], v_n_arr[5], rs_p, rp_env, plot_axis)
                #
                # self.plot_xy(ax[3], cl, min_ind, v_n_arr[6], v_n_arr[7], ts_p, tp_env, plot_axis)
                # self.plot_wind(ax[3], self.wndcls[i_sm], v_n_arr[6], v_n_arr[7], v_n_arr[8])
                #
                # self.plot_xy(ax[4], cl, min_ind, v_n_arr[9], v_n_arr[10], v_n_arr[11], ts_p, tp_env, plot_axis)
                # self.plot_wind(ax[4], self.wndcls[i_sm], v_n_arr[9], v_n_arr[10], v_n_arr[11],)
                #
                # self.plot_tau(ax[5], v_n_arr[12], self.wndcls[i_sm], cl, ts_p, tp_env, True)


                i_sm = i_sm + 1
            else:
                print('WARNING! For: {} the vel. profile  exceeds the sonic vel.'.format(self.sp_smfls[i_sm]))



        tablehead = ['log(L)', 'M(Msun)', 'Yc', 'mdot', 'r-sp', 't-sp', 'r_env', 'm_env']

        for v_n in add_sonic_vals:
            tablehead.append(v_n)

        # extended_head = tablehead + tmp
        print(tablehead)
        out_array = np.delete(out_array, 0, 0)
        return tablehead, out_array

    def analyze_wind_fls(self, ax, v_n_arr, append0=False, append1=False):
        '''
        Returns a 2d array with mdot - first col, other cols for v_n at a point where tau=2/3 (photoshphere)
        :param v_n_arr:
        :return:
        '''

        def get_arr_of_val_0_1_ph(v_n_arr, append0=False, append1=False):
            '''
            Returns a 2d array with mdot a first row, and rest - rows for v_n,
            (if append0 or append1 != False: it will append before photosph. value, also the 0 or/and 1 value.
            '''

            out_arr = np.zeros(2 + len(v_n_arr))
            if append0:
                out_arr = np.zeros(2 + len(v_n_arr) * 2) # mdot, val0 val1, val_ph
            if append1:
                out_arr = np.zeros(2 + len(v_n_arr) * 3)

            for cl in self.wndcls:
                arr = []
                arr = np.append(arr, cl.get_value('mdot', 10))   # Mass loss ( for future combining)
                tau = cl.get_value('tau', 0)
                if tau == -np.inf or tau == np.inf:
                    raise ValueError('tau == inf')
                arr = np.append(arr, tau)    # Optical depth at the sonic point

                for v_n in v_n_arr:
                    if v_n == 'mdot':
                        raise NameError('Mdot is already appended')
                    if v_n == 'tau':
                        raise NameError('Tau is appended by default as a first value of')

                    arr = np.append(arr, cl.get_value(v_n, 'ph'))
                    if append0:
                        arr = np.append(arr, cl.get_value(v_n, 0))  # mdot, val0 val1, val_ph
                    if append1:
                        arr = np.append(arr, cl.get_value(v_n, 1))

                out_arr = np.vstack((out_arr, arr))


            out_arr = np.delete(out_arr, 0, 0)

            # --- HEAD of the table (array)

            head = []
            head.append('mdot')
            head.append('tau-sp')
            for v_n in v_n_arr:
                head.append('{}-{}'.format(v_n, 'ph'))
                if append0:
                    head.append('{}-{}'.format(v_n, 0))  # mdot, val0 val1, val_ph
                if append1:
                    head.append('{}-{}'.format(v_n, 1))

            return head, out_arr
        # vals = get_arr_of_val_0_1_ph(['t','r','tau'])
        # print('a')

        # for cl in self.wndcls:
        #     atm_p = cl.get_col('gp') # the grid point where is photosphere
        #
        #     t0 = cl.get_col('t')[0]
        #     t1 = cl.get_col('t')[1]
        #     tph= cl.get_col('t')[np.int(atm_p)]
        #     r0 = cl.get_col('r')[0]
        #     r1 = cl.get_col('r')[1]
        #     rph= cl.get_col('r')[np.int(atm_p)]
        #
        #
        #
        #     ax = plt.subplot(111)
        #     x_wind = cl.get_col('t')
        #     tau = cl.get_col('tau')
        #     ax.plot(x_wind, tau, '.', color='gray')
        #     atm_p = cl.get_col('gp')
        #     x_wind_atm = cl.get_col('t')[np.int(atm_p)]
        #
        #     ax.axvline(x=x_wind_atm, label='Atmosphere')
        #
        #     ax.set_yscale("log", nonposy='clip')
        #     ax.grid()
        #     ax.set_xlabel(Labels.lbls('t'))
        #     ax.set_ylabel(Labels.lbls('tau'))
        #     # ax.set_yscale("log", nonposy='clip')
        #     ax.legend()
        #     plt.show()

        def max_ind(cl, tau_lim=(2/3)):
            tau = cl.get_col('tau')
            ind = Math.find_nearest_index(tau, tau_lim)
            if ind == 0:
                raise ValueError('tau=2/3 is not found in the')
            return ind

        def min_ind(cl, u_min):
            u = cl.get_col('u')
            return Math.find_nearest_index(u, u_min)







        # for i in range(len(self.wndcls)):
        #     if self.sp_smfls[i].split('/')[-1].split('sm.data')[0] != self.wmdfils[i].split('/')[-1].split('.wind')[0]:
        #         raise NameError('For Tau. sm.data and .wind are not the same...: {} != {}'.format(
        #             self.sp_smfls[i].split('/')[-1].split('sm.data')[0],
        #             self.wmdfils[i].split('/')[-1].split('.wind')[0]
        #         ))

            # plot_tau(ax, wndcl, smcl, limit=(2/3), inner=20)
            # tau_ = self.sp_mdl[i].get_col('tau')
            # tau_wind_0 = self


            # ax[5].set_yscale("log", nonposy='clip')
            # plot_wind(ax[5], self.wndcls[i], 'r', 'tau')
            # plot_tau(ax[5],'t',self.wndcls[i],self.sp_smdl[i],False)
            # plot_wind(ax[4], self.wndcls[i], 't', 'kappa')
            # plot_wind(ax[3], self.wndcls[i], 't', 'u')
            # plot_wind(ax[4], wcl, 'tau', None)

        return get_arr_of_val_0_1_ph(v_n_arr, append0, append1)




        # for cl in self.wndcls:
        #     atm_p = cl.get_col('gp') # the grid point where is photosphere
        #
        #     t0 = cl.get_col('t')[0]
        #     t1 = cl.get_col('t')[1]
        #     tph= cl.get_col('t')[np.int(atm_p)]
        #     r0 = cl.get_col('r')[0]
        #     r1 = cl.get_col('r')[1]
        #     rph= cl.get_col('r')[np.int(atm_p)]
        #
        #
        #
        #     ax = plt.subplot(111)
        #     x_wind = cl.get_col('t')
        #     tau = cl.get_col('tau')
        #     ax.plot(x_wind, tau, '.', color='gray')
        #     atm_p = cl.get_col('gp')
        #     x_wind_atm = cl.get_col('t')[np.int(atm_p)]
        #
        #     ax.axvline(x=x_wind_atm, label='Atmosphere')
        #
        #     ax.set_yscale("log", nonposy='clip')
        #     ax.grid()
        #     ax.set_xlabel(Labels.lbls('t'))
        #     ax.set_ylabel(Labels.lbls('tau'))
        #     # ax.set_yscale("log", nonposy='clip')
        #     ax.legend()
        #     plt.show()

    @staticmethod
    def where(arr, value):
        for i in range(len(arr)):
            if arr[i] == value:
                return i
        raise ValueError('arr[{}] == value[{}] | not found'.format(arr, value))

    @staticmethod
    def where_mdot(arr, value, precision=.2):
        for i in range(len(arr)):
            # print('======== {} != {} ======='.format(np.round(arr[i], 2),np.round(value, 2)))
            if np.round(arr[i], 2) == np.round(value, 2):
                return i
        return None

        #     if "%{}f".format(precision) % arr[i] == "%{}f".format(precision) % value:
        #         return i
        # return None
        # # raise ValueError('arr[{}] == value[{}] | not found'.format(arr, value))

    def combine_save(self, depth, add_sonic_vals, show_plot, use_wind_fls = True):

        def combine_two_tables_by_mdot(head1, table1, head2, table2):

            i_mdot1_name = self.where(head1, 'mdot')
            i_mdot2_name = self.where(head2, 'mdot')
            mdot1_arr = table1[:, i_mdot1_name]
            mdot2_arr = table2[:, i_mdot2_name]

            # if len(mdot1_arr) != len(mdot2_arr):
            #     raise ValueError('len(mdot1_arr)[{}] != len(mdot2_arr)[{}]'.format(len(mdot1_arr), len(mdot2_arr)))

            out_arr = []

            n_mdot_avl = 0
            for i in range(len(mdot1_arr)):

                i_mdot1 = self.where_mdot(mdot1_arr, mdot1_arr[i])
                i_mdot2 = self.where_mdot(mdot2_arr, mdot1_arr[i])



                if i_mdot1 == None or i_mdot2 == None:
                    pass # Mdot is not found in one of two arrays - passing by.
                    # print(mdot1_arr[i])
                    # break
                else:
                    n_mdot_avl = n_mdot_avl + 1

                    two_tables = []
                    two_tables = np.hstack((two_tables, table1[i_mdot1, :]))
                    two_tables = np.hstack((two_tables, table2[i_mdot2, 1:]))

                    out_arr = np.append(out_arr, two_tables)

            if n_mdot_avl == 0: raise ValueError('No common Mdot between sm.data and .wind data files fround: \n'
                                                 'sm.data: {} \n'
                                                 '.wind:   {} '.format(mdot1_arr, mdot2_arr))

            out_array = np.reshape(out_arr, (n_mdot_avl, len(table1[0,:])+len(table2[0, 1:])))

            head = head1
            for i in range(1, len(head2)): # starting from 1, as the table2 also has mdot, which is not needed twice
                head.append(head2[i])

            if len(head) != len(out_array[0, :]):
                raise ValueError('Something is wrong here...')
            return head, out_array
        def create_file_name(first_part='SP3'):
            '''Creates name like:
            SP3_ga_z0008_10sm_y10
            using the initial folders, where the sm.data files are located
            '''
            out_name = first_part
            for i in range(len(self.input_dirs)):
                if self.input_dirs[i] not in self.dirs_not_to_be_included and self.input_dirs[i] != '..':
                    out_name = out_name + self.input_dirs[i]
                    if i < len(self.input_dirs) - 1:
                        out_name = out_name + '_'
            out_name = out_name + '.profs'
            return out_name

        #==============================================================================



        fname = create_file_name('SP3')

        fig = plt.figure()
        ax = []
        ax.append(fig.add_subplot(231))
        ax.append(fig.add_subplot(232))
        ax.append(fig.add_subplot(233))
        ax.append(fig.add_subplot(234))
        ax.append(fig.add_subplot(235))
        ax.append(fig.add_subplot(236))

        fig.set_figheight(14)
        fig.set_figwidth(14)

        v_n_arr = [['r', 'rho', 'kappa'],           # CORE
                   ['r', 'Pg/P_total', 'L/Ledd'],   # CORE
                   ['t', 'u'],                      # CORE + WIND
                   ['t', 'kappa'],                  # CORE + WIND
                   ['t']]                           # Tau

        # set_labels_names_paramters(ax)

        sp_head, sp_table = self.analyze_sp_sm_fls(ax, depth, add_sonic_vals, show_plot, v_n_arr)
        wd_head, wd_arr   = self.analyze_wind_fls(ax, ['t', 'r'], False, False)

        # for i in range(len(self.sp_smdl)):
        #     plot_tau(ax[5], 'r', self.wndcls[i], self.sp_smdl[i])

        # # do not append 0th and 1st values on the WIND (for the SP file)

        plt.tight_layout()
        fig = plt.gcf()
        plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
        plt.show()




        head, table = combine_two_tables_by_mdot(sp_head, sp_table, wd_head, wd_arr)
        print('\n FILE: {} | SAVED IN: {} | sm.data & .wind FILES USED \n'.format(fname, 'SP3'))







        tmp = ''
        for i in range(len(head)):
            tmp = tmp + head[i] + ' '
        head__ = tmp

        np.savetxt(self.out_dir + fname, table, '%.5f', '  ', '\n', head__, '')

        self.plot_tph_teff()
        print('a')




    # def combine_ga_sp(self, depth, add_sonic_vals, show_plot, way_to_interp_tau='Uni'):
    #     '''
    #     Interpolates tau to get the critical value, based on the mdot_critical, obtained from analysis of GA models,
    #     returns the column of tau, where the first value is the critical value.
    #     OUT: 2d table with Sonic point values from SONIC-BEC and critical values from NORMAL-BEC (+ tau)
    #
    #     :param depth:
    #     :param add_sonic_vals:
    #     :param show_plot:
    #     :param way_to_interp_tau: IntUni, Uni, poly4, ploy3, poly2, poly1,
    #     :return:
    #     '''
    #     ga_head, ga_table = self.analyze_ga_sm_fls(depth, add_sonic_vals, show_plot)
    #     sp_head, sp_table = self.analyze_sp_sm_and_plot_fls(depth, add_sonic_vals, show_plot)
    #
    #     def compare_sonic_values():
    #         '''
    #         Combines the [-sp] values from ga and sp tables, estimating the error for each pair, returning the
    #         3d array [v_n:, mdot:, [mdot_val, ga_val, sp_val, err, err in %]]
    #         :return:
    #         '''
    #         ga_sp_v_n = []
    #         ga_i_val = []
    #         for i in range(len(ga_head)):
    #             print(ga_head[i].split('-'))
    #             if 'sp' in ga_head[i].split('-'):
    #                 ga_sp_v_n.append(ga_head[i].split('-')[0])
    #                 ga_i_val = np.append(ga_i_val, np.int(i))
    #
    #         sp_sp_v_n = []
    #         sp_i_val = []
    #         for i in range(len(sp_head)):
    #             print(sp_head[i].split('-'))
    #             if 'sp' in sp_head[i].split('-'):
    #                 sp_sp_v_n.append(sp_head[i].split('-')[0])
    #                 sp_i_val = np.append(sp_i_val, np.int(i))
    #
    #         if ga_sp_v_n != sp_sp_v_n:
    #             raise NameError('Sonic Point names (SP): {}\n Sonic Point names (GA): {}\n NOT THE SAME'
    #                             .format(sp_sp_v_n, ga_sp_v_n))
    #         print('\t__{} =? {}'.format(len(ga_i_val), len(sp_i_val)))
    #
    #
    #         print('Mdot | {}'.format(ga_sp_v_n))
    #
    #         mdot_row = []
    #         for i in range(len(ga_table[:,3])): # mdot
    #             ga_mdot = ga_table[i,3]
    #             if ga_mdot in sp_table[:,3]: # if [ga_mdot] is in [sp_mdot]
    #                 mdot_row = np.append(mdot_row, ga_table[i,3])
    #
    #         arr = []
    #         for i in range(len(ga_sp_v_n)):
    #             # print('\n')
    #             arr_mdots = np.zeros(5)
    #             for j in range(len(mdot_row)):
    #                 ga_mdot_i = Math.find_nearest_index(ga_table[:, 3], mdot_row[j])
    #                 sp_mdot_i = Math.find_nearest_index(sp_table[:, 3], mdot_row[j])
    #                 ga_val = ga_table[ga_mdot_i, np.int(ga_i_val[i])]
    #                 sp_val = sp_table[sp_mdot_i, np.int(sp_i_val[i])]
    #
    #                 err = np.abs(ga_val - sp_val)
    #                 err2= (ga_val - err) / 100      # in % with respect to GA
    #
    #                 arr_mdots = np.vstack((arr_mdots, [mdot_row[i], ga_val, sp_val, err, err2]))
    #
    #             arr = np.append(arr, np.delete(arr_mdots, 0, 0))
    #
    #         arr = np.reshape(arr, (len(ga_sp_v_n), len(mdot_row), 5))
    #
    #         return arr
    #
    #     def extrapolate_crit_val(v_n, way_to_interp_tau):
    #
    #         grid_mdot = sp_table[1:, 3] # mdot
    #         mdot = sp_table[:, 3]
    #         tau = []
    #
    #         if not v_n in sp_head:
    #             raise NameError('v_n: {} is not in sp_head: {}'.format(v_n, sp_head))
    #
    #         for i in range(len(sp_head)):
    #             if sp_head[i] == v_n:
    #                 tau = sp_table[:, i]
    #                 break
    #         # print(mdot, tau)
    #
    #         md_tau = []
    #         for i in range(len(mdot)):
    #             md_tau = np.append(md_tau, [mdot[i], tau[i]])
    #
    #         md_tau_sort = np.sort(md_tau.view('f8, f8'), order=['f0'], axis=0).view(np.float)
    #         md_tau_sh = np.reshape(md_tau_sort, (np.int(len(md_tau)/2), 2))
    #
    #
    #
    #         tau_grid1 = interpolate.InterpolatedUnivariateSpline(md_tau_sh[:,0], md_tau_sh[:,1])(grid_mdot)
    #         tau_cr1 = interpolate.InterpolatedUnivariateSpline(md_tau_sh[:, 0], md_tau_sh[:, 1])(ga_table[0, 3])
    #
    #         # tau_grid2 = interpolate.UnivariateSpline(md_tau_sh[:, 0], md_tau_sh[:, 1])(grid_mdot)
    #         # tau_cr2 = interpolate.UnivariateSpline(md_tau_sh[:, 0], md_tau_sh[:, 1])(ga_table[0, 3])
    #         #
    #         # tmp, tau_grid3 = Math.fit_plynomial(md_tau_sh[:,0], md_tau_sh[:,1], 4, 0, grid_mdot)
    #         # tmp, tau_cr3 = Math.fit_plynomial(md_tau_sh[:, 0], md_tau_sh[:, 1], 4, 0, np.array([ga_table[0, 3]]))
    #         #
    #         # tmp, tau_grid4 = Math.fit_plynomial(md_tau_sh[:,0], md_tau_sh[:,1], 3, 0, grid_mdot)
    #         # tmp, tau_cr4 = Math.fit_plynomial(md_tau_sh[:, 0], md_tau_sh[:, 1], 3, 0, np.array([ga_table[0, 3]]))
    #         #
    #         # tmp, tau_grid5 = Math.fit_plynomial(md_tau_sh[:,0], md_tau_sh[:,1], 2, 0, grid_mdot)
    #         # tmp, tau_cr5 = Math.fit_plynomial(md_tau_sh[:, 0], md_tau_sh[:, 1], 2, 0, np.array([ga_table[0, 3]]))
    #         #
    #         # tmp, tau_grid6 = Math.fit_plynomial(md_tau_sh[:,0], md_tau_sh[:,1], 1, 0, grid_mdot)
    #         # tmp, tau_cr6 = Math.fit_plynomial(md_tau_sh[:, 0], md_tau_sh[:, 1], 1, 0, np.array([ga_table[0, 3]]))
    #
    #
    #         plt.plot(md_tau_sh[:,0], md_tau_sh[:,1], '.', color = 'black')
    #         plt.plot(grid_mdot, tau_grid1, '--', color='gray', label = '{}_cr(IntUn)  = {}'.format(v_n, "%.2f" % ( tau_cr1)))
    #         # plt.plot(grid_mdot, tau_grid2, '--', color='blue', label = '{}_cr(Univ) = {}'.format(v_n, "%.2f" % ( tau_cr2)))
    #         # plt.plot(grid_mdot, tau_grid3, '--', color='orange', label='{}_cr(ploy4) = {}'.format(v_n, "%.2f" % ( tau_cr3)))
    #         # plt.plot(grid_mdot, tau_grid4, '--', color='red', label='{}_cr(ploy3) = {}'.format(v_n, "%.2f" % ( tau_cr4)))
    #         # plt.plot(grid_mdot, tau_grid5, '--', color='cyan', label='{}_cr(ploy2) = {}'.format(v_n, "%.2f" % ( tau_cr5)))
    #         # plt.plot(grid_mdot, tau_grid6, '--', color='black', label='{}_cr(ploy1) = {}'.format(v_n, "%.2f" % ( tau_cr6)))
    #
    #         plt.axvline(x=ga_table[0, 3], label='Critical point')
    #         plt.legend()
    #         plt.grid()
    #         plt.xlabel(Labels.lbls('mdot'))
    #         plt.ylabel(v_n)
    #         if show_plot:
    #             plt.show()
    #
    #
    #
    #         if way_to_interp_tau == 'IntUni':
    #             return tau_cr1, np.append(np.array([tau_cr1]), tau_grid1)
    #
    #         # if way_to_interp_tau == 'Uni':
    #         #     return tau_cr2, np.append(np.array([tau_cr2]), tau_grid2)
    #         #
    #         # if way_to_interp_tau == 'poly4':
    #         #     return tau_cr3, np.append(np.array([tau_cr3]), tau_grid3)
    #         #
    #         # if way_to_interp_tau == 'poly3':
    #         #     return tau_cr4, np.append(np.array([tau_cr4]), tau_grid4)
    #         #
    #         # if way_to_interp_tau == 'poly2':
    #         #     return tau_cr5, np.append(np.array([tau_cr5]), tau_grid5)
    #         #
    #         # if way_to_interp_tau == 'poly1':
    #         #     return tau_cr6, np.append(np.array([tau_cr6]), tau_grid6)
    #
    #         raise NameError('Given way_to_interp_tau ({}) is not availabel. '
    #                         'Use only: IntUni, Uni, poly4, poly3, poly2, poly1'.format(way_to_interp_tau))
    #
    #
    #
    #
    #     arr = compare_sonic_values()
    #
    #     print(way_to_interp_tau)
    #     # 'log(L)', 'M(Msun)', 'Yc', 'mdot', 'r-sp', 't-sp', 'R_wind', 'T_eff', 'Tau'
    #     tau, tau_row = extrapolate_crit_val('Tau', way_to_interp_tau)
    #
    #     ga_head.insert(6,'Tau')
    #     tmp = ''
    #     for i in range(len(ga_head)):
    #         tmp = tmp + ' ' + ga_head[i]
    #
    #     ga_head = tmp
    #     ga_table = np.insert(ga_table, 6, tau, 1)
    #
    #
    #     # --- --- --- MAKING A OUTPUT FILE NAME OUT OF FOLDERS THE SM.DATA FILES CAME FROM --- --- ---
    #     out_name = 'SP2'
    #     for i in range(len(self.input_dirs)):
    #         if self.input_dirs[i] not in self.dirs_not_to_be_included and self.input_dirs[i] != '..':
    #             out_name = out_name + self.input_dirs[i]
    #             if i < len(self.input_dirs) - 1:
    #                 out_name = out_name + '_'
    #     out_name = out_name + '.data'
    #
    #     print('Results are saved in: {}'.format(self.out_dir + out_name))
    #
    #     # np.savetxt(self.out_dir + out_name, ga_table, '%.5f', '  ', '\n', ga_head, '')
    #
    #     # sp_head.insert(6, 'Tau')
    #
    #     crit_row = np.zeros(len(sp_head))
    #     crit_row[0] = ga_table[0,0] # L
    #     crit_row[1] = ga_table[0,1] # M
    #     crit_row[2] = ga_table[0,2] # Yc
    #     crit_row[3] = ga_table[0,3] # mdot
    #     crit_row[4] = ga_table[0,4] # R
    #     crit_row[5] = ga_table[0,5] # T
    #     crit_row[6] = tau
    #
    #     tmp = ''
    #     for i in range(len(sp_head)):
    #         tmp = tmp + ' ' + sp_head[i]
    #     sp_head = tmp
    #
    #     sp_table_and_crit_row = np.insert(sp_table, 0, crit_row, 0)
    #     np.savetxt(self.out_dir + out_name, sp_table_and_crit_row, '%.5f', '  ', '\n', sp_head, '')
    #
    #
    #     # print(arr)
    #
    #     print('a')

    # def extract_mdot_tau(self, mdot1, mdot2, mdot_step):
    #
    #     self.mdot_grid = np.flip(np.arange(mdot1, mdot2, mdot_step), 0)
    #
    #     arr = np.empty((len(self.mdot_grid), 3))
    #     arr[:] = np.nan
    #
    #     for i in range(len(self.plt_files)):
    #         l = self.plotmdl[i].l_[-1]
    #         m = self.plotmdl[i].m_[-1]
    #         mdot = self.plotmdl[i].mdot_[-1]
    #         tau = self.plotmdl[i].tauatR[-1]
    #
    #         if not np.float("%.2f" % mdot) in list([np.float("%.2f" % mdot) for mdot in self.mdot_grid]):
    #             raise ValueError('Value of mdot from .plot file ({}) is not in a grid \n {}'.format(mdot, self.mdot_grid))
    #         else:
    #             # j = np.where( list([np.float("%.2f" % mdot) for mdot in self.mdot_grid]) == np.float("%.2f" % mdot) )[0]
    #             j = Math.find_nearest_index(np.array(list([np.float("%.2f" % mdot) for mdot in self.mdot_grid])), np.float("%.2f" % mdot))
    #             arr[j, 0] = l
    #             arr[j, 1] = m
    #             arr[j, 2] = tau
    #
    #     arr_ = np.vstack((self.mdot_grid, arr.T)).T
    #
    #     return arr_

# ======================================================================================================================

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

# ----------------------------------------------------------------------------------------------------------------------

dir = '/media/vnedora/HDD/sse/ga_z002/10sm/y10/sp/'


# location = '/media/vnedora/HDD/sse/' + 'tga_z002/t10sm/'
# local = '' + './'




tst_spfiles = get_files(dir, [''], [], 'sm.data')
tst_windfls = get_files(dir, [''], [], 'wind')

if len(tst_spfiles) != len(tst_windfls):
    raise IOError('Number of sm.data({}) and .wind({}) files are not the same!'.format(len(tst_spfiles), len(tst_windfls)))

cl = Criticals3(dir, tst_spfiles, tst_windfls, './', './')
cl.combine_save(1000, ['kappa-sp', 'L/Ledd-sp', 'HP-sp', 'mfp-sp', 'tpar-'], True, True) # PLOT, WIND fils



# print(tst_spfiles)