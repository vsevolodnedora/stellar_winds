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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

    light_v = np.float( 2.99792458 * (10 ** 10) )      # cm/s
    solar_m = np.float ( 1.99 * (10 ** 33)  )          # g
    solar_l = np.float ( 3.9 * (10 ** 33)  )           # erg s^-1
    solar_r = np.float ( 6.96 * (10 ** 10) )           #cm
    grav_const = np.float ( 6.67259 * (10 ** (-8) )  ) # cm3 g^-1 s^-2
    k_b     =  np.float ( 1.380658 * (10 ** (-16) ) )  # erg k^-1
    m_H     =  np.float ( 1.6733 * (10 ** (-24) ) )    # g
    c_k_edd =  np.float ( 4 * light_v * np.pi * grav_const * ( solar_m / solar_l ) )# k = c_k_edd*(M/L) (if M and L in solar units)

    yr      = np.float( 31557600. )
    smperyear = np.float(solar_m / yr)



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
    def get_max_by_interpolating(x, y):

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
        guess = x[np.where(y == y.max())]
        # print(guess)
        print('Gues:', guess)

        x_max = fmin(f2, guess)


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

        print()

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
                print('Yc: {}, y_row({} - {}), y_res: {}'.format(yc_value, y_arr.min(), y_arr.max(), y))
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

class Plots:
    def __init__(self):
        pass

    @staticmethod
    def plot_color_table(table, v_n_x, v_n_y, v_n_z, opal_used, label = None):

        plt.figure()
        ax = plt.subplot(111)


        if label != None:
            ax.text(0.8, 0.1, label, style='italic',
                    bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center',
                    verticalalignment='center', transform=ax.transAxes)


            # print('TEXT')
            # plt.text(table[0, 1:].min(), table[1:, 0].min(), label, style='italic')
            # bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10}
            # plt.text(2, 0.65, r'$\cos(2 \pi t) \exp(-t)$')

        # ax = fig.add_subplot(1, 1, 1)
        plt.xlim(table[0, 1:].min(), table[0, 1:].max())
        plt.ylim(table[1:, 0].min(), table[1:, 0].max())
        plt.ylabel(Labels.lbls(v_n_y))
        plt.xlabel(Labels.lbls(v_n_x))

        levels = Levels.get_levels(v_n_z, opal_used)


        contour_filled = plt.contourf(table[0, 1:], table[1:, 0], table[1:, 1:], levels, cmap=plt.get_cmap('RdYlBu_r'))
        plt.colorbar(contour_filled, label=Labels.lbls(v_n_z))
        contour = plt.contour(table[0, 1:], table[1:, 0], table[1:, 1:], levels, colors='k')
        plt.clabel(contour, colors='k', fmt='%2.2f', fontsize=12)
        plt.title('SONIC HR DIAGRAM')


        # plt.ylabel(l_or_lm)
        # plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
        # plt.savefig(name)
        plt.show()

    @staticmethod
    def plot_color_background(ax, table, v_n_x, v_n_y, v_n_z, opal_used, label = None):



        # if label != None:
        #     print('TEXT')

            # ax.text(table[0, 1:].min(), table[1:, 0].min(), s=label)
            # bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10}
            # plt.text(2, 0.65, r'$\cos(2 \pi t) \exp(-t)$')

        # ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(table[0, 1:].min(), table[0, 1:].max())
        ax.set_ylim(table[1:, 0].min(), table[1:, 0].max())
        ax.set_ylabel(Labels.lbls(v_n_y))
        ax.set_xlabel(Labels.lbls(v_n_x))

        levels = Levels.get_levels(v_n_z, opal_used)


        contour_filled = plt.contourf(table[0, 1:], table[1:, 0], table[1:, 1:], levels, cmap=plt.get_cmap('RdYlBu_r'))
        clb = plt.colorbar(contour_filled)
        clb.ax.set_title(Labels.lbls(v_n_z))

        # ax.colorbar(contour_filled, label=Labels.lbls(v_n_z))
        contour = plt.contour(table[0, 1:], table[1:, 0], table[1:, 1:], levels, colors='k')
        ax.clabel(contour, colors='k', fmt='%2.2f', fontsize=12)
        ax.set_title('SONIC HR DIAGRAM')

        # print('Yc:{}'.format(yc_val))
        ax.text(0.9, 0.9, label, style='italic',
                bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes)


        # plt.ylabel(l_or_lm)
        # plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)
        # plt.savefig(name)
        # plt.show()

    @staticmethod
    def plot_obs_mdot_llm(ax, obs_cls, l_or_lm, yc_val, yc1, yc2):
        '''

        :param ax:
        :param obs_cls:
        :param l_or_lm:
        :param yc_val:
        :param yc1:
        :param yc2:
        :return:
        '''
        classes = []
        classes.append('dum')
        mdot_obs = []
        llm_obs = []

        # from Phys_Math_Labels import Opt_Depth_Analythis

        for star_n in obs_cls.stars_n:
            i = -1
            mdot_obs = np.append(mdot_obs, obs_cls.get_num_par('mdot', star_n))
            llm_obs = np.append(llm_obs, obs_cls.get_num_par(l_or_lm, star_n, yc_val))
            eta = obs_cls.get_num_par('eta', star_n)

            lm_err1, lm_err2 = obs_cls.get_star_lm_obs_err(star_n, yc_val)
            mdot1, mdot2 = obs_cls.get_star_mdot_obs_err(star_n, yc_val)

            mdot_coord = [mdot1, mdot2, mdot2, mdot1]
            lm_coord = [lm_err1, lm_err1, lm_err2, lm_err2]
            ax.add_patch(patches.Polygon(xy=list(zip(mdot_coord, lm_coord)), fill=True, alpha=.4,
                                         color=obs_cls.get_class_color(star_n)))

            if l_or_lm == 'lm':

                llm1, llm2 = obs_cls.get_star_lm_err(star_n, yc_val)
                    # obs_cls.get_star_llm_evol_err(star_n, l_or_lm, yc_val, 1.0, 0.1)                  # ERRORS L/LM
                # mdot1, mdot2 = obs_cls.get_star_mdot_err(star_n, l_or_lm, yc_val, 1.0, 0.1, 'nugis')           # ERRORS Mdot
                mdot = obs_cls.get_num_par('mdot', star_n)
                plt.plot([mdot, mdot], [llm1, llm2], '-', color=obs_cls.get_class_color(star_n))

                # ax.errorbar(mdot_obs[i], llm_obs[i], yerr=[[llm1],  [llm2]], fmt='--.', color=obs_cls.get_class_color(star_n))


            plt.plot(mdot_obs[i], llm_obs[i], marker=obs_cls.get_clss_marker(star_n), markersize='9',
                     color=obs_cls.get_class_color(star_n), ls='')  # plot color dots)))
            ax.annotate('{}'.format(int(star_n)), xy=(mdot_obs[i], llm_obs[i]),textcoords='data')  # plot numbers of stars

            # t = obs_cls.get_num_par('t', star_n)
            # ax.annotate('{}'.format("%.2f" % t), xy=(mdot_obs[i], llm_obs[i]), textcoords='data')  # plot numbers of stars


            # v_inf = obs_cls.get_num_par('v_inf', star_n)
            # tau_cl = Opt_Depth_Analythis(30, v_inf, 1., 1., mdot_obs[i], 0.20)
            # tau = tau_cl.anal_eq_b1(1.)
            # # # # ax.annotate(str(int(tau)), xy=(mdot_obs[i], llm_obs[i]), textcoords='data')  # plo
            # ax.annotate('{} {}'.format(str(int(tau)), eta), xy=(mdot_obs[i], llm_obs[i]),
            #             textcoords='data')  # plot numbers of stars

            if obs_cls.get_star_class(star_n) not in classes:
                plt.plot(mdot_obs[i], llm_obs[i], marker=obs_cls.get_clss_marker(star_n), markersize='9',
                         color=obs_cls.get_class_color(star_n), ls='',
                         label='{}'.format(obs_cls.get_star_class(star_n)))  # plot color dots)))
                classes.append(obs_cls.get_star_class(star_n))

        print('\t__PLOT: total stars: {}'.format(len(obs_cls.stars_n)))
        print(len(mdot_obs), len(llm_obs))

        # fit = np.polyfit(mdot_obs, llm_obs, 1)  # fit = set of coeddicients (highest first)
        # f = np.poly1d(fit)
        # fit_x_coord = np.mgrid[(mdot_obs.min() - 1):(mdot_obs.max() + 1):1000j]

        mdot_grid = np.mgrid[(mdot_obs.min() - 1):(mdot_obs.max() + 1):100j]
        x_coord, y_coord = Math.fit_plynomial(mdot_obs, llm_obs, 1, 100, mdot_grid)
        ax.plot(x_coord, y_coord, '-.', color='blue')

        min_mdot, max_mdot = obs_cls.get_min_max('mdot')
        min_llm, max_llm = obs_cls.get_min_max(l_or_lm, yc_val)

        ax.set_xlim(min_mdot - 0.2, max_mdot + 0.2)
        ax.set_ylim(min_llm - 0.05, max_llm + 0.05)

        ax.set_ylabel(Labels.lbls(l_or_lm))
        ax.set_xlabel(Labels.lbls('mdot'))
        ax.grid(which='major', alpha=0.2)
        # ax.legend(bbox_to_anchor=(1, 1), loc='upper right', ncol=1)

        print('Yc:{}'.format(yc_val))
        ax.text(0.9, 0.9, 'Yc:{}'.format(yc_val), style='italic',
                bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes)

        # ax.text(min_mdot, max_llm, 'Yc:{}'.format(yc_val), style='italic',
        #         bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10})

        # l_grid = np.mgrid[5.2:6.:100j]
        # ax.plot(Physics.l_mdot_prescriptions(l_grid, ), l_grid, '-.', color='orange', label='Nugis & Lamers 2000')
        #
        # ax.plot(Physics.yoon(l_grid, 10 ** 0.02), l_grid, '-.', color='green', label='Yoon 2017')

    @staticmethod
    def plot_obs_t_llm_mdot_int(ax, t_llm_mdot, obs_cls, l_or_lm, yc1 = None, yc2 = None, lim_t1 = None, lim_t2 = None):

        if lim_t1 == None: lim_t1 = t_llm_mdot[0, 1:].min()
        if lim_t2 == None: lim_t2 = t_llm_mdot[0, 1:].max()

        yc_val = t_llm_mdot[0, 0] #

        classes = []
        classes.append('dum')
        x = []
        y = []
        for star_n in obs_cls.stars_n:
            xyz = obs_cls.get_xyz_from_yz(yc_val, star_n, l_or_lm, 'mdot',
                                          t_llm_mdot[0,1:], t_llm_mdot[1:,0], t_llm_mdot[1:,1:], lim_t1, lim_t2)

            if xyz.any():
                x = np.append(x, xyz[0, 0])
                y = np.append(y, xyz[1, 0])

                # print('Star {}, {} range: ({}, {})'.format(star_n,l_or_lm, llm1, llm2))

                for i in range(len(xyz[0, :])):

                    plt.plot(xyz[0, i], xyz[1, i], marker=obs_cls.get_clss_marker(star_n), markersize='9',
                             color=obs_cls.get_class_color(star_n), ls='')  # plot color dots)))
                    ax.annotate(int(star_n), xy=(xyz[0, i], xyz[1, i]),
                                textcoords='data')  # plot numbers of stars

                    if obs_cls.get_star_class(star_n) not in classes:
                        plt.plot(xyz[0, i], xyz[1, i], marker=obs_cls.get_clss_marker(star_n), markersize='9',
                                 color=obs_cls.get_class_color(star_n), ls='',
                                 label='{}'.format(obs_cls.get_star_class(star_n)))  # plot color dots)))
                        classes.append(obs_cls.get_star_class(star_n))

                    # -------------------------OBSERVABLE ERRORS FOR L and Mdot ----------------------------------------
                    lm_err1, lm_err2 = obs_cls.get_star_lm_obs_err(star_n, yc_val)
                    ts1_b, ts2_b, ts1_t, ts2_t = obs_cls.get_star_ts_obs_err(star_n, t_llm_mdot, yc_val, lim_t1, lim_t2)
                    ts_coord = [ts1_b, ts2_b, ts2_t, ts1_t]
                    lm_coord = [lm_err1, lm_err1, lm_err2, lm_err2]
                    ax.add_patch(patches.Polygon(xy=list(zip(ts_coord, lm_coord)), fill=True, alpha=.4,
                                                 color=obs_cls.get_class_color(star_n)))
                    # ax.plot([xyz[0, i], xyz[0, i]], [lm_err1, lm_err2], '-',
                    #         color='gray')

                    # ax.plot([ts_err1, ts_err2], [xyz[1, i], xyz[1, i]], '-',
                    #         color='gray')
                    # print('  :: Star {}. L/M:{}(+{}/-{}) ts:{}(+{}/-{})'
                    #       .format(star_n, "%.2f" % xyz[0, i], "%.2f" % np.abs(xyz[0, i]-lm_err2), "%.2f" % np.abs(xyz[0, i]-lm_err2),
                    #               "%.2f" % xyz[1, i],  "%.2f" % np.abs(xyz[1, i]-ts_err2), "%.2f" % np.abs(xyz[1, i]-ts_err1)))
                    # ax.plot([ts_err1, ts_err2], [lm_err1, lm_err2], '-', color='gray')

                    #



                    if l_or_lm == 'lm':
                        lm1, lm2 = obs_cls.get_star_lm_err(star_n, yc_val)
                        ts1, ts2 = obs_cls.get_star_ts_err(star_n, t_llm_mdot, yc_val, lim_t1, lim_t2)
                        ax.plot([ts1, ts2], [lm1, lm2], '-',
                                color=obs_cls.get_class_color(star_n))

                        # ax.plot([xyz[0, i], xyz[0, i]], [lm1, lm2], '-',
                        #         color=obs_cls.get_class_color(star_n))
                        # ax.plot([ts1, ts2], [xyz[1, i], xyz[1, i]], '-',
                        #         color=obs_cls.get_class_color(star_n))


                        # ax.add_patch(patches.Rectangle((xyz[0, i] - ts1, xyz[1, i] - lm1), ts2 + ts1, lm2 + lm1,
                        #                                alpha=.3, color=obs_cls.get_class_color(star_n)))

                        # ax.add_patch(patches.Rectangle((xyz[0, i] - ts1, xyz[1, i] - lm1), ts2 + ts1, lm2 + lm1,
                        #                                alpha=.3, color=obs_cls.get_class_color(star_n)))

                        # ax.plot([xyz[0, i] - ts1, xyz[1, i] - lm1], [xyz[0, i]+ts2, xyz[1, i] + lm2], '-', color=obs_cls.get_class_color(star_n))
                        # ax.plot([xyz[0, i] - ts1, xyz[0, i]+ts2], [xyz[1, i] - lm1, xyz[1, i] + lm2], '-', color=obs_cls.get_class_color(star_n))

                        # ax.errorbar(xyz[0, i], xyz[1, i], yerr=[[lm1], [lm2]], fmt='--.', color = obs_cls.get_class_color(star_n))
                        # ax.errorbar(xyz[0, i], xyz[1, i], xerr=[[ts1], [ts2]], fmt='--.', color=obs_cls.get_class_color(star_n))



        fit = np.polyfit(x, y, 1)  # fit = set of coeddicients (highest first)
        f = np.poly1d(fit)
        fit_x_coord = np.mgrid[(x.min() - 1):(x.max() + 1):1000j]
        plt.plot(fit_x_coord, f(fit_x_coord), '-.', color='blue')

        # ax.text(0.9, 0.9,'Yc:{}'.format(yc_val), style='italic',
        #         bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10}, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)

        # ax.text(x.max(), y.max(), 'Yc:{}'.format(yc_val), style='italic',
        #         bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10})
        ax.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=1)



# class Opt_Depth_Analythis():
#
#
#     def __init__(self, v0, v_inf, R, b, mdot, k = 0.20):
#         self.v0 = v0 * 100000           # km-s -> cm/s
#         self.b = b
#         self.v_inf = v_inf * 100000     # km-s -> cm/s
#         self.R = R * Constants.solar_r  # r_sol -> cm
#         self.k = k   # electron scattering assumed by default
#         self.mdot = 10**mdot * (Constants.solar_m / Constants.yr)              #
#
#         self.v0vinf = self.v0/self.v_inf
#
#
#     def b_vel_low(self, r):
#         return self.v0 + (self.v_inf - self.v0) * (1 - (self.R / (r * Constants.solar_r)))**self.b
#
#     def anal_eq_b1(self, r):
#         logs = np.log(1 - ((self.R / (r * Constants.solar_r)) * (1 - self.v0vinf)) )
#         return -((self.k * np.abs(self.mdot)) / (4 * np.pi * self.R * (self.v_inf - self.v0))) * logs

class Opt_Depth_Analythis():


    def __init__(self, v0, v_inf, R, b, mdot, k = 0.20):
        self.v0 = v0 * 100000 * Constants.yr / Constants.solar_r           # km-s -> cm/s -> r_sol / yr
        self.b = b
        self.v_inf = v_inf * 100000 * Constants.yr / Constants.solar_r     # km-s -> cm/s -> r_sol / yr
        self.R = R   # r_sol -> cm
        self.k = k  * Constants.solar_m / Constants.solar_r ** 2           # cm^2/g -> sol_r^2 / sol_m
        self.mdot = 10**mdot              #

        self.v0vinf = self.v0/self.v_inf

    def eff_kappa(self, r, k_sp, beta, v_inf, v_esc_rs, rs):
        '''
        Effective kappa formula from Japaneese paper
        :param r:
        :param k_sp:
        :param beta:
        :param v_inf:
        :param v_esc_rs:
        :param rs:
        :return:
        '''
        return k_sp * (1 + (2*self.b*(v_inf / v_esc_rs)**2) * (1-(rs/r))**(2*beta - 1))

    def esc_vel(self, m, r):
        '''
        Escape Velocity at radius 'r' in sol_r / yr
        :param m:
        :param r:
        :return:
        '''
        g = 1.90809 * 10**5 # grav const in r_sol/m_sol
        return np.sqrt(2*g*m/r)

    def b_vel_low(self, r):
        return self.v0 + (self.v_inf - self.v0) * (1 - (self.R / (r * Constants.solar_r)))**self.b

    def anal_eq_b1(self, r):
        logs = np.log(1 - ((self.R / (r)) * (1 - self.v0vinf)) )
        return -((self.k * np.abs(self.mdot)) / (4 * np.pi * self.R * (self.v_inf - self.v0))) * logs

    def kappa_test(self):
        m = 20
        r = np.mgrid[1.0:10.0:100j]
        k = []
        for i in range(len(r)):
            # print(self.esc_vel(m, r[i]) )
            k = np.append(k, self.eff_kappa(r[i],0.2, 2, 1600,self.esc_vel(m,r[i]), 1.0))

        plt.plot(r, k)
        plt.show()

class Levels:

    def __init__(self):
        pass

    @staticmethod
    def get_levels(v_n, opal_used):
        if opal_used.split('/')[-1] == 'table8.data':

            if v_n == 'r':
                return [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2., 2.1, 2.2, 2.3, 2.4, 2.5]
            if v_n == 'm':
                levels = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
            if v_n == 'mdot':
                return [-6.0, -5.75, -5.5, -5.25, -5., -4.75, -4.5, -4.25, -4, -3.75, -3.5, -3.25, -3.]
                # levels = [-6.0, -5.9, -5.8, -5.7, -5.6, -5.5, -5.4, -5.3, -5.2, -5.1, -5., -4.9, -4.8, -4.7, -4.6, -4.5]
            if v_n == 'l':
                return [5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2]
            if v_n == 'lm':
                return [4.0, 4.05, 4.1, 4.15, 4.2, 4.25, 4.3, 4.35, 4.4, 4.45,
                          4.5, 4.55, 4.6, 4.65, 4.7, 4.75, 4.8, 4.85, 4.9, 4.95, 5.0]
            if v_n == 't':
                return [5.15, 5.16, 5.17, 5.18, 5.19, 5.20, 5.21, 5.22, 5.23, 5.24, 5.25, 5.26, 5.27, 5.28, 5.29, 5.30]

            if v_n == 'k':
                return [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]  # FOR log Kappa
            if v_n == 'rho':
                return [-10, -9.5, -9, -8.5, -8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4]
            # if v_n_z == 'r':   levels = [0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6,
            #                            1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0, 2.05, 2.10, 2.15, 2.20]
            if v_n == 'tau':
                return [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
            if v_n == 'm':
                return [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

        if opal_used.split('/')[-1] == 'table_x.data':

            if v_n == 'r':
                return [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2., 2.1]
            if v_n == 'm':
                levels = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
            if v_n == 'mdot':
                return [-6.0, -5.75, -5.5, -5.25, -5., -4.75, -4.5, -4.25, -4, -3.75, -3.5, -3.25, -3.]
                # levels = [-6.0, -5.9, -5.8, -5.7, -5.6, -5.5, -5.4, -5.3, -5.2, -5.1, -5., -4.9, -4.8, -4.7, -4.6, -4.5]
            if v_n == 'l':
                return [5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4]
            if v_n == 'lm':
                return [4.0, 4.05, 4.1, 4.15, 4.2, 4.25, 4.3, 4.35, 4.4, 4.45,
                          4.5, 4.55, 4.6, 4.65, 4.7, 4.75, 4.8, 4.85, 4.9, 4.95, 5.0]
            if v_n == 't':
                return [5.15, 5.16, 5.17, 5.18, 5.19, 5.20, 5.21, 5.22, 5.23, 5.24, 5.25, 5.26, 5.27, 5.28, 5.29, 5.30]

            if v_n == 'k':
                return [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]  # FOR log Kappa
            if v_n == 'rho':
                return [-10, -9.5, -9, -8.5, -8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4]
            # if v_n_z == 'r':   levels = [0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6,
            #                            1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0, 2.05, 2.10, 2.15, 2.20]
            if v_n == 'tau':
                return [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]


        if v_n == 'Ys' or v_n == 'ys':
            return [0.5, 0.55, 0.6, 0.65, 0.7,0.75,0.8,0.85,0.9,0.95,1.0]

        raise NameError('Levels are not found for <{}> Opal:{}'.format(v_n, opal_used))

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
