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

        arr_x, arr_y = Math.x_y_z_sort(arr_x, arr_y, np.empty(0,))

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
    def fit_polynomial(x, y, order, depth, new_x = np.empty(0, )):
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

        if order == 5:
            fit = np.polyfit(x, y, order)  # fit = set of coeddicients (highest first)
            f = np.poly1d(fit)
            lbl = '({}) + ({}*x) + ({}*x**2) + ({}*x**3) + ({}*x**4) + ({}*x**5)'.format(
                "%.3f" % f.coefficients[5],
                "%.3f" % f.coefficients[4],
                "%.3f" % f.coefficients[3],
                "%.3f" % f.coefficients[2],
                "%.3f" % f.coefficients[1],
                "%.3f" % f.coefficients[0]
            )
            # fit_x_coord = np.mgrid[(x.min()):(x.max()):depth*1j]
            # plt.plot(fit_x_coord, f(fit_x_coord), '--', color='black')

        if order == 6:
            fit = np.polyfit(x, y, order)  # fit = set of coeddicients (highest first)
            f = np.poly1d(fit)
            lbl = '({}) + ({}*x) + ({}*x**2) + ({}*x**3) + ({}*x**4) + ({}*x**5) + ({}*x**6)'.format(
                "%.3f" % f.coefficients[6],
                "%.3f" % f.coefficients[5],
                "%.3f" % f.coefficients[4],
                "%.3f" % f.coefficients[3],
                "%.3f" % f.coefficients[2],
                "%.3f" % f.coefficients[1],
                "%.3f" % f.coefficients[0]
            )
            # fit_x_coord = np.mgrid[(x.min()):(x.max()):depth*1j]
            # plt.plot(fit_x_coord, f(fit_x_coord), '--', color='black')

        if not order in [1,2,3,4,5,6]:
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
    def interpolate_value_table(table, xy_val):

        x = table[0, 1:]
        y = table[1:, 0]
        z = table[1:,1:]

        x_val = xy_val[0]
        y_val = xy_val[1]


        # x_i = Math.find_nearest_index(x, x_val)
        # y_i = Math.find_nearest_index(y, y_val)
        #
        # x = x.ravel()
        # x = (x[x != np.isnan])
        # y = y.ravel()
        # y = (y[y != np.isnan])
        # z = z.ravel()
        # z = (z[z != np.isnan])
        #
        # f = interpolate.SmoothBivariateSpline(x, y, z, kx=1, ky=1)
        #
        # res = f(x_val, y_val)

        def cropp_nan_in_y(x, y):
            if len(x)!=len(y): raise ValueError('Not equal sizes len(x) {} != {} len(y)'.format(len(x), len(y)))
            new_x = []
            new_y = []
            for i in range(len(y)):
                if np.isnan(y[i]):
                    pass
                    # print('Nan? y[{}]={}'.format(i, y[i]))
                else:
                    new_x = np.append(new_x, x[i])
                    new_y = np.append(new_y, y[i])
            return new_x, new_y

        # x_grid = np.mgrid[x.min():x.max():depth * 1j]
        z_y = np.zeros(1)
        y_c = []
        for i in range(len(y)):
            x_crop, z_crop = cropp_nan_in_y(x, z[i, :])
            if x_val >= x_crop.min() and x_val <= x_crop.max():
                z_y = np.vstack((z_y, interpolate.InterpolatedUnivariateSpline(x_crop, z_crop)(x_val)))
                y_c = np.append(y_c, y[i])

        z_y = np.delete(z_y, 0, 0)

        if len(z_y) == 1:
            return z_y
        else:
            res = interpolate.InterpolatedUnivariateSpline(y_c,z_y)(y_val)

            return res
        #
        # # y_grid = np.mgrid[y.min():y.max():depth * 1j]
        # z_x = np.zeros(1)
        # for i in range(len(x_grid)):
        #     z_x = np.vstack((z_x, Math.interpolate_arr(y, z_y[:, i], y_val, method)))
        #
        # z_x = np.delete(z_x, 0, 0).T
        #
        # res = Math.combine(x_grid, y_grid, z_x)
        # res[0, 0] = table[0, 0]
        #
        # return res

    @staticmethod
    def extrapolate(table, x_left, x_right, y_down, y_up, depth, method):
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
                if method in [1, 2, 3, 4, 5, 6]:
                    new_x, new_y = Math.fit_polynomial(x_arr, y_arr, method, depth, x_grid)
                    return new_y
                if method == 'Uni':
                    new_y = interpolate.UnivariateSpline(x_arr, y_arr)(x_grid)
                    return new_y
                if method == 'IntUni':
                    new_y = interpolate.InterpolatedUnivariateSpline(x_arr, y_arr)(x_grid)
                    return new_y
                if method == 'linear':
                    new_y = interpolate.interp1d(x_arr, y_arr, kind='linear', bounds_error=False)(x_grid)
                    return new_y
                if method == 'cubic':
                    new_y = interpolate.interp1d(x_arr, y_arr, kind='cubic', bounds_error=False)(x_grid)
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

        # if use_input_table:
        #     new_res = np.zeros(res.shape)
        #
        #
        #
        return res

    @staticmethod
    def extrapolate2(table, x_left, x_right, y_down, y_up, depth, method, use_old_table):
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
        z = table[1:, 1:]

        if x_left != None:
            x1 = x.min() - (x_left * (x.max() - x.min()) / 100)  #
        else:
            x1 = x.min()

        if x_right != None:
            x2 = x.max() + (x_right * (x.max() - x.min()) / 100)  #
        else:
            x2 = x.max()

        if y_down != None:
            y1 = y.min() - (y_down * (y.max() - y.min()) / 100)  #
        else:
            y1 = y.min()

        if y_up != None:
            y2 = y.max() + (y_up * (y.max() - y.min()) / 100)  #
        else:
            y2 = y.max()

        def int_pol(x_arr, y_arr, x_grid):

            if x_grid.min() == x_arr.min() and x_arr.max() == x_grid.max():
                return interpolate.InterpolatedUnivariateSpline(x_arr, y_arr)(x_grid)
            else:
                if method in [1, 2, 3, 4, 5, 6]:
                    new_x, new_y = Math.fit_polynomial(x_arr, y_arr, method, depth, x_grid)
                    return new_y
                if method == 'Uni':
                    new_y = interpolate.UnivariateSpline(x_arr, y_arr)(x_grid)
                    return new_y
                if method == 'IntUni':
                    new_y = interpolate.InterpolatedUnivariateSpline(x_arr, y_arr)(x_grid)
                    return new_y
                if method == 'linear':
                    new_y = interpolate.interp1d(x_arr, y_arr, kind='linear', bounds_error=False)(x_grid)
                    return new_y
                if method == 'cubic':
                    new_y = interpolate.interp1d(x_arr, y_arr, kind='cubic', bounds_error=False)(x_grid)
                    return new_y

        x_grid = np.mgrid[x1:x2:depth * 1j]
        y_grid = np.mgrid[y1:y2:depth * 1j]

        # ----- POLYNOMIAL --- FULL GRID ----------------------

        x_grid = np.mgrid[x1:x2:depth * 1j]
        z_y = np.zeros(len(x_grid))
        for i in range(len(y)):
            z_y = np.vstack((z_y, int_pol(x, z[i, :], x_grid)))

        z_y = np.delete(z_y, 0, 0)

        y_grid = np.mgrid[y1:y2:depth * 1j]
        z_x = np.zeros(len(y_grid))
        for i in range(len(x_grid)):
            z_x = np.vstack((z_x, int_pol(y, z_y[:, i], y_grid)))

        z_x = np.delete(z_x, 0, 0).T

        res = Math.combine(x_grid, y_grid, z_x)
        res[0, 0] = table[0, 0]

        if not use_old_table:
            return res

        # ----- INTERPOLATIONS --- PART OF A GRID -------------

        x_grid_ = x_grid[(x_grid >= x[0]) & (x_grid <= x[-1])]  # removing the rest
        y_grid_ = y_grid[(y_grid >= y[0]) & (y_grid <= y[-1])]  # removing the rest

        z_y = np.zeros(len(x_grid_))
        for i in range(len(y)):
            z_y = np.vstack((z_y, interpolate.InterpolatedUnivariateSpline(x, z[i, :])(x_grid_)))

        z_y = np.delete(z_y, 0, 0)


        z_x = np.zeros(len(y_grid_))
        for i in range(len(x_grid_)):
            z_x = np.vstack((z_x, interpolate.InterpolatedUnivariateSpline(y, z_y[:, i])(y_grid_)))

        z_x = np.delete(z_x, 0, 0).T

        res_ = Math.combine(x_grid_, y_grid_, z_x)

        # ---- COMBINING THE TWO TABLES by replacing the polynom-obtained with interpolated in the range -------
        zz__ = np.zeros((len(y_grid), len(x_grid)))

        x_grid_ = res_[0,1:]
        y_grid_ = res_[1:,0]
        zz_ = res_[1:, 1:]

        new_res = np.array(res)
        new_zz = new_res[1:, 1:]

        # Taking the polynimial table [depth x depth] and filling the values that has not been extrapolated with the
        # interpolated ones, to preserve accuracy in that region
        for i in range(len(x_grid_)):
            for j in range(len(y_grid_)):
                if x_grid_[i] in x_grid and y_grid_[j] in y_grid:
                    i_coord = Math.find_nearest_index(x_grid, x_grid_[i])
                    j_coord = Math.find_nearest_index(y_grid, y_grid_[j])

                    new_zz[j_coord, i_coord] = zz_[j, i]

        # for i in range(len(x_grid)):
        #     for j in range(len(y_grid)):
        #         if x_grid[i] in x_grid_ and y_grid[j] in y_grid_:
        #             print('i:{} j:{}'.format(i,j))
        #             # print('ind_x: {}, ind_y: {}'.format())
        #             val = zz_[np.where(y_grid_ == y_grid[j]), np.where(x_grid_ == x_grid[i])]
        #             zz__[j, i] = val
        #         else:
        #             zz__[j, i] = zz_[j, i]


        # for i in range(len(x_grid_)):
        #     for j in range(len(y_grid_)):
        #         if x_grid_[i] in x_grid and y_grid_[j] in y_grid:
        #             print('i:{} j:{}'.format(i,j))
        #             # print('ind_x: {}, ind_y: {}'.format())
        #             val = zz_[np.where(x_grid == x_grid_[i]), np.where(y_grid == y_grid_[j])]
        #             res__[j, i] = val

        # res__[0, 0] = table[0, 0]

        # if use_input_table:
        #     new_res = np.zeros(res.shape)
        #
        #
        #

        final = Math.combine(x_grid, y_grid, new_zz, table[0,0])
        return np.array([x1,x2,y1,y2]), final

    @staticmethod
    def extrapolate_value(x, y, x_val, method, ax_plot=None):
        '''
        Select a method to extrapolate a value x_val for given x (raw) and y (raw)
        available methods: 'IntUni', 'Uni', 'poly4', 'poly3', 'poly2' , 'poly1', and 'test' to plot all of them.
        ax_plot - is a class 'plot' in plotting is needed.
        :param ax_plot:
        :param x:
        :param y:
        :param x_val:
        :param method:
        :return:
        '''

        x, y = Math.x_y_z_sort(x, y, np.empty(0, ), 0)
        y_val = None

        # x_grid = np.mgrid[np.append(x, x_val).min():np.append(x, x_val).max():100j]

        if ax_plot != None:
            ax_plot.plot(x, y, '.', color='black')

        if method == 'IntUni' or method == 'test':
            y_val = interpolate.InterpolatedUnivariateSpline(x, y)(x_val)
            if ax_plot != None:
                ax_plot.plot(x_val, y_val, 'x', color='magenta',
                             label='IntUni x_p:{} y_p:{}'.format("%.2f" % x_val, "%.2f" % y_val))
                # ax_plot.plot(np.append(x, x_val), np.append(y, y_val1), '-', color='magenta')

        # y_grid = interpolate.InterpolatedUnivariateSpline(x_grid, np.append(y, y_val))(ga_table[0, 3])

        if method == 'Uni' or method == 'test':
            y_val = interpolate.UnivariateSpline(x, y)(x_val)
            if ax_plot != None:
                ax_plot.plot(x_val, y_val, 'x', color='red',
                             label='Uni x_p:{} y_p:{}'.format("%.2f" % x_val, "%.2f" % y_val))
                # ax_plot.plot(np.append(x, x_val), np.append(y, y_val2), '-', color='red')
            # tau_cr2 = interpolate.UnivariateSpline(md_tau_sh[:, 0], md_tau_sh[:, 1])(ga_table[0, 3])

        if method == 'poly4' or method == 'test':
            # y_val2 = Math.fit_plynomial(md_tau_sh[:,0], md_tau_sh[:,1], 4, 0, grid_mdot)
            tmp, y_val = Math.fit_polynomial(x, y, 4, 0, np.array([x_val]))
            if ax_plot != None:
                ax_plot.plot(x_val, y_val, 'x', color='blue',
                             label='poly4 x_p:{} y_p:{}'.format("%.2f" % x_val, "%.2f" % y_val))
                # ax_plot.plot(np.append(x, x_val), np.append(y, y_val3), '-', color='blue')

        if method == 'poly3' or method == 'test':
            # y_val2 = Math.fit_plynomial(md_tau_sh[:,0], md_tau_sh[:,1], 4, 0, grid_mdot)
            tmp, y_val = Math.fit_polynomial(x, y, 3, 0, np.array([x_val]))
            if ax_plot != None:
                ax_plot.plot(x_val, y_val, 'x', color='orange',
                             label='poly3 x_p:{} y_p:{}'.format("%.2f" % x_val, "%.2f" % y_val))
                # ax_plot.plot(np.append(x, x_val), np.append(y, y_val4), '-', color='orange')

        if method == 'poly2' or method == 'test':
            # y_val2 = Math.fit_plynomial(md_tau_sh[:,0], md_tau_sh[:,1], 4, 0, grid_mdot)
            tmp, y_val = Math.fit_polynomial(x, y, 2, 0, np.array([x_val]))
            if ax_plot != None:
                ax_plot.plot(x_val, y_val, 'x', color='cyan',
                             label='poly2 x_p:{} y_p:{}'.format("%.2f" % x_val, "%.2f" % y_val))
                # ax_plot.plot(np.append(x, x_val), np.append(y, y_val5), '-', color='cyan')

        if method == 'poly1' or method == 'test':
            # y_val2 = Math.fit_plynomial(md_tau_sh[:,0], md_tau_sh[:,1], 4, 0, grid_mdot)
            tmp, y_val = Math.fit_polynomial(x, y, 1, 0, np.array([x_val]))
            if ax_plot != None:
                ax_plot.plot(x_val, y_val, 'x', color='green',
                             label='poly1 x_p:{} y_p:{}'.format("%.2f" % x_val, "%.2f" % y_val))
                # ax_plot.plot(np.append(x, x_val), np.append(y, y_val6), '-', color='green')

        if not method in ['poly1', 'ploy2', 'poly3', 'poly4', 'Uni', 'IntUni', 'test']:
            raise NameError('method is not recognised {} \n Available: {}'.format(method,
                                                                                  ['poly1', 'ploy2', 'poly3', 'poly4',
                                                                                   'Uni', 'IntUni', 'test']))
        if y_val == None : raise ValueError('Extrapolation Failed')
        return y_val

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

    @staticmethod
    def interpolate_arr(x, y, new_x, interp_method):
        '''

        :param x:
        :param y:
        :param new_x:
        :param interp_method: 'IntUni' 'Uni' '1dCubic' '1dLinear'
        :return:
        '''

        if interp_method == 'IntUni':
            new_y = interpolate.InterpolatedUnivariateSpline(x, y)(new_x)
            return new_y
        if interp_method == 'Uni':
            new_y = interpolate.UnivariateSpline(x, y)(new_x)
            return new_y
        if interp_method == '1dCubic':
            new_y= interpolate.interp1d(x, y, kind='cubic', bounds_error=False)(new_x)
            return new_y
        if interp_method == '1dLinear':
            new_y = interpolate.interp1d(x, y, kind='linear', bounds_error=False)(new_x)
            return new_y

        if interp_method == 'poly5':
            # y_val2 = Math.fit_plynomial(md_tau_sh[:,0], md_tau_sh[:,1], 4, 0, grid_mdot)
            new_x, new_y = Math.fit_polynomial(x, y, 5, 0, new_x)
            return new_y

        if interp_method == 'poly4':
            # y_val2 = Math.fit_plynomial(md_tau_sh[:,0], md_tau_sh[:,1], 4, 0, grid_mdot)
            new_x, new_y = Math.fit_polynomial(x, y, 4, 0, new_x)
            return new_y

        if interp_method == 'poly3':
            # y_val2 = Math.fit_plynomial(md_tau_sh[:,0], md_tau_sh[:,1], 4, 0, grid_mdot)
            new_x, new_y = Math.fit_polynomial(x, y, 3, 0, new_x)
            return new_y

        if interp_method == 'poly2':
            # y_val2 = Math.fit_plynomial(md_tau_sh[:,0], md_tau_sh[:,1], 4, 0, grid_mdot)
            new_x, new_y = Math.fit_polynomial(x, y, 2, 0, new_x)
            return new_y

        if interp_method == 'poly1':
            # y_val2 = Math.fit_plynomial(md_tau_sh[:,0], md_tau_sh[:,1], 4, 0, grid_mdot)
            new_x, new_y = Math.fit_polynomial(x, y, 1, 0, new_x)
            return new_y

        raise NameError('Interpolation method is not recognised, Use: [IntUni Uni 1dCubic 1dLinear, poly1..5]')


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
    def lm_l__to_m(lm, l):
        return 10**(l - lm)

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

        return 0.25*np.log10( l/(4*np.pi * Constants.steph_boltz* r**2))

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
                  'is beyond the mdot range ({}, {})'
                  '\n\t\t Uning Extrapolation'.format(num_of_model, "%.2f" % star_y_coord, "%.2f" % star_z_coord,
                                                             "%.3f" % z_row_for_star_z.max(), "%.2f" % z_row_for_star_z.min()))
            int_star_x_coord = [Math.extrapolate_value(z_row_for_star_z, x_1d_arr, star_z_coord, 'IntUni')]
            print('\t\t Extrapl. Result: x:{} (where x_arr: [{} {}]'.format(int_star_x_coord, "%.2f" % x_1d_arr.max(), "%.2f" % x_1d_arr.min()))

            z_fill = np.zeros(len(int_star_x_coord))
            z_fill.fill(star_z_coord)
            y_fill = np.zeros(len(int_star_x_coord))
            y_fill.fill(star_y_coord)

            # return np.vstack((np.array(int_star_x_coord), y_fill, z_fill))
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

