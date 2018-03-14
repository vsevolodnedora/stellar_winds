#-----------------------------------------------------------------------------------------------------------------------
# Set of classes including:
#   Read_Table    Reads the OPAL table
#   Row_Analyze   Supplementary, analyses rho and kappa for a given value of t.
#   OPAL_Interpol Uses Row_Analyze to go through all values of t, and interpolate between them
#   Table_Analyze Uses Row_Analyze to invert axis, from t-rho-kappa to t-kappa-rho, and interpolates.
#   New_Table     Generates new OPAL for a given Metallicity by interpolation
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
#-----------------------------------------------------------CLASSES-----------------------------------------------------
from Phys_Math_Labels import Errors
from Phys_Math_Labels import Math
from Phys_Math_Labels import Physics
from Phys_Math_Labels import Constants

from PhysPlots import PhysPlots
#-----------------------------------------------------------------------------------------------------------------------


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
            raise ValueError('t1 ({}) > t2 ({})'.format(t1,t2))
            # sys.exit('\t__Error. t1 ({}) > t2 ({}) in |check_t_lim| in |Table_Analyze|')
        # a = Errors.is_a_bigger_b(t1, t2,    '|check_t_lim|', True, ' wrong temp. limits')
        if t2 > self.t[-1]:
            raise ValueError('t2 {} > t[-1] {} '.format(t2, self.t[-1]))
            # sys.exit('\t__Error. |check_t_lim|, t2 {} > t[-1] {} '.format(t2, self.t[-1]))
        if t1 < self.t[0]:
            print('\t: t_array is: ({} , {}) consisting of {} elements' .format(self.t[0], self.t[-1], len(self.t)))
            raise ValueError('t1 {} < t[0] {}'.format(t1, self.t[0]))
            # sys.exit('t__Error. |check_t_lim| t1 {} < t[0] {}'.format(t1, self.t[0]))

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
            PhysPlots.k_vs_t(self.t, self.min_k, self.max_k, True, True, k1, k2, t1, t2, it1, it2, self.plot_dir)  # save but not show

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

    # @staticmethod
    # def interpolate_2d(x, y, z, x_coord, y_coord, depth):
    #
    #     x_coord = np.array(x_coord, dtype=float)
    #     y_coord = np.array(y_coord, dtype=float)
    #     # interpolating every row< going down in y.
    #     if len(x_coord)!=len(y_coord):
    #         raise ValueError('x and y coord must be equal in length (x:{}, y:{})'.format(len(x_coord),len(y_coord)))
    #     #
    #     # if x_coord.min() < x.min() or x_coord.max() > x.max():
    #     #     raise ValueError('x_min:{} < x.min:{} or x_max:{} > '.format(x_coord.min(), x.min()))
    #     # if
    #     #
    #     print(x.shape, y.shape, z.shape)
    #     new_z = []#np.zeros((len(y), len(x_coord)))
    #     for si in range(len(y)):
    #         # new_x[si,:] = Math.interp_row(x, z[si,:], x_coord)
    #         new_z = np.append(new_z, Math.interp_row(x, z[si,:], x_coord))
    #
    #     # inteprolating every column, going right in x.
    #     new_z2 = []#np.zeros(( len(y_coord), len(new_x[:,0]) ))
    #     for si in range(len(x)):
    #         # new_y[:,si] = Math.interp_row(y, new_x[:,si], y_coord)
    #         new_z2 = np.append(new_z2, Math.interp_row(y, new_z, y_coord))
    #
    #     print(new_z.shape, new_z2.shape)
    #
    #
    #     f = interpolate.interp2d(x, y, z, kind='cubic')
    #
    #
    #     print(f(x_coord, y_coord))
    #
    #
    #     return None


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

class New_Table:
    def __init__(self, path, tables, values, out_dir_name):
        self.plot_dir_name = out_dir_name
        self.tbls = []
        self.val = values
        if len(values)!=len(tables):
            sys.exit('\t___Error. |New_table, init| n of tables and values is different: {} != {}'.format(len(tables), len(values)))
        self.ntbls = 0
        for i in range(len(tables)):
            self.tbls.append(Read_Table(path + tables[i] + '.data'))
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
        fname = self.plot_dir_name + 'table_x.data'
        res = Math.combine(self.tbls[0].r,self.tbls[0].t, new_kappa)
        np.savetxt(fname,res,'%.3f','\t')
        return res