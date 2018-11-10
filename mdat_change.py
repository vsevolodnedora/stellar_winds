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

'''===================================================m.dat=change==================================================='''
#
#   This script is changing the m.dat file, imposing new FNAME and new DMDT (the same ones)
#
'''=================================================================================================================='''
ref_mdat = 'ref_m.dat'
var_name1 = 'FNAME'
van_name2 = 'DMDT'

def get_min_max_step():
    start = input("First mdot: ")
    stop  = input("Last  mdot: ")
    step =  input("Step  mdot: ")

    mdot_array = np.arange(float(start), float(stop), float(step))

    return mdot_array

def create_mdat_file(mdot):

    f = open(ref_mdat, 'r').readlines()
    all_lines = []
    new_lines = []
    for line in f:
        all_lines.append(line)

        # ------------------------------------Replacing the line with FNAME variable
        if var_name1 in line.split():
            # print(line.split(), len(line.split()), line.split()[2])

            new_line = line.split()

            new_line[2] = 're_' + str("%.2f" % mdot) # changing the name of the 'FNAME'

            new_spaced_line = new_line[0] + ' = ' + new_line[2] + '    NR    = ' + new_line[5] +\
                              '       IOUT  = ' + new_line[8] + '       IPRN  = ' + new_line[-1] + '\n' #combining everything back

            new_lines.append(new_spaced_line)

            # ------------------------------------Replacing the line with DMDT variable
        else:

            if van_name2 in line.split():

                half_line = line.split(van_name2) #

                new_line = half_line[-1].split()

                half_line[1] = '-' + str( "%.3e" % (10**( (-1.) * mdot)) ).replace('e','d')

                new_spaced_line = half_line[0] + van_name2 + ' = ' + half_line[1] + '  IDMDT = '+ new_line[-1] + '\n'

                new_lines.append(new_spaced_line)

            else:

                new_lines.append(line)



    # DMSDYF= 5.d-2      IKAP  = 3          DMDT  = -1.000d-4  IDMDT = 4
    # DMSDYF= 5.d-2      IKAP  = 3          DMDT = -1.000d-04  IDMDT = 4


    new_file_name = 'm.dat'
    # new_file_name = 're_' + str( "%.2f" % (mdot)) + 'm.dat'
    f = open(new_file_name, 'w').writelines(new_lines)


fname = input(".bin1 file: ")

file_name = fname

mdot_array = get_min_max_step()

print('TOTAL NUMBER OF ITERATION: {} '.format(len(mdot_array)))

yn = input(" y/n ")
print('\n')

if yn == 'y':
    for mdot in mdot_array:

        create_mdat_file(-1 * mdot)

        pass_mdot = str("%.2f" % (-1 * mdot) ) # Format 4.20

        # bashCommand = "cwm --rdf test.rdf --ntriples > test.nt"
        #
        # process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        # output, error = process.communicate()

        # subprocess.Popen("./x.sh {} {}".format(file_name, pass_mdot ), shell=True)
        subprocess.call(['./x.sh', file_name, pass_mdot])

        print('DONE!')

else:
    print('See you later :)')











# bashCommand = "chmod u+x x.sh"
# process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
# output, error = process.communicate()
#
# bashCommand = "./x.sh"
# process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
# output, error = process.communicate()





# def get_min_max_step():
#     start = input("First mdot: ")
#     stop  = input("Last  mdot: ")
#     step =  input("Step  mdot: ")
#
#     mdot_array = np.arange(float(start), float(stop), float(step))
#
#     return mdot_array
#
# if __name__ == '__main__':
#     get_min_max_step()

#
#
# print(np.linspace(-6,-4,10))
# print(np.arange(-6,-4,0.1))
# # print("%2e" % 10**np.arange(-6,-4,0.1) )
# mdotset = np.arange(-6,-4, 0.1)
#
# print( str( [ "%.3e" % 10**mdot for mdot in mdotset ] ).replace('e','d') )
#
# print( str( ["%.2f" %  (-1 * mdot) for mdot in mdotset ] ) )
#
# # print([ 10**mdot for mdot in mdotset])
#
#
#
#
# file_name = 'primordial_m.dat'
# var_name1 = 'FNAME'
# value1 = input("New_Mass_Loss")  # Python 3
# print('Your input: {}'.format(value1))
#
# # FNAME = re_1.0d-4  NR    = 9999       IOUT  = 0001       IPRN  = 0500001
# # FNAME = re_1d-4    NR    = 9999       IOUT  = 1          IPRN  = 0500001
#
#
# f = open(file_name, 'r').readlines()
# all_lines = []
# new_lines = []
# for line in f:
#     all_lines.append(line)
#     # print(line.split())
#     if var_name1 in line.split():
#         print(line.split(), len(line.split()), line.split()[2])
#
#
#         new_line = line.split()
#
#         new_line[2] = 're_' + value1 # changing the name of the FNAME
#
#         new_spaced_line = new_line[0] + ' = ' + new_line[2] + '  NR    = ' + new_line[5] +\
#                           '       IOUT  = ' + new_line[8] + '       IPRN  = ' + new_line[-1] + '\n' #combining everything back
#
#         print(new_line)
#         print(line)
#         print(new_spaced_line)
#
#         new_lines.append(new_spaced_line)
#
#     else:
#
#         new_lines.append(line)
#
#
# new_file_name = 're_' + value1 + 'm.dat'
# f = open(new_file_name, 'w').writelines(new_lines)