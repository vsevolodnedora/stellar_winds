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

import os
import subprocess


#
#   This script is changing the m.dat file, imposing new FNAME and new DMDT (the same ones)
#


import os
import subprocess
import shutil

'''============================================================================================'''

prime_fold_names = ['10sm', '11sm', '12sm', '13sm', '14sm', '15sm', '16sm', '17sm', '18sm', '19sm', '20sm', '21sm', '22sm', '23sm', '24sm', '25sm', '26sm', '27sm', '28sm', '29sm', '30sm']
secon_fold_names = ['y10', 'y9', 'y8', 'y7', 'y6', 'y5', 'y4', 'y3', 'y2', 'y1']

dir_to_copy = 'sp'

ref_mdat = 'ref_m.dat'
var_name1 = 'FNAME'
van_name2 = 'DMDT'

mdot1 = -3.
mdot2 = -6.
mdot_step = -0.05

'''==========================================================================================='''

mdot_array = np.arange(mdot1, mdot2, mdot_step)
print('In main folders: {}'.format(prime_fold_names))
print('& secondary    : {}'.format(secon_fold_names))
print('Mdot values from {} to {} with step {} (iterations {})'.format(mdot1, mdot2, mdot_step, len(mdot_array)))
print('Total number of models {}'.format(len(prime_fold_names)*len(secon_fold_names)*len(mdot_array)))

go = input("Agree y/n: ")

# ---

def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

def del_sp_dirs():
    for pfold in prime_fold_names:
        for sfold in secon_fold_names:
            directory = pfold + '/' + sfold + '/' + dir_to_copy
            shutil.rmtree(directory, ignore_errors=True)
            # os.rmdir(directory)

def create_sp_dirs():

    for pfold in prime_fold_names:
        for sfold in secon_fold_names:
            directory = pfold + '/' + sfold + '/' + dir_to_copy
            os.makedirs(directory)

def copy_sp_dir():
    for pfold in prime_fold_names:
        for sfold in secon_fold_names:
            directory = pfold + '/' + sfold + '/' + dir_to_copy
            copytree('./', directory)

# ---
def create_mdat_file2(mdot, path, ref_mdat):

    f = open(path+ref_mdat, 'r').readlines()
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


    new_file_name = path+'m.dat'
    # new_file_name = 're_' + str( "%.2f" % (mdot)) + 'm.dat'
    f = open(new_file_name, 'w').writelines(new_lines)

# --- --- --- MAIN CYCLE --- --- ---
if go == 'y':
    print('\n\n <<<< <<<< <<<< START >>>> >>>> >>>>\n\n')
    for pfold in prime_fold_names:
        for sfold in secon_fold_names:

            directory = '/media/vnedora/HDD/sse/ga_z0008/' + pfold + '/' + sfold + '/' + dir_to_copy
            os.chdir(directory)
            print('\n\n <<<< <<<< <<<< {} >>>> >>>> >>>>\n\n'.format(directory))

            for mdot in mdot_array:
                create_mdat_file2(-1 * mdot, directory+'/', ref_mdat) # creating m.dat for every mdot iteration.
                file_name = 'f' + sfold # like fy10.bin1
                pass_mdot = str("%.2f" % (-1 * mdot))

                subprocess.call(['./auto_ev_rd_bb.sh'.format(directory), file_name, pass_mdot])

                # exc = '(cd {} && /auto_ev_rd_bb.sh)'.format(directory)
                # subprocess.call([exc, file_name, pass_mdot])
                # os.system('python '+directory+'/run_mass_loss.py {} {} {} {}'.format(file_name, mdot1, mdot2, mdot_step, 'n'))
                # subprocess.call(['python '+directory+'/run_mass_loss.py',file_name, str(mdot1), str(mdot2), str(mdot_step, 'n'])
                # subprocess.call(['python '+directory+'run_mass_loss.py', file_name, -3. -6, -0.05, 'n'])
                print('\n\n <<<< <<<< <<<< DONE >>>> >>>> >>>>')
else:
    print('See you later :)')



'''=================================================================================================================='''

