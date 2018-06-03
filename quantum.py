# --- Quntum ---
#
#
# -- --- --

import numpy as np
import matplotlib.pyplot as plt
import base64

# quantum computer is solving the shrodinger equation on the side
# from initial ground state we need to obtain a certain quatum state, where later you need to do a meausrement.
# State -> evolove unitary -> computation.
# Zero state -> |0>
# degrees of freedom can be pushed back in case of unitary evolution
# the way you peak the unitary, you put is
# it selects the most correct solution out of set of outputs.
#
# conisder a set of boxes: instead of checking them one by one it
# unitary evolution:
# Consider boxes 1,2,3,4,5,6,7...
# let boll be in box number 3
# in binary it is 001, 010, 011, ... names of the boxes
# 3 qubuts - encodes - 8 solutions (8 dimensional Hilbert space, D^2^n)
#
# Let the boll be in first box. Apply the unitary: [find a hamiltonian that generates a unitary]
# |100> -> [H(x)3] -> [G]-> [G] ... [G] -> measurement
# the H = 1/sqrt(2) (1 1 / -1 1 ) matrix
# H(x)2 = (1/sqrt(2)(H H / -H H) matrix (tensor multiplication)
# Grover Unitary G = Oracle H^(x)n -> phase -> H^(x)n
# oracle checking if the linear combination of solutions is a solution for the sysstem, applyting the unitary, to check.
#
# Consider
# H^(x)|0>|0>0> =  1/sqrt(2) (|0> + ||1>)) 1/sqrt(2) (|0> + |1>)
# Dedine H^(x)n|0> = \psi
# a| correct sol> +

# n = 2
# boxes = 2**n
# steps = np.int(np.sqrt(2**n))
# rightboxes = np.random.randint(boxes)
#
#
# stringoracle='IyEvdXNyL2Jpbi9lbnYgcHl0aG9uCiNjb2Rpbmc6IHV0ZjgKCiNpbXBvcnQgbnVtcHkKI153aWxsIGJlIGltcG9ydGVkIGluIG1haW4gZmlsZSBhbnl3YXkKCmRlZiBvcmFjbGUocHNpKToKICAgIFU9bnAuZGlhZyhbMSwxLDEsMSwxLC0xLDEsMSwxLDEsMSwxLDEsMSwxLDFdKQogICAgcmV0dXJuIFUuZG90KHBzaSkKCgo='
#
# key = base64.b64decode(stringoracle)
#
# eval(compile(key,'<string>','exec'))
#
#
#
# hsqr2 = np.ones((2, 2))
# hsqr2[1,1] = -1
#
# def grov(v, h):
#     '''by Sven'''
#     v1 - oracle(v)
#     v1 = h.dot(phase.dot(h.dot(v1)))
#
#
#
# h = hsqr2
# for i in range(n):
#     h = np.kron(h,hsqr2)
#
# h = h/np.sqrt(boxes)
# # oracle = np.identity(boxes)
# # oracle[rightboxes] = -1
#
# phase = -1*np.identity(boxes)
# phase[0] = 1
#
# qbits = np.zeros(boxes)
#
# psi = np.zeros(boxes)
# psi[0] = 1
#
#
# first = psi.dot(h)
#
#
# print(oracle(psi))


# ---------------------

stringoracle='IyEvdXNyL2Jpbi9lbnYgcHl0aG9uCiNjb2Rpbmc6IHV0ZjgKCiNpbXBvcnQgbnVtcHkKI153aWxsIGJlIGltcG9ydGVkIGluIG1haW4gZmlsZSBhbnl3YXkKCmRlZiBvcmFjbGUocHNpKToKICAgIFU9bnAuZGlhZyhbMSwxLDEsMSwxLC0xLDEsMSwxLDEsMSwxLDEsMSwxLDFdKQogICAgcmV0dXJuIFUuZG90KHBzaSkKCgo='

key = base64.b64decode(stringoracle)
print(key)
eval(compile(key,'<string>','exec'))


n = 4
boxes = n**2

psi = np.zeros(boxes)
psi[0] = 1

h = np.array(([1,1],[1,-1]))

phi = -1*np.identity(boxes)
phi[0,0] = 1


h_ = h
print(h_)
for i in range(n-1):
    # print(n)
    h = np.kron(h, h_)
h = h / np.sqrt(boxes)


def get_g(h, psi, phase):
    first = h.dot( phase.dot(h.dot(oracle(psi))) )
    return first



def get_grove(h, psi):
    psi = h.dot(psi)

    return psi
    # print(psi)


psi = get_grove(h, psi)# print(get_grove(h, psi))
# print(get_g(h, psi, phi))

steps = np.int(np.sqrt(2**n))


for i in range(steps):
    psi = get_g(h, psi, phi)

print(psi)
print(psi**2)


