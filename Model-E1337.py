import random
from flask import Flask, abort, redirect, request, Response, session
from jinja2 import Template
import base64, json, os, random, re, subprocess, time, xml.sax
from io import StringIO
import numpy as np

class systemUnsolvable(Exception):
	pass

class bcolors:
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


def unlock():
	code = int(request.form['code'])
	cur = next()

	if code == cur:
		return 'Unlocked successfully.  Flag: ' + flags[1]
	else:
		return 'Code incorrect.  Expected %08i' % cur

def setup(seed):
	global state
	state = 0
	for i in range(16):
		cur = seed & 3 									# get the 2 lowest bits of seed
		seed >>= 2										# push those out of the seed
		state = (state << 4) | ((state & 3) ^ cur)		# state | 00 | state & b011 ^ (seed lowest 2)
														# old state bin | 2x0| 
		state |= cur << 2								# as before | seed lowest 2 | state & b011 ^ (seed lowest 2)

		# We meed more relations :(... So lets get some
		#
		# say state= ABCD.....XYZ
		# A = seed[1]
		# B = seed[0]
		#
		# CA = 0 (=state[0])
		# DB = 0 (=state[1])
		#
		# W = seed[1]
		# X = seed[0]
		# YW = U (=state[0])
		# ZX = V (=state[1])
		#
		# First 2 relations can be described by:
		# ABCD
		# 0000 | 0
		# 0000 | 0
		# 1010 | 0
		# 0101 | 0
		#
		# For every 4 steps (i.e. from A to E, E to I etc.) and tupel UVWXYZ we have:
		# UVWXYZ
		# 000000 | 0
		# 000000 | 0
		# 101010 | 0
		# 010101 | 0

def next(bits):
	global state

	ret = 0
	for i in range(bits):
		ret <<= 1										# ret becomes a variable where the i-th bit stores
														# the last bit of state in the i-th run
		ret |= state & 1 								# push the lowest bit of state to the right or ret
		state = (state << 1) ^ (state >> 61)			# we describe this with matrices later
		state &= 0xFFFFFFFFFFFFFFFF						# cutoff the 65th bit
		#state ^= 0xFFFFFFFFFFFFFFFF   we can ignore this since in the next calc the 1's simply cancel

		#   ABCD....WXYZ
		#=>(AD)(BD)(CA)(DA)....(WZ)(XZ)(YW)(ZW)			also described with matrices later
		for j in range(0, 64, 4):
			cur = (state >> j) & 0xF					# cur = state j-th bit to j+4
			cur = (cur >> 3) | (((cur >> 3) & 1) << 1) | ((cur & 1) << 2)  | ((cur & 1)<< 3) # 4th bit or 3rd bit 
			# ABCD => DDAA
			state ^= cur << j
			# ABCD ^ DDAA => (AD)(BD)(CA)(DA)

	return ret

#only block matrices allowed >:O
def prettyprint(matrix):
	size = matrix.shape[0]
	for i in range(size):
		for j in range(size):
			print("{} ".format(matrix[i,j]), end='')
		print()

# Make a numpy array storing the bits from an int
def intToBinVector(value):
	out = np.zeros([64],dtype=int)
	for i in range(64):
		out[63-i] = (value >> i) & 1

	return out

# And convert back again. len is used for the amount of bits our vect has
def binVectorToInt(vect,len=64):
	out = 0
	for i in range(len):
		out += int(vect[i] * 2** (len - 1 -i))

	return out

# this actually calculates something. Params are what the site feeds us with.
def simpleGauss(*args):
	result = []
	for arg in args:
		print("{:026b}".format(arg))
		result.extend(intToBinVector(arg)[-26:])


	equations=np.zeros([len(result) + len(initRelations), 65],dtype=int)
	cur = np.identity(64)
	for i in range(len(result)):
		lastrow = cur[63,:]
		lastrow = np.append(lastrow, result[i])
		equations[i,:] = lastrow
		cur = np.mod(np.matmul(cur, fullOp),2)

	equations[-len(initRelations):] = initRelations
	print("================================solving the System: ===================================")
	printGauss(equations)

	for row in range(64):
		#shift a row to row-index that has a 1 on the diagonal
		findNthRow(equations,row)

		#make all following rows 0 in that column
		innerIter(equations, row)
		#printGauss(equations)

	equations = eliminateNonDiagonalEntries(equations)
	print("================================Done: ==================================================")
	printGauss(equations)
	global state
	state = equations[:,64]
	state = binVectorToInt(state)
	print("================================Initial State is: ======================================")
	print("{:064b}".format(state)) 

# We should have a top right triagonal form at this point. Now its easy to get rid of all nondiag entries
def eliminateNonDiagonalEntries(equations):
	#we can strip all the nulls
	cols = 64
	equations = equations[:cols]

	for col in range(cols):
		current = equations[col]
		for row in range(col):
			if equations[row, col] == 1:
				equations[row] = xOr(equations[row],current)

	return equations

# Print our Gauss system for debugging or just for convenience. Color_col can be used to color a column.
def printGauss(equations, color_col=-1):
	for i in range(equations.shape[0]):
		for j in range(64):
			if j == color_col:
				if i ==color_col:
					print(bcolors.FAIL + "{} ".format(equations[i,j])+bcolors.ENDC, end='')
				else:
					print(bcolors.WARNING + "{} ".format(equations[i,j])+bcolors.ENDC, end='')
			else:
				print("{} ".format(equations[i,j]), end='')

		print("| {}".format(equations[i,64]))
	print()

# all indices nullbased. Elsiminates everything under the diagonal in the column "col"
def innerIter(equations, col):
	rows = equations.shape[0]
	current = equations[col, :]

	for row in range(col+1,rows):
		if(equations[row,col] == 1):
			equations[row,:] = xOr(equations[row,:], current)

	return equations

#same sized vectors only >:O
def xOr(vec1,vec2):
	out = []
	size = len(vec1)

	for i in range(size):
		out.append(vec1[i] ^ vec2[i])

	return out

#again n nullbased. Find the first row that has a 1 in the n'th column and make it the n'th row by swapping.
def findNthRow(equations, n):
	rows = equations.shape[0]
	for i in range(n,rows):
		if(equations[i,n] == 1):
			#swap the rows
			temp = equations[i,:].copy()
			equations[i,:] = equations[n,:]
			equations[n,:] = temp
			#print("swapped rows {} and {}.".format(i,n))
			return

	printGauss(equations,n)
	raise systemUnsolvable("In the {}th iteration. We need more input!!!".format(n))


#######Init the Matrices to simulate next and setup############################
op1 = np.zeros([64,64],dtype=int)				
op2 = np.zeros([64,64],dtype=int)

# op1 is our << 1 bitshift
# 0100...00
# 0010...00
# .
# .
# .
# 0000...01
# 0000...00

for i in range(63):
	op1[i,i+1]=1

# op2 is the >> 61 bitshift
# 0000...00
# .
# .
# .
# 1000...00
# 0100...00
# 0010...00

for i in range(61,64):
	op2[i,i-61]=1

first = np.mod(op1+op2,2) # the mod and addition commute
print("===========================the two bitshift operations: ===================")
prettyprint(first)
print()

second = np.zeros([64,64],dtype=int) 
submatrix=np.array([
	[1,0,0,1],
	[0,1,0,1],
	[1,0,1,0],
	[1,0,0,1]
	])
for i in range(0,64,4):
	second[i:i+4, i:i+4] = submatrix

print("===========================the inner iteration: ===========================")
prettyprint(second)
print()
fullOp = np.matmul(second,first) # the mod and multiplication commute
print("===========================a full iteration: ==============================")
prettyprint(second)
print()

#these are the inital relations state fulfills
initRelations = np.zeros([64, 65],dtype=int) # these relations are described in setup
block1 = [
	[0,0,0,0],
	[0,0,0,0],
	[1,0,1,0],
	[0,1,0,1]
]
rest = [
	[0,0,0,0,0,0],
	[0,0,0,0,0,0],
	[1,0,1,0,1,0],
	[0,1,0,1,0,1]
]

initRelations[0:4, 0:4] = block1

for i in range(4,64,4):
	initRelations[i:i+4, i-2:i+4] = rest

initRelations = initRelations[np.any(initRelations, axis =1)]

######################################

#crack the state
simpleGauss(24020036,7901952)
#YAY
print()
print("=======================Now i can see the future >:O ====================================")
print("%08i" %next(26))
print("%08i" %next(26))
print("%08i" %next(26))
print("%08i" %next(26))
print("%08i" %next(26))
print("%08i" %next(26))
print("%08i" %next(26))
print("%08i" %next(26))
print("%08i" %next(26))
print("%08i" %next(26))
print("%08i" %next(26))
print("%08i" %next(26))
print("%08i" %next(26))
print("%08i" %next(26))
print("%08i" %next(26))
print("%08i" %next(26))
print("%08i" %next(26))
print("%08i" %next(26))
print("%08i" %next(26))
print("%08i" %next(26))
print("%08i" %next(26))
print("%08i" %next(26))
print("%08i" %next(26))
print("%08i" %next(26))