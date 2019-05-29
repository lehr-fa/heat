import heat as ht
import numpy as np
from collections import deque
from mpi4py import MPI

Nx = 4
Ny = 4


NORTH = 0
SOUTH = 1
EAST = 2
WEST = 3


comm = MPI.COMM_WORLD
mpi_rows = 2 
mpi_cols= 2

mxloc = Nx/mpi_rows
myloc = Ny/mpi_cols

#def ind2cord(ind, ):

def cord2ind(x,y,mpi_rows, mpi_cols):
    return y + x*mpi_cols

def shiftleft(y, mpi_cols):
    if y > 0:
        return y - 1
    else:
        return mpi_cols - 1        



#print("Creating a %d x %d processor grid..." % (mpi_rows, mpi_cols) )

ccomm = comm.Create_cart( (mpi_rows, mpi_cols), periods=(True, True), reorder=True)

mpi_rows = int(np.floor(np.sqrt(comm.size)))
mpi_cols = comm.size // mpi_rows
if mpi_rows*mpi_cols > comm.size:
    mpi_cols -= 1
if mpi_rows*mpi_cols > comm.size:
    mpi_rows -= 1

neigh = [0,0,0,0]
    
neigh[NORTH], neigh[SOUTH] = ccomm.Shift(0, 1)
neigh[EAST], neigh[WEST] = ccomm.Shift(1, 1)

Nypi_row, Nypi_col = ccomm.Get_coords(ccomm.rank) 

#print(comm.rank, (Nypi_row, Nypi_col),shiftleft(Nypi_col, mpi_cols), cord2ind(Nypi_row, shiftleft(Nypi_col, mpi_cols),mpi_rows, mpi_cols))

#my_A = np.random.normal(size=(Nx, Ny)).astype(np.float32)

my_A = np.array([[1,2,3,4], [5,6,7,8],[9,10,11,12],[13,14,15,16]]).astype(np.float32)
my_B = np.array([[1,2,3,4], [5,6,7,8],[9,10,11,12],[13,14,15,16]]).astype(np.float32)

#my_B = np.random.normal(size=(Nx, Ny)).astype(np.float32)

#tmpA = np.random.normal(size=(Nx/2, Ny/2)).astype(np.float32)
#tmpB = np.random.normal(size=(Nx/2, Ny/2)).astype(np.float32)


lsx = int(Nypi_row * mxloc)
lex = int(Nypi_row * mxloc + mxloc)

lsy = int(Nypi_col * myloc)
ley = int(Nypi_col * myloc + myloc)


#print(comm.rank, lsx, lex, lsy, ley)

print(comm.rank, my_A[lsx:lex,lsy:ley])
#if comm.rank==0:
#    print(my_A)

#print(ccomm.Get_size())

#print(dir(ccomm))



"""
Nypi_row, Nypi_col = ccomm.Get_coords(ccomm.rank) 

#print(comm.rank, Nypi_row, Nypi_col)

neigh[NORTH], neigh[SOUTH] = ccomm.Shift(0, 1)
neigh[EAST], neigh[WEST] = ccomm.Shift(1, 1)

print(comm.rank, neigh[NORTH], neigh[SOUTH], neigh[EAST], neigh[WEST])

# Create matrices
my_A = np.random.normal(size=(Nx, Ny)).astype(np.float32)
my_B = np.random.normal(size=(Nx, Ny)).astype(np.float32)
my_C = np.zeros_like(my_A)

tile_A = my_A
tile_B = my_B
tile_A_ = np.empty_like(my_A)
tile_B_ = np.empty_like(my_A)
req = [None, None, None, None]


aa = ht.random.randn(4,4)
"""
