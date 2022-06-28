"""
Build R_Ninv_R for MAP sky calculations.

R is shorthand for the matrix Fprime_Fz.  Used for MAP sky calculations.

"""

import multiprocessing, os
CPU_AVAIL = int(multiprocessing.cpu_count())
os.environ["MKL_NUM_THREADS"] = str(CPU_AVAIL-1)
import numpy as np

import argparse
import h5py
import time
from astropy import units
from scipy import sparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    '--array_dir',
    type=str,
    help='Path to directory containing Fprime_Fz ("R").'
)
parser.add_argument(
    '--sky_path',
    type=str,
    help='Path to h5py compatible skyh5 file.'
)
parser.add_argument(
    '--sky_rms_frac',
    type=float,
    default=0.1,
    help='Fraction of sky RMS to use as noise amplitude.'
)
parser.add_argument(
    '--clobber',
    action='store_true',
    help='If passed, overwrite matrix if it exists.'
)
args = parser.parse_args()


def build_R_Ninv_R(bm, array_dir, matrix_name, R, Ninv):
    """
    Build R_Ninv_R = Fprime_Fz^H * Ninv * Fprime_Fz

    """
    Ninv_R = Ninv * R
    R_Ninv_R = np.dot(R.conj().T, Ninv_R)
    bm.output_data(
        R_Ninv_R, str(array_dir), matrix_name, 'R_Ninv_R'
    )
    return R_Ninv_R


array_dir = Path(args.array_dir)
print(f'\nReading MAP dict from {array_dir}', end='\n\n')
map_dict = np.load(array_dir / 'map-dict.npy', allow_pickle=True)
if not isinstance(map_dict, dict):
    map_dict = map_dict.item()
bm = map_dict['bm']

print(f'Reading sky model from {args.sky_path}')
with h5py.File(args.sky_path, 'r') as f:
    stokes = f['Data']['stokes'][()][0] * units.K
stokes_vec = stokes.to('mK').value.flatten()
noise_amp = stokes_vec.std() * args.sky_rms_frac
Ninv = sparse.diags(np.ones(stokes_vec.size) / noise_amp**2)

print('Reading R (Fprime_Fz)...')
R = bm.read_data(str(array_dir / 'Fprime_Fz'), 'Fprime_Fz')

assert stokes_vec.size == R.shape[0], \
    f'Shape mismatch.  Sky model vector size {stokes_vec.size} does not agree'\
    f' with R.shape[0] {R.shape[0]}.'

matrix_name = f'R_Ninv_R_noise_{noise_amp:.3e}'
matrix_path = array_dir / (matrix_name + '.h5')
if not matrix_path.exists() or args.clobber:
    print('Computing R_Ninv_R...')
    start = time.time()
    R_Ninv_R = build_R_Ninv_R(bm, str(array_dir), 'R_Ninv_R', R, Ninv)
    stop = time.time()
    print(f'Completed in {stop - start} seconds.', end='\n\n')
    print(f'Matrix written to {matrix_path}')
else:
    print(f'Matrix already exists.  To overwrite use --clobber.')