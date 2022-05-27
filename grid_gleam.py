"""
Grid the gleam catalog onto a HEALPix grid.

"""

import numpy as np
import time
import argparse
from pathlib import Path
from pyradiosky import SkyModel
from astropy import units
import astropy_healpix as ahp
from BayesEoR.Utils import get_git_version_info

parser = argparse.ArgumentParser()
parser.add_argument(
    'data_path',
    type=str,
    help='Path to pyradiosky skyh5 GLEAM file.'
)
parser.add_argument(
    '--nside',
    type=int,
    default=256,
    help='HEALPix nside.'
)
parser.add_argument(
    '--file_str',
    type=str,
    default=None,
    help='If passed, insert `file_str` at the end of the output file name.'
)
parser.add_argument(
    '--out_dir',
    type=str,
    default=None,
    help='Output directory.  Defaults to directory of input sky model.'
)
parser.add_argument(
    '--clobber',
    action='store_true',
    default=False,
    dest='clobber',
    help='If passed, clobber existing file.  Defaults to False.'
)
args = parser.parse_args()

data_path = Path(args.data_path)
data_dir = data_path.parent
filename = data_path.name
nside = args.nside

print(f'\nReading in data from {data_path}', end='\n\n')
sm = SkyModel()
sm.read_skyh5(data_path)
sm.jansky_to_kelvin()
sm.stokes /= ahp.nside_to_pixel_area(nside)
nf = sm.Nfreqs

print('Gridding to HEALPix')
start = time.time()
npix = ahp.nside_to_npix(nside)
pix_inds = np.arange(npix)
stokes = np.zeros((4, nf, npix)) * sm.stokes.unit
for i_src, (ra, dec) in enumerate(zip(sm.ra, sm.dec)):
    hpx_inds, weights = ahp.bilinear_interpolation_weights(ra, dec, nside)
    for ind, w in zip(hpx_inds, weights):
        stokes[0, :, ind] += w * sm.stokes[0, :, i_src]
stokes = units.Quantity(stokes.value, unit='K')

sm_out = SkyModel(
    stokes=stokes, spectral_type='full', freq_array=sm.freq_array,
    component_type='healpix', nside=nside, hpx_inds=pix_inds
)
stop = time.time()
elapsed = (stop - start) * units.s
if elapsed.value > 60:
    elapsed = elapsed.to('min')
print(f'Completed in {elapsed}', end='\n\n')

git_info = get_git_version_info(directory=Path(__file__).parent)
git_str = 'Git version info: '
git_str += ', '.join(
    [f'{key}={git_info[key]}' for key in git_info]
)
sm_out.history = git_str + '\n'

outfile = filename.replace('gleam', f'gleam-nside{nside}')
outfile = outfile.strip('.skyh5')
if args.file_str is not None:
    outfile += f'-{args.file_str}'
outfile += '.skyh5'
if args.out_dir:
    outdir = Path(args.out_dir)
else:
    outdir = data_dir
print(f'\nWriting to {outdir / outfile}', end='\n\n')
sm_out.write_skyh5(outdir / outfile, clobber=args.clobber)
