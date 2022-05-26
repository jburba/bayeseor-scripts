"""
Interpolate and/or downselect a pyradiosky SkyModel spectrally and spatially.

"""

import numpy as np
import argparse
import ast

from copy import deepcopy
from pathlib import Path
from pyradiosky import SkyModel
from astropy.units import Quantity
from BayesEoR.Linalg import Healpix
from BayesEoR.Utils import get_git_version_info

parser = argparse.ArgumentParser()
parser.add_argument(
    'data_path',
    type=str,
    help='Path(s) to pyradiosky skyh5 file(s).'
)
parser.add_argument(
    '--start_freq',
    type=float,
    help='Starting frequency in MHz.'
)
parser.add_argument(
    '--df',
    type=float,
    default=0.0,
    help='Frequency resolution in MHz.'
)
parser.add_argument(
    '--nf',
    type=int,
    default=1,
    help='Number of frequency channels.'
)
parser.add_argument(
    '--central_jd',
    type=float,
    help='Central Julian date.'
)
parser.add_argument(
    '--tele_latlonalt',
    type=str,
    default='(-30.72152777777791,21.428305555555557,1073.0000000093132)',
    help='Telescope (lat,lon,alt) passed with no spaces.  '
         'Defaults to location of HERA.'
)
parser.add_argument(
    '--fov_ra',
    type=float,
    default=None,
    help='FoV along the RA axis.  Defaults to all sky in RA.'
)
parser.add_argument(
    '--fov_dec',
    type=float,
    default=None,
    help='FoV along the DEC axis.  Defaults to the value of `fov_ra`.'
)
parser.add_argument(
    '--simple_za_filter',
    action='store_true',
    default=False,
    help='If passed, filter pixels in the sky model by zenith angle only. '
         'Otherwise, filter pixels in a rectangular region set by the FoV '
         'values along the RA and DEC axes (default).'
)
parser.add_argument(
    '--nside',
    type=int,
    default=256,
    help='HEALPix nside.  Defaults to the nside of the sky model if '
         'SkyModel.component_type == \'healpix\', otherwise 256.'
)
parser.add_argument(
    '--zero_mean',
    action='store_true',
    help='If passed, force sky model to be zero mean per frequency.'
)
parser.add_argument(
    '--prefactor',
    type=float,
    default=1.0,
    help='Factor used to scale the entire sky model stokes array.'
)
parser.add_argument(
    '--spec_ind',
    type=float,
    default=None,
    help='If passed, replace the spectrum with a power law relative to the '
         'first frequency in the sky model.'
)
parser.add_argument(
    '--spec_ind_std',
    type=float,
    default=None,
    help='If passed, draw the spectral indices from a Normal distribution '
         'with mean `spec_ind` and standard deviation `spec_ind_str`.  '
         'Otherwise, if `spec_ind` is passed use a single power law.'
)
parser.add_argument(
    '--np_seed',
    type=int,
    default=12345,
    help='Seed for numpy.random.  Defaults to 12345.'
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

print(f'Reading in data from {data_path}', end='\n\n')
sm = SkyModel()
sm.read_skyh5(data_path)
is_hpx = sm.component_type == 'healpix'

if args.fov_ra is not None:
    print('Performing spatial downselect')

    if is_hpx:
        nside = sm.nside
    else:
        nside = args.nside
    tele_latlonalt = ast.literal_eval(args.tele_latlonalt)
    hpx = Healpix(
        fov_ra_eor=args.fov_ra,
        fov_dec_eor=args.fov_dec,
        nside=nside,
        telescope_latlonalt=tele_latlonalt,
        central_jd=args.central_jd,
        simple_za_filter=args.simple_za_filter
    )
    if hpx.npix_fov != hpx.npix:
        if is_hpx:
            sm_sub = sm.select(component_inds=hpx.pix_eor, inplace=False)
        else:
            lonra, latra = hpx.get_extent_ra_dec(hpx.fov_ra, hpx.fov_dec)
            ras = sm.ra.deg.copy()
            decs = sm.dec.deg.copy()
            horizon_radius = 90.
            wrap_ra = hpx.field_center[0] - horizon_radius < 0
            if wrap_ra:
                ras[ras>180.] -= 360
            ras_mask = np.logical_and(ras >= lonra[0], ras <= lonra[1])
            decs_mask = np.logical_and(decs >= latra[0], decs <= latra[1])
            radec_mask = ras_mask * decs_mask
            sm_sub = sm.select(
                component_inds=np.where(radec_mask)[0], inplace=False
            )
        fov_str = '-fov'
        if hpx.fov_ra == hpx.fov_dec:
            fov_str += f'-{hpx.fov_ra:.1f}'
        else:
            fov_str += f'-ra-{hpx.fov_ra:.1f}-dec-{hpx.fov_dec:.1f}'
        if hpx.simple_za_filter:
            fov_str += '-za-filter'
    else:
        fov_str = ''
else:
    sm_sub = deepcopy(sm)
    fov_str = ''

if args.start_freq is not None:
    freqs = Quantity(
        args.start_freq + np.arange(args.nf) * args.df,
        unit='MHz'
    )
    freqs = freqs.to('Hz')

    freqs_in_sm = [freq in sm_sub.freq_array for freq in freqs]
    if np.any(np.logical_not(freqs_in_sm)):
        print('Interpolating in frequency')
        sm_sub.at_frequencies(freqs)
        freq_str = (
            f"-{freqs[0].to('MHz').value:.2f}-{freqs[-1].to('MHz').value:.2f}MHz"
            + f'-nf-{freqs.size}'
        )
else:
    freq_str = ''

if args.zero_mean:
    print('Forcing sky model to zero mean per frequency')
    sm_sub.stokes[0] -= sm_sub.stokes[0].mean(axis=1)[:, None]

is_point = sm_sub.component_type == 'point'
if is_point:
    sm_sub.jansky_to_kelvin()

nsrcs = sm_sub.Ncomponents
beta = args.spec_ind
beta_std = args.spec_ind_std
if beta == 0 and beta_std == 0:
    beta_std = None
if beta is not None:
    print('Fixing spectral structure to power law with ', end='')
    if beta_std is not None:
        print(f'mean={beta} and stddev={beta_std}')
        np.random.seed(args.np_seed)
        betas = np.random.normal(-beta, beta_std, nsrcs)
        beta_str = f'-pld-mean-{beta:.2f}-std-{beta_std:.2f}'
    else:
        print(f'spectral index {beta}')
        betas = -1 * np.ones(nsrcs) * beta
        if beta == 0:
            beta_str = '-flat'
        else:
            beta_str = f'-spl-{beta:.2f}'
    sm_sub.stokes[0] = (
        sm_sub.stokes[0, 0]
        * (freqs / freqs[0])[:, None]**(betas[None, :])
    )
else:
    beta_str = ''
if is_point:
    sm_sub.kelvin_to_jansky()

sm_sub.stokes *= args.prefactor

history_str = (
    '\n\nPreprocessed with preproc_sky_models.py and the '
    'following command line arguments: '
)
history_str += ', '.join(
    [f'{key}={args.__dict__[key]}' for key in args.__dict__]
)
history_str += '\n\n'
git_info = get_git_version_info(directory=Path(__file__).parent)
git_str = 'Git version info: '
git_str += ', '.join(
    [f'{key}={git_info[key]}' for key in git_info]
)
history_str += git_str
sm_sub.history += history_str

outfile = filename.strip('.skyh5')
outfile += freq_str
outfile += beta_str
outfile += fov_str
if args.file_str is not None:
    outfile += f'-{args.file_str}'
if args.prefactor != 1.0:
    outfile += f'-prefactor-{args.prefactor:.3e}'
outfile += '.skyh5'
if args.out_dir:
    outdir = Path(args.out_dir)
else:
    outdir = data_dir
print(f'\nWriting to {outdir / outfile}', end='\n\n')
sm_sub.write_skyh5(outdir / outfile, clobber=args.clobber)