import os
import subprocess
import argparse
import six
import fnmatch
import numpy as np
from pyuvdata import UVBeam
import glob
import yaml

a = argparse.ArgumentParser(description="A command-line script to convert Nicolas "
                            "Fagnoni's CST simulations to UVBeam FITS files.")
a.add_argument('settings_file', type=str, help='settings yaml file name '
               '(should be in the same directory as the CST .txt files)')
a.add_argument('--efield', help='Efield rather than power beams',
               action='store_true', default=False)
a.add_argument('--calc_cross_pols', help='Calculate cross pol power beams '
               '(i.e. xy, yx). Only applies if efield is not True.',
               action='store_true', default=False)
a.add_argument('--peak_normalize', help='Peak normalize the beam.',
               action='store_true', default=False)
a.add_argument('--no_healpix', help='Convert to HEALPix',
               action='store_true', default=False)
a.add_argument("--hp_nside", default=None, type=int, help="If converting to HEALpix, use"
               "this NSIDE. Default is closest, yet higher, resolution to input resolution.")
a.add_argument("--interp_func", type=str, default="az_za_simple", help="If converting to HEALpix, "
               "use this interpolation function in pyuvdata.UVBeam. Only az_za_simple supported currently.")
a.add_argument('--outfile', type=str, help='Output file name', default=None)
a.add_argument('-f', '--freq_range', nargs=2, type=float,
               help='Frequency range to include in MHz')

args = a.parse_args()

if not args.settings_file.endswith('yaml'):
    raise ValueError('settings file should be a yaml file.')

# try to get git info (only works for old data in simulations repo)
try:
    data_dir = os.path.dirname(args.settings_file)
    git_origin = subprocess.check_output(['git', '-C', data_dir, 'config',
                                          '--get', 'remote.origin.url'],
                                         stderr=subprocess.STDOUT).strip()
    git_hash = subprocess.check_output(['git', '-C', data_dir, 'rev-parse', 'HEAD'],
                                       stderr=subprocess.STDOUT).strip()
    git_branch = subprocess.check_output(['git', '-C', data_dir, 'rev-parse',
                                          '--abbrev-ref', 'HEAD'],
                                         stderr=subprocess.STDOUT).strip()

    if six.PY3:
        git_origin = git_origin.decode('utf8')
        git_hash = git_hash.decode('utf8')
        git_branch = git_branch.decode('utf8')

    version_str = ('  Git origin: ' + git_origin
                   + '.  Git branch: ' + git_branch
                   + '.  Git hash: ' + git_hash + '.')
except subprocess.CalledProcessError:
    version_str = ''

beam = UVBeam()

if args.freq_range is not None:
    with open(args.settings_file, 'r') as file:
        settings_dict = yaml.safe_load(file)

    frequencies = np.array(settings_dict['frequencies'], dtype=float) / 1e6
    frequency_select = np.where((frequencies >= args.freq_range[0])
                                & (frequencies <= args.freq_range[1]))[0].tolist()
    if len(frequency_select) == 0:
        raise ValueError('No frequencies in freq_range')
    frequency_select = frequencies[frequency_select] * 1e6
else:
    frequency_select = None

if args.efield or args.calc_cross_pols:
    read_beam_type = 'efield'
else:
    read_beam_type = 'power'


beam.read_cst_beam(args.settings_file, beam_type=read_beam_type,
                   frequency_select=frequency_select)

default_out_file = 'NF_HERA' + '_' + beam.feed_name

if args.efield:
    default_out_file += '_efield'
else:
    default_out_file += '_power'

default_out_file += '_beam'

if not args.efield and args.calc_cross_pols:
    beam.efield_to_power()

beam.history = 'CST simulations by Nicolas Fagnoni.' + version_str

if not args.no_healpix:
    beam.interpolation_function = args.interp_func
    beam.to_healpix(nside=args.hp_nside)
    default_out_file += '_healpix'

if args.peak_normalize:
    beam.peak_normalize()

default_out_file += '.fits'
if args.outfile is not None:
    outfile = args.outfile
else:
    outfile = default_out_file

beam.write_beamfits(outfile, clobber=True)
