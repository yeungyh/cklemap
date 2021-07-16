import yaml, pprint, argparse, errno, sys
import numpy as np
import numpy.random as npr
from test.driver_ckli import drive
from time import time

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', help='Input file')
parser.add_argument('-o', '--output', help='Output file')
parser.add_argument('--NYxi', type=int, default=0)
parser.add_argument('-N', '--Nens', type=int, default=0)
parser.add_argument('--hp_from_data', help='Estimate hyperparameters from data', action='store_true')
parser.add_argument('-k', '--kernel', help='Kernel')
args = parser.parse_args()

if args.file is None:
    print('No input file specified. Try again.')
    sys.exit(errno.EAGAIN)

with open(args.file) as f:
    data = yaml.safe_load(f)

data_printer = pprint.PrettyPrinter()
data_printer.pprint(data)

seed  = data['seed']
Nruns = data['Nruns']
rs    = npr.RandomState(seed)

if args.NYxi > 0:
    data['NYxi'] = args.NYxi
    print('Overriden NYxi to {:d}'.format(data['NYxi']))

if args.Nens > 0:
    data['Nens'] = args.Nens
    print('Overriden Nens to {:d}'.format(data['Nens']))

if args.kernel is not None:
    data['kernel'] = args.kernel
    print('Overriden kernel to {}'.format(data['kernel']))

print('***')

Nc = 32 * 32
Yref = np.zeros((Nruns, Nc))
uref = np.zeros((Nruns, Nc))
Yest = np.zeros((Nruns, Nc))
uest = np.zeros((Nruns, Nc))
Yest_MAPH1 = np.zeros((Nruns, Nc))
iYobs = np.zeros((Nruns, data['NYobs']))
iuobs = np.zeros((Nruns, data['Nuobs']))
Yobs  = np.zeros((Nruns, data['NYobs']))
uobs  = np.zeros((Nruns, data['Nuobs']))
ckli_status = np.zeros(Nruns)
MAP_status  = np.zeros(Nruns)

for i in range(Nruns):
    
    print('Run {:d}'.format(i))
    timer  = time()
    output = drive(rs, data, args.hp_from_data)
    print("Elapsed time: {:g} s".format(time() - timer))

    Yref[i] = output.Yref
    uref[i] = output.uref
    Yest[i] = output.Yest
    uest[i] = output.uest
    Yest_MAPH1[i] = output.Yest_MAPH1
    iYobs[i] = output.iYobs
    iuobs[i] = output.iuobs
    Yobs[i] = output.Yobs
    uobs[i] = output.uobs
    ckli_status[i] = output.ckli_status
    MAP_status[i]  = output.MAP_status

if args.output is None:
    outfile = 'test/output/{:s}'.format(data['name'])
else:
    outfile = args.output
    
np.savez(outfile, Yref=Yref, uref=uref, Yest=Yest, uest=uest, Yest_MAPH1=Yest_MAPH1, iYobs=iYobs, iuobs=iuobs, Yobs=Yobs, uobs=uobs, ckli_status=ckli_status, MAP_status=MAP_status)
