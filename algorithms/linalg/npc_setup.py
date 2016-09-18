from distutils.extension import Extension
from distutils.core import setup
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import platform
import os
import numpy as np

BLAS_COMPLEX_INNER = 1 #There seems to be a bug in Canopy info for our complex BLAS call; set to 0 in this case before compile.
PARALLEL_TDOT = 1

system = platform.system()
extra_compile_args = ["-O3"]
extra_link_args = []
library_dirs = []
include_dirs = [np.get_include()]
libraries = []

if PARALLEL_TDOT: #really this should be based on compiler type but pain in ass
	if system == 'Darwin':
		print "OpenMP not standard on OS-X, turning off PARALLEL_TDOT"
		PARALLEL_TDOT=0
	if system == 'Windows':
		extra_compile_args.extend(['/openmp', '/Ox', '/fp:fast','/favor:INTEL64','/Og'])
	else:
		extra_compile_args.extend(["-v", "-fopenmp"])
		extra_link_args.extend(["-v", "-lgomp"])  # -gomp necessary for gcc 
		libraries.append('pthread')

CC = os.getenv('CC')
if CC == 'icc': #Intel compilers - have to link against this library
	libraries.append('irc')
	extra_compile_args.append("-xHost")
else:
	if platform.system() != 'Darwin':
		extra_compile_args.append("-march=native")
#Check for MKL or EPD by looking for these flags in environment variable
MKL_DIR = os.getenv('MKL_DIR')
if MKL_DIR==None:
	MKL_DIR = os.getenv('MKLROOT')
	if MKL_DIR == None:
		MKL_DIR = os.getenv('MKL_HOME')

EPD_DIR = os.getenv('EPD_DIR')
CANOPY_DIR = os.getenv('CANOPY_DIR')
ATLAS_DIR = os.getenv('ATLAS_DIR')
OPENBLAS_DIR = os.getenv('OPENBLAS_DIR')

if MKL_DIR is not None:
	print "Using MKL"
	with open('cblas.h', 'w') as file: #cblas is included in mkl.h, so make a little dummy file
		file.write('#include "mkl.h"')
	include_dirs.append(MKL_DIR + '/include')
	library_dirs.append(MKL_DIR +'/lib/intel64')
	#extra_compile_args.append("-DMKL_DIRECT_CALL")
	TKlibraries=['mkl_rt', 'pthread' , 'iomp5']

elif EPD_DIR is not None:
	print "Using EPD"
	include_dirs.append('/usr/include')
	library_dirs.append(EPD_DIR+'/lib')

	TKlibraries=['mkl_rt', 'pthread','m' , 'iomp5']

elif CANOPY_DIR is not None and MKL_DIR is not None:

	print "Using EPD-CANOPY"
	include_dirs.append('/usr/include')
	library_dirs.append(CANOPY_DIR+'/lib')

	print "And using MKL"
	with open('cblas.h', 'w') as file: #cblas is included in mkl.h, so make a little dummy file
		file.write('#include "mkl.h"')
	include_dirs.append(MKL_DIR + '/include')
	library_dirs.append(MKL_DIR +'/lib/intel64')

	extra_compile_args = ["-O3"]

	TKlibraries=['mkl_rt', 'pthread', 'iomp5']


elif CANOPY_DIR is not None:
	print "Using EPD-CANOPY"
	include_dirs.append('/usr/include')
	library_dirs.append(CANOPY_DIR+'/lib')
	TKlibraries=['mkl_rt', 'pthread','m', 'iomp5']


elif ATLAS_DIR is not None:
	print "Using ATLAS"
	include_dirs.append(ATLAS_DIR + '/include')
	library_dirs.append(ATLAS_DIR +'/lib')
	TKlibraries=['blas', 'cblas', 'atlas']

elif OPENBLAS_DIR is not None:
	print "Using OPENBLAS"
	include_dirs.append(OPENBLAS_DIR + '/include')
	library_dirs.append(OPENBLAS_DIR +'/lib')
	TKlibraries=['openblas']

else:
	include_dirs.append('/usr/include')
	include_dirs.append('/System/Library/Frameworks/vecLib.framework/Headers/') # < 10.9
	include_dirs.append('/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.9.sdk/System/Library/Frameworks/Accelerate.framework/Frameworks/vecLib.framework/Headers') #10.9
	library_dirs.append('/usr/lib')
	TKlibraries=['blas', 'cblas']
	

ext_modules = [	Extension("npc_helper", ["npc_helper.pyx"],
	extra_compile_args = extra_compile_args,
	extra_link_args = extra_link_args,
	include_dirs = include_dirs, libraries = libraries+TKlibraries, library_dirs = library_dirs,
	define_macros=[('CYTHON_TRACE', '0')],
	language='c++')
	]
"""
setup(
  name = 'npc_helper',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)
"""
setup(ext_modules = cythonize(ext_modules, compile_time_env={'PARALLEL_TDOT': PARALLEL_TDOT, 'BLAS_COMPLEX_INNER':BLAS_COMPLEX_INNER}))

