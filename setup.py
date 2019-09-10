import os
import platform
import subprocess
import time
from setuptools import Extension, dist, find_packages, setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

dist.Distribution().fetch_build_eggs(['Cython', 'numpy>=1.11.1'])
import numpy as np  # noqa: E402
from Cython.Build import cythonize  # noqa: E402


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


MAJOR = 1
MINOR = 0
PATCH = ''
SUFFIX = 'rc0'
SHORT_VERSION = '{}.{}.{}{}'.format(MAJOR, MINOR, PATCH, SUFFIX)

def get_git_hash():

    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH', 'HOME']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        sha = out.strip().decode('ascii')
    except OSError:
        sha = 'unknown'

    return sha


def get_hash():
    if os.path.exists('.git'):
        sha = get_git_hash()[:7]
    else:
        sha = 'unknown'

    return sha



def make_cuda_ext(name, module, sources):

    return CUDAExtension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        extra_compile_args={
            'cxx': [],
            'nvcc': [
                '-D__CUDA_NO_HALF_OPERATORS__',
                '-D__CUDA_NO_HALF_CONVERSIONS__',
                '-D__CUDA_NO_HALF2_OPERATORS__',
            ]
        })


def make_cython_ext(name, module, sources):
    extra_compile_args = None
    if platform.system() != 'Windows':
        extra_compile_args = {
            'cxx': ['-Wno-unused-function', '-Wno-write-strings']
        }

    extension = Extension(
        '{}.{}'.format(module, name),
        [os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=[np.get_include()],
        language='c++',
        extra_compile_args=extra_compile_args)
    extension, = cythonize(extension)
    return extension


def get_requirements(filename='requirements.txt'):
    here = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(here, filename), 'r') as f:
        requires = [line.replace('\n', '') for line in f.readlines()]
    return requires


if __name__ == '__main__':
    setup(
	name='utils',	
        license='Apache License 2.0',
        setup_requires=['pytest-runner', 'cython', 'numpy'],
        tests_require=['pytest'],
	install_requires=get_requirements(),
        ext_modules=[
            make_cuda_ext(
		name='nms_cpu',
                module='detection/model/utils/nms/',
                sources=['src/nms_cpu.cpp']),
            make_cuda_ext(
                name='nms_cuda',
                module='detection/model/utils/nms/',
                sources=['src/nms_cuda.cpp', 'src/nms_kernel.cu']),
            make_cuda_ext(
                name='deform_conv_cuda',
                module='detection/model/utils/dcn/',
		sources=[
                    'src/deform_conv_cuda.cpp',
                    'src/deform_conv_cuda_kernel.cu']),
            make_cuda_ext(
                name='roi_align_cuda',
                module='detection/model/utils/roi_align',
		sources=['src/roi_align_cuda.cpp', 'src/roi_align_kernel.cu']),
        ],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False)

