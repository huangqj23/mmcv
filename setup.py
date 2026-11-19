import glob
import os
import platform
import re
from importlib.metadata import PackageNotFoundError, distribution

from setuptools import find_packages, setup

try:
    import torch
    from torch.utils.cpp_extension import BuildExtension
    cmd_class = {'build_ext': BuildExtension}
except ModuleNotFoundError:
    cmd_class = {}
    print('Skip building ext ops due to the absence of torch.')


def choose_requirement(primary, secondary):
    """If some version of primary requirement installed, return primary, else
    return secondary."""
    try:
        name = re.split(r'[!<>=]', primary)[0]
        distribution(name)
    except PackageNotFoundError:
        return secondary

    return str(primary)


def get_version():
    version_file = 'mmcv/version.py'
    with open(version_file, encoding='utf-8') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


def parse_requirements(fname='requirements/runtime.txt', with_version=True):
    """Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs

    Returns:
        List[str]: list of requirements items

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import sys
    from os.path import exists
    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip,
                                                     rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # NOQA
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath) as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    yield from parse_line(line)

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


install_requires = parse_requirements()

try:
    # OpenCV installed via conda.
    import cv2  # NOQA: F401
    major, minor, *rest = cv2.__version__.split('.')
    if int(major) < 3:
        raise RuntimeError(
            f'OpenCV >=3 is required but {cv2.__version__} is installed')
except ImportError:
    # If first not installed install second package
    CHOOSE_INSTALL_REQUIRES = [('opencv-python-headless>=3',
                                'opencv-python>=3')]
    for main, secondary in CHOOSE_INSTALL_REQUIRES:
        install_requires.append(choose_requirement(main, secondary))


def get_extensions():
    extensions = []

    if os.getenv('MMCV_WITH_OPS', '1') == '0':
        return extensions

    ext_name = 'mmcv._ext'
    from torch.utils.cpp_extension import CppExtension, CUDAExtension

    # prevent ninja from using too many resources
    try:
        import psutil
        num_cpu = len(psutil.Process().cpu_affinity())
        cpu_use = max(4, num_cpu - 1)
    except (ModuleNotFoundError, AttributeError):
        cpu_use = 4

    os.environ.setdefault('MAX_JOBS', str(cpu_use))
    define_macros = []
    if PY_LIMITED_API:
        # the ops reach Python through torch.ops, and mmcv._ext only uses the
        # limited API (csrc/pytorch/module.cpp): one wheel per (torch, CUDA)
        # covers every Python version from 3.10 on
        define_macros += [('Py_LIMITED_API', '0x030A0000')]
    if platform.system() != 'Windows':
        extra_compile_args = {'cxx': ['-std=c++17']}
    else:
        extra_compile_args = {'cxx': ['/std:c++17']}
    op_files = glob.glob('./mmcv/ops/csrc/pytorch/*.cpp') + \
        glob.glob('./mmcv/ops/csrc/pytorch/cpu/*.cpp')

    # NVIDIA GPUs and CPU only. FORCE_CUDA=1 compiles the CUDA kernels on a
    # machine without a GPU (CI, image builds).
    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('MMCV_WITH_CUDA', None)]
        cuda_args = os.getenv('MMCV_CUDA_ARGS')
        extra_compile_args['nvcc'] = [cuda_args] if cuda_args else []
        # on Windows the compiler picks the standard by itself
        if platform.system() != 'Windows':
            extra_compile_args['nvcc'] += ['-std=c++17']
        op_files += glob.glob('./mmcv/ops/csrc/pytorch/cuda/*.cu') + \
            glob.glob('./mmcv/ops/csrc/pytorch/cuda/*.cpp')
        extension = CUDAExtension
        include_dirs = [
            os.path.abspath('./mmcv/ops/csrc/pytorch'),
            os.path.abspath('./mmcv/ops/csrc/common'),
            os.path.abspath('./mmcv/ops/csrc/common/cuda')
        ]
    else:
        print(f'Compiling {ext_name} only with CPU')
        extension = CppExtension
        include_dirs = [os.path.abspath('./mmcv/ops/csrc/common')]

    ext_ops = extension(
        name=ext_name,
        sources=op_files,
        include_dirs=include_dirs,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        py_limited_api=PY_LIMITED_API)
    extensions.append(ext_ops)
    return extensions


# MMCV_PY_LIMITED_API=0 builds a regular, Python-version specific extension
PY_LIMITED_API = os.getenv('MMCV_PY_LIMITED_API', '1') == '1'


setup(
    name='mmcv' if os.getenv('MMCV_WITH_OPS', '1') == '1' else 'mmcv-lite',
    version=get_version(),
    description='OpenMMLab Computer Vision Foundation',
    keywords='computer vision',
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Utilities',
    ],
    url='https://github.com/open-mmlab/mmcv',
    author='MMCV Contributors',
    author_email='openmmlab@gmail.com',
    install_requires=install_requires,
    extras_require={
        'all': parse_requirements('requirements.txt'),
        'tests': parse_requirements('requirements/test.txt'),
        'build': parse_requirements('requirements/build.txt'),
        'optional': parse_requirements('requirements/optional.txt'),
    },
    python_requires='>=3.10',
    ext_modules=get_extensions(),
    cmdclass=cmd_class,
    options={'bdist_wheel': {
        'py_limited_api': 'cp310'
    }} if PY_LIMITED_API else {},
    zip_safe=False)
