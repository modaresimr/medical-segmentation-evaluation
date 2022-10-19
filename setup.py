from setuptools import setup,find_packages
setup(
    name='eval_seg'
    version='1.0',
    author='S.M.R. Modaresi',
    description='Multimodal Evaluation of Medical Image Segmentation',
    long_description='README.md',
    url='https://github.com/modaresimr/medical-segmentation-evaluation/',
    keywords='evaluation, medical image segmentation',
    python_requires='>=3.8, <4',
    packages=find_packages(include=['eval_seg']),
    install_requires=[
        'pandas==0.23.3',
        'numpy>=1.14.5',
        'matplotlib>=2.2.0',
        'nibabel',
        'tqdm',
        'ipywidgets',
        'diskcache',
        'compress_pickle',
        'opencv-python',
        'scipy',
        'edt',
        'skimage',
        'connected-components-3d',
        'sparse',
        'k3d',
        'ipython',
    ]
)