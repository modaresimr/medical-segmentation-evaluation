from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='eval_seg',
    version='1.0',
    author='S.M.R. Modaresi',
    description='Multimodal Evaluation of Medical Image Segmentation',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/modaresimr/medical-segmentation-evaluation/',
    keywords='evaluation, medical image segmentation',
    python_requires='>=3.7, <4',
    package_dir={"": "eval_seg"},
    packages=find_packages(),
    #   namespace_packages=[],
    zip_safe=False,
    install_requires=[
        'pandas>=0.23.3',
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
        'scikit-image',
        'connected-components-3d',
        'sparse',
        'k3d',
        'ipython',
    ])
