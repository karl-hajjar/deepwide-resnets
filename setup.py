import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='deepwide-resnets',
    distname='',
    version='0.1.0',
    author='Karl Hajjar',
    author_email='karl.hajjar@polytechnique.edu',
    description='Library for experiments with deep and wide resnets',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/karl-hajjar/deepwide-resnets',
    license='MIT',
    packages=['.'],
    python_requires='3.9',
    install_requires=['Click',
                      'clickclick',
                      'pandas',
                      'numpy',
                      'pyyaml',
                      'scikit-learn',
                      'matplotlib',
                      'seaborn',
                      'torch',
                      'torchvision',
                      'pytorch-lightning==0.8.5',
                      'tensorboard==2.2.2',
                      'jax[cpu]'],
)
