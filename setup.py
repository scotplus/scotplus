from setuptools import setup, find_packages

setup(
    name='scotplus',
    version='1.0.0',    
    description='SCOT+ aligns unpaired single-cell multi-omic datasets via optimal transport.',
    long_description='SCOT+ is a compilation of optimal transport tools for application to multi-omic datasets, culminating in a formulation that takes into account local geometry and feature correspondence when aligning data. Visit our website for in-depth tutorials on what exactly this looks like.',
    url='https://github.com/scotplus/book_source',
    author='Colin Baker',
    author_email='colin_baker@brown.edu',
    license='',
    # packages=find_packages(exclude=["backend"]),
    install_requires=['numpy', 
                      'POT',
                      'torch',
                      'scanpy',
                      'scikit-learn',
                      'matplotlib',
                      'umap-learn',
                      'scipy',
                      'matplotlib',
                      'seaborn',
                      'pandas'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.9',
        'Natural Language :: English'
    ],
)