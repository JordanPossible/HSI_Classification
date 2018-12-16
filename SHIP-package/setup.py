#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 16:07:42 2018

@author: cxy
"""

"""
setup.py
Downloads and installs dependencies, datasets, and pretrained filters.
"""
import pip
import zipfile
#import urllib.request
import urllib2
import os


def main():

    #install scikit-learn
    pip_install('sklearn')
    
    #install numpy
    pip_install('numpy')

    #install scipy
    pip_install('scipy')
    
    #install pandas
    pip_install('pandas')

    #install matplotlib
    pip_install('matplotlib')
    
    #install spectral python
    pip_install('spectral')

    #install os python
    pip_install('os')
    
    #install shutil python
    pip_install('shutil')
    
    #install math python
    pip_install('math')
    
    #install time python
    pip_install('time')
    
    #install py_gco from https://github.com/amueller/gco_python
    pip_install('git+git://github.com/amueller/gco_python')

#    #install tensorflow-gpu
#    pip_install('tensorflow-gpu')
#
#    #install pydensecrf from https://github.com/lucasb-eyer/pydensecrf
#    pip_install('git+https://github.com/lucasb-eyer/pydensecrf.git')

#    #download and install datasets and pretrained filters    
#    pretrained_filters = [['http://www.cis.rit.edu/~rmk6217/scae.zip', './feature_extraction/pretrained'],
#                          ['http://www.cis.rit.edu/~rmk6217/smcae.zip', './feature_extraction/pretrained']]
#    datasets = [['http://www.cis.rit.edu/~rmk6217/datasets.zip', './']]
#    to_download = pretrained_filters + datasets
#    for i in range(len(to_download)):
#        print('Downloading ' + to_download[i][0] + ' .....')
#        download_and_unzip(to_download[i][0],to_download[i][1])


def pip_install(pkg):
    pip.main(['install', pkg])

def download_and_unzip(url, out_folder):
    tempfile_path = './tmp_dw_file.zip'
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    urllib.request.urlretrieve(url, tempfile_path)
    zip_ref = zipfile.ZipFile(tempfile_path, 'r')
    zip_ref.extractall(out_folder)
    zip_ref.close()
    os.remove(tempfile_path)    
    

if __name__ == '__main__':
    main()
