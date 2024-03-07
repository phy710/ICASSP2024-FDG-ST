# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 11:54:45 2023

@author: pky0507
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import pywt
import shutil

def extract_amp_spectrum(trg_img):

    fft_trg_np = np.fft.fft2( trg_img, axes=(-2, -1) )
    amp_target, _ = np.abs(fft_trg_np), np.angle(fft_trg_np)

    return amp_target

def amp_spectrum_swap( amp_local, amp_target, L=0.1 , ratio=0, threshold_ratio = 0.05):
    
    a_local = np.fft.fftshift( amp_local, axes=(-2, -1) )
    a_trg = np.fft.fftshift( amp_target, axes=(-2, -1) )

    _, h, w = a_local.shape
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1
    
    threshold = np.max(np.max(a_trg, 1), 1)*threshold_ratio
    for i in range(len(amp_target)):
        a_trg[i] = pywt.threshold(a_trg[i], threshold[i])
    a_local[:,h1:h2,w1:w2] = a_local[:,h1:h2,w1:w2] * ratio + a_trg[:,h1:h2,w1:w2] * (1- ratio)
    #a_local[:,h1:h2,w1:w2] = a_local[:,h1:h2,w1:w2] * ratio + pywt.threshold(a_trg[:,h1:h2,w1:w2], T, mode='soft') * (1- ratio)
    a_local = np.fft.ifftshift( a_local, axes=(-2, -1) )
    return a_local

def freq_space_interpolation( local_img, amp_target, L=0 , ratio=0.0, threshold_ratio = 0.05):
    
    local_img_np = local_img 

    # get fft of local sample
    fft_local_np = np.fft.fft2( local_img_np, axes=(-2, -1) )

    # extract amplitude and phase of local sample
    amp_local, pha_local = np.abs(fft_local_np), np.angle(fft_local_np)

    # swap the amplitude part of local image with target amplitude spectrum
    amp_local_ = amp_spectrum_swap( amp_local, amp_target, L=L , ratio=ratio, threshold_ratio = threshold_ratio)

    # get transformed image via inverse fft
    fft_local_ = amp_local_ * np.exp( 1j * pha_local )
    local_in_trg = np.fft.ifft2( fft_local_, axes=(-2, -1) )
    local_in_trg = np.real(local_in_trg)

    return local_in_trg

def draw_image(image):
    plt.imshow(image, cmap='gray')
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)

    plt.xticks([])
    plt.yticks([])
    
    return 0

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

if __name__ == "__main__":
    
    np.random.seed(42)
    root = './Fundus'
    L = 1
    ratio_max = 1
    threshold_ratio = 0.00
    shutil.copytree(root, root+'-FDG')
    
    for a in range(4):
        for b in range(4):
            domain_source = 'Domain'+str(a+1)
            domain_target = 'Domain'+str(b+1) 
            domain_combine = 'Domain'+str(a+1)+str(b+1)
            lis_source = os.listdir(os.path.join(root, domain_source, 'train', 'ROIs', 'image'))
            lis_target = os.listdir(os.path.join(root, domain_target, 'train', 'ROIs', 'image'))
            shutil.copytree(os.path.join(root, domain_source), os.path.join(root+'-FDG', domain_combine))
            for i in range(len(lis_source)):
                im_source = Image.open(os.path.join(root, domain_source, 'train', 'ROIs', 'image', lis_source[i]))
                im_target = Image.open(os.path.join(root, domain_target, 'train', 'ROIs', 'image', lis_target[np.random.randint(len(lis_target))]))
               # im_source = im_source.resize( (256,256), Image.BICUBIC )
                im_source = np.asarray(im_source, np.float32)
                im_source = im_source.transpose((2, 0, 1))
              #  im_target = im_target.resize( (256,256), Image.BICUBIC )
                im_target = np.asarray(im_target, np.float32)
                im_target = im_target.transpose((2, 0, 1))
                amp_target = extract_amp_spectrum(im_target)
                local_in_trg = freq_space_interpolation(im_source, amp_target, L=L, ratio=1-ratio_max*np.random.rand(), threshold_ratio = threshold_ratio)
                local_in_trg = local_in_trg.transpose((1,2,0))
                
                img = Image.fromarray((np.clip(local_in_trg, 0, 255)).astype(np.uint8)).save(os.path.join(root+'-FDG', domain_combine, 'train', 'ROIs', 'image', lis_source[i]))
