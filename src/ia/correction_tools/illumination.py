# import functions from packages
import os
import time 
import numpy as np
import matplotlib.pyplot as plt 
import multiprocessing as mp 
from scipy.stats import scoreatpercentile
from scipy.ndimage import gaussian_filter

# import local variables
from .. import _allowed_colors, _distance_zxy, _image_size, _correction_folder
from ..io_tools.load import correct_fov_image



def Generate_illumination_correction(data_folder, 
                                     sel_channels=None, 
                                     num_threads=12, parallel=True,
                                     num_images=48,
                                     single_im_size=_image_size, all_channels=_allowed_colors,
                                     num_buffer_frames=10, num_empty_frames=0, 
                                     correction_folder=_correction_folder,
                                     hot_pixel_corr=True, hot_pixel_th=4, z_shift_corr=True,
                                     remove_cap=True, cap_th_per=[5, 90],
                                     gaussian_filter_size=60, 
                                     save=True, overwrite=False, save_folder=None,
                                     save_prefix='illumination_correction_', 
                                     make_plot=True, verbose=True):
    """Function to generate illumination corrections for given channels
    Inputs:
        data_folder: the folder (one hybridization) contains images, str of path
        sel_channels: selected channels to generate illumination profiles, list or None (default: None->all channels in this round)
        num_threads: number of threads to process images in parallel, int (default: 12)
        num_images: number of images to be proecessed and averaged, int (default: 40)
    Outputs:
        """
    
    ## check inputs
    _total_start = time.time()
    if sel_channels is None:
        sel_channels = all_channels
    # check save folders
    if save_folder is None:
        save_folder = os.path.join(data_folder, 'Corrections')
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    _save_filenames = [os.path.join(save_folder, f"{save_prefix}{_ch}_{single_im_size[-2]}x{single_im_size[-1]}.npy")
                       for _ch in sel_channels]
    
    # directly load these channels
    _loaded_pfs = [np.load(_fl) for _fl in _save_filenames if os.path.isfile(_fl) and not overwrite]
    _loaded_channels = [_ch for _ch, _fl in zip(sel_channels, _save_filenames) if os.path.isfile(_fl) and not overwrite]
    if verbose:
        print(f"-- directly load:{_loaded_channels} illumination profiles for files")
    # get channels to be loaded
    _sel_channels = [_ch for _ch, _fl in zip(sel_channels, _save_filenames) if not (os.path.isfile(_fl) and not overwrite)]
    _sel_filenames = [_fl for _ch, _fl in zip(sel_channels, _save_filenames) if not (os.path.isfile(_fl) and not overwrite)]
    # start load images if any channels selected
    if len(_sel_channels) > 0:
        if verbose:
            print(f"-- start calculating {_sel_channels} illumination profiles")

        ## detect dax files
        _fovs = [_fl for _fl in os.listdir(data_folder) if _fl.split('.')[-1]=='dax']
        _fovs = [_f for _f in sorted(_fovs, key=lambda v:int(v.split('.dax')[0].split('_')[-1]) )]
        _num_load = min(num_images, len(_fovs))
        if verbose:
            print(f"-- {_num_load} among {len(_fovs)} dax files will be loaded in data_folder: {data_folder}")
        # get input daxfiles
        _input_fls = [os.path.join(data_folder, _fl) for _fl in _fovs[:_num_load]]
        # load images
        #_signal_sums = [np.zeros([single_im_size[-2], single_im_size[-1]]) for _c in _sel_channels]
        #_layer_cts = [np.zeros([single_im_size[-2], single_im_size[-1]]) for _c in _sel_channels]
        
        _illumination_args = [(_fl, _sel_channels, 
                      remove_cap, cap_th_per, 
                      gaussian_filter_size,
                      single_im_size, 
                      all_channels, 
                      num_buffer_frames, num_empty_frames,
                      hot_pixel_corr, hot_pixel_th, 
                      z_shift_corr, 
                      verbose) for _fl in _input_fls]
        if parallel:
            with mp.Pool(num_threads) as _illumination_pool:
                if verbose:
                    print(f"++ start multi-processing illumination profile calculateion with {num_threads} threads for {len(_illumination_args)} images", end=' ')
                    _multi_time = time.time()
                _pfs_per_fov = _illumination_pool.starmap(_image_to_profile, _illumination_args, chunksize=1)
                _illumination_pool.close()
                _illumination_pool.join()
                _illumination_pool.terminate()
                if verbose:
                    print(f"in {time.time()-_multi_time:.2f}s.")
        else:
            if verbose:
                _multi_time = time.time()
                print(f"++ start illumination profile calculateion for {len(_illumination_args)} images, start at: {time.localtime().tm_hour}:{time.localtime().tm_min}:{time.localtime().tm_sec}")
                
            _pfs_per_fov = [_image_to_profile(*_arg) for _arg in _illumination_args]
            if verbose:
                print(f"finish in {time.time()-_multi_time:.2f}s.")
            
        # summarize results
        _sel_pfs = []
        for _i, _ch in enumerate(_sel_channels):
            _pf = np.mean([_r[_i] for _r in _pfs_per_fov], axis=0)
            _pf = gaussian_filter(_pf, gaussian_filter_size) 
            _sel_pfs.append(_pf / np.max(_pf))


        # save
        if save:
            if verbose:
                print(f"-- saving updated profiles")
            for _ch, _pf, _fl in zip(_sel_channels, _sel_pfs, _sel_filenames):
                if verbose:
                    print(f"--- saving {_ch} profile into file: {_fl}")
                np.save(_fl.split('.npy')[0], _pf)
            
    # merge illumination profiles:
    _illumination_pfs = []
    for _ch in sel_channels:
        if _ch in _sel_channels:
            _illumination_pfs.append(_sel_pfs[_sel_channels.index(_ch)])
        elif _ch in _loaded_channels:
            _illumination_pfs.append(_loaded_pfs[_loaded_channels.index(_ch)])
        else:
            raise IndexError(f"channel: {_ch} doesn't exist in either _sel_channels or _loaded_channels!")
    
    if make_plot:
        for _ch, _pf, _fl in zip(sel_channels, _illumination_pfs, _save_filenames):
            plt.figure(dpi=150, figsize=(4,3))
            plt.imshow(_pf, )
            plt.colorbar()
            plt.title(f"illumination, channel:{_ch}")
            if save:
                plt.savefig(_fl.replace('.npy', '.png'), transparent=True)
            plt.show()
    if verbose:
        print(f"-- finish generating illumination profiles, time:{time.time()-_total_start:.2f}s")
    return _illumination_pfs

## function to be called by multiprocessing:
# function to process a image into median profiles in this fov
def _image_to_profile(filename, sel_channels, 
                      remove_cap=True, cap_th_per=[5, 90], 
                      gaussian_filter_size=40,
                      single_im_size=_image_size, 
                      all_channels=_allowed_colors, 
                      num_buffer_frames=10, num_empty_frames=0,
                      hot_pixel_corr=True, hot_pixel_th=4, 
                      z_shift_corr=False, 
                      verbose=True,
                      ):
    """Function to process image into mean profiles"""

    ## step 1: load images
    if verbose:
        print(f"-- load image: {os.path.join(os.path.basename(filename))} for illumination", end=' ')
        _start_time = time.time()

    _ims,_ = correct_fov_image(filename, sel_channels,
                              single_im_size=single_im_size, 
                              all_channels=all_channels,
                              num_buffer_frames=num_buffer_frames, 
                              num_empty_frames=num_empty_frames,
                              calculate_drift=False, 
                              corr_channels=sel_channels,
                              warp_image=False, 
                              hot_pixel_corr=hot_pixel_corr, 
                              hot_pixel_th=hot_pixel_th,
                              z_shift_corr=z_shift_corr,
                              illumination_corr=False, chromatic_corr=False, 
                              bleed_corr=False,
                              return_drift=False, verbose=True)
    if verbose:
        _load_time = time.time()
        print(f"in {_load_time-_start_time:.2f}s,", end=' ')
    ## step 2: calculate mean profile
    _pfs = []
    for _im, _ch in zip(_ims, sel_channels):
        _nim = np.array(_im, dtype=np.float32)
        # remove extreme values if specified
        if remove_cap:
            _limits = [scoreatpercentile(_im, min(cap_th_per)), 
                       scoreatpercentile(_im, max(cap_th_per))]
            _nim = np.clip(_im, min(_limits), max(_limits))
        _pfs.append(gaussian_filter(np.sum(_nim, axis=0), 
                                    gaussian_filter_size) )
    if verbose:
        print(f"into profile in {time.time()-_load_time:.2f}s.")

    return _pfs                                

def illumination_correction(im, corr_profile):
    """Apply illumination correction for 2d or 3d image with 2d profile (x-y)"""

    if len(np.shape(corr_profile)) != 2:
        raise IndexError(f"corr_profile for illumination should be 2d")
        
    if len(np.shape(im)) == 3:
        return np.clip(im.astype(np.float32) / corr_profile[np.newaxis,:], 
                       a_min=np.iinfo(im.dtype).min, 
                       a_max=np.iinfo(im.dtype).max).astype(im.dtype)
         
    elif len(np.shape(im)) == 2:
        return np.clip(im.astype(np.float32) / corr_profile, 
                       a_min=np.iinfo(im.dtype).min, 
                       a_max=np.iinfo(im.dtype).max).astype(im.dtype)
    else:
        raise IndexError(f"input image should be 2d or 3d.")
