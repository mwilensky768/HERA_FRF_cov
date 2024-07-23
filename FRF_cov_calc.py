import numpy as np
import matplotlib.pyplot as plt
from pyuvdata import UVData
from hera_filters.dspec import dpss_operator
from hera_cal.redcal import predict_noise_variance_from_autos
from hera_cal.io import HERAData
import glob
import os
from astropy import units
SDAY_SEC = units.sday.to("s")

import pickle
import re
from pathlib import Path
import sys
from itertools import combinations
from matplotlib.colors import LogNorm, SymLogNorm
from yaml import safe_load
import warnings
from scipy.stats import norm, kstest
from scipy.linalg import cholesky
import time
import warnings
from scipy.signal import blackmanharris

import hashlib

import argparse

def get_filt_params(antpair, spw, filter_param_dir):
    """
    Read in the dictionary for the associated spectral window and get the FRF parameters for a given antpair.
    
    Parameters:
        antpair (tuple of int): 
            Pair of antennas to get the params for.
        spw (int):
            Index of the spectral window in question.
    Returns:
        filter_center (float): 
            Center of the FRF in Hz (not mHz!)
        filter_half_width (float):
            Filter half width in Hz (not mHz!)
    """
    filt_param_filepath = f"{filter_param_dir}/spw_{spw}.yaml"
    with open(filt_param_filepath, "r") as filt_param_file:
        filt_param_dict = safe_load(filt_param_file)
    
    try:
        filter_center = filt_param_dict["filter_centers"][str(antpair)] * 1e-3 # File is in mHz, want it in Hz
    except KeyError:
        reversed_antpair = (antpair[1], antpair[0])
        try:
            # Have to hit this with a minus sign since the baseline is conjugated
            filter_center = -filt_param_dict["filter_centers"][str(reversed_antpair)] * 1e-3
        except KeyError:
            raise ValueError(f"Neither {antpair} nor {reversed_antpair} are keys in the filter params dictionary")
        finally:
            antpair = reversed_antpair
            
    filter_half_width = filt_param_dict["filter_half_widths"][str(antpair)] * 1e-3
    
    
    return filter_center, filter_half_width

def get_var_from_hd(auto_hd, vis_hd, antpairpol, times):
    """
    Get the variance for an antpairpol from a HERAData object.
    
    Parameters:
        auto_hd (HERAData):
            The autocorrelation waterfall made by the EXTRACT_AUTOS script.
        vis_hd (HERAData):
            The (frf'd) visiblity waterfall. Just used to get the nsamples array
            for the antpairpol in question without doing an additional cornerturn.
        antpairpol (tuple):
            (ant1, ant2, pol).
        times (array):
            The julian dates to get the variance for.
            
            
    Returns:
        var (array):
            Noise variance for antpairpol in question. Shape (Ntimes, Nfreqs, Npol).
    """
    
    antpairpols = [antpairpol, 
                   (antpairpol[0], antpairpol[0], antpairpol[2]),
                   (antpairpol[1], antpairpol[1], antpairpol[2])]
    _ , _, nsamples = vis_hd.read(bls=antpairpol, times=times)

    new_auto_hd = auto_hd.select(times=times, bls=antpairpols[1:], inplace=False)
    print(f"Any times in object: {np.any(np.isin(times,new_auto_hd.times))}")
    
    data, _, _ = new_auto_hd.build_datacontainers()

    dt = nsamples[antpairpol] * vis_hd.integration_time[0]
    df = vis_hd.channel_width
    var = predict_noise_variance_from_autos(antpairpol, data, dt=dt, df=df,
                                            nsamples=nsamples)
    
    return var

def get_frop(times, filter_cent_use, filter_half_wid_use, Nfreqs, t_avg=300.,
              cutoff=1e-9, weights=None):
    """
    Get FRF operator.
    
    Parameters:
        times (array):
            (Interleaved) Julian Dates on hd, converted to seconds.
        filter_cent_use (float):
            Filter center obtained from get_filt_params.
        filter_half_wid_use (float):
            Filter half width obtained from get_filt_params
        Nfreqs (int):
            Number of frequencies in the data.
        t_avg (float):
            Desired coherent averaging length, in seconds.
        cutoff (float):
            Eigenvalue cutoff for DPSS modes.
        weights (array):
            Array of weights to use. Should either be the negation of the flags 
            or None. None uses uniform weights.
    Returns:
        frop (array):
            Filter operator. Shape (Ntimes_coarse, Ntimes_fine, Nfreqs.)
    """

    # Time handling is a slightly modified port from hera_filters/hera_cal
    
    Ntimes = len(times)
    
    dmatr, evals = dpss_operator(times, np.array([filter_cent_use, ]), np.array([filter_half_wid_use, ]),
                                 eigenval_cutoff=np.array([cutoff, ]))
    Nmodes = dmatr.shape[-1]
    dtime = np.median(np.abs(np.diff(times)))
    chunk_size = int(np.round((t_avg / dtime)))

    
    Nchunk = int(np.ceil(Ntimes / chunk_size))
    chunk_remainder = Ntimes % chunk_size

    
    if weights is None: 
        weights = np.ones([Ntimes, hd.Nfreqs])
        
    #####Index Legend#####
    # a = DPSS mode      #
    # f = frequency      #
    # t = fine time      #
    # T = coarse time    #
    #####Index Legend#####
    
    ddagW = dmatr.T.conj()[:, np.newaxis] * weights.T # aft
    ddagWd = ddagW @ dmatr # afa
    lsq = np.linalg.solve(ddagWd.swapaxes(0,1), ddagW.swapaxes(0,1)) # fat
        
    if chunk_remainder > 0: # Stack some 0s that get 0 weight so we can do the reshaping below without incident
        
        dmatr_stack_shape = [chunk_size - chunk_remainder, Nmodes]
        weights_stack_shape = [chunk_size - chunk_remainder, Nfreqs]
        dmatr = np.vstack((dmatr, np.zeros(dmatr_stack_shape, dtype=complex)))
        weights = np.vstack((weights, np.zeros(weights_stack_shape, dtype=complex)))
    
    dres = dmatr.reshape(Nchunk, chunk_size, Nmodes)
    wres = weights.reshape(Nchunk, chunk_size, Nfreqs)
    wnorm = wres.sum(axis=1)[:, np.newaxis]

    # normalize for an average
    wres = np.where(wnorm > 0, wres / wnorm, 0)
    # does "Ttf,Tta->Tfa" much faster than einsum and fancy indexing
    dchunk = np.zeros([Nchunk, Nfreqs, Nmodes], dtype=complex)
    for coarse_time_ind in range(Nchunk):
        dchunk[coarse_time_ind] = np.tensordot(wres[coarse_time_ind].T, 
                                               dres[coarse_time_ind],
                                               axes=1)
        
    # does "Tfa,fat->Ttf" faster than einsum and fancy indexing
    frop = np.zeros([Nchunk, Ntimes, Nfreqs], dtype=complex)
    for freq_ind in range(Nfreqs):
        frop[:, :, freq_ind] = np.tensordot(dchunk[:, freq_ind],
                                            lsq[freq_ind],
                                            axes=1)
        
    return frop

def get_frop_hash(filter_cent_use, filter_half_wid_use, weights):
    """
    Make a tuple out of the unique parameters that define the FRF covariance 
    operator. The weights are hashed since they are large arrays, everything 
    else is just taken as is (Technically should include t_avg but this
    will be fixed in the HERA pipeline).
    
    Parameters:
        filter_cent_use (float):
            Filter center obtained from get_filt_params.
        filter_half_wid_use (float):
            Filter half width obtained from get_filt_params
        weights (array):
            Array of weights to use. Should be the negation of the flags or uniform.
    """
    w_hash = hashlib.sha1(weights).hexdigest()
    
    return (filter_cent_use, filter_half_wid_use, w_hash)

def get_covs_antpair(auto_hd, vis_hd, spw,  antpairpol, filter_param_dir, 
                     t_avg=300., get_weights_from_flags=True, default_var=1., 
                     Ninterleave=4, cutoff=1e-9, frop_cache=None):
    """
    Get coherently averaged covariance
    
    Parameters:
        auto_hd (HERAData):
            The autocorrelation waterfall made by the EXTRACT_AUTOS script added 
            to the HERAData object with the antpairpol in question.
        vis_hd (HERAData):
            The (frf'd) visiblity waterfall. Just used to get the nsamples array
            for the antpairpol in question.
        spw (int):
            Spectral window index (0-10 I think)
        antpairpol (tuple):
            (ant1, ant2, pol) to get covariance for
        t_avg (float):
            Coherent averaging window in seconds
        get_weights_from_flags (array):
            If True the negation of the flags, otherwise use uniform weights.
        default_var (float):
            Default variance to use when nsamples=0. These samples should get 0 
            weight and so this number should not matter in theory. Needs to
            exist so that the variance doesn't have nans.
        Ninterleave (int):
            Number of interleaved time streams.
        cutoff (float):
            Eigenvalue cutoff for DPSS filter.
        frop_cache (dict):
            Cache of filters that have already been calculated.
            
    Returns:
        covs (array): 
            The (interleaved) covariances for the antpairpol in question.
        frop_cache (dict):
            Cache of filters that have already been calculated.
    """
    if frop_cache is None:
        frop_cache = {}
    filter_cent_use, filter_half_wid_use = get_filt_params(antpairpol[:2], spw,
                                                           filter_param_dir)

    times = vis_hd.times
    covs = []
    for int_ind in range(Ninterleave):
        int_times = times[int_ind::Ninterleave]
        
        if get_weights_from_flags:
            _, flags, _ = vis_hd.read(times=int_times, bls=antpairpol)
            weights = np.logical_not(flags[antpairpol]).astype(int)
            all_flagged = np.all(flags[antpairpol], axis=0)
            if np.any(all_flagged):
                warnings.warn(f"Some channels are entirely flagged. Giving them uniform weighting.")
                weights[:, all_flagged] = 1
        else:
            weights = None
        
        times_s = (int_times - int_times[0]) * SDAY_SEC # Put in seconds
        
        fropkey = get_frop_hash(filter_cent_use, filter_half_wid_use, weights)
        if fropkey in frop_cache:
            print("I already have this frop.")
            frop = frop_cache[fropkey]
        else:
            frop = get_frop(times_s, filter_cent_use, filter_half_wid_use, vis_hd.Nfreqs, 
                          cutoff=cutoff, t_avg=t_avg, weights=weights) # indices Tt or Ttf
            frop_cache[fropkey] = frop
        if frop.ndim == 2:
            frop = frop[:, :, np.newaxis]

        # get noise variance
        varis = get_var_from_hd(auto_hd, vis_hd, antpairpol, int_times) # indices tf

        if np.any(np.isnan(varis)):
            warnings.warn(f"Replacing nans with default variance of {default_var}")
            varis[np.isnan(varis)] = default_var

        # Form NF^\dag
        
        # Form FNF^\dag
        

        fvaris_left = varis * frop # Ttf
        Nfreqs = fvaris_left.shape[-1]
        Nchunk = fvaris_left.shape[0]
        cov = np.zeros([Nchunk, Nchunk, Nfreqs], dtype=complex)
        for freq_ind in range(Nfreqs):
            # for some reason tdot is OOM faster than matmul...
            cov[:, :, freq_ind] = np.tensordot(fvaris_left[:, :, freq_ind],
                                               frop[:, :, freq_ind].T.conj(),
                                               axes=1)
        # Keeping this einsum line for now since I know it's correct...
        # cov = np.einsum("ijk,jk,ljk->ilk", frop, varis, frop.conj(), optimize=True) # TTf

        covs.append(cov)

    return np.array(covs), frop_cache

def cov_wrapper_per_waterfall(auto_hd, waterfall_file, spw, filter_param_dir, cutoff=1e-9, 
                              t_avg=300., get_weights_from_flags=True, 
                              default_var=1, Ninterleave=4, frop_cache=None, 
                              profile=False):
    """
    Wrapper to calculate the FRF'd covariances for all antpairpols in a given waterfall file.
    
        auto_hd (HERAData):
            The autocorrelation waterfall made by the EXTRACT_AUTOS script added
            to the HERAData object with
            the antpairpol in question.
        waterfall_file (str):
            Path to a waterfall file for which we just need the antpairpols and nsamples. Only partial I/O is
            used within the internal functions.
        spw (int):
            Spectral window index (0-10 I think)
        cutoff (float):
            Eigenvalue cutoff for DPSS filter.
        t_avg (float):
            Coherent averaging window in seconds
        get_weights_from_flags (array):
            If True the negation of the flags, otherwise use uniform weights.
        default_var (float):
            Default variance to use when nsamples=0. These samples should get 0 weight and so this number should
            not matter in theory. Needs to exist so that the variance doesn't have nans.
        Ninterleave (int):
            Number of interleaved time streams.
        frop_cache (dict):
            Cache of filters that have already been calculated.
        profile (bool):
            If true, indicates this is a profiling run and breaks the loop after one antpairpol.
    """
    
    print("Setting up HERAData object.")
    t_begin_read = time.time()
    wf_hd = HERAData(waterfall_file)
    t_finish_read = time.time()
    t_to_read = t_finish_read - t_begin_read
    print(f"Setup took {t_to_read} s")
    print(f"wf_hd has {wf_hd.Nbls} baselines")
    print(f"wf_hd has {wf_hd.Npols} pols")
    
    print("Calculating covariances.")
    t_begin_calc = time.time()
    covs = []
    for antpairpol in wf_hd.bls:
        titer = time.time()
        covs_antpair, frop_cache = get_covs_antpair(auto_hd, wf_hd, spw, 
                                                    antpairpol, filter_param_dir, t_avg=t_avg, 
                                                    get_weights_from_flags=get_weights_from_flags, 
                                                    default_var=default_var, cutoff=cutoff, frop_cache=frop_cache,
                                                    Ninterleave=Ninterleave)
        covs.append(covs_antpair)
        titer_end = time.time()
        print(f"This iteration took {titer_end - titer} s")
        if profile:
            break
    t_end_calc = time.time()
    t_to_calc = t_end_calc - t_begin_calc
    print(f"Calc took {t_to_calc}")
    
    
    return np.array(covs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, required=True,
                        help="Path to the directory to which the output is written.")
    parser.add_argument("--filter-param-dir", type=str, required=True,
                        dest="filter_param_dir",
                        help="The path to a yaml file containing the FRF parameters, keyed by antpair.")
    parser.add_argument("--spw", required=True, type=int,
                        help="The spectral window being analyzed.")
    parser.add_argument("--auto-file", required=True, type=str, dest="auto_file",
                        help="Path to extracted autocorrelations in waterfall format.")
    parser.add_argument("--waterfall-files", nargs="*", type=str, required=True,
                        dest="waterfall_files",
                        help="(filtered) Waterfall files from which to grab nsamples")
    parser.add_argument("--cutoff", required=False, default=1e-9,
                        help="Eigenvalue cutoff for DPSS filter.")
    parser.add_argument("--ninterleave", required=False, type=int, default=4,
                        help="Number of interleaved streams for filtering.")
    parser.add_argument("--fn-out", required=True, type=str, dest="fn_out",
                        help="Where to write out the files.")
    args = parser.parse_args()

    auto_hd = HERAData(args.auto_file)
    # FIXME: These need to be changed for re-run
    spw_chans = [(0, 95), (180, 265), (265,365), (365,417), (417,497), (497, 577), (577, 657), (657, 737)]
    chan_low, chan_high = spw_chans[args.spw]
    auto_hd.read(freq_chans=np.arange(chan_low,chan_high))
    # FIXME: The do script was cribbed from something that maps many inputs to one output
    # But this needs to map 1-to-1...
    covs_total = []
    for wf_file in args.waterfall_files:
        covs = cov_wrapper_per_waterfall(auto_hd, wf_file, args.spw, 
                                         args.filter_param_dir,
                                         cutoff=args.cutoff,
                                         Ninterleave=args.ninterleave)
        
        covs_total.append(covs)
    np.save(args.fn_out, covs_total)

        

