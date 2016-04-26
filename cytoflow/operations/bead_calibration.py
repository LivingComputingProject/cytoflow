#!/usr/bin/env python2.7

# (c) Massachusetts Institute of Technology 2015-2016
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''
Created on Aug 31, 2015

@author: brian
'''

from __future__ import division, absolute_import

from traits.api import (HasStrictTraits, Str, CStr, File, Dict, Python,
                        Instance, Int, List, Float, Constant, Array, provides)
import numpy as np
import math
import warnings
import scipy.signal
import scipy.optimize
        
import matplotlib.pyplot as plt

import cytoflow.views
import cytoflow.utility as util

from .i_operation import IOperation
from .import_op import check_tube, Tube, ImportOp

@provides(IOperation)
class BeadCalibrationOp(HasStrictTraits):
    """
    Calibrate arbitrary channels to molecules-of-fluorophore using fluorescent
    beads (eg, the Spherotech RCP-30-5A rainbow beads.)
    
    To use, set the `beads_file` property to an FCS file containing the beads'
    events; specify which beads you ran by setting the `beads_type` property
    to match one of the values of BeadCalibrationOp.BEADS; and set the
    `units` dict to which channels you want calibrated and in which units.
    Then, call `estimate()` and check the peak-finding with 
    `default_view().plot()`.  If the peak-finding is wacky, try adjusting
    `bead_peak_quantile` and `bead_brightness_threshold`.  When the peaks are
    successfully identified, call apply() on your experimental data set. 
    
    If you can't make the peak finding work, please submit a bug report!
    
    This procedure works best when the beads file is very clean data.  It does
    not do its own gating (maybe a future addition?)  In the meantime, 
    I recommend gating the *acquisition* on the FSC/SSC channels in order
    to get rid of debris, cells, and other noise.
    
    Also, this module assumes that the right-most peak is the brightest peak
    in the tube.  If you have your PMT voltages so high that you have multiple
    peaks off-scale at the top, this calibration will not be valid.
    
    Finally, because you can't have a negative number of fluorescent molecules
    (MEFLs, etc) (as well as for math reasons), this module filters out
    negative values.
    
    Attributes
    ----------
    name : Str
        The operation name (for UI representation.)

    units : Dict(Str, Str)
        A dictionary specifying the channels you want calibrated (keys) and
        the units you want them calibrated in (values).  The units must be
        keys of the `beads` attribute.       
        
    beads_file : File
        A file containing the FCS events from the beads.  Must be set to use
        `estimate()`.  This isn't persisted by `pickle()`.

    beads : Dict(Str, List(Float))
        The beads' characteristics.  Keys are calibrated units (ie, MEFL or
        MEAP) and values are ordered lists of known fluorophore levels.  Common
        values for this dict are included in BeadCalibrationOp.BEADS.
        Must be set to use `estimate()`.
        
    bead_peak_quantile : Int
        The quantile threshold used to choose bead peaks.  Default == 80.
        Must be set to use `estimate()`.
        
    bead_brightness_threshold : Float
        How bright must a bead peak be to be considered?  Default == 100.
        Must be set to use `estimate()`.
        
    Notes
    -----
    The peak finding is rather sophisticated.  
    
    For each channel, a 1024-bin histogram is computed on the log-transformed
    bead data, and then the histogram is smoothed with a Savitzky-Golay 
    filter (with a window length of 7 and a polynomial order of 0).  
    
    Next, a wavelet-based peak-finding algorithm is used: it convolves the
    smoothed histogram with a series of wavelets and looks for relative 
    maxima at various length-scales.  The parameters of the smoothing 
    algorithm were arrived at empircally, using beads collected at a wide 
    range of PMT voltages.
    
    Finally, the peaks are filtered by height (the histogram bin has a quantile
    greater than `bead_peak_quantile`) and intensity (brighter than 
    `bead_brightness_threshold`).
    
    How to convert from a series of peaks to mean equivalent fluorochrome?
    We assume that the brightest peak is the brightest bead population, then
    do a linear regression of the (log-transformed) peak locations vs the
    (log-transformed) MEFs.
    
    There's a slight subtlety in the fact that we're performing the linear
    regression in log-space: if the relationship in log10-space is Y=aX + b,
    then the same relationship in linear space is x = 10**X, y = 10**y, and
    y = (10**b) * (x ** a).
    
    One more thing.  Because the beads are (log) evenly spaced across all
    the channels, we can directly compute the fluorophore equivalent in channels
    where we wouldn't usually measure that fluorophore: for example, you can
    compute MEFL (mean equivalent fluorosceine) in the PE-Texas Red channel,
    because the bead peak pattern is the same in the PE-Texas Red channel
    as it would be in the FITC channel.
    
    Examples
    --------
    >>> bead_op = flow.BeadCalibrationOp()
    >>> bead_op.beads = flow.BeadCalibrationOp.BEADS["Spherotech RCP-30-5A Lot AA01-AA04, AB01, AB02, AC01, GAA01-R"]
    >>> bead_op.units = {"Pacific Blue-A" : "MEFL",
                         "FITC-A" : "MEFL",
                         "PE-Tx-Red-YG-A" : "MEFL"}
    >>>
    >>> bead_op.beads_file = "beads.fcs"
    >>> bead_op.estimate(ex3)
    >>>
    >>> bead_op.default_view().plot(ex3)  
    >>> # check the plot!
    >>>
    >>> ex4 = bead_op.apply(ex3)  
    """
    
    # traits
    id = Constant('edu.mit.synbio.cytoflow.operations.beads_calibrate')
    friendly_id = Constant("Bead Calibration")
    
    name = CStr()
    units = Dict(Str, Str)
    unit_scale = Dict(Str, Float)
    channel_scale = Dict(Str, Float)
    
    beads_file = File(transient = True)
    bead_peak_quantile = Int(80)

    bead_brightness_threshold = Float(100)
    # TODO - bead_brightness_threshold should probably be different depending
    # on the data range of the input.
    
    beads = Dict(Str, List(Float), transient = True)

    _calibration_functions = Dict(Str, Python)
    
    # keep track of peaks and mefs for diagnostic plot
    _peaks = Dict(Str, Array)
    _mefs = Dict(Str, Array)

    def estimate(self, experiment, subset = None): 
        """
        Estimate the calibration coefficients from the beads file.
        """
        if not experiment:
            raise util.CytoflowOpError("No experiment specified")

        if not set(self.units.keys()) <= set(experiment.channels):
            raise util.CytoflowOpError("Specified channels that weren't found in "
                                  "the experiment.")
            
        if not set(self.units.values()) <= set(self.beads.keys()):
            raise util.CytoflowOpError("Units don't match beads.")
        
        # make a little Experiment
        check_tube(self.beads_file, experiment)
        beads_exp = ImportOp(tubes = [Tube(file = self.beads_file)],
                             name_metadata = experiment.metadata['name_metadata']).apply()
                             
        # unlike other operations, we DON'T apply the history to the beads (-;
        channels = self.units.keys()

        for channel in channels:
            data = beads_exp.data[channel]
            
            # TODO - this assumes the data is on a linear scale.  check it!
            
            # bin the data on a log scale
            data_range = experiment.metadata[channel]['range']
            hist_bins = np.logspace(1, math.log(data_range, 2), num = 1024, base = 2)
            hist = np.histogram(data, bins = hist_bins)
            
            # mask off-scale values
            hist[0][0] = 0
            hist[0][-1] = 0
            
            # smooth it with a Savitzky-Golay filter
            hist_smooth = scipy.signal.savgol_filter(hist[0], 7, 0)
            
            # find peaks
            peak_bins = scipy.signal.find_peaks_cwt(hist_smooth, 
                                                    widths = np.arange(5, 20),
                                                    max_distances = np.arange(5, 20) / 2)
            
            # filter by height and intensity
            peak_threshold = np.percentile(hist_smooth, self.bead_peak_quantile)
            peak_bins_filtered = \
                [x for x in peak_bins if hist_smooth[x] > peak_threshold 
                 and hist[1][x] > self.bead_brightness_threshold]
            
            scale = self.unit_scale[channel] if channel in self.unit_scale else 1.0
            peaks = [hist_bins[x] for x in peak_bins_filtered]
         
            if channel in self.unit_scale:
                peaks = [x / self.unit_scale[channel] for x  in peaks]
            
            self._peaks[channel] = peaks
            mef_unit = self.units[channel]
            
            # "mean equivalent fluorochrome"
            mefs = self.beads[mef_unit]
            
            if not mef_unit in self.beads:
                raise util.CytoflowOpError("Invalid unit {0} specified for channel {1}".format(mef_unit, channel))
            
            if len(peaks) == 0:
                raise util.CytoflowOpError("Didn't find any peaks; check the diagnostic plot")
            elif len(peaks) > len(self.beads):
                raise util.CytoflowOpError("Found too many peaks; check the diagnostic plot")
            elif len(peaks) == 1:
                # if we only have one peak, so just assume a multiplicative conversion
                self._mefs[channel] = [mefs[-1]]
                a = mefs[-1] / peaks[0]
                self._calibration_functions[channel] = lambda x, a=a: a * x
            else:
                # if there are n >= 2 peaks, assume that the brighest peak is
                # the brightest bead population and do a (log)linear regression
                # to find the transformation
                 
                # do it in log10 space because otherwise the brightest peaks
                # have an outsized influence.
 
                self._mefs[channel] = mefs[len(mefs) - len(peaks):]
                     
                # linear regression of the peak locations against mef subset,
                # just to warn of nonlinearity (eg PMT voltage too high or low)
                lr = np.polyfit(np.log10(self._peaks[channel]), 
                                np.log10(self._mefs[channel]), 
                                deg = 1, 
                                full = True)
                
                a = lr[0][0]
                if np.abs(1.0 - a) > 0.05:
                    warnings.warn("Calibration fit for channel {0} is non-linear!"
                                  .format(channel),
                                  util.CytoflowOpWarning)
               
                # we used to use the entire (log)linear regression, but the 
                # (log) part introduced nonlinearities which were bad.   
                #  b = 10 ** lr[0][1]
                #  self._calibration_functions[channel] = \
               #      lambda x, a=a, b=b: b * np.power(x, a)
                    
                # now, we use a simple multiplicative scale
                def mef_err(x):
                    return np.sqrt(np.sum(np.square(np.log10(x * self._peaks[channel]) 
                                                  - np.log10(self._mefs[channel]))))
 
                m = scipy.optimize.minimize(mef_err, 1)
 
                self._calibration_functions[channel] = \
                    lambda x, a=m.x[0]: a * x

    def apply(self, experiment):
        """Applies the bleedthrough correction to an experiment.
        
        Parameters
        ----------
        old_experiment : Experiment
            the experiment to which this op is applied
            
        Returns
        -------
            a new experiment calibrated in physical units.
        """
        if not experiment:
            raise util.CytoflowOpError("No experiment specified")
        
        channels = self.units.keys()

        if not self.units:
            raise util.CytoflowOpError("Units not specified.")
        
        if not self._calibration_functions:
            raise util.CytoflowOpError("Calibration not found. "
                                  "Did you forget to call estimate()?")
        
        if not set(channels) <= set(experiment.channels):
            raise util.CytoflowOpError("Module units don't match experiment channels")
                
        if set(channels) != set(self._calibration_functions.keys()):
            raise util.CytoflowOpError("Calibration doesn't match units. "
                                  "Did you forget to call estimate()?")

        # negative physical units don't make sense -- how can you have the 
        # equivalent of -5 molecules of fluoresceine?  so, we filter out 
        # negative values here.

        new_experiment = experiment.clone()
        
#         for channel in channels:
#             new_experiment.data = \
#                 new_experiment.data[new_experiment.data[channel] > 0]
#                 
#         new_experiment.data.reset_index(drop = True, inplace = True)
        
        for channel in channels:
            calibration_fn = self._calibration_functions[channel]
            
            new_experiment[channel] = calibration_fn(new_experiment[channel])
            
            if channel in self.channel_scale:
                new_experiment[channel] = new_experiment[channel].divide(self.channel_scale[channel])
            
            new_experiment.metadata[channel]['bead_calibration_fn'] = calibration_fn
            new_experiment.metadata[channel]['units'] = self.units[channel]
            if 'range' in experiment.metadata[channel]:
                new_experiment.metadata[channel]['range'] = calibration_fn(experiment.metadata[channel]['range'])
            
        new_experiment.history.append(self.clone_traits()) 
        return new_experiment
    
    def default_view(self, **kwargs):
        """
        Returns a diagnostic plot to see if the bleedthrough spline estimation
        is working.
        
        Returns
        -------
            IView : An IView, call plot() to see the diagnostic plots
        """

        return BeadCalibrationDiagnostic(op = self, **kwargs)
    
    BEADS = {
             # from http://www.spherotech.com/RCP-30-5a%20%20rev%20H%20ML%20071712.xls
             "Spherotech RCP-30-5A Lot AG01, AF02, AD04 and AAE01" :
                { "MECSB" : [216, 464, 1232, 2940, 7669, 19812, 35474],
                  "MEBFP" : [861, 1997, 5776, 15233, 45389, 152562, 396759],
                  "MEFL" :  [792, 2079, 6588, 16471, 47497, 137049, 271647],
                  "MEPE" :  [531, 1504, 4819, 12506, 36159, 109588, 250892],
                  "MEPTR" : [233, 669, 2179, 5929, 18219, 63944, 188785],
                  "MECY" : [1614, 4035, 12025, 31896, 95682, 353225, 1077421],
                  "MEPCY7" : [14916, 42336, 153840, 494263],
                  "MEAP" :  [373, 1079, 3633, 9896, 28189, 79831, 151008],
                  "MEAPCY7" : [2864, 7644, 19081, 37258]},
             # from http://www.spherotech.com/RCP-30-5a%20%20rev%20G.2.xls
             "Spherotech RCP-30-5A Lot AA01-AA04, AB01, AB02, AC01, GAA01-R":
                { "MECSB" : [179, 400, 993, 3203, 6083, 17777, 36331],
                  "MEBFP" : [700, 1705, 4262, 17546, 35669, 133387, 412089],
                  "MEFL" :  [692, 2192, 6028, 17493, 35674, 126907, 290983],
                  "MEPE" :  [505, 1777, 4974, 13118, 26757, 94930, 250470],
                  "MEPTR" : [207, 750, 2198, 6063, 12887, 51686, 170219],
                  "MECY" :  [1437, 4693, 12901, 36837, 76621, 261671, 1069858],
                  "MEPCY7" : [32907, 107787, 503797],
                  "MEAP" :  [587, 2433, 6720, 17962, 30866, 51704, 146080],
                  "MEAPCY7" : [718, 1920, 5133, 9324, 14210, 26735]}}
    
@provides(cytoflow.views.IView)
class BeadCalibrationDiagnostic(HasStrictTraits):
    """
    Plots diagnostic histograms of the peak finding algorithm.
    
    Attributes
    ----------
    name : Str
        The instance name (for serialization, UI etc.)
    
    op : Instance(BeadCalibrationDiagnostic)
        The op whose parameters we're viewing
        
    """
    
    # traits   
    id = "edu.mit.synbio.cytoflow.view.autofluorescencediagnosticview"
    friendly_id = "Autofluorescence Diagnostic" 
    
    name = Str
    
    # TODO - why can't I use BeadCalibrationOp here?
    op = Instance(IOperation)
    
    def plot(self, experiment, **kwargs):
        """Plot a faceted histogram view of a channel"""
        
        if not self.op._calibration_functions:
            raise util.CytoflowViewError("No calibration found. "
                                         "Did you forget to call estimate()?")
              
        # make a little Experiment
        check_tube(self.op.beads_file, experiment)
        beads_exp = ImportOp(tubes = [Tube(file = self.op.beads_file)],
                             name_metadata = experiment.metadata['name_metadata']).apply()

        plt.figure()
        
        channels = self.op.units.keys()
        
        for idx, channel in enumerate(channels):
            data = beads_exp.data[channel]
            
            # bin the data on a log scale
            data_range = experiment.metadata[channel]['range']
            hist_bins = np.logspace(1, math.log(data_range, 2), num = 1024, base = 2)
            hist = np.histogram(data, bins = hist_bins)
            
            # mask off-scale values
            hist[0][0] = 0
            hist[0][-1] = 0
            
            hist_smooth = scipy.signal.savgol_filter(hist[0], 7, 0)
            
            # find peaks
            peak_bins = scipy.signal.find_peaks_cwt(hist_smooth, 
                                                    widths = np.arange(5, 20),
                                                    max_distances = np.arange(5, 20) / 2)
            
            # filter by height and intensity
            peak_threshold = np.percentile(hist_smooth, self.op.bead_peak_quantile)
            peak_bins_filtered = \
                [x for x in peak_bins if hist_smooth[x] > peak_threshold
                 and hist[1][x] > self.op.bead_brightness_threshold]
                
            plt.subplot(len(channels), 2, 2 * idx + 1)
            plt.xscale('log')
            plt.xlabel(channel)
            plt.plot(hist_bins[1:], hist_smooth)
            for peak in peak_bins_filtered:
                plt.axvline(hist_bins[peak], color = 'r')

            plt.subplot(len(channels), 2, 2 * idx + 2)
            for mef in self.op.beads[self.op.units[channel]]:
                plt.axhline(mef, color = 'c')
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(channel)
            plt.ylabel(self.op.units[channel])
            plt.plot(self.op._peaks[channel], 
                     self.op._mefs[channel], 
                     marker = 'o',
                     linestyle = "")
              
            xmin, xmax = plt.xlim()
            x = np.logspace(np.log10(xmin), np.log10(xmax))
            plt.plot(x, 
                     self.op._calibration_functions[channel](x), 
                     color = 'r', linestyle = ':')
            
        plt.tight_layout(pad = 0.8)
            
