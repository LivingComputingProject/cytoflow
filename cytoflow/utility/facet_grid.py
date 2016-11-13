'''
Created on Nov 13, 2016

@author: brian
'''

import warnings
from itertools import product

import seaborn
import numpy as np
import matplotlib.pyplot as plt

class MultiIndexGrid(seaborn.FacetGrid):
    
    def __init__(self, data, row=None, col=None, hue=None, col_wrap=None,
             sharex=True, sharey=True, size=3, aspect=1, palette=None,
             row_order=None, col_order=None, hue_order=None, hue_kws=None,
             legend_out=True, despine=True, margin_titles=False, xlim=None, 
             ylim=None, subplot_kws=None, gridspec_kws=None):
        
        # Determine the hue facet layer information
        hue_var = hue
        if hue is None:
            hue_names = None
        else:
            hue_names = data.index.get_level_values(hue)
            hue_names = seaborn.categorical_order(hue_names, hue_order)

        colors = self._get_palette(data, hue, hue_order, palette)

        # Set up the lists of names for the row and column facet variables
        if row is None:
            row_names = []
        else:
            row_names = data.index.get_level_values(row)
            row_names = seaborn.categorical_order(row_names, row_order)

        if col is None:
            col_names = []
        else:
            col_names = data.index.get_level_values(col)
            col_names = seaborn.categorical_order(col_names, col_order)


        # Additional dict of kwarg -> list of values for mapping the hue var
        hue_kws = hue_kws if hue_kws is not None else {}

        # Make a boolean mask that is True anywhere there is an NA
        # value in one of the faceting variables, but only if dropna is True
#         none_na = np.zeros(len(data), np.bool)
#         if dropna:
#             row_na = none_na if row is None else data[row].isnull()
#             col_na = none_na if col is None else data[col].isnull()
#             hue_na = none_na if hue is None else data[hue].isnull()
#             not_na = ~(row_na | col_na | hue_na)
#         else:
#             not_na = ~none_na

        # Compute the grid shape
        ncol = 1 if col is None else len(col_names)
        nrow = 1 if row is None else len(row_names)
        self._n_facets = ncol * nrow

        self._col_wrap = col_wrap
        if col_wrap is not None:
            if row is not None:
                err = "Cannot use `row` and `col_wrap` together."
                raise ValueError(err)
            ncol = col_wrap
            nrow = int(np.ceil(len(data[col].unique()) / col_wrap))
        self._ncol = ncol
        self._nrow = nrow

        # Calculate the base figure size
        # This can get stretched later by a legend
        figsize = (ncol * size * aspect, nrow * size)

        # Validate some inputs
        if col_wrap is not None:
            margin_titles = False

        # Build the subplot keyword dictionary
        subplot_kws = {} if subplot_kws is None else subplot_kws.copy()
        gridspec_kws = {} if gridspec_kws is None else gridspec_kws.copy()
        if xlim is not None:
            subplot_kws["xlim"] = xlim
        if ylim is not None:
            subplot_kws["ylim"] = ylim

        # Initialize the subplot grid
        if col_wrap is None:
            kwargs = dict(figsize=figsize, squeeze=False,
                          sharex=sharex, sharey=sharey,
                          subplot_kw=subplot_kws,
                          gridspec_kw=gridspec_kws)

            fig, axes = plt.subplots(nrow, ncol, **kwargs)
            self.axes = axes

        else:
            # If wrapping the col variable we need to make the grid ourselves
            if gridspec_kws:
                warnings.warn("`gridspec_kws` ignored when using `col_wrap`")

            n_axes = len(col_names)
            fig = plt.figure(figsize=figsize)
            axes = np.empty(n_axes, object)
            axes[0] = fig.add_subplot(nrow, ncol, 1, **subplot_kws)
            if sharex:
                subplot_kws["sharex"] = axes[0]
            if sharey:
                subplot_kws["sharey"] = axes[0]
            for i in range(1, n_axes):
                axes[i] = fig.add_subplot(nrow, ncol, i + 1, **subplot_kws)
            self.axes = axes

            # Now we turn off labels on the inner axes
            if sharex:
                for ax in self._not_bottom_axes:
                    for label in ax.get_xticklabels():
                        label.set_visible(False)
                    ax.xaxis.offsetText.set_visible(False)
            if sharey:
                for ax in self._not_left_axes:
                    for label in ax.get_yticklabels():
                        label.set_visible(False)
                    ax.yaxis.offsetText.set_visible(False)

        # Set up the class attributes
        # ---------------------------

        # First the public API
        self.data = data
        self.fig = fig
        self.axes = axes

        self.row_names = row_names
        self.col_names = col_names
        self.hue_names = hue_names
        self.hue_kws = hue_kws

        # Next the private variables
        self._nrow = nrow
        self._row_var = row
        self._ncol = ncol
        self._col_var = col

        self._margin_titles = margin_titles
        self._col_wrap = col_wrap
        self._hue_var = hue_var
        self._colors = colors
        self._legend_out = legend_out
        self._legend = None
        self._legend_data = {}
        self._x_var = None
        self._y_var = None
        self._dropna = False
        self._not_na = np.ones(len(data), np.bool)

        # Make the axes look good
        fig.tight_layout()
        if despine:
            self.despine()    
            
    def facet_data(self):
        """Generator for name indices and data subsets for each facet.

        Yields
        ------
        (i, j, k), data_ijk : tuple of ints, DataFrame
            The ints provide an index into the {row, col, hue}_names attribute,
            and the dataframe contains a subset of the full data corresponding
            to each facet. The generator yields subsets that correspond with
            the self.axes.flat iterator, or self.axes[i, j] when `col_wrap`
            is None.

        """
        data = self.data
        if self._nrow == 1 or self._col_wrap is not None:
            rows = [None]
        else:
            rows = self.row_names
            
        if self._ncol == 1:
            cols = [None]
        else:
            cols = self.col_names
            
        if len(self._colors) == 1:
            hues = [None]
        else:
            hues = self.hue_names
            
        for (i, row), (j, col), (k, hue) in product(enumerate(rows),
                                                    enumerate(cols),
                                                    enumerate(hues)):
            idx = [x for x in (row, col, hue) if x is not None]
            level = [x[1] for x in ((row, self._row_var),
                                     (col, self._col_var),
                                     (hue, self._hue_var))
                      if x[0] is not None]
            
            data_ijk = data.xs(idx, level = level)
            yield (i, j, k), data_ijk
        