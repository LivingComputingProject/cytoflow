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

from __future__ import absolute_import

import pandas as pd
from traits.api import HasStrictTraits, Dict, List, Instance, Str, Any, \
                       Property, Tuple

import cytoflow.utility as util

class Experiment(HasStrictTraits):
    """An Experiment manages all the data and metadata for a flow experiment.
    
    An `Experiment` is `cytoflow`'s central data structure: it wraps a 
    `pandas.DataFrame` containing all the data from a flow experiment. Each 
    row in the table is an event.  Each column is a measurement from one of the 
    detectors (or a "derived" measurement such as a transformed value or a 
    ratio).  Experimental conditions are represented as levels in a hierarchical
    `pandas.MultiIndex` and are pieces of metadata associated with that event: 
    which tube it came from, what the experimental conditions for that tube 
    were, gate membership, etc.  Each condition's name must be a valid Python
    identifier.  The `Experiment` object lets you:
      - Add additional conditions to define subpopulations
      - Get events that match a particular set of conditions.
      
    Additionally, the `Experiment` object manages channel- and experiment-level
    metadata in the `metadata` field, which is a dictionary.  This allows
    the rest of the `cytoflow` package to track and enforce other constraints
    that are important in doing quantitative flow cytometry: for example,
    every tube must be collected with the same channel parameters (such 
    as PMT voltage.)

    NOTE: `Experiment` is not responsible for enforcing the constraints; 
    `cytoflow.ImportOp` and the other modules are.
    
    Attributes
    ----------

    data : pandas.DataFrame
        the `DataFrame` representing all the events and metadata.  Each event
        is a row; each column is a channel (PMT measurements or "derived"
        channels such as transformed values or ratios.)  Experimental conditions
        and metadata from the analysis (gate membership, etc) are stored as
        levels in `data`'s index, which is a `pandas.MultiIndex`.
        
    metadata : Dict(Str : Dict)
        Per-channel information about the data in the Experiment.  Each column
        in self.data has an entry whose key is the column name and whose value
        is a dict of channel-specific metadata.  Metadata is added by
        operations, which is occasionally useful if modules are expected to
        work together.  Each module's documentation should specify which
        metadata it adds.
    
    history : List(IOperation)
        A list of the operations that have been applied to the raw data that
        have led to this Experiment.
        
    statistics : Dict((Str, Str) : pandas.Series)
        A dictionary of statistics and parameters computed by models that were
        fit to the data.  The key is an (Str, Str) tuple, where the first Str
        is the name of the operation that supplied the statistic, and the second
        Str is the name of the statistic. The value is a multi-indexed 
        `pandas.Series`: each level of the index is an experimental condition, 
        and each row is a subset for which the statistic was computed.
    
    channels : List(String)
        A read-only `List` containing the channels that this experiment tracks.
    
    conditions : Dict(String : pandas.Series)
        A read-only Dict of the experimental conditions and analysis metadata 
        (gate membership, etc) and that this experiment tracks.  The key is the 
        name of the condition, and the value is a `pandas.Series` containing
        all of the values of this condition.

    Implementation details
    ----------------------
    
    The OOP programmer in me desperately wanted to subclass DataFrame, add
    some flow-specific stuff, and move on with my life.  (I may still, with
    something like https://github.com/dalejung/pandas-composition).  A few 
    things get in the way of directly subclassing pandas.DataFrame:
    
     - First, to enable some of the delicious syntactic sugar for accessing
       its contents, DataFrame redefines ``__getattribute__`` and 
       ``__setattribute__``, and making it recognize (and maintain across 
       copies) additional attributes is an unsupported (non-public) API 
       feature and introduces other subclassing weirdness.
    
     - Second, many of the operations (like appending!) don't happen in-place;
       they return copies instead.  It's cleaner to simply manage that copying
       ourselves instead of making the client deal with it.  We can pretend
       to operate on the data in-place.
       
    To maintain the ease of use, we'll override __getitem__ and pass it to
    the wrapped DataFrame.  We'll do the same with some of the more useful
    DataFrame API pieces (like query()); and of course, you can just get the
    data frame itself with `Experiment.data`
    
    Examples
    --------
    >>> import cytoflow as flow
    >>> tube1 = flow.Tube(file = 'cytoflow/tests/data/Plate01/RFP_Well_A3.fcs',
    ...                   conditions = {"Dox" : 10.0})
    >>> tube2 = flow.Tube(file='cytoflow/tests/data/Plate01/CFP_Well_A4.fcs',
    ...                   conditions = {"Dox" : 1.0})
    >>> 
    >>> import_op = flow.ImportOp(conditions = {"Dox" : "float"},
    ...                           tubes = [tube1, tube2])
    >>> 
    >>> ex = import_op.apply()
    >>> ex.data.shape
    (20000, 17)
    >>> ex.data.groupby(['Dox']).size()
    Dox
    1      10000
    10     10000
    dtype: int64

    """

    # this doesn't play nice with copy.copy(); clone it ourselves.
    data = Instance(pd.DataFrame, args=())
    
    # potentially mutable.  deep copy required
    metadata = Dict(Str, Dict(Str, Any), copy = "deep")
    
    # statistics.  mutable, deep copy required
    statistics = Dict(Tuple(Str, Str), pd.Series, copy = "deep")
    
    history = List(HasStrictTraits)
    
    channels = Property(List)
    conditions = Property(Dict(Str, pd.Series))
            
    def __getitem__(self, key):
        """Override __getitem__ so we can reference columns like ex.column"""
        return self.data.__getitem__(key)
     
    def __setitem__(self, key, value):
        """Override __setitem__ so we can assign columns like ex.column = ..."""
        return self.data.__setitem__(key, value)
    
    def __len__(self):
        return len(self.data)

    def _get_channels(self):
        return self.data.columns.values
    
    def _get_conditions(self):
        idx = self.data.index
        try:
            return {name : pd.Series(idx.get_level_values(name).unique()) 
                    for name in idx.names if name}
        except AttributeError:
            return []
        
    def subset(self, name, value):
        """
        A fast way to get a subset of the data where a condition equals a 
        particular value.
        
        Parameters
        ----------
        name : Str
            A condition; ie, a key in `self.conditions`.
            
        value : Any
            The value to look for.  Will be checked with equality, ie `==`
            
        """
        ret = self.clone()
        ret.data = self.data.xs(value, name, drop_level = False)
        return ret
    
    
    def query(self, expr, **kwargs):
        """
        Expose pandas.DataFrame.query() to the outside world.

        You can refer to both conditions and channels with `query`.  This method
        "sanitizes" column names first, replacing characters that are not valid
        in a Python identifier (`A-Z`, `a-z`, `0-9` and `_`) with an underscore
        `_`. So, the column name `FITC-A` becomes `FITC_A`, and can be queried
        with `FITC_A > 0.0` or such.
        
        Parameters
        ----------
        expr : string
            The expression to pass to `pandas.DataFrame.query()`.  Must be
            a valid Python expression, something you could pass to `eval()`.
            
        **kwargs : dict
            Other named parameters to pass to `pandas.DataFrame.query()`.
            
        Returns
        -------
        A new `Experiment`, a clone of this one with the data returned by
        `pandas.DataFrame.query()`
        """
        
        resolvers = {}
        for name, col in self.data.iteritems():
            new_name = util.sanitize_identifier(name)
            if new_name in resolvers:
                raise util.CytoflowError("Tried to sanitize column name {1} to "
                                         "{2} but it already existed in the "
                                         " DataFrame."
                                         .format(name, new_name))
            else:
                resolvers[new_name] = col
                
        ret = self.clone()
        ret.data = self.data.query(expr, resolvers = ({}, resolvers), **kwargs)
        ret.data.reset_index(drop = True, inplace = True)
        return ret
    
    def clone(self):
        """Clone this experiment"""
        new_exp = self.clone_traits()
        new_exp.data = self.data.copy()

        # shallow copy of the history
        new_exp.history = self.history[:]
        return new_exp
            
    def add_condition(self, name, dtype, data = None):
        """Add a new column of per-event metadata to this `Experiment`.  Operates
           *in place*.
        
        There are two places to call `add_condition`.
          - As you're setting up a new `Experiment`, call `add_condition()`
            with `data` set to `None` to specify the conditions the new events
            will have.
          - If you compute some new per-event metadata on an existing 
            `Experiment`, call `add_condition()` to add it. 
        
        Parameters
        ----------
        name : String
            The name of the new condition.  Must be a valid Python identifier,
            containing characters only in `A-Z`, `a-z`, `0-9` and `_`.
        
        dtype : String
            The type of the new column in `self.data`.  Must be a string that
            `pandas.Series` recognizes as a `dtype`: common types are 
            "category", "float", "int", and "bool".
            
        data : pandas.Series (default = None)
            The `pandas.Series` to add to `self.data`.  Must be the same
            length as `self.data`, and it must be convertable to a 
            `pandas.Series` of type `dtype`.  If `None`, will add an
            empty column to the `Experiment` ... but the `Experiment` must
            be empty to do so!
             
        Raises
        ------
        CytoflowError
            If the `pandas.Series` passed in `data` isn't the same length
            as `self.data`, or isn't convertable to type `dtype`.          
            
        Examples
        --------
        >>> import cytoflow as flow
        >>> ex = flow.Experiment()
        >>> ex.add_condition("Time", "float")
        >>> ex.add_condition("Strain", "category")      
        """

        if name != util.sanitize_identifier(name):
            raise util.CytoflowError("The name '{}' must be a valid Python "
                                     "identifier".format(name))
        
        if name in self.channels:
            raise util.CytoflowError("Already a channel named {}".format(name))
        
        if name in self.conditions:
            raise util.CytoflowError("Already a condition named {}".format(name))
        
        if data is None and len(self) > 0:
            raise util.CytoflowError("If data is None, self.data must be empty!")
        
        if data is not None and len(self) != len(data):
            raise util.CytoflowError("data must be the same length as self.data")
        
        if data is not None:
            self.data[name] = data.astype(dtype, copy = True)
        else:
            if dtype == "category":
                self.data[name] = pd.Series(dtype = "object")
            else:
                self.data[name] = pd.Series(dtype = dtype)
            
        self.data.set_index(name, append = True, inplace = True)
        self.data.sort_index(inplace = True)
            
    def add_channel(self, name, data = None):
        """Add a new column of per-event "data" (as opposed to metadata) to this
          `Experiment`: ie, something that was measured per cell, or derived
          from per-cell measurements.  Operates *in place*.
        
        Parameters
        ----------
        name : String
            The name of the new column in `self.data`.
            
        data : pandas.Series
            The `pandas.Series` to add to `self.data`.  Must be the same
            length as `self.data`, and it must be convertable to a 
            dtype of `float64` of type `dtype`.  If `None`, will add an
            empty column to the `Experiment` ... but the `Experiment` must
            be empty to do so!
             
        Raises
        ------
        CytoflowError
            If the `pandas.Series` passed in `data` isn't the same length
            as `self.data`, or isn't convertable to a dtype `float64`.          
            
        Examples
        --------
        >>> ex.add_channel("FSC_over_2", ex["FSC-A"] / 2.0) 
        """
        
        if name in self.channels:
            raise util.CytoflowError("Already a channel named {}".format(name))
            
        if name in self.conditions:
            raise util.CytoflowError("Already a condition named {}".format(name))

        if data is None and len(self) > 0:
            raise util.CytoflowError("If data is None, self.data must be empty!")

        if data is not None and len(self) != len(data):
            raise util.CytoflowError("data must be the same length as self.data")
        
        try:
            if data is not None:
                self.data[name] = data.astype("float64", copy = True)
            else:
                self.data[name] = pd.Series(dtype = "float64")
                
            self.metadata[name] = {}
                
        except (ValueError, TypeError):
                raise util.CytoflowError("Had trouble converting data to type \"float64\"")
        
    def add_events(self, data, conditions, check_conditions = True):
        """
        Add new events to this Experiment.
        
        This function adds appropriate index levels to `data` with values from
        `conditions`, then appends it to `self.data`.  Thus, it is particularly
        useful for adding tubes of data to new experiments, before additional
        per-event metadata is added by gates, etc.
        
        EVERY column in `self.data` must be accounted for.  Each column of
        type `channel` must appear in `data`; each column of metadata must
        have a key:value pair in `conditions`.
        
        Parameters
        ----------
        tube : pandas.DataFrame
            A single tube or well's worth of data. Must be a DataFrame with
            the same columns as `self.channels`
        
        conditions : Dict(Str, Any)
            A dictionary of the tube's metadata.  The keys must match 
            `self.conditions`, and the values must be coercable to the
            relevant `numpy.dtype`.
            
        check_conditions : Bool (default = True)
            If `True` (the default), throw an error if the conditions in
            `conditions` have already been added to the experiment.
 
        Raises
        ------
        CytoflowError
            - If there are columns in `data` that aren't channels in the 
              experiment, or vice versa. 
            - If there are keys in `conditions` that aren't conditions in
              the experiment, or vice versa.
            - If there is metadata specified in `conditions` that can't be
              converted to the corresponding metadata dtype.
            - If `check_conditions` is `True` and events with the same 
              conditions have already been added to the Experiment.
            
        Examples
        --------
        >>> import cytoflow as flow
        >>> import fcsparser
        >>> ex = flow.Experiment()
        >>> ex.add_condition("Time", "float")
        >>> ex.add_condition("Strain", "category")
        >>> tube1, _ = fcparser.parse('CFP_Well_A4.fcs')
        >>> tube2, _ = fcparser.parse('RFP_Well_A3.fcs')
        >>> ex.add_events(tube1, {"Time" : 1, "Strain" : "BL21"})
        >>> ex.add_events(tube2, {"Time" : 1, "Strain" : "Top10G"})
        """

        # make sure the new tube's channels match the rest of the 
        # channels in the Experiment
    
        if len(self) > 0 and set(data.columns) != set(self.channels):
            raise util.CytoflowError("New events don't have the same channels")
            
        # check that the conditions for this tube exist in the experiment
        # already

        if( any(True for k in conditions if k not in self.conditions) or \
            any(True for k in self.conditions if k not in conditions) ):
            raise util.CytoflowError("Conditions {} don't match the experiment's "
                                     "conditions {}"
                                     .format(conditions.keys(), self.conditions.keys()))
            
        # add the conditions to tube's internal data frame.  specify the conditions
        # dtype using self.conditions.  check for errors as we do so.
        
        # take this chance to up-convert the float32s to float64.
        # this happened automatically in DataFrame.append(), below, but 
        # only in certain cases.... :-/
        
        # TODO - the FCS standard says you can specify the precision.  
        # check with int/float/double files!
        
        new_data = data.astype("float64", copy=True)
        
        for meta_name, meta_value in conditions.iteritems():
            meta_type = self.conditions[meta_name]
            try:
                new_data[meta_name] = \
                    pd.Series(data = [meta_value] * len(new_data),
                              index = new_data.index,
                              dtype = meta_type)
                new_data.set_index(meta_name, append = True, inplace = True)
                
                # if we're categorical, merge the categories
#                 if meta_type == "category" and meta_name in self.data.columns:
#                     cats = set(self.data[meta_name].cat.categories) | set(new_data[meta_name].cat.categories)
#                     self.data[meta_name] = self.data[meta_name].cat.set_categories(cats)
#                     new_data[meta_name] = new_data[meta_name].cat.set_categories(cats)
            except (ValueError, TypeError):
                raise util.CytoflowError("Had trouble converting conditions {0}"
                                         "(value = {1}) to type {2}" \
                                         .format(meta_name,
                                                 meta_value,
                                                 meta_type))
#        self.data = self.data.append(new_data, ignore_index = True)
        self.data = self.data.append(new_data)
        del new_data

if __name__ == "__main__":
    import fcsparser
    ex = Experiment()
    ex.add_conditions({"time" : "category"})

    tube0, _ = fcsparser.parse('../cytoflow/tests/data/tasbe/BEADS-1_H7_H07_P3.fcs')
    tube1, _ = fcsparser.parse('../cytoflow/tests/data/tasbe/beads.fcs')
    tube2, _ = fcsparser.parse('../cytoflow/tests/data/Plate01/RFP_Well_A3.fcs')
    
    ex.add_tube(tube1, {"time" : "one"})
    ex.add_tube(tube2, {"time" : "two"})
    
