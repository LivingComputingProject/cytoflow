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
Created on Mar 23, 2015

@author: brian
'''

# for local debugging
if __name__ == '__main__':
    from traits.etsconfig.api import ETSConfig
    ETSConfig.toolkit = 'qt4'

    import os
    os.environ['TRAITS_DEBUG'] = "1"
    
from traits.api import HasStrictTraits, List, CFloat, Str, Interface, \
                       Property, Bool, provides
from traitsui.api import View, CheckListEditor, Item, HGroup

from cytoflowgui.value_bounds_editor import ValuesBoundsEditor
import cytoflow.utility as util

class ISubset(Interface):
    name = Str
    values = List
    str = Property(Str)
    
@provides(ISubset)
class BoolSubset(HasStrictTraits):
    name = Str
    values = List # unused
    selected_t = Bool(False)
    selected_f = Bool(False)
    
    str = Property(Str, depends_on = "name, selected_t, selected_f")
    
    def default_traits_view(self):
        return View(HGroup(Item('selected_t',
                                label = self.name + "+"), 
                           Item('selected_f',
                                label = self.name + "-")))
        
    # MAGIC: gets the value of the Property trait "str"
    def _get_str(self):
        if self.selected_t and not self.selected_f:
            return "({0} == True)".format(util.sanitize_identifier(self.name))
        elif not self.selected_t and self.selected_f:
            return "({0} == False)".format(util.sanitize_identifier(self.name))
        else:
            return ""

@provides(ISubset)
class CategorySubset(HasStrictTraits):
    name = Str
    values = List
    selected = List
    str = Property(trait = Str, depends_on = 'name, values[], selected[]')
    
    def default_traits_view(self):
        return View(Item('subset',
                         label = self.name,
                         editor = CheckListEditor(values = self.values,
                                                  cols = 2),
                         style = 'custom'))
        
    # MAGIC: gets the value of the Property trait "str"
    def _get_str(self):
        if len(self.selected) == 0:
            return ""
        
        phrase = "("
        for cat in self.selected:
            if len(phrase) > 1:
                phrase += " or "
            phrase += "{0} == \"{1}\"".format(util.sanitize_identifier(self.name), cat) 
        phrase += ")"
        
        return phrase


@provides(ISubset)
class RangeSubset(HasStrictTraits):
    name = Str
    values = List
    high = CFloat
    low = CFloat
    
    str = Property(trait = Str, depends_on = "name, values[], low, high")
    
    def default_traits_view(self):
        return View(Item('high',
                         label = self.name,
                         editor = ValuesBoundsEditor(
                                     values = self.values,
                                     low_name = 'low',
                                     high_name = 'high',
                                     auto_set = False)))
    
    def _get_str(self):
        if self.low == self.values[0] and self.high == self.values[-1]:
            return ""
        elif self.low == self.high:
            return "({0} == {1})" \
                   .format(util.sanitize_identifier(self.name), self.low)
        else:
            return "({0} >= {1} and {0} <= {2})" \
            .format(util.sanitize_identifier(self.name), self.low, self.high)
          
    
    # MAGIC: the default value for self.high
    def _high_default(self):
        if self.values:
            return max(self.values)
        else:
            return 0
    
    # MAGIC: the default value for self.low
    def _low_default(self):
        if self.values:
            return min(self.values)
        else:
            return 0
