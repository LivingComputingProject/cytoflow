#!/usr/bin/env python2.7
# coding: latin-1

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
Created on Nov 16, 2015

@author: brian
'''
import unittest

import matplotlib
matplotlib.use('Agg')

import cytoflow as flow

class Test(unittest.TestCase):


    def setUp(self):
        import os
        self.cwd = os.path.dirname(os.path.abspath(__file__))
        self.ex = flow.ImportOp(conditions = {},
                                tubes = [flow.Tube(file = self.cwd + '/data/tasbe/rby.fcs',
                                                   conditions = {})]).apply()
        
        

    def testApply(self):
        """Just run apply(); don't actually test functionality"""
        self.op = flow.BinningOp(name = "Bin",
                                 channel = "PE-Tx-Red-YG-A",
                                 num_bins = 50,
                                 bin_count_name = "Bin_Count").apply(self.ex)
                                 
        self.op = flow.BinningOp(name = "Bin",
                                 channel = "PE-Tx-Red-YG-A",
                                 bin_width = 0.1,
                                 scale = "log",
                                 bin_count_name = "Bin_Count").apply(self.ex)
                                 
        self.op = flow.BinningOp(name = "Bin",
                                 channel = "PE-Tx-Red-YG-A",
                                 num_bins = 50,
                                 scale = "logicle",
                                 bin_count_name = "Bin_Count").apply(self.ex)
        #self.op.apply(self.ex)
        
    def testView(self):
        """Just run default_view().plot(); don't actually test functionality"""
        self.op = flow.BinningOp(name = "Bin",
                                 channel = "PE-Tx-Red-YG-A",
                                 num_bins = 50,
                                 bin_count_name = "Bin_Count").default_view().plot(self.ex)
                                 
        self.op = flow.BinningOp(name = "Bin",
                                 channel = "PE-Tx-Red-YG-A",
                                 bin_width = 0.1,
                                 scale = "log",
                                 bin_count_name = "Bin_Count").default_view().plot(self.ex)
                                 
        self.op = flow.BinningOp(name = "Bin",
                                 channel = "PE-Tx-Red-YG-A",
                                 num_bins = 50,
                                 scale = "logicle",
                                 bin_count_name = "Bin_Count").default_view().plot(self.ex)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()