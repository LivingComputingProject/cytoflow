if __name__ == '__main__':
    from traits.etsconfig.api import ETSConfig
    ETSConfig.toolkit = 'qt4'

    import os
    os.environ['TRAITS_DEBUG'] = "1"

from traits.api import HasTraits, Str, provides
import matplotlib.pyplot as plt
from cytoflow.views.i_view import IView
from cytoflow.views.sns_axisgrid import FacetGrid

@provides(IView)
class HistogramView(HasTraits):
    """Plots a one-channel histogram
    
    Attributes
    ----------
    name : Str
        The HistogramView name (for serialization, UI etc.)
    
    channel : Str
        the name of the channel we're plotting
    
    xfacet : Str 
        the conditioning variable for multiple plots (horizontal)
    
    yfacet : Str
        the conditioning variable for multiple plots (vertical)
    
    huefacet : Str
        the conditioning variable for multiple plots (color)
        
    subset : Str
        a string passed to pandas.DataFrame.query() to subset the data before 
        we plot it.
        
        .. note: Should this be a param instead?
    """
    
    # traits   
    id = "edu.mit.cytoflow.view.histogram"
    friendly_id = "Histogram" 
    
    name = Str
    channel = Str
    xfacet = Str
    yfacet = Str
    huefacet = Str
    subset = Str
    
    def plot(self, experiment, fig_num = None, **kwargs):
        """Plot a faceted histogram view of a channel"""
        
        kwargs.setdefault('histtype', 'stepfilled')
        kwargs.setdefault('alpha', 0.5)
        kwargs.setdefault('bins', 200) # Do not move above
        
        if not self.subset:
            x = experiment.data
        else:
            x = experiment.query(self.subset)

        g = FacetGrid(x, 
                      col = (self.xfacet if self.xfacet else None),
                      row = (self.yfacet if self.yfacet else None),
                      hue = (self.huefacet if self.huefacet else None),
                      fig_kws={"num" : fig_num})
        
        # TODO - compute and specify the bin width!
        g.map(plt.hist, self.channel, **kwargs)
        
        
    def validate(self, experiment):
        """Validate this view against an experiment."""
        if not experiment:
            return False
        
        if self.channel not in experiment.channels:
            return False
        
        if self.xfacet and self.xfacet not in experiment.metadata:
            return False
        
        if self.yfacet and self.yfacet not in experiment.metadata:
            return False
        
        if self.huefacet and self.huefacet not in experiment.metadata:
            return False
        
        if self.subset:
            try:
                experiment.query(self.subset)
            except:
                return False
        
        return True
    
if __name__ == '__main__':
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    
    plt.ioff()
    p = plt.figure(1)
    
    import seaborn as sns
    tips = sns.load_dataset("tips")
    g = FacetGrid(tips, col="time", fig_kws={"num" : 1})
    
    plt.show()