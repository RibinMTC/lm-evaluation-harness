import abc


class Plotter(abc.ABC):

    # @abc.abstractmethod
    # def get_histograms(self, items):
    #     """Get histograms, if the task has any"""
    #     pass

    @abc.abstractmethod
    def get_plots(self):
        """Get plots, if the task has any"""
        pass
