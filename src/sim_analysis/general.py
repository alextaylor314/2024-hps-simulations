import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

### CLASSES AND FUNCTIONS FOR SIMULATION ANALYSIS ###

# to store arrays and their summary stats
class ArrayAverage:
    def __init__(self, array, use_blocking=False):
        self.array = array.copy() # 1D array of data
        self.value = self.array.mean()
        self.sigma = self.array.std()
        self.count = len(self.array)
        if use_blocking:
            last_axis = len(array.shape)-1
            self.error = np.apply_along_axis(blocking_error, last_axis, array)
        else:
            self.error = self.sigma / np.sqrt(self.count)
    # calculate a frequency distribution using a kernel density estimation (KDE)
    def KDE(self):
        x = np.linspace(np.min(self.array), np.max(self.array), num=100)
        y = gaussian_kde(self.array, bw_method='silverman').evaluate(x)
        y /= np.sum(y)
        return x, y
    # get summary statistics as an array of strings - for printing purposes
    def get_str_array(self):
        value_string = '{:.4f}'.format(self.value)
        error_string = '{:.4f}'.format(self.error)
        sigma_string = '{:.4f}'.format(self.sigma)
        return [value_string, error_string, sigma_string]
    # calculate running average, window = 2*either_side + 1
    def running_avg(self, either_side=2):
        ra = []
        for n in range(0, len(self.array)-(2*either_side)):
            ra.append(np.sum(self.array[n:n+(2*either_side)+1]))
        padding = [np.NAN]*either_side
        return np.array(padding + ra + padding)/((2*either_side)+1)
    # plot the KDE, mean, and standard error
    def create_plot(self, ax, x_name, y_name):
        # plot KDE estimate of frequency distribution
        x, y = self.KDE()
        ax.plot(x, y)
        # set axis labels
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        # determine upper limit of y axis
        y_upper = 1.1*y.max()
        ax.set_ylim(0, y_upper)
        # plot mean and standard error
        ax.vlines(self.value, 0, y_upper)
        range_lower = self.value - self.error
        range_upper = self.value + self.error
        ax.fill_between([range_lower, range_upper], 0, y_upper, alpha=0.3)
        # needs minor ticks!
        ax.minorticks_on()

# to store best-fit quantities and their errors
class FittedQuantity:
    def __init__(self, best_fit, error):
        self.value = best_fit
        self.error = error
    # get summary statistics as an array of strings - for printing purposes
    def get_str_array(self):
        value_string = '{:.4f}'.format(self.value)
        error_string = '{:.4f}'.format(self.error)
        sigma_string = ''
        return [value_string, error_string, sigma_string]

# to store Flory scaling exponent data
class FloryData:
    def __init__(self, ij, dij, R0, nu):
        self.ij = ij
        self.dij = dij
        self.R0 = R0
        self.nu = nu
    # plot the fit used to determine the Flory exponent
    def create_plot(self, ax, x_name, y_name, fontsize=7):
        # plot the distances versus sequence separation
        ax.plot(self.ij, self.dij)
        # plot the fit
        dij_fit = self.R0.value*np.power(self.ij, self.nu.value)
        ax.plot(self.ij, self.R0.value*np.power(self.ij, self.nu.value),
                c='0.3',ls='dashed',label='Fit')
        # set axis labels
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        # add the Flory exponent as text
        ax.text(0.05, 0.9, r'$\nu$={:.2f}'.format(self.nu.value),
                horizontalalignment='left',verticalalignment='center',
                transform=ax.transAxes, fontsize=fontsize)
        # add legend
        ax.legend(loc='lower right')
        # needs minor ticks!
        ax.minorticks_on()

# to store and plot contact/energy maps and profiles
class MapProfile:
    def __init__(self, map, profile):
        self.map = map
        self.profile = profile
    def plot_contact_map(self, ax, xy_label, v_label, contact_limit=1e-3):
        # make a copy to display, and truncate to fit axes
        to_display = self.map.copy()
        vmin = contact_limit
        vmax = 1.0
        to_display = np.clip(to_display, vmin, vmax)
        # plot contact map and colour bar
        im = ax.imshow(to_display,
                       extent=[1, to_display.shape[0], 1, to_display.shape[0]],
                       origin='lower', aspect='equal', cmap=plt.cm.Blues,
                       norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax))
        c_bar = plt.colorbar(im, ax=ax, label=v_label, fraction=0.05, pad=0.04)
        # set axis labels
        ax.set_xlabel(xy_label)
        ax.set_ylabel(xy_label)
    def plot_contact_profile(self, ax, x_label, y_label):
        # get residue indices
        res_indices = np.array(range(1, len(self.profile.array)+1))
        # set  axis ranges
        nan_removed = self.profile.array[~np.isnan(self.profile.array)]
        ax.set_xlim(res_indices[0]-1, res_indices[-1]+1)
        ax.set_ylim(0, 1.1*nan_removed.max())
        # plot raw contact profile as points
        ax.scatter(res_indices, self.profile.array, s=10, marker='x')
        # plot running average as a line
        ax.plot(res_indices, self.profile.running_avg(), color='limegreen',
                linewidth=1.2)
        # set axis labels
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        # needs minor ticks!
        ax.minorticks_on()
    def plot_energy_map(self, ax, xy_label, v_label):
        # convert from kJ/mol to J/mol
        to_display = 1e3*self.map
        # truncate to fit axes
        vmin = -1000
        vmax = 1000
        to_display[to_display < vmin] = vmin
        to_display[to_display > vmax] = vmax
        # plot energy map and colour bar
        im = ax.imshow(to_display,
                       extent=[1, to_display.shape[0], 1, to_display.shape[0]],
                       origin='lower', aspect='equal', cmap=plt.cm.RdBu_r,
                       norm=mpl.colors.SymLogNorm(linthresh=1,
                                                  vmin=vmin, vmax=vmax))
        c_bar = plt.colorbar(im, ax=ax, label=v_label, fraction=0.05, pad=0.04)
        # set axis labels
        ax.set_xlabel(xy_label)
        ax.set_ylabel(xy_label)
    def plot_energy_profile(self, ax, x_label, y_label):
        # cache variables; do not convert units
        ep = self.profile.array
        ra = self.profile.running_avg()
        # get residue indices
        res_indices = np.array(range(1, len(ep)+1))
        # set  axis ranges
        y_min = ep[~np.isnan(ep)].min()
        y_max = ep[~np.isnan(ep)].max()
        y_range = y_max-y_min
        padding = 0.1*y_range
        ax.set_xlim(res_indices[0]-1, res_indices[-1]+1)
        ax.set_ylim(min(y_min-padding, 0), max(y_max+padding, 0))
        # add horizontal line at zero
        ax.axhline(0, color='tab:gray', linewidth=1.2)
        # plot raw energy profile as points
        ax.scatter(res_indices, ep, s=10, marker='x')
        # plot running average as a line
        ax.plot(res_indices, ra, color='limegreen', linewidth=1.2)
        # set axis labels
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        # needs minor ticks!
        ax.minorticks_on()
