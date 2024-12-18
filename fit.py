"""
Classes and functions for performing fits an calculating limit contours
"""

from .fmath import *
from scipy.stats import chi2, norm
from scipy.signal import savgol_filter




def TwoSidedGridSearch(generator, observations, param_grid, background=None, target_cl=0.90,
                        ddof=0, verbose=False, delta_chi2=True):
    # This is a brute-force grid search for the upper and lower limit CLs.
    lower_cl = param_grid[0]
    upper_cl = param_grid[-1]

    def statistic(param):
        if background is not None:
            return np.sum(power(generator(param) + background - observations, 2)/background)
        elif background is not None and delta_chi2 == False:
            return chisquare(generator(param) + background, observations, ddof)[0]
        else:
            return np.sum(generator(param))

    stop_value = chi2.ppf(target_cl, observations.shape[0]) - observations.shape[0] \
        if background is not None else 2.3
    if delta_chi2:
        stop_value = 2.71

    # lower bound
    if verbose:
        print("getting LOWER BOUND...")
    for i, x in enumerate(param_grid):
        stat = statistic(x)
        if stat > stop_value:
            lower_cl = x
            if verbose:
                print("chi2 = ", stat)
                print("found lower cl = ", lower_cl)
            break
    
    # upper bound
    if verbose:
        print("getting UPPER BOUND...")
    for i, x in enumerate(param_grid[::-1]):
        stat = statistic(x)
        if stat > stop_value:
            upper_cl = x
            if verbose:
                print("chi2 = ", stat)
                print("found upper cl = ", upper_cl)
            break

    if lower_cl == param_grid[0] and upper_cl == param_grid[-1]:
        lower_cl = None
        upper_cl = None
    return lower_cl, upper_cl




def binary_search(test_function, stop_value, lower_edge, upper_edge, tolerance=0.1,
                    is_increasing=True, verbose=False):
    # test_function(x): single variable test statistic function of parameter x
    # stop_value: the target value of test_function(x_final)
    # tolerance: the diference |test_function(x) - test_function(x_final)| allowed
    # upper_edge and lower_edge: the range of x values to test
    # is_increasing: True if test_function is expected to increase with x, False otherwise
    # test_function is assumed to be monotonic over the range x ~ [lower_edge, upper_edge]
    if verbose:
        print("starting binary search")
    x_upper = upper_edge
    x_lower = lower_edge
    x = x_lower
    test = test_function(x)
    while abs(x_upper - x_lower) > tolerance*abs(upper_edge - lower_edge):
        test = test_function(x)
        if verbose:
            print("trying x, test_func = ", x, test)
        if test < stop_value:
            if is_increasing:
                x_lower = x
                x = (x_upper + x_lower)/2
            else:
                x_upper = x
                x = (x_upper + x_lower)/2
        else:
            if is_increasing:
                x_upper = x
                x = (x_upper + x_lower)/2
            else:
                x_lower = x
                x = (x_upper + x_lower)/2
        
        if x == x_lower or x == x_upper:
            if verbose:
                print("Ran into edge of search window, exiting")
            return x
    return x




def cleanLimitData(masses, lower_limits, upper_limits, apply_smoothing=False):
        diff_upper_lower = upper_limits - lower_limits
        upper_limits = np.delete(upper_limits, np.where(diff_upper_lower < 0))
        lower_limits = np.delete(lower_limits, np.where(diff_upper_lower < 0))
        masses = np.delete(masses, np.where(diff_upper_lower < 0))

        # Apply a savgol filter
        if apply_smoothing:
            lower_limits = savgol_filter(lower_limits, 3, 0)

        joined_limits = np.append(lower_limits, upper_limits[::-1])
        joined_masses = np.append(masses, masses[::-1])

        # Extend to the zero mass limit
        joined_masses = np.append(0.0, joined_masses)
        joined_limits = np.append(joined_limits[0], joined_limits)
        return joined_masses, joined_limits




class ChiSquareRandomizedSearch:
    """
    Perform a randomized search for the upper and lower confidence limits (CL's)
    inside of a specified interval param_range. Must pass in a signal model (signal_generator),
    which is a function of a single model parameter (theta), a set of observations, and an
    optional set of backgrounds. signal_generator, observations, and background must all have
    the same array shape.
    
    Options:
    target_cl: the target CL in decimal form
    tolerance: the percentage tolerance away from the chi2 value associated with the target_cl
    max_points: the stopping criterion; if we test more than max_points and haven't found
                any limits, exit the search and return the peak value of the chi2 at the last point
    """
    def __init__(self, signal_generator, observations, background=None, param_range=(0,1,),
                    target_cl=0.9, ddof=1, tolerance=0.05, max_points=100):
        self.range = param_range
        self.tol = tolerance
        self.signal = signal_generator
        self.obs = observations
        self.bkg = background
        self.ddof = ddof
        self.cl = target_cl
        self.max_points = max_points
        self.target_chi2 = chi2.ppf(target_cl, observations.shape[0] - ddof) - observations.shape[0] if background is not None else 2.3

        # Chi2 map
        self.chisq_list = []
        self.param_list = []

        self.lower_cl = None
        self.upper_cl = None
    
    def test_stat(self, theta):
        # Takes in model parameter theta
        if self.bkg is not None:
            return chisquare(self.signal(theta) + self.bkg, self.obs, self.ddof)[0]
        else:
            return np.sum(self.signal(theta))
    
    def update_search_window(self):
        sorted_chi2_map = self.get_sorted_chisq_dist()
        chi2_vals = sorted_chi2_map[:,1]
        param_vals = sorted_chi2_map[:,0]

        max_chi2_id = np.argmax(chi2_vals)
        max_param = param_vals[max_chi2_id]
        if max_param != param_vals[0] and max_param != param_vals[-1]:
            return param_vals[max_chi2_id-1], param_vals[max_chi2_id+1]
        else:
            return param_vals[0], param_vals[-1]

    
    def run_search(self, verbose=False):
        # Draw random variates over parameter range

        # Check test statistic until we find a high point in between two lower points
        # The max must be constrained there

        theta_lower = self.range[0]
        theta_upper = self.range[1]
        theta_mid = (theta_upper + theta_lower)/2
        theta_mid_upper = (theta_mid + theta_upper)/2
        theta_mid_lower = (theta_mid + theta_lower)/2

        candidates = [theta_lower, theta_mid_lower, theta_mid, theta_mid_upper, theta_upper]
        middle_ctrl_pts = []
        outer_ctrl_pts = []

        if verbose:
            print("Checking candidates: ", candidates)

        # loop over initial candidates
        # there is a small chance that we land within tolerance of a target CL here,
        # but we haven't differentiated if it is the lower or upper CL, so don't bother.
        for theta in candidates:
            self.param_list.append(theta)
            stat = self.test_stat(theta)
            self.chisq_list.append(stat)

            if stat > self.target_chi2:
                middle_ctrl_pts.append(theta)
            else:
                outer_ctrl_pts.append(theta)
        
        if verbose:
            print("Finished candidates. beginning control pt. search...")
        
        # Begin randomized search in window (lower_edge, upper_edge)
        upper_edge = self.range[1]
        lower_edge = self.range[0]
        # We need at least 3 points: one point in the middle of the chisquare dist.
        # between the two CL levels, and two points outside
        trial_counter = 0
        while len(middle_ctrl_pts) < 1:
            trial_counter += 1
            if trial_counter > self.max_points:
                return (upper_edge + lower_edge)/2, (upper_edge + lower_edge)/2
            if (upper_edge - lower_edge)/(self.range[1] - self.range[0]) < self.tol:
                if verbose:
                    print("Search window too narrow before finding target CL; exiting")
                return None, None #(upper_edge + lower_edge)/2, (upper_edge + lower_edge)/2
            if verbose:
                print("Checking in range ", lower_edge, upper_edge)

            theta_rnd = np.random.uniform(lower_edge, upper_edge)
            stat = self.test_stat(theta_rnd)

            if verbose:
                print("param = ", theta_rnd, "has test stat. = ", stat)

            self.param_list.append(theta_rnd)
            self.chisq_list.append(stat)

            if stat > self.target_chi2:
                middle_ctrl_pts.append(theta_rnd)
            else:
                outer_ctrl_pts.append(theta_rnd)
            

            lower_edge, upper_edge = self.update_search_window()

            # If we have a middle control point, use it to differentiate 
            # where the outer control points lie and update the search window
            if len(middle_ctrl_pts) >= 1:
                # Check if we find any chi2 within tolerance of the CL targets
                if abs(stat - self.target_chi2)/self.target_chi2 < self.tol:
                    if theta_rnd <= middle_ctrl_pts[0]:
                        self.lower_cl = theta_rnd
                    else:
                        self.upper_cl = theta_rnd

            if self.lower_cl is not None and self.upper_cl is not None:
                if verbose:
                    print("Found both CLs! No binary search needed.")
                break

        if verbose:
            print("Found control pts. Running binary search to find the upper and lower CLs.")
        theta_lower = min(outer_ctrl_pts)
        theta_upper = max(outer_ctrl_pts)
        theta_mid_lower = min(middle_ctrl_pts)
        theta_mid_upper = max(middle_ctrl_pts)

        # If not yet within tolerance of either upper or lower cl,
        # run binary searches using the control points as constraints
        if self.lower_cl is None:
            self.lower_cl = binary_search(self.test_stat, self.target_chi2, theta_lower, theta_mid_lower,
                                            tolerance=self.tol, is_increasing=True, verbose=verbose)
            self.param_list.append(self.lower_cl)
            self.chisq_list.append(self.test_stat(self.lower_cl))
        
        if self.upper_cl is None:
            self.upper_cl = binary_search(self.test_stat, self.target_chi2, theta_mid_upper, theta_upper,
                                            tolerance=self.tol, is_increasing=False, verbose=verbose)
            self.param_list.append(self.upper_cl)
            self.chisq_list.append(self.test_stat(self.upper_cl))
        
        if verbose:
            print("found upper CL and lower CL at chi2 = ", self.chisq_list[-1], self.chisq_list[-2])
        
        return self.lower_cl, self.upper_cl
    
    def get_sorted_chisq_dist(self):
        chisq_param_pairs = np.array([self.param_list, self.chisq_list]).transpose()
        return chisq_param_pairs[chisq_param_pairs[:,0].argsort()]




class PseudoExperiment:
    """
    Class for running pseudoexperiments given a set of observations 'obs', assuming
    that the errors on the observations are sqrt(obs) and are normally distributed.
    """
    def __init__(self, expectations, ddof=1, data_name="pe_output.dat"):
        self.exp_values = np.array(expectations)
        self.ddof = ddof
        self.dat_loc = data_name
        self.chi2_values = []

    def draw_variates(self):
        # Given the array of expectation values and errors, draw normally dist. variates
        n_bins = len(self.exp_values)
        u = np.random.uniform(0,1,n_bins)
        obs = norm.ppf(u, loc=self.exp_values, scale=np.sqrt(self.exp_values))
        return np.round(obs, 2)

    def run(self, n_experiments=10000):
        # draw_variates n_samples times
        outfile = open(self.dat_loc, "w")
        for i in range(n_experiments):
            obs_i = self.draw_variates()
            data_str = ' '.join([str(obs) for obs in obs_i]) + "\n"
            outfile.write(data_str)

        outfile.close()

    def get_chi2_dist(self):
        # Get the chi2 distribution for the generated samples
        self.chi2_values = []
        readfile = np.genfromtxt(self.dat_loc)
        n_dof = len(self.exp_values) - self.ddof
        for i in range(readfile.shape[0]):
            obs_i = readfile[i,:]
            self.chi2_values.append(chisquare(obs_i, self.exp_values, n_dof)[0])
        return np.array(self.chi2_values)
    
    def get_chi2_median(self):
        return np.median(self.chi2_values)

    def get_chi2_pvalue(self, p):
        chi2_sorted = self.get_chi2_dist()
        chi2_hist = np.histogram(chi2_sorted, bins=100)
        cdf = np.cumsum(chi2_hist[0])/len(chi2_sorted)
        absolute_val_array = np.abs(cdf - p)
        smallest_difference_index = absolute_val_array.argmin()
        return chi2_hist[1][smallest_difference_index]



