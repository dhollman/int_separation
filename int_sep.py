from __future__ import division
from collections import defaultdict
import math
from types import MethodType
from scipy.optimize import curve_fit
import numpy as np
from itertools import product
import psi4
from psi4 import Gaussian94BasisSetParser, BasisSet, clean as psi_clean, MintsHelper
from scipy.special import erfcinv
from grendel.gmath import Tensor, Vector
import matplotlib.pyplot as plt

xproduct = lambda *args: product(*[xrange(i) for i in args])

class IntDataSet(object):
    """
    All of the computed integrals (IntData) associated with a given set of centers.
    Used mostly to avoid recreating psi4.Molecule objects
    """

    ####################
    # Class Attributes #
    ####################

    parser = Gaussian94BasisSetParser()

    ##################
    # Initialization #
    ##################

    def __init__(self,
            centers,
            basis_name
    ):
        """
        """
        # Create the molecule object that we need
        mol_string = ""
        self.center_numbers = []
        self.centers = []
        cent_num = -1
        for i, center in enumerate(centers):
            if center not in self.centers[:i]:
                self.centers.append(Vector(center))
                mol_string += "H  {}  {}  {}\n".format(center[0], center[1], center[2])
                cent_num += 1
                self.center_numbers.append(cent_num)
            else:
                self.center_numbers.append(self.centers.index(center))
        self.molecule = psi4.geometry("symmetry c1\n" + mol_string.strip())
        #----------------------------------------#
        # Create the BasisSet objects needed
        self.basis_name = basis_name
        psi4.options.basis = self.basis_name
        self.basis = BasisSet.construct(IntDataSet.parser, self.molecule, 'BASIS')
        #----------------------------------------#
        # Create the MintsHelper and the factories
        self.mints = MintsHelper(self.basis)
        self.factory = self.mints.integral()
        #----------------------------------------#
        # Create the list of IntData objects
        self.int_data = {}

    def int_for(self, i, j, k, l, centers=(0,1,2,3)):
        key = (i,j,k,l,centers)
        if key in self.int_data:
            return self.int_data[key]
        else:
            center_nums = [self.center_numbers[x] for x in centers]
            rv = IntData(
                basis=self.basis,
                factory=self.factory,
                center_nums=center_nums,
                shells=[self.basis.shell(icent, ish) for icent, ish in zip(center_nums, key)],
                parent_set=self
            )
            self.int_data[key] = rv
            return rv

#--------------------------------------------------------------------------------#

class IntData(object):

    ####################
    # Class Attributes #
    ####################

    # Map of integral type to chunk size
    # TODO derivative integrals when needed
    oei_types = {
        "ao_overlap" : 1,
        "ao_kinetic" : 1,
        "ao_potential" : 1,
        "ao_pseudospectral" : 1,
        "ao_dipole" : 3,
        "ao_quadrupole" : 6,
        "ao_angular_momentum" : 3,
        "ao_nabla" : 3
    }
    tei_types = [
        "eri",
        "f12",
        "f12g12",
        "f12_double_commutator",
        "f12_squared"
    ]

    ##################
    # Initialization #
    ##################

    def __init__(self,
            basis,
            factory,
            center_nums,
            shells,
            parent_set = None,
            cf_exp=1.0
    ):
        """
        """
        self.basis = basis
        self.factory = factory
        self.center_nums = center_nums
        self.cf = psi4.FittedSlaterCorrelationFactor(cf_exp)
        self.parent_set = parent_set
        #----------------------------------------#
        # Shell related arrays
        self.shells = shells
        self.shell_sizes = tuple(sh.nfunction for sh in self.shells)
        self.shell_offsets = tuple(sh.function_index for sh in self.shells)
        self.shell_slices = tuple(slice(off, off + sz) for sz, off in zip(self.shell_sizes, self.shell_offsets))
        self.shell_indices = map(basis.function_to_shell, [sh.function_index for sh in shells])
        self.shell_nprimative = tuple(sh.nprimative for sh in self.shells)
        #----------------------------------------#
        # Get the TwoBodyIntegral objects that compute the integrals
        for tei_name in IntData.tei_types:
            # get the TwoBodyAOInt object
            factory_fxn = getattr(self.factory, tei_name)
            if "f12" in tei_name:
                setattr(self, tei_name + "_int", lambda fxn=factory_fxn: fxn(self.cf))
            else:
                setattr(self, tei_name + "_int", factory_fxn)
            # set the cached version to None
            setattr(self, "_" + tei_name, None)
        # Get the OneBodyIntegral objects that compute integrals
        for oei_name in IntData.oei_types:
            # get the OneBodyAOInt object
            factory_fxn = getattr(self.factory, oei_name)
            setattr(self, oei_name + "_int", factory_fxn)
            # set the cached version to None
            setattr(self, "_" + oei_name, {})

    def __getattr__(self, item):
        if "ao_" + item in IntData.oei_types:
            item = "ao_" + item
        if item in IntData.tei_types:
            rv = getattr(self, "_" + item)
            if rv is None:
                # Get the integral object
                int_obj = getattr(self, item + "_int")
                if callable(int_obj):
                    int_obj = int_obj()
                    setattr(self, item + "_int", int_obj)
                # Compute the shell
                int_obj.compute_shell(*self.shell_indices)
                # Cache the value
                rv = Tensor(int_obj.py_buffer).reshape(self.shell_sizes)
                setattr(self, "_" + item, rv)
            return rv
        elif item in IntData.oei_types:
            def _oei_getter(self, c1, c2, item=item):
                known = getattr(self, "_" + item)
                if (c1, c2) in known:
                    return known[(c1, c2)]
                else:
                    int_obj = getattr(self, item + "_int")
                    if callable(int_obj):
                        int_obj = int_obj()
                        setattr(self, item + "_int", int_obj)
                    int_obj.compute_shell(self.shell_indices[c1], self.shell_indices[c2])
                    nchunk = IntData.oei_types[item]
                    if nchunk == 1:
                        rv = Tensor(int_obj.py_buffer).reshape((self.shell_sizes[c1], self.shell_sizes[c2]))
                    else:
                        rv = Tensor(int_obj.py_buffer).reshape((self.shell_sizes[c1], self.shell_sizes[c2], nchunk))
                    known[(c1,c2)] = rv
                return rv
            return MethodType(_oei_getter, self)
        else:
            raise AttributeError, item


    def pair_extent(self, cnum1, cnum2, thresh=1e-8):
        if not self.shell_nprimative[cnum1] == 0 and self.shell_nprimative[cnum2] == 0:
            raise NotImplementedError("Contracted extents not implemented")
        exp1 = self.shells[cnum1].exp(0)
        exp2 = self.shells[cnum2].exp(0)
        return math.sqrt(2/(exp1+exp2))*erfcinv(thresh)

    def get_ints(self, int_type):
        return getattr(self, int_type)

#--------------------------------------------------------------------------------#

def tei_decays(
        ijkl_sets,
        geoms,
        basis_name,
        xpoints,
        int_type="eri",
        normalize=True,
        multi_func=None
):
    if multi_func is None:
        multi_func = lambda x: float(np.amax(abs(x)))

    xpts = []
    int_sets = []
    for geom in geoms:
        int_sets.append(IntDataSet(geom, basis_name))
        if callable(xpoints):
            xpts.append(xpoints(int_sets[-1]))
    if not callable(xpoints):
        xpts = xpoints

    rv = {}
    for i, j, k, l in ijkl_sets:
        decay = []
        print "Computing decay for " + str((i,j,k,l)) + "..."
        for int_set in int_sets:
            decay.append(multi_func(int_set.int_for(i, j, k, l).get_ints(int_type)))
        # Sort decay by xpoints
        x, decay = zip(*sorted(zip(xpts,decay)))
        decay = Vector(decay)
        if normalize:
            if decay[0] == 0:
                decay[...] = 0
                if any(d > 0 for d in decay[1:]):
                    raise NotImplementedError()
            else:
                decay /= decay[0]
        rv[(i,j,k,l)] = (x, decay)
    return rv

#--------------------------------------------------------------------------------#

def one_over_r(xdata, a, b):
    return a / (xdata - b)**3

def tei_decay_plot(
        decay_vals,
        name_template="({}a {}a | {}b {}b) decay",
        fit_function=one_over_r,
        p0=(0.3,0.3),
        fit_offset=3.0,
        **kwargs
):
    colors = "rgbcmyk"
    color_idx = 0
    ax = plt.axes()
    ax.set_yscale('log')
    for i, j, k, l in sorted(decay_vals.keys()):
        print "Computing decay fit for " + str((i,j,k,l)) + "..."
        x, decay = decay_vals[(i,j,k,l)]
        # fit the curve to f(r) = a / (r-b)
        fit_x, fit_y = zip(*[(xx, yy) for xx, yy in zip(x, decay) if xx >= fit_offset])
        popt, _ = curve_fit(
            fit_function,
            fit_x,
            fit_y,
            p0=p0,
            maxfev=6000
        )
        print "  parameters: {}".format(popt)
        # plot the fit
        line_pts = Vector(range(100))*((max(x)-min(x))/99) + min(x)
        plt.plot(
            list(line_pts),
            list(fit_function(line_pts, *popt)),
            colors[color_idx%len(colors)] + "-",
            lw=2,
            **kwargs
        )
        # now plot the data points
        print "Plotting " + str((i,j,k,l)) + "..."
        name = name_template.format(i,j,k,l)
        plt.plot(
            list(x),
            list(abs(decay)),
            colors[color_idx % len(colors)] + ":",
            marker="o",
            label=name,
            **kwargs
        )
        color_idx += 1

#--------------------------------------------------------------------------------#

def figure_1(
        xmin=1,
        xmax=8,
        npts=28,
        shells=((i,i,i,i) for i in range(15)),
):
    xpoints = Vector(range(npts)) * ((xmax-xmin)/npts) + xmin
    decay_data = tei_decays(
        shells,
        geoms=([
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, r],
                   [0, 0, r]
               ] for r in xpoints
        ),
        basis_name = "cc-pV5Z",
        xpoints=xpoints,
        int_type="eri",
    )

    plt.ylim(0,1)
    tei_decay_plot(
        decay_data
    )
    plt.legend()
    plt.show()

#--------------------------------------------------------------------------------#

def figure_2(
        xmin=1,
        xmax=8,
        npts=28,
        shell_numbers=(1,3,6,14)
):
    xpoints = Vector(range(npts)) * ((xmax-xmin)/npts) + xmin
    decay_data = tei_decays(
        ((i,i,i,i) for i in shell_numbers),
        geoms=([
                   [0, 0, 0],
                   [0, 0, r],
                   [0, 0, 0],
                   [0, 0, r]
               ] for r in xpoints
        ),
        basis_name = "cc-pV5Z",
        xpoints=xpoints,
        int_type="eri"
        )

    def negexp(xvals, a, b):
        return a * np.exp(-b * np.array(xvals)**2)
    axes = plt.axes()
    axes.set_yscale("log")
    plt.ylim(0,1)
    tei_decay_plot(
        decay_data,
        fit_function=negexp,
        name_template="({}a {}b | {}a {}b) decay",
        p0=(1.0, 1.0),
        fit_offset=2.0
    )
    plt.legend()
    plt.show()

#--------------------------------------------------------------------------------#

def figure_3(
        shell_quartets=(
            #(4,4,4,4),
            #(5,5,5,5),
            #(4,4,5,5),
            (6,6,6,6),
            (8,8,8,8),
            #(4,4,14,14),
            (14,14,14,14)
        ),
        rABmin = 3.5,
        rABmax = 10,
        rABnpts = 10,
        r34min = 0,
        r34max = 7,
        r34npts = 10,
        r12 = 0.0,
        max_func = lambda x: float(np.amax(abs(x)))
):
    colors = "rgbcmyk"
    color_idx = 0
    rABpts = Vector(range(rABnpts)) * (rABmax-rABmin)/rABnpts + rABmin
    r34pts = Vector(range(r34npts)) * (r34max-r34min)/r34npts + r34min
    xvals = defaultdict(lambda: [])
    yvals = defaultdict(lambda: [])
    rABr34 = defaultdict(lambda: [])

    scr_failures = 0
    order_failures = 0
    axes = plt.axes()
    #axes.set_xscale("log")
    for rAB in rABpts:
        for r34 in r34pts:
            curr_set = IntDataSet(
                centers=[
                    [-r12/2, 0, 0],
                    [r12/2, 0, 0],
                    [-r34/2, rAB, 0],
                    [r34/2, rAB, 0]
                ],
                basis_name="cc-pV5Z"
            )

            for q in shell_quartets:
                if not q[0] == q[1] and q[2] == q[3]:
                    raise NotImplementedError
                qint = curr_set.int_for(*q)
                val = max_func(qint.get_ints("eri"))
                est = np.amax(qint.overlap(0,1)) * np.amax(qint.overlap(2,3)) / rAB
                if abs(val) - abs(est) > 1e-7:
                    scr_failures += 1
                if abs(est) > 1e-7:
                    rABr34[q].append((rAB, r34))
                    xvals[q].append(
                        #rAB + r34
                        est
                    )
                    yvals[q].append(val)
    for q in shell_quartets:
        decay = yvals[q]
        x, decay, rab34 = zip(*sorted(zip(xvals[q], decay, rABr34[q])))
        plt.plot(
            list(x),
            list(decay),
            colors[color_idx % len(colors)] + "-",
            marker="o",
            label="({}a {}b | {}c {}d)".format(*q)
        )
        for i in range(1, len(x)):
            xpt, ypt = x[i], decay[i]
            xprev, yprev = x[i-1], decay[i-1]
            if yprev - ypt > 1e-7:
                order_failures += 1
                #plt.annotate(
                #    "R_AB = {}, R_34 = {}".format(rab34[i][0], rab34[i][1]),
                #    (xpt, ypt),
                #    #xytext=(5,-5)
                #)
        color_idx += 1
    plt.legend(loc='best')
    print "Ordering failures: {}\nScreening failures: {}".format(order_failures, scr_failures)
    plt.show()

#--------------------------------------------------------------------------------#

# Problem cases: (4,4,4,4), rAB = (1.5,15,12), r34 = (0,10,15), r12 = 0.0

def figure_4(
        shell_quartets=(
            #(4,4,4,4),
            #(2,2,2,2),
            #(5,5,5,5),
            #(4,4,5,5),
            #(6,6,6,6),
            #(8,8,8,8),
            #(4,4,14,14),
            (4,4,3,3),
            #(14,14,14,14),
        ),
        rABmin = 0,
        rABmax = 20,
        rABnpts = 12,
        r34min = 1,
        r34max = 10,
        r34npts = 10,
        r12 = 2.0,
        max_func = lambda x: float(np.amax(abs(x)))
):
    colors = "rgbcmy"
    color_idx = 0
    rABpts = Vector(range(rABnpts)) * (rABmax-rABmin)/rABnpts + rABmin
    r34pts = Vector(range(r34npts)) * (r34max-r34min)/r34npts + r34min
    xvals = defaultdict(lambda: [])
    yvals = defaultdict(lambda: [])
    rABr34 = defaultdict(lambda: [])
    yest = defaultdict(lambda: [])
    err = defaultdict(lambda: [])

    fig = plt.figure(4)
    ax = fig.add_subplot(111)
    min_x = float("inf")
    max_x = -float("inf")
    for rAB in rABpts:
        for r34 in r34pts:
            curr_set = IntDataSet(
                centers=[
                    [-r12/2, 0, 0],
                    [r12/2, 0, 0],
                    [-r34/2, rAB, 0],
                    [r34/2, rAB, 0]
                ],
                basis_name="cc-pV5Z"
            )

            theta = 1e-3
            for q in shell_quartets:
                if not q[0] == q[1] and q[2] == q[3]:
                    raise NotImplementedError
                qint = curr_set.int_for(*q)
                print qint.pair_extent(0,1,theta)
                print qint.pair_extent(2,3,theta)
                q1 = np.amax(curr_set.int_for(q[0],q[1],q[0],q[1]).get_ints("eri"))
                q2 = np.amax(curr_set.int_for(q[2],q[3],q[2],q[3]).get_ints("eri"))
                val = max_func(qint.get_ints("eri"))
                s34 = np.amax(qint.overlap(2,3))
                est = math.sqrt(q1 * q2) / max(rAB*1.889726 - qint.pair_extent(0,1,theta) - qint.pair_extent(2,3,theta), 1)
                #est = q1 * q2 / rAB
                #est = np.amax(qint.overlap(0,1)) * np.amax(qint.overlap(2,3)) / (
                #        rAB - qint.pair_extent(0,1,theta) - qint.pair_extent(2,3,theta)
                #)
                # if well-separated and significant
                if abs(val) > 1e-12: #and rAB*1.889726 - qint.pair_extent(0,1,theta) - qint.pair_extent(2,3,theta) > 0:
                    rABr34[q].append((rAB, r34))
                    xvals[q].append(est)
                    yvals[q].append(val)
                    yest[q].append(est)
                    err[q].append(abs(est-val))
    for q in shell_quartets:
        decay = yvals[q]
        x, decay, ests, rab34, er = zip(*sorted(zip(xvals[q], decay, yest[q], rABr34[q], err[q])))
        xvals[q], yvals[q], yest[q], rABr34[q], err[q] = x, decay, ests, rab34, er
        tmp = min(x)
        if tmp < min_x: min_x = tmp
        tmp = max(x)
        if tmp > max_x: max_x = tmp
        plt.plot(
            list(x),
            list(decay),
            colors[color_idx % len(colors)] + "-",
            marker="o",
            label="({}a {}b | {}c {}d)".format(*q),
            picker=5
        )
        plt.plot(
            list(x),
            list(er),
            colors[color_idx % len(colors)] + ":",
            marker="*",
            label="estimate error".format(*q),
            picker=5
        )
        color_idx += 1

    def onpick(event):
        idx = event.ind
        for q in shell_quartets:
            if len(shell_quartets) > 1:
                print 'q = {}, x = {}, y = {}, rAB = {}, r34 = {}'.format(
                    q,  xvals[q][idx],  yvals[q][idx],
                    rABr34[q][idx][0], rABr34[q][idx][1]
                )
            else:
                if len(idx) > 1:
                    print idx
                for i in idx:
                    print '{}idx= {}, x = {}, y = {}, rAB = {}, r34 = {}'.format(
                        "    " if len(idx) > 1 else "",
                        i, xvals[q][i],  yvals[q][i],
                        rABr34[q][i][0], rABr34[q][i][1]
                    )

    #plot the estimates
    #ax.set_xscale('log')
    ax.grid(True)
    ax.plot(
        list(Vector(range(100))* (max_x - min_x)/100 + min_x),
        list(Vector(range(100))* (max_x - min_x)/100 + min_x),
        "k-",
        lw=3
    )

    fig.canvas.mpl_connect('pick_event', onpick)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc='best')
    plt.show()

#figure_4()

def check_oei_tensor():

    intset = IntDataSet([
            [0,0,0],
            [0,1,0],
            [0,1,1],
            [1,1,1]
        ],
        "cc-pVDZ"
    )

    nao = intset.basis.nao()
    nsh = intset.basis.nshell()
    Spsi = intset.mints.ao_overlap()
    S = Tensor(shape=(nao, nao))
    for i, j in xproduct(nao, nao):
        S[i,j] = Spsi[0, i, j]

    for c1, c2 in xproduct(4,4):
        for sh1, sh2 in xproduct(int(nsh/4), int(nsh/4)):
            if c1 == c2 and sh1 != sh2:
                continue
            shnums = [0]*4
            shnums[c1] = sh1
            shnums[c2] = sh2
            intcomp = intset.int_for(*shnums)
            block = intcomp.overlap(c1,c2)
            sl1 = intcomp.shell_slices[c1]
            sl2 = intcomp.shell_slices[c2]
            if float(np.amax(abs(S[sl1,sl2]-block))) > 1e-10:
                print str((c1,c2)) + ", shells " + str((sh1,sh2)) + " => slices " + str((sl1, sl2)) \
                      + ", comp = " +  str(block) \
                      + ", should be = " +  str(S[sl1,sl2]) \
                      + ", difference = " +  str(float(np.amax(abs(S[sl1,sl2]-block))))

def figure_5(
        xmin=0,
        xmax=3,
        npts=50,
        #shell_numbers=range(14,15)
        shell_numbers=[4,8,9,10,11,14]

):
    fig = plt.figure("Overlap Decay")
    ax = fig.add_subplot(111)
    xpoints = np.array(range(npts)) * ((xmax-xmin)/npts) + xmin
    decay = defaultdict(lambda: [])
    for r in xpoints:
        iset = IntDataSet(
            centers=[
                (0, 0, 0),
                (0, 0, 0),
                (0, 0, r),
                (0, 0, r),
            ],
            basis_name="cc-pV5Z"
        )
        for sh in shell_numbers:
            decay[sh].append(np.amax(iset.int_for(sh, sh, sh, sh).overlap(0,2)))

    for sh in shell_numbers:
        dec = np.array(decay[sh])/decay[sh][0]
        ax.plot(list(xpoints), list(dec), marker="o", label="S_{},{} decay".format(sh,sh))
    ax.legend(loc='best')
    plt.show()

def figure_6(
        quartet=(8,8,8,8),
        rABmin = 5.5,
        rABmax = 10,
        rABnpts = 10,
        r34min = 0,
        r34max = 4,
        r34npts = 10,
        r12 = 0.0,
        max_func = lambda x: float(np.amax(abs(x))),
        exponents=[1.0]#Vector(range(1,5))*0.2
):
    colors = "rgbcmy"
    color_idx = 0
    rABpts = Vector(range(rABnpts)) * (rABmax-rABmin)/rABnpts + rABmin
    r34pts = Vector(range(r34npts)) * (r34max-r34min)/r34npts + r34min
    xvals = defaultdict(lambda: [])
    yvals = defaultdict(lambda: [])
    rABr34 = defaultdict(lambda: [])
    yest = defaultdict(lambda: [])

    fig = plt.figure(4)
    ax = fig.add_subplot(111)
    min_x = float("inf")
    max_x = -float("inf")
    for rAB in rABpts:
        for r34 in r34pts:
            curr_set = IntDataSet(
                centers=[
                    [-r12/2, 0, 0],
                    [r12/2, 0, 0],
                    [-r34/2, rAB, 0],
                    [r34/2, rAB, 0]
                ],
                basis_name="cc-pV5Z"
            )

            for q in exponents:
                qint = curr_set.int_for(*quartet)
                val = max_func(qint.get_ints("eri"))
                #est = np.amax(qint.overlap(0,1)) * np.amax(qint.overlap(2,3))**(q - 0.01 * rAB) / rAB
                est = np.amax(qint.overlap(0,1)) * np.amax(qint.overlap(2,3))*max(r34,1.0) / rAB
                #est = 1 / rAB
                if abs(val) > 1e-7:
                    rABr34[q].append((rAB, r34))
                    xvals[q].append(est)
                    yvals[q].append(val)
                    yest[q].append(est)
    for q in exponents:
        decay = yvals[q]
        x, decay, ests, rab34 = zip(*sorted(zip(xvals[q], decay, yest[q], rABr34[q])))
        xvals[q], yvals[q], yest[q], rABr34[q] = x, decay, ests, rab34
        tmp = min(x)
        if tmp < min_x: min_x = tmp
        tmp = max(x)
        if tmp > max_x: max_x = tmp
        plt.plot(
            list(x),
            list(decay),
            colors[color_idx % len(colors)] + "-",
            marker="o",
            label="exp={}, ({}a {}b | {}c {}d)".format(q, *quartet),
            picker=5
        )
        color_idx += 1

    def onpick(event):
        idx = event.ind
        for q in exponents:
            if len(exponents) > 1:
                print 'q = {}, x = {}, y = {}, rAB = {}, r34 = {}'.format(
                    q,  xvals[q][idx],  yvals[q][idx],
                    rABr34[q][idx][0], rABr34[q][idx][1]
                )
            else:
                if len(idx) > 1:
                    print idx
                for i in idx:
                    print '{}idx= {}, x = {}, y = {}, rAB = {}, r34 = {}'.format(
                        "    " if len(idx) > 1 else "",
                        i, xvals[q][i],  yvals[q][i],
                        rABr34[q][i][0], rABr34[q][i][1]
                    )

    #plot the estimates
    #ax.set_xscale('log')
    ax.grid(True)
    ax.plot(
        (min_x, max_x),
        (min_x, max_x),
        "k-",
        lw=3
    )

    fig.canvas.mpl_connect('pick_event', onpick)
    ax.legend(loc='best')
    plt.show()

def figure_7(
        shell_quartets=(
            (3,3,3,3),
            #(4,4,4,4),
            #(5,5,5,5),
            #(4,4,5,5),
            #(6,6,6,6),
            #(8,8,8,8),
            #(4,4,14,14),
            #(14,14,14,14),
        ),
        rABmin = 0,
        rABmax = 40,
        rABnpts = 20,
        r34min = 0,
        r34max = 15,
        r34npts = 10,
        r12 = 2.0,
        max_func = lambda x: float(np.amax(abs(x))),
        int_type = "f12_double_commutator"
):
    colors = "rgbcmy"
    color_idx = 0
    rABpts = Vector(range(rABnpts)) * (rABmax-rABmin)/rABnpts + rABmin
    r34pts = Vector(range(r34npts)) * (r34max-r34min)/r34npts + r34min
    xvals = defaultdict(lambda: [])
    yvals = defaultdict(lambda: [])
    rABr34 = defaultdict(lambda: [])
    yest = defaultdict(lambda: [])

    fig = plt.figure(4)
    ax = fig.add_subplot(111)
    min_x = float("inf")
    max_x = -float("inf")
    for rAB in rABpts:
        for r34 in r34pts:
            curr_set = IntDataSet(
                centers=[
                    [-r12/2, 0, 0],
                    [r12/2, 0, 0],
                    [-r34/2, rAB, 0],
                    [r34/2, rAB, 0]
                ],
                basis_name="cc-pV5Z"
            )

            theta = 1e-2
            for q in shell_quartets:
                if not q[0] == q[1] and q[2] == q[3]:
                    raise NotImplementedError
                qint = curr_set.int_for(*q)
                print qint.pair_extent(0,1,theta)
                print qint.pair_extent(2,3,theta)
                #q1 = math.sqrt(abs(np.amax(curr_set.int_for(q[0],q[1],q[0],q[1]).get_ints(int_type))))
                #q2 = math.sqrt(abs(np.amax(curr_set.int_for(q[2],q[3],q[2],q[3]).get_ints(int_type))))
                q1 = math.sqrt(abs(np.amax(curr_set.int_for(q[0],q[1],q[0],q[1]).get_ints("f12"))))
                q2 = math.sqrt(abs(np.amax(curr_set.int_for(q[2],q[3],q[2],q[3]).get_ints("f12"))))
                val = max_func(qint.get_ints("f12"))
                s34 = np.amax(qint.overlap(2,3))
                est = q1 * q2 * math.exp(-(rAB - qint.pair_extent(0,1,theta) - qint.pair_extent(2,3,theta))**2)
                #est = q1 * q2 * math.exp(-rAB**2)
                # if well-separated and significant
                if abs(val) > theta and rAB - qint.pair_extent(0,1,theta) - qint.pair_extent(2,3,theta) > 0:
                    rABr34[q].append((rAB, r34))
                    xvals[q].append(est)
                    yvals[q].append(val)
                    yest[q].append(est)
    for q in shell_quartets:
        decay = yvals[q]
        x, decay, ests, rab34 = zip(*sorted(zip(xvals[q], decay, yest[q], rABr34[q])))
        xvals[q], yvals[q], yest[q], rABr34[q] = x, decay, ests, rab34
        tmp = min(x)
        if tmp < min_x: min_x = tmp
        tmp = max(x)
        if tmp > max_x: max_x = tmp
        plt.plot(
            list(x),
            list(decay),
            colors[color_idx % len(colors)] + "-",
            marker="o",
            label="({}a {}b | {}c {}d)".format(*q),
            picker=5
        )
        color_idx += 1

    def onpick(event):
        idx = event.ind
        for q in shell_quartets:
            if len(shell_quartets) > 1:
                print 'q = {}, x = {}, y = {}, rAB = {}, r34 = {}'.format(
                    q,  xvals[q][idx],  yvals[q][idx],
                    rABr34[q][idx][0], rABr34[q][idx][1]
                )
            else:
                if len(idx) > 1:
                    print idx
                for i in idx:
                    print '{}idx= {}, x = {}, y = {}, rAB = {}, r34 = {}'.format(
                        "    " if len(idx) > 1 else "",
                        i, xvals[q][i],  yvals[q][i],
                        rABr34[q][i][0], rABr34[q][i][1]
                    )

    #plot the estimates
    #ax.set_xscale('log')
    ax.grid(True)
    ax.plot(
        list(Vector(range(100))* (max_x - min_x)/100 + min_x),
        list(Vector(range(100))* (max_x - min_x)/100 + min_x),
        "k-",
        lw=3
    )

    fig.canvas.mpl_connect('pick_event', onpick)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc='best')
    plt.show()

#figure_4()
