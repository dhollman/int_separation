from __future__ import division
from collections import Iterable, Sized
from functools import partial
from inspect import getargspec
from numbers import Number
from int_sep import *
from matplotlib import transforms
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from grendel import Bohr, Angstroms

b2a = Bohr.to(Angstroms)
a2b = Angstroms.to(Bohr)

def frange(rmin, rmax, npts):
    return list(Vector(range(npts)) * ((rmax-rmin)/npts) + rmin)

psi4.clean()

def two_center_integral_decay(**kwargs):
    return four_center_integral_decay(r12s=0, r34s=0, centers=(0,0,1,1), **kwargs)

def four_center_integral_decay(
        #region | Lots of arguments... |
        quartets,
        r12s, r34s,
        fit_function,
        fit_fxn_name_template=None,
        centers=(0,1,2,3),
        xmin=1, xmax=8,
        npts=28,
        fit_xmin=3.0, fit_xmax=None,
        chisq_xmin=None, chisq_xmax=None,
        basis="cc-pV5Z",
        int_type="eri",
        max_fxn=lambda x: float(np.amax(abs(x))),
        curve_fit_options=dict(
            maxfev=6000,
        ),
        normalize=True,
        log_y=True,
        legend_font_size=None
        #endregion
):
    #region | Set up and stuff |
    # From the "alphabet of colors"
    colors = [tuple(Vector(x) / 255) for x in [
        [255, 0, 16], [43, 206, 72],
        [240, 163, 255], [255, 168, 187], [66, 102, 0],
        [94, 241, 242], [0, 153, 143], [224, 255, 102],
        [116, 10, 255],
        [153, 0, 0], [255, 255, 128], [255, 255, 0],
        [255, 80, 5],
        [0, 117, 220], [153, 63, 0], [76, 0, 92], [25, 25, 25],
        [0, 92, 49], [255, 204, 153], [128, 128, 128],
        [148, 255, 181], [143, 124, 0], [157, 204, 0],
        [194, 0, 136],
        [0, 51, 128], [255, 164, 5]
    ]]
    color_idx = 0

    if isinstance(quartets[0], int):
        quartets = (quartets,)

    if isinstance(r12s, Number):
        r12s = [r12s] * len(quartets)
    if isinstance(r34s, Number):
        r34s = [r34s] * len(quartets)
    if len(quartets) == 1 and len(r12s) != 1:
        quartets = list(quartets) * len(r12s)
        if len(r34s) == 1: r34s *= len(quartets)
    elif len(quartets) == 1 and len(r34s) != 1:
        quartets = list(quartets) * len(r34s)
        if len(r12s) == 1: r12s *= len(quartets)
    if len(r12s) != len(r34s) or len(r34s) != len(quartets):
        raise ValueError("Dimension mismatch: {} != {} != {}".format(
            len(r12s), len(r34s), len(quartets)
        ))

    xmin *= a2b
    if fit_xmin is not None:
        fit_xmin *= a2b
    else:
        fit_xmin = xmin

    xmax *= a2b
    if fit_xmax is not None:
        fit_xmax *= a2b
    else:
        fit_xmax = xmax

    xpoints = frange(xmin, xmax, npts)
    fig = plt.figure("Decay Fits")
    ax = fig.add_subplot(111)
    if log_y:
        ax.set_yscale("log")
        #if do_error:
    #    err_axes = fig.add_subplot(212)
    #    err_axes = axes
    #    if log_yerr:
    #        err_axes.set_yscale("log")

    if fit_fxn_name_template is None:
        if hasattr(fit_function, "name_template"):
            fit_fxn_name_template = fit_function.name_template
        else:
            raise ValueError("Fit function has no name template, and no template was explicitly specified")

    #endregion
    #========================================#
    #region | Get the actual integral values |
    decays = defaultdict(lambda: [])
    int_sets = dict()
    for r in xpoints:
        for r12, r34 in zip(r12s, r34s):
            int_sets[(r12, r34, r)] = IntDataSet(
                centers=[
                    [-r12*b2a / 2, 0, 0],
                    [r12*b2a / 2, 0, 0],
                    [-r34*b2a / 2, 0, r*b2a],
                    [r34*b2a / 2, 0, r*b2a]
                ],
                basis_name=basis
            )

        for q, r12, r34 in zip(quartets, r12s, r34s):
            decays[(q, r12, r34)].append(
                max_fxn(int_sets[(r12, r34, r)].int_for(*q).get_ints(int_type))
            )
    #endregion
    #========================================#
    for q, r12, r34 in zip(quartets, r12s, r34s):
        qk = (q, r12, r34)
        #----------------------------------------#
        #region | Plot the values themselves |
        if normalize:
            decay = list(Vector(decays[qk])/max(decays[qk]))
        else:
            decay = decays[qk]
        if r12 == 0 and r34 == 0:
            name = "$( {0}_a {1}_a |\\ {2}_b {3}_b )$ decay".format(*q)
        else:
            name = "$( {0}_{4} {1}_{5} |\\ {2}_{6} {3}_{7} )$ decay\n($R_{{ab}} = {8:.2g}$, $R_{{cd}} = {9:.2g})$".format(
                *(list(q)+["abcd"[c] for c in centers]+[r12,r34])
            )
        ax.plot(
            xpoints, decay,
            color=colors[color_idx % len(colors)],
            linestyle=":",
            marker="o",
            label=name
        )
        #endregion
        #----------------------------------------#
        #region | Find the fit |
        fit_x, fit_y = zip(*[
            (xx, yy) for xx, yy in zip(xpoints, decay)
                    if (fit_xmin is None or xx >= fit_xmin) and (fit_xmax is None or xx <= fit_xmax)
        ])
        ffargs = getargspec(fit_function).args
        nparam = len(ffargs) - 1
        #----------------------------------------#
        # Give the function the Schwarz information if needed
        qq_fit = "q1" in ffargs and "q2" in ffargs
        if ("q1" in ffargs and "q2" not in ffargs) or ("q2" in ffargs and "q1" not in ffargs):
            raise NameError("Name your functions parameters to not include"
                            " 'q1' or 'q2' unless you use both and need them"
                            " to be the Schwarz q1 and q2")
        if qq_fit:
            q1 = max_fxn(int_sets[(r12,r34,xpoints[0])].int_for(q[0],q[1],q[0],q[1],centers=(0,1,0,1)).get_ints(int_type))
            q2 = max_fxn(int_sets[(r12,r34,xpoints[0])].int_for(q[2],q[3],q[2],q[3],centers=(2,3,2,3)).get_ints(int_type))
            def fit_function_wrap(*args, **kwargs):
                return fit_function(*args, q1=q1, q2=q2, **kwargs)
            # Now tell scipy curve fit about the number of parameters:
            ffuncq = fit_function_wrap
            nparam -= 2
        else:
            ffuncq = fit_function
        #----------------------------------------#
        # Give the function the zeta information if needed
        zeta_fit = "zetas" in ffargs
        if zeta_fit:
            iset = int_sets[(r12,r34,xpoints[0])]
            idata = iset.int_for(*q)
            if any(nprim > 1 for nprim in idata.shell_nprimative):
                raise NotImplementedError("Contracted shells not implemented")
            zetas = [sh.exp(0) for sh in idata.shells]
            def fit_function_wrap(*args, **kwargs):
                return ffuncq(*args, zetas=zetas, **kwargs)
            ffuncz = fit_function_wrap
            nparam -= 1
        else:
            ffuncz = ffuncq
        #----------------------------------------#
        r12_r34_fit = "r12" in ffargs and "r34" in ffargs
        if r12_r34_fit:
            def fit_function_wrap(*args, **kwargs):
                return ffuncz(*args, r12=r12, r34=r34, **kwargs)
            ffuncr = fit_function_wrap
            nparam -= 2
        else:
            ffuncr = ffuncz
        #----------------------------------------#
        ss_fit = "s1" in ffargs and "s2" in ffargs
        if ss_fit:
            idata = int_sets[(r12,r34,xpoints[0])].int_for(*q)
            s1 = max_fxn(idata.overlap(0,1))
            s2 = max_fxn(idata.overlap(2,3))
            def fit_function_wrap(*args, **kwargs):
                return ffuncr(*args, s1=s1, s2=s2, **kwargs)
                # Now tell scipy curve fit about the number of parameters:
            ffunc = fit_function_wrap
            nparam -= 2
        else:
            ffunc = ffuncr
        #----------------------------------------#
        # Figure out the number of parameters and let the optimizer know via the p0 option
        if nparam <= 0:
            raise ValueError("Fit function doesn't have enough parameters")
        if "p0" not in curve_fit_options:
            curve_fit_options["p0"] = [1]*nparam
        elif not isinstance(curve_fit_options["p0"], Sized):
            curve_fit_options["p0"] = [curve_fit_options["p0"]]*nparam
        elif len(curve_fit_options["p0"]) != nparam:
            raise ValueError("Dimension mismatch in curve fitting initial parameters: {} != {}".format(
                len(curve_fit_options["p0"]), nparam
            ))
        #----------------------------------------#
        # call scipy.optimize.curve_fit
        popt, _ = curve_fit(
            ffunc,
            fit_x,
            fit_y,
            **curve_fit_options
        )
        #endregion
        #----------------------------------------#
        #region | Compute the quality of the fit |

        #SSerr = sum((ffunc(fit_x, *popt) - np.array(fit_y))**2)
        #SStot = sum((np.array(fit_y) - np.mean(fit_y))**2)
        #Rsq = 1 - SSerr/SStot
        if chisq_xmin is None: chisq_xmin = fit_xmin
        if chisq_xmax is None: chisq_xmax = fit_xmax
        chisq_x, chisq_y = zip(*[
            (xx, yy) for xx, yy in zip(xpoints, decay)
            if (chisq_xmin is None or xx >= chisq_xmin) and (chisq_xmax is None or xx <= chisq_xmax)
        ])
        #chisq = sum((np.array(chisq_y) - ffunc(chisq_x, *popt))**2 / ffunc(chisq_x, *popt))
        chisqlog_y = np.log10(np.array(chisq_y))
        chisqlog_func = np.log10(ffunc(chisq_x, *popt))
        chisqlog = sum((chisqlog_y - chisqlog_func)**2 / abs(chisqlog_func))

        #endregion
        #----------------------------------------#
        #region | Plot the fit |

        ax.plot(
            frange(xmin, xmax, 100),
            ffunc(frange(xmin, xmax, 100), *popt),
            color=colors[color_idx % len(colors)],
            linestyle="-",
            label=fit_fxn_name_template.format(*popt)
            #+ "\n$R^2$ = {:.8f}".format(Rsq)
            #+ "\n$\chi^2$ = {:.3e}".format(chisq)
                  + "\n$\chi^2[log_{{10}}]$ = {:.3e}".format(chisqlog)
        )

        #endregion
        #----------------------------------------#
        #region | #plot error (disabled) |
        # Plot the deviation from the fit on the error axis
        #if do_error:
        #    fdev = abs(Vector(fit_y) - ffunc(fit_x, *popt))
        #    err_axes.plot(
        #        fit_x,
        #        list(fdev),
        #        colors[color_idx % len(colors)] + "--",
        #        marker="*",
        #    )
        #endregion
        #----------------------------------------#
        color_idx += 1
    #========================================#
    #region | Show the region over which the fit is being carried out |

    if fit_xmin is not None:
        fit_min_axes = (fit_xmin-xmin)/(xmax-xmin)
    else:
        fit_min_axes = 0
    if fit_xmax is not None:
        fit_max_axes = (fit_xmax-xmin)/(xmax-xmin)
    else:
        fit_max_axes = 1
    fitr_height = 0.05
    fitr = Rectangle(
        (fit_min_axes,0), fit_max_axes-fit_min_axes, fitr_height,
         transform=ax.transAxes,
         facecolor='red',
         alpha=0.38
    )

    ax.add_patch(fitr)
    ax.text(
        (fit_min_axes+fit_max_axes)/2,fitr_height/2, "Fit region",
        verticalalignment='center',
        horizontalalignment='center',
        transform=ax.transAxes
    )

    #endregion
    #========================================#
    #region | Other misc options |

    ax.set_ylim(0,1)
    ax.set_title(("Normalized d" if normalize else "D") + "ecay of selected integrals in " + basis)
    ax.set_xlabel("$R_{ab}$ (Bohr)")
    ax.set_ylabel("Integral value"
                  + (" (normalized)" if normalize else "")
                  + (", log scale" if log_y else "")
    )
    ax.grid(True)
    nlegend_col = 2
    if legend_font_size is None:
        legend_font_size = 12
        if len(quartets) > 6:
            legend_font_size = 10
        if len(quartets) > 10:
            legend_font_size = 8
    if len(quartets) % 2 == 1:
        nlegend_col = 1
    ax.legend(
        loc="best",
        ncol=nlegend_col,
        prop={
            'size':legend_font_size
        }
    )
    plt.show()

    #endregion


# cc-pV5Z shells
#   0-4: s functions (only 0 contracted)
#   5-8: p functions
#   9-11: d functions
#   12, 13: f functions
#   14: g function


#region | a/r^k functions |
def one_over_r(xpts, a, b):
    return a / (np.array(xpts) - b)

def one_over_rn_no_ab(xpts, n):
    return 1.0 / np.array(xpts)**n
one_over_rn_no_ab.name_template="$R_{{AB}}^{{-{0:.2g}}}$"

def one_over_r3(xpts, a, b):
    return a / (np.array(xpts) - b)**3

def one_over_r4(xpts, a, b):
    return a / (np.array(xpts) - b)**4

def one_over_r5(xpts, a, b):
    return a / (np.array(xpts) - b)**5
#endregion


#region | No b in denominator functions |

def one_over_rn_nob(xpts, a, n):
    return a / (np.array(xpts))**n
one_over_rn_nob.name_template="${0:.3g}R_{{AB}}^{{-{1:.2g}}}$"

def one_over_r5_nob(xpts, a):
    return a / (np.array(xpts))**5

#endregion


#region | A/r^k series... no interesting results |

def one_over_r_series_3(xpts, c1, c2, c3):
    xary = np.array(xpts)
    return c1 / xary + c2 / xary ** 2 + c3 / xary ** 3


def one_over_r_series_6(xpts, c1, c2, c3, c4, c5, c6):
    xary = np.array(xpts)
    return c1 / xary + c2 / xary ** 2 + c3 / xary ** 3 + c4 / xary ** 4 + c5 / xary ** 5 + c6 / xary ** 6


def one_over_r_series_10(xpts, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10):
    xary = np.array(xpts)
    return c1 / xary \
           + c2 / xary ** 2 \
           + c3 / xary ** 3 \
           + c4 / xary ** 4 \
           + c5 / xary ** 5 \
           + c6 / xary ** 6 \
           + c7 / xary ** 7 \
           + c8 / xary ** 8 \
           + c9 / xary ** 9 \
           + c10 / xary ** 10


def one_over_r_series_2_5(xpts, c2, c3, c4, c5):
    xary = np.array(xpts)
    return c2 / xary ** 2 \
           + c3 / xary ** 3 \
           + c4 / xary ** 4 \
           + c5 / xary ** 5


def one_over_r_series_2_5_b(xpts, c2, c3, c4, c5, b):
    xary = np.array(xpts)
    denom = np.array([max(x, 1) for x in (xary - b)])
    return c2 / denom ** 2 \
           + c3 / denom ** 3 \
           + c4 / denom ** 4 \
           + c5 / denom ** 5

def one_over_r_series_6(xpts, c1, c2, c3, c4, c5, c6):
    xary = np.array(xpts)
    return c1/xary + c2/xary**2 + c3/xary**3 + c4/xary**4 + c5/xary**5 + c6/xary**6

def one_over_r_series_10(xpts, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10):
    xary = np.array(xpts)
    return c1/xary \
           + c2/xary**2 \
           + c3/xary**3 \
           + c4/xary**4 \
           + c5/xary**5 \
           + c6/xary**6 \
           + c7/xary**7 \
           + c8/xary**8 \
           + c9/xary**9 \
           + c10/xary**10

def one_over_r_series_2_5(xpts, c2, c3, c4, c5):
    xary = np.array(xpts)
    return c2/xary**2 \
           + c3/xary**3 \
           + c4/xary**4 \
           + c5/xary**5

def one_over_r_series_2_5_b(xpts, c2, c3, c4, c5, b):
    xary = np.array(xpts)
    denom = np.array([max(x,1) for x in (xary-b)])
    return c2/denom**2 \
           + c3/denom**3 \
           + c4/denom**4 \
           + c5/denom**5
#endregion


def one_over_rn(xpts, a, b, n):
    return a / (np.array(xpts) - b)**n
one_over_rn.name_template="${0:.3g} / (R_{{AB}} - {1:.3g})^{{{2:.2g}}}$"

def q1q2_over_r_b_n(xpts, b, n, q1, q2):
    return q1*q2 / (np.array(xpts) - b)**n
q1q2_over_r_b_n.name_template="$Q_a Q_b / (R_{{AB}} - {0:.3g})^{{{1:.2g}}}$"

def s1s2_over_rn(xpts, a, n, s1, s2):
    return a*s1*s2 / np.array(xpts)**n
s1s2_over_rn.name_template="${0:.3g}S_A S_B R_{{AB}}^{{-{1:.2g}}}$"

def s1s2_over_r_b_n(xpts, a, b, n, s1, s2):
    return a*s1*s2 / (np.array(xpts)-b)**n
s1s2_over_r_b_n.name_template="${0:.3g}S_A S_B (R_{{AB}}-{1:.4g})^{{-{2:.2g}}}$"

def a_q1q2_over_rn(xpts, a, n, q1, q2):
    return a*q1*q2 / (np.array(xpts))**n
a_q1q2_over_rn.name_template="${0:.4g} Q_a Q_b R_{{AB}}^{{-{1:.2g}}}$"

def a_zetas_q1q2_over_rn(xpts, a, n, q1, q2, zetas):
    exp1 = math.sqrt((zetas[0]+zetas[1])/(zetas[0]*zetas[1]))
    exp2 = math.sqrt((zetas[2]+zetas[3])/(zetas[2]*zetas[3]))
    #exp1 = math.sqrt(1/(zetas[0]*zetas[1]))
    #exp2 = math.sqrt(1/(zetas[2]*zetas[3]))
    return a*exp1*exp2*q1*q2 / (np.array(xpts))**n
a_zetas_q1q2_over_rn.name_template=r"${0:.9g} \sqrt{{ \alpha_a \alpha_b }}Q_a Q_b R_{{AB}}^{{-{1:.2g}}}$"

def wacky_zetas_q1q2_over_rn(xpts, a, n, q1, q2, zetas):
    exp1 = math.sqrt((zetas[0]+zetas[1])/(zetas[0]*zetas[1]))
    exp2 = math.sqrt((zetas[2]+zetas[3])/(zetas[2]*zetas[3]))
    #exp1 = math.sqrt(1/(zetas[0]*zetas[1]))
    #exp2 = math.sqrt(1/(zetas[2]*zetas[3]))
    return ((2**(1/math.pi))/6)*a*exp1*exp2*q1*q2 / (np.array(xpts))**n
wacky_zetas_q1q2_over_rn.name_template=r"${0:.9g} k \sqrt{{ \alpha_a \alpha_b }}Q_a Q_b R_{{AB}}^{{-{1:.2g}}}$"


if __name__ == "__main__":

    #region | Trivial two center separation patterns |
    # One over r decay
    #   Arises from integrals of the form
    #   ( i_a j_a | k_b l_b ) where AM(i) == AM(j)
    #   and AM(k) == AM(l)
    if 0:
        two_center_integral_decay(
            quartets=(
                (1, 1, 1, 1), (4, 4, 4, 4), (6, 6, 6, 6),
                (14, 14, 14, 14), (4, 4, 6, 6), (8, 8, 12, 12),
                (6, 8, 12, 13), (1, 1, 14, 14)
            ),
            fit_function=one_over_r,
            fit_fxn_name_template="${0:.3g} / (R_{{ab}} - {1:.3g})$",
            fit_xmin=5,
            xmin=0, xmax=20, npts=50,
            log_y=False
        )

    # One over r^3 decay
    #   Arises from integrals of the form
    #   ( i_a j_a | k_b l_b ) where
    #   abs(AM(i) - AM(j)) + abs(AM(k) - AM(l)) == 2
    if 0:
        two_center_integral_decay(
            quartets=(
                (1, 5, 1, 5),
                (4, 8, 4, 8),
                (6, 9, 6, 10),
                (9, 12, 10, 13),
            ),
            fit_function=one_over_r3,
            fit_fxn_name_template="${0:.3g} / (R_{{ab}} - {1:.3g})^3$",
            fit_xmin=5,
            xmin=0.5,
            xmax=15
        )

    # 1/r^5 decay
    #   Arises from integrals of the form
    #   ( i_a j_a | k_b l_b ) where
    #   abs(AM(i) - AM(j)) + abs(AM(k) - AM(l)) == 4
    if 0:
        two_center_integral_decay(
            quartets=(
                (1, 9, 9, 1),
                (1, 2, 1, 14),
                (4, 10, 4, 10),
                (6, 13, 6, 13),
                (9, 14, 10, 14),
                (3, 5, 6, 14),
            ),
            fit_function=one_over_r5_nob,
            fit_fxn_name_template="${0:.3g} / R_{{ab}}^5$",
            fit_xmin=5,
            xmin=0.5,
            xmax=15
        )

    # etc. the general form is clearly
    #   1/r^(Delta_am1 + Delta_am2 + 1)
    if 0:
        two_center_integral_decay(
            quartets=(
                (1, 5, 1, 9),
                (4, 10, 4, 10),
                (14, 14, 14, 14),
                (1, 5, 1, 14),
                (7, 3, 13, 8),
                (14, 13, 12, 7)
            ),
            fit_function=one_over_rn,
            fit_fxn_name_template="${0:.3g} / (R_{{ab}} - {1:.3g})^{{{2:.2g}}}$",
            fit_xmin=5,
            xmin=0.5,
            xmax=15
        )

    # Notice we get similarly good results even when we get rid of the
    #  subtracted part of the denominator
    if 0:
        two_center_integral_decay(
            quartets=(
                (1, 5, 1, 9),
                (4, 10, 4, 10),
                (14, 14, 14, 14),
                (1, 5, 1, 14),
                (7, 3, 13, 8),
                (14, 13, 12, 7)
            ),
            fit_function=one_over_rn_nob,
            fit_fxn_name_template="${0:.3g} / R_{{ab}}^{{{1:.2g}}}$",
            fit_xmin=5,
            xmin=0.5,
            xmax=15
        )
    #endregion

    #================================================================================#

    #region | This leads to nonsense |
    # This leads to nonsense
    if 0:
        four_center_integral_decay(
            quartets=(
                # delta_l = 5
                (1, 14, 10, 8),
                # delta_l = 2
                (1, 5, 1, 5),
            ),
            r12=1.0, r34=1.0,
            fit_function=one_over_r_series_3,
            fit_fxn_name_template="${0:.3g}R_{{ab}}^{{-1}}"
                                  " + {1:.3g}R_{{ab}}^{{-2}}"
                                  " + {2:.3g}R_{{ab}}^{{-3}}$",
            xmin=0.5,
            xmax=20

        )

    if 0:
        four_center_integral_decay(
            quartets=(
                # delta_l = 5
                (1, 14, 10, 8),
                # delta_l = 2
                (1, 5, 1, 5),
            ),
            r12=1.0, r34=6.0,
            fit_function=one_over_r_series_6,
            fit_fxn_name_template="${0:.3g}R_{{ab}}^{{-1}}"
                                  " + {1:.3g}R_{{ab}}^{{-2}}"
                                  " + {2:.3g}R_{{ab}}^{{-3}}$\n$\\quad"
                                  " + {3:.3g}R_{{ab}}^{{-4}}"
                                  " + {4:.3g}R_{{ab}}^{{-5}}"
                                  " + {5:.3g}R_{{ab}}^{{-6}}$",
            fit_xmin=5.0, fit_xmax=20,
            xmin=0.5, xmax=50,
            npts=75,
            chisq_xmax=50

        )

    # More nonsense
    if 0:
        four_center_integral_decay(
            quartets=(
                # delta_l = 5
                (1, 14, 10, 8),
                # delta_l = 2
                #(1,5,1,5),
                # delta_l = 8
                #(1,14,1,14),
            ),
            r12=0.0, r34=0.0,
            fit_function=one_over_r_series_2_5_b,
            fit_fxn_name_template="${0:.3g}(R_{{ab}}-{4:.3g})^{{-2}}"
                                  " + {1:.3g}(R_{{ab}}-{4:.3g})^{{-3}}"
                                  " + {2:.3g}(R_{{ab}}-{4:.3g})^{{-4}}"
                                  " + {3:.3g}(R_{{ab}}-{4:.3g})^{{-5}}$",
            fit_xmin=5.0, fit_xmax=20,
            xmin=0.5, xmax=50,
            npts=75,
            chisq_xmax=50
        )
    #endregion

    ####################################################################
    #                         Useful!!!!!                              #
    ####################################################################
    if 0:
        four_center_integral_decay(
            quartets=(
                # delta_l = 5
                (14,14,1,12),
            ),
            r12s=[0.0,1.0,2.0,0.0,1.0,2.0],
            r34s=[0.0,0.0,0.0,1.0,1.0,1.0],
            fit_function=one_over_rn,
            fit_xmin=5.0, fit_xmax=20,
            xmin=0.5, xmax=20,
            npts=20,
            normalize=True
        )

    # Important conclusion:
    #   Three center integrals (where one center has large angular momentum)
    #   decay much more rapidly than four center analogs!

    #region | Two center fits using q1q2 and zeta |

    # Good figure
    if 0:
        two_center_integral_decay(
            quartets=(
                (1, 1, 1, 1),
                (3, 3, 3, 3),
                (4, 4, 3, 3),
                (1, 1, 4, 4),
                (5, 5, 5, 5),
                (8, 8, 8, 8),
                (6, 6, 7, 7),
                (5, 5, 8, 8)
            ),
            fit_function=a_zetas_q1q2_over_rn,
            fit_xmin=5, fit_xmax=20,
            xmin=0.5, xmax=20, npts=40,
            normalize=False
        )

    if 0:
        two_center_integral_decay(
            quartets=(
                (1, 1, 1, 1),
                (5, 5, 5, 5),
                (10, 10, 10, 10),
                (12, 12, 12, 12),
                (13, 13, 13, 13),
                (14, 14, 14, 14)
            ),
            fit_function=a_zetas_q1q2_over_rn,
            fit_xmin=10, fit_xmax=20,
            xmin=4, xmax=20, npts=40,
            normalize=False
        )

    if 0:
        two_center_integral_decay(
            quartets=(
                (1, 2, 1, 2),
                (1, 1, 1, 1),
                (1, 3, 1, 3),
                (2, 3, 2, 3),
            ),
            fit_function=a_q1q2_over_rn,
            fit_xmin=10, fit_xmax=20,
            xmin=4, xmax=20, npts=40,
            normalize=False
        )


    #endregion


    # Thought:  Screening using 1/R gets you nowhere.  BUT there
    #   is a point where 1/R is an accurate enough approximation,
    #   and the distributions can be treated as point charges.
    #   See, for instance, equation 9.8.29 (p. 370) in Helgaker
    #   Also, how can we use erfc to our advantage?

    # Demonstration of this idea:
    if 0:
        two_center_integral_decay(
            quartets=(
                #(1, 1, 1, 1),
                (3, 3, 3, 3),
                #(4, 4, 3, 3),
                (1, 1, 4, 4),
                (5, 5, 5, 5),
                (8, 8, 8, 8),
                #(6, 6, 7, 7),
                (5, 5, 8, 8),
                (14,14,14,14),
                (12,12,14,14),
                (12,12,12,12)
            ),
            fit_function=one_over_rn_nob,
            fit_xmin=5, fit_xmax=20,
            xmin=0.5, xmax=20, npts=30,
            normalize=False
        )

    # Now for some four center cases

    # SS/r is perfect fit in the SSSS case...
    if 0:
        four_center_integral_decay(
            quartets=(
                (1, 1, 1, 1),
            ),
            r12s=[0.0,1.0,3.0,0.0,1.0,3.0],
            r34s=[0.0,0.0,0.0,2.0,2.0,2.0],
            fit_function=s1s2_over_rn,
            fit_xmin=5, fit_xmax=20,
            xmin=0, xmax=20, npts=30,
            normalize=False
        )

    # Even for cross terms...
    if 0:
        four_center_integral_decay(
            quartets=(
                (1, 1, 4, 4),
            ),
            r12s=[0.0,1.0,3.0,0.0,1.0,3.0],
            r34s=[0.0,0.0,0.0,2.0,2.0,2.0],
            fit_function=s1s2_over_rn,
            fit_xmin=5, fit_xmax=20,
            xmin=0, xmax=20, npts=30,
            normalize=False
        )

    # And for (sp|sp) type integrals..
    if 0:
        four_center_integral_decay(
            quartets=(
                (1, 5, 1, 5),
            ),
            r12s=[0.0,1.0,3.0,0.0,1.0,3.0],
            r34s=[0.0,0.0,0.0,2.0,2.0,2.0],
            fit_function=s1s2_over_rn,
            fit_xmin=5, fit_xmax=20,
            xmin=0, xmax=20, npts=30,
            normalize=False
        )

    # And here is why the SS/R doesn't always work...
    if 0:
        # Not a perfect fit, even at super long range
        # Also, there's much more to be gained from short range...
        four_center_integral_decay(
            quartets=(
                (4, 14, 4, 14),
            ),
            r12s=3.0, r34s=3.0,
            fit_function=s1s2_over_rn,
            fit_xmin=50, fit_xmax=100,
            xmin=0.1, xmax=100, npts=200,
            normalize=False
        )

    if 1:
        four_center_integral_decay(
            quartets=(
                (4, 14, 4, 14),
            ),
            r12s=3.0, r34s=3.0,
            fit_function=s1s2_over_rn,
            fit_xmin=50, fit_xmax=100,
            xmin=0.1, xmax=100, npts=200,
            normalize=False
        )
