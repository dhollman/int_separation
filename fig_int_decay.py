from __future__ import division
from collections import Iterable
from numbers import Number
from int_sep import *
from matplotlib import transforms
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

def frange(rmin, rmax, npts):
    return list(Vector(range(npts)) * ((rmax-rmin)/npts) + rmin)


def two_center_integral_decay(**kwargs):
    return four_center_integral_decay(r12=0, r34=0, centers=(0,0,1,1), **kwargs)

def four_center_integral_decay(
        quartets,
        r12s, r34s,
        fit_function,
        fit_fxn_name_template,
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
        log_y=True
):
    colors = [tuple(Vector(x)/255) for x in [[255,0,16], [43,206,72],
        [240,163,255], [255,168,187], [66,102,0],
        [94,241,242], [0,153,143], [224,255,102], [116,10,255],
        [153,0,0], [255,255,128], [255,255,0], [255,80,5],
        [0,117,220], [153,63,0], [76,0,92], [25,25,25],
        [0,92,49],  [255,204,153], [128,128,128],
        [148,255,181], [143,124,0], [157,204,0], [194,0,136],
        [0,51,128], [255,164,5]
    ]]
    color_idx = 0

    if isinstance(r12s, Number):
        r12s = [r12s] * len(quartets)


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
    #========================================#
    decays = defaultdict(lambda: [])
    for r in xpoints:
        int_set = IntDataSet(
            centers=[
                [-r12/2, 0, 0],
                [r12/2, 0, 0],
                [-r34/2, 0, r],
                [r34/2, 0, r]
            ],
            basis_name=basis
        )

        for q in quartets:
            decays[q].append(
                max_fxn(int_set.int_for(*q).get_ints(int_type))
            )
    #========================================#
    for q in quartets:
        # Plot the values themselves
        if normalize:
            decay = list(Vector(decays[q])/max(decays[q]))
        else:
            decay = decays[q]
        name = r"$( {0}_{4} {1}_{5} |\ {2}_{6} {3}_{7} )$ decay".format(*(list(q)+["abcd"[c] for c in centers]))
        ax.plot(
            xpoints, decay,
            color=colors[color_idx % len(colors)],
            linestyle=":",
            marker="o",
            label=name
        )
        #----------------------------------------#
        # Find the fit
        fit_x, fit_y = zip(*[
            (xx, yy) for xx, yy in zip(xpoints, decay)
                    if (fit_xmin is None or xx >= fit_xmin) and (fit_xmax is None or xx <= fit_xmax)
        ])
        popt, _ = curve_fit(
            fit_function,
            fit_x,
            fit_y,
            **curve_fit_options
        )
        #----------------------------------------#
        # Compute the R^2 value of the fit
        #SSerr = sum((fit_function(fit_x, *popt) - np.array(fit_y))**2)
        #SStot = sum((np.array(fit_y) - np.mean(fit_y))**2)
        #Rsq = 1 - SSerr/SStot
        if chisq_xmin is None: chisq_xmin = fit_xmin
        if chisq_xmax is None: chisq_xmax = fit_xmax
        chisq_x, chisq_y = zip(*[
            (xx, yy) for xx, yy in zip(xpoints, decay)
            if (chisq_xmin is None or xx >= chisq_xmin) and (chisq_xmax is None or xx <= chisq_xmax)
        ])
        #chisq = sum((np.array(chisq_y) - fit_function(chisq_x, *popt))**2 / fit_function(chisq_x, *popt))
        chisqlog = sum((np.log10(np.array(chisq_y)) - np.log10(fit_function(chisq_x, *popt)))**2 / abs(np.log10(fit_function(chisq_x, *popt))))

        #----------------------------------------#
        # Plot the fit
        ax.plot(
            frange(xmin, xmax, 100),
            fit_function(frange(xmin, xmax, 100), *popt),
            color=colors[color_idx % len(colors)],
            linestyle="-",
            label = fit_fxn_name_template.format(*popt)
                    #+ "\n$R^2$ = {:.8f}".format(Rsq)
                    #+ "\n$\chi^2$ = {:.3e}".format(chisq)
                    + "\n$\chi^2[log_{{10}}]$ = {:.3e}".format(chisqlog)
        )
        #----------------------------------------#
        # Plot the deviation from the fit on the error axis
        #if do_error:
        #    fdev = abs(Vector(fit_y) - fit_function(fit_x, *popt))
        #    err_axes.plot(
        #        fit_x,
        #        list(fdev),
        #        colors[color_idx % len(colors)] + "--",
        #        marker="*",
        #    )
        #----------------------------------------#
        color_idx += 1
    #========================================#
    # Show the region over which the fit is being carried out
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
    #========================================#
    # Other misc options
    ax.set_ylim(0,1)
    ax.set_title(r"Normalized decay of selected integrals in " + basis)
    ax.set_xlabel("$R_{ab}$ (Angstroms)")
    ax.set_ylabel("Integral value (normalized)" + (", log scale" if log_y else ""))
    ax.grid(True)
    ax.legend(loc="best", ncol=2)
    plt.show()


# cc-pV5Z shells
#   0-4: s functions (only 0 contracted)
#   5-8: p functions
#   9-11: d functions
#   12, 13: f functions
#   14: g function


def one_over_r(xpts, a, b):
    return a / (np.array(xpts) - b)

def one_over_r3(xpts, a, b):
    return a / (np.array(xpts) - b)**3

def one_over_r4(xpts, a, b):
    return a / (np.array(xpts) - b)**4

def one_over_r5(xpts, a, b):
    return a / (np.array(xpts) - b)**5

def one_over_r5_nob(xpts, a):
    return a / (np.array(xpts))**5

def one_over_rn(xpts, a, b, n):
    return a / (np.array(xpts) - b)**n

def one_over_rn_nob(xpts, a, n):
    return a / (np.array(xpts))**n

def one_over_r_series_3(xpts, c1, c2, c3):
    xary = np.array(xpts)
    return c1/xary + c2/xary**2 + c3/xary**3

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

if __name__ == "__main__":

    # One over r decay
    #   Arises from integrals of the form
    #   ( i_a j_a | k_b l_b ) where AM(i) == AM(j)
    #   and AM(k) == AM(l)
    if 0:
        two_center_integral_decay(
            quartets=(
                (1,1,1,1), (4,4,4,4), (6,6,6,6),
                (14,14,14,14), (4,4,6,6), (8,8,12,12),
                (6,8,12,13), (1,1,14,14)
            ),
            fit_function=one_over_r,
            fit_fxn_name_template="${0:.3g} / (R_{{ab}} - {1:.3g})$",
            fit_xmin=5,
            xmin=3, xmax=20, npts=50
        )

    # One over r^3 decay
    #   Arises from integrals of the form
    #   ( i_a j_a | k_b l_b ) where
    #   abs(AM(i) - AM(j)) + abs(AM(k) - AM(l)) == 2
    if 0:
        two_center_integral_decay(
            quartets=(
                (1,5,1,5),
                (4,8,4,8),
                (6,9,6,10),
                (9,12,10,13),
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
                (1,9,9,1),
                (1,2,1,14),
                (4,10,4,10),
                (6,13,6,13),
                (9,14,10,14),
                (3,5,6,14),
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
                (1,5,1,9),
                (4,10,4,10),
                (14,14,14,14),
                (1,5,1,14),
                (7,3,13,8),
                (14,13,12,7)
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
                (1,5,1,9),
                (4,10,4,10),
                (14,14,14,14),
                (1,5,1,14),
                (7,3,13,8),
                (14,13,12,7)
            ),
            fit_function=one_over_rn_nob,
            fit_fxn_name_template="${0:.3g} / R_{{ab}}^{{{1:.2g}}}$",
            fit_xmin=5,
            xmin=0.5,
            xmax=15
        )

    #================================================================================#

    # This leads to nonsense
    if 0:
        four_center_integral_decay(
            quartets=(
                # delta_l = 5
                (1,14,10,8),
                # delta_l = 2
                (1,5,1,5),
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
                (1,14,10,8),
                # delta_l = 2
                (1,5,1,5),
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
                (1,14,10,8),
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


    if 1:
        four_center_integral_decay(
            quartets=(
                # delta_l = 5
                (1,14,14,14),
                # delta_l = 2
                #(1,5,1,5),
                # delta_l = 8
                #(1,14,1,14),
            ),
            r12=0.1, r34=0.0,
            fit_function=one_over_r,
            fit_fxn_name_template="${0:.3g}(R_{{ab}}-{1:.3g})^{{-1}}$",
            fit_xmin=5.0, fit_xmax=20,
            xmin=0.5, xmax=50,
            npts=75,
            chisq_xmax=50,
            normalize=False
        )
