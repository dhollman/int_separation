from __future__ import division
from int_sep import *
from matplotlib import transforms
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

def frange(rmin, rmax, npts):
    return list(Vector(range(npts)) * ((rmax-rmin)/npts) + rmin)


# cc-pV5Z shells
#   0-4: s functions (only 0 contracted)
#   5-8: p functions
#   9-11: d functions
#   12, 13: f functions
#   14: g function

def one_over_r(xpts, a, b):
    return a / (np.array(xpts) - b)

# First look at the decay of the ( i_a i_a | j_b j_b ) integrals
def integral_decay(
        quartets,
        xmin=1,
        xmax=8,
        npts=28,
        fit_xmin=3.0,
        fit_xmax=None,
        basis="cc-pV5Z",
        int_type="eri",
        fit_function=one_over_r,
        max_fxn=lambda x: float(np.amax(abs(x))),
        curve_fit_options=dict(
            maxfev=6000,
        ),
        fit_fxn_name_template="${0:.3g} / (R_{{ab}} - {1:.3g})$",
        log_y=True
):
    colors = [Vector(x)/255 for x in [[240,163,255],
        [0,117,220],
        [153,63,0],
        [76,0,92],
        [25,25,25],
        [0,92,49],
        [43,206,72],
        [255,204,153],
        [128,128,128],
        [148,255,181],
        [143,124,0],
        [157,204,0],
        [194,0,136],
        [0,51,128],
        [255,164,5],
        [255,168,187],
        [66,102,0],
        [255,0,16],
        [94,241,242],
        [0,153,143],
        [224,255,102],
        [116,10,255],
        [153,0,0],
        [255,255,128],
        [255,255,0],
        [255,80,5]]
    ]
    color_idx = 0

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
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, r],
                [0, 0, r]
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
        decay = list(Vector(decays[q])/max(decays[q]))
        name = r"$( {}_a {}_a |\ {}_b {}_b )$ decay".format(*q)
        ax.plot(
            xpoints, decay,
            colors[color_idx % len(colors)] + ":",
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
        chisq = sum((np.array(fit_y) - fit_function(fit_x, *popt))**2 / fit_function(fit_x, *popt))

        #----------------------------------------#
        # Plot the fit
        ax.plot(
            frange(xmin, xmax, 100),
            fit_function(frange(xmin, xmax, 100), *popt),
            colors[color_idx % len(colors)] + "-",
            label = fit_fxn_name_template.format(*popt)
                    #+ "\n$R^2$ = {:.8f}".format(Rsq)
                    + "\n$\chi^2$ = {:.3e}".format(chisq)
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
    ax.set_title(r"Normalized decay of selected $( \mu_a \mu_a | \nu_b \nu_b )$ integrals in " + basis)
    ax.set_xlabel("$R_{ab}$ (Angstroms)")
    ax.set_ylabel("Integral value (normalized)" + (", log scale" if log_y else ""))
    ax.grid(True)
    ax.legend(loc="best", ncol=2)
    plt.show()







if __name__ == "__main__":
    integral_decay(
        quartets=(
         (1,1,1,1),
         (4,4,4,4),
         (6,6,6,6),
         (14,14,14,14),
         (4,4,6,6),
         (8,8,12,12),
         (6,8,12,13)
     ),
        curve_fit_options=dict(
            p0=(0.3,0.3)
        )
    )
