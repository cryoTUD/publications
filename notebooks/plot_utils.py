"""
Shared plotting utilities for Faraday Discussions figure notebooks.

All notebooks import this module via:
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))  # or notebooks/ dir
    from plot_utils import setup_style, plot_radial_profiles, plot_fsc_curves
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


# ── Style ────────────────────────────────────────────────────────────────────

def setup_style(font_scale=2.0, font="Helvetica"):
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"]  = 42
    sns.set_theme(context="paper", font=font, font_scale=font_scale)
    sns.set_style("white")


# ── Dual-axis helpers ────────────────────────────────────────────────────────

def add_resolution_axis(ax, fontsize=10):
    """Add resolution (Å) twin x-axis on top of a spatial-frequency axis."""
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    xticks = [t for t in ax.get_xticks() if t > 0]
    ax2.set_xticks(xticks)
    ax2.set_xticklabels([f"{1/t:.1f}" for t in xticks], fontsize=fontsize)
    ax2.set_xlabel("Resolution [Å]", fontsize=fontsize)
    return ax2


# ── Radial-profile plot ───────────────────────────────────────────────────────

def plot_radial_profiles(
    freq, profiles, labels=None,
    log_scale=False, normalized=False, squared=False,
    crop_freq=None, ylims=None,
    linewidth=2, figsize=(7, 5), fontsize=12,
):
    """
    Plot one or more radial profiles with a dual x-axis (freq / resolution).

    Parameters
    ----------
    freq      : 1-D array  spatial frequency [1/Å]
    profiles  : list of 1-D arrays
    labels    : list of str or None
    log_scale : bool – use semilogy
    normalized: bool – divide each profile by its max
    squared   : bool – plot |F|²
    crop_freq : (lo_Å, hi_Å) – resolution range in Angstroms to keep
    ylims     : (ymin, ymax) or None
    """
    if crop_freq is not None:
        lo = 1.0 / max(crop_freq)
        hi = 1.0 / min(crop_freq)
        mask = (freq >= lo) & (freq <= hi)
    else:
        mask = np.ones(len(freq), dtype=bool)

    freq_plot = freq[mask]
    fig, ax = plt.subplots(figsize=figsize)

    for i, profile in enumerate(profiles):
        label = labels[i] if labels else None
        y = np.array(profile)[mask].astype(float)
        if squared:
            y = y ** 2
        if normalized:
            ymax = np.nanmax(np.abs(y))
            if ymax > 0:
                y = y / ymax
        if log_scale:
            ax.semilogy(freq_plot, y, label=label, linewidth=linewidth)
        else:
            ax.plot(freq_plot, y, label=label, linewidth=linewidth)

    ax.set_xlabel("Spatial Frequency [Å⁻¹]", fontsize=fontsize)
    ylabel = "|F|²" if squared else "|F|"
    if normalized:
        ylabel += " (norm.)"
    ax.set_ylabel(ylabel, fontsize=fontsize)
    if ylims:
        ax.set_ylim(ylims)
    if labels:
        ax.legend(fontsize=fontsize * 0.75)

    add_resolution_axis(ax, fontsize=fontsize * 0.8)
    plt.tight_layout()
    return fig

def pretty_plot_radial_profile(freq,list_of_profiles_native,normalise=True, squared_amplitudes=True, discrete=True, legends=None,figsize=(14,8), fontsize=14,linewidth=1, marker="o", font="Helvetica",fontscale=1, showlegend=True, showPoints=False, alpha=0.05, variation=None, yticks=None, logScale=True, ylims=None, xlims=None, crop_freq=None):
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import cm
    from locscale.include.emmer.ndimage.profile_tools import crop_profile_between_frequency
    import seaborn as sns

    import matplotlib as mpl
    mpl.rcParams['pdf.fonttype'] = 42
    
    sns.set(rc={'figure.figsize':figsize})
    sns.set_theme(context="paper", font=font, font_scale=fontscale)
    sns.set_style("white")
    
    if normalise:
        list_of_profiles = []
        for profile in list_of_profiles_native:
            normalised_profile = profile/profile.max()
            list_of_profiles.append(normalised_profile)
    else:
        list_of_profiles = list_of_profiles_native
        

    i = 0
    colors = cm.rainbow(np.linspace(0,1,len(list_of_profiles)))
    xlabel_bottom_log = r'Spatial Frequency, $d^{-2} (\AA^{-2})$'
    xlabel_bottom_norm = r'Spatial Frequency, $d^{-1} (\AA^{-1})$'
    xlabel_top = r'Resolution, $d (\AA)$'
    if squared_amplitudes:
        ylabel_log = r'$ln  \langle \mid F \mid ^{2} \rangle $ '
    else:
        ylabel_log = r'$ln  \langle \mid F \mid \rangle $ '
    ylabel_norm = r'Normalised $ \langle \mid F \mid \rangle $'
    if discrete:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.grid(False)
        ax2 = ax1.twiny()

        if logScale:
            for profile in list_of_profiles:
                if crop_freq is not None:
                    frequency, profile = crop_profile_between_frequency(freq, profile, crop_freq[0], crop_freq[1])
                    if showPoints:
                        if squared_amplitudes:
                            ax1.plot(frequency**2,np.log(profile**2),c=colors[i], linewidth=linewidth, marker=marker)
                        else:
                            ax1.plot(frequency**2,np.log(profile),c=colors[i], linewidth=linewidth, marker=marker)
                    else:
                        if squared_amplitudes:
                            ax1.plot(frequency**2,np.log(profile**2),c=colors[i], linewidth=linewidth)
                        else:
                            ax1.plot(frequency**2,np.log(profile),c=colors[i], linewidth=linewidth)
                    
                else:
                    print("crop_freq is None")
                    print("Squared amp",squared_amplitudes)
                    if squared_amplitudes:
                        ax1.plot(freq**2,np.log(profile**2),c=colors[i], linewidth=linewidth)
                    else:
                        print("this line here")
                        ax1.plot(freq**2,np.log(profile),c=colors[i], linewidth=linewidth)
                i += 1
            
            ax2.set_xticks(ax1.get_xticks())
            ax2.set_xbound(ax1.get_xbound())
            ax2.set_xticklabels([round(1/np.sqrt(x),1) for x in ax1.get_xticks()])
            if showlegend:
                ax1.legend(legends)
            ax1.set_xlabel(xlabel_bottom_log)
            ax1.set_ylabel(ylabel_log)
            ax2.set_xlabel(xlabel_top)
        else:
            for profile in list_of_profiles:
                if crop_freq is not None:
                    frequency, profile = crop_profile_between_frequency(freq, profile, crop_freq[0], crop_freq[1])
                    if showPoints:
                        ax1.plot(frequency,profile,c=colors[i], linewidth=linewidth, marker=marker)
                    else:
                        ax1.plot(frequency,profile,c=colors[i], linewidth=linewidth)
                else:
                    ax1.plot(freq,profile,c=colors[i], linewidth=linewidth)
                i += 1
            
            
            if showlegend:
                ax1.legend(legends)
        
                
            ax2.set_xticks(ax1.get_xticks())
            ax2.set_xbound(ax1.get_xbound())
            ax2.set_xticklabels([round(1/x,1) for x in ax1.get_xticks()])
            
    
            ax1.set_xlabel(xlabel_bottom_norm)
            ax1.set_ylabel(ylabel_norm)
            ax2.set_xlabel(xlabel_top)

    else:
        
        profile_list = np.array(list_of_profiles)
        average_profile = np.einsum("ij->j", profile_list) / len(profile_list)
        
        variation = []
        for col_index in range(profile_list.shape[1]):
            col_extract = profile_list[:,col_index]
            variation.append(col_extract.std())

        variation = np.array(variation)
        
        y_max = average_profile + variation
        y_min = average_profile - variation

        fig = plt.figure()
        
        ax1 = fig.add_subplot(111)
        ax1.grid(False)
        ax2 = ax1.twiny()
        
        if logScale:
            if crop_freq is not None:
                frequency, average_profile = crop_profile_between_frequency(freq, average_profile, crop_freq[0], crop_freq[1])
                frequency, y_max = crop_profile_between_frequency(freq, y_max, crop_freq[0], crop_freq[1])
                frequency, y_min = crop_profile_between_frequency(freq, y_min, crop_freq[0], crop_freq[1])
            
            ax1.plot(frequency**2, np.log(average_profile), 'k',alpha=1)
            ax1.fill_between(frequency**2,np.log(y_max), np.log(y_min), color="grey", alpha=0.5)
            if showlegend:
                ax1.legend(["N={}".format(len(profile_list))])
        
            
            ax2.set_xticks(ax1.get_xticks())
            ax2.set_xbound(ax1.get_xbound())
            ax2.set_xticklabels([round(1/np.sqrt(x),1) for x in ax1.get_xticks()])
            
    
            ax1.set_xlabel(xlabel_bottom_log)
            ax1.set_ylabel(ylabel_log)
            ax2.set_xlabel(xlabel_top)
        else:
            if crop_freq is not None:
                frequency, average_profile = crop_profile_between_frequency(freq, average_profile, crop_freq[0], crop_freq[1])
                frequency, y_max = crop_profile_between_frequency(freq, y_max, crop_freq[0], crop_freq[1])
                frequency, y_min = crop_profile_between_frequency(freq, y_min, crop_freq[0], crop_freq[1])
            ax1.plot(frequency, average_profile, 'k',alpha=1)
            ax1.fill_between(frequency,y_max, y_min,color="grey", alpha=0.5)
            
            if showlegend:
                ax1.legend(["N={}".format(len(profile_list))])
        
                
            ax2.set_xticks(ax1.get_xticks())
            ax2.set_xbound(ax1.get_xbound())
            ax2.set_xticklabels([round(1/x,1) for x in ax1.get_xticks()])
            
    
            ax1.set_xlabel(xlabel_bottom_norm)
            ax1.set_ylabel(ylabel_norm)
            ax2.set_xlabel(xlabel_top)

    if ylims is not None:
        plt.ylim(ylims)
    if yticks is not None:
        plt.yticks(yticks)
    if xlims is not None:
        plt.xlim(xlims)
    
    
    #plt.tight_layout()
    return fig
    

# ── FSC curve plot ────────────────────────────────────────────────────────────

def plot_fsc_curves(
    freq, fsc_dict,
    labels=None, linewidth=2,
    figsize=(7, 5), fontsize=12, ylims=None,
):
    """
    Plot FSC curves vs spatial frequency with resolution axis on top.

    Parameters
    ----------
    fsc_dict : dict  {key: fsc_array}
    labels   : dict  {key: label_str}  or None (keys used as labels)
    """
    fig, ax = plt.subplots(figsize=figsize)
    for key, fsc in fsc_dict.items():
        label = (labels or {}).get(key, str(key))
        ax.plot(freq, fsc, label=label, linewidth=linewidth)
    ax.axhline(0.143, color="gray", linestyle="--", linewidth=1, label="FSC=0.143")
    ax.set_xlabel("Spatial Frequency [Å⁻¹]", fontsize=fontsize)
    ax.set_ylabel("FSC", fontsize=fontsize)
    ax.set_ylim(ylims or [-0.1, 1.05])
    ax.legend(fontsize=fontsize * 0.75)
    add_resolution_axis(ax, fontsize=fontsize * 0.8)
    plt.tight_layout()
    return fig

# ──────────────────────────────────────────────────────────────────────────────
def plot_correlations(x_array, y_array, scatter=False, figsize=(14,8),font="Helvetica",fontscale=3,hue=None, x_label=None, y_label=None, title_text=None, output_folder=None, filename=None, find_correlation=True, alpha=0.3):
    
    import matplotlib as mpl
    import seaborn as sns
    import os
    import matplotlib.pyplot as plt
    from scipy import stats
    import pandas as pd

    import matplotlib as mpl
    mpl.rcParams['pdf.fonttype'] = 42
    sns.set(rc={'figure.figsize':figsize})
    sns.set_theme(context="paper", font=font, font_scale=fontscale)
    sns.set_style("white")
    
    fig = plt.figure(1)
    
    if x_label is None:
        x_label = "x"
    
    if y_label is None:
        y_label = "y"
    
    
    if find_correlation:
        data = pd.DataFrame(data=[x_array,y_array], index=[x_label,y_label]).T
      
        def annotate(data, **kws):
            r, p = stats.pearsonr(data[x_label], data[y_label])
            ax = plt.gca()
            ax.text(.05, .8, 'R$^2$={:.2f}'.format(r),
                    transform=ax.transAxes)
        g = sns.lmplot(data=data, x=x_label, y=y_label, scatter=scatter)
        g.map_dataframe(annotate)
        plt.tight_layout()
        #plt.show()
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.tight_layout()
    if filename is not None:
        if output_folder is None:
            figure_output_folder = os.getcwd()
        else:
            figure_output_folder = output_folder
    
    
        output_filename = os.path.join(figure_output_folder, filename)
        plt.savefig(output_filename, dpi=600, bbox_inches="tight",transparency=True)
    else:
        return fig
    

def plot_correlations_multiple_single_plot(list_of_xy_tuple, scatter=False, hue=None, figsize=(14,8),fontscale=3, x_label=None, y_label=None, ylims=None, title_text=None, output_folder=None, filename=None, find_correlation=True, alpha=0.3, ci=95):
    import seaborn as sns
    import os
    import matplotlib.pyplot as plt
    from scipy import stats
    import pandas as pd
    import matplotlib as mpl
    mpl.rcParams['pdf.fonttype'] = 42
    
    sns.set(rc={'figure.figsize':figsize})
    sns.set_theme(context="paper", font="Helvetica", font_scale=fontscale)
    sns.set_style("white")
    
    fig, ax = plt.subplots(1,len(list_of_xy_tuple), sharex=True, sharey=True)   

    if x_label is None:
        x_label = "x"
    
    if y_label is None:
        y_label = "y"
    

    for i,xy_tuple in enumerate(list_of_xy_tuple):
        data_dictionary={}
        all_stacks = []
        for xy in xy_tuple:
            print(type(xy))
            print(len(xy))
            x_array, y_array, category_label = xy
            category_array = np.repeat(category_label, len(x_array))
            stack = np.vstack((x_array,y_array,category_array)).T
            all_stacks.append(stack)
        
        stack_arrays = np.concatenate(tuple([x for x in all_stacks]))
        
        data = pd.DataFrame(data=stack_arrays, columns=[x_label,y_label, "Category"])
        data[x_label] = data[x_label].astype(np.float32)
        data[y_label] = data[y_label].astype(np.float32)
        data["Category"] = data["Category"].astype(str)
    
        
   
        g = sns.scatterplot(data=data, x=x_label, y=y_label,legend=False,ax=ax[i], s=2)
        g.set(xlabel=None)
        g.set(ylabel=None)
        g.set(title=category_label)
        plt.legend(loc="lower right")   
    
        if i==0:
            continue
            #ax[i].get_yaxis().set_visible(False)
            #ax[i].get_xaxis().set_visible(False)
        else:
            continue
            #ax[i].get_yaxis().set_visible(False)
            #ax[i].get_xaxis().set_visible(False)
        
    
    if ylims is not None:
        plt.ylim(ylims)
    
    fig.add_subplot(1, 1, 1, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    
    plt.tight_layout()
    if filename is not None:
        if output_folder is None:
            figure_output_folder = os.getcwd()
        else:
            figure_output_folder = output_folder
    
    
        output_filename = os.path.join(figure_output_folder, filename)
        plt.savefig(output_filename, dpi=600,bbox_inches='tight')
    else:
        return fig        
    
def plot_correlations_multiple(xy_tuple, scatter=False, hue=None, figsize=(14,8),fontscale=3, x_label=None, y_label=None, ylims=None, title_text=None, output_folder=None, filename=None, find_correlation=True, alpha=0.3, ci=95):
    import seaborn as sns
    import os
    import matplotlib.pyplot as plt
    from scipy import stats
    import pandas as pd
    import matplotlib as mpl
    mpl.rcParams['pdf.fonttype'] = 42
    
    sns.set(rc={'figure.figsize':figsize})
    sns.set_theme(context="paper", font="Helvetica", font_scale=fontscale)
    sns.set_style("white")
    

    if x_label is None:
        x_label = "x"
    
    if y_label is None:
        y_label = "y"
    
    data_dictionary={}
    all_stacks = []
    for xy in xy_tuple:
        x_array, y_array, category_label = xy
        category_array = np.repeat(category_label, len(x_array))
        stack = np.vstack((x_array,y_array,category_array)).T
        all_stacks.append(stack)
    
    stack_arrays = np.concatenate(tuple([x for x in all_stacks]))
    
    data = pd.DataFrame(data=stack_arrays, columns=[x_label,y_label, "Category"])
    data[x_label] = data[x_label].astype(np.float32)
    data[y_label] = data[y_label].astype(np.float32)
    data["Category"] = data["Category"].astype(str)

        

    def annotate(data, **kws):
        r, p = stats.pearsonr(data[x_label], data[y_label])
        ax = plt.gca(figsize=(16,8))
        ax.text('R$^2$={:.2f}'.format(r),
                        transform=ax.transAxes)
    g = sns.lmplot(data=data, x=x_label, y=y_label, scatter=scatter, hue="Category", ci=ci, legend=False)
    # add to ax
        
    plt.legend(loc="lower right")   

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    if ylims is not None:
        plt.ylim(ylims)
    
        
    plt.tight_layout()
    if filename is not None:
        if output_folder is None:
            figure_output_folder = os.getcwd()
        else:
            figure_output_folder = output_folder
    
    
        output_filename = os.path.join(figure_output_folder, filename)
        plt.savefig(output_filename, dpi=600,bbox_inches='tight')
    else:
        # return figure 
        return plt.gcf()
    

def pretty_lineplot_XY(xdata, ydata, xlabel, ylabel, filename=None,figsize=(14,8), marker="o", markersize=12,fontscale=2.5,font="Helvetica",linewidth=2,legends=None):
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import cm
    import seaborn as sns    
    from matplotlib.pyplot import cm
    import matplotlib as mpl
    ## Function not generic
    mpl.rcParams['pdf.fonttype'] = 42
    plt.figure(1)
    sns.set(rc={'figure.figsize':figsize})
    sns.set_theme(context="paper", font=font, font_scale=fontscale)
    sns.set_style("white")

    sns.lineplot(x=xdata,y=ydata,linewidth=linewidth,marker=marker,markersize=markersize)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel, rotation=90, ha="center")

    if legends is not None:        
        plt.legend(legends)
    plt.tight_layout()
    plt.savefig(filename, dpi=600, bbox_inches="tight", transparency=True)
