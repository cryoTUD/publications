
## Plot tools 
import numpy as np
import seaborn as sns
from contextlib import contextmanager
from matplotlib import rcParams

@contextmanager
def temporary_rcparams(updates):
    original_rcparams = rcParams.copy()
    try:
        rcParams.update(updates)
        yield
    finally:
        rcParams.update(original_rcparams)

def configure_plot_scaling(figsize_mm, fontsize=None):
    reference_figure_size_mm = (162.56, 121.92)
    figsize_in = (figsize_mm[0] / 25.4, figsize_mm[1] / 25.4)  # Convert mm to inches
    scale_factor = figsize_mm[0] / figsize_mm[0]
    # add the scale factor to the rcParams
    updates_dict = {
        'pdf.fonttype': 42,
        'figure.figsize': figsize_in,
    }

    if fontsize is not None:
        updates_dict['font.size'] = fontsize 

    return updates_dict

# Add a decorator which prints the value of the function each time it is called
def print_function_call(func):
    # Get all arguments of the called functions
    argnames = func.__code__.co_varnames[:func.__code__.co_argcount] 
    # Get the names of the default arguments
    func_defaults = func.__defaults__
    # Get the length of the default arguments
    func_defaults_length = len(func_defaults) if func_defaults is not None else 0
    # Get the names of the default arguments
    func_defaults_names = argnames[-func_defaults_length:]
    # Get a dictionary of the default arguments and their values
    func_defaults_dict = dict(zip(func_defaults_names, func_defaults))
    
    # Get the names of the function
    fname = func.__name__
    print("Function name: ", fname)
    print("=====================================")
    def inner_func(*args, **kwargs):

        function_arguments_passed = {}
        for i, arg in enumerate(args):
            # if arg is an iterable then print first element and "..."
            if hasattr(arg, "__iter__"):
                # if the iterable is a string then print the whole string
                if isinstance(arg, str):
                    #print(f"{argnames[i]} = {arg}")
                    function_arguments_passed[argnames[i]] = arg
                else:
                    #print(f"{argnames[i]} = ...")
                    function_arguments_passed[argnames[i]] = "..."
            else:
                #print(f"{argnames[i]} = {arg}")
                function_arguments_passed[argnames[i]] = arg
        
        # Default arguments
        #print("Default arguments:")
        

        print("Passed arguments:")
        for key, value in function_arguments_passed.items():
            print(f"{key} = {value}")
        print("Default arguments:")
        for key, value in func_defaults_dict.items():
            print(f"{key} = {value}")


        # Return the value of the called function
        return func(*args, **kwargs)
          
        # printing the function arguments 
        # print(', '.join( '% s = % r' % entry
        #     for entry in zip(argnames, args[:len(argnames)])), end = ", ")
          
        # # Printing the variable length Arguments
        # print("args =", list(args[len(argnames):]), end = ", ")
          
        # # Printing the variable length keyword
        # # arguments
        # print("kwargs =", kwargs, end = "")
        
    return inner_func


def r2(y_fit, y_data):
    y_mean = y_data.mean()
    residual_squares = (y_data-y_fit)**2
    variance = (y_data-y_mean)**2
    
    residual_sum_of_squares = residual_squares.sum()
    sum_of_variance = variance.sum()
    
    r_squared = 1 - residual_sum_of_squares/sum_of_variance
    
    return r_squared

def crop_data_to_map(input_data_map, mask, mask_threshold, skip_zeros=True):
    from locscale.include.emmer.ndimage.map_utils import parse_input
    
    input_data_map = parse_input(input_data_map)
    mask = parse_input(mask)
    
    binarised_mask = (mask>=mask_threshold).astype(np.int_)
    flattend_array = (binarised_mask * input_data_map).flatten()
    
    nonzero_array = flattend_array[flattend_array>0]
    
    return nonzero_array

# decorate the function with the decorator

def plot_correlations(x_array, y_array,  x_label, y_label, title_text, \
                    scatter=False, figsize_cm=(14,8),font="Helvetica",fontsize=10,\
                    fontscale=1,hue=None,find_correlation=True, alpha=0.3, filepath=None):

    import matplotlib.pyplot as plt
    from matplotlib.pyplot import cm
    from locscale.include.emmer.ndimage.profile_tools import crop_profile_between_frequency
    import seaborn as sns
    from scipy import stats
    import matplotlib 
    import pandas as pd
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    # set the global font size for the plot

        
    plt.rcParams.update({'font.size': fontsize})
    figsize = (figsize_cm[0]/2.54, figsize_cm[1]/2.54) # convert cm to inches
    
    fig = plt.figure(figsize=figsize, dpi=600) # dpi=600 for publication quality
    sns.set_theme(context="paper", font=font, font_scale=fontscale)
    # Set font size for all text in the figure
    sns.set_style("white")
    
    def annotate(data, **kws):
        pearson_correlation = stats.pearsonr(x_array, y_array)
        r2_text = f"$R$ = {pearson_correlation[0]:.2f}"
        ax = plt.gca()
        ax.text(.05, .8, r2_text,transform=ax.transAxes)
    # Create a pandas dataframe for the data
    data = pd.DataFrame({x_label: x_array, y_label: y_array})
    # Plot the data    
    g = sns.lmplot(x=x_label, y=y_label, data=data, scatter=scatter)
    g.map_dataframe(annotate)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()

    if filepath is not None:
        plt.savefig(filepath, bbox_inches='tight')
    


def plot_correlations_multiple_single_plot(list_of_xy_tuple, scatter=False, hue=None, figsize=(14,8),\
                                            fontscale=1, x_label=None, y_label=None, ylims=None, \
                                            title_text=None,find_correlation=True, alpha=0.3, ci=95):
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
    
    return fig        

#    
def plot_correlations_multiple(xy_tuple, scatter=False, hue=None, figsize=(14,8),\
                                fontscale=3, x_label=None, y_label=None, ylims=None, \
                                title_text=None, find_correlation=True, alpha=0.3, ci=95):
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
    
    plt.legend(loc="lower right")   

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    if ylims is not None:
        plt.ylim(ylims)
    
        
    plt.tight_layout()
    return fig
    
#
def plot_linear_regression(data_input, x_col, y_col, x_label=None, y_label=None, title_text=None):
    import matplotlib.pyplot as plt
    import pandas as pd
    
    def linear(x,a,b):
        return a * x + b
    from matplotlib.offsetbox import AnchoredText
    fig, ax = plt.subplots(1,1)

            
    data_unsort = data_input.copy()
    data=data_unsort.sort_values(by=x_col)
    x_data = data[x_col]
    y_data = data[y_col]
    
    y_fit = x_data ## When assuming y=x as ideal equation
    
    r_squared = data[x_col].corr(data[y_col], method="spearman")
    
    ax.plot(x_data, y_data,'bo')
    ax.plot(x_data, x_data, 'r-')
    equation = "y = x \nCorrelation = {}".format(round(r_squared,2))
    legend_text = equation
    anchored_text=AnchoredText(legend_text, loc=2)
    ax.add_artist(anchored_text)
    if x_label is not None:
        ax.set_xlabel(x_label)
    else:
        ax.set_xlabel(x_col)
        
    if y_label is not None:
        ax.set_ylabel(y_label)
    else:
        ax.set_ylabel(y_col)
    ax.set_title(title_text)    
    return fig
    
def plot(plot_properties):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    sns.set_theme(context="paper", font="Helvetica", font_scale=1.5)
    sns.set_style("white")
    kwargs = dict(linewidth=3)    
    
#
def plot_radial_profile_seaborn(freq, list_of_profiles, font=16, ylims=None, crop_first=10, crop_end=1, legends=None):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    import matplotlib as mpl
    mpl.rcParams['pdf.fonttype'] = 42
    
    freq = freq[crop_first:-crop_end]
    
    sns.set_theme(context="paper", font="Helvetica", font_scale=1.5)
    sns.set_style("white")
    kwargs = dict(linewidth=3)

    profile_list = np.array(list_of_profiles)
    average_profile = np.einsum("ij->j", profile_list) / len(profile_list)

    variation = []
    for col_index in range(profile_list.shape[1]):
        col_extract = profile_list[:,col_index]
        variation.append(col_extract.std())

    variation = np.array(variation)
        
    y_max = average_profile + variation
    y_min = average_profile - variation
    
    fig, ax = plt.subplots()
    ax = sns.lineplot(x=freq, y=average_profile[crop_first:-crop_end], **kwargs)
    ax.fill_between(freq, y_min[crop_first:-crop_end], y_max[crop_first:-crop_end], alpha=0.3)
    ax.set_xlabel('Spatial Frequency $1/d [\AA^{-1}]$',fontsize=font)
    ax.set_ylabel('$\mid F \mid $',fontsize=font)
    if legends is not None:
        ax.legend(legends)
    ax2 = ax.twiny()
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xbound(ax.get_xbound())
    ax2.set_xticklabels([round(1/x,1) for x in ax.get_xticks()])
    ax2.set_xlabel('$d [\AA]$',fontsize=font)
    if ylims is not None:
        plt.ylim(ylims)
    plt.tight_layout()
    plt.show()
    
    return fig

def pretty_lineplot_XY(xdata, ydata, xlabel, ylabel, figsize_mm=(14,8), \
                        marker="o", markersize=12,fontscale=2.5,font="Helvetica", \
                        linewidth=2,legends=None):
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import cm
    import seaborn as sns    
    from matplotlib.pyplot import cm
    import matplotlib as mpl
    ## Function not generic
    mpl.rcParams['pdf.fonttype'] = 42
    figsize = (figsize_mm[0]/25.4, figsize_mm[1]/25.4)
    fig, ax = plt.subplots(figsize=figsize)
    sns.set_theme(context="paper", font=font, font_scale=fontscale)
    sns.set_style("white")
    sns.lineplot(x=xdata,y=ydata,linewidth=linewidth,marker=marker,markersize=markersize, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, rotation=90, ha="center")

    if legends is not None:        
        ax.legend(legends)
    plt.tight_layout()

    return fig

def find_freq_value_where_fsc_drops_below_threshold(freq, *fsc_curves, threshold=0.5):
    from scipy.interpolate import interp1d
    from scipy.ndimage import uniform_filter1d
    freq_values = []
    for fsc_curve in fsc_curves:
        xarray = np.array(fsc_curve)
        yarray = np.array(freq)
        xarray_smoothed = uniform_filter1d(xarray, size=5)
        g = interp1d(xarray_smoothed, yarray)
        freq_values.append(g(threshold))
        f = interp1d(yarray, xarray_smoothed)
    return freq_values


def pretty_lineplot_multiple_fsc_curves(fsc_arrays_perturb, two_xaxis=True, figsize=(14,8),\
                                        fontscale=2.5,font="Helvetica",linewidth=2,legends=None):
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import cm
    import seaborn as sns    
    from matplotlib.pyplot import cm
    import matplotlib as mpl
    ## Function not generic
    mpl.rcParams['pdf.fonttype'] = 42
    fig = plt.figure()
    sns.set(rc={'figure.figsize':figsize})
    sns.set_theme(context="paper", font=font, font_scale=fontscale)
    sns.set_style("white")
    colors_rainbow = cm.rainbow(np.linspace(0,1,len(fsc_arrays_perturb.keys())))
    
    if two_xaxis:
        # print(';)
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.grid(False)
        
        
        for i,rmsd in enumerate(fsc_arrays_perturb.keys()):
            sns.lineplot(x=fsc_arrays_perturb[rmsd][0],y=fsc_arrays_perturb[rmsd][1], linewidth=linewidth, color=colors_rainbow[i], ax=ax1)
            ax1.set_xlabel(r" Spatial Frequency, $d^{-1}(\AA^{-1}$)")
            ax1.set_ylabel("FSC")
        
        if legends is not None:        
            ax1.legend(legends)
        
        ax2 = ax1.twiny()
        ax2.set_xticks(ax1.get_xticks())
        ax2.set_xbound(ax1.get_xbound())        
        ax2.set_xticklabels([round(1/x,1) for x in ax1.get_xticks()])            
        ax2.set_xlabel(r'Resolution, $d (\AA)$')
        
        if legends is not None:   
            print("Legends print")
            plt.legend(legends)
    else:
        for i,rmsd in enumerate(fsc_arrays_perturb.keys()):
            sns.lineplot(x=fsc_arrays_perturb[rmsd][0],y=fsc_arrays_perturb[rmsd][1], linewidth=linewidth, color=colors_rainbow[i])
            plt.xlabel(r" Spatial Frequency, $d^{-1}(\AA^{-1}$)")
            plt.ylabel("FSC")
    

    plt.tight_layout()
    return fig
    
#
def pretty_violinplots(list_of_series, xticks, ylabel,xlabel=None, figsize=(14,8),\
                        fontscale=3,font="Helvetica",linewidth=2):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import cm
    import matplotlib as mpl
    ## Function not generic
    mpl.rcParams['pdf.fonttype'] = 42
    fig = plt.figure()
    sns.set(rc={'figure.figsize':figsize})
    sns.set_theme(context="paper", font=font, font_scale=fontscale)
    sns.set_style("white")
    
    ax = sns.violinplot(data=list_of_series, scale_hue=False)
    ax.set_xticklabels(xticks)
    ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    
    plt.tight_layout()
    return fig
        
#
def pretty_boxplots(list_of_series, xticks, ylabel,xlabel=None, figsize_cm=(14,8),\
                    fontscale=3,font="Helvetica",linewidth=2):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import cm
    import matplotlib
    
    ## Headers
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    # set the global font size for the plot

        
    plt.rcParams.update({'font.size': fontsize})
    figsize = (figsize_cm[0]/2.54, figsize_cm[1]/2.54) # convert cm to inches
    
    fig, ax1 = plt.subplots(figsize=figsize, dpi=600)  # DPI is fixed to 600 for publication quality
    sns.set_theme(context="paper", font=font, font_scale=fontscale)
    # Set font size for all text in the figure
    sns.set_style("white")

    ## Plot the data


    ax1.boxplot(list_of_series)
    ax1.set_xticklabels(xticks)
    ax1.set_ylabel(ylabel)
    if xlabel is not None:
        ax1.set_xlabel(xlabel)
    fig.tight_layout()
    
    return fig

#    
def pretty_plot_radial_profile(freq,list_of_profiles_native, plot_type="make_log", \
                                legends=None,figsize_mm=(14,8), fontsize=10,linewidth=1, \
                                marker="o", markersize=5,font="Helvetica",fontscale=1, showlegend=True, showPoints=False, \
                                alpha=1, variation=None, yticks=None, ylims=None, xlims=None, crop_freq=None, labelsize=None, title=None, xticks=None):
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import cm
    from locscale.include.emmer.ndimage.profile_tools import crop_profile_between_frequency
    import seaborn as sns
    import matplotlib 
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    # set the global font size for the plot

        
    plt.rcParams.update({'font.size': fontsize})
    figsize = (figsize_mm[0]/25.4, figsize_mm[1]/25.4) # convert cm to inches
    
    fig, ax1 = plt.subplots(figsize=figsize, dpi=600)  # DPI is fixed to 600 for publication quality
    sns.set_theme(context="paper", font=font, font_scale=fontscale)
    # Set font size for all text in the figure
    sns.set_style("white")

    if isinstance(freq, list):
        list_of_freq = freq
    else:
        list_of_freq = [freq]*len(list_of_profiles_native)
    
    # Crop frequencies if required
    if crop_freq is not None:
        cropped_frequency_list = [crop_profile_between_frequency(f, list_of_profiles_native[0], crop_freq[0], crop_freq[1])[0] for f in list_of_freq]
        cropped_profiles = [crop_profile_between_frequency(list_of_freq[0], profile, crop_freq[0], crop_freq[1])[1] for profile in list_of_profiles_native]
    else:
        cropped_frequency_list = list_of_freq
        cropped_profiles = list_of_profiles_native
    
    final_list_of_profiles = []

    for profile in cropped_profiles:
        if plot_type=="make_log":
            profile = np.log(profile)
            plot_frequency_axis_list = [cropped_frequency**2 for cropped_frequency in cropped_frequency_list]
        elif plot_type=="squared_amp":
            profile = np.log(profile**2)
            plot_frequency_axis_list = [cropped_frequency**2 for cropped_frequency in cropped_frequency_list]
        elif plot_type=="normalise":
            profile = profile/profile.max()
            plot_frequency_axis_list = cropped_frequency_list
        else:
            plot_frequency_axis_list = cropped_frequency_list        
    
        final_list_of_profiles.append(profile)
        
    
    # Add labels to the plot
    xlabel_top = r'Resolution, $d (\AA)$'
    if plot_type=="normalise":
        xlabel = r'Spatial Frequency, $d^{-1} (\AA^{-1})$'
        ylabel = r'Normalised $ \langle \mid F \mid \rangle $'
    elif plot_type=="squared_amp":
        xlabel = r'Spatial Frequency, $d^{-2} (\AA^{-2})$'
        ylabel = r'$ln  \langle \mid F \mid ^{2} \rangle $ '
    elif plot_type=="make_log":
        xlabel = r'Spatial Frequency, $d^{-2} (\AA^{-2})$'
        ylabel = r'$ln  \langle \mid F \mid \rangle $'
    else:
        xlabel = r'Spatial Frequency, $d^{-1} (\AA^{-1})$'
        ylabel = r'$ \langle \mid F \mid \rangle $'
    # Map the colors
    
    #colors = cm.rainbow(np.linspace(0,1,len(final_list_of_profiles)))
    colors = cm.turbo(np.linspace(0,1,len(final_list_of_profiles)))
    
    ax1.grid(False)
    ax2 = ax1.twiny()

    for i, profile in enumerate(final_list_of_profiles):
        if showPoints:
            ax1.plot(plot_frequency_axis_list[i], profile, marker=marker, markersize=markersize, color=colors[i], alpha=alpha, \
                        linewidth=linewidth, label=legends[i])
        else:
            ax1.plot(plot_frequency_axis_list[i], profile, color=colors[i], alpha=alpha, linewidth=linewidth, label=legends[i])

    if xticks is not None:
        ax2.set_xticks(xticks)
        ax2.set_xticklabels([round(x,1) for x in xticks])
    else:                
        ax2.set_xticks(ax1.get_xticks())
        ax2.set_xbound(ax1.get_xbound())
        ax2.set_xticklabels([round(1/np.sqrt(x),1) for x in ax1.get_xticks()])
    #ax2.tick_params(axis="both", which="both", labelsize=labelsize)

    if showlegend:
        ax1.legend(loc="best")
    ax1.set_xlabel(xlabel)#, fontsize=fontsize)
    ax1.set_ylabel(ylabel)#, fontsize=fontsize)
    #ax1.tick_params(axis="both", which="both", labelsize=labelsize)
    ax2.set_xlabel(xlabel_top)#, fontsize=fontsize)
    
    if ylims is not None:
        plt.ylim(ylims)
    if yticks is not None:
        plt.yticks(yticks)
  
    if xlims is not None:
        plt.xlim(xlims)

    if title is not None:
        plt.title(title)
    plt.tight_layout()
    return fig, ax1, ax2

def pretty_plot_fsc_curve(freq,list_of_profiles_native,  \
                                legends=None, figsize_mm=(14,8), fontsize=10,linewidth=1, \
                                marker="o", markersize=5,font="Helvetica",fontscale=1, showlegend=True, showPoints=False, \
                                alpha=1, xlabelpad=None, ylabelpad=None, yticks=None, ylims=None, xlims=None, crop_freq=None, labelsize=None, title=None):
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import cm
    from locscale.include.emmer.ndimage.profile_tools import crop_profile_between_frequency
    import seaborn as sns
    import matplotlib 
    import os
        # Convert cm to inches for figsize
    figsize = (figsize_mm[0] / 25.4, figsize_mm[1] / 25.4)
    rcParams_updates = configure_plot_scaling(figsize_mm)
    rcParams_updates['legend.fontsize'] *= 0.8
    rcParams_updates['lines.linewidth'] *= 0.5
    rcParams_updates['lines.markersize'] *= 0.8
    rcParams_updates['legend.borderpad'] *= 0.2
    rcParams_updates['legend.fontsize'] *= 0.3

    with temporary_rcparams(rcParams_updates):
        fig, ax1 = plt.subplots(figsize=figsize, dpi=600)  # DPI is fixed to 600 for publication quality
        sns.set_theme(context="paper", font=font)
        sns.set_style("white")

        # Crop frequencies if required
        if crop_freq is not None:
            cropped_frequency = crop_profile_between_frequency(freq, list_of_profiles_native[0], crop_freq[0], crop_freq[1])[0]
            cropped_profiles = [crop_profile_between_frequency(freq, profile, crop_freq[0], crop_freq[1])[1] for profile in list_of_profiles_native]
        else:
            cropped_frequency = freq
            cropped_profiles = list_of_profiles_native
        
        final_list_of_profiles = cropped_profiles
        plot_frequency_axis = cropped_frequency
        
        # Add labels to the plot
        xlabel_top = r'Resolution, $d (\AA)$'

        xlabel = r'Spatial Frequency, $d^{-1} (\AA^{-1})$'
        ylabel = r'FSC'
        
        colors = cm.rainbow(np.linspace(0,1,len(final_list_of_profiles)))
        
        ax1.grid(False)
        ax2 = ax1.twiny()

        for i, profile in enumerate(final_list_of_profiles):
            if showPoints:
                ax1.plot(plot_frequency_axis, profile, marker=marker, color=colors[i], alpha=alpha, \
                        label=legends[i])
            else:
                ax1.plot(plot_frequency_axis, profile, color=colors[i], alpha=alpha, label=legends[i])
                    
        ax2.set_xticks(ax1.get_xticks())
        ax2.set_xbound(ax1.get_xbound())
        ax2.set_xticklabels([round(1/x,1) for x in ax1.get_xticks()])
        #ax2.tick_params(axis="both", which="both", labelsize=labelsize)

        if showlegend:
            ax1.legend(loc="upper right")
        ax1.set_xlabel(xlabel)#, fontsize=fontsize)
        ax1.set_ylabel(ylabel)#, fontsize=fontsize)
        #ax1.tick_params(axis="both", which="both", labelsize=labelsize)
        ax2.set_xlabel(xlabel_top)#, fontsize=fontsize)

        if yticks is not None:
            ax1.set_yticks(yticks)
            ax1.set_yticklabels(yticks)
            ax2.set_yticks(yticks)
            ax2.set_yticklabels(yticks)
        
        if ylims is not None:
            plt.ylim(ylims)
        # if yticks is not None:
        #     plt.yticks(yticks)
        if xlims is not None:
            plt.xlim(xlims)

        if title is not None:
            plt.title(title)
        
        if xlabelpad is not None:
            ax1.xaxis.labelpad = xlabelpad
        if ylabelpad is not None:
            ax1.yaxis.labelpad = ylabelpad

        plt.tight_layout()
    return fig

#    
def pretty_plot_continious_radial_profiles(freq,list_of_profiles_native,normalise=False, squared_amplitudes=True, \
                                legends=None,figsize=(14,8), fontsize=14,linewidth=1, \
                                marker="o", font="Helvetica",fontscale=1, showlegend=True, showPoints=False, \
                                alpha=0.05, variation=None, yticks=None, logScale=True, ylims=None, xlims=None, crop_freq=None):
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import cm
    from locscale.include.emmer.ndimage.profile_tools import crop_profile_between_frequency
    import seaborn as sns

    #import matplotlib as mpl
    #mpl.rcParams['pdf.fonttype'] = 42
    
    fig, ax1 = plt.subplots(figsize=figsize)
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
    ax1.grid(False)
    ax2 = ax1.twiny()

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


def pretty_plot_series(x_array,list_of_y_array, figsize_cm=(14,8), fontsize=10,linewidth=1, \
                                font="Helvetica",fontscale=1, alpha=0.2, xticks=None, num_xticks=5, yticks=None, ylims=None, xlims=None, labelsize=None, title=None):
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import cm
    from locscale.include.emmer.ndimage.profile_tools import crop_profile_between_frequency
    import seaborn as sns
    import matplotlib 
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    # set the global font size for the plot

        
    plt.rcParams.update({'font.size': fontsize})
    figsize = (figsize_cm[0]/2.54, figsize_cm[1]/2.54) # convert cm to inches
    
    fig, ax1 = plt.subplots(figsize=figsize, dpi=600)  # DPI is fixed to 600 for publication quality
    sns.set_theme(context="paper", font=font, font_scale=fontscale)
    # Set font size for all text in the figure
    sns.set_style("white")

    """
    Plots a series of y arrays with a common x array. 
    """
    mean_y = np.mean(list_of_y_array, axis=0)
    std_y = np.std(list_of_y_array, axis=0)
    extreme_y_min = mean_y - std_y
    extreme_y_max = mean_y + std_y

    if len(list_of_y_array) > 1:
        ax1.plot(x_array, mean_y, color='black', linewidth=linewidth)
        ax1.fill_between(x_array, extreme_y_min, extreme_y_max, color='grey', alpha=alpha)
    else:
        ax1.plot(x_array, list_of_y_array[0], color='black', linewidth=linewidth)

    if xticks is not None:
        ax1.set_xticks([round(xtick,2) for xtick in xticks])
    else:
        ax1.set_xticks(np.linspace(x_array[0], x_array[-1], num_xticks).round(2))
    if yticks is not None:
        ax1.set_yticks(yticks)
    if ylims is not None:
        ax1.set_ylim(ylims)
    if xlims is not None:
        ax1.set_xlim(xlims)
    if labelsize is not None:
        ax1.tick_params(axis='both', which='major', labelsize=labelsize)

    ax1.set_xlabel(r'Spatial frequency ($\AA^{-1}$)')
    ax1.set_ylabel('Phase correlation')

    # Obtain a second X axis with the resolution in Angstroms
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(ax1.get_xticks())
    ax2.set_xticklabels(np.round(1/ax1.get_xticks(), 1))
    ax2.set_xlabel(r'Resolution ($\AA$)')

    if title is not None:
        ax1.set_title(title)

    plt.tight_layout()

    return fig

def pretty_plot_rainbow_series(x_array, list_of_y_array, figsize_mm=(14, 8), fontsize=10, linewidth=1,
                       font="Helvetica", xlabel=None, ylabel=None, xlabel2= None, xticks=None, num_xticks=5, 
                       yticks=None, ylims=None, xlims=None, labelsize=None, title=None):
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import cm
    import numpy as np
    import seaborn as sns
    import matplotlib
    
    # Convert cm to inches for figsize
    figsize = (figsize_mm[0] / 25.4, figsize_mm[1] / 25.4)
    rcParams_updates = configure_plot_scaling(figsize_mm)
    # increase yticks width for better visibility
    rcParams_updates['ytick.major.width'] *= 2
    rcParams_updates['ytick.minor.width'] *= 2
    with temporary_rcparams(rcParams_updates):
        # DPI is fixed to 600 for publication quality
        fig, ax1 = plt.subplots(figsize=figsize, dpi=600)
        sns.set_theme(context="paper", font=font)
        sns.set_style("white")

        # Set the color palette
        color = cm.rainbow(np.linspace(0, 1, len(list_of_y_array)))
        
        for idx, y in enumerate(list_of_y_array):
            ax1.plot(x_array, y, color=color[idx])

        # Set x and y ticks if provided
        if xticks is not None:
            ax1.set_xticks([round(xtick, 1) for xtick in xticks])
        else:
            ax1.set_xticks(np.linspace(x_array[0], x_array[-1], num_xticks).round(1))
        if yticks is not None:
            ax1.set_yticks(yticks)

        # Set x and y limits if provided
        if ylims is not None:
            ax1.set_ylim(ylims)
        if xlims is not None:
            ax1.set_xlim(xlims)
        # Set the axis labels
        if xlabel is None:
            ax1.set_xlabel(r'Spatial frequency ($\AA^{-1}$)')
        else:
            ax1.set_xlabel(xlabel)
        if ylabel is None:
            ax1.set_ylabel('Phase correlation')
        else:
            ax1.set_ylabel(ylabel)

        # Add a second X axis for resolution
        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xticks(ax1.get_xticks())
        ax2.set_xticklabels(np.round(1 / ax1.get_xticks(), 1))
        if xlabel2 is None:
            ax2.set_xlabel(r'Resolution ($\AA$)')
        else:
            ax2.set_xlabel(xlabel2)

        # Ensure ticks are visible
        for tick in ax1.get_xticklabels():
            tick.set_visible(True)
        for tick in ax1.get_yticklabels():
            tick.set_visible(True)
        for tick in ax2.get_xticklabels():
            tick.set_visible(True)
        for tick in ax2.get_yticklabels():
            tick.set_visible(True)

        if title is not None:
            ax1.set_title(title)

        plt.tight_layout()

    return fig


def jsonify_dictionary(input_dict):
    # convert pickle object to json object
    new_dict = {}
    for key, value in input_dict.items():
        key = str(key) 
        value_is_iterable = isinstance(value, (list, tuple, np.ndarray))
        value_is_dict = isinstance(value, dict)
        value_is_float = isinstance(value, float)
        value_is_int = isinstance(value, (np.int64, int, np.int32))
        value_is_string = isinstance(value, str)
        
        print("key: {}, value_is_iterable: {}, value_is_dict: {}, value_is_float: {}, \
              value_is_int: {}, value_is_string: {}".format(key, value_is_iterable, \
                                                            value_is_dict, value_is_float, value_is_int, value_is_string))
        
        if value_is_dict:
            new_value = jsonify_dictionary(value)
        elif value_is_iterable:
            new_value = [str(x) for x in value]
        elif not value_is_string:
            new_value = str(value)
        
        new_dict[key] = new_value
        
    
    return new_dict         

def pretty_plot_confidence_interval(x_array, *list_of_y_arrays, confidence_interval=95,  \
                                    figsize_cm=(14,8), fontsize=10,linewidth=1, \
                                font="Helvetica",fontscale=1, alpha=0.2, xticks=None, \
                                num_xticks=5, yticks=None, ylims=None, xlims=None, labelsize=None, title=None, \
                                xlabel=None, ylabel=None, showlegend=True):
    
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import cm
    from locscale.include.emmer.ndimage.profile_tools import crop_profile_between_frequency
    import seaborn as sns
    import matplotlib 
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    # set the global font size for the plot
    from scipy import stats
        
    plt.rcParams.update({'font.size': fontsize})
    figsize = (figsize_cm[0]/2.54, figsize_cm[1]/2.54) # convert cm to inches
    
    fig, ax1 = plt.subplots(figsize=figsize, dpi=600)  # DPI is fixed to 600 for publication quality
    sns.set_theme(context="paper", font=font, font_scale=fontscale)
    # Set font size for all text in the figure
    sns.set_style("white")
    
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'cyan', 'magenta', 'yellow']
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    for idx, y in enumerate(list_of_y_arrays):
        y = np.array(y)

        # Calculate the means and standard error
        mean_y = np.mean(y, axis=0)
        stderr_y = stats.sem(y, axis=0)

        # Calculate the t-values for the given confidence interval
        df = len(y) - 1
        t_value = stats.t.ppf((1 + confidence_interval/100) / 2, df)
        margin_of_error = stderr_y * t_value

        # Plot
        ax.plot(x_array, mean_y, color=colors[idx % len(colors)], alpha=1, label=f"Curve {idx + 1}", linewidth=linewidth)
        ax.fill_between(x_array, mean_y - margin_of_error, mean_y + margin_of_error, color=colors[idx % len(colors)], alpha=alpha)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize)
        
    if xticks is not None:
        ax.set_xticks([round(xtick,2) for xtick in xticks])
    else:
        ax.set_xticks(np.linspace(x_array[0], x_array[-1], num_xticks).round(2))
    
    if yticks is not None:
        ax.set_yticks(yticks)
    if ylims is not None:
        ax.set_ylim(ylims)
    if xlims is not None:
        ax.set_xlim(xlims)
    if labelsize is not None:
        ax.tick_params(axis='both', which='major', labelsize=labelsize)
    if title is not None:
        ax.set_title(title)
    if showlegend:
        ax.legend(loc="best")
    plt.tight_layout()
    
    return fig, ax



def create_colormap_rectangle(rect_width_mm=20, rect_height_mm=80):
    """
    Creates a rectangle filled with the Turbo colormap, mapping values from 1 to 50.

    Parameters:
    rect_width_mm (float): Width of the rectangle in millimeters.
    rect_height_mm (float): Height of the rectangle in millimeters.

    Returns:
    fig, ax: Matplotlib figure and axis objects.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import cm
    # Convert dimensions from millimeters to inches (1 mm = 0.0393701 inches)
    rect_width_in = rect_width_mm * 0.0393701
    rect_height_in = rect_height_mm * 0.0393701

    # Create figure and axis with specified dimensions
    fig, ax = plt.subplots(figsize=(rect_width_in, rect_height_in))

    # Generate a linear gradient from 1 to 50
    gradient = np.linspace(1, 50, 256).reshape(-1, 1)  # 256 points from 1 to 50

    # Display the gradient using imshow with the Turbo colormap
    ax.imshow(gradient, aspect='auto', cmap=cm.turbo, extent=[0, rect_width_mm, 1, 50])

    # Customize the axis
    ax.set_xlim(0, rect_width_mm)
    ax.set_ylim(1, 50)
    ax.set_xticks([])  # Hide x-axis ticks
    ax.set_ylabel("Values", fontsize=10)  # Y-axis label
    ax.tick_params(axis='y', labelsize=8)  # Adjust y-axis tick label size

    # Remove spines (borders) for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Adjust layout to fit elements properly
    plt.tight_layout()
    plt.show()

    return fig, ax

def plot_ridgeplot(values_dictionary, sorting_dictionary, ax=None, clabel="Label", xlabel="X-axis", ylabel="Y-axis"):
    """
    Plots a ridge plot for given values and sorting dictionaries.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    common_keys = list(set(values_dictionary.keys()) & set(sorting_dictionary.keys()))
    if len(common_keys) == 0:
        raise ValueError("No common keys found between values_dictionary and sorting_dictionary.")
    
    # Sort keys by increasing order of sorting values
    common_keys.sort(key=lambda x: sorting_dictionary[x])

    # Normalize sorting values for coloring
    min_sort, max_sort = min(sorting_dictionary.values()), max(sorting_dictionary.values())
    print(min_sort, max_sort)
    norm_sorting = [(sorting_dictionary[e] - min_sort) / (max_sort - min_sort) if max_sort > min_sort else 0.5 for e in common_keys]

    # Generate colors using turbo colormap
    colors = cm.turbo(norm_sorting)

    # X-axis (assuming all values have the same length)
    max_x_value = max([max(values_dictionary[e]) for e in common_keys])
    x = np.linspace(0, max_x_value, len(values_dictionary[common_keys[0]]))

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each value with an increasing vertical offset
    offset = 0
    spacing = 2  # Adjust this if necessary

    for i, key in enumerate(common_keys):
        y_offset = offset + i * spacing
        ax.plot(x, values_dictionary[key] + y_offset, color=colors[i])
    
    # Formatting
    ax.set_yticks([0, 0.5, 1])  # Hide y-axis labels
    ax.set_xticks([0, max_x_value / 2, max_x_value])  # Set x-axis ticks

    # Create a colorbar
    sm = cm.ScalarMappable(cmap="turbo", norm=plt.Normalize(min_sort, max_sort))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label=clabel)

    cbar.ax.yaxis.set_ticks([0.3, 0.6, 0.9])

    # Hide tick marks but not labels
    cbar.ax.tick_params(size=0)

    # Set axis labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if ax is None:
        plt.show()
    else:
        return ax
    




def plot_kde_ridge(f1_scores, kde_values, means=None, ax=None, clabel="Label", xlabel="X-axis", ylabel="Y-axis"):
    """
    Plots KDE ridge plots for given f1_scores and kde_values.
    
    Args:
    f1_scores (dict): {emdb_id (int): f1_score (float)}
    kde_values (dict): {emdb_id (str): kde_array (1D numpy array of length 1000)}
    
    Returns:
    None (displays the plot)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    # Convert EMDB IDs in kde_values to int for matching
    kde_values = {int(k): v for k, v in kde_values.items()}
    f1_scores = {int(k): v for k, v in f1_scores.items()}
    means = {int(k): v for k, v in means.items()} if means is not None else None
    # Ensure all EMDB IDs are in both dictionaries
    common_ids = list(set(f1_scores.keys()) & set(kde_values.keys()))
    if len(common_ids) == 0:
        print("No matching EMDB IDs found between f1_scores and kde_values.")
        return
    
    # Sort EMDB IDs by increasing F1 score
    common_ids.sort(key=lambda x: f1_scores[x])

    # Normalize F1 scores for coloring
    min_f1, max_f1 = min(f1_scores.values()), max(f1_scores.values())
    norm_f1_scores = [(f1_scores[e] - min_f1) / (max_f1 - min_f1) if max_f1 > min_f1 else 0.5 for e in common_ids]
    
    # Generate colors using turbo colormap
    colors = cm.turbo(norm_f1_scores)

    # X-axis (assuming all KDE arrays have the same length of 1000)
    x = np.linspace(0, 1, 1000)  # Assume this is the voxel intensity range
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each KDE with an increasing vertical offset
    offset = 0
    spacing = 2  # Adjust this if necessary

    for i, emdb_id in enumerate(common_ids):
        y_offset = offset + i * spacing
        ax.plot(x, kde_values[emdb_id] + y_offset, color=colors[i])
        # compute mean value by integrating the KDE 
        if means is not None and emdb_id in means:
            mean_value = means[emdb_id]
            ax.scatter(mean_value, y_offset, color='black', s=10, marker='x')
            print(f"EMDB: {emdb_id} & F1 score: {f1_scores[emdb_id]} & Mean KDE value: {mean_value:.2f}")
        else:
            print(f"EMDB: {emdb_id} & F1 score: {f1_scores[emdb_id]}")

    # Formatting
    ax.set_yticks([])  # Hide y-axis labels
    ax.set_xticks([0, 0.5, 1])  # Set x-axis ticks

    # Create a colorbar to the left of the plot
    sm = cm.ScalarMappable(cmap="turbo", norm=plt.Normalize(min_f1, max_f1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label=clabel, location="left")

    cbar.ax.yaxis.set_ticks([0.3, 0.6, 0.9])

    # Hide tick marks but not labels
    cbar.ax.tick_params(size=0)

    # Set axis labels
    ax.set_xlabel(xlabel)
    ax.set_title(ylabel)

    if ax is None:
        plt.show()
    else:
        return ax










################ GRAVEYARD ################


# def pretty_plot_rainbow_series(x_array, list_of_y_array, figsize_mm=(14, 8), fontsize=10, linewidth=1,
#                        font="Helvetica", fontscale=1, alpha=0.2, xticks=None, num_xticks=5, 
#                        yticks=None, ylims=None, xlims=None, labelsize=None, title=None):
#     import matplotlib.pyplot as plt
#     from matplotlib.pyplot import cm
#     import numpy as np
#     import seaborn as sns
#     import matplotlib
    
#     # Convert cm to inches for figsize
#     figsize = (figsize_mm[0] / 25.4, figsize_mm[1] / 25.4)
#     rcParams_updates = configure_plot_scaling(figsize_mm)

#     with temporary_rcparams(rcParams_updates):
#         # DPI is fixed to 600 for publication quality
#         fig, ax1 = plt.subplots(figsize=figsize, dpi=600)
#         sns.set_theme(context="paper", font=font, font_scale=fontscale)
#         sns.set_style("white")

#         # Set the color palette
#         color = cm.rainbow(np.linspace(0, 1, len(list_of_y_array)))
        
#         for idx, y in enumerate(list_of_y_array):
#             ax1.plot(x_array, y, color=color[idx], linewidth=linewidth)

#         # Set x and y ticks if provided
#         if xticks is not None:
#             ax1.set_xticks([round(xtick, 2) for xtick in xticks])
#         else:
#             ax1.set_xticks(np.linspace(x_array[0], x_array[-1], num_xticks).round(2))
#         if yticks is not None:
#             ax1.set_yticks(yticks)

#         # Set x and y limits if provided
#         if ylims is not None:
#             ax1.set_ylim(ylims)
#         if xlims is not None:
#             ax1.set_xlim(xlims)

#         # Set label size if provided
#         if labelsize is not None:
#             ax1.tick_params(axis='both', which='major', labelsize=labelsize)

#         # Set the axis labels
#         ax1.set_xlabel(r'Spatial frequency ($\AA^{-1}$)')
#         ax1.set_ylabel('Phase correlation')

#         # Add a second X axis for resolution
#         ax2 = ax1.twiny()
#         ax2.set_xlim(ax1.get_xlim())
#         ax2.set_xticks(ax1.get_xticks())
#         ax2.set_xticklabels(np.round(1 / ax1.get_xticks(), 1))
#         ax2.set_xlabel(r'Resolution ($\AA$)')

#         if title is not None:
#             ax1.set_title(title)

#         plt.tight_layout()

#     return fig
