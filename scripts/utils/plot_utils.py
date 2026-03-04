
## Plot tools 
import numpy as np
import seaborn as sns
from contextlib import contextmanager
from matplotlib import rcParams

@contextmanager
def temporary_rcparams(updates):
    original_rcparams = rcParams.copy()
    # print font families present in matplotlib
    
    try:
        rcParams.update(updates)
        yield
    finally:
        rcParams.update(original_rcparams)

def configure_plot_scaling(figsize_mm, fontsize=None):
    #import seaborn as sns
    import os
    from matplotlib import font_manager as fm
    arial_font_path = "/home/abharadwaj1/.fonts/linux-fonts/arial.ttf"
    if os.path.exists(arial_font_path):
        fm.fontManager.addfont(arial_font_path)

    figsize_in = (figsize_mm[0] / 25.4, figsize_mm[1] / 25.4)  # Convert mm to inches
    #sns.set_theme(font="Liberation Sans")
    updates_dict = {
        # font
        'pdf.fonttype': 42,
        'figure.figsize': figsize_in,

        # dpi
        'figure.dpi': 600,
        # font 
        # 'font.family': 'sans-serif',
        # 'font.sans-serif': ['Arial'],
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

def plot_correlations(x_array, y_array,  x_label, y_label, title_text=None, \
                    scatter=False, figsize_cm=(14,8),font="Helvetica",fontsize=10,\
                    xticks=None, yticks=None, xlims=None, ylims=None,\
                    fontscale=1,hue=None,find_correlation=True, alpha=0.3, filepath=None):

    import matplotlib.pyplot as plt
    from matplotlib.pyplot import cm
    from locscale.include.emmer.ndimage.profile_tools import crop_profile_between_frequency
    import seaborn as sns
    from scipy import stats
    import matplotlib 
    import pandas as pd

    # set the global font size for the plot

        
    plt.rcParams.update({'font.size': fontsize})
    figsize = (figsize_cm[0]/2.54, figsize_cm[1]/2.54) # convert cm to inches
    fig = plt.figure(figsize=figsize, dpi=600) # dpi=600 for publication quality

    def annotate(data, **kws):
        pearson_correlation = stats.pearsonr(x_array, y_array)
        print("Pearson correlation: ", pearson_correlation)
        r2_text = f"$R$ = {pearson_correlation[0]:.3f}"
        ax = plt.gca()
        ax.text(.05, .8, r2_text,transform=ax.transAxes)
    # Create a pandas dataframe for the data
    data = pd.DataFrame({x_label: x_array, y_label: y_array})
    # Plot the data    
    g = sns.lmplot(x=x_label, y=y_label, data=data, scatter=scatter, ci=95, markers=".")
    g.map_dataframe(annotate)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if title_text is not None:
        plt.title(title_text)
    if xlims is not None:
        plt.xlim(xlims)
    if ylims is not None:
        plt.ylim(ylims)
    if xticks is not None:
        plt.xticks(xticks)
    if yticks is not None:
        plt.yticks(yticks)

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
                        linewidth=2,legends=None, yticks=None, ylim=None, save_path=None):
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import cm
    import seaborn as sns    
    from matplotlib.pyplot import cm
    import matplotlib as mpl
    ## Function not generic
    figsize = (figsize_mm[0]/25.4, figsize_mm[1]/25.4)
    fig, ax = plt.subplots(figsize=figsize, dpi=600)

    sns.lineplot(x=xdata,y=ydata,linewidth=linewidth,marker=marker,markersize=markersize, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, rotation=90, ha="center")
    if yticks is not None:
        plt.yticks(yticks)
    if ylim is not None:
        plt.ylim(ylim)
    if legends is not None:        
        ax.legend(legends)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

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


def pretty_lineplot_XY_multiple(xdata_list, ydata_list, xlabel, ylabel, figsize_cm=(14,8),fontsize=10, \
                        marker="o", markersize=3,fontscale=1,font="Helvetica", \
                        linewidth=1,legends=None, title=None, yticks=None, ylim=None, save_path=None):
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import cm
    from locscale.include.emmer.ndimage.profile_tools import crop_profile_between_frequency
    import seaborn as sns
    import matplotlib 

    figsize = (figsize_cm[0]/2.54, figsize_cm[1]/2.54) # convert cm to inches
    
    fig, ax = plt.subplots(figsize=figsize, dpi=600)  # DPI is fixed to 600 for publication quality
    # Set font size for all text in the figure
    i = 0
    for xdata, ydata in zip(xdata_list, ydata_list):
        sns.lineplot(x=xdata,y=ydata,linewidth=linewidth, ax=ax, label=legends[i])
        i += 1
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, rotation=90, ha="center")
    # Show the legend
    if legends is not None:
        ax.legend(loc="best", fontsize=fontsize)
    plt.tight_layout()
    if ylim is not None:
        plt.ylim(ylim)
    if yticks is not None:
        plt.yticks(yticks)

    if title is not None:
        plt.title(title)
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')


    return fig

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
    rcParams_updates.update({
        'legend.fontsize': fontsize,
        'lines.linewidth': linewidth,
        'lines.markersize': markersize,
        'legend.borderpad': 0.2,
        'legend.fontsize': 0.3
    })
    # rcParams_updates['legend.fontsize'] *= 0.8
    # rcParams_updates['lines.linewidth'] *= 0.5
    # rcParams_updates['lines.markersize'] *= 0.8
    # rcParams_updates['legend.borderpad'] *= 0.2
    # rcParams_updates['legend.fontsize'] *= 0.3

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
    #rcParams_updates['ytick.major.width'] *= 2
    #rcParams_updates['ytick.minor.width'] *= 2
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
        xticklabels = [round(1 / x, 1) for x in ax1.get_xticks()]
        xticklabels[0] = f"$\infty$"
        ax2.set_xticklabels(xticklabels)
        #ax2.set_xticklabels(np.round(1 / ax1.get_xticks(), 1))
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
    # matplotlib.rcParams['pdf.fonttype'] = 42
    # matplotlib.rcParams['ps.fonttype'] = 42
    # set the global font size for the plot
    from scipy import stats
        
    #plt.rcParams.update({'font.size': fontsize})
    figsize = (figsize_cm[0]/2.54, figsize_cm[1]/2.54) # convert cm to inches
    
    fig, ax1 = plt.subplots(figsize=figsize, dpi=600)  # DPI is fixed to 600 for publication quality
    #sns.set_theme(context="paper", font=font, font_scale=fontscale)
    # Set font size for all text in the figure
    #sns.set_style("white")
    
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'cyan', 'magenta', 'yellow']
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    for idx, y in enumerate(list_of_y_arrays):
        y = np.array(y)

        # Calculate the means and standard error
        mean_y = np.mean(y, axis=0)
        stderr_y = stats.sem(y, axis=0)

        # Calculate the t-values for the given confidence interval
        df = len(y) - 1
        print("Degrees of freedom: ", df)
        t_value = stats.t.ppf((1 + confidence_interval / 100) / 2, df)
        print(t_value)
        margin_of_error = stderr_y * t_value

        ax.plot(x_array, mean_y, color=colors[idx % len(colors)], alpha=1, label=f"Curve {idx + 1}", linewidth=linewidth)
        ax.fill_between(x_array, mean_y - margin_of_error, mean_y + margin_of_error, color=colors[idx % len(colors)], alpha=alpha)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
        
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

def pretty_lineplot_XY_multiple_with_shade(xdata_list, list_of_ydata_list, xlabel, ylabel, figsize_cm=(14,8),fontsize=10, \
                        marker="o", markersize=3,fontscale=1,font="Helvetica", save_path=None, \
                        linewidth=1,legends=None, title=None, ylims=None, yticks=None, confidence_interval=95):
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import cm
    from locscale.include.emmer.ndimage.profile_tools import crop_profile_between_frequency
    import seaborn as sns
    import matplotlib 
    from scipy import stats
    figsize = (figsize_cm[0]/2.54, figsize_cm[1]/2.54) # convert cm to inches
    
    fig, ax = plt.subplots(figsize=figsize, dpi=600)  # DPI is fixed to 600 for publication quality

    colors_list = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]
    for i,ydata_list in enumerate(list_of_ydata_list):
        # Plot the data by shading the mean and standard deviation for each data set
        xdata_array = np.array(xdata_list)
        ydata_array = np.array(ydata_list)

        ydata_mean = np.mean(ydata_array, axis=0)
        ydata_se = stats.sem(ydata_array, axis=0)
        # Calculate the t-values for the given confidence interval
        df = len(ydata_array) - 1
        t_value = stats.t.ppf((1 + confidence_interval / 100) / 2, df)
        margin_of_error = ydata_se * t_value
        
        ydata_max = ydata_mean + margin_of_error
        ydata_min = ydata_mean - margin_of_error


        # Plot the mean and standard deviation as a shaded region
        ax.fill_between(xdata_array, ydata_min, ydata_max, alpha=0.2, color=colors_list[i])
        ax.plot(xdata_array, ydata_mean, linewidth=linewidth, color=colors_list[i], label=colors_list[i])

        # Add a text stating the number of data sets
        ax.text(0.75, 0.95, f"N = {ydata_array.shape[0]}", ha="left", va="top", transform=ax.transAxes)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel, rotation=90, ha="center")
    
    plt.tight_layout()
    if ylims is not None:
        plt.ylim(ylims)
    if yticks is not None:
        plt.yticks(yticks)
    
    if title is not None:
        plt.title(title)
    
    if legends is not None:
        ax.legend(legends)
    if save_path is not None:
        plt.savefig(save_path, dpi=600)
    return fig


def plot_binned_residuals(variances_combined_sampled, squared_residuals_sampled, \
                        num_bins=50, xlabel='Variance', ylabel='Mean Squared Residuals', title='Mean Squared Residuals by Variance Bin',\
                        save_path=None, figsize_cm=(8, 8), font='Helvetica', fontscale=1.5, linewidth=2, marker='o', markersize=10):
    """
    Plot the binned squared residuals as a function of variance.

    Parameters:
    - variances_combined_sampled: numpy array of variances.
    - squared_residuals_sampled: numpy array of squared residuals.
    - num_bins: number of bins for the x-axis (default is 50).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import cm
    import seaborn as sns    
    from matplotlib.pyplot import cm
    import matplotlib as mpl
    ## Function not generic
    figsize = (figsize_cm[0] / 2.54, figsize_cm[1] / 2.54)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, rotation=90, ha="center")
    import warnings
    warnings.filterwarnings('ignore')
    variances_combined_sampled = np.array(variances_combined_sampled)
    squared_residuals_sampled = np.array(squared_residuals_sampled)
    # Binning the variances_combined_sampled data
    bins = np.linspace(variances_combined_sampled.min(), variances_combined_sampled.max(), num_bins + 1)
    bin_indices = np.digitize(variances_combined_sampled, bins)

    # Compute the statistics for each bin
    bin_weight_threshold = 1e-9
    bin_not_empty = lambda i: len(variances_combined_sampled[bin_indices == i]) > 0
    bin_weight_check = lambda i: True#len(variances_combined_sampled[bin_indices == i]) / len(variances_combined_sampled) > bin_weight_threshold
    weight_per_bin = [len(variances_combined_sampled[bin_indices == i]) / len(variances_combined_sampled) for i in range(1, len(bins)) if bin_not_empty(i)]
    uncertainty_bin_means = [variances_combined_sampled[bin_indices == i].mean() for i in range(1, len(bins)) if bin_not_empty(i) and bin_weight_check(i)]
    residual_means = [squared_residuals_sampled[bin_indices == i].mean() for i in range(1, len(bins)) if bin_not_empty(i) and bin_weight_check(i)]
    residual_stds = [squared_residuals_sampled[bin_indices == i].std() for i in range(1, len(bins)) if bin_not_empty(i) and bin_weight_check(i)]
    #residual_standard_errors = [squared_residuals_sampled[bin_indices == i].std() / np.sqrt(len(squared_residuals_sampled[bin_indices == i])) for i in range(1, len(bins))]
    confidence_interval = 95
    z_score = 1.96
    residual_standard_errors = [z_score * squared_residuals_sampled[bin_indices == i].std() / np.sqrt(len(squared_residuals_sampled[bin_indices == i])) for i in range(1, len(bins)) if bin_not_empty(i) and bin_weight_check(i)]
    
    uncertainty_bin_values = [variances_combined_sampled[bin_indices == i] for i in range(1, len(bins)) if bin_not_empty(i) and bin_weight_check(i)]
    residual_bin_values = [squared_residuals_sampled[bin_indices == i] for i in range(1, len(bins)) if bin_not_empty(i) and bin_weight_check(i)]
    
    # Plot the results

    ax.plot(uncertainty_bin_means, residual_means, color='blue', marker=marker, markersize=markersize, linewidth=linewidth)
    # plot diagonal line for reference
    ax.plot([0, 1], [0, 1], color='red', linestyle='--')
    ax.fill_between(uncertainty_bin_means, np.array(residual_means) - np.array(residual_standard_errors), np.array(residual_means) + np.array(residual_standard_errors), color='skyblue', alpha=0.4, label='1 Standard Deviation')
    
    residues_max = np.max(residual_means)
    uncertainty_max = np.max(uncertainty_bin_means)
    ymax = np.max([residues_max, uncertainty_max])
    ax.set_xlim(-0.01, 0.25)
    ax.set_ylim(-0.01, 0.5)
    xticks = [0, 0.1, 0.2]
    yticks = [0, 0.25, 0.5]
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(xticks)
    ax.set_yticklabels(yticks)
    
    plt.tight_layout()
    plt.show()
    
    # compute the Uncertainty Calibration Error (UCE) and Expected Calibration Error (ECE)
    uce = 0
    mce = 0
    total_number_of_elements = len(squared_residuals_sampled)
    for i in range(len(uncertainty_bin_means)):
        number_of_elements_in_bin = len(squared_residuals_sampled[bin_indices == i + 1])
        absolute_calibration_error = np.abs(residual_means[i] - uncertainty_bin_means[i])
        maximum_calibration_error = np.max([residual_means[i] - uncertainty_bin_means[i]])
        if np.isnan(absolute_calibration_error):
            absolute_calibration_error = 0
        uncertainty_calibration_error = absolute_calibration_error * number_of_elements_in_bin / total_number_of_elements
        maximum_uncertainty_calibration_error = maximum_calibration_error * number_of_elements_in_bin / total_number_of_elements
        
        uce += uncertainty_calibration_error
        mce += maximum_uncertainty_calibration_error
    
    
    print('Uncertainty Calibration Error (UCE):', uce)
    print('Maximum Calibration Error (MCE):', mce)

    # calculate unweighted calibration error
    unweighted_calibration_error = np.mean(np.abs(np.array(residual_means) - np.array(uncertainty_bin_means)))
    print('Unweighted Calibration Error:', unweighted_calibration_error)
    
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')
    
    return uncertainty_bin_values, residual_bin_values

def plot_binned_residuals_2(variances_combined_sampled, squared_residuals_sampled, \
                        num_bins=50, xlabel='Variance', ylabel='Mean Squared Residuals', title='Mean Squared Residuals by Variance Bin',\
                        save_path=None, figsize_cm=(8, 8), font='Helvetica', fontscale=1.5, linewidth=2, marker='o', markersize=10):
    """
    Plot the binned squared residuals as a function of variance.

    Parameters:
    - variances_combined_sampled: numpy array of variances.
    - squared_residuals_sampled: numpy array of squared residuals.
    - num_bins: number of bins for the x-axis (default is 50).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import cm
    import seaborn as sns    
    from matplotlib.pyplot import cm
    import matplotlib as mpl
    ## Function not generic
    figsize = (figsize_cm[0] / 2.54, figsize_cm[1] / 2.54)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, rotation=90, ha="center")
    import warnings
    warnings.filterwarnings('ignore')
    variances_combined_sampled = np.array(variances_combined_sampled)
    squared_residuals_sampled = np.array(squared_residuals_sampled)
    # Binning the variances_combined_sampled data
    bins = np.linspace(variances_combined_sampled.min(), variances_combined_sampled.max(), num_bins + 1)
    bin_indices = np.digitize(variances_combined_sampled, bins)

    # Compute the statistics for each bin
    bin_weight_threshold = 1e-9
    bin_not_empty = lambda i: len(variances_combined_sampled[bin_indices == i]) > 0
    bin_weight_check = lambda i: True#len(variances_combined_sampled[bin_indices == i]) / len(variances_combined_sampled) > bin_weight_threshold
    weight_per_bin = [len(variances_combined_sampled[bin_indices == i]) / len(variances_combined_sampled) for i in range(1, len(bins)) if bin_not_empty(i)]
    uncertainty_bin_means = [variances_combined_sampled[bin_indices == i].mean() for i in range(1, len(bins)) if bin_not_empty(i) and bin_weight_check(i)]
    residual_means = [squared_residuals_sampled[bin_indices == i].mean() for i in range(1, len(bins)) if bin_not_empty(i) and bin_weight_check(i)]
    residual_stds = [squared_residuals_sampled[bin_indices == i].std() for i in range(1, len(bins)) if bin_not_empty(i) and bin_weight_check(i)]
    #residual_standard_errors = [squared_residuals_sampled[bin_indices == i].std() / np.sqrt(len(squared_residuals_sampled[bin_indices == i])) for i in range(1, len(bins))]
    confidence_interval = 95
    z_score = 1.96
    residual_standard_errors = [z_score * squared_residuals_sampled[bin_indices == i].std() / np.sqrt(len(squared_residuals_sampled[bin_indices == i])) for i in range(1, len(bins)) if bin_not_empty(i) and bin_weight_check(i)]
    
    uncertainty_bin_values = [variances_combined_sampled[bin_indices == i] for i in range(1, len(bins)) if bin_not_empty(i) and bin_weight_check(i)]
    residual_bin_values = [squared_residuals_sampled[bin_indices == i] for i in range(1, len(bins)) if bin_not_empty(i) and bin_weight_check(i)]
    
    # Plot the results

    ax.plot(uncertainty_bin_means, residual_means, color='blue', marker=marker, markersize=markersize, linewidth=linewidth)
    # plot diagonal line for reference
    ax.plot([0, 1], [0, 1], color='red', linestyle='--')
    ax.fill_between(uncertainty_bin_means, np.array(residual_means) - np.array(residual_standard_errors), np.array(residual_means) + np.array(residual_standard_errors), color='skyblue', alpha=0.4, label='1 Standard Deviation')
    
    residues_max = np.max(residual_means)
    uncertainty_max = np.max(uncertainty_bin_means)
    ymax = np.max([residues_max, uncertainty_max])
    ax.set_xlim(-0.01, 0.15)
    ax.set_ylim(-0.01, 0.25)
    xticks = [0, 0.1]
    yticks = [0, 0.1, 0.2]

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(xticks)
    ax.set_yticklabels(yticks)
    
    plt.tight_layout()
    plt.show()
    
    # compute the Uncertainty Calibration Error (UCE) and Expected Calibration Error (ECE)
    uce = 0
    mce = 0
    total_number_of_elements = len(squared_residuals_sampled)
    for i in range(len(uncertainty_bin_means)):
        number_of_elements_in_bin = len(squared_residuals_sampled[bin_indices == i + 1])
        absolute_calibration_error = np.abs(residual_means[i] - uncertainty_bin_means[i])
        maximum_calibration_error = np.max([residual_means[i] - uncertainty_bin_means[i]])
        if np.isnan(absolute_calibration_error):
            absolute_calibration_error = 0
        uncertainty_calibration_error = absolute_calibration_error * number_of_elements_in_bin / total_number_of_elements
        maximum_uncertainty_calibration_error = maximum_calibration_error * number_of_elements_in_bin / total_number_of_elements
        
        uce += uncertainty_calibration_error
        mce += maximum_uncertainty_calibration_error
    
    
    print('Uncertainty Calibration Error (UCE):', uce)
    print('Maximum Calibration Error (MCE):', mce)

    # calculate unweighted calibration error
    unweighted_calibration_error = np.mean(np.abs(np.array(residual_means) - np.array(uncertainty_bin_means)))
    print('Unweighted Calibration Error:', unweighted_calibration_error)
    
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')
    
    return uncertainty_bin_values, residual_bin_values

def get_2d_jointplot_with_text(x_array, y_array, x_label, y_label,save_path, figsize_mm=(40, 60), \
                                yticks=None, xticks=None,fontsize=10, mark_emdb_ids=None, emdb_id_list=None, \
                                probability_levels_required = [0, 0.5, 0.68, 0.8, 0.9]):
    import numpy as np
    import seaborn as sns
    import scipy.stats as st
    import matplotlib.pyplot as plt
    from scipy import stats

    def annotate(data, **kws):
        pearson_correlation = stats.pearsonr(x_array, y_array)
        r2_text = f"$R$ = {pearson_correlation[0]:.2f}"
        ax = plt.gca()
        ax.text(-0.8, .8, r2_text,transform=ax.transAxes)
    
    sns.set_context("paper", font_scale=2)
    figsize_in = (figsize_mm[0]/25.4, figsize_mm[1]/25.4)
    plt.figure(figsize=figsize_in, dpi=600)

    # Create a grid over data range 
    xi = np.linspace(0, max(x_array), 100)
    yi = np.linspace(0, max(y_array), 100)
    xi, yi = np.meshgrid(xi, yi)

    # Create the plot
    g = sns.JointGrid(x=x_array, y=y_array, height=6)
  
    # # draw a scatterplot
    g.plot_joint(sns.regplot, scatter_kws={'color': 'black', 'alpha': 0.5, 'marker':'.'}, line_kws={'color': 'black'}, ci=95)
    #annotate(None)
    # dram lmplot with confidence interval
    r, _ = stats.pearsonr(x_array, y_array)
    g.ax_joint.text(0.05, 0.9, f"$R$ = {r:.2f}", transform=g.ax_joint.transAxes, fontsize=fontsize)
    # Plot the marginal distributions and leave 2 vertical and horizontal spaces for the contour labels
    g.plot_marginals(sns.histplot, color='black', alpha=0.1, bins=20, kde=True)

    # annotate the plot with the correlation coefficient
    #g.ax_joint.annotate(annotate, xy=(0.05, 0.8), xycoords='axes fraction', fontsize=fontsize)
    #g.ax_marg_x.set_xlim(0, max(x_array))
    #g.ax_marg_y.set_ylim(0, max(y_array))


    # Set axis labels
    g.set_axis_labels(xlabel=x_label, ylabel=y_label)
    # Set axis ticks
    if yticks:
        g.ax_joint.set_yticks(yticks)
    if xticks:
        g.ax_joint.set_xticks(xticks)


    # Set x limit
    #g.ax_joint.set_xlim(0.25, max(x_array)*1.1)
    # Set y limit
    #g.ax_joint.set_ylim(0, max(y_array)*1.1)
    #plt.ylim(0, 1.2)
    # Add a vertical transparent band (red) between x=0.55 and x=0.65, and a horizontal transparent band (blue) y = 0.18 and 0.22 
    #g.ax_joint.axvspan(0.53, 0.67, color='red', alpha=0.25)
    # g.ax_joint.axhspan(0.17, 0.23, color='blue', alpha=0.25)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    
        plt.close()




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
