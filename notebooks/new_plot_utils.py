

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


def pretty_lineplot_multiple_fsc_curves(fsc_arrays_perturb, two_xaxis=True, filename=None,figsize=(14,8),fontscale=2.5,font="Helvetica",linewidth=2,legends=None):
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
    fsc_filename = filename
    colors_rainbow = cm.rainbow(np.linspace(0,1,len(fsc_arrays_perturb.keys())))
    
    if two_xaxis:
        # print(';)
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.grid(False)
        
        
        for i,rmsd in enumerate(fsc_arrays_perturb.keys()):
            if legends is not None:
                legend_text = legends[i]
            else:
                legend_text = None
            sns.lineplot(x=fsc_arrays_perturb[rmsd][0],y=fsc_arrays_perturb[rmsd][1], \
                        linewidth=linewidth, \
                        color=colors_rainbow[i], \
                        ax=ax1, \
                        label=legend_text
                        )
            ax1.set_xlabel(r" Spatial Frequency, $d^{-1}(\AA^{-1}$)")
            ax1.set_ylabel("FSC")
        
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
    

    #plt.tight_layout()
    fig.savefig(fsc_filename, dpi=600, bbox_inches="tight")

def pretty_violinplots(list_of_series, xticks, ylabel,xlabel=None, figsize=(14,8),fontscale=3,font="Helvetica",linewidth=2, filename=None):
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
    fig.tight_layout()
    
    if filename is not None:
        plt.savefig(filename, dpi=600, bbox_inches="tight", transparency=True)
        

def pretty_boxplots(list_of_series, xticks, ylabel,xlabel=None, figsize=(14,8),fontscale=3,font="Helvetica",linewidth=2, filename=None):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import cm
    import matplotlib as mpl
    ## Function not generic
    mpl.rcParams['pdf.fonttype'] = 42
    fig,ax = plt.subplots()
    sns.set(rc={'figure.figsize':figsize})
    sns.set_theme(context="paper", font=font, font_scale=fontscale)
    sns.set_style("white")
    
    ax.boxplot(list_of_series)
    ax.set_xticklabels(xticks)
    ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    fig.tight_layout()
    
    if filename is not None:
        plt.savefig(filename, dpi=600, bbox_inches="tight", transparency=True)
    
    
