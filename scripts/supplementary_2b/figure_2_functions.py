## FUNCTIONS FOR FIGURE 
import os
import sys 
sys.path.append(os.environ["LOCSCALE_2_SCRIPTS_PATH"]) ## This is mandatory
from scripts.utils.plot_utils import configure_plot_scaling, temporary_rcparams

def gradient_solver(emmap,gx,gy,gz,model_initial,g,friction,min_dist_in_angst,apix,save_file_folder,
                  dt=0.05,capmagnitude_lj=400,epsilon=1,scale_lj=1,lj_factor=1,capmagnitude_map=100,scale_map=1,total_iterations=50, 
                  compute_map=False,emmap_path=None,mask_path=None,returnPointsOnly=True,verbose=False,
                  integration='verlet',myoutput=None):
    '''
    Function to solve pseudoatomic model using gradient descent approach. 
    
    emmap : numpy.ndarray
        Numpy array containing the 3D volume of the map
    gx,gy,gz : numpy.ndarray
        Gradients obtained using numpy.gradient() method to get gradient information in x,y and z
    model_initial : pseudomodel_analysis.Model()
        Is a custom built class which has the coordinate information of all atoms. Also has several useful custom functions 
    g : float
        Gradient scaling parameter to scale the "accelerations" uniformly across the model
    friction : float
        friction coefficient to converge the model
    min_dist_in_angst : float
        Minimum distance between two atoms in the pseudo-atomic model, constrained by the bond lengths
    apix : float
        apix of the emmap
    
    -- special note for the following parameters --
    capmagnitude_lj, capmagnitude_map : float
        These values truncate the maximum acceleration felt by an atom during each iteration so that the analysis becomes bounded
        
    '''
    import os
    import numpy as np
    import gemmi
    from locscale.include.emmer.ndimage.map_tools import compute_real_space_correlation
    from locscale.include.emmer.pdb.pdb_to_map import pdb2map
    from locscale.include.emmer.ndimage.profile_tools import compute_radial_profile
    from locscale.include.emmer.pdb.pdb_utils import set_atomic_bfactors
    from locscale.include.emmer.pdb.modify_pdb import set_pdb_cell_based_on_gradient
    from locscale.preprocessing.pseudomodel_classes import Vector, add_Vector
    from locscale.preprocessing.pseudomodel_solvers import get_neighborhood, average_map_value, get_acceleration_from_gradient, get_acceleration_from_lj_potential
    from locscale.utils.plot_tools import tab_print
    from tqdm import tqdm

    tabbed_print = tab_print(2)
    tprint = tabbed_print.tprint
    map_values = []
    pseudomodel = model_initial.copy()
    gradient_magnitude = np.sqrt(gx**2+gy**2+gz**2)
    
    # convert the following into a dictionary solver_properties = 'Solver started with the following properties: \n'+'\n Number of atoms = '+str(len(pseudomodel.list))+'\n Map potential: \n'+'\n g = '+str(g)+'\n Max gradient magnitude  = '+str(gradient_magnitude.max())+'\n Map value range  = '+str((emmap.min(),emmap.max()))+'\n Cap magnitude at  = '+str(capmagnitude_map)+'\n LJ Potential: \n'+'\n Equilibrium distance = '+str(min_dist_in_angst)+'\n apix, in A = '+str(apix)+'\n LJ Factor = '+str(lj_factor)+'\n Epsilon = '+str(epsilon)+'\n Cap magnitude at  = '+str(capmagnitude_lj)+'\n Friction: \n'+ '\n Friction Coefficient = '+str(friction)+'\n Solver properties: \n'+'\n Total Iterations = '+str(total_iterations)+'\n Time step = '+str(dt)
    solver_properties_dictionary = {
        'num_atoms':len(pseudomodel.list),
        'map_potential':{
            'g':g,
            'max_gradient_magnitude':gradient_magnitude.max(),
            'map_value_range':(emmap.min(),emmap.max()),
            'cap_magnitude_at':capmagnitude_map
        },
        'lj_potential':{
            'equilibrium_distance':min_dist_in_angst,
            'apix_in_A':apix,
            'lj_factor':lj_factor,
            'epsilon':epsilon,
            'cap_magnitude_at':capmagnitude_lj
        },
        'friction':{
            'friction_coefficient':friction
        },
        'solver_properties':{
            'total_iterations':total_iterations,
            'time_step':dt

        }
    }
    
    ## print the solver properties in a nice format
    print('='*50,file=myoutput)
    print('Solver started with the following properties: ',file=myoutput)
    for key,value in solver_properties_dictionary.items():
        print(key+' = '+str(value), file=myoutput)
    print('='*50,file=myoutput)
  
    emmap_shape = emmap.shape
    unitcell = gemmi.UnitCell(emmap_shape[0]*apix,emmap_shape[1]*apix,emmap_shape[2]*apix,90,90,90)
    for iter in tqdm(range(total_iterations),desc="Building Pseudo-atomic model"):
        # Save a  copy of the model as PDB
        save_file_name = "pseudoatomic_model_"+str(iter)+".mmcif"
        save_file_path = os.path.join(save_file_folder,save_file_name)

        pseudomodel.write_pdb(save_file_path, apix=apix,unitcell=unitcell) # saves as mmcif confusingly

        neighborhood = get_neighborhood(pseudomodel.list,min_dist_in_angst/apix)
        
        point_id = 0
        for atom in pseudomodel.list:            
            lj_neighbors = [pseudomodel.list[k] for k in neighborhood[point_id][1]]
            
            gradient_acceleration,map_value = get_acceleration_from_gradient(gx,gy,gz,emmap, g, point=atom, capmagnitude_map=capmagnitude_map)
            if len(lj_neighbors)==0:
                lj_potential_acceleration,_ = Vector(np.array([0,0,0])),0
            else:
                lj_potential_acceleration,_ = get_acceleration_from_lj_potential(atom, lj_neighbors, epsilon=1, min_dist_in_pixel=min_dist_in_angst/apix,lj_factor=lj_factor,capmagnitude_lj=capmagnitude_lj)
            
            gradient_acceleration,lj_potential_acceleration = gradient_acceleration.scale(scale_map),lj_potential_acceleration.scale(scale_lj)
            acceleration = add_Vector(gradient_acceleration,lj_potential_acceleration)
            # add friction 
            atom.acceleration = add_Vector(acceleration, atom.velocity.scale(-friction))
            atom.map_value = map_value
            point_id += 1
        
        if not returnPointsOnly:
            map_values.append(average_map_value(pseudomodel.list))

        if integration == 'euler':
            for atom in pseudomodel.list:
                atom.velocity_from_acceleration(dt)        
                atom.position_from_velocity(dt)
                atom.update_history()
        
        elif integration == 'verlet':
            ''' 
            For the first iteration, use Euler integration since we have no information about -1'th time step
            ''' 
            if iter == 0: 
                for atom in pseudomodel.list:
                    atom.velocity_from_acceleration(dt)        
                    atom.position_from_velocity(dt)
                    atom.update_history()
            else:
                for atom in pseudomodel.list:
                    atom.verlet_integration(dt)
                    atom.update_history()
        else:
            continue 
    pseudomodel.apix = apix
    pseudomodel.update_pdb_positions(apix)
    if returnPointsOnly:
        return pseudomodel    
    else:
        return pseudomodel, map_values

# 
def create_modmap(input_pdb_path, apix, size, output_folder, global_bfactor_map):
    import gemmi
    from locscale.include.emmer.pdb.pdb_utils import set_atomic_bfactors
    from locscale.include.emmer.pdb.pdb_to_map import pdb2map
    from locscale.include.emmer.ndimage.map_utils import save_as_mrc
    uniform_bfactor = global_bfactor_map
    input_pdb_st = gemmi.read_structure(input_pdb_path)
    uniform_bfactor_st = set_atomic_bfactors(input_gemmi_st=input_pdb_st, b_iso=uniform_bfactor)
    simmap = pdb2map(uniform_bfactor_st, apix=apix, size=size)
    input_file_extension = os.path.splitext(input_pdb_path)[1]
    output_file_name = os.path.basename(input_pdb_path).replace(input_file_extension,".mrc")
    output_map_path = os.path.join(output_folder, output_file_name)
    save_as_mrc(simmap, output_map_path,apix=apix)

def refine_model(model_path, map_path, resolution, num_iter):
    """Refine model using servalcat."""
    from locscale.preprocessing.headers import run_servalcat_iterative
    refined_model_path = run_servalcat_iterative(
        model_path=model_path,
        map_path=map_path,
        pseudomodel_refinement=True,
        resolution=resolution,
        num_iter=num_iter
    )
    return refined_model_path


def simulate_maps(apix, shape, output_folder, num_iter):
    """Simulate maps from refined models."""
    from locscale.include.emmer.pdb.pdb_to_map import pdb2map
    from locscale.include.emmer.ndimage.map_utils import save_as_mrc
    import os
    import numpy as np
    from tqdm import tqdm

    refmac_iterations = np.arange(1, num_iter + 1, dtype=int)
    simulated_maps = {}
    for i in tqdm(refmac_iterations, desc="Simulating maps"):
        refined_model_file = os.path.join(output_folder, f"servalcat_refinement_cycle_{i}.cif")
        simulated_map = pdb2map(refined_model_file, apix=apix, size=shape)
        simulated_map_path = os.path.join(output_folder, f"simulated_map_cycle_{i}.mrc")
        save_as_mrc(simulated_map, simulated_map_path, apix=apix)
        simulated_maps[i] = simulated_map_path

    return simulated_maps


# def compute_fsc(refined_map, halfmap, softmask):
#     """Compute FSC for a given map."""
#     from locscale.include.emmer.ndimage.fsc_util import calculate_fsc_maps
#     import numpy as np

#     fsc_masked = calculate_fsc_maps(refined_map * softmask, halfmap * softmask)
#     fsc_unmasked = calculate_fsc_maps(refined_map, halfmap)
#     return {"masked": (fsc_masked, np.mean(fsc_masked)), "unmasked": (fsc_unmasked, np.mean(fsc_unmasked))}

def compute_fsc_all_cycles(output_folder, halfmap1_path, halfmap2_path, mask_path, num_iter, n_jobs=10):
    """Compute FSC for all cycles."""
    import joblib

    model_map_paths_with_averaging = {k : os.path.join(output_folder, f"simulated_map_cycle_{k}.mrc") for k in range(1, num_iter + 1)}
    results_with_averaging = joblib.Parallel(n_jobs=n_jobs, verbose=10)(
                joblib.delayed(compute_fsc_cycle)(cycle, halfmap1_path, halfmap2_path, model_map_paths_with_averaging[cycle], mask_path)\
                                                            for cycle in range(1, num_iter + 1))

    return results_with_averaging
    
def compute_fsc_cycle(cycle, halfmap1_path, halfmap2_path,refined_model_map_path, mask_path):
    """Compute FSC for a given cycle."""
    import numpy as np
    from locscale.include.emmer.ndimage.fsc_util import calculate_fsc_maps
    from locscale.include.emmer.ndimage.map_utils import load_map
    from locscale.emmernet.emmernet_functions import load_smoothened_mask

    assert os.path.exists(refined_model_map_path), f"Refined model map path {refined_model_map_path} does not exist"
    assert os.path.exists(halfmap1_path), f"Halfmap1 path {halfmap1_path} does not exist"
    assert os.path.exists(halfmap2_path), f"Halfmap2 path {halfmap2_path} does not exist"
    
    softmask, apix = load_smoothened_mask(mask_path)
    halfmap1, apix = load_map(halfmap1_path)
    halfmap2, apix = load_map(halfmap2_path)
    refined_model_map, apix = load_map(refined_model_map_path)
    
    #fsc_vals_halfmap1 = calculate_fsc_maps(refined_model_map_path, halfmap1_path)
    #fsc_vals_halfmap2 = calculate_fsc_maps(refined_model_map_path, halfmap2_path)
    fsc_vals_halfmap1 = calculate_fsc_maps(refined_model_map * softmask, halfmap1 * softmask)
    fsc_vals_halfmap2 = calculate_fsc_maps(refined_model_map * softmask, halfmap2 * softmask)
    

    fsc_average_halfmap1 = (cycle, np.mean(fsc_vals_halfmap1), fsc_vals_halfmap1)
    fsc_average_halfmap2 = (cycle, np.mean(fsc_vals_halfmap2), fsc_vals_halfmap2)

    results = { 
        "cycle" : cycle,
        "halfmap1" : fsc_average_halfmap1,
        "halfmap2" : fsc_average_halfmap2
    }
    return results

def save_results(results, output_folder, num_iter):
    """Save FSC results to pickle files."""
    import pickle
    import os


    fsc_cycles_halfmap1_with_averaging = {}
    fsc_cycles_halfmap2_with_averaging = {}
    for result in results:
        cycle = result["cycle"]
        fsc_cycles_halfmap1_with_averaging[cycle] = result["halfmap1"]
        fsc_cycles_halfmap2_with_averaging[cycle] = result["halfmap2"]    
    
    with open(os.path.join(output_folder, f'fsc_average_halfmap_2_cycle{num_iter}_masked.pickle'), 'wb') as f:
        pickle.dump(fsc_cycles_halfmap2_with_averaging, f)
    with open(os.path.join(output_folder, f'fsc_average_halfmap_1_cycle{num_iter}_masked.pickle'), 'wb') as f:
        pickle.dump(fsc_cycles_halfmap1_with_averaging, f)


def plot_fsc_curves_one_cycle(freq, list_of_fsc, figsize_mm=(45,40)):
    '''Plot FSC curves for one cycle'''
    import os
    import sys 
    sys.path.append(os.environ["LOCSCALE_2_SCRIPTS_PATH"])
    from scripts.utils.plot_utils import pretty_plot_fsc_curve, find_freq_value_where_fsc_drops_below_threshold, configure_plot_scaling, temporary_rcparams

    # rcParams_updates = configure_plot_scaling(figsize_mm)
    # rcParams_updates['legend.fontsize'] *= 0.8  # Reduce legend font size
    # rcParams_updates['lines.linewidth'] *= 0.5  # Reduce line width
    # rcParams_updates['lines.markersize'] *= 0.8  # Reduce marker size
    # rcParams_updates['legend.borderpad'] *= 0.2  # Reduce border padding
    # with temporary_rcparams(rcParams_updates):
    fsc_cycle_1_halfmap1, fsc_cycle_1_halfmap2 = list_of_fsc 
    ylims = (0,1)
    ytick = [0.2, 0.5, 0.8]

    fig_cycle_1 = pretty_plot_fsc_curve(freq, list_of_fsc, legends=["Halfmap 1", "Halfmap 2"], \
                                        showPoints=False, showlegend=False, figsize_mm=figsize_mm, crop_freq=(15,1.8), ylims=ylims, yticks=ytick)
    # add a horizontal line at 0.5 and draw a vertical line at the intersection of the FSC curve with the horizontal line
    freq_1, freq_2 = find_freq_value_where_fsc_drops_below_threshold(freq, fsc_cycle_1_halfmap1, fsc_cycle_1_halfmap2, threshold=0.5)
    freq_max = max(freq_1, freq_2)
    fig_cycle_1.axes[0].axhline(y=0.5, color='k', linestyle='--')
    fig_cycle_1.axes[0].axvline(x=freq_1, color='r', linestyle='--', ymax=0.55)
    fig_cycle_1.axes[0].axvline(x=freq_2, color='b', linestyle='--', ymax=0.55)
    fig_cycle_1.axes[0].set_yticks(ytick)
    fig_cycle_1.axes[0].set_ylim(ylims)
    # Set pad for x and y labels
    fig_cycle_1.axes[0].xaxis.labelpad = 2
    fig_cycle_1.axes[0].yaxis.labelpad = 2
    # show y ticks  
    fig_cycle_1.axes[0].tick_params(axis='y', which='major', labelsize=8)
    fig_cycle_1.tight_layout()

    return fig_cycle_1


def plot_fsc_average(fsc_average_halfmap_1, fsc_average_halfmap_2, figsize_mm=(45,26.5)):
    """
    Generate and save a plot of FSC averages for halfmaps 1 and 2.

    Parameters:
    - fsc_average_halfmap_1: dict, FSC average data for halfmap 1
    - fsc_average_halfmap_2: dict, FSC average data for halfmap 2
    - output_plot_folder: str, directory to save the output plot
    """
    
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import sys 
    sys.path.append(os.environ["LOCSCALE_2_SCRIPTS_PATH"])

    # rcParams_updates = configure_plot_scaling(figsize_mm)
    # rcParams_updates['legend.fontsize'] *= 0.8  # Reduce legend font size
    # rcParams_updates['lines.linewidth'] *= 0.5  # Reduce line width
    # rcParams_updates['lines.markersize'] *= 0.8  # Reduce marker size
    # rcParams_updates['legend.borderpad'] *= 0.2  # Reduce border padding

    # with temporary_rcparams(rcParams_updates):   
        # Prepare data for plotting
    xarray = list(fsc_average_halfmap_1.keys())
    yarray_halfmap2 = [fsc_average_halfmap_2[i][1] for i in fsc_average_halfmap_2.keys()]
    yarray_halfmap1 = [fsc_average_halfmap_1[i][1] for i in fsc_average_halfmap_1.keys()]

    # Plot data
    figsize = (figsize_mm[0] / 25.4, figsize_mm[1] / 25.4)  # Convert mm to inches
    fig, ax = plt.subplots(figsize=figsize, dpi = 600)
    # plot halfmap 1 in white squares
    ax.plot(xarray, yarray_halfmap1, "ks-", label="Halfmap 1", \
            markeredgecolor="k", markerfacecolor="w")
    # plot halfmap 2 in black circles
    ax.plot(xarray, yarray_halfmap2, "ko-", label="Halfmap 2")
    xticks = [0, 15, 30]
    yticks = [0.45, 0.5, 0.55]
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xlim(0, 30)
    ax.set_ylim(0.43, 0.57)

    

    # Customize plot
    ax.set_xlabel("Refinement Cycle")
    ax.set_ylabel(r'$\langle FSC \rangle$')
    ax.legend(loc="lower right")
    fig.tight_layout()
    #ax.xaxis.labelpad = 2
    #ax.yaxis.labelpad = 2



    return fig

def combine_fsc_plots(freq, list_of_cycle_1_fsc, list_of_cycle_5_fsc, list_of_cycle_10_fsc, output_plot_folder, legends=["Halfmap 1", "Halfmap 2"], ylims=(-0.05, 1), figsize_mm=(24, 8), dpi=600, font="Helvetica", fontsize=10, ticksize=8, linewidth=1, markersize=5, linestyle="--", hline_color="k", vline_colors=("r", "b")):
    """
    Create a combined plot with three subplots for FSC curves of cycles 1, 5, and 10, matching the style of pretty_plot_fsc_curve.

    Parameters:
    - freq: list, frequency array
    - list_of_cycle_1_fsc: list, FSC curves for cycle 1
    - list_of_cycle_5_fsc: list, FSC curves for cycle 5
    - list_of_cycle_10_fsc: list, FSC curves for cycle 10
    - output_plot_folder: str, folder to save the output plot
    - legends: list, legend labels for the plots
    - ylims: tuple, y-axis limits for the plots
    - figsize: tuple, figure size (width, height) in inches
    - dpi: int, resolution of the output figure
    - font: str, font used in the plots
    - fontsize: int, font size for text in the plots
    - ticksize: int, size of ticks on axes
    - linewidth: int, line width for plotted lines
    - markersize: int, size of markers on points
    - linestyle: str, style for horizontal and vertical lines
    - hline_color: str, color of horizontal lines
    - vline_colors: tuple, colors of vertical lines (x1, x2)
    """
    import matplotlib.pyplot as plt
    import matplotlib
    import seaborn as sns
    import numpy as np
    import os
    import sys
    sys.path.append(os.environ["LOCSCALE_2_SCRIPTS_PATH"])
    from scripts.utils.plot_utils import find_freq_value_where_fsc_drops_below_threshold

    figsize = (figsize_mm[0] / 25.4, figsize_mm[1] / 25.4)  # Convert mm to inches

    rcParams_updates = configure_plot_scaling(figsize_mm)
    #rcParams_updates['legend.fontsize'] *= 0.8  # Reduce legend font size
    rcParams_updates['lines.linewidth'] *= 0.5  # Reduce line width
    #rcParams_updates['lines.markersize'] *= 0.8  # Reduce marker size
    #rcParams_updates['legend.borderpad'] *= 0.2  # Reduce border padding

    with temporary_rcparams(rcParams_updates):
        # Initialize the figure and subplots
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        # Helper function to plot each cycle
        def plot_cycle(ax, cycle_fsc, title, hide_xaxis=False, hide_yaxis=True):
            colors = sns.color_palette("husl", len(cycle_fsc))
            for curve, color, legend in zip(cycle_fsc, colors, legends):
                ax.plot(freq, curve, label=legend, color=color)
            freq_1, freq_2 = find_freq_value_where_fsc_drops_below_threshold(freq, *cycle_fsc, threshold=0.5)
            ax.axhline(y=0.5, color=hline_color, linestyle=linestyle)
            ax.axvline(x=freq_1, color=vline_colors[0], linestyle=linestyle)
            ax.axvline(x=freq_2, color=vline_colors[1], linestyle=linestyle)
            ax.set_ylim(ylims)
            ax.set_title(title)
           #ax.set_xlabel(r"Spatial Frequency, $d^{-1} (\AA^{-1})$")
           # ax.set_ylabel("FSC")
            ax.tick_params(axis="both", which="major")
            # Hide axis labels for subplots
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.xaxis.labelpad = 2
            ax.yaxis.labelpad = 2
            if hide_xaxis:
                # Hide the x-axis labels
                ax.set_xticklabels([])
            if hide_yaxis:
                # Hide the y-axis labels
                ax.set_yticklabels([])

        # Plot for each cycle
        plot_cycle(axes[0], list_of_cycle_1_fsc, "Cycle 1", hide_yaxis=False)
        plot_cycle(axes[1], list_of_cycle_5_fsc, "Cycle 5", hide_yaxis=True)
        plot_cycle(axes[2], list_of_cycle_10_fsc, "Cycle 10", hide_yaxis=True)

        # Add legends and adjust layout
        axes[0].legend()
        # Add y label for first subplot
        axes[0].set_ylabel("FSC")
        # Add common x label
        fig.text(0.5, 0.04, r"Spatial Frequency, $d^{-1} (\AA^{-1})$", ha="center")
    

        plt.tight_layout()

        # Save the figure
        output_file_path = os.path.join(output_plot_folder, "combined_fsc_plots_cycles_1_5_10.eps")
        output_file_path_png = os.path.join(output_plot_folder, "combined_fsc_plots_cycles_1_5_10.png")
        plt.savefig(output_file_path, bbox_inches="tight", dpi=dpi)
        plt.savefig(output_file_path_png, bbox_inches="tight", dpi=dpi)
        plt.close()

        print(f"Combined plot saved to {output_file_path_png}")
