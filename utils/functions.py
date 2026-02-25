
def get_cumulative_probability_threshold_levels(x_array, y_array, probability_thresholds=[0.5, 0.68, 0.9, 0.99], grid_size=100):
    """
    Compute density thresholds for a 2D distribution such that the contours 
    at these levels enclose the given cumulative probability thresholds.
    
    Parameters:
        x_array, y_array : array-like
            1D arrays of data points.
        probability_thresholds : list of float
            List of cumulative probability thresholds (e.g., [0.5, 0.68, 0.9, 0.99]).
        grid_size : int, optional
            Resolution of the grid on which the KDE is evaluated.
    
    Returns:
        list of float: Density thresholds corresponding to the provided cumulative probabilities.
    """
    import numpy as np
    from scipy.stats import gaussian_kde

    # Compute the 2D KDE.
    data = np.vstack([x_array, y_array])
    kde = gaussian_kde(data)
    
    # Create a grid over the range of the data.
    xmin, xmax = np.min(x_array), np.max(x_array)
    ymin, ymax = np.min(y_array), np.max(y_array)
    X_grid = np.linspace(xmin, xmax, grid_size)
    Y_grid = np.linspace(ymin, ymax, grid_size)
    X, Y = np.meshgrid(X_grid, Y_grid)
    grid_coords = np.vstack([X.ravel(), Y.ravel()])
    
    # Evaluate the KDE on the grid.
    Z = kde(grid_coords).reshape(X.shape)
    
    # Calculate grid cell area.
    dx = X_grid[1] - X_grid[0]
    dy = Y_grid[1] - Y_grid[0]
    area = dx * dy
    
    # Flatten and sort the density values in descending order.
    Z_flat = Z.flatten()
    idx = np.argsort(Z_flat)[::-1]
    Z_sorted = Z_flat[idx]
    
    # Compute the cumulative probability.
    cumsum = np.cumsum(Z_sorted) * area
    cumsum /= cumsum[-1]  # Normalize to 1.
    
    # For each probability threshold, find the corresponding density level.
    contour_levels = []
    for p in probability_thresholds:
        ind = np.searchsorted(cumsum, p)
        level = Z_sorted[ind]
        contour_levels.append(level)
        
    return contour_levels    
    
def get_2d_jointplot_with_text(x_array, y_array, x_label, y_label,save_path, figsize_mm=(40, 60), \
                                yticks=None, xticks=None,fontsize=10, mark_emdb_ids=None, emdb_id_list=None, \
                                probability_levels_required = [0, 0.5, 0.68, 0.8, 0.9]):
    import numpy as np
    import seaborn as sns
    import scipy.stats as st
    import matplotlib.pyplot as plt
    
    sns.set_context("paper", font_scale=2)
    figsize_in = (figsize_mm[0]/25.4, figsize_mm[1]/25.4)
    plt.figure(figsize=figsize_in, dpi=600)

    # Create a grid over data range 
    xi = np.linspace(0, max(x_array), 100)
    yi = np.linspace(0, max(y_array), 100)
    xi, yi = np.meshgrid(xi, yi)

    # Create a 2D KDE
    kde = st.gaussian_kde([x_array, y_array])
    zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)

    # Compute contour levels based on cumulative probability
    levels = get_cumulative_probability_threshold_levels(x_array, y_array, probability_levels_required)
    print(f"Levels: {levels}")
    levels_increasing = sorted(levels)
      # Rename the contour levels to match required probability levels
    contour_labels = [f'{p*100:.1f}' for p in probability_levels_required]
    print(f"Contour labels: {contour_labels}")
    labels_dictionary = {levels[i]: contour_labels[i] for i in range(len(levels))}

    # Create the plot
    g = sns.JointGrid(x=x_array, y=y_array, height=6)

    # Plot the KDE
    cf = g.ax_joint.contourf(xi, yi, zi, levels=levels_increasing, cmap='turbo', linewidths=1, alpha=0.5)

    # Overlay the contour lines
    c = g.ax_joint.contour(xi, yi, zi, levels=levels_increasing, colors='black', linewidths=1)

    # Add labels to contour lines
    g.ax_joint.clabel(c, inline=True, fmt=labels_dictionary, fontsize=fontsize)
  
  
    # Plot the marginal distributions and leave 2 vertical and horizontal spaces for the contour labels
    g.plot_marginals(sns.histplot, color='black', alpha=0.1, bins=20, kde=True)
    g.ax_marg_x.set_xlim(0, max(x_array))
    g.ax_marg_y.set_ylim(0, max(y_array))


    # Set axis labels
    g.set_axis_labels(xlabel=x_label, ylabel=y_label)
    # Set axis ticks
    if yticks:
        g.ax_joint.set_yticks(yticks)
    if xticks:
        g.ax_joint.set_xticks(xticks)

      # Plot the scatter plot and color based on EMDB ID if provided
    if mark_emdb_ids is None:
      g.plot_joint(sns.scatterplot, color='black', alpha=0.5, marker='o')
    else:
        emdb_ids_to_mark_red = mark_emdb_ids["red"]
        emdb_ids_to_mark_blue = mark_emdb_ids["blue"]
        emdb_ids_to_mark_green = mark_emdb_ids["green"]
        emdbs_not_to_mark = [x for x in emdb_id_list if x not in emdb_ids_to_mark_red and x not in emdb_ids_to_mark_blue and x not in emdb_ids_to_mark_green]
        for emdb_id in emdb_ids_to_mark_red:
            index = emdb_id_list.index(emdb_id)
            g.ax_joint.scatter(x_array[index], y_array[index], color='red', alpha=1, marker='o', s=40)
        for emdb_id in emdb_ids_to_mark_blue:
            index = emdb_id_list.index(emdb_id)
            g.ax_joint.scatter(x_array[index], y_array[index], color='blue', alpha=1, marker='s', s=40)
        for emdb_id in emdb_ids_to_mark_green:
            index = emdb_id_list.index(emdb_id)
            g.ax_joint.scatter(x_array[index], y_array[index], color='green', alpha=1, marker='+', s=40)
        
        for emdb_id in emdbs_not_to_mark:
            index = emdb_id_list.index(emdb_id)
            g.ax_joint.scatter(x_array[index], y_array[index], color='black', alpha=0.5, marker='o', s=5)
      

    # Set x limit
    g.ax_joint.set_xlim(0.25, max(x_array)*1.1)
    # Set y limit
    g.ax_joint.set_ylim(0, max(y_array)*1.1)
    #plt.ylim(0, 1.2)
    # Add a vertical transparent band (red) between x=0.55 and x=0.65, and a horizontal transparent band (blue) y = 0.18 and 0.22 
    g.ax_joint.axvspan(0.53, 0.67, color='red', alpha=0.25)
    g.ax_joint.axhspan(0.17, 0.23, color='blue', alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path)
    
    plt.close()

def get_2d_jointplot_with_series(x_array, y_array, x_array_series, y_array_series, color_array, x_label, y_label, save_path, figsize_mm=(40, 60), \
                                yticks=None, xticks=None,fontsize=10, mark_emdb_ids=None, emdb_id_list=None, \
                                probability_levels_required = [0, 0.5, 0.68, 0.8, 0.9], cmax=1, cmin=0):
    import numpy as np
    import seaborn as sns
    import scipy.stats as st
    import matplotlib.pyplot as plt
    from matplotlib.cm import RdYlGn

    sns.set_context("paper", font_scale=2)
    figsize_in = (figsize_mm[0]/25.4, figsize_mm[1]/25.4)
    plt.figure(figsize=figsize_in, dpi=600)

    # Create a grid over data range 
    xi = np.linspace(0, max(x_array), 100)
    yi = np.linspace(0, max(y_array), 100)
    xi, yi = np.meshgrid(xi, yi)

    # Create a 2D KDE
    kde = st.gaussian_kde([x_array, y_array])
    zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)

    # Compute contour levels based on cumulative probability
    levels = get_cumulative_probability_threshold_levels(x_array, y_array, probability_levels_required)
    print(f"Levels: {levels}")
    levels_increasing = sorted(levels)
      # Rename the contour levels to match required probability levels
    contour_labels = [f'{(1-p)*100:.1f}' for p in probability_levels_required]
    print(f"Contour labels: {contour_labels}")
    labels_dictionary = {levels[i]: contour_labels[i] for i in range(len(levels))}

    # Create the plot
    g = sns.JointGrid(x=x_array, y=y_array, height=6)

    # Plot the KDE
    cf = g.ax_joint.contourf(xi, yi, zi, levels=levels_increasing, cmap='viridis', linewidths=1, alpha=0.5)

    # Overlay the contour lines
    c = g.ax_joint.contour(xi, yi, zi, levels=levels_increasing, colors='black', linewidths=1)

    # Add labels to contour lines
    g.ax_joint.clabel(c, inline=True, fmt=labels_dictionary, fontsize=fontsize)
  

    # Set axis labels
    g.set_axis_labels(xlabel=x_label, ylabel=y_label)
    # Set axis ticks
    if yticks:
        g.ax_joint.set_yticks(yticks)
    if xticks:
        g.ax_joint.set_xticks(xticks)

    # Plot the series of X and Y arrays and color based on series color
    assert len(x_array_series) == len(y_array_series) == len(color_array)
    x_array_series = np.array(x_array_series)
    y_array_series = np.array(y_array_series)
    color_array = np.array(color_array)
    norm_range_colormap = (color_array - cmin) / (cmax - cmin)
    
    colors = RdYlGn(np.linspace(0, 1, len(norm_range_colormap)))
    # Draw a line plot for the series
    for i in range(len(x_array_series)):
        f1_score = color_array[i]
        normalised_color = norm_range_colormap[i]
        color_index = int(normalised_color * (len(colors)-1))
        if i == 0:
            marker = "s"
        elif i == len(x_array_series) - 1:
            marker = "x"
        else:
            continue
        g.ax_joint.plot(x_array_series[i], y_array_series[i], color="black", alpha=0.5, marker=marker, markersize=5, linestyle='-', linewidth=2)
    g.ax_joint.plot(x_array_series, y_array_series, color="black", alpha=0.5, linestyle='-', linewidth=2)
    

    
    # Set x limit
    g.ax_joint.set_xlim(0.25, max(x_array)*1.1)
    # Set y limit
    g.ax_joint.set_ylim(0, max(y_array)*1.1)
    #plt.ylim(0, 1.2)

    plt.tight_layout()
    plt.savefig(save_path)
    
    plt.close()

def get_2d_jointplot_with_list_of_series(x_array, y_array, list_of_x_array_series, list_of_y_array_series, list_of_color_array, x_label, y_label, save_path, figsize_mm=(40, 60), \
                                yticks=None, xticks=None,fontsize=10, mark_emdb_ids=None, emdb_id_list=None, \
                                probability_levels_required = [0, 0.5, 0.68, 0.8, 0.9], cmax=1, cmin=0, get_trajectory=False):
    import numpy as np
    import seaborn as sns
    import scipy.stats as st
    import matplotlib.pyplot as plt
    from matplotlib.cm import RdYlGn

    sns.set_context("paper", font_scale=2)
    figsize_in = (figsize_mm[0]/25.4, figsize_mm[1]/25.4)
    plt.figure(figsize=figsize_in, dpi=600)

    # Create a grid over data range 
    xi = np.linspace(0, max(x_array), 100)
    yi = np.linspace(0, max(y_array), 100)
    xi, yi = np.meshgrid(xi, yi)

    # Create a 2D KDE
    kde = st.gaussian_kde([x_array, y_array])
    zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)

    # Compute contour levels based on cumulative probability
    levels = get_cumulative_probability_threshold_levels(x_array, y_array, probability_levels_required)
    print(f"Levels: {levels}")
    levels_increasing = sorted(levels)
      # Rename the contour levels to match required probability levels
    contour_labels = [f'{(p)*100:.1f}' for p in probability_levels_required]
    print(f"Contour labels: {contour_labels}")
    labels_dictionary = {levels[i]: contour_labels[i] for i in range(len(levels))}

    # Create the plot
    g = sns.JointGrid(x=x_array, y=y_array, height=6)

    # Plot the KDE
    cf = g.ax_joint.contourf(xi, yi, zi, levels=levels_increasing, cmap='viridis', linewidths=1, alpha=0.5)

    # Overlay the contour lines
    c = g.ax_joint.contour(xi, yi, zi, levels=levels_increasing, colors='black', linewidths=1)

    # Add labels to contour lines
    g.ax_joint.clabel(c, inline=True, fmt=labels_dictionary, fontsize=fontsize)
  

    # Set axis labels
    g.set_axis_labels(xlabel=x_label, ylabel=y_label)
    # Set axis ticks
    if yticks:
        g.ax_joint.set_yticks(yticks)
    if xticks:
        g.ax_joint.set_xticks(xticks)

    for series_index in range(len(list_of_x_array_series)):
        x_array_series = list_of_x_array_series[series_index]
        y_array_series = list_of_y_array_series[series_index]
        color_array = list_of_color_array[series_index]
        # Plot the series of X and Y arrays and color based on series color
        print(len(x_array_series), "len x array series")
        print(len(y_array_series), "len y array series")
        print(len(color_array), "len color array")
        assert len(x_array_series) == len(y_array_series) == len(color_array)
        x_array_series = np.array(x_array_series)
        y_array_series = np.array(y_array_series)
        color_array = np.array(color_array)
        cmin = min(color_array)
        cmax = max(color_array)
        norm_range_colormap = (color_array - cmin) / (cmax - cmin)
        
        colors = RdYlGn(np.linspace(0, 1, len(norm_range_colormap)))
        index_color_max = np.argmax(color_array)
        # Draw a line plot for the series
        for i in range(len(x_array_series)):
            f1_score = color_array[i]
            normalised_color = norm_range_colormap[i]
            color_index = int(normalised_color * (len(colors)-1))
            if get_trajectory:
                if i == 0:
                    marker = "s"
                    color = colors[color_index]
                    #print(color_index, color, f1_score, "start")
                elif i == len(x_array_series) - 1:
                    marker = "*"
                    color = colors[color_index]
                    print(color_index, color, f1_score, "end")
                elif i == index_color_max:
                    marker = "o"
                    color = colors[color_index]
                    #print(color_index, color, f1_score)
                else:
                    continue
            else:        
                if i == index_color_max:
                    marker = "s"
                    # choose color based on the color array
                    color = colors[color_index]
                else:
                    continue
            g.ax_joint.plot(x_array_series[i], y_array_series[i], color=color, alpha=1, marker=marker, markersize=5, linestyle='-', linewidth=1)
            marker_position_x = x_array_series[i]
            marker_position_y = y_array_series[i]
            # add a text label just above the marker
        #text_label = f"EMD-{emdb_id_list[series_index]}"
        #g.ax_joint.text(marker_position_x, marker_position_y, text_label, fontsize=8, verticalalignment='bottom', horizontalalignment='right')
        if get_trajectory:
            g.ax_joint.plot(x_array_series, y_array_series, color="black", alpha=0.5, linestyle='-', linewidth=1)
        
    # # Draw a colorbar
    # cmin_cbar = round(cmin,1)
    # cmax_cbar = round(cmax,1)
    # # cmin_cbar = 0
    # # cmax_cbar = 1
    # cmid_cbar = round((cmin_cbar + cmax_cbar) / 2, 2)
    # sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(vmin=cmin_cbar, vmax=cmax_cbar))
    # sm._A = []
    # cbar = plt.colorbar(sm, ax=g.ax_joint)
    # cbar.set_label("F1 Score")

    # cbar.set_ticks([cmin_cbar, cmid_cbar, cmax_cbar])
    
    # Set x limit
    g.ax_joint.set_xlim(0.1, max(x_array)*1.1)
    # Set y limit
    g.ax_joint.set_ylim(0, max(y_array)*1.1)
    #plt.ylim(0, 1.2)

    plt.tight_layout()
    plt.savefig(save_path)
    
    plt.close()

def calculate_radiomic_features(mask_array, apix, settings_file):
    import SimpleITK as sitk
    from radiomics.featureextractor import RadiomicsFeatureExtractor    
    
    mask = sitk.GetImageFromArray(mask_array.astype(int))      # Ensure to convert to int

    # Create PyRadiomics feature extractor

    extractor = RadiomicsFeatureExtractor(settings_file)
    extractor.settings['resampledPixelSpacing'] = [apix, apix, apix]
    # Get surface area to volume ratio
    extractor.settings['enableCExtensions'] = True


    # Extract features
    featurevector = extractor.execute(mask, mask)

    return featurevector
   


