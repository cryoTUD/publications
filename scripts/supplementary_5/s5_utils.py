def monte_carlo_cubes_analysis(X, model, iterations, cx=16, cy=16, cz=16): 
    import numpy as np
    import warnings
    warnings.filterwarnings("ignore")
    monte_carlo_cubes =  [model(X, training=True) for i in range(iterations)]
    monte_carlo_cubes = np.array([cube.numpy().reshape(32,32,32) for cube in monte_carlo_cubes])
    intensity_value_at_center = [monte_carlo_cubes[i][cz,cy,cx] for i in range(iterations)]
    return intensity_value_at_center

def monte_carlo_runs(X, iterations, model):
    import numpy as np 
    import warnings
    warnings.filterwarnings("ignore")

    monte_carlo_cubes =  [model(X, training=True) for i in range(iterations)]
    monte_carlo_cubes = np.array([cube.numpy() for cube in monte_carlo_cubes])
    return monte_carlo_cubes