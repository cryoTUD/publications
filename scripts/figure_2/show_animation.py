## IMPORTS
import os
import sys
import glob
import warnings
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image

sys.path.append(os.environ["LOCSCALE_2_SCRIPTS_PATH"])
warnings.filterwarnings("ignore")

from scripts.utils.general import setup_environment, assert_paths_exist, create_folders_if_they_do_not_exist
animate_for = "bfactor_refinement" # or "pseudomodel_refinement"
folder_name = "fsc_bfactor_refinement_animation" if animate_for == "bfactor_refinement" else "animation_radial_profile"
frame_pattern = "figure_2_radial_profiles_iteration_cycle_{}.png" if animate_for == "pseudomodel_refinement" else "figure_2_fsc_curves_average_cycle_{}.png"
start_index = 0 if animate_for == "bfactor_refinement" else 1
end_index = 30 if animate_for == "bfactor_refinement" else 51
def main():
    data_archive_path = setup_environment()

    # Define inputs
    input_folder = os.path.join(
        data_archive_path, "outputs", "figure_2", folder_name
    )
    output_filename = os.path.join(input_folder, "fsc_refinement_animation.gif")

    assert_paths_exist(input_folder)

    # frame_paths = sorted(glob.glob(os.path.join(input_folder, frame_pattern)))
    # if not frame_paths:
    #     raise FileNotFoundError(f"No PNG files found in {input_folder} matching pattern {frame_pattern}")

    # # Print the names of frames paths
    # print("Frame paths:")
    # for frame_path in frame_paths:
    #     print(frame_path)
    frame_paths = []
    for cycle in range(start_index, end_index):
        frame_paths.append(
            os.path.join(
                input_folder,
                frame_pattern.format(cycle),
            )
        )
    # Load images into frames
    fig = plt.figure()
    img = plt.imshow(Image.open(frame_paths[0]))
    plt.axis("off")

    def update(frame):
        img.set_data(Image.open(frame))
        return [img]

    ani = animation.FuncAnimation(
        fig, update, frames=frame_paths, blit=True, repeat=False, interval=800
    )

    ani.save(output_filename, writer="ffmpeg", fps=1)
    print(f"Animation saved to {output_filename}")

if __name__ == "__main__":
    main()
