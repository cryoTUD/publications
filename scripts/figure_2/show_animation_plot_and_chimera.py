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

animate_for = "pseudomodel_refinement"  # or "pseudomodel_refinement"
folder_name = "fsc_bfactor_refinement_animation" if animate_for == "bfactor_refinement" else "animation_radial_profile"
frame_pattern = "figure_2_radial_profiles_iteration_cycle_{}.png" if animate_for == "pseudomodel_refinement" else "figure_2_fsc_curves_average_cycle_{}.png"
chimera_frame_pattern = "servalcat_refinement_cycle_{}.png" if animate_for == "bfactor_refinement" else "pseudoatomic_model_{}.png"
subfolder_name = "animation_frames"
start_index = 1 if animate_for == "bfactor_refinement" else 1
end_index = 31 if animate_for == "bfactor_refinement" else 50


def combine_images_horizontally(image1_path, image2_path):
    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)

    # Resize img2 to half the height of img1 while maintaining aspect ratio
    new_height = img1.height // 2
    aspect_ratio = img2.width / img2.height
    new_width = int(new_height * aspect_ratio)
    img2_resized = img2.resize((new_width, new_height))

    # Pad img2_resized to match height of img1
    img2_padded = Image.new("RGB", (new_width, img1.height), color="white")
    img2_padded.paste(img2_resized, (0, (img1.height - new_height) // 2))

    total_width = img1.width + new_width
    new_img = Image.new("RGB", (total_width, img1.height), color="white")
    new_img.paste(img1, (0, 0))
    new_img.paste(img2_padded, (img1.width, 0))

    return new_img


def main():
    data_archive_path = setup_environment()

    input_folder = os.path.join(data_archive_path, "outputs", "figure_2", folder_name)
    subfolder_path = os.path.join(input_folder, subfolder_name)
    output_filename = os.path.join(input_folder, "fsc_refinement_animation_with_chimera.gif")

    assert_paths_exist(input_folder, subfolder_path)

    frame_paths = []
    chimera_paths = []
    for cycle in range(start_index, end_index):
        frame_paths.append(os.path.join(input_folder, frame_pattern.format(cycle)))
        chimera_paths.append(os.path.join(subfolder_path, chimera_frame_pattern.format(cycle)))

    # Verify that all image files exist
    for path in frame_paths + chimera_paths:
        assert os.path.isfile(path), f"Missing image file: {path}"

    # Create initial combined image
    initial_combined = combine_images_horizontally(chimera_paths[0], frame_paths[0])

    fig = plt.figure()
    img = plt.imshow(initial_combined)
    plt.axis("off")

    def update(frame_index):
        combined = combine_images_horizontally(chimera_paths[frame_index], frame_paths[frame_index])
        img.set_data(combined)
        return [img]

    ani = animation.FuncAnimation(
        fig, update, frames=range(len(frame_paths)), blit=True, repeat=False, interval=300
    )

    ani.save(output_filename, writer="pillow", fps=2)
    print(f"Animation saved to {output_filename}")


if __name__ == "__main__":
    main()
