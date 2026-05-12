import os 
import subprocess

main_folder = "/home/abharadwaj1/thesis/data_archive/inputs/3_surfer/low_pass_filtered/emd_33888"
mask_path = os.path.join(main_folder, "EMD_33888_unsharpened_fullmap_confidenceMap.mrc")
locscalesurfer = "/home/abharadwaj1/dev/gitrepo/locscale-surfer/locscale_surfer.py"

cutoff_res = [5, 6, 7, 8, 9, 10, 15, 20, 25]
for cutoff in cutoff_res:
    subfolder_path = os.path.join(main_folder, f"low_pass_{cutoff}A")
    emmap_path = os.path.join(subfolder_path, f"emd_33888_filtered_{cutoff}A.mrc")
    cmd = [f"python {locscalesurfer}"]
    cmd.append("--input")
    cmd.append(emmap_path)
    cmd.append("--target")
    cmd.append(emmap_path)
    cmd.append("--mask_path")
    cmd.append(mask_path)
    cmd.append("--gpu_ids")
    cmd.append("6")
    print(f"Running LocScale Surfer for cutoff {cutoff}A...")
    print(cmd)
    cmd_str = " ".join(cmd)
        
    subprocess.run(cmd_str, shell=True, check=True)
