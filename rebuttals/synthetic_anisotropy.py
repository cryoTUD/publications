import os 
import subprocess

folder = "/home/abharadwaj1/papers/elife_paper/figure_information/data/synthetic_anisotropy_spike_protein"

for jobname in os.listdir(folder):
    jobfolder = os.path.join(folder, jobname)
    try:
        #halfmap1_file = [os.path.join(jobfolder, f) for f in os.listdir(jobfolder) if "_half1_class001.mrc" in f][0]
        #halfmap2_file = [os.path.join(jobfolder, f) for f in os.listdir(jobfolder) if "_half2_class001.mrc" in f][0]
        emmap_path = os.path.join(jobfolder, "run_class001.mrc")
        copied_emmap_path = os.path.join(jobfolder, "locscale_run", "run_class001.mrc")
        if not os.path.exists(os.path.dirname(copied_emmap_path)):
            os.makedirs(os.path.dirname(copied_emmap_path))
        if not os.path.exists(copied_emmap_path):
            import shutil
            shutil.copyfile(emmap_path, copied_emmap_path)
        
    except IndexError:
        print(f"{jobname}\tFalse")
        continue
    #both_maps_present = os.path.exists(halfmap1_file) and os.path.exists(halfmap2_file)
    both_maps_present = os.path.exists(copied_emmap_path)
    print(f"{jobname}\t{both_maps_present}")

    if both_maps_present:
        locscale_cmd = f"locscale feature_enhance -em {copied_emmap_path} -v -gpus 5 6 7 -np 12"
        print(f"Running locscale for {jobname}")
        print(locscale_cmd)
        print("-----------------------------------")
        subprocess.run(locscale_cmd, shell=True)
        print(f"Processed locscale for {jobname}")
        print("-----------------------------------")

