import os
import sys
import argparse
from argparse import RawTextHelpFormatter
from datetime import datetime
#from mpi4py import MPI

progname = os.path.basename(sys.argv[0])
datmod = "2022-04-15"  # to be updated by gitlab after every commit
author = '\n\nAuthors: Alok Bharadwaj'
version = progname + '  0.1' + '  (;' + datmod+ ')'

simple_cmd = 'python faraday_figure_scripts.py'

description = "*** Performs all calculations required to generate figures for the Faraday Discussions paper  *** \n"

cmdl_parser = argparse.ArgumentParser(
description="*** Performs all calculations required to generate figures for the Faraday Discussions paper  *** \n" + \
('\nExample usage: \"{0}\". {1} on {2}'.format(simple_cmd, author, datmod)),formatter_class=RawTextHelpFormatter)



cmdl_parser.add_argument('-em', '--em_map',  help='Input filename EM map')
 
        
def print_start_banner(start_time):
    print("Starting to compute the figures\n")
    print(start_time)
    print("**************************************************")
    print("*               Faraday Discussions              *")
    print("**************************************************")

def print_end_banner(time_now, start_time):
    print("Finished printing all figures\n")
    print("Processing time: {}".format(time_now-start_time))
    print("**************************************************")
    print("*                Have a nice day                 *")
    print("**************************************************")


def launch_figure_printing(args):   
    start_time = datetime.now()
    print_start_banner(start_time)
    import os 
    local_faraday_folder = os.environ['LOCAL_FARADAY_PATH']
    local_faraday_folder = os.path.join(os.environ['LOCAL_FARADAY_PATH'],"faraday_discussions")
    from subprocess import run
    ## Get all paths
    
    figure_1ab_path = os.path.join(local_faraday_folder,"figure_scripts","figure_1","Figure_1ab_plot_radial_profiles.py")
    figure_2f_path = os.path.join(local_faraday_folder,"figure_scripts","figure_2","Figure_2f_atomic_and_guinier_bfactors.py")
    figure_2g_path = os.path.join(local_faraday_folder,"figure_scripts","figure_2","Figure_2g_plot_local_profile.py")
    
    figure_3c_path = os.path.join(local_faraday_folder,"figure_scripts","figure_3","Figure_3c_plot_radial_profile_perturbed_modemaps.py")
    figure_4b_path = os.path.join(local_faraday_folder,"figure_scripts","figure_4","Figure_4b_bfactor_distribution_scattered.py")
    figure_5a_path = os.path.join(local_faraday_folder,"figure_scripts","figure_5","Figure_5a_plot_ensemble_profiles.py")
    figure_5b_path = os.path.join(local_faraday_folder,"figure_scripts","figure_5","Figure_5b_get_helix_sheet_violinplots.py")
    figure_5c_path = os.path.join(local_faraday_folder,"figure_scripts","figure_5","Figure_5c_apply_average_profiles.py")
    figure_S2_path = os.path.join(local_faraday_folder,"figure_scripts","supplements","plot_profile_fit.py")
    figure_S3_path = os.path.join(local_faraday_folder,"figure_scripts","supplements","fsc_dip_analysis_with_rmsd.py")
    
    list_of_figure_paths = [figure_2f_path,figure_2g_path,figure_3c_path,figure_4b_path,figure_5a_path,figure_5b_path,figure_5c_path,
                            figure_S2_path,figure_S3_path]
    
    for figure_path in list_of_figure_paths:
        run(["python",figure_path])
    
    print_end_banner(datetime.now(), start_time=start_time)
    
            
        

def main():
    args = cmdl_parser.parse_args()

    launch_figure_printing(args)

if __name__ == '__main__':
    main()
