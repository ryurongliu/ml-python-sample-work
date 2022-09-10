import os
from pathlib import Path
import shutil
import glob

def setup_folders(num_vincs=6, num_sites=6):
    """
    DESCRIPTION:
    Sets up directory structure for storing plotfiles.
    
    
    CALLING SEQUENCE: 
    setup_folders(num_vincs=6, num_sites=6)
    
    KEYWORDS:
    ## num_vincs: number of velocity increments (default 6; +0-5 km/s)
    ## num_sites: number of specific collision sites (default 6)
    
    
    Directory Structure:
    Plots
        - all_ejecta
            - vincs_separate
                - 0vinc
                    - all_planets
                    - per_planet
                        - cols_v_time
                        - cols_v_time_fits
                        - inc_v_a
                        - e_v_a
                - 1vinc
                - 2vinc
                  ...
                  ...
            - vincs_compared
                - histograms
                - cols_v_time
                - inc_v_a
                - e_v_a
                
        - specific_collision_sites
            - site1
                - vincs_separate
                    - 0vinc
                        - all_planets
                        - per_planet
                            - cols_v_time
                            - cols_v_time_fits
                            - inc_v_a
                            - e_v_a
                    - 1vinc
                    - 2vinc
                      ...
                      ...
                - vincs_compared
                    - histograms
                    - cols_v_time
                    - inc_v_a
                    - e_v_a
            - site2
              ...
              ...
              
        - single_ejecta
            - 0vinc
            - 1vinc
              ...
              ...
    
    """
    
    object_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    
    parent = os.getcwd()
    plotspath = parent + "/Plots"
    all_ejecta_path = plotspath + "/all_ejecta"
    specific_sites_path = plotspath + "/specific_collision_sites"
    single_ejecta_path = plotspath + "/single_ejecta"
    
    #create Plots directory
    Path(plotspath).mkdir(parents=True, exist_ok=True)
    
    
    
    #create all_ejecta folder
    Path(all_ejecta_path).mkdir(parents=True, exist_ok=True)
     
    #populate all_ejecta_folder:
    
    ###1. vincs_separate folder
    Path(all_ejecta_path + "/vincs_separate").mkdir(parents=True, exist_ok=True)
    for i in range(num_vincs):
        
        #make vincs_separate
        vinc_folder = all_ejecta_path + "/vincs_separate/" + str(i) + "vinc"
        Path(vinc_folder).mkdir(parents=True, exist_ok=True)
        
        #make all_planets
        Path(vinc_folder + "/all_planets").mkdir(parents=True, exist_ok=True)
        Path(vinc_folder + "/all_planets/inc_v_a_snapshots").mkdir(parents=True, exist_ok=True)
        Path(vinc_folder + "/all_planets/e_v_a_snapshots").mkdir(parents=True, exist_ok=True)
        
        #make and populate per_planet
        per_p_folder = vinc_folder + "/per_planet"
        Path(per_p_folder).mkdir(parents=True, exist_ok=True)
        Path(per_p_folder + "/cols_v_time").mkdir(parents=True, exist_ok=True)
        Path(per_p_folder + "/cols_v_time_fits").mkdir(parents=True, exist_ok=True)
        Path(per_p_folder + "/inc_v_a").mkdir(parents=True, exist_ok=True)
        Path(per_p_folder + "/e_v_a").mkdir(parents=True, exist_ok=True)
        for o in object_names[1:]:
            Path(per_p_folder + "/inc_v_a/" + o + "_inc_v_a_snapshots").mkdir(parents=True, exist_ok=True)
            Path(per_p_folder + "/e_v_a/" + o + "_e_v_a_snapshots").mkdir(parents=True, exist_ok=True)
        
        Path(per_p_folder + "/inc_v_a/remaining_inc_v_a_snapshots").mkdir(parents=True, exist_ok=True)
        Path(per_p_folder + "/e_v_a/remaining_e_v_a_snapshots").mkdir(parents=True, exist_ok=True)
        Path(per_p_folder + "/inc_v_a/esc_inc_v_a_snapshots").mkdir(parents=True, exist_ok=True)
        Path(per_p_folder + "/e_v_a/esc_e_v_a_snapshots").mkdir(parents=True, exist_ok=True)
        Path(per_p_folder + "/inc_v_a/mixed_inc_v_a_snapshots").mkdir(parents=True, exist_ok=True)
        Path(per_p_folder + "/e_v_a/mixed_e_v_a_snapshots").mkdir(parents=True, exist_ok=True)
    
    
    ###2. vincs_compared folder
    Path(all_ejecta_path + "/vincs_compared").mkdir(parents=True, exist_ok=True)
    Path(all_ejecta_path + "/vincs_compared/histograms").mkdir(parents=True, exist_ok=True)
    Path(all_ejecta_path + "/vincs_compared/cols_v_time").mkdir(parents=True, exist_ok=True)
    Path(all_ejecta_path + "/vincs_compared/inc_v_a").mkdir(parents=True, exist_ok=True)
    Path(all_ejecta_path + "/vincs_compared/e_v_a").mkdir(parents=True, exist_ok=True)
    
    
    
    #create specific_collision_sites folder
    Path(specific_sites_path).mkdir(parents=True, exist_ok=True)
    
    #populate specific_collision_sites folder
    for j in range(num_sites):
        
        #folder for each site
        site_path = specific_sites_path + "/site" + str(j) 
        Path(site_path).mkdir(parents=True, exist_ok=True)
        
        #1. vincs_separate folder
        for i in range(num_vincs):
        
            #make vincs_separate
            vinc_folder = site_path + "/vincs_separate/" + str(i) + "vinc"
            Path(vinc_folder).mkdir(parents=True, exist_ok=True)

            #make all_planets
            Path(vinc_folder + "/all_planets").mkdir(parents=True, exist_ok=True)
            Path(vinc_folder + "/all_planets/inc_v_a_snapshots").mkdir(parents=True, exist_ok=True)
            Path(vinc_folder + "/all_planets/e_v_a_snapshots").mkdir(parents=True, exist_ok=True)
            
            #make and populate per_planet
            per_p_folder = vinc_folder + "/per_planet"
            Path(per_p_folder).mkdir(parents=True, exist_ok=True)
            Path(per_p_folder + "/cols_v_time").mkdir(parents=True, exist_ok=True)
            Path(per_p_folder + "/cols_v_time_fits").mkdir(parents=True, exist_ok=True)
            Path(per_p_folder + "/inc_v_a").mkdir(parents=True, exist_ok=True)
            Path(per_p_folder + "/e_v_a").mkdir(parents=True, exist_ok=True)
            for o in object_names[1:]:
                Path(per_p_folder + "/inc_v_a/" + o + "_inc_v_a_snapshots").mkdir(parents=True, exist_ok=True)
                Path(per_p_folder + "/e_v_a/" + o + "_e_v_a_snapshots").mkdir(parents=True, exist_ok=True)
            Path(per_p_folder + "/inc_v_a/remaining_inc_v_a_snapshots").mkdir(parents=True, exist_ok=True)
            Path(per_p_folder + "/e_v_a/remaining_e_v_a_snapshots").mkdir(parents=True, exist_ok=True)
            Path(per_p_folder + "/inc_v_a/esc_inc_v_a_snapshots").mkdir(parents=True, exist_ok=True)
            Path(per_p_folder + "/e_v_a/esc_e_v_a_snapshots").mkdir(parents=True, exist_ok=True)
            Path(per_p_folder + "/inc_v_a/mixed_inc_v_a_snapshots").mkdir(parents=True, exist_ok=True)
            Path(per_p_folder + "/e_v_a/mixed_e_v_a_snapshots").mkdir(parents=True, exist_ok=True)
        
        ###2. vincs_compared folder
        Path(site_path + "/vincs_compared").mkdir(parents=True, exist_ok=True)
        Path(site_path + "/vincs_compared/histograms").mkdir(parents=True, exist_ok=True)
        Path(site_path + "/vincs_compared/cols_v_time").mkdir(parents=True, exist_ok=True)
        Path(site_path + "/vincs_compared/inc_v_a").mkdir(parents=True, exist_ok=True)
        Path(site_path + "/vincs_compared/e_v_a").mkdir(parents=True, exist_ok=True)
        
    
    #create single_ejecta_path folder
    Path(single_ejecta_path).mkdir(parents=True, exist_ok=True)
    #populate
    for i in range(num_vincs):
        Path(single_ejecta_path + '/' + str(i) + 'vinc').mkdir(parents=True, exist_ok=True)
        
        
def sort_data(num_vincs=6):
    """
    DESCRIPTION:
    Sorts data folders in Ejecta_Simulation_Data by vinc.
    
    CALLING SEQUENCE:
    sort_data(num_vincs=6)
    
    KEYWORDS:
    ## num_vincs: number of velocity increments (default 6; +0-5 km/s)
    """
    
    parent = os.getcwd()
    folders = sorted(glob.glob(parent + '/Ejecta_Simulation_Data/5000e*'))
    for i in range(num_vincs):
        Path(parent + '/Ejecta_Simulation_Data/'+str(i)+'vinc').mkdir(parents=True, exist_ok=True)
    for folder in folders:
        vincnum = folder.split('/')[-1].split('_')[2][0]
        shutil.move(folder, parent + '/Ejecta_Simulation_Data/' + str(vincnum) + 'vinc')
    
    
    