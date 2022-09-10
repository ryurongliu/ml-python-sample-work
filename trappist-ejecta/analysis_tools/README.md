****PLOTTING PIPELINE******

In the terminal, cd into the directory that contains your Ejecta_Simulation_Data folder. Boot up python from here.

1. Set up directory structure (setup.py)     
    `setup_folders(num_vincs=6, num_sites=6)`    
    Creates and populates a Plots folder.    
    
    
    
2. Split into specific sites [IN PROGRESS]          
   `specific_sites(num_sites=6)`       
    Separates ejecta into specific sites (nominally forward, backward, up, down, left and right). Info saved in a new folder.     
    
    
    
3. Histograms (histograms.py)   
    `histograms(num_vincs=6, num_sites=6, all=True, specific=None)`  
    
    [SPECIFIC HISTOGRAMS NOT YET IMPLEMENTED]   
    
    Creates histogram plots. Use keyword 'all=True' to create ALL histograms, or specify using keyword 'specific' and a list with the histograms desired, encoded in this fashion:    
            as[vinc_num] : all_ejecta/vincs_separate/[vinc_num]/all_planets/   
            ac[planet_name, o] : all_ejecta/vincs_compared/histograms/[planet_name, o]     
            c[site_num]s[vinc_num] : specific_collision_sites/[site_num]/vincs_separate/[vinc_num]/all_planets 
            c[site_num]c[planet_name, o] : specific_collision_sites/[site_num]/vincs_compared/[planet_name, o]  
            ex. 'as1' : all_ejecta/vincs_separate/1vinc/all_planets/300000e_2000y_1vinc_histogram.png  
            ex. 'aco' : all_ejecta/vincs_compared/histograms/vincs_compared_overall_histogram.png 
            ex. 'acd' : all_ejecta/vincs_compared/histograms/vincs_compared_planet_d_histogram.png    
            ex. 'c3s4' : specific_collision_sites/site3/vincs_separate/4vinc/all_planets/site3_4vinc_histogram.png   
            ex. 'c5ce' : specific_collision_sites/site5/vincs_compared/histograms/vincs_compared_planet_e_histogram.png    
            
            
4. Cols v. Time (cols_v_time.py)  
    `cols_v_time(num_vincs=6, num_sites=6, all=True, specific=None)`  
    
    [SPECIFIC PLOTS NOT YET IMPLEMENTED]   
    
    Creates cols v. time plots. Use keyword 'all' to create all plots, or specify using keyword 'specific':   
        as[vinc_num][planet_name] : all_ejecta/vincs_separate/[vinc_num]/per_planet/cols_v_time/[planet_name]  
        ac[planet_name] : all_ejecta/vincs_compared/col_v_time/planet_name   
        c[site_num]s[vinc_num][planet_name] :     
                        specific_collision_sites/[site_num]/vincs_separate/[vinc_num]/per_planet/col_v_time[planet_name]   
        c[site_num]c[planet_name] : specific_collision_sites/[site_num]/vincs_compared/cols_v_time/[vinc_num]    



5. Cols v. Time Fits   
 
   [IN PROGRESS]   
     
    *HAVE TO DO THIS MANUALLY TO GET THE BEST FIT!*   
    Sets up a jupyter notebook for manual curve fitting of collisions vs. time.       
     


6. Orbital Elements (orbital_elements.py)     
   
   First, sort all particles by collisions/escaped/remaining:        
   `sort_particles_all()`     
   Next, get orbital element information for all particles:     
   `get_orbital_elements_all()`        
   To make ALL videos for inc vs. a and e vs. a, use:    
   `orbital_elements_videos_all()`   
   To make a specific snapshot in time of inc vs. a/e vs. a, use      
   `inc_snapshot_allp(vinc, timeslice, rem=True, esc=True, b=True, c=True, d=True, e=True, f=True, g=True, h=True)`   
    
    KEYWORDS:
    - vinc: the vinc you want snapshotted   
    - timeslice: the timeslice you want snapshotted   
    - rem/esc/b/c/.....h: toggling types on/off
   
    

8. Single Ejecta [IN PROGRESS....]
    
        
            
            

                    
