import math
import numpy as np
import matplotlib.pyplot as plt 
import rebound
import os
import glob
import csv
import sys
import statistics

def cols_v_time(num_vincs=6, num_sites=6, all=True, specific=None):
    """
    DESCRIPTION:
    Creates collisions vs. time graphs of data.
    
    CALLING SEQUENCE:
    cols_v_time(num_vincs=6, num_sites=6, all=True, specific=None)
    
    KEYWORDS:
    ## num_vincs: number of velocity increments (default 6; +0-5 km/s)
    ## all: when TRUE, will create ALL col v. time graphs within the plotting directory; when set to FALSE, you must
            supply the 'specific' keyword
    ## specific: indicates which specific col_v_time graphs you want to create, in a list encoded as follows:
            
            as[vinc_num][planet_name] : all_ejecta/vincs_separate/[vinc_num]/per_planet/cols_v_time/[planet_name]
            ac[planet_name] : all_ejecta/vincs_compared/col_v_time/planet_name
            c[site_num]s[vinc_num][planet_name] : 
                            specific_collision_sites/[site_num]/vincs_separate/[vinc_num]/per_planet/col_v_time[planet_name]
            c[site_num]c[planet_name] : specific_collision_sites/[site_num]/vincs_compared/cols_v_time/[vinc_num]
            
    """
    
    #****CONSTANTS****(hardwired to TRAPPIST for now, to be fixed later)
    num_planets = 7
    object_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    colors = ['w','r', 'c', 'm', 'gold', 'darkgrey', 'b', 'g']
    #******************
    
    if all is True:
        if specific is not None:
            print('Error: use EITHER \'all=True\' to generate all col v. time graphs, OR use specific=[] to specify which graphs you want to generate.')
            return
        else:
            #generate all col v time graphs
            as_targets = 'ALL'
            ac_targets = 'ALL'
            
    
    elif all is False:
        if specific is None:
            print('Error: please specify which col v time graphs you want to generate, or use \'all=True\' to generate all graphs.')
            return
        else:
            for target in specific:
                if len(target)==3:
                    pass
                    #all ejecta
                elif len(target)==4:
                    pass
                    #specific collision site
                else:
                    print('Error: incorrect format for graph specifier. Please refer to the docstring for proper formatting.')
                    return
                
    
    parent = os.getcwd()
    
    
    #Loop structure:
    #[vincs]
    # -----[objects]
    # ----------[files]
    
    binned_collisions_all = []
    err_all = []
    #binned_collisions_all[vinc][object][time_interval]
    
    #loop over num_vincs to get data
    for v in range(num_vincs):
        folder_paths = sorted(glob.glob(parent + '/Ejecta_Simulation_Data/' + str(v) + 'vinc/5000e*'))
        
        #bin collisions into 10-year chunks
        
        times = np.linspace(0, 2000, 201, dtype=int)
        binned_collisions = [0]*len(object_names)
        err = [0] *len(object_names)
        
        #loop over each object
        for o in range(len(object_names)):
            binned_collisions[o] = [0]*(len(times)-1)
            err[o] = [0] *(len(times)-1)
            
            #loop over each folder
            for f in range(len(folder_paths)):
                file = folder_paths[f] + '/' + folder_paths[f].split('/')[-1] + '_' + object_names[o] + '.csv'
                with open(file, mode='r') as f:
                    r = csv.reader(f)
                    rows = list(r)
                    lines = len(rows)
                    if (lines!=2):      #if there are collision entries, read them
                        
                        rownum = 2  #track row
                        k = 0       #track bin
                        #bin collisions
                        while((rownum < lines) & (k < len(times)-1)):
                            rowtime = float(rows[rownum][4])
                            if (rowtime < times[k+1]):
                                binned_collisions[o][k] += 1
                                rownum += 1
                            else:
                                k += 1
            for j in range(len(times)-1):
                err[o][j] = math.sqrt(binned_collisions[o][j])
        binned_collisions_all.append(binned_collisions)
        err_all.append(err)
        
        print('data collected for ' + str(v) + 'vinc...')
    
    xvals = np.arange(200)*10 + 5
    
    
    
    #plot!
    
    #plot as targets: all_ejecta/vincs_separate/[num_vinc]/per_planet/cols_v_time/[planets]
    if as_targets=='ALL':
        
        xtix = np.linspace (0, 2000, 21, dtype=int)
        
        #each vinc
        for v in range(num_vincs):

            #each object separately
            for o in range(len(object_names)-1):  #not plotting for the star
                fig, ax = plt.subplots()
                fig.set_size_inches(18, 10)
                ax.errorbar(xvals, binned_collisions_all[v][o+1], yerr = err_all[v][o+1], color = colors[o+1], ecolor = 'k', capsize = 3)
                ax.set_xlim(0, 2000)
                ax.set_xticks(xtix)
                ax.xaxis.grid(True, color = 'gainsboro')
                ax.set_xlabel("Time (years)", fontsize = 13)
                ax.set_ylim(0)
                ax.yaxis.grid(True, color = 'gainsboro')
                ax.set_ylabel("Collisions", fontsize = 13)
                ax.set_title("TRAPPIST1-" + str(object_names[o+1]) + 
                             ": collisions over time (300000 ejecta, 2000 years, v_inc = +" + str(v) + ' km/s)', 
                             fontsize = 16, y = 1.02)
                
                filepath = parent + '/Plots/all_ejecta/vincs_separate/' + str(v) + 'vinc/per_planet/cols_v_time/'
                
                plt.savefig(filepath + "300000e_2000y_" + str(v) + "vinc_" + str(object_names[o+1]) + "_cols_v_time.png", dpi = 200)
                plt.close()
            
            print(str(v) + 'vinc cols v. time plots per planet complete....')
        
            #single vinc, all objects together
            fig1, ax1 = plt.subplots()
            fig1.set_size_inches(18, 10)
            for i in range (1, len(object_names)):
                ax1.errorbar(xvals, binned_collisions_all[v][i], yerr = err_all[v][i], color = colors[i], ecolor = 'k', capsize = 3)
            ax1.xaxis.grid(True, color = 'gainsboro')
            ax1.set_xlabel("Time (years)", fontsize = 13)
            #ax1.set_xscale('log')
            ax1.set_xlim(0, 2000)
            ax1.set_xticks(xtix)
            #ax1.set_ylim(0)
            ax1.yaxis.grid(True, color = 'gainsboro')
            ax1.set_yscale('log')
            #ax1.set_ylabel("Collisions", fontsize = 13)
            ax1.set_title("All Planets: collisions over time (300000 ejecta, 2000 years, v_inc = +" + str(v) + ' km/s)', 
                             fontsize = 16, y = 1.02)
            filepath = parent + '/Plots/all_ejecta/vincs_separate/' + str(v) + 'vinc/all_planets/'
            plt.savefig(filepath + "300000e_2000y_" + str(v) + "vinc_all_planets_cols_v_time.jpg", dpi = 200)
            plt.close()
            
            print(str(v) + 'vinc cols v. time plots all planets complete....')
    
    #plot ac targets: all_ejecta/vincs_compared
    if ac_targets=='ALL':
        xtix = np.linspace (0, 2000, 21, dtype=int)
        
        #for each object:
        for i in range(1, len(object_names)):
            fig, ax = plt.subplots()
            fig.set_size_inches(18, 10)
            eb0 = ax.errorbar(xvals, binned_collisions_all[0][i], yerr = err_all[0][i], color = colors[i], 
                        ecolor = 'k', capsize = 3)
            #eb1 = ax.errorbar(xvals, binned_collisions_all[1][i], yerr = err_all[1][i], color = colors[i], 
                       # ecolor = 'k', capsize = 3, marker=1, alpha=.5)
            #eb2 = ax.errorbar(xvals, binned_collisions_all[2][i], yerr = err_all[2][i], color = colors[i], 
                       # ecolor = 'k', capsize = 3, marker=2, alpha=.5)
            #eb3 = ax.errorbar(xvals, binned_collisions_all[3][i], yerr = err_all[3][i], color = colors[i], 
                       # ecolor = 'k', capsize = 3, marker=3, alpha=.5)
            #eb4 = ax.errorbar(xvals, binned_collisions_all[4][i], yerr = err_all[4][i], color = colors[i], 
                      #  ecolor = 'k', capsize = 3, marker=4, alpha=.5)
            eb5 = ax.errorbar(xvals, binned_collisions_all[5][i], yerr = err_all[5][i], color = colors[i], 
                        ecolor = 'k', capsize = 3, alpha = .7, ls=':')
            eb5[-1][0].set_linestyle(':')
            ax.set_xlim(0, 2000)
            ax.set_xticks(xtix)
            ax.xaxis.grid(True, color = 'gainsboro')
            ax.set_xlabel("Time (years)", fontsize = 13)
            ax.set_yscale('log')
            ax.yaxis.grid(True, color = 'gainsboro')
            ax.set_ylabel("Collisions", fontsize = 13)
            ax.set_title("TRAPPIST1-" + str(object_names[i]) + ": collisions over time (300000 ejecta, 2000 years)", 
                         fontsize = 16, y = 1.02)
            
            ax.legend((eb0, eb5), 
                      ('v_inc = 0 km/s', 'v_inc = 5 km/s'),prop={'size': 15})

            filepath = parent + '/Plots/all_ejecta/vincs_compared/cols_v_time/'

            plt.savefig(filepath + "300000e_2000y_all_vincs_" + str(object_names[i]) + "_cols_v_time.png", dpi = 200)
            plt.close()
            
            print('vincs_compared col v. time plot for planet ' + object_names[i] + ' complete....')
                
            
        
        
        
        
        
        
        
        