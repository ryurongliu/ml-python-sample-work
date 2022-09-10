import math
import numpy as np
import matplotlib.pyplot as plt 
import rebound
import os
import glob
import csv
import sys
import statistics

def histograms(num_vincs=6, num_sites=6, all=True, specific=None):
    """
    DESCRIPTION:
    Creates histograms of collision data.
    
    CALLING SEQUENCE:
    histograms(num_vincs=6, all=True, specific=None)
    
    KEYWORDS:
    ## num_vincs: number of velocity increments (default 6; +0-5 km/s)
    ## all: when TRUE, will create ALL histograms within the plotting directory; when set to FALSE, you must
            supply the 'specific' keyword
    ## specific: indicates which specific histograms you want to create, in a list encoded as follows:
            
            as[vinc_num] : all_ejecta/vincs_separate/[vinc_num]
            ac[planet_name, o] : all_ejecta/vincs_compared/[planet_name, o] 
            c[site_num]s[vinc_num] : specific_collision_sites/[site_num]/vincs_separate/[vinc_num]
            c[site_num]c[planet_name, o] : specific_collision_sites/[site_num]/vincs_compared/[planet_name, o]


            ex. 'as1' : all_ejecta/vincs_separate/1vinc/300000e_2000y_1vinc_histogram.png
            ex. 'aco' : all_ejecta/vincs_compared/vincs_compared_overall_histogram.png
            ex. 'acd' : all_ejecta/vincs_compared/vincs_compared_planet_d_histogram.png

            ex. 'c3s4' : specific_collision_sites/site3/vincs_separate/4vinc/site3_4vinc_histogram.png
            ex. 'c5ce' : specific_collision_sites/site5/vincs_compared/vincs_compared_planet_e_histogram.png
            
    """
    
    #****CONSTANTS****(hardwired to TRAPPIST for now, to be fixed later)
    num_planets = 7
    object_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    colors = ['w','r', 'c', 'm', 'gold', 'darkgrey', 'b', 'g']
    #******************
    
    if all is True:
        if specific is not None:
            print('Error: use EITHER \'all=True\' to generate all histograms, OR use specific=[] to specify which histograms you want to generate.')
            return
        else:
            #generate all histos
            as_targets = np.arange(num_vincs)
            ac_targets = list.copy(object_names)
            ac_targets.append('all')
            ac_targets.append('overview')
            ac_targets.append('esc')
            ac_targets.remove('a')
            
    
    elif all is False:
        if specific is None:
            print('Error: please specify which histogram you want to generate, or use \'all=True\' to generate all histograms.')
            return
        else:
            for target in specific:
                if len(target)==3:
                    pass
                    #all ejecta
                elif len(target)==4:
                    #specific collision site
                    pass
                else:
                    print('Error: incorrect format for histogram specifier. Please refer to the docstring for proper formatting.')
                    return
                
    
    parent = os.getcwd()
    
    aggregate_vals_all = []
    error_vals_all = []
    error_percent_all = []
    
    #loop over vincs to get data
    for v in range(num_vincs):
        folder_paths = sorted(glob.glob(parent + '/Ejecta_Simulation_Data/' + str(v) + 'vinc/5000e*'))
        
        #aggregate the collisions into arrays and errorval arrays

        #dictionaries to keep track of values
        aggregate_vals = {                       
            "total_ejecta": 0,     #total number of ejecta present
            "num_collisions": 0, #total number of collisions
            "num_remaining": 0,  #number of remaining ejecta
            "esc": 0             #number of escaped particles
        }
        error_vals = {
            "num_collisions": 0,   
            "num_remaining": 0,    
            "esc": 0               
        }
        error_percent = {
            "num_collisions": 0,  
            "num_remaining": 0,    
            "esc": 0              
        }
        #dictionary entries for each planet
        for i in range (len(object_names)):
            aggregate_vals[object_names[i]] = 0 
            error_vals[object_names[i]] = 0 
            error_percent[object_names[i]] = 0 


        #add up total collisions from each simulation 
        for i in range(len(folder_paths)):
            file = folder_paths[i] + '/' + folder_paths[i].split('/')[-1] + '_overview.csv'
            with open(file, mode = 'r') as f:
                r = csv.reader(f)
                rows = list(r)
                aggregate_vals["total_ejecta"] += int(rows[2][1])         #total num of ejecta
                aggregate_vals["esc"] += int(rows[8][1])                  #num escaped
                for j in range(9, 17):
                    aggregate_vals[rows[j][0]] += int(rows[j][1])         #num collided with each planet
                    aggregate_vals['num_collisions'] += int(rows[j][1])   #add to overall collision count
         
        #calculate number of remaining particles
        aggregate_vals['num_remaining'] = aggregate_vals['total_ejecta'] - aggregate_vals['num_collisions']
        
        
        #sqrt(n) error value for:
        #- each planet's total
        #- total num of collisions
        #- and num remaining
        for i in range(len(object_names)):
            error_vals[object_names[i]] = math.sqrt(aggregate_vals[object_names[i]])
        error_vals['num_collisions'] = math.sqrt(aggregate_vals['num_collisions'])
        error_vals['num_remaining'] = math.sqrt(aggregate_vals['num_remaining'])
        error_vals['esc'] = math.sqrt(aggregate_vals['esc'])
                                                 
        #error as percentage
        for i in range(len(object_names)):
            if (error_vals[object_names[i]] != 0):
                error_percent[object_names[i]] = error_vals[object_names[i]]/aggregate_vals[object_names[i]] * 100
        error_percent['num_collisions'] = error_vals['num_collisions']/aggregate_vals['num_collisions'] * 100
        error_percent['num_remaining'] = error_vals['num_remaining']/aggregate_vals['num_remaining'] * 100
        if (aggregate_vals['esc'] !=0):
            error_percent['esc'] = error_vals['esc']/aggregate_vals['esc'] * 100
        
        #save all values
        aggregate_vals_all.append(aggregate_vals)
        error_vals_all.append(error_vals)
        error_percent_all.append(error_percent)
        
        print('Data from ' + str(v) + 'vinc gathered....')
    #end loop over num_vincs    
        
        
    
    #make plots for as_targets:
    for v in as_targets:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [4, 1, .4]})
        fig.set_size_inches(18,10)
        fig.suptitle("Planetary Collisions: 300,000 Ejecta over 2,000 years ("
                     + "v_inc = +" + str(v) + " km/s)", fontsize = 16, y= .95)


        #plot 1 --------------------------------------

        #set x marks
        x_vals1 = np.arange(len(object_names)-1)

        #get collision data
        collisions = []
        errs = []
        errs_percents = []
        for i in range(len(object_names)):
            collisions.append(aggregate_vals_all[v][object_names[i]])
            errs.append(error_vals_all[v][object_names[i]])
            errs_percents.append(error_percent_all[v][object_names[i]])

        #plot
        rects1 = ax1.bar(x_vals1, collisions[1:], yerr = errs[1:], align = 'center', 
                         color = colors[1:], alpha = .75, capsize = 10)

        #prettify plot
        ax1.set_xticks(x_vals1)
        ax1.set_xticklabels(object_names[1:], fontsize = 13)
        ax1.set_yticks([0, 30000, 60000, 90000, 120000, 150000, 180000, 210000, 240000, 270000, 300000])
        ax1.set_ylabel("Number of Collisions", fontsize = 13)
        ax1.yaxis.grid(True, color = 'gainsboro')


        #labelling each bar
        i = 0
        for rect in rects1:
            height = rect.get_height()
            err = errs_percents[i]
            ax1.annotate(str(height) + u"\u00B1" + '{:.2f}'.format(err) + "%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize = 10)
            i += 1

        #plot2 ___-_________

        #xticks for second plot
        x_vals2 = np.arange(2)

        #first layer
        rects2 = ax2.bar(x_vals2, [collisions[0], aggregate_vals_all[v]['num_remaining']], yerr = [0, error_vals_all[v]['num_remaining']], 
                align = 'center', color = ['w', 'palegreen'], alpha = .75, capsize = 10)

        #stack each planet's data on top
        b = collisions[0]
        for i in range(num_planets-1):
            ax2.bar(x_vals2, [collisions[i+1], 0], align = 'center', 
                    color = [colors[i+1], 'r'], alpha = .75, capsize = 10, bottom = b)
            b += collisions[i+1]

        #last stack
        ax2.bar(x_vals2[0], collisions[num_planets], yerr = error_vals_all[v]['num_collisions'], align = 'center', 
                    color = [colors[num_planets], 'r'], alpha = .75, capsize = 10, bottom = b)

        #prettify plot
        ax2.set_xticks(x_vals2)
        ax2.set_xticklabels(["Collisions", "Remaining"], fontsize = 13)
        ax2.set_ylim(0, 300000)
        ax2.yaxis.grid(True, color = 'gainsboro')
        ax2.set_yticks([0, 30000, 60000, 90000, 120000, 150000, 180000, 210000, 240000, 270000, 300000])


        #label plot 2

        #heights for percentage labels
        totalheights = []
        height = 0
        for i in range(num_planets + 1):
            totalheights.append(collisions[i]/2 + height)
            height += collisions[i]
        totalheights[7] = sum(collisions)
        totalheights.append(aggregate_vals_all[v]['num_remaining']/2)
        totalheights.append(aggregate_vals_all[v]['num_remaining'])

        #calculate each planets' collisions as percentage of total ejecta
        percents = []
        for i in range(num_planets + 1):
            percents.append(collisions[i]/300000*100)
        percents.append(aggregate_vals_all[v]['num_remaining']/300000*100)

        print(percents)
        for i in range(1, num_planets):
            ax2.annotate(object_names[i] + ": " + '{:.2f}'.format(percents[i]) + "%", 
                         xy = (rects2[0].get_x() + rects2[0].get_width() / 2, totalheights[i]), 
                         xytext = (0, 0), textcoords = "offset points", 
                         ha = 'center', va = 'center', fontsize = 10)
        ax2.annotate(object_names[num_planets] + ": " + '{:.2f}'.format(percents[num_planets]) + "%", 
                         xy = (rects2[0].get_x() + 1.1, totalheights[num_planets]), 
                         xytext = (0, 0), textcoords = "offset points", 
                         ha = 'center', va = 'center', fontsize = 10)

        ax2.annotate(str(aggregate_vals_all[v]['num_collisions']) + u"\u00B1" + '{:.2f}'.format(error_percent_all[v]['num_collisions']) + "%", 
                         xy = (rects2[0].get_x() + rects2[0].get_width() / 2, aggregate_vals_all[v]['num_collisions']), 
                         xytext = (0, 5), textcoords = "offset points", 
                         ha = 'center', va = 'bottom', fontsize = 10)

        ax2.annotate('{:.2f}'.format(percents[8]) + "%",
                         xy = (rects2[1].get_x() + rects2[1].get_width() / 2, totalheights[8]), 
                         xytext = (0, 0), textcoords = "offset points", 
                         ha = 'center', va = 'center', fontsize = 10)

        ax2.annotate(str(aggregate_vals_all[v]['num_remaining']) + u"\u00B1" + '{:.2f}'.format(error_percent_all[v]['num_remaining']) + "%", 
                         xy = (rects2[1].get_x() + rects2[1].get_width() / 2, totalheights[9]), 
                         xytext = (0, 5), textcoords = "offset points", 
                         ha = 'center', va = 'bottom', fontsize = 10) 


        #Plot 3 ---------

        ax3.set_ylim(0, 6)
        ax3.bar(0, aggregate_vals_all[v]['esc'], width=0.4, yerr = 0, align = 'center', 
                    color = 'k', alpha = .75)
        ax3.set_xticks([0])
        ax3.set_xticklabels(["Escaped"], fontsize = 13)
        ax3.yaxis.grid(True, color = 'gainsboro')
     
        
        plotpath = parent + '/Plots/all_ejecta/vincs_separate/' + str(v) +'vinc/all_planets/'
        plt.savefig(plotpath + '300000e_2000y_' + str(v)+'vinc_histograms.png', dpi=200)
        plt.close()
        
        print('vinc_separated plots for ' + str(v) + 'vinc complete...')
    
    #make plots for ac_targets:
    
    #1. all planets together
    if 'all' in ac_targets:
        fig, ax1 = plt.subplots()
        fig.set_size_inches(18,10)
        fig.suptitle('All Planetary Collisions (compared across v_incs)', fontsize = 16, y= .95)

        x_vals_hist = np.arange(num_planets)
        width =0.4

        collisions_all = []
        err_percents_all = []
        for i in range(num_vincs):
            collisions = [] 
            err_percents = []
            for j in range(num_planets):
                collisions.append(aggregate_vals_all[i][object_names[j+1]])
                err_percents.append(error_percent_all[i][object_names[j+1]])
            collisions_all.append(collisions)
            err_percents_all.append(err_percents)

        ecolors = ['grey', 'grey', 'gainsboro', 'grey', 'grey', 'gainsboro']

        rects0 = ax1.bar(x_vals_hist - 5*width/6, collisions_all[0], width/3, align = 'center', 
                         color = colors[1:], edgecolor = 'grey', alpha = 1)
        rects1 = ax1.bar(x_vals_hist - 3*width/6, collisions_all[1], width/3, align = 'center', 
                         color = colors[1:], edgecolor = 'grey', alpha = .8)
        rects2 = ax1.bar(x_vals_hist - 1*width/6, collisions_all[2], width/3, align = 'center', 
                         color = colors[1:], edgecolor = 'grey', alpha = .6)
        rects3 = ax1.bar(x_vals_hist + 1*width/6, collisions_all[3], width/3, align = 'center', 
                         color = colors[1:], edgecolor = 'grey', alpha = .4)
        rects4 = ax1.bar(x_vals_hist + 3*width/6, collisions_all[4], width/3, align = 'center', 
                         color = colors[1:], edgecolor = 'grey', alpha = .2)
        rects5 = ax1.bar(x_vals_hist + 5*width/6, collisions_all[5], width/3, align = 'center', 
                         color = colors[1:], edgecolor = 'grey', alpha = .1)

        rects0 = ax1.bar(x_vals_hist - 5*width/6, collisions_all[0], width/3, align = 'center', 
                         color = colors[1:], edgecolor = 'k', alpha = 1, fill=False)
        rects1 = ax1.bar(x_vals_hist - 3*width/6, collisions_all[1], width/3, align = 'center', 
                         color = colors[1:], edgecolor = 'k', alpha = 1, hatch = '+', fill=False)
        rects2 = ax1.bar(x_vals_hist - 1*width/6, collisions_all[2], width/3, align = 'center', 
                         color = colors[1:], edgecolor = 'k', alpha = 1, hatch = '.', fill=False)
        rects3 = ax1.bar(x_vals_hist + 1*width/6, collisions_all[3], width/3, align = 'center', 
                         color = colors[1:], edgecolor = 'k', alpha = 1, hatch = 'x', fill=False)
        rects4 = ax1.bar(x_vals_hist + 3*width/6, collisions_all[4], width/3, align = 'center', 
                         color = colors[1:], edgecolor = 'k', alpha = 1, hatch = '*', fill=False)
        rects5 = ax1.bar(x_vals_hist + 5*width/6, collisions_all[5], width/3, align = 'center', 
                         color = colors[1:], edgecolor = 'k', alpha = 1, hatch='/', fill=False)

        rects = [rects0, rects1, rects2, rects3, rects4, rects5]

        for i in range(len(rects)):
            j=0
            for rect in rects[i]:
                if not ((i==0) and (j==2)):
                    height = rect.get_height()
                    err = err_percents_all[1][i]
                    ax1.annotate(str(height) + u"\u00B1" + '{:.2f}'.format(err) + "%",
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 5),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize = 10, rotation=90)
                else:
                    height = rect.get_height()
                    err = err_percents_all[1][i]
                    ax1.annotate(str(height) + '\n' + u"\u00B1" + '\n' + '{:.2f}'.format(err) + "%",
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 5),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize = 10, rotation=90)
                j +=1


        ax1.set_xticklabels(object_names, fontsize = 13)
        ax1.set_ylim(0, 200000)
        ax1.yaxis.grid(True, color = 'gainsboro')
        ax1.set_yticks([0, 20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000, 180000, 200000])
        ax1.set_ylabel('Number of Collisions', fontsize=13)

        ax1.legend((rects0[0], rects1[0], rects2[0], rects3[0], rects4[0], rects5[0]), 
                   ('v_inc = 0 km/s', 'v_inc = 1 km/s', 'v_inc = 2 km/s', 'v_inc = 3 km/s',
                   'v_inc = 4 km/s', 'v_inc = 5 km/s'), prop={'size': 20})
        
        filepath = parent + '/Plots/all_ejecta/vincs_compared/histograms/'
        plt.savefig(filepath + 'all_vincs_all_planets_histogram.png', dpi=200)
        plt.close()
        
        ac_targets.remove('all')
        
        print('vincs_compared/all_vincs_all_planets plot complete...')
        
    
    #2. overviews
    if 'overview' in ac_targets:
        fig, ax1 = plt.subplots()
        fig.set_size_inches(18, 10)
        fig.suptitle('Total Planetary Collisions as Percentage of Total Ejecta (compared across v_incs)', fontsize= 16, y= .95)

        #gather data
        collisions_all = []
        err_all = []
        percents_all = []
        totalheights_all = []

        #[num_vinc][object]

        for i in range(num_vincs):
            collisions = [] 
            err = []
            percents = []
            totalheights = []
            height = 0
            for j in range(len(object_names)):
                collisions.append(aggregate_vals_all[i][object_names[j]])
                err.append(error_vals_all[i][object_names[j]])
                percents.append(collisions[j]/300000*100)
                if j==0:
                    totalheights.append(collisions[j])
                else:
                    totalheights.append(height + collisions[j]/2)
                    height += collisions[j]
            percents.append(aggregate_vals_all[i]['num_remaining']/300000*100)
            collisions_all.append(collisions)
            err_all.append(err)
            percents_all.append(percents)
            totalheights_all.append(totalheights)

        x_vals = np.arange(2)
        width = 0.4
        widths = [-5, -3, -1, 1, 3, 5]
        alphas = [1, .8, .6, .4, .2, .1]
        hatches = ['', '+', '.', 'x', '*', '/']


        rects_for_legend = []
        for v in range(num_vincs):   
            rects0 = ax1.bar(x_vals + widths[v]*width/6, [collisions_all[v][0], aggregate_vals_all[v]['num_remaining']], 
                             width/3, yerr = [0, error_vals_all[v]['num_remaining']], align = 'center', 
                             color = ['w', 'palegreen'], alpha = alphas[v], capsize = 10)

            #stack each planet's data on top
            b = collisions_all[v][0]
            for i in range(num_planets-1):
                ax1.bar(x_vals + widths[v]*width/6, [collisions_all[v][i+1], 0], width/3, align = 'center', 
                        color = [colors[i+1], 'r'], alpha = alphas[v], capsize = 10, bottom = b)
                ax1.bar(x_vals[0] + widths[v]*width/6, collisions_all[v][i+1], width/3, align = 'center',
                       alpha = 1, bottom = b, edgecolor='k', fill=False)
                b += collisions_all[v][i+1]

            #last stack
            ax1.bar(x_vals[0] + widths[v]*width/6, collisions_all[v][num_planets], width/3, 
                    yerr = error_vals_all[v]['num_collisions'], align = 'center', 
                    color = [colors[num_planets], 'r'], alpha = alphas[v], capsize = 10, bottom = b)
            rects1 = ax1.bar(x_vals + widths[v]*width/6, 
                    [aggregate_vals_all[v]['num_collisions'], aggregate_vals_all[v]['num_remaining']], width/3,
                    align = 'center', edgecolor='k', fill=False, hatch=hatches[v], alpha = .9)
            ax1.bar(x_vals + widths[v]*width/6, 
                    [aggregate_vals_all[v]['num_collisions'], aggregate_vals_all[v]['num_remaining']], width/3,
                    align = 'center', edgecolor='k', fill=False)

            rects_for_legend.append(rects1[1])

            #prettify plot
            ax1.set_xticks(x_vals)
            ax1.set_xticklabels(["Collisions", "Remaining"], fontsize = 13)
            ax1.set_ylim(0, 330000)
            ax1.yaxis.grid(True, color = 'gainsboro')
            ax1.set_yticks([0, 30000, 60000, 90000, 120000, 150000, 180000, 210000, 240000, 270000, 300000])
            ax1.set_ylabel("Number of Collisions", fontsize = 13)

            #label
            for i in range(1, num_planets):
                #planets
                ax1.annotate(object_names[i] + ": " + '{:.2f}'.format(percents_all[v][i]) + "%", 
                             xy = (rects0[0].get_x() + rects0[0].get_width() / 2, totalheights_all[v][i]), 
                             xytext = (0, 0), textcoords = "offset points", 
                             ha = 'center', va = 'center', fontsize = 10, 
                             bbox=dict(boxstyle="square,pad=0.1", fc="w", ec="b", lw=0))
                #h 
                ax1.annotate(object_names[num_planets] + ": " + '{:.2f}'.format(percents_all[v][num_planets]) + "%", 
                             xy = (rects0[0].get_x() + rects0[0].get_width() / 2, totalheights_all[v][num_planets]), 
                             xytext = (0, 5), textcoords = "offset points", 
                             ha = 'center', va = 'bottom', fontsize = 10, color='w', 
                            bbox=dict(boxstyle="square,pad=0.1", fc=colors[-1], ec="b", lw=0, alpha = .1))
                #total
                ax1.annotate(str(aggregate_vals_all[v]['num_collisions']) + '\n' + u"\u00B1" + '\n{:.2f}'.format(error_percent_all[v]['num_collisions']) + "%", 
                             xy = (rects0[0].get_x() + rects0[0].get_width() / 2, aggregate_vals_all[v]['num_collisions']), 
                             xytext = (0, 25), textcoords = "offset points", 
                             ha = 'center', va = 'bottom', fontsize = 10, 
                            bbox=dict(boxstyle="square,pad=0.0", fc="w", ec="b", lw=0)) 
                #remaining
                ax1.annotate(str(aggregate_vals_all[v]['num_remaining']) + '\n'+u"\u00B1" + '\n{:.2f}'.format(error_percent_all[v]['num_remaining']) + "%", 
                             xy = (rects0[1].get_x() + rects0[1].get_width() / 2, aggregate_vals_all[v]['num_remaining']), 
                             xytext = (0, 5), textcoords = "offset points", 
                             ha = 'center', va = 'bottom', fontsize = 10,
                            bbox=dict(boxstyle="square,pad=0.", fc="w", ec="b", lw=0))
                #remaining percent
                ax1.annotate('{:.2f}'.format(percents_all[v][-1]) + "%",
                             xy = (rects0[1].get_x() + rects0[1].get_width() / 2, aggregate_vals_all[v]['num_remaining']/2), 
                             xytext = (0, 0), textcoords = "offset points", 
                             ha = 'center', va = 'center', fontsize = 10, 
                             bbox=dict(boxstyle="square,pad=0.1", fc="w", ec="b", lw=0)) 

        ax1.legend(rects_for_legend, ('v_inc = 0 km/s', 'v_inc = 1 km/s', 'v_inc = 2 km/s', 'v_inc = 3 km/s',
                   'v_inc = 4 km/s', 'v_inc = 5 km/s'), prop={'size': 20})

        filepath = parent + '/Plots/all_ejecta/vincs_compared/histograms/'
        plt.savefig(filepath + 'overview_percentages_of_total_histogram.png', dpi=200)
        plt.close()
        
        ac_targets.remove('overview')
        print('vincs_compared/overview plot complete...')
    
    #4. escaped
    if 'esc' in ac_targets:
        fig, ax = plt.subplots()
        fig.suptitle('Escaped Particles (compared across v_incs)', fontsize= 16, y= .95)
        fig.set_size_inches(9, 10)
        xticks = [1]
        width = 0.4
        widths = [-5, -3, -1, 1, 3, 5]
        ax.set_ylim(0, 6)
        ax.set_ylabel('Number of Particles', fontsize=15)
        ax.set_xlabel('v_inc', fontsize=15)
        xtix = [0.6666666666666667, 0.7999999999999999, 0.9333333333333333, 1.0666666666666667, 1.2, 1.3333333333333333]
        ax.set_xticks(xtix)
        ax.set_xticklabels(('0 km/s', '1 km/s', '2 km/s', '3 km/s', '4 km/s', '5 km/s'), fontsize = 13)
        ax.yaxis.grid(True, color = 'gainsboro')

        for v in range(num_vincs):
            ax.bar(xticks[0] + widths[v]*width/6, aggregate_vals_all[v]['esc'], 
                   width=width/3, color='k', align = 'center', alpha = .75)
            ax.bar(xticks[0] + widths[v]*width/6, aggregate_vals_all[v]['esc'], 
                   width=width/3, color='k', align = 'center', alpha = 1, edgecolor='k', fill=False)
        
        filepath = parent + '/Plots/all_ejecta/vincs_compared/histograms/'
        plt.savefig(filepath + 'all_vincs_escaped_histogram.png', dpi=200)
        plt.close()
        
        ac_targets.remove('esc')
        print('vincs_compared/escaped plot complete...')
        
    #3. planets by themselves
    ylims = {
        'a': 0, 
        'b': 60000, 
        'c': 120000, 
        'd': 200000, 
        'e': 80000, 
        'f': 40000, 
        'g': 40000, 
        'h': 20000
    }
    yticks = {
        'a': [0], 
        'b': [0, 20000, 40000, 60000],
        'c': [0, 20000, 40000, 60000, 80000, 100000, 120000], 
        'd': [0, 20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000, 180000, 200000],
        'e': [0, 20000, 40000, 60000, 80000], 
        'f': [0, 20000, 40000], 
        'g': [0, 20000, 40000], 
        'h': [0, 20000]
    }

    for o in range(len(ac_targets)):
        fig, ax1 = plt.subplots()
        fig.set_size_inches(18, 10)
        fig.suptitle('TRAPPIST1-' + ac_targets[o] +': Total Collisions (compared across v_incs)', fontsize= 16, y= .95)

        xvals = 1
        width = 0.4

        collisions = []
        err = []
        for i in range(num_vincs):
            collisions.append(aggregate_vals_all[i][ac_targets[o]])
            err.append(error_vals_all[i][ac_targets[o]])
            
        obj_ind = object_names.index(ac_targets[o])

        rects0 = ax1.bar(xvals - 5*width/6, collisions[0], width/3, yerr = err[0], align = 'center', 
                         color = colors[obj_ind], edgecolor = 'grey', alpha = 1, capsize=10)
        rects1 = ax1.bar(xvals - 3*width/6, collisions[1], width/3, yerr = err[1], align = 'center', 
                         color = colors[obj_ind], edgecolor = 'grey', alpha = .8, capsize=10)
        rects2 = ax1.bar(xvals - 1*width/6, collisions[2], width/3, yerr = err[2], align = 'center', 
                         color = colors[obj_ind], edgecolor = 'grey', alpha = .6, capsize=10)
        rects3 = ax1.bar(xvals + 1*width/6, collisions[3], width/3, yerr = err[3], align = 'center', 
                         color = colors[obj_ind], edgecolor = 'grey', alpha = .4, capsize=10)
        rects4 = ax1.bar(xvals + 3*width/6, collisions[4], width/3, yerr = err[4], align = 'center', 
                         color = colors[obj_ind], edgecolor = 'grey', alpha = .2, capsize=10)
        rects5 = ax1.bar(xvals + 5*width/6, collisions[5], width/3, yerr = err[5], align = 'center', 
                         color = colors[obj_ind], edgecolor = 'grey', alpha = .1, capsize=10)
        
        #hatches
        rects0 = ax1.bar(xvals - 5*width/6, collisions[0], width/3, align = 'center', 
                         color = colors[obj_ind], edgecolor = 'k', alpha = 1, fill=False)
        rects1 = ax1.bar(xvals - 3*width/6, collisions[1], width/3, align = 'center', 
                         color = colors[obj_ind], edgecolor = 'k', alpha = 1, hatch = '+', fill=False)
        rects2 = ax1.bar(xvals - 1*width/6, collisions[2], width/3, align = 'center', 
                         color = colors[obj_ind], edgecolor = 'k', alpha = 1, hatch = '.', fill=False)
        rects3 = ax1.bar(xvals + 1*width/6, collisions[3], width/3, align = 'center', 
                         color = colors[obj_ind], edgecolor = 'k', alpha = 1, hatch = 'x', fill=False)
        rects4 = ax1.bar(xvals + 3*width/6, collisions[4], width/3, align = 'center', 
                         color = colors[obj_ind], edgecolor = 'k', alpha = 1, hatch = '*', fill=False)
        rects5 = ax1.bar(xvals + 5*width/6, collisions[5], width/3, align = 'center', 
                         color = colors[obj_ind], edgecolor = 'k', alpha = 1, hatch='/', fill=False)

        rects = [rects0, rects1, rects2, rects3, rects4, rects5]

        #prettify plot
        ax1.set_ylim(0, ylims[ac_targets[o]])
        ax1.yaxis.grid(True, color = 'gainsboro')
        ax1.set_yticks(yticks[ac_targets[o]])
        ax1.set_ylabel('Number of Collisions', fontsize=15)
        ax1.set_xlabel('Velocity Increment', fontsize=15)
        xtix = []
        for i in range(len(rects)):
            xtix.append(rects[i][0].get_x() + rects[i][0].get_width() / 2)
        ax1.set_xticks(xtix)
        ax1.set_xticklabels(('0 km/s', '1 km/s', '2 km/s', '3 km/s', '4 km/s', '5 km/s'), fontsize = 13)
        
        #label
        for i in range(len(rects)):
            j=0
            for rect in rects[i]:
                height = rect.get_height()
                err = err_percents_all[1][i]
                ax1.annotate(str(height) + u"\u00B1" + '{:.2f}'.format(err) + "%",
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 5),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize = 10)
                j +=1
                
        filepath = parent + '/Plots/all_ejecta/vincs_compared/histograms/'
        plt.savefig(filepath + 'all_vincs_' + ac_targets[o] + '_histogram.png', dpi=200)
        plt.clf()
        
        print('vincs_compared plot for planet ' + ac_targets[o] + ' complete...')
        
    
                 
   
        
        
    