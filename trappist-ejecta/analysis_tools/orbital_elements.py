import math
import numpy as np
import matplotlib.pyplot as plt 
import rebound
import os
import csv
import sys
import statistics
import glob
import pickle

from rebound import hash as h
from matplotlib.animation import FuncAnimation, FFMpegWriter

#****CONSTANTS****(hardwired to TRAPPIST for now, to be fixed later)
num_planets = 7
object_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
colors = ('palegreen', 'w','r', 'c', 'm', 'gold', 'darkgrey', 'b', 'g')
groups = ('remaining', 'escaped', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h')
semimaj = (1.154e-2, 1.580e-2, 2.227e-2, 2.925e-2, 3.849e-2, 4.683e-2, 6.189e-2)

bin_slices = 201
#******************

"""

FUNCTIONS:

For all_ejecta plots:
    sort_particles_all()
        - sort particles into types (by hash)
        
    get_orbital_elements_all()
        - get orbital element info of all particles
        
    orbital_elements_videos_all()
        - create all animations for inc v. a and e v. a
        
    ecc_snapshot_allp(vinc, timeslice, rem=True, esc=True, b=True, c=True, d=True, e=True, f=True, g=True, h=True)
        - create a single snapshot of e vs. a
        
    inc_snapshot_allp(vinc, timeslice, rem=True, esc=True, b=True, c=True, d=True, e=True, f=True, g=True, h=True)
        - create a single snapshot of inc vs. a

"""

def sort_particles_all(num_vincs=6, num_sims=60):
    
    """
    DESCRIPTION:
        Sorts all particles into escaped, remaining, and planetary collisions by hash.
        Saves a sorted list of dictionaries as particles_sorted_all.pkl, accessed by
        particles_sorted_all[num_vinc][num_sim][type]
        Where num_vinc: 0-5,
              num_sim: 0-59,
              type: 'escaped', 'remaining', 'a'-'h'
    
    CALLING SEQUENCE:
        sort_particles_all(num_vincs=6)
    
    """
    
    parent = os.getcwd()
    particles_sorted_all = []   #stored by hash; [num_vinc][num_sim][type] 
    print('getting data...')
    
    for v in range(num_vincs):
        
        particles_sorted_per_vinc = [0]*num_sims
        folder_paths = sorted(glob.glob(parent + '/Ejecta_Simulation_Data/' + str(v) + 'vinc/*'))
        for folder in folder_paths:
            
            #sorted into collision types
            particles_sorted_per_sim = {
                'escaped': [],
                'remaining': list(range(1, 5001)),
            }
            for o in object_names:
                particles_sorted_per_sim[o] = []
                
            
            #get escaped hashes
            esc_file = folder + '/' + folder.split('/')[-1] + '_escaped.csv'
            with open(esc_file, 'r') as e_file:
                r = csv.reader(e_file)
                rows = list(r)
                if len(rows)!=2:
                    for k in range(2, len(rows)):
                        hashval = int(rows[k][0])
                        particles_sorted_per_sim['escaped'].append(hashval)
                        particles_sorted_per_sim['remaining'].remove(hashval)
            
            #get planet collision hashes
            for o in object_names:
                o_file = folder + '/' + folder.split('/')[-1] + '_' + o + '.csv'
                with open(o_file, 'r') as f:
                    r = csv.reader(f)
                    rows = list(r)
                    if len(rows)!=2:
                        for k in range(2, len(rows)):
                            hashval = int(rows[k][0])
                            particles_sorted_per_sim[o].append(hashval)
                            particles_sorted_per_sim['remaining'].remove(hashval)
            
            sim_num = int(folder.split('/')[-1].split('_')[-1].split('.')[0])
            particles_sorted_per_vinc[sim_num-1] = particles_sorted_per_sim
        
        particles_sorted_all.append(particles_sorted_per_vinc)
        print(str(v) + 'vinc particles sorted...')
        
    with open(parent + '/Plots/all_ejecta/particles_sorted_all.pkl', 'wb') as f:
        pickle.dump(particles_sorted_all, f, pickle.HIGHEST_PROTOCOL)

        
        
        
        
def get_orbital_elements_all(num_vincs=6, num_sims=60):
    
    """
    DESCRIPTION:
        Saves the inclination, eccentricity, and semimajor-axis of all particles into lists of dictionaries.
            eccs_all.pkl
            incs_all.pkl
            axes_all.pkl
        Accessed by [num_vinc][num_sim][bin_slice][type]
        
    CALLING SEQUENCE:
        get_orbital_elements_all(num_vincs=6)
        
    """
    parent = os.getcwd()
    eccs_all = [] #[num_vinc][num_sim][bin_slice][type]
    incs_all = []
    axes_all = []
    
    with open(parent + '/Plots/all_ejecta/particles_sorted_all.pkl', 'rb') as f:
        particles_sorted_all = pickle.load(f)
    
    #get orbital element data
    print('getting orbital elements data...')    
    for v in range(num_vincs):
        folder_paths = sorted(glob.glob(parent + '/Ejecta_Simulation_Data/' + str(v) + 'vinc/*'))
        bin_paths = [x + '/' + x.split('/')[-1] + '.bin' for x in folder_paths]
        eccs_per_vinc = [0]*num_sims
        incs_per_vinc = [0]*num_sims
        axes_per_vinc = [0]*num_sims
        for bin in bin_paths:
            eccs_per_sim = []
            incs_per_sim = []
            axes_per_sim = []
            
            sim_num = int(bin.split('/')[-1].split('_')[-1].split('.')[0])
            sa = rebound.SimulationArchive(bin)
            for i in range(bin_slices):
                snapshot = sa[i]
                eccs_per_binslice = {
                    'escaped': [],
                    'remaining': [],
                }
                incs_per_binslice = {
                    'escaped': [],
                    'remaining': [],
                }
                axes_per_binslice = {
                    'escaped': [],
                    'remaining': [],
                }
                for o in object_names:
                    eccs_per_binslice[o] = []
                    incs_per_binslice[o] = []
                    axes_per_binslice[o] = []
                    
                existing_particles = [x.hash.value for x in snapshot.particles[:]]
                
                #get escaped orbital elements
                for hashval in particles_sorted_all[v][sim_num-1]['escaped']:
                    if hashval in existing_particles:
                        orbit = snapshot.particles[h(hashval)].orbit
                        eccs_per_binslice['escaped'].append(orbit.e)
                        incs_per_binslice['escaped'].append(orbit.inc)
                        axes_per_binslice['escaped'].append(orbit.a)
                
                #get remaining orbital elements
                for hashval in particles_sorted_all[v][sim_num-1]['remaining']:
                    if hashval in existing_particles:
                        orbit = snapshot.particles[h(hashval)].orbit
                        eccs_per_binslice['remaining'].append(orbit.e)
                        incs_per_binslice['remaining'].append(orbit.inc)
                        axes_per_binslice['remaining'].append(orbit.a)
                    
                #get planet collision orbital elements
                for o in object_names:
                    for hashval in particles_sorted_all[v][sim_num-1][o]:
                        if hashval in existing_particles:
                            orbit = snapshot.particles[h(hashval)].orbit
                            eccs_per_binslice[o].append(orbit.e)
                            incs_per_binslice[o].append(orbit.inc)
                            axes_per_binslice[o].append(orbit.a)

                eccs_per_sim.append(eccs_per_binslice)
                incs_per_sim.append(incs_per_binslice)
                axes_per_sim.append(axes_per_binslice)
            
            eccs_per_vinc[sim_num-1] = eccs_per_sim
            incs_per_vinc[sim_num-1] = incs_per_sim
            axes_per_vinc[sim_num-1] = axes_per_sim
            
            print('orbital elements from simulation ' + str(sim_num) + 'done...')
        
        eccs_all.append(eccs_per_vinc)
        incs_all.append(incs_per_vinc)
        axes_all.append(axes_per_vinc)
        
        print('orbital elements for ' + str(v) + 'vinc done...')
    
    with open(parent + '/Plots/all_ejecta/incs_all.pkl', 'wb') as f:
        pickle.dump(incs_all, f, pickle.HIGHEST_PROTOCOL)
    with open(parent + '/Plots/all_ejecta/eccs_all.pkl', 'wb') as f:
        pickle.dump(eccs_all, f, pickle.HIGHEST_PROTOCOL)
    with open(parent + '/Plots/all_ejecta/axes_all.pkl', 'wb') as f:
        pickle.dump(axes_all, f, pickle.HIGHEST_PROTOCOL)
    
    

def orbital_elements_videos_all(num_vincs=6, num_sims=60):
    """
    DESCRIPTION:
    Creates animated plots for inc vs. a and e vs. a for all particles:
            inc_vs_a_all.mp4
            inc_vs_a_remaining.mp4
            inc_vs_a_escaped.mp4
            inc_vs_a_b.mp4
            ...
            ...
            inc_vs_a_h.mp4
            
            [same for ecc_vs_a]
            
        
    
    CALLING SEQUENCE:
        inc_vs_a(num_vincs=6, num_sims=60)
         
    """
    
    parent = os.getcwd()
    #with open(parent + '/Plots/all_ejecta/particles_sorted_all.pkl', 'rb') as f:
        #particles_sorted_all = pickle.load(f)
        
    print('loading data...')
    with open(parent + '/Plots/all_ejecta/incs_all.pkl', 'rb') as f:
        incs_all = pickle.load(f)
    print('incs loaded...')
    with open(parent + '/Plots/all_ejecta/eccs_all.pkl', 'rb') as f:
        eccs_all = pickle.load(f)
    print('eccs loaded...')
    with open(parent + '/Plots/all_ejecta/axes_all.pkl', 'rb') as f:
        axes_all = pickle.load(f)
        
    print('axes loaded...')
        
        
    mp4colors = ('palegreen','r', 'c', 'm', 'gold', 'darkgrey', 'b', 'g', 'k')
    mp4labels = ('remaining', 'b collisions', 'c collisions', 'd collisions', 'e collisions', 'f collisions', 
         'g collisions', 'h collisions', 'escaped')
    mp4data = ["remaining", 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'escaped']
    mp4titles = ('remaining particles', 'b collisions', 'c collisions', 'd collisions', 'e collisions', 'f collisions', 
         'g collisions', 'h collisions', 'escaped particles')
    mp4filenames = ('remaining', 'b-collisions', 'c-collisions', 'd-collisions', 'e-collisions', 'f-collisions', 
         'g-collisions', 'h-collisions', 'escaped')

    for v in range(num_vincs):
    #make inc_vs_a all together video
        
        fig = plt.figure()
        ax = fig.add_subplot(111, ylim = (0., 0.8), xlim = (0., 0.25))
        fig.set_size_inches(18, 10)
        xrem, yrem = [],[]
        rem = ax.scatter(xrem,yrem, color = mp4colors[0], s=10, alpha = 0.8,label = mp4labels[0])
        xb, yb = [],[]
        b = ax.scatter(xb,yb, color = mp4colors[1], s=10, alpha = 0.8,label = mp4labels[1])
        xc, yc = [],[]
        c = ax.scatter(xc,yc, color = mp4colors[2], s=10, alpha = 0.8,label = mp4labels[2])
        xd, yd = [],[]
        d = ax.scatter(xd,yd, color = mp4colors[3], s=10, alpha = 0.8,label = mp4labels[3])
        xe, ye = [],[]
        e = ax.scatter(xe,ye, color = mp4colors[4], s=10, alpha = 0.8,label = mp4labels[4])
        xf, yf = [],[]
        fcol = ax.scatter(xf,yf, color = mp4colors[5], s=10, alpha = 0.8,label = mp4labels[5])
        xg, yg = [],[]
        g = ax.scatter(xg,yg, color = mp4colors[6], s=10, alpha = 0.8,label = mp4labels[6])
        xh, yh = [],[]
        h = ax.scatter(xh,yh, color = mp4colors[7], s=10, alpha = 0.8,label = mp4labels[7])
        xesc, yesc = [], []
        esc = ax.scatter(xesc, yesc, color = mp4colors[8], s=10, alpha = 0.8, label = mp4labels[8])
        for j in range(len(semimaj)):
                ax.vlines(semimaj[j], 0, 2*math.pi, colors=colors[j+2], linestyle = 'dotted')
                ax.text(semimaj[j], 0.95, object_names[j+1], ha = 'center', fontsize = 12, rotation = 0)
        plt.xlim(0,.25)
        plt.ylim(0,1)
        ax.set_xlabel("Semimajor Axis (AU)", fontsize = 13)
        ax.set_ylabel("Inclination (radians)", fontsize = 13)
        plt.legend(loc=1)
        
        def animate_incs(i):
            for j in range(num_sims):
                if j == 0:
                    xrem = list.copy(axes_all[v][j][i]['remaining'])
                    yrem = list.copy(incs_all[v][j][i]['remaining'])
                    xb = list.copy(axes_all[v][j][i]['b'])
                    yb = list.copy(incs_all[v][j][i]['b'])
                    xc = list.copy(axes_all[v][j][i]['c'])
                    yc = list.copy(incs_all[v][j][i]['c'])
                    xd = list.copy(axes_all[v][j][i]['d'])
                    yd = list.copy(incs_all[v][j][i]['d'])
                    xe = list.copy(axes_all[v][j][i]['e'])
                    ye = list.copy(incs_all[v][j][i]['e'])
                    xf = list.copy(axes_all[v][j][i]['f'])
                    yf = list.copy(incs_all[v][j][i]['f'])
                    xg = list.copy(axes_all[v][j][i]['g'])
                    yg = list.copy(incs_all[v][j][i]['g'])
                    xh = list.copy(axes_all[v][j][i]['h'])
                    yh = list.copy(incs_all[v][j][i]['h'])
                    xesc = list.copy(axes_all[v][j][i]['escaped'])
                    yesc = list.copy(incs_all[v][j][i]['escaped'])
                else:
                    xrem = xrem + axes_all[v][j][i]['remaining']
                    yrem = yrem + incs_all[v][j][i]['remaining']
                    xb = xb + axes_all[v][j][i]['b']
                    yb = yb + incs_all[v][j][i]['b']
                    xc = xc + axes_all[v][j][i]['c']
                    yc = yc + incs_all[v][j][i]['c']
                    xd = xd + axes_all[v][j][i]['d']
                    yd = yd + incs_all[v][j][i]['d']
                    xe = xe + axes_all[v][j][i]['e']
                    ye = ye + incs_all[v][j][i]['e']
                    xf = xf + axes_all[v][j][i]['f']
                    yf = yf + incs_all[v][j][i]['f']
                    xg = xg + axes_all[v][j][i]['g']
                    yg = yg + incs_all[v][j][i]['g']
                    xh = xh + axes_all[v][j][i]['h']
                    yh = yh + incs_all[v][j][i]['h']
                    xesc = xesc + axes_all[v][j][i]['escaped']
                    yesc = yesc + incs_all[v][j][i]['escaped']

            rem.set_offsets(np.c_[xrem,yrem]) 
            b.set_offsets(np.c_[xb,yb])
            c.set_offsets(np.c_[xc,yc])
            d.set_offsets(np.c_[xd,yd])
            e.set_offsets(np.c_[xe,ye])
            fcol.set_offsets(np.c_[xf,yf])
            g.set_offsets(np.c_[xg,yg])
            h.set_offsets(np.c_[xh,yh])
            esc.set_offsets(np.c_[xesc, yesc])
            ax.set_title("all particles, inc vs. a, time = " + str(i*10) + " years (v_inc = +" + str(v) + " km/s)", fontsize = 16, y = 1.04)

        ani_incs = FuncAnimation(fig, animate_incs, 
                        frames=np.arange(0, 201, 1), interval=200, repeat=True) 

        savefolder = parent + '/Plots/all_ejecta/vincs_separate/' + str(v) + 'vinc/all_planets/'
        f ="inc_v_a_" + str(v) + "vinc_all_particles.mp4"
        plt.rcParams['animation.ffmpeg_path'] = r'/Users/larkpie/opt/anaconda3/envs/venv/bin/ffmpeg'
        writervideo = FFMpegWriter(fps=5) 
        ani_incs.save(savefolder + f, writer=writervideo, dpi=200)
        plt.close()
        
        print(str(v) + 'vinc inc v. a all-together animated...')
        
    #make e vs a all together video
        fig = plt.figure()
        ax = fig.add_subplot(111, ylim = (0., 0.8), xlim = (0., 0.25))
        fig.set_size_inches(18, 10)
        xrem, yrem = [],[]
        rem = ax.scatter(xrem,yrem, color = mp4colors[0], s=10, alpha = 0.8,label = mp4labels[0])
        xb, yb = [],[]
        b = ax.scatter(xb,yb, color = mp4colors[1], s=10, alpha = 0.8,label = mp4labels[1])
        xc, yc = [],[]
        c = ax.scatter(xc,yc, color = mp4colors[2], s=10, alpha = 0.8,label = mp4labels[2])
        xd, yd = [],[]
        d = ax.scatter(xd,yd, color = mp4colors[3], s=10, alpha = 0.8,label = mp4labels[3])
        xe, ye = [],[]
        e = ax.scatter(xe,ye, color = mp4colors[4], s=10, alpha = 0.8,label = mp4labels[4])
        xf, yf = [],[]
        fcol = ax.scatter(xf,yf, color = mp4colors[5], s=10, alpha = 0.8,label = mp4labels[5])
        xg, yg = [],[]
        g = ax.scatter(xg,yg, color = mp4colors[6], s=10, alpha = 0.8,label = mp4labels[6])
        xh, yh = [],[]
        h = ax.scatter(xh,yh, color = mp4colors[7], s=10, alpha = 0.8,label = mp4labels[7])
        xesc, yesc = [], []
        esc = ax.scatter(xesc, yesc, color = mp4colors[8], s=10, alpha = 0.8, label = mp4labels[8])
        for j in range(len(semimaj)):
                ax.vlines(semimaj[j], 0, 2*math.pi, colors=colors[j+2], linestyle = 'dotted')
                ax.text(semimaj[j], 0.95, object_names[j+1], ha = 'center', fontsize = 12, rotation = 0)
        plt.xlim(0,.25)
        plt.ylim(0,1)
        ax.set_xlabel("Semimajor Axis (AU)", fontsize = 13)
        ax.set_ylabel("Eccentricity", fontsize = 13)
        plt.legend(loc=1)
        
        def animate_eccs(i):
            for j in range(num_sims):
                if j == 0:
                    xrem = list.copy(axes_all[v][j][i]['remaining'])
                    yrem = list.copy(eccs_all[v][j][i]['remaining'])
                    xb = list.copy(axes_all[v][j][i]['b'])
                    yb = list.copy(eccs_all[v][j][i]['b'])
                    xc = list.copy(axes_all[v][j][i]['c'])
                    yc = list.copy(eccs_all[v][j][i]['c'])
                    xd = list.copy(axes_all[v][j][i]['d'])
                    yd = list.copy(eccs_all[v][j][i]['d'])
                    xe = list.copy(axes_all[v][j][i]['e'])
                    ye = list.copy(eccs_all[v][j][i]['e'])
                    xf = list.copy(axes_all[v][j][i]['f'])
                    yf = list.copy(eccs_all[v][j][i]['f'])
                    xg = list.copy(axes_all[v][j][i]['g'])
                    yg = list.copy(eccs_all[v][j][i]['g'])
                    xh = list.copy(axes_all[v][j][i]['h'])
                    yh = list.copy(eccs_all[v][j][i]['h'])
                    xesc = list.copy(axes_all[v][j][i]['escaped'])
                    yesc = list.copy(eccs_all[v][j][i]['escaped'])
                else:
                    xrem = xrem + axes_all[v][j][i]['remaining']
                    yrem = yrem + eccs_all[v][j][i]['remaining']
                    xb = xb + axes_all[v][j][i]['b']
                    yb = yb + eccs_all[v][j][i]['b']
                    xc = xc + axes_all[v][j][i]['c']
                    yc = yc + eccs_all[v][j][i]['c']
                    xd = xd + axes_all[v][j][i]['d']
                    yd = yd + eccs_all[v][j][i]['d']
                    xe = xe + axes_all[v][j][i]['e']
                    ye = ye + eccs_all[v][j][i]['e']
                    xf = xf + axes_all[v][j][i]['f']
                    yf = yf + eccs_all[v][j][i]['f']
                    xg = xg + axes_all[v][j][i]['g']
                    yg = yg + eccs_all[v][j][i]['g']
                    xh = xh + axes_all[v][j][i]['h']
                    yh = yh + eccs_all[v][j][i]['h']
                    xesc = xesc + axes_all[v][j][i]['escaped']
                    yesc = yesc + eccs_all[v][j][i]['escaped']

            rem.set_offsets(np.c_[xrem,yrem]) 
            b.set_offsets(np.c_[xb,yb])
            c.set_offsets(np.c_[xc,yc])
            d.set_offsets(np.c_[xd,yd])
            e.set_offsets(np.c_[xe,ye])
            fcol.set_offsets(np.c_[xf,yf])
            g.set_offsets(np.c_[xg,yg])
            h.set_offsets(np.c_[xh,yh])
            esc.set_offsets(np.c_[xesc, yesc])
            ax.set_title("all particles, e vs. a, time = " + str(i*10) + " years (v_inc = +" + str(v) + " km/s)", fontsize = 16, y = 1.04)

        ani_eccs = FuncAnimation(fig, animate_eccs, 
                        frames=np.arange(0, 201, 1), interval=200, repeat=True) 

        savefolder = parent + '/Plots/all_ejecta/vincs_separate/' + str(v) + 'vinc/all_planets/'
        f ="e_v_a_" + str(v) + "vinc_all_particles.mp4"
        plt.rcParams['animation.ffmpeg_path'] = r'/Users/larkpie/opt/anaconda3/envs/venv/bin/ffmpeg'
        writervideo = FFMpegWriter(fps=5) 
        ani_eccs.save(savefolder + f, writer=writervideo, dpi=200)
        plt.close()
        
        print(str(v) + 'vinc e v. a all-together animated...')
        
    
    #make separates for eccentricity
        for q in range(9):
            fig = plt.figure()
            ax = fig.add_subplot(111, ylim = (0., 1.), xlim = (0., 0.25))
            fig.set_size_inches(18, 10)
            x, y = [],[]
            sc = ax.scatter(x,y, color = mp4colors[q], s=10, alpha = 0.8, label = mp4labels[q])
            for j in range(len(semimaj)):
                    ax.vlines(semimaj[j], 0, 2*math.pi, colors=colors[j+2], linestyle = 'dotted')
                    ax.text(semimaj[j], 0.95, object_names[j+1], ha = 'center', fontsize = 12, rotation = 0)
            plt.xlim(0,.25)
            plt.ylim(0,1)
            ax.set_xlabel("Semimajor Axis (AU)", fontsize = 13)
            ax.set_ylabel("Eccentricity", fontsize = 13)
            plt.legend(loc=1)


            def animate_eccs_separate(i):
                for j in range(num_sims):
                    if j==0:
                        x = list.copy(axes_all[v][j][i][mp4data[q]])
                        y = list.copy(eccs_all[v][j][i][mp4data[q]])
                    else:
                        x = x + axes_all[v][j][i][mp4data[q]]
                        y = y + eccs_all[v][j][i][mp4data[q]]
                sc.set_offsets(np.c_[x,y])
                ax.set_title(mp4titles[q] + ", e vs. a, time = " + str(i*10) + " years (v_inc = +" + str(v) + " km/s)", fontsize = 16, y = 1.04)

            ani = FuncAnimation(fig, animate_eccs_separate, 
                            frames=np.arange(0, 201, 1), interval=200, repeat=True) 
            
            savefolder = parent + '/Plots/all_ejecta/vincs_separate/' + str(v) + 'vinc/per_planet/e_v_a/'
            f = 'e_v_a_' + str(v) + 'vinc_' + mp4filenames[q] + ".mp4"
            plt.rcParams['animation.ffmpeg_path'] = r'/Users/larkpie/opt/anaconda3/envs/venv/bin/ffmpeg'
            writervideo = FFMpegWriter(fps=5) 
            ani.save(savefolder + f, writer=writervideo, dpi=200)
            plt.close()
            
            print(str(v) + 'vinc e v. a for planet' + mp4filenames[q] + ' animated...')

            
    #make separates for inclination
        for q in range(9):
            fig = plt.figure()
            ax = fig.add_subplot(111, ylim = (0., 1.), xlim = (0., 0.25))
            fig.set_size_inches(18, 10)
            x, y = [],[]
            sc = ax.scatter(x,y, color = mp4colors[q], s=10, alpha = 0.8, label = mp4labels[q])
            for j in range(len(semimaj)):
                    ax.vlines(semimaj[j], 0, 2*math.pi, colors=colors[j+2], linestyle = 'dotted')
                    ax.text(semimaj[j], 0.95, object_names[j+1], ha = 'center', fontsize = 12, rotation = 0)
            plt.xlim(0,.25)
            plt.ylim(0,1)
            ax.set_xlabel("Semimajor Axis (AU)", fontsize = 13)
            ax.set_ylabel("Inclination (radians)", fontsize = 13)
            plt.legend(loc=1)


            def animate_incs_separate(i):
                for j in range(num_sims):
                    if j==0:
                        x = list.copy(axes_all[v][j][i][mp4data[q]])
                        y = list.copy(incs_all[v][j][i][mp4data[q]])
                    else:
                        x = x + axes_all[v][j][i][mp4data[q]]
                        y = y + incs_all[v][j][i][mp4data[q]]
                sc.set_offsets(np.c_[x,y])
                ax.set_title(mp4titles[q] + ", inc vs. a, time = " + str(i*10) + " years (v_inc = +" + str(v) + " km/s)", fontsize = 16, y = 1.04)

            ani = FuncAnimation(fig, animate_incs_separate, 
                            frames=np.arange(0, 201, 1), interval=200, repeat=True) 
            
            savefolder = parent + '/Plots/all_ejecta/vincs_separate/' + str(v) + 'vinc/per_planet/inc_v_a/'
            f = 'inc_v_a_' + str(v) + 'vinc_' + mp4filenames[q] + ".mp4"
            plt.rcParams['animation.ffmpeg_path'] = r'/Users/larkpie/opt/anaconda3/envs/venv/bin/ffmpeg'
            writervideo = FFMpegWriter(fps=5) 
            ani.save(savefolder + f, writer=writervideo, dpi=200)
            plt.close()
            
            print(str(v) + 'vinc inc v. a for planet' + mp4filenames[q] + ' animated...')

            
            

def ecc_snapshot_allp(vinc, timeslice, rem=True, esc=True, a=False, b=True, c=True, d=True, e=True, f=True, g=True, h=True):
    """
    DESCRIPTION:
        Outputs a snapshot e vs. a plot for a particular vinc, time, and type(s), using all particles. 

    CALLING SEQUENCE:
        ecc_snapshot_allp(vinc, timeslice, rem=True, esc=True, b=True, c=True, d=True, e=True, f=True, g=True, h=True)
    
    KEYWORDS:
    ## vinc: the vinc you want snapshotted
    ## timeslice: the timeslice you want snapshotted
    ## rem/esc/b/c/.....h: toggling types on/off
    ## label: out of 'all', 'mixed', or the single type (e.g. 'remaining', 'a', etc.) 
    """
    
    types = [rem, esc, b, c, d, e, f, g, h]
    folders = ['remaining', 'esc', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    if sum(types)==1:
        folder = folders[types.index(1)]
    elif sum(types)==len(types):
        folder = 'all'
    else:
        folder = 'mixed'
    
    parent = os.getcwd()
    
    print('loading data...')
    with open(parent + '/Plots/all_ejecta/eccs_all.pkl', 'rb') as file:
        eccs_all = pickle.load(file)
    print ('eccs loaded...')
    with open(parent + '/Plots/all_ejecta/axes_all.pkl', 'rb') as file:
        axes_all = pickle.load(file)
    print ('axes loaded...')
        
    num_sims = 60
        
    colors = ('palegreen', 'k', 'w','r', 'c', 'm', 'gold', 'darkgrey', 'b', 'g')
    groups = ('remaining', 'escaped', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h')
    
    data = []
    co = []
    gr = []
    
    if(rem):
        for j in range(num_sims):
            if j==0:
                xrem = list.copy(axes_all[vinc][j][timeslice]["remaining"])
                yrem = list.copy(eccs_all[vinc][j][timeslice]["remaining"])
            else:
                xrem = xrem + axes_all[vinc][j][timeslice]["remaining"]
                yrem = yrem + eccs_all[vinc][j][timeslice]["remaining"]
        drem = (xrem, yrem)
        data.append(drem)
        co.append(colors[0])
        gr.append(groups[0])
    if(esc):
        for j in range(num_sims):
            if j==0:
                xesc = list.copy(axes_all[vinc][j][timeslice]["escaped"])
                yesc = list.copy(eccs_all[vinc][j][timeslice]["escaped"])
            else:
                xesc = xesc + axes_all[vinc][j][timeslice]["escaped"]
                yesc = yesc + eccs_all[vinc][j][timeslice]["escaped"]
        desc = (xesc, yesc)
        data.append(desc)
        co.append(colors[1])
        gr.append(groups[1])
    if(a):
        for j in range(num_sims):
            if j==0:
                xa = list.copy(axes_all[vinc][j][timeslice]["a"])
                ya = list.copy(eccs_all[vinc][j][timeslice]["a"])
            else:
                xa = xa + axes_all[vinc][j][timeslice]["a"]
                ya = ya + eccs_all[vinc][j][timeslice]["a"]
        da = (xa, ya)
        data.append(da)
        co.append(colors[2])
        gr.append(groups[2])
    if(b):
        for j in range(num_sims):
            if j==0:
                xb = list.copy(axes_all[vinc][j][timeslice]["b"])
                yb = list.copy(eccs_all[vinc][j][timeslice]["b"])
            else:
                xb = xb + axes_all[vinc][j][timeslice]["b"]
                yb = yb + eccs_all[vinc][j][timeslice]["b"]
        db = (xb, yb)
        data.append(db)
        co.append(colors[3])
        gr.append(groups[3])
    if(c):
        for j in range(num_sims):
            if j==0:
                xc = list.copy(axes_all[vinc][j][timeslice]["c"])
                yc = list.copy(eccs_all[vinc][j][timeslice]["c"])
            else:
                xc = xc + axes_all[vinc][j][timeslice]["c"]
                yc = yc + eccs_all[vinc][j][timeslice]["c"]
        dc = (xc, yc)
        data.append(dc)
        co.append(colors[4])
        gr.append(groups[4])
    if(d):
        for j in range(num_sims):
            if j==0:
                xd = list.copy(axes_all[vinc][j][timeslice]["d"])
                yd = list.copy(eccs_all[vinc][j][timeslice]["d"])
            else:
                xd = xd + axes_all[vinc][j][timeslice]["d"]
                yd = yd + eccs_all[vinc][j][timeslice]["d"]
        dd = (xd, yd)
        data.append(dd)
        co.append(colors[5])
        gr.append(groups[5])
    if(e):
        for j in range(num_sims):
            if j==0:
                xe = list.copy(axes_all[vinc][j][timeslice]["e"])
                ye = list.copy(eccs_all[vinc][j][timeslice]["e"])
            else:
                xe = xe + axes_all[vinc][j][timeslice]["e"]
                ye = ye + eccs_all[vinc][j][timeslice]["e"]
        de = (xe, ye)
        data.append(de)
        co.append(colors[6])
        gr.append(groups[6])
    if(f):
        for j in range(num_sims):
            if j==0:
                xf = list.copy(axes_all[vinc][j][timeslice]["f"])
                yf = list.copy(eccs_all[vinc][j][timeslice]["f"])
            else:
                xf = xf + axes_all[vinc][j][timeslice]["f"]
                yf = yf + eccs_all[vinc][j][timeslice]["f"]
        df = (xf, yf)
        data.append(df)
        co.append(colors[7])
        gr.append(groups[7])
    if(g):
        for j in range(num_sims):
            if j==0:
                xg = list.copy(axes_all[vinc][j][timeslice]["g"])
                yg = list.copy(eccs_all[vinc][j][timeslice]["g"])
            else:
                xg = xg + axes_all[vinc][j][timeslice]["g"]
                yg = yg + eccs_all[vinc][j][timeslice]["g"]
        dg = (xg, yg)
        data.append(dg)
        co.append(colors[8])
        gr.append(groups[8])
    if(h):
        for j in range(num_sims):
            if j==0:
                xh = list.copy(axes_all[vinc][j][timeslice]["h"])
                yh = list.copy(eccs_all[vinc][j][timeslice]["h"])
            else:
                xh = xh + axes_all[vinc][j][timeslice]["h"]
                yh = yh + eccs_all[vinc][j][timeslice]["h"]
        dh = (xh, yh)
        data.append(dh)
        co.append(colors[9])
        gr.append(groups[9])

    fig = plt.figure()
    ax = fig.add_subplot(111, ylim = (0., 1.), xlim = (0., 0.25))
    fig.set_size_inches(18,10)
    for j in range(len(semimaj)):
        ax.vlines(semimaj[j], 0, 1, colors=colors[j+3], linestyle = 'dotted')
        ax.text(semimaj[j], 0.95, object_names[j+1], ha = 'center', fontsize = 12, rotation = 0)
        
    for datum, color, group in zip(data, co, gr):
        x, y = datum
        ax.scatter(x, y, alpha = 0.8, c=color, edgecolors = 'none', s=10, label=group)
        
    ax.set_xlabel("Semimajor Axis (AU)", fontsize = 13)
    ax.set_ylabel("Eccentricity", fontsize = 13)
    ax.set_title("e vs. a, time = " + str(timeslice*10) + " years (v_inc = +" + str(vinc) + " km/s)", fontsize = 16, y = 1.02)

    plt.legend(loc=1)
    
    if folder=='all':
        #into all_planets/e_v_a_snapshots
        folderpath = parent + '/Plots/all_ejecta/vincs_separate/' + str(vinc) + 'vinc/all_planets/e_v_a_snapshots/'
    
    else:
        #into per_planet/e_v_a/[folder]_e_v_a_snapshots
        folderpath = parent + '/Plots/all_ejecta/vincs_separate/' + str(vinc) + 'vinc/per_planet/e_v_a/' + folder + '_e_v_a_snapshots/'
    
    filename = str(vinc) + 'vinc_' + str(timeslice) + 'slice_['
    if folder=='all':
        filename += 'all]_e_v_a.png'
        
    elif folder=='mixed':
        if rem:
            filename += 'r'
        if esc:
            filename += 'x'
        if a:
            filename += 'a'
        if b:
            filename += 'b'
        if c:
            filename += 'c'
        if d:
            filename += 'd'
        if e:
            filename += 'e'
        if f:
            filename += 'f'
        if g:
            filename += 'g'
        if h:
            filename += 'h'
        
        filename += ']_e_v_a.png'
    
    else:
        filename += folder + ']_e_v_a.png'
    
    plt.savefig(folderpath + filename, dpi=200)
    plt.close()
    
    print ('plot saved to ' + folderpath + filename)
  


        
def inc_snapshot_allp(vinc, timeslice, rem=True, esc=True, a=False, b=True, c=True, d=True, e=True, f=True, g=True, h=True):
    """
    DESCRIPTION:
        Outputs a snapshot inc vs. a plot for a particular vinc, time, and type(s), using all particles. 

    CALLING SEQUENCE:
        inc_snapshot_allp(vinc, timeslice, rem=True, esc=True, b=True, c=True, d=True, e=True, f=True, g=True, h=True)
    
    KEYWORDS:
    ## vinc: the vinc you want snapshotted
    ## timeslice: the timeslice you want snapshotted
    ## rem/esc/b/c/.....h: toggling types on/off
    ## label: out of 'all', 'mixed', or the single type (e.g. 'remaining', 'a', etc.) 
    """
    
    types = [rem, esc, b, c, d, e, f, g, h]
    folders = ['remaining', 'esc', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    if sum(types)==1:
        folder = folders[types.index(1)]
    elif sum(types)==len(types):
        folder = 'all'
    else:
        folder = 'mixed'
        
    
    parent = os.getcwd()
    
    print('loading data...')
    with open(parent + '/Plots/all_ejecta/incs_all.pkl', 'rb') as file:
        incs_all = pickle.load(file)
    print('incs loaded...')
    with open(parent + '/Plots/all_ejecta/axes_all.pkl', 'rb') as file:
        axes_all = pickle.load(file)
    print('axes loaded...')
        
    num_sims = 60
        
    colors = ('palegreen', 'k', 'w','r', 'c', 'm', 'gold', 'darkgrey', 'b', 'g')
    groups = ('remaining', 'escaped', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h')
    
    data = []
    co = []
    gr = []
    
    if(rem):
        for j in range(num_sims):
            if j==0:
                xrem = list.copy(axes_all[vinc][j][timeslice]["remaining"])
                yrem = list.copy(incs_all[vinc][j][timeslice]["remaining"])
            else:
                xrem = xrem + axes_all[vinc][j][timeslice]["remaining"]
                yrem = yrem + incs_all[vinc][j][timeslice]["remaining"]
        drem = (xrem, yrem)
        data.append(drem)
        co.append(colors[0])
        gr.append(groups[0])
    if(esc):
        for j in range(num_sims):
            if j==0:
                xesc = list.copy(axes_all[vinc][j][timeslice]["escaped"])
                yesc = list.copy(incs_all[vinc][j][timeslice]["escaped"])
            else:
                xesc = xesc + axes_all[vinc][j][timeslice]["escaped"]
                yesc = yesc + incs_all[vinc][j][timeslice]["escaped"]
        desc = (xesc, yesc)
        data.append(desc)
        co.append(colors[1])
        gr.append(groups[1])
    if(a):
        for j in range(num_sims):
            if j==0:
                xa = list.copy(axes_all[vinc][j][timeslice]["a"])
                ya = list.copy(incs_all[vinc][j][timeslice]["a"])
            else:
                xa = xa + axes_all[vinc][j][timeslice]["a"]
                ya = ya + incs_all[vinc][j][timeslice]["a"]
        da = (xa, ya)
        data.append(da)
        co.append(colors[2])
        gr.append(groups[2])
    if(b):
        for j in range(num_sims):
            if j==0:
                xb = list.copy(axes_all[vinc][j][timeslice]["b"])
                yb = list.copy(incs_all[vinc][j][timeslice]["b"])
            else:
                xb = xb + axes_all[vinc][j][timeslice]["b"]
                yb = yb + incs_all[vinc][j][timeslice]["b"]
        db = (xb, yb)
        data.append(db)
        co.append(colors[3])
        gr.append(groups[3])
    if(c):
        for j in range(num_sims):
            if j==0:
                xc = list.copy(axes_all[vinc][j][timeslice]["c"])
                yc = list.copy(incs_all[vinc][j][timeslice]["c"])
            else:
                xc = xc + axes_all[vinc][j][timeslice]["c"]
                yc = yc + incs_all[vinc][j][timeslice]["c"]
        dc = (xc, yc)
        data.append(dc)
        co.append(colors[4])
        gr.append(groups[4])
    if(d):
        for j in range(num_sims):
            if j==0:
                xd = list.copy(axes_all[vinc][j][timeslice]["d"])
                yd = list.copy(incs_all[vinc][j][timeslice]["d"])
            else:
                xd = xd + axes_all[vinc][j][timeslice]["d"]
                yd = yd + incs_all[vinc][j][timeslice]["d"]
        dd = (xd, yd)
        data.append(dd)
        co.append(colors[5])
        gr.append(groups[5])
    if(e):
        for j in range(num_sims):
            if j==0:
                xe = list.copy(axes_all[vinc][j][timeslice]["e"])
                ye = list.copy(incs_all[vinc][j][timeslice]["e"])
            else:
                xe = xe + axes_all[vinc][j][timeslice]["e"]
                ye = ye + incs_all[vinc][j][timeslice]["e"]
        de = (xe, ye)
        data.append(de)
        co.append(colors[6])
        gr.append(groups[6])
    if(f):
        for j in range(num_sims):
            if j==0:
                xf = list.copy(axes_all[vinc][j][timeslice]["f"])
                yf = list.copy(incs_all[vinc][j][timeslice]["f"])
            else:
                xf = xf + axes_all[vinc][j][timeslice]["f"]
                yf = yf + incs_all[vinc][j][timeslice]["f"]
        df = (xf, yf)
        data.append(df)
        co.append(colors[7])
        gr.append(groups[7])
    if(g):
        for j in range(num_sims):
            if j==0:
                xg = list.copy(axes_all[vinc][j][timeslice]["g"])
                yg = list.copy(incs_all[vinc][j][timeslice]["g"])
            else:
                xg = xg + axes_all[vinc][j][timeslice]["g"]
                yg = yg + incs_all[vinc][j][timeslice]["g"]
        dg = (xg, yg)
        data.append(dg)
        co.append(colors[8])
        gr.append(groups[8])
    if(h):
        for j in range(num_sims):
            if j==0:
                xh = list.copy(axes_all[vinc][j][timeslice]["h"])
                yh = list.copy(incs_all[vinc][j][timeslice]["h"])
            else:
                xh = xh + axes_all[vinc][j][timeslice]["h"]
                yh = yh + incs_all[vinc][j][timeslice]["h"]
        dh = (xh, yh)
        data.append(dh)
        co.append(colors[9])
        gr.append(groups[9])

    fig = plt.figure()
    ax = fig.add_subplot(111, ylim = (0., 1.), xlim = (0., 0.25))
    fig.set_size_inches(18,10)
    for j in range(len(semimaj)):
        ax.vlines(semimaj[j], 0, 1, colors=colors[j+3], linestyle = 'dotted')
        ax.text(semimaj[j], 0.95, object_names[j+1], ha = 'center', fontsize = 12, rotation = 0)
        
    for datum, color, group in zip(data, co, gr):
        x, y = datum
        ax.scatter(x, y, alpha = 0.8, c=color, edgecolors = 'none', s=10, label=group)
        
    ax.set_xlabel("Semimajor Axis (AU)", fontsize = 13)
    ax.set_ylabel("Inclination (radians)", fontsize = 13)
    ax.set_title("inc vs. a, time = " + str(timeslice*10) + " years (v_inc = +" + str(vinc) + " km/s)", fontsize = 16, y = 1.02)
    
    plt.legend(loc=1)
    
    if folder=='all':
        #into all_planets/e_v_a_snapshots
        folderpath = parent + '/Plots/all_ejecta/vincs_separate/' + str(vinc) + 'vinc/all_planets/inc_v_a_snapshots/'
    
    else:
        #into per_planet/e_v_a/[folder]_e_v_a_snapshots
        folderpath = parent + '/Plots/all_ejecta/vincs_separate/' + str(vinc) + 'vinc/per_planet/inc_v_a/' + folder + '_inc_v_a_snapshots/'
    
    filename = str(vinc) + 'vinc_' + str(timeslice) + 'slice_['
    if folder=='all':
        filename += 'all]_inc_v_a.png'
        
    elif folder=='mixed':
        if rem:
            filename += 'r'
        if esc:
            filename += 'x'
        if a:
            filename += 'a'
        if b:
            filename += 'b'
        if c:
            filename += 'c'
        if d:
            filename += 'd'
        if e:
            filename += 'e'
        if f:
            filename += 'f'
        if g:
            filename += 'g'
        if h:
            filename += 'h'
        
        filename += ']_inc_v_a.png'
    
    else:
        filename += folder + ']_inc_v_a.png'
    
    plt.savefig(folderpath + filename, dpi=200)
    plt.close()
    
    print ('plot saved to ' + folderpath + filename)
    





        

        
    
    
    
    
    
    
    
    
    
    
    
    
