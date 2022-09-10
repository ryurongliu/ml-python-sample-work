#!/usr/bin/env python
# coding: utf-8

"""
TRAPPIST-1 PLANETARY EJECTA SIMULATION

Methodology:
1. Create the TRAPPIST-1 system.
2. Generate a random spherical distribution of massless ejecta around the source planet, at a distance 1km from the surface.
3. Integrate over the specified timescale. 

Output Data:
1. BINS - snapshots of simulation at designated time intervals.
2. CSV - records orbital and positional information about ejecta at the moment of collision, sorted by planet collided.


"""

import math
import numpy as np
import matplotlib 
import rebound
import os
import csv
import time
import random
import sys
from rebound import hash as h

#define constants
Rsun_to_AU = 0.00465047
Rearth_to_AU = 4.2635e-5
days_to_yr = 1/365.25
Mearth_to_Msun = 3.00274e-6 
deg_to_rad = math.pi/180
km_to_AU = 6.68459e-9
sec_to_yr = 1/31557600
G = 39.441677           #in units of AU, M_Sun, yr

"""FUNCTION DEFINITIONS"""

###simulation-related functions

def draw_sim():
    
    """
    Draw TRAPPIST-1 system. 
    """
    
    sim = rebound.Simulation()
    sim.units = ('yr', 'AU', 'Msun')
    sim.G = G
    sim.integrator = "mercurius"
    sim.collision = "direct"
    sim.collision_resolve = c             #custom collision resolve function
    sim.ri_mercurius.safe_mode = 1        #synchronize after timesteps (solves the major issue!)
    sim.ri_mercurius.hillfac = 3          #default hill radii to switch at
    sim.dt = dt                           #integration timestep 
    
    #add TRAPPIST-1 star and planets
    sim.add(m = 0.0898, r = 0.1192*Rsun_to_AU, hash = 'a')
    sim.add(m = 1.374*Mearth_to_Msun, a=1.154e-2, e=0, inc=(90-89.728)*deg_to_rad, 
            Omega=1, omega=1, M=320.2328615005874, r = 1.116*Rearth_to_AU, hash='b')
    sim.add(m = 1.308*Mearth_to_Msun, a=1.580e-2, e=0, inc=(90-89.778)*deg_to_rad,
            Omega=1, omega=1, M=302.03085481266083, r = 1.097*Rearth_to_AU, hash='c')
    sim.add(m = 0.388*Mearth_to_Msun, a=2.227e-2, e=0, inc=(90-89.896)*deg_to_rad,
            Omega=1, omega=1, M=-421.17961121323583, r = 0.788*Rearth_to_AU, hash='d')
    sim.add(m = 0.692*Mearth_to_Msun, a=2.925e-2, e=0, inc=(90-89.793)*deg_to_rad,
            Omega=1, omega=1, M=-270.15332650367066, r = 0.920*Rearth_to_AU, hash='e')
    sim.add(m = 1.039*Mearth_to_Msun, a=3.849e-2, e=0, inc=(90-89.740)*deg_to_rad,
            Omega=1, omega=1, M=-187.20059131394225, r = 1.045*Rearth_to_AU, hash='f')
    sim.add(m = 1.321*Mearth_to_Msun, a=4.683e-2, e=0, inc=(90-89.742)*deg_to_rad,
            Omega=1, omega=1, M=-136.97239841027107, r = 1.129*Rearth_to_AU, hash='g')
    sim.add(m = 0.326*Mearth_to_Msun, a=6.189e-2, e=0, inc=(90-89.805)*deg_to_rad,
            Omega=1, omega=1, M=-89.87573477315452, r = 0.755*Rearth_to_AU, hash='h')
    
    sim.N_active = len(object_names)         #only planets and star influence each other
    sim.testparticle_type = 0                #test particles do not influence other particles
    sim.move_to_com()                        #move to center of mass frame 
    return sim


def generate_ejecta(sim, planetname, n, v_increment, genseed):
    
    """
    Generates a random spherical distribution of massless ejecta around the source planet.
    
    Calling sequence:
        generate_ejecta(sim, planetname, n, v_increment, genseed)
    
    Arguments:
        *sim ------------the simulation within which to generate ejecta
        *planetname -----the hashname associated with the source planet particle
        *n --------------the number of particles to generate
        *v_increment ----the velocity increment to give each particle, in addition to v_esc of the source planet
        *genseed --------the random generation seed
        
    """
    
    random.seed(genseed)
    
    planet = sim.particles[planetname]              #get planet particle
    
    M = planet.m                                    #planetary mass
    r = planet.r                                    #radius
    v_esc = math.sqrt(2*G*M/r)                      #escape velocity
    
    px = planet.x                                   #planetary position and velocity                
    py = planet.y
    pz = planet.z
    pvx = planet.vx
    pvy = planet.vy
    pvz = planet.vz
    
    #creating each ejecta
    for i in range(n):
        
        name = i+1                                                                #hashname by number
        
        dir_v = [random.random()*2-1, random.random()*2-1, random.random()*2-1]   #random direction
        n_v = dir_v/np.linalg.norm(dir_v)                                         #unit vector in random direction
        vi = n_v*(v_esc + v_increment)                                            #initial velocity vector
        
        vx = vi[0]+pvx                         #add to planetary velocity for total initial velocity wrt the star                 
        vy = vi[1]+pvy
        vz = vi[2]+pvz
        
        x = px+n_v[0]*(r + 1*km_to_AU)        #place particle 1km above the surface radially outward
        y = py+n_v[1]*(r + 1*km_to_AU)        #particle position = planetary position + directional unit vector * (radius + 1 km)
        z = pz+n_v[2]*(r + 1*km_to_AU)
        
        #add to simulation
        sim.add(m=0, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, hash=name)
        
        #add initial data to list, for output
        #inclination, velocity wrt planet, velocity wrt star
        inc = sim.particles[h(name)].inc
        init_data = [name, inc, vi[0], vi[1], vi[2], vx, vy, vz]
        vals['init'].append(init_data)

        
        
def c(sim, c):
    """
    Custom collision resolve function.
    Removes the massless ejecta and stores data for output. 
    """
    
    i, j = c.p1, c.p2
    p1val = sim.contents.particles[i].hash.value                          
    p2val = sim.contents.particles[j].hash.value
 
    for x in range(len(object_names)):                 #compare to each planet/star
        if h(object_names[x]).value == p1val:          #if first particle is a planet or the star 
            part = sim.contents.particles[h(p2val)]    #we want the second particle 
            coldata = [p2val, part.vx, part.vy, part.vz, sim.contents.t]   
            vals[object_names[x]].append(coldata)
            return (2)                                 #remove second particle                          
        
        if h(object_names[x]).value == p2val:          #if second particle is a planet or the star        
            part = sim.contents.particles[h(p1val)]    #we want the first particle
            coldata = [p1val, part.vx, part.vy, part.vz, sim.contents.t]
            vals[object_names[x]].append(coldata)
            return (1)                                 #remove first particle
    
    return 0

def escape_check(sim):
    
    """
    Checks for escaped particles, based on eccentricity.
    """
    ps = sim.particles
    esc_parts = [] 

    for i in range (len(object_names), sim.N):
        particle = ps[i]
        orbit = particle.calculate_orbit(sim.particles[object_names[0]]) #orbital elements around sun 
        ecc = orbit.e
        if (ecc >= 1):                                                   #if eccentricity > 1, consider escaped
            name = particle.hash.value
            data = [name, orbit.a, orbit.e, orbit.inc, orbit.Omega, orbit.omega, orbit.f]
            vals['esc'].append(data)
            

            
###Data recording functions            
            
def make_datalists():
    """
    Creates a dictionary of lists for each recorded dataset. 
    """
    
    vals = {                              
       'init':[],                          #initial hash, inc, velocity, position
        'esc':[],                          #escaped particles (records time + velocity)
    }
    for i in range (len(object_names)):    #planet/star collisions (records time + velocity)
        vals[object_names[i]]=[]
    return vals


def make_filedir(label):
    
    """
    Sets up directory structure for data writing.
    """
    
    parent = os.getcwd()                      #make overall Ejecta Simulation Data folder
    data_fold = "Ejecta_Simulation_Data"
    path = os.path.join(parent, data_fold)
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        pass
    
    new_fold = label                          #make folder for the particular simulation
    newpath = os.path.join(path, new_fold)
    if not os.path.exists(newpath):
        os.mkdir(newpath)
    else:
        sys.exit("Error: please choose a different simulation label.")
    
    return newpath


def make_overall_dict():
    """
    Gathers data for overview file. 
    """
    overall = {
        'Source Planet': sourceplanet,
        'Number of Ejecta': num_ejecta,
        'Velocity Increment (10km/s)': v_increment, 
        'Timesteps': num_years/dt,
        'Step amount (yrs)': dt,
        'Total Time (yrs)': num_years, 
        'Generation seed': genseed,
        'Escaped Particles': len(vals['esc']),
    }
    for x in range (len(object_names)):
        overall[object_names[x]] = (len(vals[object_names[x]])) #number of collisions per planet
    return overall



def write_datafiles(label):
    
    """
    Writes data to csv files. 
    """
    parent = os.getcwd()
    folderpath = parent + '/Ejecta_Simulation_Data/' + label
    
    #overview file
    overall = make_overall_dict()
    overallpath = os.path.join(folderpath, label + '_overview.csv')
    w = csv.writer(open(overallpath, 'w'))
    w.writerow([label])
    for key, val in overall.items():
        w.writerow([key,val])
        
    #initial conditions file
    initpath = os.path.join(folderpath, label + '_particle_inits.csv')
    init_header = ['hash','inc', 'vxPlanet', 'vyPlanet', 'vzPlanet', 'vxStar', 'vyStar', 'vzStar']
    with open(initpath, 'w') as f:
        write = csv.writer(f)
        write.writerow([label])
        write.writerow(init_header)
        write.writerows(vals['init'])
    
    #planetary collisions files
    for x in range (len(object_names)):
        objcol_path = os.path.join(folderpath, label + '_' + object_names[x]+'.csv')
        obj_header = ['hash','vx', 'vy', 'vz', 't']
        with open(objcol_path, 'w') as f:
            write = csv.writer(f)
            write.writerow([label])
            write.writerow(obj_header)
            write.writerows(vals[object_names[x]])
    
    #escaped particles file
    esc_path = os.path.join(folderpath, label + '_escaped.csv')
    esc_header = ['hash', 'semi-maj axis', 'eccentricity', 'inclination', 'long. asc. node', 'arg. pericenter', 'true anomaly']
    with open(esc_path, 'w') as f:
        write = csv.writer(f)
        write.writerow([label])
        write.writerow(esc_header)
        write.writerows(vals['esc'])
        
#ONLY NECESSARY FOR SLURM
def move_outfile(label):
    parent = os.getcwd()
    oldout = parent + '/'+label+'.out'
    newout = parent + '/Ejecta_Simulation_Data/'+label+'/' + label + '.out'
    os.rename(oldout, newout)

#####################################



"""SIMULATION PARAMETERS"""
          
dt = days_to_yr/10
object_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
jobno = str(sys.argv[1])

#***********MOST IMPORTANT!************
num_ejecta = 5000 
num_years = 2000
v_increment =   1      #in km/s
sourceplanet = 'd'

archive_int = 10                    #archive snapshot intervals

#for time interval integration:
chunk = 10                          #how many years to do at a time
proj_chunktime =  5400                 #how long it should take to do that many years (in seconds)
maxtime = 432000                       #maximum integration time (in seconds)
#**************************************

v_inc = v_increment*km_to_AU/sec_to_yr
genseed = random.randint(0, 2**32-1)
label = str(num_ejecta) + 'e_' + str(num_years) + 'y_' + str(v_increment) + 'vinc_' + str(jobno)                

#####################################################


"""ACTUAL SIMULATION"""

data_folder = make_filedir(label)                                                #set up file structure
sim = draw_sim()                                                                 #set up TRAPPIST-1 system
vals = make_datalists()                                                          #set up datalists
generate_ejecta(sim, sourceplanet, num_ejecta, v_inc, genseed)                   #generate ejecta
archive = data_folder + '/' + label + '.bin'                                     #set up bin archive
sim.automateSimulationArchive(archive, interval=archive_int, deletefile = True)


#INTEGRATION LOOP
#keeps track of total time, so we don't exceed cluster computing time limits

maxloops = int(num_years/chunk)                        #total number of loops to do
loops = 0
tic = time.perf_counter()
for i in range (maxloops):
    sim.integrate((i+1)*chunk, exact_finish_time=1)
    toc = time.perf_counter()    
    elapsed = toc-tic
    loops = loops + 1
    print(elapsed, (i+1)*chunk)
    if(elapsed + proj_chunktime >= maxtime):          #if projected time exceeds max time, stop integrating
        break 
         
            
num_years = loops*chunk                                     #actual total time integrated
escape_check(sim)                                           #check for escaped particles
sim.save(data_folder + "/" + label + "_end.bin")            #save final state for restarting
write_datafiles(label)                                      #write data to files
move_outfile(label)                                       #SLURM ONLY - move outfile
