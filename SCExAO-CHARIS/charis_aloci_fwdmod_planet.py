import numpy as np
from astropy.io import ascii, fits
import os
import glob
from tkinter.filedialog import askopenfilename

from prelims.param import get_param
from setup.charis_path import get_path
from setup.charis_get_constant import get_constant
from cals.get_charis_wvlh import get_wvlh
from subroutines.nbrlist import nbrlist
from subroutines.filelist import filelist
from subroutines.subarr import subarr
from subroutines.charis_myaper import myaper
from subroutines.psf_gaussian import psf_gaussian
from planetspec.charis_insert_planet_into_cube import insert_planet_into_cube
from filt.charis_imrsub import imrsub
from subroutines.aper import aper
from subroutines.cntrd import cntrd

from psfsub.charis_adialoci import adialoci
from psfsub.charis_sdialoci import sdialoci

from reg.charis_register_cube import cart2pol

def aloci_fwdmod_planet(pfname, reducname=None, sdi_reducname=None,
                        nonorthup=None, prefname=None, gauspsf=None, method=None,
                        planetmethod=None, planetmodel=None, targfile=None,
                        pickpsf=None, reductype=None, adi=None, sdi=None, 
                        adipsdi=None, sdipadi=None, asdi=None, rdi=None, 
                        refpath=None):
    
    parent = os.getcwd()
    reducdir = parent + '/reduc/'
    subdir = 'proc/'
    reducdir1 = reducdir + 'reg/'
    datadir = reducdir1
    reducdir += subdir
    
    reducdirorg = reducdir
    
    #*******
    
    if prefname is None:
        #***Prefixes and such***
        prefname = 'n'
    
    suffname = 'reg_cal'
    
    #****Reduction Type **** 0=ADI, 1=SDI, 2=ADIpSDI, 3=SDIpADI, 4=ASDI
    if reductype is None:
        reductype = 0 #set ADI as default
    if sdi is not None:
        reductype = 1 #set to SDI
    if adipsdi is not None:
        reductype = 2 #set to ADIpSDI
    if sdipadi is not None:
        reductype = 3 #set to SDIpADI
   
    if reducname is None:
        #file name, with full path
        reducname_full_path = askopenfilename(initialdir=parent, title="Select Your Processed Image", filetypes=(("fits files", "*.fits"),("all files", "*.*")))
        
        your_path = reducname_full_path.rfind('/')+1
        
        #determining the long-form of the reduction directory
        reducdirorg = reducname_full_path[:your_path]
        
        #now determining the base name of your reduced file
        reducname = reducname_full_path[your_path:]
    
    
    if (reductype==2) or (reductype==3) and (sdi_reducname is None):
        #SDI
        
        #file name, with full path
        sdi_reducname_full_path = askopenfilename(initialdir=parent, title="Select Your Processed Image (Assuming SDI Subtraction)", filetypes=(("fits files", "*.fits"),("all files", "*.*")))
        
        sdi_your_path = sdi_reducname_full_path.rfind('/')+1
        
        #determining the long-form of the reduction directory
        sdi_reducdirorg = sdi_reducname_full_path[:your_path]
        
        #now determining the base name of your reduced file
        sdi_reducname = sdi_reducname_full_path[your_path:]

    print(planetmethod)    
    if planetmethod is None:
        planetmethod=0
    if planetmodel is not None:
        planetmethod = 1
    print(planetmethod)
    if planetmethod==0:
        inputmodel0 = get_param('planmod', pfname=pfname).strip('\'') 
        modelpath = get_path('planetdir')
        inputmodel = modelpath + inputmodel0
    elif planetmethod==1:
        modelpath = get_path('planetdir')
        inputmodel = modelpath + planetmodel
    else:
        modelpath = get_path('modeldir')
        inputmodel = askopenfilename(initialdir=modelpath, title="Select Planet Model", filetypes=(("fits files", "*.fits"),("all files", "*.*")))
    

    #Not in operation yet
    if (rdi is not None) and (refpath is None):
        refpath = input('Enter the Path of the Reference Library (including quotes):')
    
    #***Location of Simulated Planet***
    #******Method*******
    #0. You manually imput a log(contrast), x position, y position (default)
    #1. You read a file which has log(contrast), x position, y position
    #reading in planet position, brightness ...
    
    if method is None:
        #***default is a manually inputing value
        contrast = input('Enter the log(contrast) at H band for the model planet: ')
        xp0 = float(input('Enter the x position for the model planet in the first image: '))
        yp0 = float(input('Enter the y position for the model planet in the first image: '))
     
    else:
        #get type name
        if method=='manual' or method==0:
            #***manually entering some H-band contrast and x,y position
            contrast = float(input('Enter the log (contrast) at H band for the model planet: '))
            xp0 = float(input('Enter the x position for the model planet in the first image: '))
            yp0 = float(input('Enter the y position for the model planet in the first image: '))
        elif method=='targfile' or method==1:
            targdata = ascii.read(targfile)
            xp0 = targdata['xp0']
            yp0 = targdata['yp0']
            contrast = targdata['contrast']
            
            if len(xp0)==1:
                xp0 = xp0[0]
                yp0 = yp0[0]
                contrast = contrast[0]
        
        elif method=='gui' or method==2:
            magfile = askopenfilename(initialdir=parent, title="Choose the input file (X,Y,H, band Log(Contrast())", filetypes=(("fits files", "*.fits"),("all files", "*.*")))
            magdata = ascii.read(magfile)
            xp0 = magdata['xp0']
            yp0 = magdata['yp0']
            contrast = magdata['contrast']
            
            if len(xp0)==1:
                xp0 = xp0[0]
                yp0 = yp0[0]
                contrast = contrast[0]
        
    #number of planets
    if np.isscalar(xp0):
        nplanet = 1
    else:
        nplanet = len(xp0)
    
    print('The Planet Position is', xp0, yp0, 'with a Log H-band Contrast of', contrast)
    
    #***stuff for CHARIS***
    #telescope diameter for Subaru
    Dtel = get_constant('Dtel') #7.9d0; visible pupil for SCExAO
    pixscale = get_constant('pixscale') #0.0162 nominally
    
    #****the data cubes
    hdul = fits.open(reducdirorg+reducname)
    header0 = hdul[0].header
    refcube = hdul[1].data
    header = hdul[1].header
    reducname_col = reducname.split('.')[0] + '_collapsed.fits'
    refdul = fits.open(reducdirorg + reducname_col)
    hrefcol = refdul[0].header
    refcol = refdul[1].data
    h1col = refdul[1].header
    #**ADI treatment
    
    #******************
    #ALOCI parameters from fits header in output file
    
    #for ADI or SDI-only
    znfwhm = header['loci_nfw']
    zna = header['loci_na']
    zgeom = header['loci_geo']
    zdrsub = header['loci_drs']
    zrmin = header['loci_rmi']
    zrmax = header['loci_rma']
    
    zsvd = header['svd']
    znref = header['loci_nre']
    zpixmask = header['pixmaskv']
    zzero = header['zero']
    zmeanadd = header['meanadd']
    zrsub = header['rsub']
    #zadiztype = header['adiztype']
    
    if zsvd is None:
        zcutoff = 1e-99
    
    #for ADI + SDII on residuals [2], SDI + ADI on residuals [3]
    if (reductype==2) or (reductype==3):
        hdul = fits.open(reducdirorg+sdi_reducname)
        sdiheader0 = hdul[0].header
        sdirefcube = hdul[1].data
        sdiheader = hdul[1].header
        sdi_reducname_col = sdi_reducname.split('.')[0] + '_collapsed.fits'
        refdul = fits.open(reducdirorg+sdi_reducname_col)
        sdihrefcol = refdul[0].header
        sdirefcol = refdul[1].data
        sdi_h1col = refdul[1].header
        hrefcol = sdihrefcol
        refcol = sdirefcol
        
        sdiznfwm = sdiheader['loci_nfw']
        sdizna = sidheader['loci_na']
        sdizgeom = sdiheader['loci_geo']
        sdizdrsub = sdiheader['loci_drs']
        sdizrmin = sdiheader['loci_rmi']
        sdizrmax = sdiheader['loci_rma']
        zsdiztype = sdiheader['sdiztype']
        
        sdizsvd = sdiheader['svd']
        sdiznref = sdiheader['loci_nre']
        sdizpixmask = sdiheader['pixmaskv']
        sdizzero = sdiheader['zero']
        sdizmeanadd = sdiheader['meanadd']
        if sdizsvd is None:
            sdizcutoff = 1e-99
        
    if reductype==3:
        zrsub = sdiheader['rsub']
    
    print('LOCI parameters are', znfwhm, zdrsub, zna, zgeom, zrmin, zrmax)
    
    
    imx = header['naxis1']
    dimx = header['naxis2']
    dimy = dimx #assume square arrays
    xc = dimx//2
    yc = dimy//2
    
    #Now Get the Wavelength Vector
    Lambda = get_wvlh(header)
    Lambda *=1e-3
    nlambda = len(Lambda)
    #determin arra of FWHM
    fwhm = 1.0*(1e-6*Lambda/Dtel)*(180.*3600./np.pi)/pixscale
    
    #***extraction radius*****
    aperrad = np.zeros(len(Lambda))
    for ir in range(len(Lambda)):
        aperrad[ir] = header['R_AP' + str(ir)]
    
    
    #**Now get basic image properties
    
    #the below is actually not needed since the charis_northup function now modifies ROTANG to compensate for the north angle offset
    #angoffset = get_constant('angoffset')
    if reductype==0 or reductype==1 or reductype==3 or reductype==4:
        #ADI, SDI, SDI p-ADI, ASDI
        northpa = header['ROTANG']
    elif reductype==2:
        northpa = sdiheader['ROTANG']
    
    
    #define the temporary directory
    tmpdir = reducdir + 'tmp/'
    
    #parameters of the sequence 
    date = get_param('obsdate', pfname=pfname)
    flist = get_param('fnum_sat', pfname=pfname)
    
    filenum = nbrlist(flist)
    files = filelist(filenum, prefname, suffname)
    filesfc = filelist(filenum, prefname, suffname + '_fc')
    nfiles = len(filesfc)
    
    #reading hour angle and parallactic angle at the beginning of exposures
    
    reducdata = ascii.read('reduc.log')
    filenum = reducdata['File_no']
    allxc = reducdata['XC']
    allyc = reducdata['YC']
    allrsat = reducdata['RSAT']
    allha = reducdata['HA']
    allpa = np.array(reducdata['Par_Angle'])
    
    #***now define PSF size
    #***PSF
    dpsf = 21
    intr_psf = np.zeros((nlambda, dpsf, dpsf))
    #***
    #for now we are going to do just one planet
    print('loop size is', nplanet)
    
    #****Define Your Prior: the Intrinsic PSF****
    #****
    
    #1. 
    #use an empirical PSF
    
    if gauspsf is None:
        if pickpsf is None:
            #automatically
            fname = glob.glob(parent + '/psfmodel/psfcube_med.fits')
            c = len(fname)
            
            if c==0:
                #manually:
                fname = askopenfilename(initialdir=parent, title="Pick Reference PSF Files", filetypes=(("fits files", "*.fits"),("all files", "*.*")))
                c = n_elements(fname) 
        else:
            fname = askopenfilename(initialdir=parent, title="Pick Reference PSF Files", filetypes=(("fits files", "*.fits"),("all files", "*.*")))
            c = n_elements(fname) 
        
        #loop in wavelength
        for il in range(nlambda):
            #since you have coronographic data and/or saturated data
            
            #best to just use a model PSF, e.g. c=0
            fpeak = 0.
            for n in range(c):
                hdul = fits.open(fname[n])
                imcube = hdul[1].data
                im = imcube[il, :, :]
                im = subarr(im, dpsf)
                #make the PSF a unit PSF: signal of 1 within a FWHM-sized aperture
                im = im/(myaper(im, dpsf//2, dpsf//2, aperrad[il]))[0]
                if n==0:
                    psf = im/c
                else:
                    psf += im/c
            intr_psf[il, :, :] = psf
        
    #2
    #use a simples gaussian
    else:
        for il in range(nlambda):
            #approximate the PSF by a gaussian; should be ok for high Strehl and simple PSF structure (e.g. Subaru or VLT, not Keck)
            a = psf_gaussian(npixel=dpsf, fwhm=fwhm[il], normalize=True)
            a = a/(myaper(a, dpsf//2, dpsf//2, 0.5*fwhm[il]))[0]
            
            intr_psf[il, :, :] = a
    
    #now should have [xp0, yp0] in first image, contrast in H band, planet model, intrinsic PSF
    
    #x, y coordinates of pixels of the PSF centered on pixel 0.0
    
    xpsf = np.outer(np.ones(dpsf), np.arange(dpsf)) - dpsf//2
    ypsf = np.outer(np.arange(dpsf), np.ones(dpsf)) - dpsf//2
    
    #indices of pixels of the psf centered on pixel 0.0 in the big picture
    ipsf = xpsf + ypsf*dimx
    
    imt = np.zeros((nplanet, dpsf, dpsf))
    nsc = 21 #number of the sub-companion
    
    #****Keywords for RA, DEC, individual exposure time (exp1time), and coadds (so exp1time*ncoadds = exptime)****
    radeg = get_param('RA', pfname=pfname)
    decdeg = get_param('DEC', pfname=pfname)
    
    res_flux = np.zeros((nplanet, nlambda)) #residual flux (1-attenuation)
    
    planet_input_spectrum = np.zeros((nfiles, nplanet, nlambda)) #the array of input planet spectra
    
    #loop calculates the subtraction residuals for fake planets of a given brightness
    
    #define the brightness steps
    
    #xp and yp are given in first image, now calculate new xp and yp positions in subsequent images
    dx0 = np.array([xp0 - xc]) #dx in first image
    dy0 = np.array([yp0 - yc]) #dy in first image
    rc0 = np.sqrt(dx0**2. + dy0**2.)
    
    contrast = np.array([contrast])
    
    for n in range(nfiles):
        hdul = fits.open(datadir + files[n])
        h0sci = hdul[0].header
        im = hdul[1].data
        h1sci = hdul[1].header
        #set to empty data cube
        im[:, :, :] = 1e-15
        outputcube = np.copy(im)
        #now, add planet to data cube
        outputcube0 = np.copy(im)
        
        for iplanet in range(nplanet):
            plan_coords0 = np.rad2deg(cart2pol(dx0[iplanet], dy0[iplanet]))
            asc = np.deg2rad(plan_coords0[0]-(allpa[n]-allpa[0])) #array of angles
            
            if nonorthup is None:
                asc += northpa*np.pi/180.
            xposarr = rc0[iplanet]*np.cos(asc)+xc #array of x positions
            yposarr = rc0[iplanet]*np.sin(asc)+yc #array of y positions
            
            outputcube_indiv, outputspec = insert_planet_into_cube(pfname, im, h0sci, h1sci, xposarr, yposarr, contrast[iplanet], intr_psf, inputmodel)
            
            planet_input_spectrum[n, iplanet, :] = outputspec
            outputcube0 += outputcube_indiv
        
        outputcube += outputcube0
        #register image
        
        fits.HDUList([fits.PrimaryHDU(header=h0sci), fits.ImageHDU(outputcube, header=h1sci)]).writeto(datadir + filesfc[n],overwrite=True)
    
    
    #average input planet spectrum
    planet_avg_input_spectrum = np.median(planet_input_spectrum, axis=0)
    #####PLOT
    
    
    #planet-to-star contrast
    starspectrum = np.zeros(len(Lambda))
    #a simple loop to pull out the star's spectrum
    
    for islice in range(len(Lambda)):
        starspectrum[islice]  = h1sci['FSTAR_'+str(islice)]
    star_avg_input_spectrum = np.nanmedian(starspectrum)
    
    for iplanet in range(nplanet):
        planet_to_star_contrast = np.nanmedian(planet_avg_input_spectrum[iplanet, :])/star_avg_input_spectrum
        print('The Planet-To-Star Contrast in the Filter is', planet_to_star_contrast, 'for planet', iplanet+1)
        print('Planet Avg. Flux Is', np.median(planet_avg_input_spectrum[iplanet, :]), 'for planet', iplanet+1)
    
    print('Star Avg. Flux Is', star_avg_input_spectrum)
    
    if zrsub>0:
        imrsub(pfname, prefname=prefname, suffname=suffname, rmax=80, prad=True, fc=True)
        suffname = 'rsub_fc'
    
    print('LOCI parameters are', znfwhm, zdrsub, zna, zgeom, zrmin, zrmax, znref)
    
    itc = 0
    
    print('xp0', xp0)
    print('is scalar', np.isscalar(xp0))

    if reductype==0:
        #just ADI
        #A-LOCI with Forward-Model
        adialoci(pfname, prefname=prefname, rsubval=zrsub, nfwhm=znfwhm, drsub=zdrsub, na=zna, geom=zgeom, rmin=zrmin, rmax=zrmax, svd=zsvd, nref=znref, pixmask=zpixmask, zero=zzero, meanadd=zmeanadd, fc=True, fwdmod=True, outfile='res'+str(itc)+'.fits')
    
    
    elif reductype==1:
        #just SDI
        sdialoci(pfname, fwdmod=True, prefname=prefname, rsubval=zrsub, nfwhm=znfwhm, drsub=zdrsub, na=zna, geom=zgeom, rmin=zrmin, rmax=zrmax, svd=zsvd, nref=znref, pixmask=zpixmask, zero=zzero, meanadd=zmeanadd, fc=True, outfile='res'+str(itc)+'.fits')
    
    
    elif reductype==2:
        #ADI, SDI on post-ADI residuals
        adialoci(pfname, prefname=prefname, rsubval=zrsub, nfwhm=znfwhm, drsub=zdrsub, na=zna, geom=zgeom, rmin=zrmin, rmax=zrmax, svd=zsvd, nref=znref, pixmask=zpixmask, zero=zzero, meanadd=zmeanadd, fc=True, fwdmod=True, norot=True)
        
        sdialoci(pfname, fwdmod=True, prefname=prefname, suffname='_alocisub', postadi=True, nfwhm=sdiznfwhm, drsub=sdizdrsub, na=sdizna, geom=sdizgeom, rmin=sdizrmin, rmax=sdizrmax, svd=sdizsvd, nref=sdiznref, pixmask=sdizpixmask, zero=sdizzero, meanadd=sdizmeanadd, fc=True, outfile='res'+str(itc)+'.fits')
    
    
    elif reductype==3:
        #SDI, ADI on post-SDI residuals
        sdialoci(pfname, fwdmod=True, prefname=prefname, norot=True, nfwhm=sdiznfwhm, drsub=sdizdrsub, na=sdizna, geom=sdizgeom, rmin=sdizrmin, rmax=sdizrmax, svd=sdizvd, mref=sdiznref, pixmask=sdizpixmask, zero=sdizzero, meanadd=sdizmeanadd, fc=True)
        
        adialoci(pfname, prefname=prefname, rsubval=zrusb, nfwhm=znfwhm, drsub=zdrsub, na=zna, geom=zgeom, rmin=zrmin, rmax=zrmax, svd=zsvd, nref=znref, pixmask=zpixmask, zero=zzero, meanadd=zmeanadd, suffname='_sdialocisub', fc=True, fwdmod=True, postsdi=True, outfile='res'+str(itc)+'.fits')
    
    
    else:
        print('cannot find reduction method')
        print('FWD-Mod Failure')
        sys.exit()  
    
    #Now read back in the files, rescale them to minimize chi-squared, add fits header information, and save them with unique input file name
    
    #your original file
    #remember......
    
    #the empty cube, after processing
    hdul = fits.open(reducdir + 'res' + str(itc) + '.fits')
    h0cube = hdul[0].header
    imcube = hdul[1].data
    h1cube = hdul[1].header
    
    modelname = 'res' + str(itc) + '.fits'
    modelbasename = modelname.split('.')[0]
    
    #wavelength-collapsed version
    hdul = fits.open(reducdir + 'res' + str(itc) + '_collapsed.fits')
    h0col = hdul[0].header
    imcol = hdul[1].data
    h1col = hdul[1].header
    
    #for now do just one planet
    #***now Loop on Wavelength and then Planet to get Attenuation vs. Channel
    for iplanet in range(nplanet):
        for ilambda in range(nlambda):
            
            imslice = imcube[ilambda, :, :]
            
            #compute the flux at the original position
            if np.isscalar(xp0):
                print('scalar xp0')
                x, x, flux, eflux, sky, skyerr, x, x = aper(imslice, xp0, yp0, phpadu=1, apr=aperrad[ilambda], skyrad=np.array([2,6])*2*aperrad[ilambda], setskyval=0, exact=True, verbose=False)
            else:
                x, x, flux, eflux, sky, skyerr, x, x = aper(imslice, xp0[iplanet], yp0[iplanet], phpadu=1, apr=aperrad[ilambda], skyrad=np.array([2,6])*2*aperrad[ilambda], setskyval=0, exact=True, verbose=False)
            
            #now compare to the original, input flux to get the attenuation
            
            print('flux', flux)
            print('planet_avg_spectrum', planet_avg_input_spectrum)
            
            res_flux[iplanet, ilambda] = flux/planet_avg_input_spectrum[iplanet, ilambda]
            print('res_flux', res_flux[iplanet, ilambda])
            print('Throughput for Planet', str(iplanet+1), 'at Wavelength', str(Lambda[ilambda]), 'with contrast of', planet_avg_input_spectrum[iplanet, ilambda]/starspectrum[ilambda], 'is ...', res_flux[iplanet, ilambda])
    
    for iplanet in range(nplanet):
        file = open('synth_throughput'+str(iplanet)+'.txt', 'w')
        for i in range(len(Lambda)):
            file.write("{:f} {:f}\n".format(Lambda[i], res_flux[iplanet, i]))
        file.close()
        
      
        
    #calculate astrometry offsets from cntrd
    reducname_col = reducname.split('.')[0] + '_collapsed.fits'
    refdul = fits.open(reducdirorg + reducname_col)
    refcol = refdul[1].data
    
    if np.isscalar(xp0):
        x_est, y_est = cntrd(refcol, xp0, yp0, 2.64)
        delta_x, delta_y = xp0-x_est, yp0-y_est
    
        file = open('synth_astrom_offset.txt', 'w')
        file.write("{:f} {:f} {:f} {:f} {:f} {:f} {:f}\n".format(0, xp0, yp0, x_est, y_est, delta_x, delta_y))
    
    else:
        x_est, y_est = cntrd(refcol, xp0, yp0, 2.64)
        delta_x, delta_y = xp0-x_est, yp0-y_est
        
        file = open('synth_astrom_offset.txt', 'w')
        for i in range (len(xp0)):
            file.write("{:f} {:f} {:f} {:f} {:f} {:f} {:f}\n".format(i, xp0[i], yp0[i], x_est[i], y_est[i], delta_x[i], delta_y[i]))
    
    file.close()
        

        
    
