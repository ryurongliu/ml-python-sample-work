import numpy as np
from astropy.io import fits, ascii
import os
import glob
import sys
import time

#np.set_printoptions(threshold=sys.maxsize)

from prelims.param import get_param
from tkinter.filedialog import askopenfilename
from specphotcal.charis_specphot_cal import specphot_cal
from cals.get_charis_wvlh import get_wvlh
from setup.charis_get_constant import get_constant
from subroutines.get_xycind import get_xycind
from subroutines.dist_circle import dist_circle
from subroutines.gcntrd import gcntrd
from subroutines.cntrd import cntrd
from subroutines.aper import aper
from subroutines.profrad_tc import profrad_tc
from subroutines.charis_snratio_sub import snratio_sub
#from charis_snratio_sub import snratio_sub
from subroutines.planefit import planefit

#from scipy.signal import medfilt2d
#from scipy.ndimage import median_filter
from subroutines.filter_image import filter_image


import matplotlib.pyplot as plt

def extract_1d_spectrum(pfname, datacube=None, coords=None, fitcollapse=None, 
                        fitslice=None, fitbckgd=None, extended=None, 
                        throughputcor=None, fitrange=None, nozerosky=None, 
                        noprad=None, filt_slice=None, filtsnr=None,
                        nocalerror=None, bklim=None, bklir=None, pick=None,
                        r_ap=None, calcube=None, fname=None, startname=None,
                        mag=None, oplotstar=None, clipoutliers=None,
                        plotfromzero=None, yclip=None, prefname=None,
                        suffname=None, test=None, verbose=None, breakout=None, 
                        snrout=None, fluxunits=None,
                        outputspec=None):
    
    #Built off of GPI's spectral extraction program but probably more robust
    
    """Preliminary"""

    bigtic = time.perf_counter()

    #define data directory
    parent = os.getcwd()
    reducdir = parent + '/reduc/'
    
    #data directory
    datadir = reducdir + 'reg/'
    subdir = 'proc/'
    reducdir+= subdir
    
    #information about the target properties
    hmag = get_param('HMAG', pfname=pfname)
    ehmag = get_param('EHMAG', pfname=pfname)
    spt = get_param('SPTYPE', pfname=pfname)
    
    if bklim is None:
        bklim = 4
    if bklir is None:
        bklir = 2
    bklim = max(bklim, 2.999999) #ensure that the background region is greater than 4*r
    bklir = max(bklir, .99999)
    
    if datacube is not None:
        extractcube = datacube
        extractcube = reducdir + datacube
    else:
        extractcube = askopenfilename(initialdir=parent, title="Select the Data Cube From Which You Want to Extract a Spectrum", filetypes=(("fits files", "*.fits"),("all files", "*.*")))
    
    #read in data cube, if the cube has been flux-calibrated then proceed, if not then do a flux calibration
    
    hdul = fits.open(extractcube)
    h1 = hdul[1].header
    
    #if 'FLUXUNIT' has been found, then your datacube is flux-calibrated. proceed with extracting a spectrum. If it has not been found, then do a flux calibration first.
    if 'FLUXUNIT' not in h1:
        """porting charis_specphot_cal.py here"""
        specphot_cal(pfname, pick=pick, r_ap=r_ap, datacube=extractcube, calcube=calcube, filtername=fname, starname=starname, mag=mag, prefname=prefname, suffname=suffname, test=test, fluxunits=fluxunits, outfilename=outfilename)
        print('outfile name is', outfilename)
        
        extractcube=outfilename
        print('extractcube is..', extractcube)
        
        #the outfile is now the file from which you will extract a cube
        
        
    """Extract a Spectrum"""
    
    #1. Define position
    if coords is None:
        xpos = int(input("Input X Coordinate "))
        ypos = int(input("Input Y Coordinate "))
        #print(type(xpos))
    else:
        xpos = coords[0] #x position
        ypos = coords[1] #y position
    xposf = xpos
    yposf = ypos
    
    #2. Read in the datacube
    hdul = fits.open(extractcube)
    data = hdul[1].data
    h1 = hdul[1].header
    h0 = hdul[0].header
    wvlh_charis = get_wvlh(h0) #pull wavelengths
    #Telescope Diameter for Subaru
    Dtel = get_constant('Dtel') #7.9d0 visible pupil for SCExAO
    #pixel scale 16.4 mas/pixel
    pixscale = get_constant('pixscale') #0.0164 nominally 
    fwhm = 1.0*(1e-6*(wvlh_charis*1e-3)/Dtel)*(180.*3600./np.pi)/pixscale
    
    nlambda = data.shape[0]
    aperrad = np.zeros(nlambda)
    
    #3. Define extraction radius
    for i in range(nlambda):
        aperrad[i] = h1['R_AP' + str(i)]
    
    #4. If you want to fit the position of the companion in the collapsed cube, then throw the fitcollapse switch
    data_col = np.nanmedian(data, axis=0)
    med_wvlh = np.nanmedian(wvlh_charis)
    fwhm_col = 1.0*(1e-6*(med_wvlh*1e-3)/Dtel)*(180.*3600./np.pi)/pixscale
    
    if fitrange is None:
        fitrange = np.arange(len(aperrad))
        
    if fitcollapse is not None:
        apmed = np.nanmedian(aperrad[fitrange])
    
        if fitbckgd is not None:
            if fitrange is None:
                image = data_col
            else:
                image = np.nanmedian(data[fitrange, :, :], axis=0)
                data_col = image

            fits.writeto('imagefits.fits', image, overwrite=True)

            #to start, get indices of the "source", defined as < 2*rad
            sourceind = get_xycind(201, 201, xposf, yposf, 2*apmed)
            bckind0 = get_xycind(201, 201 ,xposf, yposf, bklim*apmed)
            sourcedist = np.sqrt((xposf-201//2)**2. + (yposf-201//2)**2.)
            gdist = dist_circle(201)
            annulus = np.flatnonzero((gdist>sourcedist-bklir*apmed) & (gdist<sourcedist+bklir*apmed)) #an annulus +/- 1.5 aperture radii
            bckind0 = np.intersect1d(bckind0, annulus)
            background = np.setxor1d(bckind0, sourceind)
            coeff, yfit = planefit(background % 201, background // 201, image.flatten()[background])

            #plane-fit's estimate of the background underneath the source
            xinds = sourceind % 201
            yinds = sourceind // 201
            src_bkg_plane = coeff[0] + coeff[1]*xinds + coeff[2]*yinds
            image[np.unravel_index(sourceind, (201, 201))] -= src_bkg_plane
            xposf, yposf = gcntrd(image, xpos, ypos, fwhm_col)
            if (xposf<0) or (yposf<0):
                xposf, yposf = cntrd(image, xpos, ypos, fwhm_col)

            if extended is not None:
                #if the source is extended
                sourceind = get_xycind(201, 201, xposf, yposf, 2*apmed)
                xinds = sourceind % 201
                yinds = sourceind // 201
                sourceind = np.unravel_index(sourceind, (201, 201))
                #center of mass
                xcen = np.divide(np.sum(1.*xinds*image[sourceind]), (np.sum(image[sourceind])*1.))
                ycen = np.divide(np.sum(1.*yinds*image[sourceind]), (np.sum(image[sourceind])*1.))
                print('xposf', xposf, yposf)
                xposf = xcen
                yposf = ycen

        else:
            xposf, yposf = cntrd(data_col, xpos, ypos, fwhm_col)

            if(xposf<0) or (ypos<0):
                xposf, yposf = gcntrd(data_col, xpos, ypos, fwhm_col)

            if extended is not None:
                #if the source is extended
                sourceind = get_xycind(201, 201, xposf, yposf, 2*apmed)
                xinds = sourceind % 201
                yinds = sourceind // 201
                
                sourceind = np.unravel_index(sourceind, (201, 201))
                #center of mass
                xcen = np.divide(np.sum(1.*xinds*image[sourceind]), (np.sum(image[sourceind])*1.))
                ycen = np.divide(np.sum(1.*yinds*image[sourceind]), (np.sum(image[sourceind])*1.))
                xposf = xcen
                yposf = ycen

            print('centroid is', xposf, yposf)

        print('Fitted Centroid Position is ...', xposf, yposf)
    
    
    if breakout is not None:
        if np.isfinite(xposf)==0 or np.isfinite(yposf)==0:
            xposout = np.nan
            yposout = np.nan
            snrout = np.nan
            return xposout, yposout
    
    fluxdens_spectrum = np.zeros(nlambda)
    efluxdens_spectrum = np.zeros(nlambda)
    snrslice = np.zeros(nlambda)
    
    for j in range(nlambda):
        fwhm_slice = fwhm[j]
        aperradf = aperrad[j]
        sat_skyrad = aperradf*np.array([2, bklim])
        
        #if you want to fit the companion position per wavelength slice then throw the fitslice switch
        if fitslice is not None:
            xposf, yposf = gcntrd(data[j, :, :], xpos, ypos, fwhm_slice)
            print('Fitted Centroid Position is ...', xposf, yposf, ' ... for slice', j)
            
        
        #how to treat the background ....
        #1. nominal: do nothing except subtract radial profile: assume background is zero'd by lstsquares psf sub (usually valid)
        #2. fitbckgd: fit the background to an annular region excluding the companion, interpolate over companion's position
        #3. apply a spatial filter of 10 psf footprints
        #4. do absolutely nothing: usually valid, usually indistinguishable from #1
        
        if noprad is None:
            pr, x, x = profrad_tc(data[j, :, :])
            data [j, :, :] -= pr
            
        if fitbckgd is not None:
            image = data[j, :, :]
            #to start, get indices of the "source", defined as <2*rad
            sourceind = get_xycind(201, 201, xposf, yposf, 2*aperradf)
            bckind0 = get_xycind(201, 201 ,xposf, yposf, sat_skyrad[1])
            sourcedist = np.sqrt((xposf-201//2)**2. + (yposf-201//2)**2.)
            gdist = dist_circle(201)
            annulus = np.flatnonzero((gdist>sourcedist-bklir*aperradf) & (gdist<sourcedist+bklir*aperradf)) #an annulus +/- 1.5 aperture radii
            bckind0 = np.intersect1d(bckind0, annulus)
            background = np.setxor1d(bckind0, sourceind)
            coeff, yfit = planefit(background % 201, background // 201, image.flatten()[background])

            #plane-fit's estimate of the background underneath the source
            xinds = sourceind % 201
            yinds = sourceind // 201
            src_bkg_plane = coeff[0] + coeff[1]*xinds + coeff[2]*yinds
            
            #plane-fit's estimate of the background around the source
            xwing = background % 201
            ywing = background // 201 
            wings = coeff[0] + coeff[1]*xwing + coeff[2]*ywing
            
            bcksub = np.zeros((201, 201))
            bcksub[np.unravel_index(sourceind, (201, 201))] = src_bkg_plane
            
            if verbose is not None:
                blah = np.zeros((201, 201))
                blah2 = np.zeros((201, 201))
                blah[np.unravel_index(sourceind, (201, 201))] = src_bkg_plane
                blah2[np.unravel_index(background, (201, 201))] = wings
                fits.writeto('blah.fits', blah, overwrite=True)
                fits.writeto('blah2.fits', blah2, overwrite=True)
                fits.writeto('total.fits', data[j, :, :] - (blah+blah2), overwrite=True)
                
                print('background-subbing slice number', j+1)
                fits.writeto('fitted_' + str(j) + '.fits', data[j, :, :] - (blah+blah2))
            
            
        if filt_slice is not None:
            noisyslice = data[j, :, :]
            #noisyslice -= medfilt2d(noisyslice, kernel_size=10*fwhm_slice)
            #noisyslice -= median_filter(noisyslice, size=int(np.floor(10*fwhm_slice)))
            noisyslice -= filter_image(noisyslice, size=int(np.floor(10*fwhm_slice)))
            data[j, :, :] = noisyslice

            if verbose is not None:
                fits.writeto('filtered_' + str(j) + '.fits', noisyslice, overwrite=True)


        #Aperture Photometry
        phpadu = 1.0 #dummy number
        if nozerosky is not None:
            x, x, flux, eflux, sky, skyerr, x, x = aper(data[j, :, :], xposf, yposf, phpadu=phpadu, apr=aperradf, skyrad=sat_skyrad, badpix = [0,0], exact=True, verbose=False)

        else:
            if fitbckgd is None:
                #print(xposf, yposf)
                inputdata = data[j,:,:]
                #notnan = np.nan_to_num(inputdata)
                #print(notnan.shape)
                x, x, flux, eflux, sky, skyerr, x, x = aper(data[j,:,:], np.array([xposf]),
                                                            np.array([yposf]), phpadu=phpadu, apr=aperradf,setskyval=0,
                                                            skyrad=sat_skyrad, badpix = [0,0], exact=True, verbose=False)
                #print(flux, eflux, sky, skyerr)
                
            else:
                x, x, flux, eflux, sky, skyerr, x, x = aper(data[j, :, :]-bcksub, xposf, yposf, phpadu=phpadu,
                                                            apr=aperradf, skyrad=sat_skyrad, badpix = [0,0],
                                                            exact=True, verbose=False)

        if verbose is not None:
                print('in channel ',j,' the flux is ',flux,data[j,int(np.floor(yposf)),int(np.floor(xposf))])
        #Now pull the spot calibration uncertainty from the fits header
        spotcal_uncert = h1['CERR' + str(j)]

        #now compute the SNR per res element 
        ####major new assumption#####
        #1. if a point source ,the commented out and new versions are the same
        #2. if extended source, it covers more than one res element and you use a larger ap-radius
        #   what you are really doing is summing over mult res elements. 
        #   the calc then implicity assumes that the SNR in the 'core' is a good prxy for the SNR 'out of' the core. 
        #   e.g. that the intensity distribution is flat and these are independent samples. 
        #   the alternative, summing over larger ap, overestimates finite element correction

        #fits.writeto("duh.fits",data[j,:,:],overwrite=True)
        if filtsnr is None:
            #x, x, snrval, x, x, x = snratio_sub(data[j, :, :], fwhm=2*aperradf, coord=[xposf, yposf], finite=True, zero=1, fixpos=1, silent=True)
            x, convimage, snrval, snrmap, noisemap, x = snratio_sub(data[j, :, :], fwhm=fwhm[j], coord=[xposf,yposf], finite=True, zero=1, silent=True, fixpos=True, verbose=1)

        else:
            #x, x, snrval, x, x, x = snratio_sub(data[j, :, :], fwhm=2*aperradf, coord=[xposf, yposf], finite=True, zero=1, silent=True, fixpos=True, filt=1, expfilt=3)
            x, convimage, snrval, snrmap, noisemap, x = snratio_sub(data[j, :, :], fwhm=fwhm[j], coord=[xposf, yposf], finite=True, zero=1, silent=True, fixpos=True, filt=1, expfilt=2)
        #fits.writeto('duh2.fits',data[j,:,:],overwrite=True)
        #fits.writeto('snrmap1d.fits',snrmap,overwrite=True)
        #fits.writeto('noisemap1d.fits',noisemap,overwrite=True)
        #fits.writeto('convimage1d.fits',convimage,overwrite=True)
        #print('j is ',j)
        #if j == 0:
        #  return
        fluxdens_spectrum[j] = flux
        #print("!", snrval)
        if nocalerror is None:
            efluxdens_spectrum[j] = np.sqrt((fluxdens_spectrum[j]/snrval)**2. + (flux*spotcal_uncert)**2.)
            #print('??', efluxdens_spectrum[j])
        else:
            efluxdens_spectrum[j] = np.sqrt((fluxdens_spectrum[j]/snrval)**2. + (0*flux*spotcal_uncert)**2.)
        snrslice[j] = snrval
        #print("???", efluxdens_spectrum)
            
    #ENDFOR line 369
    
    #Now get SNR of collapsed cube
    if filtsnr is None:
        x, x, snrval, snrmapcol, x, x = snratio_sub(data_col, fwhm=fwhm_col, coord=[xposf, yposf], finite=True, zero=1, silent=True, fixpos=1)
    else:
        x, x, snrval, snrmapcol, x, x = snratio_sub(data_col, fwhm=fwhm_col, coord=[xposf, yposf], finite=True, zero=1, silent=True, fixpos=1, filt=1, expfilt=2)
    
    fits.writeto('snrmapcol.fits', snrmapcol, overwrite=True)
    
    if throughputcor is not None:
        if throughputcor !=0:
            #1. search for synth_throughput*txt files
            #2. if you find no files, stop program and tell user to run fwdmod first
            #3. if you only find one file, then use this one.
            #4. if you find more than one, then ask user which one to use
            test = glob.glob('synth_throughput*txt')
            ntest = len(test)
            print(test)
            if ntest==0:
                print('ERROR! You need to run forward-modeling to get a throughput correction estimate first!')
                sys.exit()
            elif ntest==1:
                throughput_est = glob.glob('synth_throughput*txt')
                print(throughput_est)
                f = open(throughput_est[0],"r")
                lines = f.readlines()
                throughput = np.zeros(nlambda)
                for i in range(nlambda):
                    throughput[i] = (lines[i].split()[1])
                f.close()
            else:
                throughput_est = askopenfilename(initialdir=parent, title="Select YourThroughput Correction Map", filetypes=(("txt files", "*.txt"),("all files", "*.*")))
                f = open(throughput_est,"r")
                lines = f.readlines()
                throughput = np.zeros(nlambda)
                for i in range(nlambda):
                    throughput[i] = (lines[i].split()[1])
                #for x in lines:
                #    throughput[x] = (x.split()[1])
                f.close()
                
            fluxdens_spectrum/=throughput
            efluxdens_spectrum/=throughput
    
    maxval = max(fluxdens_spectrum)
    minval = min(fluxdens_spectrum)
    print(fluxdens_spectrum)
    
    if yclip is None:
        yclip = 5
    if clipoutliers is not None:
        maxval = min(maxval, yclip*np.median(fluxdens_spectrum))
        goodclip = np.where(fluxdens_spectrum <= yclip*np.median(fluxdens_spectrum))
        maxval = max(flxudens_spectrum[goodclip])
    
    #decision tree to decide the minimum and maximum values
    #broadband
    if min(wvlh_charis) < 1170 and max(wvlh_charis) > 2300:
        minxrange = 1.0
        maxxrange = 2.5
    
    #J
    if min(wvlh_charis) < 1170 and max(wvlh_charis) <1400:
        minxrange = 1.1
        maxxrange = 1.4
    
    #H
    if min(wvlh_charis) > 1400 and min(wvlh_charis) < 1600:
        minxrange = 1.4
        maxxrange = 1.9
    
    #K
    if min(wvlh_charis) > 1850: 
        minxrange = 1.9
        maxxrange = 2.45
        
    
    #3. Define extraction radius 
    if oplotstar is not None:
        starflux = np.zeros(nlambda)
        for ir in range(nlambda):
            starflux[ir] = h1['FSTAR_' + str(ir)]
        starscalefac = 1*np.median(starflux)/np.median(fluxdens_spectrum)
        print(fluxdens_spectrum)
        print(starflux)
        maxval = max(max(fluxdens_spectrum), max(starflux)/starscalefac)
        minval = min(min(fluxdens_spectrum), min(starflux)/starscalefac)
    
    if plotfromzero is not None:
        minval = 0
    if outputspec is None:
        outputspec = 'spectrum.dat'
    #np.savetxt(outputspec,(wvlh_charis*1e-3, fluxdens_spectrum, efluxdens_spectrum, snrslice))
    ascii.write([wvlh_charis*1e-3, fluxdens_spectrum, efluxdens_spectrum, snrslice], outputspec,
                overwrite=True, formats={'col0': '%1.7f', 'col1': '%1.7f', 'col2':'%1.7f', 'col3':'%1.7f'})
    
    if yclip is None:
        yclip = 5
    
    if clipoutliers is None:
        ####PLOT
        #plt.plot(wvlh_charis*1e-3, fluxdens_spectrum, label='Companion')
        plt.errorbar(wvlh_charis*1e-3, fluxdens_spectrum, yerr=efluxdens_spectrum, ecolor='k', elinewidth=1, capsize=3)
        #plt.errorbar(wvlh_charis*1e-3, fluxdens_spectrum, yerr=efluxdens_spectrum, label='Companion', ecolor='k', elinewidth=1, capsize=3)
        plt.xlim(1, 2.5)
        print(minval, maxval)
        plt.ylim(0.8*minval, 1.1*maxval)
        plt.xlabel('Wavelength (Microns)')
        plt.ylabel('Flux Density (mJy)')
        if oplotstar:
           plt.plot(wvlh_charis*1e-3, starflux/starscalefac, label='Star')
        plt.legend()
        plt.savefig('spectrum.png')
        ####PLOT
        pass
    
    else:
        if throughput is None:
            outliers = np.where(abs(fluxdens_spectrum)>yclip*np.median(fluxdens_spectrum))
            good = np.where(abs(fluxdens_spectrum)<=yclip*np.median(fluxdens_spectrum))
        else:
            outliers = np.where((abs(fluxdens_spectrum)>yclip*median(fluxdens_spectrum)) | (throughput < 0.03))
            good = np.where((abs(fluxdens_spectrum)<=yclip*median(fluxdens_spectrum)) & (throughput >= 0.03))
        
        #PLOT
        #PLOT
    
    if oplotstar is not None:
        print('scale factor is', starscalefac)
    
    errcen = (1/2.35)*fwhm_col/snrval
    
    print('Done!')
    print('Spectrum Extracted at Position', xposf, yposf, '[E,N]', (np.array([xposf, yposf]) - 201//2)*0.0162, 'with rough errors of', errcen*0.0162)
    print('the estimated SNR of the collapsed cube is', snrval)
    
    xposout = (xposf-201//2)*0.0162
    yposout = (yposf-201//2)*0.0162
   
    bigtoc = time.perf_counter()

    print('total time:', bigtoc-bigtic)
 
    return xposout, yposout

if __name__ == '__main__':
    datacube = 'final.fits'
    xs, ys = extract_1d_spectrum('HD33632_low.info', datacube=datacube, coords = [146,95])#, fitcollapse=True)
        
 



                    
                      
    
    
    
    
    
