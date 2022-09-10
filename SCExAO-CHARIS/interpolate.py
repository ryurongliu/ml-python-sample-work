import numpy as np

def interpolate(image, xi, yi, cubic=-0.5, missing = np.nan):
    
    """cubic convolution interpolation, with cubic = kernel coefficient
       if cubic == 0, bilinear interpolation
       if cubic > 0, cubic = -1
       
       use 'missing' to set out-of-bounds values, default NaN
    
    adapted from Keys, R.G. "Cubic Convolution Interpolation for Digital Image Processing", 1981
    
    adapted from IDL interpolate()
    """
    
    #base coordinates for interpolation sample points
    x_j = np.floor(xi).astype(int)
    y_k = np.floor(yi).astype(int) 
    
    interped = np.empty(len(xi))
    
    sz = image.shape
    dimx = sz[0]
    dimy = sz[1]
    
    if(len(xi)!=len(yi)): #check length
        print("INTERPOLATE error: Coordinate arrays must have same length")
        return None
    
    if cubic==0: #do bilinear interp
        for i in range(len(xi)):
            xj = x_j[i]
            yk = y_k[i]
            
            if xj>=dimx-1 or xj<0 or yk>=dimy-1 or yk<0: #out of bounds
                interped[i]= missing
            
            else:
                trip1 = (xj, yk, test[xj, yk])
                trip2 = (xj+1, yk, test[xj+1, yk])
                trip3 = (xj, yk+1, test[xj, yk+1])
                trip4 = (xj+1, yk+1, test[xj+1, yk+1])
                triplets = [trip1, trip2, trip3, trip4]

                interped[i] = bilinear_interpolation(xi[i], yi[i], triplets)
        #endfor
    #end bilinear interp
    
    else: #do cubic convolve interp 
        
        if cubic>0:
            cubic = -1
            
        L = [-1, 0, 1, 2]
        M = [-1, 0, 1, 2]
        

        for i in range(len(xi)):
            x = xi[i]
            y = yi[i]
            xj = x_j[i]
            yk = y_k[i]

            c = np.empty((4,4))
            
            tic = time.perf_counter()
            
            #find coefficients; various boundary conditions
            if xj==0:
                if yk==0: #pad top row and left column
                    c[0,0] = image[0,0]
                    c[0, 1:] = image[0, 0:3]
                    c[1:, 0] = image[0:3, 0]
                    c[1:, 1:] = image[0:3, 0:3]

                elif yk>0 and yk<dimy-2: #pad top row
                    c[0, :] = image[0, yk-1:yk+2+1] 
                    c[1:, :] = image[0:3, yk-1:yk+2+1]

                elif yk==dimy-2: #pad top row and right column 
                    c[0, :3] = image[0, dimy-1-2:dimy]
                    c[0, 3] = image[0, dimy-1]
                    c[1:, 3] = image[0:2+1, dimy-1]
                    c[1:, 0:3] = image[0:2+1, dimy-1-2:dimy]
                else:
                    c[:,:] = missing

            elif xj>0 and xj<dimx-2:
                if yk==0: #pad left column
                    c[:, 0] = image[xj-1:xj+2+1, 0]
                    c[:, 1:4] = image[xj-1:xj+2+1, 0:2+1]

                elif yk>0 and yk<dimy-2: #within bounds
                    c = image[xj-1:xj+2+1, yk-1:yk+2+1]

                elif yk==dimy-2: #pad right column 
                    c[:, 3] = image[xj-1:xj+2+1, dimy-1]
                    c[:, 0:3] = image[xj-1:xj+2+1, dimy-1-2:dimy]
                    
                else:
                    c[:,:] = missing

            elif xj==dimx-2:
                if yk==0: #pad bottom row and left column 
                    c[:3, 0] = image[dimx-1-2:dimx, 0]
                    c[3, 0] = image[dimx-1, 0]
                    c[3, 1:] = image[dimx-1, 0:3]
                    c[:3, 1:] = image[dimx-1-2:dimx, 0:2+1]

                elif yk>0 and yk<dimy-2: #pad bottom row
                    c[3, :] = image[dimx-1, yk-1:yk+2+1]
                    c[0:3, :] = image[dimx-1-2:dimx, yk-1:yk+2+1]

                elif yk==dimy-2: #pad bottom row and right column
                    c[:3, 3] = image[dimx-1-2:dimx, dimy-1]
                    c[3, 3] = image[dimx-1, dimy-1]
                    c[3, :3] = image[dimx-1, dimy-1-2:dimy]
                    c[:3, :3] = image[dimx-1-2:dimx, dimy-1-2:dimy]
                
                else:
                    c[:,:] = missing

            else: #point out of bounds
                c[:, :] = missing
            
            
            total=0 #interpolated value
            
            
            for l in L:
                for m in M:
                    coef = c[l+1, m+1]
                    u_y = u(y-(yk+m), cubic) #interpolation kernel
                    #print(x, xj, l)
                    u_x = u(x-(xj+l), cubic)
                    #print(type(u_y))
                    #print(type(u_x))
                    total += coef*u_y*u_x
            
            interped[i] = total
        #endfor
    #end cubic convolve
            
    return interped



def u(coord, a): #interpolation kernel
    s = np.abs(coord)
    if s>=0 and s<=1:
        return((a+2)*(s**3) - (a+3)*(s**2) + 1)
    elif s>1 and s<2:
        return(a*(s**3) - (5*a)*(s**2) +(8*a)*s -(4*a))
    elif s>=2:
        return 0
    
    
    
def bilinear_interpolation(x, y, points): #bilinear interp from stackoverflow
    #https://stackoverflow.com/questions/8661537/how-to-perform-bilinear-interpolation-in-python
    '''Interpolate (x,y) from values associated with four points.

    The four points are a list of four triplets:  (x, y, value).
    The four points can be in any order.  They should form a rectangle.

        >>> bilinear_interpolation(12, 5.5,
        ...                        [(10, 4, 100),
        ...                         (20, 4, 200),
        ...                         (10, 6, 150),
        ...                         (20, 6, 300)])
        165.0

    '''
    # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation

    points = sorted(points)               # order points by x, then by y
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError('points do not form a rectangle')
    if not x1 <= x <= x2 or not y1 <= y <= y2:
        raise ValueError('(x, y) not within the rectangle')

    return (q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)
           ) / ((x2 - x1) * (y2 - y1) + 0.0)
