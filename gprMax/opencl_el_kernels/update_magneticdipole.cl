//  This function updates magnetic field values for magnetic dipole sources.
//
//  Args:
//      NMAGDIPOLE: Total number of magnetic dipoles in the model
//      iteration: Iteration number of simulation
//      dx, dy, dz: Spatial discretisations
//      srcinfo1: Source cell coordinates and polarisation information
//      srcinfo2: Other source information, e.g. length, resistance etc...
//      srcwaveforms: Source waveform values
//      ID, H: Access to ID and field component arrays

if (i < NMAGDIPOLE) {

    int x, y, z, polarisation;

    x = srcinfo1[INDEX2D_SRCINFO(i,0)];
    y = srcinfo1[INDEX2D_SRCINFO(i,1)];
    z = srcinfo1[INDEX2D_SRCINFO(i,2)];
    polarisation = srcinfo1[INDEX2D_SRCINFO(i,3)];

    // 'x' polarised source
    if (polarisation == 0) {
        int materialHx = ID[INDEX4D_ID(3,x,y,z)];
        Hx[INDEX3D_FIELDS(x,y,z)] = Hx[INDEX3D_FIELDS(x,y,z)] - updatecoeffsH[INDEX2D_MAT(materialHx,4)] * srcwaveforms[INDEX2D_SRCWAVES(i,iteration)] * (1 / (dx * dy * dz));
    }

    // 'y' polarised source
    else if (polarisation == 1) {
        int materialHy = ID[INDEX4D_ID(4,x,y,z)];
        Hy[INDEX3D_FIELDS(x,y,z)] = Hy[INDEX3D_FIELDS(x,y,z)] - updatecoeffsH[INDEX2D_MAT(materialHy,4)] * srcwaveforms[INDEX2D_SRCWAVES(i,iteration)] * (1 / (dx * dy * dz));
    }

    // 'z' polarised source
    else if (polarisation == 2) {
        int materialHz = ID[INDEX4D_ID(5,x,y,z)];
        Hz[INDEX3D_FIELDS(x,y,z)] = Hz[INDEX3D_FIELDS(x,y,z)] - updatecoeffsH[INDEX2D_MAT(materialHz,4)] * srcwaveforms[INDEX2D_SRCWAVES(i,iteration)] * (1 / (dx * dy * dz));
    }
}