//  This function updates electric field values for voltage sources.
//
//  Args:
//      NVOLTSRC: Total number of voltage sources in the model
//      iteration: Iteration number of simulation
//      dx, dy, dz: Spatial discretisations
//      srcinfo1: Source cell coordinates and polarisation information
//      srcinfo2: Other source information, e.g. length, resistance etc...
//      srcwaveforms: Source waveform values
//      ID, E: Access to ID and field component arrays

if (i < NVOLTSRC) {

    {{REAL}} resistance;
    int x, y, z, polarisation;

    x = srcinfo1[INDEX2D_SRCINFO(i,0)];
    y = srcinfo1[INDEX2D_SRCINFO(i,1)];
    z = srcinfo1[INDEX2D_SRCINFO(i,2)];
    polarisation = srcinfo1[INDEX2D_SRCINFO(i,3)];
    resistance = srcinfo2[i];

    // 'x' polarised source
    if (polarisation == 0) {
        if (resistance != 0) {
            int materialEx = ID[INDEX4D_ID(0,x,y,z)];
            Ex[INDEX3D_FIELDS(x,y,z)] = Ex[INDEX3D_FIELDS(x,y,z)] - updatecoeffsE[INDEX2D_MAT(materialEx,4)] * srcwaveforms[INDEX2D_SRCWAVES(i,iteration)] * (1 / (resistance * dy * dz));
        }
        else {
            Ex[INDEX3D_FIELDS(x,y,z)] = -1 * srcwaveforms[INDEX2D_SRCWAVES(i,iteration)] / dx;
        }
    }

    // 'y' polarised source
    else if (polarisation == 1) {
        if (resistance != 0) {
            int materialEy = ID[INDEX4D_ID(1,x,y,z)];
            Ey[INDEX3D_FIELDS(x,y,z)] = Ey[INDEX3D_FIELDS(x,y,z)] - updatecoeffsE[INDEX2D_MAT(materialEy,4)] * srcwaveforms[INDEX2D_SRCWAVES(i,iteration)] * (1 / (resistance * dx * dz));
        }
        else {
            Ey[INDEX3D_FIELDS(x,y,z)] = -1 * srcwaveforms[INDEX2D_SRCWAVES(i,iteration)] / dy;
        }
    }

    // 'z' polarised source
    else if (polarisation == 2) {
        if (resistance != 0) {
            int materialEz = ID[INDEX4D_ID(2,x,y,z)];
            Ez[INDEX3D_FIELDS(x,y,z)] = Ez[INDEX3D_FIELDS(x,y,z)] - updatecoeffsE[INDEX2D_MAT(materialEz,4)] * srcwaveforms[INDEX2D_SRCWAVES(i,iteration)] * (1 / (resistance * dx * dy));
        }
        else {
            Ez[INDEX3D_FIELDS(x,y,z)] = -1 * srcwaveforms[INDEX2D_SRCWAVES(i,iteration)] / dz;
        }
    }
}