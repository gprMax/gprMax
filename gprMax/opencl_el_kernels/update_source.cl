if (i < NHERTZDIPOLE) {
    {{REAL}} dl;
    int x, y, z, polarisation;

    x = srcinfo1[INDEX2D_SRCINFO(i,0)];
    y = srcinfo1[INDEX2D_SRCINFO(i,1)];
    z = srcinfo1[INDEX2D_SRCINFO(i,2)];
    
    polarisation = srcinfo1[INDEX2D_SRCINFO(i,3)];
    dl = srcinfo2[i];
    
    // 'x' polarised source
    if (polarisation == 0) {
        int materialEx = ID[INDEX4D_ID(0,x,y,z)];
        Ex[INDEX3D_FIELDS(x,y,z)] = Ex[INDEX3D_FIELDS(x,y,z)] - updatecoeffsE[INDEX2D_MAT(materialEx,4)] * srcwaveforms[INDEX2D_SRCWAVES(i,iteration)] * dl * (1 / (dx * dy * dz));
    }

    // 'y' polarised source
    else if (polarisation == 1) {
        int materialEy = ID[INDEX4D_ID(1,x,y,z)];
        Ey[INDEX3D_FIELDS(x,y,z)] = Ey[INDEX3D_FIELDS(x,y,z)] - updatecoeffsE[INDEX2D_MAT(materialEy,4)] * srcwaveforms[INDEX2D_SRCWAVES(i,iteration)] * dl * (1 / (dx * dy * dz));
    }

    // 'z' polarised source
    else if (polarisation == 2) {
        int materialEz = ID[INDEX4D_ID(2,x,y,z)];
        Ez[INDEX3D_FIELDS(x,y,z)] = Ez[INDEX3D_FIELDS(x,y,z)] - updatecoeffsE[INDEX2D_MAT(materialEz,4)] * srcwaveforms[INDEX2D_SRCWAVES(i,iteration)] * dl * (1 / (dx * dy * dz));
    }
}