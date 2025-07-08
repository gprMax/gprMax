from string import Template

update_e =  Template("""
    // Macros for converting subscripts to linear index:
    #define INDEX2D_MAT(m, n) (m)*($NY_MATCOEFFS)+(n)
    #define INDEX2D_MATDISP(m, n) (m)*($NY_MATDISPCOEFFS)+(n)
    #define INDEX3D_FIELDS(i, j, k) (i)*($NY_FIELDS)*($NZ_FIELDS)+(j)*($NZ_FIELDS)+(k)
    #define INDEX4D_ID(p, i, j, k) (p)*($NX_ID)*($NY_ID)*($NZ_ID)+(i)*($NY_ID)*($NZ_ID)+(j)*($NZ_ID)+(k)
    #define INDEX4D_T(p, i, j, k) (p)*($NX_T)*($NY_T)*($NZ_T)+(i)*($NY_T)*($NZ_T)+(j)*($NZ_T)+(k)
    #define IDX2D_MAT(m, n) (m)*($NY_MATCOEFFS)+(n)
    #define IDX2D_MATDISP(m, n) (m)*($NY_MATDISPCOEFFS)+(n)
    #define IDX3D_FIELDS(i, j, k) (i)*($NY_FIELDS)*($NZ_FIELDS)+(j)*($NZ_FIELDS)+(k)
    #define IDX4D_ID(p, i, j, k) (p)*($NX_ID)*($NY_ID)*($NZ_ID)+(i)*($NY_ID)*($NZ_ID)+(j)*($NZ_ID)+(k)
    #define IDX4D_T(p, i, j, k) (p)*($NX_T)*($NY_T)*($NZ_T)+(i)*($NY_T)*($NZ_T)+(j)*($NZ_T)+(k)
                       
    /////////////////////////////////////////////////
    // Electric field updates - standard materials //
    /////////////////////////////////////////////////

    extern "C" __global__ void update_e(int NX,
                                        int NY, 
                                        int NZ, 
                                        const unsigned int* __restrict__ ID, 
                                        $REAL *Ex, 
                                        $REAL *Ey, 
                                        $REAL *Ez, 
                                        const $REAL* __restrict__ Hx, 
                                        const $REAL* __restrict__ Hy, 
                                        const $REAL* __restrict__ Hz,
                                        const $REAL* __restrict__ updatecoeffsE,
                                        const $REAL* __restrict__ updatecoeffsH) {

    // Electric field updates - normal materials.
    //
    //  Args:
    //      NX, NY, NZ: Number of cells of the model domain.
    //      ID, E, H: Access to ID and field component arrays.

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for 3D field arrays
    int x = i / ($NY_FIELDS * $NZ_FIELDS);
    int y = (i % ($NY_FIELDS * $NZ_FIELDS)) / $NZ_FIELDS;
    int z = (i % ($NY_FIELDS * $NZ_FIELDS)) % $NZ_FIELDS;

    // Convert the linear index to subscripts for 4D material ID array
    int x_ID = (i % ($NX_ID * $NY_ID * $NZ_ID)) / ($NY_ID * $NZ_ID);
    int y_ID = ((i % ($NX_ID * $NY_ID * $NZ_ID)) % ($NY_ID * $NZ_ID)) / $NZ_ID;
    int z_ID = ((i % ($NX_ID * $NY_ID * $NZ_ID)) % ($NY_ID * $NZ_ID)) % $NZ_ID;

    // Ex component
    if ((NY != 1 || NZ != 1) && x >= 0 && x < NX && y > 0 && y < NY && z > 0 && z < NZ) {
        int materialEx = ID[IDX4D_ID(0,x_ID,y_ID,z_ID)];
        Ex[IDX3D_FIELDS(x,y,z)] = updatecoeffsE[IDX2D_MAT(materialEx,0)] * Ex[IDX3D_FIELDS(x,y,z)] +
                                    updatecoeffsE[IDX2D_MAT(materialEx,2)] * (Hz[IDX3D_FIELDS(x,y,z)] - Hz[IDX3D_FIELDS(x,y-1,z)]) -
                                updatecoeffsE[IDX2D_MAT(materialEx,3)] * (Hy[IDX3D_FIELDS(x,y,z)] - Hy[IDX3D_FIELDS(x,y,z-1)]);
    }

    // Ey component
    if ((NX != 1 || NZ != 1) && x > 0 && x < NX && y >= 0 && y < NY && z > 0 && z < NZ) {
        int materialEy = ID[IDX4D_ID(1,x_ID,y_ID,z_ID)];
        Ey[IDX3D_FIELDS(x,y,z)] = updatecoeffsE[IDX2D_MAT(materialEy,0)] * Ey[IDX3D_FIELDS(x,y,z)] +
                                    updatecoeffsE[IDX2D_MAT(materialEy,3)] * (Hx[IDX3D_FIELDS(x,y,z)] - Hx[IDX3D_FIELDS(x,y,z-1)]) -
                                    updatecoeffsE[IDX2D_MAT(materialEy,1)] * (Hz[IDX3D_FIELDS(x,y,z)] - Hz[IDX3D_FIELDS(x-1,y,z)]);
                       }

    // Ez component
    if ((NX != 1 || NY != 1) && x > 0 && x < NX && y > 0 && y < NY && z >= 0 && z < NZ) {
        int materialEz = ID[IDX4D_ID(2,x_ID,y_ID,z_ID)];
        Ez[IDX3D_FIELDS(x,y,z)] = updatecoeffsE[IDX2D_MAT(materialEz,0)] * Ez[IDX3D_FIELDS(x,y,z)] +
                                    updatecoeffsE[IDX2D_MAT(materialEz,1)] * (Hy[IDX3D_FIELDS(x,y,z)] - Hy[IDX3D_FIELDS(x-1,y,z)]) -
                                    updatecoeffsE[IDX2D_MAT(materialEz,2)] * (Hx[IDX3D_FIELDS(x,y,z)] - Hx[IDX3D_FIELDS(x,y-1,z)]);
    }

}
    """)

update_m = Template("""
    // Macros for converting subscripts to linear index:
    #define IDX2D_MAT(m, n) (m)*($NY_MATCOEFFS)+(n)
    #define IDX2D_MATDISP(m, n) (m)*($NY_MATDISPCOEFFS)+(n)
    #define IDX3D_FIELDS(i, j, k) (i)*($NY_FIELDS)*($NZ_FIELDS)+(j)*($NZ_FIELDS)+(k)
    #define IDX4D_ID(p, i, j, k) (p)*($NX_ID)*($NY_ID)*($NZ_ID)+(i)*($NY_ID)*($NZ_ID)+(j)*($NZ_ID)+(k)
    #define IDX4D_T(p, i, j, k) (p)*($NX_T)*($NY_T)*($NZ_T)+(i)*($NY_T)*($NZ_T)+(j)*($NZ_T)+(k)
    #define A ($NY_FIELDS * $NZ_FIELDS)
    extern "C" __global__ void update_magnetic(int NX,
                                                int NY,
                                                int NZ,
                                                const unsigned int* __restrict__ ID,
                                                $REAL *Hx,
                                                $REAL *Hy,
                                                $REAL *Hz,
                                                const $REAL* __restrict__ Ex,
                                                const $REAL* __restrict__ Ey,
                                                const $REAL* __restrict__ Ez,
                                                const $REAL* __restrict__ updatecoeffsE,
                                                const $REAL* __restrict__ updatecoeffsH) {

        // Magnetic field updates - normal materials.
        //
        //  Args:
        //      NX, NY, NZ: Number of cells of the model domain.
        //      ID, E, H: Access to ID and field component arrays.

        int i = blockIdx.x * blockDim.x + threadIdx.x;

        // Convert the linear index to subscripts for 3D field arrays
        int x = i / ($NY_FIELDS * $NZ_FIELDS);
        int y = (i % ($NY_FIELDS * $NZ_FIELDS)) / $NZ_FIELDS;
        int z = (i % ($NY_FIELDS * $NZ_FIELDS)) % $NZ_FIELDS;

        // Convert the linear index to subscripts for 4D material ID array
        int x_ID = (i % ($NX_ID * $NY_ID * $NZ_ID)) / ($NY_ID * $NZ_ID);
        int y_ID = ((i % ($NX_ID * $NY_ID * $NZ_ID)) % ($NY_ID * $NZ_ID)) / $NZ_ID;
        int z_ID = ((i % ($NX_ID * $NY_ID * $NZ_ID)) % ($NY_ID * $NZ_ID)) % $NZ_ID;

        // Hx component
        if (NX != 1 && x > 0 && x < NX && y >= 0 && y < NY && z >= 0 && z < NZ) {
            int materialHx = ID[IDX4D_ID(3,x_ID,y_ID,z_ID)];
            Hx[IDX3D_FIELDS(x,y,z)] = updatecoeffsH[IDX2D_MAT(materialHx,0)] * Hx[IDX3D_FIELDS(x,y,z)] -
                                        updatecoeffsH[IDX2D_MAT(materialHx,2)] * (Ez[IDX3D_FIELDS(x,y+1,z)] - Ez[IDX3D_FIELDS(x,y,z)]) +
                                        updatecoeffsH[IDX2D_MAT(materialHx,3)] * (Ey[IDX3D_FIELDS(x,y,z+1)] - Ey[IDX3D_FIELDS(x,y,z)]);
        }

        // Hy component
        if (NY != 1 && x >= 0 && x < NX && y > 0 && y < NY && z >= 0 && z < NZ) {
            int materialHy = ID[IDX4D_ID(4,x_ID,y_ID,z_ID)];
            Hy[IDX3D_FIELDS(x,y,z)] = updatecoeffsH[IDX2D_MAT(materialHy,0)] * Hy[IDX3D_FIELDS(x,y,z)] -
                                        updatecoeffsH[IDX2D_MAT(materialHy,3)] * (Ex[IDX3D_FIELDS(x,y,z+1)] - Ex[IDX3D_FIELDS(x,y,z)]) +
                                        updatecoeffsH[IDX2D_MAT(materialHy,1)] * (Ez[IDX3D_FIELDS(x+1,y,z)] - Ez[IDX3D_FIELDS(x,y,z)]);
        }

        // Hz component
        if (NZ != 1 && x >= 0 && x < NX && y >= 0 && y < NY && z > 0 && z < NZ) {
            int materialHz = ID[IDX4D_ID(5,x_ID,y_ID,z_ID)];
            Hz[IDX3D_FIELDS(x,y,z)] = updatecoeffsH[IDX2D_MAT(materialHz,0)] * Hz[IDX3D_FIELDS(x,y,z)] -
                                        updatecoeffsH[IDX2D_MAT(materialHz,1)] * (Ey[IDX3D_FIELDS(x+1,y,z)] - Ey[IDX3D_FIELDS(x,y,z)]) +
                                        updatecoeffsH[IDX2D_MAT(materialHz,2)] * (Ex[IDX3D_FIELDS(x,y+1,z)] - Ex[IDX3D_FIELDS(x,y,z)]);
        }
    }
    """)

update_hertzian_dipole = Template("""    
    // Macros for converting subscripts to linear index:
    #define IDX2D_MAT(m, n) (m)*($NY_MATCOEFFS)+(n)
    #define IDX3D_FIELDS(i, j, k) (i)*($NY_FIELDS)*($NZ_FIELDS)+(j)*($NZ_FIELDS)+(k)
    #define IDX4D_ID(p, i, j, k) (p)*($NX_ID)*($NY_ID)*($NZ_ID)+(i)*($NY_ID)*($NZ_ID)+(j)*($NZ_ID)+(k)
    #define IDX2D_SRCINFO(m, n) (m)*$NY_SRCINFO+(n)
    #define IDX2D_SRCWAVES(m, n) (m)*($NY_SRCWAVES)+(n)

    extern "C" __global__ void update_hertzian_dipole(int NHERTZDIPOLE,
                                            int iteration,
                                            $REAL dx,
                                            $REAL dy,
                                            $REAL dz,
                                            const int* __restrict__ srcinfo1,
                                            const $REAL* __restrict__ srcinfo2,
                                            const $REAL* __restrict__ srcwaveforms,
                                            const unsigned int* __restrict__ ID,
                                            $REAL *Ex,
                                            $REAL *Ey,
                                            $REAL *Ez,
                                            const $REAL* __restrict__ updatecoeffsE,
                                            const $REAL* __restrict__ updatecoeffsH) {
        // Updates electric field values for Hertzian dipole sources.
        //
        //  Args:
        //      NHERTZDIPOLE: Total number of Hertzian dipoles in the model.
        //      iteration: Iteration number of simulation.
        //      dx, dy, dz: Spatial discretisations.
        //      srcinfo1: Source cell coordinates and polarisation information.
        //      srcinfo2: Other source information, e.g. length, resistance etc...
        //      srcwaveforms: Source waveform values.
        //      ID, E: Access to ID and field component arrays.

        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < NHERTZDIPOLE) {
            $REAL dl;
            int x, y, z, polarisation;

            x = srcinfo1[IDX2D_SRCINFO(i,0)];
            y = srcinfo1[IDX2D_SRCINFO(i,1)];
            z = srcinfo1[IDX2D_SRCINFO(i,2)];
            polarisation = srcinfo1[IDX2D_SRCINFO(i,3)];
            dl = srcinfo2[i];
            // 'x' polarised source
            if (polarisation == 0) {
                int materialEx = ID[IDX4D_ID(0,x,y,z)];
                Ex[IDX3D_FIELDS(x,y,z)] = Ex[IDX3D_FIELDS(x,y,z)] - updatecoeffsE[IDX2D_MAT(materialEx,4)] *
                                            srcwaveforms[IDX2D_SRCWAVES(i,iteration)] * dl * (1 / (dx * dy * dz));
            }

            // 'y' polarised source
            else if (polarisation == 1) {
                int materialEy = ID[IDX4D_ID(1,x,y,z)];
                Ey[IDX3D_FIELDS(x,y,z)] = Ey[IDX3D_FIELDS(x,y,z)] - updatecoeffsE[IDX2D_MAT(materialEy,4)] *
                                            srcwaveforms[IDX2D_SRCWAVES(i,iteration)] * dl * (1 / (dx * dy * dz));
            }

            // 'z' polarised source
            else if (polarisation == 2) {
                int materialEz = ID[IDX4D_ID(2,x,y,z)];
                Ez[IDX3D_FIELDS(x,y,z)] = Ez[IDX3D_FIELDS(x,y,z)] - updatecoeffsE[IDX2D_MAT(materialEz,4)] *
                                            srcwaveforms[IDX2D_SRCWAVES(i,iteration)] * dl * (1 / (dx * dy * dz));
            }
        }
    }
"""
)

store_outputs = Template("""
    #define IDX3D_RXS(i, j, k) (i)*($NY_RXS)*($NZ_RXS)+(j)*($NZ_RXS)+(k)
    #define IDX2D_RXCOORDS(m, n) (m)*($NY_RXCOORDS)+(n)
    #define IDX3D_FIELDS(i, j, k) (i)*($NY_FIELDS)*($NZ_FIELDS)+(j)*($NZ_FIELDS)+(k)

    extern "C" __global__ void store_outputs(int NRX,
                                                    int iteration,
                                                    const int* __restrict__ rxcoords,
                                                    $REAL *rxs,
                                                    const $REAL* __restrict__ Ex,
                                                    const $REAL* __restrict__ Ey,
                                                    const $REAL* __restrict__ Ez,
                                                    const $REAL* __restrict__ Hx,
                                                    const $REAL* __restrict__ Hy,
                                                    const $REAL* __restrict__ Hz){
    // Stores field component values for every receiver in the model.
    //
    // Args:
    //    NRX: total number of receivers in the model.
    //    rxs: array to store field components for receivers - rows
    //          are field components; columns are iterations; pages are receiver.

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < NRX) {
        int x, y, z;
        x = rxcoords[IDX2D_RXCOORDS(i,0)];
        y = rxcoords[IDX2D_RXCOORDS(i,1)];
        z = rxcoords[IDX2D_RXCOORDS(i,2)];
        rxs[IDX3D_RXS(0,iteration,i)] = Ex[IDX3D_FIELDS(x,y,z)];
        rxs[IDX3D_RXS(1,iteration,i)] = Ey[IDX3D_FIELDS(x,y,z)];
        rxs[IDX3D_RXS(2,iteration,i)] = Ez[IDX3D_FIELDS(x,y,z)];
        rxs[IDX3D_RXS(3,iteration,i)] = Hx[IDX3D_FIELDS(x,y,z)];
        rxs[IDX3D_RXS(4,iteration,i)] = Hy[IDX3D_FIELDS(x,y,z)];
        rxs[IDX3D_RXS(5,iteration,i)] = Hz[IDX3D_FIELDS(x,y,z)];
    }
}

""")

update_voltage_source = Template("""
    // Macros for converting subscripts to linear index:
    #define IDX2D_MAT(m, n) (m)*($NY_MATCOEFFS)+(n)
    #define IDX3D_FIELDS(i, j, k) (i)*($NY_FIELDS)*($NZ_FIELDS)+(j)*($NZ_FIELDS)+(k)
    #define IDX4D_ID(p, i, j, k) (p)*($NX_ID)*($NY_ID)*($NZ_ID)+(i)*($NY_ID)*($NZ_ID)+(j)*($NZ_ID)+(k)
    #define IDX2D_SRCINFO(m, n) (m)*$NY_SRCINFO+(n)
    #define IDX2D_SRCWAVES(m, n) (m)*($NY_SRCWAVES)+(n)

    extern "C" __global__ void update_voltage_source(int NVOLTSRC,
                                            int iteration,
                                            $REAL dx,
                                            $REAL dy,
                                            $REAL dz,
                                            const int* __restrict__ srcinfo1,
                                            const $REAL* __restrict__ srcinfo2,
                                            const $REAL* __restrict__ srcwaveforms,
                                            const unsigned int* __restrict__ ID,
                                            $REAL *Ex,
                                            $REAL *Ey,
                                            $REAL *Ez,
                                            const $REAL* __restrict__ updatecoeffsE,
                                            const $REAL* __restrict__ updatecoeffsH) {

    // Updates electric field values for voltage sources.
    //
    //  Args:
    //      NVOLTSRC: Total number of voltage sources in the model.
    //      iteration: Iteration number of simulation.
    //      dx, dy, dz: Spatial discretisations.
    //      srcinfo1: Source cell coordinates and polarisation information.
    //      srcinfo2: Other source information, e.g. length, resistance etc...
    //      srcwaveforms: Source waveform values.
    //      ID, E: Access to ID and field component arrays.

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < NVOLTSRC) {

        $REAL resistance;
        int x, y, z, polarisation;

        x = srcinfo1[IDX2D_SRCINFO(i,0)];
        y = srcinfo1[IDX2D_SRCINFO(i,1)];
        z = srcinfo1[IDX2D_SRCINFO(i,2)];
        polarisation = srcinfo1[IDX2D_SRCINFO(i,3)];
        resistance = srcinfo2[i];

        // 'x' polarised source
        if (polarisation == 0) {
            if (resistance != 0) {
                int materialEx = ID[IDX4D_ID(0,x,y,z)];
                Ex[IDX3D_FIELDS(x,y,z)] = Ex[IDX3D_FIELDS(x,y,z)] - updatecoeffsE[IDX2D_MAT(materialEx,4)] *
                                            srcwaveforms[IDX2D_SRCWAVES(i,iteration)] * (1 / (resistance * dy * dz));
            }
            else {
                Ex[IDX3D_FIELDS(x,y,z)] = -1 * srcwaveforms[IDX2D_SRCWAVES(i,iteration)] / dx;
            }
        }

        // 'y' polarised source
        else if (polarisation == 1) {
            if (resistance != 0) {
                int materialEy = ID[IDX4D_ID(1,x,y,z)];
                Ey[IDX3D_FIELDS(x,y,z)] = Ey[IDX3D_FIELDS(x,y,z)] - updatecoeffsE[IDX2D_MAT(materialEy,4)] *
                                            srcwaveforms[IDX2D_SRCWAVES(i,iteration)] * (1 / (resistance * dx * dz));
            }
            else {
                Ey[IDX3D_FIELDS(x,y,z)] = -1 * srcwaveforms[IDX2D_SRCWAVES(i,iteration)] / dy;
            }
        }

        // 'z' polarised source
        else if (polarisation == 2) {
            if (resistance != 0) {
                int materialEz = ID[IDX4D_ID(2,x,y,z)];
                Ez[IDX3D_FIELDS(x,y,z)] = Ez[IDX3D_FIELDS(x,y,z)] - updatecoeffsE[IDX2D_MAT(materialEz,4)] *
                                            srcwaveforms[IDX2D_SRCWAVES(i,iteration)] * (1 / (resistance * dx * dy));
            }
            else {
                Ez[IDX3D_FIELDS(x,y,z)] = -1 * srcwaveforms[IDX2D_SRCWAVES(i,iteration)] / dz;
            }
        }
    }
}
""")

update_magnetic_dipole = Template("""
    #define IDX2D_MAT(m, n) (m)*($NY_MATCOEFFS)+(n)
    #define IDX3D_FIELDS(i, j, k) (i)*($NY_FIELDS)*($NZ_FIELDS)+(j)*($NZ_FIELDS)+(k)
    #define IDX4D_ID(p, i, j, k) (p)*($NX_ID)*($NY_ID)*($NZ_ID)+(i)*($NY_ID)*($NZ_ID)+(j)*($NZ_ID)+(k)
    #define IDX2D_SRCINFO(m, n) (m)*$NY_SRCINFO+(n)
    #define IDX2D_SRCWAVES(m, n) (m)*($NY_SRCWAVES)+(n)

    extern "C"  __global__ void update_magnetic_dipole(int NMAGDIPOLE,
                                            int iteration,
                                            $REAL dx,
                                            $REAL dy,
                                            $REAL dz,
                                            const int* __restrict__ srcinfo1,
                                            const $REAL* __restrict__ srcinfo2,
                                            const $REAL* __restrict__ srcwaveforms,
                                            const unsigned int* __restrict__ ID,
                                            $REAL *Hx,
                                            $REAL *Hy,
                                            $REAL *Hz,
                                            const $REAL* __restrict__ updatecoeffsE,
                                            const $REAL* __restrict__ updatecoeffsH) {

    // Updates electric field values for Hertzian dipole sources.
    //
    //  Args:
    //      NMAGDIPOLE: Total number of magnetic dipoles in the model.
    //      iteration: Iteration number of simulation.
    //      dx, dy, dz: Spatial discretisations.
    //      srcinfo1: Source cell coordinates and polarisation information.
    //      srcinfo2: Other source information, e.g. length, resistance etc...
    //      srcwaveforms: Source waveform values.
    //      ID, H: Access to ID and field component arrays.

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < NMAGDIPOLE) {

        int x, y, z, polarisation;

        x = srcinfo1[IDX2D_SRCINFO(i,0)];
        y = srcinfo1[IDX2D_SRCINFO(i,1)];
        z = srcinfo1[IDX2D_SRCINFO(i,2)];
        polarisation = srcinfo1[IDX2D_SRCINFO(i,3)];

        // 'x' polarised source
        if (polarisation == 0) {
            int materialHx = ID[IDX4D_ID(3,x,y,z)];
            Hx[IDX3D_FIELDS(x,y,z)] = Hx[IDX3D_FIELDS(x,y,z)] - updatecoeffsH[IDX2D_MAT(materialHx,4)] *
                                        srcwaveforms[IDX2D_SRCWAVES(i,iteration)] * (1 / (dx * dy * dz));
        }

        // 'y' polarised source
        else if (polarisation == 1) {
            int materialHy = ID[IDX4D_ID(4,x,y,z)];
            Hy[IDX3D_FIELDS(x,y,z)] = Hy[IDX3D_FIELDS(x,y,z)] - updatecoeffsH[IDX2D_MAT(materialHy,4)] *
                                        srcwaveforms[IDX2D_SRCWAVES(i,iteration)] * (1 / (dx * dy * dz));
        }

        // 'z' polarised source
        else if (polarisation == 2) {
            int materialHz = ID[IDX4D_ID(5,x,y,z)];
            Hz[IDX3D_FIELDS(x,y,z)] = Hz[IDX3D_FIELDS(x,y,z)] - updatecoeffsH[IDX2D_MAT(materialHz,4)] *
                                        srcwaveforms[IDX2D_SRCWAVES(i,iteration)] * (1 / (dx * dy * dz));
        }
    }
}
""")

update_electric_dispersive_A = Template("""
    #define INDEX2D_MAT(m, n) (m)*($NY_MATCOEFFS)+(n)
    #define INDEX2D_MATDISP(m, n) (m)*($NY_MATDISPCOEFFS)+(n)
    #define INDEX3D_FIELDS(i, j, k) (i)*($NY_FIELDS)*($NZ_FIELDS)+(j)*($NZ_FIELDS)+(k)
    #define INDEX4D_ID(p, i, j, k) (p)*($NX_ID)*($NY_ID)*($NZ_ID)+(i)*($NY_ID)*($NZ_ID)+(j)*($NZ_ID)+(k)
    #define INDEX4D_T(p, i, j, k) (p)*($NX_T)*($NY_T)*($NZ_T)+(i)*($NY_T)*($NZ_T)+(j)*($NZ_T)+(k)
    #define IDX2D_MAT(m, n) (m)*($NY_MATCOEFFS)+(n)
    #define IDX2D_MATDISP(m, n) (m)*($NY_MATDISPCOEFFS)+(n)
    #define IDX3D_FIELDS(i, j, k) (i)*($NY_FIELDS)*($NZ_FIELDS)+(j)*($NZ_FIELDS)+(k)
    #define IDX4D_ID(p, i, j, k) (p)*($NX_ID)*($NY_ID)*($NZ_ID)+(i)*($NY_ID)*($NZ_ID)+(j)*($NZ_ID)+(k)
    #define IDX4D_T(p, i, j, k) (p)*($NX_T)*($NY_T)*($NZ_T)+(i)*($NY_T)*($NZ_T)+(j)*($NZ_T)+(k)
    #include <hip/hip_complex.h>
    extern "C" __global__ void update_electric_dispersive_A(int NX,
                                    int NY,
                                    int NZ,
                                    int MAXPOLES,
                                    const $COMPLEX* __restrict__ updatecoeffsdispersive,
                                    $COMPLEX *Tx,
                                    $COMPLEX *Ty,
                                    $COMPLEX *Tz,
                                    const unsigned int* __restrict__ ID,
                                    $REAL *Ex,
                                    $REAL *Ey,
                                    $REAL *Ez,
                                    const $REAL* __restrict__ Hx,
                                    const $REAL* __restrict__ Hy,
                                    const $REAL* __restrict__ Hz,
                                    const $REAL* __restrict__ updatecoeffsE,
                                    const $REAL* __restrict__ updatecoeffsH) {

    // Electric field updates - dispersive materials - part A of updates to electric
    //                              field values when dispersive materials
    //                              (with multiple poles) are present.
    //
    //  Args:
    //      NX, NY, NZ: Number of cells of the model domain.
    //      MAXPOLES: Maximum number of dispersive material poles present in model.
    //      updatedispersivecoeffs, T, ID, E, H: Access to update coefficients,
    //                                              dispersive, ID and field
    //                                              component arrays.

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for 3D field arrays
    int x = i / ($NY_FIELDS * $NZ_FIELDS);
    int y = (i % ($NY_FIELDS * $NZ_FIELDS)) / $NZ_FIELDS;
    int z = (i % ($NY_FIELDS * $NZ_FIELDS)) % $NZ_FIELDS;

    // Convert the linear index to subscripts for 4D material ID array
    int x_ID = (i % ($NX_ID * $NY_ID * $NZ_ID)) / ($NY_ID * $NZ_ID);
    int y_ID = ((i % ($NX_ID * $NY_ID * $NZ_ID)) % ($NY_ID * $NZ_ID)) / $NZ_ID;
    int z_ID = ((i % ($NX_ID * $NY_ID * $NZ_ID)) % ($NY_ID * $NZ_ID)) % $NZ_ID;

    // Convert the linear index to subscripts for 4D dispersive array
    int x_T = (i % ($NX_T * $NY_T * $NZ_T)) / ($NY_T * $NZ_T);
    int y_T = ((i % ($NX_T * $NY_T * $NZ_T)) % ($NY_T * $NZ_T)) / $NZ_T;
    int z_T = ((i % ($NX_T * $NY_T * $NZ_T)) % ($NY_T * $NZ_T)) % $NZ_T;

    // Ex component
    if ((NY != 1 || NZ != 1) && x >= 0 && x < NX && y > 0 && y < NY && z > 0 && z < NZ) {
        int materialEx = ID[IDX4D_ID(0,x_ID,y_ID,z_ID)];
        $REAL phi = 0;
        for (int pole = 0; pole < MAXPOLES; pole++) {
            phi = phi + hipCrealf(updatecoeffsdispersive[IDX2D_MATDISP(materialEx,pole*3)]) * hipCrealf(Tx[IDX4D_T(pole,x_T,y_T,z_T)]);
            Tx[IDX4D_T(pole,x_T,y_T,z_T)] = updatecoeffsdispersive[IDX2D_MATDISP(materialEx,1+(pole*3))] * Tx[IDX4D_T(pole,x_T,y_T,z_T)] +
                                            updatecoeffsdispersive[IDX2D_MATDISP(materialEx,2+(pole*3))] * Ex[IDX3D_FIELDS(x,y,z)];
        }
        Ex[IDX3D_FIELDS(x,y,z)] = updatecoeffsE[IDX2D_MAT(materialEx,0)] * Ex[IDX3D_FIELDS(x,y,z)] +
                                    updatecoeffsE[IDX2D_MAT(materialEx,2)] * (Hz[IDX3D_FIELDS(x,y,z)] - Hz[IDX3D_FIELDS(x,y-1,z)]) -
                                    updatecoeffsE[IDX2D_MAT(materialEx,3)] * (Hy[IDX3D_FIELDS(x,y,z)] - Hy[IDX3D_FIELDS(x,y,z-1)]) -
                                    updatecoeffsE[IDX2D_MAT(materialEx,4)] * phi;
    }

    // Ey component
    if ((NX != 1 || NZ != 1) && x > 0 && x < NX && y >= 0 && y < NY && z > 0 && z < NZ) {
        int materialEy = ID[IDX4D_ID(1,x_ID,y_ID,z_ID)];
        $REAL phi = 0;
        for (int pole = 0; pole < MAXPOLES; pole++) {
            phi = phi + hipCrealf(updatecoeffsdispersive[IDX2D_MATDISP(materialEy,pole*3)]) * hipCrealf(Ty[IDX4D_T(pole,x_T,y_T,z_T)]);
            Ty[IDX4D_T(pole,x_T,y_T,z_T)] = updatecoeffsdispersive[IDX2D_MATDISP(materialEy,1+(pole*3))] * Ty[IDX4D_T(pole,x_T,y_T,z_T)] +
                                            updatecoeffsdispersive[IDX2D_MATDISP(materialEy,2+(pole*3))] * Ey[IDX3D_FIELDS(x,y,z)];
        }
        Ey[IDX3D_FIELDS(x,y,z)] = updatecoeffsE[IDX2D_MAT(materialEy,0)] * Ey[IDX3D_FIELDS(x,y,z)] +
                                    updatecoeffsE[IDX2D_MAT(materialEy,3)] * (Hx[IDX3D_FIELDS(x,y,z)] - Hx[IDX3D_FIELDS(x,y,z-1)]) -
                                    updatecoeffsE[IDX2D_MAT(materialEy,1)] * (Hz[IDX3D_FIELDS(x,y,z)] - Hz[IDX3D_FIELDS(x-1,y,z)]) -
                                    updatecoeffsE[IDX2D_MAT(materialEy,4)] * phi;
    }

    // Ez component
    if ((NX != 1 || NY != 1) && x > 0 && x < NX && y > 0 && y < NY && z >= 0 && z < NZ) {
        int materialEz = ID[IDX4D_ID(2,x_ID,y_ID,z_ID)];
        $REAL phi = 0;
        for (int pole = 0; pole < MAXPOLES; pole++) {
            phi = phi + hipCrealf(updatecoeffsdispersive[IDX2D_MATDISP(materialEz,pole*3)]) * hipCrealf(Tz[IDX4D_T(pole,x_T,y_T,z_T)]);
            Tz[IDX4D_T(pole,x_T,y_T,z_T)] = updatecoeffsdispersive[IDX2D_MATDISP(materialEz,1+(pole*3))] * Tz[IDX4D_T(pole,x_T,y_T,z_T)] +
                                            updatecoeffsdispersive[IDX2D_MATDISP(materialEz,2+(pole*3))] * Ez[IDX3D_FIELDS(x,y,z)];
        }
        Ez[IDX3D_FIELDS(x,y,z)] = updatecoeffsE[IDX2D_MAT(materialEz,0)] * Ez[IDX3D_FIELDS(x,y,z)] +
                                    updatecoeffsE[IDX2D_MAT(materialEz,1)] * (Hy[IDX3D_FIELDS(x,y,z)] - Hy[IDX3D_FIELDS(x-1,y,z)]) -
                                    updatecoeffsE[IDX2D_MAT(materialEz,2)] * (Hx[IDX3D_FIELDS(x,y,z)] - Hx[IDX3D_FIELDS(x,y-1,z)]) -
                                    updatecoeffsE[IDX2D_MAT(materialEz,4)] * phi;
    }
}

""")

update_electric_dispersive_B = Template("""
    #define INDEX2D_MAT(m, n) (m)*($NY_MATCOEFFS)+(n)
    #define INDEX2D_MATDISP(m, n) (m)*($NY_MATDISPCOEFFS)+(n)
    #define INDEX3D_FIELDS(i, j, k) (i)*($NY_FIELDS)*($NZ_FIELDS)+(j)*($NZ_FIELDS)+(k)
    #define INDEX4D_ID(p, i, j, k) (p)*($NX_ID)*($NY_ID)*($NZ_ID)+(i)*($NY_ID)*($NZ_ID)+(j)*($NZ_ID)+(k)
    #define INDEX4D_T(p, i, j, k) (p)*($NX_T)*($NY_T)*($NZ_T)+(i)*($NY_T)*($NZ_T)+(j)*($NZ_T)+(k)
    #define IDX2D_MAT(m, n) (m)*($NY_MATCOEFFS)+(n)
    #define IDX2D_MATDISP(m, n) (m)*($NY_MATDISPCOEFFS)+(n)
    #define IDX3D_FIELDS(i, j, k) (i)*($NY_FIELDS)*($NZ_FIELDS)+(j)*($NZ_FIELDS)+(k)
    #define IDX4D_ID(p, i, j, k) (p)*($NX_ID)*($NY_ID)*($NZ_ID)+(i)*($NY_ID)*($NZ_ID)+(j)*($NZ_ID)+(k)
    #define IDX4D_T(p, i, j, k) (p)*($NX_T)*($NY_T)*($NZ_T)+(i)*($NY_T)*($NZ_T)+(j)*($NZ_T)+(k)
    #include <hip/hip_complex.h>

    extern "C" __global__ void update_electric_dispersive_B(int NX,
                                    int NY,
                                    int NZ,
                                    int MAXPOLES,
                                    const $COMPLEX* __restrict__ updatecoeffsdispersive,
                                    $COMPLEX *Tx,
                                    $COMPLEX *Ty,
                                    $COMPLEX *Tz,
                                    const unsigned int* __restrict__ ID,
                                    const $REAL* __restrict__ Ex,
                                    const $REAL* __restrict__ Ey,
                                    const $REAL* __restrict__ Ez,
                                    const $REAL* __restrict__ updatecoeffsE,
                                    const $REAL* __restrict__ updatecoeffsH){
    // Electric field updates - dispersive materials - part B of updates to electric
    //                              field values when dispersive materials
    //                              (with multiple poles) are present.
    //
    //  Args:
    //      NX, NY, NZ: Number of cells of the model domain.
    //      MAXPOLES: Maximum number of dispersive material poles present in model.
    //      updatedispersivecoeffs, T, ID, E, H: Access to update coefficients,
    //                                              dispersive, ID and field
    //                                              component arrays.

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for 3D field arrays
    int x = i / ($NY_FIELDS * $NZ_FIELDS);
    int y = (i % ($NY_FIELDS * $NZ_FIELDS)) / $NZ_FIELDS;
    int z = (i % ($NY_FIELDS * $NZ_FIELDS)) % $NZ_FIELDS;

    // Convert the linear index to subscripts for 4D material ID array
    int x_ID = (i % ($NX_ID * $NY_ID * $NZ_ID)) / ($NY_ID * $NZ_ID);
    int y_ID = ((i % ($NX_ID * $NY_ID * $NZ_ID)) % ($NY_ID * $NZ_ID)) / $NZ_ID;
    int z_ID = ((i % ($NX_ID * $NY_ID * $NZ_ID)) % ($NY_ID * $NZ_ID)) % $NZ_ID;

    // Convert the linear index to subscripts for 4D dispersive array
    int x_T = (i % ($NX_T * $NY_T * $NZ_T)) / ($NY_T * $NZ_T);
    int y_T = ((i % ($NX_T * $NY_T * $NZ_T)) % ($NY_T * $NZ_T)) / $NZ_T;
    int z_T = ((i % ($NX_T * $NY_T * $NZ_T)) % ($NY_T * $NZ_T)) % $NZ_T;


    // Ex component
    if ((NY != 1 || NZ != 1) && x >= 0 && x < NX && y > 0 && y < NY && z > 0 && z < NZ) {
        int materialEx = ID[IDX4D_ID(0,x_ID,y_ID,z_ID)];
        for (int pole = 0; pole < MAXPOLES; pole++) {
            Tx[IDX4D_T(pole,x_T,y_T,z_T)] = Tx[IDX4D_T(pole,x_T,y_T,z_T)] -
                                                updatecoeffsdispersive[IDX2D_MATDISP(materialEx,2+(pole*3))] * Ex[IDX3D_FIELDS(x,y,z)];
        }
    }

    // Ey component
    if ((NX != 1 || NZ != 1) && x > 0 && x < NX && y >= 0 && y < NY && z > 0 && z < NZ) {
        int materialEy = ID[IDX4D_ID(1,x_ID,y_ID,z_ID)];
        for (int pole = 0; pole < MAXPOLES; pole++) {
            Ty[IDX4D_T(pole,x_T,y_T,z_T)] = Ty[IDX4D_T(pole,x_T,y_T,z_T)] -
                                                updatecoeffsdispersive[IDX2D_MATDISP(materialEy,2+(pole*3))] * Ey[IDX3D_FIELDS(x,y,z)];
        }
    }

    // Ez component
    if ((NX != 1 || NY != 1) && x > 0 && x < NX && y > 0 && y < NY && z >= 0 && z < NZ) {
        int materialEz = ID[IDX4D_ID(2,x_ID,y_ID,z_ID)];
        for (int pole = 0; pole < MAXPOLES; pole++) {
            Tz[IDX4D_T(pole,x_T,y_T,z_T)] = Tz[IDX4D_T(pole,x_T,y_T,z_T)] -
                                                updatecoeffsdispersive[IDX2D_MATDISP(materialEz,2+(pole*3))] * Ez[IDX3D_FIELDS(x,y,z)];
        }
    }
}
""")