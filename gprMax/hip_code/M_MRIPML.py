from string import Template

x_args = {
    "hip": Template(
        """
                    __global__ void $FUNC(int xs,
                                          int xf,
                                          int ys,
                                          int yf,
                                          int zs,
                                          int zf,
                                          int NX_PHI1,
                                          int NY_PHI1,
                                          int NZ_PHI1,
                                          int NX_PHI2,
                                          int NY_PHI2,
                                          int NZ_PHI2,
                                          int NY_R,
                                          const unsigned int* __restrict__ ID,
                                          const $REAL* __restrict__ Ex,
                                          const $REAL* __restrict__ Ey,
                                          const $REAL* __restrict__ Ez,
                                          const $REAL* __restrict__ Hx,
                                          $REAL *Hy,
                                          $REAL *Hz,
                                          $REAL *PHI1,
                                          $REAL *PHI2,
                                          const $REAL* __restrict__ RA,
                                          const $REAL* __restrict__ RB,
                                          const $REAL* __restrict__ RE,
                                          const $REAL* __restrict__ RF,
                                          $REAL d,
                                          const $REAL* __restrict__ updatecoeffsE,
                                          const $REAL* __restrict__ updatecoeffsH)
                    """
    )
}

y_args = {
    "hip": Template(
        """
                    __global__ void $FUNC(int xs,
                                          int xf,
                                          int ys,
                                          int yf,
                                          int zs,
                                          int zf,
                                          int NX_PHI1,
                                          int NY_PHI1,
                                          int NZ_PHI1,
                                          int NX_PHI2,
                                          int NY_PHI2,
                                          int NZ_PHI2,
                                          int NY_R,
                                          const unsigned int* __restrict__ ID,
                                          const $REAL* __restrict__ Ex,
                                          const $REAL* __restrict__ Ey,
                                          const $REAL* __restrict__ Ez,
                                          $REAL *Hx,
                                          const $REAL* __restrict__ Hy,
                                          $REAL *Hz,
                                          $REAL *PHI1,
                                          $REAL *PHI2,
                                          const $REAL* __restrict__ RA,
                                          const $REAL* __restrict__ RB,
                                          const $REAL* __restrict__ RE,
                                          const $REAL* __restrict__ RF,
                                          $REAL d,
                                          const $REAL* __restrict__ updatecoeffsE,
                                          const $REAL* __restrict__ updatecoeffsH)
                    """
    )
}

z_args = {
    "hip": Template(
        """
                    __global__ void $FUNC(int xs,
                                          int xf,
                                          int ys,
                                          int yf,
                                          int zs,
                                          int zf,
                                          int NX_PHI1,
                                          int NY_PHI1,
                                          int NZ_PHI1,
                                          int NX_PHI2,
                                          int NY_PHI2,
                                          int NZ_PHI2,
                                          int NY_R,
                                          const unsigned int* __restrict__ ID,
                                          const $REAL* __restrict__ Ex,
                                          const $REAL* __restrict__ Ey,
                                          const $REAL* __restrict__ Ez,
                                          $REAL *Hx,
                                          $REAL *Hy,
                                          const $REAL* __restrict__ Hz,
                                          $REAL *PHI1,
                                          $REAL *PHI2,
                                          const $REAL* __restrict__ RA,
                                          const $REAL* __restrict__ RB,
                                          const $REAL* __restrict__ RE,
                                          const $REAL* __restrict__ RF,
                                          $REAL d,
                                          const $REAL* __restrict__ updatecoeffsE,
                                          const $REAL* __restrict__ updatecoeffsH)
                    """
    )
}

order1_xminus = {
    "args_hip": x_args["hip"],

    "func": Template(
        """
    //  This function updates the Hy and Hz field components for the xminus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_PHI, NY_PHI, NZ_PHI, NY_R: Dimensions of PHI1, PHI2, and R PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      Phi, RA, RB, RE, RF: Access to PML magnetic coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    int p1 = i / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    int i1 = (i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    int j1 = ((i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    int k1 = ((i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    int p2 = i / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    int i2 = (i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    int j2 = ((i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    int k2 = ((i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

    $REAL IRA, IRA1, RB0, RC0, RE0, RF0, dEy, dEz;
    $REAL dx = d;
    int ii, jj, kk, materialHy, materialHz;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 == 0 && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = xf - (i1 + 1);
        jj = j1 + ys;
        kk = k1 + zs;

        // PML coefficients
        IRA = 1 / RA[IDX2D_R(0,i1)];
        IRA1 = IRA - 1;
        RB0 = RB[IDX2D_R(0,i1)];
        RE0 = RE[IDX2D_R(0,i1)];
        RF0 = RF[IDX2D_R(0,i1)];
        RC0 = IRA * RB0 * RF0;

        // Hy
        materialHy = ID[IDX4D_ID(4,ii,jj,kk)];
        dEz = (Ez[IDX3D_FIELDS(ii+1,jj,kk)] - Ez[IDX3D_FIELDS(ii,jj,kk)]) / dx;
        Hy[IDX3D_FIELDS(ii,jj,kk)] = Hy[IDX3D_FIELDS(ii,jj,kk)] + updatecoeffsH[IDX2D_MAT(materialHy,4)] *
                                     (IRA1 * dEz - IRA * PHI1[IDX4D_PHI1(0,i1,j1,k1)]);
        PHI1[IDX4D_PHI1(0,i1,j1,k1)] = RE0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] + RC0 * dEz -
                                       RC0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)];
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = xf - (i2 + 1);
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        IRA = 1 / RA[IDX2D_R(0,i2)];
        IRA1 = IRA - 1;
        RB0 = RB[IDX2D_R(0,i2)];
        RE0 = RE[IDX2D_R(0,i2)];
        RF0 = RF[IDX2D_R(0,i2)];
        RC0 = IRA * RB0 * RF0;

        // Hz
        materialHz = ID[IDX4D_ID(5,ii,jj,kk)];
        dEy = (Ey[IDX3D_FIELDS(ii+1,jj,kk)] - Ey[IDX3D_FIELDS(ii,jj,kk)]) / dx;
        Hz[IDX3D_FIELDS(ii,jj,kk)] = Hz[IDX3D_FIELDS(ii,jj,kk)] - updatecoeffsH[IDX2D_MAT(materialHz,4)] *
                                     (IRA1 * dEy - IRA * PHI2[IDX4D_PHI2(0,i2,j2,k2)]);
        PHI2[IDX4D_PHI2(0,i2,j2,k2)] = RE0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] + RC0 * dEy -
                                       RC0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)];
    }
"""
    ),
}

order2_xminus = {
    "args_hip": x_args["hip"],

    "func": Template(
        """
    //  This function updates the Hy and Hz field components for the xminus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_PHI, NY_PHI, NZ_PHI, NY_R: Dimensions of PHI1, PHI2, and R PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      Phi, RA, RB, RE, RF: Access to PML magnetic coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    int p1 = i / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    int i1 = (i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    int j1 = ((i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    int k1 = ((i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    int p2 = i / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    int i2 = (i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    int j2 = ((i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    int k2 = ((i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

    $REAL IRA, IRA1, RB0, RC0, RE0, RF0, RB1, RC1, RE1, RF1, Psi1, Psi2, dEy, dEz;
    $REAL dx = d;
    int ii, jj, kk, materialHy, materialHz;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 == 0 && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = xf - (i1 + 1);
        jj = j1 + ys;
        kk = k1 + zs;

        // PML coefficients
        IRA = 1 / (RA[IDX2D_R(0,i1)] + RA[IDX2D_R(1,i1)]);
        IRA1 = IRA - 1;
        RB0 = RB[IDX2D_R(0,i1)];
        RE0 = RE[IDX2D_R(0,i1)];
        RF0 = RF[IDX2D_R(0,i1)];
        RC0 = IRA * RF0;
        RB1 = RB[IDX2D_R(1,i1)];
        RE1 = RE[IDX2D_R(1,i1)];
        RF1 = RF[IDX2D_R(1,i1)];
        RC1 = IRA * RF1;

        // Hy
        Psi1 = RB0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] + RB1 * PHI1[IDX4D_PHI1(1,i1,j1,k1)];
        materialHy = ID[IDX4D_ID(4,ii,jj,kk)];
        dEz = (Ez[IDX3D_FIELDS(ii+1,jj,kk)] - Ez[IDX3D_FIELDS(ii,jj,kk)]) / dx;
        Hy[IDX3D_FIELDS(ii,jj,kk)] = Hy[IDX3D_FIELDS(ii,jj,kk)] + updatecoeffsH[IDX2D_MAT(materialHy,4)] *
                                     (IRA1 * dEz - IRA * Psi1);
        PHI1[IDX4D_PHI1(1,i1,j1,k1)] = RE1 * PHI1[IDX4D_PHI1(1,i1,j1,k1)] + RC1 * (dEz - Psi1);
        PHI1[IDX4D_PHI1(0,i1,j1,k1)] = RE0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] + RC0 * (dEz - Psi1);
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = xf - (i2 + 1);
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        IRA = 1 / (RA[IDX2D_R(0,i2)] + RA[IDX2D_R(1,i2)]);
        IRA1 = IRA - 1;
        RB0 = RB[IDX2D_R(0,i2)];
        RE0 = RE[IDX2D_R(0,i2)];
        RF0 = RF[IDX2D_R(0,i2)];
        RC0 = IRA * RF0;
        RB1 = RB[IDX2D_R(1,i2)];
        RE1 = RE[IDX2D_R(1,i2)];
        RF1 = RF[IDX2D_R(1,i2)];
        RC1 = IRA * RF1;

        // Hz
        Psi2 = RB0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] + RB1 * PHI2[IDX4D_PHI2(1,i2,j2,k2)];
        materialHz = ID[IDX4D_ID(5,ii,jj,kk)];
        dEy = (Ey[IDX3D_FIELDS(ii+1,jj,kk)] - Ey[IDX3D_FIELDS(ii,jj,kk)]) / dx;
        Hz[IDX3D_FIELDS(ii,jj,kk)] = Hz[IDX3D_FIELDS(ii,jj,kk)] - updatecoeffsH[IDX2D_MAT(materialHz,4)] *
                                     (IRA1 * dEy - IRA * Psi2);
        PHI2[IDX4D_PHI2(1,i2,j2,k2)] = RE1 * PHI2[IDX4D_PHI2(1,i2,j2,k2)] + RC1 * (dEy - Psi2);
        PHI2[IDX4D_PHI2(0,i2,j2,k2)] = RE0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] + RC0 * (dEy - Psi2);
    }
"""
    ),
}

order1_xplus = {
    "args_hip": x_args["hip"],

    "func": Template(
        """
    //  This function updates the Hy and Hz field components for the xplus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_PHI, NY_PHI, NZ_PHI, NY_R: Dimensions of PHI1, PHI2, and R PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      Phi, RA, RB, RE, RF: Access to PML magnetic coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    int p1 = i / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    int i1 = (i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    int j1 = ((i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    int k1 = ((i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    int p2 = i / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    int i2 = (i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    int j2 = ((i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    int k2 = ((i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

    $REAL IRA, IRA1, RB0, RC0, RE0, RF0, dEy, dEz;
    $REAL dx = d;
    int ii, jj, kk, materialHy, materialHz;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 == 0 && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = j1 + ys;
        kk = k1 + zs;

        // PML coefficients
        IRA = 1 / RA[IDX2D_R(0,i1)];
        IRA1 = IRA - 1;
        RB0 = RB[IDX2D_R(0,i1)];
        RE0 = RE[IDX2D_R(0,i1)];
        RF0 = RF[IDX2D_R(0,i1)];
        RC0 = IRA * RB0 * RF0;

        // Hy
        materialHy = ID[IDX4D_ID(4,ii,jj,kk)];
        dEz = (Ez[IDX3D_FIELDS(ii+1,jj,kk)] - Ez[IDX3D_FIELDS(ii,jj,kk)]) / dx;
        Hy[IDX3D_FIELDS(ii,jj,kk)] = Hy[IDX3D_FIELDS(ii,jj,kk)] + updatecoeffsH[IDX2D_MAT(materialHy,4)] *
                                     (IRA1 * dEz - IRA * PHI1[IDX4D_PHI1(0,i1,j1,k1)]);
        PHI1[IDX4D_PHI1(0,i1,j1,k1)] = RE0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] + RC0 * dEz -
                                       RC0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)];
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        IRA = 1 / RA[IDX2D_R(0,i2)];
        IRA1 = IRA - 1;
        RB0 = RB[IDX2D_R(0,i2)];
        RE0 = RE[IDX2D_R(0,i2)];
        RF0 = RF[IDX2D_R(0,i2)];
        RC0 = IRA * RB0 * RF0;

        // Hz
        materialHz = ID[IDX4D_ID(5,ii,jj,kk)];
        dEy = (Ey[IDX3D_FIELDS(ii+1,jj,kk)] - Ey[IDX3D_FIELDS(ii,jj,kk)]) / dx;
        Hz[IDX3D_FIELDS(ii,jj,kk)] = Hz[IDX3D_FIELDS(ii,jj,kk)] - updatecoeffsH[IDX2D_MAT(materialHz,4)] *
                                     (IRA1 * dEy - IRA * PHI2[IDX4D_PHI2(0,i2,j2,k2)]);
        PHI2[IDX4D_PHI2(0,i2,j2,k2)] = RE0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] + RC0 * dEy -
                                       RC0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)];
    }
"""
    ),
}

order2_xplus = {
    "args_hip": x_args["hip"],

    "func": Template(
        """
    //  This function updates the Hy and Hz field components for the xplus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_PHI, NY_PHI, NZ_PHI, NY_R: Dimensions of PHI1, PHI2, and R PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      Phi, RA, RB, RE, RF: Access to PML magnetic coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    int p1 = i / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    int i1 = (i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    int j1 = ((i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    int k1 = ((i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    int p2 = i / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    int i2 = (i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    int j2 = ((i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    int k2 = ((i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

    $REAL IRA, IRA1, RB0, RC0, RE0, RF0, RB1, RC1, RE1, RF1, Psi1, Psi2, dEy, dEz;
    $REAL dx = d;
    int ii, jj, kk, materialHy, materialHz;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 == 0 && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = j1 + ys;
        kk = k1 + zs;

        // PML coefficients
        IRA = 1 / (RA[IDX2D_R(0,i1)] + RA[IDX2D_R(1,i1)]);
        IRA1 = IRA - 1;
        RB0 = RB[IDX2D_R(0,i1)];
        RE0 = RE[IDX2D_R(0,i1)];
        RF0 = RF[IDX2D_R(0,i1)];
        RC0 = IRA * RF0;
        RB1 = RB[IDX2D_R(1,i1)];
        RE1 = RE[IDX2D_R(1,i1)];
        RF1 = RF[IDX2D_R(1,i1)];
        RC1 = IRA * RF1;

        // Hy
        Psi1 = RB0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] + RB1 * PHI1[IDX4D_PHI1(1,i1,j1,k1)];
        materialHy = ID[IDX4D_ID(4,ii,jj,kk)];
        dEz = (Ez[IDX3D_FIELDS(ii+1,jj,kk)] - Ez[IDX3D_FIELDS(ii,jj,kk)]) / dx;
        Hy[IDX3D_FIELDS(ii,jj,kk)] = Hy[IDX3D_FIELDS(ii,jj,kk)] + updatecoeffsH[IDX2D_MAT(materialHy,4)] *
                                     (IRA1 * dEz - IRA * Psi1);
        PHI1[IDX4D_PHI1(1,i1,j1,k1)] = RE1 * PHI1[IDX4D_PHI1(1,i1,j1,k1)] + RC1 * (dEz - Psi1);
        PHI1[IDX4D_PHI1(0,i1,j1,k1)] = RE0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] + RC0 * (dEz - Psi1);
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        IRA = 1 / (RA[IDX2D_R(0,i2)] + RA[IDX2D_R(1,i2)]);
        IRA1 = IRA - 1;
        RB0 = RB[IDX2D_R(0,i2)];
        RE0 = RE[IDX2D_R(0,i2)];
        RF0 = RF[IDX2D_R(0,i2)];
        RC0 = IRA * RF0;
        RB1 = RB[IDX2D_R(1,i2)];
        RE1 = RE[IDX2D_R(1,i2)];
        RF1 = RF[IDX2D_R(1,i2)];
        RC1 = IRA * RF1;

        // Hz
        Psi2 = RB0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] + RB1 * PHI2[IDX4D_PHI2(1,i2,j2,k2)];
        materialHz = ID[IDX4D_ID(5,ii,jj,kk)];
        dEy = (Ey[IDX3D_FIELDS(ii+1,jj,kk)] - Ey[IDX3D_FIELDS(ii,jj,kk)]) / dx;
        Hz[IDX3D_FIELDS(ii,jj,kk)] = Hz[IDX3D_FIELDS(ii,jj,kk)] - updatecoeffsH[IDX2D_MAT(materialHz,4)] *
                                     (IRA1 * dEy - IRA * Psi2);
        PHI2[IDX4D_PHI2(1,i2,j2,k2)] = RE1 * PHI2[IDX4D_PHI2(1,i2,j2,k2)] + RC1 * (dEy - Psi2);
        PHI2[IDX4D_PHI2(0,i2,j2,k2)] = RE0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] + RC0 * (dEy - Psi2);
    }
"""
    ),
}

order1_yminus = {
    "args_hip": y_args["hip"],

    "func": Template(
        """
    //  This function updates the Hx and Hz field components for the yminus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_PHI, NY_PHI, NZ_PHI, NY_R: Dimensions of PHI1, PHI2, and R PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      Phi, RA, RB, RE, RF: Access to PML magnetic coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    int p1 = i / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    int i1 = (i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    int j1 = ((i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    int k1 = ((i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    int p2 = i / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    int i2 = (i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    int j2 = ((i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    int k2 = ((i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

    $REAL IRA, IRA1, RB0, RC0, RE0, RF0, dEx, dEz;
    $REAL dy = d;
    int ii, jj, kk, materialHx, materialHz;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 == 0 && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = yf - (j1 + 1);
        kk = k1 + zs;

        // PML coefficients
        IRA = 1 / RA[IDX2D_R(0,j1)];
        IRA1 = IRA - 1;
        RB0 = RB[IDX2D_R(0,j1)];
        RE0 = RE[IDX2D_R(0,j1)];
        RF0 = RF[IDX2D_R(0,j1)];
        RC0 = IRA * RB0 * RF0;

        // Hx
        materialHx = ID[IDX4D_ID(3,ii,jj,kk)];
        dEz = (Ez[IDX3D_FIELDS(ii,jj+1,kk)] - Ez[IDX3D_FIELDS(ii,jj,kk)]) / dy;
        Hx[IDX3D_FIELDS(ii,jj,kk)] = Hx[IDX3D_FIELDS(ii,jj,kk)] - updatecoeffsH[IDX2D_MAT(materialHx,4)] *
                                     (IRA1 * dEz - IRA * PHI1[IDX4D_PHI1(0,i1,j1,k1)]);
        PHI1[IDX4D_PHI1(0,i1,j1,k1)] = RE0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] + RC0 * dEz -
                                       RC0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)];
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = yf - (j2 + 1);
        kk = k2 + zs;

        // PML coefficients
        IRA = 1 / RA[IDX2D_R(0,j2)];
        IRA1 = IRA - 1;
        RB0 = RB[IDX2D_R(0,j2)];
        RE0 = RE[IDX2D_R(0,j2)];
        RF0 = RF[IDX2D_R(0,j2)];
        RC0 = IRA * RB0 * RF0;

        // Hz
        materialHz = ID[IDX4D_ID(5,ii,jj,kk)];
        dEx = (Ex[IDX3D_FIELDS(ii,jj+1,kk)] - Ex[IDX3D_FIELDS(ii,jj,kk)]) / dy;
        Hz[IDX3D_FIELDS(ii,jj,kk)] = Hz[IDX3D_FIELDS(ii,jj,kk)] + updatecoeffsH[IDX2D_MAT(materialHz,4)] *
                                     (IRA1 * dEx - IRA * PHI2[IDX4D_PHI2(0,i2,j2,k2)]);
        PHI2[IDX4D_PHI2(0,i2,j2,k2)] = RE0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] + RC0 * dEx -
                                       RC0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)];
    }
"""
    ),
}

order2_yminus = {
    "args_hip": y_args["hip"],

    "func": Template(
        """
    //  This function updates the Hx and Hz field components for the yminus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_PHI, NY_PHI, NZ_PHI, NY_R: Dimensions of PHI1, PHI2, and R PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      Phi, RA, RB, RE, RF: Access to PML magnetic coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    int p1 = i / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    int i1 = (i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    int j1 = ((i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    int k1 = ((i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    int p2 = i / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    int i2 = (i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    int j2 = ((i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    int k2 = ((i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

    $REAL IRA, IRA1, RB0, RC0, RE0, RF0, RB1, RC1, RE1, RF1, Psi1, Psi2, dEx, dEz;
    $REAL dy = d;
    int ii, jj, kk, materialHx, materialHz;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 == 0 && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = yf - (j1 + 1);
        kk = k1 + zs;

        // PML coefficients
        IRA = 1 / (RA[IDX2D_R(0,j1)] + RA[IDX2D_R(1,j1)]);
        IRA1 = IRA - 1;
        RB0 = RB[IDX2D_R(0,j1)];
        RE0 = RE[IDX2D_R(0,j1)];
        RF0 = RF[IDX2D_R(0,j1)];
        RC0 = IRA * RF0;
        RB1 = RB[IDX2D_R(1,j1)];
        RE1 = RE[IDX2D_R(1,j1)];
        RF1 = RF[IDX2D_R(1,j1)];
        RC1 = IRA * RF1;

        // Hx
        Psi1 = RB0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] + RB1 * PHI1[IDX4D_PHI1(1,i1,j1,k1)];
        materialHx = ID[IDX4D_ID(3,ii,jj,kk)];
        dEz = (Ez[IDX3D_FIELDS(ii,jj+1,kk)] - Ez[IDX3D_FIELDS(ii,jj,kk)]) / dy;
        Hx[IDX3D_FIELDS(ii,jj,kk)] = Hx[IDX3D_FIELDS(ii,jj,kk)] - updatecoeffsH[IDX2D_MAT(materialHx,4)] *
                                     (IRA1 * dEz - IRA * Psi1);
        PHI1[IDX4D_PHI1(1,i1,j1,k1)] = RE1 * PHI1[IDX4D_PHI1(1,i1,j1,k1)] + RC1 * (dEz - Psi1);
        PHI1[IDX4D_PHI1(0,i1,j1,k1)] = RE0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] + RC0 * (dEz - Psi1);
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = yf - (j2 + 1);
        kk = k2 + zs;

        // PML coefficients
        IRA = 1 / (RA[IDX2D_R(0,j2)] + RA[IDX2D_R(1,j2)]);
        IRA1 = IRA - 1;
        RB0 = RB[IDX2D_R(0,j2)];
        RE0 = RE[IDX2D_R(0,j2)];
        RF0 = RF[IDX2D_R(0,j2)];
        RC0 = IRA * RF0;
        RB1 = RB[IDX2D_R(1,j2)];
        RE1 = RE[IDX2D_R(1,j2)];
        RF1 = RF[IDX2D_R(1,j2)];
        RC1 = IRA * RF1;

        // Hz
        Psi2 = RB0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] + RB1 * PHI2[IDX4D_PHI2(1,i2,j2,k2)];
        materialHz = ID[IDX4D_ID(5,ii,jj,kk)];
        dEx = (Ex[IDX3D_FIELDS(ii,jj+1,kk)] - Ex[IDX3D_FIELDS(ii,jj,kk)]) / dy;
        Hz[IDX3D_FIELDS(ii,jj,kk)] = Hz[IDX3D_FIELDS(ii,jj,kk)] + updatecoeffsH[IDX2D_MAT(materialHz,4)] *
                                     (IRA1 * dEx - IRA * Psi2);
        PHI2[IDX4D_PHI2(1,i2,j2,k2)] = RE1 * PHI2[IDX4D_PHI2(1,i2,j2,k2)] + RC1 * (dEx - Psi2);
        PHI2[IDX4D_PHI2(0,i2,j2,k2)] = RE0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] + RC0 * (dEx - Psi2);
    }
"""
    ),
}

order1_yplus = {
    "args_hip": y_args["hip"],

    "func": Template(
        """
    //  This function updates the Hx and Hz field components for the yplus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_PHI, NY_PHI, NZ_PHI, NY_R: Dimensions of PHI1, PHI2, and R PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      Phi, RA, RB, RE, RF: Access to PML magnetic coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    int p1 = i / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    int i1 = (i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    int j1 = ((i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    int k1 = ((i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    int p2 = i / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    int i2 = (i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    int j2 = ((i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    int k2 = ((i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

    $REAL IRA, IRA1, RB0, RC0, RE0, RF0, dEx, dEz;
    $REAL dy = d;
    int ii, jj, kk, materialHx, materialHz;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 == 0 && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = j1 + ys;
        kk = k1 + zs;

        // PML coefficients
        IRA = 1 / RA[IDX2D_R(0,j1)];
        IRA1 = IRA - 1;
        RB0 = RB[IDX2D_R(0,j1)];
        RE0 = RE[IDX2D_R(0,j1)];
        RF0 = RF[IDX2D_R(0,j1)];
        RC0 = IRA * RB0 * RF0;

        // Hx
        materialHx = ID[IDX4D_ID(3,ii,jj,kk)];
        dEz = (Ez[IDX3D_FIELDS(ii,jj+1,kk)] - Ez[IDX3D_FIELDS(ii,jj,kk)]) / dy;
        Hx[IDX3D_FIELDS(ii,jj,kk)] = Hx[IDX3D_FIELDS(ii,jj,kk)] - updatecoeffsH[IDX2D_MAT(materialHx,4)] *
                                     (IRA1 * dEz - IRA * PHI1[IDX4D_PHI1(0,i1,j1,k1)]);
        PHI1[IDX4D_PHI1(0,i1,j1,k1)] = RE0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] + RC0 * dEz -
                                       RC0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)];
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        IRA = 1 / RA[IDX2D_R(0,j2)];
        IRA1 = IRA - 1;
        RB0 = RB[IDX2D_R(0,j2)];
        RE0 = RE[IDX2D_R(0,j2)];
        RF0 = RF[IDX2D_R(0,j2)];
        RC0 = IRA * RB0 * RF0;

        // Hz
        materialHz = ID[IDX4D_ID(5,ii,jj,kk)];
        dEx = (Ex[IDX3D_FIELDS(ii,jj+1,kk)] - Ex[IDX3D_FIELDS(ii,jj,kk)]) / dy;
        Hz[IDX3D_FIELDS(ii,jj,kk)] = Hz[IDX3D_FIELDS(ii,jj,kk)] + updatecoeffsH[IDX2D_MAT(materialHz,4)] *
                                     (IRA1 * dEx - IRA * PHI2[IDX4D_PHI2(0,i2,j2,k2)]);
        PHI2[IDX4D_PHI2(0,i2,j2,k2)] = RE0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] + RC0 * dEx -
                                       RC0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)];
    }
"""
    ),
}

order2_yplus = {
    "args_hip": y_args["hip"],

    "func": Template(
        """
    //  This function updates the Hx and Hz field components for the yplus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_PHI, NY_PHI, NZ_PHI, NY_R: Dimensions of PHI1, PHI2, and R PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      Phi, RA, RB, RE, RF: Access to PML magnetic coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    int p1 = i / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    int i1 = (i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    int j1 = ((i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    int k1 = ((i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    int p2 = i / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    int i2 = (i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    int j2 = ((i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    int k2 = ((i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

    $REAL IRA, IRA1, RB0, RC0, RE0, RF0, RB1, RC1, RE1, RF1, Psi1, Psi2, dEx, dEz;
    $REAL dy = d;
    int ii, jj, kk, materialHx, materialHz;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 == 0 && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = j1 + ys;
        kk = k1 + zs;

        // PML coefficients
        IRA = 1 / (RA[IDX2D_R(0,j1)] + RA[IDX2D_R(1,j1)]);
        IRA1 = IRA - 1;
        RB0 = RB[IDX2D_R(0,j1)];
        RE0 = RE[IDX2D_R(0,j1)];
        RF0 = RF[IDX2D_R(0,j1)];
        RC0 = IRA * RF0;
        RB1 = RB[IDX2D_R(1,j1)];
        RE1 = RE[IDX2D_R(1,j1)];
        RF1 = RF[IDX2D_R(1,j1)];
        RC1 = IRA * RF1;

        // Hx
        Psi1 = RB0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] + RB1 * PHI1[IDX4D_PHI1(1,i1,j1,k1)];
        materialHx = ID[IDX4D_ID(3,ii,jj,kk)];
        dEz = (Ez[IDX3D_FIELDS(ii,jj+1,kk)] - Ez[IDX3D_FIELDS(ii,jj,kk)]) / dy;
        Hx[IDX3D_FIELDS(ii,jj,kk)] = Hx[IDX3D_FIELDS(ii,jj,kk)] - updatecoeffsH[IDX2D_MAT(materialHx,4)] *
                                     (IRA1 * dEz - IRA * Psi1);
        PHI1[IDX4D_PHI1(1,i1,j1,k1)] = RE1 * PHI1[IDX4D_PHI1(1,i1,j1,k1)] + RC1 * (dEz - Psi1);
        PHI1[IDX4D_PHI1(0,i1,j1,k1)] = RE0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] + RC0 * (dEz - Psi1);
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        IRA = 1 / (RA[IDX2D_R(0,j2)] + RA[IDX2D_R(1,j2)]);
        IRA1 = IRA - 1;
        RB0 = RB[IDX2D_R(0,j2)];
        RE0 = RE[IDX2D_R(0,j2)];
        RF0 = RF[IDX2D_R(0,j2)];
        RC0 = IRA * RF0;
        RB1 = RB[IDX2D_R(1,j2)];
        RE1 = RE[IDX2D_R(1,j2)];
        RF1 = RF[IDX2D_R(1,j2)];
        RC1 = IRA * RF1;

        // Hz
        Psi2 = RB0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] + RB1 * PHI2[IDX4D_PHI2(1,i2,j2,k2)];
        materialHz = ID[IDX4D_ID(5,ii,jj,kk)];
        dEx = (Ex[IDX3D_FIELDS(ii,jj+1,kk)] - Ex[IDX3D_FIELDS(ii,jj,kk)]) / dy;
        Hz[IDX3D_FIELDS(ii,jj,kk)] = Hz[IDX3D_FIELDS(ii,jj,kk)] + updatecoeffsH[IDX2D_MAT(materialHz,4)] *
                                     (IRA1 * dEx - IRA * Psi2);
        PHI2[IDX4D_PHI2(1,i2,j2,k2)] = RE1 * PHI2[IDX4D_PHI2(1,i2,j2,k2)] + RC1 * (dEx - Psi2);
        PHI2[IDX4D_PHI2(0,i2,j2,k2)] = RE0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] + RC0 * (dEx - Psi2);
    }
"""
    ),
}

order1_zminus = {
    "args_hip": z_args["hip"],

    "func": Template(
        """
    //  This function updates the Hx and Hy field components for the zminus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_PHI, NY_PHI, NZ_PHI, NY_R: Dimensions of PHI1, PHI2, and R PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      Phi, RA, RB, RE, RF: Access to PML magnetic coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    int p1 = i / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    int i1 = (i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    int j1 = ((i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    int k1 = ((i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    int p2 = i / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    int i2 = (i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    int j2 = ((i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    int k2 = ((i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

    $REAL IRA, IRA1, RB0, RC0, RE0, RF0, dEx, dEy;
    $REAL dz = d;
    int ii, jj, kk, materialHx, materialHy;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 == 0 && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = j1 + ys;
        kk = zf - (k1 + 1);

        // PML coefficients
        IRA = 1 / RA[IDX2D_R(0,k1)];
        IRA1 = IRA - 1;
        RB0 = RB[IDX2D_R(0,k1)];
        RE0 = RE[IDX2D_R(0,k1)];
        RF0 = RF[IDX2D_R(0,k1)];
        RC0 = IRA * RB0 * RF0;

        // Hx
        materialHx = ID[IDX4D_ID(3,ii,jj,kk)];
        dEy = (Ey[IDX3D_FIELDS(ii,jj,kk+1)] - Ey[IDX3D_FIELDS(ii,jj,kk)]) / dz;
        Hx[IDX3D_FIELDS(ii,jj,kk)] = Hx[IDX3D_FIELDS(ii,jj,kk)] + updatecoeffsH[IDX2D_MAT(materialHx,4)] *
                                     (IRA1 * dEy - IRA * PHI1[IDX4D_PHI1(0,i1,j1,k1)]);
        PHI1[IDX4D_PHI1(0,i1,j1,k1)] = RE0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] + RC0 * dEy -
                                       RC0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)];
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = zf - (k2 + 1);

        // PML coefficients
        IRA = 1 / RA[IDX2D_R(0,k2)];
        IRA1 = IRA - 1;
        RB0 = RB[IDX2D_R(0,k2)];
        RE0 = RE[IDX2D_R(0,k2)];
        RF0 = RF[IDX2D_R(0,k2)];
        RC0 = IRA * RB0 * RF0;

        // Hy
        materialHy = ID[IDX4D_ID(4,ii,jj,kk)];
        dEx = (Ex[IDX3D_FIELDS(ii,jj,kk+1)] - Ex[IDX3D_FIELDS(ii,jj,kk)]) / dz;
        Hy[IDX3D_FIELDS(ii,jj,kk)] = Hy[IDX3D_FIELDS(ii,jj,kk)] - updatecoeffsH[IDX2D_MAT(materialHy,4)] *
                                     (IRA1 * dEx - IRA * PHI2[IDX4D_PHI2(0,i2,j2,k2)]);
        PHI2[IDX4D_PHI2(0,i2,j2,k2)] = RE0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] + RC0 * dEx -
                                       RC0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)];
    }
"""
    ),
}

order2_zminus = {
    "args_hip": z_args["hip"],

    "func": Template(
        """
    //  This function updates the Hx and Hy field components for the zminus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_PHI, NY_PHI, NZ_PHI, NY_R: Dimensions of PHI1, PHI2, and R PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      Phi, RA, RB, RE, RF: Access to PML magnetic coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    int p1 = i / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    int i1 = (i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    int j1 = ((i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    int k1 = ((i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    int p2 = i / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    int i2 = (i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    int j2 = ((i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    int k2 = ((i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

    $REAL IRA, IRA1, RB0, RC0, RE0, RF0, RB1, RC1, RE1, RF1, Psi1, Psi2, dEx, dEy;
    $REAL dz = d;
    int ii, jj, kk, materialHx, materialHy;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 == 0 && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = j1 + ys;
        kk = zf - (k1 + 1);

        // PML coefficients
        IRA = 1 / (RA[IDX2D_R(0,k1)] + RA[IDX2D_R(1,k1)]);
        IRA1 = IRA - 1;
        RB0 = RB[IDX2D_R(0,k1)];
        RE0 = RE[IDX2D_R(0,k1)];
        RF0 = RF[IDX2D_R(0,k1)];
        RC0 = IRA * RF0;
        RB1 = RB[IDX2D_R(1,k1)];
        RE1 = RE[IDX2D_R(1,k1)];
        RF1 = RF[IDX2D_R(1,k1)];
        RC1 = IRA * RF1;

        // Hx
        Psi1 = RB0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] + RB1 * PHI1[IDX4D_PHI1(1,i1,j1,k1)];
        materialHx = ID[IDX4D_ID(3,ii,jj,kk)];
        dEy = (Ey[IDX3D_FIELDS(ii,jj,kk+1)] - Ey[IDX3D_FIELDS(ii,jj,kk)]) / dz;
        Hx[IDX3D_FIELDS(ii,jj,kk)] = Hx[IDX3D_FIELDS(ii,jj,kk)] + updatecoeffsH[IDX2D_MAT(materialHx,4)] *
                                     (IRA1 * dEy - IRA * Psi1);
        PHI1[IDX4D_PHI1(1,i1,j1,k1)] = RE1 * PHI1[IDX4D_PHI1(1,i1,j1,k1)] + RC1 * (dEy - Psi1);
        PHI1[IDX4D_PHI1(0,i1,j1,k1)] = RE0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] + RC0 * (dEy - Psi1);
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = zf - (k2 + 1);

        // PML coefficients
        IRA = 1 / (RA[IDX2D_R(0,k2)] + RA[IDX2D_R(1,k2)]);
        IRA1 = IRA - 1;
        RB0 = RB[IDX2D_R(0,k2)];
        RE0 = RE[IDX2D_R(0,k2)];
        RF0 = RF[IDX2D_R(0,k2)];
        RC0 = IRA * RF0;
        RB1 = RB[IDX2D_R(1,k2)];
        RE1 = RE[IDX2D_R(1,k2)];
        RF1 = RF[IDX2D_R(1,k2)];
        RC1 = IRA * RF1;

        // Hy
        Psi2 = RB0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] + RB1 * PHI2[IDX4D_PHI2(1,i2,j2,k2)];
        materialHy = ID[IDX4D_ID(4,ii,jj,kk)];
        dEx = (Ex[IDX3D_FIELDS(ii,jj,kk+1)] - Ex[IDX3D_FIELDS(ii,jj,kk)]) / dz;
        Hy[IDX3D_FIELDS(ii,jj,kk)] = Hy[IDX3D_FIELDS(ii,jj,kk)] - updatecoeffsH[IDX2D_MAT(materialHy,4)] *
                                     (IRA1 * dEx - IRA * Psi2);
        PHI2[IDX4D_PHI2(1,i2,j2,k2)] = RE1 * PHI2[IDX4D_PHI2(1,i2,j2,k2)] + RC1 * (dEx - Psi2);
        PHI2[IDX4D_PHI2(0,i2,j2,k2)] = RE0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] + RC0 * (dEx - Psi2);
    }
"""
    ),
}

order1_zplus = {
    "args_hip": z_args["hip"],

    "func": Template(
        """
    //  This function updates the Hx and Hy field components for the zplus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_PHI, NY_PHI, NZ_PHI, NY_R: Dimensions of PHI1, PHI2, and R PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      Phi, RA, RB, RE, RF: Access to PML magnetic coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    int p1 = i / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    int i1 = (i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    int j1 = ((i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    int k1 = ((i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    int p2 = i / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    int i2 = (i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    int j2 = ((i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    int k2 = ((i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

    $REAL IRA, IRA1, RB0, RC0, RE0, RF0, dEx, dEy;
    $REAL dz = d;
    int ii, jj, kk, materialHx, materialHy;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 == 0 && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = j1 + ys;
        kk = k1 + zs;

        // PML coefficients
        IRA = 1 / RA[IDX2D_R(0,k1)];
        IRA1 = IRA - 1;
        RB0 = RB[IDX2D_R(0,k1)];
        RE0 = RE[IDX2D_R(0,k1)];
        RF0 = RF[IDX2D_R(0,k1)];
        RC0 = IRA * RB0 * RF0;

        // Hx
        materialHx = ID[IDX4D_ID(3,ii,jj,kk)];
        dEy = (Ey[IDX3D_FIELDS(ii,jj,kk+1)] - Ey[IDX3D_FIELDS(ii,jj,kk)]) / dz;
        Hx[IDX3D_FIELDS(ii,jj,kk)] = Hx[IDX3D_FIELDS(ii,jj,kk)] + updatecoeffsH[IDX2D_MAT(materialHx,4)] *
                                     (IRA1 * dEy - IRA * PHI1[IDX4D_PHI1(0,i1,j1,k1)]);
        PHI1[IDX4D_PHI1(0,i1,j1,k1)] = RE0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] + RC0 * dEy -
                                       RC0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)];
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        IRA = 1 / RA[IDX2D_R(0,k2)];
        IRA1 = IRA - 1;
        RB0 = RB[IDX2D_R(0,k2)];
        RE0 = RE[IDX2D_R(0,k2)];
        RF0 = RF[IDX2D_R(0,k2)];
        RC0 = IRA * RB0 * RF0;

        // Hy
        materialHy = ID[IDX4D_ID(4,ii,jj,kk)];
        dEx = (Ex[IDX3D_FIELDS(ii,jj,kk+1)] - Ex[IDX3D_FIELDS(ii,jj,kk)]) / dz;
        Hy[IDX3D_FIELDS(ii,jj,kk)] = Hy[IDX3D_FIELDS(ii,jj,kk)] - updatecoeffsH[IDX2D_MAT(materialHy,4)] *
                                     (IRA1 * dEx - IRA * PHI2[IDX4D_PHI2(0,i2,j2,k2)]);
        PHI2[IDX4D_PHI2(0,i2,j2,k2)] = RE0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] + RC0 * dEx -
                                       RC0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)];
    }
"""
    ),
}

order2_zplus = {
    "args_hip": z_args["hip"],

    "func": Template(
        """
    //  This function updates the Hx and Hy field components for the zplus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_PHI, NY_PHI, NZ_PHI, NY_R: Dimensions of PHI1, PHI2, and R PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      Phi, RA, RB, RE, RF: Access to PML magnetic coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    int p1 = i / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    int i1 = (i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    int j1 = ((i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    int k1 = ((i % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    int p2 = i / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    int i2 = (i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    int j2 = ((i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    int k2 = ((i % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

    $REAL IRA, IRA1, RB0, RC0, RE0, RF0, RB1, RC1, RE1, RF1, Psi1, Psi2, dEx, dEy;
    $REAL dz = d;
    int ii, jj, kk, materialHx, materialHy;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 == 0 && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = j1 + ys;
        kk = k1 + zs;

        // PML coefficients
        IRA = 1 / (RA[IDX2D_R(0,k1)] + RA[IDX2D_R(1,k1)]);
        IRA1 = IRA - 1;
        RB0 = RB[IDX2D_R(0,k1)];
        RE0 = RE[IDX2D_R(0,k1)];
        RF0 = RF[IDX2D_R(0,k1)];
        RC0 = IRA * RF0;
        RB1 = RB[IDX2D_R(1,k1)];
        RE1 = RE[IDX2D_R(1,k1)];
        RF1 = RF[IDX2D_R(1,k1)];
        RC1 = IRA * RF1;

        // Hx
        Psi1 = RB0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] + RB1 * PHI1[IDX4D_PHI1(1,i1,j1,k1)];
        materialHx = ID[IDX4D_ID(3,ii,jj,kk)];
        dEy = (Ey[IDX3D_FIELDS(ii,jj,kk+1)] - Ey[IDX3D_FIELDS(ii,jj,kk)]) / dz;
        Hx[IDX3D_FIELDS(ii,jj,kk)] = Hx[IDX3D_FIELDS(ii,jj,kk)] + updatecoeffsH[IDX2D_MAT(materialHx,4)] *
                                     (IRA1 * dEy - IRA * Psi1);
        PHI1[IDX4D_PHI1(1,i1,j1,k1)] = RE1 * PHI1[IDX4D_PHI1(1,i1,j1,k1)] + RC1 * (dEy - Psi1);
        PHI1[IDX4D_PHI1(0,i1,j1,k1)] = RE0 * PHI1[IDX4D_PHI1(0,i1,j1,k1)] + RC0 * (dEy - Psi1);
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        IRA = 1 / (RA[IDX2D_R(0,k2)] + RA[IDX2D_R(1,k2)]);
        IRA1 = IRA - 1;
        RB0 = RB[IDX2D_R(0,k2)];
        RE0 = RE[IDX2D_R(0,k2)];
        RF0 = RF[IDX2D_R(0,k2)];
        RC0 = IRA * RF0;
        RB1 = RB[IDX2D_R(1,k2)];
        RE1 = RE[IDX2D_R(1,k2)];
        RF1 = RF[IDX2D_R(1,k2)];
        RC1 = IRA * RF1;

        // Hy
        Psi2 = RB0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] + RB1 * PHI2[IDX4D_PHI2(1,i2,j2,k2)];
        materialHy = ID[IDX4D_ID(4,ii,jj,kk)];
        dEx = (Ex[IDX3D_FIELDS(ii,jj,kk+1)] - Ex[IDX3D_FIELDS(ii,jj,kk)]) / dz;
        Hy[IDX3D_FIELDS(ii,jj,kk)] = Hy[IDX3D_FIELDS(ii,jj,kk)] - updatecoeffsH[IDX2D_MAT(materialHy,4)] *
                                     (IRA1 * dEx - IRA * Psi2);
        PHI2[IDX4D_PHI2(1,i2,j2,k2)] = RE1 * PHI2[IDX4D_PHI2(1,i2,j2,k2)] + RC1 * (dEx - Psi2);
        PHI2[IDX4D_PHI2(0,i2,j2,k2)] = RE0 * PHI2[IDX4D_PHI2(0,i2,j2,k2)] + RC0 * (dEx - Psi2);
    }
"""
    ),
}
