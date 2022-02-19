// Macros for converting subscripts to linear index:
#define INDEX2D_R(m, n) (m)*(NY_R)+(n)
#define INDEX2D_MAT(m, n) (m)*({{NY_MATCOEFFS}})+(n)
#define INDEX3D_FIELDS(i, j, k) (i)*({{NY_FIELDS}})*({{NZ_FIELDS}})+(j)*({{NZ_FIELDS}})+(k)
#define INDEX4D_ID(p, i, j, k) (p)*({{NX_ID}})*({{NY_ID}})*({{NZ_ID}})+(i)*({{NY_ID}})*({{NZ_ID}})+(j)*({{NZ_ID}})+(k)
#define INDEX4D_PHI1(p, i, j, k) (p)*(NX_PHI1)*(NY_PHI1)*(NZ_PHI1)+(i)*(NY_PHI1)*(NZ_PHI1)+(j)*(NZ_PHI1)+(k)
#define INDEX4D_PHI2(p, i, j, k) (p)*(NX_PHI2)*(NY_PHI2)*(NZ_PHI2)+(i)*(NY_PHI2)*(NZ_PHI2)+(j)*(NZ_PHI2)+(k)

__constant {{REAL}} updatecoeffsE[{{N_updatecoeffsE}}] = 
{
    {% for i in updateEVal %}
    {{i}},
    {% endfor %}
};


__kernel void order1_xminus(int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global const {{REAL}}* restrict Ex, __global {{REAL}} *Ey, __global {{REAL}} *Ez, __global const {{REAL}}* restrict Hx, __global const {{REAL}}* restrict Hy, __global const {{REAL}}* restrict Hz, __global {{REAL}} *PHI1, __global {{REAL}} *PHI2, __global const {{REAL}}* restrict RA, __global const {{REAL}}* restrict RB, __global const {{REAL}}* restrict RE, __global const {{REAL}}* restrict RF, {{REAL}} d){

    //  This function updates the Ey and Ez field components for the xminus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_PHI, NY_PHI, NZ_PHI, NY_R: Dimensions of PHI1, PHI2, and R PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      Phi, RA, RB, RE, RF: Access to PML electric coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    // obtain the linear index corresponding to the current thread
    int idx = get_global_id(2) * get_global_size(0) * get_global_size(1) + get_global_id(1) * get_global_size(0) + get_global_id(0);

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    int p1 = idx / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    int i1 = (idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    int j1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    int k1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    int p2 = idx / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    int i2 = (idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    int j2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    int k2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

    {{REAL}} RA01, RB0, RE0, RF0, dHy, dHz;
    {{REAL}} dx = d;
    int ii, jj, kk, materialEy, materialEz;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 == 0 && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = xf - i1;
        jj = j1 + ys;
        kk = k1 + zs;

        // PML coefficients
        RA01 = RA[INDEX2D_R(0,i1)] - 1;
        RB0 = RB[INDEX2D_R(0,i1)];
        RE0 = RE[INDEX2D_R(0,i1)];
        RF0 = RF[INDEX2D_R(0,i1)];

        // Ey
        materialEy = ID[INDEX4D_ID(1,ii,jj,kk)];
        dHz = (Hz[INDEX3D_FIELDS(ii,jj,kk)] - Hz[INDEX3D_FIELDS(ii-1,jj,kk)]) / dx;
        Ey[INDEX3D_FIELDS(ii,jj,kk)] = Ey[INDEX3D_FIELDS(ii,jj,kk)] - updatecoeffsE[INDEX2D_MAT(materialEy,4)] * (RA01 * dHz + RB0 * PHI1[INDEX4D_PHI1(0,i1,j1,k1)]);
        PHI1[INDEX4D_PHI1(0,i1,j1,k1)] = RE0 * PHI1[INDEX4D_PHI1(0,i1,j1,k1)] - RF0 * dHz;
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = xf - i2;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        RA01 = RA[INDEX2D_R(0,i2)] - 1;
        RB0 = RB[INDEX2D_R(0,i2)];
        RE0 = RE[INDEX2D_R(0,i2)];
        RF0 = RF[INDEX2D_R(0,i2)];

        // Ez
        materialEz = ID[INDEX4D_ID(2,ii,jj,kk)];
        dHy = (Hy[INDEX3D_FIELDS(ii,jj,kk)] - Hy[INDEX3D_FIELDS(ii-1,jj,kk)]) / dx;
        Ez[INDEX3D_FIELDS(ii,jj,kk)] = Ez[INDEX3D_FIELDS(ii,jj,kk)] + updatecoeffsE[INDEX2D_MAT(materialEz,4)] * (RA01 * dHy + RB0 * PHI2[INDEX4D_PHI2(0,i2,j2,k2)]);
        PHI2[INDEX4D_PHI2(0,i2,j2,k2)] = RE0 * PHI2[INDEX4D_PHI2(0,i2,j2,k2)] - RF0 * dHy;
    }
}

__kernel void order2_xminus(int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global const {{REAL}}* restrict Ex, __global {{REAL}} *Ey, __global {{REAL}} *Ez, __global const {{REAL}}* restrict Hx, __global const {{REAL}}* restrict Hy, __global const {{REAL}}* restrict Hz, __global {{REAL}} *PHI1, __global {{REAL}} *PHI2, __global const {{REAL}}* restrict RA, __global const {{REAL}}* restrict RB, __global const {{REAL}}* restrict RE, __global const {{REAL}}* restrict RF, {{REAL}} d){
    //  This function updates the Ey and Ez field components for the xminus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_PHI, NY_PHI, NZ_PHI, NY_R: Dimensions of PHI1, PHI2, and R PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      Phi, RA, RB, RE, RF: Access to PML electric coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    // Obtain the linear index corresponding to the current thread
    int idx = get_global_id(2) * get_global_size(0) * get_global_size(1) + get_global_id(1) * get_global_size(0) + get_global_id(0);

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    int p1 = idx / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    int i1 = (idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    int j1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    int k1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    int p2 = idx / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    int i2 = (idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    int j2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    int k2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

    {{REAL}} RA0, RB0, RE0, RF0, RA1, RB1, RE1, RF1, RA01, dHy, dHz;
    {{REAL}} dx = d;
    int ii, jj, kk, materialEy, materialEz;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 == 0 && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = xf - i1;
        jj = j1 + ys;
        kk = k1 + zs;

        // PML coefficients
        RA0 = RA[INDEX2D_R(0,i1)];
        RB0 = RB[INDEX2D_R(0,i1)];
        RE0 = RE[INDEX2D_R(0,i1)];
        RF0 = RF[INDEX2D_R(0,i1)];
        RA1 = RA[INDEX2D_R(1,i1)];
        RB1 = RB[INDEX2D_R(1,i1)];
        RE1 = RE[INDEX2D_R(1,i1)];
        RF1 = RF[INDEX2D_R(1,i1)];
        RA01 = RA[INDEX2D_R(0,i1)] * RA[INDEX2D_R(1,i1)] - 1;

        // Ey
        materialEy = ID[INDEX4D_ID(1,ii,jj,kk)];
        dHz = (Hz[INDEX3D_FIELDS(ii,jj,kk)] - Hz[INDEX3D_FIELDS(ii-1,jj,kk)]) / dx;
        Ey[INDEX3D_FIELDS(ii,jj,kk)] = Ey[INDEX3D_FIELDS(ii,jj,kk)] - updatecoeffsE[INDEX2D_MAT(materialEy,4)] * (RA01 * dHz + RA1 * RB0 * PHI1[INDEX4D_PHI1(0,i1,j1,k1)] + RB1 * PHI1[INDEX4D_PHI1(1,i1,j1,k1)]);
        PHI1[INDEX4D_PHI1(1,i1,j1,k1)] = RE1 * PHI1[INDEX4D_PHI1(1,i1,j1,k1)] - RF1 * (RA0 * dHz + RB0 * PHI1[INDEX4D_PHI1(0,i1,j1,k1)]);
        PHI1[INDEX4D_PHI1(0,i1,j1,k1)] = RE0 * PHI1[INDEX4D_PHI1(0,i1,j1,k1)] - RF0 * dHz;
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = xf - i2;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        RA0 = RA[INDEX2D_R(0,i2)];
        RB0 = RB[INDEX2D_R(0,i2)];
        RE0 = RE[INDEX2D_R(0,i2)];
        RF0 = RF[INDEX2D_R(0,i2)];
        RA1 = RA[INDEX2D_R(1,i2)];
        RB1 = RB[INDEX2D_R(1,i2)];
        RE1 = RE[INDEX2D_R(1,i2)];
        RF1 = RF[INDEX2D_R(1,i2)];
        RA01 = RA[INDEX2D_R(0,i2)] * RA[INDEX2D_R(1,i2)] - 1;

        // Ez
        materialEz = ID[INDEX4D_ID(2,ii,jj,kk)];
        dHy = (Hy[INDEX3D_FIELDS(ii,jj,kk)] - Hy[INDEX3D_FIELDS(ii-1,jj,kk)]) / dx;
        Ez[INDEX3D_FIELDS(ii,jj,kk)] = Ez[INDEX3D_FIELDS(ii,jj,kk)] + updatecoeffsE[INDEX2D_MAT(materialEz,4)] * (RA01 * dHy + RA1 * RB0 * PHI2[INDEX4D_PHI2(0,i2,j2,k2)] + RB1 * PHI2[INDEX4D_PHI2(1,i2,j2,k2)]);
        PHI2[INDEX4D_PHI2(1,i2,j2,k2)] = RE1 * PHI2[INDEX4D_PHI2(1,i2,j2,k2)] - RF1 * (RA0 * dHy + RB0 * PHI2[INDEX4D_PHI2(0,i2,j2,k2)]);
        PHI2[INDEX4D_PHI2(0,i2,j2,k2)] = RE0 * PHI2[INDEX4D_PHI2(0,i2,j2,k2)] - RF0 * dHy;
    }
}

__kernel void order1_xplus(int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global const {{REAL}}* restrict Ex, __global {{REAL}} *Ey, __global {{REAL}} *Ez, __global const {{REAL}}* restrict Hx, __global const {{REAL}}* restrict Hy, __global const {{REAL}}* restrict Hz, __global {{REAL}} *PHI1, __global {{REAL}} *PHI2, __global const {{REAL}}* restrict RA, __global const {{REAL}}* restrict RB, __global const {{REAL}}* restrict RE, __global const {{REAL}}* restrict RF, {{REAL}} d){
    //  This function updates the Ey and Ez field components for the xplus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_PHI, NY_PHI, NZ_PHI, NY_R: Dimensions of PHI1, PHI2, and R PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      Phi, RA, RB, RE, RF: Access to PML electric coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    // Obtain the linear index corresponding to the current thread
    int idx = get_global_id(2) * get_global_size(0) * get_global_size(1) + get_global_id(1) * get_global_size(0) + get_global_id(0);

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    int p1 = idx / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    int i1 = (idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    int j1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    int k1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    int p2 = idx / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    int i2 = (idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    int j2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    int k2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

    {{REAL}} RA01, RB0, RE0, RF0, dHy, dHz;
    {{REAL}} dx = d;
    int ii, jj, kk, materialEy, materialEz;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 == 0 && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = j1 + ys;
        kk = k1 + zs;

        // PML coefficients
        RA01 = RA[INDEX2D_R(0,i1)] - 1;
        RB0 = RB[INDEX2D_R(0,i1)];
        RE0 = RE[INDEX2D_R(0,i1)];
        RF0 = RF[INDEX2D_R(0,i1)];

        // Ey
        materialEy = ID[INDEX4D_ID(1,ii,jj,kk)];
        dHz = (Hz[INDEX3D_FIELDS(ii,jj,kk)] - Hz[INDEX3D_FIELDS(ii-1,jj,kk)]) / dx;
        Ey[INDEX3D_FIELDS(ii,jj,kk)] = Ey[INDEX3D_FIELDS(ii,jj,kk)] - updatecoeffsE[INDEX2D_MAT(materialEy,4)] * (RA01 * dHz + RB0 * PHI1[INDEX4D_PHI1(0,i1,j1,k1)]);
        PHI1[INDEX4D_PHI1(0,i1,j1,k1)] = RE0 * PHI1[INDEX4D_PHI1(0,i1,j1,k1)] - RF0 * dHz;
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        RA01 = RA[INDEX2D_R(0,i2)] - 1;
        RB0 = RB[INDEX2D_R(0,i2)];
        RE0 = RE[INDEX2D_R(0,i2)];
        RF0 = RF[INDEX2D_R(0,i2)];

        // Ez
        materialEz = ID[INDEX4D_ID(2,ii,jj,kk)];
        dHy = (Hy[INDEX3D_FIELDS(ii,jj,kk)] - Hy[INDEX3D_FIELDS(ii-1,jj,kk)]) / dx;
        Ez[INDEX3D_FIELDS(ii,jj,kk)] = Ez[INDEX3D_FIELDS(ii,jj,kk)] + updatecoeffsE[INDEX2D_MAT(materialEz,4)] * (RA01 * dHy + RB0 * PHI2[INDEX4D_PHI2(0,i2,j2,k2)]);
        PHI2[INDEX4D_PHI2(0,i2,j2,k2)] = RE0 * PHI2[INDEX4D_PHI2(0,i2,j2,k2)] - RF0 * dHy;
    }
}

__kernel void order2_xplus(int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global const {{REAL}}* restrict Ex, __global {{REAL}} *Ey, __global {{REAL}} *Ez, __global const {{REAL}}* restrict Hx, __global const {{REAL}}* restrict Hy, __global const {{REAL}}* restrict Hz, __global {{REAL}} *PHI1, __global {{REAL}} *PHI2, __global const {{REAL}}* restrict RA, __global const {{REAL}}* restrict RB, __global const {{REAL}}* restrict RE, __global const {{REAL}}* restrict RF, {{REAL}} d) {

    //  This function updates the Ey and Ez field components for the xplus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_PHI, NY_PHI, NZ_PHI, NY_R: Dimensions of PHI1, PHI2, and R PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      Phi, RA, RB, RE, RF: Access to PML electric coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    // Obtain the linear index corresponding to the current thread
    int idx = get_global_id(2) * get_global_size(0) * get_global_size(1) + get_global_id(1) * get_global_size(0) + get_global_id(0);

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    int p1 = idx / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    int i1 = (idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    int j1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    int k1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    int p2 = idx / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    int i2 = (idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    int j2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    int k2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

    {{REAL}} RA0, RB0, RE0, RF0, RA1, RB1, RE1, RF1, RA01, dHy, dHz;
    {{REAL}} dx = d;
    int ii, jj, kk, materialEy, materialEz;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 == 0 && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = j1 + ys;
        kk = k1 + zs;

        // PML coefficients
        RA0 = RA[INDEX2D_R(0,i1)];
        RB0 = RB[INDEX2D_R(0,i1)];
        RE0 = RE[INDEX2D_R(0,i1)];
        RF0 = RF[INDEX2D_R(0,i1)];
        RA1 = RA[INDEX2D_R(1,i1)];
        RB1 = RB[INDEX2D_R(1,i1)];
        RE1 = RE[INDEX2D_R(1,i1)];
        RF1 = RF[INDEX2D_R(1,i1)];
        RA01 = RA[INDEX2D_R(0,i1)] * RA[INDEX2D_R(1,i1)] - 1;

        // Ey
        materialEy = ID[INDEX4D_ID(1,ii,jj,kk)];
        dHz = (Hz[INDEX3D_FIELDS(ii,jj,kk)] - Hz[INDEX3D_FIELDS(ii-1,jj,kk)]) / dx;
        Ey[INDEX3D_FIELDS(ii,jj,kk)] = Ey[INDEX3D_FIELDS(ii,jj,kk)] - updatecoeffsE[INDEX2D_MAT(materialEy,4)] * (RA01 * dHz + RA1 * RB0 * PHI1[INDEX4D_PHI1(0,i1,j1,k1)] + RB1 * PHI1[INDEX4D_PHI1(1,i1,j1,k1)]);
        PHI1[INDEX4D_PHI1(1,i1,j1,k1)] = RE1 * PHI1[INDEX4D_PHI1(1,i1,j1,k1)] - RF1 * (RA0 * dHz + RB0 * PHI1[INDEX4D_PHI1(0,i1,j1,k1)]);
        PHI1[INDEX4D_PHI1(0,i1,j1,k1)] = RE0 * PHI1[INDEX4D_PHI1(0,i1,j1,k1)] - RF0 * dHz;
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        RA0 = RA[INDEX2D_R(0,i2)];
        RB0 = RB[INDEX2D_R(0,i2)];
        RE0 = RE[INDEX2D_R(0,i2)];
        RF0 = RF[INDEX2D_R(0,i2)];
        RA1 = RA[INDEX2D_R(1,i2)];
        RB1 = RB[INDEX2D_R(1,i2)];
        RE1 = RE[INDEX2D_R(1,i2)];
        RF1 = RF[INDEX2D_R(1,i2)];
        RA01 = RA[INDEX2D_R(0,i2)] * RA[INDEX2D_R(1,i2)] - 1;

        // Ez
        materialEz = ID[INDEX4D_ID(2,ii,jj,kk)];
        dHy = (Hy[INDEX3D_FIELDS(ii,jj,kk)] - Hy[INDEX3D_FIELDS(ii-1,jj,kk)]) / dx;
        Ez[INDEX3D_FIELDS(ii,jj,kk)] = Ez[INDEX3D_FIELDS(ii,jj,kk)] + updatecoeffsE[INDEX2D_MAT(materialEz,4)] * (RA01 * dHy + RA1 * RB0 * PHI2[INDEX4D_PHI2(0,i2,j2,k2)] + RB1 * PHI2[INDEX4D_PHI2(1,i2,j2,k2)]);
        PHI2[INDEX4D_PHI2(1,i2,j2,k2)] = RE1 * PHI2[INDEX4D_PHI2(1,i2,j2,k2)] - RF1 * (RA0 * dHy + RB0 * PHI2[INDEX4D_PHI2(0,i2,j2,k2)]);
        PHI2[INDEX4D_PHI2(0,i2,j2,k2)] = RE0 * PHI2[INDEX4D_PHI2(0,i2,j2,k2)] - RF0 * dHy;
    }
}


__kernel void order1_yminus(int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global {{REAL}} *Ex, __global const {{REAL}}* restrict Ey, __global {{REAL}} *Ez, __global const {{REAL}}* restrict Hx, __global const {{REAL}}* restrict Hy, __global const {{REAL}}* restrict Hz, __global {{REAL}} *PHI1, __global {{REAL}} *PHI2, __global const {{REAL}}* restrict RA, __global const {{REAL}}* restrict RB, __global const {{REAL}}* restrict RE, __global const {{REAL}}* restrict RF, {{REAL}} d) {

    //  This function updates the Ex and Ez field components for the yminus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_PHI, NY_PHI, NZ_PHI, NY_R: Dimensions of PHI1, PHI2, and R PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      Phi, RA, RB, RE, RF: Access to PML electric coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    // Obtain the linear index corresponding to the current thread
    int idx = get_global_id(2) * get_global_size(0) * get_global_size(1) + get_global_id(1) * get_global_size(0) + get_global_id(0);

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    int p1 = idx / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    int i1 = (idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    int j1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    int k1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    int p2 = idx / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    int i2 = (idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    int j2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    int k2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

    {{REAL}} RA01, RB0, RE0, RF0, dHx, dHz;
    {{REAL}} dy = d;
    int ii, jj, kk, materialEx, materialEz;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 == 0 && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = yf - j1;
        kk = k1 + zs;

        // PML coefficients
        RA01 = RA[INDEX2D_R(0,j1)] - 1;
        RB0 = RB[INDEX2D_R(0,j1)];
        RE0 = RE[INDEX2D_R(0,j1)];
        RF0 = RF[INDEX2D_R(0,j1)];

        // Ex
        materialEx = ID[INDEX4D_ID(0,ii,jj,kk)];
        dHz = (Hz[INDEX3D_FIELDS(ii,jj,kk)] - Hz[INDEX3D_FIELDS(ii,jj-1,kk)]) / dy;
        Ex[INDEX3D_FIELDS(ii,jj,kk)] = Ex[INDEX3D_FIELDS(ii,jj,kk)] + updatecoeffsE[INDEX2D_MAT(materialEx,4)] * (RA01 * dHz + RB0 * PHI1[INDEX4D_PHI1(0,i1,j1,k1)]);
        PHI1[INDEX4D_PHI1(0,i1,j1,k1)] = RE0 * PHI1[INDEX4D_PHI1(0,i1,j1,k1)] - RF0 * dHz;
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = yf - j2;
        kk = k2 + zs;

        // PML coefficients
        RA01 = RA[INDEX2D_R(0,j2)] - 1;
        RB0 = RB[INDEX2D_R(0,j2)];
        RE0 = RE[INDEX2D_R(0,j2)];
        RF0 = RF[INDEX2D_R(0,j2)];

        // Ez
        materialEz = ID[INDEX4D_ID(2,ii,jj,kk)];
        dHx = (Hx[INDEX3D_FIELDS(ii,jj,kk)] - Hx[INDEX3D_FIELDS(ii,jj-1,kk)]) / dy;
        Ez[INDEX3D_FIELDS(ii,jj,kk)] = Ez[INDEX3D_FIELDS(ii,jj,kk)] - updatecoeffsE[INDEX2D_MAT(materialEz,4)] * (RA01 * dHx + RB0 * PHI2[INDEX4D_PHI2(0,i2,j2,k2)]);
        PHI2[INDEX4D_PHI2(0,i2,j2,k2)] = RE0 * PHI2[INDEX4D_PHI2(0,i2,j2,k2)] - RF0 * dHx;
    }
}

__kernel void order2_yminus(int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global {{REAL}} *Ex, __global const {{REAL}}* restrict Ey, __global {{REAL}} *Ez, __global const {{REAL}}* restrict Hx, __global const {{REAL}}* restrict Hy, __global const {{REAL}}* restrict Hz, __global {{REAL}} *PHI1, __global {{REAL}} *PHI2, __global const {{REAL}}* restrict RA, __global const {{REAL}}* restrict RB, __global const {{REAL}}* restrict RE, __global const {{REAL}}* restrict RF, {{REAL}} d) {

    //  This function updates the Ex and Ez field components for the yminus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_PHI, NY_PHI, NZ_PHI, NY_R: Dimensions of PHI1, PHI2, and R PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      Phi, RA, RB, RE, RF: Access to PML electric coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    // Obtain the linear index corresponding to the current thread
    int idx = get_global_id(2) * get_global_size(0) * get_global_size(1) + get_global_id(1) * get_global_size(0) + get_global_id(0);

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    int p1 = idx / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    int i1 = (idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    int j1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    int k1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    int p2 = idx / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    int i2 = (idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    int j2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    int k2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

    {{REAL}} RA0, RB0, RE0, RF0, RA1, RB1, RE1, RF1, RA01, dHx, dHz;
    {{REAL}} dy = d;
    int ii, jj, kk, materialEx, materialEz;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 == 0 && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = yf - j1;
        kk = k1 + zs;

        // PML coefficients
        RA0 = RA[INDEX2D_R(0,j1)];
        RB0 = RB[INDEX2D_R(0,j1)];
        RE0 = RE[INDEX2D_R(0,j1)];
        RF0 = RF[INDEX2D_R(0,j1)];
        RA1 = RA[INDEX2D_R(1,j1)];
        RB1 = RB[INDEX2D_R(1,j1)];
        RE1 = RE[INDEX2D_R(1,j1)];
        RF1 = RF[INDEX2D_R(1,j1)];
        RA01 = RA[INDEX2D_R(0,j1)] * RA[INDEX2D_R(1,j1)] - 1;

        // Ex
        materialEx = ID[INDEX4D_ID(0,ii,jj,kk)];
        dHz = (Hz[INDEX3D_FIELDS(ii,jj,kk)] - Hz[INDEX3D_FIELDS(ii,jj-1,kk)]) / dy;
        Ex[INDEX3D_FIELDS(ii,jj,kk)] = Ex[INDEX3D_FIELDS(ii,jj,kk)] + updatecoeffsE[INDEX2D_MAT(materialEx,4)] * (RA01 * dHz + RA1 * RB0 * PHI1[INDEX4D_PHI1(0,i1,j1,k1)] + RB1 * PHI1[INDEX4D_PHI1(1,i1,j1,k1)]);
        PHI1[INDEX4D_PHI1(1,i1,j1,k1)] = RE1 * PHI1[INDEX4D_PHI1(1,i1,j1,k1)] - RF1 * (RA0 * dHz + RB0 * PHI1[INDEX4D_PHI1(0,i1,j1,k1)]);
        PHI1[INDEX4D_PHI1(0,i1,j1,k1)] = RE0 * PHI1[INDEX4D_PHI1(0,i1,j1,k1)] - RF0 * dHz;
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = yf - j2;
        kk = k2 + zs;

        // PML coefficients
        RA0 = RA[INDEX2D_R(0,j2)];
        RB0 = RB[INDEX2D_R(0,j2)];
        RE0 = RE[INDEX2D_R(0,j2)];
        RF0 = RF[INDEX2D_R(0,j2)];
        RA1 = RA[INDEX2D_R(1,j2)];
        RB1 = RB[INDEX2D_R(1,j2)];
        RE1 = RE[INDEX2D_R(1,j2)];
        RF1 = RF[INDEX2D_R(1,j2)];
        RA01 = RA[INDEX2D_R(0,j2)] * RA[INDEX2D_R(1,j2)] - 1;

        // Ez
        materialEz = ID[INDEX4D_ID(2,ii,jj,kk)];
        dHx = (Hx[INDEX3D_FIELDS(ii,jj,kk)] - Hx[INDEX3D_FIELDS(ii,jj-1,kk)]) / dy;
        Ez[INDEX3D_FIELDS(ii,jj,kk)] = Ez[INDEX3D_FIELDS(ii,jj,kk)] - updatecoeffsE[INDEX2D_MAT(materialEz,4)] * (RA01 * dHx + RA1 * RB0 * PHI2[INDEX4D_PHI2(0,i2,j2,k2)] + RB1 * PHI2[INDEX4D_PHI2(1,i2,j2,k2)]);
        PHI2[INDEX4D_PHI2(1,i2,j2,k2)] = RE1 * PHI2[INDEX4D_PHI2(1,i2,j2,k2)] - RF1 * (RA0 * dHx + RB0 * PHI2[INDEX4D_PHI2(0,i2,j2,k2)]);
        PHI2[INDEX4D_PHI2(0,i2,j2,k2)] = RE0 * PHI2[INDEX4D_PHI2(0,i2,j2,k2)] - RF0 * dHx;
    }
}


__kernel void order1_yplus(int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global {{REAL}} *Ex, __global const {{REAL}}* restrict Ey, __global {{REAL}} *Ez, __global const {{REAL}}* restrict Hx, __global const {{REAL}}* restrict Hy, __global const {{REAL}}* restrict Hz, __global {{REAL}} *PHI1, __global {{REAL}} *PHI2, __global const {{REAL}}* restrict RA, __global const {{REAL}}* restrict RB, __global const {{REAL}}* restrict RE, __global const {{REAL}}* restrict RF, {{REAL}} d) {

    //  This function updates the Ex and Ez field components for the yplus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_PHI, NY_PHI, NZ_PHI, NY_R: Dimensions of PHI1, PHI2, and R PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      Phi, RA, RB, RE, RF: Access to PML electric coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    // Obtain the linear index corresponding to the current thread
    int idx = get_global_id(2) * get_global_size(0) * get_global_size(1) + get_global_id(1) * get_global_size(0) + get_global_id(0);

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    int p1 = idx / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    int i1 = (idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    int j1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    int k1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    int p2 = idx / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    int i2 = (idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    int j2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    int k2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

    {{REAL}} RA01, RB0, RE0, RF0, dHx, dHz;
    {{REAL}} dy = d;
    int ii, jj, kk, materialEx, materialEz;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 == 0 && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = j1 + ys;
        kk = k1 + zs;

        // PML coefficients
        RA01 = RA[INDEX2D_R(0,j1)] - 1;
        RB0 = RB[INDEX2D_R(0,j1)];
        RE0 = RE[INDEX2D_R(0,j1)];
        RF0 = RF[INDEX2D_R(0,j1)];

        // Ex
        materialEx = ID[INDEX4D_ID(0,ii,jj,kk)];
        dHz = (Hz[INDEX3D_FIELDS(ii,jj,kk)] - Hz[INDEX3D_FIELDS(ii,jj-1,kk)]) / dy;
        Ex[INDEX3D_FIELDS(ii,jj,kk)] = Ex[INDEX3D_FIELDS(ii,jj,kk)] + updatecoeffsE[INDEX2D_MAT(materialEx,4)] * (RA01 * dHz + RB0 * PHI1[INDEX4D_PHI1(0,i1,j1,k1)]);
        PHI1[INDEX4D_PHI1(0,i1,j1,k1)] = RE0 * PHI1[INDEX4D_PHI1(0,i1,j1,k1)] - RF0 * dHz;
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        RA01 = RA[INDEX2D_R(0,j2)] - 1;
        RB0 = RB[INDEX2D_R(0,j2)];
        RE0 = RE[INDEX2D_R(0,j2)];
        RF0 = RF[INDEX2D_R(0,j2)];

        // Ez
        materialEz = ID[INDEX4D_ID(2,ii,jj,kk)];
        dHx = (Hx[INDEX3D_FIELDS(ii,jj,kk)] - Hx[INDEX3D_FIELDS(ii,jj-1,kk)]) / dy;
        Ez[INDEX3D_FIELDS(ii,jj,kk)] = Ez[INDEX3D_FIELDS(ii,jj,kk)] - updatecoeffsE[INDEX2D_MAT(materialEz,4)] * (RA01 * dHx + RB0 * PHI2[INDEX4D_PHI2(0,i2,j2,k2)]);
        PHI2[INDEX4D_PHI2(0,i2,j2,k2)] = RE0 * PHI2[INDEX4D_PHI2(0,i2,j2,k2)] - RF0 * dHx;
    }
}


__kernel void order2_yplus(int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global {{REAL}} *Ex, __global const {{REAL}}* restrict Ey, __global {{REAL}} *Ez, __global const {{REAL}}* restrict Hx, __global const {{REAL}}* restrict Hy, __global const {{REAL}}* restrict Hz, __global {{REAL}} *PHI1, __global {{REAL}} *PHI2, __global const {{REAL}}* restrict RA, __global const {{REAL}}* restrict RB, __global const {{REAL}}* restrict RE, __global const {{REAL}}* restrict RF, {{REAL}} d) {

    //  This function updates the Ex and Ez field components for the yplus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_PHI, NY_PHI, NZ_PHI, NY_R: Dimensions of PHI1, PHI2, and R PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      Phi, RA, RB, RE, RF: Access to PML electric coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    // Obtain the linear index corresponding to the current thread
    int idx = get_global_id(2) * get_global_size(0) * get_global_size(1) + get_global_id(1) * get_global_size(0) + get_global_id(0);

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    int p1 = idx / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    int i1 = (idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    int j1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    int k1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    int p2 = idx / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    int i2 = (idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    int j2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    int k2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

    {{REAL}} RA0, RB0, RE0, RF0, RA1, RB1, RE1, RF1, RA01, dHx, dHz;
    {{REAL}} dy = d;
    int ii, jj, kk, materialEx, materialEz;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 == 0 && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = j1 + ys;
        kk = k1 + zs;

        // PML coefficients
        RA0 = RA[INDEX2D_R(0,j1)];
        RB0 = RB[INDEX2D_R(0,j1)];
        RE0 = RE[INDEX2D_R(0,j1)];
        RF0 = RF[INDEX2D_R(0,j1)];
        RA1 = RA[INDEX2D_R(1,j1)];
        RB1 = RB[INDEX2D_R(1,j1)];
        RE1 = RE[INDEX2D_R(1,j1)];
        RF1 = RF[INDEX2D_R(1,j1)];
        RA01 = RA[INDEX2D_R(0,j1)] * RA[INDEX2D_R(1,j1)] - 1;

        // Ex
        materialEx = ID[INDEX4D_ID(0,ii,jj,kk)];
        dHz = (Hz[INDEX3D_FIELDS(ii,jj,kk)] - Hz[INDEX3D_FIELDS(ii,jj-1,kk)]) / dy;
        Ex[INDEX3D_FIELDS(ii,jj,kk)] = Ex[INDEX3D_FIELDS(ii,jj,kk)] + updatecoeffsE[INDEX2D_MAT(materialEx,4)] * (RA01 * dHz + RA1 * RB0 * PHI1[INDEX4D_PHI1(0,i1,j1,k1)] + RB1 * PHI1[INDEX4D_PHI1(1,i1,j1,k1)]);
        PHI1[INDEX4D_PHI1(1,i1,j1,k1)] = RE1 * PHI1[INDEX4D_PHI1(1,i1,j1,k1)] - RF1 * (RA0 * dHz + RB0 * PHI1[INDEX4D_PHI1(0,i1,j1,k1)]);
        PHI1[INDEX4D_PHI1(0,i1,j1,k1)] = RE0 * PHI1[INDEX4D_PHI1(0,i1,j1,k1)] - RF0 * dHz;
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        RA0 = RA[INDEX2D_R(0,j2)];
        RB0 = RB[INDEX2D_R(0,j2)];
        RE0 = RE[INDEX2D_R(0,j2)];
        RF0 = RF[INDEX2D_R(0,j2)];
        RA1 = RA[INDEX2D_R(1,j2)];
        RB1 = RB[INDEX2D_R(1,j2)];
        RE1 = RE[INDEX2D_R(1,j2)];
        RF1 = RF[INDEX2D_R(1,j2)];
        RA01 = RA[INDEX2D_R(0,j2)] * RA[INDEX2D_R(1,j2)] - 1;

        // Ez
        materialEz = ID[INDEX4D_ID(2,ii,jj,kk)];
        dHx = (Hx[INDEX3D_FIELDS(ii,jj,kk)] - Hx[INDEX3D_FIELDS(ii,jj-1,kk)]) / dy;
        Ez[INDEX3D_FIELDS(ii,jj,kk)] = Ez[INDEX3D_FIELDS(ii,jj,kk)] - updatecoeffsE[INDEX2D_MAT(materialEz,4)] * (RA01 * dHx + RA1 * RB0 * PHI2[INDEX4D_PHI2(0,i2,j2,k2)] + RB1 * PHI2[INDEX4D_PHI2(1,i2,j2,k2)]);
        PHI2[INDEX4D_PHI2(1,i2,j2,k2)] = RE1 * PHI2[INDEX4D_PHI2(1,i2,j2,k2)] - RF1 * (RA0 * dHx + RB0 * PHI2[INDEX4D_PHI2(0,i2,j2,k2)]);
        PHI2[INDEX4D_PHI2(0,i2,j2,k2)] = RE0 * PHI2[INDEX4D_PHI2(0,i2,j2,k2)] - RF0 * dHx;
    }
}

__kernel void order1_zminus(int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global {{REAL}} *Ex, __global {{REAL}} *Ey, __global const {{REAL}}* restrict Ez, __global const {{REAL}}* restrict Hx, __global const {{REAL}}* restrict Hy, __global const {{REAL}}* restrict Hz, __global {{REAL}} *PHI1, __global {{REAL}} *PHI2, __global const {{REAL}}* restrict RA, __global const {{REAL}}* restrict RB, __global const {{REAL}}* restrict RE, __global const {{REAL}}* restrict RF, {{REAL}} d) {

    //  This function updates the Ex and Ey field components for the zminus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_PHI, NY_PHI, NZ_PHI, NY_R: Dimensions of PHI1, PHI2, and R PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      Phi, RA, RB, RE, RF: Access to PML electric coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    // Obtain the linear index corresponding to the current thread
    int idx = get_global_id(2) * get_global_size(0) * get_global_size(1) + get_global_id(1) * get_global_size(0) + get_global_id(0);

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    int p1 = idx / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    int i1 = (idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    int j1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    int k1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    int p2 = idx / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    int i2 = (idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    int j2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    int k2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

    {{REAL}} RA01, RB0, RE0, RF0, dHx, dHy;
    {{REAL}} dz = d;
    int ii, jj, kk, materialEx, materialEy;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 == 0 && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = j1 + ys;
        kk = zf - k1;

        // PML coefficients
        RA01 = RA[INDEX2D_R(0,k1)] - 1;
        RB0 = RB[INDEX2D_R(0,k1)];
        RE0 = RE[INDEX2D_R(0,k1)];
        RF0 = RF[INDEX2D_R(0,k1)];

        // Ex
        materialEx = ID[INDEX4D_ID(0,ii,jj,kk)];
        dHy = (Hy[INDEX3D_FIELDS(ii,jj,kk)] - Hy[INDEX3D_FIELDS(ii,jj,kk-1)]) / dz;
        Ex[INDEX3D_FIELDS(ii,jj,kk)] = Ex[INDEX3D_FIELDS(ii,jj,kk)] - updatecoeffsE[INDEX2D_MAT(materialEx,4)] * (RA01 * dHy + RB0 * PHI1[INDEX4D_PHI1(0,i1,j1,k1)]);
        PHI1[INDEX4D_PHI1(0,i1,j1,k1)] = RE0 * PHI1[INDEX4D_PHI1(0,i1,j1,k1)] - RF0 * dHy;
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = zf - k2;

        // PML coefficients
        RA01 = RA[INDEX2D_R(0,k2)] - 1;
        RB0 = RB[INDEX2D_R(0,k2)];
        RE0 = RE[INDEX2D_R(0,k2)];
        RF0 = RF[INDEX2D_R(0,k2)];

        // Ey
        materialEy = ID[INDEX4D_ID(1,ii,jj,kk)];
        dHx = (Hx[INDEX3D_FIELDS(ii,jj,kk)] - Hx[INDEX3D_FIELDS(ii,jj,kk-1)]) / dz;
        Ey[INDEX3D_FIELDS(ii,jj,kk)] = Ey[INDEX3D_FIELDS(ii,jj,kk)] + updatecoeffsE[INDEX2D_MAT(materialEy,4)] * (RA01 * dHx + RB0 * PHI2[INDEX4D_PHI2(0,i2,j2,k2)]);
        PHI2[INDEX4D_PHI2(0,i2,j2,k2)] = RE0 * PHI2[INDEX4D_PHI2(0,i2,j2,k2)] - RF0 * dHx;
    }
}

__kernel void order2_zminus(int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global {{REAL}} *Ex, __global {{REAL}} *Ey, __global const {{REAL}}* restrict Ez, __global const {{REAL}}* restrict Hx, __global const {{REAL}}* restrict Hy, __global const {{REAL}}* restrict Hz, __global {{REAL}} *PHI1, __global {{REAL}} *PHI2, __global const {{REAL}}* restrict RA, __global const {{REAL}}* restrict RB, __global const {{REAL}}* restrict RE, __global const {{REAL}}* restrict RF, {{REAL}} d) {

    //  This function updates the Ex and Ey field components for the zminus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_PHI, NY_PHI, NZ_PHI, NY_R: Dimensions of PHI1, PHI2, and R PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      Phi, RA, RB, RE, RF: Access to PML electric coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    // Obtain the linear index corresponding to the current thread
    int idx = get_global_id(2) * get_global_size(0) * get_global_size(1) + get_global_id(1) * get_global_size(0) + get_global_id(0);

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    int p1 = idx / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    int i1 = (idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    int j1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    int k1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    int p2 = idx / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    int i2 = (idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    int j2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    int k2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

    {{REAL}} RA0, RB0, RE0, RF0, RA1, RB1, RE1, RF1, RA01, dHx, dHy;
    {{REAL}} dz = d;
    int ii, jj, kk, materialEx, materialEy;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 == 0 && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = j1 + ys;
        kk = zf - k1;

        // PML coefficients
        RA0 = RA[INDEX2D_R(0,k1)];
        RB0 = RB[INDEX2D_R(0,k1)];
        RE0 = RE[INDEX2D_R(0,k1)];
        RF0 = RF[INDEX2D_R(0,k1)];
        RA1 = RA[INDEX2D_R(1,k1)];
        RB1 = RB[INDEX2D_R(1,k1)];
        RE1 = RE[INDEX2D_R(1,k1)];
        RF1 = RF[INDEX2D_R(1,k1)];
        RA01 = RA[INDEX2D_R(0,k1)] * RA[INDEX2D_R(1,k1)] - 1;

        // Ex
        materialEx = ID[INDEX4D_ID(0,ii,jj,kk)];
        dHy = (Hy[INDEX3D_FIELDS(ii,jj,kk)] - Hy[INDEX3D_FIELDS(ii,jj,kk-1)]) / dz;
        Ex[INDEX3D_FIELDS(ii,jj,kk)] = Ex[INDEX3D_FIELDS(ii,jj,kk)] - updatecoeffsE[INDEX2D_MAT(materialEx,4)] * (RA01 * dHy + RA1 * RB0 * PHI1[INDEX4D_PHI1(0,i1,j1,k1)] + RB1 * PHI1[INDEX4D_PHI1(1,i1,j1,k1)]);
        PHI1[INDEX4D_PHI1(1,i1,j1,k1)] = RE1 * PHI1[INDEX4D_PHI1(1,i1,j1,k1)] - RF1 * (RA0 * dHy + RB0 * PHI1[INDEX4D_PHI1(0,i1,j1,k1)]);
        PHI1[INDEX4D_PHI1(0,i1,j1,k1)] = RE0 * PHI1[INDEX4D_PHI1(0,i1,j1,k1)] - RF0 * dHy;
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = zf - k2;

        // PML coefficients
        RA0 = RA[INDEX2D_R(0,k2)];
        RB0 = RB[INDEX2D_R(0,k2)];
        RE0 = RE[INDEX2D_R(0,k2)];
        RF0 = RF[INDEX2D_R(0,k2)];
        RA1 = RA[INDEX2D_R(1,k2)];
        RB1 = RB[INDEX2D_R(1,k2)];
        RE1 = RE[INDEX2D_R(1,k2)];
        RF1 = RF[INDEX2D_R(1,k2)];
        RA01 = RA[INDEX2D_R(0,k2)] * RA[INDEX2D_R(1,k2)] - 1;

        // Ey
        materialEy = ID[INDEX4D_ID(1,ii,jj,kk)];
        dHx = (Hx[INDEX3D_FIELDS(ii,jj,kk)] - Hx[INDEX3D_FIELDS(ii,jj,kk-1)]) / dz;
        Ey[INDEX3D_FIELDS(ii,jj,kk)] = Ey[INDEX3D_FIELDS(ii,jj,kk)] + updatecoeffsE[INDEX2D_MAT(materialEy,4)] * (RA01 * dHx + RA1 * RB0 * PHI2[INDEX4D_PHI2(0,i2,j2,k2)] + RB1 * PHI2[INDEX4D_PHI2(1,i2,j2,k2)]);
        PHI2[INDEX4D_PHI2(1,i2,j2,k2)] = RE1 * PHI2[INDEX4D_PHI2(1,i2,j2,k2)] - RF1 * (RA0 * dHx + RB0 * PHI2[INDEX4D_PHI2(0,i2,j2,k2)]);
        PHI2[INDEX4D_PHI2(0,i2,j2,k2)] = RE0 * PHI2[INDEX4D_PHI2(0,i2,j2,k2)] - RF0 * dHx;
    }
}


__kernel void order1_zplus(int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global {{REAL}} *Ex, __global {{REAL}} *Ey, __global const {{REAL}}* restrict Ez, __global const {{REAL}}* restrict Hx, __global const {{REAL}}* restrict Hy, __global const {{REAL}}* restrict Hz, __global {{REAL}} *PHI1, __global {{REAL}} *PHI2, __global const {{REAL}}* restrict RA, __global const {{REAL}}* restrict RB, __global const {{REAL}}* restrict RE, __global const {{REAL}}* restrict RF, {{REAL}} d) {

    //  This function updates the Ex and Ey field components for the zplus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_PHI, NY_PHI, NZ_PHI, NY_R: Dimensions of PHI1, PHI2, and R PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      Phi, RA, RB, RE, RF: Access to PML electric coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    // Obtain the linear index corresponding to the current thread
    int idx = get_global_id(2) * get_global_size(0) * get_global_size(1) + get_global_id(1) * get_global_size(0) + get_global_id(0);

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    int p1 = idx / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    int i1 = (idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    int j1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    int k1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    int p2 = idx / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    int i2 = (idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    int j2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    int k2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

    {{REAL}} RA01, RB0, RE0, RF0, dHx, dHy;
    {{REAL}} dz = d;
    int ii, jj, kk, materialEx, materialEy;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 == 0 && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = j1 + ys;
        kk = k1 + zs;

        // PML coefficients
        RA01 = RA[INDEX2D_R(0,k1)] - 1;
        RB0 = RB[INDEX2D_R(0,k1)];
        RE0 = RE[INDEX2D_R(0,k1)];
        RF0 = RF[INDEX2D_R(0,k1)];

        // Ex
        materialEx = ID[INDEX4D_ID(0,ii,jj,kk)];
        dHy = (Hy[INDEX3D_FIELDS(ii,jj,kk)] - Hy[INDEX3D_FIELDS(ii,jj,kk-1)]) / dz;
        Ex[INDEX3D_FIELDS(ii,jj,kk)] = Ex[INDEX3D_FIELDS(ii,jj,kk)] - updatecoeffsE[INDEX2D_MAT(materialEx,4)] * (RA01 * dHy + RB0 * PHI1[INDEX4D_PHI1(0,i1,j1,k1)]);
        PHI1[INDEX4D_PHI1(0,i1,j1,k1)] = RE0 * PHI1[INDEX4D_PHI1(0,i1,j1,k1)] - RF0 * dHy;
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        RA01 = RA[INDEX2D_R(0,k2)] - 1;
        RB0 = RB[INDEX2D_R(0,k2)];
        RE0 = RE[INDEX2D_R(0,k2)];
        RF0 = RF[INDEX2D_R(0,k2)];

        // Ey
        materialEy = ID[INDEX4D_ID(1,ii,jj,kk)];
        dHx = (Hx[INDEX3D_FIELDS(ii,jj,kk)] - Hx[INDEX3D_FIELDS(ii,jj,kk-1)]) / dz;
        Ey[INDEX3D_FIELDS(ii,jj,kk)] = Ey[INDEX3D_FIELDS(ii,jj,kk)] + updatecoeffsE[INDEX2D_MAT(materialEy,4)] * (RA01 * dHx + RB0 * PHI2[INDEX4D_PHI2(0,i2,j2,k2)]);
        PHI2[INDEX4D_PHI2(0,i2,j2,k2)] = RE0 * PHI2[INDEX4D_PHI2(0,i2,j2,k2)] - RF0 * dHx;
    }
}


__kernel void order2_zplus(int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, __global const unsigned int* restrict ID, __global {{REAL}} *Ex, __global {{REAL}} *Ey, __global const {{REAL}}* restrict Ez, __global const {{REAL}}* restrict Hx, __global const {{REAL}}* restrict Hy, __global const {{REAL}}* restrict Hz, __global {{REAL}} *PHI1, __global {{REAL}} *PHI2, __global const {{REAL}}* restrict RA, __global const {{REAL}}* restrict RB, __global const {{REAL}}* restrict RE, __global const {{REAL}}* restrict RF, {{REAL}} d) {

    //  This function updates the Ex and Ey field components for the zplus slab.
    //
    //  Args:
    //      xs, xf, ys, yf, zs, zf: Cell coordinates of PML slab
    //      NX_PHI, NY_PHI, NZ_PHI, NY_R: Dimensions of PHI1, PHI2, and R PML arrays
    //      ID, E, H: Access to ID and field component arrays
    //      Phi, RA, RB, RE, RF: Access to PML electric coefficient arrays
    //      d: Spatial discretisation, e.g. dx, dy or dz

    // Obtain the linear index corresponding to the current thread
    int idx = get_global_id(2) * get_global_size(0) * get_global_size(1) + get_global_id(1) * get_global_size(0) + get_global_id(0);

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    int p1 = idx / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    int i1 = (idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    int j1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    int k1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    int p2 = idx / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    int i2 = (idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    int j2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    int k2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

    {{REAL}} RA0, RB0, RE0, RF0, RA1, RB1, RE1, RF1, RA01, dHx, dHy;
    {{REAL}} dz = d;
    int ii, jj, kk, materialEx, materialEy;
    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    if (p1 == 0 && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = j1 + ys;
        kk = k1 + zs;

        // PML coefficients
        RA0 = RA[INDEX2D_R(0,k1)];
        RB0 = RB[INDEX2D_R(0,k1)];
        RE0 = RE[INDEX2D_R(0,k1)];
        RF0 = RF[INDEX2D_R(0,k1)];
        RA1 = RA[INDEX2D_R(1,k1)];
        RB1 = RB[INDEX2D_R(1,k1)];
        RE1 = RE[INDEX2D_R(1,k1)];
        RF1 = RF[INDEX2D_R(1,k1)];
        RA01 = RA[INDEX2D_R(0,k1)] * RA[INDEX2D_R(1,k1)] - 1;

        // Ex
        materialEx = ID[INDEX4D_ID(0,ii,jj,kk)];
        dHy = (Hy[INDEX3D_FIELDS(ii,jj,kk)] - Hy[INDEX3D_FIELDS(ii,jj,kk-1)]) / dz;
        Ex[INDEX3D_FIELDS(ii,jj,kk)] = Ex[INDEX3D_FIELDS(ii,jj,kk)] - updatecoeffsE[INDEX2D_MAT(materialEx,4)] * (RA01 * dHy + RA1 * RB0 * PHI1[INDEX4D_PHI1(0,i1,j1,k1)] + RB1 * PHI1[INDEX4D_PHI1(1,i1,j1,k1)]);
        PHI1[INDEX4D_PHI1(1,i1,j1,k1)] = RE1 * PHI1[INDEX4D_PHI1(1,i1,j1,k1)] - RF1 * (RA0 * dHy + RB0 * PHI1[INDEX4D_PHI1(0,i1,j1,k1)]);
        PHI1[INDEX4D_PHI1(0,i1,j1,k1)] = RE0 * PHI1[INDEX4D_PHI1(0,i1,j1,k1)] - RF0 * dHy;
    }

    if (p2 == 0 && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        RA0 = RA[INDEX2D_R(0,k2)];
        RB0 = RB[INDEX2D_R(0,k2)];
        RE0 = RE[INDEX2D_R(0,k2)];
        RF0 = RF[INDEX2D_R(0,k2)];
        RA1 = RA[INDEX2D_R(1,k2)];
        RB1 = RB[INDEX2D_R(1,k2)];
        RE1 = RE[INDEX2D_R(1,k2)];
        RF1 = RF[INDEX2D_R(1,k2)];
        RA01 = RA[INDEX2D_R(0,k2)] * RA[INDEX2D_R(1,k2)] - 1;

        // Ey
        materialEy = ID[INDEX4D_ID(1,ii,jj,kk)];
        dHx = (Hx[INDEX3D_FIELDS(ii,jj,kk)] - Hx[INDEX3D_FIELDS(ii,jj,kk-1)]) / dz;
        Ey[INDEX3D_FIELDS(ii,jj,kk)] = Ey[INDEX3D_FIELDS(ii,jj,kk)] + updatecoeffsE[INDEX2D_MAT(materialEy,4)] * (RA01 * dHx + RA1 * RB0 * PHI2[INDEX4D_PHI2(0,i2,j2,k2)] + RB1 * PHI2[INDEX4D_PHI2(1,i2,j2,k2)]);
        PHI2[INDEX4D_PHI2(1,i2,j2,k2)] = RE1 * PHI2[INDEX4D_PHI2(1,i2,j2,k2)] - RF1 * (RA0 * dHx + RB0 * PHI2[INDEX4D_PHI2(0,i2,j2,k2)]);
        PHI2[INDEX4D_PHI2(0,i2,j2,k2)] = RE0 * PHI2[INDEX4D_PHI2(0,i2,j2,k2)] - RF0 * dHx;
    }
}

