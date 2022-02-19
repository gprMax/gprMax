// Macros for converting subscripts to linear index:
#define INDEX3D_FIELDS(i, j, k) (i)*({{NY_FIELDS}})*({{NZ_FIELDS}})+(j)*({{NZ_FIELDS}})+(k)
#define INDEX4D_SNAPS(p, i, j, k) (p)*({{NX_SNAPS}})*({{NY_SNAPS}})*({{NZ_SNAPS}})+(i)*({{NY_SNAPS}})*({{NZ_SNAPS}})+(j)*({{NZ_SNAPS}})+(k)

////////////////////
// Store snapshot //
////////////////////

__kernel void store_snapshot(int p, int xs, int xf, int ys, int yf, int zs, int zf, int dx, int dy, int dz,
                                __global const {{REAL}}* __restrict__ Ex, __global const {{REAL}}* __restrict__ Ey,
                                __global const {{REAL}}* __restrict__ Ez, __global const {{REAL}}* __restrict__ Hx,
                                __global const {{REAL}}* __restrict__ Hy, __global const {{REAL}}* __restrict__ Hz,
                                __global {{REAL}} *snapEx, __global {{REAL}} *snapEy, __global {{REAL}} *snapEz,
                                __global {{REAL}} *snapHx, __global {{REAL}} *snapHy, __global {{REAL}} *snapHz) {

    //  This function stores field values for a snapshot.
    //
    //  Args:
    //      p: Snapshot number
    //      xs, xf, ys, yf, xs, xf: Start and finish cell coordinates for snapshot
    //      dx, dy, dz: Sampling interval in cell coordinates for snapshot
    //      E, H: Access to field component arrays
    //      snapEx, snapEy, snapEz, snapHx, snapHy, snapHz: Access to arrays to store snapshots

    // Obtain the linear index corresponding to the current thread
    int idx = get_global_id(2) * get_global_size(0) * get_global_size(1) + get_global_id(1) * get_global_size(0) + get_global_id(0);

    // Convert the linear index to subscripts for 4D SNAPS array
    int i = (idx % ({{NX_SNAPS}} * {{NY_SNAPS}} * {{NZ_SNAPS}})) / ({{NY_SNAPS}} * {{NZ_SNAPS}});
    int j = ((idx % ({{NX_SNAPS}} * {{NY_SNAPS}} * {{NZ_SNAPS}})) % ({{NY_SNAPS}} * {{NZ_SNAPS}})) / {{NZ_SNAPS}};
    int k = ((idx % ({{NX_SNAPS}} * {{NY_SNAPS}} * {{NZ_SNAPS}})) % ({{NY_SNAPS}} * {{NZ_SNAPS}})) % {{NZ_SNAPS}};

    // Subscripts for field arrays
    int ii, jj, kk;

    if (i >= xs && i < xf && j >= ys && j < yf && k >= zs && k < zf) {

        // Increment subscripts for field array to account for spatial sampling of snapshot
        ii = (xs + i) * dx;
        jj = (ys + j) * dy;
        kk = (zs + k) * dz;

        // The electric field component value at a point comes from an average of
        // the 4 electric field component values in that cell
        snapEx[INDEX4D_SNAPS(p,i,j,k)] = (Ex[INDEX3D_FIELDS(ii,jj,kk)] + Ex[INDEX3D_FIELDS(ii,jj+1,kk)] + Ex[INDEX3D_FIELDS(ii,jj,kk+1)] + Ex[INDEX3D_FIELDS(ii,jj+1,kk+1)]) / 4;
        snapEy[INDEX4D_SNAPS(p,i,j,k)] = (Ey[INDEX3D_FIELDS(ii,jj,kk)] + Ey[INDEX3D_FIELDS(ii+1,jj,kk)] + Ey[INDEX3D_FIELDS(ii,jj,kk+1)] + Ey[INDEX3D_FIELDS(ii+1,jj,kk+1)]) / 4;
        snapEz[INDEX4D_SNAPS(p,i,j,k)] = (Ez[INDEX3D_FIELDS(ii,jj,kk)] + Ez[INDEX3D_FIELDS(ii+1,jj,kk)] + Ez[INDEX3D_FIELDS(ii,jj+1,kk)] + Ez[INDEX3D_FIELDS(ii+1,jj+1,kk)]) / 4;

        // The magnetic field component value at a point comes from average of
        // 2 magnetic field component values in that cell and the following cell
        snapHx[INDEX4D_SNAPS(p,i,j,k)] = (Hx[INDEX3D_FIELDS(ii,jj,kk)] + Hx[INDEX3D_FIELDS(ii+1,jj,kk)]) / 2;
        snapHy[INDEX4D_SNAPS(p,i,j,k)] = (Hy[INDEX3D_FIELDS(ii,jj,kk)] + Hy[INDEX3D_FIELDS(ii,jj+1,kk)]) / 2;
        snapHz[INDEX4D_SNAPS(p,i,j,k)] = (Hz[INDEX3D_FIELDS(ii,jj,kk)] + Hz[INDEX3D_FIELDS(ii,jj,kk+1)]) / 2;
    }
}