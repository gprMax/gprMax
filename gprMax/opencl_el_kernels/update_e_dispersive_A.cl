//  This function is part A of updates to electric field values when dispersive materials (with multiple poles) are present.
//
//  Args:
//      NX, NY, NZ: Number of cells of the model domain
//      MAXPOLES: Maximum number of dispersive material poles present in model
//      updatedispersivecoeffs, T, ID, E, H: Access to update coefficients, dispersive, ID and field component arrays

// Convert the linear index to subscripts for 3D field arrays
int x = i / ({{NY_FIELDS}} * {{NZ_FIELDS}});
int y = (i % ({{NY_FIELDS}} * {{NZ_FIELDS}})) / {{NZ_FIELDS}};
int z = (i % ({{NY_FIELDS}} * {{NZ_FIELDS}})) % {{NZ_FIELDS}};

// Convert the linear index to subscripts for 4D material ID array
int i_ID = (i % ({{NX_ID}} * {{NY_ID}} * {{NZ_ID}})) / ({{NY_ID}} * {{NZ_ID}});
int j_ID = ((i % ({{NX_ID}} * {{NY_ID}} * {{NZ_ID}})) % ({{NY_ID}} * {{NZ_ID}})) / {{NZ_ID}};
int k_ID = ((i % ({{NX_ID}} * {{NY_ID}} * {{NZ_ID}})) % ({{NY_ID}} * {{NZ_ID}})) % {{NZ_ID}};

// Convert the linear index to subscripts for 4D dispersive array
int i_T = (i % ({{NX_T}} * {{NY_T}} * {{NZ_T}})) / ({{NY_T}} * {{NZ_T}});
int j_T = ((i % ({{NX_T}} * {{NY_T}} * {{NZ_T}})) % ({{NY_T}} * {{NZ_T}})) / {{NZ_T}};
int k_T = ((i % ({{NX_T}} * {{NY_T}} * {{NZ_T}})) % ({{NY_T}} * {{NZ_T}})) % {{NZ_T}};

// Ex component
if ((NY != 1 || NZ != 1) && x >= 0 && x < NX && y > 0 && y < NY && z > 0 && z < NZ) {
    int materialEx = ID[INDEX4D_ID(0,i_ID,j_ID,k_ID)];
    {{REAL}} phi = 0;
    for (int pole = 0; pole < MAXPOLES; pole++) {
        phi = phi + updatecoeffsdispersive[INDEX2D_MATDISP(materialEx,pole*3)].real * Tx[INDEX4D_T(pole,i_T,j_T,k_T)].real;
        Tx[INDEX4D_T(pole,i_T,j_T,k_T)] = cfloat_add(cfloat_mul(updatecoeffsdispersive[INDEX2D_MATDISP(materialEx,1+(pole*3))], Tx[INDEX4D_T(pole,i_T,j_T,k_T)]), cfloat_mulr(updatecoeffsdispersive[INDEX2D_MATDISP(materialEx,2+(pole*3))], Ex[INDEX3D_FIELDS(x,y,z)]));
    }
    Ex[INDEX3D_FIELDS(x,y,z)] = updatecoeffsE[INDEX2D_MAT(materialEx,0)] * Ex[INDEX3D_FIELDS(x,y,z)] + updatecoeffsE[INDEX2D_MAT(materialEx,2)] * (Hz[INDEX3D_FIELDS(x,y,z)] - Hz[INDEX3D_FIELDS(x,y-1,z)]) - updatecoeffsE[INDEX2D_MAT(materialEx,3)] * (Hy[INDEX3D_FIELDS(x,y,z)] - Hy[INDEX3D_FIELDS(x,y,z-1)]) - updatecoeffsE[INDEX2D_MAT(materialEx,4)] * phi;
}

// Ey component
if ((NX != 1 || NZ != 1) && x > 0 && x < NX && y >= 0 && y < NY && z > 0 && z < NZ) {
    int materialEy = ID[INDEX4D_ID(1,i_ID,j_ID,k_ID)];
    {{REAL}} phi = 0;
    for (int pole = 0; pole < MAXPOLES; pole++) {
        phi = phi + updatecoeffsdispersive[INDEX2D_MATDISP(materialEy,pole*3)].real * Ty[INDEX4D_T(pole,i_T,j_T,k_T)].real;
        Ty[INDEX4D_T(pole,i_T,j_T,k_T)] = cfloat_add(cfloat_mul(updatecoeffsdispersive[INDEX2D_MATDISP(materialEy,1+(pole*3))], Ty[INDEX4D_T(pole,i_T,j_T,k_T)]), cfloat_mulr(updatecoeffsdispersive[INDEX2D_MATDISP(materialEy,2+(pole*3))], Ey[INDEX3D_FIELDS(x,y,z)]));
    }
    Ey[INDEX3D_FIELDS(x,y,z)] = updatecoeffsE[INDEX2D_MAT(materialEy,0)] * Ey[INDEX3D_FIELDS(x,y,z)] + updatecoeffsE[INDEX2D_MAT(materialEy,3)] * (Hx[INDEX3D_FIELDS(x,y,z)] - Hx[INDEX3D_FIELDS(x,y,z-1)]) - updatecoeffsE[INDEX2D_MAT(materialEy,1)] * (Hz[INDEX3D_FIELDS(x,y,z)] - Hz[INDEX3D_FIELDS(x-1,y,z)]) - updatecoeffsE[INDEX2D_MAT(materialEy,4)] * phi;
}

// Ez component
if ((NX != 1 || NY != 1) && x > 0 && x < NX && y > 0 && y < NY && z >= 0 && z < NZ) {
    int materialEz = ID[INDEX4D_ID(2,i_ID,j_ID,k_ID)];
    {{REAL}} phi = 0;
    for (int pole = 0; pole < MAXPOLES; pole++) {
        phi = phi + updatecoeffsdispersive[INDEX2D_MATDISP(materialEz,pole*3)].real * Tz[INDEX4D_T(pole,i_T,j_T,k_T)].real;
        Tz[INDEX4D_T(pole,i_T,j_T,k_T)] = cfloat_add(cfloat_mul(updatecoeffsdispersive[INDEX2D_MATDISP(materialEz,1+(pole*3))], Tz[INDEX4D_T(pole,i_T,j_T,k_T)]), cfloat_mulr(updatecoeffsdispersive[INDEX2D_MATDISP(materialEz,2+(pole*3))], Ez[INDEX3D_FIELDS(x,y,z)]));
    }
    Ez[INDEX3D_FIELDS(x,y,z)] = updatecoeffsE[INDEX2D_MAT(materialEz,0)] * Ez[INDEX3D_FIELDS(x,y,z)] + updatecoeffsE[INDEX2D_MAT(materialEz,1)] * (Hy[INDEX3D_FIELDS(x,y,z)] - Hy[INDEX3D_FIELDS(x-1,y,z)]) - updatecoeffsE[INDEX2D_MAT(materialEz,2)] * (Hx[INDEX3D_FIELDS(x,y,z)] - Hx[INDEX3D_FIELDS(x,y-1,z)]) - updatecoeffsE[INDEX2D_MAT(materialEz,4)] * phi;
}    