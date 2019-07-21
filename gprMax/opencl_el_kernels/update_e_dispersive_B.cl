//  This function is part B which updates the dispersive field arrays when dispersive materials (with multiple poles) are present.
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
    for (int pole = 0; pole < MAXPOLES; pole++) {
        Tx[INDEX4D_T(pole,i_T,j_T,k_T)] = cfloat_sub(Tx[INDEX4D_T(pole,i_T,j_T,k_T)], cfloat_mulr(updatecoeffsdispersive[INDEX2D_MATDISP(materialEx,2+(pole*3))], Ex[INDEX3D_FIELDS(x,y,z)]));
    }
}

// Ey component
if ((NX != 1 || NZ != 1) && x > 0 && x < NX && y >= 0 && y < NY && z > 0 && z < NZ) {
    int materialEy = ID[INDEX4D_ID(1,i_ID,j_ID,k_ID)];
    for (int pole = 0; pole < MAXPOLES; pole++) {
        Ty[INDEX4D_T(pole,i_T,j_T,k_T)] = cfloat_sub(Ty[INDEX4D_T(pole,i_T,j_T,k_T)], cfloat_mulr(updatecoeffsdispersive[INDEX2D_MATDISP(materialEy,2+(pole*3))], Ey[INDEX3D_FIELDS(x,y,z)]));
    }
}

// Ez component
if ((NX != 1 || NY != 1) && x > 0 && x < NX && y > 0 && y < NY && z >= 0 && z < NZ) {
    int materialEz = ID[INDEX4D_ID(2,i_ID,j_ID,k_ID)];
    for (int pole = 0; pole < MAXPOLES; pole++) {
        Tz[INDEX4D_T(pole,i_T,j_T,k_T)] = cfloat_sub(Tz[INDEX4D_T(pole,i_T,j_T,k_T)], cfloat_mulr(updatecoeffsdispersive[INDEX2D_MATDISP(materialEz,2+(pole*3))], Ez[INDEX3D_FIELDS(x,y,z)]));
    }
}