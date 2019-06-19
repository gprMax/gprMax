// convert the linear index to subscripts for 3D field arrays
int x = i / ({{NY_FIELDS}} * {{NZ_FIELDS}});
int y = (i % ({{NY_FIELDS}}*{{NZ_FIELDS}})) / {{NZ_FIELDS}};
int z = (i % ({{NY_FIELDS}}*{{NZ_FIELDS}})) % {{NZ_FIELDS}};

//convert the linear index to subscripts for 4D material ID arrays
int x_ID = (i%({{NX_ID}} * {{NY_ID}} * {{NZ_ID}})) / ({{NY_ID}} * {{NZ_ID}});
int y_ID = ((i%({{NX_ID}} * {{NY_ID}} * {{NZ_ID}})) % ({{NY_ID}} * {{NZ_ID}})) / {{NZ_ID}};
int z_ID = ((i%({{NX_ID}} * {{NY_ID}} * {{NZ_ID}})) % ({{NY_ID}} * {{NZ_ID}})) % {{NZ_ID}};

// Ex component
if ((NY != 1 || NZ != 1) && x >= 0 && x < NX && y > 0 && y < NY && z > 0 && z < NZ) {
    int materialEx = ID[INDEX4D_ID(0,x_ID,y_ID,z_ID)];
    Ex[INDEX3D_FIELDS(x,y,z)] = updatecoeffsE[INDEX2D_MAT(materialEx,0)] * Ex[INDEX3D_FIELDS(x,y,z)] + updatecoeffsE[INDEX2D_MAT(materialEx,2)] * (Hz[INDEX3D_FIELDS(x,y,z)] - Hz[INDEX3D_FIELDS(x,y-1,z)]) - updatecoeffsE[INDEX2D_MAT(materialEx,3)] * (Hy[INDEX3D_FIELDS(x,y,z)] - Hy[INDEX3D_FIELDS(x,y,z-1)]);
}

// Ey component
if ((NX != 1 || NZ != 1) && x > 0 && x < NX && y >= 0 && y < NY && z > 0 && z < NZ) {
    int materialEy = ID[INDEX4D_ID(1,x_ID,y_ID,z_ID)];
    Ey[INDEX3D_FIELDS(x,y,z)] = updatecoeffsE[INDEX2D_MAT(materialEy,0)] * Ey[INDEX3D_FIELDS(x,y,z)] + updatecoeffsE[INDEX2D_MAT(materialEy,3)] * (Hx[INDEX3D_FIELDS(x,y,z)] - Hx[INDEX3D_FIELDS(x,y,z-1)]) - updatecoeffsE[INDEX2D_MAT(materialEy,1)] * (Hz[INDEX3D_FIELDS(x,y,z)] - Hz[INDEX3D_FIELDS(x-1,y,z)]);
}

// Ez component
if ((NX != 1 || NY != 1) && x > 0 && x < NX && y > 0 && y < NY && z >= 0 && z < NZ) {
    int materialEz = ID[INDEX4D_ID(2,x_ID,y_ID,z_ID)];
    Ez[INDEX3D_FIELDS(x,y,z)] = updatecoeffsE[INDEX2D_MAT(materialEz,0)] * Ez[INDEX3D_FIELDS(x,y,z)] + updatecoeffsE[INDEX2D_MAT(materialEz,1)] * (Hy[INDEX3D_FIELDS(x,y,z)] - Hy[INDEX3D_FIELDS(x-1,y,z)]) - updatecoeffsE[INDEX2D_MAT(materialEz,2)] * (Hx[INDEX3D_FIELDS(x,y,z)] - Hx[INDEX3D_FIELDS(x,y-1,z)]);
}