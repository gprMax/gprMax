int x = i / ({{NY_FIELDS}} * {{NZ_FIELDS}});
int y = (i%({{NY_FIELDS}}*{{NZ_FIELDS}})) / {{NZ_FIELDS}};
int z = (i%({{NY_FIELDS}}*{{NZ_FIELDS}})) % {{NZ_FIELDS}};

// convert the linear index to subscripts to 4D material ID arrays
int x_ID = ( i % ({{NX_ID}} * {{NY_ID}} * {{NZ_ID}})) / ({{NY_ID}} * {{NZ_ID}});
int y_ID = (( i % ({{NX_ID}} * {{NY_ID}} * {{NZ_ID}})) % ({{NY_ID}} * {{NZ_ID}})) / {{NZ_ID}};
int z_ID = (( i % ({{NX_ID}} * {{NY_ID}} * {{NZ_ID}})) % ({{NY_ID}} * {{NZ_ID}})) % {{NZ_ID}};

// Hx component
if (NX != 1 && x > 0 && x < NX && y >= 0 && y < NY && z >= 0 && z < NZ) {
    int materialHx = ID[INDEX4D_ID(3,x_ID,y_ID,z_ID)];
    Hx[INDEX3D_FIELDS(x,y,z)] = updatecoeffsH[INDEX2D_MAT(materialHx,0)] * Hx[INDEX3D_FIELDS(x,y,z)] - updatecoeffsH[INDEX2D_MAT(materialHx,2)] * (Ez[INDEX3D_FIELDS(x,y+1,z)] - Ez[INDEX3D_FIELDS(x,y,z)]) + updatecoeffsH[INDEX2D_MAT(materialHx,3)] * (Ey[INDEX3D_FIELDS(x,y,z+1)] - Ey[INDEX3D_FIELDS(x,y,z)]);
}

// Hy component
if (NY != 1 && x >= 0 && x < NX && y > 0 && y < NY && z >= 0 && z < NZ) {
    int materialHy = ID[INDEX4D_ID(4,x_ID,y_ID,z_ID)];
    Hy[INDEX3D_FIELDS(x,y,z)] = updatecoeffsH[INDEX2D_MAT(materialHy,0)] * Hy[INDEX3D_FIELDS(x,y,z)] - updatecoeffsH[INDEX2D_MAT(materialHy,3)] * (Ex[INDEX3D_FIELDS(x,y,z+1)] - Ex[INDEX3D_FIELDS(x,y,z)]) + updatecoeffsH[INDEX2D_MAT(materialHy,1)] * (Ez[INDEX3D_FIELDS(x+1,y,z)] - Ez[INDEX3D_FIELDS(x,y,z)]);
}

// Hz component
if (NZ != 1 && x >= 0 && x < NX && y >= 0 && y < NY && z > 0 && z < NZ) {
    int materialHz = ID[INDEX4D_ID(5,x_ID,y_ID,z_ID)];
    Hz[INDEX3D_FIELDS(x,y,z)] = updatecoeffsH[INDEX2D_MAT(materialHz,0)] * Hz[INDEX3D_FIELDS(x,y,z)] - updatecoeffsH[INDEX2D_MAT(materialHz,1)] * (Ey[INDEX3D_FIELDS(x+1,y,z)] - Ey[INDEX3D_FIELDS(x,y,z)]) + updatecoeffsH[INDEX2D_MAT(materialHz,2)] * (Ex[INDEX3D_FIELDS(x,y+1,z)] - Ex[INDEX3D_FIELDS(x,y,z)]);
}