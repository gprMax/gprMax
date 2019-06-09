{#
// kernel for updating the sources
#}

// Macros for converting subscripts to linear index:
#define INDEX2D_MAT(m, n) (m)*({{NY_MATCOEFFS}}) + (n)
#define INDEX2D_SRCINFO(m, n) (m)*({{NY_SRCINFO}}) + (n)
#define INDEX2D_SRCWAVES(m, n) (m)*({{NY_SRCWAVES}}) + (n)
#define INDEX3D_FIELDS(i, j, k) (i)*({{NY_FIELDS}})*({{NZ_FIELDS}}) + (j)*({{NZ_FIELDS}}) + (k)
#define INDEX4D_ID(p, i, j, k) (p)*({{NX_ID}})*({{NY_ID}})*({{NZ_ID}}) + (i)*({{NY_ID}})*({{NZ_ID}}) + (j)*({{NZ_ID}}) + (k)

// material update coefficients to be declared in constant memory
__constant {{REAL}} updatecoeffsE[{{N_updatecoeffsE}}] = {0.0};
__constant {{REAL}} updatecoeffsH[{{N_updatecoeffsH}}] = {0.0};

// kernel to update the coefficients
__kernel void setUpdateCoeffs(__constant {{REAL}} * updatecoeffsE, __constant {{REAL}} * updatecoeffsH){
    // do nothing else
}


// simplest source which is Hertizan Dipole will be used
__kernel void update_hertzian_dipole(int NHERTZDIPOLE, int iteration, {{REAL}} dx, {{REAL}} dy, {{REAL}} dz, __global const int* restrict srcinfo1, __global const {{REAL}}* restrict srcinfo2, __global const {{REAL}}* restrict srcwaveforms, __global const unsigned int* restrict ID, __global {{REAL}} *Ex, __global {{REAL}} *Ey, __global {{REAL}} *Ez){
    // updates electric field values for Hertizan Dipole Source
    // Args:
    //     NHERTZDIPOLE: total number of hertizan dipole in the model
    //     iteration
    //     dx, dy, dz: spatial discretization
    //     srcinfo1: source cell coordinates and polarisation information
    //     srcinfo2: other source info, length, resistance, etc
    //     srcwaveforms : source waveforms values
    //     ID, E: access to ID and field component values 

    // get linear index 
    int src = get_global_id(0); 

    if (src < NHERTZDIPOLE) {

        {{REAL}} dl;
        int i, j, k, polarisation;

        i = srcinfo1[INDEX2D_SRCINFO(src,0)];
        j = srcinfo1[INDEX2D_SRCINFO(src,1)];
        k = srcinfo1[INDEX2D_SRCINFO(src,2)];
        polarisation = srcinfo1[INDEX2D_SRCINFO(src,3)];
        dl = srcinfo2[src];

        // 'x' polarised source
        if (polarisation == 0) {
            int materialEx = ID[INDEX4D_ID(0,i,j,k)];
            Ex[INDEX3D_FIELDS(i,j,k)] = Ex[INDEX3D_FIELDS(i,j,k)] - updatecoeffsE[INDEX2D_MAT(materialEx,4)] * srcwaveforms[INDEX2D_SRCWAVES(src,iteration)] * dl * (1 / (dx * dy * dz));
        }

        // 'y' polarised source
        else if (polarisation == 1) {
            int materialEy = ID[INDEX4D_ID(1,i,j,k)];
            Ey[INDEX3D_FIELDS(i,j,k)] = Ey[INDEX3D_FIELDS(i,j,k)] - updatecoeffsE[INDEX2D_MAT(materialEy,4)] * srcwaveforms[INDEX2D_SRCWAVES(src,iteration)] * dl * (1 / (dx * dy * dz));
        }

        // 'z' polarised source
        else if (polarisation == 2) {
            int materialEz = ID[INDEX4D_ID(2,i,j,k)];
            Ez[INDEX3D_FIELDS(i,j,k)] = Ez[INDEX3D_FIELDS(i,j,k)] - updatecoeffsE[INDEX2D_MAT(materialEz,4)] * srcwaveforms[INDEX2D_SRCWAVES(src,iteration)] * dl * (1 / (dx * dy * dz));
        }
}