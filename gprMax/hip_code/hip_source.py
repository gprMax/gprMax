from string import Template
vec_scale = b"""\
    extern "C" __global__ void scale_vector(float factor, int n, short unused1, int unused2, float unused3, float *x) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if ( tid == 0 ) {
        printf("tid: %d, factor: %f, x*: %lu, n: %lu, unused1: %d, unused2: %d, unused3: %f\\n",tid,factor,x,n,(int) unused1,unused2,unused3);
    }
    if (tid < n) {
        x[tid] *= factor;
    }
    }
    """

update_e =  Template("""
    // Macros for converting subscripts to linear index:
    #define INDEX2D_MAT(m, n) (m)*($NY_MATCOEFFS)+(n)
    #define INDEX2D_MATDISP(m, n) (m)*($NY_MATDISPCOEFFS)+(n)
    #define INDEX3D_FIELDS(i, j, k) (i)*($NY_FIELDS)*($NZ_FIELDS)+(j)*($NZ_FIELDS)+(k)
    #define INDEX4D_ID(p, i, j, k) (p)*($NX_ID)*($NY_ID)*($NZ_ID)+(i)*($NY_ID)*($NZ_ID)+(j)*($NZ_ID)+(k)
    #define INDEX4D_T(p, i, j, k) (p)*($NX_T)*($NY_T)*($NZ_T)+(i)*($NY_T)*($NZ_T)+(j)*($NZ_T)+(k)

    // Material coefficients (read-only) in constant memory (64KB)_
    __device__ __constant__ $REAL updatecoeffsE[$N_updatecoeffsE];
    __device__ __constant__ $REAL updatecoeffsH[$N_updatecoeffsH];

    /////////////////////////////////////////////////
    // Electric field updates - standard materials //
    /////////////////////////////////////////////////

    extern "C" __global__ void update_e(int NX, int NY, int NZ, const unsigned int* __restrict__ ID, $REAL *Ex, $REAL *Ey, $REAL *Ez, const $REAL* __restrict__ Hx, const $REAL* __restrict__ Hy, const $REAL* __restrict__ Hz) {

        //  This function updates electric field values.
        //
        //  Args:
        //      NX, NY, NZ: Number of cells of the model domain
        //      ID, E, H: Access to ID and field component arrays

        // Obtain the linear index corresponding to the current thread
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // Convert the linear index to subscripts for 3D field arrays
        int i = idx / ($NY_FIELDS * $NZ_FIELDS);
        int j = (idx % ($NY_FIELDS * $NZ_FIELDS)) / $NZ_FIELDS;
        int k = (idx % ($NY_FIELDS * $NZ_FIELDS)) % $NZ_FIELDS;

        // Convert the linear index to subscripts for 4D material ID array
        int i_ID = (idx % ($NX_ID * $NY_ID * $NZ_ID)) / ($NY_ID * $NZ_ID);
        int j_ID = ((idx % ($NX_ID * $NY_ID * $NZ_ID)) % ($NY_ID * $NZ_ID)) / $NZ_ID;
        int k_ID = ((idx % ($NX_ID * $NY_ID * $NZ_ID)) % ($NY_ID * $NZ_ID)) % $NZ_ID;

        // Ex component
        if ((NY != 1 || NZ != 1) && i >= 0 && i < NX && j > 0 && j < NY && k > 0 && k < NZ) {
            int materialEx = ID[INDEX4D_ID(0,i_ID,j_ID,k_ID)];
            Ex[INDEX3D_FIELDS(i,j,k)] = updatecoeffsE[INDEX2D_MAT(materialEx,0)] * Ex[INDEX3D_FIELDS(i,j,k)] + updatecoeffsE[INDEX2D_MAT(materialEx,2)] * (Hz[INDEX3D_FIELDS(i,j,k)] - Hz[INDEX3D_FIELDS(i,j-1,k)]) - updatecoeffsE[INDEX2D_MAT(materialEx,3)] * (Hy[INDEX3D_FIELDS(i,j,k)] - Hy[INDEX3D_FIELDS(i,j,k-1)]);
        }

        // Ey component
        if ((NX != 1 || NZ != 1) && i > 0 && i < NX && j >= 0 && j < NY && k > 0 && k < NZ) {
            int materialEy = ID[INDEX4D_ID(1,i_ID,j_ID,k_ID)];
            Ey[INDEX3D_FIELDS(i,j,k)] = updatecoeffsE[INDEX2D_MAT(materialEy,0)] * Ey[INDEX3D_FIELDS(i,j,k)] + updatecoeffsE[INDEX2D_MAT(materialEy,3)] * (Hx[INDEX3D_FIELDS(i,j,k)] - Hx[INDEX3D_FIELDS(i,j,k-1)]) - updatecoeffsE[INDEX2D_MAT(materialEy,1)] * (Hz[INDEX3D_FIELDS(i,j,k)] - Hz[INDEX3D_FIELDS(i-1,j,k)]);
        }

        // Ez component
        if ((NX != 1 || NY != 1) && i > 0 && i < NX && j > 0 && j < NY && k >= 0 && k < NZ) {
            int materialEz = ID[INDEX4D_ID(2,i_ID,j_ID,k_ID)];
            Ez[INDEX3D_FIELDS(i,j,k)] = updatecoeffsE[INDEX2D_MAT(materialEz,0)] * Ez[INDEX3D_FIELDS(i,j,k)] + updatecoeffsE[INDEX2D_MAT(materialEz,1)] * (Hy[INDEX3D_FIELDS(i,j,k)] - Hy[INDEX3D_FIELDS(i-1,j,k)]) - updatecoeffsE[INDEX2D_MAT(materialEz,2)] * (Hx[INDEX3D_FIELDS(i,j,k)] - Hx[INDEX3D_FIELDS(i,j-1,k)]);
        }
    }
    """)
