from string import Template

kernel_fields_fluxes_gpu = Template("""

#include <pycuda-complex.hpp>
#include <math.h>                         
#include <math_constants.h>

// Macros for converting subscripts to linear index:
#define INDEX3D_FIELDS(i, j, k) (i)*($NY_FIELDS)*($NZ_FIELDS)+(j)*($NZ_FIELDS)+(k)
#define INDEX5D_FFT(f, i, j, k, c) (f)*($NX_FLUX)*($NY_FLUX)*($NZ_FLUX)*($NC)+(i)*($NY_FLUX)*($NZ_FLUX)*($NC)+(j)*($NZ_FLUX)*($NC)+(k)*($NC)+(c)
                     
__device__ __constant__ $REAL $OMEGA[$NF];

////////////////////////////////////
// Saving fft transform of fields //
////////////////////////////////////

__global__ void save_fields_flux(int NF, int NX, int NY, int NZ,
                                 int x_begin, int y_begin, int z_begin,
                                 $REAL *Ex, $REAL *Ey, $REAL *Ez, $REAL *Hx, $REAL *Hy, $REAL *Hz,
                                 $COMPLEX *FFT_E, $COMPLEX *FFT_H, int iteration, $REAL dt) {

    //  This function computes fft of the fields for the wavelengths in wavelengths.
    //
    //  Args:
    //      NF: Number of wavelengths
    //      NX, NY, NZ: Number of cells of the model domain
    //      E, H: Access to field component arrays
                                   
    // Obtain the linear index corresponding to the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                                    
    //printf("idx= %d", idx);
    //printf("x_begin= %d, y_begin= %d, z_begin= %d \\n", $x_begin, $y_begin, $z_begin);

    assert($NX_FLUX == NX);
    assert($NY_FLUX == NY);
    assert($NZ_FLUX == NZ);

    // FFT indices
    int i_fft = idx / (NY * NZ);
    int j_fft = (idx % (NY * NZ)) / NZ;
    int k_fft = (idx % (NY * NZ)) % NZ;
                                    
    //printf("x_begin= %d, y_begin= %d, z_begin= %d \\n", x_begin, y_begin, z_begin);

                                    
    // Convert the linear index to subscripts for 3D field arrays
    int i = i_fft + x_begin;
    //printf("i= %d\\n", i);
    int j = j_fft + y_begin;
    //printf("j= %d\\n", j);
    int k = k_fft + z_begin;
    //printf("k= %d\\n", k);
                                    
    //printf("i= %d, j= %d, k= %d \\n", i, j, k);
       
                                    
    float $SIN_OMEGA, $COS_OMEGA;
    int $IDX_5DX, $IDX_5DY, $IDX_5DZ;

    if (i_fft >= NX || j_fft >= NY || k_fft >= NZ) return;

    //printf("\\n");
                                    
    int $IDX_3D = INDEX3D_FIELDS(i, j, k);
                                    
    //printf("NX_FIELDS = $NX_FIELDS, NY_FIEDS= $NY_FIELDS, NZ_FIELDS= $NZ_FIELDS \\n");
    //printf("NX_FLUX = %d, NY_FLUX= %d, NZ_FLUX= %d \\n", NX, NY, NZ);
                                    
    //printf("i_fft= %d, j_fft= %d, k_fft= %d ---- INDEX_3D= %d \\n", i_fft, j_fft, k_fft, idx_3d);

                                    
    float $NORM = dt / sqrt(2 * CUDART_PI);
                                    
    for (int f = 0; f < NF; f++){
                                    
        $REAL $PHASE = $OMEGA[f]*iteration*dt;
                                          
        sincosf($PHASE, &$SIN_OMEGA, &$COS_OMEGA);      
        $COMPLEX $EXP_FACTOR($REAL($COS_OMEGA * $NORM), $REAL($SIN_OMEGA * $NORM));     
        //printf("%e", $EXP_FACTOR.real());             

        $IDX_5DX = INDEX5D_FFT(f,i_fft,j_fft,k_fft, 0);
        $IDX_5DY = INDEX5D_FFT(f,i_fft,j_fft,k_fft, 1);
        $IDX_5DZ = INDEX5D_FFT(f,i_fft,j_fft,k_fft, 2);   
                                    
    //    printf("i_fft= %d, j_fft= %d, k_fft= %d ---- INDEX_5D_x= %d ---- INDEX_5D_y= %d ---- INDEX_5D_z= %d \\n", i_fft, j_fft, k_fft, $IDX_5DX, $IDX_5DY, $IDX_5DZ);
    //    printf("idx_3d= %d", $IDX_3D);
    //    printf("%e", Ex[$IDX_3D]);
    
        FFT_E[$IDX_5DX] += Ex[$IDX_3D] * $EXP_FACTOR;  
        FFT_E[$IDX_5DY] += Ey[$IDX_3D] * $EXP_FACTOR;
        FFT_E[$IDX_5DZ] += Ez[$IDX_3D] * $EXP_FACTOR;
                                    
        FFT_H[$IDX_5DX] += Hx[$IDX_3D] * $EXP_FACTOR;
        FFT_H[$IDX_5DY] += Hy[$IDX_3D] * $EXP_FACTOR;
        FFT_H[$IDX_5DZ] += Hz[$IDX_3D] * $EXP_FACTOR;
        if (f == 0 && i_fft ==  0 && j_fft == 0 && k_fft == 0 && iteration == 500){                               
            printf("Ex= %e \\n", Ex[$IDX_3D]);
            printf("Ey= %e \\n", Ey[$IDX_3D]);
            printf("Ez= %e \\n", Ez[$IDX_3D]);                              
            printf("Hx= %e \\n", Hx[$IDX_3D]);
            printf("Hy= %e \\n", Hy[$IDX_3D]);
            printf("Hz= %e \\n", Hz[$IDX_3D]);}
        }
    }
""")