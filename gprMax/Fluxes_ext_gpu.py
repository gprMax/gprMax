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
                                 const $REAL *Ex, const $REAL *Ey, const $REAL *Ez, const $REAL *Hx, const $REAL *Hy, const $REAL *Hz,
                                 $COMPLEX *FFT_E, $COMPLEX *FFT_H, int iteration, $REAL dt) {

    //  This function computes fft of the fields for the wavelengths in wavelengths.
    //
    //  Args:
    //      NF: Number of wavelengths
    //      NX, NY, NZ: Number of cells of the model domain
    //      E, H: Access to field component arrays
                                   
    // Obtain the linear index corresponding to the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

                                    
    // FFT indices
    int i_fft = idx / (NY * NZ);
    int j_fft = (idx % (NY * NZ)) / NZ;
    int k_fft = (idx % (NY * NZ)) % NZ;

                                    
    // Convert the linear index to subscripts for 3D field arrays
    int i = i_fft + x_begin;
    int j = j_fft + y_begin;
    int k = k_fft + z_begin;
                                    
                                    
    float $SIN_OMEGA, $COS_OMEGA;
    int $IDX_5DX, $IDX_5DY, $IDX_5DZ;

    if (i_fft >= NX || j_fft >= NY || k_fft >= NZ) return;
                                    
    int $IDX_3D = INDEX3D_FIELDS(i, j, k);

                                    
    float $NORM = dt / sqrt(2 * CUDART_PI);
                                    
    for (int f = 0; f < NF; f++){
                                    
        $REAL $PHASE = $OMEGA[f]*iteration*dt;
                                          
        sincosf($PHASE, &$SIN_OMEGA, &$COS_OMEGA);      
        $COMPLEX $EXP_FACTOR($REAL($COS_OMEGA * $NORM), $REAL($SIN_OMEGA * $NORM));        

        $IDX_5DX = INDEX5D_FFT(f,i_fft,j_fft,k_fft, 0);
        $IDX_5DY = INDEX5D_FFT(f,i_fft,j_fft,k_fft, 1);
        $IDX_5DZ = INDEX5D_FFT(f,i_fft,j_fft,k_fft, 2);
    
        FFT_E[$IDX_5DX] += Ex[$IDX_3D] * $EXP_FACTOR;  
        FFT_E[$IDX_5DY] += Ey[$IDX_3D] * $EXP_FACTOR;
        FFT_E[$IDX_5DZ] += Ez[$IDX_3D] * $EXP_FACTOR;
                                    
        FFT_H[$IDX_5DX] += Hx[$IDX_3D] * $EXP_FACTOR;
        FFT_H[$IDX_5DY] += Hy[$IDX_3D] * $EXP_FACTOR;
        FFT_H[$IDX_5DZ] += Hz[$IDX_3D] * $EXP_FACTOR;
    }
}
""")