from string import Template

kernels_template_fields = Template("""

#include <pycuda-complex.hpp>
#include <math.h>                         
#include <math_constants.h>

// Macros for converting subscripts to linear index:
#define INDEX3D_FIELDS(i, j, k) (i)*($NY_FIELDS)*($NZ_FIELDS)+(j)*($NZ_FIELDS)+(k)
#define INDEX5D_FFT(f, i, j, k, c) (f)*($NX_FIELDS)*($NY_FIELDS)*($NZ_FIELDS)*($NC)+(i)*($NY_FIELDS)*($NZ_FIELDS)*($NC)+(j)*($NZ_FIELDS)*($NC)+(k)*($NC)+(c)
                       

////////////////////////////////////
// Saving fft transform of fields //
////////////////////////////////////

__global__ void save_fields_flux(int NF, int NX, int NY, int NZ, $COMPLEX *Ex, $COMPLEX *Ey, $COMPLEX *Ez, $COMPLEX *Hx, $COMPLEX *Hy, $COMPLEX *Hz,
                                   $REAL *wavelength, $COMPLEX *FFT_E, $COMPLEX *FFT_H, int iteration, $REAL dt) {

    //  This function computes fft of the fields for the wavelengths in wavelengths.
    //
    //  Args:
    //      NF: Number of wavelengths
    //      NX, NY, NZ: Number of cells of the model domain
    //      E, H: Access to field component arrays
                                   
    // Obtain the linear index corresponding to the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for 3D field arrays
    int i = idx / ($NY_FIELDS * $NZ_FIELDS);
    int j = (idx % ($NY_FIELDS * $NZ_FIELDS)) / $NZ_FIELDS;
    int k = (idx % ($NY_FIELDS * $NZ_FIELDS)) % $NZ_FIELDS;
    
    // Convert the linear index to subscripts for 4D field arrays      
    int i_fft = (idx % ($NX_FIELDS * $NY_FIELDS * $NZ_FIELDS * $NC)) / ($NY_FIELDS * $NZ_FIELDS * $NC);
    int j_fft = ((idx % ($NX_FIELDS * $NY_FIELDS * $NZ_FIELDS * $NC)) % ($NY_FIELDS * $NZ_FIELDS * $NC)) / ($NZ_FIELDS * $NC);
    int k_fft = (((idx % ($NX_FIELDS * $NY_FIELDS * $NZ_FIELDS * $NC)) % ($NY_FIELDS * $NZ_FIELDS * $NC)) % ($NZ_FIELDS * $NC)) / $NC;

    double sin_omega, cos_omega;
    int idx_5dx, idx_5dy, idx_5dz;                                                                           
    double pi = CUDART_PI; 

    if (i >= 0 && i < NX && j > 0 && j < NY && k > 0 && k < NZ) {
        int idx_3d = INDEX3D_FIELDS(i, j, k);
        for (int f = 0; f < NF; f++){
            sincos(wavelength[f]*iteration*dt, &sin_omega, &cos_omega);
            idx_5dx = INDEX5D_FFT(f,i_fft,j_fft,k_fft, 0);
            FFT_E[idx_5dx].x += (Ex[idx_3d].x * cos_omega - Ex[idx_3d].y * sin_omega) * dt / sqrt(2 * pi);
            FFT_E[idx_5dx].y += (Ex[idx_3d].y * sin_omega + Ex[idx_3d].x * cos_omega) * dt / sqrt(2 * pi);
                                   
            idx_5dy = INDEX5D_FFT(f,i_fft,j_fft,k_fft, 1);
            FFT_E[idx_5dy].x += (Ey[idx_3d].x * cos_omega - Ey[idx_3d].y * sin_omega) * dt / sqrt(2 * pi);
            FFT_E[idx_5dy].y += (Ey[idx_3d].y * sin_omega + Ey[idx_3d].x * cos_omega) * dt / sqrt(2 * pi);
                                   
            idx_5dz = INDEX5D_FFT(f,i_fft,j_fft,k_fft, 2);
            FFT_E[idx_5dz].x += (Ez[idx_3d].x * cos_omega - Ez[idx_3d].y * sin_omega) * dt / sqrt(2 * pi);
            FFT_E[idx_5dz].y += (Ez[idx_3d].y * sin_omega + Ez[idx_3d].x * cos_omega) * dt / sqrt(2 * pi);
                                   
                                   
            FFT_H[idx_5dx].x += (Hx[idx_3d].x * cos_omega - Hx[idx_3d].y * sin_omega) * dt / sqrt(2 * pi);
            FFT_H[idx_5dx].y += (Hx[idx_3d].y * sin_omega + Hx[idx_3d].x * cos_omega) * dt / sqrt(2 * pi);
                                   
            FFT_H[idx_5dy].x += (Hy[idx_3d].x * cos_omega - Hy[idx_3d].y * sin_omega) * dt / sqrt(2 * pi);
            FFT_H[idx_5dy].y += (Hy[idx_3d].y * sin_omega + Hy[idx_3d].x * cos_omega) * dt / sqrt(2 * pi);
                                   
            FFT_H[idx_5dz].x += (Hz[idx_3d].x * cos_omega - Hz[idx_3d].y * sin_omega) * dt / sqrt(2 * pi);
            FFT_H[idx_5dz].y += (Hz[idx_3d].y * sin_omega + Hz[idx_3d].x * cos_omega) * dt / sqrt(2 * pi);
        }
    }

                                   """)