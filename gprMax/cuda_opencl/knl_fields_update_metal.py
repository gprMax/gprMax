import Metal
import ctypes

#####################################
# 1. Setup the Metal kernel itself.
#####################################

# Define Metal kernel functions
kernel_electric = """
#include <metal_stdlib>
using namespace metal;

kernel void update_electric (device int NX,
                             device int NY,
                             device int NZ,
                             device const uint* ID [[buffer(0)]],
                             device float* Ex [[buffer(1)]],
                             device float* Ey [[buffer(2)]],
                             device float* Ez [[buffer(3)]],
                             device const float* Hx [[buffer(4)]],
                             device const float* Hy [[buffer(5)]],
                             device const float* Hz [[buffer(6)]],
                             uint3 gid [[thread_position_in_grid]]){
                             
            uint x = uint(thread_position_in_grid().x);
            uint y = uint(thread_position_in_grid().y);
            uint z = uint(thread_position_in_grid().z);

            // Convert the linear index to subscripts for 4D material ID array
            uint x_ID = (x % uint(($NX_ID * $NY_ID * $NZ_ID))) / (uint($NY_ID * $NZ_ID));
            uint y_ID = ((x % uint(($NX_ID * $NY_ID * $NZ_ID))) % (uint($NY_ID * $NZ_ID))) / uint($NZ_ID);
            uint z_ID = ((x % uint(($NX_ID * $NY_ID * $NZ_ID))) % (uint($NY_ID * $NZ_ID))) % uint($NZ_ID);

            // Ex component
            if ((NY != 1 || NZ != 1) && x < NX && y > 0 && y < NY && z > 0 && z < NZ) {
                uint materialEx = ID[(x_ID * uint($NY_ID * $NZ_ID)) + (y_ID * uint($NZ_ID)) + z_ID];
                Ex[x * uint(($NY_FIELDS * $NZ_FIELDS)) + y * uint($NZ_FIELDS) + z] =
                    updatecoeffsE[(materialEx * 5) + 0] * Ex[x * uint(($NY_FIELDS * $NZ_FIELDS)) + y * uint($NZ_FIELDS) + z] +
                    updatecoeffsE[(materialEx * 5) + 2] * (Hz[x * uint(($NY_FIELDS * $NZ_FIELDS)) + y * uint($NZ_FIELDS) + z] - Hz[x * uint(($NY_FIELDS * $NZ_FIELDS)) + (y-1) * uint($NZ_FIELDS) + z]) -
                    updatecoeffsE[(materialEx * 5) + 3] * (Hy[x * uint(($NY_FIELDS * $NZ_FIELDS)) + y * uint($NZ_FIELDS) + z] - Hy[x * uint(($NY_FIELDS * $NZ_FIELDS)) + y * uint($NZ_FIELDS) + (z-1)]);
            }

            // Ey component
            if ((NX != 1 || NZ != 1) && x > 0 && x < NX && y < NY && z > 0 && z < NZ) {
                uint materialEy = ID[(x_ID * uint($NY_ID * $NZ_ID)) + (y_ID * uint($NZ_ID)) + z_ID];
                Ey[x * uint(($NY_FIELDS * $NZ_FIELDS)) + y * uint($NZ_FIELDS) + z] =
                    updatecoeffsE[(materialEy * 5) + 0] * Ey[x * uint(($NY_FIELDS * $NZ_FIELDS)) + y * uint($NZ_FIELDS) + z] +
                    updatecoeffsE[(materialEy * 5) + 3] * (Hx[x * uint(($NY_FIELDS * $NZ_FIELDS)) + y * uint($NZ_FIELDS) + z] - Hx[x * uint(($NY_FIELDS * $NZ_FIELDS)) + y * uint($NZ_FIELDS) + (z-1)]) -
                    updatecoeffsE[(materialEy * 5) + 1] * (Hz[x * uint(($NY_FIELDS * $NZ_FIELDS)) + y * uint($NZ_FIELDS) + z] - Hz[(x-1) * uint(($NY_FIELDS * $NZ_FIELDS)) + y * uint($NZ_FIELDS) + z]);
            }

            // Ez component
            if ((NX != 1 || NY != 1) && x > 0 && x < NX && y > 0 && y < NY && z < NZ) {
                uint materialEz = ID[(x_ID * uint($NY_ID * $NZ_ID)) + (y_ID * uint($NZ_ID)) + z_ID];
                Ez[x * uint(($NY_FIELDS * $NZ_FIELDS)) + y * uint($NZ_FIELDS) + z] =
                    updatecoeffsE[(materialEz * 5) + 0] * Ez[x * uint(($NY_FIELDS * $NZ_FIELDS)) + y * uint($NZ_FIELDS) + z] +
                    updatecoeffsE[(materialEz * 5) + 1] * (Hy[x * uint(($NY_FIELDS * $NZ_FIELDS)) + y * uint($NZ_FIELDS) + z] - Hy[(x-1) * uint(($NY_FIELDS * $NZ_FIELDS)) + y * uint($NZ_FIELDS) + z]) -
                    updatecoeffsE[(materialEz * 5) + 2] * (Hx[x * uint(($NY_FIELDS * $NZ_FIELDS)) + y * uint($NZ_FIELDS) + z] - Hx[x * uint(($NY_FIELDS * $NZ_FIELDS)) + (y-1) * uint($NZ_FIELDS) + z]);
            }
            """

kernel_magnetic = """
#include <metal_stdlib>
kernel void update_magnetic(device int NX,
                            device int NY,
                            device int NZ,
                            device const uint* ID [[buffer(0)]],
                            device float* Hx [[ buffer(1) ]],
                            device float* Hy [[ buffer(2) ]],
                            device float* Hz [[ buffer(3) ]],
                            const device float* Ex [[ buffer(4) ]],
                            const device float* Ey [[ buffer(5) ]],
                            const device float* Ez [[ buffer(6) ]]
                            uint3 gid [[thread_position_in_grid]]){
            uint x = uint(thread_position_in_grid().x);
            uint y = uint(thread_position_in_grid().y);
            uint z = uint(thread_position_in_grid().z);

            // Convert the linear index to subscripts for 4D material ID array
            uint x_ID = (x % uint(($NX_ID * $NY_ID * $NZ_ID))) / (uint($NY_ID * $NZ_ID));
            uint y_ID = ((x % uint(($NX_ID * $NY_ID * $NZ_ID))) % (uint($NY_ID * $NZ_ID))) / uint($NZ_ID);
            uint z_ID = ((x % uint(($NX_ID * $NY_ID * $NZ_ID))) % (uint($NY_ID * $NZ_ID))) % uint($NZ_ID);

            // Hx component
            if (x < NX && y < NY && z > 0 && z < NZ) {
                uint materialHx = ID[(x_ID * uint($NY_ID * $NZ_ID)) + (y_ID * uint($NZ_ID)) + z_ID];
                Hx[x * uint(($NY_FIELDS * $NZ_FIELDS)) + y * uint($NZ_FIELDS) + z] =
                    updatecoeffsH[(materialHx * 4) + 0] * Hx[x * uint(($NY_FIELDS * $NZ_FIELDS)) + y * uint($NZ_FIELDS) + z] -
                    updatecoeffsH[(materialHx * 4) + 2] * (Ez[x * uint(($NY_FIELDS * $NZ_FIELDS)) + (y+1) * uint($NZ_FIELDS) + z] - Ez[x * uint(($NY_FIELDS * $NZ_FIELDS)) + y * uint($NZ_FIELDS) + z]) +
                    updatecoeffsH[(materialHx * 4) + 3] * (Ey[x * uint(($NY_FIELDS * $NZ_FIELDS)) + y * uint($NZ_FIELDS) + (z+1)] - Ey[x * uint(($NY_FIELDS * $NZ_FIELDS)) + y * uint($NZ_FIELDS) + z]);
            }

            // Hy component
            if (x < NX && y > 0 && y < NY && z < NZ) {
                uint materialHy = ID[(x_ID * uint($NY_ID * $NZ_ID)) + (y_ID * uint($NZ_ID)) + z_ID];
                Hy[x * uint(($NY_FIELDS * $NZ_FIELDS)) + y * uint($NZ_FIELDS) + z] =
                    updatecoeffsH[(materialHy * 4) + 0] * Hy[x * uint(($NY_FIELDS * $NZ_FIELDS)) + y * uint($NZ_FIELDS) + z] -
                    updatecoeffsH[(materialHy * 4) + 3] * (Ex[x * uint(($NY_FIELDS * $NZ_FIELDS)) + y * uint($NZ_FIELDS) + (z+1)] - Ex[x * uint(($NY_FIELDS * $NZ_FIELDS)) + y * uint($NZ_FIELDS) + z]) +
                    updatecoeffsH[(materialHy * 4) + 1] * (Ez[(x+1) * uint(($NY_FIELDS * $NZ_FIELDS)) + y * uint($NZ_FIELDS) + z] - Ez[x * uint(($NY_FIELDS * $NZ_FIELDS)) + y * uint($NZ_FIELDS) + z]);
            }

            // Hz component
            if (x > 0 && x < NX && y < NY && z < NZ) {
                uint materialHz = ID[(x_ID * uint($NY_ID * $NZ_ID)) + (y_ID * uint($NZ_ID)) + z_ID];
                Hz[x * uint(($NY_FIELDS * $NZ_FIELDS)) + y * uint($NZ_FIELDS) + z] =
                    updatecoeffsH[(materialHz * 4) + 0] * Hz[x * uint(($NY_FIELDS * $NZ_FIELDS)) + y * uint($NZ_FIELDS) + z] -
                    updatecoeffsH[(materialHz * 4) + 1] * (Ey[(x+1) * uint(($NY_FIELDS * $NZ_FIELDS)) + y * uint($NZ_FIELDS) + z] - Ey[x * uint(($NY_FIELDS * $NZ_FIELDS)) + y * uint($NZ_FIELDS) + z]) +
                    updatecoeffsH[(materialHz * 4) + 2] * (Ex[x * uint(($NY_FIELDS * $NZ_FIELDS)) + (y+1) * uint($NZ_FIELDS) + z] - Ex[x * uint(($NY_FIELDS * $NZ_FIELDS)) + y * uint($NZ_FIELDS) + z]);
                            }
"""


# Create a Metal device, library, and kernel function
device = Metal.MTLCreateSystemDefaultDevice()
library1 = device.newLibraryWithSource_options_error_(kernel_electric, None, None)[0]
library2 = device.newLibraryWithSource_options_error_(kernel_magnetic, None, None)[0]
electric_function = library1.newFunctionWithName_("update_electric")
magnetic_function = library2.newFunctionWithName_("update_magnetic")

#########################################
# 2. Setup the input and output buffers.
#########################################

# Create input and output buffers

buffer_length = 1024
# NUM_ELEMENTS = NX * NY * NZ   

# input buffers
ID_buffer = device.newBufferWithLength_options_(buffer_length * ctypes.sizeof(ctypes.c_uint), Metal.MTLResourceStorageModeShared)
Hx_buffer = device.newBufferWithLength_options_(buffer_length * ctypes.sizeof(ctypes.c_float), Metal.MTLResourceStorageModeShared)
Hy_buffer = device.newBufferWithLength_options_(buffer_length * ctypes.sizeof(ctypes.c_float), Metal.MTLResourceStorageModeShared)
Hz_buffer = device.newBufferWithLength_options_(buffer_length * ctypes.sizeof(ctypes.c_float), Metal.MTLResourceStorageModeShared)
Ex_buffer = device.newBufferWithLength_options_(buffer_length * ctypes.sizeof(ctypes.c_float), Metal.MTLResourceStorageModeShared)
Ey_buffer = device.newBufferWithLength_options_(buffer_length * ctypes.sizeof(ctypes.c_float), Metal.MTLResourceStorageModeShared)
Ez_buffer = device.newBufferWithLength_options_(buffer_length * ctypes.sizeof(ctypes.c_float), Metal.MTLResourceStorageModeShared)


#####################################
# 3. Call the Metal kernel function.
#####################################

# Create a command queue and command buffer
commandQueue = device.newCommandQueue()
commandBuffer = commandQueue.commandBuffer()

# Set the kernel function and buffers
pso = device.newComputePipelineStateWithFunction_error_(electric_function, None)[0]
pso2 = device.newComputePipelineStateWithFunction_error_(magnetic_function, None)[1]
computeEncoder = commandBuffer.computeCommandEncoder()
computeEncoder.setComputePipelineState_(pso)
computeEncoder.setComputePipelineState_(pso2)
# computeEncoder.setBuffer_offset_atIndex_(input_buffer, 0, 0)
# computeEncoder.setBuffer_offset_atIndex_(output_buffer, 0, 1)

# Define threadgroup size
threadsPerThreadgroup = Metal.MTLSizeMake(1024, 1, 1)
threadgroupSize = Metal.MTLSizeMake(pso.maxTotalThreadsPerThreadgroup(), 1, 1)
threadgroupSize = Metal.MTLSizeMake(pso2.maxTotalThreadsPerThreadgroup(), 1, 1)
# Dispatch the kernel
computeEncoder.dispatchThreads_threadsPerThreadgroup_(threadsPerThreadgroup, threadgroupSize)
computeEncoder.endEncoding()

# Commit the command buffer
commandBuffer.commit()
commandBuffer.waitUntilCompleted()


