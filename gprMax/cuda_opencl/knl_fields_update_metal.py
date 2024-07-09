import Metal
import ctypes

#####################################
# 1. Setup the Metal kernel itself.
#####################################

# Define Metal kernel functions
kernel_electric = """
#include <metal_stdlib>
using namespace metal;
kernel void update_electric (
                            device const uint* ID [[buffer(0)]],
                            device float* Ex [[buffer(1)]],
                            device float* Ey [[buffer(2)]],
                            device float* Ez [[buffer(3)]],
                            device const float* Hx [[buffer(4)]],
                            device const float* Hy [[buffer(5)]],
                            device const float* Hz [[buffer(6)]],
                            device const float* updatecoeffsE [[buffer(7)]],
                            uint3 gid [[thread_position_in_grid]])
{
   // Retrieve dimensions from buffers
   uint NX = ID[0];
   uint NY = ID[1];
   uint NZ = ID[2];

   uint x = gid.x;
   uint y = gid.y;
   uint z = gid.z;

   // Convert the linear index to subscripts for 3D arrays
   uint x_ID = (x % NX);  
   uint y_ID = (y % NY);  
   uint z_ID = (z % NZ);

   // Ex component
   if ((NY != 1 || NZ != 1) && x < NX && y > 0 && y < NY && z > 0 && z < NZ) {
       uint materialEx = ID[(x_ID * (NY * NZ)) + (y_ID * NZ) + z_ID];
       Ex[x * (NY * NZ) + y * NZ + z] =
           updatecoeffsE[(materialEx * 5) + 0] * Ex[x * (NY * NZ) + y * NZ + z] +
           updatecoeffsE[(materialEx * 5) + 2] * (Hz[x * (NY * NZ) + y * NZ + z] - Hz[x * (NY * NZ) + (y-1) * NZ + z]) -
           updatecoeffsE[(materialEx * 5) + 3] * (Hy[x * (NY * NZ) + y * NZ + z] - Hy[x * (NY * NZ) + y * NZ + (z-1)]);
   }

   // Ey component
   if ((NX != 1 || NZ != 1) && x > 0 && x < NX && y < NY && z > 0 && z < NZ) {
       uint materialEy = ID[(x_ID * (NY * NZ)) + (y_ID * NZ) + z_ID];
       Ey[x * (NY * NZ) + y * NZ + z] =
           updatecoeffsE[(materialEy * 5) + 0] * Ey[x * (NY * NZ) + y * NZ + z] +
           updatecoeffsE[(materialEy * 5) + 3] * (Hx[x * (NY * NZ) + y * NZ + z] - Hx[x * (NY * NZ) + y * NZ + (z-1)]) -
           updatecoeffsE[(materialEy * 5) + 1] * (Hz[x * (NY * NZ) + y * NZ + z] - Hz[(x-1) * (NY * NZ) + y * NZ + z]);
   }

   // Ez component
   if ((NX != 1 || NY != 1) && x > 0 && x < NX && y > 0 && y < NY && z < NZ) {
       uint materialEz = ID[(x_ID * (NY * NZ)) + (y_ID * NZ) + z_ID];
       Ez[x * (NY * NZ) + y * NZ + z] =
           updatecoeffsE[(materialEz * 5) + 0] * Ez[x * (NY * NZ) + y * NZ + z] +
           updatecoeffsE[(materialEz * 5) + 1] * (Hy[x * (NY * NZ) + y * NZ + z] - Hy[(x-1) * (NY * NZ) + y * NZ + z]) -
           updatecoeffsE[(materialEz * 5) + 2] * (Hx[x * (NY * NZ) + y * NZ + z] - Hx[x * (NY * NZ) + (y-1) * NZ + z]);
   }
}
           """

kernel_magnetic = """
#include <metal_stdlib>
kernel void update_magnetic(
                           device const uint* ID [[buffer(0)]],
                           device float* Hx [[ buffer(1) ]],
                           device float* Hy [[ buffer(2) ]],
                           device float* Hz [[ buffer(3) ]],
                           const device float* Ex [[ buffer(4) ]],
                           const device float* Ey [[ buffer(5) ]],
                           const device float* Ez [[ buffer(6) ]],
                           device const float* updatecoeffsH [[buffer(7)]],
                           uint3 gid [[thread_position_in_grid]])
{
   // Retrieve dimensions from buffers
   uint NX = ID[0];
   uint NY = ID[1];
   uint NZ = ID[2];

   uint x = gid.x;
   uint y = gid.y;
   uint z = gid.z;

   // Convert the linear index to subscripts for 3D arrays
   uint x_ID = (x % NX);  
   uint y_ID = (y % NY);  
   uint z_ID = (z % NZ);

   // Hx component
   if ((NY != 1 || NZ != 1) && x > 0 && x < NX && y < NY && z < NZ) {
       uint materialHx = ID[(x_ID * (NY * NZ)) + (y_ID * NZ) + z_ID];
       Hx[x * (NY * NZ) + y * NZ + z] =
           updatecoeffsH[(materialHx * 5) + 0] * Hx[x * (NY * NZ) + y * NZ + z] +
           updatecoeffsH[(materialHx * 5) + 2] * (Ez[x * (NY * NZ) + y * NZ + z] - Ez[x * (NY * NZ) + (y-1) * NZ + z]) -
           updatecoeffsH[(materialHx * 5) + 3] * (Ey[x * (NY * NZ) + y * NZ + z] - Ey[x * (NY * NZ) + y * NZ + (z-1)]);
   }

   // Hy component
   if ((NX != 1 || NZ != 1) && x < NX && y > 0 && y < NY && z < NZ) {
       uint materialHy = ID[(x_ID * (NY * NZ)) + (y_ID * NZ) + z_ID];
       Hy[x * (NY * NZ) + y * NZ + z] =
           updatecoeffsH[(materialHy * 5) + 0] * Hy[x * (NY * NZ) + y * NZ + z] +
           updatecoeffsH[(materialHy * 5) + 3] * (Ex[x * (NY * NZ) + y * NZ + z] - Ex[x * (NY * NZ) + y * NZ + (z-1)]) -
           updatecoeffsH[(materialHy * 5) + 1] * (Ez[x * (NY * NZ) + y * NZ + z] - Ez[(x-1) * (NY * NZ) + y * NZ + z]);
   }

   // Hz component
   if ((NX != 1 || NY != 1) && x < NX && y < NY && z > 0 && z < NZ) {
       uint materialHz = ID[(x_ID * (NY * NZ)) + (y_ID * NZ) + z_ID];
       Hz[x * (NY * NZ) + y * NZ + z] =
           updatecoeffsH[(materialHz * 5) + 0] * Hz[x * (NY * NZ) + y * NZ + z] +
           updatecoeffsH[(materialHz * 5) + 1] * (Ey[x * (NY * NZ) + y * NZ + z] - Ey[(x-1) * (NY * NZ) + y * NZ + z]) -
           updatecoeffsH[(materialHz * 5) + 2] * (Ex[x * (NY * NZ) + y * NZ + z] - Ex[x * (NY * NZ) + (y-1) * NZ + z]);
   }
}
"""

# Create a Metal device, library, and kernel function
device = Metal.MTLCreateSystemDefaultDevice()
# library1 = device.newLibraryWithSource_options_error_(kernel_electric, None, None)[0]
# library2 = device.newLibraryWithSource_options_error_(kernel_magnetic, None, None)[0]
# electric_function = library1.newFunctionWithName_("update_electric")
# magnetic_function = library2.newFunctionWithName_("update_magnetic")

library1, error1 = device.newLibraryWithSource_options_error_(kernel_electric, None, None)
if error1:
   print("Error creating electric library:", error1)
   exit()

library2, error2 = device.newLibraryWithSource_options_error_(kernel_magnetic, None, None)
if error2:
   print("Error creating magnetic library:", error2)
   exit()

# Check if libraries were created successfully
if not library1 or not library2:
   print("Failed to create Metal libraries")
   exit()

# Retrieve kernel functions
electric_function = library1.newFunctionWithName_("update_electric")
magnetic_function = library2.newFunctionWithName_("update_magnetic")
if not electric_function:
   print("Failed to retrieve electric function from library")
   exit()
if not magnetic_function:
   print("Failed to retrieve magnetic function from library")
   exit()

#########################################
# 2. Setup the input and output buffers.
#########################################

# Create input and output buffers
buffer_length = 1024

# input buffers
ID_buffer = device.newBufferWithLength_options_(buffer_length * ctypes.sizeof(ctypes.c_uint), Metal.MTLResourceStorageModeShared)
Hx_buffer = device.newBufferWithLength_options_(buffer_length * ctypes.sizeof(ctypes.c_float), Metal.MTLResourceStorageModeShared)
Hy_buffer = device.newBufferWithLength_options_(buffer_length * ctypes.sizeof(ctypes.c_float), Metal.MTLResourceStorageModeShared)
Hz_buffer = device.newBufferWithLength_options_(buffer_length * ctypes.sizeof(ctypes.c_float), Metal.MTLResourceStorageModeShared)
Ex_buffer = device.newBufferWithLength_options_(buffer_length * ctypes.sizeof(ctypes.c_float), Metal.MTLResourceStorageModeShared)
Ey_buffer = device.newBufferWithLength_options_(buffer_length * ctypes.sizeof(ctypes.c_float), Metal.MTLResourceStorageModeShared)
Ez_buffer = device.newBufferWithLength_options_(buffer_length * ctypes.sizeof(ctypes.c_float), Metal.MTLResourceStorageModeShared)
updatecoeffsE_buffer = device.newBufferWithLength_options_(buffer_length * ctypes.sizeof(ctypes.c_float), Metal.MTLResourceStorageModeShared)
updatecoeffsH_buffer = device.newBufferWithLength_options_(buffer_length * ctypes.sizeof(ctypes.c_float), Metal.MTLResourceStorageModeShared)

#####################################
# 3. Call the Metal kernel function.
#####################################

# Create a command queue and command buffer
commandQueue = device.newCommandQueue()
commandBuffer = commandQueue.commandBuffer()

# Set the kernel function and buffers
pso1 = device.newComputePipelineStateWithFunction_error_(electric_function, None)[0]
pso2 = device.newComputePipelineStateWithFunction_error_(magnetic_function, None)[0]

# Check for errors in creating pipeline states
if not pso1 or not pso2:
   print("Failed to create one or more compute pipeline states")
   exit()

# Create command encoder
computeEncoder = commandBuffer.computeCommandEncoder()

# Set compute pipeline state for update_electric
computeEncoder.setComputePipelineState_(pso1)
computeEncoder.setBuffer_offset_atIndex_(ID_buffer, 0, 0)
computeEncoder.setBuffer_offset_atIndex_(Ex_buffer, 0, 1)
computeEncoder.setBuffer_offset_atIndex_(Ey_buffer, 0, 2)
computeEncoder.setBuffer_offset_atIndex_(Ez_buffer, 0, 3)
computeEncoder.setBuffer_offset_atIndex_(Hx_buffer, 0, 4)
computeEncoder.setBuffer_offset_atIndex_(Hy_buffer, 0, 5)
computeEncoder.setBuffer_offset_atIndex_(Hz_buffer, 0, 6)
computeEncoder.setBuffer_offset_atIndex_(updatecoeffsE_buffer, 0, 7)

# Define threadgroup size for pso1
threadsPerThreadgroup1 = Metal.MTLSizeMake(1024, 1, 1)
threadgroupSize1 = Metal.MTLSizeMake(pso1.maxTotalThreadsPerThreadgroup(), 1, 1)

# Dispatch threads for pso1
computeEncoder.dispatchThreads_threadsPerThreadgroup_(threadsPerThreadgroup1, threadgroupSize1)

# Set compute pipeline state for update_magnetic
computeEncoder.setComputePipelineState_(pso2)
computeEncoder.setBuffer_offset_atIndex_(ID_buffer, 0, 0)
computeEncoder.setBuffer_offset_atIndex_(Ex_buffer, 0, 4)
computeEncoder.setBuffer_offset_atIndex_(Ey_buffer, 0, 5)
computeEncoder.setBuffer_offset_atIndex_(Ez_buffer, 0, 6)
computeEncoder.setBuffer_offset_atIndex_(Hx_buffer, 0, 1)
computeEncoder.setBuffer_offset_atIndex_(Hy_buffer, 0, 2)
computeEncoder.setBuffer_offset_atIndex_(Hz_buffer, 0, 3)
computeEncoder.setBuffer_offset_atIndex_(updatecoeffsH_buffer, 0, 7)

# Define threadgroup size for pso2
threadsPerThreadgroup2 = Metal.MTLSizeMake(1024, 1, 1)
threadgroupSize2 = Metal.MTLSizeMake(pso2.maxTotalThreadsPerThreadgroup(), 1, 1)

# Dispatch threads for pso2
computeEncoder.dispatchThreads_threadsPerThreadgroup_(threadsPerThreadgroup2, threadgroupSize2)

# End encoding and commit the command buffer
computeEncoder.endEncoding()
commandBuffer.commit()
commandBuffer.waitUntilCompleted()