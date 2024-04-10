cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(GPU_Denoiser LANGUAGES CXX CUDA)

#set(CMAKE_CUDA_FLAGS "-Wall")
set(CMAKE_CUDA_ARCHITECTURES 70)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

include(CTest)
 
add_library(GPU_Denoiser Fonctions.cu)

add_executable(Denoiser Denoiser.cu Fonctions.cu)
 
# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to 
# particles will also build with -std=c++11
target_compile_features(GPU_Denoiser PUBLIC cxx_std_11)
 
# We need to explicitly state that we need all CUDA files in the 
# particle library to be built with -dc as the member functions 
# could be called by other libraries and executables
set_target_properties( GPU_Denoiser
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
 
#add_executable(GPU_Denoiser_test test.cu)
 
#set_property(TARGET GPU_Denoiser_test 
#             PROPERTY CUDA_SEPARABLE_COMPILATION ON)
#target_link_libraries(GPU_Denoiser_test PRIVATE GPU_Denoiser)