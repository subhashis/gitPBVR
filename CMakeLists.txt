project(PBVR CXX C)

cmake_minimum_required(VERSION 2.6.0)  # FindBoost.cmake is broken in older versions

set(CMAKE_MODULE_PATH "$ENV{CMAKE_MODULE_PATH}")

FILE(GLOB C_HEADERS "*.h" "*.hpp")
FILE(GLOB C_SOURCES "*.c" "*.cpp" "*.cxx")
FILE(GLOB CUDA_HEADERS "*.cuh")
FILE(GLOB CUDA_SOURCES "*.cu")




########################################
#     OpenGL      #
########################################

find_package(OpenGL REQUIRED)
INCLUDE_DIRECTORIES(${OPENGL_INCLUDE_DIR})
message(!!OPENGL_INCLUDE_DIR=${OPENGL_INCLUDE_DIR})
message(!!GLEW_LIBRARIES=${GLEW_LIBRARIES})
if(UNIX)
  SET(GLEW_LIBRARIES "GLEW")
endif()
message(!!GLEW_LIBRARIES=${GLEW_LIBRARIES})

find_package(GLUT REQUIRED)
include_directories(${GLUT_INCLUDE_DIRS})
message(!!GLUT_INCLUDE_DIRS=${GLUT_INCLUDE_DIRS})

message(Starting Cuda!!!)
########################################
#     CUDA      #
########################################


find_package(CUDA REQUIRED COMPONENTS sdk)
include_directories(${CUDA_INCLUDE_DIRS})
message(!!CUDA_INCLUDE_DIRS=${CUDA_INCLUDE_DIRS})
SET(CUDA_SDK_ROOT_DIR "/root/NVIDIA_CUDA-5.0_Samples")
SET(CUDA_CUT_INCLUDE_DIR "/root/NVIDIA_CUDA-5.0_Samples/common/inc")
message(%%CUDA_SDK_ROOT_DIR = ${CUDA_SDK_ROOT_DIR})
message(!!CUDA_CUT_INCLUDE_DIRS=${CUDA_CUT_INCLUDE_DIR})
message(!!CUDA_DIR=${CUDA_VERSION})

cuda_include_directories(${CUDA_INCLUDE_DIRS} ${CUDA_CUT_INCLUDE_DIR})
message(!!include_directories=${CUDA_INCLUDE_DIRS})
set(CUDA_NVCC_FLAGS "-arch=sm_21;-use_fast_math;")


message("CMAKE_SIZEOF_VOID_P: ${CMAKE_SIZEOF_VOID_P}")


message("cutil found: ${CUDA_CUT_INCLUDE_DIR}")


if(CMAKE_SIZEOF_VOID_P EQUAL 8)
message("64 bit system ${CUDA_SDK_ROOT_DIR}/../shared/lib/x64/")
   
 
else(CMAKE_SIZEOF_VOID_P EQUAL 8)
message("32 bit system")
   
  
endif(CMAKE_SIZEOF_VOID_P EQUAL 8)

###############Libraries#################
set(GLEW_CUDA_LIBRARY "/usr/lib/x86_64-linux-gnu/libGLEW.so")
#set(freeglut_LIBRARY "/root/NVIDIA_CUDA-5.0_Samples/common/lib/x64/freeglut.lib")
set(GL_LIBRARY "/usr/lib/x86_64-linux-gnu/libGL.so")
set(GLU_LIBRARY "/usr/lib/x86_64-linux-gnu/libGLU.so")
set(GLUT_LIBRARY "/usr/lib/x86_64-linux-gnu/libglut.so")
message(1. CUDA library = ${CUDA_LIBRARIES})
message(2. GLEW_CUDA_LIBRARY = ${GLEW_CUDA_LIBRARY})
message(3. freeglut_LIBRARY = ${freeglut_LIBRARY})


########################################
#     CUDA Templates      #
########################################





########################################
#     VTK      #
########################################

find_package(VTK REQUIRED)
SET(QT_QMAKE_EXECUTABLE ${VTK_QT_QMAKE_EXECUTABLE} CACHE FILEPATH "")
include(${VTK_USE_FILE})
include_directories(${VTK_DIR}/include)


########################################
#     QT       #
########################################

set(QT_USE_QTOPENGL TRUE)
find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})
file(GLOB UI "*.ui")
QT4_WRAP_CPP(MOC_SRCS ${C_HEADERS})
QT4_WRAP_UI(UI_WRAP ${UI})

message("UI: " ${UI})
message("UI_WRAP: " ${UI_WRAP})

message("MOC: " ${MOC_SRCS})


########################################
#     BOOST       #
########################################
set(Boost_ADDITIONAL_VERSIONS "1.41")
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
message(Boost_INCLUDE_DIR=${Boost_INCLUDE_DIR})


include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} )



SET(HEADERS ${C_HEADERS} ${CUDA_HEADERS} ${UI_WRAP})
SET(SOURCES ${C_SOURCES} ${CUDA_SOURCES} ${MOC_SRCS})
message("headers: " ${HEADERS})
message("sources: " ${SOURCES})

cuda_add_executable(PBVR 
	${HEADERS} 
	${SOURCES})
target_link_libraries(PBVR 
  ${CUDA_LIBRARIES} 
  ${GLEW_CUDA_LIBRARY}
  ${GL_LIBRARY}
  ${GLU_LIBRARY}
  ${GLUT_LIBRARY}
  vtkIO
  vtkCommon
  vtkHybrid
  vtkRendering
  QVTK
  )
  
