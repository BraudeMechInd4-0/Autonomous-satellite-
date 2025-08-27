cmake_minimum_required(VERSION 3.10)
project(SatellitePropagator)

find_package(Eigen3 REQUIRED)

set(CMAKE_CXX_STANDARD 17)

# Add source files
set(SOURCES
    Main_to_print_executable_time.cpp
    RK4.cpp
    RK8.cpp
    ODE45.cpp
    ODE78.cpp
    ODE113.cpp
    MPCI.cpp
    CommonFunctions.cpp
    coefficients78.cpp
)

add_executable(SatellitePropagator ${SOURCES})
target_link_libraries(SatellitePropagator Eigen3::Eigen)

# If you need to link external libraries, add them here
# target_link_libraries(SatellitePropagator <lib1> <lib2>)