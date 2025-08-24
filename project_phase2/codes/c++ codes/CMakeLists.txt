cmake_minimum_required(VERSION 3.10)
project(SatellitePropagator)

set(CMAKE_CXX_STANDARD 17)

include_directories(../AhmedMAtallah/Chebyshev-Picard-Method/include)

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
    ../AhmedMAtallah/Chebyshev-Picard-Method/src/SHM.cpp
    ../AhmedMAtallah/Chebyshev-Picard-Method/src/MCPIIOM.cpp
    ../AhmedMAtallah/Chebyshev-Picard-Method/src/Basic.cpp
)

add_executable(SatellitePropagator ${SOURCES})

# If you need to link external libraries, add them here
# target_link_libraries(SatellitePropagator <lib1> <lib2>)