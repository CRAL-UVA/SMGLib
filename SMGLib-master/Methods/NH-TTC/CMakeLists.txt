cmake_minimum_required( VERSION 3.0 )

## Use the variable PROJECT_NAME for changing the target name
set( PROJECT_NAME "NH_TTC" )

## Set our project name
project( ${PROJECT_NAME} )

# Build GLFW
# Also disable building some of the extra things GLFW has (examples, tests, docs)
set(GLFW_BUILD_EXAMPLES  OFF CACHE BOOL " " FORCE)
set(GLFW_BUILD_TESTS     OFF CACHE BOOL " " FORCE)
set(GLFW_BUILD_DOCS      OFF CACHE BOOL " " FORCE)

# Now actually run cmake on the CMakeLists.txt file found inside of the GLFW directory
add_subdirectory(ext/glfw)

include_directories( SYSTEM ext )
include_directories( SYSTEM ext/eigen )
include_directories( SYSTEM ext/glm )
include_directories( SYSTEM ext/glad/include )
include_directories( SYSTEM ext/glfw/include )

add_subdirectory( ext )

add_library(nativefd STATIC IMPORTED)
if (WIN32)
	set_target_properties(nativefd PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/ext/nativefd/lib/nfd.lib)
elseif(APPLE)
	set_target_properties(nativefd PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/ext/nativefd/lib/libnfd_osx.a)
else()
	set_target_properties(nativefd PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/ext/nativefd/lib/libnfd_linux.a)
endif()

find_package( Threads REQUIRED )

if(WIN32)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
else()
	set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -g -Wall -Wextra -Wpedantic -Wno-unused-parameter" )
endif()

include_directories( src )
add_subdirectory( src )

if (WIN32)
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT run_nhttc)
endif()
