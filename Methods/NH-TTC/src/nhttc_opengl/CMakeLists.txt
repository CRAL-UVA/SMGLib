if (UNIX AND NOT APPLE)
FIND_PACKAGE(PkgConfig REQUIRED)
PKG_CHECK_MODULES(GTK3 REQUIRED gtk+-3.0)

# Setup CMake to use GTK+, tell the compiler where to look for headers
# and to the linker where to look for libraries
INCLUDE_DIRECTORIES(${GTK3_INCLUDE_DIRS})
LINK_DIRECTORIES(${GTK3_LIBRARY_DIRS})

# Add other flags to the compiler
ADD_DEFINITIONS(${GTK3_CFLAGS_OTHER})
endif()

set( RUN_SGD_HEADERS opengl_utils.h nhttc_sim.h test_utils.h )
set( RUN_SGD_SOURCE run_nhttc.cpp opengl_utils.cpp nhttc_sim.cpp test_utils.cpp )

add_executable(run_nhttc ${RUN_SGD_SOURCE} ${RUN_SGD_HEADERS} )
if (UNIX AND NOT APPLE)
target_link_libraries( run_nhttc SGD nativefd glad glfw ${CMAKE_THREAD_LIBS_INIT} ${GTK3_LIBRARIES})
else()
target_link_libraries( run_nhttc SGD nativefd glad glfw ${CMAKE_THREAD_LIBS_INIT} )
endif()

file(COPY shaders DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY textures DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
