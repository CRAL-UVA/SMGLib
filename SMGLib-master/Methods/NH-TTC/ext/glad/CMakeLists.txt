add_library( glad src/glad.c include/glad/glad.h )
target_link_libraries( glad ${CMAKE_DL_LIBS} )