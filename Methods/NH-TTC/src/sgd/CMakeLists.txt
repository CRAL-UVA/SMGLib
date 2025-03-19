set( SGD_HEADERS sgd.h ttc_obstacles.h ttc_sgd_problem.h ttc_sgd_problem_models.h )
set( SGD_SOURCE sgd.cpp ttc_obstacles.cpp ttc_sgd_problem.cpp ttc_sgd_problem_models.cpp )

add_library(SGD ${SGD_SOURCE} ${SGD_HEADERS})