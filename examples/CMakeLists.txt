add_executable(demo_jq      demo_jq.cpp)
add_executable(demo_jhq     demo_jhq.cpp)
add_executable(demo_jq_ivf  demo_jq_ivf.cpp)
add_executable(demo_jhq_ivf demo_jhq_ivf.cpp)

target_link_libraries(demo_jq      PRIVATE jhq_impl)
target_link_libraries(demo_jhq     PRIVATE jhq_impl)
target_link_libraries(demo_jq_ivf  PRIVATE jhq_impl)
target_link_libraries(demo_jhq_ivf PRIVATE jhq_impl)
