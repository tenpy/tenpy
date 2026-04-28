message(STATUS "")
message(STATUS "------------------------------------------------------------------------")
message(STATUS " Project cyten, A Cross-section of Python & C++, Tensor network library ")
message(STATUS "------------------------------------------------------------------------")
message(STATUS "")

# #####################################################################
# ## CMAKE and CXX VERSION
# #####################################################################
cmake_minimum_required(VERSION 3.20...3.27)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

message(STATUS " Generator: ${CMAKE_GENERATOR}")
message(STATUS " Build Target: ${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS " Installation Prefix: ${CMAKE_INSTALL_PREFIX}")

# #####################################################################
# ## Version
# #####################################################################

# Scikit-build-core sets version
if(NOT SKBUILD_PROJECT_VERSION)
  set(SKBUILD_PROJECT_VERSION 0.0.1)
endif()

# #####################################################################
# ## COMMAND LINE OPTIONS
# #####################################################################
option(BUILD_DOC "Build API documentation" OFF)


# #####################################################################
# ## PROJECT
# #####################################################################
project(
    cyten
    VERSION ${SKBUILD_PROJECT_VERSION}
    LANGUAGES CXX
)

include(CMakePrintHelpers)
cmake_print_variables(PROJECT_SOURCE_DIR)
cmake_print_variables(CMAKE_CXX_STANDARD)


# #####################################################################
# ## C++ LIBRARY
# #####################################################################

add_library(cyten STATIC)
set_property(TARGET cyten PROPERTY C_VISIBILITY_PRESET hidden)
set_property(TARGET cyten PROPERTY VISIBILITY_INLINES_HIDDEN ON)

target_include_directories(cyten
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  PUBLIC
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
add_subdirectory(src)


# #####################################################################
# ## PYTHON BINDINGS
# #####################################################################

set(Python_FIND_VIRTUALENV ONLY)
set(Python_FIND_STRATEGY LOCATION)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

find_package(pybind11 CONFIG REQUIRED)
# Find the module development requirements (requires FindPython from 3.17 or
# scikit-build-core's built-in backport)
cmake_print_variables(Python_VERSION)
cmake_print_variables(Python_INCLUDE_DIRS)
# cmake_print_variables(Python_NumPy_VERSION)
# cmake_print_variables(Python_NumPy_INCLUDE_DIRS)

pybind11_add_module(_core pybind/core.cpp)

target_include_directories(_core
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/pybind
  PUBLIC "${PROJECT_SOURCE_DIR}/include")
add_subdirectory(pybind)  # adds source files to _core

target_link_libraries(_core PUBLIC cyten)


# The install directory is the output (wheel) directory
install(TARGETS _core LIBRARY DESTINATION cyten)


# #####################################################################
# ## DOCUMENTATION
# #####################################################################
IF(BUILD_DOC)
  FIND_PACKAGE(Doxygen REQUIRED)

  if(DOXYGEN_FOUND)
    configure_file(cyten_Doxyfile.in cytenDoxyfile)
    add_custom_target(doc ALL
      ${DOXYGEN_EXECUTABLE} ${PROJECT_SOURCE_DIR}/docs/Doxyfile
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/docs
      COMMENT "Generating API documentation with Doxygen in docs/build_docs/" VERBATIM
    )
  endif(DOXYGEN_FOUND)

  message(STATUS " Build Documentation: YES")
  message(STATUS "  - Doxygen: ${DOXYGEN_EXECUTABLE}")
else()
  message(STATUS " Build Documentation: NO")
endif()

if(DOXYGEN_FOUND)
  install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/docs/
    DESTINATION docs
    COMPONENT documentation
  )
endif()



# enable_testing()
# add_test(NAME import_check COMMAND python -c "import cyten")
