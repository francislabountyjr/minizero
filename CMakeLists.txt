cmake_minimum_required(VERSION 3.16)

project(minizero)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Torch REQUIRED)
find_package(Boost COMPONENTS system thread iostreams)
if(GAME_TYPE STREQUAL "ATARI")
    find_package(ale REQUIRED)
    find_package(OpenCV REQUIRED)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -g -Wall -mpopcnt -O3 -pthread")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -Wall -Wno-unused-function -O0 -pthread")

# for git info
include_directories(${PROJECT_BINARY_DIR}/git_info)
find_package(Git QUIET)
set(GIT_INFO_DIR ${PROJECT_BINARY_DIR}/git_info)
file(MAKE_DIRECTORY ${GIT_INFO_DIR})

set(GIT_HASH_VALUE "unknown")
set(GIT_SHORT_HASH_VALUE "unknown")
if(GIT_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_HASH_VALUE
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --abbrev=6 --dirty --always --exclude *
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_SHORT_HASH_VALUE
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
endif()

file(WRITE ${GIT_INFO_DIR}/git_info.h
    "#pragma once\n\n"
    "#define GIT_HASH \"${GIT_HASH_VALUE}\"\n"
    "#define GIT_SHORT_HASH \"${GIT_SHORT_HASH_VALUE}\"\n"
)

add_subdirectory(minizero)
add_subdirectory(minizero/actor)
add_subdirectory(minizero/config)
add_subdirectory(minizero/console)
add_subdirectory(minizero/environment)
add_subdirectory(minizero/learner)
add_subdirectory(minizero/network)
add_subdirectory(minizero/utils)
add_subdirectory(minizero/zero)

string(TOLOWER "${PROJECT_NAME}_${GAME_TYPE}" EXE_FILE_NAME)
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME ${EXE_FILE_NAME})
