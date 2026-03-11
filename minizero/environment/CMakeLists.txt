set(SRCS base/base_env.cpp)
if(GAME_TYPE STREQUAL "PUZZLE2048")
    file(GLOB GAME_SRCS stochastic/puzzle2048/*.cpp)
elseif(GAME_TYPE STREQUAL "TETRISBLOCKPUZZLE")
    file(GLOB GAME_SRCS stochastic/tetrisblockpuzzle/*.cpp)
else()
    string(TOLOWER "${GAME_TYPE}" GAME_TYPE_LOWER)
    file(GLOB GAME_SRCS ${GAME_TYPE_LOWER}/*.cpp)
endif()
list(APPEND SRCS ${GAME_SRCS})

add_library(environment ${SRCS})
target_include_directories(
    environment PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    base
    amazons
    atari
    breakthrough
    clobber
    conhex
    connect6
    dotsandboxes
    gungi
    go
    gomoku
    havannah
    hex
    killallgo
    linesofaction
    nogo
    othello
    rubiks
    santorini
    surakarta
    tictactoe
    stochastic
    stochastic/puzzle2048
    stochastic/tetrisblockpuzzle
)
target_link_libraries(
    environment
    config
    utils
)
if(GAME_TYPE STREQUAL "ATARI")
    target_link_libraries(
        environment
        ale::ale-lib
        ${OpenCV_LIBS}
    )
endif()
