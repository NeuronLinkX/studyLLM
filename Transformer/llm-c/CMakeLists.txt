cmake_minimum_required(VERSION 3.12)
project(llm_c C)

set(CMAKE_C_STANDARD 11)

# -------------------------------
# 컴파일 옵션(아키텍처별 안전 처리)
# -------------------------------
# 공통 옵션
add_compile_options(-O3 -ffast-math -Wall -Wextra)

# x86_64에서만 -march=native 사용
if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    # macOS
    if (CMAKE_OSX_ARCHITECTURES MATCHES "x86_64")
        add_compile_options(-march=native)
    endif()
else()
    # Linux/기타: x86_64일 때만 -march=native 시도
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
        add_compile_options(-march=native)
    endif()
endif()

# 절대 넣지 마세요(arm64에서 깨짐):
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native -ffast-math")

# -------------------------------
# 라이브러리/소스
# -------------------------------
add_library(llm STATIC
    src/llm_tensor.c
    src/llm_math.c
    src/llm_pe.c
    src/llm_attention.c
    src/llm_ffn.c
    src/llm_block.c
    src/llm_model.c
    src/llm_weights_io.c
)

target_include_directories(llm PUBLIC include)
# 수학함수 필요시 링크 (macOS/clang는 보통 필요 없지만 호환 유지)
target_link_libraries(llm m)

# -------------------------------
# 예제 실행 파일
# -------------------------------
add_executable(demo examples/demo_forward.c)
target_link_libraries(demo llm m)

add_executable(hello_min examples/hello_min.c)
target_link_libraries(hello_min llm m)
