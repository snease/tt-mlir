source env/activate

cmake -B env/build env
cmake --build env/build

build_type=${BUILD_TYPE:-Release}
c_compiler=${C_COMPILER:-clang}
cxx_compiler=${CXX_COMPILER:-clang++}

if [[ -n $TTMLIR_ENABLE_RUNTIME ]]; then
    cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=$build_type -DCMAKE_C_COMPILER=$c_compiler -DCMAKE_CXX_COMPILER=$cxx_compiler -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DTTMLIR_ENABLE_RUNTIME=ON
else
    cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=$build_type -DCMAKE_C_COMPILER=$c_compiler -DCMAKE_CXX_COMPILER=$cxx_compiler -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
fi

cmake --build build
