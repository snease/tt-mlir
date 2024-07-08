source env/activate

cmake -B env/build env
cmake --build env/build

build_type=${BUILD_TYPE:-Release}
c_compiler=${C_COMPILER:-clang}
cxx_compiler=${CXX_COMPILER:-clang++}

cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=$build_type -DCMAKE_C_COMPILER=$c_compiler -DCMAKE_CXX_COMPILER=$cxx_compiler
cmake --build build
