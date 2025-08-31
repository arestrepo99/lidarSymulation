cp ./build/liblidar_denoiser.dylib .

export DYLD_LIBRARY_PATH=.:$DYLD_LIBRARY_PATH

# Find where LibTorch is installed
LIBTORCH_PATH=$(brew --prefix pytorch)

# Compile the C test file
gcc -o test_c test_c.c \
    -L./build -llidar_denoiser \
    -I. \
    -L${LIBTORCH_PATH}/lib -ltorch -lc10 \
    -arch arm64 \
    -rpath ${LIBTORCH_PATH}/lib \
    -Wl,-rpath,@loader_path