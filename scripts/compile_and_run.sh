gcc -o lidar_simulation main.c lidar_system.c rendering.c -I/opt/homebrew/opt/sdl2/include/SDL2 -L/opt/homebrew/opt/sdl2/lib -lSDL2 -lm && ./lidar_simulation


# gcc -o data_generator data_generator.c lidar_system.c -I/opt/homebrew/opt/sdl2/include/SDL2 -L/opt/homebrew/opt/sdl2/lib -lSDL2 -lm && ./data_generator




mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=$(brew --prefix pytorch) ..
make


gcc -o test_c test_c.c -L. -llidar_denoiser -I$(brew --prefix pytorch)/include -I./include