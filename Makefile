# Compiler and flags
CC = gcc
CXX = g++
CFLAGS = -Wall -Wextra -std=c11
CXXFLAGS = -Wall -Wextra -std=c++17

# Library paths (customize these based on your system)
SDL2_PREFIX = /opt/homebrew/opt/sdl2
PYTORCH_PREFIX = /opt/homebrew/opt/pytorch
RAYLIB_PREFIX = /opt/homebrew/opt/raylib

# Include paths
INCLUDES = -I$(SDL2_PREFIX)/include/SDL2 -I$(PYTORCH_PREFIX)/include -I$(RAYLIB_PREFIX)/include -Isrc/headers

# Library flags
LDFLAGS = -L$(SDL2_PREFIX)/lib -L$(PYTORCH_PREFIX)/lib -L$(RAYLIB_PREFIX)/lib
LIBS = -lSDL2 -lm -ltorch -lc10 -ltorch_cpu -lstdc++ -lraylib

# Source directories
C_SRCDIR = src/c
CPP_SRCDIR = src/cpp
HEADERDIR = src/headers
BINDIR = bin
OBJDIR = build/obj

# Create object file lists
C_SOURCES = $(wildcard $(C_SRCDIR)/*.c)
CPP_SOURCES = $(wildcard $(CPP_SRCDIR)/*.cpp)
C_OBJECTS = $(patsubst $(C_SRCDIR)/%.c,$(OBJDIR)/%.o,$(C_SOURCES))
CPP_OBJECTS = $(patsubst $(CPP_SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(CPP_SOURCES))

# Executables
EXECUTABLES = $(BINDIR)/lidar_simulation $(BINDIR)/data_generator $(BINDIR)/test_c

# Default target
all: $(EXECUTABLES)

# Create bin directory if it doesn't exist
$(BINDIR):
	mkdir -p $(BINDIR)

# Create obj directory if it doesn't exist
$(OBJDIR):
	mkdir -p $(OBJDIR)

# C compilation rule
$(OBJDIR)/%.o: $(C_SRCDIR)/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# C++ compilation rule
$(OBJDIR)/%.o: $(CPP_SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# lidar_simulation (C version)
$(BINDIR)/lidar_simulation: $(OBJDIR)/main.o $(OBJDIR)/lidar_system.o $(OBJDIR)/rendering.o $(OBJDIR)/lidar_denoiser_integration.o $(OBJDIR)/lidar_denoiser_c.o $(OBJDIR)/lidar_denoiser.o  | $(BINDIR)
	$(CC) $^ -o $@ $(LDFLAGS) $(LIBS)

# data_generator
$(BINDIR)/data_generator: $(OBJDIR)/data_generator.o $(OBJDIR)/lidar_system.o | $(BINDIR)
	$(CC) $^ -o $@ $(LDFLAGS) $(LIBS)

# data_generator_temporal
$(BINDIR)/data_generator_temporal: $(OBJDIR)/data_generator_temporal.o $(OBJDIR)/lidar_system.o | $(BINDIR)
	$(CC) $^ -o $@ $(LDFLAGS) $(LIBS)

# test_c
$(BINDIR)/test_c: $(OBJDIR)/test_c.o $(OBJDIR)/lidar_denoiser_c.o $(OBJDIR)/lidar_denoiser.o | $(BINDIR)
	$(CC) $^ -o $@ $(LDFLAGS) $(LIBS)


# Clean build artifacts
clean:
	rm -rf $(OBJDIR) $(BINDIR)

# Install dependencies (macOS-specific)
install-deps:
	brew install sdl2 pytorch raylib
	
# Run lidar simulation
run-simulation: $(BINDIR)/lidar_simulation
	./$(BINDIR)/lidar_simulation

# Run data generator
run-data-gen: $(BINDIR)/data_generator
	./$(BINDIR)/data_generator

run-data-gen-temporal: $(BINDIR)/data_generator_temporal
	./$(BINDIR)/data_generator_temporal

run-test: $(BINDIR)/test_c
	./$(BINDIR)/test_c

print-sources:
	@echo "C_SRCDIR: $(C_SRCDIR)"
	@echo "CPP_SRCDIR: $(CPP_SRCDIR)"
	@echo "C_SOURCES: $(C_SOURCES)"
	@echo "CPP_SOURCES: $(CPP_SOURCES)"
	@echo "C_OBJECTS: $(C_OBJECTS)"
	@echo "CPP_OBJECTS: $(CPP_OBJECTS)"
	
.PHONY: all clean install-deps run-simulation run-data-gen