# Compiler
#CXX = /opt/rocm/llvm/bin/clang++
CXX = /home/yaxunl/git/llvm2/assert/bin/clang++
# Compiler flags
#CXXFLAGS = --offload-arch=gfx1100 -save-temps -Wall -Wextra -std=c++17 -O3 -mllvm -enable-misched=false -mllvm -enable-post-misched=false -ggdb -Wno-unused-parameter
CXXFLAGS = --offload-arch=gfx1100 --rocm-path=/home/yaxunl/buildbot/Externals/hip/rocm-6.0.0 -DNDEBUG -Wall -Wextra -std=c++17 -O3 -Wno-unused-parameter

# Program name
PROGRAM = TheNextWeek

# Source files
SOURCES = main.cc
# Object files
OBJECTS = $(SOURCES:.cc=.o)

# Build target
$(PROGRAM): $(OBJECTS)
	$(CXX) $(CXXFLAGS) --rocm-path=/home/yaxunl/buildbot/Externals/hip/rocm-6.0.0 --hip-link -rtlib=compiler-rt -unwindlib=libgcc -frtlib-add-rpath -o $(PROGRAM) $(OBJECTS)

# Rule for building object files
%.o: %.cc
	$(CXX) $(CXXFLAGS) -xhip -c $< -o $@

main.o : main.cc camera.h  color.h  hittable.h  hittable_list.h  interval.h  material.h  \
   PPMImageFile.h  ray.h  rtweekend.h  rtw_stb_image.h  sphere.h  vec3.h DeviceArray.h \
   Makefile SharedPtr.h Vector.h aabb.h String.h Map.h List.h bvh.h constant_medium.h \
   hipUtils.h quad.h texture.h

# Clean target
clean:
	rm -f $(PROGRAM) $(OBJECTS)

debug: $(PROGRAM)
	/opt/rocm/bin/rocgdb --args ./TheNextWeek -s two_spheres

test: $(PROGRAM)
	AMD_LOG_LEVEL=3 ./TheNextWeek -s two_spheres

# PHONY targets
.PHONY: clean debug test

