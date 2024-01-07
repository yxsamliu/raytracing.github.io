//==============================================================================================
// Originally written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================

#include "rtweekend.h"

#include "bvh.h"
#include "camera.h"
#include "color.h"
#include "constant_medium.h"
#include "hittable_list.h"
#include "material.h"
#include "quad.h"
#include "sphere.h"
#include "texture.h"
#include <functional>
#include <map>

class Scene {
public:
  void render(const std::string &file_name) { cam.render(world, file_name); }

protected:
  hittable_list world;
  camera cam;
};

TestConfig Cfg;

template <typename S> class Test {
public:
  static void run(int image_width, int samples_per_pixel, int max_depth,
                  const std::string &file_name) {
    S s(image_width, samples_per_pixel, max_depth);
    s.render(file_name);
  }
  static void run(const std::string &file_name) {
    S s;
    s.render(file_name);
  }
};

class random_spheres : public Scene {
public:
  random_spheres() {
    unsigned rng = 0;
    auto checker =
        makeShared<checker_texture>(0.32, color(.2, .3, .1), color(.9, .9, .9));
    world.add(makeShared<sphere>(point3(0, -1000, 0), 1000,
                                 makeShared<lambertian>(checker)));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
          auto choose_mat = random_double(rng);
          point3 center(a + 0.9 * random_double(rng), 0.2,
                        b + 0.9 * random_double(rng));

          if ((center - point3(4, 0.2, 0)).length() > 0.9) {
            SharedPtr<material> sphere_material;

            if (choose_mat < 0.8) {
              // diffuse
              auto albedo = color::random(rng) * color::random(rng);
              sphere_material = makeShared<lambertian>(albedo);
              auto center2 = center + vec3(0, random_double(0, .5, rng), 0);
              world.add(
                  makeShared<sphere>(center, center2, 0.2, sphere_material));
            } else if (choose_mat < 0.95) {
              // metal
              auto albedo = color::random(0.5, 1, rng);
              auto fuzz = random_double(0, 0.5, rng);
              sphere_material = makeShared<metal>(albedo, fuzz);
              world.add(makeShared<sphere>(center, 0.2, sphere_material));
            } else {
              // glass
              sphere_material = makeShared<dielectric>(1.5);
              world.add(makeShared<sphere>(center, 0.2, sphere_material));
            }
            }
        }
    }

    auto material1 = makeShared<dielectric>(1.5);
    world.add(makeShared<sphere>(point3(0, 1, 0), 1.0, material1));

    auto material2 = makeShared<lambertian>(color(0.4, 0.2, 0.1));
    world.add(makeShared<sphere>(point3(-4, 1, 0), 1.0, material2));

    auto material3 = makeShared<metal>(color(0.7, 0.6, 0.5), 0.0);
    world.add(makeShared<sphere>(point3(4, 1, 0), 1.0, material3));

    world = hittable_list(makeShared<bvh_node>(world, rng));

    cam.aspect_ratio      = 16.0 / 9.0;
    cam.image_width       = 400;
    cam.samples_per_pixel = 100;
    cam.max_depth         = 50;
    cam.background        = color(0.70, 0.80, 1.00);

    cam.vfov     = 20;
    cam.lookfrom = point3(13,2,3);
    cam.lookat   = point3(0,0,0);
    cam.vup      = vec3(0,1,0);

    cam.defocus_angle = 0.02;
    cam.focus_dist    = 10.0;
  }
};

class two_spheres : public Scene {
public:
  two_spheres() {
    auto checker =
        makeShared<checker_texture>(0.8, color(.2, .3, .1), color(.9, .9, .9));

    world.add(makeShared<sphere>(point3(0, -10, 0), 10,
                                 makeShared<lambertian>(checker)));
    world.add(makeShared<sphere>(point3(0, 10, 0), 10,
                                 makeShared<lambertian>(checker)));

    cam.aspect_ratio      = 16.0 / 9.0;
    cam.image_width       = 400;
    cam.samples_per_pixel = 100;
    cam.max_depth         = 50;
    cam.background        = color(0.70, 0.80, 1.00);

    cam.vfov     = 20;
    cam.lookfrom = point3(13,2,3);
    cam.lookat   = point3(0,0,0);
    cam.vup      = vec3(0,1,0);

    cam.defocus_angle = 0;
  }
};

class earth : public Scene {
public:
  earth() {
    auto earth_texture = makeShared<image_texture>("earthmap.jpg");
    auto earth_surface = makeShared<lambertian>(earth_texture);
    auto globe = makeShared<sphere>(point3(0, 0, 0), 2, earth_surface);

    cam.aspect_ratio      = 16.0 / 9.0;
    cam.image_width       = 400;
    cam.samples_per_pixel = 100;
    cam.max_depth         = 50;
    cam.background        = color(0.70, 0.80, 1.00);

    cam.vfov     = 20;
    cam.lookfrom = point3(0,0,12);
    cam.lookat   = point3(0,0,0);
    cam.vup      = vec3(0,1,0);

    cam.defocus_angle = 0;

    world.add(globe);
  }
};

class two_perlin_spheres : public Scene {
public:
  two_perlin_spheres() {
    unsigned rng = 0;
    auto pertext = makeShared<noise_texture>(4, rng);
    world.add(makeShared<sphere>(point3(0, -1000, 0), 1000,
                                 makeShared<lambertian>(pertext)));
    world.add(makeShared<sphere>(point3(0, 2, 0), 2,
                                 makeShared<lambertian>(pertext)));

    cam.aspect_ratio      = 16.0 / 9.0;
    cam.image_width       = 400;
    cam.samples_per_pixel = 100;
    cam.max_depth         = 50;
    cam.background        = color(0.70, 0.80, 1.00);

    cam.vfov     = 20;
    cam.lookfrom = point3(13,2,3);
    cam.lookat   = point3(0,0,0);
    cam.vup      = vec3(0,1,0);

    cam.defocus_angle = 0;
  }
};

class quads : public Scene {
public:
  quads() {
    // Materials
    auto left_red = makeShared<lambertian>(color(1.0, 0.2, 0.2));
    auto back_green = makeShared<lambertian>(color(0.2, 1.0, 0.2));
    auto right_blue = makeShared<lambertian>(color(0.2, 0.2, 1.0));
    auto upper_orange = makeShared<lambertian>(color(1.0, 0.5, 0.0));
    auto lower_teal = makeShared<lambertian>(color(0.2, 0.8, 0.8));

    // Quads
    world.add(makeShared<quad>(point3(-3, -2, 5), vec3(0, 0, -4), vec3(0, 4, 0),
                               left_red));
    world.add(makeShared<quad>(point3(-2, -2, 0), vec3(4, 0, 0), vec3(0, 4, 0),
                               back_green));
    world.add(makeShared<quad>(point3(3, -2, 1), vec3(0, 0, 4), vec3(0, 4, 0),
                               right_blue));
    world.add(makeShared<quad>(point3(-2, 3, 1), vec3(4, 0, 0), vec3(0, 0, 4),
                               upper_orange));
    world.add(makeShared<quad>(point3(-2, -3, 5), vec3(4, 0, 0), vec3(0, 0, -4),
                               lower_teal));

    cam.aspect_ratio      = 1.0;
    cam.image_width       = 400;
    cam.samples_per_pixel = 100;
    cam.max_depth         = 50;
    cam.background        = color(0.70, 0.80, 1.00);

    cam.vfov     = 80;
    cam.lookfrom = point3(0,0,9);
    cam.lookat   = point3(0,0,0);
    cam.vup      = vec3(0,1,0);

    cam.defocus_angle = 0;
  }
};

class simple_light : public Scene {
public:
  simple_light() {
    unsigned rng = 0;
    auto pertext = makeShared<noise_texture>(4, rng);
    world.add(makeShared<sphere>(point3(0, -1000, 0), 1000,
                                 makeShared<lambertian>(pertext)));
    world.add(makeShared<sphere>(point3(0, 2, 0), 2,
                                 makeShared<lambertian>(pertext)));

    auto difflight = makeShared<diffuse_light>(color(4, 4, 4));
    world.add(makeShared<sphere>(point3(0, 7, 0), 2, difflight));
    world.add(makeShared<quad>(point3(3, 1, -2), vec3(2, 0, 0), vec3(0, 2, 0),
                               difflight));

    cam.aspect_ratio      = 16.0 / 9.0;
    cam.image_width       = 400;
    cam.samples_per_pixel = 100;
    cam.max_depth         = 50;
    cam.background        = color(0,0,0);

    cam.vfov     = 20;
    cam.lookfrom = point3(26,3,6);
    cam.lookat   = point3(0,2,0);
    cam.vup      = vec3(0,1,0);

    cam.defocus_angle = 0;
  }
};

class cornell_box : public Scene {
public:
  cornell_box() {
    auto red = makeShared<lambertian>(color(.65, .05, .05));
    auto white = makeShared<lambertian>(color(.73, .73, .73));
    auto green = makeShared<lambertian>(color(.12, .45, .15));
    auto light = makeShared<diffuse_light>(color(15, 15, 15));

    world.add(makeShared<quad>(point3(555, 0, 0), vec3(0, 555, 0),
                               vec3(0, 0, 555), green));
    world.add(makeShared<quad>(point3(0, 0, 0), vec3(0, 555, 0),
                               vec3(0, 0, 555), red));
    world.add(makeShared<quad>(point3(343, 554, 332), vec3(-130, 0, 0),
                               vec3(0, 0, -105), light));
    world.add(makeShared<quad>(point3(0, 0, 0), vec3(555, 0, 0),
                               vec3(0, 0, 555), white));
    world.add(makeShared<quad>(point3(555, 555, 555), vec3(-555, 0, 0),
                               vec3(0, 0, -555), white));
    world.add(makeShared<quad>(point3(0, 0, 555), vec3(555, 0, 0),
                               vec3(0, 555, 0), white));

    SharedPtr<hittable> box1 =
        box(point3(0, 0, 0), point3(165, 330, 165), white);
    box1 = makeShared<rotate_y>(box1, 15);
    box1 = makeShared<translate>(box1, vec3(265, 0, 295));
    world.add(box1);

    SharedPtr<hittable> box2 =
        box(point3(0, 0, 0), point3(165, 165, 165), white);
    box2 = makeShared<rotate_y>(box2, -18);
    box2 = makeShared<translate>(box2, vec3(130, 0, 65));
    world.add(box2);

    cam.aspect_ratio      = 1.0;
    cam.image_width       = 600;
    cam.samples_per_pixel = 200;
    cam.max_depth         = 50;
    cam.background        = color(0,0,0);

    cam.vfov     = 40;
    cam.lookfrom = point3(278, 278, -800);
    cam.lookat   = point3(278, 278, 0);
    cam.vup      = vec3(0,1,0);

    cam.defocus_angle = 0;
  }
};

class cornell_smoke : public Scene {
public:
  cornell_smoke() {
    auto red = makeShared<lambertian>(color(.65, .05, .05));
    auto white = makeShared<lambertian>(color(.73, .73, .73));
    auto green = makeShared<lambertian>(color(.12, .45, .15));
    auto light = makeShared<diffuse_light>(color(7, 7, 7));

    world.add(makeShared<quad>(point3(555, 0, 0), vec3(0, 555, 0),
                               vec3(0, 0, 555), green));
    world.add(makeShared<quad>(point3(0, 0, 0), vec3(0, 555, 0),
                               vec3(0, 0, 555), red));
    world.add(makeShared<quad>(point3(113, 554, 127), vec3(330, 0, 0),
                               vec3(0, 0, 305), light));
    world.add(makeShared<quad>(point3(0, 555, 0), vec3(555, 0, 0),
                               vec3(0, 0, 555), white));
    world.add(makeShared<quad>(point3(0, 0, 0), vec3(555, 0, 0),
                               vec3(0, 0, 555), white));
    world.add(makeShared<quad>(point3(0, 0, 555), vec3(555, 0, 0),
                               vec3(0, 555, 0), white));

    SharedPtr<hittable> box1 =
        box(point3(0, 0, 0), point3(165, 330, 165), white);
    box1 = makeShared<rotate_y>(box1, 15);
    box1 = makeShared<translate>(box1, vec3(265, 0, 295));

    SharedPtr<hittable> box2 =
        box(point3(0, 0, 0), point3(165, 165, 165), white);
    box2 = makeShared<rotate_y>(box2, -18);
    box2 = makeShared<translate>(box2, vec3(130, 0, 65));

    world.add(makeShared<constant_medium>(box1, 0.01, color(0, 0, 0)));
    world.add(makeShared<constant_medium>(box2, 0.01, color(1, 1, 1)));

    cam.aspect_ratio      = 1.0;
    cam.image_width       = 600;
    cam.samples_per_pixel = 200;
    cam.max_depth         = 50;
    cam.background        = color(0,0,0);

    cam.vfov     = 40;
    cam.lookfrom = point3(278, 278, -800);
    cam.lookat   = point3(278, 278, 0);
    cam.vup      = vec3(0,1,0);

    cam.defocus_angle = 0;
  }
};

class final_scene : public Scene {
public:
  final_scene(int image_width, int samples_per_pixel, int max_depth) {
    unsigned rng = 0;
    hittable_list boxes1;
    auto ground = makeShared<lambertian>(color(0.48, 0.83, 0.53));

    int boxes_per_side = 20;
    for (int i = 0; i < boxes_per_side; i++) {
        for (int j = 0; j < boxes_per_side; j++) {
            auto w = 100.0;
            auto x0 = -1000.0 + i*w;
            auto z0 = -1000.0 + j*w;
            auto y0 = 0.0;
            auto x1 = x0 + w;
            auto y1 = random_double(1, 101, rng);
            auto z1 = z0 + w;

            boxes1.add(box(point3(x0,y0,z0), point3(x1,y1,z1), ground));
        }
    }

    world.add(makeShared<bvh_node>(boxes1, rng));

    auto light = makeShared<diffuse_light>(color(7, 7, 7));
    world.add(makeShared<quad>(point3(123, 554, 147), vec3(300, 0, 0),
                               vec3(0, 0, 265), light));

    auto center1 = point3(400, 400, 200);
    auto center2 = center1 + vec3(30,0,0);
    auto sphere_material = makeShared<lambertian>(color(0.7, 0.3, 0.1));
    world.add(makeShared<sphere>(center1, center2, 50, sphere_material));

    world.add(makeShared<sphere>(point3(260, 150, 45), 50,
                                 makeShared<dielectric>(1.5)));
    world.add(makeShared<sphere>(point3(0, 150, 145), 50,
                                 makeShared<metal>(color(0.8, 0.8, 0.9), 1.0)));

    auto boundary = makeShared<sphere>(point3(360, 150, 145), 70,
                                       makeShared<dielectric>(1.5));
    world.add(boundary);
    world.add(makeShared<constant_medium>(boundary, 0.2, color(0.2, 0.4, 0.9)));
    boundary =
        makeShared<sphere>(point3(0, 0, 0), 5000, makeShared<dielectric>(1.5));
    world.add(makeShared<constant_medium>(boundary, .0001, color(1, 1, 1)));

    auto emat =
        makeShared<lambertian>(makeShared<image_texture>("earthmap.jpg"));
    world.add(makeShared<sphere>(point3(400, 200, 400), 100, emat));
    auto pertext = makeShared<noise_texture>(0.1, rng);
    world.add(makeShared<sphere>(point3(220, 280, 300), 80,
                                 makeShared<lambertian>(pertext)));

    hittable_list boxes2;
    auto white = makeShared<lambertian>(color(.73, .73, .73));
    int ns = 1000;
    for (int j = 0; j < ns; j++) {
      boxes2.add(makeShared<sphere>(point3::random(0, 165, rng), 10, white));
    }

    world.add(makeShared<translate>(
        makeShared<rotate_y>(makeShared<bvh_node>(boxes2, rng), 15),
        vec3(-100, 270, 395)));

    cam.aspect_ratio      = 1.0;
    cam.image_width       = image_width;
    cam.samples_per_pixel = samples_per_pixel;
    cam.max_depth         = max_depth;
    cam.background        = color(0,0,0);

    cam.vfov     = 40;
    cam.lookfrom = point3(478, 278, -600);
    cam.lookat   = point3(278, 278, 0);
    cam.vup      = vec3(0,1,0);

    cam.defocus_angle = 0;
  }
};

int main(int argc, char *argv[]) {
  std::string scene = "final_coarse"; // default scene
  bool runAll = false;

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "-s") {
      if (i + 1 < argc) {
        scene = argv[++i];
        if (scene == "all") {
          runAll = true;
          break;
        }
      }
    } else if (strcmp(argv[i], "-t") == 0) {
      Cfg.output_time = true;
    } else if (strcmp(argv[i], "-c") == 0) {
      Cfg.compare_cpu = true;
    }
  }
  // Vector of pairs to store scene names and their corresponding functions
  std::vector<std::pair<std::string, std::function<void()>>> scenes = {
      {"quads", []() { Test<quads>::run("quads"); }},
      {"two_spheres", []() { Test<two_spheres>::run("two_spheres"); }},
      {"earth", []() { Test<earth>::run("earth"); }},
      {"two_perlin_spheres",
       []() { Test<two_perlin_spheres>::run("two_perlin_spheres"); }},
      {"simple_light", []() { Test<simple_light>::run("simple_light"); }},
      {"random_spheres", []() { Test<random_spheres>::run("random_spheres"); }},
      {"cornell_box", []() { Test<cornell_box>::run("cornell_box"); }},
      {"cornell_smoke", []() { Test<cornell_smoke>::run("cornell_smoke"); }},
      {"final_coarse",
       []() { Test<final_scene>::run(400, 250, 4, "final_coarse"); }},
      {"final_detailed",
       []() { Test<final_scene>::run(800, 10000, 40, "final_detailed"); }},
  };

  if (runAll) {
    // Run all tests
    for (auto &test : scenes) {
      std::cout << "Running " << test.first << std::endl;
      test.second();
    }
  } else {
    // Find and run the specified scene
    bool found = false;
    for (const auto &pair : scenes) {
      if (pair.first == scene) {
        pair.second(); // Run the scene
        found = true;
        break;
      }
    }
    if (!found) {
      std::cerr << "Unknown scene: " << scene << std::endl;
      return 1;
    }
  }

  return 0;
}
