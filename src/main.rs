extern crate rand;
extern crate find_folder;
extern crate image;
extern crate time;
extern crate ocl;
#[macro_use] extern crate colorify;

mod gpu;
mod tracer;

fn main() {
    gpu::run_gpu_loop();
}
