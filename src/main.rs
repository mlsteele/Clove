#[macro_use] extern crate colorify;
extern crate find_folder;
extern crate image;
extern crate ocl;
extern crate piston_window;
extern crate rand;
extern crate time;

use piston_window::{
    PistonWindow, WindowSettings, OpenGL, EventLoop,
    Texture, TextureSettings,
};
use std::thread;
use std::sync::{Arc,RwLock};

mod gpu;
mod tracer;

fn main() {
    let dims: (u32, u32) = (512, 512);

    let black: image::Rgba<u8> = image::Rgba{data: [0u8, 0u8, 0u8, 255u8]};
    let img_canvas_shared: Arc<_> = {
        let img_canvas: image::ImageBuffer<image::Rgba<u8>, Vec<u8>> = image::ImageBuffer::from_pixel(
            dims.0, dims.1, black);
        Arc::new(RwLock::new(img_canvas))
    };

    {
        let img_canvas_shared = Arc::clone(&img_canvas_shared);
        thread::spawn(move || {
            gpu::run_gpu_loop(img_canvas_shared);
        });
    }

    let opengl = OpenGL::V3_2;
    let mut window: PistonWindow =
        WindowSettings::new("piston: image", [300, 300])
        .exit_on_esc(true)
        .opengl(opengl)
        .build()
        .unwrap();

    // window.set_lazy(true);
    while let Some(e) = window.next() {
        let img_canvas = img_canvas_shared.read().unwrap();

        let texture = Texture::from_image(
                &mut window.factory,
                &img_canvas,
                &TextureSettings::new()
            ).unwrap();

        window.draw_2d(&e, |c, g| {
            piston_window::clear([1.,1.,0.,1.], g);
            piston_window::image(&texture, c.transform, g);
        });
    }
}

