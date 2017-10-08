#[macro_use] extern crate colorify;
extern crate find_folder;
extern crate image;
extern crate ocl;
extern crate piston_window;
extern crate rand;
extern crate time;

use piston_window::{
    PistonWindow, WindowSettings, OpenGL, EventLoop,
    Flip, Texture, TextureSettings,
};

mod gpu;
mod tracer;

fn main() {
    // gpu::run_gpu_loop();

    let opengl = OpenGL::V3_2;
    let mut window: PistonWindow =
        WindowSettings::new("piston: image", [300, 300])
        .exit_on_esc(true)
        .opengl(opengl)
        .build()
        .unwrap();

    let black: image::Rgba<u8> = image::Rgba{data: [0u8, 0u8, 0u8, 255u8]};
    let img_canvas: image::ImageBuffer<image::Rgba<u8>, Vec<u8>> = image::ImageBuffer::from_pixel(
        512, 512, black);

    // let assets = find_folder::Search::ParentsThenKids(3, 3)
    //     .for_folder("assets").unwrap();
    // let rust_logo = assets.join("rust.png");
    // let rust_logo = Texture::from_path(
    //         &mut window.factory,
    //         &rust_logo,
    //         Flip::None,
    //         &TextureSettings::new()
    //     ).unwrap();
    let texture = Texture::from_image(
            &mut window.factory,
            &img_canvas,
            &TextureSettings::new()
        ).unwrap();
    window.set_lazy(true);
    while let Some(e) = window.next() {
        window.draw_2d(&e, |c, g| {
            piston_window::clear([1.,1.,0.,1.], g);
            piston_window::image(&texture, c.transform, g);
        });
    }
}

