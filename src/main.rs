#[macro_use] extern crate colorify;
extern crate find_folder;
extern crate image;
extern crate ocl;
extern crate piston_window;
extern crate rand;
extern crate time;

use piston_window::{
    PistonWindow, WindowSettings, OpenGL,
    Texture, TextureSettings,
    Transformed, MouseCursorEvent, RenderEvent,
};
use std::thread;
use std::sync::{Arc,Mutex};

mod gpu;
mod tracer;

fn main() {
    let dims: (u32, u32) = (512, 512);

    let black: image::Rgba<u8> = image::Rgba{data: [0u8, 0u8, 0u8, 255u8]};
    let img_blank: image::ImageBuffer<image::Rgba<u8>, Vec<u8>> = image::ImageBuffer::from_pixel(
        dims.0, dims.1, black);

    // This shared canvas is written to by the GPU thread
    // whenver it is None. And is read by the gui whenever
    // it is Some.
    let img_canvas_shared: Arc<Mutex<Option<_>>> = {
        Arc::new(Mutex::new(Some(img_blank.clone())))
    };

    let cursor_shared: Arc<Mutex<Option<(u32, u32)>>> = Arc::new(Mutex::new(None));

    {
        let img_canvas_shared = Arc::clone(&img_canvas_shared);
        let cursor_shared = Arc::clone(&cursor_shared);
        let _gpu_thread = thread::spawn(move || {
            gpu::run_gpu_loop(img_canvas_shared, cursor_shared);
        });
        // gpu_thread.join().unwrap();
    }

    // // Skip opengl
    // return;

    let opengl = OpenGL::V3_2;
    let mut window: PistonWindow =
        WindowSettings::new("piston: image", [300, 300])
        .exit_on_esc(true)
        .opengl(opengl)
        .build()
        .unwrap();

    let mut texture = Texture::from_image(&mut window.factory,
                                    &img_blank,
                                    &TextureSettings::new()
    ).unwrap();

    // window.set_lazy(true);
    let scaleup = 2.5;
    while let Some(e) = window.next() {
        e.mouse_cursor(|x,y| {
            *cursor_shared.lock().unwrap() = Some(
                ((x / scaleup) as u32,
                 (y / scaleup) as u32
                ));
        });

        e.render(|_| {
            let img_canvas: Option<_> = img_canvas_shared.lock().unwrap().take();
            if let Some(img_canvas) = img_canvas {
                texture = Texture::from_image(&mut window.factory,
                                            &img_canvas,
                                            &TextureSettings::new()
                ).unwrap()
            }

            window.draw_2d(&e, |c, g| {
                piston_window::clear([1.,1.,0.,1.], g);
                piston_window::image(&texture, c.transform.scale(scaleup, scaleup), g);
            });
        });
    }
}

