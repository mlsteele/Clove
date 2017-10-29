#[macro_use] extern crate colorify;
extern crate find_folder;
extern crate image;
extern crate ocl;
extern crate piston_window;
extern crate rand;
extern crate time;

mod gpu;
mod tracer;
mod common;

use rand::Rng;
use piston_window::{
    PistonWindow, WindowSettings, OpenGL,
    Texture, TextureSettings,
    Transformed, MouseCursorEvent, RenderEvent, UpdateEvent
};
use std::thread;
use std::sync::{Arc,Mutex};
use std::time::Duration;
use common::{Turn, Cursor};

fn main() {
    let dims: (u32, u32) = (1000, 500);

    let black: image::Rgba<u8> = image::Rgba{data: [0u8, 0u8, 0u8, 255u8]};
    let bg_color = [0.3, 0.0, 0.3, 1.];
    let img_blank: image::ImageBuffer<image::Rgba<u8>, Vec<u8>> = image::ImageBuffer::from_pixel(
        dims.0, dims.1, black);

    // This shared canvas is written to by the GPU thread
    // and read by the gui.
    let img_canvas_shared: Arc<Mutex<_>> = {
        Arc::new(Mutex::new(img_blank.clone()))
    };
    let turn_shared = Arc::new(Mutex::new(Turn::WantDisplay));

    let cursor_shared = Arc::new(Mutex::new(Default::default()));

    {
        let img_canvas_shared = Arc::clone(&img_canvas_shared);
        let cursor_shared = Arc::clone(&cursor_shared);
        let turn_shared = Arc::clone(&turn_shared);
        thread::spawn(move || {
            loop {
                let img_canvas_shared = Arc::clone(&img_canvas_shared);
                let cursor_shared = Arc::clone(&cursor_shared);
                let turn_shared = Arc::clone(&turn_shared);
                let gpu_thread = thread::spawn(move || {
                    gpu::run_gpu_loop(dims, img_canvas_shared, turn_shared, cursor_shared);
                });
                let _ = gpu_thread.join();
                thread::sleep(Duration::from_millis(150));
            }
        });
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
    let scaleup = 1.5;
    while let Some(e) = window.next() {
        e.update(|_| {
            // Fake raindrop cursor
            if rand::thread_rng().next_f32() < 0.9 {
                *cursor_shared.lock().unwrap() = Cursor{
                    enabled: true,
                    x: rand::thread_rng().next_u32() % dims.0,
                    y: rand::thread_rng().next_u32() % dims.1,
                    pressed: false,
                };
            }
        });

        e.mouse_cursor(|x,y| {
            *cursor_shared.lock().unwrap() = Cursor{
                enabled: true,
                x: (x / scaleup) as u32,
                y: (y / scaleup) as u32,
                pressed: false,
            };
        });

        e.render(|_| {
            let turn = {*turn_shared.lock().unwrap()};
            if turn == Turn::WantDisplay {
                {
                    let img_canvas = img_canvas_shared.lock().unwrap();
                    texture = Texture::from_image(&mut window.factory,
                                                &img_canvas,
                                                &TextureSettings::new()
                    ).unwrap()
                }
                *turn_shared.lock().unwrap() = Turn::WantData;
            }

            window.draw_2d(&e, |c, g| {
                piston_window::clear(bg_color, g);
                piston_window::image(&texture, c.transform.scale(scaleup, scaleup), g);
            });
        });
    }
}

