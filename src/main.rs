extern crate camera_capture;
#[macro_use] extern crate colorify;
extern crate find_folder;
extern crate image;
extern crate ocl;
extern crate piston_window;
extern crate rand;
extern crate time;

mod gpu;
mod tracer;
mod cam;
mod common;

use rand::Rng;
use piston_window::{
    PistonWindow, WindowSettings, OpenGL,
    Texture, TextureSettings, Transformed,
    MouseCursorEvent, RenderEvent, UpdateEvent, ReleaseEvent, ButtonEvent,
    Button, ButtonState, MouseButton, Key,
};
use std::thread;
use std::sync::{Arc,Mutex};
use std::time::Duration;
use std::sync::mpsc;
use common::{Turn, Cursor};

const CAM_ENABLE: bool = false;

fn main() {
    // let dims: (u32, u32) = (848, 480); // cam dims
    // let dims: (u32, u32) = (1000, 500); // elephant dims
    // let dims: (u32, u32) = (1920 / 2, 1080 / 2);
    let dims: (u32, u32) = (1000, 1000);

    #[allow(unused_variables)]
    let black: image::Rgba<u8> = image::Rgba{data: [0u8, 0u8, 0u8, 255u8]};
    #[allow(unused_variables)]
    let white: image::Rgba<u8> = image::Rgba{data: [255u8, 255u8, 255u8, 255u8]};
    let bg_color = [0.3, 0.0, 0.3, 1.];
    let img_blank: image::ImageBuffer<image::Rgba<u8>, Vec<u8>> = image::ImageBuffer::from_pixel(
        dims.0, dims.1, white);

    // This shared canvas is written to by the GPU thread
    // and read by the gui.
    let img_canvas_shared: Arc<Mutex<_>> = {
        Arc::new(Mutex::new(img_blank.clone()))
    };
    let turn_shared = Arc::new(Mutex::new(Turn::WantDisplay));

    let cursor_shared = Arc::new(Mutex::new(Default::default()));

    // Start the cam loop
    let cam_receiver = {
        let (tx, rx) = mpsc::sync_channel(1);
        thread::Builder::new().name("cam".to_owned()).spawn(move || {
            if CAM_ENABLE {
                // Disable the cam
                cam::cam_loop(dims, tx);
            }
        }).unwrap();
        rx
    };

    let (gpu_stop_sender, gpu_stop_receiver) = mpsc::sync_channel(1);

    // Test the cam. Exceptions are easier to debug from out here.
    // let _ = cam_receiver.recv().expect("cam img");

    // Start the gpu loop (with supervisor wrapper)
    {
        let img_canvas_shared = Arc::clone(&img_canvas_shared);
        let cursor_shared = Arc::clone(&cursor_shared);
        let turn_shared = Arc::clone(&turn_shared);
        let cam_receiver = Arc::new(Mutex::new(cam_receiver));
        let stop_receiver = Arc::new(Mutex::new(gpu_stop_receiver));
        thread::Builder::new().name("gpu-outer".to_owned()).spawn(move || {
            let cam_receiver = Arc::clone(&cam_receiver);
            loop {
                let img_canvas_shared = Arc::clone(&img_canvas_shared);
                let cursor_shared = Arc::clone(&cursor_shared);
                let turn_shared = Arc::clone(&turn_shared);
                let cam_receiver = Arc::clone(&cam_receiver);
                let stop_receiver = Arc::clone(&stop_receiver);
                let gpu_thread = thread::Builder::new().name("gpu-inner".to_owned()).spawn(move || {
                    gpu::run_gpu_loop(
                        dims,
                        img_canvas_shared,
                        turn_shared,
                        cursor_shared,
                        cam_receiver,
                        Some(stop_receiver),
                    );
                }).unwrap();
                let _ = gpu_thread.join();
                thread::sleep(Duration::from_millis(50));
            }
        }).unwrap();
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
    // let scaleup = 1.5;
    let scaleup = 1.0;
    while let Some(e) = window.next() {
        e.update(|_| {
            const FAKE_MOUSE: bool = false;
            // Fake raindrop cursor
            if FAKE_MOUSE && rand::thread_rng().next_f32() < 0.9 {
                *cursor_shared.lock().unwrap() = Cursor{
                    enabled: true,
                    x: rand::thread_rng().next_u32() % dims.0,
                    y: rand::thread_rng().next_u32() % dims.1,
                    pressed: false,
                };
            }
        });

        e.mouse_cursor(|x,y| {
            let mut c = cursor_shared.lock().unwrap();
            c.enabled = true;
            c.x = (x / scaleup) as u32;
            c.y = (y / scaleup) as u32;
        });

        e.button(|arg| {
            if arg.button == Button::Mouse(MouseButton::Left) {
                cursor_shared.lock().unwrap().pressed = match arg.state {
                    ButtonState::Press   => true,
                    ButtonState::Release => false,
                }
            }
        });

        e.release(|button| {
            if button == Button::Keyboard(Key::R) {
                let _ = gpu_stop_sender.send(());

                printlnc!(red: "reload");
            }
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

    let _ = gpu_stop_sender.send(());
}
