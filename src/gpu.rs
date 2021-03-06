use image;
use ocl;
use rand;
use std::path::Path;
use std::sync::mpsc::TryRecvError;
use ocl::{Context, Queue, Device, Program, Image, Kernel};
use ocl::enums::{ImageChannelOrder, ImageChannelDataType, MemObjectType};
use find_folder::Search;
use rand::Rng;
use std::collections::vec_deque::VecDeque;
use tracer::TimeTracer;
use std::sync::{Arc,Mutex};
use std::time;
use common::{Turn, Cursor};
use std::sync::mpsc;
use cam;
use cam::{CamImg};

type MaskVal = u8;
const MASK_ZERO: image::Luma<MaskVal> = image::Luma([0]);

#[allow(dead_code)]
fn read_source_image(loco : &str) -> image::ImageBuffer<image::Rgba<u8>, Vec<u8>> {
    let dyn = image::open(&Path::new(loco)).unwrap();
    let img = dyn.to_rgba();
    img
}

#[allow(dead_code)]
fn min_pixel<I>(img: &I) -> (u32, u32, u16)
    where I: image::GenericImage<Pixel=image::Luma<u16>>
{
    let (x,y,px) = img.pixels().min_by_key(|&(_,_,v)| v[0]).unwrap();
    (x, y, px[0])
}

// Pixels sorted by value.
// Only consider those pixels where mask is hot.
// Returns list of (x, y, pixel_value)
fn sort_pixels_with_mask<I,M>(img: &I, mask: &M) -> Vec<(u32, u32, u16)>
    where I: image::GenericImage<Pixel=image::Luma<u16>>,
          M: image::GenericImage<Pixel=image::Luma<u8>>
{
    let mut choices: Vec<(u32,u32,u16)> = img.pixels().zip(mask.pixels()).filter_map(|((x,y,px), (_,_,map_px))| {
        if map_px[0] > 127u8 {
            Some((x,y,px[0]))
        } else {
            None
        }
    }).collect();
    // .min_by_key(|&(_,_,px)| px)
    choices.sort_unstable_by_key(|&(_,_,px)| px);
    choices
}

// Minimum value pixel in the img.
// Only consider those pixels where mask is hot.
// Returns None if there are no viable pixels.
#[allow(dead_code)]
fn min_pixel_with_mask<I,M>(img: &I, mask: &M) -> Option<(u32, u32, u16)>
    where I: image::GenericImage<Pixel=image::Luma<u16>>,
          M: image::GenericImage<Pixel=image::Luma<u8>>
{
    let choices = sort_pixels_with_mask(img, mask);
    choices.first().map(|x| *x)
}

// The neighbors of the position that are not filled
// and in bounds.
fn neighbors_empty<M>(x: u32, y: u32, mask_filled: &M) -> Vec<(u32,u32)>
    where M: image::GenericImage<Pixel=image::Luma<u8>>
{
    let x = x as i32;
    let y = y as i32;
    let mut neighbors = vec![];
    let (w, h) = mask_filled.dimensions();
    let w = w as i32;
    let h = h as i32;
    for dx in -1..2 {
        for dy in -1..2 {
            let nx = x + dx;
            let ny = y + dy;
            let is_self = dx == 0 && dy == 0;
            let in_bounds = nx >= 0 && ny >= 0 && nx < w && ny < h;
            // printlnc!(green: "n {} {}    (self:{})  (bounds:{})", nx, ny, is_self, in_bounds);
            if !is_self && in_bounds {
                let nx = nx as u32;
                let ny = ny as u32;
                if mask_filled.get_pixel(nx,ny)[0] <= 127u8 {
                    neighbors.push((nx, ny));
                }
            }
        }
    }
    neighbors
}

// Check whether the point (x,y) falls within bounds of a
// rect from (0,0) to (width,height).
#[allow(dead_code)]
fn in_bounds<T>(width: T, height: T, x: T, y: T) -> bool
    where T: Ord + Default
{
    let zero: T = Default::default();
    if x >= zero && y >= zero && x < width && y < height {
        return true;
    }
    return false;
}

fn duration_millis(d: &time::Duration) -> i64 {
    const MILLIS_PER_SEC: i64 = 1000;
    const NANOS_PER_MILLI: i32 = 1000_000;
    return (d.as_secs() as i64 * MILLIS_PER_SEC) + (d.subsec_nanos() as i64 / NANOS_PER_MILLI as i64);
}

pub fn run_gpu_loop(
    dims: (u32, u32),
    img_canvas_shared: Arc<Mutex<image::ImageBuffer<image::Rgba<u8>, Vec<u8>>>>,
    turn_shared: Arc<Mutex<Turn>>,
    cursor_shared: Arc<Mutex<Cursor>>,
    cam_rx: Arc<Mutex<mpsc::Receiver<CamImg>>>,
    stop_rx: Option<Arc<Mutex<mpsc::Receiver<()>>>>,
) {
    let mut rng = rand::thread_rng();
    let compute_program = Search::ParentsThenKids(3, 3)
        .for_folder("cl").expect("Error locating 'cl'")
        .join("main.cl");

    println!("getting ocl context...");
    let context = Context::builder().devices(Device::specifier()
        .type_flags(ocl::flags::DEVICE_TYPE_GPU).first()).build().unwrap();
    println!("devices: ({}) {:?}", context.devices().len(), context.devices());
    let device = context.devices()[0];
    println!("device: {:?}", device);
    println!("  {:?}", device.info(ocl::enums::DeviceInfo::Name));
    println!("  {:?}", device.info(ocl::enums::DeviceInfo::Vendor));
    println!("  {:?}", device.info(ocl::enums::DeviceInfo::VendorId));
    println!("  {:?}", device.info(ocl::enums::DeviceInfo::Type));
    println!("  {:?}", device.info(ocl::enums::DeviceInfo::Extensions));
    println!("  {:?}", device.info(ocl::enums::DeviceInfo::OpenclCVersion));
    let queue = Queue::new(&context, device, None).unwrap();

    let program = Program::builder()
        .src_file(compute_program)
        .devices(device)
        .build(&context)
        .unwrap();

    let mut img_subject: image::ImageBuffer<image::Rgba<u8>, Vec<u8>> = {
        // TODO subject may not be correct dims
        if false {
            // Subject is an elephant
            image::open(&Path::new("resources/elephant.jpg"))
                .expect("load subject")
                .to_rgba()
        } else {
            // Subject is the camera
            // TODO subject may not be correct dims
            cam::convert(cam_rx.lock().unwrap().recv().expect("cam recv"))
        }
    };

    let center = (dims.0 / 2, dims.1 / 2);

    #[allow(unused_variables)]
    let black: image::Rgba<u8> = image::Rgba([0u8, 0u8, 0u8, 255u8]);
    #[allow(unused_variables)]
    let white: image::Rgba<u8> = image::Rgba([255u8, 255u8, 255u8, 255u8]);
    #[allow(unused_variables)]
    let red: image::Rgba<u8> = image::Rgba([255u8, 0u8, 0u8, 255u8]);
    #[allow(unused_variables)]
    let green: image::Rgba<u8> = image::Rgba([0u8, 255u8, 0u8, 255u8]);
    #[allow(unused_variables)]
    let blue: image::Rgba<u8> = image::Rgba([0u8, 127u8, 255u8, 255u8]);
    #[allow(unused_variables)]
    let orange: image::Rgba<u8> = image::Rgba([255u8, 127u8, 0u8, 255u8]);
    // #[allow(unused_variables)]
    // let xmas_green: image::Rgba<u8> = image::Rgba{data: [7u8, 86u8, 0u8, 255u8]};
    #[allow(unused_variables)]
    let xmas_green: image::Rgba<u8> = image::Rgba([100u8, 186u8, 100u8, 255u8]);
    #[allow(unused_variables)]
    let xmas_red: image::Rgba<u8> = image::Rgba([170u8, 0u8, 0u8, 255u8]);
    #[allow(unused_variables)]
    let xmas_yellow: image::Rgba<u8> = image::Rgba([240u8, 219u8, 77u8, 255u8]);

    printlnc!(white_bold: "initializing color queue");
    #[allow(unused_variables)]
    let color_queue: VecDeque<image::Rgba<u8>> = {
        let ncolors = (dims.0 * dims.1) as usize;
        let mut q = VecDeque::with_capacity(ncolors);
        for _ in 0..ncolors {
            let mut buf = [0u8; 4];
            rng.fill(&mut buf);
            buf[3] = 255u8;
            q.push_back(image::Rgba(buf));
        }
        q
    };

    let mut img_canvas: image::ImageBuffer<image::Rgba<u8>, Vec<u8>> = {
        img_canvas_shared.lock().unwrap().clone()
    };
    // temporary destination buffer
    let img_canvas_dest = img_canvas.clone();

    // Glider:
    // img_canvas.put_pixel(4, 3, white);
    // img_canvas.put_pixel(4, 4, white);
    // img_canvas.put_pixel(4, 5, white);
    // img_canvas.put_pixel(3, 5, white);
    // img_canvas.put_pixel(2, 4, white);

    // Which pixels are filled.
    let mut img_mask_filled: image::ImageBuffer<image::Luma<MaskVal>, Vec<MaskVal>> = image::ImageBuffer::from_pixel(
        dims.0, dims.1, image::Luma([0]));
    // temporary destination buffer
    let img_mask_filled_dest = img_mask_filled.clone();

    // let mut img_score: image::ImageBuffer<image::Luma<u16>, Vec<u16>> = image::ImageBuffer::from_pixel(
    //     dims.0, dims.1, image::Luma{data: [0u16]});

    fn place_pixel<I,M>(x: u32, y: u32, color: image::Rgba<u8>, mask_value: MaskVal,
                   canvas: &mut I, mask_filled: &mut M)
        where I: image::GenericImage<Pixel=image::Rgba<u8>>,
              M: image::GenericImage<Pixel=image::Luma<MaskVal>>
    {
        // printlnc!(red: "placing {} {}", x, y);
        canvas.put_pixel(x, y, color);
        // Mark as filled
        mask_filled.put_pixel(x, y, image::Luma([mask_value]));
        // Add neighbors to frontier
    };

    // Draw a sunset bar
    // for x in 0..dims.0 {
    //     place_pixel(x, center.1, orange,
    //                 &mut img_canvas, &mut img_mask_filled, &mut img_mask_frontier);
    //     place_pixel(x, center.1 + 1, blue,
    //                 &mut img_canvas, &mut img_mask_filled, &mut img_mask_frontier);
    // }

    // Start with a pixel in the middle.
    place_pixel(center.0, center.1, *img_subject.get_pixel(center.0, center.1), 1,
                &mut img_canvas, &mut img_mask_filled);

    place_pixel(center.0+80, center.1, *img_subject.get_pixel(center.0, center.1), 2,
                &mut img_canvas, &mut img_mask_filled);

    // Initialize the canvas
    // printlnc!(white_bold: "setting up board");
    // for x in 0..dims.0 {
    //     for y in 0..dims.1 {
    //         let mut buf = [0u8; 4];
    //         rand::thread_rng().fill_bytes(&mut buf);
    //         buf[1] = buf[0];
    //         buf[2] = buf[0];
    //         buf[3] = 255u8;
    //         let drop = image::Rgba{data: buf};
    //         img_canvas.put_pixel(x, y, drop);
    //     }
    // }

    let save_enabled = false;
    if save_enabled {
        printlnc!(white_bold: "saving start image");
        img_canvas.save(&Path::new(&format!("result_{:06}.png", 0))).unwrap();
    }

    let cl_in_canvas = {
        let builder = Image::<u8>::builder()
            .channel_order(ImageChannelOrder::Rgba)
            .channel_data_type(ImageChannelDataType::UnormInt8)
            .image_type(MemObjectType::Image2d)
            .dims(&dims)
            .flags(ocl::flags::MEM_READ_ONLY | ocl::flags::MEM_HOST_WRITE_ONLY)
            .queue(queue.clone());
        unsafe { builder.use_host_slice(&img_canvas) }.build().unwrap()
    };

    let cl_in_mask_filled = {
        let builder = Image::<u8>::builder()
            .channel_order(ImageChannelOrder::Luminance)
            .channel_data_type(ImageChannelDataType::UnsignedInt8)
            .image_type(MemObjectType::Image2d)
            .dims(&dims)
            .flags(ocl::flags::MEM_READ_ONLY | ocl::flags::MEM_HOST_WRITE_ONLY)
            .queue(queue.clone());
        unsafe { builder.use_host_slice(&img_mask_filled) }.build().unwrap()
    };

    let cl_in_subject = {
        let builder = Image::<u8>::builder()
            .channel_order(ImageChannelOrder::Rgba)
            .channel_data_type(ImageChannelDataType::UnormInt8)
            .image_type(MemObjectType::Image2d)
            .dims(&dims)
            .flags(ocl::flags::MEM_READ_ONLY | ocl::flags::MEM_HOST_WRITE_ONLY)
            .queue(queue.clone());
        unsafe { builder.use_host_slice(&img_subject) }.build().unwrap()
    };

    let cl_out_canvas = {
        let builder = Image::<u8>::builder()
            .channel_order(ImageChannelOrder::Rgba)
            .channel_data_type(ImageChannelDataType::UnormInt8)
            .image_type(MemObjectType::Image2d)
            .dims(&dims)
            .flags(ocl::flags::MEM_WRITE_ONLY | ocl::flags::MEM_HOST_READ_ONLY)
            .queue(queue.clone());

        unsafe { builder.use_host_slice(&img_canvas_dest) }.build().unwrap()
    };

    let cl_out_mask_filled = {
        let builder = Image::<u8>::builder()
            .channel_order(ImageChannelOrder::Luminance)
            .channel_data_type(ImageChannelDataType::UnsignedInt8)
            .image_type(MemObjectType::Image2d)
            .dims(&dims)
            .flags(ocl::flags::MEM_WRITE_ONLY | ocl::flags::MEM_HOST_READ_ONLY)
            .queue(queue.clone());
        unsafe { builder.use_host_slice(&img_mask_filled_dest) }.build().unwrap()
    };

    let mut kernel = Kernel::builder()
        .name("pastiche")
        .program(&program)
        .queue(queue.clone())
        .global_work_size(&dims)
        .arg_named("canvas", Some(&cl_in_canvas))
        .arg_named("mask_filled", Some(&cl_in_mask_filled))
        .arg_named("subject", Some(&cl_in_subject))
        .arg_buf_named::<_, _, ocl::Buffer<ocl::prm::Uint>>("rand", None)
        .arg_vec_named("time_ms", ocl::prm::Uint::new(0)) // placeholder value
        .arg_vec_named("cursor_enabled", ocl::prm::Uint::new(0)) // placeholder value
        .arg_vec_named("cursor_pressed", ocl::prm::Uint::new(0)) // placeholder value
        .arg_vec_named("cursor_xy", ocl::prm::Uint2::new(0, 0)) // placeholder value
        // .arg_vec_named::<ocl::prm::Float4>("goal", None)
        .arg_img(&cl_out_canvas)
        .arg_img(&cl_out_mask_filled)
        .build().unwrap();

    let talk_every = 200;
    #[allow(unused_variables)]
    let cam_every = 10;
    let save_every = 1000;
    let mut last_drop = 1;

    let start = time::Instant::now();

    'outer: for frame in 0.. {
        let talk: bool = frame % talk_every == 0;
        let cam: bool = frame % cam_every == 0;
        // let cam: bool = false;

        let mut tracer = TimeTracer::new("frame");

        if talk { printlnc!(white_bold: "\nFrame: {}", frame) };

        if let Some(ref stop_rx) = stop_rx {
            match stop_rx.lock().unwrap().try_recv() {
                Ok(()) => {
                    printlnc!(red: "gpu stopped");
                    return;
                }
                Err(TryRecvError::Empty) => {},
                Err(TryRecvError::Disconnected) => {
                    printlnc!(red: "gpu stop receiver disconnected");
                    return;
                },
            }
        }

        if talk { tracer.stage("cam") };

        if cam {
            match cam_rx.lock().unwrap().try_recv() {
                Ok(img) => {
                    img_subject = cam::convert(img);
                    printlnc!(royal_blue: "cam frame");
                },
                Err(mpsc::TryRecvError::Empty) => {},
                Err(mpsc::TryRecvError::Disconnected) => panic!("cam receiver disconnected"),
            };
        }

        if talk { tracer.stage("create memory bindings") };

        let cl_in_canvas = {
            let builder = Image::<u8>::builder()
                .channel_order(ImageChannelOrder::Rgba)
                .channel_data_type(ImageChannelDataType::UnormInt8)
                .image_type(MemObjectType::Image2d)
                .dims(&dims)
                .flags(ocl::flags::MEM_READ_ONLY | ocl::flags::MEM_HOST_WRITE_ONLY)
                .queue(queue.clone());
            unsafe { builder.use_host_slice(&img_canvas) }.build().unwrap()
        };

        let cl_in_mask_filled = {
            let builder = Image::<u8>::builder()
                .channel_order(ImageChannelOrder::Luminance)
                .channel_data_type(ImageChannelDataType::UnsignedInt8)
                .image_type(MemObjectType::Image2d)
                .dims(&dims)
                .flags(ocl::flags::MEM_READ_ONLY | ocl::flags::MEM_HOST_WRITE_ONLY)
                .queue(queue.clone());
            unsafe { builder.use_host_slice(&img_mask_filled) }.build().unwrap()
        };

        let cl_in_subject = {
            let builder = Image::<u8>::builder()
                .channel_order(ImageChannelOrder::Rgba)
                .channel_data_type(ImageChannelDataType::UnormInt8)
                .image_type(MemObjectType::Image2d)
                .dims(&dims)
                .flags(ocl::flags::MEM_READ_ONLY | ocl::flags::MEM_HOST_WRITE_ONLY)
                .queue(queue.clone());
            unsafe { builder.use_host_slice(&img_subject) }.build().unwrap()
        };

        // cl_in_canvas.write(&img_canvas).enq().unwrap();
        // cl_in_mask_filled.write(&img_mask_filled).enq().unwrap();

        kernel.set_arg_img_named("canvas", Some(&cl_in_canvas)).unwrap();
        kernel.set_arg_img_named("mask_filled", Some(&cl_in_mask_filled)).unwrap();
        kernel.set_arg_img_named("subject", Some(&cl_in_subject)).unwrap();

        // let target = color_queue.pop_front();
        // if target.is_none() {
        //     printlnc!(royal_blue: "color queue drained");
        //     break;
        // }
        // let target = target.unwrap();
        // let goal = ocl::prm::Float4::new(
        //     (target[0] as f32) / 256.,
        //     (target[1] as f32) / 256.,
        //     (target[2] as f32) / 256.,
        //     (target[3] as f32) / 256.
        // );

        const RAND_PM_M: u32 = 2147483647; // 2**31-1
        let host_rands: Vec<u32> = std::iter::repeat_with(||rand::thread_rng().gen::<u32>() % RAND_PM_M)
            .take((dims.0 * dims.1) as usize)
            .collect();
        let in_rands = {
            let builder = ocl::Buffer::builder()
                .flags(ocl::flags::MEM_READ_ONLY | ocl::flags::MEM_HOST_WRITE_ONLY)
                .len(host_rands.len())
                .queue(queue.clone());
            unsafe{ builder.use_host_slice(&host_rands) }.build().unwrap()
        };

        kernel.set_arg_buf_named("rand", Some(&in_rands)).unwrap();

        let since = start.elapsed();
        kernel.set_arg_vec_named("time_ms", ocl::prm::Uint::new(duration_millis(&since) as u32)).unwrap();

        {
            let cursor = cursor_shared.lock().unwrap();
            let enabled = if cursor.enabled { 1 } else { 0 };
            let pressed = if cursor.pressed { 1 } else { 0 };
            kernel.set_arg_vec_named("cursor_enabled", enabled).unwrap();
            kernel.set_arg_vec_named("cursor_pressed", pressed).unwrap();
            kernel.set_arg_vec_named("cursor_xy", ocl::prm::Uint2::new(cursor.x, cursor.y)).unwrap();
        }

        if talk { printlnc!(royal_blue: "Running kernel..."); }
        if talk { printlnc!(white_bold: "image dims: {:?}", &dims); }

        if talk { tracer.stage("kernel enqueue"); }
        unsafe{ kernel.enq().unwrap() };

        if talk { tracer.stage("finish queue"); }
        queue.finish().unwrap();

        if talk { tracer.stage("read image"); }
        cl_out_canvas.read(&mut img_canvas).enq().unwrap();
        cl_out_mask_filled.read(&mut img_mask_filled).enq().unwrap();
        if talk { tracer.stage("pick"); }

        if talk { tracer.stage("place"); }

        let xmas_tree = false;
        if xmas_tree {
            // xmas tree works with this size: let dims: (u32, u32) = (1000, 1000);
            let place_every: u32 = 50;
            if frame % place_every == 0 {
                let i = frame / place_every;
                if i < 10 {
                    place_pixel(center.0, center.1 - (i - 5) * 50 - 100, xmas_green, 1,
                                &mut img_canvas, &mut img_mask_filled);
                }
            }
        }

        let random_drops = true;
        if random_drops && frame % 100 == 0 {
            let (x, y) = (rng.gen_range(0,dims.0), rng.gen_range(0,dims.1));
            place_pixel(x, y, *img_subject.get_pixel(x, y), last_drop+1,
                        &mut img_canvas, &mut img_mask_filled);
            last_drop += 1;
        }

        // if let Some((x, y, _)) = min_pixel_with_mask(&img_score, &img_mask_frontier) {
        //     place_pixel(x, y, target,
        //                 &mut img_canvas, &mut img_mask_filled, &mut img_mask_frontier);
        // } else {
        //     printlnc!(royal_blue: "no viable pixels");
        //     break 'outer;
        // }

        // let mut choices = sort_pixels_with_mask(&img_score, &img_mask_frontier);
        // let nplace = match frame {
        //     0 ... 1000 => 1,
        //     0 ... 2000 => 4,
        //     0 ... 4000 => 8,
        //     _ => 8,
        // };
        // choices.truncate(nplace);
        // if choices.len() == 0 {
        //     printlnc!(royal_blue: "no viable pixels");
        //     break 'outer;
        // }
        // for &(x,y,_) in choices.iter() {
        //     place_pixel(x, y, target,
        //                 &mut img_canvas, &mut img_mask_filled, &mut img_mask_frontier);
        // }

        // if frame == 400 {
        //     place_pixel(30, 30, target,
        //                 &mut img_canvas, &mut img_mask_filled, &mut img_mask_frontier);
        // }

        if talk { tracer.stage("cursor"); }
        // {
        //     let cursor = cursor_shared.lock().unwrap();
        //     let (x, y) = (cursor.x, cursor.y);
        //     if cursor.enabled && in_bounds(img_canvas.width(), img_canvas.height(), x, y) {
        //         let color = if cursor.pressed {
        //             white
        //         } else {
        //             black
        //         };
        //         place_pixel(x, y, color,
        //                     &mut img_canvas, &mut img_mask_filled, &mut img_mask_frontier);
        //     }
        // }

        if talk { tracer.stage("save"); }

        if save_enabled && frame % save_every == 0 {
            img_canvas.save(&Path::new(&format!("result_{:06}.png", frame))).unwrap();
            // img_mask_filled.save(&Path::new(&format!("mask_{:06}.png", frame))).unwrap();

            // img_mask_frontier.save(&Path::new(&format!("mask_frontier_{:06}.png", frame))).unwrap();
            // img_mask_filled.save(&Path::new(&format!("mask_filled_{:06}.png", frame))).unwrap();

            // {
            //     let buf: Vec<u8> = img_score.clone().into_raw().iter().map(|px| {
            //         (px >> 8) as u8
            //     }).collect();
            //     let img2: image::ImageBuffer<image::Luma<u8>, Vec<u8>> = image::ImageBuffer::from_raw(
            //         dims.0, dims.1, buf).unwrap();
            //     img2
            // }.save(&Path::new(&format!("score_{:06}.png", frame))).unwrap();
        }

        if talk { tracer.stage("share"); }
        {
            let turn = {*turn_shared.lock().unwrap()};
            if turn == Turn::WantData {
                {
                    let mut out = img_canvas_shared.lock().unwrap();
                    *out = img_canvas.clone();
                }
                *turn_shared.lock().unwrap() = Turn::WantDisplay;
            }
        }

        if talk { tracer.finish(); }
    }

    if save_enabled {
        printlnc!(white_bold: "saving final");
        img_canvas.save(&Path::new("result.png")).unwrap();
    }
}