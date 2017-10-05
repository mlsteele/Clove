extern crate rand;
extern crate find_folder;
extern crate image;
extern crate time;
extern crate ocl;
#[macro_use] extern crate colorify;

use std::path::Path;
use ocl::{Context, Queue, Device, Program, Image, Kernel};
use ocl::enums::{ImageChannelOrder, ImageChannelDataType, MemObjectType};
use find_folder::Search;
// use image::GenericImage;
use rand::Rng;

fn print_elapsed(title: &str, start: time::Timespec) {
    let time_elapsed = time::get_time() - start;
    let elapsed_ms = time_elapsed.num_milliseconds();
    let separator = if title.len() > 0 { ": " } else { "" };
    println!("    {}{}{}.{:03}", title, separator, time_elapsed.num_seconds(), elapsed_ms);
}


fn read_source_image(loco : &str) -> image::ImageBuffer<image::Rgba<u8>, Vec<u8>> {
    let dyn = image::open(&Path::new(loco)).unwrap();
    let img = dyn.to_rgba();
    img
}


fn main() {
    let compute_program = Search::ParentsThenKids(3, 3)
        .for_folder("cl_src").expect("Error locating 'cl_src'")
        .join("cl/clove.cl");

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

    let dims = (200, 200);

    let black: image::Rgba<u8> = image::Rgba{data: [0u8, 0u8, 0u8, 255u8]};
    let white: image::Rgba<u8> = image::Rgba{data: [255u8, 255u8, 255u8, 255u8]};
    // let start_pixel: image::Rgba<u8> = image::Rgba{data: [255u8, 255u8, 255u8, 255u8]};
    let mut src_image: image::ImageBuffer<image::Rgba<u8>, Vec<u8>> = image::ImageBuffer::from_pixel(
        dims.0, dims.1, black);
    src_image.put_pixel(4, 3, white);
    src_image.put_pixel(4, 4, white);
    src_image.put_pixel(4, 5, white);
    src_image.put_pixel(3, 5, white);
    src_image.put_pixel(2, 4, white);

    printlnc!(white_bold: "setting up board");
    for x in 0..dims.0 {
        for y in 0..dims.1 {
            let mut drop = black;
            if rand::thread_rng().next_f64() > 0.4 {
                drop = white;
            }
            src_image.put_pixel(x, y, drop);
        }
    }

    // let img = read_source_image("test.jpg");

    // let dims = img.dimensions();

    // ##################################################
    // #################### UNROLLED ####################
    // ##################################################

    let mut result_image: image::ImageBuffer<image::Rgba<u8>, Vec<u8>> = image::ImageBuffer::new(
        dims.0, dims.1);

    let cl_dest = Image::<u8>::builder()
        .channel_order(ImageChannelOrder::Rgba)
        .channel_data_type(ImageChannelDataType::UnormInt8)
        .image_type(MemObjectType::Image2d)
        .dims(&dims)
        .flags(ocl::flags::MEM_WRITE_ONLY | ocl::flags::MEM_HOST_READ_ONLY | ocl::flags::MEM_COPY_HOST_PTR)
        .queue(queue.clone())
        .host_data(&result_image)
        .build().unwrap();

    printlnc!(white_bold: "saving start image");
    src_image.save(&Path::new(&format!("result_{:08}.png", 0))).unwrap();

    for frame in 1..6000 {
        let talk = frame % 200 == 0;

        if talk { printlnc!(white_bold: "\nFrame: {}", frame); }

        let start_time = time::get_time();

        let cl_source = Image::<u8>::builder()
            .channel_order(ImageChannelOrder::Rgba)
            .channel_data_type(ImageChannelDataType::UnormInt8)
            .image_type(MemObjectType::Image2d)
            .dims(&dims)
            .flags(ocl::flags::MEM_READ_ONLY | ocl::flags::MEM_HOST_WRITE_ONLY | ocl::flags::MEM_COPY_HOST_PTR)
            .queue(queue.clone())
            .host_data(&src_image)
            .build().unwrap();

        if talk { print_elapsed("create source", start_time); }

        let kernel = Kernel::new("life", &program).unwrap()
            .queue(queue.clone())
            .gws(&dims)
            .arg_img(&cl_source)
            .arg_img(&cl_dest);

        if talk { printlnc!(royal_blue: "Running kernel..."); }
        if talk { printlnc!(white_bold: "image dims: {:?}", &dims); }

        kernel.enq().unwrap();
        if talk { print_elapsed("kernel enqueued", start_time); }

        queue.finish().unwrap();
        if talk { print_elapsed("queue finished", start_time); }

        cl_dest.read(&mut result_image).enq().unwrap();
        if talk { print_elapsed("read finished", start_time); }

        // src_image.copy_from(&result_image, 0, 0);
        src_image = result_image.clone();
        if talk { print_elapsed("copy", start_time); }

        if frame % 10 == 0 {
            result_image.save(&Path::new(&format!("result_{:08}.png", frame))).unwrap();
            if talk { print_elapsed("save", start_time); }
        }
    }

    result_image.save(&Path::new("result.png")).unwrap();
}
