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

#[allow(dead_code)]
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

    let dims = (100, 100);

    let black: image::Rgba<u8> = image::Rgba{data: [0u8, 0u8, 0u8, 255u8]};
    let white: image::Rgba<u8> = image::Rgba{data: [255u8, 255u8, 255u8, 255u8]};
    // let start_pixel: image::Rgba<u8> = image::Rgba{data: [255u8, 255u8, 255u8, 255u8]};
    let mut img_src: image::ImageBuffer<image::Rgba<u8>, Vec<u8>> = image::ImageBuffer::from_pixel(
        dims.0, dims.1, black);
    img_src.put_pixel(4, 3, white);
    img_src.put_pixel(4, 4, white);
    img_src.put_pixel(4, 5, white);
    img_src.put_pixel(3, 5, white);
    img_src.put_pixel(2, 4, white);

    let img_mask: image::ImageBuffer<image::Luma<u8>, Vec<u8>> = image::ImageBuffer::from_pixel(
        dims.0, dims.1, image::Luma{data: [255u8]});

    let mut img_score: image::ImageBuffer<image::Luma<u16>, Vec<u16>> = image::ImageBuffer::from_pixel(
        dims.0, dims.1, image::Luma{data: [0u16]});

    printlnc!(white_bold: "setting up board");
    for x in 0..dims.0 {
        for y in 0..dims.1 {
            let mut drop = black;
            if rand::thread_rng().next_f64() > 0.4 {
                drop = white;
            }
            img_src.put_pixel(x, y, drop);
        }
    }

    // let img = read_source_image("test.jpg");

    // let dims = img.dimensions();

    let mut img_dest: image::ImageBuffer<image::Rgba<u8>, Vec<u8>> = image::ImageBuffer::new(
        dims.0, dims.1);

    printlnc!(white_bold: "saving start image");
    img_src.save(&Path::new(&format!("result_{:06}.png", 0))).unwrap();

    for frame in 1..2 {
        printlnc!(white_bold: "\nFrame: {}", frame);

        let start_time = time::get_time();

        let cl_in_source = Image::<u8>::builder()
            .channel_order(ImageChannelOrder::Rgba)
            .channel_data_type(ImageChannelDataType::UnormInt8)
            .image_type(MemObjectType::Image2d)
            .dims(&dims)
            .flags(ocl::flags::MEM_READ_ONLY | ocl::flags::MEM_HOST_WRITE_ONLY | ocl::flags::MEM_COPY_HOST_PTR)
            .queue(queue.clone())
            .host_data(&img_src)
            .build().unwrap();

        let cl_in_mask = Image::<u8>::builder()
            .channel_order(ImageChannelOrder::Luminance)
            .channel_data_type(ImageChannelDataType::UnormInt8)
            .image_type(MemObjectType::Image2d)
            .dims(&dims)
            .flags(ocl::flags::MEM_READ_ONLY | ocl::flags::MEM_HOST_WRITE_ONLY | ocl::flags::MEM_COPY_HOST_PTR)
            .queue(queue.clone())
            .host_data(&img_mask)
            .build().unwrap();

        let cl_out_score = Image::<u16>::builder()
            .channel_order(ImageChannelOrder::Luminance)
            .channel_data_type(ImageChannelDataType::UnormInt16)
            .image_type(MemObjectType::Image2d)
            .dims(&dims)
            .flags(ocl::flags::MEM_WRITE_ONLY | ocl::flags::MEM_HOST_READ_ONLY | ocl::flags::MEM_COPY_HOST_PTR)
            .queue(queue.clone())
            .host_data(&img_score)
            .build().unwrap();

        let cl_out_dest = Image::<u8>::builder()
            .channel_order(ImageChannelOrder::Rgba)
            .channel_data_type(ImageChannelDataType::UnormInt8)
            .image_type(MemObjectType::Image2d)
            .dims(&dims)
            .flags(ocl::flags::MEM_WRITE_ONLY | ocl::flags::MEM_HOST_READ_ONLY | ocl::flags::MEM_COPY_HOST_PTR)
            .queue(queue.clone())
            .host_data(&img_dest)
            .build().unwrap();

        print_elapsed("created memory bindings", start_time);

        let goal = ocl::prm::Float4::new(0., 0., 0., 1.);
        let kernel = Kernel::new("score", &program).unwrap()
            .queue(queue.clone())
            .gws(&dims)
            .arg_img(&cl_in_source)
            .arg_img(&cl_in_mask)
            .arg_vec(goal)
            .arg_img(&cl_out_score);

        printlnc!(royal_blue: "Running kernel...");
        printlnc!(white_bold: "image dims: {:?}", &dims);

        kernel.enq().unwrap();
        print_elapsed("kernel enqueued", start_time);

        queue.finish().unwrap();
        print_elapsed("queue finished", start_time);

        cl_out_dest.read(&mut img_dest).enq().unwrap();
        cl_out_score.read(&mut img_score).enq().unwrap();
        print_elapsed("read finished", start_time);

        // img_src.copy_from(&img_dest, 0, 0);
        img_src = img_dest.clone();
        print_elapsed("copy", start_time);

        if frame % 1 == 0 {
            img_dest.save(&Path::new(&format!("result_{:06}.png", frame))).unwrap();

            {
                let buf: Vec<u8> = img_score.clone().into_raw().iter().map(|px| {
                    (px >> 8) as u8
                }).collect();
                let img2: image::ImageBuffer<image::Luma<u8>, Vec<u8>> = image::ImageBuffer::from_raw(
                    dims.0, dims.1, buf).unwrap();
                img2
            }.save(&Path::new(&format!("score_{:06}.png", frame))).unwrap();

            print_elapsed("save", start_time);
        }
    }

    img_dest.save(&Path::new("result.png")).unwrap();
}
