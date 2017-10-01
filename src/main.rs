extern crate find_folder;
extern crate image;
extern crate time;
extern crate ocl;
#[macro_use] extern crate colorify;

use std::path::Path;
use ocl::{Context, Queue, Device, Program, Image, Kernel};
use ocl::enums::{ImageChannelOrder, ImageChannelDataType, MemObjectType};
use find_folder::Search;

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

    let context = Context::builder().devices(Device::specifier()
        .type_flags(ocl::flags::DEVICE_TYPE_GPU).first()).build().unwrap();
    let device = context.devices()[0];
    let queue = Queue::new(&context, device, None).unwrap();

    let program = Program::builder()
        .src_file(compute_program)
        .devices(device)
        .build(&context)
        .unwrap();

    // let img = read_source_image("test.jpg");

    // let dims = img.dimensions();
    let dims = (200, 200);

    // let cl_source = Image::<u8>::builder()
    //     .channel_order(ImageChannelOrder::Rgba)
    //     .channel_data_type(ImageChannelDataType::UnormInt8)
    //     .image_type(MemObjectType::Image2d)
    //     .dims(&dims)
    //     .flags(ocl::flags::MEM_READ_ONLY | ocl::flags::MEM_HOST_WRITE_ONLY | ocl::flags::MEM_COPY_HOST_PTR)
    //     .queue(queue.clone())
    //     .host_data(&img)
    //     .build().unwrap();

    // ##################################################
    // #################### UNROLLED ####################
    // ##################################################

    let mut result_unrolled: image::ImageBuffer<image::Rgba<u8>, Vec<u8>> = image::ImageBuffer::new(dims.0, dims.1);

    let cl_dest_unrolled = Image::<u8>::builder()
        .channel_order(ImageChannelOrder::Rgba)
        .channel_data_type(ImageChannelDataType::UnormInt8)
        .image_type(MemObjectType::Image2d)
        .dims(&dims)
        .flags(ocl::flags::MEM_WRITE_ONLY | ocl::flags::MEM_HOST_READ_ONLY | ocl::flags::MEM_COPY_HOST_PTR)
        .queue(queue.clone())
        .host_data(&result_unrolled)
        .build().unwrap();

    let kernel = Kernel::new("clove", &program).unwrap()
        .queue(queue.clone())
        .gws(&dims)
        .arg_img(&cl_dest_unrolled);

    printlnc!(royal_blue: "\nRunning kernel (unrolled)...");
    printlnc!(white_bold: "image dims: {:?}", &dims);
    let start_time = time::get_time();

    kernel.enq().unwrap();
    print_elapsed("kernel enqueued", start_time);

    queue.finish().unwrap();
    print_elapsed("queue finished", start_time);

    cl_dest_unrolled.read(&mut result_unrolled).enq().unwrap();
    print_elapsed("read finished", start_time);

    result_unrolled.save(&Path::new("result_unrolled.png")).unwrap();
}
