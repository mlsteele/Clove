use camera_capture;
use image;
use std::sync::mpsc;
use image::buffer::ConvertBuffer;

pub type CamImg = image::ImageBuffer<image::Rgb<u8>, camera_capture::Frame>;

pub fn cam_loop(dims: (u32, u32), sender: mpsc::SyncSender<CamImg>) {
    let cam = camera_capture::create(0).expect("get camera");
    let cam = cam
        .fps(30.0).expect("cam fps")
        .resolution(dims.0, dims.1).expect("cam resolution")
        .start().expect("start camera");
    for image in cam {
        match sender.try_send(image) {
            Ok(_) => {},
            Err(mpsc::TrySendError::Full(_)) => {},
            Err(mpsc::TrySendError::Disconnected(_)) => {
                eprintln!("cam disconnected");
                return
            }
        }
    }
}

pub fn convert(img: CamImg) -> image::ImageBuffer<image::Rgba<u8>, Vec<u8>> {
    img.convert()
}
