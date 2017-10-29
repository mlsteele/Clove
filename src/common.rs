#[derive(PartialEq, Clone, Copy)]
pub enum Turn {
    WantData,
    WantDisplay,
}

#[derive(Default, Clone)]
pub struct Cursor {
    pub enabled: bool,
    pub x: u32,
    pub y: u32,
    pub pressed: bool,
}

