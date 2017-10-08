use time;
use time::{Timespec};

pub struct TimeTracer {
    label: String,
    stage: String,
    staged: bool,      // whether any stages were used
    start: Timespec, // when the tracer started
    prev: Timespec, // when the active stage started
}

impl TimeTracer {
    pub fn new(label: &str) -> TimeTracer {
        let start_time = time::get_time();
	TimeTracer{
	    label:  label.to_owned(),
	    stage:  "init".to_owned(),
	    staged: false,
	    start:  start_time,
	    prev:   start_time,
	}
    }

    pub fn stage(&mut self, label: &str) {
        let now = time::get_time();
        self.finish_stage();
	self.stage = label.to_owned();
	self.prev = now;
	self.staged = true;
    }

    // consume self to indicate the object is no longer usable.
    pub fn finish(self) {
        let now = time::get_time();
        let since_start = now - self.start;
        if self.staged {
            self.finish_stage();
        }
        if let Some(us) = since_start.num_microseconds() {
            printlnc!(green: "- {} [time={:?}us]", self.label, us);
        } else {
            let ms = since_start.num_milliseconds();
            printlnc!(green: "- {} [time={}MS]", self.label, ms);
        }
    }

    fn finish_stage(&self) {
        let now = time::get_time();
        let since_prev = now - self.prev;
        if let Some(us) = since_prev.num_microseconds() {
            printlnc!(green: "| {}:{} [time={:?}us]", self.label, self.stage, us);
        } else {
            let ms = since_prev.num_milliseconds();
            printlnc!(green: "| {}:{} [time={}MS]", self.label, self.stage, ms);
        }
    }
}
