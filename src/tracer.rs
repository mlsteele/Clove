use std::time::Instant;

pub struct TimeTracer {
    label: String,
    stage: String,
    staged: bool,      // whether any stages were used
    start: Instant, // when the tracer started
    prev: Instant, // when the active stage started
}

impl TimeTracer {
    pub fn new(label: &str) -> TimeTracer {
        let start_time = Instant::now();
        TimeTracer{
            label:  label.to_owned(),
            stage:  "init".to_owned(),
            staged: false,
            start:  start_time,
            prev:   start_time,
        }
    }

    pub fn stage(&mut self, label: &str) {
        let now = Instant::now();
        self.finish_stage();
	self.stage = label.to_owned();
	self.prev = now;
	self.staged = true;
    }

    // consume self to indicate the object is no longer usable.
    pub fn finish(self) {
        let now = Instant::now();
        let since_start = now - self.start;
        if self.staged {
            self.finish_stage();
        }
        let us = since_start.as_micros();
        printlnc!(green: "- {} [time={:?}us]", self.label, us);
    }

    fn finish_stage(&self) {
        let now = Instant::now();
        let since_prev = now - self.prev;
        let us = since_prev.as_micros();
        printlnc!(green: "| {}:{} [time={:?}us]", self.label, self.stage, us);
    }
}
