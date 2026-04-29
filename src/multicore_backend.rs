#![no_main]
#![no_std]

use ariel_os::{debug::log::*, thread::CoreAffinity};
use ariel_os::thread::sync::Channel;
use portable_atomic::{AtomicUsize, Ordering};
use microflow::backend::Backend;

#[derive(Copy, Clone)]
pub struct Job {
    pub func: fn(usize),
    pub arg: usize,
}

static WORK_QUEUE: Channel<Job> = Channel::new();
static JOB_REMAINING: AtomicUsize = AtomicUsize::new(0);

pub struct ArielBackend;

impl Backend for ArielBackend {
    fn defer_job(func: fn(usize), arg: usize) {
        JOB_REMAINING.fetch_add(1, Ordering::Relaxed);
        WORK_QUEUE.send(&Job { func, arg });
    }

    fn wait() {
        while JOB_REMAINING.load(Ordering::Acquire) > 0 {
            ariel_os::thread::yield_same();
        }
    }        
}

fn worker() {
    let my_id = ariel_os::thread::current_tid().unwrap();
    let core = ariel_os::thread::core_id();
    info!("[{:?}] Worker running at [{:?}] ...", my_id, core);

    loop {
        let job = WORK_QUEUE.recv();
        (job.func)(job.arg);
        JOB_REMAINING.fetch_sub(1, Ordering::Release);
    }
}

#[ariel_os::thread(
    autostart, 
    priority = 1, 
    affinity = ariel_os::thread::CoreAffinity::one(ariel_os::thread::CoreId::new(0)),
    stacksize = 17000
)]
fn thread0() {
    worker();
}

#[ariel_os::thread(
    autostart, 
    priority = 1, 
    affinity = ariel_os::thread::CoreAffinity::one(ariel_os::thread::CoreId::new(1)),
    stacksize = 17000
)]
fn thread1() {
    worker();
}