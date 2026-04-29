use ariel_os::debug::log::*;
use portable_atomic::{AtomicUsize, Ordering};
use microflow::backend::Backend;

#[derive(Copy, Clone)]
pub struct Job {
    pub func: fn(usize),
    pub arg: usize,
}

#[repr(align(64))]
struct PaddedAtomic(AtomicUsize);

const JOB_QUEUE_SIZE: usize = 32;

static mut JOB_QUEUE: [Option<Job>; JOB_QUEUE_SIZE] = [None; JOB_QUEUE_SIZE];
static Q_HEAD: PaddedAtomic = PaddedAtomic(AtomicUsize::new(0));
static Q_TAIL: PaddedAtomic = PaddedAtomic(AtomicUsize::new(0));
static JOB_REMAINING: PaddedAtomic = PaddedAtomic(AtomicUsize::new(0));

pub struct ArielBackend;

impl Backend for ArielBackend {
    fn defer_job(func: fn(usize), arg: usize) {
        JOB_REMAINING.0.fetch_add(1, Ordering::Relaxed);
        
        let head = Q_HEAD.0.load(Ordering::Relaxed);
        unsafe {
            JOB_QUEUE[head % JOB_QUEUE_SIZE] = Some(Job { func, arg });
        }
        Q_HEAD.0.store(head + 1, Ordering::Release);
    }

    fn wait() {
        while JOB_REMAINING.0.load(Ordering::Acquire) > 0 {
            if !process_one_job() {
                core::hint::spin_loop();
            }
        }
    }        
}

fn process_one_job() -> bool {
    let tail = Q_TAIL.0.load(Ordering::Relaxed);
    let head = Q_HEAD.0.load(Ordering::Acquire);
    
    if tail < head {
        if Q_TAIL.0.compare_exchange(tail, tail + 1, Ordering::Acquire, Ordering::Relaxed).is_ok() {
            let job = unsafe { JOB_QUEUE[tail % JOB_QUEUE_SIZE].unwrap() };
            (job.func)(job.arg);
            
            JOB_REMAINING.0.fetch_sub(1, Ordering::Release);
            return true;
        }
    }
    false
}

fn worker() {
    let my_id = ariel_os::thread::current_tid().unwrap();
    let core = ariel_os::thread::core_id();
    info!("[{:?}] Worker running at [{:?}] ...", my_id, core);

    loop {
        if !process_one_job() {
            core::hint::spin_loop();
        }
    }
}

#[ariel_os::thread(autostart, priority = 1, stacksize = 160000)]
fn thread0() {
    worker();
}