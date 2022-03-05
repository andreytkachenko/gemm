pub trait Executor {
    fn execute<F: FnOnce(usize) + Send + 'static + Copy + Sync>(
        &self,
        start: usize,
        end: usize,
        step: usize,
        f: F,
    );

    fn synchronize(&self) {}
}

pub struct DefaultExecutor;
impl Executor for DefaultExecutor {
    #[inline]
    fn execute<F: FnOnce(usize) + Send + 'static + Copy + Sync>(
        &self,
        start: usize,
        end: usize,
        step: usize,
        f: F,
    ) {
        for i in (start..end).step_by(step) {
            f(i);
        }
    }
}

pub struct RayonExecutor;
impl Executor for RayonExecutor {
    #[inline]
    fn execute<F: FnOnce(usize) + Send + 'static + Copy + Sync>(
        &self,
        start: usize,
        end: usize,
        step: usize,
        f: F,
    ) {
        let end = 1 + ((end - 1) / step);

        use rayon::prelude::*;
        (start..end)
            .into_par_iter()
            .map(|x| x * step)
            .for_each(|x| f(x));
    }
}

pub struct ThreadPoolExecutor {
    thread_pool: threadpool::ThreadPool,
}

impl ThreadPoolExecutor {
    pub fn new() -> ThreadPoolExecutor {
        ThreadPoolExecutor {
            thread_pool: threadpool::Builder::new().build(),
        }
    }
}

impl Executor for ThreadPoolExecutor {
    #[inline]
    fn execute<F: FnOnce(usize) + Send + 'static + Copy>(
        &self,
        start: usize,
        end: usize,
        step: usize,
        f: F,
    ) {
        let thread_count = self.thread_pool.max_count();

        let len = end - start;
        let num_steps = len / step;

        let mut left_steps = num_steps % thread_count;
        let main_steps = num_steps - left_steps;

        let job_size = main_steps / thread_count;

        let mut prev_end = 0;

        for _ in 0..thread_count {
            let mut now_end = prev_end + job_size;
            if left_steps > 0 {
                now_end += 1;
                left_steps -= 1;
            }
            self.thread_pool.execute(move || {
                for j in prev_end..now_end {
                    f(start + j * step);
                }
            });

            prev_end = now_end
        }
    }

    #[inline]
    fn synchronize(&self) {
        self.thread_pool.join();
    }
}
