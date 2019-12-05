use std::alloc;

pub struct Alloc {
    ptr: *mut u8,
    layout: alloc::Layout,
}

impl Alloc {
    pub fn new(size: usize) -> Alloc {
        const ALIGN: usize = 32;
        let layout = alloc::Layout::from_size_align(size, ALIGN).unwrap();
        let ptr = unsafe { alloc::alloc(layout) };
        Alloc { ptr, layout }
    }

    pub fn ptr<F>(&self) -> *mut F {
        self.ptr as *mut F
    }
}

impl Drop for Alloc {
    fn drop(&mut self) {
        unsafe {
            alloc::dealloc(self.ptr, self.layout);
        }
    }
}
