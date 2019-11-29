pub(crate) trait Matrix: Copy {
    unsafe fn sub(&self, row: usize, col: usize) -> Self;
    unsafe fn is_transposed(&self) -> bool;
    unsafe fn stride(&self) -> usize;
    unsafe fn get(&self, row: usize, col: usize) -> f32;
    unsafe fn index(&self, row: usize, col: usize) -> *const f32;
    unsafe fn row(&self, row: usize) -> *const f32;
    unsafe fn col(&self, col: usize) -> *const f32;
}

pub(crate) trait MatrixMut: Matrix {
    unsafe fn set(&self, row: usize, col: usize, val: f32);
    unsafe fn index_mut(&self, row: usize, col: usize) -> *mut f32;
    unsafe fn row_mut(&self, row: usize) -> *mut f32;
    unsafe fn col_mut(&self, col: usize) -> *mut f32;
}

#[derive(Copy, Clone)]
pub(crate) struct ConstMatrix {
    pub stride: usize,
    pub ptr: *const f32
}

unsafe impl Send for ConstMatrix {}
unsafe impl Sync for ConstMatrix {}

impl ConstMatrix {
    pub fn new(ptr: *const f32, stride: usize) -> Self {
        Self { ptr, stride }
    }
}

impl Matrix for ConstMatrix {
    #[inline]
    unsafe fn sub(&self, row: usize, col: usize) -> Self {
        Self { 
            ptr: self.index(row, col), 
            stride: self.stride 
        }
    }

    #[inline]
    unsafe fn is_transposed(&self) -> bool { 
        false 
    }

    #[inline]
    unsafe fn stride(&self) -> usize {
        self.stride
    }

    #[inline]
    unsafe fn get(&self, row: usize, col: usize) -> f32 {
        *self.ptr.add(row * self.stride + col)
    }

    #[inline]
    unsafe fn index(&self, row: usize, col: usize) -> *const f32 {
        self.ptr.add(row * self.stride + col)
    }

    #[inline]
    unsafe fn row(&self, row: usize) -> *const f32 {
        self.ptr.add(row * self.stride)
    }

    #[inline]
    unsafe fn col(&self, col: usize) -> *const f32 {
        self.ptr.add(col)
    }
}

#[derive(Copy, Clone)]
pub(crate) struct ConstTransposedMatrix {
    pub stride: usize,
    pub ptr: *const f32
}

unsafe impl Send for ConstTransposedMatrix {}
unsafe impl Sync for ConstTransposedMatrix {}

impl ConstTransposedMatrix {
    pub fn new(ptr: *const f32, stride: usize) -> Self {
        Self { ptr, stride }
    }
}

impl Matrix for ConstTransposedMatrix {
    #[inline]
    unsafe fn sub(&self, row: usize, col: usize) -> Self {
        Self { 
            ptr: self.index(row, col), 
            stride: self.stride 
        }
    }

    #[inline]
    unsafe fn is_transposed(&self) -> bool {
        true
    }

    #[inline]
    unsafe fn stride(&self) -> usize {
        self.stride
    }

    #[inline]
    unsafe fn get(&self, row: usize, col: usize) -> f32 {
        *self.ptr.add(col * self.stride + row)
    }

    #[inline]
    unsafe fn index(&self, row: usize, col: usize) -> *const f32 {
        self.ptr.add(col * self.stride + row)
    }

    #[inline]
    unsafe fn row(&self, row: usize) -> *const f32 {
        self.ptr.add(row)
    }

    #[inline]
    unsafe fn col(&self, col: usize) -> *const f32 {
        self.ptr.add(col * self.stride)
    }
}

#[derive(Copy, Clone)]
pub(crate) struct MutMatrix {
    pub stride: usize,
    pub ptr: *mut f32
}

unsafe impl Send for MutMatrix {}
unsafe impl Sync for MutMatrix {}

impl MutMatrix {
    pub fn new(ptr: *mut f32, stride: usize) -> Self {
        Self { ptr, stride }
    }
}

impl Matrix for MutMatrix {
    #[inline]
    unsafe fn sub(&self, row: usize, col: usize) -> Self {
        Self { 
            ptr: self.index_mut(row, col), 
            stride: self.stride 
        }
    }

    #[inline]
    unsafe fn is_transposed(&self) -> bool { 
        false 
    }

    #[inline]
    unsafe fn stride(&self) -> usize {
        self.stride
    }
    
    #[inline]
    unsafe fn get(&self, row: usize, col: usize) -> f32 {
        *self.ptr.add(row * self.stride + col)
    }

    #[inline]
    unsafe fn index(&self, row: usize, col: usize) -> *const f32 {
        self.ptr.add(row * self.stride + col)
    }

    #[inline]
    unsafe fn row(&self, row: usize) -> *const f32 {
        self.ptr.add(row * self.stride)
    }

    #[inline]
    unsafe fn col(&self, col: usize) -> *const f32 {
        self.ptr.add(col)
    }
}

impl MatrixMut for MutMatrix {
    #[inline]
    unsafe fn set(&self, row: usize, col: usize, value: f32) {
        *self.ptr.add(row * self.stride + col) = value;
    }

    #[inline]
    unsafe fn index_mut(&self, row: usize, col: usize) -> *mut f32 {
        self.ptr.add(row * self.stride + col)
    }

    #[inline]
    unsafe fn row_mut(&self, row: usize) -> *mut f32 {
        self.ptr.add(row * self.stride)
    }

    #[inline]
    unsafe fn col_mut(&self, col: usize) -> *mut f32 {
        self.ptr.add(col)
    }
}

#[derive(Copy, Clone)]
pub(crate) struct MutTransposedMatrix {
    pub stride: usize,
    pub ptr: *mut f32
}

unsafe impl Send for MutTransposedMatrix {}
unsafe impl Sync for MutTransposedMatrix {}

impl MutTransposedMatrix {
    pub fn new(ptr: *mut f32, stride: usize) -> Self {
        Self { ptr, stride }
    }
}

impl Matrix for MutTransposedMatrix {
    #[inline]
    unsafe fn sub(&self, row: usize, col: usize) -> Self {
        Self { 
            ptr: self.index_mut(row, col), 
            stride: self.stride 
        }
    }

    #[inline]
    unsafe fn is_transposed(&self) -> bool {
        true
    }

    #[inline]
    unsafe fn stride(&self) -> usize {
        self.stride
    }

    #[inline]
    unsafe fn get(&self, row: usize, col: usize) -> f32 {
        *self.ptr.add(col * self.stride + row)
    }

    #[inline]
    unsafe fn index(&self, row: usize, col: usize) -> *const f32 {
        self.ptr.add(col * self.stride + row) as *const f32
    }

    #[inline]
    unsafe fn row(&self, row: usize) -> *const f32 {
        self.ptr.add(row) as *const f32
    }

    #[inline]
    unsafe fn col(&self, col: usize) -> *const f32 {
        self.ptr.add(col * self.stride) as *const f32
    }
}


impl MatrixMut for MutTransposedMatrix {
    #[inline]
    unsafe fn set(&self, row: usize, col: usize, value: f32) {
        *self.ptr.add(col * self.stride + row) = value;
    }

    #[inline]
    unsafe fn index_mut(&self, row: usize, col: usize) -> *mut f32 {
        self.ptr.add(col * self.stride + row)
    }

    #[inline]
    unsafe fn row_mut(&self, row: usize) -> *mut f32 {
        self.ptr.add(row)
    }

    #[inline]
    unsafe fn col_mut(&self, col: usize) -> *mut f32 {
        self.ptr.add(col * self.stride)
    }
}


