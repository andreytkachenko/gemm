
pub trait Number: core::fmt::Display + Copy + Send + Sync + 'static  {}

impl Number for f32 {}
impl Number for f64 {}

pub trait Matrix<F: Number>: Copy + Send + Sync + 'static {
    unsafe fn sub(&self, row: usize, col: usize) -> Self;
    unsafe fn sub_col(&self, col: usize) -> Self;
    unsafe fn sub_row(&self, row: usize) -> Self;
    unsafe fn is_transposed(&self) -> bool;
    unsafe fn stride(&self) -> usize;
    unsafe fn get(&self, row: usize, col: usize) -> F;
    unsafe fn index(&self, row: usize, col: usize) -> *const F;
    unsafe fn row(&self, row: usize) -> *const F;
    unsafe fn col(&self, col: usize) -> *const F;
    unsafe fn ptr(&self) -> *const F;
    unsafe fn inc_row(&mut self);
    unsafe fn inc_col(&mut self);
    unsafe fn shift_row(&mut self, rows: usize);
    unsafe fn shift_col(&mut self, cols: usize);
}

pub trait MatrixMut<F: Number>: Matrix<F> {
    unsafe fn set(&self, row: usize, col: usize, val: F);
    unsafe fn index_mut(&self, row: usize, col: usize) -> *mut F;
    unsafe fn row_mut(&self, row: usize) -> *mut F;
    unsafe fn col_mut(&self, col: usize) -> *mut F;
    unsafe fn ptr_mut(&self) -> *mut F;
}

#[derive(Copy, Clone)]
pub struct ConstMatrix<F: Number> {
    pub stride: usize,
    pub ptr: *const F
}

unsafe impl<F: Number> Send for ConstMatrix<F> {}
unsafe impl<F: Number> Sync for ConstMatrix<F> {}

impl<F: Number> ConstMatrix<F> {
    pub fn new(ptr: *const F, stride: usize) -> Self {
        Self { ptr, stride }
    }
}

impl<F: Number> Matrix<F> for ConstMatrix<F> {
    #[inline]
    unsafe fn sub(&self, row: usize, col: usize) -> Self {
        Self { 
            ptr: self.index(row, col), 
            stride: self.stride 
        }
    }

    #[inline]
    unsafe fn sub_col(&self, col: usize) -> Self {
        Self { 
            ptr: self.col(col), 
            stride: self.stride 
        }
    }

    #[inline]
    unsafe fn sub_row(&self, row: usize) -> Self {
        Self { 
            ptr: self.row(row), 
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
    unsafe fn get(&self, row: usize, col: usize) -> F {
        *self.ptr.add(row * self.stride + col)
    }

    #[inline]
    unsafe fn index(&self, row: usize, col: usize) -> *const F {
        self.ptr.add(row * self.stride + col)
    }

    #[inline]
    unsafe fn row(&self, row: usize) -> *const F {
        self.ptr.add(row * self.stride)
    }

    #[inline]
    unsafe fn col(&self, col: usize) -> *const F {
        self.ptr.add(col)
    }

    #[inline]
    unsafe fn ptr(&self) -> *const F {
        self.ptr
    }

    #[inline]
    unsafe fn inc_row(&mut self) {
        self.ptr = self.row(1);
    }

    #[inline]
    unsafe fn inc_col(&mut self) {
        self.ptr = self.col(1);
    }

    #[inline]
    unsafe fn shift_row(&mut self, rows: usize) {
        self.ptr = self.row(rows);
    }

    #[inline]
    unsafe fn shift_col(&mut self, cols: usize) {
        self.ptr = self.col(cols);
    }
}

#[derive(Copy, Clone)]
pub struct ConstTransposedMatrix<F: Number> {
    pub stride: usize,
    pub ptr: *const F
}

unsafe impl<F: Number> Send for ConstTransposedMatrix<F> {}
unsafe impl<F: Number> Sync for ConstTransposedMatrix<F> {}

impl<F: Number> ConstTransposedMatrix<F> {
    pub fn new(ptr: *const F, stride: usize) -> Self {
        Self { ptr, stride }
    }
}

impl<F: Number> Matrix<F> for ConstTransposedMatrix<F> {
    #[inline]
    unsafe fn sub(&self, row: usize, col: usize) -> Self {
        Self { 
            ptr: self.index(row, col), 
            stride: self.stride 
        }
    }

    #[inline]
    unsafe fn sub_col(&self, col: usize) -> Self {
        Self { 
            ptr: self.col(col), 
            stride: self.stride 
        }
    }

    #[inline]
    unsafe fn sub_row(&self, row: usize) -> Self {
        Self { 
            ptr: self.row(row), 
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
    unsafe fn get(&self, row: usize, col: usize) -> F {
        *self.ptr.add(col * self.stride + row)
    }

    #[inline]
    unsafe fn index(&self, row: usize, col: usize) -> *const F {
        self.ptr.add(col * self.stride + row)
    }

    #[inline]
    unsafe fn row(&self, row: usize) -> *const F {
        self.ptr.add(row)
    }

    #[inline]
    unsafe fn col(&self, col: usize) -> *const F {
        self.ptr.add(col * self.stride)
    }

    #[inline]
    unsafe fn ptr(&self) -> *const F {
        self.ptr
    }

    #[inline]
    unsafe fn inc_row(&mut self) {
        self.ptr = self.row(1);
    }

    #[inline]
    unsafe fn inc_col(&mut self) {
        self.ptr = self.col(1);
    }

    #[inline]
    unsafe fn shift_row(&mut self, rows: usize) {
        self.ptr = self.row(rows);
    }

    #[inline]
    unsafe fn shift_col(&mut self, cols: usize) {
        self.ptr = self.col(cols);
    }
}

#[derive(Copy, Clone)]
pub struct MutMatrix<F: Number> {
    pub stride: usize,
    pub ptr: *mut F
}

unsafe impl<F: Number> Send for MutMatrix<F> {}
unsafe impl<F: Number> Sync for MutMatrix<F> {}

impl<F: Number> MutMatrix<F> {
    pub fn new(ptr: *mut F, stride: usize) -> Self {
        Self { ptr, stride }
    }
}

impl<F: Number> Matrix<F> for MutMatrix<F> {
    #[inline]
    unsafe fn sub(&self, row: usize, col: usize) -> Self {
        Self { 
            ptr: self.index_mut(row, col), 
            stride: self.stride 
        }
    }

    #[inline]
    unsafe fn sub_col(&self, col: usize) -> Self {
        Self { 
            ptr: self.col_mut(col), 
            stride: self.stride 
        }
    }

    #[inline]
    unsafe fn sub_row(&self, row: usize) -> Self {
        Self { 
            ptr: self.row_mut(row), 
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
    unsafe fn get(&self, row: usize, col: usize) -> F {
        *self.ptr.add(row * self.stride + col)
    }

    #[inline]
    unsafe fn index(&self, row: usize, col: usize) -> *const F {
        self.ptr.add(row * self.stride + col)
    }

    #[inline]
    unsafe fn row(&self, row: usize) -> *const F {
        self.ptr.add(row * self.stride)
    }

    #[inline]
    unsafe fn col(&self, col: usize) -> *const F {
        self.ptr.add(col)
    }

    #[inline]
    unsafe fn ptr(&self) -> *const F {
        self.ptr
    }

    #[inline]
    unsafe fn inc_row(&mut self) {
        self.ptr = self.row_mut(1);
    }

    #[inline]
    unsafe fn inc_col(&mut self) {
        self.ptr = self.col_mut(1);
    }

    #[inline]
    unsafe fn shift_row(&mut self, rows: usize) {
        self.ptr = self.row_mut(rows);
    }

    #[inline]
    unsafe fn shift_col(&mut self, cols: usize) {
        self.ptr = self.col_mut(cols);
    }
}

impl<F: Number> MatrixMut<F> for MutMatrix<F> {
    #[inline]
    unsafe fn set(&self, row: usize, col: usize, value: F) {
        *self.ptr.add(row * self.stride + col) = value;
    }

    #[inline]
    unsafe fn index_mut(&self, row: usize, col: usize) -> *mut F {
        self.ptr.add(row * self.stride + col)
    }

    #[inline]
    unsafe fn row_mut(&self, row: usize) -> *mut F {
        self.ptr.add(row * self.stride)
    }

    #[inline]
    unsafe fn col_mut(&self, col: usize) -> *mut F {
        self.ptr.add(col)
    }

    #[inline]
    unsafe fn ptr_mut(&self) -> *mut F {
        self.ptr
    }
}

#[derive(Copy, Clone)]
pub struct MutTransposedMatrix<F: Number> {
    pub stride: usize,
    pub ptr: *mut F
}

unsafe impl<F: Number> Send for MutTransposedMatrix<F> {}
unsafe impl<F: Number> Sync for MutTransposedMatrix<F> {}

impl<F: Number> MutTransposedMatrix<F> {
    pub fn new(ptr: *mut F, stride: usize) -> Self {
        Self { ptr, stride }
    }
}

impl<F: Number> Matrix<F> for MutTransposedMatrix<F> {
    #[inline]
    unsafe fn sub(&self, row: usize, col: usize) -> Self {
        Self { 
            ptr: self.index_mut(row, col), 
            stride: self.stride 
        }
    }

    #[inline]
    unsafe fn sub_col(&self, col: usize) -> Self {
        Self { 
            ptr: self.col_mut(col), 
            stride: self.stride 
        }
    }

    #[inline]
    unsafe fn sub_row(&self, row: usize) -> Self {
        Self { 
            ptr: self.row_mut(row), 
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
    unsafe fn get(&self, row: usize, col: usize) -> F {
        *self.ptr.add(col * self.stride + row)
    }

    #[inline]
    unsafe fn index(&self, row: usize, col: usize) -> *const F {
        self.ptr.add(col * self.stride + row) as *const F
    }

    #[inline]
    unsafe fn row(&self, row: usize) -> *const F {
        self.ptr.add(row) as *const F
    }

    #[inline]
    unsafe fn col(&self, col: usize) -> *const F {
        self.ptr.add(col * self.stride) as *const F
    }

    #[inline]
    unsafe fn ptr(&self) -> *const F {
        self.ptr
    }

    #[inline]
    unsafe fn inc_row(&mut self) {
        self.ptr = self.row_mut(1);
    }

    #[inline]
    unsafe fn inc_col(&mut self) {
        self.ptr = self.col_mut(1);
    }

    #[inline]
    unsafe fn shift_row(&mut self, rows: usize) {
        self.ptr = self.row_mut(rows);
    }

    #[inline]
    unsafe fn shift_col(&mut self, cols: usize) {
        self.ptr = self.col_mut(cols);
    }
}


impl<F: Number> MatrixMut<F> for MutTransposedMatrix<F> {
    #[inline]
    unsafe fn set(&self, row: usize, col: usize, value: F) {
        *self.ptr.add(col * self.stride + row) = value;
    }

    #[inline]
    unsafe fn index_mut(&self, row: usize, col: usize) -> *mut F {
        self.ptr.add(col * self.stride + row)
    }

    #[inline]
    unsafe fn row_mut(&self, row: usize) -> *mut F {
        self.ptr.add(row)
    }

    #[inline]
    unsafe fn col_mut(&self, col: usize) -> *mut F {
        self.ptr.add(col * self.stride)
    }

    #[inline]
    unsafe fn ptr_mut(&self) -> *mut F {
        self.ptr
    }
}


