pub trait Dim {
    const DIM: usize;
}

pub struct A1; 
impl Dim for A1 { 
    const DIM: usize = 1;
}
pub struct A2; 
impl Dim for A2 { 
    const DIM: usize = 2;
}
pub struct A3; 
impl Dim for A3 { 
    const DIM: usize = 3;
}
pub struct A4; 
impl Dim for A4 { 
    const DIM: usize = 4;
}
pub struct A5; 
impl Dim for A5 { 
    const DIM: usize = 5;
}
pub struct A6; 
impl Dim for A6 { 
    const DIM: usize = 6;
}
pub struct A7; 
impl Dim for A7 { 
    const DIM: usize = 7;
}

pub struct A8; 
impl Dim for A8 { 
    const DIM: usize = 8;
}

pub struct A16; 
impl Dim for A16 { 
    const DIM: usize = 16;
}

/*
macro_rules! gen {
    (($name: ident, $val: lit)) => {
        
    };
}

gen! {
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11),
    (A12, 12),
    (A14, 14),
    (A16, 16),
    (A18, 18),
    (A20, 20),
    (A22, 22),
    (A24, 24),
    (A26, 26),
    (A28, 28),
    (A30, 30),
    (A32, 32),
}*/