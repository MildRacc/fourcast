#![allow(non_snake_case)]

use ndarray::{Array1, Array2};
use rand::{random_range};

pub struct Cell {
    w_ih: Array2<f32>,
    w_hh: Array2<f32>,
    b_ih: Array1<f32>,
    b_hh: Array1<f32>,
}

impl Cell {
    pub fn new(hidden_size: usize, input_shape: usize) -> Cell
    {

        // Weights
        let w_ih = Array2::from_shape_fn( (3 * hidden_size, input_shape), |_| random_range(-0.1..0.1));
        let w_hh = Array2::from_shape_fn( (3 * hidden_size, input_shape), |_| random_range(-0.1..0.1));

        // Biases
        let b_ih = Array1::from_shape_fn(3 * hidden_size, |_| random_range(-0.1..0.1));
        let b_hh = Array1::from_shape_fn(3 * hidden_size, |_| random_range(-0.1..0.1));

        Self {
            w_ih: w_ih,
            w_hh: w_hh,
            b_ih: b_ih,
            b_hh: b_hh
        }
    }

    pub fn tuneParams() {
        
    }
}



pub struct LSTM {
    activationFunc: &'static dyn Fn(&mut Array2<f32>)
}

impl LSTM {
    pub fn new(actFunc: Functions) -> LSTM
    {
        Self {
            activationFunc: match actFunc {
                Functions::ReLu => &matrix_functions::ReLu_Mat,
                Functions::LReLu => &matrix_functions::LReLu_Mat,
                Functions::Sigmoid => &matrix_functions::Sigmoid_Mat,
                Functions::Tanh => &matrix_functions::Tanh_Mat,
                Functions::Linear => &matrix_functions::Linear_Mat
            }
        }
    }
}


pub enum Functions
{
    ReLu,
    LReLu,
    Sigmoid,
    Tanh,
    Linear,
}

mod matrix_functions{
    use crate::fourcast::reg_functions;
    use ndarray::{Array2};

    pub fn ReLu_Mat(x: &mut Array2<f32>)
    {
        let iter = x.iter_mut();
    
        for value in iter
        {
            reg_functions::ReLu(value);
        }
    }
    
    pub fn LReLu_Mat(x: &mut Array2<f32>)
    {
        let iter = x.iter_mut();
    
        for value in iter
        {
            reg_functions::LReLu(value);
        }
    }
    
    pub fn Sigmoid_Mat(x: &mut Array2<f32>)
    {
        let iter = x.iter_mut();
    
        for value in iter
        {
            reg_functions::Sigmoid(value);
        }
    }
    
    pub fn Tanh_Mat(x: &mut Array2<f32>)
    {
        let iter = x.iter_mut();
    
        for value in iter
        {
            reg_functions::Tanh(value);
        }
    }
    
    pub fn Linear_Mat(_x: &mut Array2<f32>){}
}

mod reg_functions {
    use std::f32::consts::E;

    #[allow(unused_assignments)]
    pub fn ReLu(x: &mut f32)
    {
        *x = x.max(0.0);
    }

    #[allow(unused_assignments)]
    pub fn LReLu(x: &mut f32)
    {
        *x = x.max(*x * 0.001);
    }

    #[allow(unused_assignments)]
    pub fn Tanh(x: &mut f32)
    {
        *x = x.tanh();
    }

    #[allow(unused_assignments)]
    pub fn Sigmoid(x: &mut f32)
    {
        *x = 1.0 / (1.0 + E.powf(-*x));
    }

    pub fn Linear(_x: &mut f32){}
}