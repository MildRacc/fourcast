#![allow(non_snake_case)]


use ndarray::{Array1, Array2};
use rand::{random_range};

pub struct GruCell {
    w_ih: Array2<f32>,
    w_hh: Array2<f32>,
    b_ih: Array1<f32>,
    b_hh: Array1<f32>,

    h_t: Array2<f32>
}

impl GruCell {
    pub fn new(hidden_size: usize, input_shape: usize) -> GruCell
    {

        // Weights
        let w_ih = Array2::from_shape_fn( (3 * hidden_size, input_shape), |_| random_range(-0.1..0.1));
        let w_hh = Array2::from_shape_fn( (3 * hidden_size, input_shape), |_| random_range(-0.1..0.1));

        // Biases
        let b_ih = Array1::from_shape_fn(3 * hidden_size, |_| random_range(-0.1..0.1));
        let b_hh = Array1::from_shape_fn(3 * hidden_size, |_| random_range(-0.1..0.1));

        let h_t = Array2::from_shape_fn( (3 * hidden_size, input_shape), |_| random_range(-0.1..0.1));

        Self {
            w_ih: w_ih,
            w_hh: w_hh,
            b_ih: b_ih,
            b_hh: b_hh,

            h_t: h_t
        }
    }

    pub fn tuneParams() {
        
    }
}



pub struct LSTM {
    activationFunc: &'static dyn Fn(&mut Array2<f32>),

    hiddenLayers: i32,

    inputSize: usize,
    inputShape: usize,
    outputSize: usize,
    hiddenSize: usize,
    batchSize: usize,

    cells: Vec<GruCell>,

    isConfigured: bool
}


impl LSTM {
    
    
    pub fn new() -> LSTM
    {
        Self {
            activationFunc: &matrix_functions::Linear_Mat,
            hiddenLayers: 0,

            inputSize: 0,
            inputShape: 0,
            outputSize: 0,
            hiddenSize: 0,
            batchSize: 0,

            cells: Vec::new(),

            isConfigured: false
        }
    }



    pub fn configure(&mut self, conf: ModelConfig) -> bool
    {

        self.activationFunc = match conf.activation_function {
            Functions::ReLu => &matrix_functions::ReLu_Mat,
            Functions::LReLu => &matrix_functions::LReLu_Mat,
            Functions::Logistic => &matrix_functions::Logistic_Mat,
            Functions::Logistic_Approx_16 => &matrix_functions::Logistic_Approx_16_Mat,
            Functions::Tanh => &matrix_functions::Tanh_Mat,
            Functions::Linear => &matrix_functions::Linear_Mat
        };

        self.hiddenLayers = conf.hidden_layers;
        self.inputSize = conf.input_size;
        self.outputSize = conf.output_size;
        self.hiddenSize = conf.hidden_size;
        self.batchSize = conf.batch_size;

        
        if self.hiddenLayers == 0 || self.inputSize == 0 || self.outputSize == 0 || self.hiddenSize == 0 || self.batchSize == 0
        {
            println!("ERROR: The following parameters can NOT be equal to zero: Hidden Layers, Input Size, Output Size, Hidden Size, Batch Size");

            return false;
        }

        for _i in 0..self.hiddenLayers
        {
            let newCell = GruCell::new(self.hiddenSize, self.inputShape);
            self.cells.push(newCell);
        }


        self.isConfigured = true;
        return true;
    }

    pub fn train(&self)
    {
        if !self.isConfigured
        {
            println!("ERROR: CANNOT TRAIN WHILE UNCONFIGURED");
        }



        println!("Training Successful");
    }


}



pub struct ModelConfig
{
    pub activation_function: Functions,
    pub hidden_layers: i32,

    pub input_size: usize,
    pub input_shape: usize,
    pub output_size: usize,
    pub hidden_size: usize,
    pub batch_size: usize,

    pub num_epochs: i32
    
}



pub enum Functions
{
    ReLu,
    LReLu,
    Logistic,
    Logistic_Approx_16,
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
    
    pub fn Logistic_Mat(x: &mut Array2<f32>)
    {
        let iter = x.iter_mut();
    
        for value in iter
        {
            reg_functions::Logistic(value);
        }
    }
    
    pub fn Logistic_Approx_16_Mat(x: &mut Array2<f32>)
    {
        let iter = x.iter_mut();
    
        for value in iter
        {
            reg_functions::Logistic_Approx_16_Mat(value);
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
    use std::{f32::consts::E};

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
    pub fn Logistic(x: &mut f32)
    {
        *x = 1.0 / (1.0 + E.powf(-*x));
    }

    #[allow(unused_assignments)]
    pub fn Logistic_Approx_16_Mat(x: &mut f32)
    {
        *x = 1.0/(1.0 + (1.0 - f32::powi(*x / 16.0, 16)));
    }

    #[allow(unused_assignments)]
    pub fn Tanh(x: &mut f32)
    {
        *x = x.tanh();
    }

    pub fn Linear(_x: &mut f32){}
}