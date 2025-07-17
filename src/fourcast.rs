#![allow(non_snake_case)]


use ndarray::{s, Array1, Array2, Array3};
use rand::{random_range};

struct CellForwardCache
{
    previous: Array2<f32>,
    f_t: Array2<f32>, // Forget
    h_tilde: Array2<f32>, // Candidate
}

pub struct MGUCell {
    // Forget
    w_f: Array2<f32>, // Input to hidden weights
    u_f: Array2<f32>, // hidden to hidden weights
    b_f: Array1<f32>, // Bias

    w_h: Array2<f32>, // Input to hidden weights
    u_h: Array2<f32>, // hidden to hidden weight
    b_h: Array1<f32>, // Bias

    h_t: Array2<f32>,

    activationFunction: &'static dyn Fn(Array2<f32>) -> Array2<f32>,
    gateFunction: &'static dyn Fn(Array2<f32>) -> Array2<f32>,

    cache: CellForwardCache

}

impl MGUCell {
    pub fn new(hidden_size: usize, input_shape: usize) -> MGUCell
    {

        // Forget
        let w_f = Array2::from_shape_fn( (hidden_size, input_shape), |_| random_range(-0.1..0.1));
        let u_f = Array2::from_shape_fn( (hidden_size, hidden_size), |_| random_range(-0.1..0.1));
        let b_f= Array1::from_shape_fn(hidden_size, |_| random_range(-0.1..0.1));

        // Hidden
        let w_h = Array2::from_shape_fn( (hidden_size, input_shape), |_| random_range(-0.1..0.1));
        let u_h = Array2::from_shape_fn( (hidden_size, hidden_size), |_| random_range(-0.1..0.1));
        let b_h= Array1::from_shape_fn(hidden_size, |_| random_range(-0.1..0.1));

        // Output
        let h_t = Array2::zeros((1, hidden_size));

        Self {
            w_f: w_f,
            u_f: u_f,
            b_f: b_f,

            w_h: w_h,
            u_h: u_h,
            b_h: b_h,

            h_t: h_t,

            activationFunction: &matrix_functions::Tanh_Mat,
            gateFunction: &matrix_functions::Logistic_Mat,

            cache: CellForwardCache { f_t: (), h_tilde: () }
        }
    }

    
    pub fn forward(&mut self, input_t: &Array2<f32>) -> Array2<f32>
    {
        // Step 1: Forget Gate
        let forgetInput = input_t.dot(&self.w_f.t()) + self.h_t.dot(&self.u_f.t()) + &self.b_f;
        let f_t = (self.gateFunction)(forgetInput);

        // Step 2: Candidate Hidden State
        let gated_hidden = &f_t * &self.h_t;
        let candidate_input = input_t.dot(&self.w_h.t()) + gated_hidden.dot(&self.u_h.t()) + &self.b_h;
        let h_tilde = (self.activationFunction)(candidate_input);

        // Step 3: Final
        let complement_f = f_t.mapv(|v| 1.0 - v);
        let output = (&complement_f * &self.h_t) + (&f_t * &h_tilde);

        self.h_t = output.clone();

        output
    }

    pub fn tuneParams() {
        
    }
}



pub struct LSTM {
    activationFunc: &'static dyn Fn(Array2<f32>) -> Array2<f32>,
    gateFunc: &'static dyn Fn(Array2<f32>) -> Array2<f32>,
    lossFunc: &'static dyn Fn(&Array2<f32>, &Array2<f32>) -> f32,

    hiddenLayers: i32,
    inputSize: usize,
    inputShape: usize,
    outputSize: usize,
    hiddenSize: usize,
    batchSize: usize,

    learningRate: f32,

    epochs: usize,

    cells: Vec<MGUCell>,

    isConfigured: bool
}


impl LSTM {
    
    
    pub fn new() -> LSTM
    {
        Self {
            activationFunc: &matrix_functions::Linear_Mat,
            gateFunc: &matrix_functions::Logistic_Mat,
            lossFunc: &loss_functions::MSE,
            hiddenLayers: 0,

            inputSize: 0,
            inputShape: 0,
            outputSize: 0,
            hiddenSize: 0,
            batchSize: 0,

            learningRate: 0.0,

            epochs: 0,

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
            Functions::LogisticApprox16 => &matrix_functions::Logistic_Approx_16_Mat,
            Functions::Tanh => &matrix_functions::Tanh_Mat,
            Functions::Linear => &matrix_functions::Linear_Mat
        };
        self.gateFunc = match conf.gate_function {
            Functions::ReLu => &matrix_functions::ReLu_Mat,
            Functions::LReLu => &matrix_functions::LReLu_Mat,
            Functions::Logistic => &matrix_functions::Logistic_Mat,
            Functions::LogisticApprox16 => &matrix_functions::Logistic_Approx_16_Mat,
            Functions::Tanh => &matrix_functions::Tanh_Mat,
            Functions::Linear => &matrix_functions::Linear_Mat
        };
        self.lossFunc = match conf.loss_function {
            LossFunctions::MSE => &loss_functions::MSE,
            LossFunctions::RMSE => &loss_functions::RMSE,
            LossFunctions::MAE => &loss_functions::MAE
        };

        self.hiddenLayers = conf.hidden_layers;
        self.inputSize = conf.input_size;
        self.inputShape = conf.input_shape;
        self.outputSize = conf.output_size;
        self.hiddenSize = conf.hidden_size;
        self.batchSize = conf.batch_size;
        self.epochs = conf.num_epochs;

        // Freak out if configuration is bad
        if self.hiddenLayers == 0 || self.inputSize == 0 || self.outputSize == 0 || self.hiddenSize == 0 || self.batchSize == 0
        {
            println!("ERROR: The following parameters can NOT be equal to zero: Hidden Layers, Input Size, Output Size, Hidden Size, Batch Size");

            return false;
        }

        // Populate model with cells
        for _i in 0..self.hiddenLayers
        {
            let newCell = MGUCell::new(self.hiddenSize, self.inputShape);
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


        // Actual training begins here
        for _ in 0..self.epochs
        {
            // Forward here
        }

        println!("Training Successful");
    }

    fn forward(&mut self, inputSequence: &Array3<f32>) -> Array2<f32>
    {
        let sequenceLength: i32 = inputSequence.shape()[1] as i32;

        for t in 0..sequenceLength
        {
            let x_t = inputSequence.slice(s![.., t, ..]).to_owned();

            for cell in &mut self.cells
            {
                let _layer_output = cell.forward(&x_t);
            }
        }


        self.cells.last().unwrap().h_t.clone()
    }

    fn backward(&mut self, prediction: &Array2<f32>, target: &Array2<f32>, cache: &CellForwardCache)
    {
        let dl_dht;

        let f_t_complement = &cache.f_t.mapv(|x| 1.0 - x);
        let h_tilde_squared_complement = (&cache.h_tilde * &cache.h_tilde).mapv(|x| 1.0 - x);

        // Calc ∂L/∂W_f
        let dwf_pre_t: Array2<f32> = (&dl_dht * (&cache.h_tilde - &cache.previous)) * &cache.f_t * f_t_complement;
        let dl_dwf = dwf_pre_t.t();

        // Calc ∂L/∂W_h
        let dwh_pre_t = (&dl_dht * &cache.f_t) * h_tilde_squared_complement;
        let dl_dwh = dwh_pre_t.t();

        // Calc ∂L/∂U_f
        let duf_pre_t = (&dl_dht * &cache.f_t) * f_t_complement;
        let dl_duf = duf_pre_t.t();

        // Calc ∂L/∂U_h
        let duh_pre_t = (&dl_dht * &cache.f_t) * h_tilde_squared_complement;
        let dl_duh;

        // Calc ∂L/∂B_f
        let dl_dbf;

        // Calc ∂L/∂B_h
        let dl_dbh;

    }



}



pub struct ModelConfig
{
    pub activation_function: Functions,
    pub gate_function: Functions,
    pub loss_function: LossFunctions,
    pub hidden_layers: i32,

    pub input_size: usize,
    pub input_shape: usize,
    pub output_size: usize,
    pub hidden_size: usize,
    pub batch_size: usize,

    pub num_epochs: usize
}



pub enum Functions
{
    ReLu,
    LReLu,
    Logistic,
    LogisticApprox16,
    Tanh,
    Linear,
}

pub enum LossFunctions
{
    MSE,
    RMSE,
    MAE,
}



mod matrix_functions
{
    use ndarray::{Array2};

    pub fn ReLu_Mat(mut x: Array2<f32>) -> Array2<f32>
    {
        x.mapv_inplace(|v: f32| v.max(0.0));
        x
    }
    
    pub fn LReLu_Mat(mut x: Array2<f32>) -> Array2<f32>
    {
        x.mapv_inplace(|v: f32| v.max(v * 0.001));
        x
    }
    
    pub fn Logistic_Mat(mut x: Array2<f32>) -> Array2<f32>
    {
        x.mapv_inplace(|v: f32| 1.0 / (1.0 + (-v).exp()));
        x
    }
    
    pub fn Logistic_Approx_16_Mat(mut x: Array2<f32>) -> Array2<f32>
    {
        x.mapv_inplace(|v: f32| 1.0/(1.0 + (1.0 - f32::powi(v / 16.0, 16))));
        x
    }

    pub fn Tanh_Mat(mut x: Array2<f32>) -> Array2<f32>
    {
        x.mapv_inplace(|v: f32| v.tanh());
        x
    }
    
    pub fn Linear_Mat(x: Array2<f32>) -> Array2<f32> {x}
}

mod loss_functions
{
    use ndarray::Array2;

    pub fn MSE(predictions: &Array2<f32>, targets: &Array2<f32>) -> f32
    {
        let diff = predictions - targets;
        let squared = &diff * &diff;
        squared.mean().unwrap()
    }

    pub fn RMSE(predictions: &Array2<f32>, targets: &Array2<f32>) -> f32
    {
        let diff = predictions - targets;
        let squared = &diff * &diff;
        squared.mean().unwrap().sqrt()
    }

    pub fn MAE(predictions: &Array2<f32>, targets: &Array2<f32>) -> f32
    {
        let diff = predictions - targets;
        let abs_diff = diff.mapv(|v| v.abs());
        abs_diff.mean().unwrap()
    }

}