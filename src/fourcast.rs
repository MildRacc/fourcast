#![allow(non_snake_case)]

use ndarray::{Array1, Array2, Array3, Axis, s};
use rand::random_range;

#[derive(Clone)]
struct CellForwardCache {
    input: Array2<f32>,                   // Input at this timestep
    previous_h: Array2<f32>,              // Previous hidden state
    f_t: Array2<f32>,                     // Forget gate output
    h_tilde: Array2<f32>,                 // Candidate hidden state
    gated_hidden: Array2<f32>,            // f_t * h_prev (needed for gradients)
    forget_preactivation: Array2<f32>,    // Pre-activation for forget gate
    candidate_preactivation: Array2<f32>, // Pre-activation for candidate
}

pub struct MGUCell {
    // Forget gate parameters
    w_f: Array2<f32>, // Input to hidden weights
    u_f: Array2<f32>, // Hidden to hidden weights
    b_f: Array1<f32>, // Bias

    // Candidate hidden state parameters
    w_h: Array2<f32>, // Input to hidden weights
    u_h: Array2<f32>, // Hidden to hidden weights
    b_h: Array1<f32>, // Bias

    h_t: Array2<f32>, // Current hidden state

    activationFunction: &'static dyn Fn(Array2<f32>) -> Array2<f32>,
    gateFunction: &'static dyn Fn(Array2<f32>) -> Array2<f32>,

    cache: CellForwardCache,
}

impl MGUCell {
    pub fn new(hidden_size: usize, input_shape: usize) -> MGUCell {
        // Initialize weights with Xavier/Glorot initialization
        let xavier_bound = (6.0 / (input_shape + hidden_size) as f32).sqrt();

        // Forget gate parameters
        let w_f = Array2::from_shape_fn((hidden_size, input_shape), |_| {
            random_range(-xavier_bound..xavier_bound)
        });
        let u_f = Array2::from_shape_fn((hidden_size, hidden_size), |_| {
            random_range(-xavier_bound..xavier_bound)
        });
        let b_f = Array1::zeros(hidden_size);

        // Candidate hidden state parameters
        let w_h = Array2::from_shape_fn((hidden_size, input_shape), |_| {
            random_range(-xavier_bound..xavier_bound)
        });
        let u_h = Array2::from_shape_fn((hidden_size, hidden_size), |_| {
            random_range(-xavier_bound..xavier_bound)
        });
        let b_h = Array1::zeros(hidden_size);

        // Initialize hidden state (will be resized in forward pass)
        let h_t = Array2::zeros((1, hidden_size));

        Self {
            w_f,
            u_f,
            b_f,
            w_h,
            u_h,
            b_h,
            h_t,
            activationFunction: &matrix_functions::Tanh_Mat,
            gateFunction: &matrix_functions::Logistic_Mat,
            cache: CellForwardCache {
                input: Array2::zeros((1, input_shape)),
                previous_h: Array2::zeros((1, hidden_size)),
                f_t: Array2::zeros((1, hidden_size)),
                h_tilde: Array2::zeros((1, hidden_size)),
                gated_hidden: Array2::zeros((1, hidden_size)),
                forget_preactivation: Array2::zeros((1, hidden_size)),
                candidate_preactivation: Array2::zeros((1, hidden_size)),
            },
        }
    }

    pub fn forward(&mut self, input_t: &Array2<f32>) -> Array2<f32> {
        let previous_h = self.h_t.clone();


        // Step 1: Forget Gate
        // input_t: [batch_size, input_dim]
        // w_f: [hidden_size, input_dim] -> w_f.t(): [input_dim, hidden_size]
        // previous_h: [batch_size, hidden_size]
        // u_f: [hidden_size, hidden_size] -> u_f.t(): [hidden_size, hidden_size]
        let forget_preactivation =
            input_t.dot(&self.w_f.t()) + previous_h.dot(&self.u_f.t()) + &self.b_f;

        let f_t = (self.gateFunction)(forget_preactivation.clone());

        // Step 2: Candidate Hidden State
        let gated_hidden = &f_t * &previous_h;

        let candidate_preactivation =
            input_t.dot(&self.w_h.t()) + gated_hidden.dot(&self.u_h.t()) + &self.b_h;

        let h_tilde = (self.activationFunction)(candidate_preactivation.clone());

        // Step 3: Final Hidden State
        let complement_f = f_t.mapv(|v| 1.0 - v);
        let output = (&complement_f * &previous_h) + (&f_t * &h_tilde);

        // Cache forward pass data for backprop
        self.cache = CellForwardCache {
            input: input_t.clone(),
            previous_h,
            f_t: f_t.clone(),
            h_tilde: h_tilde.clone(),
            gated_hidden: gated_hidden.clone(),
            forget_preactivation,
            candidate_preactivation,
        };

        self.h_t = output.clone();
        output
    }

    pub fn reset_hidden_state(&mut self) {
        self.h_t.fill(0.0);
    }
}

// Forward cache for storing data needed during BPTT
#[derive(Clone)]
struct ForwardCache {
    inputs: Vec<Array2<f32>>,
    cell_caches: Vec<Vec<CellForwardCache>>, // [timestep][layer]
    layer_outputs: Vec<Vec<Array2<f32>>>,    // [timestep][layer]
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
    // Add output layer for forecasting
    output_weights: Array2<f32>,
    output_bias: Array1<f32>,

    isConfigured: bool,
    forward_cache: ForwardCache,
}

impl LSTM {
    pub fn new() -> LSTM {
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
            learningRate: 0.001,
            epochs: 0,
            cells: Vec::new(),
            output_weights: Array2::zeros((0, 0)),
            output_bias: Array1::zeros(0),
            isConfigured: false,
            forward_cache: ForwardCache {
                inputs: Vec::new(),
                cell_caches: Vec::new(),
                layer_outputs: Vec::new(),
            },
        }
    }

    pub fn configure(&mut self, conf: ModelConfig) -> bool {
        self.activationFunc = match conf.activation_function {
            Functions::ReLu => &matrix_functions::ReLu_Mat,
            Functions::LReLu => &matrix_functions::LReLu_Mat,
            Functions::Logistic => &matrix_functions::Logistic_Mat,
            Functions::LogisticApprox16 => &matrix_functions::Logistic_Approx_16_Mat,
            Functions::Tanh => &matrix_functions::Tanh_Mat,
            Functions::Linear => &matrix_functions::Linear_Mat,
        };

        self.gateFunc = match conf.gate_function {
            Functions::ReLu => &matrix_functions::ReLu_Mat,
            Functions::LReLu => &matrix_functions::LReLu_Mat,
            Functions::Logistic => &matrix_functions::Logistic_Mat,
            Functions::LogisticApprox16 => &matrix_functions::Logistic_Approx_16_Mat,
            Functions::Tanh => &matrix_functions::Tanh_Mat,
            Functions::Linear => &matrix_functions::Linear_Mat,
        };

        self.lossFunc = match conf.loss_function {
            LossFunctions::MSE => &loss_functions::MSE,
            LossFunctions::RMSE => &loss_functions::RMSE,
            LossFunctions::MAE => &loss_functions::MAE,
        };

        self.hiddenLayers = conf.hidden_layers;
        self.inputSize = conf.input_size;
        self.inputShape = conf.input_shape;
        self.outputSize = conf.output_size;
        self.hiddenSize = conf.hidden_size;
        self.batchSize = conf.batch_size;
        self.epochs = conf.num_epochs;
        self.learningRate = conf.learning_rate;

        // Validate configuration
        if self.hiddenLayers == 0
            || self.inputSize == 0
            || self.outputSize == 0
            || self.hiddenSize == 0
            || self.batchSize == 0
        {
            println!(
                "ERROR: The following parameters can NOT be equal to zero: Hidden Layers, Input Size, Output Size, Hidden Size, Batch Size"
            );
            return false;
        }

        // Create MGU cells
        self.cells.clear();
        for i in 0..self.hiddenLayers {
            let input_dim = if i == 0 {
                self.inputShape
            } else {
                self.hiddenSize
            };
            let cell = MGUCell::new(self.hiddenSize, input_dim);
            self.cells.push(cell);
        }

        self.output_weights = Array2::from_shape_fn((self.outputSize, self.hiddenSize), |_| {
            rand::random::<f32>() * 0.01 // Small random values
        });

        self.output_bias = Array1::zeros(self.outputSize);

        self.isConfigured = true;
        true
    }

    ///  inputSequence: Array3<f32>
    ///     [i]:        batch           - Index of sequences in a batch (batch size)
    ///     [i][j]:     timestamp       - Index of timestamps in a batch (sequence length)
    ///     [i][j][k]:  feature         - Index of features per timestamp (input dimension)
    ///
    ///  targetSequence: Array2<f32>
    ///     [i]:    batch   - Index of the sequence in the batch (batch size)
    ///     [i][j]: output  - Index of the output feature for that sequence (output size)
    pub fn train(&mut self, inputSequence: &Array3<f32>, targetSequence: &Array2<f32>) {
        if !self.isConfigured {
            println!("ERROR: CANNOT TRAIN WHILE UNCONFIGURED");
            return;
        }

        // Reset hidden states at start of training
        for cell in &mut self.cells {
            cell.reset_hidden_state();
        }

        for epoch in 0..self.epochs {
            let prediction = self.forward(inputSequence);
            self.backward(&prediction, targetSequence);
            let loss = (self.lossFunc)(&prediction, targetSequence);

            if epoch % 10 == 0 || epoch == self.epochs - 1 {
                println!("Epoch {}: Loss = {:.6}", epoch, loss);
            }

            if loss < 0.001 {
                println!(
                    "Training converged at epoch {} with loss {:.6}",
                    epoch, loss
                );
                break;
            }
        }

    }

    fn forward(&mut self, inputSequence: &Array3<f32>) -> Array2<f32> {
        let sequence_length = inputSequence.shape()[1];
        let batch_size = inputSequence.shape()[0];

        // Clear forward cache
        self.forward_cache.inputs.clear();
        self.forward_cache.cell_caches.clear();
        self.forward_cache.layer_outputs.clear();

        // Reset hidden states for all cells to correct batch size
        for cell in &mut self.cells {
            cell.h_t = Array2::zeros((batch_size, self.hiddenSize));
        }

        // Process each timestep
        for t in 0..sequence_length {
            // Extract input at timestep t: shape [batch_size, input_dim]
            let x_t = inputSequence.slice(s![.., t, ..]).to_owned();

            self.forward_cache.inputs.push(x_t.clone());

            let mut timestep_caches = Vec::new();
            let mut timestep_outputs = Vec::new();
            let mut layer_input = x_t;

            // Forward through all layers for this timestep
            for (_layer_idx, cell) in self.cells.iter_mut().enumerate() {
                let layer_output = cell.forward(&layer_input);

                timestep_caches.push(cell.cache.clone());
                timestep_outputs.push(layer_output.clone());
                layer_input = layer_output;
            }

            self.forward_cache.cell_caches.push(timestep_caches);
            self.forward_cache.layer_outputs.push(timestep_outputs);
        }

        // Get final hidden state from last layer at last timestep
        let final_hidden = self
            .forward_cache
            .layer_outputs
            .last() // last timestep
            .unwrap()
            .last() // last layer
            .unwrap()
            .clone();

        // Apply output layer transformation
        let prediction = final_hidden.dot(&self.output_weights.t()) + &self.output_bias;

        prediction
    }

    fn backward(&mut self, prediction: &Array2<f32>, target: &Array2<f32>) {
        let sequence_length = self.forward_cache.inputs.len();
        let num_layers = self.cells.len();

        // Calculate gradient from output layer
        let dl_dpred = prediction - target;
        let final_hidden = &self.cells.last().unwrap().h_t;

        // Output layer gradients
        let dl_dw_out = dl_dpred.t().dot(final_hidden);
        let dl_db_out = dl_dpred.sum_axis(Axis(0));
        let dl_dh_final = dl_dpred.dot(&self.output_weights);

        // Initialize gradient accumulators for each layer
        let mut grad_accumulators: Vec<GradientAccumulator> = (0..num_layers)
            .map(|_| GradientAccumulator::new())
            .collect();

        // Initialize gradients for BPTT
        let mut dl_dh_next: Vec<Array2<f32>> =
            vec![Array2::zeros((1, self.hiddenSize)); num_layers];
        dl_dh_next[num_layers - 1] = dl_dh_final;

        // Backpropagate through time
        for t in (0..sequence_length).rev() {
            let mut dl_dh_current = dl_dh_next.clone();

            // Backpropagate through layers (from last to first)
            for layer_idx in (0..num_layers).rev() {
                let cache = &self.forward_cache.cell_caches[t][layer_idx];

                // Get input to this layer at this timestep
                let layer_input = if layer_idx == 0 {
                    &self.forward_cache.inputs[t]
                } else {
                    &self.forward_cache.layer_outputs[t][layer_idx - 1]
                };

                // Compute gradients for this layer at this timestep
                let gradients = self.compute_mgu_gradients(
                    &dl_dh_current[layer_idx],
                    cache,
                    layer_input,
                    layer_idx,
                );

                // Accumulate gradients
                grad_accumulators[layer_idx].accumulate(&gradients);

                // Set gradient for previous layer in current timestep
                if layer_idx > 0 {
                    dl_dh_current[layer_idx - 1] = &dl_dh_current[layer_idx - 1] + &gradients.dl_dx;
                }

                // Set gradient for same layer in previous timestep
                if t > 0 {
                    dl_dh_next[layer_idx] = gradients.dl_dh_prev;
                } else {
                    dl_dh_next[layer_idx] = Array2::zeros((1, self.hiddenSize));
                }
            }
        }

        // Apply accumulated gradients with gradient clipping
        for (layer_idx, accumulator) in grad_accumulators.iter().enumerate() {
            self.apply_gradients(layer_idx, accumulator);
        }

        // Update output layer
        let clipped_dw_out = clip_gradients(&dl_dw_out, 5.0);
        let clipped_db_out = clip_gradients(&dl_db_out, 5.0);

        self.output_weights = &self.output_weights - &(self.learningRate * &clipped_dw_out);
        self.output_bias = &self.output_bias - &(self.learningRate * &clipped_db_out);
    }

    fn compute_mgu_gradients(
        &self,
        dl_dht: &Array2<f32>,
        cache: &CellForwardCache,
        input: &Array2<f32>,
        layer_idx: usize,
    ) -> MGUGradients {

        let cell = &self.cells[layer_idx];

        // Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
        let df_dnet_f = &cache.f_t * &cache.f_t.mapv(|x| 1.0 - x);

        // Tanh derivative: 1 - tanh²(x)
        let dh_dnet_h = cache
            .candidate_preactivation
            .mapv(|x| 1.0 - x.tanh().powi(2));

        // Gradients w.r.t. forget gate
        let dl_df = dl_dht * (&cache.h_tilde - &cache.previous_h);
        let dl_dnet_f = &dl_df * &df_dnet_f;

        // Gradients w.r.t. candidate hidden state
        let dl_dh_tilde = dl_dht * &cache.f_t;
        let dl_dnet_h = &dl_dh_tilde * &dh_dnet_h;

        // Weight gradients - ensure correct dimensions
        // For weight matrices, we need: [output_dim, input_dim]
        // dl_dnet has shape [batch_size, hidden_size]
        // input has shape [batch_size, input_dim]
        // So we need: input.t().dot(dl_dnet) to get [input_dim, hidden_size], then transpose

        let dl_dw_f = input.t().dot(&dl_dnet_f).t().to_owned(); // [hidden_size, input_dim]
        let dl_du_f = cache.previous_h.t().dot(&dl_dnet_f).t().to_owned(); // [hidden_size, hidden_size]
        let dl_db_f = dl_dnet_f.sum_axis(Axis(0)); // [hidden_size]

        let dl_dw_h = input.t().dot(&dl_dnet_h).t().to_owned(); // [hidden_size, input_dim]
        let dl_du_h = cache.gated_hidden.t().dot(&dl_dnet_h).t().to_owned(); // [hidden_size, hidden_size]
        let dl_db_h = dl_dnet_h.sum_axis(Axis(0)); // [hidden_size]

        // Gradients w.r.t. inputs and previous hidden state
        let dl_dx = dl_dnet_f.dot(&cell.w_f) + dl_dnet_h.dot(&cell.w_h);

        let dl_dh_prev = dl_dht * &cache.f_t.mapv(|x| 1.0 - x) +  // Direct connection
                    dl_dnet_f.dot(&cell.u_f) +           // Through forget gate
                    dl_dnet_h.dot(&cell.u_h) * &cache.f_t; // Through candidate gate

        MGUGradients {
            dl_dw_f,
            dl_du_f,
            dl_db_f,
            dl_dw_h,
            dl_du_h,
            dl_db_h,
            dl_dx,
            dl_dh_prev,
        }
    }

    fn apply_gradients(&mut self, layer_idx: usize, accumulator: &GradientAccumulator) {
        let cell = &mut self.cells[layer_idx];
        let clip_value = 5.0; // Gradient clipping threshold

        // Apply gradient clipping and update parameters
        let clipped_dw_f = clip_gradients(&accumulator.dw_f, clip_value);
        let clipped_du_f = clip_gradients(&accumulator.du_f, clip_value);
        let clipped_db_f = clip_gradients(&accumulator.db_f, clip_value);
        let clipped_dw_h = clip_gradients(&accumulator.dw_h, clip_value);
        let clipped_du_h = clip_gradients(&accumulator.du_h, clip_value);
        let clipped_db_h = clip_gradients(&accumulator.db_h, clip_value);

        cell.w_f = &cell.w_f - &(self.learningRate * &clipped_dw_f);
        cell.u_f = &cell.u_f - &(self.learningRate * &clipped_du_f);
        cell.b_f = &cell.b_f - &(self.learningRate * &clipped_db_f);
        cell.w_h = &cell.w_h - &(self.learningRate * &clipped_dw_h);
        cell.u_h = &cell.u_h - &(self.learningRate * &clipped_du_h);
        cell.b_h = &cell.b_h - &(self.learningRate * &clipped_db_h);
    }

    pub fn predict(&mut self, inputSequence: &Array3<f32>) -> Array2<f32> {
        self.forward(inputSequence)
    }
}

// Helper structs for gradient computation
struct MGUGradients {
    dl_dw_f: Array2<f32>,
    dl_du_f: Array2<f32>,
    dl_db_f: Array1<f32>,
    dl_dw_h: Array2<f32>,
    dl_du_h: Array2<f32>,
    dl_db_h: Array1<f32>,
    dl_dx: Array2<f32>,
    dl_dh_prev: Array2<f32>,
}

struct GradientAccumulator {
    dw_f: Array2<f32>,
    du_f: Array2<f32>,
    db_f: Array1<f32>,
    dw_h: Array2<f32>,
    du_h: Array2<f32>,
    db_h: Array1<f32>,
    initialized: bool,
}

impl GradientAccumulator {
    fn new() -> Self {
        Self {
            dw_f: Array2::zeros((0, 0)),
            du_f: Array2::zeros((0, 0)),
            db_f: Array1::zeros(0),
            dw_h: Array2::zeros((0, 0)),
            du_h: Array2::zeros((0, 0)),
            db_h: Array1::zeros(0),
            initialized: false,
        }
    }

    fn accumulate(&mut self, gradients: &MGUGradients) {
        if !self.initialized {
            self.dw_f = gradients.dl_dw_f.clone();
            self.du_f = gradients.dl_du_f.clone();
            self.db_f = gradients.dl_db_f.clone();
            self.dw_h = gradients.dl_dw_h.clone();
            self.du_h = gradients.dl_du_h.clone();
            self.db_h = gradients.dl_db_h.clone();
            self.initialized = true;
        } else {
            self.dw_f = &self.dw_f + &gradients.dl_dw_f;
            self.du_f = &self.du_f + &gradients.dl_du_f;
            self.db_f = &self.db_f + &gradients.dl_db_f;
            self.dw_h = &self.dw_h + &gradients.dl_dw_h;
            self.du_h = &self.du_h + &gradients.dl_du_h;
            self.db_h = &self.db_h + &gradients.dl_db_h;
        }
    }
}

// Gradient clipping function
fn clip_gradients<D>(
    gradients: &ndarray::ArrayBase<ndarray::OwnedRepr<f32>, D>,
    clip_value: f32,
) -> ndarray::ArrayBase<ndarray::OwnedRepr<f32>, D>
where
    D: ndarray::Dimension,
    ndarray::ArrayBase<ndarray::OwnedRepr<f32>, D>: Clone,
{
    gradients.mapv(|x| x.max(-clip_value).min(clip_value))
}

pub struct ModelConfig {
    pub activation_function: Functions,
    pub gate_function: Functions,
    pub loss_function: LossFunctions,
    pub hidden_layers: i32,
    pub input_size: usize,
    pub input_shape: usize,
    pub output_size: usize,
    pub hidden_size: usize,
    pub batch_size: usize,
    pub num_epochs: usize,
    pub learning_rate: f32,
}

pub enum Functions {
    ReLu,
    LReLu,
    Logistic,
    LogisticApprox16,
    Tanh,
    Linear,
}

pub enum LossFunctions {
    MSE,
    RMSE,
    MAE,
}

mod matrix_functions {
    use ndarray::Array2;

    pub fn ReLu_Mat(mut x: Array2<f32>) -> Array2<f32> {
        x.mapv_inplace(|v: f32| v.max(0.0));
        x
    }

    pub fn LReLu_Mat(mut x: Array2<f32>) -> Array2<f32> {
        x.mapv_inplace(|v: f32| if v > 0.0 { v } else { v * 0.01 });
        x
    }

    pub fn Logistic_Mat(mut x: Array2<f32>) -> Array2<f32> {
        x.mapv_inplace(|v: f32| 1.0 / (1.0 + (-v.max(-500.0).min(500.0)).exp()));
        x
    }

    pub fn Logistic_Approx_16_Mat(mut x: Array2<f32>) -> Array2<f32> {
        x.mapv_inplace(|v: f32| {
            let clamped = v.max(-16.0).min(16.0);
            1.0 / (1.0 + (1.0 - (clamped / 16.0).powi(16)))
        });
        x
    }

    pub fn Tanh_Mat(mut x: Array2<f32>) -> Array2<f32> {
        x.mapv_inplace(|v: f32| v.max(-500.0).min(500.0).tanh());
        x
    }

    pub fn Linear_Mat(x: Array2<f32>) -> Array2<f32> {
        x
    }
}

mod loss_functions {
    use ndarray::Array2;

    pub fn MSE(predictions: &Array2<f32>, targets: &Array2<f32>) -> f32 {
        let diff = predictions - targets;
        let squared = &diff * &diff;
        squared.mean().unwrap()
    }

    pub fn RMSE(predictions: &Array2<f32>, targets: &Array2<f32>) -> f32 {
        let diff = predictions - targets;
        let squared = &diff * &diff;
        squared.mean().unwrap().sqrt()
    }

    pub fn MAE(predictions: &Array2<f32>, targets: &Array2<f32>) -> f32 {
        let diff = predictions - targets;
        let abs_diff = diff.mapv(|v| v.abs());
        abs_diff.mean().unwrap()
    }
}
