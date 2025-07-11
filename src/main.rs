mod fourcast;

fn main() {
    println!("Hello, world!");

    let mut model = fourcast::LSTM::new();

    let config = fourcast::ModelConfig
    {
        activation_function: fourcast::Functions::Tanh,
        gate_function: fourcast::Functions::Logistic_Approx_16,
        hidden_layers: 64,
        input_size: 16,
        input_shape: 4,
        output_size: 16,
        hidden_size: 16,
        batch_size: 32,
        num_epochs: 128
    };

    model.configure(config);

    model.train();
    
}
