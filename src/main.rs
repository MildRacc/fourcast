mod fourcast;

fn main() {
    println!("Hello, world!");

    let model = fourcast::LSTM::new();

    let config = fourcast::ModelConfig {
        fourcast::Functions::Sigmoid,
        128,
        32,
        32,
        32,
        64,
        2048
    };

    model.train();
    
}
