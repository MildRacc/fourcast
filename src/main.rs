#![allow(non_snake_case)]

use csv::Reader;
use ndarray::{s, Array2, Array3};
use ndarray_rand::rand::seq::SliceRandom;
use ndarray_rand::rand::thread_rng;
use std::fs::File;
use std::io::{BufWriter, Write};

mod fourcast;

const COL_OFFSET: usize = 1;
const BATCH_SIZE: usize = 87;
const NUM_TIMESTAMPS: usize = 64;
const NUM_FEATURES: usize = 4;
const DATA_PATH: &str = "/home/sashad/Documents/Code/fourcast/data/MSFT.csv";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("== FORECAST ==");

    let mut model = fourcast::LSTM::new();

    let config = fourcast::ModelConfig {
        activation_function: fourcast::Functions::LReLu,
        gate_function: fourcast::Functions::Tanh,
        loss_function: fourcast::LossFunctions::MAE,
        hidden_layers: 2,
        input_size: 4,
        input_shape: NUM_FEATURES,
        output_size: 4,
        hidden_size: 12,
        batch_size: BATCH_SIZE,
        num_epochs: 128,
        learning_rate: 0.001,
    };

    let mut inputSequence: Array3<f32> = Array3::zeros((BATCH_SIZE, NUM_TIMESTAMPS, NUM_FEATURES));
    let mut targetSequence: Array2<f32> = Array2::zeros((BATCH_SIZE, NUM_FEATURES));

    model.configure(config);

    let mut csvData: Vec<[f32; NUM_FEATURES]> = Vec::new();

    let mut rdr = Reader::from_path(&DATA_PATH)?;
    for result in rdr.records() {
        let record = result?;
        csvData.push([
            record[0 + COL_OFFSET].parse()?,
            record[1 + COL_OFFSET].parse()?,
            record[2 + COL_OFFSET].parse()?,
            record[3 + COL_OFFSET].parse()?,
        ]);
    }

    normalize_data(&mut csvData);

    for batch in 0..BATCH_SIZE {
        for t in 0..NUM_TIMESTAMPS {
            let idx = batch * NUM_TIMESTAMPS + t;
            for f in 0..NUM_FEATURES {
                inputSequence[[batch, t, f]] = csvData[idx][f];
            }
        }
    }

    for batch in 0..BATCH_SIZE {
        for f in 0..NUM_FEATURES {
            targetSequence[[batch, f]] = csvData[(batch + 1) * NUM_TIMESTAMPS][f];
        }
    }


    let mut indices: Vec<usize> = (0..BATCH_SIZE).collect();
    indices.shuffle(&mut thread_rng());

    let shuffledInput = Array3::from_shape_fn(inputSequence.raw_dim(), |(i, t, f)| inputSequence[(indices[i], t, f)]);
    let shuffledTarget = Array2::from_shape_fn(targetSequence.raw_dim(), |(i, j)| targetSequence[(indices[i], j)]);



    model.train(&shuffledInput, &shuffledTarget);

    // Replace the testing section with this debug version:


    let mut testInput: Array3<f32> = Array3::zeros((BATCH_SIZE, NUM_TIMESTAMPS, NUM_FEATURES));
    let mut testTargets: Array2<f32> = Array2::zeros((BATCH_SIZE, NUM_FEATURES));

    // Start test data after all training data is used
    let test_start_idx = 200;


    // Fill test input sequences
    for batch in 0..BATCH_SIZE {

        for t in 0..NUM_TIMESTAMPS {
            let idx = test_start_idx + batch * (NUM_TIMESTAMPS + 1) + t;

            if idx < csvData.len() {
                for f in 0..NUM_FEATURES {
                    testInput[[batch, t, f]] = csvData[idx][f];
                }
                if t == 0 || t == NUM_TIMESTAMPS - 1 {

                }
            }
        }
    }

    // Fill test targets (the next value after each test sequence)

    for batch in 0..BATCH_SIZE {
        let target_idx = test_start_idx + batch * (NUM_TIMESTAMPS + 1) + NUM_TIMESTAMPS;

        if target_idx < csvData.len() {
            for f in 0..NUM_FEATURES {
                testTargets[[batch, f]] = csvData[target_idx][f];
            }
        } else {
            println!("  WARNING: Target index {} is out of bounds!", target_idx);
        }
    }



    // Get predictions

    let predictions: Array2<f32> = model.predict(&testInput);


    // Check if all predictions are identical
    let first_pred = predictions.row(0);
    let mut all_identical = true;
    for batch in 1..BATCH_SIZE {
        let current_pred = predictions.row(batch);
        for f in 0..NUM_FEATURES {
            if (first_pred[f] - current_pred[f]).abs() > 1e-8 {
                all_identical = false;
                break;
            }
        }
        if !all_identical {
            break;
        }
    }


    // Check if input sequences are different
    let first_input = testInput.slice(s![0, .., ..]);
    let mut all_inputs_identical = true;
    for batch in 1..BATCH_SIZE {
        let current_input = testInput.slice(s![batch, .., ..]);
        if first_input != current_input {
            all_inputs_identical = false;
            break;
        }
    }


    // Write results to CSV with actual test targets
    let mut file = BufWriter::new(File::create(
        "/home/sashad/Documents/Code/fourcast/data/predictions.csv",
    )?);
    writeln!(
        file,
        "batch,pred_open,pred_high,pred_low,pred_close,actual_open,actual_high,actual_low,actual_close"
    )?;

    for batch in 0..BATCH_SIZE {
        let pred = predictions.row(batch);
        let actual = testTargets.row(batch);
        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{}",
            batch, pred[0], pred[1], pred[2], pred[3], actual[0], actual[1], actual[2], actual[3]
        )?;
    }

    Ok(())
}



fn normalize_data(data: &mut Vec<[f32; NUM_FEATURES]>) {
    for f in 0..NUM_FEATURES {
        let min = data.iter().map(|row| row[f]).fold(f32::INFINITY, f32::min);
        let max = data
            .iter()
            .map(|row| row[f])
            .fold(f32::NEG_INFINITY, f32::max);
        for row in data.iter_mut() {
            if max > min {
                row[f] = (row[f] - min) / (max - min);
            } else {
                row[f] = 0.0; // Avoid division by zero if all values are the same
            }
        }
    }
}
