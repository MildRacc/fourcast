#![allow(non_snake_case)]

use dioxus_core::{fc_to_builder, IntoDynNode};
use freya::elements::rect;
use freya::hooks::{theme_with, use_canvas, use_node_signal, use_platform, FontTheme, FontThemeWith, InputTheme, InputThemeWith};
use freya::launch::{launch_with_props};
use freya::prelude::{component, dioxus_elements, rsx, use_signal, Checkbox, Element, GlobalSignal, Input, Props, Readable, ScrollView, Signal, Tile, Writable};
use csv::Reader;
use ndarray::{s, Array2, Array3};
use ndarray_rand::rand::seq::SliceRandom;
use ndarray_rand::rand::thread_rng;
use skia_safe::{Color, Color4f, ColorSpace, Paint};
use std::borrow::Cow;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::ops::Not;

mod fourcast;

const COL_OFFSET: usize = 1;
const BATCH_SIZE: usize = 87;
const NUM_TIMESTAMPS: usize = 64;
const NUM_FEATURES: usize = 4;
const DATA_PATH: &str = "/home/sashad/Documents/Code/fourcast/data/MSFT.csv";



#[derive(Clone, PartialEq)]
struct Config
{
    COL_OFFSET: usize,
    BATCH_SIZE: usize,
    NUM_TIMESTAMPS: usize,
    NUM_FEATURES: usize,
    DATA_PATH: String,
}

impl Config
{
    fn new() -> Config
    {
        Config
        {
            COL_OFFSET: 0,
            BATCH_SIZE: 16,
            NUM_TIMESTAMPS: 64,
            NUM_FEATURES: 4,
            DATA_PATH: "~/Documents/data.csv".to_string()
        }
    }
}




fn main() -> Result<(), Box<dyn std::error::Error>> {

    launch_with_props(app, "4Cast", (1920.0, 1028.0));

    return Ok(());

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



#[derive(Clone, PartialEq)]
struct InputData
{
    pub value: &'static str,
    pub valueType: ValueType
}

impl InputData
{
    pub fn new(val: &'static str, valType: ValueType) -> InputData
    {
        InputData { value: val, valueType: valType }
    }
}



#[derive(Props, PartialEq, Clone)]
struct ModelOptionProps
{
    pub text: &'static str,
    pub data: InputData,
    pub placeholder: &'static str,
    pub saveto: Signal<String>
}




fn ModelOption(ModelOptionProps {text, data, placeholder, mut saveto}: ModelOptionProps) -> Element
{
    let mut value = use_signal(String::new);
    let mut checked = use_signal(|| if placeholder.to_lowercase() == "true" {true} else {false});
    
    
    rsx!(
        rect {
            width: "100%",
            padding: "5",
            margin: "5",
            direction: "horizontal",
            spacing: "25",

            label {
                color: "#DBDBDF",
                "{text}"
            }
            if data.valueType != ValueType::Bool {
                Input {
                    width: "35%",
                    theme: theme_with!(InputTheme {
                        background: Cow::Borrowed("#1C1C1F"),
                        hover_background: Cow::Borrowed("#242427"),
                        border_fill: Cow::Borrowed("#343437"),
                        font_theme: theme_with!(FontTheme {
                            color: Cow::Borrowed("#DADAEF")
                        }),
                    }),
                    placeholder: placeholder,
                    
                    value,
                    onchange: move |e: String| {

                        let filtered: String;

                        match data.valueType {
                            ValueType::U32 => filtered = e.chars().filter(|c| c.is_ascii_digit()).collect(),
                            ValueType::Float => {
                                let mut periods: i32 = 0;
                                filtered = e.chars().filter(|c| c.is_ascii_digit() || {
                                    if *c == '.' && periods == 0
                                    {
                                        periods+=1;
                                        true
                                    }
                                    else {
                                        false
                                    }
                                }).collect();
                            },
                            _ => filtered = e
                        }
                        

                        value.set(filtered.clone());
                        saveto.set(filtered);
                    }
                }
            }
            else {
                Tile {
                    onselect: move |_| {
                        checked.set(!{*checked.read()});                        
                    },

                    leading: rsx!(
                        Checkbox { 
                            selected: *checked.read_unchecked(),
                        }
                    )
                }
                
            }
        }
    )
}



struct ModelParameters
{
    pub colOffset: Signal<String>,
    pub featToGraph: Signal<String>,
    pub numFeats: Signal<String>,
    pub batchSize: Signal<String>,
    pub numTimestamps: Signal<String>,
    pub dataPath: Signal<String>,
    pub learningRate: Signal<String>,
    pub useCyclical: Signal<String>,
}
impl ModelParameters {
    pub fn new() -> ModelParameters
    {
        ModelParameters {
            colOffset:  use_signal(|| String::new()),
            featToGraph: use_signal(|| String::new()),
            numFeats: use_signal(|| String::new()),
            batchSize: use_signal(|| String::new()),
            numTimestamps: use_signal(|| String::new()),
            dataPath: use_signal(|| String::new()),
            learningRate: use_signal(|| String::new()),
            useCyclical: use_signal(|| String::new()),
        }
    }
}



fn app() -> Element
{

    let mut training_percent = use_signal(|| 0);
    let mut is_training = use_signal(|| false);


    let (reference, size) = use_node_signal();
    let mut value = use_signal(|| 0);
    let platform = use_platform();

    let graph_canvas = use_canvas(move || {
        let curr = value();
        platform.invalidate_drawing_area(size.peek().area);
        platform.request_animation_frame();
        move |ctx| {
            let canvas = ctx.canvas;
            let area = ctx.area;

            let clrspc_srgb = &ColorSpace::new_srgb();

            let minx = area.min_x();
            let miny = area.min_y();
            let maxx = area.max_x();
            let maxy = area.max_y();

            canvas.draw_line((minx, maxy), (maxx, miny), &Paint::new(Color4f::new(0.5, 0.5, 0.75, 1.0), clrspc_srgb));
            
        }
    });

    let parameters = ModelParameters::new();
    
    
    rsx!(
        rect {
            width: "100%",
            height: "100%",
            background: "#1C1C1F",
            direction: "horizontal",
            
            // Settings Window
            rect {
                width: "35%",
                height: "100%",
                corner_radius: "8",
                background: "#1C1C1F",
                border: "0 3 0 0 center #17171D",

                ScrollView {   
                    width: "100%",
                    height: "100%",

                    ModelOption { 
                        text: "Column Offset",
                        data: InputData::new("", ValueType::U32),
                        placeholder: "0",
                        saveto: parameters.colOffset,
                    }

                    ModelOption { 
                        text: "Feature to Graph",
                        data: InputData::new("", ValueType::U32),
                        placeholder: "0",
                        saveto: parameters.featToGraph
                    }

                    ModelOption { 
                        text: "Num Features",
                        data: InputData::new("", ValueType::U32),
                        placeholder: "4",
                        saveto: parameters.numFeats,
                    }

                    ModelOption { 
                        text: "Batch Size",
                        data: InputData::new("", ValueType::U32),
                        placeholder: "16",
                        saveto: parameters.batchSize,
                    }

                    ModelOption { 
                        text: "Num Timestamps",
                        data: InputData::new("", ValueType::U32),
                        placeholder: "64",
                        saveto: parameters.numTimestamps,
                    }

                    ModelOption { 
                        text: "Data Path",
                        data: InputData::new("", ValueType::String),
                        placeholder: "~/Documents/data.csv",
                        saveto: parameters.dataPath,
                    }

                    ModelOption { 
                        text: "Learning Rate",
                        data: InputData::new("", ValueType::String),
                        placeholder: "0.005",
                        saveto: parameters.learningRate,
                    }

                    ModelOption { 
                        text: "Use Cyclical",
                        data: InputData::new("", ValueType::Bool),
                        placeholder: "False",
                        saveto: parameters.useCyclical,
                    }

                    
                }
            }

            // Action Window
            rect {
                width: "65%",
                height: "100%",
                corner_radius: "8",
                background: "#1C1C1F",

                direction: "vertical",

                rect {
                    width: "100%",
                    height: "70%",

                    // Canvas
                    rect {
                        border: "2 center red",

                        padding: "5",
                        margin: "5",

                        onclick: move |_| {
                            value += 1;
                        },
                        canvas_reference: graph_canvas.attribute(),
                        reference,
                        width: "fill",
                        height: "fill" 
                    }
                }
            }

        }
    )
}


// TODO
fn generate_graph_points(min: (u32, u32), max: (u32, u32), data: Vec<f32>)
{
    let (minx, miny) = min;
    let (maxx, maxy) = max;

    let domain = maxx - minx;
    let range = maxy - miny;

    let sectionSize = domain / (data.len() as u32);

        
}


#[derive(Clone, PartialEq)]
enum ValueType
{
    U32,
    String,
    Float,
    Bool,
}

