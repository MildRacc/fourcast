#![allow(non_snake_case)]

use dioxus_core::{fc_to_builder, use_hook, IntoDynNode};
use freya::elements::{label, rect};
use freya::hooks::{theme_with, use_canvas, use_node_signal, use_platform, FontTheme, FontThemeWith, InputTheme, InputThemeWith};
use freya::launch::{launch_with_props};
use freya::prelude::{component, dioxus_elements, rsx, use_signal, Button, Checkbox, Dropdown, DropdownItem, Element, GlobalSignal, Input, Props, Readable, ScrollView, Signal, Tile, Writable};
use csv::Reader;
use ndarray::{range, s, Array2, Array3};
use ndarray_rand::rand::seq::SliceRandom;
use ndarray_rand::rand::thread_rng;
use skia_safe::yuva_pixmaps::DataType;
use skia_safe::{Canvas, Color, Color4f, ColorSpace, Paint};
use std::borrow::Cow;
use std::fs::File;
use std::io::{BufWriter, Write};

mod fourcast;

// const colOff: usize = 1;
// const batchSize: usize = 87;
// const numTimestamps: usize = 64;
// const numFeats: usize = 4;
// const DATA_PATH: &str = "/home/sashad/Documents/Code/fourcast/data/MSFT.csv";



#[derive(Clone, PartialEq)]
struct Config
{
    colOff: usize,
    batchSize: usize,
    numTimestamps: usize,
    numFeats: usize,
    DATA_PATH: String,
}

impl Config
{
    fn new() -> Config
    {
        Config
        {
            colOff: 0,
            batchSize: 16,
            numTimestamps: 64,
            numFeats: 4,
            DATA_PATH: "~/Documents/data.csv".to_string()
        }
    }
}




fn main() -> Result<(), Box<dyn std::error::Error>> {

    launch_with_props(app, "4Cast", (1920.0, 1028.0));

    return Ok(());

    println!("== FORECAST ==");

    

    

    Ok(())
}



fn normalize_data(data: &mut Vec<[f32; numFeats]>) {
    for f in 0..numFeats {
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

#[derive(Clone, Copy, Debug)]
struct DataPoint
{
    x: f32,
    y: f32
}

impl DataPoint
{
    fn new(x: f32, y: f32) -> DataPoint
    {
        return DataPoint{x, y}
    }
}

#[derive(Clone, Copy, Debug)]
struct Coordinate
{
    x: i32,
    y: i32
}

impl Coordinate
{
    fn new(x: i32, y: i32) -> Coordinate
    {
        Coordinate {x, y}
    }
}



fn ModelOption(ModelOptionProps {text, data, placeholder, mut saveto}: ModelOptionProps) -> Element
{
    let mut value = use_signal(String::new);
    let mut checked = use_signal(|| if placeholder.to_lowercase() == "true" {true} else {false});
    
    let vecOptions = use_hook( || if data.valueType == ValueType::Vector {data.value.split_whitespace().collect()} else {Vec::new()} );
    let mut vecSelected = use_signal(|| placeholder);
    
    if data.valueType == ValueType::Bool || data.valueType == ValueType::Vector
    {
        saveto.set(placeholder.to_string());
    }
    
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
            

            if data.valueType == ValueType::Bool // Check for input type
            {
                Tile {
                    onselect: move |_| {
                        checked.set(!{*checked.read()});
                        saveto.set(checked.peek().to_string());                        
                    },

                    leading: rsx!(
                        Checkbox { 
                            selected: *checked.read_unchecked(),
                        }
                    )
                }
            }    
            else if data.valueType == ValueType::Vector
            {
                Dropdown {
                    value: vecSelected.read().clone(),
                    for ch in vecOptions {
                        DropdownItem {
                            value: ch,
                            onpress: {
                                move |_| {
                                    vecSelected.set(ch);
                                    saveto.set(ch.to_string());
                                }
                            },
                            label { "{ch}" }
                        }
                    }
                }
            }
            else
            {
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
                } // Input
            } // Ifelse
        } // Rect
    )
}



struct ModelParameters
{
    pub colOffset: Signal<String>,
    pub xAxisFeat: Signal<String>,
    pub yAxisFeat: Signal<String>,
    pub numEpochs: Signal<String>,
    pub numFeats: Signal<String>,
    pub hiddenLayers: Signal<String>,
    pub hiddenSize: Signal<String>,
    pub batchSize: Signal<String>,
    pub numTimestamps: Signal<String>,
    pub dataPath: Signal<String>,
    pub learningRate: Signal<String>,
    pub useCyclical: Signal<String>,
    pub activationFunc: Signal<String>,
    pub gateFunc: Signal<String>,
    pub lossFunc: Signal<String>
}
impl ModelParameters {
    pub fn new() -> ModelParameters
    {
        ModelParameters {
            colOffset:  use_signal(|| String::new()),
            xAxisFeat: use_signal(|| String::new()),
            yAxisFeat: use_signal(|| String::new()),
            numEpochs: use_signal(|| String::new()),
            numFeats: use_signal(|| String::new()),
            hiddenLayers: use_signal(|| String::new()),
            hiddenSize: use_signal(|| String::new()),
            batchSize: use_signal(|| String::new()),
            numTimestamps: use_signal(|| String::new()),
            dataPath: use_signal(|| String::new()),
            learningRate: use_signal(|| String::new()),
            useCyclical: use_signal(|| String::new()),
            activationFunc: use_signal(|| String::new()),
            gateFunc: use_signal(|| String::new()),
            lossFunc: use_signal(|| String::new()),
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

            let mut TestData: Vec<DataPoint> = Vec::new();
            for i in range(-13.0, 13.0, 0.0625)
            {
                TestData.push(DataPoint::new(i as f32, ( (i as f32).powf(i as f32) ).sin() * i));
            }

            let coordinates =  generate_graph_points((minx as u32, miny as u32), (maxx as u32, maxy as u32), TestData);

            graph_points(coordinates, canvas);

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
                        text: "Feature to Graph over X-Axis",
                        data: InputData::new("", ValueType::U32),
                        placeholder: "0",
                        saveto: parameters.xAxisFeat
                    }

                    ModelOption { 
                        text: "Feature to Graph over Y-Axis",
                        data: InputData::new("", ValueType::U32),
                        placeholder: "0",
                        saveto: parameters.yAxisFeat
                    }

                    ModelOption { 
                        text: "Num Features",
                        data: InputData::new("", ValueType::U32),
                        placeholder: "4",
                        saveto: parameters.numFeats,
                    }

                    ModelOption { 
                        text: "Hidden Layer Size",
                        data: InputData::new("", ValueType::U32),
                        placeholder: "16",
                        saveto: parameters.hiddenSize,
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
                        data: InputData::new("", ValueType::Float),
                        placeholder: "0.005",
                        saveto: parameters.learningRate,
                    }

                    ModelOption { 
                        text: "Epochs",
                        data: InputData::new("", ValueType::U32),
                        placeholder: "127",
                        saveto: parameters.numEpochs,
                    }

                    ModelOption { 
                        text: "Hidden Layers",
                        data: InputData::new("", ValueType::U32),
                        placeholder: "127",
                        saveto: parameters.hiddenLayers,
                    }

                    ModelOption { 
                        text: "Use Cyclical",
                        data: InputData::new("", ValueType::Bool),
                        placeholder: "False",
                        saveto: parameters.useCyclical,
                    }

                    ModelOption { 
                        text: "Activation Function",
                        data: InputData::new("ReLu LReLu Logistic LogisticApprox16 Tanh Linear", ValueType::Vector),
                        placeholder: "LReLu",
                        saveto: parameters.activationFunc,
                    }

                    ModelOption { 
                        text: "Gate Function",
                        data: InputData::new("ReLu LReLu Logistic LogisticApprox16 Tanh Linear", ValueType::Vector),
                        placeholder: "Tanh",
                        saveto: parameters.gateFunc,
                    }

                    ModelOption { 
                        text: "Loss Function",
                        data: InputData::new("MSE RMSE MAE", ValueType::Vector),
                        placeholder: "MAE",
                        saveto: parameters.lossFunc,
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
                        border: "2 center #2A2A2A",

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
                    
                    Button{
                        onpress: move |_| {

                            let colOff: usize = parameters.colOffset.peek().parse().unwrap(); // Unwrap not safe — temporary

                            let xAxisFeat: usize = parameters.xAxisFeat.peek().parse().unwrap(); //
                            let yAxisFeat: usize = parameters.yAxisFeat.peek().parse().unwrap(); //

                            let numEpochs: usize = parameters.numEpochs.peek().parse().unwrap(); //

                            let numFeats: usize = parameters.numFeats.peek().parse().unwrap(); //
                            
                            let hiddenLayers: i32 = parameters.hiddenLayers.peek().parse().unwrap(); //

                            let hiddenSize: usize = parameters.hiddenSize.peek().parse().unwrap(); //

                            let batchSize: usize = parameters.batchSize.peek().parse().unwrap(); //

                            let numTimestamps: usize = parameters.numTimestamps.peek().parse().unwrap(); //

                            let dataPathBinding = parameters.dataPath.peek();
                            let dataPath = dataPathBinding.as_str();

                            let learningRate: f32 = parameters.learningRate.peek().parse().unwrap();

                            let useCyclicalBinding = parameters.useCyclical.peek();
                            let useCyclical = useCyclicalBinding.as_str();

                            let activationBinding = parameters.activationFunc.peek();
                            let activation = activationBinding.as_str();

                            let gateBinding = parameters.gateFunc.peek();
                            let gate = gateBinding.as_str();

                            let lossBinding = parameters.lossFunc.peek();
                            let loss = lossBinding.as_str();

                            let mut model = fourcast::LSTM::new();

                            let config = fourcast::ModelConfig {
                                activation_function: match activation
                                {
                                    "LReLu" => fourcast::Functions::LReLu,
                                    "Linear" => fourcast::Functions::Linear,
                                    "Logistic" => fourcast::Functions::Logistic,
                                    "LogisticApprox16" => fourcast::Functions::LogisticApprox16,
                                    "ReLu" => fourcast::Functions::ReLu,
                                    "Tanh" => fourcast::Functions::Tanh,
                                    &_ => fourcast::Functions::ReLu
                                },
                                gate_function: match gate
                                {
                                    "LReLu" => fourcast::Functions::LReLu,
                                    "Linear" => fourcast::Functions::Linear,
                                    "Logistic" => fourcast::Functions::Logistic,
                                    "LogisticApprox16" => fourcast::Functions::LogisticApprox16,
                                    "ReLu" => fourcast::Functions::ReLu,
                                    "Tanh" => fourcast::Functions::Tanh,
                                    &_ => fourcast::Functions::ReLu
                                },
                                loss_function: match loss
                                {
                                    "MAE" => fourcast::LossFunctions::MAE,
                                    "MSE" => fourcast::LossFunctions::MSE,
                                    "RMSE" => fourcast::LossFunctions::RMSE,
                                    &_ => fourcast::LossFunctions::MSE
                                },
                                hidden_layers: hiddenLayers,
                                input_size: 4,
                                input_shape: numFeats,
                                output_size: 4,
                                hidden_size: hiddenSize,
                                batch_size: batchSize,
                                num_epochs: numEpochs,
                                learning_rate: learningRate,
                            };
                            // Done configuring

                            let mut inputSequence: Array3<f32> = Array3::zeros((batchSize, numTimestamps, numFeats));
                            let mut targetSequence: Array2<f32> = Array2::zeros((batchSize, numFeats));

                            model.configure(config); // Configure the model

                            let mut csvData: Vec<Vec<f32>> = Vec::new();

                            let mut rdr = Reader::from_path(&dataPath).unwrap();
                            for result in rdr.records() {
                                let record = result.unwrap();
                                csvData.push([
                                    record[0 + colOff].parse().unwrap(),
                                    record[1 + colOff].parse().unwrap(),
                                    record[2 + colOff].parse().unwrap(),
                                    record[3 + colOff].parse().unwrap(),
                                ].to_vec());
                            }

                            normalize_data(&mut csvData);

                            for batch in 0..batchSize {
                                for t in 0..numTimestamps {
                                    let idx = batch * numTimestamps + t;
                                    for f in 0..numFeats {
                                        inputSequence[[batch, t, f]] = csvData[idx][f];
                                    }
                                }
                            }

                            for batch in 0..batchSize {
                                for f in 0..numFeats {
                                    targetSequence[[batch, f]] = csvData[(batch + 1) * numTimestamps][f];
                                }
                            }


                            let mut indices: Vec<usize> = (0..batchSize).collect();
                            indices.shuffle(&mut thread_rng());

                            let shuffledInput = Array3::from_shape_fn(inputSequence.raw_dim(), |(i, t, f)| inputSequence[(indices[i], t, f)]);
                            let shuffledTarget = Array2::from_shape_fn(targetSequence.raw_dim(), |(i, j)| targetSequence[(indices[i], j)]);



                            model.train(&shuffledInput, &shuffledTarget);

                            // Replace the testing section with this debug version:


                            let mut testInput: Array3<f32> = Array3::zeros((batchSize, numTimestamps, numFeats));
                            let mut testTargets: Array2<f32> = Array2::zeros((batchSize, numFeats));

                            // Start test data after all training data is used
                            let test_start_idx = 200;


                            // Fill test input sequences
                            for batch in 0..batchSize {

                                for t in 0..numTimestamps {
                                    let idx = test_start_idx + batch * (numTimestamps + 1) + t;

                                    if idx < csvData.len() {
                                        for f in 0..numFeats {
                                            testInput[[batch, t, f]] = csvData[idx][f];
                                        }
                                        if t == 0 || t == numTimestamps - 1 {

                                        }
                                    }
                                }
                            }

                            // Fill test targets (the next value after each test sequence)

                            for batch in 0..batchSize {
                                let target_idx = test_start_idx + batch * (numTimestamps + 1) + numTimestamps;

                                if target_idx < csvData.len() {
                                    for f in 0..numFeats {
                                        testTargets[[batch, f]] = csvData[target_idx][f];
                                    }
                                } else {
                                    println!("  WARNING: Target index {} is out of bounds!", target_idx);
                                }
                            }



                            // Get predictions

                            let predictions: Array2<f32> = model.predict(&testInput);


                            // Check if to make sure all predictions are not identical
                            let first_pred = predictions.row(0);
                            let mut all_identical = true;
                            for batch in 1..batchSize {
                                let current_pred = predictions.row(batch);
                                for f in 0..numFeats {
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
                            for batch in 1..batchSize {
                                let current_input = testInput.slice(s![batch, .., ..]);
                                if first_input != current_input {
                                    all_inputs_identical = false;
                                    break;
                                }
                            }

                            for i in 0..batchSize
                            {
                                let pred = predictions.row(i);
                                let actual = testTargets.row(i);
                            }


                            // Write results to CSV with actual test targets
                            let mut file = BufWriter::new(File::create(
                                "/home/sashad/Documents/Code/fourcast/data/predictions.csv",
                            ).unwrap());
                            writeln!(
                                file,
                                "batch,pred_open,pred_high,pred_low,pred_close,actual_open,actual_high,actual_low,actual_close"
                            ).unwrap();

                            for batch in 0..batchSize {
                                let pred = predictions.row(batch);
                                let actual = testTargets.row(batch);
                                writeln!(
                                    file,
                                    "{},{},{},{},{},{},{},{},{}",
                                    batch, pred[0], pred[1], pred[2], pred[3], actual[0], actual[1], actual[2], actual[3]
                                );
                            }

                        },

                        label{"Click my ahh"}

                    }
                } // Canvas Rect
            } // Action rect
        } // Rect
    )
}


// TODO
// Returns the coordinates of each vertex of the plot in worldspace
fn generate_graph_points(minXY: (u32, u32), maxXY: (u32, u32), data: Vec<DataPoint>) -> Vec<Coordinate>
{

    let mut inputMinX = data[0].x;
    let mut inputMaxX = data[0].x;
    let mut inputMinY = data[0].y;
    let mut inputMaxY = data[0].y;

    for dp in &data {
        if dp.x < inputMinX { inputMinX = dp.x; }
        if dp.x > inputMaxX { inputMaxX = dp.x; }
        if dp.y < inputMinY { inputMinY = dp.y; }
        if dp.y > inputMaxY { inputMaxY = dp.y; }
    }

    let (minx, miny) = minXY;
    let (maxx, maxy) = maxXY;

    let domain = maxx - minx;
    let range = maxy - miny;

    let sectionSize = domain / (data.len() as u32);

    let mut outputPoints: Vec<Coordinate> = Vec::new();


    for dp in data
    {
        let x = remap_range(dp.x, inputMinX, inputMaxX, minx as f32, maxx as f32);
        let y = remap_range(dp.y, inputMinY, inputMaxY, miny as f32, maxy as f32);
        outputPoints.push(Coordinate::new(x as i32, y as i32));
    }
    
    outputPoints.reverse();

    return outputPoints
}



fn graph_points(coordinates: Vec<Coordinate>, canvas: &Canvas)
{
    let clrspc_srgb = &ColorSpace::new_srgb();

    for i in 1..coordinates.len()
        {
            let c1 = match coordinates.get(i-1)
            {
                Some(coord) => coord,
                None => &Coordinate::new(0, 0)
            };
            let c2 = match coordinates.get(i)
            {
                Some(coord) => coord,
                None => &Coordinate::new(0, 0)
            };

            let p1 = (c1.x, c1.y);
            let p2 = (c2.x, c2.y);

            canvas.draw_line(p1, p2, &Paint::new(Color4f::new(0.5, 0.5, 0.75, 1.0), clrspc_srgb));
    }
}



// Equivalent to Arduino's Map(), or GDScript's range_lerp() function. Takes a value from one range of values, and maps it to another range
fn remap_range(x: f32, inputRangeMin: f32, inputRangeMax: f32, outputRangeMin: f32, outputRangeMax: f32) -> f32
{
    outputRangeMin + ( (outputRangeMax - outputRangeMin) * ( (x - inputRangeMin) / (inputRangeMax - inputRangeMin) ) )
}



#[derive(Clone, PartialEq)]
enum ValueType
{
    U32,
    String,
    Float,
    Bool,
    Vector
}

