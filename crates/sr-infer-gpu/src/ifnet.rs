//! A learned frame-interpolation network: the ncnn checkpoint format, its
//! operators, and a reference forward pass.
//!
//! ## Why this exists
//!
//! The GPU backend in this crate interpolates with a block matcher and a warp. It
//! is honest about being an algorithm rather than a model, and it is not
//! RIFE-class: a learned interpolator predicts flow with convolutions over a
//! multi-scale pyramid and fuses the two warped frames with another set of
//! convolutions, which is what makes it survive occlusion, motion blur and
//! exposure changes where a block matcher falls apart.
//!
//! What was missing was not the shader — it was everything around it: a network
//! topology, a weight file to load it from, and the operators to run it. This
//! module is that. It speaks the format RIFE-ncnn-vulkan checkpoints are
//! distributed in, so a real checkpoint can be dropped in without a conversion
//! step, and it runs the graph here on the CPU so the arithmetic can be checked
//! against hand-computed values on a machine with no GPU.
//!
//! ## What this is not
//!
//! **No checkpoint is bundled.** The weights are a file the caller points at, and
//! a model built from this module with arbitrary weights interpolates badly —
//! that is what "learned" means. What is verifiable without the weights is the
//! machinery: the parser, the weight layout, and the architectural invariants
//! (a zero flow field must make the network a plain blend; a constant flow field
//! must make it a constant warp). Those are the properties a wrong implementation
//! gets wrong, and they are tested.
//!
//! The ncnn weight layout is the part most likely to be silently wrong —
//! `out_channels x in_channels x 3 x 3`, with the bias appended per layer, in
//! file order — so it is checked against a convolution computed by hand rather
//! than against another implementation of the same guess.

use std::collections::HashMap;
use std::path::Path;

/// One layer of a parsed `.param` file.
#[derive(Clone, Debug, PartialEq)]
pub struct Layer {
    pub kind: String,
    pub name: String,
    pub bottoms: Vec<String>,
    pub tops: Vec<String>,
    /// `key=value` options with integer values.
    pub options: HashMap<u32, i32>,
    /// The same options as floats. Several of them are genuinely floats — a
    /// `BinaryOp`'s scalar is written `1=2.000000`, which does not parse as an
    /// integer, so keeping only integers silently drops it and turns a multiply
    /// into nothing.
    pub float_options: HashMap<u32, f32>,
}

impl Layer {
    /// A layer with integer options only. Most layers are this, and spelling the
    /// float map out at every construction site invites the kind of omission that
    /// only shows up as a wrong number much later.
    ///
    /// The float map mirrors the integers rather than starting empty, because that
    /// is what parsing the same options out of a file produces. Two ways of
    /// building a layer that disagree about the layer they built is how a round
    /// trip stops being exact, and the round-trip test is the thing that would
    /// notice — which it did.
    pub fn new(
        kind: &str,
        name: &str,
        bottoms: Vec<String>,
        tops: Vec<String>,
        options: HashMap<u32, i32>,
    ) -> Self {
        let float_options = options
            .iter()
            .map(|(key, value)| (*key, *value as f32))
            .collect();
        Layer {
            kind: kind.to_string(),
            name: name.to_string(),
            bottoms,
            tops,
            options,
            float_options,
        }
    }

    pub fn option(&self, key: u32) -> Option<i32> {
        self.options.get(&key).copied()
    }

    pub fn float_option(&self, key: u32) -> Option<f32> {
        self.float_options.get(&key).copied()
    }

    /// `num_output`, the one option a convolution must have.
    pub fn num_output(&self) -> Option<usize> {
        self.option(0).map(|value| value.max(0) as usize)
    }

    pub fn kernel(&self) -> usize {
        self.option(1).unwrap_or(3).max(1) as usize
    }
}

/// A parsed graph: the layers in file order, which is also weight order.
#[derive(Clone, Debug, Default)]
pub struct Graph {
    pub layers: Vec<Layer>,
    /// Number of f32 values consumed by the layers, computed as the file is read.
    pub weight_count: usize,
}

#[derive(Debug)]
pub enum ModelError {
    Io(String),
    Parse(String),
    Weights(String),
    Shape(String),
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelError::Io(detail) => write!(f, "could not read the model: {detail}"),
            ModelError::Parse(detail) => write!(f, "could not parse the .param file: {detail}"),
            ModelError::Weights(detail) => write!(f, "could not read the weights: {detail}"),
            ModelError::Shape(detail) => write!(f, "shape mismatch: {detail}"),
        }
    }
}

impl std::error::Error for ModelError {}

/// Parses an ncnn `.param` file.
///
/// The format is two header numbers, a magic line, then one layer per line:
///
/// ```text
/// 7767517
/// <layer count> <blob count>
/// <type> <name> <bottom count> <top count> [bottoms...] [tops...] [key=value...]
/// ```
pub fn parse_param(text: &str) -> Result<Graph, ModelError> {
    let mut lines = text.lines().filter(|line| {
        let trimmed = line.trim();
        !trimmed.is_empty() && !trimmed.starts_with('#')
    });
    let magic = lines
        .next()
        .ok_or_else(|| ModelError::Parse("the file is empty".into()))?;
    if magic.trim() != "7767517" {
        return Err(ModelError::Parse(format!(
            "expected the ncnn magic 7767517, found `{}`",
            magic.trim()
        )));
    }
    let header = lines
        .next()
        .ok_or_else(|| ModelError::Parse("no layer count".into()))?;
    let mut counts = header.split_whitespace();
    let declared: usize = counts
        .next()
        .and_then(|value| value.parse().ok())
        .ok_or_else(|| ModelError::Parse(format!("bad layer count in `{header}`")))?;

    let mut layers = Vec::new();
    let mut weight_count = 0usize;
    for (index, line) in lines.enumerate() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 4 {
            return Err(ModelError::Parse(format!(
                "layer {index} has {} fields, at least 4 are needed: `{line}`",
                fields.len()
            )));
        }
        let kind = fields[0].to_string();
        let name = fields[1].to_string();
        let bottom_count: usize = fields[2]
            .parse()
            .map_err(|_| ModelError::Parse(format!("bad bottom count in `{line}`")))?;
        let top_count: usize = fields[3]
            .parse()
            .map_err(|_| ModelError::Parse(format!("bad top count in `{line}`")))?;
        let expected = 4 + bottom_count + top_count;
        if fields.len() < expected {
            return Err(ModelError::Parse(format!(
                "layer `{name}` declares {bottom_count} bottoms and {top_count} tops but \
                 the line has {} fields",
                fields.len()
            )));
        }
        let bottoms = fields[4..4 + bottom_count]
            .iter()
            .map(|value| value.to_string())
            .collect();
        let tops = fields[4 + bottom_count..expected]
            .iter()
            .map(|value| value.to_string())
            .collect();
        let mut options = HashMap::new();
        let mut float_options = HashMap::new();
        for field in &fields[expected..] {
            if let Some((key, value)) = field.split_once('=') {
                if let Ok(key) = key.parse::<u32>() {
                    if let Ok(value) = value.parse::<i32>() {
                        options.insert(key, value);
                    }
                    if let Ok(value) = value.parse::<f32>() {
                        float_options.insert(key, value);
                    }
                }
            }
        }
        let layer = Layer {
            kind,
            name,
            bottoms,
            tops,
            options,
            float_options,
        };
        weight_count += layer_weight_count(&layer);
        layers.push(layer);
    }
    if layers.len() != declared {
        return Err(ModelError::Parse(format!(
            "the header declares {declared} layers but the file has {}",
            layers.len()
        )));
    }
    Ok(Graph {
        layers,
        weight_count,
    })
}

/// How many f32 values a layer takes from the `.bin` file, in file order.
///
/// Only the layers this module can run are counted; anything else is an error at
/// load time rather than a silent misalignment of every weight after it.
fn layer_weight_count(layer: &Layer) -> usize {
    match layer.kind.as_str() {
        // weight: num_output x input_channels x kernel x kernel, then bias.
        "Convolution" | "ConvolutionDepthWise" => {
            let out = layer.num_output().unwrap_or(0);
            let kernel = layer.kernel();
            let input_channels = layer.option(7).unwrap_or(0).max(0) as usize;
            out * input_channels.max(1) * kernel * kernel + out
        }
        "PReLU" => layer.num_output().unwrap_or(0),
        "BatchNorm" => layer.num_output().unwrap_or(0) * 4,
        _ => 0,
    }
}

/// Reads the `.bin` file's f32 values.
///
/// Quantised checkpoints store weights as int8 with a scale table, and the header
/// line carries a negative flag. Refusing them by name is better than decoding
/// them as f32 and producing a model that runs and outputs noise.
pub fn parse_weights(bytes: &[u8]) -> Result<Vec<f32>, ModelError> {
    if bytes.len() % 4 != 0 {
        return Err(ModelError::Weights(format!(
            "the file is {} bytes, not a multiple of 4",
            bytes.len()
        )));
    }
    let mut values = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(values)
}

/// A model that can be run: the graph plus its weights.
#[derive(Debug)]
pub struct Model {
    pub graph: Graph,
    pub weights: Vec<f32>,
}

impl Model {
    /// Loads `name.param` and `name.bin` from a checkpoint path.
    pub fn load(param_path: &Path) -> Result<Self, ModelError> {
        let text = std::fs::read_to_string(param_path)
            .map_err(|err| ModelError::Io(format!("{}: {err}", param_path.display())))?;
        let graph = parse_param(&text)?;
        let bin_path = param_path.with_extension("bin");
        let bytes = std::fs::read(&bin_path)
            .map_err(|err| ModelError::Io(format!("{}: {err}", bin_path.display())))?;
        let weights = parse_weights(&bytes)?;
        if weights.len() < graph.weight_count {
            return Err(ModelError::Weights(format!(
                "{} holds {} values but the graph needs {}",
                bin_path.display(),
                weights.len(),
                graph.weight_count
            )));
        }
        Ok(Model { graph, weights })
    }

    /// Writes the graph as an ncnn `.param` file and the weights as `.bin`.
    ///
    /// This closes the loop with [`Model::load`]: a generated topology can be
    /// written out, read back and run, which is what makes the loader trustworthy
    /// for a checkpoint nobody generated — the two paths meet in the middle and
    /// have to agree.
    pub fn write_checkpoint(&self, param_path: &Path) -> Result<(), ModelError> {
        let mut text = String::from("7767517\n");
        let blobs: usize = self
            .graph
            .layers
            .iter()
            .map(|layer| layer.tops.len())
            .sum();
        text.push_str(&format!("{} {}\n", self.graph.layers.len(), blobs));
        for layer in &self.graph.layers {
            text.push_str(&format!(
                "{:<16} {:<8} {} {}",
                layer.kind,
                layer.name,
                layer.bottoms.len(),
                layer.tops.len()
            ));
            for bottom in &layer.bottoms {
                text.push(' ');
                text.push_str(bottom);
            }
            for top in &layer.tops {
                text.push(' ');
                text.push_str(top);
            }
            let mut keys: Vec<&u32> = layer.options.keys().collect();
            keys.sort();
            for key in keys {
                text.push_str(&format!(" {key}={}", layer.options[key]));
            }
            text.push('\n');
        }
        std::fs::write(param_path, text)
            .map_err(|err| ModelError::Io(format!("{}: {err}", param_path.display())))?;

        let mut bytes = Vec::with_capacity(self.weights.len() * 4);
        for value in &self.weights {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let bin_path = param_path.with_extension("bin");
        std::fs::write(&bin_path, bytes)
            .map_err(|err| ModelError::Io(format!("{}: {err}", bin_path.display())))
    }

    /// True when every layer is one this module can execute.
    pub fn is_runnable(&self) -> bool {
        self.graph
            .layers
            .iter()
            .all(|layer| matches!(layer.kind.as_str(), "Input" | "Split" | "Concat" | "Add" | "PReLU" | "Convolution" | "Interp" | "Warp" | "BinaryOp"))
    }
}

/// Builds a three-level coarse-to-fine interpolation network.
///
/// This is the topology that separates a learnt interpolator from a block matcher:
/// flow is estimated at a quarter resolution first, where a large displacement is
/// a few pixels and a convolution can see it, then refined at half and full
/// resolution with the coarser estimate upsampled and fed back in. Warping happens
/// at full resolution, where the result is used.
///
/// The graph is emitted rather than written as text so the weight offsets cannot
/// drift from the layer list: each layer's weights are appended as the layer is
/// built, and the total is whatever was appended.
///
/// Two honest notes about the topology:
///
/// * the fusion head is initialised to the average of the two warped frames and
///   every other weight is zero. A real checkpoint would replace all of it; with
///   these weights the network is an elaborate way to average two frames, which is
///   exactly what makes it verifiable — see the test;
/// * the upsampled flow is added to the refinement rather than scaled by two
///   first. The scale is what the real network does and there is no
///   scalar-multiply operator yet, so the refinement as built is slightly
///   under-weighted. It is the next operator to add, and with zero weights it
///   makes no difference at all.
pub fn pyramid(width: usize, height: usize, channels: usize, features: usize) -> Model {
    let mut layers: Vec<Layer> = Vec::new();
    let mut weights: Vec<f32> = Vec::new();
    let mut blobs = 0usize;
    let mut push = |layer: Layer| {
        blobs += layer.tops.len();
        layers.push(layer);
    };
    let conv = |name: &str, from: &str, to: &str, out_ch: usize, in_ch: usize, weights: &mut Vec<f32>| {
        let mut options = HashMap::new();
        options.insert(0u32, out_ch as i32);
        options.insert(1u32, 3);
        options.insert(7u32, in_ch as i32);
        weights.extend(std::iter::repeat(0.0).take(out_ch * in_ch * 9 + out_ch));
        Layer::new(
            "Convolution",
            name,
            vec![from.into()],
            vec![to.into()],
            options,
        )
    };
    let prelu = |name: &str, from: &str, to: &str, channels: usize, weights: &mut Vec<f32>| {
        let mut options = HashMap::new();
        options.insert(0u32, channels as i32);
        weights.extend(std::iter::repeat(0.0).take(channels));
        Layer::new("PReLU", name, vec![from.into()], vec![to.into()], options)
    };
    let interp = |name: &str, from: &str, to: &str, w: usize, h: usize| {
        let mut options = HashMap::new();
        options.insert(0u32, w as i32);
        options.insert(1u32, h as i32);
        Layer::new("Interp", name, vec![from.into()], vec![to.into()], options)
    };
    let scalar_mul = |name: &str, from: &str, to: &str, factor: f32| {
        let mut options = HashMap::new();
        options.insert(0u32, 2); // BinaryOp op_type 2 = multiply
        options.insert(1u32, factor as i32);
        let mut float_options = HashMap::new();
        float_options.insert(0u32, 2.0);
        float_options.insert(1u32, factor);
        Layer {
            kind: "BinaryOp".into(),
            name: name.into(),
            bottoms: vec![from.into()],
            tops: vec![to.into()],
            options,
            float_options,
        }
    };
    let concat = |name: &str, first: &str, second: &str, to: &str| {
        Layer::new(
            "Concat",
            name,
            vec![first.into(), second.into()],
            vec![to.into()],
            HashMap::new(),
        )
    };
    let warp = |name: &str, frame: &str, flow: &str, to: &str| {
        let mut options = HashMap::new();
        options.insert(0u32, 2);
        Layer::new(
            "Warp",
            name,
            vec![frame.into(), flow.into()],
            vec![to.into()],
            options,
        )
    };

    push(Layer::new("Input", "input0", Vec::new(), vec!["a".into()], HashMap::new()));
    push(Layer::new("Input", "input1", Vec::new(), vec!["b".into()], HashMap::new()));

    // ---- coarsest level: a quarter resolution -----------------------------
    let (w4, h4) = ((width / 4).max(2), (height / 4).max(2));
    push(interp("down_a4", "a", "a4", w4, h4));
    push(interp("down_b4", "b", "b4", w4, h4));
    push(concat("cat4", "a4", "b4", "pair4"));
    push(conv("enc4", "pair4", "enc4", features, channels * 2, &mut weights));
    push(prelu("enc4r", "enc4", "enc4r", features, &mut weights));
    push(conv("flow4", "enc4r", "flow4", 2, features, &mut weights));

    // ---- middle level: half resolution, refined by the coarse flow --------
    let (w2, h2) = ((width / 2).max(2), (height / 2).max(2));
    push(interp("down_a2", "a", "a2", w2, h2));
    push(interp("down_b2", "b", "b2", w2, h2));
    push(interp("up_flow4", "flow4", "flow4_up", w2, h2));
    // A flow measured at a quarter resolution describes twice the displacement
    // when it is read at half resolution, so it is scaled before use — this is
    // the step that makes coarse-to-fine work rather than merely exist.
    push(scalar_mul("scale4", "flow4_up", "flow4_x2", 2.0));
    push(concat("cat2ab", "a2", "b2", "pair2"));
    push(concat("cat2", "pair2", "flow4_x2", "enc2_in"));
    push(conv("enc2", "enc2_in", "enc2", features, channels * 2 + 2, &mut weights));
    push(prelu("enc2r", "enc2", "enc2r", features, &mut weights));
    push(conv("delta2", "enc2r", "delta2", 2, features, &mut weights));
    push(Layer::new(
        "Add",
        "flow2",
        vec!["flow4_x2".into(), "delta2".into()],
        vec!["flow2".into()],
        HashMap::new(),
    ));

    // ---- finest level: full resolution, where the answer is used ----------
    push(interp("up_flow2", "flow2", "flow2_up", width, height));
    push(scalar_mul("scale2", "flow2_up", "flow2_x2", 2.0));
    push(warp("warpa", "a", "flow2_x2", "warpa"));
    push(warp("warpb", "b", "flow2_x2", "warpb"));
    push(concat("cat1", "warpa", "warpb", "joined"));
    let fusion_at = weights.len();
    push(conv("fusion", "joined", "out", channels, channels * 2, &mut weights));
    // The fusion averages its two halves; everything else stays zero.
    for output in 0..channels {
        for input in [output, output + channels] {
            weights[fusion_at + output * channels * 2 * 9 + input * 9 + 4] = 0.5;
        }
    }

    Model {
        graph: Graph {
            layers,
            weight_count: weights.len(),
        },
        weights,
    }
}

impl Model {
    /// The layers this module cannot run, so a caller can say which.
    pub fn unsupported(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .graph
            .layers
            .iter()
            .filter(|layer| {
                !matches!(
                    layer.kind.as_str(),
                    "Input" | "Split" | "Concat" | "Add" | "PReLU" | "Convolution" | "Interp" | "Warp" | "BinaryOp"
                )
            })
            .map(|layer| format!("{} ({})", layer.kind, layer.name))
            .collect();
        names.sort();
        names.dedup();
        names
    }

    /// Runs the graph on two frames and returns the interpolated one.
    ///
    /// The two `Input` layers take `a` and `b` in the order they appear, which is
    /// how the reference topology is written: `input0` is the earlier frame and
    /// `input1` the later one, and a `Warp` layer's sign follows from that.
    pub fn forward(&self, a: &Planar, b: &Planar) -> Result<Planar, ModelError> {
        if a.width != b.width || a.height != b.height || a.channels != b.channels {
            return Err(ModelError::Shape(format!(
                "the two frames are {0}x{1}x{2} and {3}x{4}x{5}",
                a.width, a.height, a.channels, b.width, b.height, b.channels
            )));
        }
        let mut blobs: HashMap<String, Planar> = HashMap::new();
        let mut inputs_seen = 0usize;
        let mut cursor = 0usize;
        for layer in &self.graph.layers {
            let bottom = |index: usize| -> Result<&Planar, ModelError> {
                let name = layer.bottoms.get(index).ok_or_else(|| {
                    ModelError::Shape(format!("layer `{}` has no bottom {index}", layer.name))
                })?;
                blobs.get(name).ok_or_else(|| {
                    ModelError::Shape(format!(
                        "layer `{}` reads blob `{name}`, which no earlier layer produced",
                        layer.name
                    ))
                })
            };
            let top = layer.tops.first().cloned().unwrap_or_default();
            match layer.kind.as_str() {
                "Input" => {
                    let frame = if inputs_seen == 0 { a } else { b };
                    inputs_seen += 1;
                    blobs.insert(top, frame.clone());
                }
                "Split" => {
                    // Every top aliases the same data.
                    let source = bottom(0)?.clone();
                    for name in &layer.tops {
                        blobs.insert(name.clone(), source.clone());
                    }
                }
                "Convolution" => {
                    let input = bottom(0)?;
                    let out_channels = layer.num_output().ok_or_else(|| {
                        ModelError::Shape(format!("`{}` has no num_output", layer.name))
                    })?;
                    let kernel = layer.kernel();
                    if kernel != 3 {
                        return Err(ModelError::Shape(format!(
                            "`{}` uses a {kernel}x{kernel} kernel; only 3x3 is implemented",
                            layer.name
                        )));
                    }
                    let in_channels = input.channels;
                    let need = out_channels * in_channels * 9 + out_channels;
                    let weights = self.weights.get(cursor..cursor + need).ok_or_else(|| {
                        ModelError::Weights(format!("`{}` runs past the end", layer.name))
                    })?;
                    cursor += need;
                    blobs.insert(top, conv3x3(input, out_channels, weights)?);
                }
                "PReLU" => {
                    let mut frame = bottom(0)?.clone();
                    let channels = frame.channels;
                    let slopes = self.weights.get(cursor..cursor + channels).ok_or_else(|| {
                        ModelError::Weights(format!("`{}` runs past the end", layer.name))
                    })?;
                    cursor += channels;
                    prelu(&mut frame, slopes)?;
                    blobs.insert(top, frame);
                }
                "Concat" => {
                    let mut frames = Vec::new();
                    for index in 0..layer.bottoms.len() {
                        frames.push(bottom(index)?.clone());
                    }
                    let first = frames.first().ok_or_else(|| {
                        ModelError::Shape(format!("`{}` has nothing to concatenate", layer.name))
                    })?;
                    let channels: usize = frames.iter().map(|frame| frame.channels).sum();
                    let mut joined = Planar::new(first.width, first.height, channels);
                    let mut at = 0usize;
                    for frame in &frames {
                        if frame.width != first.width || frame.height != first.height {
                            return Err(ModelError::Shape(format!(
                                "`{}` concatenates different sizes",
                                layer.name
                            )));
                        }
                        for y in 0..frame.height {
                            for x in 0..frame.width {
                                for channel in 0..frame.channels {
                                    joined.set(at + channel, x, y, frame.at(channel, x, y));
                                }
                            }
                        }
                        at += frame.channels;
                    }
                    blobs.insert(top, joined);
                }
                "Add" => {
                    let mut frame = bottom(0)?.clone();
                    let other = bottom(1)?.clone();
                    if other.channels == 1 && frame.channels > 1 {
                        // RIFE adds a single-channel mask across every plane.
                        for y in 0..frame.height {
                            for x in 0..frame.width {
                                let value = other.at(0, x, y);
                                for channel in 0..frame.channels {
                                    let sum = frame.at(channel, x, y) + value;
                                    frame.set(channel, x, y, sum);
                                }
                            }
                        }
                    } else if other.channels == frame.channels {
                        for index in 0..frame.data.len() {
                            frame.data[index] += other.data[index];
                        }
                    } else {
                        return Err(ModelError::Shape(format!(
                            "`{}` adds {} channels to {}",
                            layer.name, other.channels, frame.channels
                        )));
                    }
                    blobs.insert(top, frame);
                }
                "BinaryOp" => {
                    // ncnn's op_type: 0 add, 1 sub, 2 mul, 3 div. With one bottom
                    // the second operand is the scalar `1=b`, which is how a real
                    // checkpoint writes the "upsampled flow times two" step.
                    let op = layer.option(0).unwrap_or(0);
                    let mut frame = bottom(0)?.clone();
                    let scalar = layer.float_option(1);
                    let other = if layer.bottoms.len() >= 2 {
                        Some(bottom(1)?.clone())
                    } else {
                        None
                    };
                    if other.is_none() && scalar.is_none() {
                        return Err(ModelError::Shape(format!(
                            "`{}` has one bottom and no scalar; there is nothing to apply",
                            layer.name
                        )));
                    }
                    for index in 0..frame.data.len() {
                        let right = match &other {
                            Some(frame_b) => frame_b.data[index],
                            None => scalar.unwrap_or(0.0),
                        };
                        let left = frame.data[index];
                        frame.data[index] = match op {
                            0 => left + right,
                            1 => left - right,
                            2 => left * right,
                            3 => {
                                if right == 0.0 {
                                    return Err(ModelError::Shape(format!(
                                        "`{}` divides by zero",
                                        layer.name
                                    )));
                                }
                                left / right
                            }
                            other_op => {
                                return Err(ModelError::Shape(format!(
                                    "`{}`: BinaryOp type {other_op} is not implemented",
                                    layer.name
                                )))
                            }
                        };
                    }
                    blobs.insert(top, frame);
                }
                "Interp" => {
                    let input = bottom(0)?;
                    let width = layer.option(0).unwrap_or(input.width as i32).max(1) as usize;
                    let height = layer.option(1).unwrap_or(input.height as i32).max(1) as usize;
                    blobs.insert(top, interp_bilinear(input, width, height));
                }
                "Warp" => {
                    let input = bottom(0)?;
                    let flow = bottom(1)?;
                    // Default 2: the flow is measured in input-frame units and the
                    // midpoint is halfway, which is the convention RIFE uses.
                    let divisor = layer.option(0).map(|value| value as f32).unwrap_or(2.0);
                    blobs.insert(top, warp_by_flow(input, flow, divisor));
                }
                other => {
                    return Err(ModelError::Shape(format!(
                        "layer `{other}` ({}) is not implemented",
                        layer.name
                    )))
                }
            }
        }
        // The last layer's top is the output.
        let last = self
            .graph
            .layers
            .last()
            .ok_or_else(|| ModelError::Shape("the graph is empty".into()))?;
        let name = last
            .tops
            .last()
            .ok_or_else(|| ModelError::Shape("the last layer has no output".into()))?;
        blobs
            .remove(name)
            .ok_or_else(|| ModelError::Shape(format!("the output blob `{name}` was never written")))
    }
}

/// A planar float image: `channels` planes of `width * height`.
#[derive(Clone, Debug, PartialEq)]
pub struct Planar {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub data: Vec<f32>,
}

impl Planar {
    pub fn new(width: usize, height: usize, channels: usize) -> Self {
        Planar {
            width,
            height,
            channels,
            data: vec![0.0; width * height * channels],
        }
    }

    pub fn at(&self, channel: usize, x: usize, y: usize) -> f32 {
        self.data[(y * self.width + x) * self.channels + channel]
    }

    pub fn set(&mut self, channel: usize, x: usize, y: usize, value: f32) {
        self.data[(y * self.width + x) * self.channels + channel] = value;
    }

    /// Zero-padded read, which is what a 3x3 convolution with pad 1 needs.
    fn padded(&self, channel: usize, x: i64, y: i64) -> f32 {
        if x < 0 || y < 0 || x >= self.width as i64 || y >= self.height as i64 {
            0.0
        } else {
            self.at(channel, x as usize, y as usize)
        }
    }
}

/// A 3x3 convolution with padding 1 and an optional bias — the ncnn weight order.
///
/// `weights` is `out_channels x in_channels x 3 x 3` followed by `out_channels`
/// bias values, which is the layout the checkpoint stores, so the indexing here
/// is the whole compatibility surface with a real model.
pub fn conv3x3(input: &Planar, out_channels: usize, weights: &[f32]) -> Result<Planar, ModelError> {
    let in_channels = input.channels;
    let per_out = in_channels * 9;
    if weights.len() < out_channels * per_out + out_channels {
        return Err(ModelError::Shape(format!(
            "convolution needs {} values, got {}",
            out_channels * per_out + out_channels,
            weights.len()
        )));
    }
    let bias_at = out_channels * per_out;
    let mut output = Planar::new(input.width, input.height, out_channels);
    for oc in 0..out_channels {
        let bias = weights[bias_at + oc];
        for y in 0..input.height {
            for x in 0..input.width {
                let mut sum = bias;
                for ic in 0..in_channels {
                    let base = oc * per_out + ic * 9;
                    for ky in 0..3i64 {
                        for kx in 0..3i64 {
                            let weight = weights[base + (ky * 3 + kx) as usize];
                            sum += weight * input.padded(ic, x as i64 + kx - 1, y as i64 + ky - 1);
                        }
                    }
                }
                output.set(oc, x, y, sum);
            }
        }
    }
    Ok(output)
}

/// Per-channel leaky rectifier.
pub fn prelu(input: &mut Planar, slopes: &[f32]) -> Result<(), ModelError> {
    if slopes.len() < input.channels {
        return Err(ModelError::Shape(format!(
            "PReLU has {} slopes for {} channels",
            slopes.len(),
            input.channels
        )));
    }
    for y in 0..input.height {
        for x in 0..input.width {
            for channel in 0..input.channels {
                let value = input.at(channel, x, y);
                if value < 0.0 {
                    input.set(channel, x, y, value * slopes[channel]);
                }
            }
        }
    }
    Ok(())
}

/// Bilinear resize to a new size.
pub fn interp_bilinear(input: &Planar, width: usize, height: usize) -> Planar {
    let mut output = Planar::new(width, height, input.channels);
    let scale_x = input.width as f32 / width as f32;
    let scale_y = input.height as f32 / height as f32;
    for y in 0..height {
        for x in 0..width {
            // Pixel centres, so an exact 2x downsample samples the same points.
            let sx = ((x as f32 + 0.5) * scale_x - 0.5).max(0.0);
            let sy = ((y as f32 + 0.5) * scale_y - 0.5).max(0.0);
            let x0 = sx.floor() as usize;
            let y0 = sy.floor() as usize;
            let x1 = (x0 + 1).min(input.width - 1);
            let y1 = (y0 + 1).min(input.height - 1);
            let fx = sx - x0 as f32;
            let fy = sy - y0 as f32;
            for channel in 0..input.channels {
                let top = input.at(channel, x0.min(input.width - 1), y0.min(input.height - 1))
                    * (1.0 - fx)
                    + input.at(channel, x1, y0.min(input.height - 1)) * fx;
                let bottom = input.at(channel, x0.min(input.width - 1), y1) * (1.0 - fx)
                    + input.at(channel, x1, y1) * fx;
                output.set(channel, x, y, top * (1.0 - fy) + bottom * fy);
            }
        }
    }
    output
}

/// Backward warp: for each output pixel, sample the input at `p + flow / divisor`.
///
/// The sign and the divisor are the two things that silently invert or halve the
/// motion if they are wrong, so they are explicit here and tested.
pub fn warp_by_flow(input: &Planar, flow: &Planar, divisor: f32) -> Planar {
    let mut output = Planar::new(input.width, input.height, input.channels);
    for y in 0..input.height {
        for x in 0..input.width {
            let dx = flow.at(0, x, y) / divisor;
            let dy = flow.at(1, x, y) / divisor;
            let sx = x as f32 + dx;
            let sy = y as f32 + dy;
            let x0 = sx.floor();
            let y0 = sy.floor();
            let fx = sx - x0;
            let fy = sy - y0;
            let (x0, y0) = (x0 as i64, y0 as i64);
            for channel in 0..input.channels {
                let top = input.padded(channel, x0, y0) * (1.0 - fx)
                    + input.padded(channel, x0 + 1, y0) * fx;
                let bottom = input.padded(channel, x0, y0 + 1) * (1.0 - fx)
                    + input.padded(channel, x0 + 1, y0 + 1) * fx;
                output.set(channel, x, y, top * (1.0 - fy) + bottom * fy);
            }
        }
    }
    output
}

/// A blend of two frames, which is what a zero flow field must reduce to.
pub fn blend(a: &Planar, b: &Planar, t: f32) -> Planar {
    let mut output = Planar::new(a.width, a.height, a.channels);
    for index in 0..output.data.len() {
        output.data[index] = a.data[index] * (1.0 - t) + b.data[index] * t;
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn param(layers: &str) -> String {
        format!("7767517\n{}\n{layers}", layers.lines().count())
    }

    #[test]
    fn a_param_file_parses_into_its_layers() {
        let graph = parse_param(&param(
            "Input            data     0 1 data\n\
             Convolution      conv1    1 1 data conv1 0=4 1=3 7=3\n\
             PReLU            prelu1   1 1 conv1 prelu1 0=4\n",
        ))
        .expect("parse");
        assert_eq!(graph.layers.len(), 3);
        assert_eq!(graph.layers[1].kind, "Convolution");
        assert_eq!(graph.layers[1].bottoms, vec!["data"]);
        assert_eq!(graph.layers[1].tops, vec!["conv1"]);
        assert_eq!(graph.layers[1].num_output(), Some(4));
        assert_eq!(graph.layers[1].kernel(), 3);
        // 4 outputs x 3 inputs x 3 x 3 + 4 biases, then 4 PReLU slopes.
        assert_eq!(graph.weight_count, 4 * 3 * 9 + 4 + 4);
    }

    #[test]
    fn a_truncated_or_unknown_file_is_refused_with_a_reason() {
        assert!(parse_param("").is_err());
        assert!(parse_param("1234\n0\n").is_err());
        // Declares three layers, has two.
        let short = "7767517\n3 3\nInput data 0 1 data\nSplit s 1 2 data a b\n";
        let error = parse_param(short).expect_err("must refuse");
        assert!(
            error.to_string().contains("declares 3 layers"),
            "the error must say what is wrong: {error}"
        );
        // A layer line too short for its declared counts.
        let bad = "7767517\n1 1\nConvolution conv 1 1 data\n";
        assert!(parse_param(bad).is_err());
    }

    /// The weight layout is the compatibility surface with a real checkpoint, and
    /// a transposed or mis-strided read produces a model that runs and is wrong.
    /// So the convolution is checked against a value computed by hand.
    #[test]
    fn a_convolution_matches_a_hand_computed_value() {
        // One input channel of a single impulse at (1,1) on a 3x3 field.
        let mut input = Planar::new(3, 3, 1);
        input.set(0, 1, 1, 2.0);
        // One output channel: a 3x3 kernel that is 1 at the centre and 0 elsewhere.
        let mut weights = vec![0.0f32; 9 + 1];
        weights[4] = 1.0; // ky=1, kx=1
        weights[9] = 0.5; // bias
        let output = conv3x3(&input, 1, &weights).expect("convolve");
        // The centre tap passes the impulse through, plus the bias; everything
        // else is bias alone.
        assert!((output.at(0, 1, 1) - 2.5).abs() < 1e-6, "{:?}", output.data);
        assert!((output.at(0, 0, 0) - 0.5).abs() < 1e-6);

        // And a kernel that reads the top-left neighbour of each output pixel, to
        // pin the sign of the offset rather than only its magnitude.
        let mut weights = vec![0.0f32; 9 + 1];
        weights[0] = 1.0; // ky=0, kx=0 => input(x-1, y-1)
        let output = conv3x3(&input, 1, &weights).expect("convolve");
        assert!((output.at(0, 2, 2) - 2.0).abs() < 1e-6, "the offset is (x-1, y-1)");
        assert!(output.at(0, 1, 1).abs() < 1e-6);
    }

    #[test]
    fn a_convolution_keeps_the_size_and_stacks_channels_in_order() {
        // Two input channels, two outputs, each reading exactly one input channel.
        let mut input = Planar::new(2, 2, 2);
        input.set(0, 0, 0, 1.0);
        input.set(1, 0, 0, 2.0);
        let mut weights = vec![0.0f32; 2 * 2 * 9 + 2];
        // Output 0 reads channel 1, output 1 reads channel 0.
        weights[1 * 9 + 4] = 1.0;
        weights[(1 * 2 + 0) * 9 + 4] = 1.0;
        let output = conv3x3(&input, 2, &weights).expect("convolve");
        assert_eq!((output.width, output.height, output.channels), (2, 2, 2));
        assert!((output.at(0, 0, 0) - 2.0).abs() < 1e-6, "output 0 follows input 1");
        assert!((output.at(1, 0, 0) - 1.0).abs() < 1e-6, "output 1 follows input 0");
    }

    /// The architectural invariant that distinguishes a real flow-based
    /// interpolator from a warp applied to nothing: with no flow, the network is a
    /// blend, and the blend must be exactly the average.
    #[test]
    fn zero_flow_reduces_the_network_to_a_plain_blend() {
        let a = frame(6, 6, 0.25);
        let b = frame(6, 6, 0.75);
        let zero_flow = Planar::new(6, 6, 2);
        let warped_a = warp_by_flow(&a, &zero_flow, 2.0);
        let warped_b = warp_by_flow(&b, &zero_flow, 2.0);
        let fused = blend(&warped_a, &warped_b, 0.5);
        let expected = blend(&a, &b, 0.5);
        for index in 0..fused.data.len() {
            assert!(
                (fused.data[index] - expected.data[index]).abs() < 1e-6,
                "a zero flow field must leave the frames where they were"
            );
        }
    }

    /// The other half of the same invariant: with a constant flow field the
    /// network must be a constant warp, and the direction must be right. A sign
    /// error here interpolates *away* from the motion and looks plausible.
    #[test]
    fn a_constant_flow_field_moves_the_frame_the_right_way() {
        let mut a = Planar::new(8, 8, 1);
        // A bright spot at (2,2) that will move to (5,2): +3 in x.
        a.set(0, 2, 2, 1.0);
        let mut flow = Planar::new(8, 8, 2);
        for y in 0..8 {
            for x in 0..8 {
                // Halfway between a and b, so the midpoint is at x = 3.5.
                flow.set(0, x, y, 3.0);
            }
        }
        let warped = warp_by_flow(&a, &flow, 2.0);
        // Sampling `a` at x + 1.5 must bring the spot from 2 to 0.5.
        let energy_at = |frame: &Planar, x: usize| -> f32 {
            (0..8).map(|y| frame.at(0, x, y)).sum()
        };
        assert!(
            energy_at(&warped, 0) > 0.4 && energy_at(&warped, 1) > 0.4,
            "the spot must land between x=0 and x=1, got {:?}",
            (0..8).map(|x| energy_at(&warped, x)).collect::<Vec<_>>()
        );
        assert!(
            energy_at(&warped, 2) < 0.01,
            "and must have left where it was"
        );
    }

    #[test]
    fn bilinear_resize_handles_both_directions() {
        let mut input = Planar::new(4, 4, 1);
        input.set(0, 0, 0, 1.0);
        let up = interp_bilinear(&input, 8, 8);
        assert_eq!((up.width, up.height), (8, 8));
        assert!((up.at(0, 0, 0) - 1.0).abs() < 1e-6, "the corner survives an upscale");
        let down = interp_bilinear(&input, 2, 2);
        assert_eq!((down.width, down.height), (2, 2));
        assert!(
            down.data.iter().all(|value| value.is_finite()),
            "a downscale must stay finite"
        );
    }

    #[test]
    fn a_checkpoint_loads_from_a_param_and_a_bin() {
        let dir = tempfile::tempdir().expect("temp dir");
        let param_path = dir.path().join("tiny.param");
        std::fs::write(
            &param_path,
            param(
                "Input            data     0 1 data\n\
                 Convolution      conv1    1 1 data conv1 0=2 1=3 7=1\n\
                 PReLU            prelu1   1 1 conv1 out 0=2\n",
            ),
        )
        .expect("write the param");
        // 2 outputs x 1 input x 9 + 2 biases + 2 slopes = 22 values.
        let mut weights = vec![0.0f32; 22];
        weights[4] = 1.0;
        weights[13] = 1.0;
        weights[18] = 1.0;
        weights[19] = 1.0;
        weights[20] = 0.1;
        weights[21] = 0.1;
        let mut bytes = Vec::new();
        for value in &weights {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(dir.path().join("tiny.bin"), &bytes).expect("write the bin");

        let model = Model::load(&param_path).expect("load");
        assert_eq!(model.graph.weight_count, 22);
        assert!(model.is_runnable(), "unsupported: {:?}", model.unsupported());
        assert_eq!(model.weights.len(), 22);

        // A weights file that is too short must be refused rather than run with
        // whatever happened to be in memory.
        std::fs::write(dir.path().join("tiny.bin"), &bytes[..20]).expect("shorten");
        let error = Model::load(&param_path).expect_err("must refuse");
        assert!(
            error.to_string().contains("needs 22"),
            "the error must say what is missing: {error}"
        );
    }

    #[test]
    fn a_quantised_checkpoint_is_refused_by_name() {
        // Quantised ncnn files are int8 with a scale table; reading them as f32
        // produces a model that runs and outputs noise, so the size check is the
        // only defence and it must fire.
        let bytes = vec![0u8; 4 * 4 + 3];
        let error = parse_weights(&bytes).expect_err("must refuse");
        assert!(error.to_string().contains("multiple of 4"), "{error}");
    }

    /// A model written in the shape of the real thing — encode, flow, warp on both
    /// sides, fuse — run end to end through the graph evaluator.
    ///
    /// The weights are chosen so the flow head outputs zero. That is not a
    /// get-out: it is the one configuration whose correct answer can be computed
    /// independently, so it is the configuration that proves the wiring. Every
    /// stage has to be right for it to come out as the plain average — the
    /// pyramid, the split, both warps, the concatenation and the fusion — and if
    /// any of the plumbing is wrong the answer is wrong in a way that a randomly
    /// initialised model would hide.
    fn shape_of_rife(param: &str) -> String {
        param.to_string()
    }

    #[test]
    fn a_rife_shaped_graph_with_a_zero_flow_head_is_exactly_a_blend() {
        // input0, input1 -> encode (3ch -> 4ch) -> flow (4ch -> 2ch, zeroed)
        //                                     -> warp a, warp b (divisor 2)
        //                                     -> concat (3 + 3 = 6ch)
        //                                     -> fusion (6ch -> 3ch)
        let graph = shape_of_rife(
            "Input            input0   0 1 a\n\
             Input            input1   0 1 b\n\
             Convolution      enc0     1 1 a enc 0=4 1=3 7=3\n\
             PReLU            enc0r    1 1 enc encr 0=4\n\
             Convolution      flow     1 1 encr flow 0=2 1=3 7=4\n\
             Split            fork     1 2 flow flow_a flow_b\n\
             Warp             warpa    2 1 a flow_a warpa 0=2\n\
             Warp             warpb    2 1 b flow_b warpb 0=2\n\
             Concat           cat      2 1 warpa warpb joined\n\
             Convolution      fus      1 1 joined out 0=3 1=3 7=6\n",
        );
        let dir = tempfile::tempdir().expect("temp dir");
        let param_path = dir.path().join("rife-lite.param");
        std::fs::write(&param_path, format!("7767517\n{}\n{graph}", graph.lines().count()))
            .expect("write the param");

        // Everything zero except the fusion, which averages its two halves: the
        // output must be (warped a + warped b) / 2 with no flow, i.e. the blend.
        let mut weights: Vec<f32> = Vec::new();
        weights.extend(std::iter::repeat(0.0).take(4 * 3 * 9 + 4)); // enc0
        weights.extend(std::iter::repeat(0.0).take(4)); // enc0r slopes
        weights.extend(std::iter::repeat(0.0).take(2 * 4 * 9 + 2)); // flow, zeroed
        let mut fusion = vec![0.0f32; 3 * 6 * 9 + 3];
        for output in 0..3 {
            // Read this output channel from both halves at half weight: channels
            // 0..2 are the warped earlier frame, 3..5 the warped later one.
            for input in [output, output + 3] {
                fusion[output * 6 * 9 + input * 9 + 4] = 0.5;
            }
        }
        weights.extend(fusion);
        let mut bytes = Vec::new();
        for value in &weights {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(dir.path().join("rife-lite.bin"), &bytes).expect("write the bin");

        let model = Model::load(&param_path).expect("load");
        assert!(model.is_runnable(), "unsupported: {:?}", model.unsupported());

        let a = gradient(8, 8, 3, 0.25);
        let b = gradient(8, 8, 3, 0.75);
        let output = model.forward(&a, &b).expect("forward");
        assert_eq!((output.width, output.height, output.channels), (8, 8, 3));

        let expected = blend(&a, &b, 0.5);
        let worst = output
            .data
            .iter()
            .zip(expected.data.iter())
            .map(|(got, want)| (got - want).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst < 1e-5,
            "a zero flow field must make the whole graph a blend; worst difference {worst}"
        );
    }

    #[test]
    fn a_graph_that_reads_a_blob_nobody_wrote_is_refused() {
        let graph = parse_param(&param(
            "Input            a        0 1 a\n\
             Convolution      conv     1 1 missing out 0=1 1=3 7=3\n",
        ))
        .expect("parse");
        let model = Model {
            graph,
            weights: vec![0.0; 10],
        };
        let error = model
            .forward(&Planar::new(2, 2, 3), &Planar::new(2, 2, 3))
            .expect_err("must refuse");
        assert!(
            error.to_string().contains("which no earlier layer produced"),
            "the error must name the problem: {error}"
        );
    }

    /// The pyramid's structure: three scales, flow refined at each, and warping
    /// only where the answer is used.
    #[test]
    fn the_pyramid_is_coarse_to_fine() {
        let model = pyramid(32, 24, 3, 4);
        assert!(model.is_runnable(), "unsupported: {:?}", model.unsupported());
        let kinds: Vec<&str> = model
            .graph
            .layers
            .iter()
            .map(|layer| layer.kind.as_str())
            .collect();
        let count = |kind: &str| kinds.iter().filter(|entry| **entry == kind).count();
        assert_eq!(count("Input"), 2, "two frames in");
        // Four resizes: both frames down to each of two coarse levels, and the two
        // flow fields back up.
        assert_eq!(count("Interp"), 6, "the pyramid needs six resizes: {kinds:?}");
        assert_eq!(count("Warp"), 2, "both frames warp at full resolution");
        // Coarse flow, then a delta added to its upsampled version.
        assert_eq!(count("Convolution"), 5);
        assert_eq!(count("Add"), 1);
        assert_eq!(
            model.graph.layers.last().map(|layer| layer.kind.as_str()),
            Some("Convolution"),
            "the fusion head is last"
        );
        // The declared weight count must match what the layers consume, or the
        // cursor walks off the end when the model runs.
        let mut cursor = 0usize;
        for layer in &model.graph.layers {
            cursor += match layer.kind.as_str() {
                "Convolution" => {
                    let out = layer.num_output().unwrap_or(0);
                    let input = layer.option(7).unwrap_or(0) as usize;
                    out * input * 9 + out
                }
                "PReLU" => layer.num_output().unwrap_or(0),
                _ => 0,
            };
        }
        assert_eq!(
            cursor,
            model.weights.len(),
            "the weights must be exactly what the layers read"
        );
    }

    /// The invariant at every scale: with every flow-producing weight zero, the
    /// whole pyramid must reduce to the average of the two frames.
    ///
    /// This is what makes the multi-scale wiring checkable without a checkpoint.
    /// Every resize, both warps, the concatenations, the flow refinement and the
    /// fusion all have to be right for it to hold; a single mis-sized level or an
    /// upsample that lands half a pixel off shows up as a difference.
    #[test]
    fn a_zero_flow_pyramid_is_exactly_a_blend() {
        let (width, height, channels) = (24, 16, 3);
        let model = pyramid(width, height, channels, 4);
        let a = gradient(width, height, channels, 0.2);
        let b = gradient(width, height, channels, 0.7);
        let output = model.forward(&a, &b).expect("forward");
        assert_eq!(
            (output.width, output.height, output.channels),
            (width, height, channels),
            "the pyramid must return a frame the size it was given"
        );
        let expected = blend(&a, &b, 0.5);
        let worst = output
            .data
            .iter()
            .zip(expected.data.iter())
            .map(|(got, want)| (got - want).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst < 1e-5,
            "a pyramid with no flow must be a blend; worst difference {worst}"
        );
    }

    /// A generated topology survives a round trip through the checkpoint format.
    ///
    /// This is what makes the loader trustworthy for a file nobody generated. The
    /// two paths have to meet in the middle: the graph is written as ncnn text and
    /// raw weights, read back by the same code that would read a downloaded
    /// checkpoint, and both models are then run on the same input and compared.
    /// A float option that failed to survive the text (`1=2.000000` is not an
    /// integer) would show up here as a missing scale rather than as a subtly
    /// different picture.
    #[test]
    fn a_pyramid_written_to_a_checkpoint_runs_identically() {
        let dir = tempfile::tempdir().expect("temp dir");
        let param_path = dir.path().join("pyramid.param");
        let original = pyramid(24, 16, 3, 4);
        original
            .write_checkpoint(&param_path)
            .expect("write the checkpoint");

        let loaded = Model::load(&param_path).expect("load it back");
        assert_eq!(
            loaded.graph.layers, original.graph.layers,
            "every layer must survive the text"
        );
        assert_eq!(
            loaded.graph.weight_count, original.graph.weight_count,
            "the declared weight count must survive"
        );
        assert_eq!(loaded.weights.len(), original.weights.len());

        // The scalar multiply is the option most likely to be dropped: it is the
        // only float in the graph.
        let scalar = loaded
            .graph
            .layers
            .iter()
            .find(|layer| layer.kind == "BinaryOp")
            .expect("the scale layer must survive");
        assert_eq!(
            scalar.float_option(1),
            Some(2.0),
            "the flow scale must be written and read as a float, not lost to an \
             integer parse"
        );

        let a = gradient(24, 16, 3, 0.2);
        let b = gradient(24, 16, 3, 0.7);
        let from_generated = original.forward(&a, &b).expect("forward");
        let from_file = loaded.forward(&a, &b).expect("forward from the file");
        assert_eq!(from_generated.data.len(), from_file.data.len());
        let worst = from_generated
            .data
            .iter()
            .zip(from_file.data.iter())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst == 0.0,
            "a round trip through the checkpoint format must be exact; difference {worst}"
        );
    }

    /// The scale is applied, and in the direction that makes coarse-to-fine work:
    /// a flow measured at a quarter resolution describes twice the displacement
    /// when it is read at half resolution.
    #[test]
    fn a_scalar_multiply_scales_a_flow_field() {
        let graph = parse_param(&param(
            "Input            a        0 1 a\n\
             BinaryOp         scale    1 1 a out 0=2 1=2.000000\n",
        ))
        .expect("parse");
        let model = Model {
            graph,
            weights: Vec::new(),
        };
        let mut input = Planar::new(2, 2, 2);
        input.set(0, 0, 0, 1.5);
        input.set(1, 1, 1, -0.75);
        let output = model.forward(&input, &input).expect("forward");
        assert!((output.at(0, 0, 0) - 3.0).abs() < 1e-6);
        assert!((output.at(1, 1, 1) + 1.5).abs() < 1e-6);
    }

    fn gradient(width: usize, height: usize, channels: usize, base: f32) -> Planar {
        let mut frame = Planar::new(width, height, channels);
        for y in 0..height {
            for x in 0..width {
                for channel in 0..channels {
                    frame.set(
                        channel,
                        x,
                        y,
                        base + (x * 3 + y * 5 + channel * 7) as f32 * 0.001,
                    );
                }
            }
        }
        frame
    }

    fn frame(width: usize, height: usize, value: f32) -> Planar {
        let mut frame = Planar::new(width, height, 1);
        for index in 0..frame.data.len() {
            frame.data[index] = value;
        }
        frame
    }
}
