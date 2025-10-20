use std::path::PathBuf;

use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::VarBuilder;
use candle_transformers::models::vit::{Config, Model};
use clap::{arg, command, Parser};

const NUM_LABELS: usize = 2; // number of classifications. normal and nsfw means we have two
const MODEL_NAME: &'static str = "Falconsai/nsfw_image_detection";

pub const IMAGENET_MEAN: [f32; 3] = [0.485f32, 0.456, 0.406];
pub const IMAGENET_STD: [f32; 3] = [0.229f32, 0.224, 0.225];

/// Loads an image from disk using the image crate at the requested resolution,
/// using the given std and mean parameters.
/// This returns a tensor with shape (3, res, res). imagenet normalization is applied.
pub fn load_image_with_std_mean<P: AsRef<std::path::Path>>(
    p: P,
    res: usize,
    mean: &[f32; 3],
    std: &[f32; 3],
) -> candle_core::Result<Tensor> {
    let img = image::ImageReader::open(p)?
        .decode()
        .map_err(candle_core::Error::wrap)?
        .resize_to_fill(
            res as u32,
            res as u32,
            image::imageops::FilterType::Triangle,
        );
    let img = img.to_rgb8();
    let data = img.into_raw();
    let data = Tensor::from_vec(data, (res, res, 3), &Device::Cpu)?.permute((2, 0, 1))?;
    let mean = Tensor::new(mean, &Device::Cpu)?.reshape((3, 1, 1))?;
    let std = Tensor::new(std, &Device::Cpu)?.reshape((3, 1, 1))?;
    (data.to_dtype(DType::F32)? / 255.)?
        .broadcast_sub(&mean)?
        .broadcast_div(&std)
}

/// Loads an image from disk using the image crate at the requested resolution.
/// This returns a tensor with shape (3, res, res). imagenet normalization is applied.
pub fn load_image<P: AsRef<std::path::Path>>(p: P, res: usize) -> candle_core::Result<Tensor> {
    load_image_with_std_mean(p, res, &IMAGENET_MEAN, &IMAGENET_STD)
}

/// Loads an image from disk using the image crate, this returns a tensor with shape
/// (3, 224, 224). imagenet normalization is applied.
pub fn load_image224<P: AsRef<std::path::Path>>(p: P) -> candle_core::Result<Tensor> {
    load_image(p, 224)
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    path: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let device = Device::Cpu; // switch this to cuda if you have an nvidia gpu
    let image = load_image224(args.path)?.to_device(&device)?;

    let api = hf_hub::api::sync::Api::new()?;
    let api = api.model(MODEL_NAME.into());
    let model_file = api.get("model.safetensors")?;

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)? };
    let config = Config::vit_base_patch16_224();
    let model = Model::new(&config, NUM_LABELS, vb)?;
    println!("model built");

    let logits = model.forward(&image.unsqueeze(0)?)?;
    let prs = candle_nn::ops::softmax(&logits, D::Minus1)?
        .i(0)?
        .to_vec1::<f32>()?;
    println!("normal: {}", prs.get(0).copied().unwrap_or(f32::NAN));
    println!("nsfw: {}", prs.get(1).copied().unwrap_or(f32::NAN));

    Ok(())
}
