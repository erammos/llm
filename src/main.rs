use candle_core::{D, DType, Device, Result, Tensor, Var};
use candle_nn::{LayerNorm, Linear, Module, VarBuilder};
use std::fs::read_to_string;
use tiktoken_rs::{r50k_base, r50k_base_singleton, tokenizer};

struct LayerNorm {
    eps: f64,
    scale: Tensor,
    shift: Tensor,
}
impl LayerNorm {
    pub fn new(emb_dim: usize, vs: VarBuilder) -> Result<Self> {
        Ok(Self {
            eps: 1e-5,
            scale: vs.get(emb_dim, "scale")?,
            shift: vs.get(emb_dim, "shift")?,
        })
    }
}
impl Module for LayerNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mean = x.mean_keepdim(D::Minus1)?;
        let var = x.var_keepdim(D::Minus1)?;
        let eps_tensor = Tensor::new(self.eps as f32, x.device())?;
        let denom = (var.broadcast_add(&eps_tensor)?).sqrt()?;
        let norm_x = x.broadcast_sub(&mean)?.broadcast_div(&denom)?;

        let scaled = norm_x.broadcast_mul(&self.scale)?;
        scaled.broadcast_add(&self.shift)
    }
}
struct FeedForward {
    l1: Linear,
    l2: Linear,
}
impl Module for FeedForward {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let l = self.l1.forward(xs)?;
        let o = l.gelu()?;
        Ok(self.l2.forward(&o)?)
    }
}

pub fn main() {
    let tokenizer = r50k_base_singleton();

    let text = read_to_string("the-verdict.txt").unwrap();
    let tokens = tokenizer.encode_with_special_tokens(&text);

    let l: LayerNorm;

    let vocab_size = 50257;
    let output_dim = 256;
    let batch_size = 8;
    let window_size = 4;
}
