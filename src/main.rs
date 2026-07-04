use candle_core::{D, DType, Device, IndexOp, Result, Tensor, Var};
use candle_nn::{
    Dropout, Embedding, Linear, Module, Sequential, VarBuilder, VarMap, attention::AttnMask,
    embedding, linear, linear_no_bias, ops::softmax,
};
use core::f32;
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
impl FeedForward {
    pub fn new(emb_dim: usize, vs: VarBuilder) -> Result<Self> {
        Ok(Self {
            l1: linear(emb_dim, 4 * emb_dim, vs.pp("linear1"))?,
            l2: linear(4 * emb_dim, emb_dim, vs.pp("linear2"))?,
        })
    }
}
impl Module for FeedForward {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let l = self.l1.forward(xs)?;
        let o = l.gelu()?;
        Ok(self.l2.forward(&o)?)
    }
}

struct MultiHeadAttention {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    dropout: Dropout,
    mask: Tensor,
    out_proj: Linear,
    num_heads: usize,
    d_in: usize,
    d_out: usize,
    context_length: usize,
}

impl MultiHeadAttention {
    pub fn new(
        d_in: usize,
        d_out: usize,
        context_length: usize,
        dropout: f32,
        num_heads: usize,
        vs: VarBuilder,
    ) -> Result<MultiHeadAttention> {
        let wq = linear_no_bias(d_in, d_out, vs.pp("wq"))?;
        let wk = linear_no_bias(d_in, d_out, vs.pp("wk"))?;
        let wv = linear_no_bias(d_in, d_out, vs.pp("wv"))?;
        let out_proj = linear(d_out, d_out, vs.pp("out_proj"))?;
        let dropout = Dropout::new(dropout);
        let mask = Tensor::tril2(context_length, DType::U8, vs.device())?;
        Ok(Self {
            wq,
            wk,
            wv,
            dropout,
            out_proj,
            num_heads,
            d_in,
            d_out,
            context_length,
            mask,
        })
    }
}
impl Module for MultiHeadAttention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let head_dims = self.d_out / self.num_heads;
        let (b, num_tokens, d_in) = xs.dims3()?;
        let queries = self.wq.forward(xs)?;
        let keys = self.wk.forward(xs)?;
        let values = self.wv.forward(xs)?;

        let keys = keys.reshape((b, num_tokens, self.num_heads, head_dims))?;
        let queries = queries.reshape((b, num_tokens, self.num_heads, head_dims))?;
        let values = values.reshape((b, num_tokens, self.num_heads, head_dims))?;
        let keys = keys.transpose(1, 2)?;
        let queries = queries.transpose(1, 2)?;
        let values = values.transpose(1, 2)?;

        let atten_scores = queries.matmul(&keys.transpose(2, 3)?)?;
        let mask_bool = self.mask.i((..num_tokens, ..num_tokens))?.broadcast_as((
            b,
            self.num_heads,
            num_tokens,
            num_tokens,
        ))?;
        let inf_tensor = Tensor::new(f32::NEG_INFINITY, xs.device())?.broadcast_as((
            b,
            self.num_heads,
            num_tokens,
            num_tokens,
        ))?;

        let atten_scores = mask_bool.where_cond(&atten_scores, &inf_tensor)?;

        let attn_weights = softmax(
            &(atten_scores / (keys.dim(D::Minus1)? as f64).sqrt())?,
            D::Minus1,
        )?;
        let attn_weights = self.dropout.forward(&attn_weights, true)?;
        let context_vec = &attn_weights.matmul(&values)?.transpose(1, 2)?;

        let context_vec = context_vec
            .contiguous()?
            .reshape((b, num_tokens, self.d_out))?;
        let context_vec = self.out_proj.forward(&context_vec)?;
        Ok(context_vec)
    }
}

struct TransfomerBlock {
    ff: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
    drop_shortcut: Dropout,
}
impl TransfomerBlock {
    pub fn new(
        emb_dim: usize,
        context_length: usize,
        n_heads: usize,
        dropout: f32,
        vs: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            ff: FeedForward::new(emb_dim, vs.pp("ff"))?,
            norm1: LayerNorm::new(emb_dim, vs.pp("norm1"))?,
            norm2: LayerNorm::new(emb_dim, vs.pp("norm2"))?,
            drop_shortcut: Dropout::new(dropout),
        })
    }
}

struct GPTModel {
    tok_emb: Embedding,
    pos_emb: Embedding,
    drop_emb: Dropout,
    tranformers: Sequential,
}

pub fn main() {
    let tokenizer = r50k_base_singleton();

    let text = read_to_string("the-verdict.txt").unwrap();
    let tokens = tokenizer.encode_with_special_tokens(&text);

    let l: LayerNorm;

    let vocab_size = 50257;
    let context_length = 1024;
    let emb_dim = 768;
    let n_heads = 12;
    let n_layers = 12;
    let drop_rate = 0.1;

    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
}
