use candle_core::{D, DType, Device, IndexOp, Result, Tensor, Var};
use candle_nn::{
    Dropout, Embedding, Linear, Module, Sequential, VarBuilder, VarMap, attention::AttnMask,
    embedding, linear, linear_no_bias, ops::softmax, seq, sequential,
};
use core::f32;
use std::{any::Any, fs::read_to_string};
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
        let keys = keys.transpose(1, 2)?.contiguous()?;
        let queries = queries.transpose(1, 2)?.contiguous()?;
        let values = values.transpose(1, 2)?.contiguous()?;

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
    att: MultiHeadAttention,
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
            att: MultiHeadAttention::new(emb_dim, emb_dim, context_length, dropout, n_heads, vs)?,
        })
    }
}
impl Module for TransfomerBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let shortcut = &x;
        let mut x = self.norm1.forward(&x)?;
        x = self.att.forward(&x)?;
        x = self.drop_shortcut.forward(&x, true)?;
        x = x.broadcast_add(shortcut)?;

        let shortcut = &x;
        let mut x = self.norm2.forward(&x)?;
        x = self.ff.forward(&x)?;
        x = self.drop_shortcut.forward(&x, true)?;
        x = x.broadcast_add(&shortcut)?;

        Ok(x)
    }
}

struct GPTModel {
    tok_emb: Embedding,
    pos_emb: Embedding,
    drop_emb: Dropout,
    tranformers: Sequential,
    final_norm: LayerNorm,
    out_head: Linear,
}

impl Module for GPTModel {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len) = xs.shape().dims2()?;
        let tok_embeds = self.tok_emb.forward(&xs)?;
        let pos_embeds = self
            .pos_emb
            .forward(&Tensor::arange(0, seq_len as u32, &xs.device())?)?;
        let mut x = tok_embeds.broadcast_add(&pos_embeds)?;
        let x = self.drop_emb.forward(&x, true)?;
        let x = self.tranformers.forward(&x)?;
        let x = self.final_norm.forward(&x)?;
        let logits = self.out_head.forward(&x)?;
        Ok(logits)
    }
}

impl GPTModel {
    pub fn new(
        vocab_size: usize,
        context_length: usize,
        emb_dim: usize,
        n_heads: usize,
        n_layers: usize,
        drop_rate: f32,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut seq = seq();
        for i in 0..n_layers {
            seq = seq.add(TransfomerBlock::new(
                emb_dim,
                context_length,
                n_heads,
                drop_rate,
                vb.pp(format!("transformer {}", i)),
            )?);
        }
        Ok(Self {
            tok_emb: embedding(vocab_size, emb_dim, vb.pp("tok_emb"))?,
            pos_emb: embedding(context_length, emb_dim, vb.pp("pos_enb"))?,
            drop_emb: Dropout::new(drop_rate),
            tranformers: seq,
            final_norm: LayerNorm::new(emb_dim, vb.pp("final_norm"))?,
            out_head: linear_no_bias(emb_dim, vocab_size, vb.pp("out_head"))?,
        })
    }
}
fn print_total_parameters(varmap: &VarMap) {
    let mut total_params = 0;

    // Get access to the inner map of named variables
    let data = varmap.data().lock().unwrap();

    for (name, var) in data.iter() {
        // Retrieve the shape dimensions of the individual tensor
        let shape = var.shape();
        let param_count = shape.elem_count();
        total_params += param_count;

        println!(
            "{}: {} elements (shape: {:?}",
            name,
            param_count,
            shape.dims()
        );
    }

    println!("\n==========================================");
    println!("Total Trainable Parameters: {}", total_params);
    println!("==========================================");
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

    let gpt = GPTModel::new(
        vocab_size,
        context_length,
        emb_dim,
        n_heads,
        n_layers,
        drop_rate,
        vb,
    )
    .unwrap();
    print_total_parameters(&varmap);
}
