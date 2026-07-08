use candle_core::{D, DType, Device, IndexOp, Result, Tensor, Var};
use candle_nn::{
    AdamW, Dropout, Embedding, Linear, Module, ModuleT, Optimizer, ParamsAdamW, Sequential,
    VarBuilder, VarMap, attention::AttnMask, embedding, layer_norm, linear, linear_no_bias,
    loss::cross_entropy, ops::softmax, seq, sequential,
};
use core::f32;
use std::{any::Any, fs::read_to_string};
use tiktoken_rs::{r50k_base, r50k_base_singleton, tokenizer};

/* struct LayerNorm {
    eps: f64,
    scale: Tensor,
    shift: Tensor,
}
impl LayerNorm {
    pub fn new(emb_dim: usize, vs: VarBuilder) -> Result<Self> {
        let scale = vs.get_with_hints(emb_dim, "scale", candle_nn::Init::Const(1.0))?;
        let shift = vs.get_with_hints(emb_dim, "shift", candle_nn::Init::Const(0.0))?;
        Ok(Self {
            eps: 1e-5,
            scale,
            shift,
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
*/

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
impl ModuleT for MultiHeadAttention {
    fn forward_t(&self, xs: &Tensor, train_mode: bool) -> Result<Tensor> {
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
        let attn_weights = self.dropout.forward(&attn_weights, train_mode)?;
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
    norm1: candle_nn::LayerNorm,
    norm2: candle_nn::LayerNorm,
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
            norm1: layer_norm(emb_dim, 1e-5, vs.pp("norm1"))?,
            norm2: layer_norm(emb_dim, 1e-5, vs.pp("norm2"))?,
            drop_shortcut: Dropout::new(dropout),
            att: MultiHeadAttention::new(emb_dim, emb_dim, context_length, dropout, n_heads, vs)?,
        })
    }
}
impl ModuleT for TransfomerBlock {
    fn forward_t(&self, x: &Tensor, train_mode: bool) -> Result<Tensor> {
        let shortcut = &x;
        let mut x = self.norm1.forward(&x)?;
        x = self.att.forward_t(&x, train_mode)?;
        x = self.drop_shortcut.forward(&x, train_mode)?;
        x = x.broadcast_add(shortcut)?;

        let shortcut = &x;
        let mut x = self.norm2.forward(&x)?;
        x = self.ff.forward(&x)?;
        x = self.drop_shortcut.forward(&x, train_mode)?;
        x = x.broadcast_add(&shortcut)?;

        Ok(x)
    }
}

struct GPTModel {
    tok_emb: Embedding,
    pos_emb: Embedding,
    drop_emb: Dropout,
    tranformers: Vec<TransfomerBlock>,
    final_norm: candle_nn::LayerNorm,
    out_head: Linear,
}

impl ModuleT for GPTModel {
    fn forward_t(&self, xs: &Tensor, train_mode: bool) -> Result<Tensor> {
        let (_batch_size, seq_len) = xs.shape().dims2()?;
        let tok_embeds = self.tok_emb.forward(&xs)?;
        let pos_embeds = self
            .pos_emb
            .forward(&Tensor::arange(0, seq_len as u32, &xs.device())?)?;
        let mut x = tok_embeds.broadcast_add(&pos_embeds)?;
        x = self.drop_emb.forward(&x, train_mode)?;

        for block in &self.tranformers {
            x = block.forward_t(&x, train_mode)?;
        }
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
        let mut seq = vec![];
        for i in 0..n_layers {
            seq.push(TransfomerBlock::new(
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
            final_norm: layer_norm(emb_dim, 1e-5, vb.pp("final_norm"))?,
            out_head: linear_no_bias(emb_dim, vocab_size, vb.pp("out_head"))?,
        })
    }
}

fn get_batch(
    tokens: &[u32],
    batch_idx: usize,
    batch_size: usize,
    max_length: usize,
    stride: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let mut inputs = Vec::new();
    let mut targets = Vec::new();

    for b in 0..batch_size {
        // Calculate where this specific batch element starts based on stride
        let start_pos = (batch_idx * batch_size + b) * stride;

        // Extract the input chunk (length of max_length)
        let input_chunk = tokens[start_pos..start_pos + max_length].to_vec();

        // Extract the target chunk (shifted right by 1 token!)
        let target_chunk = tokens[start_pos + 1..start_pos + max_length + 1].to_vec();

        inputs.extend(input_chunk);
        targets.extend(target_chunk);
    }

    // Create the final [batch_size, max_length] tensors
    let x = Tensor::from_vec(inputs, (batch_size, max_length), device)?;
    let y = Tensor::from_vec(targets, (batch_size, max_length), device)?;

    Ok((x, y))
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

fn generate_text_simple(
    model: &GPTModel,
    mut idx: Tensor,
    max_new_tokens: usize,
    context_size: usize,
) -> Result<Tensor> {
    for _ in 0..max_new_tokens {
        let (_batch_size, seq_len) = idx.dims2()?;

        let idx_cond = if seq_len > context_size {
            idx.i((0.., (seq_len - context_size)..))?
        } else {
            idx.clone()
        };
        let logits = model.forward_t(&idx_cond, false)?;
        let (_, cond_len, _) = logits.dims3()?;

        let logits = logits.i((.., cond_len - 1, ..))?;
        let probas = softmax(&logits, D::Minus1)?;
        let idx_next = probas.argmax_keepdim(D::Minus1)?;
        idx = Tensor::cat(&[&idx, &idx_next], 1)?;
    }
    Ok(idx)
}

fn calc_loss_batch(
    input_batch: &Tensor,
    target_batch: &Tensor,
    model: &GPTModel,
) -> Result<Tensor> {
    let logits = model.forward_t(&input_batch, false)?;
    let loss = cross_entropy(&logits.flatten(0, 1)?, &target_batch.flatten_all()?)?;
    Ok(loss)
}
pub fn main() {
    let tokenizer = r50k_base_singleton();

    let text = read_to_string("the-verdict.txt").unwrap();
    let tokens = tokenizer.encode_with_special_tokens(&text);

    let vocab_size = 50257;
    let context_length = 256; //1024;
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
    let batch_size = 2;
    let stride = context_length;

    let train_ratio = 0.9;
    let split_idx = (train_ratio * tokens.len() as f32) as usize;
    let train_data = &tokens[0..split_idx];
    let val_data = &tokens[split_idx..];

    let num_train_batches = (train_data.len() - context_length) / (batch_size * stride);
    let mut total_loss = 0.0f32;
    let params = ParamsAdamW {
        lr: 0.0004,
        ..Default::default()
    };
    let mut opt = AdamW::new(varmap.all_vars(), params).unwrap();

    for epoch in 0..10 {
        let mut total_epoch_loss = 0.0f32;
        for i in 0..num_train_batches {
            let (x, y) =
                get_batch(&train_data, i, batch_size, context_length, stride, &device).unwrap();

            let logits = gpt.forward_t(&x, true).unwrap();
            let loss =
                cross_entropy(&logits.flatten(0, 1).unwrap(), &y.flatten_all().unwrap()).unwrap();
            opt.backward_step(&loss).unwrap();
            let loss_val = loss.to_scalar::<f32>().unwrap();
            total_epoch_loss += loss_val;
            if i % 2 == 0 {
                println!(
                    "Epoch {}, Batch {}/ {} - Loss {:.4}",
                    epoch + 1,
                    i,
                    num_train_batches,
                    loss_val
                );
            }
        }
        let avg_epoch_loss = total_epoch_loss / (num_train_batches as f32);
        println!(
            "====> Epoch {} Complete! Average Loss: {:.4}",
            epoch + 1,
            avg_epoch_loss
        );
        // Put this at the bottom of your epoch loop to watch it learn live!

        let sample_tokens = tokenizer.encode_with_special_tokens("Hello, I am").to_vec();
        let len = sample_tokens.len();
        let sample_input = Tensor::from_vec(sample_tokens, (1, len), &device).unwrap();
        let generated_tokens =
            generate_text_simple(&gpt, sample_input, 15, context_length).unwrap();

        // Flatten down to a 1D vector to decode back to a string
        let token_ids: Vec<u32> = generated_tokens.flatten_all().unwrap().to_vec1().unwrap();
        // let generated_text = tokenizer.decode(&token_ids).unwrap();

        println!("--- Epoch {} Sample Generation: ---", epoch + 1);
        // println!("{}\n-----------------------------------", generated_text);
    }

    // get_batch(&train_data, 0, 2, context_length, stride, &device);
}
