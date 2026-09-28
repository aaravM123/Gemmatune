use candle_core::{DType, Device, Module, Result as CandleResult, Tensor, Var, D};
use candle_nn::{optim::Optimizer, AdamW, Embedding, ParamsAdamW, VarBuilder};
use candle_transformers::utils::repeat_kv;
use gemmatune_core::Device as RequestedDevice;
use gemmatune_lora::{TrainableAdapter, TrainableTensor};
use std::{path::Path, sync::Arc};
use crate::training::CausalBatch;

pub const GEMMA_3_1B_LAYERS: usize = 26;
const HIDDEN: usize = 1152;
const HEAD_DIM: usize = 256;
const HEADS: usize = 4;
const KV_HEADS: usize = 1;
const QUERY_WIDTH: usize = HEADS * HEAD_DIM;
const VOCABULARY: usize = 262_144;
const MAX_POSITIONS: usize = 32_768;
type Result<T> = CandleResult<T>;
struct RmsNorm {
    weight: Tensor,
    dimension: usize,
}
impl RmsNorm {
    fn load(vb: VarBuilder, dimension: usize) -> Result<Self> {
        Ok(Self {
            weight: vb.get(dimension, "weight")?.to_dtype(DType::F32)?,
            dimension,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let variance = (x.sqr()?.sum_keepdim(D::Minus1)? / self.dimension as f64)?;
        x.broadcast_div(&(variance + 1e-6)?.sqrt()?)?
            .broadcast_mul(&(&self.weight + 1.0)?)
    }
}

fn attention_projection_shape(module: &str) -> Result<(usize, usize)> {
    match module {
        "q_proj" => Ok((HIDDEN, QUERY_WIDTH)),
        "k_proj" | "v_proj" => Ok((HIDDEN, KV_HEADS * HEAD_DIM)),
        "o_proj" => Ok((QUERY_WIDTH, HIDDEN)),
        _ => candle_core::bail!("unsupported Gemma attention projection `{module}`"),
    }
}

fn validate_projection(adapter: &TrainableTensor) -> Result<()> {
    let (input, output) = attention_projection_shape(&adapter.spec.module)?;
    if adapter.spec.input_features != input || adapter.spec.output_features != output {
        candle_core::bail!(
            "Gemma 3 1B {} expects {} -> {}, got {} -> {}",
            adapter.spec.module,
            input,
            output,
            adapter.spec.input_features,
            adapter.spec.output_features,
        );
    }
    Ok(())
}
struct FrozenLinear {
    weight: Tensor,
}
impl FrozenLinear {
    fn load(vb: VarBuilder, input: usize, output: usize) -> Result<Self> {
        Ok(Self {
            weight: vb.get((output, input), "weight")?.to_dtype(DType::F32)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, tokens, input) = x.dims3()?;
        x.reshape((batch * tokens, input))?
            .matmul(&self.weight.t()?)?
            .reshape((batch, tokens, self.weight.dim(0)?))
    }
}
struct LoraLinear {
    base: FrozenLinear,
    a: Var,
    b: Var,
    scale: f64,
}
impl LoraLinear {
    fn load(vb: VarBuilder, adapter: &TrainableTensor, variables: &(Var, Var)) -> Result<Self> {
        let spec = &adapter.spec;
        let base = FrozenLinear::load(vb, spec.input_features, spec.output_features)?;
        Ok(Self {
            base,
            a: variables.0.clone(),
            b: variables.1.clone(),
            scale: adapter.scaling() as f64,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, tokens, input) = x.dims3()?;
        let update = (x
            .reshape((batch * tokens, input))?
            .matmul(&self.a.t()?)?
            .matmul(&self.b.t()?)?
            .reshape((batch, tokens, self.b.dim(0)?))?
            * self.scale)?;
        self.base.forward(x)?.add(&update)
    }

}
struct Rotary {
    sin: Tensor,
    cos: Tensor,
}
impl Rotary {
    fn new(theta: f64, device: &Device) -> Result<Self> {
        let frequencies = (0..HEAD_DIM)
            .step_by(2)
            .map(|index| 1f32 / theta.powf(index as f64 / HEAD_DIM as f64) as f32)
            .collect::<Vec<_>>();
        let positions = Tensor::arange(0u32, MAX_POSITIONS as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((MAX_POSITIONS, 1))?;
        let values =
            positions.matmul(&Tensor::from_vec(frequencies, (1, HEAD_DIM / 2), device)?)?;
        Ok(Self {
            sin: values.sin()?,
            cos: values.cos()?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let length = q.dim(2)?;
        let cos = self.cos.narrow(0, 0, length)?;
        let sin = self.sin.narrow(0, 0, length)?;
        Ok((
            candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?,
            candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?,
        ))
    }
}
struct Attention {
    q: LoraLinear,
    k: LoraLinear,
    v: LoraLinear,
    o: LoraLinear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    rotary: Arc<Rotary>,
}
impl Attention {
    fn load(
        vb: VarBuilder,
        adapter: &TrainableAdapter,
        variables: &[(String, Var, Var)],
        rotary: Arc<Rotary>,
    ) -> Result<Self> {
        let q = adapter
            .tensor("q_proj")
            .expect("validated adapter has q_proj");
        let k = adapter
            .tensor("k_proj")
            .expect("validated adapter has k_proj");
        let v = adapter
            .tensor("v_proj")
            .expect("validated adapter has v_proj");
        let o = adapter
            .tensor("o_proj")
            .expect("validated adapter has o_proj");
        for projection in [q, k, v, o] {
            validate_projection(projection)?;
        }
        let variables_for = |module| {
            variables
                .iter()
                .find(|(name, _, _)| name == module)
                .map(|(_, a, b)| (a.clone(), b.clone()))
                .expect("validated adapter has LoRA variables")
        };
        Ok(Self {
            q: LoraLinear::load(vb.pp("q_proj"), q, &variables_for("q_proj"))?,
            k: LoraLinear::load(vb.pp("k_proj"), k, &variables_for("k_proj"))?,
            v: LoraLinear::load(vb.pp("v_proj"), v, &variables_for("v_proj"))?,
            o: LoraLinear::load(vb.pp("o_proj"), o, &variables_for("o_proj"))?,
            q_norm: RmsNorm::load(vb.pp("q_norm"), HEAD_DIM)?,
            k_norm: RmsNorm::load(vb.pp("k_norm"), HEAD_DIM)?,
            rotary,
        })
    }

    fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (batch, tokens, _) = x.dims3()?;
        let q = self
            .q
            .forward(x)?
            .reshape((batch, tokens, HEADS, HEAD_DIM))?
            .transpose(1, 2)?;
        let k = self
            .k
            .forward(x)?
            .reshape((batch, tokens, 1, HEAD_DIM))?
            .transpose(1, 2)?;
        let v = self
            .v
            .forward(x)?
            .reshape((batch, tokens, 1, HEAD_DIM))?
            .transpose(1, 2)?;
        let (q, k) = self
            .rotary
            .apply(&self.q_norm.forward(&q)?, &self.k_norm.forward(&k)?)?;
        let scores =
            (q.matmul(&repeat_kv(k, HEADS)?.transpose(2, 3)?)? / (HEAD_DIM as f64).sqrt())?;
        let weights = candle_nn::ops::softmax_last_dim(
            &((scores / 50.0)?.tanh()? * 50.0)?.broadcast_add(mask)?,
        )?;
        let output = weights
            .matmul(&repeat_kv(v, HEADS)?)?
            .transpose(1, 2)?
            .reshape((batch, tokens, QUERY_WIDTH))?;
        self.o.forward(&output)
    }

}
struct Mlp {
    gate: FrozenLinear,
    up: FrozenLinear,
    down: FrozenLinear,
}
impl Mlp {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate: FrozenLinear::load(vb.pp("gate_proj"), HIDDEN, 6912)?,
            up: FrozenLinear::load(vb.pp("up_proj"), HIDDEN, 6912)?,
            down: FrozenLinear::load(vb.pp("down_proj"), 6912, HIDDEN)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hidden = (self.gate.forward(x)?.gelu_erf()? * self.up.forward(x)?)?;
        self.down.forward(&hidden)
    }
}
struct Layer {
    attention: Attention,
    mlp: Mlp,
    input_norm: RmsNorm,
    pre_ff_norm: RmsNorm,
    post_ff_norm: RmsNorm,
    post_attn_norm: RmsNorm,
    local: bool,
}
impl Layer {
    fn load(
        vb: VarBuilder,
        adapter: &TrainableAdapter,
        variables: &[(String, Var, Var)],
        rotary: Arc<Rotary>,
        local: bool,
    ) -> Result<Self> {
        Ok(Self {
            attention: Attention::load(vb.pp("self_attn"), adapter, variables, rotary)?,
            mlp: Mlp::load(vb.pp("mlp"))?,
            input_norm: RmsNorm::load(vb.pp("input_layernorm"), HIDDEN)?,
            pre_ff_norm: RmsNorm::load(vb.pp("pre_feedforward_layernorm"), HIDDEN)?,
            post_ff_norm: RmsNorm::load(vb.pp("post_feedforward_layernorm"), HIDDEN)?,
            post_attn_norm: RmsNorm::load(vb.pp("post_attention_layernorm"), HIDDEN)?,
            local,
        })
    }

    fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let attention = self.attention.forward(&self.input_norm.forward(x)?, mask)?;
        let x = (x + self.post_attn_norm.forward(&attention)?)?;
        let feed_forward = self.mlp.forward(&self.pre_ff_norm.forward(&x)?)?;
        &x + self.post_ff_norm.forward(&feed_forward)?
    }
}
pub struct TrainableGemmaDecoder {
    adapter: TrainableAdapter,
    variables: Vec<(String, Var, Var)>,
    embeddings: Embedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    device: Device,
}
impl TrainableGemmaDecoder {
    pub fn load_1b_it(
        root: impl AsRef<Path>,
        requested: RequestedDevice,
        adapter: &TrainableAdapter,
    ) -> std::result::Result<Self, String> {
        if adapter.base_checkpoint != "gemma-3-1b-it" {
            return Err("trainable decoder requires a gemma-3-1b-it adapter".into());
        }
        for module in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            if adapter.tensor(module).is_none() {
                return Err(format!("trainable decoder requires {module} LoRA weights"));
            }
        }
        let (device, _) = super::select_backend(requested)?;
        let paths = super::safetensor_paths(root.as_ref())?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&paths, DType::BF16, &device)
                .map_err(|error| format!("cannot map Gemma safetensors: {error}"))?
        }
        .pp("model");
        let embedding_weights = vb
            .pp("embed_tokens")
            .get((VOCABULARY, HIDDEN), "weight")
            .and_then(|weight| weight.to_dtype(DType::F32))
            .map_err(|error| format!("cannot load Gemma token embeddings: {error}"))?;
        let embeddings = Embedding::new(embedding_weights, HIDDEN);
        let variables = adapter
            .tensors
            .iter()
            .map(|tensor| {
                Ok((
                    tensor.spec.module.clone(),
                    Var::from_vec(tensor.a.clone(), tensor.spec.shape_a, &device)?,
                    Var::from_vec(tensor.b.clone(), tensor.spec.shape_b, &device)?,
                ))
            })
            .collect::<Result<Vec<_>>>()
            .map_err(|error| format!("cannot initialize LoRA variables: {error}"))?;
        let global =
            Arc::new(Rotary::new(1_000_000.0, &device).map_err(|error| error.to_string())?);
        let local = Arc::new(Rotary::new(10_000.0, &device).map_err(|error| error.to_string())?);
        let mut layers = Vec::with_capacity(GEMMA_3_1B_LAYERS);
        for index in 0..GEMMA_3_1B_LAYERS {
            let uses_local = (index + 1) % 6 != 0;
            layers.push(
                Layer::load(
                    vb.pp("layers").pp(index),
                    adapter,
                    &variables,
                    if uses_local {
                        local.clone()
                    } else {
                        global.clone()
                    },
                    uses_local,
                )
                .map_err(|error| format!("cannot load Gemma layer {index}: {error}"))?,
            );
        }
        let norm = RmsNorm::load(vb.pp("norm"), HIDDEN)
            .map_err(|error| format!("cannot load Gemma output norm: {error}"))?;
        Ok(Self {
            adapter: adapter.clone(),
            variables,
            embeddings,
            layers,
            norm,
            device,
        })
    }

    pub fn adapter_variables(&self) -> Vec<Var> {
        self.variables
            .iter()
            .flat_map(|(_, a, b)| [a.clone(), b.clone()])
            .collect()
    }

    pub fn trained_adapter(&self) -> std::result::Result<TrainableAdapter, String> {
        let mut adapter = self.adapter.clone();
        for tensor in &mut adapter.tensors {
            let (_, a, b) = self
                .variables
                .iter()
                .find(|(module, _, _)| module == &tensor.spec.module)
                .expect("every adapter tensor has Candle variables");
            tensor.a = a
                .as_tensor()
                .to_vec1()
                .map_err(|error| format!("cannot read updated LoRA A: {error}"))?;
            tensor.b = b
                .as_tensor()
                .to_vec1()
                .map_err(|error| format!("cannot read updated LoRA B: {error}"))?;
        }
        Ok(adapter)
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (batch, tokens) = input_ids.dims2()?;
        if tokens == 0 || tokens > MAX_POSITIONS {
            candle_core::bail!("sequence length must be in 1..={MAX_POSITIONS}");
        }
        let mut x = (self.embeddings.forward(input_ids)? * (HIDDEN as f64).sqrt())?;
        let global_mask = causal_mask(batch, tokens, None, &self.device)?;
        let local_mask = causal_mask(batch, tokens, Some(512), &self.device)?;
        for layer in &self.layers {
            x = layer.forward(
                &x,
                if layer.local {
                    &local_mask
                } else {
                    &global_mask
                },
            )?;
        }
        let normalized = self.norm.forward(&x)?;
        let (batch, tokens, hidden) = normalized.dims3()?;
        let logits = normalized
            .reshape((batch * tokens, hidden))?
            .matmul(&self.embeddings.embeddings().t()?)?
            .reshape((batch, tokens, VOCABULARY))?;
        (logits / 30.0)?.tanh()?.affine(30.0, 0.0)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FineTuneResult {
    pub adapter: TrainableAdapter,
    pub steps: usize,
    pub mean_loss: f32,
}

pub fn fine_tune(
    root: impl AsRef<Path>,
    requested: RequestedDevice,
    adapter: TrainableAdapter,
    sequences: &[Vec<u32>],
) -> std::result::Result<FineTuneResult, String> {
    if sequences.is_empty() {
        return Err("the training split contains no conversations".into());
    }
    let decoder = TrainableGemmaDecoder::load_1b_it(root, requested, &adapter)?;
    let mut optimizer = AdamW::new(
        decoder.adapter_variables(),
        ParamsAdamW {
            lr: adapter.config.learning_rate as f64,
            ..ParamsAdamW::default()
        },
    )
    .map_err(|error| format!("cannot initialize AdamW: {error}"))?;
    let mut total_loss = 0.0;
    let mut steps = 0;
    for _ in 0..adapter.config.epochs {
        for sequence in sequences {
            let batch = CausalBatch::from_tokens(sequence)?;
            let token_count = batch.token_count();
            let inputs = Tensor::from_vec(
                batch.inputs,
                (1, token_count),
                &decoder.device,
            )
            .map_err(|error| format!("cannot create training inputs: {error}"))?;
            let targets = Tensor::from_vec(batch.targets, token_count, &decoder.device)
                .map_err(|error| format!("cannot create training targets: {error}"))?;
            let logits = decoder
                .forward(&inputs)
                .and_then(|logits| logits.reshape((token_count, VOCABULARY)))
                .map_err(|error| format!("Gemma training forward pass failed: {error}"))?;
            let loss = candle_nn::loss::cross_entropy(&logits, &targets)
                .map_err(|error| format!("cannot calculate full-token cross entropy: {error}"))?;
            total_loss += loss
                .to_scalar::<f32>()
                .map_err(|error| format!("cannot read training loss: {error}"))?;
            optimizer
                .backward_step(&loss)
                .map_err(|error| format!("Candle AdamW update failed: {error}"))?;
            steps += 1;
        }
    }
    Ok(FineTuneResult {
        adapter: decoder.trained_adapter()?,
        steps,
        mean_loss: total_loss / steps as f32,
    })
}
fn causal_mask(
    batch: usize,
    tokens: usize,
    window: Option<usize>,
    device: &Device,
) -> Result<Tensor> {
    let values = (0..tokens)
        .flat_map(|row| {
            (0..tokens).map(move |column| {
                if column > row || window.is_some_and(|width| column + width < row) {
                    f32::NEG_INFINITY
                } else {
                    0.0
                }
            })
        })
        .collect::<Vec<_>>();
    Tensor::from_vec(values, (tokens, tokens), device)?.expand((batch, 1, tokens, tokens))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn lora_variables_receive_a_real_candle_gradient() -> Result<()> {
        let device = Device::Cpu;
        let projection = LoraLinear {
            base: FrozenLinear {
                weight: Tensor::zeros((2, 2), DType::F32, &device)?,
            },
            a: Var::from_vec(vec![1f32, 2.], (1, 2), &device)?,
            b: Var::from_vec(vec![3f32, 4.], (2, 1), &device)?,
            scale: 1.0,
        };
        let input = Tensor::from_vec(vec![1f32, 1.], (1, 1, 2), &device)?;
        let gradients = projection.forward(&input)?.sum_all()?.backward()?;
        assert!(gradients.get(&projection.a).is_some());
        assert!(gradients.get(&projection.b).is_some());
        Ok(())
    }

    #[test]
    fn candle_adamw_updates_the_lora_variables() -> Result<()> {
        let device = Device::Cpu;
        let projection = LoraLinear {
            base: FrozenLinear {
                weight: Tensor::zeros((2, 2), DType::F32, &device)?,
            },
            a: Var::from_vec(vec![1f32, 2.], (1, 2), &device)?,
            b: Var::from_vec(vec![3f32, 4.], (2, 1), &device)?,
            scale: 1.0,
        };
        let before = projection.b.as_tensor().to_vec2::<f32>()?;
        let input = Tensor::from_vec(vec![1f32, 1.], (1, 1, 2), &device)?;
        let loss = projection.forward(&input)?.sum_all()?;
        let mut optimizer = AdamW::new_lr(
            vec![projection.a.clone(), projection.b.clone()],
            0.001,
        )?;
        optimizer.backward_step(&loss)?;
        assert_ne!(before, projection.b.as_tensor().to_vec2::<f32>()?);
        Ok(())
    }

    #[test]
    fn creates_a_causal_local_mask() -> Result<()> {
        let mask = causal_mask(1, 3, Some(1), &Device::Cpu)?
            .squeeze(0)?
            .squeeze(0)?;
        assert_eq!(mask.to_vec2::<f32>()?[0][1], f32::NEG_INFINITY);
        assert_eq!(mask.to_vec2::<f32>()?[2][0], f32::NEG_INFINITY);
        Ok(())
    }

    #[test]
    fn token_ids_use_embedding_lookup_not_matrix_multiplication() -> Result<()> {
        let device = Device::Cpu;
        let embedding = candle_nn::embedding(4, 3, VarBuilder::zeros(DType::F32, &device))?;
        let token_ids = Tensor::from_vec(vec![0u32, 3, 1, 2], (1, 4), &device)?;
        assert_eq!(embedding.forward(&token_ids)?.dims3()?, (1, 4, 3));
        Ok(())
    }

    #[test]
    fn tied_lm_head_flattens_tokens_for_candle_matmul() -> Result<()> {
        let device = Device::Cpu;
        let embedding = candle_nn::embedding(5, 3, VarBuilder::zeros(DType::F32, &device))?;
        let hidden = Tensor::zeros((1, 4, 3), DType::F32, &device)?;
        let (batch, tokens, width) = hidden.dims3()?;
        let logits = hidden
            .reshape((batch * tokens, width))?
            .matmul(&embedding.embeddings().t()?)?
            .reshape((batch, tokens, 5))?;
        assert_eq!(logits.dims3()?, (1, 4, 5));
        Ok(())
    }

    #[test]
    fn uses_gemma_1b_head_dim_for_q_and_k_norms() -> Result<()> {
        assert_eq!(HEAD_DIM, 256);
        assert_eq!(attention_projection_shape("q_proj")?, (1152, 1024));
        assert_eq!(attention_projection_shape("k_proj")?, (1152, 256));
        assert_eq!(attention_projection_shape("v_proj")?, (1152, 256));
        assert_eq!(attention_projection_shape("o_proj")?, (1024, 1152));
        Ok(())
    }

    #[test]
    fn merges_query_heads_before_the_output_projection() -> Result<()> {
        assert_eq!(QUERY_WIDTH, 1024);
        assert_eq!(attention_projection_shape("o_proj")?, (QUERY_WIDTH, HIDDEN));
        Ok(())
    }
}
