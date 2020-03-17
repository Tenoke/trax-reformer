train_config = """
import trax.layers
import trax.models
import trax.optimizers
import trax.supervised.inputs
import trax.supervised.trainer_lib

# Parameters that will vary between experiments:
# ==============================================================================
train.model = @trax.models.ReformerLM
# attn_type = @TimeBinCausalAttention
attn_type = [
    @TimeBinCausalAttention,
    @TimeBinCausalAttention,
    @LSHCausalAttention,
    @TimeBinCausalAttention,
]
share_qk = False  # LSHCausalAttention ignores this flag and always shares q & k
attn_kv = 128
n_layers = 12
dropout = 0.2

# MemoryEfficientCausalAttention: full attention
# (no hparams to vary between experiments)

# TimeBinCausalAttention: attend to nearby items
TimeBinCausalAttention.n_bins = 512

# LSHCausalAttention: locality-sensitive hashing (LSH) attention
LSHCausalAttention.n_bins = 256
LSHCausalAttention.n_buckets = 512  # Always 2 * n_bins
LSHCausalAttention.n_hashes = 2
LSHCausalAttention.drop_for_hash_rate = 0.0


# Parameters for MultifactorSchedule:
# ==============================================================================
# 0.03125 ~= 1024^-0.5 = d_model^-0.5
MultifactorSchedule.constant = 0.03125
MultifactorSchedule.factors = 'constant * linear_warmup * rsqrt_decay'
MultifactorSchedule.warmup_steps = 2000

# Parameters for Adam:
# ==============================================================================
Adam.weight_decay_rate=0.0
Adam.b1 = 0.9
Adam.b2 = 0.98
Adam.eps = 1e-9


# Parameters for MemoryEfficientCausalAttention:
# ==============================================================================
MemoryEfficientCausalAttention.dropout = 0.0
MemoryEfficientCausalAttention.loop_stride = 256
MemoryEfficientCausalAttention.share_qk = %share_qk

# Parameters for TimeBinCausalAttention:
# ==============================================================================
TimeBinCausalAttention.dropout = 0.2
# TimeBinCausalAttention.n_bins: see top
TimeBinCausalAttention.share_qk = %share_qk

# Parameters for LSHCausalAttention:
# ==============================================================================
LSHCausalAttention.allow_duplicate_attention = False
LSHCausalAttention.attend_across_buckets = True
LSHCausalAttention.rehash_each_round = True
# LSHCausalAttention.n_bins: see top
# LSHCausalAttention.n_buckets: see top
# LSHCausalAttention.n_hashes: see top
LSHCausalAttention.one_rng = False
LSHCausalAttention.hard_k = 0
LSHCausalAttention.dropout = 0.2
# LSHCausalAttention.drop_for_hash_rate: see top


# Parameters for ReformerLM:
# ==============================================================================
ReformerLM.attention_type = %attn_type
ReformerLM.d_attention_key = %attn_kv
ReformerLM.d_attention_value = %attn_kv
ReformerLM.d_model = 1024
ReformerLM.d_ff = 2048
ReformerLM.dropout = %dropout
ReformerLM.ff_activation = @trax.layers.Relu
ReformerLM.max_len = 65536
ReformerLM.mode = 'train'
ReformerLM.n_heads = 8
ReformerLM.n_layers = %n_layers
ReformerLM.vocab_size = 258  # Includes pad token and unused EOS token
ReformerLM.share_qk = %share_qk
ReformerLM.axial_pos_shape = (128, 512)
ReformerLM.d_axial_pos_embs= (256, 768)
"""
