// ============================================================================
// Native In-Browser WebGPU Engine (`MoS INT4+ASVD` + `SCache O(1)` + `ScarStore`)
// Zero Python Backend, Zero ONNX / O(T) KV Facades, 100% Custom WGSL 1.0 Shaders
// ============================================================================

import { QwenBPETokenizer, selectChatSystemPrompt } from "./qwen_bpe_tokenizer.js";
import { BrowserScarStoreHTTP206, findSubsequenceLastEnd } from "./scarstore_http206_browser.js";

const MAX_ROUTING_USER_TOKENS = 192;
const PREFILL_BATCH_TOKENS = 128;

function packKernelParams(ints = {}, floats = {}) {
  const buf = new ArrayBuffer(64);
  const u32 = new Uint32Array(buf);
  const f32 = new Float32Array(buf);
  u32[0] = ints.p0 >>> 0;
  u32[1] = ints.p1 >>> 0;
  u32[2] = ints.p2 >>> 0;
  u32[3] = ints.p3 >>> 0;
  u32[4] = ints.p4 >>> 0;
  u32[5] = ints.p5 >>> 0;
  u32[6] = ints.p6 >>> 0;
  u32[7] = ints.p7 >>> 0;
  f32[8] = floats.f0 ?? 0.0;
  f32[9] = floats.f1 ?? 0.0;
  f32[10] = floats.f2 ?? 0.0;
  f32[11] = floats.f3 ?? 0.0;
  return buf;
}

export class BrowserMoSWebGPUEngine {
  constructor({
    device,
    adapterLabel,
    manifest,
    wgslSource,
    tokenizer,
    scarStore,
    shardBuffers,
  }) {
    this.device = device;
    this.adapterLabel = adapterLabel;
    this.manifest = manifest;
    this.tokenizer = tokenizer;
    this.scarStore = scarStore;
    this.shardBuffers = shardBuffers;

    const arch = manifest.architecture;
    const scDef = manifest.scache_defaults;

    this.hiddenDim = arch.hidden_dim;
    this.numLayers = arch.num_layers;
    this.numQHeads = arch.num_q_heads;
    this.numKvHeads = arch.num_kv_heads;
    this.headDim = arch.head_dim;
    this.interDim = arch.intermediate_size;
    this.vocabSize = arch.vocab_size;
    this.rmsNormEps = arch.rms_norm_eps;
    this.mosRank = arch.mos_rank;
    this.eosTokenIds = new Set(arch.eos_token_ids || [151645, 151643]);
    this.userTurnPrefixIds = arch.user_turn_prefix_ids || [151644, 872, 198];

    this.windowSize = scDef.window_size;
    this.numSink = Math.min(scDef.num_sink_tokens, Math.floor(this.windowSize / 2));
    this.slideLen = Math.max(1, this.windowSize - this.numSink);
    this.numSlots = scDef.num_slots;
    this.deltaBlend = scDef.delta_blend;

    this.totalWeightGpuBytes = shardBuffers.reduce((acc, b) => acc + b.size, 0);

    this.shaderModule = device.createShaderModule({
      label: "mos_scache_kernels",
      code: wgslSource,
    });

    const entryPoints = [
      "embed_int4_lookup",
      "rmsnorm_f32",
      "fp16_gemv_bias",
      "mos_fused_int4_bscar_gemv",
      "qwen_rope_inplace",
      "scache_o1_update_widrow_hoff",
      "scache_o1_gqa_attention",
      "swiglu_mul_f32",
      "vec_add_inplace_f32",
      "vec_copy_f32",
      "head_expanded_scar_inject",
      "int4_vocab_logits_gemv",
      "logits_top2_reduce_stage1",
      "logits_top2_reduce_stage2",
    ];
    this.pipelines = {};
    for (const ep of entryPoints) {
      this.pipelines[ep] = device.createComputePipeline({
        label: `pipe_${ep}`,
        layout: "auto",
        compute: { module: this.shaderModule, entryPoint: ep },
      });
    }

    this._allocateFixedBuffers(arch);
    this._prebuildDispatchPlan();
    this.resetSCache();
  }

  static async create(baseUrl = ".", onProgress = null) {
    if (typeof navigator === "undefined" || !navigator.gpu) {
      throw new Error(
        "WebGPU não está disponível neste navegador. Use Chrome 113+, Edge 113+ ou Safari 18+ com WebGPU habilitado.",
      );
    }

    const report = (phase, pct, detail = "") => {
      if (onProgress) onProgress({ phase, pct, detail });
    };

    report("adapter", 2, "Solicitando adaptador físico WebGPU (high-performance)...");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) {
      throw new Error("Nenhum adaptador de hardware WebGPU foi encontrado pelo navegador.");
    }

    let info = adapter.info || {};
    if (!info.vendor && typeof adapter.requestAdapterInfo === "function") {
      try {
        info = await adapter.requestAdapterInfo();
      } catch (_) {
        // Ignore if blocked by privacy settings
      }
    }
    const adapterLabel = [info.description || info.device, info.vendor, info.architecture]
      .filter(Boolean)
      .join(" • ") || "WebGPU Hardware Adapter";

    // Universal WebGPU 1.0 default limits (every shard is strictly < 80 MiB <= 128 MiB!)
    const device = await adapter.requestDevice();

    const cleanBase = baseUrl.replace(/\/$/, "");
    report("manifest", 5, "Carregando manifesto WebGPU + kernels WGSL 1.0...");
    const [manifestResp, wgslResp] = await Promise.all([
      fetch(`${cleanBase}/webgpu_manifest.json`),
      fetch(`${cleanBase}/mos_scache_kernels.wgsl`),
    ]);
    if (!manifestResp.ok) throw new Error(`HTTP ${manifestResp.status} em webgpu_manifest.json`);
    if (!wgslResp.ok) throw new Error(`HTTP ${wgslResp.status} em mos_scache_kernels.wgsl`);

    const manifest = await manifestResp.json();
    const wgslSource = await wgslResp.text();

    // Check WGSL compilation messages upfront
    const testMod = device.createShaderModule({ code: wgslSource });
    if (typeof testMod.getCompilationInfo === "function") {
      const comp = await testMod.getCompilationInfo();
      const errs = (comp.messages || []).filter((m) => m.type === "error");
      if (errs.length > 0) {
        throw new Error(
          `Erro de compilação WGSL: ${errs.map((e) => `L${e.lineNum}: ${e.message}`).join(" | ")}`,
        );
      }
    }

    report("tokenizer", 10, "Inicializando tokenizador BPE Qwen2.5 + roteador ScarStore (42 KB)...");
    const [tokenizer, scarStore] = await Promise.all([
      QwenBPETokenizer.fromUrl(`${cleanBase}/${manifest.tokenizer_file}`),
      manifest.scarstore_router_file
        ? BrowserScarStoreHTTP206.load(`${cleanBase}/${manifest.scarstore_router_file}`)
        : Promise.resolve(null),
    ]);

    const shards = manifest.shards || [];
    const totalBytes = manifest.total_weights_bytes || 1;
    let loadedBytes = 0;
    const shardBuffers = [];

    for (let i = 0; i < shards.length; i++) {
      const s = shards[i];
      report(
        "shards",
        Math.round(12 + (loadedBytes / totalBytes) * 82),
        `Carregando fatia MoS ${i + 1}/${shards.length}: ${s.filename} (${s.mib.toFixed(1)} MiB)...`,
      );
      const sResp = await fetch(`${cleanBase}/${s.filename}`);
      if (!sResp.ok) {
        throw new Error(`Falha ao baixar ${s.filename}: HTTP ${sResp.status}`);
      }
      const arrBuf = await sResp.arrayBuffer();
      const gpuBuf = device.createBuffer({
        label: s.filename,
        size: arrBuf.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(gpuBuf, 0, arrBuf, 0, arrBuf.byteLength);
      shardBuffers.push(gpuBuf);
      loadedBytes += arrBuf.byteLength;
    }

    report("compile", 96, "Construindo pipelines WGSL e buffers fixos SCache O(1)...");
    const engine = new BrowserMoSWebGPUEngine({
      device,
      adapterLabel,
      manifest,
      wgslSource,
      tokenizer,
      scarStore,
      shardBuffers,
    });
    report("ready", 100, "Motor WebGPU MoS + SCache O(1) pronto!");
    return engine;
  }

  get scacheGpuBytes() {
    return this.scacheKvBuf.size + this.scacheWScarBuf.size + this.scacheSalienceBuf.size;
  }

  _allocateFixedBuffers(arch) {
    const U = GPUBufferUsage;
    const totalKvHeads = this.numLayers * this.numKvHeads;
    this.countsU32Total = totalKvHeads * 4;
    const capPerHead = this.numSink + this.numSlots + this.slideLen;
    const vecU32 = this.headDim >> 1;
    const kvU32Words = this.countsU32Total + totalKvHeads * 2 * capPerHead * vecU32;

    // 100% Constant O(1) SCache GPU Buffers — NEVER resized across infinite context!
    this.scacheKvBuf = this.device.createBuffer({
      label: "scache_kv_o1",
      size: kvU32Words * 4,
      usage: U.STORAGE | U.COPY_DST | U.COPY_SRC,
    });
    this.scacheWScarBuf = this.device.createBuffer({
      label: "scache_w_scar_o1",
      size: totalKvHeads * this.headDim * this.headDim * 4,
      usage: U.STORAGE | U.COPY_DST | U.COPY_SRC,
    });
    this.scacheSalienceBuf = this.device.createBuffer({
      label: "scache_salience_o1",
      size: Math.max(16, totalKvHeads * Math.max(1, this.numSlots) * 4),
      usage: U.STORAGE | U.COPY_DST | U.COPY_SRC,
    });
    this._zeroKvBytes = new Uint8Array(this.scacheKvBuf.size);
    this._zeroWScarBytes = new Uint8Array(this.scacheWScarBuf.size);
    this._zeroSalBytes = new Uint8Array(this.scacheSalienceBuf.size);

    const qDim = this.numQHeads * this.headDim;
    const kvDim = this.numKvHeads * this.headDim;

    this.stepStateBuf = this.device.createBuffer({
      label: "step_state_u32",
      size: 16,
      usage: U.STORAGE | U.COPY_DST,
    });
    this.prefillStepsBuf = this.device.createBuffer({
      label: "prefill_steps_u32",
      size: PREFILL_BATCH_TOKENS * 16,
      usage: U.COPY_SRC | U.COPY_DST,
    });
    this.prefillXInReadBuf = this.device.createBuffer({
      label: "prefill_xin_readback",
      size: MAX_ROUTING_USER_TOKENS * this.hiddenDim * 4,
      usage: U.COPY_DST | U.MAP_READ,
    });

    const invFreqF32 = new Float32Array(arch.rope_inv_freq);
    this.invFreqBuf = this.device.createBuffer({
      label: "rope_inv_freq",
      size: Math.max(16, invFreqF32.byteLength),
      usage: U.STORAGE | U.COPY_DST,
    });
    this.device.queue.writeBuffer(this.invFreqBuf, 0, invFreqF32);

    this.hCurBuf = this.device.createBuffer({ size: this.hiddenDim * 4, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });
    this.normBuf = this.device.createBuffer({ size: this.hiddenDim * 4, usage: U.STORAGE | U.COPY_SRC });
    this.qBuf = this.device.createBuffer({ size: qDim * 4, usage: U.STORAGE | U.COPY_SRC });
    this.kBuf = this.device.createBuffer({ size: kvDim * 4, usage: U.STORAGE });
    this.vBuf = this.device.createBuffer({ size: kvDim * 4, usage: U.STORAGE });
    this.qUnropedLastBuf = this.device.createBuffer({ size: qDim * 4, usage: U.STORAGE | U.COPY_SRC });
    this.attnOutBuf = this.device.createBuffer({ size: qDim * 4, usage: U.STORAGE });
    this.projOutBuf = this.device.createBuffer({ size: this.hiddenDim * 4, usage: U.STORAGE });
    this.gateBuf = this.device.createBuffer({ size: this.interDim * 4, usage: U.STORAGE });
    this.upBuf = this.device.createBuffer({ size: this.interDim * 4, usage: U.STORAGE });
    this.asvdUBuf = this.device.createBuffer({ size: Math.max(64, this.mosRank) * 4, usage: U.STORAGE });
    this.logitsBuf = this.device.createBuffer({ size: this.vocabSize * 4, usage: U.STORAGE | U.COPY_SRC });
    this.top2S1F32Buf = this.device.createBuffer({ size: 256 * 3 * 4, usage: U.STORAGE });
    this.top2S1U32Buf = this.device.createBuffer({ size: 256 * 4, usage: U.STORAGE });
    this.top2OutBuf = this.device.createBuffer({ size: 16, usage: U.STORAGE | U.COPY_SRC });
    this.readbackTop2Buf = this.device.createBuffer({ size: 16, usage: U.COPY_DST | U.MAP_READ });

    // ScarStore calibration & `+M` dynamic capsules buffers
    const calF32 = this.scarStore
      ? this.scarStore.getCalibrationBufferF32()
      : new Float32Array(this.hiddenDim + qDim);
    this.scarCalibBuf = this.device.createBuffer({
      size: calF32.byteLength,
      usage: U.STORAGE | U.COPY_DST,
    });
    this.device.queue.writeBuffer(this.scarCalibBuf, 0, calF32);

    this.scarRank = this.scarStore ? this.scarStore.rank : 60;
    this.scarNumProto = this.scarStore ? this.scarStore.numPrototypes : 8;
    this.scarBeta = this.scarStore ? this.scarStore.attentionBeta : 14.0;
    this.scarCapStride = 1 + 2 * (this.scarRank * this.hiddenDim) + this.scarNumProto * qDim;

    this.activeCapsulesBuf = this.device.createBuffer({
      size: Math.max(64, 8 * this.scarCapStride * 4),
      usage: U.STORAGE | U.COPY_DST,
    });
    this.scarTelemetryBuf = this.device.createBuffer({
      size: 64,
      usage: U.STORAGE | U.COPY_SRC | U.COPY_DST,
    });
    this.readbackScarTelBuf = this.device.createBuffer({
      size: 64,
      usage: U.COPY_DST | U.MAP_READ,
    });
    this.scarInjectUniform = this.device.createBuffer({
      size: 64,
      usage: U.UNIFORM | U.COPY_DST,
    });
    this._setActiveCapsulesGpu([]);
  }

  _makeUniform(buf64) {
    const u = this.device.createBuffer({
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(u, 0, buf64);
    return u;
  }

  _bg(ep, bindingMap) {
    const entries = Object.entries(bindingMap).map(([bIdx, buf]) => ({
      binding: Number(bIdx),
      resource: { buffer: buf, offset: 0, size: buf.size },
    }));
    return this.device.createBindGroup({
      layout: this.pipelines[ep].getBindGroupLayout(0),
      entries,
    });
  }

  _makeLinearOps(shardBuf, spec, xBuf, yBuf) {
    const ops = [];
    const outDim = spec.out_dim;
    const inDim = spec.in_dim;
    const wgs = Math.ceil(outDim / 8);

    if (spec.mode === "fp16") {
      const u = this._makeUniform(
        packKernelParams({
          p0: outDim,
          p1: inDim,
          p2: spec.w_u32_off,
          p3: spec.bias_u32_off,
          p4: spec.has_bias,
        }),
      );
      ops.push(["fp16_gemv_bias", this._bg("fp16_gemv_bias", { 0: u, 1: shardBuf, 3: xBuf, 6: yBuf }), wgs]);
    } else {
      const rank = spec.rank;
      const uDown = this._makeUniform(
        packKernelParams({ p0: rank, p1: inDim, p2: spec.ascar_u32_off, p3: 0, p4: 0 }),
      );
      ops.push([
        "fp16_gemv_bias",
        this._bg("fp16_gemv_bias", { 0: uDown, 1: shardBuf, 3: xBuf, 6: this.asvdUBuf }),
        Math.ceil(rank / 8),
      ]);

      const uUp = this._makeUniform(
        packKernelParams({
          p0: outDim,
          p1: inDim,
          p2: rank,
          p3: spec.packed_u32_off,
          p4: spec.scales_u32_off,
          p5: spec.bscar_u32_off,
          p6: spec.bias_u32_off,
          p7: spec.has_bias,
        }),
      );
      ops.push([
        "mos_fused_int4_bscar_gemv",
        this._bg("mos_fused_int4_bscar_gemv", {
          0: uUp,
          1: shardBuf,
          3: xBuf,
          4: this.asvdUBuf,
          6: yBuf,
        }),
        wgs,
      ]);
    }
    return ops;
  }

  _prebuildDispatchPlan() {
    this.embedOps = [];
    for (const ch of this.manifest.embed_chunks) {
      const sb = this.shardBuffers[ch.shard_index];
      const u = this._makeUniform(
        packKernelParams({
          p0: this.hiddenDim,
          p1: ch.packed_u32_off,
          p2: ch.scales_u32_off,
          p3: ch.vocab_start,
          p4: ch.vocab_rows,
        }),
      );
      this.embedOps.push([
        "embed_int4_lookup",
        this._bg("embed_int4_lookup", { 0: u, 1: sb, 2: this.stepStateBuf, 6: this.hCurBuf }),
        Math.ceil(this.hiddenDim / 256),
      ]);
    }

    const qDim = this.numQHeads * this.headDim;
    const totalRopePairs = (this.numQHeads + this.numKvHeads) * (this.headDim >> 1);
    const uRope = this._makeUniform(
      packKernelParams({ p0: this.numQHeads, p1: this.numKvHeads, p2: this.headDim }),
    );
    const bgRope = this._bg("qwen_rope_inplace", {
      0: uRope,
      2: this.stepStateBuf,
      3: this.invFreqBuf,
      6: this.qBuf,
      7: this.kBuf,
    });

    const uAddH = this._makeUniform(packKernelParams({ p0: this.hiddenDim }));
    const bgAddH = this._bg("vec_add_inplace_f32", { 0: uAddH, 3: this.projOutBuf, 6: this.hCurBuf });

    const uSwiglu = this._makeUniform(packKernelParams({ p0: this.interDim }));
    const bgSwiglu = this._bg("swiglu_mul_f32", { 0: uSwiglu, 3: this.upBuf, 6: this.gateBuf });

    const uCopyQ = this._makeUniform(packKernelParams({ p0: qDim }));
    const bgCopyQ = this._bg("vec_copy_f32", { 0: uCopyQ, 3: this.qBuf, 6: this.qUnropedLastBuf });

    const attnScale = 1.0 / Math.sqrt(this.headDim);

    this.layerOps = [];
    for (let lIdx = 0; lIdx < this.manifest.layers.length; lIdx++) {
      const lMeta = this.manifest.layers[lIdx];
      const sb = this.shardBuffers[lMeta.shard_index];
      const projs = lMeta.projections;
      const ops = [];

      const uLn1 = this._makeUniform(
        packKernelParams({ p0: this.hiddenDim, p1: lMeta.input_layernorm_u32_off }, { f0: this.rmsNormEps }),
      );
      ops.push(["rmsnorm_f32", this._bg("rmsnorm_f32", { 0: uLn1, 1: sb, 3: this.hCurBuf, 6: this.normBuf }), 1]);

      ops.push(...this._makeLinearOps(sb, projs["self_attn.q_proj"], this.normBuf, this.qBuf));
      ops.push(...this._makeLinearOps(sb, projs["self_attn.k_proj"], this.normBuf, this.kBuf));
      ops.push(...this._makeLinearOps(sb, projs["self_attn.v_proj"], this.normBuf, this.vBuf));

      if (lIdx === this.numLayers - 1) {
        ops.push(["vec_copy_f32", bgCopyQ, Math.ceil(qDim / 256)]);
      }

      ops.push(["qwen_rope_inplace", bgRope, Math.ceil(totalRopePairs / 64)]);

      const uScUpd = this._makeUniform(
        packKernelParams({
          p0: lIdx,
          p1: this.numKvHeads,
          p2: this.headDim,
          p3: this.numSink,
          p4: this.numSlots,
          p5: this.slideLen,
          p6: this.countsU32Total,
        }),
      );
      ops.push([
        "scache_o1_update_widrow_hoff",
        this._bg("scache_o1_update_widrow_hoff", {
          0: uScUpd,
          2: this.stepStateBuf,
          3: this.kBuf,
          4: this.vBuf,
          6: this.scacheWScarBuf,
          7: this.scacheSalienceBuf,
          8: this.scacheKvBuf,
        }),
        this.numKvHeads,
      ]);

      const uScAtt = this._makeUniform(
        packKernelParams(
          {
            p0: lIdx,
            p1: this.numQHeads,
            p2: this.numKvHeads,
            p3: this.headDim,
            p4: this.numSink,
            p5: this.numSlots,
            p6: this.slideLen,
            p7: this.countsU32Total,
          },
          { f0: attnScale, f1: this.deltaBlend },
        ),
      );
      ops.push([
        "scache_o1_gqa_attention",
        this._bg("scache_o1_gqa_attention", {
          0: uScAtt,
          1: this.scacheKvBuf,
          3: this.qBuf,
          4: this.scacheWScarBuf,
          6: this.attnOutBuf,
        }),
        this.numQHeads,
      ]);

      ops.push(...this._makeLinearOps(sb, projs["self_attn.o_proj"], this.attnOutBuf, this.projOutBuf));
      ops.push(["vec_add_inplace_f32", bgAddH, Math.ceil(this.hiddenDim / 256)]);

      const uLn2 = this._makeUniform(
        packKernelParams({ p0: this.hiddenDim, p1: lMeta.post_attention_layernorm_u32_off }, { f0: this.rmsNormEps }),
      );
      ops.push(["rmsnorm_f32", this._bg("rmsnorm_f32", { 0: uLn2, 1: sb, 3: this.hCurBuf, 6: this.normBuf }), 1]);
      ops.push(...this._makeLinearOps(sb, projs["mlp.gate_proj"], this.normBuf, this.gateBuf));
      ops.push(...this._makeLinearOps(sb, projs["mlp.up_proj"], this.normBuf, this.upBuf));
      ops.push(["swiglu_mul_f32", bgSwiglu, Math.ceil(this.interDim / 256)]);
      ops.push(...this._makeLinearOps(sb, projs["mlp.down_proj"], this.gateBuf, this.projOutBuf));
      ops.push(["vec_add_inplace_f32", bgAddH, Math.ceil(this.hiddenDim / 256)]);

      this.layerOps.push(ops);
    }

    const fnMeta = this.manifest.final_norm;
    const fnSb = this.shardBuffers[fnMeta.shard_index];
    const uFn = this._makeUniform(
      packKernelParams({ p0: this.hiddenDim, p1: fnMeta.w_u32_off }, { f0: this.rmsNormEps }),
    );
    this.finalNormOp = [
      "rmsnorm_f32",
      this._bg("rmsnorm_f32", { 0: uFn, 1: fnSb, 3: this.hCurBuf, 6: this.normBuf }),
      1,
    ];

    this.scarInjectOp = [
      "head_expanded_scar_inject",
      this._bg("head_expanded_scar_inject", {
        0: this.scarInjectUniform,
        3: this.scarCalibBuf,
        4: this.qUnropedLastBuf,
        5: this.activeCapsulesBuf,
        6: this.normBuf,
        7: this.scarTelemetryBuf,
      }),
      1,
    ];

    this.lmHeadOps = [];
    for (const ch of this.manifest.lm_head_chunks) {
      const sb = this.shardBuffers[ch.shard_index];
      const u = this._makeUniform(
        packKernelParams({
          p0: ch.vocab_rows,
          p1: this.hiddenDim,
          p2: ch.packed_u32_off,
          p3: ch.scales_u32_off,
          p4: ch.vocab_start,
        }),
      );
      this.lmHeadOps.push([
        "int4_vocab_logits_gemv",
        this._bg("int4_vocab_logits_gemv", { 0: u, 1: sb, 3: this.normBuf, 6: this.logitsBuf }),
        Math.ceil(ch.vocab_rows / 8),
      ]);
    }

    const uR1 = this._makeUniform(packKernelParams({ p0: this.vocabSize }));
    this.reduceOps = [
      [
        "logits_top2_reduce_stage1",
        this._bg("logits_top2_reduce_stage1", {
          0: uR1,
          3: this.logitsBuf,
          6: this.top2S1F32Buf,
          8: this.top2S1U32Buf,
        }),
        256,
      ],
      [
        "logits_top2_reduce_stage2",
        this._bg("logits_top2_reduce_stage2", {
          1: this.top2S1U32Buf,
          3: this.top2S1F32Buf,
          8: this.top2OutBuf,
        }),
        1,
      ],
    ];
  }

  _setActiveCapsulesGpu(activeCapsules) {
    const mAct = Math.min(8, activeCapsules.length);
    this.currentMActive = mAct;
    if (mAct > 0) {
      const packed = new Float32Array(mAct * this.scarCapStride);
      for (let i = 0; i < mAct; i++) {
        const base = i * this.scarCapStride;
        packed[base] = activeCapsules[i].gate;
        packed.set(activeCapsules[i].floats, base + 1);
      }
      this.device.queue.writeBuffer(this.activeCapsulesBuf, 0, packed);
    }
    const uBuf = packKernelParams(
      {
        p0: this.hiddenDim,
        p1: this.numQHeads,
        p2: this.headDim,
        p3: this.scarRank,
        p4: mAct,
        p5: this.scarNumProto,
      },
      { f0: this.scarBeta },
    );
    this.device.queue.writeBuffer(this.scarInjectUniform, 0, uBuf);
  }

  resetSCache() {
    this.device.queue.writeBuffer(this.scacheKvBuf, 0, this._zeroKvBytes);
    this.device.queue.writeBuffer(this.scacheWScarBuf, 0, this._zeroWScarBytes);
    this.device.queue.writeBuffer(this.scacheSalienceBuf, 0, this._zeroSalBytes);
    this._setActiveCapsulesGpu([]);
  }

  _recordOps(cpass, ops) {
    for (let i = 0; i < ops.length; i++) {
      const [ep, bg, wgs] = ops[i];
      cpass.setPipeline(this.pipelines[ep]);
      cpass.setBindGroup(0, bg);
      cpass.dispatchWorkgroups(wgs, 1, 1);
    }
  }

  async _readTop2AndTelemetry() {
    await this.readbackTop2Buf.mapAsync(GPUMapMode.READ, 0, 16);
    const raw = this.readbackTop2Buf.getMappedRange(0, 16).slice(0);
    this.readbackTop2Buf.unmap();

    const u32 = new Uint32Array(raw);
    const f32 = new Float32Array(raw);
    const top1Id = u32[0];
    const v1 = f32[1];
    const v2 = f32[2];
    const sexp = Math.max(f32[3], 1e-12);
    const p1 = 1.0 / sexp;
    const p2 = Math.exp(Math.max(-80.0, v2 - v1)) / sexp;

    let spread = 0.0;
    let stepWeights = null;
    if (this.currentMActive >= 2) {
      await this.readbackScarTelBuf.mapAsync(GPUMapMode.READ, 0, 64);
      const telRaw = this.readbackScarTelBuf.getMappedRange(0, 64).slice(0);
      this.readbackScarTelBuf.unmap();
      const telF32 = new Float32Array(telRaw);
      spread = telF32[8];
      stepWeights = Array.from(telF32.slice(0, this.currentMActive));
    }
    return { top1Id, p1, margin: p1 - p2, spread, stepWeights };
  }

  async generate(userQuery, options = {}) {
    const {
      maxNewTokens = 72,
      enableScarStore = true,
      systemPrompt = null,
      onToken = null,
      onRouting = null,
      signal = null,
    } = options;

    const sysTxt = systemPrompt ?? selectChatSystemPrompt(userQuery);
    const promptIds = this.tokenizer.applyChatTemplate(
      [
        { role: "system", content: sysTxt },
        { role: "user", content: userQuery },
      ],
      true,
    );
    const lp = promptIds.length;
    const rawQStart = Math.min(findSubsequenceLastEnd(promptIds, this.userTurnPrefixIds), Math.max(0, lp - 1));
    const qStart = Math.max(rawQStart, lp - MAX_ROUTING_USER_TOKENS);
    const lu = Math.max(1, lp - qStart);

    this.resetSCache();
    const scacheBytesInitial = this.scacheGpuBytes;

    const t0 = performance.now();
    // Chunked prefill submission (<= PREFILL_BATCH_TOKENS per command encoder) with zero mapAsync stalls!
    for (let batchStart = 0; batchStart < lp; batchStart += PREFILL_BATCH_TOKENS) {
      const batchEnd = Math.min(lp, batchStart + PREFILL_BATCH_TOKENS);
      const batchLen = batchEnd - batchStart;
      const stepsU32 = new Uint32Array(batchLen * 4);
      for (let i = 0; i < batchLen; i++) {
        const pos = batchStart + i;
        stepsU32[i * 4 + 0] = promptIds[pos];
        stepsU32[i * 4 + 1] = pos;
      }
      this.device.queue.writeBuffer(this.prefillStepsBuf, 0, stepsU32);

      const enc = this.device.createCommandEncoder();
      for (let i = 0; i < batchLen; i++) {
        const pos = batchStart + i;
        enc.copyBufferToBuffer(this.prefillStepsBuf, i * 16, this.stepStateBuf, 0, 16);

        if (pos < lp - 1) {
          if (pos >= qStart) {
            const cpassPre = enc.beginComputePass();
            this._recordOps(cpassPre, this.embedOps);
            for (let l = 0; l < this.numLayers - 1; l++) {
              this._recordOps(cpassPre, this.layerOps[l]);
            }
            cpassPre.end();

            const uSlot = pos - qStart;
            enc.copyBufferToBuffer(
              this.hCurBuf,
              0,
              this.prefillXInReadBuf,
              uSlot * this.hiddenDim * 4,
              this.hiddenDim * 4,
            );

            const cpassLast = enc.beginComputePass();
            this._recordOps(cpassLast, this.layerOps[this.numLayers - 1]);
            cpassLast.end();
          } else {
            const cpass = enc.beginComputePass();
            this._recordOps(cpass, this.embedOps);
            for (let l = 0; l < this.numLayers; l++) {
              this._recordOps(cpass, this.layerOps[l]);
            }
            cpass.end();
          }
        } else {
          // Final prompt token `lp - 1`: stop right before `layers[L-1]` to route ScarStore!
          const cpassPre = enc.beginComputePass();
          this._recordOps(cpassPre, this.embedOps);
          for (let l = 0; l < this.numLayers - 1; l++) {
            this._recordOps(cpassPre, this.layerOps[l]);
          }
          cpassPre.end();

          const uSlot = pos - qStart;
          enc.copyBufferToBuffer(
            this.hCurBuf,
            0,
            this.prefillXInReadBuf,
            uSlot * this.hiddenDim * 4,
            this.hiddenDim * 4,
          );
        }
      }
      this.device.queue.submit([enc.finish()]);
    }

    // Read `x_in` at `layers[L-1]` across user tokens and execute Span-Max + HTTP 206 Range fetch!
    let routeResult = { active: [], turnBytesFetched: 0, httpEvents: [] };
    if (enableScarStore && this.scarStore) {
      const readBytes = lu * this.hiddenDim * 4;
      await this.prefillXInReadBuf.mapAsync(GPUMapMode.READ, 0, readBytes);
      const xinRaw = this.prefillXInReadBuf.getMappedRange(0, readBytes).slice(0);
      this.prefillXInReadBuf.unmap();
      const xinF32 = new Float32Array(xinRaw);
      routeResult = await this.scarStore.routeAndFetchActiveCapsules(xinF32, lu);
      this._setActiveCapsulesGpu(routeResult.active);
    } else {
      if (this.scarStore) {
        this.scarStore.lastActiveGates.clear();
      }
      this._setActiveCapsulesGpu([]);
    }

    if (onRouting) {
      onRouting(routeResult);
    }

    // Finish `layers[L-1] -> final_norm -> head_expanded_scar_inject -> lm_head -> top2` on `lp - 1`
    const encTail = this.device.createCommandEncoder();
    const cpassTail = encTail.beginComputePass();
    this._recordOps(cpassTail, this.layerOps[this.numLayers - 1]);
    this._recordOps(cpassTail, [this.finalNormOp, this.scarInjectOp]);
    this._recordOps(cpassTail, this.lmHeadOps);
    this._recordOps(cpassTail, this.reduceOps);
    cpassTail.end();
    encTail.copyBufferToBuffer(this.top2OutBuf, 0, this.readbackTop2Buf, 0, 16);
    if (this.currentMActive >= 2) {
      encTail.copyBufferToBuffer(this.scarTelemetryBuf, 0, this.readbackScarTelBuf, 0, 64);
    }
    this.device.queue.submit([encTail.finish()]);

    const firstStep = await this._readTop2AndTelemetry();
    const prefillMs = performance.now() - t0;
    const scacheBytesAfterPrefill = this.scacheGpuBytes;

    const generatedIds = [];
    const confs = [];
    const margins = [];
    let maxSpread = firstStep.spread || 0.0;

    if (!this.eosTokenIds.has(firstStep.top1Id)) {
      generatedIds.push(firstStep.top1Id);
      confs.push(firstStep.p1);
      margins.push(firstStep.margin);
      if (onToken) {
        onToken({
          text: this.tokenizer.decode(generatedIds, true),
          generatedTokens: generatedIds.length,
          tokPerSec: 0,
          scacheBytes: this.scacheGpuBytes,
          scacheDeltaBytes: this.scacheGpuBytes - scacheBytesInitial,
          confidence: firstStep.p1,
          margin: firstStep.margin,
        });
      }
    }

    const tDec0 = performance.now();
    let curPos = lp;
    const singleStepArr = new Uint32Array(4);

    while (
      generatedIds.length > 0 &&
      generatedIds.length < maxNewTokens &&
      !this.eosTokenIds.has(generatedIds[generatedIds.length - 1])
    ) {
      if (signal && signal.aborted) break;

      singleStepArr[0] = generatedIds[generatedIds.length - 1];
      singleStepArr[1] = curPos;
      singleStepArr[2] = this.currentMActive;
      this.device.queue.writeBuffer(this.stepStateBuf, 0, singleStepArr);

      const enc = this.device.createCommandEncoder();
      const cpass = enc.beginComputePass();
      this._recordOps(cpass, this.embedOps);
      for (let l = 0; l < this.numLayers; l++) {
        this._recordOps(cpass, this.layerOps[l]);
      }
      this._recordOps(cpass, [this.finalNormOp, this.scarInjectOp]);
      this._recordOps(cpass, this.lmHeadOps);
      this._recordOps(cpass, this.reduceOps);
      cpass.end();
      enc.copyBufferToBuffer(this.top2OutBuf, 0, this.readbackTop2Buf, 0, 16);
      if (this.currentMActive >= 2) {
        enc.copyBufferToBuffer(this.scarTelemetryBuf, 0, this.readbackScarTelBuf, 0, 64);
      }
      this.device.queue.submit([enc.finish()]);

      const stepOut = await this._readTop2AndTelemetry();
      curPos += 1;
      if (stepOut.spread > maxSpread) maxSpread = stepOut.spread;
      if (this.eosTokenIds.has(stepOut.top1Id)) break;

      generatedIds.push(stepOut.top1Id);
      confs.push(stepOut.p1);
      margins.push(stepOut.margin);

      const decElapsedS = Math.max((performance.now() - tDec0) / 1000.0, 1e-4);
      const liveTokS = (generatedIds.length - 1) / decElapsedS;
      if (onToken) {
        onToken({
          text: this.tokenizer.decode(generatedIds, true),
          generatedTokens: generatedIds.length,
          tokPerSec: liveTokS,
          scacheBytes: this.scacheGpuBytes,
          scacheDeltaBytes: this.scacheGpuBytes - scacheBytesInitial,
          confidence: stepOut.p1,
          margin: stepOut.margin,
        });
      }
    }

    const decodeMs = performance.now() - tDec0;
    const scacheBytesFinal = this.scacheGpuBytes;
    const decSteps = Math.max(1, generatedIds.length - 1);
    const decodeTokPerSec = generatedIds.length > 1 && decodeMs > 0 ? decSteps / (decodeMs / 1000.0) : 0.0;
    const meanConf = confs.length > 0 ? confs.reduce((a, b) => a + b, 0) / confs.length : 0.0;
    const meanMargin = margins.length > 0 ? margins.reduce((a, b) => a + b, 0) / margins.length : 0.0;

    return {
      answer: this.tokenizer.decode(generatedIds, true).trim(),
      generatedTokenIds: generatedIds,
      prefillTokens: lp,
      generatedTokens: generatedIds.length,
      prefillMs: Math.round(prefillMs * 100) / 100,
      decodeMs: Math.round(decodeMs * 100) / 100,
      decodeTokPerSec: Math.round(decodeTokPerSec * 100) / 100,
      scacheBytesInitial,
      scacheBytesAfterPrefill,
      scacheBytesFinal,
      scacheDeltaBytes: scacheBytesFinal - scacheBytesInitial,
      vramWeightsMiB: Math.round((this.totalWeightGpuBytes / (1024 * 1024)) * 100) / 100,
      activeCapsules: routeResult.active.map((a) => ({
        capsuleId: a.capsuleId,
        gate: Math.round(a.gate * 10000) / 10000,
        fromCache: a.fromCache,
      })),
      turnHttp206Bytes: routeResult.turnBytesFetched,
      totalHttp206Bytes: this.scarStore ? this.scarStore.totalHttp206Bytes : 0,
      httpEvents: routeResult.httpEvents,
      meanConfidenceP1: Math.round(meanConf * 10000) / 10000,
      meanMarginP1P2: Math.round(meanMargin * 10000) / 10000,
      maxInterHeadSpread: Math.round(maxSpread * 10000) / 10000,
    };
  }
}
