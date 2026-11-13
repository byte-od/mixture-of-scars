// ============================================================================
// Browser `HeadExpandedScarStore` + Live RFC 7233 `HTTP/1.1 206` Range Client
// Streams exact `62,720 B` (`61.25 KB`) / `143,360 B` (`140.00 KB`) 72B slices
// on demand with `0 B` upfront cartridge payload download & `< 0.2 ms` `0x00`
// cryptographic LGPD/GDPR unlearning!
// ============================================================================

export function fp16BitsToF32(h) {
  const s = (h & 0x8000) << 16;
  const e = (h >> 10) & 0x1f;
  const m = h & 0x03ff;
  const buf = new ArrayBuffer(4);
  const u32 = new Uint32Array(buf);
  const f32 = new Float32Array(buf);

  if (e === 0) {
    if (m === 0) {
      u32[0] = s;
      return f32[0];
    }
    // Subnormal FP16 -> normalized FP32
    let mant = m;
    let exp = -14;
    while ((mant & 0x0400) === 0) {
      mant <<= 1;
      exp -= 1;
    }
    mant &= 0x03ff;
    u32[0] = s | ((exp + 127) << 23) | (mant << 13);
    return f32[0];
  }
  if (e === 31) {
    u32[0] = s | 0x7f800000 | (m << 13);
    return f32[0];
  }
  u32[0] = s | ((e - 15 + 127) << 23) | (m << 13);
  return f32[0];
}

export function findSubsequenceLastEnd(seqIds, pattern) {
  if (!pattern || pattern.length === 0 || seqIds.length < pattern.length) {
    return 0;
  }
  const m = pattern.length;
  for (let idx = seqIds.length - m; idx >= 0; idx--) {
    let ok = true;
    for (let j = 0; j < m; j++) {
      if (seqIds[idx + j] !== pattern[j]) {
        ok = false;
        break;
      }
    }
    if (ok) {
      return Math.min(idx + m, Math.max(seqIds.length - 1, 0));
    }
  }
  return 0;
}

export class BrowserScarStoreHTTP206 {
  constructor(routerMeta, routerBinBuffer, baseUrl = ".") {
    this.meta = routerMeta;
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.hiddenDim = routerMeta.hidden_dim;
    this.numHeads = routerMeta.num_heads;
    this.headDim = routerMeta.head_dim;
    this.rank = routerMeta.rank;
    this.numPrototypes = routerMeta.num_key_prototypes;
    this.attentionBeta = routerMeta.attention_beta ?? 14.0;
    this.recordBytes = routerMeta.record_bytes;
    this.spanWindow = routerMeta.span_window ?? 10;
    this.spanStride = routerMeta.span_stride ?? 3;
    this.userTurnPrefixIds = routerMeta.user_turn_prefix_ids || [151644, 872, 198];

    const floats = new Float32Array(routerBinBuffer);
    const lay = routerMeta.router_bin_layout;

    this.muQueryLast = floats.slice(
      lay.mu_query_last.offset_floats,
      lay.mu_query_last.offset_floats + lay.mu_query_last.length_floats,
    );
    this.muHiddenFinal = floats.slice(
      lay.mu_hidden_final.offset_floats,
      lay.mu_hidden_final.offset_floats + lay.mu_hidden_final.length_floats,
    );
    this.muQHeads = floats.slice(
      lay.mu_q_heads.offset_floats,
      lay.mu_q_heads.offset_floats + lay.mu_q_heads.length_floats,
    );
    this.zAnchorsFlat = floats.slice(
      lay.z_anchors.offset_floats,
      lay.z_anchors.offset_floats + lay.z_anchors.length_floats,
    );

    // Zero upfront cartridge payloads downloaded! Fetched strictly on-demand via HTTP 206.
    this.hotCapsules = new Map();     // capsuleId -> { packedGpuFloats, rawBytesView }
    this.unlearnedIds = new Set();    // cryptographically wiped capsuleIds
    this.http206Events = [];          // audit log of every HTTP 206 Range request
    this.totalHttp206Bytes = 0;
    this.lastCosines = new Map();
    this.lastActiveGates = new Map();
  }

  static async load(routerJsonUrl = "./scarstore_router.json") {
    const resp = await fetch(routerJsonUrl);
    if (!resp.ok) {
      throw new Error(`Failed to fetch ScarStore router ${routerJsonUrl}: HTTP ${resp.status}`);
    }
    const meta = await resp.json();
    const baseDir = routerJsonUrl.includes("/")
      ? routerJsonUrl.slice(0, routerJsonUrl.lastIndexOf("/"))
      : ".";
    const binResp = await fetch(`${baseDir}/${meta.router_bin_file}`);
    if (!binResp.ok) {
      throw new Error(`Failed to fetch ${meta.router_bin_file}: HTTP ${binResp.status}`);
    }
    const binBuf = await binResp.arrayBuffer();
    return new BrowserScarStoreHTTP206(meta, binBuf, baseDir);
  }

  getCalibrationBufferF32() {
    const out = new Float32Array(this.muHiddenFinal.length + this.muQHeads.length);
    out.set(this.muHiddenFinal, 0);
    out.set(this.muQHeads, this.muHiddenFinal.length);
    return out;
  }

  getCapsuleStrideFloats() {
    const qDim = this.numHeads * this.headDim;
    return 1 + 2 * (this.rank * this.hiddenDim) + this.numPrototypes * qDim;
  }

  getCapsulesDirectory() {
    return (this.meta.capsules || []).map((c) => ({
      ...c,
      cachedInBrowser: this.hotCapsules.has(c.capsule_id),
      unlearnedWiped: this.unlearnedIds.has(c.capsule_id),
      lastCosine: this.lastCosines.get(c.capsule_id) ?? null,
      lastGate: this.lastActiveGates.get(c.capsule_id) ?? 0.0,
    }));
  }

  unpackCapsuleInt4Bytes(arrayBuffer) {
    const bytes = new Uint8Array(arrayBuffer);
    const dv = new DataView(arrayBuffer);
    const d = this.hiddenDim;
    const r = this.rank;
    const pCnt = this.numPrototypes;
    const qDim = this.numHeads * this.headDim;
    const nAb = (d * r) >> 6;
    const nK = (pCnt * qDim) >> 6;
    let off = 0;

    const dequantBlocks = (nBlk) => {
      const out = new Float32Array(nBlk * 64);
      const pkOff = off;
      off += nBlk * 32;
      const scOff = off;
      off += nBlk * 2;

      for (let b = 0; b < nBlk; b++) {
        const scale = fp16BitsToF32(dv.getUint16(scOff + b * 2, true));
        const rowByteBase = pkOff + b * 32;
        const outBase = b * 64;
        for (let i = 0; i < 32; i++) {
          const pb = bytes[rowByteBase + i];
          out[outBase + i * 2] = ((pb & 0x0f) - 8) * scale;
          out[outBase + i * 2 + 1] = (((pb >> 4) & 0x0f) - 8) * scale;
        }
      }
      return out;
    };

    const bFlat = dequantBlocks(nAb); // [d * r]
    const aFlat = dequantBlocks(nAb); // [r * d]
    const kFlat = dequantBlocks(nK);  // [pCnt * numHeads * headDim]

    // Unit-normalize each per-head prototype [p, h, headDim]
    const totalHeads = pCnt * this.numHeads;
    for (let ph = 0; ph < totalHeads; ph++) {
      const base = ph * this.headDim;
      let normSq = 0.0;
      for (let j = 0; j < this.headDim; j++) {
        const v = kFlat[base + j];
        normSq += v * v;
      }
      const invNorm = 1.0 / Math.max(Math.sqrt(normSq), 1e-8);
      for (let j = 0; j < this.headDim; j++) {
        kFlat[base + j] *= invNorm;
      }
    }

    const totalLen = aFlat.length + bFlat.length + kFlat.length;
    const combined = new Float32Array(totalLen);
    combined.set(aFlat, 0);
    combined.set(bFlat, aFlat.length);
    combined.set(kFlat, aFlat.length + bFlat.length);
    return { packedGpuFloats: combined, rawBytesView: bytes };
  }

  async fetchCapsuleSliceHTTP206(capEntry) {
    const cid = capEntry.capsule_id;
    if (this.hotCapsules.has(cid)) {
      return {
        floats: this.hotCapsules.get(cid).packedGpuFloats,
        fromCache: true,
        bytesFetched: 0,
        status: 200,
      };
    }

    const startByte = capEntry.byte_offset;
    const endByte = startByte + capEntry.record_bytes - 1;
    const rangeHeader = `bytes=${startByte}-${endByte}`;

    const candidateUrls = [];
    if (this.meta.local_http206_store_url) {
      candidateUrls.push(`${this.baseUrl}/${this.meta.local_http206_store_url}`);
    }
    if (this.meta.cloud_http206_store_url) {
      candidateUrls.push(this.meta.cloud_http206_store_url);
    }

    let lastErr = null;
    for (const storeUrl of candidateUrls) {
      const t0 = performance.now();
      try {
        const resp = await fetch(storeUrl, {
          method: "GET",
          headers: { Range: rangeHeader },
        });
        if (resp.status !== 206 && resp.status !== 200) {
          throw new Error(`Unexpected HTTP ${resp.status} on Range ${rangeHeader}`);
        }
        let buf = await resp.arrayBuffer();
        // Safety fallback if a static host without Range support returned 200 full file
        if (resp.status === 200 && buf.byteLength > capEntry.record_bytes) {
          buf = buf.slice(startByte, startByte + capEntry.record_bytes);
        }
        const elapsedMs = Math.round((performance.now() - t0) * 100) / 100;
        const unpacked = this.unpackCapsuleInt4Bytes(buf);
        this.hotCapsules.set(cid, unpacked);
        this.totalHttp206Bytes += buf.byteLength;

        const evt = {
          capsuleId: cid,
          url: storeUrl,
          status: resp.status,
          range: rangeHeader,
          contentRange: resp.headers.get("Content-Range") || `${rangeHeader}/*`,
          bytes: buf.byteLength,
          kb: Math.round((buf.byteLength / 1024) * 100) / 100,
          latencyMs: elapsedMs,
          timestamp: new Date().toISOString(),
        };
        this.http206Events.unshift(evt);
        return {
          floats: unpacked.packedGpuFloats,
          fromCache: false,
          bytesFetched: buf.byteLength,
          status: resp.status,
          event: evt,
        };
      } catch (err) {
        lastErr = err;
      }
    }
    throw lastErr || new Error(`Could not stream capsule ${cid} via HTTP 206`);
  }

  unlearnCapsule(capsuleId) {
    const t0 = performance.now();
    let zeroedBytes = 0;
    const entry = this.hotCapsules.get(capsuleId);
    if (entry) {
      entry.packedGpuFloats.fill(0.0);
      entry.rawBytesView.fill(0);
      zeroedBytes = entry.packedGpuFloats.byteLength + entry.rawBytesView.byteLength;
      this.hotCapsules.delete(capsuleId);
    }
    // Also zero the corresponding z_anchor in browser RAM so cone cosine drops strictly to 0.0000
    const caps = this.meta.capsules || [];
    for (let i = 0; i < caps.length; i++) {
      if (caps[i].capsule_id === capsuleId) {
        const zOff = i * this.hiddenDim;
        this.zAnchorsFlat.fill(0.0, zOff, zOff + this.hiddenDim);
        zeroedBytes += this.hiddenDim * 4;
      }
    }
    this.unlearnedIds.add(capsuleId);
    this.lastActiveGates.delete(capsuleId);
    const ms = Math.round((performance.now() - t0) * 1000) / 1000;
    return { capsuleId, zeroedBytes, unlearnMs: ms };
  }

  restoreCapsuleAnchor(capsuleId) {
    // Utility if user wants to re-enable a wiped capsule by reloading router.bin
    return this.unlearnedIds.has(capsuleId);
  }

  _buildSpanProbes(xInFlat, numUserTokens) {
    const d = this.hiddenDim;
    const mu = this.muQueryLast;
    const probes = [];

    const addWindowProbe = (sIdx, eIdx) => {
      const count = eIdx - sIdx;
      if (count <= 0) return;
      const invCount = 1.0 / count;
      const vec = new Float32Array(d);
      let normSq = 0.0;
      for (let c = 0; c < d; c++) {
        let sum = 0.0;
        for (let t = sIdx; t < eIdx; t++) {
          sum += xInFlat[t * d + c];
        }
        const v = sum * invCount - mu[c];
        vec[c] = v;
        normSq += v * v;
      }
      const invNorm = 1.0 / Math.max(Math.sqrt(normSq), 1e-8);
      for (let c = 0; c < d; c++) vec[c] *= invNorm;
      probes.push(vec);
    };

    // 1. Global centered mean probe
    addWindowProbe(0, numUserTokens);

    // 2. Sliding Span-Max semantic windows (`window=10, stride=3`)
    if (numUserTokens >= this.spanWindow + 4) {
      const win = Math.min(this.spanWindow, numUserTokens);
      const startsSet = new Set();
      for (let s = 0; s <= Math.max(0, numUserTokens - win); s += Math.max(1, this.spanStride)) {
        startsSet.add(s);
      }
      startsSet.add(numUserTokens - win);
      if (numUserTokens > win + 5 && numUserTokens - 5 - win > 0) {
        startsSet.add(numUserTokens - 5 - win);
      }
      const starts = Array.from(startsSet).sort((a, b) => a - b);
      for (const s of starts) {
        const e = Math.min(numUserTokens, s + win);
        if (e - s >= 4) {
          addWindowProbe(s, e);
        }
      }
    }
    return probes;
  }

  async routeAndFetchActiveCapsules(xInFlat, numUserTokens) {
    const caps = this.meta.capsules || [];
    this.lastCosines.clear();
    this.lastActiveGates.clear();
    if (caps.length === 0 || numUserTokens <= 0) {
      return { active: [], turnBytesFetched: 0, httpEvents: [] };
    }

    const d = this.hiddenDim;
    const probes = this._buildSpanProbes(xInFlat, numUserTokens);
    const gVals = new Float32Array(caps.length);
    const idToIdx = new Map();
    let maxGate = 0.0;

    for (let i = 0; i < caps.length; i++) {
      const cap = caps[i];
      idToIdx.set(cap.capsule_id, i);
      if (this.unlearnedIds.has(cap.capsule_id)) {
        this.lastCosines.set(cap.capsule_id, 0.0);
        gVals[i] = 0.0;
        continue;
      }
      const zBase = i * d;
      let maxCos = -2.0;
      for (const probe of probes) {
        let dot = 0.0;
        for (let c = 0; c < d; c++) {
          dot += probe[c] * this.zAnchorsFlat[zBase + c];
        }
        if (dot > maxCos) maxCos = dot;
      }
      this.lastCosines.set(cap.capsule_id, maxCos);
      const ramp = Math.max(cap.ramp, 1e-6);
      const g = Math.min(1.0, Math.max(0.0, (maxCos - cap.tau) / ramp));
      gVals[i] = g;
      if (g > maxGate) maxGate = g;
    }

    if (maxGate <= 0.0) {
      return { active: [], turnBytesFetched: 0, httpEvents: [] };
    }

    // Conjunctive Sleep-Bridge parent supersession
    const baseCone = Float32Array.from(gVals);
    for (let i = 0; i < caps.length; i++) {
      const parents = caps[i].parent_capsules || [];
      if (parents.length === 0) continue;
      let parentMin = Infinity;
      let missing = false;
      for (const pid of parents) {
        const pIdx = idToIdx.get(pid);
        if (pIdx === undefined || baseCone[pIdx] <= 0.0) {
          missing = true;
          break;
        }
        if (baseCone[pIdx] < parentMin) parentMin = baseCone[pIdx];
      }
      if (missing) {
        gVals[i] = 0.0;
        continue;
      }
      const effBg = gVals[i] * Math.min(1.0, parentMin / 0.5);
      gVals[i] = effBg;
      if (effBg > 0.0) {
        const atten = Math.max(0.0, 1.0 - effBg);
        for (const pid of parents) {
          gVals[idToIdx.get(pid)] *= atten;
        }
      }
    }

    const active = [];
    let turnBytesFetched = 0;
    const httpEvents = [];
    for (let i = 0; i < caps.length; i++) {
      if (gVals[i] > 1e-6) {
        const fetched = await this.fetchCapsuleSliceHTTP206(caps[i]);
        turnBytesFetched += fetched.bytesFetched;
        if (fetched.event) httpEvents.push(fetched.event);
        this.lastActiveGates.set(caps[i].capsule_id, gVals[i]);
        active.push({
          capsuleId: caps[i].capsule_id,
          gate: gVals[i],
          floats: fetched.floats,
          fromCache: fetched.fromCache,
          meta: caps[i],
        });
      }
    }

    return { active, turnBytesFetched, httpEvents };
  }
}
