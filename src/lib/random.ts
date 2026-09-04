/**
 * Deterministic pseudo-random numbers (sfc32) with Box–Muller normals.
 * Every simulation in the dashboard is reproducible from its seed.
 */

export interface RNG {
  /** uniform on [0, 1) */
  uniform(): number;
  /** standard normal N(0, 1) */
  normal(): number;
  /** integer in [0, n) */
  int(n: number): number;
  /** sample an index according to probabilities (must sum to ~1) */
  categorical(probs: number[]): number;
  shuffle<T>(arr: T[]): T[];
}

function splitmix32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x9e3779b9) | 0;
    let t = a ^ (a >>> 16);
    t = Math.imul(t, 0x21f0aaad);
    t = t ^ (t >>> 15);
    t = Math.imul(t, 0x735a2d97);
    return ((t = t ^ (t >>> 15)) >>> 0) / 4294967296;
  };
}

export function makeRNG(seed: number): RNG {
  // seed sfc32 state from splitmix32
  const sm = splitmix32(seed);
  let a = Math.floor(sm() * 4294967296) >>> 0;
  let b = Math.floor(sm() * 4294967296) >>> 0;
  let c = Math.floor(sm() * 4294967296) >>> 0;
  let d = Math.floor(sm() * 4294967296) >>> 0;
  const next = () => {
    a >>>= 0;
    b >>>= 0;
    c >>>= 0;
    d >>>= 0;
    let t = (a + b) | 0;
    a = b ^ (b >>> 9);
    b = (c + (c << 3)) | 0;
    c = (c << 21) | (c >>> 11);
    d = (d + 1) | 0;
    t = (t + d) | 0;
    c = (c + t) | 0;
    return (t >>> 0) / 4294967296;
  };
  for (let i = 0; i < 12; i++) next();

  let spare: number | null = null;
  const rng: RNG = {
    uniform: next,
    normal() {
      if (spare !== null) {
        const v = spare;
        spare = null;
        return v;
      }
      let u = 0;
      let v = 0;
      let s = 0;
      do {
        u = 2 * next() - 1;
        v = 2 * next() - 1;
        s = u * u + v * v;
      } while (s >= 1 || s === 0);
      const f = Math.sqrt((-2 * Math.log(s)) / s);
      spare = v * f;
      return u * f;
    },
    int(n: number) {
      return Math.min(n - 1, Math.floor(next() * n));
    },
    categorical(probs: number[]) {
      const u = next();
      let acc = 0;
      for (let i = 0; i < probs.length; i++) {
        acc += probs[i];
        if (u < acc) return i;
      }
      return probs.length - 1;
    },
    shuffle<T>(arr: T[]): T[] {
      const out = arr.slice();
      for (let i = out.length - 1; i > 0; i--) {
        const j = Math.floor(next() * (i + 1));
        [out[i], out[j]] = [out[j], out[i]];
      }
      return out;
    },
  };
  return rng;
}
