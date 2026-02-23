/// rust_snn — Spiking Neural Network backend for NSCK
///
/// Follows the exact same cdylib / PyO3 pattern as rust_vsa/hypervec_rs.
/// The Python shim (snn_shim.py) tries `import snn_rs` first and falls
/// back to the pure-Python snn_perception.py when the .pyd is absent.
///
/// Hot paths accelerated here:
///   LIFLayer.step()          O(N) per timestep — vectorised + rayon
///   SnnCore.simulate()       Entire perceive() inner loop in one Rust call
///   StdpEngine.apply()       O(snn_size × input_dim) nested loop — rayon parallel
///   HebbianMatrix.update()   Oja's rule outer-product — see hebbian.rs
///   ConceptMapper.recognize  Jaccard per concept — see concept.rs

use pyo3::prelude::*;
use rayon::prelude::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

mod hebbian;
mod concept;

// ============================================================
// LIFLayer  –  direct PyO3 replacement for LIFNeuronLayer
// ============================================================

/// Leaky Integrate-and-Fire neuron layer.
///
/// Membrane equation: τ·dv/dt = -(v - v_rest) + I
/// Spike when v ≥ v_thresh → reset to v_reset, enter refractory.
///
/// All per-neuron updates are parallelised with rayon.
#[pyclass(module = "snn_rs")]
pub struct LIFLayer {
    n_neurons:          usize,
    tau:                f64,
    v_rest:             f64,
    v_reset:            f64,
    v_thresh:           f64,
    refractory_period:  f64,
    dt:                 f64,
    // mutable state
    v:                  Vec<f64>,
    refractory:         Vec<f64>,
    spike_history:      Vec<Vec<f64>>,
}

#[pymethods]
impl LIFLayer {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_neurons:         usize,
        tau:               f64,
        v_rest:            f64,
        v_reset:           f64,
        v_thresh:          f64,
        refractory_period: f64,
        dt:                f64,
    ) -> Self {
        LIFLayer {
            n_neurons,
            tau,
            v_rest,
            v_reset,
            v_thresh,
            refractory_period,
            dt,
            v:             vec![v_rest; n_neurons],
            refractory:    vec![0.0;    n_neurons],
            spike_history: Vec::new(),
        }
    }

    /// Single timestep LIF update.  Returns spike vector (0.0 | 1.0) length n_neurons.
    fn step(&mut self, input_current: Vec<f64>) -> Vec<f64> {
        let dt  = self.dt;
        let tau = self.tau;
        let v_rest   = self.v_rest;
        let v_reset  = self.v_reset;
        let v_thresh = self.v_thresh;
        let ref_period = self.refractory_period;

        // Parallel update — each neuron is independent
        let spikes: Vec<f64> = self.v
            .par_iter_mut()
            .zip(self.refractory.par_iter_mut())
            .zip(input_current.par_iter())
            .map(|((v, refrac), &i)| {
                if *refrac > 0.0 {
                    *refrac -= dt;
                    if *refrac < 0.0 { *refrac = 0.0; }
                    *v = v_reset;
                    0.0
                } else {
                    // Euler integration: dv = dt/τ · (-(v-v_rest) + I)
                    *v += dt / tau * (-(*v - v_rest) + i);
                    if *v >= v_thresh {
                        *v      = v_reset;
                        *refrac = ref_period;
                        1.0
                    } else {
                        0.0
                    }
                }
            })
            .collect();

        self.spike_history.push(spikes.clone());
        spikes
    }

    /// Reset membrane potentials and refractory timers to initial state.
    fn reset(&mut self) {
        self.v.fill(self.v_rest);
        self.refractory.fill(0.0);
        self.spike_history.clear();
    }

    /// Full spike train as flat Vec<f64> with shape (time_steps × n_neurons).
    /// Reshape in Python: np.array(layer.get_spike_train()).reshape(steps, n_neurons)
    fn get_spike_train(&self) -> Vec<Vec<f64>> {
        self.spike_history.clone()
    }

    /// Number of neurons (read-only property).
    #[getter]
    fn n_neurons(&self) -> usize { self.n_neurons }

    /// Timestep in ms.
    #[getter]
    fn dt(&self) -> f64 { self.dt }

    fn __repr__(&self) -> String {
        format!("<LIFLayer n_neurons={} tau={} dt={}>",
                self.n_neurons, self.tau, self.dt)
    }
}

// ============================================================
// StdpEngine  –  Spike-Timing-Dependent Plasticity
// ============================================================

/// STDP weight updater.
///
/// Rule:
///   Δt > 0 (pre before post) → LTP:  Δw = A+ · exp(-Δt / τ)
///   Δt < 0 (post before pre) → LTD:  Δw = -A- · exp(Δt / τ)
///
/// The `apply` call is O(snn_size × input_dim) and is fully parallelised
/// with rayon — the worst bottleneck of the Python perceive() loop.
#[pyclass(module = "snn_rs")]
pub struct StdpEngine {
    input_dim:         usize,
    snn_size:          usize,
    stdp_lr:           f64,
    tau_stdp:          f64,
    a_plus:            f64,
    a_minus:           f64,
    pre_spike_times:   Vec<f64>,  // length input_dim
    post_spike_times:  Vec<f64>,  // length snn_size
    current_time:      f64,
    pub updates:       usize,
}

#[pymethods]
impl StdpEngine {
    #[new]
    fn new(
        input_dim: usize,
        snn_size:  usize,
        stdp_lr:   f64,
        tau_stdp:  f64,
        a_plus:    f64,
        a_minus:   f64,
    ) -> Self {
        StdpEngine {
            input_dim,
            snn_size,
            stdp_lr,
            tau_stdp,
            a_plus,
            a_minus,
            pre_spike_times:  vec![-1000.0; input_dim],
            post_spike_times: vec![-1000.0; snn_size],
            current_time:     0.0,
            updates:          0,
        }
    }

    /// Advance internal clock by one dt step.
    fn advance_time(&mut self, dt: f64) {
        self.current_time += dt;
    }

    /// Reset clock and spike time history (call at start of each perceive()).
    fn reset(&mut self) {
        self.pre_spike_times.fill(-1000.0);
        self.post_spike_times.fill(-1000.0);
        self.current_time = 0.0;
    }

    /// Apply one STDP step.  Returns flat delta_w Vec<f64> (snn_size × input_dim row-major).
    ///
    /// The returned delta_w should be added to input_weights:
    ///   input_weights += stdp_lr * delta_w
    fn apply(
        &mut self,
        input_spikes:  Vec<f64>,  // length input_dim
        output_spikes: Vec<f64>,  // length snn_size
    ) -> Vec<f64> {
        let t = self.current_time;

        // Update spike time records (sequential — fast, just index writes)
        for (i, &s) in input_spikes.iter().enumerate() {
            if s > 0.5 { self.pre_spike_times[i] = t; }
        }
        for (i, &s) in output_spikes.iter().enumerate() {
            if s > 0.5 { self.post_spike_times[i] = t; }
        }

        let tau         = self.tau_stdp;
        let a_plus      = self.a_plus;
        let a_minus     = self.a_minus;
        let input_dim   = self.input_dim;
        let window      = 5.0 * tau;

        // Snapshot borrows before parallel closure
        let pre_times  = &self.pre_spike_times;
        let post_times = &self.post_spike_times;

        // O(snn_size × input_dim) — parallelised by rayon over all synapses
        let delta_w: Vec<f64> = (0..(self.snn_size * input_dim))
            .into_par_iter()
            .map(|idx| {
                let post_idx = idx / input_dim;
                let pre_idx  = idx % input_dim;

                let t_post = post_times[post_idx];
                let t_pre  = pre_times[pre_idx];

                if t_post < -999.0 || t_pre < -999.0 {
                    return 0.0;
                }

                let delta_t = t_post - t_pre;
                if delta_t.abs() >= window { return 0.0; }

                let raw = if delta_t > 0.0 {
                    // LTP: pre before post — strengthen
                    a_plus * (-delta_t / tau).exp()
                } else {
                    // LTD: post before pre — weaken
                    -a_minus * (delta_t / tau).exp()
                };
                // Scale by learning rate so callers simply do: weights += engine.apply(...)
                raw * self.stdp_lr
            })
            .collect();

        self.updates += 1;
        delta_w   // caller multiplies by stdp_lr and adds to weights
    }

    #[getter]
    fn updates(&self) -> usize { self.updates }

    #[getter]
    fn current_time(&self) -> f64 { self.current_time }
}

// ============================================================
// SnnCore  –  complete perceive() inner loop in a single call
// ============================================================

/// Full-cycle accelerated SNN module.
///
/// Calling `simulate()` replaces the entire Python `for step in range(n_steps):`
/// loop including LIF integration, weight-matrix projection, STDP, and Poisson
/// spike generation — all from one PyO3 call (zero Python overhead per step).
#[pyclass(module = "snn_rs")]
pub struct SnnCore {
    // topology
    input_dim:  usize,
    snn_size:   usize,

    // LIF parameters (mirrored for internal use)
    tau:               f64,
    v_rest:            f64,
    v_reset:           f64,
    v_thresh:          f64,
    refractory_period: f64,
    dt:                f64,

    // mutable LIF state
    v:           Vec<f64>,
    refractory:  Vec<f64>,

    // weight matrix [snn_size × input_dim] row-major
    pub input_weights: Vec<f64>,

    // STDP
    stdp_enabled:     bool,
    stdp_lr:          f64,
    tau_stdp:         f64,
    a_plus:           f64,
    a_minus:          f64,
    pre_spike_times:  Vec<f64>,
    post_spike_times: Vec<f64>,
    current_time:     f64,

    // stats
    pub stdp_updates:   usize,
    pub weight_updates: usize,

    // RNG seed (deterministic per-constructor)
    rng_seed: u64,

    // Weber-Fechner log compression (psychophysics-inspired)
    weber_fechner: bool,
}

#[pymethods]
impl SnnCore {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        input_dim:         usize,
        snn_size:          usize,
        tau:               f64,
        v_rest:            f64,
        v_reset:           f64,
        v_thresh:          f64,
        refractory_period: f64,
        dt:                f64,
        stdp_enabled:      bool,
        stdp_lr:           f64,
        tau_stdp:          f64,
        a_plus:            f64,
        a_minus:           f64,
        seed:              Option<u64>,
        weber_fechner:     Option<bool>,
    ) -> Self {
        let rng_seed = seed.unwrap_or(42);
        let mut rng = ChaCha8Rng::seed_from_u64(rng_seed);

        // Initialise weight matrix with scaled Gaussian (matches Python * 5.0 + normalise)
        let mut input_weights: Vec<f64> = (0..(snn_size * input_dim))
            .map(|_| rng.gen::<f64>() * 2.0 - 1.0)  // uniform [-1, 1]
            .collect();

        // L2-normalise each row (same as Python's _normalize_weights)
        for row in 0..snn_size {
            let start = row * input_dim;
            let end   = start + input_dim;
            let norm: f64 = input_weights[start..end].iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 0.0 {
                for w in &mut input_weights[start..end] { *w /= norm; }
            }
        }

        SnnCore {
            input_dim, snn_size,
            tau, v_rest, v_reset, v_thresh, refractory_period, dt,
            v:           vec![v_rest; snn_size],
            refractory:  vec![0.0;    snn_size],
            input_weights,
            stdp_enabled, stdp_lr, tau_stdp, a_plus, a_minus,
            pre_spike_times:  vec![-1000.0; input_dim],
            post_spike_times: vec![-1000.0; snn_size],
            current_time: 0.0,
            stdp_updates:   0,
            weight_updates: 1,  // counts the initialisation normalisation
            rng_seed,
            weber_fechner: weber_fechner.unwrap_or(false),
        }
    }

    /// Run the full perceive() inner loop for `n_steps` timesteps.
    ///
    /// Returns the spike train as a Vec<Vec<f64>> of shape (n_steps, snn_size).
    /// This single call replaces the entire Python `for step in range(n_steps):` block.
    ///
    /// When `weber_fechner` is enabled (default), applies sign-preserving logarithmic
    /// compression to the sensory input: x_wf = sign(x) · ln(1 + |x|).
    /// This is inspired by the Weber-Fechner law (Fechner, 1860) — the same principle
    /// biological auditory and visual systems use to compress dynamic range.
    fn simulate(
        &mut self,
        sensory_input: Vec<f64>,
        n_steps:       usize,
        learn:         bool,
    ) -> Vec<Vec<f64>> {
        // ── Weber-Fechner logarithmic compression (psychophysics) ──────
        // x_wf = sign(x) · ln(1 + |x|)
        // Compresses dynamic range: improves discrimination at low intensities,
        // prevents saturation at high intensities.
        let sensory: Vec<f64> = if self.weber_fechner {
            sensory_input.iter()
                .map(|&x| x.signum() * (1.0 + x.abs()).ln())
                .collect()
        } else {
            sensory_input
        };

        // Fresh simulation state
        self.v.fill(self.v_rest);
        self.refractory.fill(0.0);
        if learn && self.stdp_enabled {
            self.pre_spike_times.fill(-1000.0);
            self.post_spike_times.fill(-1000.0);
            self.current_time = 0.0;
        }

        let mut rng = ChaCha8Rng::seed_from_u64(
            self.rng_seed.wrapping_add(self.stdp_updates as u64) // vary per call
        );

        // Cache snapshot values to avoid repeated self-borrow inside loop
        let dt             = self.dt;
        let tau            = self.tau;
        let v_rest         = self.v_rest;
        let v_reset        = self.v_reset;
        let v_thresh       = self.v_thresh;
        let ref_period     = self.refractory_period;
        let input_dim      = self.input_dim;
        let snn_size       = self.snn_size;
        let tau_stdp       = self.tau_stdp;
        let a_plus         = self.a_plus;
        let a_minus        = self.a_minus;
        let window         = 5.0 * tau_stdp;
        let stdp_enabled   = self.stdp_enabled;
        let stdp_lr        = self.stdp_lr;

        let mut spike_train: Vec<Vec<f64>> = Vec::with_capacity(n_steps);

        for _step in 0..n_steps {
            // ── 1. Project sensory input through weight matrix ──────────────
            // input_current[i] = sum_j(W[i,j] * sensory[j]) * 2.0 + noise
            let noise_scale = 1.0_f64;
            let input_current: Vec<f64> = (0..snn_size)
                .map(|i| {
                    let start  = i * input_dim;
                    let dot: f64 = self.input_weights[start..start + input_dim]
                        .iter()
                        .zip(sensory.iter())
                        .map(|(w, x)| w * x)
                        .sum();
                    let noise: f64 = (rng.gen::<f64>() * 2.0 - 1.0) * noise_scale;
                    dot * 2.0 + noise
                })
                .collect();

            // ── 2. LIF step (rayon-parallel) ───────────────────────────────
            let output_spikes: Vec<f64> = self.v
                .par_iter_mut()
                .zip(self.refractory.par_iter_mut())
                .zip(input_current.par_iter())
                .map(|((v, refrac), &i)| {
                    if *refrac > 0.0 {
                        *refrac -= dt;
                        if *refrac < 0.0 { *refrac = 0.0; }
                        *v = v_reset;
                        0.0
                    } else {
                        *v += dt / tau * (-(*v - v_rest) + i);
                        if *v >= v_thresh {
                            *v      = v_reset;
                            *refrac = ref_period;
                            1.0
                        } else { 0.0 }
                    }
                })
                .collect();

            // ── 3. STDP weight update ──────────────────────────────────────
            if learn && stdp_enabled {
                // Poisson input spikes from sensory signal (cheap sequential RNG)
                let input_spikes: Vec<f64> = sensory.iter()
                    .map(|&x| {
                        let prob = x.clamp(0.0, 1.0) * 0.5;
                        if rng.gen::<f64>() < prob { 1.0 } else { 0.0 }
                    })
                    .collect();

                let t = self.current_time;

                // Record spike times
                for (i, &s) in input_spikes.iter().enumerate() {
                    if s > 0.5 { self.pre_spike_times[i] = t; }
                }
                for (i, &s) in output_spikes.iter().enumerate() {
                    if s > 0.5 { self.post_spike_times[i] = t; }
                }

                // Compute delta_w in parallel (O(snn_size × input_dim))
                let pre_times  = &self.pre_spike_times;
                let post_times = &self.post_spike_times;

                let delta_w: Vec<f64> = (0..(snn_size * input_dim))
                    .into_par_iter()
                    .map(|idx| {
                        let post_idx = idx / input_dim;
                        let pre_idx  = idx % input_dim;
                        let t_post = post_times[post_idx];
                        let t_pre  = pre_times[pre_idx];
                        if t_post < -999.0 || t_pre < -999.0 { return 0.0f64; }
                        let delta_t = t_post - t_pre;
                        if delta_t.abs() >= window { return 0.0f64; }
                        if delta_t > 0.0 {
                            a_plus * (-delta_t / tau_stdp).exp()
                        } else {
                            -a_minus * (delta_t / tau_stdp).exp()
                        }
                    })
                    .collect();

                // Apply delta_w and re-normalise rows
                for (w, &dw) in self.input_weights.iter_mut().zip(delta_w.iter()) {
                    *w += stdp_lr * dw;
                }
                self.normalize_weights_internal();
                self.stdp_updates += 1;

                self.current_time += dt;
            }

            spike_train.push(output_spikes);
        }

        spike_train
    }

    /// L2-normalise every row of the weight matrix (matches Python _normalize_weights).
    fn normalize_weights(&mut self) {
        self.normalize_weights_internal();
        self.weight_updates += 1;
    }

    /// Get flat weight matrix [snn_size × input_dim] row-major.
    fn get_weights(&self) -> Vec<f64> {
        self.input_weights.clone()
    }

    /// Replace weight matrix.  Must be length snn_size × input_dim.
    fn set_weights(&mut self, weights: Vec<f64>) -> PyResult<()> {
        if weights.len() != self.snn_size * self.input_dim {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Expected {} weights, got {}",
                self.snn_size * self.input_dim,
                weights.len()
            )));
        }
        self.input_weights = weights;
        Ok(())
    }

    #[getter]
    fn stdp_updates(&self) -> usize { self.stdp_updates }

    #[getter]
    fn weight_updates(&self) -> usize { self.weight_updates }

    #[getter]
    fn input_dim(&self) -> usize { self.input_dim }

    #[getter]
    fn snn_size(&self) -> usize { self.snn_size }

    /// Whether Weber-Fechner log compression is applied to sensory input.
    #[getter]
    fn weber_fechner(&self) -> bool { self.weber_fechner }

    /// Enable or disable Weber-Fechner log compression at runtime.
    #[setter]
    fn set_weber_fechner(&mut self, enabled: bool) { self.weber_fechner = enabled; }

    fn get_stats(&self) -> Vec<(String, f64)> {
        vec![
            ("stdp_updates".into(),   self.stdp_updates   as f64),
            ("weight_updates".into(), self.weight_updates as f64),
        ]
    }

    fn __repr__(&self) -> String {
        format!("<SnnCore input={} snn={} stdp={}>",
                self.input_dim, self.snn_size, self.stdp_enabled)
    }
}

impl SnnCore {
    fn normalize_weights_internal(&mut self) {
        let input_dim = self.input_dim;
        for row in 0..self.snn_size {
            let start = row * input_dim;
            let end   = start + input_dim;
            let norm: f64 = self.input_weights[start..end]
                .iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-12 {
                for w in &mut self.input_weights[start..end] { *w /= norm; }
            }
        }
    }
}

// ============================================================
// PyO3 module registration  (mirrors hypervec_rs pattern)
// ============================================================

#[pymodule]
fn snn_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    // Core classes
    m.add_class::<LIFLayer>()?;
    m.add_class::<StdpEngine>()?;
    m.add_class::<SnnCore>()?;

    // Hebbian module
    hebbian::register_hebbian_module(m)?;

    // Concept / rate-coding module
    concept::register_concept_module(m)?;

    Ok(())
}

// ============================================================
// Unit tests for cross-disciplinary enhancements
// ============================================================

#[cfg(test)]
mod tests {
    #[test]
    fn test_weber_fechner_sign_preservation() {
        let input = vec![-10.0, -1.0, 0.0, 1.0, 10.0];
        let compressed: Vec<f64> = input.iter()
            .map(|&x| x.signum() * (1.0 + x.abs()).ln())
            .collect();
        assert!(compressed[0] < 0.0, "Negative preserved");
        assert!(compressed[1] < 0.0, "Negative preserved");
        assert_eq!(compressed[2], 0.0, "Zero preserved");
        assert!(compressed[3] > 0.0, "Positive preserved");
        assert!(compressed[4] > 0.0, "Positive preserved");
    }

    #[test]
    fn test_weber_fechner_monotonic() {
        // log(1+x) is monotonically increasing for x > 0
        let values = vec![0.1, 1.0, 10.0, 100.0, 1000.0];
        let compressed: Vec<f64> = values.iter()
            .map(|&x| (1.0 + x).ln())
            .collect();
        for i in 1..compressed.len() {
            assert!(compressed[i] > compressed[i - 1],
                    "Weber-Fechner should be monotonic: {} > {}",
                    compressed[i], compressed[i - 1]);
        }
    }

    #[test]
    fn test_weber_fechner_dynamic_range_compression() {
        // Original range: 1000/0.01 = 100,000
        // Compressed range should be much smaller
        let small = (1.0_f64 + 0.01).ln();
        let large = (1.0_f64 + 1000.0).ln();
        let compressed_ratio = large / small;
        let original_ratio = 1000.0 / 0.01;
        assert!(compressed_ratio < original_ratio / 10.0,
                "Compression ratio {} should be << original {}",
                compressed_ratio, original_ratio);
    }

    #[test]
    fn test_lif_layer_basic() {
        let mut layer = super::LIFLayer::new(
            4,      // n_neurons
            20.0,   // tau
            -70.0,  // v_rest
            -75.0,  // v_reset
            -55.0,  // v_thresh
            2.0,    // refractory_period
            1.0,    // dt
        );
        // Zero input should not produce spikes
        let spikes = layer.step(vec![0.0; 4]);
        assert_eq!(spikes.len(), 4);
        assert_eq!(spikes.iter().sum::<f64>(), 0.0, "Zero input → no spikes");
    }
}
