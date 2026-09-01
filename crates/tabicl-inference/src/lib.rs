//! TabICL's required input preprocessing and inference wrappers. The
//! checkpoint only performs on inputs normalized exactly as the tabicl
//! Python wrapper normalizes them, and TabICL is an in-context learner —
//! "fit" happens at serving time on every request. Fit statistics are
//! computed host-side in f64 with numpy/sklearn/scipy semantics; the
//! model sees the f32 cast.
//!
//! ## Output stability classes
//!
//! Continuous outputs (bands, probabilities, scores) are
//! environment-dependent at ~1e-4 (backend, SIMD dispatch, libm):
//! compare with tolerance, never bit-equality. Discrete outputs
//! (labels, rankings, coverage indicators) are stable within a pinned
//! environment but can flip near thresholds across environments — do
//! not persist, dedupe, or cross-compare them expecting equality
//! unless the deployment pins one backend and platform. Row NLL over
//! near-deterministic conditionals is ordinal, not cardinal.

pub use candle_core::Device; // consumers pick a device without a candle dep

pub mod classifier; // the classifier wrapper, pinned to one member
pub mod ensemble; // multi-member regressor: shuffles x norms, mean bands
pub mod power; // Yeo-Johnson power stage (sklearn PowerTransformer)
pub mod regressor; // the sklearn-wrapper mirror + the norm pipelines
pub mod unsupervised; // chain-rule density read over the wrapper
