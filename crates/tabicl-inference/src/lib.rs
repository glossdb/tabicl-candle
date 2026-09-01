//! TabICL's required input preprocessing and inference wrappers. The
//! checkpoint only performs on inputs normalized exactly as the tabicl
//! Python wrapper normalizes them, and TabICL is an in-context learner —
//! "fit" happens at serving time on every request. Fit statistics are
//! computed host-side in f64 with numpy/sklearn/scipy semantics; the
//! model sees the f32 cast.

pub use candle_core::Device; // consumers pick a device without a candle dep

pub mod classifier; // the classifier wrapper, pinned to one member
pub mod ensemble; // multi-member regressor: shuffles x norms, mean bands
pub mod power; // Yeo-Johnson power stage (sklearn PowerTransformer)
pub mod regressor; // the sklearn-wrapper mirror + the norm pipelines
pub mod unsupervised; // chain-rule density read over the wrapper
