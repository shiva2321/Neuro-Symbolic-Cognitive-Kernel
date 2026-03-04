use pyo3::prelude::*;

pub mod living_hv_store;
pub mod spectral_rg;
pub mod societal_hnsw;
pub mod percolation;
pub mod tda_ripser;

#[pymodule]
fn societal_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<living_hv_store::LivingHVStore>()?;
    m.add_class::<spectral_rg::SpectralRG>()?;
    m.add_class::<societal_hnsw::SocietalHNSW>()?;
    m.add_class::<percolation::PercolationDetector>()?;
    m.add_class::<tda_ripser::TDARipser>()?;
    Ok(())
}
