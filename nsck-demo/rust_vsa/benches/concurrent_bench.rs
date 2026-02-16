/// Performance benchmarks for concurrent cognitive operations

use criterion::{black_box, criterion_group, criterion_main, Criterion};
extern crate hypervec_rs;
use hypervec_rs::HyperVector;

fn bench_hypervector_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("hypervector_ops");
    
    // Benchmark HyperVector creation
    group.bench_function("create_hv", |b| {
        b.iter(|| {
            let hv = HyperVector::new(black_box(Some(42)));
            black_box(hv);
        });
    });
    
    // Benchmark XOR
    let hv1 = HyperVector::new(Some(1));
    let hv2 = HyperVector::new(Some(2));
    group.bench_function("xor", |b| {
        b.iter(|| {
            let result = hv1.xor(black_box(&hv2));
            black_box(result);
        });
    });
    
    group.finish();
}

criterion_group!(benches, bench_hypervector_operations);
criterion_main!(benches);
