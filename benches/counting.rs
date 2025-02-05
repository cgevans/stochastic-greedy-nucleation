use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array2;
use stochastic_greedy_model::{calculate_n, calculate_n_add};

fn counting_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("counting");
    
    // Create test arrays of different sizes
    let sizes = [10, 50, 100, 500];
    
    for size in sizes {
        let array = Array2::from_shape_fn((size, size), |_| rand::random::<bool>());
        
        group.bench_function(format!("calculate_n_{size}x{size}"), |b| {
            b.iter(|| calculate_n(black_box(array.view())))
        });
        
        group.bench_function(format!("calculate_n_add_{size}x{size}"), |b| {
            b.iter(|| calculate_n_add(black_box(array.view())))
        });
    }
    
    group.finish();
}

criterion_group!(benches, counting_benchmark);
criterion_main!(benches); 