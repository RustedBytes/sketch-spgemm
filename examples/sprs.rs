use sketch_spgemm::interop::sprs::auto_spgemm;
use sketch_spgemm::{AutoSpGemmConfig, SpGemmError};
use sprs::CsMatI;

fn main() -> Result<(), SpGemmError> {
    // The adapter borrows these CSR buffers directly. u16 demonstrates that
    // sprs index types are preserved in the returned matrix.
    let left = CsMatI::<i64, u16>::new((2, 3), vec![0, 2, 3], vec![0, 2, 1], vec![2, 3, 4]);
    let right =
        CsMatI::<i64, u16>::new((3, 2), vec![0, 1, 2, 4], vec![0, 1, 0, 1], vec![5, 6, 7, 8]);

    let (product, stats) = auto_spgemm(left.view(), right.view(), AutoSpGemmConfig::default())?;

    assert_eq!(product.get(0, 0), Some(&31));
    assert_eq!(product.get(0, 1), Some(&24));
    assert_eq!(product.get(1, 1), Some(&24));

    println!("selected path: {:?}", stats.choice);
    println!("sprs product: {product:?}");
    Ok(())
}
