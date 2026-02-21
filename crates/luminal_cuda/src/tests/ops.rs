use proptest::prelude::*;

use super::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(5))]

    #[test]
    fn test_add(x in 1usize..100, y in 1usize..5) {
        test_binary(x, x, |a, b| a + b, |a, b| (&a + &b).unwrap());
        test_binary((y, x), (y, x), |a, b| a + b, |a, b| (&a + &b).unwrap());
    }

    #[test]
    fn test_mul(x in 1usize..100, y in 1usize..5) {
        test_binary(x, x, |a, b| a * b, |a, b| (&a * &b).unwrap());
        test_binary((y, x), (y, x), |a, b| a * b, |a, b| (&a * &b).unwrap());
    }

    #[test]
    fn test_max(rows in 1usize..8, cols in 1usize..8) {
        test_unary((rows, cols), |a| a.max(1), |a| a.max(1).unwrap());
    }

    #[test]
    fn test_mean(rows in 1usize..8, cols in 1usize..8) {
        test_unary((rows, cols), |a| a.mean(1), |a| a.mean(1).unwrap());
    }

    #[test]
    fn test_matmul(m in 1usize..128, n in 1usize..128, k in 1usize..128) {
        // a_shape: (m, k), b_shape: (n, k) - b gets transposed to (k, n) with k-contiguous strides
        test_binary(
            (m, k),
            (n, k),
            |a, b| a.matmul(b.t()),
            |a, b| a.matmul(&b.t().unwrap()).unwrap(),
        );
    }

    // Unary ops tests
    #[test]
    fn test_exp2(x in 1usize..100, y in 1usize..5) {
        // exp2(x) = 2^x, verified by computing 2^x using exp(x * ln(2))
        test_unary(x, |a| a.exp2(), |a| (a * 2.0f64.ln()).unwrap().exp().unwrap());
        test_unary((y, x), |a| a.exp2(), |a| (a * 2.0f64.ln()).unwrap().exp().unwrap());
    }

    #[test]
    fn test_log2(x in 1usize..100, y in 1usize..5) {
        // log2(x) = ln(x) / ln(2)
        test_unary_positive(x, |a| a.log2(), |a| (a.log().unwrap() / 2.0f64.ln()).unwrap());
        test_unary_positive((y, x), |a| a.log2(), |a| (a.log().unwrap() / 2.0f64.ln()).unwrap());
    }

    #[test]
    fn test_sin(x in 1usize..100, y in 1usize..5) {
        test_unary(x, |a| a.sin(), |a| a.sin().unwrap());
        test_unary((y, x), |a| a.sin(), |a| a.sin().unwrap());
    }

    #[test]
    fn test_recip(x in 1usize..100, y in 1usize..5) {
        test_unary_nonzero(x, |a| a.reciprocal(), |a| a.recip().unwrap());
        test_unary_nonzero((y, x), |a| a.reciprocal(), |a| a.recip().unwrap());
    }

    #[test]
    fn test_sqrt(x in 1usize..100, y in 1usize..5) {
        test_unary_positive(x, |a| a.sqrt(), |a| a.sqrt().unwrap());
        test_unary_positive((y, x), |a| a.sqrt(), |a| a.sqrt().unwrap());
    }

    // Binary ops tests
    #[test]
    fn test_mod_op(size in 1usize..100, rows in 1usize..5) {
        test_mod(size, size, |a, b| a % b);
        test_mod((rows, size), (rows, size), |a, b| a % b);
    }

    #[test]
    fn test_less_than(x in 1usize..100, y in 1usize..5) {
        test_binary(x, x, |a, b| a.lt(b), |a, b| a.lt(&b).unwrap().to_dtype(candle_core::DType::F32).unwrap());
        test_binary((y, x), (y, x), |a, b| a.lt(b), |a, b| a.lt(&b).unwrap().to_dtype(candle_core::DType::F32).unwrap());
    }

    // --- Additional robust tests ---

    // Softmax: verify output sums to ~1 and matches candle reference
    #[test]
    fn test_softmax(rows in 1usize..8, cols in 2usize..32) {
        test_unary((rows, cols), |a| a.softmax(1), |a| {
            // Manual softmax: exp(x - max) / sum(exp(x - max))
            let max_val = a.max(1).unwrap();
            let shifted = a.broadcast_sub(&max_val.unsqueeze(1).unwrap()).unwrap();
            let exps = shifted.exp().unwrap();
            let sum_exps = exps.sum(1).unwrap();
            exps.broadcast_div(&sum_exps.unsqueeze(1).unwrap()).unwrap()
        });
    }

    // Layer norm (std_norm): normalize so std ~ 1
    #[test]
    fn test_layer_norm(rows in 1usize..8, cols in 2usize..32) {
        test_unary((rows, cols), |a| a.layer_norm(1, 1e-5), |a| {
            let mean = a.mean_keepdim(1).unwrap();
            let centered = a.broadcast_sub(&mean).unwrap();
            let var = centered.sqr().unwrap().mean_keepdim(1).unwrap();
            let std = (var + 1e-5f64).unwrap().sqrt().unwrap();
            centered.broadcast_div(&std).unwrap()
        });
    }

    // ReLU activation
    #[test]
    fn test_relu(x in 1usize..100, y in 1usize..5) {
        test_unary(x, |a| a.relu(), |a| a.relu().unwrap());
        test_unary((y, x), |a| a.relu(), |a| a.relu().unwrap());
    }

    // Sum reduction along axis 0 and 1
    #[test]
    fn test_sum_reduce(rows in 1usize..8, cols in 1usize..32) {
        test_unary((rows, cols), |a| a.sum(0), |a| a.sum(0).unwrap());
        test_unary((rows, cols), |a| a.sum(1), |a| a.sum(1).unwrap());
    }

    // Max reduction along axis 0
    #[test]
    fn test_max_reduce_axis0(rows in 2usize..8, cols in 1usize..32) {
        test_unary((rows, cols), |a| a.max(0), |a| a.max(0).unwrap());
    }

    // 3D tensor operations: test matmul with batch-like leading dimensions
    #[test]
    fn test_3d_sum(batch in 1usize..4, rows in 1usize..8, cols in 1usize..16) {
        test_unary((batch, rows, cols), |a| a.sum(2), |a| a.sum(2).unwrap());
    }

    #[test]
    fn test_3d_max(batch in 1usize..4, rows in 1usize..8, cols in 1usize..16) {
        test_unary((batch, rows, cols), |a| a.max(2), |a| a.max(2).unwrap());
    }

    #[test]
    fn test_3d_mean(batch in 1usize..4, rows in 1usize..8, cols in 1usize..16) {
        test_unary((batch, rows, cols), |a| a.mean(2), |a| a.mean(2).unwrap());
    }

    // Chained unary: exp2 then log2 should be ~identity (round-trip)
    #[test]
    fn test_exp2_log2_roundtrip(x in 1usize..50, y in 1usize..4) {
        test_unary_positive(
            (y, x),
            |a| a.log2().exp2(),
            |a| {
                let logged = (a.log().unwrap() / 2.0f64.ln()).unwrap();
                (logged * 2.0f64.ln()).unwrap().exp().unwrap()
            },
        );
    }

    // Chained binary: (a + b) * (a - b) = a^2 - b^2
    #[test]
    fn test_add_mul_chain(x in 1usize..50, y in 1usize..4) {
        test_binary(
            (y, x),
            (y, x),
            |a, b| (a + b) * (a - b),
            |a, b| {
                let sum = (&a + &b).unwrap();
                let diff = (&a - &b).unwrap();
                (&sum * &diff).unwrap()
            },
        );
    }

    // Subtraction (a - b)
    #[test]
    fn test_sub(x in 1usize..100, y in 1usize..5) {
        test_binary(x, x, |a, b| a - b, |a, b| (&a - &b).unwrap());
        test_binary((y, x), (y, x), |a, b| a - b, |a, b| (&a - &b).unwrap());
    }

    // Scalar multiply: tensor * constant
    #[test]
    fn test_scalar_mul(x in 1usize..100, y in 1usize..5) {
        test_unary(x, |a| a * 2.5, |a| (a * 2.5f64).unwrap());
        test_unary((y, x), |a| a * 0.1, |a| (a * 0.1f64).unwrap());
    }

    // Scalar add: tensor + constant
    #[test]
    fn test_scalar_add(x in 1usize..100, y in 1usize..5) {
        test_unary(x, |a| a + 1.0, |a| (a + 1.0f64).unwrap());
        test_unary((y, x), |a| a + 3.14, |a| (a + 3.14f64).unwrap());
    }

    // Transpose + matmul: ensures stride permutations work
    #[test]
    fn test_matmul_transposed_lhs(m in 1usize..32, n in 1usize..32, k in 1usize..32) {
        test_binary(
            (k, m),
            (n, k),
            |a, b| a.t().matmul(b.t()),
            |a, b| a.t().unwrap().matmul(&b.t().unwrap()).unwrap(),
        );
    }
}
