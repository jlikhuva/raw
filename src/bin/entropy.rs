//! A new perspective on Entropy

fn main() {}

/// A probability distribution
pub trait ProbabilityDistribution<const N: usize> {
    fn entropy(&self) -> f32;
}
