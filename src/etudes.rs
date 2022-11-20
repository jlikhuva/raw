//! ## Rust Algorithmic Etudes
//!
//! A collection of small, nifty algorithmic procedures

pub mod shuffling {
    use itertools::Itertools;
    use rand::{self, thread_rng, Rng};

    /// Shuffle the given data slice by reducing the `shuffling` problem
    /// to a `sorting` problem. The reduction is  done in `O(n)` while
    /// the sorting is done in `O(n lg n)`.
    pub fn sorting_shuffler<T>(items: &mut [T]) -> () {
        let mut tagged_items: Vec<(usize, i64)> = Vec::with_capacity(items.len());
        for (idx, _) in items.iter().enumerate() {
            let rand_int = rand::random::<i64>();
            tagged_items.push((idx, rand_int));
        }
        let shuffled_indexes: Vec<usize> = tagged_items
            .iter()
            .enumerate()
            .sorted_by_key(|(_, &x)| x.1)
            .map(|(_, &x)| x.0)
            .collect();
        for (new_index, &old_index) in shuffled_indexes.iter().enumerate() {
            items.swap(new_index, old_index);
        }
    }

    /// Shuffle the given data slice in a single pass. This procedure runs in `O(n)`
    /// if we assume that the procedure to generate a random number runs in constant
    /// time.
    pub fn fisher_yates_shuffle<T>(items: &mut [T]) -> () {
        for shuffled_portion_end in 0..=items.len() - 2 {
            let random_index = thread_rng().gen_range(shuffled_portion_end + 1..items.len());
            items.swap(random_index, shuffled_portion_end);
        }
    }
}

pub mod discrete_optimization {}

pub mod streaming_algorithms_traits {
    /// A cardinality estimator will be any object that is able to observe a possibly
    /// infinite stream of items and, at any point, estimate the number of unique items
    /// seen so far.
    pub trait CardinalityEstimator<T> {
        /// We observe each item as it comes in.
        fn observe(&mut self, item: &T);

        /// Return an estimation of the number of unique items see thus far
        fn current_estimate(&self) -> u64;
    }

    /// A sampler will be anything that can observe a possibly infinite
    /// number of items and produce a finite random sample from that
    /// stream
    pub trait Sampler<T> {
        /// We observe each item as it comes in
        fn observe(&mut self, item: T);

        /// Produce a random uniform sample of the all the items that
        /// have been observed so far.
        fn sample(&self) -> &[T];
    }

    /// Indicates that the filter has probably seen a given
    /// item before
    pub struct ProbablyYes;

    /// Indicates that a filter hasn't seen a given item before.
    pub struct DefinitelyNot;

    /// A filter will be any object that is able to observe a possibly
    /// infinite stream of items and, at any point, answer if a given
    /// item has been seen before
    pub trait Filter<T> {
        /// We observe each item as it comes in.
        fn observe(&mut self, item: &T);

        /// Tells us whether we've seen the given item before. This method
        /// can produce false positives. That is why instead of returning a
        /// boolean, it returns `Result<ProbablyYes, DefinitelyNot>`
        fn has_been_observed_before(&self, item: &T) -> Result<ProbablyYes, DefinitelyNot>;
    }

    pub trait StatisticsTracker<T> {
        /// We observe each item as it comes in.
        fn observe(&mut self, item: &T);

        /// Returns the current median element
        fn current_median(&self) -> &T;
    }
}

pub mod naive_estimator {
    use super::streaming_algorithms_traits::CardinalityEstimator;
    use std::collections::hash_map::DefaultHasher;
    use std::{hash::Hash, hash::Hasher, marker::PhantomData};

    /// A naïve estimator that estimates the cardinality
    /// to simply be 1 divided by the smallest hash value
    /// observed thus far
    ///  ## Example
    /// 
    /// ```rust
    /// pub struct Car;
    /// 
    /// let mut estimator = NaiveEstimator<Car>::default();
    /// 
    /// let estimate = estimator.current_estimate();
    /// ```
    pub struct NaiveEstimator<T> {
        // The smallest hash value seen so far.
        cur_hash_value_min: Option<u64>,
        /// We want our estimator to be parameterized by the type
        /// it is estimating, but we don't need to store anything
        /// so we never use the provided type. To keep the compiler
        /// happy, we sacrifice a single ghost as an offering
        _marker: PhantomData<T>,
    }

    impl<T> Default for NaiveEstimator<T> {
        fn default() -> Self {
            Self {
                cur_hash_value_min: Option::default(),
                _marker: PhantomData::default(),
            }
        }
    }

    impl<T: Hash> CardinalityEstimator<T> for NaiveEstimator<T> {
        fn observe(&mut self, item: &T) {
            let mut hasher = DefaultHasher::new();
            item.hash(&mut hasher);
            let hash_value = hasher.finish();

            match self.cur_hash_value_min {
                None => self.cur_hash_value_min = Some(hash_value),
                Some(cur_min) if cur_min > hash_value => self.cur_hash_value_min = Some(hash_value),
                Some(_) => {}
            }
        }

        fn current_estimate(&self) -> u64 {
            match self.cur_hash_value_min {
                None => 0,
                Some(cur_min) => 1 / (cur_min / u64::MAX),
            }
        }
    }
}

pub mod probabilistic_counting {
    use super::streaming_algorithms_traits::CardinalityEstimator;
    use std::collections::hash_map::DefaultHasher;
    use std::{hash::Hash, hash::Hasher, marker::PhantomData};

    /// Estimate the number of distinct items in the stream as
    /// `2^k` where `k` is the longest trail of leading zeroes
    /// among the hash values of all the items observed so far
    pub struct MaxTrailEstimator<T> {
        // The smallest hash value seen so far.
        cur_max_trail_length: Option<u64>,
        /// We want our estimator to be parameterized by the type
        /// it is estimating, but we don't need to store anything
        /// so we never use the provided type. To keep the compiler
        /// happy, we sacrifice a single ghost as an offering to the
        /// compiler gods
        _marker: PhantomData<T>,
    }

    impl<T> CardinalityEstimator<T> for MaxTrailEstimator<T> {
        fn observe(&mut self, item: &T) {
            todo!()
        }

        fn current_estimate(&self) -> u64 {
            // match self.cur_max_trail_length {
            //     None => 0,
            //     Some(trail_length) =>
            // }
            todo!()
        }
    }


}
#[cfg(test)]
mod test_streaming {
    use rand::distributions::{Distribution, Uniform};
    use rand::random;

    use super::shuffling;
    /// Generate data for testing streaming algorithms
    ///
    /// Generates a random collection of randomly replicated numbers
    fn generate_random_data(n: u64) -> Vec<i64> {
        let mut data = Vec::new();
        for _ in 0..n {
            data.push(random())
        }

        // For each value, throw a fair coin and if it turns up heads
        // replicate that value a random number of times
        let mut rng = rand::thread_rng();
        let fair_coin = Uniform::from(1..=2);
        let replicates = Uniform::from(10..=1000);

        for idx in 0..n {
            if fair_coin.sample(&mut rng) == 1 {
                data.extend(replicate(data[idx as usize], replicates.sample(&mut rng)).iter());
            }
        }
        shuffling::fisher_yates_shuffle(&mut data);
        return data;
    }

    fn replicate(val: i64, times: u32) -> Vec<i64> {
        let mut replicated_values = Vec::with_capacity(times as usize);
        for _ in 0..times {
            replicated_values.push(val);
        }
        return replicated_values;
    }

    #[test]
    fn test_generate_random_data() -> () {
        let data = generate_random_data(10000);
        // println!("{:#?}", data);
        println!("{:#?}", data.len())
    }
}
