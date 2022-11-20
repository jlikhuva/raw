mod parsing {
    //! In this exercise, you're going to decompress a compressed string.
    //! Your input is a compressed string of the format number[string]
    //! and the decompressed output form should be the string written number times.
    //! For example: The input
    //!    3[abc]4[ab]c
    //! Would be output as
    //!    abcabcabcababababc
    //!
    //! Other rules
    //!    Number can have more than one digit. For example, 10[a] is allowed, and just means aaaaaaaaaa
    //!    One repetition can occur inside another. For example, 2[3[a]b] decompresses into aaabaaab
    //!    Characters allowed as input include digits, small English letters and brackets [ ].
    //!
    //!    Digits are only to represent amount of repetitions.
    //!    Letters are just letters.
    //!    Brackets are only part of syntax of writing repeated substring.
    //!
    //! Input is always valid, so no need to check its validity.
    fn parse_string(s: &str) -> String {
        let mut flattener = Flattener::of(s);
        flattener.flatten();
        flattener.cur_res
    }

    /// Flattens a string written in the problem's syntax
    /// We'll use this on both the global string and the
    /// block substrings
    #[derive(Debug)]
    struct Flattener<'a> {
        underlying: &'a str,
        cur_pos: usize,
        cur_res: String,
    }

    impl<'a> Flattener<'a> {
        /// Creates a new flattener of the given string
        pub fn of(s: &'a str) -> Self {
            Flattener {
                underlying: s,
                cur_pos: 0,
                cur_res: String::new(),
            }
        }

        /// To flatten a string, we walk along it and
        ///    a. Anytime we encounter a digit, we create a digit parser to parse the digit
        ///       and then immediately create a block parser to parse the block. We then add
        ///       the resulting string to current result and keep walking
        ///    b. If a character is not a digit, we simply add it to cur_res and keep walking
        pub fn flatten(&mut self) {
            while self.cur_pos < self.underlying.len() {
                let c = self.underlying[self.cur_pos..=self.cur_pos].chars().next().unwrap();
                if c.is_digit(10) {
                    let mut cur_digit_parser = DigitParser::new(self.underlying, self.cur_pos);
                    let repeats = cur_digit_parser.parse_digit();
                    let span = Self::calculate_block_span(self.underlying, cur_digit_parser.cur_pos);
                    let cur_block_parser: BlockParser = (self.underlying, span, repeats).into();
                    let parsed_block = cur_block_parser.parse_block();
                    self.cur_res.push_str(&parsed_block);
                    self.cur_pos = span.1 + 1;
                } else {
                    self.cur_res.push(c);
                    self.cur_pos += 1;
                }
            }
        }

        fn calculate_block_span(underlying: &str, mut cur_pos: usize) -> BlockSpan {
            let mut stack = Vec::with_capacity(underlying[cur_pos..].len());
            let start = cur_pos;
            let opening_brace = underlying[start..=start].chars().next().unwrap();
            stack.push(opening_brace);
            cur_pos += 1;
            while stack.len() > 0 {
                let c = underlying[cur_pos..=cur_pos].chars().next().unwrap();
                if c == ']' {
                    stack.pop();
                } else if c == '[' {
                    stack.push(c);
                }
                cur_pos += 1;
            }
            let span_end = cur_pos;
            (start + 1, span_end - 1)
        }
    }

    #[derive(Debug)]
    struct DigitParser<'a> {
        underlying: &'a str,
        cur_pos: usize,
    }

    impl<'a> DigitParser<'a> {
        pub fn new(s: &'a str, pos: usize) -> Self {
            DigitParser {
                underlying: s,
                cur_pos: pos,
            }
        }

        /// Walks the underlying string from the current
        /// position until you find an opening bracket.
        /// Reinterpret the span of characters as a digit and return
        pub fn parse_digit(&mut self) -> usize {
            let digit_start = self.cur_pos;
            for c in self.underlying[digit_start..].chars() {
                if c == '[' {
                    break;
                }
                self.cur_pos += 1;
            }
            let digit_end = self.cur_pos;
            self.underlying[digit_start..digit_end].parse::<usize>().unwrap()
        }
    }

    #[derive(Debug)]
    struct BlockParser<'a> {
        underlying: &'a str,

        /// This span excludes the starting
        block_span: (usize, usize),

        /// The number of times to repeat the corresponding
        /// block
        n: usize,
    }
    type BlockSpan = (usize, usize);

    impl<'a> From<(&'a str, BlockSpan, usize)> for BlockParser<'a> {
        fn from((s, span, n): (&'a str, BlockSpan, usize)) -> Self {
            BlockParser {
                underlying: s,
                block_span: span,
                n,
            }
        }
    }

    impl<'a> BlockParser<'a> {
        /// To parse a block, we first flatten the substring in the provided span
        /// then we repeat that flattened string `n` times
        pub fn parse_block(&self) -> String {
            let (start, end) = self.block_span;
            let block = &self.underlying[start..end];
            let mut block_flattener = Flattener::of(block);
            block_flattener.flatten();
            str::repeat(&block_flattener.cur_res, self.n)
        }
    }

    #[test]
    fn test_parser() {
        let x = "abcde";
        let x_parsed = parse_string(x);
        assert_eq!("abcde", x_parsed);

        let a = "3[abc]4[ab]c";
        let a_parsed = parse_string(a);
        assert_eq!("abcabcabcababababc", a_parsed);

        let b = "2[3[a]b]";
        let b_parsed = parse_string(b);
        assert_eq!("aaabaaab", b_parsed);

        let c = "10[a]a";
        let c_parsed = parse_string(c);
        assert_eq!("aaaaaaaaaaa", c_parsed)
    }
}

mod graph {
    use std::collections::HashMap;

    /// A graph represented as an adjacency list. This is an directed
    /// unweighted graph
    #[derive(Debug)]
    pub struct DirectedUnweighted<T>(HashMap<T, Vec<T>>);

    impl<T> DirectedUnweighted<T> {
        pub fn with_capacity(capacity: usize) -> Self {
            todo!()
        }

        pub fn add_node(&mut self, node_id: T) {
            todo!()
        }

        pub fn add_edge(&mut self, from: T, to: T) {
            todo!()
        }

        pub fn bfs(&self, src: T) -> HashMap<T, f32> {
            todo!()
        }
    }

    #[derive(Debug)]
    struct Edge<T> {
        sink: T,
        weight: f32,
    }
    /// A graph represented as an adjacency list. This is an directed
    /// weighted graph
    #[derive(Debug)]
    pub struct DirectedWeighted<T>(HashMap<T, Vec<Edge<T>>>);
}

mod secret_word {
    use std::collections::HashMap;

    struct Master;

    struct Solution;

    struct HammingGraph {
        /// An adjacency list for a weighted undirected graph
        list: HashMap<String, HashMap<usize, Vec<String>>>,
    }

    impl HammingGraph {
        /// Creates a new Hamming Graph from the given list of words
        pub fn new(words: &[String]) -> Self {
            let mut list = HashMap::with_capacity(words.len());
            for (idx, word) in words.iter().enumerate() {
                for other_word in &words[idx + 1..] {
                    let similarity = Self::hamming_similarity(&word, &other_word);
                    Self::add_to_graph(&word, &other_word, similarity);
                }
            }
            HammingGraph { list }
        }

        /// The hamming distance between two sequences is simply the count
        /// of the number of positions where the two sequences match.
        fn hamming_similarity(a: &str, b: &str) -> usize {
            a.chars().zip(b.chars()).filter(|(ac, bc)| ac == bc).count()
        }

        /// Creates an edge from a -> b and from b -> a. Both edges
        /// have a weight of `similarity`
        fn add_to_graph(a: &str, b: &str, similarity: usize) {}
    }

    impl Solution {
        /// Here's the scheme that we'll adopt:
        /// - First, we'll create a dense undirected, weighted graph. In the graph,
        ///   the weight between any two nodes is the hamming similarity between them
        ///
        /// - To answer the query, we pick an arbitrary word `w` and call the Masters.guess(w)
        ///
        /// - This will tell us the hamming similarity between `w` and the correct word.
        ///
        /// -  We can then only focus on the neighbors with that similarity
        pub fn find_secret_word(words: Vec<String>, master: &Master) {
            todo!()
        }
    }
}

mod majority_element {
    use std::collections::{BinaryHeap, HashMap};

    #[derive(PartialEq, Eq, PartialOrd, Ord)]
    struct CountTuple {
        frequency: usize,
        number: i32,
    }

    pub fn majority_element(nums: Vec<i32>) -> Vec<i32> {
        let mut majority = Vec::with_capacity(3);
        let mut counter = HashMap::with_capacity(nums.len());
        let mut cur_top_k = BinaryHeap::new();
        let threshold = nums.len() / 3;

        // Count up how many times each items occurs
        for num in nums {
            let entry = counter.entry(num).or_insert(0_usize);
            *entry += 1;
        }
        // Place the <freq, number> pairs into the heap
        for (number, frequency) in counter {
            let heap_entry = CountTuple { number, frequency };
            cur_top_k.push(heap_entry);
        }

        // Extract the top-3 items from the heap
        while majority.len() < 3 && !cur_top_k.is_empty() {
            let cur = cur_top_k.pop().unwrap();
            if cur.frequency <= threshold {
                break;
            }
            majority.push(cur.number);
        }

        majority
    }

    #[test]
    fn majority() {
        majority_element(vec![3, 2, 3]);
    }
}

mod short_routines {
    use std::collections::{HashMap, HashSet};

    /// Find a subset of the input set `s` s.t the sum of `s` is as
    /// large as possible but not larger than the target value
    pub fn subset_sum(set: &[i32], target: i32) -> i32 {
        let mut sum_vec = vec![0];
        for &new_item in set {
            let new_set = sum_vec.iter().map(|&old_item| old_item + new_item).collect::<Vec<_>>();
            sum_vec = merge(&sum_vec, &new_set)
                .into_iter()
                .filter(|&item| item <= target)
                .collect();
        }
        *sum_vec.last().unwrap()
    }

    #[test]
    fn test_subset_sum() {
        let inp = [1, 4, 5];
        assert_eq!(subset_sum(&inp, 7), 6);
    }

    pub fn square_root(x: usize) -> usize {
        todo!()
    }

    pub fn odd_even_jump(a: &[i32]) -> usize {
        let mut num_valid = 0;
        // For each possible starting index `i`, test if we can reach the
        // end of the array from `i`
        for (i, _) in a.iter().enumerate() {
            if can_reach_end_from(i, a) {
                num_valid += 1;
            }
        }
        num_valid
    }

    /// Tells us if we can reach the end of the input array `a`
    /// by making a series of odd-even jumps
    fn can_reach_end_from(i: usize, a: &[i32]) -> bool {
        if i == a.len() - 1 {
            return true;
        } else {
            todo!()
        }
    }
    impl Solution {
        pub fn three_sum(nums: Vec<i32>) -> Vec<Vec<i32>> {
            let mut three_sum_triples = Vec::new();
            for (idx, &fixed) in nums.iter().enumerate() {
                let two_sum_pairs = Self::all_pairs_with_sum(0 - fixed, &nums[idx + 1..]);
                for mut pair in two_sum_pairs {
                    pair.push(fixed);
                    three_sum_triples.push(pair);
                }
            }
            three_sum_triples
        }

        fn all_pairs_with_sum(target: i32, idx: &[i32]) -> Vec<Vec<i32>> {
            todo!()
        }

        pub fn k_closest(points: Vec<Vec<i32>>, k: i32) -> Vec<Vec<i32>> {
            let mut k_closest = Vec::with_capacity(k as usize);
            let euclid = |v: &Vec<i32>| ((v[0] * v[0] + v[1] * v[1]) as f32).sqrt();
            let mut points_dist: Vec<(f32, Vec<i32>)> = points.into_iter().map(|v| (euclid(&v), v)).collect();
            points_dist.sort_by(|a, b| a.0.total_cmp(&b.0));

            for i in 0..k as usize {
                k_closest.push(std::mem::take(&mut points_dist[i].1));
            }

            k_closest
        }

        pub fn is_valid(s: String) -> bool {
            let mut stack = Vec::with_capacity(s.len());
            for bracket in s.chars() {
                if bracket == '{' || bracket == '(' || bracket == '[' {
                    stack.push(bracket);
                } else if Self::open_and_close_match(bracket, &stack) {
                    stack.pop();
                } else {
                    return false;
                }
            }
            stack.len() == 0
        }

        fn open_and_close_match(closing: char, stack: &Vec<char>) -> bool {
            match (stack.last(), closing) {
                (Some(&c), '}') => c == '{',
                (Some(&c), ')') => c == '(',
                (Some(&c), ']') => c == '[',
                _ => false,
            }
        }
    }

    /// You have two baskets, and each basket can carry any quantity of fruit,
    /// but you want each basket to only carry one type of fruit each.
    ///
    /// You start at any tree of your choice, then repeatedly perform the following steps:
    ///
    ///     Add one piece of fruit from this tree to your baskets.  If you cannot, stop.
    ///     Move to the next tree to the right of the current tree.  If there is no tree to the right, stop.
    ///
    pub fn max_total_fruit(tree: &[i32]) -> i32 {
        let mut cur_max = 0;
        for (i, _) in tree.iter().enumerate() {
            let cur_count = calculate_max_possible_from(i, tree);
            if cur_count > cur_max {
                cur_max = cur_count;
            }
        }
        cur_max
    }

    fn calculate_max_possible_from(start_idx: usize, tree_types: &[i32]) -> i32 {
        let mut counter = HashMap::with_capacity(2);
        for i in start_idx..tree_types.len() {
            let cur_fruit_type = tree_types[i];
            if counter.len() == 2 && !counter.contains_key(&cur_fruit_type) {
                break;
            }
            let entry = counter.entry(cur_fruit_type).or_insert(0);
            *entry += 1;
        }
        counter.iter().fold(0, |cur, (k, &v)| cur + v)
    }

    #[test]
    fn test_fruits() {
        let input = [3, 3, 3, 1, 2, 1, 1, 2, 3, 3, 4];
        assert_eq!(5, max_total_fruit(&input))
    }

    fn median_by_merging(a: &[i32], b: &[i32]) -> f64 {
        let merged_array = merge(a, b);
        let mid = merged_array.len() / 2;
        // Calculate teh median
        if merged_array.len() & 1 == 0 {
            // We have an even number of items. in this case, the median
            // is the average of the middle two items
            (merged_array[mid] + merged_array[mid - 1]) as f64 / 2 as f64
        } else {
            merged_array[mid] as f64
        }
    }

    fn merge(a: &[i32], b: &[i32]) -> Vec<i32> {
        let mut merged = Vec::with_capacity(a.len() + b.len());
        let (mut a_idx, mut b_idx) = (0, 0);
        while a_idx < a.len() && b_idx < b.len() {
            if a[a_idx] < b[b_idx] {
                merged.push(a[a_idx]);
                a_idx += 1;
            } else {
                merged.push(b[b_idx]);
                b_idx += 1;
            }
        }
        merged.extend(&a[a_idx..]);
        merged.extend(&b[b_idx..]);
        merged
    }

    impl Solution {
        pub fn search_range(nums: Vec<i32>, target: i32) -> Vec<i32> {
            let mut res = Self::search_range_linear(&nums, target);
            if res[1] == -1 {
                res[1] = res[0];
            }
            vec![res[0], res[1]]
        }

        /// Given an array of integers nums sorted in ascending order,
        /// find the starting and ending position of a given target value.
        fn search_range_linear(nums: &[i32], target: i32) -> [i32; 2] {
            let mut res = [-1, -1];
            for (idx, &val) in nums.iter().enumerate() {
                if val == target {
                    if res[0] == -1 {
                        res[0] = idx as i32;
                    } else {
                        res[1] = idx as i32;
                    }
                }
            }
            res
        }

        fn search_range_logarithmic(nums: &[i32], target: i32) -> [i32; 2] {
            let first_idx = Self::bsearch_first(nums, target);
            let last_idx = Self::bsearch_last(nums, target);
            [first_idx, last_idx]
        }

        fn bsearch_first(nums: &[i32], target: i32) -> i32 {
            todo!()
        }

        fn bsearch_last(nums: &[i32], target: i32) -> i32 {
            todo!()
        }

        pub fn is_anagram(s: String, t: String) -> bool {
            let mut a: Vec<&u8> = s.as_bytes().iter().collect();
            let mut b: Vec<&u8> = t.as_bytes().iter().collect();
            a.sort();
            b.sort();
            a == b
        }

        pub fn merge(intervals: &mut [(i32, i32)]) -> Vec<(i32, i32)> {
            let merged = Vec::with_capacity(intervals.len());
            intervals.sort_by_key(|interval| interval.0);
            // Scan over sorted merging overlaps which should be clumped together
            merged
        }
    }

    /// Finds the sub array of array with the largest sum in linear time.
    ///
    /// As with all DP algorithms, we need by reasoning about the base cases.
    /// That is, we need to reason about the smallest possible input to
    /// the procedure and how we may solve the problem on such an input.
    /// More specifically, we need to consider inputs for which the solution
    /// to the problem is trivially defined.
    ///
    /// In this instance, the base case is a subarray with only one element.
    /// With only one element, the sum us simply that element's value (This is
    /// assuming that we are OK with negative values. If not, then the max is 0)
    ///
    /// We then think about how we can incrementally expand the size of this
    /// base case to get the answer for larger and larger sub-arrays
    /// until we've considered all possible sub arrays
    /// ```rust
    /// def max_subarray(numbers):
    ///     best_sum = 0  // or: float('-inf')
    ///     current_sum = 0
    ///     for x in numbers:
    ///         /// we can extend the current sub array or start a new one
    ///         current_sum = max(x, current_sum + x)
    ///         best_sum = max(best_sum, current_sum)
    ///     return best_sum
    ///```
    ///
    pub fn maximum_subarray_kadane(array: &[i32]) -> (usize, usize, i32) {
        let (mut best_sum, mut current_sum) = (i32::MIN, 0);
        let (mut best_start, mut best_end) = (0, 0);
        let mut current_start = 0;

        // Scan over the input array from L -> R. at each step, we calculate
        // the subarray with the largest sum ending at `current_end`
        for (current_end, &current_value) in array.iter().enumerate() {
            // At each iteration, we have two actions to choose from: we can
            // choose to extend the current best sum or we can start a new
            // sum. Because we do not know which is best, we try both and
            // pick the one with the greater value
            let extended_sum = current_sum + current_value;

            if extended_sum < current_value {
                current_sum = current_value;
                current_start = current_end;
            } else {
                current_sum = extended_sum;
            }

            // Make updates the the global max variables
            if current_sum > best_sum {
                best_sum = current_sum;
                best_start = current_start;
                best_end = current_end;
            }
        }
        (best_start, best_end, best_sum)
    }

    #[test]
    fn test_max_sub_arr() {
        let a = [-2, 1, -3, 4, -1, 2, 1, -5, 4];
        assert_eq!((3, 6, 6), maximum_subarray_kadane(&a))
    }

    pub fn backspace_compare(s: String, t: String) -> bool {
        let mut first = String::new();
        let mut second = String::new();
        // Simulate the typing of both strings
        populate(&mut first, &s);
        populate(&mut second, &t);
        return first == second;
    }

    fn populate(buffer: &mut String, s: &str) {
        for c in s.chars() {
            if c == '#' {
                buffer.pop();
            } else {
                buffer.push(c);
            }
        }
    }

    pub fn first_uniq_char(s: String) -> i32 {
        let mut charset = std::collections::HashMap::<char, i32>::new();
        for c in s.chars() {
            let e = charset.entry(c).or_default();
            *e += 1;
        }
        for (idx, c) in s.chars().enumerate() {
            if charset.get(&c).unwrap() == &1 {
                return idx as i32;
            }
        }
        return -1;
    }

    pub fn next_greater_element(nums1: Vec<i32>, nums2: Vec<i32>) -> Vec<i32> {
        let hashtbl = preprocess_input(&nums2);
        let mut next_greater = Vec::with_capacity(nums1.len());
        'outer: for num in nums1 {
            let location_in_nums2 = hashtbl.get(&num).unwrap();
            for i in location_in_nums2 + 1..nums2.len() {
                let cur_right_num = nums2[i];
                if cur_right_num > num {
                    next_greater.push(cur_right_num);
                    continue 'outer;
                }
            }
            next_greater.push(-1);
        }
        next_greater
    }

    fn preprocess_input(items: &Vec<i32>) -> HashMap<i32, usize> {
        let mut map = HashMap::with_capacity(items.len());
        for (idx, &value) in items.iter().enumerate() {
            map.insert(value, idx);
        }
        map
    }

    /// The pivot index is an index of the array such that
    /// the sum of items to the left of the index is equal
    /// to the sum of items to the right of the index
    pub fn pivot_index(nums: Vec<i32>) -> i32 {
        // We begin by finding the total sum of all the items
        // in the inputs list
        let mut right_sum: i32 = nums.iter().sum();
        let mut left_sum = 0;

        // After that, we check each index, to see if it could be
        // the pivot index
        for (idx, &cur_num) in nums.iter().enumerate() {
            right_sum -= cur_num;
            if left_sum == left_sum {
                return idx as i32;
            }
            left_sum += cur_num;
        }
        return -1;
    }

    pub fn dominant_index(nums: Vec<i32>) -> i32 {
        // Start by finding the location and value of the largest item
        // in the list. This is done in linear time
        let (max_idx, &max) = nums.iter().enumerate().max_by(|a, b| a.1.cmp(b.1)).unwrap();

        // Next, we do another linear scan to check if max >= nums[i] * 2
        for (idx, &num) in nums.iter().enumerate() {
            if idx != max_idx && max < num * 2 {
                return -1;
            }
        }
        return max_idx as i32;
    }

    /// Checks is `query` is a subsequence of `s` in O(|query|)
    fn is_subsequence(s: &str, query: &str) -> bool {
        let mut pos_in_s = 0;
        let mut pos_in_query = 0;
        while pos_in_s < s.len() && pos_in_query < query.len() {
            let c = s[pos_in_s..=pos_in_s].chars().next().unwrap();
            if c == query[pos_in_query..=pos_in_query].chars().next().unwrap() {
                pos_in_query += 1;
            }
            pos_in_s += 1;
        }
        pos_in_query == query.len()
    }

    /// Given an array of integers nums containing n + 1 integers where each integer is in the range [1, n] inclusive.
    ///
    /// There is only one repeated number in nums, return this repeated number.
    ///
    /// How can we prove that at least one duplicate number must exist in nums? -- Pigeonhole Principle
    /// Can you solve the problem without modifying the array nums? -- HashTable O(n) space
    /// Can you solve the problem using only constant, O(1) extra space? -- Sort (In place Quicksort), then Scan
    /// Can you solve the problem with runtime complexity less than O(n2)? -- as above (n lg n)
    ///
    /// There is definitely a way to do it in O(n) with O(1) space -- counting sort
    /// Floyd's Hare and Tortoise
    pub fn find_duplicate(mut nums: Vec<i32>) -> i32 {
        nums.sort();
        for window in nums.windows(2) {
            if window[0] == window[1] {
                return window[0];
            }
        }
        panic!("No duplicate found in {:?}", nums);
    }

    /// Returns the intersection of the given collection of iterators
    fn intersection<T: Ord>(collections: &[impl Iterator<Item = T>]) -> Vec<T> {
        todo!()
    }

    /// Given an array A of non-negative integers, return an array
    /// consisting of all the even elements of A, followed by
    /// all the odd elements of A. You may return any answer array
    /// that satisfies this condition.
    ///
    /// 1 <= A.length <= 5000
    /// 0 <= A[i] <= 5000
    fn sort_by_parity(a: &[i32]) -> Vec<i32> {
        let mut evens = Vec::with_capacity(a.len());
        let mut odds = Vec::with_capacity(a.len());
        for &val in a {
            if val & 1 != 1 {
                evens.push(val);
            } else {
                odds.push(val);
            }
        }
        for odd_val in odds {
            evens.push(odd_val);
        }
        evens
    }

    /// Alice has n candies, where the ith candy is of type candyType[i].
    /// Alice noticed that she started to gain weight, so she visited a doctor.

    /// The doctor advised Alice to only eat n / 2 of the candies she has (n is always even).
    /// Alice likes her candies very much, and she wants to eat the maximum number
    /// of different types of candies while still following the doctor's advice.

    /// Given the integer array candyType of length n, return the maximum number of
    /// different types of candies she can eat if she only eats n / 2 of them.
    pub fn distribute_candies(candy_type: Vec<i32>) -> i32 {
        // We must not exceed out budget
        let mut remaining_budget = candy_type.len() / 2;

        // We need to count up how many types we've eaten
        let mut counter = 0;

        // To maximize for consuming as many different candy types as
        // possible, we're going to select each type exactly once. This
        // stems from teh observation that the  value we can ever
        // return is upper bound by the distinct number of items in
        // `candy_type`.
        //
        // Therefore, whenever we have selected a candy,
        // we add it to the set below. We only select candy types
        // that are not already in this set
        let mut already_eaten = HashSet::with_capacity(remaining_budget);

        // We make a single L-R scan over the candy types we have,
        // selecting those that we have not tried before. We only proceed
        // until our budget is exhausted
        for flavor in candy_type {
            // If we've exhausted our budget, then we stop
            if remaining_budget == 0 {
                break;
            }

            // Sample the current flavor
            if !already_eaten.contains(&flavor) {
                already_eaten.insert(flavor);
                counter += 1;
                remaining_budget -= 1;
            }
        }
        counter
    }

    /// Given a string represented as a sequence of characters <s_1, s_2, ..., s_n>
    /// returns a the same sequence reversed, That is it returns <s_n, ..., s_2, s_1>
    pub fn reverse_string(s: &mut Vec<char>) {
        if s.len() > 0 {
            let (mut start, mut end) = (0, s.len() - 1);
            while start < end {
                s.swap(start, end);
                start += 1;
                end -= 1;
            }
        }
    }

    /// Given an integer array nums sorted in non-decreasing order,
    /// return an array of the squares of each number sorted in
    /// non-decreasing order.
    pub fn sorted_squares(nums: Vec<i32>) -> Vec<i32> {
        let mut squares: Vec<i32> = nums.iter().map(|num| num * num).collect();
        squares.sort();
        squares
    }

    /// The n-th fibonacci number
    pub fn fib(n: i32) -> i32 {
        let mut lookup = Vec::with_capacity(n as usize);
        lookup.push(0);
        lookup.push(1);
        for i in 2..n {
            let i = i as usize;
            let fib_i = lookup[i - 1] + lookup[i - 2];
            lookup.push(fib_i);
        }
        *lookup.last().unwrap()
    }

    /// Given a non-empty array of integers nums, every element
    /// appears twice except for one. Find that single one.
    ///
    /// Follow up: Could you implement a solution with a linear
    /// runtime complexity and without using extra memory?
    pub fn single_number(nums: Vec<i32>) -> i32 {
        // The straightforward <O(n), O(n)> solution is to keep a hash set if
        // the values. We insert into this set the first time we see an item
        // and remove from it if it's already in. Since numbers appear only twice,
        // the number that appears once will be the only thing remaining in the
        // set at the end of the linear scan
        //
        // To implement this in linear time without using extra
        // memory, we make two scan -- one from L-R and another from R-L.
        // during the L-R scan, we keep a counter that we increment
        // and in the R-L scan, we decrement the counter as we iterate.
        //
        // To use the counting method, we need to somehow figure out
        // when we're seeing an item for the second time. That requires
        // us to keep around the first item that we found automatically
        // implying the need for O(n) space.
        //
        // We can, however, used XOR. In particular, XOR has the following key
        // properties:
        //      (a) Its associative and commutative
        //      (b) 0 XOR x = x. This is the identity function of XOR
        //      (c) x XOR x = 0
        // We implement the solution that uses XOR below
        let mut counter = 0;
        for number in nums {
            counter ^= number;
        }
        counter
    }

    /// Given a sorted array of integers nums and integer values a, b and c.
    /// Apply a quadratic function of the form f(x) = ax2 + bx + c to each element x in the array.
    /// The returned array must be in sorted order.
    ///
    /// Expected time complexity: O(n)
    ///
    ///
    ///    
    /// Parabola
    pub fn sort_transformed_array(mut nums: &[i32], a: i32, b: i32, c: i32) -> Vec<i32> {
        let f = |x: &i32| a * (x * x) + b * x + c;
        let transformed = nums.iter().map(f).collect::<Vec<_>>();
        println!("{:?}", transformed);
        transformed
    }

    #[test]
    fn test_transformed() {
        let nums = [-4, -2, 2, 4, 6];
        let (a, b, c) = (-1, 3, 5);
        sort_transformed_array(&nums, a, b, c);
    }

    struct Solution;

    impl Solution {
        pub fn num_unique_emails(emails: Vec<&str>) -> i32 {
            // Create a new set to hold our parsed strings. We initialize
            // the capacity to be the upper bound on the the number of items
            // it'll hold. This is done to avoid reallocations later on thus
            // ensuring that the runtime of adding into a set if O(1) expected
            // not expected amortized
            let mut unique_emails = HashSet::with_capacity(emails.len());
            for email in emails {
                // For each "provisional email", we want to first get the parsed version of
                // the email, then store that parsed version in our set of unique emails
                let parsed_email = Self::parse_email(&email);
                unique_emails.insert(parsed_email);
            }
            unique_emails.len() as i32
        }

        /// Given an e-mail, parse it by applying the two rules to
        /// the local name. The e-mails have this general structure
        ///
        /// +-----------------------------------+
        /// | <local name> @  < domain name>    |
        /// +-----------------------------------+
        ///
        /// The parsed e-mail is returned as a new string
        fn parse_email(email: &str) -> String {
            let mut parsed_email = String::with_capacity(email.len());

            // To parse the e-mail, we first split it into a prefix and
            // a suffix as determined by the location of the '@' symbol.
            // the prefix is the local name. We need to parse this further.
            // the suffix is the domain name that needs no further parsing
            let email_parts: Vec<&str> = email.split('@').collect();

            // At this point, email_parts must have a length of 2
            // because we assume the e-mail is valid and thus has only
            // one '@'
            let parsed_local_name = Self::parse_local_name(email_parts[0]);
            parsed_email.push_str(&parsed_local_name);
            parsed_email.push('@');
            parsed_email.push_str(email_parts[1]);
            parsed_email
        }

        /// Applies the two rules to the local portion of an e-mail
        fn parse_local_name(local_name: &str) -> String {
            let mut s = String::with_capacity(local_name.len());

            // We  first split the string into parts separated by '+' and
            // select only the first one
            let portions: Vec<&str> = local_name.split('+').collect();
            s.push_str(portions.first().unwrap());

            // We then replace all occurrences of '.'
            s = s.replace('.', "");
            s
        }
    }

    #[test]
    fn test_email_parser() {
        let input = vec![
            "test.email+alex@leetcode.com",
            "test.e.mail+bob.cathy@leetcode.com",
            "testemail+david@lee.tcode.com",
        ];
        Solution::num_unique_emails(input);
    }

    #[test]
    fn test_is_sub_sequence() {
        let s = "abppplee";
        assert!(is_subsequence(s, "apple"));
        assert!(is_subsequence(s, "ale"));
        assert!(is_subsequence(s, "able"));
        assert!(!is_subsequence(s, "bale"));
    }

    /// This is LC-1021.
    fn remove_outer_parentheses(s: &str) -> String {
        // We first generate a list of all the primitive blocks in the string
        let primitive_blocks = generate_primitive_blocks(s);

        // Once we know where those blocks are, we can scan over them
        // making the needed modifications as we go along
        let mut pruned_string = String::with_capacity(s.len());
        for block in primitive_blocks {
            let end = block.len() - 1;
            pruned_string.push_str(&block[1..end]);
        }
        pruned_string
    }

    /// Generates all the primitive blocks in the string
    fn generate_primitive_blocks(mut s: &str) -> Vec<&str> {
        let mut blocks = Vec::new();
        let mut counter = 0;
        let mut block_start = 0;

        // To find where blocks end, we maintain a counter that
        // simulates pushing to and popping from a stack.
        // When the counter goes to 0, we know we've reached
        // the end of a block
        for (idx, symbol) in s.char_indices() {
            // Observe the current symbol and update the state
            if symbol == '(' {
                counter += 1;
            } else {
                counter -= 1;
            }

            // Decide if the symbol we just observed marks the end of a
            // primitive block
            if counter == 0 {
                blocks.push(&s[block_start..=idx]);
                block_start = idx + 1;
            }
        }
        blocks
    }

    #[test]
    fn test_parens() {
        assert_eq!(remove_outer_parentheses("(()())(())(()(()))"), "()()()()(())");
        assert_eq!(remove_outer_parentheses("()()"), "");
    }

    /// An object that will provide the moving average functionality.
    ///
    /// Remember, the average of a collection `A` of `k` items is
    /// simply the sum of the collection divided by k.
    struct MovingAverage {
        /// The target window size
        k: i32,

        /// The current window
        window: Vec<i32>,

        /// We keep track of the total value of the items
        /// in the window
        current_sum: i32,
    }

    impl MovingAverage {
        /// Creates a new moving average object
        fn new(size: i32) -> Self {
            MovingAverage {
                k: size,
                current_sum: 0,
                window: Vec::with_capacity(size as usize),
            }
        }

        /// Observe a new item and update the moving average object accordingly
        fn next(&mut self, val: i32) -> f64 {
            if self.window.len() == self.k as usize {
                // Update the state of the object to reflect the fact
                // that the first value has left the window and that a
                // new value has entered the window
                let exiting_value = self.window.remove(0);
                self.current_sum -= exiting_value;
            }
            self.window.push(val);
            self.current_sum += val;
            (self.current_sum as f64) / (self.window.len() as f64)
        }
    }

    type Pair<'a> = (usize, &'a f32);

    /// Given a list of numbers and a target Z, return the number of pairs
    /// according to following definition: (X,Y) where X+Y >= Z
    pub fn count(items: &[f32], target: f32) -> usize {
        let mut cnt = 0;
        let mut items: Vec<Pair> = items.iter().enumerate().collect();
        items.sort_by(|left, right| left.1.partial_cmp(right.1).unwrap());
        for (idx, &num) in &items {
            let remainder = target - num;
            let possible_partner = first_elem_greater_than(remainder, &items);
            if partner_is_valid(possible_partner, *idx, &items) {
                cnt += 1;
            }
        }
        cnt
    }

    fn partner_is_valid(possible_partner: Pair, idx: usize, items: &Vec<Pair>) -> bool {
        todo!()
    }

    fn first_elem_greater_than<'a>(remainder: f32, items: &[Pair<'a>]) -> Pair<'a> {
        todo!()
    }
}

mod change {
    //! You are given coins of different denominations and a total amount of money amount.
    //! Write a function to compute the fewest number of coins that you need to make up that amount.
    //! If that amount of money cannot be made up by any combination of the coins, return -1.
    //! You may assume that you have an infinite number of each kind of coin.

    use std::cmp::min;

    pub fn coin_change(coins: Vec<i32>, amount: i32) -> i32 {
        let mut counter = Vec::with_capacity((amount + 1) as usize);
        counter.push(0);
        for cur_amount in 1..=amount {
            let min_counts_for_cur = calculate_cur_min(&counter, &coins, cur_amount);
            counter.push(min_counts_for_cur);
        }

        *counter.last().unwrap()
    }

    fn calculate_cur_min(counter: &Vec<i32>, coins: &Vec<i32>, cur_amount: i32) -> i32 {
        // We do not know which choice of coin would result in our
        // using the fewest number of coins. So we try all of them
        // and pick the one that is both feasible and minimal
        let mut cur_min_count = -1;
        for &coin in coins {
            let remainder = cur_amount - coin;

            // If whatever remains after choosing this coin is non-negative
            // and feasible, we know that this coin is feasible.
            // We choose it and update `cur_min_count` if smaller
            if remainder >= 0 && counter[remainder as usize] >= 0 {
                let count = counter[remainder as usize] + 1;
                if cur_min_count == -1 {
                    cur_min_count = count;
                } else {
                    cur_min_count = min(cur_min_count, count);
                }
            }
        }
        cur_min_count
    }

    #[test]
    fn test_coin_change() {
        let coins = vec![1, 2, 5];
        let amount = 11;
        assert_eq!(coin_change(coins, amount), 3);

        let coins = vec![2];
        let amount = 3;
        assert_eq!(coin_change(coins, amount), -1);

        let coins = vec![1];
        let amount = 0;
        assert_eq!(coin_change(coins, amount), 0);

        let coins = vec![1];
        let amount = 1;
        assert_eq!(coin_change(coins, amount), 1);

        let coins = vec![1];
        let amount = 2;
        assert_eq!(coin_change(coins, amount), 2);
    }
}

mod most_frequent_i {
    //! Given a non-empty array of integers, return the k most frequent elements.
    //!
    //! You may assume k is always valid, 1 ≤ k ≤ number of unique elements.
    //! Your algorithm's time complexity must be better than O(n log n), where n is the array's size.
    //! It's guaranteed that the answer is unique, in other words the set of the top k frequent elements is unique.
    //! You can return the answer in any order.
    use std::collections::{BinaryHeap, HashMap};

    #[derive(PartialEq, Eq, PartialOrd, Ord)]
    struct CountTuple {
        frequency: usize,
        number: i32,
    }

    pub fn top_k_frequent(nums: Vec<i32>, k: i32) -> Vec<i32> {
        let k = k as usize;
        let mut res = Vec::with_capacity(k);
        let mut counter = HashMap::new();
        let mut cur_top_k = BinaryHeap::new();

        // Count up how many times each number appears
        for num in nums {
            let cur_value = counter.entry(num).or_insert(0);
            *cur_value += 1;
        }

        // Place the <freq, number> pairs into the heap
        for (number, frequency) in counter {
            let heap_entry = CountTuple { number, frequency };
            cur_top_k.push(heap_entry);
        }

        // Extract the top-k items from the heap
        while res.len() < k && !cur_top_k.is_empty() {
            let cur = cur_top_k.pop().unwrap();
            res.push(cur.number);
        }
        res
    }
}

mod most_frequent_ii {
    //! Given a non-empty list of words, return the k most frequent elements.
    //!
    //! Your answer should be sorted by frequency from highest to lowest.
    //! If two words have the same frequency,
    //! then the word with the lower alphabetical order comes first.
    use std::{
        cmp::Reverse,
        collections::{BinaryHeap, HashMap},
    };
    #[derive(PartialEq, Eq, PartialOrd, Ord)]
    struct CountTuple {
        frequency: usize,
        word: Reverse<String>,
    }

    /// Finds and returns the `k` most frequent words in the given list.
    pub fn top_k_frequent(words: Vec<&str>, k: i32) -> Vec<String> {
        let k = k as usize;
        let mut counter = HashMap::with_capacity(words.len());
        let mut top_k = BinaryHeap::with_capacity(words.len());
        let mut res = Vec::with_capacity(k);

        // Count the times each word appears. This is done in liner time
        for w in words {
            let cur_entry = counter.entry(w).or_insert(0);
            *cur_entry += 1;
        }

        // Put the counts in partial order by putting them in a Heap.
        for (word, frequency) in counter {
            let count_tuple = CountTuple {
                word: Reverse(word.to_string()),
                frequency,
            };
            // The expected cost of push, averaged over every possible ordering
            // of the elements being pushed, and over a sufficiently large number of pushes, is O(1).
            // This is the most meaningful cost metric when pushing elements that are not already in any sorted pattern.
            //
            // The time complexity degrades if elements are pushed in predominantly ascending order.
            // In the worst case, elements are pushed in ascending sorted order and the amortized
            // cost per push is O(log(n)) against a heap containing n elements.
            //
            // The worst case cost of a single call to push is O(n).
            // The worst case occurs when capacity is exhausted and needs a resize.
            // The resize cost has been amortized in the previous figures.
            top_k.push(count_tuple);
        }

        // Each iteration takes `O(lg k)`
        while res.len() < k && !top_k.is_empty() {
            let cur = top_k.pop().unwrap();
            res.push(cur.word.0)
        }
        res
    }

    #[test]
    fn test_top_k() {
        let input = vec!["i", "love", "code", "i", "love", "coding"];
        let k = 2;
        let expected = vec!["i", "love"];
        assert_eq!(top_k_frequent(input, k), expected);

        let input = vec!["the", "day", "is", "sunny", "the", "the", "the", "sunny", "is", "is"];
        let k = 4;
        let expected = vec!["the", "is", "sunny", "day"];
        assert_eq!(top_k_frequent(input, k), expected)
    }
}

mod longest_substring_non_repeating {
    use std::collections::HashSet;

    // Given a string s, find the length of the longest substring without repeating characters.
    pub fn length_of_longest_substring(s: String) -> i32 {
        let mut cur_longest = 0;
        for start in 0..s.len() {
            let longest_at_prefix = get_longest_at_cur_start(&s, start);
            if longest_at_prefix > cur_longest {
                cur_longest = longest_at_prefix;
            }
        }
        cur_longest as i32
    }

    fn get_longest_at_cur_start(s: &str, start: usize) -> usize {
        let mut longest_at_start = 0;
        let mut counter = HashSet::with_capacity(s[start..].len());
        // We attempt to increase the substring starting at the current
        // start index without introducing duplicates
        for end in start..s.len() {
            let new_char = s[end..=end].chars().next().unwrap();
            if counter.contains(&new_char) {
                break;
            }
            counter.insert(new_char);
            longest_at_start += 1;
        }
        longest_at_start
    }

    #[test]
    fn test_longest_no_duplicates() {
        let s = "abcabcbb";
        let expected = 3;
        assert_eq!(expected, length_of_longest_substring(s.to_string()));

        let s = "bbbbb";
        let expected = 1;
        assert_eq!(expected, length_of_longest_substring(s.to_string()));

        let s = "pwwkew";
        let expected = 3;
        assert_eq!(expected, length_of_longest_substring(s.to_string()));

        let s = "dvdf";
        let expected = 3;
        assert_eq!(expected, length_of_longest_substring(s.to_string()));

        let s = "c";
        let expected = 1;
        assert_eq!(expected, length_of_longest_substring(s.to_string()));
    }
}

mod climbing_stairs {
    //! You are climbing a staircase. It takes n steps to reach the top.
    //! Each time you can either climb 1 or 2 steps.
    //! In how many distinct ways can you climb to the top?
    pub fn climb_stairs(n: i32) -> i32 {
        let n = n as usize;
        let mut num_ways = Vec::with_capacity(n + 1);
        // With 0 stairs, there's only one way to climb
        num_ways.push(1);

        // With 1 stair, there's also only one way to climb
        num_ways.push(1);

        for k in 2..=n {
            let ways_to_climb_k = num_ways[k - 2] + num_ways[k - 1];
            num_ways.push(ways_to_climb_k)
        }
        *num_ways.last().unwrap()
    }
}

mod matrix {
    //! A collection of procedures and abstractions for dealing with 2-D
    //! arrays

    #[derive(Debug)]
    pub struct Matrix<T: Ord, const ROWS: usize, const COLS: usize> {
        /// [[--row-1--], [--row-2--], [--row-3--], ...., [--row-n]]
        /// To access a single item, we do `data[row_id][col_id]`
        data: [[T; COLS]; ROWS],
    }
}

mod split_array {
    //! LC-659
    //!     Given an integer array nums that is sorted in ascending order,
    //!     return true if and only if you can split it into one or more subsequences such
    //!     that each subsequence consists of consecutive integers and has a length of at least 3.
    //! LC-1296
    //!     Given an array of integers nums and a positive integer k,
    //!     find whether it's possible to divide this array into sets of k consecutive numbers
    //!     Return True if it is possible. Otherwise, return False.
    //!

    /// LC-659 is a subarray problem. Because the input is sorted, any possible answer
    /// will occupy a contiguous portion of the array
    pub fn is_possible(nums: &[i32]) -> bool {
        for window in nums.windows(3) {
            if has_consecutive_integers(window) {
                return true;
            }
        }
        return false;
    }

    /// Given a slice of length 3 return true if the integers it contains
    /// are all consecutive
    fn has_consecutive_integers(window: &[i32]) -> bool {
        let (first, second, third) = (window[0], window[1], window[2]);
        (first + 1) == second && (second + 1 == third)
    }
}

mod sub_array_k_different_integers {
    //! LC-992
    //!     Given an array A of positive integers,
    //!     call a (contiguous, not necessarily distinct) sub-array of A
    //!     good if the number of different integers in that sub-array is exactly K.
    //!     (For example, [1,2,3,1,2] has 3 different integers: 1, 2, and 3.)
    //!     Return the number of good sub-arrays of A.
}

mod nested_list_weight_sum {
    #[derive(Debug, PartialEq, Eq)]
    pub enum NestedInteger {
        Int(i32),
        List(Vec<NestedInteger>),
    }

    struct Solution;

    impl Solution {
        /// You are given a nested list of integers nestedList.
        /// Each element is either an integer or a list whose elements may also be integers or other lists.
        ///
        /// The depth of an integer is the number of lists that it is inside of.
        /// For example, the nested list [1,[2,2],[[3],2],1]
        /// has each integer's value set to its depth.
        ///
        /// Return the sum of each integer in nestedList multiplied by its depth.
        pub fn dfs(nested_list: Vec<NestedInteger>, depth: i32) -> i32 {
            let mut sum = 0;
            for item in nested_list {
                match item {
                    NestedInteger::Int(num) => sum += depth * num,
                    NestedInteger::List(sub_list) => sum += Self::dfs(sub_list, depth + 1),
                }
            }
            return sum;
        }
    }
}

mod coarse_scheduling {
    //! LC-210
    //!     There are a total of n courses you have to take labelled from 0 to n - 1.
    //!     Some courses may have prerequisites, for example, if prerequisites\[i] = [ai, bi]
    //!     this means you must take the course bi before the course ai.
    //!
    //!     Given the total number of courses numCourses and a list of the prerequisite pairs,
    //!     return the ordering of courses you should take to finish all courses.
    //!
    //!     If there are many valid answers, return any of them.
    //!     If it is impossible to finish all courses, return an empty array.
    //! LC-207
    //!     There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1.
    //!     You are given an array prerequisites where prerequisites[i] = [ai, bi]
    //!     indicates that you must take course bi first if you want to take course ai.
    //!     For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.
    //!     Return true if you can finish all courses. Otherwise, return false.
    //! LC-630
    //!     There are n different online courses numbered from 1 to n.
    //!     Each course has some duration(course length) t and closed on dth day.
    //!     A course should be taken continuously for t days and must be finished before or on the dth day.
    //!     You will start at the 1st day.
    //!     Given n online courses represented by pairs (t,d), your task is to
    //!     find the maximal number of courses that can be taken.

    pub fn can_finish(num_courses: i32, prerequisites: Vec<Vec<i32>>) -> bool {
        todo!()
    }

    pub fn find_order(num_courses: i32, prerequisites: Vec<Vec<i32>>) -> Vec<i32> {
        todo!()
    }

    pub fn maximal_course_number(courses: Vec<Vec<i32>>) -> i32 {
        todo!()
    }

    #[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy)]
    pub struct Course(usize);

    impl From<usize> for Course {
        fn from(id: usize) -> Self {
            Course(id)
        }
    }

    #[derive(Debug)]
    struct PrerequisiteConstraint {
        /// This should be done before `after` is taken
        src: Course,

        /// This can only be done after `before`  is taken
        dest: Course,
    }
}

mod minimum_height_trees {
    //! LC-310
    //! A tree is an undirected graph in which any two vertices are connected by exactly one path.
    //! In other words, any connected graph without simple cycles is a tree.
    //!
    //! Given a tree of n nodes labelled from 0 to n - 1, and an array of n - 1 edges where edges[i] = [ai, bi] indicates that
    //! there is an undirected edge between the two nodes ai and bi in the tree, you can choose any node of the tree as the root.
    //! When you select a node x as the root, the result tree has height h. Among all possible rooted trees,
    //! those with minimum height (i.e. min(h))  are called minimum height trees (MHTs).
    //! Return a list of all MHTs' root labels. You can return the answer in any order.
    //! The height of a rooted tree is the number of edges on the longest downward path between the root and a leaf.
}

mod word_ladder_i {
    //! A transformation sequence from word beginWord to word endWord using a
    //! dictionary wordList is a sequence of words such that:
    //!
    //! - The first word in the sequence is beginWord.
    //! - The last word in the sequence is endWord.
    //! - Only one letter is different between each adjacent pair of words in the sequence.
    //! - Every word in the sequence is in wordList.
    //!
    //! Given two words, beginWord and endWord, and a dictionary wordList,
    //! return the number of words in the shortest transformation sequence
    //! from beginWord to endWord, or 0 if no such sequence exists.

    use std::collections::{HashMap, HashSet, VecDeque};

    /// a list of all lowercase letters
    const LOWER_CASE_ALPHABET: [char; 26] = [
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
        'w', 'x', 'y', 'z',
    ];

    /// We can easily solve this problem by modelling it as a graph.
    /// Given a graph where the nodes are the words in the `word_list` and
    /// a node x is connected to another node y iff y is formed by changing
    /// a single letter in x, we can find the length of shortest path
    /// by doing BFS in this graph.
    ///
    /// This graph is directed and unweighted
    pub fn ladder_length(begin_word: String, end_word: String, mut word_list: Vec<String>) -> i32 {
        // As defined, begin_word is not in word_list. So first order of business
        // is to add it
        word_list.push(begin_word.clone());
        let dictionary = word_list.iter().collect::<HashSet<_>>();
        let graph = construct_graph_from(&dictionary);
        len_shortest_path_in(&graph, begin_word, end_word)
    }

    #[test]
    fn test_word_ladders() {
        let (src, sink) = ("hit", "cog");
        let dict = vec!["hot", "dot", "dog", "lot", "log", "cog"];
        assert_eq!(
            ladder_length(
                src.to_string(),
                sink.to_string(),
                dict.iter().map(|s| s.to_string()).collect()
            ),
            5
        );

        let (src, sink) = ("a", "c");
        let dict = vec!["a", "b", "c"];
        assert_eq!(
            ladder_length(
                src.to_string(),
                sink.to_string(),
                dict.iter().map(|s| s.to_string()).collect()
            ),
            2
        );
    }

    /// To find the length of the shortest path, we simply run bfs with begin_word
    /// as the source and end_word as the sink
    fn len_shortest_path_in(graph: &Graph, begin_word: String, end_word: String) -> i32 {
        graph.bfs_path(begin_word, end_word) as i32
    }

    /// A graph of strings represented using an adjacency list.
    #[derive(Debug)]
    struct Graph(HashMap<String, Vec<String>>);

    impl Graph {
        /// Finds the length of the shorted unweighted path from the source to the sink
        pub fn bfs_path(&self, src: String, sink: String) -> usize {
            // We keep track of the shortest distance from src to all
            // other nodes using a hash_map <string, distance_from_src>
            let mut path_lens = HashMap::with_capacity(self.0.len());

            // The queue to implement BFS
            let mut queue = VecDeque::with_capacity(self.0.len());

            // A set of already explored nodes to ensure that
            // we not reprocess nodes that have already been seen
            let mut already_seen = HashSet::with_capacity(self.0.len());

            // BFS Start
            queue.push_back(&src);
            already_seen.insert(&src);
            path_lens.insert(&src, 1);
            while !queue.is_empty() {
                let cur_node = queue.pop_front().unwrap();
                if cur_node.eq(&sink) {
                    return *path_lens.get(&sink).unwrap();
                }
                let &cur_node_distance = path_lens.get(&cur_node).unwrap();
                let cur_node_neighbors = self.0.get(cur_node).unwrap();
                for neighbor in cur_node_neighbors {
                    if already_seen.contains(neighbor) {
                        continue;
                    }
                    already_seen.insert(neighbor);
                    queue.push_back(neighbor);
                    path_lens.insert(neighbor, cur_node_distance + 1);
                }
            }
            return 0;
        }

        /// Given the label of a node that is already in the graph,
        /// this procedure adds all words in the dictionary
        /// that are one character away from `s` into the neighbor list
        /// of `s`
        pub fn add_neighbors_of(&mut self, s: &str, dict: &HashSet<&String>) {
            // We can implement this in two ways:
            //  In the first way, we can enumerate all words formed by substituting
            //  a single character in `s` with one of the letters in our alphabet Sigma, then
            //  adding a word iff it's in the dictionary. This approach takes O(|s| * |Sigma|)
            //
            // The Second option is to iterate over all the other dictionary words and
            // add as neighbors those that are only one character away. That is, those with a
            // Hamming Distance of `1`. This approach takes O(|dict| * |s|)
            //
            // Below, we implement the first option
            for &alphabet_char in LOWER_CASE_ALPHABET.iter() {
                // We try to substitute `alphabet_char` at all possible locations
                // in `s` and if the resulting word is in our dictionary,
                // we add it as a neighbor
                for (idx_to_replace, _) in s.char_indices() {
                    let possible_word = form_possible_word(s, idx_to_replace, alphabet_char);
                    if dict.contains(&possible_word) && !possible_word.eq(s) {
                        self.0
                            .get_mut(s)
                            .and_then(|cur_neighbors| Some(cur_neighbors.push(possible_word)));
                    }
                }
            }
        }
    }

    /// Created a new word by replacing the character at `ids_to_replace` with `char_to_replace_with
    /// in the given string `s`.
    fn form_possible_word(s: &str, idx_to_replace: usize, char_to_replace_with: char) -> String {
        let mut new_word = String::with_capacity(s.len());
        new_word.push_str(&s[0..idx_to_replace]);
        new_word.push(char_to_replace_with);
        new_word.push_str(&s[idx_to_replace + 1..]);
        new_word
    }

    /// To construct the graph, we simply scan the list of words from left to
    /// right, and for each word, we first insert it in the graph
    /// with no neighbors. Then, in the next step, we add all the
    /// valid 1-word neighbors of that word into its neighbor list
    fn construct_graph_from(word_list: &HashSet<&String>) -> Graph {
        let mut graph = Graph(HashMap::with_capacity(word_list.len()));
        for &word in word_list {
            graph.0.insert(word.clone(), Vec::new());
            graph.add_neighbors_of(word, word_list);
        }
        graph
    }
}

mod word_ladder_ii {
    //! A transformation sequence from word beginWord to word endWord using a dictionary
    //! wordList is a sequence of words such that:
    //!    The first word in the sequence is beginWord.
    //!    The last word in the sequence is endWord.
    //!    Only one letter is different between each adjacent pair of words in the sequence.
    //!    Every word in the sequence is in wordList.
    //!
    //! Given two words, beginWord and endWord, and a dictionary wordList,
    //! return all the shortest transformation sequences from beginWord to endWord,
    //! or an empty list if no such sequence exists.
}

mod minimum_genetic_mutation {
    //! A gene string can be represented by an 8-character long string,
    //! with choices from "A", "C", "G", "T".
    //!
    //! Suppose we need to investigate about a mutation (mutation from "start" to "end"),
    //! where ONE mutation is defined as ONE single character changed in the gene string.
    //! For example, "AACCGGTT" -> "AACCGGTA" is 1 mutation.
    //!
    //! Also, there is a given gene "bank", which records all the valid gene mutations.
    //! A gene must be in the bank to make it a valid gene string.
    //!
    //! Now, given 3 things - <start, end, bank>, your task is to
    //! determine what is the minimum number of mutations needed to mutate from "start" to "end".
    //! If there is no such a mutation, return -1.
    //!
    //! Note:
    //!    Starting point is assumed to be valid, so it might not be included in the bank.
    //!    If multiple mutations are needed, all mutations during in the sequence must be valid.
    //!    You may assume start and end string is not the same.
}

mod eulerian_path {}

mod redundant_connection {
    //! LC-684
    //! In this problem, a tree is an undirected graph that is connected and has no cycles.
    //!
    //! The given input is a graph that started as a tree with N nodes
    //! (with distinct values 1, 2, ..., N), with one additional edge added.
    //! The added edge has two different vertices chosen from 1 to N, and was not an edge that already existed.
    //!
    //! The resulting graph is given as a 2D-array of edges.
    //! Each element of edges is a pair [u, v] with u < v, that represents an undirected edge connecting nodes u and v.
    //!
    //! Return an edge that can be removed so that the resulting graph is a tree of N nodes.
    //! If there are multiple answers, return the answer that occurs last in the given 2D-array.
    //! The answer edge [u, v] should be in the same format, with u < v.

    use std::collections::HashMap;

    pub fn find_redundant_connection(edges: Vec<Vec<i32>>) -> Vec<i32> {
        todo!()
    }

    #[derive(Debug)]
    struct SingleNodeInfo<T> {
        visited: bool,
        predecessor: T,
        discovery_time: usize,
        finishing_time: usize,
        current_distance_estimate: usize,
    }

    #[derive(Debug)]
    struct Graph<T> {
        /// A mapping from <node_id, [neighbor_nodes_ids]
        adjacency_list: HashMap<T, Vec<T>>,

        /// A mapping from node_id to the auxillary for that node
        book_keeping: HashMap<T, SingleNodeInfo<T>>,
    }
}

mod replace_words {
    //! In English, we have a concept called root, which can be followed by some other word to
    //! form another longer word - let's call this word successor. For example,
    //! when the root "an" is followed by the successor word "other", we can form a new word "another".
    //!
    //! Given a dictionary consisting of many roots and a sentence consisting of words
    //! separated by spaces, replace all the successors in the sentence with the root forming it.
    //! If a successor can be replaced by more than one root, replace it
    //! with the root that has the shortest length.
    //!
    //! Return the sentence after the replacement.
}

mod island_perimeter {
    //! You are given row x col grid representing a map where grid[i][j] = 1 represents land and grid[i][j] = 0 represents water.
    //! Grid cells are connected horizontally/vertically (not diagonally).
    //! The grid is completely surrounded by water, and there is exactly one island (i.e., one or more connected land cells).
    //!
    //! The island doesn't have "lakes", meaning the water inside isn't connected to the water around the island.
    //! One cell is a square with side length 1.
    //! The grid is rectangular, width and height don't exceed 100.
    //! Determine the perimeter of the island.
}

mod coloring_a_border {
    //! Given a 2-dimensional grid of integers, each value in the grid represents
    //! the color of the grid square at that location.
    //!
    //! Two squares belong to the same connected component if
    //! and only if they have the same color and are next to
    //! each other in any of the 4 directions.
    //!
    //! The border of a connected component is all the squares
    //! in the connected component that are either 4-directionally
    //! adjacent to a square not in the component, or on the boundary
    //! of the grid (the first or last row or column).
    //!
    //! Given a square at location (r0, c0) in the grid and a color,
    //! color the border of the connected component of that square
    //! with the given color, and return the final grid.
}

mod k_sum {
    //! LC-18
    //! Given an array nums of n integers and an integer target,
    //! are there elements a, b, c, and d in nums such that a + b + c + d = target?
    //! Find all unique quadruplets in the array which gives the sum of target.
    //!
    //! Notice that the solution set must not contain duplicate quadruplets.
}

mod trie {
    //! LC-208
    //! Implement a trie with insert, search, and startsWith methods.
    //!
    //! Example:
    //!
    //! ```rust
    //! Trie trie = new Trie();
    //!
    //! trie.insert("apple");
    //! trie.search("apple");   // returns true
    //! trie.search("app");     // returns false
    //! trie.startsWith("app"); // returns true
    //! trie.insert("app");   
    //! trie.search("app");     // returns true
    //! ```
    //!
    //! Note:
    //!
    //! You may assume that all inputs consist of lowercase letters a-z.
    //! All inputs are guaranteed to be non-empty strings.
}

mod lcp {
    //! As a warmup, write a function to find the length of the longest common prefix
    //! among a collection of strings
    pub fn lcp(strings: &[&str]) -> String {
        // If we sort the strings, those that share a prefix will clump together.
        // At that point, we can calculate the longest common prefix with one
        // L-R scan of the sorted list
        //
        // Actually, come to think of it, that is more work than we need. We would do
        // that if the goal was to create the LCP Array. We solve this task by a simple
        // L-R scan on the unsorted list
        //
        // Actually, now that I think about it even more, the question asks for the
        // lcp among all the strings, not just between pairs of strings. We solve it incrementally.
        //
        // At each iteration `i`, longest holds the longest_prefix among all strings seen so
        // far
        match strings.first() {
            None => "".to_string(),
            Some(&longest) => {
                let mut longest = longest.to_string();
                for tuple_window in strings.windows(2) {
                    let cur_lcp = calculate_window_lcp(tuple_window);
                    if cur_lcp.len() < longest.len() {
                        longest = cur_lcp;
                    }
                }
                longest
            }
        }
    }

    /// Calculates the longest common prefix between the two strings in the
    /// given tuple_window. This runs in O(|shorter_string_in_window|)
    fn calculate_window_lcp(tuple_window: &[&str]) -> String {
        let mut len = 0;
        let left = tuple_window.first().unwrap();
        let right = tuple_window.last().unwrap();
        for (l, r) in left.chars().zip(right.chars()) {
            if l != r {
                break;
            }
            len += 1;
        }
        left[0..len].to_string()
    }

    #[test]
    fn test_lcp() {
        let s = vec!["flower", "flow", "flight"];
        assert_eq!(lcp(&s), "fl");
    }
}

mod longest_common_sub_sequence {
    //! LC-1143
    //!
    //! Given two strings text1 and text2, return the length of their longest common subsequence.
    //!
    //! A subsequence of a string is a new string generated from the original string with some
    //! characters(can be none) deleted without changing the relative order of the remaining characters.
    //! (eg, "ace" is a subsequence of "abcde" while "aec" is not).
    //!
    //! A common subsequence of two strings is a subsequence that is common to both strings.
    //! If there is no common subsequence, return 0.

    use std::collections::{HashSet, VecDeque};

    /// We can model this task a single player game. At each turn, the player choose one of these
    /// actions to play: 1. Remove the current first char from one of the strings or 2.
    /// Remove the first character from both strings. If (2) is played, the player gets
    /// a reward of + 1 if the removed characters match. If (1) is played, the player
    /// always gets an immediate reward of 0. After an action is played, the game transitions
    /// to a new state where:
    ///     - The number of characters in the strings has decreased. If the action played was (2)
    ///       then both strings' character count decreased by 1. If (1) was played, only the string
    ///       from which the player removed a character decreased in length
    ///     - The cumulative reward also
    pub fn longest_common_subsequence(a: &str, b: &str) -> usize {
        let mut lookup = HashSet::<State>::with_capacity(a.len() * b.len());
        let mut states = VecDeque::<State>::with_capacity(a.len() * b.len());
        let actions = [Action::RemoveFromLeft, Action::RemoveFromRight, Action::RemoveFromBoth];
        let start_state = (0, a, b, false).into();
        states.push_back(start_state);
        let mut last_popped: Option<State> = None;
        while !states.is_empty() {
            // TODO: play
        }

        // This will not panic because we know at the very least the queue has
        // the start_state
        last_popped.unwrap().subsequence_len
    }

    /// The actions available to a player at any given turn
    #[derive(Debug)]
    enum Action {
        /// Remove the first character from the string on the left
        RemoveFromLeft,

        /// Remove the first character from the string on the right
        RemoveFromRight,

        /// Remove the first character from both strings
        RemoveFromBoth,
    }

    #[derive(Debug, PartialEq, Eq)]
    struct State<'a> {
        subsequence_len: usize,
        left: &'a str,
        right: &'a str,

        /// A flag to indicate if we've exhausted one of the strings.
        /// This could be encoded in a type state
        done: bool,
    }

    impl<'a> std::hash::Hash for State<'a> {
        fn hash<H: std::hash::Hasher>(&self, hasher: &mut H) {
            self.left.hash(hasher);
            self.right.hash(hasher);
        }
    }

    impl<'a> From<(usize, &'a str, &'a str, bool)> for State<'a> {
        fn from((len, left, right, done): (usize, &'a str, &'a str, bool)) -> State<'a> {
            State {
                left,
                right,
                done,
                subsequence_len: len,
            }
        }
    }

    impl<'a> State<'a> {
        /// Play a single turn of the game by taking the action provided in
        /// the current state. Doing so transitions to a new state
        pub fn play(&mut self, action: &Action) -> State<'a> {
            match action {
                // Remove the first character from the string on the left
                Action::RemoveFromLeft => {
                    let done = self.left.len() == 0;
                    if !done {
                        self.left = &self.left[1..];
                    }
                    (self.subsequence_len, self.left, self.right, done).into()
                }

                // Remove the first character from the string on the right
                Action::RemoveFromRight => {
                    let done = self.right.len() == 0;
                    if !done {
                        self.right = &self.right[1..];
                    }
                    (self.subsequence_len, self.left, self.right, done).into()
                }

                // Remove the first character from both strings and if the removed
                // characters match, increment the field `sequence_len`
                Action::RemoveFromBoth => {
                    let done = self.left.len() == 0 || self.right.len() == 0;
                    if !done {
                        let l = self.left.chars().next().unwrap();
                        let r = self.right.chars().next().unwrap();
                        if l == r {
                            self.subsequence_len += 1;
                        }
                        self.left = &self.left[1..];
                        self.right = &self.right[1..];
                    }
                    (self.subsequence_len, self.left, self.right, done).into()
                }
            }
        }
    }
}

mod longest_common_sup_sequence {
    //! LC-1092
    //! Given two strings str1 and str2, return the shortest string that has both str1 and str2 as subsequences.  
    //! If multiple answers exist, you may return any of them.
    //!
    //! (A string S is a subsequence of string T if deleting some number of characters from T
    //! (possibly 0, and the characters are chosen anywhere from T) results in the string S.)
}

mod longest_increasing_sub_sequence {
    //! LC-300
    //! Given an integer array nums, return the length of the longest strictly increasing subsequence.
    //!
    //! A subsequence is a sequence that can be derived from an array by deleting some or no elements
    //! without changing the order of the remaining elements.
    //!
    //! For example, [3,6,2,7] is a subsequence of the array [0,3,1,6,2,2,7].
    #[derive(Debug)]
    struct GameState {
        /// The index of the next number
        next_symbol_idx: usize,

        /// The longest increasing sub sequence thus far
        cur_longest: Vec<usize>,
    }

    impl From<(usize, Vec<usize>)> for GameState {
        fn from((next_symbol_idx, cur_longest): (usize, Vec<usize>)) -> Self {
            GameState {
                cur_longest,
                next_symbol_idx,
            }
        }
    }

    impl GameState {
        pub fn add_symbol(self) -> Self {
            todo!()
        }

        pub fn skip_symbol(self) -> Self {
            todo!()
        }
    }
}

mod longest_path_in_dag {}

mod longest_palindromic_subsequence {
    //! LC-516
}

mod longest_palindromic_substring {
    //! LC-5
}

mod longest_valid_parenthesis {
    //! LC-32
    //! Given a string containing just the characters '(' and ')',
    //! find the length of the longest valid (well-formed) parentheses substring.
}

mod longest_increasing_path_in_matrix {
    //! LC-329
}

mod edit_distance {}

mod planning_an_investment_strategy {}

mod bitonic_euclidean_tsp {}

mod game_of_life {
    //! LC-289
    //! According to Wikipedia's article: "The Game of Life, also known simply as Life,
    //! is a cellular automaton devised by the British mathematician John Horton Conway in 1970."
    //!
    //! The board is made up of an m x n grid of cells, where each cell has an initial state:
    //! live (represented by a 1) or dead (represented by a 0). Each cell interacts with its
    //! eight neighbors (horizontal, vertical, diagonal) using the following four rules (taken from the above Wikipedia article):
    //!
    //! Any live cell with fewer than two live neighbors dies as if caused by under-population.
    //! Any live cell with two or three live neighbors lives on to the next generation.
    //! Any live cell with more than three live neighbors dies, as if by over-population.
    //! Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction.
    //!
    //! The next state is created by applying the above rules simultaneously to every cell in the current state,
    //! where births and deaths occur simultaneously.
    //!
    //! Given the current state of the m x n grid board, return the next state.
}

mod word_squares {
    //! A “word square” is an ordered sequence of K different words of length K that,
    //! when written one word per line, reads the same horizontally and vertically. For example:
    //!```
    //!    BALL
    //!    AREA
    //!    LEAD
    //!    LADY
    //! ```  
    //! In this exercise you're going to create a way to find word squares.
    //!
    //! First, design a way to return true if a given sequence of words is a word square.
    //! Second, given an arbitrary list of words, return all the possible word squares it contains. Reordering is allowed.
    //!
    //! For example, the input list
    //!    [AREA, BALL, DEAR, LADY, LEAD, YARD]
    //! should output
    //!    [(BALL, AREA, LEAD, LADY), (LADY, AREA, DEAR, YARD)]
    //! Finishing the first task should help you accomplish the second task.
    //!
    //! Learning objectives
    //!    This problem gives practice with algorithms, recursion,
    //!    arrays, progressively optimizing from an initial solution, and testing.

    /// Returns a collection of words all possible word squares in the collection
    fn generate_all_word_squares<'a>(collection: &[&'a str]) -> Vec<Vec<&'a str>> {
        // Check all subsequences of length collection[0].length
        // Check if each is a word square
        let word_squares = Vec::new();
        if let Some(&first_item) = collection.first() {
            let expected_square_size = first_item.len();
            todo!()
        }
        word_squares
    }

    /// returns true if the given collection of words forms a word square
    fn is_word_square(words: &[&str]) -> bool {
        let num_words = words.len();

        // If we have enough words to form a square, then lets
        // proceed to check if the square formed is a word square
        if num_words > 0 && num_words == words[0].len() {
            // To check if a list of words is a word square,
            // we proceed word by word and check if an
            // alignment of the other words can form the current
            // word
            for (idx, &word) in words.iter().enumerate() {
                let induced_word = induce_word_from_index(words, idx);
                if !induced_word.eq(word) {
                    return false;
                }
            }

            // If we were able to go through all words and validate
            // that the alignments all work out correctly,
            // then we are sure that this is a word square
            return true;
        }
        false
    }

    /// Generate the word that would be created if we took the `n`-th
    /// character from each word
    fn induce_word_from_index(words: &[&str], n: usize) -> String {
        let mut induced_word = String::with_capacity(words.len());
        for &word in words {
            let nth_char = word.chars().nth(n);
            debug_assert!(nth_char.is_some());
            induced_word.push(nth_char.unwrap());
        }
        induced_word
    }
}

mod volume_of_lakes {
    //! Imagine an island that is in the shape of a bar graph.
    //! When it rains, certain areas of the island fill up with rainwater to form lakes.
    //! Any excess rainwater the island cannot hold in lakes will run off the island to
    //! the west or east and drain into the ocean.
    //!
    //! Given an array of positive integers representing 2-D bar heights,
    //! design an algorithm (or write a function) that can compute the
    //! total volume (capacity) of water that could be held in all
    //! lakes on such an island given an array of the heights of the bars.
    //! Assume an elevation map where the width of each bar is 1.
    //!
    //! Example: Given [1,3,2,4,1,3,1,4,5,2,2,1,4,2,2],
    //!          return 15 (3 bodies of water with volumes of 1,7,7 yields total volume of 15)
    //!
    //!
    //! Learning objectives
    //!     This question offers practice with algorithms, data structures, Big-O,
    //!     defining functions, generalization, efficiency, time and space complexity,
    //!     and anticipating edge cases.
}

mod code_jam {
    //! Solutions to Google Codejam problems from 2008, 2009, 2010, and 2015
    mod standing_ovation {}

    mod ihop {}

    mod dijkstra {}

    mod ominous_omino {}

    mod fly_swatter {}

    mod saving_the_universe {}

    mod train_time_table {}

    mod minimum_scalar_product {}

    mod milkshakes {}

    mod numbers {}

    mod crop_triangles {}

    mod number_sets {}

    mod mouse_trap {}

    mod text_messaging_outrage {}

    mod ugly_numbers {}

    mod increasing_speed_limits {}

    mod cheating_a_boolean_tree {}

    mod triangle_areas {}

    mod star_wars {}

    mod perm_rle {}

    mod how_big_are_pockets {}

    mod endless_knight {}

    mod portal {}

    mod no_cheating {}

    mod what_are_birds {}

    mod apocalypse_soon {}

    mod millionaire {}

    mod modern_art_plagiarism {}

    mod mixing_bowls {}

    mod test_passing_probability {}

    mod code_sequence {}

    mod king {}

    mod painting_a_fence {}

    mod scaled_triangle {}

    mod rainbow_trees {}

    mod bus_stops {}

    mod juice {}

    mod ping_pong_balls {}

    mod mine_layer {}

    mod bridge_builders {}

    mod the_year_of_code_jam {}

    mod alien_language {}

    mod watersheds {}

    mod welcome_to_code_jam {}

    mod multi_base_happiness {}

    mod crossing_the_road {}
    mod collecting_cards {}

    mod decision_tree {}

    mod the_next_number {}

    mod square_math {}

    mod all_your_base {}

    mod center_of_mass {}

    mod bribe_the_prisoners {}

    mod fair_warning {}

    mod snapper_chain {}

    mod theme_park {}

    mod rotate {}

    mod make_it_smooth {}

    mod number_game {}

    mod file_fix_it {}

    mod picking_up_chicks {}

    mod your_rank_is_pure {}

    mod rope_intranet {}

    mod load_testing {}

    mod making_chess_boards {}

    mod elegant_diamond {}

    mod world_cup_2010 {}

    mod bacteria {}

    mod grazing_google_goats {}

    mod deranged {}

    mod hot_dog_proliferation {}

    mod different_sum {}

    mod fence {}

    mod candy_store {}

    mod city_tour {}

    mod letter_stamper {}

    mod travel_plan {}

    mod ninjutsu {}

    mod the_paths_of_yin_yang {}

    mod mushroom_monster {
        //! Kaylin loves mushrooms. Put them on her plate and she'll eat them up!
        //! n this problem she's eating a plate of mushrooms, and
        //! Bartholomew is putting more pieces on her plate.
        //!
        //! In this problem, we'll look at how many pieces of mushroom are on her plate at 10-second intervals.
        //! Bartholomew could put any non-negative integer number of mushroom pieces down at any time,
        //! and the only way they can leave the plate is by being eaten.
        //!
        //! Figure out the minimum number of mushrooms that Kaylin could have eaten
        //! using two different methods of computation:
        //!    Assume Kaylin could eat any number of mushroom pieces at any time.
        //!    Assume that, starting with the first time we look at the plate, Kaylin eats mushrooms at a constant rate whenever there are mushrooms on her plate.
        //!
        //! For example, if the input is 10 5 15 5:
        //!    With the first method, Kaylin must have eaten at least 15 mushroom pieces:
        //!        first she eats 5, then 10 more are put on her plate,
        //!        then she eats another 10.
        //!    There's no way she could have eaten fewer pieces.
        //!
        //!    With the second method, Kaylin must have eaten at least 25 mushroom pieces.
        //!        We can determine that she must eat mushrooms at a rate of at least 1 piece per second.
        //!        She starts with 10 pieces on her plate. In the first 10 seconds, she eats 10 pieces,
        //!        and 5 more are put on her plate. In the next 5 seconds, she eats 5 pieces,
        //!        then her plate stays empty for 5 seconds, and then Bartholomew puts 15 more pieces on her plate.
        //!        Then she eats 10 pieces in the last 10 seconds.
        //!
        //! Input
        //!    The first line of the input gives the number of test cases, T.
        //!    T test cases follow. Each will consist of one line containing a single integer N,
        //!    followed by a line containing N space-separated integers mi; the number of mushrooms o
        //!    n Kaylin's plate at the start, and at 10-second intervals.
        //!
        //! Output
        //!    For each test case, output one line containing "Case #x: y z",
        //!    where x is the test case number (starting from 1), y
        //!    is the minimum number of mushrooms Kaylin could have eaten using the first method of computation,
        //!    and z is the minimum number of mushrooms Kaylin could have eaten using the second method of computation.
    }

    mod haircut {
        //! You are waiting in a long line to get a haircut at a trendy barber shop.
        //! The shop has B barbers on duty, and they are numbered 1 through B.
        //! It always takes the kth barber exactly Mk minutes to cut a customer's hair,
        //! and a barber can only cut one customer's hair at a time.
        //! Once a barber finishes cutting hair, he is immediately free to help another customer.
        //!
        //! While the shop is open, the customer at the head of the queue always goes to the
        //! lowest-numbered barber who is available. When no barber is available,
        //! that customer waits until at least one becomes available.
        //!
        //! You are the Nth person in line, and the shop has just opened. Which barber will cut your hair?
        //! Input
        //!    The first line of the input gives the number of test cases, T.
        //!    T test cases follow; each consists of two lines.
        //!        The first contains two space-separated integers B and N -- the number of barbers and your place in line.
        //!        The customer at the head of the line is number 1, the next one is number 2, and so on.
        //!    The second line contains M1, M2, ..., MB.
        //!
        //!Output
        //!    For each test case, output one line containing "Case #x: y",
        //!    where x is the test case number (starting from 1)
        //!    and y is the number of the barber who will cut your hair.
    }

    mod convex_hull {
        //! A certain forest consists of N trees, each of which is inhabited by a squirrel.
        //!
        //! The boundary of the forest is the convex polygon of smallest area which contains every tree,
        //! as if a giant rubber band had been stretched around the outside of the forest.
        //!
        //! Formally, every tree is a single point in two-dimensional space with unique coordinates (Xi, Yi),
        //! and the boundary is the convex hull of those points.
        //!
        //! Some trees are on the boundary of the forest, which means they are on an edge or a corner of the polygon.
        //! The squirrels wonder how close their trees are to being on the boundary of the forest.
        //!
        //! One at a time, each squirrel climbs down from its tree, examines the forest,
        //! and determines the minimum number of trees that would need to be cut down for
        //! its own tree to be on the boundary. It then writes that number down on a log.
        //!
        //! Determine the list of numbers written on the log.
        //!
        //! Input
        //!    The first line of the input gives the number of test cases, T.
        //!    T test cases follow; each consists of a single line with an integer N,
        //!    the number of trees, followed by N lines with two space-separated integers Xi and Yi,
        //!    the coordinates of each tree. No two trees will have the same coordinates.
        //!
        //!Output
        //!    For each test case, output one line containing "Case #x:",
        //!    followed by N lines with one integer each, where line i contains the
        //!    number of trees that the squirrel living in tree i would need to cut down.
    }

    mod counter_culture {
        //! In the Counting Poetry Slam, a performer takes the microphone, chooses a number N,
        //! and counts aloud from 1 to N. That is, she starts by saying 1, and then
        //! repeatedly says the number that is 1 greater than the previous
        //! number she said, stopping after she has said N.
        //!
        //! It's your turn to perform, but you find this process tedious,
        //! and you want to add a twist to speed it up: sometimes,
        //! instead of adding 1 to the previous number, you might reverse the digits of the number
        //! (removing any leading zeroes that this creates). For example, after saying "16",
        //! you could next say either "17" or "61"; after saying "2300",
        //! you could next say either "2301" or "32". You may reverse as many times as you want (or not at all) within a performance.
        //!
        //! The first number you say must be 1; what is the fewest number of
        //! numbers you will need to say in order to reach the number N? 1 and N count
        //! toward this total. If you say the same number multiple times, each of those times counts separately.
        //!
        //! Input
        //!    The first line of the input gives the number of test cases, T. T lines follow. Each has one integer N, the number you must reach.
        //!
        //! Output
        //!    For each test case, output one line containing "Case #x: y",
        //!    where x is the test case number (starting from 1) and y is the minimum number of numbers you need to say.
    }

    mod noisy_neighbors {
        //! You are a landlord who owns a building that is an R x C grid of apartments;
        //! each apartment is a unit square cell with four walls. You want to rent out N of
        //! these apartments to tenants, with exactly one tenant per apartment, and leave the others empty.
        //! Unfortunately, all of your potential tenants are noisy, so whenever any two occupied apartments
        //! share a wall (and not just a corner), this will add one point of unhappiness to the building.
        //! For example, a 2x2 building in which every apartment is occupied has four walls that are
        //! shared by neighboring tenants, and so the building's unhappiness score is 4.
        //!
        //! If you place your N tenants optimally, what is the minimum unhappiness value for your building?
        //!
        //! Input
        //!    The first line of the input gives the number of test cases, T.
        //!    T lines follow; each contains three space-separated integers: R, C, and N.
        //!
        //! Output
        //!    For each test case, output one line containing "Case #x: y", where x is
        //!    the test case number (starting from 1) and y is the minimum possible unhappiness for the building.
    }

    mod hiking_deer {
        //! Herbert Hooves the deer is going for a hike:
        //!    one clockwise loop around his favorite circular trail, starting at degree zero.
        //! Herbert has perfect control over his speed, which can be any nonnegative value (not necessarily an integer)
        //! at any time -- he can change his speed instantaneously whenever he wants.
        //! When Herbert reaches his starting point again, the hike is over.
        //!
        //! The trail is also used by human hikers, who also walk clockwise around the trail.
        //! Each hiker has a starting point and moves at her own constant speed.
        //! Humans continue to walk around and around the trail forever.
        //!
        //! Herbert is a skittish deer who is afraid of people. He does not like to have encounters with hikers.
        //! An encounter occurs whenever Herbert and a hiker are in exactly the same place at the same time.
        //! You should consider Herbert and the hikers to be points on the circumference of a circle.
        //!
        //! Herbert can have multiple separate encounters with the same hiker.
        //!    If more than one hiker is encountered at the same instant, all of them count as separate encounters.
        //!    Any encounter at the exact instant that Herbert finishes his hike still counts as an encounter.
        //!    If Herbert were to have an encounter with a hiker and then change his speed to exactly match that hiker's speed and follow along, he would have infinitely many encounters! Of course, he must never do this.
        //!    Encounters do not change the hikers' behavior, and nothing happens when hikers encounter each other.
        //!
        //! Herbert knows the starting position and speed of each hiker.
        //! What is the minimum number of encounters with hikers that he can possibly have?
        //!
        //! Solving this problem
        //! Usually, Google Code Jam problems have 1 Small input and 1 Large input.
        //! This problem has 2 Small inputs and 1 Large input.
        //! You must solve the first Small input before you can attempt the second Small input;
        //! as usual, you will be able to retry the Small inputs (with a time penalty).
        //! Once you have solved both Small inputs, you will be able to download the Large input;
        //! as usual, you will get only one chance at the Large input.
        //!
        //! Input
        //!    The first line of the input gives the number of test cases, T.
        //!    T test cases follow. Each begins with one line with an integer N,
        //!    and is followed by N lines, each of which represents a group of hikers starting
        //!    at the same position on the trail. The ith of these lines has three space-separated integers:
        //!    a starting position Di (representing Di/360ths of the way around the trail from the deer's starting point),
        //!    the number Hi of hikers in the group, and Mi, the amount of time (in minutes)
        //!    it takes for the fastest hiker in that group to make each complete revolution around the circle.
        //!    The other hikers in that group each complete a revolution in Mi+1, Mi+2, ..., Mi+Hi-1 minutes. For example, the line
        //!
        //! 180 3 4
        //!    would mean that three hikers begin halfway around the trail from the deer's starting point,
        //!    and that they take 4, 5, and 6 minutes, respectively, to complete each full revolution around the trail.
        //!
        //! Herbert always starts at position 0 (0/360ths of the way around the circle), and no group of hikers does.
        //! Multiple groups of hikers may begin in the same place, but no two hikers will both begin in the same place and have the same speed.
        //!
        //!Output
        //!    For each test case, output one line containing "Case #x: y", where x is the test case number (starting from 1) and y
        //!    is the minimum number of encounters with hikers that the deer can have.
    }

    mod type_writer_monkey {}

    mod brattle_ship {}

    mod less_money_more_problems {}

    mod pegman {}

    mod kiddie_pool {}

    mod bilingual {}

    mod drum_decorator {}

    mod fair_land {}

    mod smoothing_window {}

    mod runaway_quail {}

    mod log_set {}

    mod river_flow {}

    mod compinatorics {}

    mod costly_binary_search {}

    mod pretty_good_proportion {}

    mod taking_over_the_world {}

    mod merlin_qa {}

    mod crane_truck {}
}
