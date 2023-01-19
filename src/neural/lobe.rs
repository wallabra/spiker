use super::base::NeuralObject;
use crate::types::Amount;
use itertools::izip;
use std::slice::{self, Chunks, ChunksMut};

/// A rectangular cluster of spiking neurons.
#[derive(Clone, Eq, PartialEq)]
pub struct Lobe {
    dims: (usize, usize),
    values: Vec<Amount>,
    strengths: Vec<Amount>,
    weights: Vec<Amount>,
    thresholds: Vec<Amount>,
    falloff: Amount,
}

impl Lobe {
    /// Create a new Lobe from a pair of dimensions and a falloff value.
    pub fn new(breadth: usize, width: usize, falloff: Amount) -> Self {
        Lobe {
            dims: (width, breadth),
            values: vec![Amount::from_num(0); breadth * (width + 1)],
            weights: vec![Amount::from_num(0); breadth * width * 3],
            strengths: vec![Amount::from_num(0); breadth * width],
            thresholds: vec![Amount::from_num(0); breadth * width],
            falloff,
        }
    }

    /// References a column of the Lobe's values.
    pub fn value_column_ref(&self, which: usize) -> &[Amount] {
        &self.values[which * self.dims.1..(which + 1) * self.dims.1]
    }

    /// Mutably references a column of the Lobe's values.
    pub fn value_column_mut(&mut self, which: usize) -> &mut [Amount] {
        &mut self.values[which * self.dims.1..(which + 1) * self.dims.1]
    }

    /// Iterates on the columns of the Lobe's values.
    pub fn values_chunked(&self) -> Chunks<Amount> {
        self.values.chunks(self.dims.1)
    }

    /// Mutably iterates on the columns of the Lobe's values.
    pub fn values_chunked_mut(&mut self) -> ChunksMut<Amount> {
        self.values.chunks_mut(self.dims.1)
    }

    /// References a column of the Lobe's firing strengths.
    pub fn strength_column_ref(&self, which: usize) -> &[Amount] {
        &self.strengths[which * self.dims.1..(which + 1) * self.dims.1]
    }

    /// Mutably references a column of the Lobe's firing strengths.
    pub fn strength_column_mut(&mut self, which: usize) -> &mut [Amount] {
        &mut self.strengths[which * self.dims.1..(which + 1) * self.dims.1]
    }

    /// Iterates on the columns of the Lobe's firing strengths.
    pub fn strengths_chunked(&self) -> Chunks<Amount> {
        self.strengths.chunks(self.dims.1)
    }

    /// References a column of the Lobe's firing thresholds.
    pub fn threshold_column_ref(&self, which: usize) -> &[Amount] {
        &self.thresholds[which * self.dims.1..(which + 1) * self.dims.1]
    }

    /// Mutably references a column of the Lobe's firing thresholds.
    pub fn threshold_column_mut(&mut self, which: usize) -> &mut [Amount] {
        &mut self.thresholds[which * self.dims.1..(which + 1) * self.dims.1]
    }

    /// Iterates on the columns of the Lobe's firing thresholds.
    pub fn thresholds_chunked(&self) -> Chunks<Amount> {
        self.thresholds.chunks(self.dims.1)
    }

    /// References a column of the Lobe's forward weights.
    pub fn weight_column_ref(&self, which: usize) -> &[Amount] {
        &self.weights[which * 3 * self.dims.1..(which + 1) * 3 * self.dims.1]
    }

    /// Mutably references a column of the Lobe's forward weights.
    pub fn weight_column_mut(&mut self, which: usize) -> &mut [Amount] {
        &mut self.weights[which * 3 * self.dims.1..(which + 1) * 3 * self.dims.1]
    }

    /// Iterates on the columns of the Lobe's forward weights.
    pub fn weight_column_chunks(&self, which: usize) -> std::slice::Chunks<Amount> {
        self.weight_column_ref(which).chunks(3)
    }

    /// Returns mutable slices into all parameters, useful for training.
    pub fn all_parameters_slices<'a>(&'a mut self) -> Vec<&'a mut [Amount]> {
        vec![
            &mut self.weights,
            &mut self.thresholds,
            &mut self.strengths,
            slice::from_mut(&mut self.falloff),
        ]
    }

    /// Returns owned vector copies of all parameters, useful for training.
    pub fn all_parameters_owned(&self) -> Vec<Amount> {
        let mut vec = vec![];

        vec.extend(&self.thresholds);
        vec.extend(&self.weights);
        vec.extend(&self.strengths);
        vec.push(self.falloff);

        vec
    }

    /// Returns the dimensions of this Lobe, useful for training.
    pub fn get_dims(&self) -> (usize, usize) {
        self.dims
    }

    /// Creates a new Lobe from a list of parameters, loaded in the same order
    /// they would be concatenated in [all_parameters_owned].
    pub fn from_parameters(dims: (usize, usize), params: &[Amount]) -> Self {
        let area = dims.0 * dims.1;

        debug_assert!(params.len() == area * 5 + 1);

        Self {
            dims,
            thresholds: params[0..area].to_vec(),
            weights: params[area..area * 4].to_vec(),
            strengths: params[area * 4..area * 5].to_vec(),
            falloff: *params.last().unwrap(),
            values: vec![Amount::from_num(0); area + dims.0],
        }
    }
}

impl NeuralObject for Lobe {
    fn input_size(&self) -> usize {
        self.dims.1
    }

    fn apply_input(&mut self, inputs: &[Amount]) {
        self.value_column_mut(0)
            .iter_mut()
            .zip(inputs)
            .for_each(|(into, from)| *into += *from)
    }

    fn tick(&mut self, duration_secs: f64) {
        let duration_secs = Amount::from_num(duration_secs);
        let breadth = self.dims.1;
        let area = self.dims.1 * self.dims.0;

        let mut outputs = vec![Amount::from_num(0.0); area];

        for (value_source, weights, strengths, thresholds, value_sink) in izip!(
            self.values_chunked(),
            self.weights.chunks(self.dims.1 * 3),
            self.strengths_chunked(),
            self.thresholds_chunked(),
            outputs.chunks_mut(breadth),
        ) {
            for offset in 0..=2 {
                let to_skip_output = (offset as isize - 1).max(0) as usize;
                let to_skip_input = (1 - offset as isize).max(0) as usize;

                let weights_iter = weights.chunks(3).skip(to_skip_input);

                for (input, weight_chunk, strength, threshold, output) in izip!(
                    value_source.iter().skip(to_skip_input),
                    weights_iter,
                    strengths.iter().skip(to_skip_input),
                    thresholds.iter().skip(to_skip_input),
                    value_sink.iter_mut().skip(to_skip_output),
                ) {
                    let weight = weight_chunk[offset];

                    *output += if *input < *threshold {
                        Amount::from_num(0)
                    } else {
                        input * weight * strength * duration_secs
                    };
                }
            }
        }

        for (value, threshold) in izip!(&mut self.values, &self.thresholds) {
            if *value >= *threshold {
                *value = Amount::from_num(0);
            }
        }

        for (into, from) in izip!(&mut self.values[breadth..], &outputs) {
            *into += *from;
        }

        for value in &mut self.values {
            *value -= *value * self.falloff * duration_secs;
        }
    }

    fn get_output(&self) -> &[Amount] {
        self.value_column_ref(self.dims.0)
    }

    fn reward(&mut self, _reward: Amount) {
        // TODO
    }
}
