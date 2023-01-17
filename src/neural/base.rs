use crate::types::Amount;

pub trait NeuralObject {
    fn input_size(&self) -> usize;
    fn apply_input(&mut self, inputs: &[Amount]);
    fn tick(&mut self, duration_secs: f64);
    fn get_output(&self) -> &[Amount];
    fn reward(&mut self, reward: Amount);
}
