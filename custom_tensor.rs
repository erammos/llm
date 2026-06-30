#[derive(Clone, Debug)]
struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}
impl Tensor {
    pub fn new(data: Vec<f32>, shape: &[usize]) -> Self {
        let mut strides = Vec::with_capacity(shape.len());

        let mut curr = 1;
        for dim in shape.iter().rev() {
            strides.push(curr);
            curr *= dim;
        }
        strides.reverse();

        Self {
            data,
            shape: shape.to_vec(),
            strides,
        }
    }
    pub fn from_random(shape: &[usize]) -> Self {
        let num_weights = shape.iter().fold(1, |acc, dim| acc * dim);
        //TODO fill random
        Tensor::new(vec![0.1; num_weights], shape)
    }

    pub fn get(&self, targets: &[usize]) -> f32 {
        let mut flat_index = 0;
        for (i, t) in targets.iter().enumerate() {
            flat_index += t * self.strides[i];
        }
        self.data[flat_index]
    }

    pub fn matmul2d(&self, t: &Tensor) -> Self {
        assert!(t.shape.len() == 2);
        assert!(self.shape.len() == 2);
        assert!(self.shape[1] == t.shape[0]);
        let rows = self.shape[0];
        let cols = t.shape[1];

        let mut data: Vec<f32> = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                let mut sum = 0.0;
                for i in 0..self.shape[1] {
                    sum += self.get(&[r, i]) * t.get(&[i, c]);
                }
                data.push(sum);
            }
        }

        Self::new(data, &[rows, cols])
    }
    pub fn tranpose(&mut self) {}
    pub fn add(&self, t: &Tensor) -> Option<Self> {
        if self.shape == t.shape {
            let mut tensor: Tensor = self.clone();
            for i in 0..tensor.data.len() {
                tensor.data[i] += t.data[i];
            }
            return Some(tensor);
        } else if self.shape.len() == 2 && t.shape.len() == 1 {
            assert!(self.shape[1] == t.shape[0]);
            let mut tensor: Tensor = self.clone();
            for r in 0..self.shape[0] {
                for c in 0..self.shape[1] {
                    tensor.data[r * self.shape[1] + c] += t.get(&[c]);
                }
            }
            return Some(tensor);
        }
        None
    }
}
