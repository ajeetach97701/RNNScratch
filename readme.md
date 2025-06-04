# RNN Scratch Implementation (Feed Forward + Backpropagation Through Time)

This repository contains a scratch implementation of a Recurrent Neural Network (RNN), including both the feed forward pass and backpropagation through time (BPTT).

## Structure

- **Data/clean_weather.csv**: Contains the datasets used for training and testing.
- **src/FeedForward.py**: Core implementation of the RNN feed forward logic.
- **src/BackPropThroughTime.py**: Core implementation of the RNN backpropagation through time (BPTT).
- **src/BackProp/**: Contains utility modules for gradient and weight update logic.
- **test.ipynb**: Jupyter notebooks to run and test both feed forward and backpropagation implementations.

## Usage

1. Data file is located in the `Data/` directory.
2. Open and run `test.ipynb` to execute the feed forward and BPTT processes using the implementations in `src/`.

## Requirements

- Python 3.11 (the version in which this file was built)
- Jupyter Notebook
- (Optional) Any dependencies listed in `requirements.txt`

## Notes

- This implementation now includes both feed forward and backpropagation for a simple RNN.
- Designed for educational purposes to understand the internals of RNN training and inference.
