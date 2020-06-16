# Numbers recognition

## Archeticture

Inspired by [End-to-end-ASR-Pytorch](https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch)


The main class is ASR model, that contains VGGExtractor, that extract high-level features from MELs, then I apply GRU to these features, and after that get 6 self-attention vectors for every digit in a number. These vectors pass to Linear layer to predict every digit in a number.


## How to run
`python3 input.csv output.csv`