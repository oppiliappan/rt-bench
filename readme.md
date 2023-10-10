# rt-bench

`rt-bench` contains benchmark code to compare the following ML frameworks:

- ggml
- ggml on gpu
- onnx
- candle

## Usage

Clone this repository with `git-lfs` installed:

```bash
git clone https://github.com/bloopai/rt-bench
```

This should download necessary model & tokenizer files. You can now run
benchmarks with:

```bash
cargo bench
```

The generated violin plots are stored under `target/criterion/embedding`.
