use benches::{Acceleration, Candle, Ggml, Ort};

fn main() {
    let seq = "My favourite animal is the dog";
    let ort = Ort::new().unwrap();
    let ggml = Ggml::new(Acceleration::Gpu);
    let mut candle = Candle::new(true).unwrap();

    let o = ort.embed(seq).unwrap();
    let g = ggml.embed(seq).unwrap();
    let c = candle.embed(seq).unwrap();

    println!("ort : {:.04?}", &o[..10]);
    println!("ggml: {:.04?}", &g[..10]);
    println!("cndl: {:.04?}", &c[..10]);
}
