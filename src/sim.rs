use benches::{Acceleration, Ggml, Ort};

fn main() {
    let seq = "My favourite animal is the dog";
    let ort = Ort::new().unwrap();
    let ggml = Ggml::new(Acceleration::Gpu);

    let o = ort.embed(seq).unwrap();
    let g = ggml.embed(seq).unwrap();

    println!("ort : {:.04?}", &o[..10]);
    println!("ggml: {:.04?}", &g[..10]);
}
