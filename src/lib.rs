use ndarray::Axis;
use ort::{
    tensor::{FromArray, InputTensor, OrtOwnedTensor},
    Environment, ExecutionProvider, GraphOptimizationLevel, LoggingLevel, SessionBuilder,
};
use std::{path::PathBuf, sync::Arc};

pub struct Ort {
    tokenizer: tokenizers::Tokenizer,
    session: ort::Session,
}

impl Ort {
    pub fn new() -> anyhow::Result<Self> {
        let tokenizer = tokenizers::Tokenizer::from_file("model/tokenizer.json").unwrap();
        let environment = Arc::new(
            Environment::builder()
                .with_name("Encode")
                .with_log_level(LoggingLevel::Warning)
                .with_execution_providers([ExecutionProvider::cpu()])
                .with_telemetry(false)
                .build()?,
        );

        let session = SessionBuilder::new(&environment)
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .with_model_from_file("model/model.onnx")?;

        Ok(Self { tokenizer, session })
    }

    pub fn embed(&self, sequence: &str) -> anyhow::Result<Vec<f32>> {
        let tokenizer_output = self.tokenizer.encode(sequence, true).unwrap();

        let input_ids = tokenizer_output.get_ids();
        let attention_mask = tokenizer_output.get_attention_mask();
        let token_type_ids = tokenizer_output.get_type_ids();
        let length = input_ids.len();
        // trace!("embedding {} tokens {:?}", length, sequence);

        let inputs_ids_array = ndarray::Array::from_shape_vec(
            (1, length),
            input_ids.iter().map(|&x| x as i64).collect(),
        )?;

        let attention_mask_array = ndarray::Array::from_shape_vec(
            (1, length),
            attention_mask.iter().map(|&x| x as i64).collect(),
        )?;

        let token_type_ids_array = ndarray::Array::from_shape_vec(
            (1, length),
            token_type_ids.iter().map(|&x| x as i64).collect(),
        )?;

        let outputs = self.session.run([
            InputTensor::from_array(inputs_ids_array.into_dyn()),
            InputTensor::from_array(attention_mask_array.into_dyn()),
            InputTensor::from_array(token_type_ids_array.into_dyn()),
        ])?;

        let output_tensor: OrtOwnedTensor<f32, _> = outputs[0].try_extract().unwrap();
        let sequence_embedding = &*output_tensor.view();
        let pooled = sequence_embedding.mean_axis(Axis(1)).unwrap();
        Ok(pooled.to_owned().as_slice().unwrap().to_vec())
    }
}

pub struct Ggml {
    model: Box<dyn llm::Model>,
}

pub enum Acceleration {
    None,
    Gpu,
}

impl Ggml {
    pub fn new(acc: Acceleration) -> Self {
        let tokenizer_source =
            llm::TokenizerSource::HuggingFaceTokenizerFile(PathBuf::from("model/tokenizer.json"));
        let model_architecture = llm::ModelArchitecture::Bert;
        let model_path = PathBuf::from("model/ggml-model-q4_0.bin");

        // Load model
        let mut model_params = llm::ModelParameters::default();
        model_params.context_size = 256;
        model_params.use_gpu = match acc {
            Acceleration::None => false,
            Acceleration::Gpu => true,
        };
        let model = llm::load_dynamic(
            Some(model_architecture),
            &model_path,
            tokenizer_source,
            model_params,
            llm::load_progress_callback_stdout,
        )
        .unwrap_or_else(|err| {
            panic!("Failed to load {model_architecture} model from {model_path:?}: {err}")
        });

        Self { model }
    }

    pub fn embed(&self, sequence: &str) -> anyhow::Result<Vec<f32>> {
        let session_config = llm::InferenceSessionConfig {
            ..Default::default()
        };
        let mut session = self.model.start_session(session_config);
        let mut output_request = llm::OutputRequest {
            all_logits: None,
            embeddings: Some(Vec::new()),
        };
        let vocab = self.model.tokenizer();
        let beginning_of_sentence = true;
        let query_token_ids = vocab
            .tokenize(sequence, beginning_of_sentence)?
            .iter()
            .map(|(_, tok)| *tok)
            .collect::<Vec<_>>();
        self.model
            .evaluate(&mut session, &query_token_ids, &mut output_request);
        output_request
            .embeddings
            .ok_or(anyhow::anyhow!("failed to embed"))
    }
}
