use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use benches::{Acceleration, Candle, Ggml, Ort};

static CHUNKS: [&str; 10] = [
  "impl RepoFile {\n    #[allow(clippy::too_many_arguments)]\n    fn build_document(\n        mut self,\n        schema: &File,\n        repo_name: &str,\n        relative_path: &Path,\n        repo_disk_path: &Path,\n        semantic_cache_key: String,\n        tantivy_cache_key: String,\n        entry_pathbuf: &Path,\n        repo_ref: &str,\n        last_commit: u64,\n        repo_metadata: &RepoMetadata,\n        file_cache: &FileCache,\n    ) -> Option<tantivy::schema::Document> {\n        let relative_path_str = relative_path.to_string_lossy().to_string();\n        #[cfg(windows)]\n        let relative_path_str = relative_path_str.replace('\\\\', \"/\");\n\n        let branches = self.branches.join(\"\\n\");\n        let lang_str = repo_metadata\n            .langs\n            ",
  "let path = source.directory().join(name).with_extension(\"json\");\n        Ok(Self {\n            state: Arc::new(read_file_or_default(&path)?),\n            path,\n        })\n    }\n\n    fn load_or(name: &'static str, source: &StateSource, val: T) -> Self {\n        let path = source.directory().join(name).with_extension(\"json\");\n        let new = Self {\n            state: Arc::new(read_file(&path).unwrap_or(val)),\n            path,\n        };\n\n        new.store().unwrap();\n        new\n    }\n\n    pub fn store(&self) -> Result<()> {\n        Ok(pretty_write_file(&self.path, self.state.as_ref())?)\n    }\n}\n\nimpl<T> Deref for PersistedState<T> {\n    type Target = T;\n\n    ",
  "protected loadValidator(\n    validatorPackage?: ValidatorPackage,\n  ): ValidatorPackage {\n    return (\n      validatorPackage ??\n      loadPackage('class-validator', 'ValidationPipe', () =>\n        require('class-validator'),\n      )\n    );\n  }\n\n  protected loadTransformer(\n    transformerPackage?: TransformerPackage,\n  ): TransformerPackage {\n    return (\n      transformerPackage ??\n      loadPackage('class-transformer', 'ValidationPipe', () =>\n        require('class-transformer'),\n      )\n    );\n  }\n\n  public async transform(value: any, metadata: ArgumentMetadata) {\n    if (this.expectedType) {\n      metadata = { ...metadata, metatype: this.expectedType };\n    }\n\n    const metatype = metadata.metatype;\n    if (!metatype || !this.toValidate(metadata)) {\n      return this.isTransformEnabled\n        ? this.transformPrimitive(value, metadata)\n        : value;\n    }\n    const originalValue = value;\n    ",
  "}\n                      />\n                    );\n                  })}\n                </div>\n              )}\n            {filters.includes(TypeMap.REF) && !!data.data.references.length && (\n              <div>\n                <div className=\"bg-bg-base flex gap-1 items-center py-2 px-3 text-bg-danger select-none\">\n                  <Ref raw sizeClassName=\"w-3.5 h-3.5\" />\n                  <p className=\"caption text-label-base\">\n                    <Trans>References</Trans>\n                  </p>\n                </div>\n                {data.data.references.map((item, i) => {\n                  return (\n                    <RefDefItem\n                      onRefDefClick={onRefDefClick}\n                      data={item.data}\n                      file={item.file}\n                      relativePath={relativePath}\n                      repoName={repoName}\n                      language={language}\n                      ",
  "export { default as ArrowPushTop } from './ArrowPushTop';\nexport { default as ArrowRevert } from './ArrowRevert';\nexport { default as ArrowRight } from './ArrowRight';\nexport { default as ArrowUp } from './ArrowUp';\nexport { default as Branch } from './Branch';\nexport { default as Bug } from './Bug';\nexport { default as ChatBubble } from './ChatBubble';\nexport { default as CheckIcon } from './CheckIcon';\nexport { default as ChevronDoubleIntersected } from './ChevronDoubleIntersected';\nexport { default as ChevronDown } from './ChevronDown';\nexport { default as ChevronDownFilled } from './ChevronDownFilled';\nexport { default as ChevronFoldIn } from './ChevronFoldIn';\nexport { default as ChevronFoldOut } from './ChevronFoldOut';\n",
  "{\n  font-family: 'Inter', sans-serif;\n  font-style: normal;\n  font-weight: 600;\n  font-size: 3rem;\n  line-height: 110%;\n  letter-spacing: -0.02em;\n\n  -webkit-user-select: none;\n  -moz-user-select: none;\n  user-select: none;\n  cursor: default;\n}\n\nh2, .h2 {\n  font-family: 'Inter', sans-serif;\n  font-style: normal;\n  font-weight: 600;\n  font-size: 2.375rem;\n  line-height: 110%;\n  letter-spacing: -0.02em;\n\n  -webkit-user-select: none;\n  -moz-user-select: none;\n  user-select: none;\n  cursor: default;\n}\n\nh3, .h3 {\n  font-family: 'Inter', sans-serif;\n  font-style: normal;\n  font-weight: 600;\n  font-size: 1.75rem;\n  line-height: 110%;\n  letter-spacing: -0.02em;\n\n  ",
  "service = module.get(CatsService);\n  });\n\n  it('should be defined', () => {\n    expect(controller).toBeDefined();\n  });\n\n  describe('create()', () => {\n    it('should create a new cat', async () => {\n      const createCatDto: CreateCatDto = {\n        name: 'Cat #1',\n        breed: 'Breed #1',\n        age: 4,\n      };\n\n      expect(controller.create(createCatDto)).resolves.toEqual({\n        _id: '1',\n        ...createCatDto,\n      });\n    });\n  });\n\n  describe('findAll()', () => {\n    it('should get an array of cats', () => {\n      expect(controller.findAll()).resolves.toEqual([\n        {\n          name: 'Cat #1',\n          breed: 'Bread #1',\n          age: 4,\n        },\n        {\n          ",
  "{\n    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>\n    where\n        D: Deserializer<'de>,\n    {\n        String::deserialize(deserializer).and_then(|s| {\n            RepoRef::from_str(s.as_str()).map_err(|e| D::Error::custom(e.to_string()))\n        })\n    }\n}\n\n#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone)]\n#[serde(rename_all = \"snake_case\")]\npub enum BranchFilter {\n    All,\n    Head,\n    Select(Vec<String>),\n}\n\nimpl BranchFilter {\n    pub(crate) fn patch(&self, old: Option<&BranchFilter>) -> Option<BranchFilter> {\n        let Some(BranchFilter::Select(ref old_list)) = old\n        else {\n\t    ",
  "import { REDIRECT_METADATA } from '../../constants';\n\n/**\n * Redirects request to the specified URL.\n *\n * @publicApi\n */\nexport function Redirect(url = '', statusCode?: number): MethodDecorator {\n  return (\n    target: object,\n    key: string | symbol,\n    descriptor: TypedPropertyDescriptor<any>,\n  ) => {\n    Reflect.defineMetadata(\n      REDIRECT_METADATA,\n      { statusCode, url },\n      descriptor.value,\n    );\n    return descriptor;\n  };\n}\n",
  "import { mapFileResult, mapRanges } from '../../mappers/results';\nimport { getHoverables } from '../../services/api';\nimport { buildRepoQuery } from '../../utils';\nimport { SearchContext } from '../../context/searchContext';\nimport { FileModalContext } from '../../context/fileModalContext';\nimport { AppNavigationContext } from '../../context/appNavigationContext';\nimport ResultModal from './index';\n\ntype Props = {\n  repoName: string;\n};\n\nconst FileModalContainer = ({ repoName }: Props) => {\n  const { path, closeFileModalOpen, isFileModalOpen } =\n    useContext(FileModalContext);\n  "
];

pub fn bench_embedding(c: &mut Criterion) {
    let mut group = c.benchmark_group("embedding");
    let ort = Ort::new().unwrap();
    let mut ggml_gpu = Ggml::new(Acceleration::Gpu);
    let mut ggml = Ggml::new(Acceleration::None);
    let mut candle = Candle::new(true).unwrap();

    group.bench_with_input(
        BenchmarkId::new("ggml_gpu_batched", 0),
        &CHUNKS,
        |b, chunks| b.iter(|| ggml_gpu.batch_embed(chunks)),
    );

    group.bench_with_input(BenchmarkId::new("onnx", 0), &CHUNKS, |b, chunks| {
        b.iter(|| {
            for c in chunks {
                ort.embed(c);
            }
        })
    });

    group.bench_with_input(BenchmarkId::new("ggml_gpu", 0), &CHUNKS, |b, chunks| {
        b.iter(|| {
            for c in chunks {
                ggml_gpu.embed(c);
            }
        })
    });

    group.bench_with_input(BenchmarkId::new("ggml", 0), &CHUNKS, |b, chunks| {
        b.iter(|| {
            for c in chunks {
                ggml.embed(c);
            }
        })
    });

    group.bench_with_input(BenchmarkId::new("candle", 0), &CHUNKS, |b, chunks| {
        b.iter(|| {
            for c in chunks {
                candle.embed(c);
            }
        })
    });

    group.finish();
}

criterion_group!(benches, bench_embedding);
criterion_main!(benches);
