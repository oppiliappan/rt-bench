use benches::{Acceleration, Ggml};

fn main() {
    let seq = std::env::args().skip(1).next().unwrap().to_string();
    //    let seq = r#"
    // '    @VisibleForTesting
    //    public KafkaConsumer getOrCreateKafkaConsumer(KafkaConsumer existingConsumer, Properties consumerProperties,
    //NotificationType notificationType, int idxConsumer) {
    //        KafkaConsumer ret = existingConsumer;
    //
    //        try {
    //            if (ret == null || !isKafkaConsumerOpen(ret)) {
    //                String[] topics = CONSUMER_TOPICS_MAP.get(notificationType);
    //                String   topic  = topics[idxConsumer % topics.length];
    //
    //                LOG.debug("Creating new KafkaConsumer for topic : {}, index : {}", topic, idxConsumer);
    //
    //                ret = new KafkaConsumer(consumerProperties);
    //
    //                ret.subscribe(Arrays.asList(topic));
    //            }
    //        } catch (Exception ee) {
    //            LOG.error("Exception in getKafkaConsumer ", ee);
    //        }
    //
    //        return ret;
    //    }'
    //
    //"#;
    let mut ggml = Ggml::new(Acceleration::None);
    let g = ggml.embed(&seq).unwrap();
    println!("{:?}", &g[..]);
}
