use crate::models::{Candidate, Chunk, Query};
use crate::retrievers::Retriever;
use anyhow::Result;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Schema, STORED, TEXT};
use tantivy::{Index, IndexWriter, TantivyDocument};
use tantivy::schema::Value;

pub struct Bm25Retriever {
    id: String,
    index: Index,
    schema: Schema,
    chunks: Vec<Chunk>,
}

impl Bm25Retriever {
    pub fn new(id: &str) -> Result<Self> {
        let mut schema_builder = Schema::builder();
        schema_builder.add_text_field("body", TEXT | STORED);
        schema_builder.add_text_field("chunk_id", STORED);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema.clone());

        Ok(Self {
            id: id.to_string(),
            index,
            schema,
            chunks: Vec::new(),
        })
    }

    pub fn index(&mut self, chunks: Vec<Chunk>) -> Result<()> {
        let mut writer: IndexWriter = self.index.writer(50_000_000)?;
        let body_field = self.schema.get_field("body").unwrap();
        let id_field = self.schema.get_field("chunk_id").unwrap();

        for chunk in &chunks {
            let mut doc = TantivyDocument::default();
            doc.add_text(body_field, &chunk.text);
            doc.add_text(id_field, &chunk.id);
            writer.add_document(doc)?;
        }

        writer.commit()?;
        self.chunks = chunks;
        Ok(())
    }
}

impl Retriever for Bm25Retriever {
    fn id(&self) -> &str {
        &self.id
    }

    fn retrieve(&self, query: &Query, top_k: usize) -> Result<Vec<Candidate>> {
        let reader = self.index.reader()?;
        let searcher = reader.searcher();
        let body_field = self.schema.get_field("body").unwrap();
        let id_field = self.schema.get_field("chunk_id").unwrap();
        let query_parser = QueryParser::for_index(&self.index, vec![body_field]);
        let sanitized: String = query.text
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect();
        let tantivy_query = query_parser.parse_query(&sanitized)?;
        let top_docs = searcher.search(&tantivy_query, &TopDocs::with_limit(top_k))?;

        let mut candidates = Vec::new();
        for (score, doc_address) in top_docs {
            let doc: TantivyDocument = searcher.doc(doc_address)?;
            let chunk_id = doc
                .get_first(id_field)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            if let Some(chunk) = self.chunks.iter().find(|c| c.id == chunk_id) {
                candidates.push(Candidate {
                    chunk: chunk.clone(),
                    score: score as f32,
                    retriever_id: self.id.to_string(),
                });
            }
        }

        Ok(candidates)
    }
}
