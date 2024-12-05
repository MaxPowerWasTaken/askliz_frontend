from lancedb.pydantic import LanceModel, Vector
from config import embedding_model

class DocumentChunk(LanceModel):
    text: str = embedding_model.SourceField()
    section_title: str
    chunk_idx: int
    doc_name: str

class DocumentChunkLanceRecord(DocumentChunk):
    """LanceDB will create our vector embeddings for us, as long as we 
       provide a LanceModel schema which defines a 'vector' attr w/ a 
       suitable embedding model & ndims"""
    vector: Vector(embedding_model.ndims()) = embedding_model.VectorField()

class RetrievedDocumentChunk(DocumentChunkLanceRecord):
    relevance_score: float  # rename from _relevance_score to relevance_score
                            # to avoid certain 'magic' on handling of leading underscore attribs