-- ============================================================================
-- CREATE VECTOR SIMILARITY SEARCH FUNCTION
-- RPC function for finding similar content chunks using vector embeddings
-- ============================================================================

BEGIN;

-- Create the vector similarity search function
CREATE OR REPLACE FUNCTION match_content_chunks(
  query_embedding vector(1536),
  website_id uuid,
  match_threshold float DEFAULT 0.7,
  match_count int DEFAULT 5
)
RETURNS TABLE (
  id uuid,
  chunk_text text,
  chunk_index int,
  scraped_page_id uuid,
  similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    scraped_content_chunks.id,
    scraped_content_chunks.chunk_text,
    scraped_content_chunks.chunk_index,
    scraped_content_chunks.scraped_page_id,
    1 - (scraped_content_chunks.embedding_vector <=> query_embedding) as similarity
  FROM scraped_content_chunks
  WHERE
    scraped_content_chunks.website_id = match_content_chunks.website_id
    AND 1 - (scraped_content_chunks.embedding_vector <=> query_embedding) > match_threshold
  ORDER BY scraped_content_chunks.embedding_vector <=> query_embedding
  LIMIT match_count;
END;
$$;

COMMIT;