-- ============================================================================
-- ADD website_id TO scraped_content_chunks FOR FASTER QUERIES
-- This denormalizes the data structure but makes RAG queries much faster
-- ============================================================================

BEGIN;

-- Add website_id column
ALTER TABLE scraped_content_chunks
  ADD COLUMN IF NOT EXISTS website_id UUID;

-- Backfill website_id from scraped_pages → scraped_websites
UPDATE scraped_content_chunks scc
SET website_id = sw.website_id
FROM scraped_pages sp
JOIN scraped_websites sw ON sp.scraped_website_id = sw.id
WHERE scc.scraped_page_id = sp.id;

-- Make it NOT NULL after backfill
ALTER TABLE scraped_content_chunks
  ALTER COLUMN website_id SET NOT NULL;

-- Add foreign key constraint
ALTER TABLE scraped_content_chunks
  ADD CONSTRAINT scraped_content_chunks_website_id_fkey
  FOREIGN KEY (website_id) REFERENCES websites(id) ON DELETE CASCADE;

-- Add index for fast lookups
CREATE INDEX IF NOT EXISTS idx_scraped_content_chunks_website_id
  ON scraped_content_chunks(website_id);

COMMIT;