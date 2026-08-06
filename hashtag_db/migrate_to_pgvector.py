"""
pgvector Migration — PreViral Hashtag Database
================================================
When ready to scale to 500K hashtags, run this script to migrate
from SQLite to PostgreSQL + pgvector.

Prerequisites:
  1. Install PostgreSQL (https://www.postgresql.org/download/)
  2. Install pgvector extension: CREATE EXTENSION vector;
  3. pip install psycopg2-binary pgvector sqlalchemy

Why pgvector?
  - SQLite CANNOT do cosine similarity on 384-dim vectors at query time
  - At 500K hashtags: 500K × 384 floats × 4 bytes = ~768MB embeddings
  - pgvector handles nearest-neighbor search in O(log n) via HNSW index
  - Query time: <10ms for top-10 similar hashtags vs ~3s full table scan in SQLite
"""
import os
import sqlite3
import json

SQLITE_PATH = os.path.join(os.path.dirname(__file__), "hashtags.db")
PG_DSN = os.environ.get("PREVIRAL_PG_DSN", "postgresql://localhost/previral")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS hashtags (
    id            SERIAL PRIMARY KEY,
    hashtag       TEXT NOT NULL,
    platform      TEXT NOT NULL,
    niche         TEXT,
    total_posts   BIGINT DEFAULT 0,
    posts_48h     INT DEFAULT 0,
    posts_7d      INT DEFAULT 0,
    competition_ratio   FLOAT,
    trend_velocity      FLOAT,
    trend_status        TEXT,
    embedding     vector(384),       -- MiniLM sentence embedding
    last_updated  TIMESTAMP DEFAULT NOW(),
    UNIQUE(hashtag, platform)
);

-- HNSW index for fast approximate nearest-neighbor search
CREATE INDEX IF NOT EXISTS hashtags_embedding_idx
    ON hashtags USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Standard indexes
CREATE INDEX IF NOT EXISTS hashtags_platform_idx ON hashtags(platform);
CREATE INDEX IF NOT EXISTS hashtags_niche_idx ON hashtags(niche);
CREATE INDEX IF NOT EXISTS hashtags_velocity_idx ON hashtags(trend_velocity DESC);
"""

PGVECTOR_QUERY_EXAMPLE = """
-- Example: Find top-10 semantically similar hashtags to a caption embedding
-- (Replace :embedding with a 384-float array from sentence-transformers)

SELECT hashtag, competition_ratio, trend_velocity, trend_status,
       1 - (embedding <=> :embedding) AS relevance_score
FROM hashtags
WHERE platform = :platform
  AND competition_ratio < 0.6
  AND trend_velocity > 0.5
ORDER BY embedding <=> :embedding   -- cosine distance
LIMIT 10;
"""

def migrate_sqlite_to_pg():
    """
    Reads all rows from SQLite hashtags.db and inserts into PostgreSQL.
    Embeddings are re-generated since SQLite doesn't store them yet.
    """
    try:
        import psycopg2
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Install: pip install psycopg2-binary pgvector")
        return

    model = SentenceTransformer("all-MiniLM-L6-v2")

    sqlite_conn = sqlite3.connect(SQLITE_PATH)
    rows = sqlite_conn.execute("SELECT * FROM hashtags").fetchall()
    sqlite_conn.close()

    pg_conn = psycopg2.connect(PG_DSN)
    cur = pg_conn.cursor()

    # Enable pgvector
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute(SCHEMA_SQL)

    print(f"Migrating {len(rows)} rows from SQLite to pgvector...")
    batch_size = 500
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]
        texts = [row[0].replace("_", " ") for row in batch]
        embeddings = model.encode(texts).tolist()

        for j, row in enumerate(batch):
            hashtag, platform, niche, comp, velocity, status = row[0], row[1], row[5] if len(row) > 5 else None, row[2], row[3], row[4]
            emb_str = "[" + ",".join(f"{x:.6f}" for x in embeddings[j]) + "]"
            cur.execute("""
                INSERT INTO hashtags (hashtag, platform, niche, competition_ratio, trend_velocity, trend_status, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
                ON CONFLICT (hashtag, platform) DO UPDATE
                SET trend_velocity = EXCLUDED.trend_velocity,
                    trend_status = EXCLUDED.trend_status,
                    embedding = EXCLUDED.embedding,
                    last_updated = NOW()
            """, (hashtag, platform, niche, comp, velocity, status, emb_str))

        pg_conn.commit()
        print(f"  Migrated {min(i+batch_size, len(rows))}/{len(rows)}")

    cur.close()
    pg_conn.close()
    print("Migration complete!")
    print(f"\nExample pgvector query:\n{PGVECTOR_QUERY_EXAMPLE}")


if __name__ == "__main__":
    print("PreViral — pgvector Migration Script")
    print("="*50)
    print(f"Source: {SQLITE_PATH}")
    print(f"Target: {PG_DSN}")
    print("\nThis requires PostgreSQL + pgvector to be installed.")
    print("For now, SQLite is fine up to ~50K hashtags.")
    print("Run this when you're ready to scale to 500K+.\n")

    if input("Proceed? (y/N): ").lower() == 'y':
        migrate_sqlite_to_pg()
    else:
        print("Aborted. Use SQLite until ready to scale.")
        print("\nSchema for when you're ready:")
        print(SCHEMA_SQL)
