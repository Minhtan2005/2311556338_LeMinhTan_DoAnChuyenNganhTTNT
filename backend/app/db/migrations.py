from sqlalchemy import text
from sqlalchemy.engine import Engine


def run_startup_migrations(engine: Engine) -> None:
    """Apply additive SQLite-safe migrations needed by current app versions."""
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS video_keyframes (
                    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                    video_id VARCHAR(36) NOT NULL,
                    frame_id INTEGER NOT NULL,
                    timestamp_sec FLOAT NOT NULL,
                    image_path VARCHAR(1024) NOT NULL,
                    created_at DATETIME NOT NULL,
                    FOREIGN KEY(video_id) REFERENCES videos (id),
                    CONSTRAINT uq_video_keyframes_video_frame UNIQUE (video_id, frame_id)
                )
                """
            )
        )
        connection.execute(text("CREATE INDEX IF NOT EXISTS ix_video_keyframes_video_id ON video_keyframes (video_id)"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS ix_video_keyframes_frame_id ON video_keyframes (frame_id)"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS ix_video_keyframes_timestamp_sec ON video_keyframes (timestamp_sec)"))

        connection.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS semantic_embeddings (
                    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                    keyframe_id INTEGER NOT NULL,
                    video_id VARCHAR(36) NOT NULL,
                    model_name VARCHAR(128) NOT NULL,
                    model_version VARCHAR(128) NOT NULL DEFAULT '',
                    vector_dim INTEGER NOT NULL,
                    vector_json TEXT NOT NULL,
                    descriptor_text TEXT NOT NULL DEFAULT '',
                    created_at DATETIME NOT NULL,
                    FOREIGN KEY(keyframe_id) REFERENCES video_keyframes (id),
                    FOREIGN KEY(video_id) REFERENCES videos (id),
                    CONSTRAINT uq_semantic_embeddings_keyframe_model UNIQUE (keyframe_id, model_name)
                )
                """
            )
        )
        connection.execute(text("CREATE INDEX IF NOT EXISTS ix_semantic_embeddings_keyframe_id ON semantic_embeddings (keyframe_id)"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS ix_semantic_embeddings_video_id ON semantic_embeddings (video_id)"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS ix_semantic_embeddings_model_name ON semantic_embeddings (model_name)"))
        _add_column_if_missing(
            connection,
            "semantic_embeddings",
            "model_version",
            "VARCHAR(128) NOT NULL DEFAULT ''",
        )
        connection.execute(text("CREATE INDEX IF NOT EXISTS ix_semantic_embeddings_model_version ON semantic_embeddings (model_version)"))


def _add_column_if_missing(connection, table_name: str, column_name: str, column_sql: str) -> None:
    columns = {row[1] for row in connection.execute(text(f"PRAGMA table_info({table_name})")).fetchall()}
    if column_name not in columns:
        connection.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}"))
