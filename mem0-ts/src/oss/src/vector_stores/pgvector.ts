import { Client } from "pg";
import { VectorStore } from "./base";
import { SearchFilters, VectorStoreConfig, VectorStoreResult } from "../types";

interface PGVectorConfig extends VectorStoreConfig {
  dbname?: string;
  user: string;
  password: string;
  host: string;
  port: number;
  embeddingModelDims: number;
  diskann?: boolean;
  hnsw?: boolean;
}

export class PGVector implements VectorStore {
  private client: Client;
  private collectionName: string;
  private sanitizedCollectionName: string;
  private quotedTableName: string | null = null;
  private useDiskann: boolean;
  private useHnsw: boolean;
  private readonly dbName: string;
  private sanitizedDbName: string;
  private config: PGVectorConfig;

  constructor(config: PGVectorConfig) {
    this.collectionName = config.collectionName || "memories";
    this.useDiskann = config.diskann || false;
    this.useHnsw = config.hnsw || false;
    this.dbName = config.dbname || "vector_store";
    this.config = config;

    // Sanitize identifiers to prevent SQL injection
    this.sanitizedCollectionName = this.sanitizeIdentifier(this.collectionName);
    this.sanitizedDbName = this.sanitizeIdentifier(this.dbName);

    this.client = new Client({
      database: "postgres", // Initially connect to default postgres database
      user: config.user,
      password: config.password,
      host: config.host,
      port: config.port,
    });

    // Auto-initialize like other vector stores
    this.initialize().catch((err) => {
      console.error("Failed to initialize PGVector:", err);
      throw err;
    });
  }

  /**
   * Sanitize SQL identifier to prevent SQL injection
   * Only allows alphanumeric characters and underscores
   */
  private sanitizeIdentifier(identifier: string): string {
    if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(identifier)) {
      throw new Error(
        `Invalid identifier: "${identifier}". Identifiers must start with a letter or underscore and contain only alphanumeric characters and underscores.`
      );
    }
    return identifier;
  }

  /**
   * Sanitize filter key to prevent SQL injection in JSONB path expressions
   * Since -> operator requires literal keys, we validate and sanitize the key
   */
  private sanitizeFilterKey(key: string): string {
    // Validate key format - allow alphanumeric, underscores, dots, and hyphens
    if (!/^[a-zA-Z0-9_.-]+$/.test(key)) {
      throw new Error(
        `Invalid filter key: "${key}". Keys must contain only alphanumeric characters, underscores, dots, and hyphens.`
      );
    }
    // Escape single quotes to prevent SQL injection
    return key.replace(/'/g, "''");
  }

  /**
   * Build filter conditions for SQL queries
   * Supports both equality and range filters (gte, gt, lte, lt)
   */
  private buildFilterConditions(
    filters: SearchFilters,
    filterValues: any[],
    startIndex: number
  ): { conditions: string[]; nextIndex: number } {
    const conditions: string[] = [];
    let currentIndex = startIndex;

    for (const [key, value] of Object.entries(filters)) {
      const sanitizedKey = this.sanitizeFilterKey(key);

      // Check if it's a range filter (object with gte, gt, lte, lt)
      if (
        typeof value === "object" &&
        value !== null &&
        ("gte" in value || "gt" in value || "lte" in value || "lt" in value)
      ) {
        // Handle range filters
        // Try to determine if values are numeric or text by checking the first range value
        const rangeValue = value.gte ?? value.gt ?? value.lte ?? value.lt;
        const isNumeric =
          typeof rangeValue === "number" ||
          (typeof rangeValue === "string" && !isNaN(Number(rangeValue)));

        const castType = isNumeric ? "numeric" : "text";

        if ("gte" in value) {
          conditions.push(
            `(payload->>'${sanitizedKey}')::${castType} >= $${currentIndex}`
          );
          filterValues.push(value.gte);
          currentIndex++;
        }
        if ("gt" in value) {
          conditions.push(
            `(payload->>'${sanitizedKey}')::${castType} > $${currentIndex}`
          );
          filterValues.push(value.gt);
          currentIndex++;
        }
        if ("lte" in value) {
          conditions.push(
            `(payload->>'${sanitizedKey}')::${castType} <= $${currentIndex}`
          );
          filterValues.push(value.lte);
          currentIndex++;
        }
        if ("lt" in value) {
          conditions.push(
            `(payload->>'${sanitizedKey}')::${castType} < $${currentIndex}`
          );
          filterValues.push(value.lt);
          currentIndex++;
        }
      } else {
        // Handle equality filter
        conditions.push(`payload->>'${sanitizedKey}' = $${currentIndex}`);
        filterValues.push(value);
        currentIndex++;
      }
    }

    return { conditions, nextIndex: currentIndex };
  }

  async initialize(): Promise<void> {
    try {
      await this.client.connect();

      // Check if database exists
      const dbExists = await this.checkDatabaseExists(this.sanitizedDbName);
      if (!dbExists) {
        await this.createDatabase(this.sanitizedDbName);
      }

      // Disconnect from postgres database
      await this.client.end();

      // Connect to the target database
      this.client = new Client({
        database: this.sanitizedDbName,
        user: this.config.user,
        password: this.config.password,
        host: this.config.host,
        port: this.config.port,
      });
      await this.client.connect();

      // Create vector extension
      await this.client.query("CREATE EXTENSION IF NOT EXISTS vector");

      // Create memory_migrations table
      await this.client.query(`
        CREATE TABLE IF NOT EXISTS memory_migrations (
          id SERIAL PRIMARY KEY,
          user_id TEXT NOT NULL UNIQUE
        )
      `);

      // Check if the collection exists
      const collections = await this.listCols();
      if (!collections.includes(this.sanitizedCollectionName)) {
        await this.createCol(this.config.embeddingModelDims);
      }

      // Cache the quoted table name for performance
      this.quotedTableName = await this.getQuotedIdentifier(
        this.sanitizedCollectionName
      );
    } catch (error) {
      console.error("Error during PGVector initialization:", error);
      throw new Error(
        `Failed to initialize PGVector: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

  private async checkDatabaseExists(dbName: string): Promise<boolean> {
    try {
      const result = await this.client.query(
        "SELECT 1 FROM pg_database WHERE datname = $1",
        [dbName]
      );
      return result.rows.length > 0;
    } catch (error) {
      console.error("Error checking database existence:", error);
      throw new Error(
        `Failed to check database existence: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

  private async createDatabase(dbName: string): Promise<void> {
    try {
      // Use quote_ident via a parameterized approach - validate first, then use quote_ident
      // Since CREATE DATABASE cannot be parameterized, we validate the name format
      const result = await this.client.query(
        `SELECT quote_ident($1) as quoted_name`,
        [dbName]
      );
      const quotedName = result.rows[0].quoted_name;
      await this.client.query(`CREATE DATABASE ${quotedName}`);
    } catch (error) {
      console.error("Error creating database:", error);
      throw new Error(
        `Failed to create database: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

  private async createCol(embeddingModelDims: number): Promise<void> {
    try {
      // Get quoted identifier for table name
      const quotedTableName = await this.getQuotedIdentifier(
        this.sanitizedCollectionName
      );

      // Create the table
      await this.client.query(`
        CREATE TABLE IF NOT EXISTS ${quotedTableName} (
          id UUID PRIMARY KEY,
          vector vector(${embeddingModelDims}),
          payload JSONB
        );
      `);

      // Create indexes based on configuration
      if (this.useDiskann && embeddingModelDims < 2000) {
        try {
          // Check if vectorscale extension is available
          const result = await this.client.query(
            "SELECT * FROM pg_extension WHERE extname = 'vectorscale'"
          );
          if (result.rows.length > 0) {
            const quotedIndexName = await this.getQuotedIdentifier(
              `${this.sanitizedCollectionName}_diskann_idx`
            );
            await this.client.query(`
              CREATE INDEX IF NOT EXISTS ${quotedIndexName}
              ON ${quotedTableName}
              USING diskann (vector);
            `);
          }
        } catch (error) {
          console.warn("DiskANN index creation failed:", error);
        }
      } else if (this.useHnsw) {
        try {
          const quotedIndexName = await this.getQuotedIdentifier(
            `${this.sanitizedCollectionName}_hnsw_idx`
          );
          await this.client.query(`
            CREATE INDEX IF NOT EXISTS ${quotedIndexName}
            ON ${quotedTableName}
            USING hnsw (vector vector_cosine_ops);
          `);
        } catch (error) {
          console.warn("HNSW index creation failed:", error);
        }
      }
    } catch (error) {
      console.error("Error creating collection:", error);
      throw new Error(
        `Failed to create collection: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

  /**
   * Get quoted identifier using PostgreSQL's quote_ident function
   * Caches the result for the table name to avoid repeated queries
   */
  private async getQuotedIdentifier(identifier: string): Promise<string> {
    // Cache the quoted table name since it doesn't change
    if (identifier === this.sanitizedCollectionName && this.quotedTableName) {
      return this.quotedTableName;
    }

    const result = await this.client.query(`SELECT quote_ident($1) as quoted`, [
      identifier,
    ]);
    const quoted = result.rows[0].quoted;

    // Cache if it's the table name
    if (identifier === this.sanitizedCollectionName) {
      this.quotedTableName = quoted;
    }

    return quoted;
  }

  async insert(
    vectors: number[][],
    ids: string[],
    payloads: Record<string, any>[]
  ): Promise<void> {
    try {
      if (vectors.length !== ids.length || vectors.length !== payloads.length) {
        throw new Error(
          "Vectors, ids, and payloads arrays must have the same length"
        );
      }

      const quotedTableName = await this.getQuotedIdentifier(
        this.sanitizedCollectionName
      );
      const values = vectors.map((vector, i) => ({
        id: ids[i],
        vector: `[${vector.join(",")}]`, // Format vector as string with square brackets
        payload: payloads[i],
      }));

      const query = `
        INSERT INTO ${quotedTableName} (id, vector, payload)
        VALUES ($1, $2::vector, $3::jsonb)
      `;

      // Execute inserts in parallel using Promise.all
      await Promise.all(
        values.map((value) =>
          this.client.query(query, [value.id, value.vector, value.payload])
        )
      );
    } catch (error) {
      console.error("Error inserting vectors:", error);
      throw new Error(
        `Failed to insert vectors: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

  async search(
    query: number[],
    limit: number = 5,
    filters?: SearchFilters
  ): Promise<VectorStoreResult[]> {
    try {
      const quotedTableName = await this.getQuotedIdentifier(
        this.sanitizedCollectionName
      );
      const queryVector = `[${query.join(",")}]`; // Format query vector as string with square brackets
      const filterValues: any[] = [queryVector, limit];

      let filterConditions: string[] = [];
      if (filters) {
        const filterResult = this.buildFilterConditions(
          filters,
          filterValues,
          3
        );
        filterConditions = filterResult.conditions;
      }

      const filterClause =
        filterConditions.length > 0
          ? "WHERE " + filterConditions.join(" AND ")
          : "";

      const searchQuery = `
        SELECT id, vector <=> $1::vector AS distance, payload
        FROM ${quotedTableName}
        ${filterClause}
        ORDER BY distance
        LIMIT $2
      `;

      const result = await this.client.query(searchQuery, filterValues);

      // Convert distance to similarity (1 - distance) for consistency with other stores
      // Distance: lower is better, Similarity: higher is better
      return result.rows.map((row) => ({
        id: row.id,
        payload: row.payload,
        score: 1 - row.distance, // Convert distance to similarity
      }));
    } catch (error) {
      console.error("Error searching vectors:", error);
      throw new Error(
        `Failed to search vectors: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

  async get(vectorId: string): Promise<VectorStoreResult | null> {
    try {
      const quotedTableName = await this.getQuotedIdentifier(
        this.sanitizedCollectionName
      );
      const result = await this.client.query(
        `SELECT id, payload FROM ${quotedTableName} WHERE id = $1`,
        [vectorId]
      );

      if (result.rows.length === 0) return null;

      return {
        id: result.rows[0].id,
        payload: result.rows[0].payload,
      };
    } catch (error) {
      console.error("Error getting vector:", error);
      throw new Error(
        `Failed to get vector: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

  async update(
    vectorId: string,
    vector: number[],
    payload: Record<string, any>
  ): Promise<void> {
    try {
      const quotedTableName = await this.getQuotedIdentifier(
        this.sanitizedCollectionName
      );
      const vectorStr = `[${vector.join(",")}]`; // Format vector as string with square brackets
      await this.client.query(
        `
        UPDATE ${quotedTableName}
        SET vector = $1::vector, payload = $2::jsonb
        WHERE id = $3
        `,
        [vectorStr, payload, vectorId]
      );
    } catch (error) {
      console.error("Error updating vector:", error);
      throw new Error(
        `Failed to update vector: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

  async delete(vectorId: string): Promise<void> {
    try {
      const quotedTableName = await this.getQuotedIdentifier(
        this.sanitizedCollectionName
      );
      await this.client.query(`DELETE FROM ${quotedTableName} WHERE id = $1`, [
        vectorId,
      ]);
    } catch (error) {
      console.error("Error deleting vector:", error);
      throw new Error(
        `Failed to delete vector: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

  async deleteCol(): Promise<void> {
    try {
      const quotedTableName = await this.getQuotedIdentifier(
        this.sanitizedCollectionName
      );
      await this.client.query(`DROP TABLE IF EXISTS ${quotedTableName}`);
    } catch (error) {
      console.error("Error deleting collection:", error);
      throw new Error(
        `Failed to delete collection: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

  private async listCols(): Promise<string[]> {
    try {
      const result = await this.client.query(`
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
      `);
      return result.rows.map((row) => row.table_name);
    } catch (error) {
      console.error("Error listing collections:", error);
      throw new Error(
        `Failed to list collections: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

  async list(
    filters?: SearchFilters,
    limit: number = 100
  ): Promise<[VectorStoreResult[], number]> {
    try {
      const quotedTableName = await this.getQuotedIdentifier(
        this.sanitizedCollectionName
      );
      const filterValues: any[] = [];
      let filterConditions: string[] = [];
      let paramIndex = 1;

      if (filters) {
        const filterResult = this.buildFilterConditions(
          filters,
          filterValues,
          1
        );
        filterConditions = filterResult.conditions;
        paramIndex = filterResult.nextIndex;
      }

      const filterClause =
        filterConditions.length > 0
          ? "WHERE " + filterConditions.join(" AND ")
          : "";

      const listQuery = `
        SELECT id, payload
        FROM ${quotedTableName}
        ${filterClause}
        LIMIT $${paramIndex}
      `;

      const countQuery = `
        SELECT COUNT(*)
        FROM ${quotedTableName}
        ${filterClause}
      `;

      filterValues.push(limit); // Add limit as the last parameter

      const [listResult, countResult] = await Promise.all([
        this.client.query(listQuery, filterValues),
        this.client.query(countQuery, filterValues.slice(0, -1)), // Remove limit parameter for count query
      ]);

      const results = listResult.rows.map((row) => ({
        id: row.id,
        payload: row.payload,
      }));

      return [results, parseInt(countResult.rows[0].count)];
    } catch (error) {
      console.error("Error listing vectors:", error);
      throw new Error(
        `Failed to list vectors: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

  async close(): Promise<void> {
    try {
      await this.client.end();
    } catch (error) {
      console.error("Error closing connection:", error);
      throw new Error(
        `Failed to close connection: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

  async getUserId(): Promise<string> {
    try {
      const result = await this.client.query(
        "SELECT user_id FROM memory_migrations LIMIT 1"
      );

      if (result.rows.length > 0) {
        return result.rows[0].user_id;
      }

      // Generate a random user_id if none exists
      const randomUserId =
        Math.random().toString(36).substring(2, 15) +
        Math.random().toString(36).substring(2, 15);
      await this.client.query(
        "INSERT INTO memory_migrations (user_id) VALUES ($1)",
        [randomUserId]
      );
      return randomUserId;
    } catch (error) {
      console.error("Error getting user ID:", error);
      throw new Error(
        `Failed to get user ID: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

  async setUserId(userId: string): Promise<void> {
    try {
      await this.client.query("DELETE FROM memory_migrations");
      await this.client.query(
        "INSERT INTO memory_migrations (user_id) VALUES ($1)",
        [userId]
      );
    } catch (error) {
      console.error("Error setting user ID:", error);
      throw new Error(
        `Failed to set user ID: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }
}
