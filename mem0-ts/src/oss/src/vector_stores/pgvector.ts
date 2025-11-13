import { Client, Pool, PoolClient } from "pg";
import { VectorStore } from "./base";
import { SearchFilters, VectorStoreConfig, VectorStoreResult } from "../types";

// Base configuration shared by all connection types
interface PGVectorConfigBase extends VectorStoreConfig {
  // Required configuration
  embeddingModelDims: number;

  // Optional configuration
  diskann?: boolean;
  hnsw?: boolean;
  minconn?: number; // Minimum connections in pool (default: 1)
  maxconn?: number; // Maximum connections in pool (default: 5)
}

// Connection via pre-built pool (highest priority)
interface PGVectorConfigWithPool extends PGVectorConfigBase {
  connectionPool: Pool;
  dbname?: string; // Optional, defaults to "vector_store"
  // Cannot have connectionString or individual parameters
  connectionString?: never;
  user?: never;
  password?: never;
  host?: never;
  port?: never;
  sslmode?: never;
}

// Connection via connection string
interface PGVectorConfigWithConnectionString extends PGVectorConfigBase {
  connectionString: string;
  dbname?: string; // Optional, extracted from connection string if not provided
  sslmode?: string; // SSL mode (e.g., 'require', 'prefer', 'disable')
  // Cannot have connectionPool or individual parameters
  connectionPool?: never;
  user?: never;
  password?: never;
  host?: never;
  port?: never;
}

// Connection via individual parameters
interface PGVectorConfigWithIndividualParams extends PGVectorConfigBase {
  user: string;
  password: string;
  host: string;
  port: number;
  dbname?: string; // Optional, defaults to "vector_store"
  sslmode?: string; // SSL mode (e.g., 'require', 'prefer', 'disable')
  // Cannot have connectionPool or connectionString
  connectionPool?: never;
  connectionString?: never;
}

// Union type: exactly one connection method must be provided
export type PGVectorConfig =
  | PGVectorConfigWithPool
  | PGVectorConfigWithConnectionString
  | PGVectorConfigWithIndividualParams;

export class PGVector implements VectorStore {
  private pool: Pool;
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
    this.config = config;

    // Determine database name and connection setup
    // TypeScript union type ensures exactly one connection method is provided
    if ("connectionPool" in config && config.connectionPool) {
      // Use provided connection pool
      const poolConfig = config as PGVectorConfigWithPool;
      this.pool = poolConfig.connectionPool;
      // Use dbname from config or default
      this.dbName = poolConfig.dbname || "vector_store";
    } else if ("connectionString" in config && config.connectionString) {
      // Use connection string
      const stringConfig = config as PGVectorConfigWithConnectionString;
      this.dbName =
        this.extractDbNameFromConnectionString(stringConfig.connectionString) ||
        stringConfig.dbname ||
        "vector_store";
      const connectionString = this.buildConnectionString(
        stringConfig.connectionString,
        stringConfig.sslmode,
      );
      this.pool = new Pool({
        connectionString,
        min: stringConfig.minconn || 1,
        max: stringConfig.maxconn || 5,
      });
    } else if (
      "user" in config &&
      "password" in config &&
      "host" in config &&
      "port" in config
    ) {
      // Use individual parameters (TypeScript ensures all are present)
      const paramsConfig = config as PGVectorConfigWithIndividualParams;
      this.dbName = paramsConfig.dbname || "vector_store";
      const connectionString = this.buildConnectionStringFromParams(
        paramsConfig.user,
        paramsConfig.password,
        paramsConfig.host,
        paramsConfig.port,
        this.dbName,
        paramsConfig.sslmode,
      );
      this.pool = new Pool({
        connectionString,
        min: paramsConfig.minconn || 1,
        max: paramsConfig.maxconn || 5,
      });
    } else {
      // This should never happen due to TypeScript type checking, but provide a helpful error
      throw new Error(
        "PGVectorConfig must provide either connectionPool, connectionString, or all individual parameters (user, password, host, port)",
      );
    }

    // Sanitize identifiers to prevent SQL injection
    this.sanitizedCollectionName = this.sanitizeIdentifier(this.collectionName);
    this.sanitizedDbName = this.sanitizeIdentifier(this.dbName);

    // Auto-initialize like other vector stores
    this.initialize().catch((err) => {
      console.error("Failed to initialize PGVector:", err);
      throw err;
    });
  }

  /**
   * Extract database name from PostgreSQL connection string
   */
  private extractDbNameFromConnectionString(
    connectionString: string,
  ): string | null {
    try {
      const url = new URL(
        connectionString.replace(/^postgresql:\/\//, "http://"),
      );
      const pathname = url.pathname;
      return pathname ? pathname.replace(/^\//, "") : null;
    } catch {
      // Try parsing as key-value pairs
      const params = new URLSearchParams(connectionString.split(" ").join("&"));
      return params.get("dbname") || null;
    }
  }

  /**
   * Build connection string with optional sslmode
   */
  private buildConnectionString(
    connectionString: string,
    sslmode?: string,
  ): string {
    if (!sslmode) {
      return connectionString;
    }

    // Check if sslmode already exists in connection string
    if (connectionString.includes("sslmode=")) {
      // Replace existing sslmode
      return connectionString.replace(/sslmode=[^ ]*/g, `sslmode=${sslmode}`);
    } else {
      // Add sslmode to connection string
      const separator = connectionString.includes("?") ? "&" : "?";
      return `${connectionString}${separator}sslmode=${sslmode}`;
    }
  }

  /**
   * Build connection string from individual parameters
   */
  private buildConnectionStringFromParams(
    user: string,
    password: string,
    host: string,
    port: number,
    dbname: string,
    sslmode?: string,
  ): string {
    let connectionString = `postgresql://${user}:${password}@${host}:${port}/${dbname}`;
    if (sslmode) {
      connectionString += `?sslmode=${sslmode}`;
    }
    return connectionString;
  }

  /**
   * Sanitize SQL identifier to prevent SQL injection
   * Only allows alphanumeric characters and underscores
   */
  private sanitizeIdentifier(identifier: string): string {
    if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(identifier)) {
      throw new Error(
        `Invalid identifier: "${identifier}". Identifiers must start with a letter or underscore and contain only alphanumeric characters and underscores.`,
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
        `Invalid filter key: "${key}". Keys must contain only alphanumeric characters, underscores, dots, and hyphens.`,
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
    startIndex: number,
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
            `(payload->>'${sanitizedKey}')::${castType} >= $${currentIndex}`,
          );
          filterValues.push(value.gte);
          currentIndex++;
        }
        if ("gt" in value) {
          conditions.push(
            `(payload->>'${sanitizedKey}')::${castType} > $${currentIndex}`,
          );
          filterValues.push(value.gt);
          currentIndex++;
        }
        if ("lte" in value) {
          conditions.push(
            `(payload->>'${sanitizedKey}')::${castType} <= $${currentIndex}`,
          );
          filterValues.push(value.lte);
          currentIndex++;
        }
        if ("lt" in value) {
          conditions.push(
            `(payload->>'${sanitizedKey}')::${castType} < $${currentIndex}`,
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
      // Only check/create database if using individual connection parameters
      if (
        "user" in this.config &&
        "password" in this.config &&
        "host" in this.config &&
        "port" in this.config
      ) {
        // Create a temporary client to connect to 'postgres' database for database creation
        const paramsConfig = this.config as PGVectorConfigWithIndividualParams;
        const tempClient = new Client({
          database: "postgres",
          user: paramsConfig.user,
          password: paramsConfig.password,
          host: paramsConfig.host,
          port: paramsConfig.port,
        });

        try {
          await tempClient.connect();

          // Check if database exists
          const dbExists = await this.checkDatabaseExists(
            this.sanitizedDbName,
            tempClient,
          );
          if (!dbExists) {
            await this.createDatabase(this.sanitizedDbName, tempClient);
          }
        } finally {
          await tempClient.end();
        }
      }

      // Use pool for all operations
      const client = await this.pool.connect();

      try {
        // Create vector extension
        await client.query("CREATE EXTENSION IF NOT EXISTS vector");

        // Create memory_migrations table
        await client.query(`
          CREATE TABLE IF NOT EXISTS memory_migrations (
            id SERIAL PRIMARY KEY,
            user_id TEXT NOT NULL UNIQUE
          )
        `);

        // Check if the collection exists
        const collections = await this.listCols(client);
        if (!collections.includes(this.sanitizedCollectionName)) {
          await this.createCol(this.config.embeddingModelDims, client);
        }

        // Cache the quoted table name for performance
        this.quotedTableName = await this.getQuotedIdentifier(
          this.sanitizedCollectionName,
          client,
        );
      } finally {
        client.release();
      }
    } catch (error) {
      console.error("Error during PGVector initialization:", error);
      throw new Error(
        `Failed to initialize PGVector: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  private async checkDatabaseExists(
    dbName: string,
    client: Client | PoolClient,
  ): Promise<boolean> {
    try {
      const result = await client.query(
        "SELECT 1 FROM pg_database WHERE datname = $1",
        [dbName],
      );
      return result.rows.length > 0;
    } catch (error) {
      console.error("Error checking database existence:", error);
      throw new Error(
        `Failed to check database existence: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  private async createDatabase(
    dbName: string,
    client: Client | PoolClient,
  ): Promise<void> {
    try {
      // Use quote_ident via a parameterized approach - validate first, then use quote_ident
      // Since CREATE DATABASE cannot be parameterized, we validate the name format
      const result = await client.query(
        `SELECT quote_ident($1) as quoted_name`,
        [dbName],
      );
      const quotedName = result.rows[0].quoted_name;
      await client.query(`CREATE DATABASE ${quotedName}`);
    } catch (error) {
      console.error("Error creating database:", error);
      throw new Error(
        `Failed to create database: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  private async createCol(
    embeddingModelDims: number,
    client: PoolClient,
  ): Promise<void> {
    try {
      // Get quoted identifier for table name
      const quotedTableName = await this.getQuotedIdentifier(
        this.sanitizedCollectionName,
        client,
      );

      // Create the table
      await client.query(`
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
          const result = await client.query(
            "SELECT * FROM pg_extension WHERE extname = 'vectorscale'",
          );
          if (result.rows.length > 0) {
            const quotedIndexName = await this.getQuotedIdentifier(
              `${this.sanitizedCollectionName}_diskann_idx`,
              client,
            );
            await client.query(`
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
            `${this.sanitizedCollectionName}_hnsw_idx`,
            client,
          );
          await client.query(`
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
        `Failed to create collection: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  /**
   * Get quoted identifier using PostgreSQL's quote_ident function
   * Caches the result for the table name to avoid repeated queries
   */
  private async getQuotedIdentifier(
    identifier: string,
    client: PoolClient,
  ): Promise<string> {
    // Cache the quoted table name since it doesn't change
    if (identifier === this.sanitizedCollectionName && this.quotedTableName) {
      return this.quotedTableName;
    }

    const result = await client.query(`SELECT quote_ident($1) as quoted`, [
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
    payloads: Record<string, any>[],
  ): Promise<void> {
    const client = await this.pool.connect();
    try {
      if (vectors.length !== ids.length || vectors.length !== payloads.length) {
        throw new Error(
          "Vectors, ids, and payloads arrays must have the same length",
        );
      }

      const quotedTableName = await this.getQuotedIdentifier(
        this.sanitizedCollectionName,
        client,
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
          client.query(query, [value.id, value.vector, value.payload]),
        ),
      );
    } catch (error) {
      console.error("Error inserting vectors:", error);
      throw new Error(
        `Failed to insert vectors: ${error instanceof Error ? error.message : String(error)}`,
      );
    } finally {
      client.release();
    }
  }

  async search(
    query: number[],
    limit: number = 5,
    filters?: SearchFilters,
  ): Promise<VectorStoreResult[]> {
    const client = await this.pool.connect();
    try {
      const quotedTableName = await this.getQuotedIdentifier(
        this.sanitizedCollectionName,
        client,
      );
      const queryVector = `[${query.join(",")}]`; // Format query vector as string with square brackets
      const filterValues: any[] = [queryVector, limit];

      let filterConditions: string[] = [];
      if (filters) {
        const filterResult = this.buildFilterConditions(
          filters,
          filterValues,
          3,
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

      const result = await client.query(searchQuery, filterValues);

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
        `Failed to search vectors: ${error instanceof Error ? error.message : String(error)}`,
      );
    } finally {
      client.release();
    }
  }

  async get(vectorId: string): Promise<VectorStoreResult | null> {
    const client = await this.pool.connect();
    try {
      const quotedTableName = await this.getQuotedIdentifier(
        this.sanitizedCollectionName,
        client,
      );
      const result = await client.query(
        `SELECT id, payload FROM ${quotedTableName} WHERE id = $1`,
        [vectorId],
      );

      if (result.rows.length === 0) return null;

      return {
        id: result.rows[0].id,
        payload: result.rows[0].payload,
      };
    } catch (error) {
      console.error("Error getting vector:", error);
      throw new Error(
        `Failed to get vector: ${error instanceof Error ? error.message : String(error)}`,
      );
    } finally {
      client.release();
    }
  }

  async update(
    vectorId: string,
    vector: number[],
    payload: Record<string, any>,
  ): Promise<void> {
    const client = await this.pool.connect();
    try {
      const quotedTableName = await this.getQuotedIdentifier(
        this.sanitizedCollectionName,
        client,
      );
      const vectorStr = `[${vector.join(",")}]`; // Format vector as string with square brackets
      await client.query(
        `
        UPDATE ${quotedTableName}
        SET vector = $1::vector, payload = $2::jsonb
        WHERE id = $3
        `,
        [vectorStr, payload, vectorId],
      );
    } catch (error) {
      console.error("Error updating vector:", error);
      throw new Error(
        `Failed to update vector: ${error instanceof Error ? error.message : String(error)}`,
      );
    } finally {
      client.release();
    }
  }

  async delete(vectorId: string): Promise<void> {
    const client = await this.pool.connect();
    try {
      const quotedTableName = await this.getQuotedIdentifier(
        this.sanitizedCollectionName,
        client,
      );
      await client.query(`DELETE FROM ${quotedTableName} WHERE id = $1`, [
        vectorId,
      ]);
    } catch (error) {
      console.error("Error deleting vector:", error);
      throw new Error(
        `Failed to delete vector: ${error instanceof Error ? error.message : String(error)}`,
      );
    } finally {
      client.release();
    }
  }

  async deleteCol(): Promise<void> {
    const client = await this.pool.connect();
    try {
      const quotedTableName = await this.getQuotedIdentifier(
        this.sanitizedCollectionName,
        client,
      );
      await client.query(`DROP TABLE IF EXISTS ${quotedTableName}`);
    } catch (error) {
      console.error("Error deleting collection:", error);
      throw new Error(
        `Failed to delete collection: ${error instanceof Error ? error.message : String(error)}`,
      );
    } finally {
      client.release();
    }
  }

  private async listCols(client: PoolClient): Promise<string[]> {
    try {
      const result = await client.query(`
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
      `);
      return result.rows.map((row) => row.table_name);
    } catch (error) {
      console.error("Error listing collections:", error);
      throw new Error(
        `Failed to list collections: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  async list(
    filters?: SearchFilters,
    limit: number = 100,
  ): Promise<[VectorStoreResult[], number]> {
    const client = await this.pool.connect();
    try {
      const quotedTableName = await this.getQuotedIdentifier(
        this.sanitizedCollectionName,
        client,
      );
      const filterValues: any[] = [];
      let filterConditions: string[] = [];
      let paramIndex = 1;

      if (filters) {
        const filterResult = this.buildFilterConditions(
          filters,
          filterValues,
          1,
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
        client.query(listQuery, filterValues),
        client.query(countQuery, filterValues.slice(0, -1)), // Remove limit parameter for count query
      ]);

      const results = listResult.rows.map((row) => ({
        id: row.id,
        payload: row.payload,
      }));

      return [results, parseInt(countResult.rows[0].count)];
    } catch (error) {
      console.error("Error listing vectors:", error);
      throw new Error(
        `Failed to list vectors: ${error instanceof Error ? error.message : String(error)}`,
      );
    } finally {
      client.release();
    }
  }

  async close(): Promise<void> {
    try {
      // Only close pool if we created it (not if it was provided)
      if (!("connectionPool" in this.config && this.config.connectionPool)) {
        await this.pool.end();
      }
    } catch (error) {
      console.error("Error closing connection:", error);
      throw new Error(
        `Failed to close connection: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  async getUserId(): Promise<string> {
    const client = await this.pool.connect();
    try {
      const result = await client.query(
        "SELECT user_id FROM memory_migrations LIMIT 1",
      );

      if (result.rows.length > 0) {
        return result.rows[0].user_id;
      }

      // Generate a random user_id if none exists
      const randomUserId =
        Math.random().toString(36).substring(2, 15) +
        Math.random().toString(36).substring(2, 15);
      await client.query(
        "INSERT INTO memory_migrations (user_id) VALUES ($1)",
        [randomUserId],
      );
      return randomUserId;
    } catch (error) {
      console.error("Error getting user ID:", error);
      throw new Error(
        `Failed to get user ID: ${error instanceof Error ? error.message : String(error)}`,
      );
    } finally {
      client.release();
    }
  }

  async setUserId(userId: string): Promise<void> {
    const client = await this.pool.connect();
    try {
      await client.query("DELETE FROM memory_migrations");
      await client.query(
        "INSERT INTO memory_migrations (user_id) VALUES ($1)",
        [userId],
      );
    } catch (error) {
      console.error("Error setting user ID:", error);
      throw new Error(
        `Failed to set user ID: ${error instanceof Error ? error.message : String(error)}`,
      );
    } finally {
      client.release();
    }
  }
}
