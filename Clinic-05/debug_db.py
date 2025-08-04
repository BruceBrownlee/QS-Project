#!/usr/bin/env python3

import duckdb
import os
from pathlib import Path
from qsconnect import Client

# Check database directly
db_path = Path('data/qsconnect-demo/database/qsconnect.duckdb')
print(f"Database path: {db_path}")
print(f"Database exists: {db_path.exists()}")

if db_path.exists():
    print(f"Database size: {db_path.stat().st_size:,} bytes")
    
    # Direct DuckDB connection
    conn = duckdb.connect(str(db_path))
    try:
        tables = conn.execute('SHOW TABLES').fetchall()
        print(f"Available tables: {tables}")
        
        for table in tables:
            table_name = table[0]
            count = conn.execute(f'SELECT COUNT(*) FROM {table_name}').fetchone()[0]
            print(f"  {table_name}: {count:,} rows")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

print("\n" + "="*50)
print("Now testing QSConnect Client:")

# Set environment variable
os.environ["QSCONNECT_ROOT"] = str(Path.cwd() / "data" / "qsconnect-demo")

# Test QSConnect
try:
    client = Client()
    conn = client.connect_to_database()
    print("QSConnect connection successful")
    
    # Try to get the table
    data = client.collect_database_table("historical_prices_fmp")
    print(f"Successfully loaded data: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    
    client.close_database_connection()
    
except Exception as e:
    print(f"QSConnect error: {e}")
    import traceback
    traceback.print_exc()
