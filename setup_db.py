import sqlite3
import pandas as pd
import os

def setup_database():
    # 1. Idempotency: Nuke the old database before starting so you always have a clean slate
    if os.path.exists('retail_analytics.db'):
        os.remove('retail_analytics.db')
        print("Old database removed. Starting fresh...")
        
    print("Loading CSV data...")
    df = pd.read_csv('data/sales_data.csv')
    
    print("Cleaning data...")
    # Drop empty ghost rows and duplicates
    df = df.dropna(how='all')
    df = df.drop_duplicates()
    
    # Parse dates, slice off the "future" 2024 data, and format for SQLite
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)
    df = df[df['Date'] <= '2024-01-31'] 
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
  
    # 2. Connect to (and automatically create) the SQLite database file
    conn = sqlite3.connect('retail_analytics.db')
    cursor = conn.cursor()

    print("Building schema...")
    # 3. Create the Dimension Tables (The Entities)
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS Stores (
            store_id TEXT PRIMARY KEY,
            region TEXT NOT NULL,
            location_type TEXT NOT NULL CHECK (location_type IN ('Retail_Branch', 'Warehouse'))
        );

        CREATE TABLE IF NOT EXISTS Store_Products (
            store_id TEXT,
            product_id TEXT,
            category TEXT NOT NULL,
            PRIMARY KEY (store_id, product_id),
            FOREIGN KEY (store_id) REFERENCES Stores(store_id)
        );
    """)

    # 4. Create the Fact Table & B+ Tree Index (The Transactions)
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS Daily_Operations_Log (
            record_date DATE,
            store_id TEXT,
            product_id TEXT,
            
            -- Core Metrics
            inventory_level INTEGER NOT NULL,
            units_sold INTEGER DEFAULT 0,
            units_ordered INTEGER DEFAULT 0,
            demand INTEGER NOT NULL,
            
            -- Economics
            price REAL NOT NULL,
            discount INTEGER DEFAULT 0,
            competitor_pricing REAL,
            
            -- Environmental Factors
            weather_condition TEXT,
            seasonality TEXT,
            promotion INTEGER CHECK (promotion IN (0, 1)),
            epidemic INTEGER CHECK (epidemic IN (0, 1)),
            
            PRIMARY KEY (record_date, store_id, product_id),
            FOREIGN KEY (store_id, product_id) REFERENCES Store_Products(store_id, product_id)
        );
        
        -- Clustered index equivalent for fast time-series retrieval by your ML models
        CREATE INDEX IF NOT EXISTS idx_time_series ON Daily_Operations_Log(store_id, product_id, record_date);
    """)

    # 5. Create the Current Inventory View (The "Right Now" Snapshot)
    cursor.executescript("""
        CREATE VIEW IF NOT EXISTS Current_Warehouse_Stock AS
        SELECT store_id, product_id, inventory_level
        FROM Daily_Operations_Log
        WHERE (store_id, product_id, record_date) IN (
            SELECT store_id, product_id, MAX(record_date)
            FROM Daily_Operations_Log
            GROUP BY store_id, product_id
        );
    """)
    conn.commit()

    print("Extracting and inserting dimension data...")
    # 6. Isolate and insert Stores
    stores_df = df[['Store ID', 'Region']].drop_duplicates().rename(columns={'Store ID': 'store_id', 'Region': 'region'})
    stores_df['location_type'] = 'Retail_Branch' 
    stores_df.to_sql('Stores', conn, if_exists='append', index=False)

    # 7. Isolate and insert Products
    products_df = df[['Store ID', 'Product ID', 'Category']].drop_duplicates().rename(
        columns={'Store ID': 'store_id', 'Product ID': 'product_id', 'Category': 'category'}
    )
    products_df.to_sql('Store_Products', conn, if_exists='append', index=False)

    print("Preparing and inserting historical fact data...")
    # 8. Prepare and insert the Historical Fact Table
    fact_df = df.rename(columns={
        'Date': 'record_date', 'Store ID': 'store_id', 'Product ID': 'product_id',
        'Inventory Level': 'inventory_level', 'Units Sold': 'units_sold', 
        'Units Ordered': 'units_ordered', 'Demand': 'demand', 'Price': 'price',
        'Discount': 'discount', 'Competitor Pricing': 'competitor_pricing',
        'Weather Condition': 'weather_condition', 'Seasonality': 'seasonality',
        'Promotion': 'promotion', 'Epidemic': 'epidemic'
    })
    
    # Drop columns that are now safely stored in the dimension tables to avoid redundancy
    fact_df = fact_df.drop(columns=['Category', 'Region'])
    
    print(f"Inserting {len(fact_df)} transactional records. This may take a few seconds...")
    fact_df.to_sql('Daily_Operations_Log', conn, if_exists='append', index=False)

    # 9. Build the unified Master_View with all advanced metrics and CAST safety nets
    print("Building Master_View for AI...")
    cursor.execute("DROP TABLE IF EXISTS Master_View;")
    cursor.execute("""
        CREATE TABLE Master_View AS 
        SELECT 
            d.record_date, 
            d.store_id, 
            d.product_id, 
            d.inventory_level, 
            d.units_sold, 
            d.demand, 
            d.price, 
            d.discount, 
            d.promotion,
            
            -- Core Metrics (Explicitly Cast for SQLAlchemy)
            CAST((d.units_sold * d.price * (1.0 - (d.discount / 100.0))) AS REAL) AS revenue,
            CAST((CASE WHEN d.inventory_level < d.demand THEN 1 ELSE 0 END) AS INTEGER) AS is_stockout,
            
            -- Rescued Advanced Metrics (Explicitly Cast for SQLAlchemy)
            CAST(d.inventory_level * d.price AS REAL) AS inventory_value,
            CAST(MAX(d.demand - d.units_sold, 0) AS INTEGER) AS lost_demand,
            CAST(CASE WHEN d.inventory_level > (2 * d.demand) THEN 1 ELSE 0 END AS INTEGER) AS overstock,
            CAST(d.units_sold AS REAL) / MAX(d.inventory_level, 1) AS sell_through_rate,
            CAST(d.inventory_level AS REAL) / MAX(d.demand, 1) AS coverage_days,
            CAST(d.price - d.competitor_pricing AS REAL) AS price_gap,

            -- Metadata
            s.region, 
            s.location_type, 
            sp.category 
        FROM Daily_Operations_Log d
        LEFT JOIN Stores s ON d.store_id = s.store_id
        LEFT JOIN Store_Products sp ON d.store_id = sp.store_id AND d.product_id = sp.product_id;
    """)
    conn.commit()
    conn.close()
    
    print("Database setup complete! 'retail_analytics.db' created successfully.")

if __name__ == "__main__":
    setup_database()