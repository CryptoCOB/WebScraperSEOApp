import asyncio
import aiomysql
import logging
from dotenv import load_dotenv
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(filename='database.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Database credentials from environment variables
MYSQL_HOST = os.getenv("MYSQL_HOST", 'localhost')
MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3306))
MYSQL_USER = os.getenv("MYSQL_USER", 'your_username')
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", 'your_password')
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", 'seo_data')



class DatabaseManager:
    """Manages database connections and interactions."""
    def __init__(self):
        self._conn = None

    async def connect(self):
        """Asynchronously establishes a connection to the MySQL database."""
        if not self._conn:
            try:
                self._conn = await aiomysql.connect(host=MYSQL_HOST, port=MYSQL_PORT,
                                                     user=MYSQL_USER, password=MYSQL_PASSWORD,
                                                     db=MYSQL_DATABASE)
            except aiomysql.Error as e:
                logging.error(f"Error connecting to MySQL database: {e}")
                raise

    async def close(self):
        """Asynchronously closes the database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None

    @property
    async def conn(self):
        """Returns the current database connection."""
        if not self._conn:
            await self.connect()
        return self._conn


async def create_tables(pool):
    """Creates necessary tables in the database if they don't exist."""
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INT AUTO_INCREMENT PRIMARY KEY,
                full_path VARCHAR(255) NOT NULL UNIQUE,
                description TEXT,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );""")

            await cur.execute("""
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INT AUTO_INCREMENT PRIMARY KEY,
                content TEXT NOT NULL,
                feedback TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );""")
            await conn.commit()

async def insert_image_description(pool, image_full_path, description):
    """Inserts a record for an image and its description into the database.
    Args:
        db: The database connection pool object.
        image_full_path: The full path to the image file.
        description: The description of the image.
    Raises:
        ValueError: If image_full_path is empty or description is too long.
    """
    if not image_full_path:
        raise ValueError("Image full path cannot be empty.")

    if len(description) > 4096:  # Adjust limit based on needs
        logging.warning("Description exceeds recommended length, truncating.")
        description = description[:4095]
        
    async with pool.acquire() as conn:
        async with pool.acquire() as cur:
            sql = "SELECT description FROM images WHERE full_path =%s"
            await cur.execute(sql, (image_full_path, ))
            await conn.commit()

async def retrieve_image_description(pool, image_full_path):
    """Retrieves the description for a given image path from the database.
    Args:
        db: The database connection pool object.
        image_full_path: The full path to the image file.
    Returns:
        The description of the image (string), or None if not found.
    """
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            sql = "SELECT description FROM images WHERE full_path = %s"
            await cur.execute(sql, (image_full_path,))
            result = await cur.fetchone()
            return result['description'] if result else None

async def insert_user_feedback(pool, content, feedback):
    """Inserts user feedback into the database."""
    if not content or not feedback:
        raise ValueError("Content and feedback cannot be empty.")

    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            sql = "INSERT INTO user_feedback (content, feedback) VALUES (%s, %s)"
            await cur.execute(sql, (content, feedback))
            await conn.commit()

async def main():
    """Demonstrate using the database management."""
    pool = await aiomysql.create_pool(host=MYSQL_HOST, port=MYSQL_PORT,
                                    user=MYSQL_USER, password=MYSQL_PASSWORD,
                                    db=MYSQL_DATABASE, autocommit=True)
    await create_tables(pool)
    # Insert sample data, retrieve, etc.

    await pool.close()

if __name__ == "__main__":
    asyncio.run(main())

