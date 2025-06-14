import os
import pandas as pd
from dotenv import load_dotenv
from typing import List
from langchain_core.documents import Document
from langchain_astradb import AstraDBVectorStore
from utils.model_loader import ModelLoader
from config.config_loader import load_config

class DataIngestion:
    """
    Class to handle data transformation and ingestion into AstraDB vector store.
    """

    def __init__(self):
        """
        Initialize environment variables, embedding model, and set CSV file path.
        """
        print("Initializing DataIngestion pipeline...")
        self.model_loader = ModelLoader()
        self._load_env_variables()
        self.csv_path = self._get_csv_path()
        self.product_data = self._load_csv()
        self.config = load_config()

    def _load_env_variables(self):
        """
        Load and validate required environment variables.
        """
        load_dotenv()

        required_vars = ["GOOGLE_API_KEY", "ASTRA_DB_ENDPOINT", "ASTRA_DB_APPLICATION_TOKEN", "ASTRA_DB_KEYSPACE"]
        missing_vars = [var for var in required_vars if os.getenv(var) is None]
        if missing_vars:
            raise EnvironmentError(f"Missing environment variables: {missing_vars}")

        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.db_api_endpoint = os.getenv("ASTRA_DB_ENDPOINT")
        self.db_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        self.db_keyspace = os.getenv("ASTRA_DB_KEYSPACE")

    def _get_csv_path(self):
        """
        Get path to the updated CSV file inside 'data' folder.
        """
        current_dir = os.getcwd()
        csv_path = os.path.join(current_dir, 'data', 'flipkart_product_review.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at: {csv_path}")
        return csv_path

    def _load_csv(self):
        """
        Load updated product data from CSV.
        """
        df = pd.read_csv(self.csv_path)
        expected_columns = {'product_name', 'price', 'stock_status', 'rating', 'num_reviews'}

        if not expected_columns.issubset(set(df.columns)):
            raise ValueError(f"CSV must contain columns: {expected_columns}")

        return df

    def transform_data(self):
        """
        Transform product data into list of LangChain Document objects.
        Create pseudo-reviews from product metadata.
        """
        documents = []

        for _, row in self.product_data.iterrows():
            pseudo_review = (
                f"The product '{row['product_name']}' is priced at ₹{row['price']}. "
                f"It has a rating of {row['rating']} from {row['num_reviews']} reviews. "
                f"Stock status: {row['stock_status']}."
            )

            metadata = {
                "product_name": row['product_name'],
                "price": row['price'],
                "stock_status": row['stock_status'],
                "rating": row['rating'],
                "num_reviews": row['num_reviews']
            }

            doc = Document(page_content=pseudo_review, metadata=metadata)
            documents.append(doc)

        print(f"Transformed {len(documents)} documents.")
        return documents

    def store_in_vector_db(self, documents: List[Document]):
        """
        Store documents into AstraDB vector store.
        """
        collection_name = self.config["astra_db"]["collection_name"]
        vstore = AstraDBVectorStore(
            embedding=self.model_loader.load_embeddings(),
            collection_name=collection_name,
            api_endpoint=self.db_api_endpoint,
            token=self.db_application_token,
            namespace=self.db_keyspace,
        )

        inserted_ids = vstore.add_documents(documents)
        print(f"Successfully inserted {len(inserted_ids)} documents into AstraDB.")
        return vstore, inserted_ids

    def run_pipeline(self):
        """
        Run the full data ingestion pipeline: transform data and store into vector DB.
        """
        documents = self.transform_data()
        vstore, inserted_ids = self.store_in_vector_db(documents)

        # Optionally do a quick search
        query = "Can you suggest budget-friendly products?"
        results = vstore.similarity_search(query)

        print(f"\nSample search results for query: '{query}'")
        for res in results:
            print(f"Content: {res.page_content}\nMetadata: {res.metadata}\n")


# Run if this file is executed directly
if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.run_pipeline()

