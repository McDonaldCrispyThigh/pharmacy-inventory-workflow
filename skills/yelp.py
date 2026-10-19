"""Legacy Yelp experiment; excluded from the current empirical workflow."""

import requests
import os
from dotenv import load_dotenv

# Load API Key
load_dotenv()
API_KEY = os.getenv("YELP_API_KEY")

class YelpAgent:
    def __init__(self):
        if not API_KEY:
            print("Warning: YELP_API_KEY is missing in .env")
            self.headers = None
        else:
            self.headers = {'Authorization': f'Bearer {API_KEY}'}
        self.base_url = "https://api.yelp.com/v3/businesses"

    def search_businesses(self, term, location):
        """
        Search Yelp for businesses.
        """
        if not self.headers:
            return []

        url = f"{self.base_url}/search"
        params = {
            'term': term,
            'location': location,
            'limit': 5  # Limit to 5 to save quota
        }

        try:
            response = requests.get(url, headers=self.headers, params=params)
            if response.status_code == 200:
                return response.json().get('businesses', [])
            else:
                print(f"Yelp Error {response.status_code}: {response.text}")
                return []
        except Exception as e:
            print(f"Error calling Yelp: {e}")
            return []

    def get_business_details(self, yelp_id):
        """
        Get detailed info (categories, hours, etc.)
        """
        if not self.headers:
            return {}

        url = f"{self.base_url}/{yelp_id}"
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                return response.json()
            else:
                return {}
        except Exception as e:
            print(f"Error getting details: {e}")
            return {}

# Test
if __name__ == "__main__":
    agent = YelpAgent()
    results = agent.search_businesses("Starbucks", "Boulder, CO")
    print(f"Found {len(results)} shops on Yelp.")
