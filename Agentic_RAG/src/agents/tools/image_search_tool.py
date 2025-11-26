from langchain.tools import tool
from serpapi import GoogleSearch
import os

@tool
def image_search(query: str) -> str:
    """
    Searches for relevant images using SerpAPI Google Images.
    Use this when you need to find images that illustrate concepts or topics.
    
    Args:
        query: The image search query
        
    Returns:
        List of image URLs with descriptions
    """
    try:
        # Get API key from environment
        api_key = os.getenv("SERPAPI_API_KEY")
        
        if not api_key:
            return "Error: SERPAPI_API_KEY not found in .env file. Please add it to use image search."
        
        # Search using SerpAPI
        params = {
            "engine": "google_images",
            "q": query,
            "api_key": api_key,
            "num": 5  # Get 5 images
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
        if "images_results" in results and results["images_results"]:
            images = results["images_results"][:5]
            
            result = f"Found {len(images)} images for '{query}':\n\n"
            for i, img in enumerate(images, 1):
                title = img.get("title", "Untitled")
                url = img.get("original", img.get("thumbnail", ""))
                source = img.get("source", "Unknown source")
                
                result += f"{i}. {title}\n"
                result += f"   URL: {url}\n"
                result += f"   Source: {source}\n\n"
            
            return result
        else:
            return f"No images found for: {query}"
            
    except Exception as e:
        return f"Image search error: {str(e)}. Make sure SERPAPI_API_KEY is set in .env file."
