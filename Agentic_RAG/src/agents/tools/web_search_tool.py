from langchain.tools import tool
import requests
from bs4 import BeautifulSoup

@tool
def web_search(query: str) -> str:
    """
    Performs a web search using DuckDuckGo to find current information.
    Use this when you need up-to-date information not in the documents.
    
    Args:
        query: The search query string
        
    Returns:
        Search results with relevant information
    """
    try:
        # Use DuckDuckGo HTML search (no API key needed)
        url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        results = []
        result_divs = soup.find_all('div', class_='result__body', limit=3)
        
        for div in result_divs:
            title_elem = div.find('a', class_='result__a')
            snippet_elem = div.find('a', class_='result__snippet')
            
            if title_elem and snippet_elem:
                title = title_elem.get_text(strip=True)
                snippet = snippet_elem.get_text(strip=True)
                results.append(f"• {title}\n  {snippet}")
        
        if results:
            return "Web search results:\n\n" + "\n\n".join(results)
        else:
            return f"No web results found for: {query}"
            
    except Exception as e:
        return f"Web search error: {str(e)}. Query was: {query}"
