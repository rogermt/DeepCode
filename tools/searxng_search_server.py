"""  
SearXNG MCP Server for DeepCode Integration  
  
This module implements a Model Context Protocol (MCP) server that provides search  
functionality to DeepCode by communicating with a local SearXNG instance. It serves  
as a replacement for Brave/Bocha search providers, offering privacy-respecting  
metasearch capabilities without requiring API keys.  
  
Features:  
--------  
- Web search via SearXNG's JSON API  
- Image search with multiple engine support  
- Bot detection bypass with proper headers  
- Error handling and timeout management  
- Response formatting compatible with DeepCode's expectations  
  
Configuration:  
-------------  
The server requires a running SearXNG instance at http://localhost:8888 with  
JSON format enabled. The following headers are automatically included to bypass 

formats:
    - html
    - json

SearXNG's bot detection:  
    - X-Forwarded-For: 127.0.0.1  
    - X-Real-IP: 127.0.0.1  
  
Usage:  
-----  
    python3 tools/searxng_search_server.py  
  
The server registers two MCP tools:  
    - searxng_web_search(query, count, time_range)  
    - searxng_image_search(query, count)  
  
Integration with DeepCode:  
-------------------------  
Add to mcp_agent.config.yaml:  
    searxng:  
      args:  
        - tools/searxng_search_server.py  
      command: python3  
      env:  
        PYTHONPATH: .  
  
Dependencies:  
------------  
- httpx: HTTP client for async requests  
- mcp: Model Context Protocol framework  
- lxml: HTML parsing (for fallback parsing)  
  
Notes:  
-----  
- Requires SearXNG instance to be running before starting  
- Uses POST requests to /search endpoint with format=json  
- Returns results in DeepCode-compatible text format  
"""
import os  
import sys  
import asyncio  
from typing import Optional  
import httpx  
from mcp.server.fastmcp import FastMCP  
from urllib.parse import urlencode, urlparse 
  
server = FastMCP("searxng-search")  
  
# Local SearXNG instance configuration  
SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://localhost:8888")  
BASE_URL = SEARXNG_URL  # Use the configurable URL  
  
# Extract host for headers  
parsed_url = urlparse(SEARXNG_URL)  
host_header = f"{parsed_url.hostname}:{parsed_url.port}" if parsed_url.port else parsed_url.hostname

HEADERS = {  
    'Content-Type': 'application/x-www-form-urlencoded',  
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',  
    'X-Forwarded-For': '127.0.0.1',  
    'X-Real-IP': '127.0.0.1',  
    'Host': host_header  # Dynamic host from SEARXNG_URL  
}
TIME_RANGE_MAP = {  
    "day": "day",  
    "week": "week",   
    "month": "month",  
    "year": "year",  
}  

  
@server.tool()  
async def searxng_web_search(  
    query: str,   
    count: int = 10,  
    time_range: Optional[str] = None  
) -> str:  
    """Search using local SearXNG instance.  
      
    Args:  
        query: Search query (required)  
        count: Number of results (1-20, default 10)  
        time_range: Time filter (day, week, month, year)  
    """  
      
    # Build request data for SearXNG API  
    data = {  
        'q': query,  
        'pageno': 1,  
        'language': 'en-US',  
        'time_range': time_range or '',  
        'category': 'general',  
        'format': 'json',  
    }        

    try:  
        async with httpx.AsyncClient() as client:  
            # Make POST request to local SearXNG  
            response = await client.post(  
                f"{BASE_URL}/search",  
                data=data,  
                headers=HEADERS,  
                timeout=10.0  
            )  
            response.raise_for_status()  
              
            # Parse JSON response from SearXNG  
            resp_json = response.json()  
            results = resp_json.get('results', [])  
              
            if not results:  
                return "No results found."  
              
            # Format results to match DeepCode's expected format  
            formatted_results = []  
            for result in results[:count]:  
                formatted_results.append(  
                    f"Title: {result.get('title', 'No title')}\n"  
                    f"URL: {result.get('url', 'No URL')}\n"  
                    f"Description: {result.get('content', 'No description')}\n"  
                    f"Published date: N/A\n"  
                    f"Site name: {result.get('engine', 'SearXNG')}"  
                )  
              
            return "\n\n".join(formatted_results)  
              
    except httpx.HTTPStatusError as e:  
        return f"SearXNG API HTTP error: {e.response.status_code}"  
    except Exception as e:  
        return f"SearXNG API error: {str(e)}"  
  
@server.tool()  
async def searxng_image_search(  
    query: str,  
    count: int = 10  
) -> str:  
    """Search for images using local SearXNG instance.  
      
    Args:  
        query: Search query (required)  
        count: Number of results (1-20, default 10)  
    """  
      
    data = {  
        'q': query,  
        'pageno': 1,  
        'language': 'en-US',  
        'category': 'images',  
        'format': 'json',  
    }  
      
    try:  
        async with httpx.AsyncClient() as client:  
            response = await client.post(  
                f"{BASE_URL}/search",  
                data=data,  
                headers=HEADERS,  
                timeout=10.0  
            )  
            response.raise_for_status()  
              
            resp_json = response.json()  
            results = resp_json.get('results', []) 

            if not results:  
                return "No image results found."  
            
            # Filter for actual image results only  
            image_results = []  
            for result in results[:count]:  
                if result.get('template') == 'images.html':  
                    # Try multiple possible image URL fields  
                    img_url = (  
                        result.get('img_src') or   
                        result.get('thumbnail_src') or   
                        result.get('src') or  
                        'No image URL'  
                    )  
                    image_results.append(  
                        f"Title: {result.get('title', 'No title')}\n"  
                        f"URL: {result.get('url', 'No URL')}\n"  
                        f"Image URL: {img_url}\n"  
                        f"Source: SearXNG Images"  
                    )  
  
            if not image_results:  
                return "No direct image results found. All results were web pages."  
            
            return "\n\n".join(image_results) 
               
    except Exception as e:  
        return f"Image search error: {str(e)}"  
  
def main():  
    """Initialize and run the SearXNG MCP server."""  
    print("Starting SearXNG Search MCP server...", file=sys.stderr)  
    print("Connecting to: http://localhost:8888", file=sys.stderr)  
    server.run(transport="stdio")  
  
if __name__ == "__main__":  
    main()