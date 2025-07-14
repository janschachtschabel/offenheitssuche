"""
Alternative Search Engine Tool for Offenheitscrawler
Uses DuckDuckGo search, proxy rotation, and web crawling as alternative to Gemini CLI
"""

import os
import time
import random
import requests
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from loguru import logger
from openai import OpenAI
import re
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Global shared proxy rotator instance to prevent multiple proxy refreshes
_shared_proxy_rotator = None
_proxy_rotator_lock = threading.Lock()


def get_shared_proxy_rotator() -> 'ProxyRotator':
    """Get or create the shared ProxyRotator instance to prevent multiple proxy refreshes"""
    global _shared_proxy_rotator
    
    with _proxy_rotator_lock:
        if _shared_proxy_rotator is None:
            logger.info("Creating shared ProxyRotator instance (first time)")
            _shared_proxy_rotator = ProxyRotator()
        else:
            logger.debug("Reusing existing shared ProxyRotator instance")
        return _shared_proxy_rotator


class ProxyRotator:
    """Manages proxy rotation for web requests with enhanced proxy fetching and testing"""
    
    def __init__(self, min_proxies: int = 3, timeout: int = 5, use_proxies: bool = True):
        self.min_proxies = min_proxies  # Reduced to 3 - prefer direct connection
        self.timeout = timeout
        self.use_proxies = use_proxies  # Option to disable proxies completely
        self.proxies: List[str] = []  # Working proxies only
        self.current_index = 0
        self.lock = threading.Lock()
        self.last_refresh = 0
        self.refresh_interval = 14400  # 240 minutes (4 hours)
        self._current_proxy = None
        self.last_request_time = 0
        self.crawl_rate_limit = 1.0  # 1 second between crawl requests (60/min)
        logger.info(f"Initializing ProxyRotator with target of {min_proxies} working proxies (minimum: 5)")
        self._refresh_proxies()

    def _refresh_proxies(self):
        """Refresh proxy list if needed"""
        if not self.use_proxies:
            logger.debug("Proxies disabled - using direct connections only")
            self.proxies = []
            return
            
        current_time = time.time()
        
        # Check if refresh is needed
        if (len(self.proxies) >= self.min_proxies and 
            current_time - self.last_refresh < self.refresh_interval):
            logger.debug(f"Proxies are fresh ({len(self.proxies)} available, "
                        f"last refresh {current_time - self.last_refresh:.1f}s ago, threshold: {self.min_proxies})")
            return
        
        logger.info(f"Refreshing proxy list ({len(self.proxies)} current, target {self.min_proxies})...")
        logger.info(f"Aktualisiere Proxy-Liste ({len(self.proxies)} aktuell, Ziel {self.min_proxies})...")
        working_proxies = []
        
        # Try different proxy sources sequentially until we have enough
        proxy_sources = [
            self._get_free_proxy_list_proxies,
            self._get_proxy_list_org_proxies,
            self._get_spys_one_proxies,
            self._get_fallback_proxies  # High-quality DE/EU/Western proxies
        ]
        
        for source_func in proxy_sources:
            if len(working_proxies) >= self.min_proxies:
                logger.info(f"Found enough proxies ({len(working_proxies)}), stopping source testing")
                break
                
            try:
                logger.info(f"Trying proxy source: {source_func.__name__}")
                source_proxies = source_func()
                
                # Test limited number of proxies per source (max 5 for fastest testing)
                test_proxies = source_proxies[:5]
                logger.info(f"Testing {len(test_proxies)} proxies from {source_func.__name__}")
                
                for proxy in test_proxies:
                    if len(working_proxies) >= self.min_proxies:
                        break
                    if self._test_proxy(proxy):
                        working_proxies.append(proxy)
                        logger.info(f"✓ Working proxy found: {proxy}")
                        
            except Exception as e:
                logger.warning(f"Proxy source {source_func.__name__} failed: {e}")
                continue
        
        if working_proxies:
            self.proxies = working_proxies
            self.last_refresh = current_time
            self.current_index = 0
            logger.info(f"Successfully loaded {len(working_proxies)} working proxies")
            logger.info(f"Erfolgreich {len(working_proxies)} funktionierende Proxies geladen")
        else:
            logger.warning("No working proxies found from any source")
            logger.warning("Keine funktionierenden Proxies aus allen Quellen gefunden")
        
        # Only warn if we have very few proxies (less than 5)
        if len(self.proxies) == 0:
            logger.warning("No working proxies found, using direct connection")
        elif len(self.proxies) < 5:
            logger.warning(f"Only {len(self.proxies)} working proxies found (minimum recommended: 5)")
        elif len(self.proxies) < self.min_proxies:
            logger.info(f"Found {len(self.proxies)} working proxies (target was {self.min_proxies}, but this is sufficient)")

    def _get_german_proxies(self) -> List[str]:
        """Get German-specific proxies"""
        try:
            url = "https://www.proxyscan.io/api/proxy?format=txt&country=DE"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code != 200:
                return []
            
            proxies = response.text.splitlines()
            logger.info(f"Found {len(proxies)} German proxy candidates")
            return proxies[:15]  # Max 15 candidates
            
        except Exception as e:
            logger.warning(f"Failed to get German proxies: {e}")
            return []

    def _get_free_proxy_list_proxies(self) -> List[str]:
        """Get proxies from free-proxy-list.net"""
        try:
            url = "https://free-proxy-list.net/"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code != 200:
                return []
            
            # Enhanced regex extraction for better matching
            proxy_pattern = r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})</td><td>(\d{2,5})'
            matches = re.findall(proxy_pattern, response.text)
            
            proxies = [f"http://{ip}:{port}" for ip, port in matches]
            logger.info(f"Found {len(proxies)} proxy candidates from free-proxy-list.net")
            return proxies[:50]  # Increased to 50 candidates
            
        except Exception as e:
            logger.warning(f"Failed to get free-proxy-list proxies: {e}")
            return []
    
    def _get_proxy_list_org_proxies(self) -> List[str]:
        """Get proxies from proxy-list.org"""
        try:
            url = "https://proxy-list.org/english/index.php"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code != 200:
                return []
            
            # Extract proxies from proxy-list.org format
            proxy_pattern = r'Proxy\(\'([^\']*)\'\'\)'
            matches = re.findall(proxy_pattern, response.text)
            
            proxies = []
            for match in matches:
                try:
                    # Decode base64 if needed
                    import base64
                    decoded = base64.b64decode(match).decode('utf-8')
                    if ':' in decoded:
                        ip, port = decoded.split(':', 1)
                        proxies.append(f"http://{ip}:{port}")
                except:
                    continue
            
            logger.info(f"Found {len(proxies)} proxy candidates from proxy-list.org")
            return proxies[:30]  # Max 30 candidates
            
        except Exception as e:
            logger.warning(f"Failed to get proxy-list.org proxies: {e}")
            return []
    
    def _get_spys_one_proxies(self) -> List[str]:
        """Get proxies from spys.one"""
        try:
            url = "http://spys.one/en/http-proxy-list/"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code != 200:
                return []
            
            # Extract proxies from spys.one format
            proxy_pattern = r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d{2,5})'
            matches = re.findall(proxy_pattern, response.text)
            
            proxies = [f"http://{ip}:{port}" for ip, port in matches]
            logger.info(f"Found {len(proxies)} proxy candidates from spys.one")
            return proxies[:30]  # Max 30 candidates
            
        except Exception as e:
            logger.warning(f"Failed to get spys.one proxies: {e}")
            return []
    
    def _get_fallback_proxies(self) -> List[str]:
        """Get a small list of very reliable fallback proxies - prefer direct connection"""
        # Reduced to only the most reliable proxies - prefer direct connection over unreliable proxies
        fallback_proxies = [
            # Only very reliable, tested proxies
            "http://178.62.201.21:8080",     # Niederlande, sehr zuverlässig
            "http://139.59.1.14:80",         # Niederlande, funktioniert oft
            "http://165.22.81.188:8080",     # Niederlande, backup
            "http://104.248.90.212:80",      # USA, backup
            "http://159.89.195.14:8080",     # USA, backup
        ]
        logger.info(f"Using {len(fallback_proxies)} minimal reliable fallback proxies (prefer direct connection)")
        return fallback_proxies

    def _test_proxy(self, proxy: str) -> bool:
        """Test if a proxy is working with a simple HTTP request"""
        proxies = {'http': proxy, 'https': proxy}
        
        # Simple test with httpbin - if this works, proxy is good
        try:
            response = requests.get(
                "http://httpbin.org/ip", 
                proxies=proxies, 
                timeout=8,  # Slightly longer timeout
                allow_redirects=True,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            )
            return response.status_code == 200
        except:
            # If httpbin fails, try a backup test
            try:
                response = requests.get(
                    "http://www.google.com", 
                    proxies=proxies, 
                    timeout=8,
                    allow_redirects=True,
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                )
                return response.status_code in [200, 301, 302]
            except:
                return False
    
    def _test_proxy_with_speed(self, proxy: str) -> tuple:
        """Test proxy and return (is_working, response_time)"""
        test_urls = [
            "http://httpbin.org/ip",
            "http://www.google.com"
        ]
        
        proxies = {'http': proxy, 'https': proxy}
        total_time = 0
        success_count = 0
        
        for test_url in test_urls:
            try:
                start_time = time.time()
                response = requests.get(
                    test_url, 
                    proxies=proxies, 
                    timeout=self.timeout,
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                )
                response_time = time.time() - start_time
                
                if response.status_code in [200, 301, 302]:
                    success_count += 1
                    total_time += response_time
                    if success_count >= 1:  # At least 1 successful test for speed measurement
                        avg_time = total_time / success_count
                        return (True, avg_time)
            except:
                continue
                
        return (False, float('inf'))
    
    def _test_proxies_parallel(self, proxy_list: List[str]) -> List[str]:
        """Test multiple proxies in parallel and sort by speed"""
        working_proxies = []
        
        logger.info(f"Testing {len(proxy_list)} proxy candidates in parallel...")
        
        # Use ThreadPoolExecutor for parallel testing
        with ThreadPoolExecutor(max_workers=20) as executor:
            # Submit all proxy tests with speed measurement
            future_to_proxy = {executor.submit(self._test_proxy_with_speed, proxy): proxy for proxy in proxy_list}
            
            # Collect results as they complete
            proxy_results = []
            for future in as_completed(future_to_proxy, timeout=60):
                proxy = future_to_proxy[future]
                try:
                    is_working, response_time = future.result()
                    if is_working:
                        proxy_results.append((proxy, response_time))
                        logger.debug(f"✓ Working proxy: {proxy} ({response_time:.2f}s)")
                        
                        # Stop when we have enough working proxies
                        if len(proxy_results) >= self.min_proxies * 1.5:  # Get 50% more than needed
                            logger.info(f"Found enough working proxies ({len(proxy_results)}), stopping tests")
                            break
                    else:
                        logger.debug(f"✗ Failed proxy: {proxy}")
                except Exception as e:
                    logger.debug(f"✗ Error testing proxy {proxy}: {e}")
        
        # Sort by response time (fastest first) and extract proxy URLs
        proxy_results.sort(key=lambda x: x[1])
        working_proxies = [proxy for proxy, _ in proxy_results]
        
        if working_proxies:
            fastest_time = proxy_results[0][1]
            slowest_time = proxy_results[-1][1]
            logger.info(f"Proxy testing completed: {len(working_proxies)} working proxies (fastest: {fastest_time:.2f}s, slowest: {slowest_time:.2f}s)")
        else:
            logger.info(f"Proxy testing completed: 0 working proxies found")
            
        return working_proxies

    @property
    def current_proxy(self) -> Optional[str]:
        """Get the currently used proxy"""
        return self._current_proxy

    def get_proxy(self) -> Optional[str]:
        """Get the next working proxy"""
        self._refresh_proxies()
        
        if not self.proxies:
            return None

        with self.lock:
            self._current_proxy = self.proxies[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.proxies)
            logger.debug(f"Using proxy: {self._current_proxy}")
            return self._current_proxy

    def get_request_kwargs(self) -> dict:
        """Get request kwargs with proxy settings and rate limiting"""
        # Apply rate limiting for crawling (60 requests per minute = 1 second pause)
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.crawl_rate_limit:
            sleep_time = self.crawl_rate_limit - time_since_last
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        
        proxy = self.get_proxy()
        if not proxy:
            return {"timeout": self.timeout}  # Keep timeout even without proxy
            
        return {
            "proxies": {
                "http": proxy,
                "https": proxy
            },
            "timeout": self.timeout
        }


class DuckDuckGoSearcher:
    """DuckDuckGo search functionality with proxy support using DDGS library"""
    
    def __init__(self, proxy_rotator: ProxyRotator, max_results: int = 10):
        self.proxy_rotator = proxy_rotator
        self.max_results = max_results
        self.last_search_time = 0  # Track last search time for rate limiting
        self.rate_limit_seconds = 2  # 2 seconds between searches
        
    def search(self, query: str, region: str = "de-de") -> List[Dict[str, str]]:
        """Perform DuckDuckGo search - try direct connection first, then proxies as fallback"""
        # Rate limiting: Wait if necessary
        current_time = time.time()
        time_since_last_search = current_time - self.last_search_time
        
        if time_since_last_search < self.rate_limit_seconds:
            wait_time = self.rate_limit_seconds - time_since_last_search
            logger.info(f"Rate limiting: waiting {wait_time:.1f} seconds before search")
            time.sleep(wait_time)
        
        self.last_search_time = time.time()
        
        results = []
        timeout = 30
        
        logger.info(f"Searching DuckDuckGo for: {query}")
        
        # Use DDGS library for reliable search
        from ddgs import DDGS
        
        # Strategy 1: Try direct connection first (fastest and most reliable)
        try:
            logger.info("Attempting direct connection (no proxy)")
            with DDGS(proxy=None, timeout=timeout) as ddgs:
                search_results = list(ddgs.text(
                    keywords=query,
                    region=region,
                    safesearch="moderate",
                    max_results=self.max_results
                ))
                
                # Convert to expected format
                for result in search_results:
                    results.append({
                        'title': result.get('title', ''),
                        'href': result.get('href', ''),
                        'body': result.get('body', '')
                    })
                
                if results:
                    logger.info(f"✅ Direct connection successful - found {len(results)} results")
                    return results
                    
        except Exception as e:
            logger.warning(f"Direct connection failed: {str(e)}")
            logger.info("Falling back to proxy connections...")
        
        # Strategy 2: Fallback to proxies if direct connection fails
        if self.proxy_rotator.use_proxies:
            max_proxy_attempts = 3
            for attempt in range(max_proxy_attempts):
                try:
                    # Get proxy settings
                    request_kwargs = self.proxy_rotator.get_request_kwargs()
                    proxies = request_kwargs.get('proxies', None)
                    
                    if not proxies:
                        logger.warning("No proxies available for fallback")
                        break
                    
                    proxy = proxies.get('https') or proxies.get('http')
                    logger.info(f"Trying proxy fallback (attempt {attempt + 1}/{max_proxy_attempts}): {proxy}")
                    
                    with DDGS(proxy=proxy, timeout=timeout) as ddgs:
                        search_results = list(ddgs.text(
                            keywords=query,
                            region=region,
                            safesearch="moderate",
                            max_results=self.max_results
                        ))
                        
                        # Convert to expected format
                        for result in search_results:
                            results.append({
                                'title': result.get('title', ''),
                                'href': result.get('href', ''),
                                'body': result.get('body', '')
                            })
                        
                        if results:
                            logger.info(f"✅ Proxy fallback successful - found {len(results)} results")
                            return results
                            
                except Exception as e:
                    logger.warning(f"Proxy attempt {attempt + 1} failed: {str(e)}")
                    if attempt < max_proxy_attempts - 1:
                        # Try next proxy
                        self.proxy_rotator.get_proxy()
                        time.sleep(1)  # Brief pause between proxy attempts
        else:
            logger.info("Proxies disabled - no fallback available")
            
        logger.info(f"Found {len(results)} search results")
        return results


class WebCrawler:
    """Web crawler with proxy support"""
    
    def __init__(self, proxy_rotator: ProxyRotator, max_content_length: int = 50000):
        self.proxy_rotator = proxy_rotator
        self.max_content_length = max_content_length
        self.session = requests.Session()
        
    def crawl_url(self, url: str) -> Dict[str, Any]:
        """Crawl a single URL - try direct connection first, then proxies as fallback"""
        try:
            # Headers to mimic browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'de-DE,de;q=0.9,en;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
            }
            
            response = None
            
            # Strategy 1: Try direct connection first (fastest and most reliable)
            try:
                logger.debug(f"Crawling URL with direct connection: {url}")
                response = self.session.get(url, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    logger.debug(f"✅ Direct connection successful for {url}")
                else:
                    logger.warning(f"Direct connection failed for {url}: Status {response.status_code}")
                    response = None  # Reset for proxy fallback
                    
            except Exception as e:
                logger.warning(f"Direct connection failed for {url}: {str(e)}")
                logger.debug("Falling back to proxy connections...")
                response = None
            
            # Strategy 2: Fallback to proxies if direct connection fails
            if not response and self.proxy_rotator.use_proxies:
                max_proxy_attempts = 2
                for attempt in range(max_proxy_attempts):
                    try:
                        # Get proxy settings
                        request_kwargs = self.proxy_rotator.get_request_kwargs()
                        request_kwargs['headers'] = headers
                        
                        if not request_kwargs.get('proxies'):
                            logger.warning("No proxies available for fallback")
                            break
                        
                        proxy_info = request_kwargs['proxies'].get('https') or request_kwargs['proxies'].get('http')
                        logger.debug(f"Trying proxy fallback for {url} (attempt {attempt + 1}/{max_proxy_attempts}): {proxy_info}")
                        
                        response = self.session.get(url, **request_kwargs)
                        
                        if response.status_code == 200:
                            logger.debug(f"✅ Proxy fallback successful for {url}")
                            break
                        else:
                            logger.warning(f"Proxy attempt {attempt + 1} failed for {url}: Status {response.status_code}")
                            response = None
                            if attempt < max_proxy_attempts - 1:
                                # Try next proxy
                                self.proxy_rotator.get_proxy()
                                
                    except Exception as e:
                        logger.warning(f"Proxy attempt {attempt + 1} failed for {url}: {str(e)}")
                        response = None
                        if attempt < max_proxy_attempts - 1:
                            # Try next proxy
                            self.proxy_rotator.get_proxy()
                            time.sleep(1)  # Brief pause between proxy attempts
            elif not response:
                logger.info(f"Proxies disabled - no fallback available for {url}")
            
            if not response or response.status_code != 200:
                status_code = response.status_code if response else "No response"
                return {
                    'url': url,
                    'title': '',
                    'content': '',
                    'error': f'HTTP {status_code}'
                }
            
            # Parse HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title_tag = soup.find('title')
            title = title_tag.get_text(strip=True) if title_tag else ''
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Extract text content
            text_content = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text_content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text_content = ' '.join(chunk for chunk in chunks if chunk)
            
            # Limit content length
            if len(text_content) > self.max_content_length:
                text_content = text_content[:self.max_content_length] + "..."
            
            return {
                'url': url,
                'title': title,
                'content': text_content,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Error crawling {url}: {e}")
            return {
                'url': url,
                'title': '',
                'content': '',
                'error': str(e)
            }
    
    def crawl_multiple_urls(self, urls: List[str], max_workers: int = 3) -> List[Dict[str, Any]]:
        """Crawl multiple URLs concurrently"""
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all crawling tasks
            future_to_url = {executor.submit(self.crawl_url, url): url for url in urls}
            
            # Collect results as they complete
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error in concurrent crawling of {url}: {e}")
                    results.append({
                        'url': url,
                        'title': '',
                        'content': '',
                        'error': str(e)
                    })
        
        return results


class SearchEngineTool:
    """Main search engine tool combining DuckDuckGo search, web crawling, and OpenAI analysis"""
    
    def __init__(self, openai_api_key: str, 
                 openai_base_url: str = "https://api.openai.com/v1",
                 openai_model: str = "gpt-4o-mini",
                 max_results: int = 20):
        # Initialize OpenAI client with provided parameters
        self.openai_client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_base_url
        )
        self.openai_model = openai_model
        self.max_results = max_results
        
        # Initialize components with shared proxy rotator to prevent multiple proxy refreshes
        self.proxy_rotator = get_shared_proxy_rotator()
        self.searcher = DuckDuckGoSearcher(self.proxy_rotator, max_results)
        self.crawler = WebCrawler(self.proxy_rotator)
        
        logger.info(f"SearchEngineTool initialized successfully with model: {openai_model} (using shared proxy rotator)")
    
    def search_and_analyze(self, organization_name: str, organization_url: str, 
                          criterion_description: str, search_patterns: List[str] = None) -> str:
        """
        Main method: Search for information about an organization's criterion and analyze with OpenAI
        
        Args:
            organization_name: Name of the organization
            organization_url: URL of the organization
            criterion_description: Description of the criterion to check
            search_patterns: Optional search patterns/keywords
            
        Returns:
            Comprehensive analysis with sources
        """
        try:
            # Construct search query
            search_query = f"{organization_name} {criterion_description}"
            if search_patterns:
                search_query += " " + " ".join(search_patterns)
            
            logger.info(f"Starting search and analysis for {organization_name} - {criterion_description}")
            
            # Step 1: Perform DuckDuckGo search
            search_results = self.searcher.search(search_query)
            
            if not search_results:
                logger.warning(f"No search results found for {organization_name}")
                return f"Keine Suchergebnisse für {organization_name} gefunden."
            
            # Step 2: Extract URLs to crawl (prioritize organization's own website)
            urls_to_crawl = []
            
            # Add organization's main URL first
            if organization_url:
                urls_to_crawl.append(organization_url)
            
            # Add search result URLs
            for result in search_results[:self.max_results]:
                url = result.get('href', '')  # Use 'href' key from DDGS results
                if url and url not in urls_to_crawl:
                    urls_to_crawl.append(url)
            
            # Limit to top 20 URLs
            urls_to_crawl = urls_to_crawl[:20]
            
            # Step 3: Crawl the URLs
            logger.info(f"Crawling {len(urls_to_crawl)} URLs")
            crawled_content = []
            for url in urls_to_crawl:
                crawl_result = self.crawler.crawl_url(url)
                if crawl_result.get('content') and not crawl_result.get('error'):
                    # Find the corresponding search result for title
                    title = url  # Default to URL
                    for result in search_results:
                        if result.get('href', '') == url:
                            title = result.get('title', url)
                            break
                    
                    crawled_content.append({
                        'title': title,
                        'url': url,
                        'content': crawl_result['content'][:2000]  # Limit content length
                    })
            
            # Step 4: Prepare content for OpenAI analysis
            content_for_analysis = []
            sources = []
            
            for content in crawled_content:
                if content.get('content'):  # content from crawled_content already filtered
                    content_for_analysis.append({
                        'url': content['url'],
                        'title': content['title'],
                        'content': content['content'][:5000]  # Limit per source
                    })
                    sources.append(content['url'])
            
            if not content_for_analysis:
                return f"Keine verwertbaren Inhalte für {organization_name} gefunden."
            
            # Step 5: Analyze with OpenAI
            analysis_result = self._analyze_with_openai(
                organization_name, 
                criterion_description, 
                content_for_analysis
            )
            
            # Step 6: Format final response with sources
            final_response = f"{analysis_result}\n\n**Quellen:**\n"
            for i, source in enumerate(sources[:5], 1):  # Limit to 5 sources
                final_response += f"{i}. {source}\n"
            
            logger.info(f"Analysis completed for {organization_name}")
            return final_response
            
        except Exception as e:
            logger.error(f"Error in search_and_analyze for {organization_name}: {e}")
            return f"Fehler bei der Analyse von {organization_name}: {str(e)}"
    
    def _analyze_with_openai(self, organization_name: str, criterion_description: str, 
                           content_data: List[Dict[str, str]]) -> str:
        """Analyze crawled content with OpenAI"""
        try:
            # Prepare content summary for OpenAI
            content_summary = ""
            for i, content in enumerate(content_data[:5], 1):  # Limit to 5 sources
                content_summary += f"\n--- Quelle {i}: {content['title']} ({content['url']}) ---\n"
                content_summary += content['content'][:3000]  # Limit per source
                content_summary += "\n"
            
            # Create prompt for OpenAI
            prompt = f"""Analysiere die folgenden Webinhalte über die Organisation "{organization_name}" bezüglich des Kriteriums: {criterion_description}

Webinhalte:
{content_summary}

Fasse alle relevanten Informationen zu diesem Kriterium zusammen:

1. Welche konkreten Informationen gibt es zu diesem Kriterium in den Texten?
2. Welche Belege und Details sind verfügbar?
3. Auf welchen Seiten/URLs wurden die relevanten Informationen gefunden?
4. Welche spezifischen Maßnahmen, Programme oder Initiativen werden erwähnt?

Antworte als strukturierte Zusammenfassung auf Deutsch. Zitiere konkrete Textpassagen als Belege. Bewerte NICHT, sondern sammle nur Informationen."""

            # Call OpenAI API
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,  # Use the configured model
                messages=[
                    {"role": "system", "content": "Du bist ein Experte für die Sammlung und Zusammenfassung von Informationen über Organisationen. Sammle alle relevanten Informationen zu den gegebenen Kriterien aus den Webinhalten. Bewerte nicht, sondern fasse nur zusammen."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=16000,  # Use reasonable max tokens
                temperature=0.3
            )
            
            analysis_result = response.choices[0].message.content
            logger.info(f"OpenAI analysis completed for {organization_name}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"OpenAI analysis failed for {organization_name}: {e}")
            return f"Fehler bei der OpenAI-Analyse: {str(e)}"


# Example usage and testing
if __name__ == "__main__":
    # Test the search engine tool
    try:
        # Initialize the tool
        search_tool = SearchEngineTool()
        
        # Test search and analysis
        result = search_tool.search_and_analyze(
            organization_name="Wikimedia Deutschland",
            organization_url="https://www.wikimedia.de",
            criterion_description="Open Educational Resources",
            search_patterns=["OER", "Bildungsressourcen", "freie Lernmaterialien"]
        )
        
        print("=== Test Result ===")
        print(result)
        
    except Exception as e:
        print(f"Test failed: {e}")
