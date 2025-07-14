import streamlit as st
import pandas as pd
import yaml
import os
import subprocess
import json
from pathlib import Path
from loguru import logger
from openai import OpenAI
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
import time
import re
from search_engine_tool import SearchEngineTool

# Configure loguru
logger.add("offenheitscrawler.log", rotation="10 MB", retention="7 days", level="INFO")

class OffenheitsCrawler:
    def __init__(self):
        self.criteria_dir = Path("criteria")
        self.data_dir = Path("data")
        self.openai_client = None
        self.search_engine_tool = None
        self.setup_openai()
        # SearchEngineTool wird nur bei Bedarf initialisiert
        
    def setup_openai(self):
        """Initialize OpenAI client with configuration from session state"""
        try:
            api_key = st.session_state.get('openai_api_key', os.getenv('OPENAI_API_KEY'))
            base_url = st.session_state.get('openai_base_url', 'https://api.openai.com/v1')
            
            if api_key:
                self.openai_client = OpenAI(
                    api_key=api_key,
                    base_url=base_url
                )
                logger.info("OpenAI client initialized successfully")
            else:
                logger.warning("No OpenAI API key found")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            st.error(f"OpenAI-Initialisierung fehlgeschlagen: {e}")
    
    def setup_search_engine_tool(self):
        """Initialize SearchEngineTool as alternative to Gemini CLI (only when needed)"""
        if self.search_engine_tool is not None:
            return True  # Already initialized
            
        try:
            # Get API key and settings directly from Streamlit session state
            api_key = st.session_state.get('openai_api_key', '')
            base_url = st.session_state.get('openai_base_url', 'https://api.openai.com/v1')
            model = st.session_state.get('openai_model', 'gpt-4o-mini')
            
            logger.info(f"Initializing SearchEngineTool with API key: {'Found (' + str(len(api_key)) + ' chars)' if api_key else 'None'}")
            logger.info(f"Using model: {model}, base_url: {base_url}")
            
            if not api_key or len(api_key.strip()) == 0:
                logger.warning("No OpenAI API key found in session state")
                return False
                
            logger.info("Initializing SearchEngineTool (including proxy setup)...")
            
            # Pass all settings to SearchEngineTool
            self.search_engine_tool = SearchEngineTool(
                openai_api_key=api_key.strip(),
                openai_base_url=base_url,
                openai_model=model,
                max_results=10
            )
            
            logger.info("SearchEngineTool initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize SearchEngineTool: {e}")
            st.error(f"SearchEngineTool-Initialisierung fehlgeschlagen: {e}")
            return False

    def load_criteria_catalogs(self) -> Dict[str, str]:
        """Load available criteria catalogs"""
        catalogs = {}
        try:
            for yaml_file in self.criteria_dir.glob("*.yaml"):
                catalogs[yaml_file.stem] = str(yaml_file)
            logger.info(f"Loaded {len(catalogs)} criteria catalogs")
        except Exception as e:
            logger.error(f"Error loading criteria catalogs: {e}")
        return catalogs

    def load_criteria_from_yaml(self, yaml_path: str) -> Dict[str, Any]:
        """Load criteria from YAML file"""
        try:
            with open(yaml_path, 'r', encoding='utf-8') as file:
                criteria = yaml.safe_load(file)
            logger.info(f"Loaded criteria from {yaml_path}")
            return criteria
        except Exception as e:
            logger.error(f"Error loading criteria from {yaml_path}: {e}")
            return {}

    def load_organizations_csv(self, csv_path: str) -> pd.DataFrame:
        """Load organizations from CSV file with robust encoding handling"""
        # Try different encodings to handle various CSV formats
        encodings = ['utf-8-sig', 'utf-8', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(csv_path, sep=';', encoding=encoding, header=None, names=['Name', 'URL'])
                logger.info(f"Loaded {len(df)} organizations from {csv_path} (encoding: {encoding})")
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"Error loading organizations from {csv_path} with {encoding}: {e}")
                continue
        
        logger.error(f"Failed to load {csv_path} with any supported encoding")
        return pd.DataFrame()

    def get_csv_files(self) -> List[str]:
        """Get list of available CSV files"""
        csv_files = []
        try:
            for csv_file in self.data_dir.glob("*.csv"):
                csv_files.append(str(csv_file))
        except Exception as e:
            logger.error(f"Error getting CSV files: {e}")
        return csv_files

    def extract_criteria_list(self, criteria_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract flat list of criteria from nested YAML structure"""
        criteria_list = []
        
        def extract_recursive(data, path=""):
            if isinstance(data, dict):
                if 'criteria' in data:
                    for criterion_key, criterion_data in data['criteria'].items():
                        criterion_info = {
                            'key': criterion_key,
                            'name': criterion_data.get('name', criterion_key),
                            'description': criterion_data.get('description', ''),
                            'type': criterion_data.get('type', 'operational'),
                            'patterns': criterion_data.get('patterns', {}),
                            'weight': criterion_data.get('weight', 1.0),
                            'path': f"{path}.{criterion_key}" if path else criterion_key
                        }
                        criteria_list.append(criterion_info)
                else:
                    for key, value in data.items():
                        if key not in ['metadata']:
                            new_path = f"{path}.{key}" if path else key
                            extract_recursive(value, new_path)
        
        extract_recursive(criteria_data)
        return criteria_list

    def call_gemini_for_criterion(self, organization_name: str, organization_url: str, 
                                 criterion: Dict[str, Any]) -> str:
        """Call Gemini via command line for a specific criterion"""
        try:
            patterns = criterion.get('patterns', {}).get('text', [])
            pattern_text = ', '.join(patterns) if patterns else ''
            
            prompt = f"""Bietet {organization_name} ({organization_url}) {criterion['description']}? 
            
Suchbegriffe: {pattern_text}

Antworte genau und umfassend und nenne die Quellen wie z.B. URLs auf Seiten und Unterseiten. 
Prüfe die Hauptseite und relevante Unterseiten der Organisation."""

            logger.info(f"Calling Gemini for {organization_name} - {criterion['name']}")
            
            # Call gemini command with longer timeout using -p flag for non-interactive mode
            # Updated to use gemini-2.5-flash model
            gemini_path = r"C:\Users\jan\AppData\Roaming\npm\gemini.cmd"
            
            # Try Windows-specific path first, then fallback to generic command
            try:
                result = subprocess.run(
                    [gemini_path, '-m', 'gemini-2.5-flash', '-p', prompt], 
                    capture_output=True, 
                    text=True, 
                    timeout=120,
                    encoding='utf-8'
                )
            except (FileNotFoundError, subprocess.SubprocessError):
                # Fallback to generic gemini command
                result = subprocess.run(
                    ['gemini', '-m', 'gemini-2.5-flash', '-p', prompt], 
                    capture_output=True, 
                    text=True, 
                    timeout=120,
                    encoding='utf-8'
                )
            
            if result.returncode == 0:
                response = result.stdout.strip()
                cleaned_response = self.clean_gemini_response(response)
                logger.info(f"Gemini call successful for {organization_name} - {criterion['name']} (length: {len(cleaned_response)})")
                
                # Rate limiting: Wait 6 seconds between calls for gemini-2.5-flash (10 req/min limit)
                logger.info("Waiting 6 seconds for Gemini rate limiting (gemini-2.5-flash)...")
                time.sleep(6)
                
                return cleaned_response
            else:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                logger.error(f"Gemini call failed for {organization_name} - {criterion['name']}: {error_msg}")
                
                # Check for quota exceeded error
                if "Quota exceeded" in error_msg or "429" in error_msg:
                    return "Gemini API Quota überschritten - bitte später versuchen"
                
                return f"Fehler beim Gemini-Aufruf: {error_msg}"
                
        except subprocess.TimeoutExpired:
            logger.error(f"Gemini call timed out for {organization_name} - {criterion['name']}")
            return "Timeout beim Gemini-Aufruf (120s)"
        except Exception as e:
            logger.error(f"Error calling Gemini for {organization_name} - {criterion['name']}: {e}")
            return f"Fehler: {e}"
    
    def call_search_engine_for_criterion(self, organization_name: str, organization_url: str, 
                                        criterion: Dict[str, Any]) -> str:
        """Alternative method using SearchEngineTool instead of Gemini CLI"""
        try:
            logger.info(f"Starting call_search_engine_for_criterion for {organization_name}")
            
            # Initialize SearchEngineTool only when needed (lazy loading)
            if not self.search_engine_tool:
                logger.info("SearchEngineTool not initialized, calling setup...")
                success = self.setup_search_engine_tool()
                logger.info(f"Setup result: {success}, SearchEngineTool is now: {self.search_engine_tool is not None}")
                if not success:
                    return "SearchEngineTool-Initialisierung fehlgeschlagen - OpenAI API Key erforderlich oder Fehler beim Setup"
            
            if not self.search_engine_tool:
                logger.error("SearchEngineTool is still None after setup attempt")
                return "SearchEngineTool-Initialisierung fehlgeschlagen - bitte Logs prüfen"
            
            patterns = criterion.get('patterns', {}).get('text', [])
            
            logger.info(f"Calling SearchEngineTool for {organization_name} - {criterion['name']}")
            
            # Use the search engine tool
            response = self.search_engine_tool.search_and_analyze(
                organization_name=organization_name,
                organization_url=organization_url,
                criterion_description=criterion['description'],
                search_patterns=patterns
            )
            
            logger.info(f"SearchEngineTool call successful for {organization_name} - {criterion['name']} (length: {len(response)})")
            
            # No rate limiting needed - proxies handle this automatically
            return response
            
        except Exception as e:
            logger.error(f"Error calling SearchEngineTool for {organization_name} - {criterion['name']}: {e}")
            return f"Fehler bei SearchEngineTool: {e}"
    
    def clean_gemini_response(self, response: str) -> str:
        """Clean Gemini response to extract only the relevant answer"""
        try:
            # Remove "Loaded cached credentials." line
            if response.startswith("Loaded cached credentials."):
                lines = response.split('\n')
                response = '\n'.join(lines[1:]).strip()
            
            # If response contains WebFetchTool output, extract the final answer
            if '[WebFetchTool]' in response:
                # Look for the final answer after all the technical output
                # The pattern is usually: technical output, then the actual answer at the end
                
                # Split into sections by looking for the final answer pattern
                sections = response.split('Sources:')
                if len(sections) > 1:
                    # Take everything after the last "Sources:" section
                    final_section = sections[-1]
                    
                    # Clean up the final section
                    lines = final_section.split('\n')
                    answer_lines = []
                    
                    for line in lines:
                        line = line.strip()
                        # Skip reference markers like [1], [2], etc.
                        if re.match(r'^\[\d+\]', line):
                            continue
                        # Skip URLs in parentheses
                        if line.startswith('(http') and line.endswith(')'):
                            continue
                        # Skip empty lines
                        if not line:
                            continue
                        # Skip lines that are just technical artifacts
                        if any(skip in line for skip in ['"', '{', '}', 'candidates', 'finishReason']):
                            continue
                        
                        # This looks like actual content
                        if len(line) > 10:  # Minimum meaningful length
                            answer_lines.append(line)
                    
                    if answer_lines:
                        # Join the answer lines and return
                        return ' '.join(answer_lines)
                
                # Fallback: look for text after the last meaningful content marker
                # Find the last occurrence of actual German text (not technical output)
                lines = response.split('\n')
                answer_lines = []
                collecting_answer = False
                
                for line in lines:
                    line = line.strip()
                    
                    # Start collecting after we see a meaningful German sentence
                    if any(german_word in line.lower() for german_word in [
                        'ja,', 'nein,', 'die', 'der', 'das', 'eine', 'einen', 'universität', 
                        'bietet', 'verfügt', 'gibt', 'existiert', 'vorhanden'
                    ]) and len(line) > 20:
                        collecting_answer = True
                        answer_lines = [line]  # Start fresh
                        continue
                    
                    # If we're collecting and this looks like continuation
                    if collecting_answer and len(line) > 10:
                        # Stop if we hit technical content again
                        if any(tech in line for tech in [
                            '[WebFetchTool]', 'Full response', 'candidates', 'finishReason',
                            'usageMetadata', 'promptTokenCount'
                        ]):
                            break
                        answer_lines.append(line)
                
                if answer_lines:
                    return ' '.join(answer_lines)
            
            # If no WebFetchTool structure, clean simple response
            cleaned = response
            
            # Remove common technical artifacts
            artifacts_to_remove = [
                'Loaded cached credentials.',
                '"text": "',
                '"role": "model"',
                '"parts":'
            ]
            
            for artifact in artifacts_to_remove:
                cleaned = cleaned.replace(artifact, '')
            
            # Clean up excessive quotes and brackets if they're not part of content
            if cleaned.count('"') > 10:  # Likely technical artifacts
                cleaned = re.sub(r'"[^"]*":', '', cleaned)  # Remove JSON keys
                cleaned = re.sub(r'[{}\[\]]', '', cleaned)  # Remove brackets
            
            return cleaned.strip()
            
        except Exception as e:
            logger.warning(f"Error cleaning Gemini response: {e}")
            return response  # Return original if cleaning fails

    def analyze_with_openai(self, gemini_response: str, criterion: Dict[str, Any], 
                           scale_type: str) -> Dict[str, Any]:
        """Analyze Gemini response with OpenAI for structured output"""
        if not self.openai_client:
            return {"error": "OpenAI client not initialized"}

        # Define scale options (only binary and 3-point)
        scale_options = {
            "binary": ["nicht vorhanden", "vorhanden"],
            "3-point": ["nicht vorhanden", "teilweise vorhanden", "vollständig"]
        }
        
        scale_values = scale_options.get(scale_type, scale_options["binary"])
        
        # Create scale-specific instructions
        if scale_type == "binary":
            scale_instruction = """WICHTIG für binäre Bewertung (0/1 System):
- "nicht vorhanden" (0): KEIN Hinweis auf Erfüllung des Kriteriums gefunden
- "vorhanden" (1): JEDER Hinweis zur Erfüllung des Kriteriums, auch indirekte oder schwache Belege

Regeln: Kein Hinweis = 0, Hinweise zur Erfüllung = 1"""
        elif scale_type == "3-point":
            scale_instruction = """WICHTIG für 3-Punkt-Bewertung (0/1/2 System):
- "nicht vorhanden" (0): KEIN Hinweis auf Erfüllung des Kriteriums
- "teilweise vorhanden" (1): Indirekte oder schwache Hinweise zur Erfüllung
- "vollständig" (2): Deutliche Hinweise zur vollständigen Erfüllung

Regeln: Kein Hinweis (0), indirekte/schwache Hinweise (1), deutliche Hinweise (2)"""
        else:
            scale_instruction = f"""Bewertungsskala:
{chr(10).join([f'- "{val}": {self._get_scale_description(val)}' for val in scale_values])}"""
        
        prompt = f"""Analysiere die folgende Antwort zu dem Kriterium "{criterion['name']}" und bewerte es strukturiert:

Kriterium: {criterion['name']}
Beschreibung: {criterion['description']}

{scale_instruction}

Antwort zu analysieren:
{gemini_response}

Bewerte das Kriterium auf der Skala: {', '.join(scale_values)}

WICHTIG: Antworte NUR mit reinem JSON - KEINE Code-Fences (```), KEINE Markdown-Formatierung, KEINE zusätzlichen Erklärungen!

Erwartetes JSON-Format (Beispiel):
{{
    "bewertung": "{scale_values[1] if len(scale_values) > 1 else scale_values[0]}",
    "begruendung": "Die Organisation zeigt konkrete Maßnahmen wie XYZ, die das Kriterium erfüllen.",
    "quellen": ["https://example.org/transparency", "https://example.org/data"],
    "konfidenz": "hoch"
}}

Deine Antwort (nur JSON, keine anderen Zeichen):"""

        # Retry logic for robust JSON parsing
        max_retries = 3
        for attempt in range(max_retries):
            try:
                model = st.session_state.get('openai_model', 'gpt-4o-mini')
                max_tokens = st.session_state.get('max_tokens', 15000)
                
                # Enhanced system prompt based on scale type
                base_json_instruction = "Antworte AUSSCHLIESSLICH mit reinem JSON. NIEMALS Code-Fences (```), NIEMALS Markdown, NIEMALS zusätzliche Erklärungen. Nur valides JSON."
                
                if scale_type == "binary":
                    system_prompt = f"Du bist ein Experte für die Bewertung von Organisationen hinsichtlich Offenheitskriterien. {base_json_instruction} Bei binärer Bewertung: Jeder Hinweis auf Kriterium-Erfüllung = vorhanden, kein Hinweis = nicht vorhanden."
                elif scale_type == "3-point":
                    system_prompt = f"Du bist ein Experte für die Bewertung von Organisationen hinsichtlich Offenheitskriterien. {base_json_instruction} Bei 3-Punkt-Bewertung: Kein Hinweis (0), schwache Hinweise (1), deutliche Hinweise (2)."
                else:
                    system_prompt = f"Du bist ein Experte für die Bewertung von Organisationen hinsichtlich Offenheitskriterien. {base_json_instruction}"
                
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=0.1
                )
                
                result_text = response.choices[0].message.content.strip()
                logger.info(f"OpenAI response received for {criterion['name']} (attempt {attempt + 1})")
                
                # Clean potential code fences or markdown formatting
                cleaned_text = self._clean_json_response(result_text)
                
                # Try to parse JSON
                try:
                    result = json.loads(cleaned_text)
                    logger.info(f"JSON parsing successful for {criterion['name']} on attempt {attempt + 1}")
                    return result
                except json.JSONDecodeError as json_error:
                    logger.warning(f"JSON parsing failed for {criterion['name']} on attempt {attempt + 1}: {json_error}")
                    logger.debug(f"Raw response: {result_text[:500]}...")
                    
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying OpenAI request for {criterion['name']} (attempt {attempt + 2}/{max_retries})")
                        continue
                    else:
                        # Final attempt failed, extract information manually
                        logger.error(f"All {max_retries} JSON parsing attempts failed for {criterion['name']}, falling back to manual extraction")
                        return {
                            "bewertung": "nicht vorhanden",
                            "begruendung": result_text,
                            "quellen": [],
                            "konfidenz": "niedrig"
                        }
                
            except Exception as e:
                logger.error(f"OpenAI analysis failed: {e}")
                return {
                    "error": str(e),
                    "bewertung": "nicht vorhanden",
                    "begruendung": f"Fehler bei der Analyse: {e}",
                    "quellen": [],
                    "konfidenz": "niedrig"
                }
    
    def _clean_json_response(self, response_text: str) -> str:
        """Clean JSON response by removing code fences and markdown formatting"""
        cleaned = response_text.strip()
        
        # Remove code fences (```json, ```, etc.)
        if cleaned.startswith('```'):
            lines = cleaned.split('\n')
            # Remove first line if it's a code fence
            if lines[0].startswith('```'):
                lines = lines[1:]
            # Remove last line if it's a closing code fence
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            cleaned = '\n'.join(lines).strip()
        
        # Remove any remaining markdown formatting
        cleaned = cleaned.replace('```json', '').replace('```', '').strip()
        
        # Remove any leading/trailing text that's not JSON
        # Find the first { and last }
        start_idx = cleaned.find('{')
        end_idx = cleaned.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            cleaned = cleaned[start_idx:end_idx + 1]
        
        return cleaned
    
    def _get_scale_description(self, scale_value: str) -> str:
        """Get description for scale values (binary and 3-point only)"""
        descriptions = {
            "nicht vorhanden": "Kriterium ist nicht erfüllt - keine Belege gefunden (0)",
            "vorhanden": "Kriterium ist erfüllt - konkrete Belege vorhanden (1)",
            "teilweise vorhanden": "Kriterium ist teilweise erfüllt - schwache/indirekte Belege (1)",
            "vollständig": "Kriterium ist vollständig erfüllt - deutliche Belege (2)"
        }
        return descriptions.get(scale_value, "Bewertung basierend auf gefundenen Informationen")

    def process_organizations(self, organizations: pd.DataFrame, criteria_list: List[Dict], 
                            scale_type: str, selected_catalog: str, progress_callback=None, 
                            use_search_engine_tool: bool = False) -> tuple[pd.DataFrame, Dict[str, str]]:
        """Process all organizations and criteria"""
        results = []
        organization_reports = {}  # Store markdown reports per organization
        total_tasks = len(organizations) * len(criteria_list)
        current_task = 0
        
        for idx, org in organizations.iterrows():
            org_name = org['Name']
            org_url = org['URL']
            
            logger.info(f"Processing organization: {org_name}")
            
            # Initialize markdown report for this organization
            markdown_report = f"# Offenheitsbericht: {org_name}\n\n"
            markdown_report += f"**URL:** {org_url}\n\n"
            markdown_report += f"**Kriterienkatalog:** {selected_catalog}\n\n"
            markdown_report += f"**Analysedatum:** {time.strftime('%d.%m.%Y %H:%M:%S')}\n\n"
            markdown_report += "---\n\n"
            
            for criterion in criteria_list:
                current_task += 1
                if progress_callback:
                    progress_callback(current_task, total_tasks, f"Verarbeite {org_name} - {criterion['name']}")
                
                logger.info(f"Processing criterion {current_task}/{total_tasks}: {criterion['name']} for {org_name}")
                
                # Step 1: Call search method based on selection
                if use_search_engine_tool:
                    # Use DuckDuckGo + OpenAI alternative
                    search_response = self.call_search_engine_for_criterion(org_name, org_url, criterion)
                    # Rate limiting: 2 seconds between requests for DuckDuckGo
                    logger.info("Waiting 2 seconds for DuckDuckGo rate limiting...")
                    time.sleep(2)
                else:
                    # Use Gemini CLI (standard)
                    search_response = self.call_gemini_for_criterion(org_name, org_url, criterion)
                    # Rate limiting: 10 requests per minute = 6 seconds between requests
                    logger.info("Waiting 6 seconds for Gemini rate limiting (gemini-2.5-flash)...")
                    time.sleep(6)
                    
                # Step 2: Always analyze with OpenAI for structured evaluation on scale
                # Both Gemini CLI and DuckDuckGo responses need structured evaluation
                search_method = "DuckDuckGo + OpenAI" if use_search_engine_tool else "Gemini CLI"
                logger.info(f"Analyzing {search_method} response with OpenAI for structured evaluation: {criterion['name']}")
                analysis = self.analyze_with_openai(search_response, criterion, scale_type)
                    
                # Store result (ohne Gemini_Response für saubere CSV)
                result = {
                    'Organisation': org_name,
                    'URL': org_url,
                    'Kriterium': criterion['name'],
                    'Kriterium_Key': criterion['key'],
                    'Beschreibung': criterion['description'],
                    'Typ': criterion['type'],
                    'Bewertung': analysis.get('bewertung', 'nicht vorhanden'),
                    'Begründung': analysis.get('begruendung', ''),
                    'Quellen': '; '.join(analysis.get('quellen', [])),
                    'Konfidenz': analysis.get('konfidenz', 'niedrig'),
                    'Gewichtung': criterion['weight']
                }
                results.append(result)
                
                # Add to markdown report
                markdown_report += f"## {criterion['name']}\n\n"
                markdown_report += f"**Beschreibung:** {criterion['description']}\n\n"
                markdown_report += f"**Bewertung:** {analysis.get('bewertung', 'nicht vorhanden')}\n\n"
                markdown_report += f"**Begründung:** {analysis.get('begruendung', '')}\n\n"
                
                if analysis.get('quellen'):
                    markdown_report += f"**Quellen:**\n"
                    for quelle in analysis.get('quellen', []):
                        markdown_report += f"- {quelle}\n"
                    markdown_report += "\n"
                
                markdown_report += f"**Konfidenz:** {analysis.get('konfidenz', 'niedrig')}\n\n"
                # Add search response to markdown report
                search_method = "DuckDuckGo + OpenAI" if use_search_engine_tool else "Gemini CLI"
                markdown_report += f"**{search_method}-Antwort:**\n```\n{search_response}\n```\n\n"
                markdown_report += "---\n\n"
            
            # Store the complete report for this organization
            organization_reports[org_name] = markdown_report
            logger.info(f"Completed processing organization: {org_name}")
        
        return pd.DataFrame(results), organization_reports

    def create_criteria_analysis(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Create analysis table by criteria with frequency and ratings"""
        # Bewertungen zu numerischen Werten konvertieren
        bewertung_mapping = {
            'vorhanden': 1,
            'teilweise vorhanden': 0.5,
            'nicht vorhanden': 0
        }
        
        results_df['Bewertung_Numeric'] = results_df['Bewertung'].map(bewertung_mapping).fillna(0)
        
        criteria_analysis = results_df.groupby(['Kriterium', 'Typ']).agg({
            'Bewertung': ['count', lambda x: (x == 'vorhanden').sum(), 
                         lambda x: (x == 'teilweise vorhanden').sum(),
                         lambda x: (x == 'nicht vorhanden').sum()],
            'Bewertung_Numeric': ['mean', 'std'],
            'Gewichtung': 'first'
        }).round(3)
        
        # Spalten umbenennen
        criteria_analysis.columns = ['Anzahl_Gesamt', 'Anzahl_Vorhanden', 'Anzahl_Teilweise', 
                                   'Anzahl_Nicht_Vorhanden', 'Durchschnitt', 'Standardabweichung', 'Gewichtung']
        
        # Prozentuale Verteilung hinzufügen
        criteria_analysis['Prozent_Vorhanden'] = (criteria_analysis['Anzahl_Vorhanden'] / criteria_analysis['Anzahl_Gesamt'] * 100).round(1)
        criteria_analysis['Prozent_Teilweise'] = (criteria_analysis['Anzahl_Teilweise'] / criteria_analysis['Anzahl_Gesamt'] * 100).round(1)
        criteria_analysis['Prozent_Nicht_Vorhanden'] = (criteria_analysis['Anzahl_Nicht_Vorhanden'] / criteria_analysis['Anzahl_Gesamt'] * 100).round(1)
        
        return criteria_analysis.reset_index()
    
    def create_organization_summary(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Create summary table by organization with weighted scores"""
        # Bewertungen zu numerischen Werten konvertieren
        bewertung_mapping = {
            'vorhanden': 1,
            'teilweise vorhanden': 0.5,
            'nicht vorhanden': 0
        }
        
        results_df['Bewertung_Numeric'] = results_df['Bewertung'].map(bewertung_mapping).fillna(0)
        
        # Gewichtete Scores berechnen
        results_df['Gewichteter_Score'] = results_df['Bewertung_Numeric'] * results_df['Gewichtung']
        
        org_summary = results_df.groupby(['Organisation', 'URL']).agg({
            'Bewertung_Numeric': ['count', 'mean', 'std'],
            'Gewichteter_Score': 'sum',
            'Gewichtung': 'sum',
            'Bewertung': [lambda x: (x == 'vorhanden').sum(),
                         lambda x: (x == 'teilweise vorhanden').sum(), 
                         lambda x: (x == 'nicht vorhanden').sum()]
        }).round(3)
        
        # Spalten umbenennen
        org_summary.columns = ['Anzahl_Kriterien', 'Durchschnitt_Ungewichtet', 'Standardabweichung',
                              'Gewichteter_Gesamtscore', 'Summe_Gewichtungen',
                              'Anzahl_Vorhanden', 'Anzahl_Teilweise', 'Anzahl_Nicht_Vorhanden']
        
        # Gewichteten Durchschnitt berechnen
        org_summary['Durchschnitt_Gewichtet'] = (org_summary['Gewichteter_Gesamtscore'] / org_summary['Summe_Gewichtungen']).round(3)
        
        # Prozentuale Verteilung
        org_summary['Prozent_Vorhanden'] = (org_summary['Anzahl_Vorhanden'] / org_summary['Anzahl_Kriterien'] * 100).round(1)
        org_summary['Prozent_Teilweise'] = (org_summary['Anzahl_Teilweise'] / org_summary['Anzahl_Kriterien'] * 100).round(1)
        org_summary['Prozent_Nicht_Vorhanden'] = (org_summary['Anzahl_Nicht_Vorhanden'] / org_summary['Anzahl_Kriterien'] * 100).round(1)
        
        return org_summary.reset_index().sort_values('Durchschnitt_Gewichtet', ascending=False)

    def save_results_automatically(self, results_df: pd.DataFrame, organization_reports: Dict[str, str], 
                                  selected_catalog: str, selected_orgs: pd.DataFrame) -> Dict[str, str]:
        """Automatically save results with meaningful names and timestamps"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        saved_files = {}
        
        try:
            # Create results directory if it doesn't exist
            results_dir = Path('results')
            results_dir.mkdir(exist_ok=True)
            
            # Save main CSV results with UTF-8 BOM for Excel compatibility
            csv_filename = f"offenheitsanalyse_{selected_catalog}_{len(selected_orgs)}orgs_{timestamp}.csv"
            csv_path = results_dir / csv_filename
            results_df.to_csv(csv_path, sep=';', index=False, encoding='utf-8-sig')
            saved_files['csv'] = str(csv_path)
            logger.info(f"CSV results saved to: {csv_path} (UTF-8 with BOM)")
            
            # Create and save criteria analysis
            criteria_analysis = self.create_criteria_analysis(results_df.copy())
            criteria_csv_filename = f"kriterienanalyse_{selected_catalog}_{len(selected_orgs)}orgs_{timestamp}.csv"
            criteria_csv_path = results_dir / criteria_csv_filename
            criteria_analysis.to_csv(criteria_csv_path, sep=';', index=False, encoding='utf-8-sig')
            saved_files['criteria_csv'] = str(criteria_csv_path)
            logger.info(f"Criteria analysis saved to: {criteria_csv_path} (UTF-8 with BOM)")
            
            # Create and save organization summary
            org_summary = self.create_organization_summary(results_df.copy())
            org_csv_filename = f"organisationszusammenfassung_{selected_catalog}_{len(selected_orgs)}orgs_{timestamp}.csv"
            org_csv_path = results_dir / org_csv_filename
            org_summary.to_csv(org_csv_path, sep=';', index=False, encoding='utf-8-sig')
            saved_files['organization_csv'] = str(org_csv_path)
            logger.info(f"Organization summary saved to: {org_csv_path}")
            
            # Save markdown reports for each organization
            reports_dir = results_dir / f"berichte_{timestamp}"
            reports_dir.mkdir(exist_ok=True)
            
            for org_name, report_content in organization_reports.items():
                # Clean organization name for filename
                clean_org_name = re.sub(r'[<>:"/\\|?*]', '_', org_name)
                report_filename = f"bericht_{clean_org_name}_{timestamp}.md"
                report_path = reports_dir / report_filename
                
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                
                logger.info(f"Markdown report saved for {org_name}: {report_path}")
            
            saved_files['reports_dir'] = str(reports_dir)
            
            return saved_files
            
        except Exception as e:
            logger.error(f"Error saving results automatically: {e}")
            return {}
    
    def create_visualizations(self, results_df: pd.DataFrame, selected_catalog: str, timestamp: str) -> Dict[str, str]:
        """Create and save visualizations for the results"""
        saved_plots = {}
        
        if results_df.empty:
            return saved_plots
        
        # Create results directory if it doesn't exist
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        # Convert ratings to numeric for analysis
        rating_map = {
            'nicht vorhanden': 0,
            'teilweise vorhanden': 0.5,
            'vorhanden': 1
        }
        
        results_df['Bewertung_Numeric'] = results_df['Bewertung'].map(rating_map).fillna(0)
        
        try:
            # 1. Top Criteria Chart
            criteria_avg = results_df.groupby('Kriterium')['Bewertung_Numeric'].mean().sort_values(ascending=False)
            top_criteria = criteria_avg.head(10)
            
            fig_top = px.bar(
                x=top_criteria.values,
                y=top_criteria.index,
                orientation='h',
                title="Top 10 Kriterien (Durchschnittsbewertung)",
                color=top_criteria.values,
                color_continuous_scale="Greens",
                labels={'x': 'Durchschnittsbewertung', 'y': 'Kriterium'}
            )
            fig_top.update_layout(height=500, margin=dict(l=200))
            
            plot_path = results_dir / f"top_kriterien_{selected_catalog}_{timestamp}.png"
            fig_top.write_image(str(plot_path), width=1200, height=600)
            saved_plots['top_criteria'] = str(plot_path)
            
            # 2. Flop Criteria Chart
            flop_criteria = criteria_avg.tail(10)
            
            fig_flop = px.bar(
                x=flop_criteria.values,
                y=flop_criteria.index,
                orientation='h',
                title="Flop 10 Kriterien (Durchschnittsbewertung)",
                color=flop_criteria.values,
                color_continuous_scale="Reds",
                labels={'x': 'Durchschnittsbewertung', 'y': 'Kriterium'}
            )
            fig_flop.update_layout(height=500, margin=dict(l=200))
            
            plot_path = results_dir / f"flop_kriterien_{selected_catalog}_{timestamp}.png"
            fig_flop.write_image(str(plot_path), width=1200, height=600)
            saved_plots['flop_criteria'] = str(plot_path)
            
            # 3. Organizations Comparison
            org_avg = results_df.groupby('Organisation')['Bewertung_Numeric'].mean().sort_values(ascending=False)
            
            fig_org = px.bar(
                x=org_avg.index,
                y=org_avg.values,
                title="Organisationen im Vergleich (Durchschnittsbewertung)",
                color=org_avg.values,
                color_continuous_scale="Blues",
                labels={'x': 'Organisation', 'y': 'Durchschnittsbewertung'}
            )
            fig_org.update_xaxes(tickangle=45)
            fig_org.update_layout(height=600, margin=dict(b=150))
            
            plot_path = results_dir / f"organisationen_vergleich_{selected_catalog}_{timestamp}.png"
            fig_org.write_image(str(plot_path), width=1200, height=700)
            saved_plots['organizations'] = str(plot_path)
            
            # 4. Heatmap: Organizations vs Criteria
            pivot_df = results_df.pivot_table(
                index='Organisation',
                columns='Kriterium',
                values='Bewertung_Numeric',
                aggfunc='mean'
            )
            
            fig_heatmap = px.imshow(
                pivot_df.values,
                x=pivot_df.columns,
                y=pivot_df.index,
                title="Heatmap: Organisationen vs Kriterien",
                color_continuous_scale="RdYlGn",
                aspect="auto",
                labels={'x': 'Kriterium', 'y': 'Organisation', 'color': 'Bewertung'}
            )
            fig_heatmap.update_layout(
                height=max(600, len(pivot_df.index) * 30),
                margin=dict(l=200, b=200)
            )
            fig_heatmap.update_xaxes(tickangle=45)
            
            plot_path = results_dir / f"heatmap_{selected_catalog}_{timestamp}.png"
            fig_heatmap.write_image(str(plot_path), width=1400, height=max(800, len(pivot_df.index) * 40))
            saved_plots['heatmap'] = str(plot_path)
            
            # 5. Criteria Distribution Profile
            bewertung_counts = results_df['Bewertung'].value_counts()
            
            fig_dist = px.pie(
                values=bewertung_counts.values,
                names=bewertung_counts.index,
                title="Verteilung aller Kriterienbewertungen",
                color_discrete_map={
                    'vorhanden': '#2E8B57',
                    'teilweise vorhanden': '#FFD700', 
                    'nicht vorhanden': '#DC143C'
                }
            )
            fig_dist.update_layout(height=500)
            
            plot_path = results_dir / f"kriterienverteilung_{selected_catalog}_{timestamp}.png"
            fig_dist.write_image(str(plot_path), width=800, height=600)
            saved_plots['distribution'] = str(plot_path)
            
            logger.info(f"All visualizations saved successfully to {results_dir}")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
        
        return saved_plots

def main():
    st.set_page_config(
        page_title="Offenheitscrawler",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Offenheitscrawler")
    st.markdown("Tool zur Analyse von Organisationen hinsichtlich ihrer Offenheitskriterien")
    
    # Initialize crawler
    crawler = OffenheitsCrawler()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Konfiguration")
        
        # OpenAI Configuration
        st.subheader("OpenAI Einstellungen")
        
        openai_api_key = st.text_input(
            "API Key",
            value=os.getenv('OPENAI_API_KEY', ''),
            type="password",
            help="OpenAI API Key (Standard aus Umgebungsvariable)"
        )
        st.session_state['openai_api_key'] = openai_api_key
        
        openai_base_url = st.text_input(
            "Base URL",
            value="https://api.openai.com/v1",
            help="OpenAI API Base URL"
        )
        st.session_state['openai_base_url'] = openai_base_url
        
        openai_model = st.selectbox(
            "Modell",
            options=["gpt-4.1-mini", "gpt-4.1", "gpt-4o", "gpt-4o-mini"],
            index=0,
            help="OpenAI Modell für die Analyse"
        )
        st.session_state['openai_model'] = openai_model
        
        max_tokens = st.number_input(
            "Max Tokens",
            min_value=1000,
            max_value=50000,
            value=15000,
            help="Maximale Anzahl Tokens pro Anfrage"
        )
        st.session_state['max_tokens'] = max_tokens
        
        # Scale Configuration
        st.subheader("Bewertungsskala")
        scale_type = st.selectbox(
            "Skala wählen",
            options=["binary", "3-point"],
            format_func=lambda x: {
                "binary": "Binär (0/1): nicht vorhanden/vorhanden",
                "3-point": "3-Punkt (0/1/2): nicht/teilweise/vollständig"
            }[x]
        )
        
        # Criteria Catalog Selection
        st.subheader("📋 Kriterienkatalog")
        catalogs = crawler.load_criteria_catalogs()
        
        if not catalogs:
            st.error("Keine Kriterienkataloge gefunden!")
            selected_catalog = None
            criteria_list = []
        else:
            selected_catalog = st.selectbox(
                "Katalog wählen",
                options=list(catalogs.keys()),
                format_func=lambda x: x.replace('_', ' ').title()
            )
            
            if selected_catalog:
                criteria_data = crawler.load_criteria_from_yaml(catalogs[selected_catalog])
                criteria_list = crawler.extract_criteria_list(criteria_data)
                st.success(f"✅ {len(criteria_list)} Kriterien geladen")
            else:
                criteria_list = []
        
        # Search Engine Selection
        st.subheader("🔍 Suchmaschine")
        search_engine_option = st.radio(
            "Suchmaschine wählen",
            options=["Gemini CLI (Standard)", "DuckDuckGo + OpenAI (Alternative)"],
            help="Gemini CLI: Verwendet Google's Gemini über Kommandozeile\n\nDuckDuckGo + OpenAI: Verwendet DuckDuckGo-Suche mit Proxy-Rotation und OpenAI GPT-4.1-mini für die Analyse"
        )
        
        use_search_engine_tool = search_engine_option == "DuckDuckGo + OpenAI (Alternative)"
        
        if use_search_engine_tool:
            # Check if OpenAI API key is available for SearchEngineTool
            api_key = st.session_state.get('openai_api_key', '')
            if not api_key or len(api_key.strip()) == 0:
                st.warning("⚠️ OpenAI API Key erforderlich")
                use_search_engine_tool = False
            else:
                st.info("ℹ️ DuckDuckGo + OpenAI wird verwendet")
        
        if not use_search_engine_tool:
            st.info("ℹ️ Gemini CLI wird verwendet")
    
    # Main content
    st.header("📊 Analyse-Übersicht")
    if selected_catalog and criteria_list:
        st.info(f"📋 **Gewählter Katalog:** {selected_catalog.replace('_', ' ').title()} ({len(criteria_list)} Kriterien)")
    else:
        st.warning("⚠️ Bitte wählen Sie zuerst einen Kriterienkatalog in der Seitenleiste aus.")
    
    # Load organizations
    st.header("🏢 Organisationen laden")
    csv_files = crawler.get_csv_files()
    
    if not csv_files:
        st.error("Keine CSV-Dateien gefunden!")
        return
    
    selected_csv = st.selectbox(
        "CSV-Datei wählen",
        options=csv_files,
        format_func=lambda x: Path(x).name
    )
    
    if selected_csv:
        organizations_df = crawler.load_organizations_csv(selected_csv)
        
        if not organizations_df.empty:
            st.success(f"✅ {len(organizations_df)} Organisationen geladen")
            
            # Organization selection
            st.subheader("Organisationen auswählen")
            selection_type = st.radio(
                "Auswahl",
                options=["Alle", "Bereich", "Einzelne"]
            )
            
            if selection_type == "Alle":
                selected_orgs = organizations_df
            elif selection_type == "Bereich":
                col_start, col_end = st.columns(2)
                with col_start:
                    start_idx = st.number_input("Von Zeile", min_value=1, max_value=len(organizations_df), value=1) - 1
                with col_end:
                    end_idx = st.number_input("Bis Zeile", min_value=1, max_value=len(organizations_df), value=min(10, len(organizations_df))) - 1
                selected_orgs = organizations_df.iloc[start_idx:end_idx+1]
            else:  # Einzelne
                selected_indices = st.multiselect(
                    "Organisationen wählen",
                    options=range(len(organizations_df)),
                    format_func=lambda x: organizations_df.iloc[x]['Name'],
                    default=[0] if len(organizations_df) > 0 else []
                )
                selected_orgs = organizations_df.iloc[selected_indices] if selected_indices else pd.DataFrame()
            
            # Preview selected organizations
            if not selected_orgs.empty:
                st.subheader(f"📋 Vorschau der geladenen Organisationen ({len(selected_orgs)})")
                st.dataframe(selected_orgs.head(10), use_container_width=True)
    

    
    # Analysis section
    if 'selected_orgs' in locals() and not selected_orgs.empty and criteria_list:
        st.header("🚀 Analyse starten")
        
        if st.button("Analyse durchführen", type="primary"):
            if not openai_api_key:
                st.error("Bitte OpenAI API Key eingeben!")
                return
            
            # Reinitialize OpenAI client with current settings
            crawler.setup_openai()
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(current, total, message):
                progress = current / total
                progress_bar.progress(progress)
                status_text.text(f"{message} ({current}/{total})")
            
            # Process organizations
            with st.spinner("Verarbeitung läuft..."):
                results_df, organization_reports = crawler.process_organizations(
                    selected_orgs, 
                    criteria_list, 
                    scale_type,
                    selected_catalog,
                    update_progress,
                    use_search_engine_tool
                )
            
            if not results_df.empty:
                st.success("✅ Analyse abgeschlossen!")
                
                # Automatically save results and reports
                with st.spinner("Speichere Ergebnisse..."):
                    saved_files = crawler.save_results_automatically(
                        results_df, organization_reports, selected_catalog, selected_orgs
                    )
                    
                    # Generate timestamp for visualizations
                    timestamp = time.strftime('%Y%m%d_%H%M%S')
                    
                    # Create and save visualizations
                    saved_plots = crawler.create_visualizations(results_df, selected_catalog, timestamp)
                
                # Show info about saved files
                if saved_files:
                    st.info(f"📁 Ergebnisse automatisch gespeichert:")
                    if 'csv' in saved_files:
                        st.write(f"• Hauptergebnisse CSV: `{saved_files['csv']}`")
                    if 'criteria_csv' in saved_files:
                        st.write(f"• Kriterienanalyse CSV: `{saved_files['criteria_csv']}`")
                    if 'organization_csv' in saved_files:
                        st.write(f"• Organisationszusammenfassung CSV: `{saved_files['organization_csv']}`")
                    if 'reports_dir' in saved_files:
                        st.write(f"• Markdown-Berichte: `{saved_files['reports_dir']}`")
                    if saved_plots:
                        st.write(f"• Visualisierungen: {len(saved_plots)} Diagramme gespeichert")
                        for plot_name, plot_path in saved_plots.items():
                            st.write(f"  - {plot_name}: `{plot_path}`")
                
                # Store results in session state
                st.session_state['results_df'] = results_df
                st.session_state['organization_reports'] = organization_reports
                
                # Show results
                st.header("📊 Ergebnisse")
                
                # Create tabs for different views
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["Übersicht", "Detailergebnisse", "Analysetabellen", "Visualisierungen", "Export"])
                
                with tab1:
                    st.subheader("Zusammenfassung")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Organisationen", len(selected_orgs))
                    with col2:
                        st.metric("Kriterien", len(criteria_list))
                    with col3:
                        st.metric("Bewertungen", len(results_df))
                    with col4:
                        # Use correct rating mapping for average calculation
                        rating_map = {'nicht vorhanden': 0, 'teilweise vorhanden': 0.5, 'vorhanden': 1}
                        results_df_temp = results_df.copy()
                        results_df_temp['Bewertung_Numeric'] = results_df_temp['Bewertung'].map(rating_map).fillna(0)
                        avg_score = results_df_temp['Bewertung_Numeric'].mean()
                        st.metric("Ø Bewertung", f"{avg_score:.2f}")
                
                with tab2:
                    st.subheader("Detaillierte Ergebnisse")
                    st.dataframe(results_df, use_container_width=True)
                
                with tab3:
                    st.subheader("📊 Analysetabellen")
                    
                    # Criteria Analysis
                    st.write("**Analyse nach Kriterien**")
                    criteria_analysis = crawler.create_criteria_analysis(results_df.copy())
                    st.dataframe(criteria_analysis, use_container_width=True)
                    
                    st.write("---")
                    
                    # Organization Summary
                    st.write("**Zusammenfassung nach Organisationen**")
                    org_summary = crawler.create_organization_summary(results_df.copy())
                    st.dataframe(org_summary, use_container_width=True)
                
                with tab4:
                    st.subheader("📊 Visualisierungen")
                    st.info("Visualisierungen wurden automatisch als PNG-Dateien gespeichert.")
                    
                    if 'saved_plots' in locals() and saved_plots:
                        st.write("**Gespeicherte Diagramme:**")
                        for plot_name, plot_path in saved_plots.items():
                            st.write(f"• {plot_name}: `{plot_path}`")
                    
                    # Show interactive visualizations in the app
                    if not results_df.empty:
                        # Convert ratings to numeric for analysis
                        rating_map = {'nicht vorhanden': 0, 'teilweise vorhanden': 0.5, 'vorhanden': 1}
                        results_df_viz = results_df.copy()
                        results_df_viz['Bewertung_Numeric'] = results_df_viz['Bewertung'].map(rating_map).fillna(0)
                        
                        # 1. Top and Flop Criteria
                        st.subheader("🏆 Top & Flop Kriterien")
                        criteria_avg = results_df_viz.groupby('Kriterium')['Bewertung_Numeric'].mean().sort_values(ascending=False)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Top 10 Kriterien**")
                            top_criteria = criteria_avg.head(10)
                            fig_top = px.bar(
                                x=top_criteria.values,
                                y=top_criteria.index,
                                orientation='h',
                                title="Beste Kriterien (Durchschnittsbewertung)",
                                color=top_criteria.values,
                                color_continuous_scale="Greens"
                            )
                            fig_top.update_layout(height=400)
                            st.plotly_chart(fig_top, use_container_width=True, key="top_criteria_chart")
                        
                        with col2:
                            st.write("**Flop 10 Kriterien**")
                            flop_criteria = criteria_avg.tail(10)
                            fig_flop = px.bar(
                                x=flop_criteria.values,
                                y=flop_criteria.index,
                                orientation='h',
                                title="Schlechteste Kriterien (Durchschnittsbewertung)",
                                color=flop_criteria.values,
                                color_continuous_scale="Reds"
                            )
                            fig_flop.update_layout(height=400)
                            st.plotly_chart(fig_flop, use_container_width=True, key="flop_criteria_chart")
                        
                        # 2. Analysis by Organization
                        st.subheader("🏢 Auswertung je Organisation")
                        org_avg = results_df_viz.groupby('Organisation')['Bewertung_Numeric'].mean().sort_values(ascending=False)
                        
                        fig_org = px.bar(
                            x=org_avg.index,
                            y=org_avg.values,
                            title="Durchschnittsbewertung je Organisation",
                            color=org_avg.values,
                            color_continuous_scale="Blues"
                        )
                        fig_org.update_xaxes(tickangle=45)
                        fig_org.update_layout(height=500)
                        st.plotly_chart(fig_org, use_container_width=True, key="org_comparison_chart")
                        
                        # 3. Criteria Distribution
                        st.subheader("📊 Verteilung der Bewertungen")
                        bewertung_counts = results_df['Bewertung'].value_counts()
                        
                        fig_dist = px.pie(
                            values=bewertung_counts.values,
                            names=bewertung_counts.index,
                            title="Verteilung aller Kriterienbewertungen",
                            color_discrete_map={
                                'vorhanden': '#2E8B57',
                                'teilweise vorhanden': '#FFD700', 
                                'nicht vorhanden': '#DC143C'
                            }
                        )
                        st.plotly_chart(fig_dist, use_container_width=True, key="distribution_chart")
                
                with tab5:
                    st.subheader("📥 Export & Downloads")
                    
                    st.write("**Automatisch gespeicherte Dateien:**")
                    if 'saved_files' in locals() and saved_files:
                        for file_type, file_path in saved_files.items():
                            if file_type == 'csv':
                                st.write(f"• Hauptergebnisse: `{file_path}`")
                            elif file_type == 'criteria_csv':
                                st.write(f"• Kriterienanalyse: `{file_path}`")
                            elif file_type == 'organization_csv':
                                st.write(f"• Organisationszusammenfassung: `{file_path}`")
                            elif file_type == 'reports_dir':
                                st.write(f"• Markdown-Berichte: `{file_path}`")
                    
                    if 'saved_plots' in locals() and saved_plots:
                        st.write("**Gespeicherte Visualisierungen:**")
                        for plot_name, plot_path in saved_plots.items():
                            st.write(f"• {plot_name}: `{plot_path}`")
                    
                    st.write("---")
                    
                    # Manual CSV Export
                    st.write("**Manuelle Downloads:**")
                    csv_data = results_df.to_csv(sep=';', index=False, encoding='utf-8')
                    st.download_button(
                        label="📄 Hauptergebnisse CSV herunterladen",
                        data=csv_data,
                        file_name=f"offenheitsanalyse_{selected_catalog}_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    # Additional CSV downloads
                    criteria_analysis = crawler.create_criteria_analysis(results_df.copy())
                    criteria_csv = criteria_analysis.to_csv(sep=';', index=False, encoding='utf-8')
                    st.download_button(
                        label="📈 Kriterienanalyse CSV herunterladen",
                        data=criteria_csv,
                        file_name=f"kriterienanalyse_{selected_catalog}_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    org_summary = crawler.create_organization_summary(results_df.copy())
                    org_csv = org_summary.to_csv(sep=';', index=False, encoding='utf-8')
                    st.download_button(
                        label="🏢 Organisationszusammenfassung CSV herunterladen",
                        data=org_csv,
                        file_name=f"organisationszusammenfassung_{selected_catalog}_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

if __name__ == "__main__":
    main()
