import os
import random
import re
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain 
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
import json
import datetime
import asyncio
from typing import List, Dict, Any, Optional, Union

# Rich for beautiful terminal output
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.syntax import Syntax
from rich.rule import Rule

# --- CONFIGURATION ---
SELECTED_MODEL_NAME = "qwen2:7b" # Default, user can override
OLLAMA_BASE_URL = "http://localhost:11434"
MAX_EXTRACTION_ATTEMPTS = 2 
MAX_PROSE_REVISION_ATTEMPTS = 1 # Max times to run editor/reviser loop per chapter

# --- RICH CONSOLE ---
console = Console(width=120)

# --- NOVEL WRITING PRINCIPLES (From User) ---
NOVEL_WRITING_PRINCIPLES = """
A novel, at its heart, is a journey—for both the writer and the reader. To make that journey compelling and complete, several key ingredients are necessary. These elements work together to create a rich and immersive experience, transforming a simple story into a world that readers can lose themselves in.
At the forefront are characters. A novel needs relatable or at least understandable individuals who drive the story forward. This includes a protagonist, the central figure whose journey the reader follows, and often an antagonist, who creates opposition. Beyond these, supporting characters add depth and dimension to the fictional world. Crucially, characters should ideally undergo some form of development or change as the story progresses, reacting to the events around them in believable ways.
The plot forms the backbone of the novel – it's the sequence of events that make up the story. A well-structured plot typically involves an inciting incident that sets the story in motion, rising action where tension and conflict build, a climax which is the turning point or peak of the conflict, falling action where the consequences of the climax unfold, and a resolution that brings the story to a satisfying close.
Integral to the plot is conflict. This is the struggle that the protagonist faces. Conflict can be external (against another character, nature, or society) or internal (a struggle within the character's own mind). It's the conflict that creates tension, drives the story forward, and keeps readers engaged.
The setting—the time and place where the story unfolds—is more than just a backdrop. A well-developed setting can influence the mood, shape the characters' actions and motivations, and even become a character in itself. It helps to ground the story in a tangible reality, whether that reality is a familiar cityscape or an imagined fantasy world.
Every novel explores a theme, which is the central idea, message, or insight about life or human nature that the author conveys. The theme often emerges subtly through the plot, characters, and setting, rather than being stated directly. It’s what gives the story a deeper meaning and resonates with readers long after they’ve finished the book.
The point of view (POV) determines through whose eyes the story is told. Common POVs include first person (narrated by a character using "I"), third-person limited (following one character's perspective), and third-person omniscient (an all-knowing narrator). The choice of POV significantly impacts how the reader experiences the story and understands the characters and events.
Finally, style and voice refer to the author's unique way of using language. This includes word choice, sentence structure, tone, and imagery. A distinctive voice can make a novel memorable and engaging, drawing the reader into the story's world and enhancing the overall reading experience. Effective dialogue is also a crucial part of style, revealing character, advancing the plot, and making the interactions between characters feel real.
Beyond these core components, elements like pacing (the speed at which the story unfolds), a strong opening that hooks the reader, and a satisfying ending that provides a sense of closure contribute significantly to a novel's success. Ultimately, what is necessary in a novel are the elements that work in harmony to create a cohesive, engaging, and meaningful story that resonates with its readers.
"""

# --- NOVEL GENERATOR CLASS ---
class NovelGenerator:
    def __init__(self, subject: str, author_style: str, genre: str,
                 ollama_model_name: str = "llama3",
                 ollama_base_url: str = "http://localhost:11434",
                 resume_text: Optional[str] = None):
        self.subject = subject
        self.author_style = author_style
        self.genre = genre
        self.ollama_model_name = ollama_model_name
        self.ollama_base_url = ollama_base_url
        self.resume_text = resume_text
        self.console = console
        
        safe_subject_for_dir = self._sanitize_filename(subject[:50]) if subject else "novel_project"
        if not safe_subject_for_dir:
            safe_subject_for_dir = "novel_project"

        self.output_dir = os.path.join("generated_novels_ollama_v7_refined", safe_subject_for_dir) 
        self.log_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.foundational_elements: Optional[Dict[str, Any]] = None
        self.detailed_chapter_plans: List[Dict[str, Any]] = []
        self.generated_chapters_prose: Dict[int, str] = {} # Stores FINALIZED prose for each chapter
        self.character_states_after_chapter: Dict[int, Dict[str, Any]] = {}

        self.console.print(f"NovelGenerator initialized for Ollama (Model: [bold cyan]{self.ollama_model_name}[/bold cyan]). Output: [green]{self.output_dir}[/green]")
        self._log_to_file("initialization.log", f"Subject: {self.subject}\nAuthor Style: {self.author_style}\nGenre: {self.genre}\nModel: {self.ollama_model_name}")

    def _sanitize_filename(self, name: str) -> str:
        name = re.sub(r'[^\w\s-]', '', name).strip()
        name = re.sub(r'[-\s]+', '_', name)
        name = name.replace('/', '_')
        return name

    def _log_to_file(self, filename: str, content: str, prefix: Optional[str] = None):
        log_entry = f"--- {datetime.datetime.now()} ---\n"
        if prefix:
            log_entry += f"[{prefix.upper()}]\n"
        log_entry += f"{content}\n\n"
        try:
            with open(os.path.join(self.log_dir, filename), "a", encoding="utf-8") as f:
                f.write(log_entry)
        except Exception as e:
            self.console.print(f"[bold red]Error writing to log file {filename}: {e}[/bold red]")

    def _get_ollama_llm(self, temperature=0.7, num_predict=-1, timeout=300):
        return ChatOllama(
            model=self.ollama_model_name,
            base_url=self.ollama_base_url,
            temperature=temperature,
            num_predict=num_predict, 
            timeout=timeout,
        )

    async def _ollama_generate_text(self, prompt: str, system_message: Optional[str] = None, temperature: Optional[float]=None, json_mode: bool = False) -> str:
        messages = []
        if system_message:
            messages.append(SystemMessagePromptTemplate.from_template(system_message).format())
        messages.append(HumanMessagePromptTemplate.from_template(prompt).format())
        
        # Use provided temperature or default to 0.7
        current_temp = temperature if temperature is not None else 0.7
        llm_instance = self._get_ollama_llm(temperature=current_temp)
        
        llm_call_kwargs = {}
        if json_mode:
            llm_call_kwargs['format'] = "json"

        self._log_to_file("ollama_requests.log", f"PROMPT:\nSystem: {system_message}\nUser: {prompt}\nJSON Mode: {json_mode}, Temp: {current_temp}", prefix="OLLAMA_GEN_TEXT")
        try:
            response = await llm_instance.ainvoke(messages, **llm_call_kwargs)
            content = response.content
            self._log_to_file("ollama_requests.log", f"RESPONSE:\n{content}", prefix="OLLAMA_GEN_TEXT")
            return content
        except Exception as e:
            self.console.print(f"[bold red]Ollama text generation error: {e}[/bold red]")
            self._log_to_file("ollama_errors.log", f"TEXT GEN ERROR: {e}\nPROMPT:\nSystem: {system_message}\nUser: {prompt}\nJSON Mode: {json_mode}")
            return f"Error: Ollama text generation failed: {str(e)}"


    @retry(stop=stop_after_attempt(MAX_EXTRACTION_ATTEMPTS + 1), wait=wait_exponential(multiplier=1, min=2, max=30))
    async def _extract_data_with_llm(self, raw_text_from_llm: str, extraction_target_description: str, 
                                     desired_format_description: str, example_json_output: str, attempt: int = 1) -> Union[Dict, List, str]:
        self.console.print(f"[dim]Attempting LLM-based extraction for '{extraction_target_description}' (Attempt {attempt})...[/dim]")
        
        extraction_prompt = f"""
You are a data extraction expert. Your task is to parse the following raw text and extract specific information, formatting it as a valid JSON object or array.

**Extraction Target:** {extraction_target_description}
**Desired JSON Output Format:** {desired_format_description}
**Example of desired JSON output structure:**
```json
{example_json_output}
```

**Raw Text to Parse:**
```text
{raw_text_from_llm}
```

Please extract the information and provide ONLY the valid JSON object/array as your response. Do not include any explanations, apologies, or surrounding text.
Begin JSON output now:
"""
        system_extraction_prompt = "You are an AI assistant that converts unstructured or semi-structured text into perfectly valid JSON according to provided instructions."
        
        # Use low temp for extraction
        extracted_json_str = await self._ollama_generate_text(
            extraction_prompt, 
            system_message=system_extraction_prompt, 
            temperature=0.1, 
            json_mode=True
        )
        self._log_to_file("llm_extraction_requests.log", f"TARGET: {extraction_target_description}\nRAW_TEXT_INPUT_SNIPPET:\n{raw_text_from_llm[:500]}...\nEXTRACTED_JSON_STRING:\n{extracted_json_str}", prefix="EXTRACTION_ATTEMPT")
            
        if extracted_json_str.startswith("Error:"):
             raise RuntimeError(f"LLM call for extraction failed: {extracted_json_str}")

        try:
            parsed_data = json.loads(extracted_json_str)
            self.console.print(f"[green]Successfully extracted and parsed data for '{extraction_target_description}'.[/green]")
            return parsed_data
        except json.JSONDecodeError as e:
            self.console.print(f"[bold orange_red1]JSON Decode Error after LLM extraction for '{extraction_target_description}' (Attempt {attempt}): {e}[/bold orange_red1]")
            self.console.print(f"Problematic JSON string: {extracted_json_str[:500]}...")
            self._log_to_file("llm_extraction_errors.log", f"TARGET: {extraction_target_description}\nJSONDecodeError: {e}\nData:\n{extracted_json_str}")
            if attempt < MAX_EXTRACTION_ATTEMPTS:
                raise 
            return {"error": f"Failed to parse JSON from LLM extraction after {attempt} attempts. Last error: {e}", "raw_output": extracted_json_str}


    async def generate_foundational_elements(self):
        self.console.print(Rule("[bold_blue]Generating Foundational Elements (LLM Text Extraction)[/bold_blue]", style="blue"))

        # --- 1. Character Profiles ---
        self.console.print("[cyan]Step 1.1: Generating Character Profiles...[/cyan]")
        num_chars_to_generate = Prompt.ask(
            Text("How many primary characters to generate (e.g., 3-5)?", style="yellow"),
            default="3", 
            choices=["2", "3", "4", "5"]
        )
        resume_text_for_prompt = f"\n- Resume Snippet (for inspiration, if relevant):\n```\n{self.resume_text}\n```" if self.resume_text else ""

        char_gen_prompt = f"""
### Input Novel Details:
- Subject: "{self.subject}"
- Genre: "{self.genre}"
- Author Style Influence: "{self.author_style}"{resume_text_for_prompt}

### Character Generation Task:
Generate detailed profiles for {num_chars_to_generate} distinct characters for the novel described above.
Include a mix of protagonist(s), antagonist(s), and key supporting characters.
For each character, provide the following information clearly labeled:
- Name: [Character's Name]
- Role: [e.g., Protagonist, Antagonist, Mentor]
- Age: [e.g., 25, 'late 30s', 'ageless']
- Appearance: [Brief physical description]
- Personality Traits: [Comma-separated list, e.g., brave, cynical, witty]
- Backstory Summary: [Concise summary of their history relevant to the plot]
- Core Motivation: [What fundamentally drives this character?]
- Primary Goal: [Their main objective within the story's timeframe]
- Internal Conflict: [Optional: Their inner struggles or moral dilemmas]
- External Conflict Source: [Optional: Main source of external conflict they face or cause]
- Initial Relationship to Protagonist: [Optional: How they initially relate to the protagonist]
- Potential Arc Keywords: [Comma-separated list, e.g., redemption, fall_from_grace, self_discovery]
- Speech Mannerisms/Voice: [Optional: How they speak]

Use '---CHARACTER BREAK---' as a separator between each character profile.
Example for one character:
Name: Elara Vance
Role: Protagonist
Age: 23
Appearance: Lithe, with cybernetic eye implant that glows faintly blue, short-cropped raven hair.
Personality Traits: Curious, resourceful, initially hesitant, fiercely loyal.
Backstory Summary: Grew up in the lower sectors, orphaned early, self-taught tech skills.
Core Motivation: To uncover the truth about her parents' disappearance.
Primary Goal: To find the hidden data archive rumored to hold corporate secrets.
Internal Conflict: Trusting others vs. her ingrained self-reliance.
Potential Arc Keywords: Empowerment, uncovering conspiracy, finding belonging.
---CHARACTER BREAK---
Name: Marcus Thorne
Role: Antagonist
... (and so on for other characters)
"""
        system_char_gen_prompt = "You are an expert character creator. Generate character profiles as clearly formatted text, using the specified labels and separator."
        
        raw_character_text = await self._ollama_generate_text(char_gen_prompt, system_message=system_char_gen_prompt, temperature=0.7)
        if raw_character_text.startswith("Error:"):
            raise RuntimeError(f"Failed to generate raw character text: {raw_character_text}")
        self._log_to_file("foundational_raw_character_text.txt", raw_character_text)

        character_profiles_list = await self._extract_data_with_llm(
            raw_text_from_llm=raw_character_text,
            extraction_target_description=f"{num_chars_to_generate} character profiles",
            desired_format_description="A Python list of dictionaries. Each dictionary represents a character and should have keys: 'name', 'role', 'age', 'appearance', 'personality_traits' (as a list of strings), 'backstory_summary', 'core_motivation', 'primary_goal', and optionally 'internal_conflict', 'external_conflict_source', 'initial_relationship_to_protagonist', 'potential_arc_keywords' (as a list of strings), 'speech_mannerisms_or_voice'. Ensure 'personality_traits' and 'potential_arc_keywords' are lists of strings.",
            example_json_output='[{"name": "Elara", "role": "Protagonist", "age": "20s", "appearance": "...", "personality_traits": ["curious", "resourceful"], "backstory_summary": "...", "core_motivation": "...", "primary_goal": "..."}, {"name": "Thorne", ...}]'
        )

        if isinstance(character_profiles_list, dict) and 'error' in character_profiles_list:
            raise RuntimeError(f"Failed to extract character profiles: {character_profiles_list.get('error')}")
        if not isinstance(character_profiles_list, list) or not all(isinstance(item, dict) for item in character_profiles_list):
            if isinstance(character_profiles_list, dict) and 'name' in character_profiles_list: 
                self.console.print("[yellow]Warning: Expected list of characters, got single character dict. Wrapping in list.[/yellow]")
                character_profiles_list = [character_profiles_list]
            else:
                raise RuntimeError(f"Character profile extraction did not yield a list of dictionaries. Got: {type(character_profiles_list)}\nContent: {str(character_profiles_list)[:500]}")
        
        for profile in character_profiles_list:
            for key in ['personality_traits', 'potential_arc_keywords']:
                if key in profile and isinstance(profile[key], str):
                    profile[key] = [s.strip() for s in profile[key].split(',') if s.strip()]

        self.console.print(f"[green]Extracted {len(character_profiles_list)} character profiles.[/green]")
        for profile_dict in character_profiles_list:
            self.console.print(Panel(json.dumps(profile_dict, indent=2), title=f"Character: {profile_dict.get('name', 'Unknown')}", border_style="magenta", expand=False))

        # --- 2. World Details ---
        self.console.print("\n[cyan]Step 1.2: Generating World Details...[/cyan]")
        world_gen_prompt = f"""
Based on the novel's subject ("{self.subject}"), genre ("{self.genre}"), and author style ("{self.author_style}"), generate detailed world-building elements.
{resume_text_for_prompt}
Provide the following information, clearly labeled:
- World Name: [Name of the primary world or setting, if applicable]
- Setting Description: [Overall description of the time period, environment, and key locations]
- Key Technologies or Magic Systems: [List and brief explanation of important technologies or magic systems. Use bullet points for multiple items.]
- Cultural Norms: [Key societal customs, beliefs, or rules relevant to the story. Use bullet points.]
- Political Landscape: [Brief overview of the political situation, factions, or governance]
"""
        system_world_gen_prompt = "You are an expert world-builder. Generate world details as clearly formatted text, using the specified labels."
        raw_world_text = await self._ollama_generate_text(world_gen_prompt, system_message=system_world_gen_prompt, temperature=0.6)
        if raw_world_text.startswith("Error:"):
            raise RuntimeError(f"Failed to generate raw world details text: {raw_world_text}")
        self._log_to_file("foundational_raw_world_text.txt", raw_world_text)

        world_details_dict = await self._extract_data_with_llm(
            raw_text_from_llm=raw_world_text,
            extraction_target_description="world details",
            desired_format_description="A Python dictionary with keys: 'world_name' (optional string), 'setting_description' (string), 'key_technologies_or_magic_systems' (list of strings), 'cultural_norms' (list of strings), 'political_landscape' (optional string). Ensure list fields are actual lists of strings.",
            example_json_output='{"world_name": "Aethelgard", "setting_description": "A sprawling medieval kingdom...", "key_technologies_or_magic_systems": ["Rune Magic: based on ancient symbols", "Steam-powered automatons: recent controversial invention"], "cultural_norms": ["Annual harvest festival", "Strict hierarchical society"], "political_landscape": "Ruled by a council of elders."}'
        )
        if isinstance(world_details_dict, dict) and 'error' in world_details_dict:
             raise RuntimeError(f"Failed to extract world details: {world_details_dict.get('error')}")
        if not isinstance(world_details_dict, dict):
            raise RuntimeError(f"World details extraction did not yield a dictionary. Got: {type(world_details_dict)}")
        
        for key in ['key_technologies_or_magic_systems', 'cultural_norms']:
            if key in world_details_dict and isinstance(world_details_dict[key], str):
                world_details_dict[key] = [s.strip().lstrip('- ') for s in world_details_dict[key].split('\n') if s.strip()]

        self.console.print(Panel(json.dumps(world_details_dict, indent=2), title="World Details", border_style="yellow", expand=False))

        # --- 3. Themes and Motifs ---
        self.console.print("\n[cyan]Step 1.3: Generating Themes and Motifs...[/cyan]")
        themes_gen_prompt = f"""
Based on the novel's subject ("{self.subject}"), genre ("{self.genre}"), the generated character profiles, and world details, identify and articulate the core themes and potential recurring motifs.
Character Names & Roles: {[(p.get('name'), p.get('role')) for p in character_profiles_list]}
World Setting Summary: {world_details_dict.get('setting_description', 'N/A')}
{resume_text_for_prompt}

Provide the following, clearly labeled:
- Primary Theme: [The central, overarching theme]
- Secondary Themes: [List of other significant themes. Use bullet points.]
- Recurring Motifs or Symbols: [List of motifs or symbols. Use bullet points.]
"""
        system_themes_gen_prompt = "You are a literary analyst. Generate themes and motifs as clearly formatted text."
        raw_themes_text = await self._ollama_generate_text(themes_gen_prompt, system_message=system_themes_gen_prompt, temperature=0.5)
        if raw_themes_text.startswith("Error:"):
            raise RuntimeError(f"Failed to generate raw themes text: {raw_themes_text}")
        self._log_to_file("foundational_raw_themes_text.txt", raw_themes_text)

        themes_motifs_dict = await self._extract_data_with_llm(
            raw_text_from_llm=raw_themes_text,
            extraction_target_description="themes and motifs",
            desired_format_description="A Python dictionary with keys: 'primary_theme' (string), 'secondary_themes' (list of strings), 'recurring_motifs_or_symbols' (list of strings). Ensure list fields are actual lists of strings.",
            example_json_output='{"primary_theme": "The price of progress", "secondary_themes": ["Identity in a digital age", "Man vs. Machine"], "recurring_motifs_or_symbols": ["Broken mirrors", "Whispering winds"]}'
        )
        if isinstance(themes_motifs_dict, dict) and 'error' in themes_motifs_dict:
            raise RuntimeError(f"Failed to extract themes and motifs: {themes_motifs_dict.get('error')}")
        if not isinstance(themes_motifs_dict, dict):
            raise RuntimeError(f"Themes/motifs extraction did not yield a dictionary. Got: {type(themes_motifs_dict)}")

        for key in ['secondary_themes', 'recurring_motifs_or_symbols']:
            if key in themes_motifs_dict and isinstance(themes_motifs_dict[key], str):
                themes_motifs_dict[key] = [s.strip().lstrip('- ') for s in themes_motifs_dict[key].split('\n') if s.strip()]

        self.console.print(Panel(json.dumps(themes_motifs_dict, indent=2), title="Themes & Motifs", border_style="blue", expand=False))

        # --- 4. Plot Outline ---
        self.console.print("\n[cyan]Step 1.4: Generating Plot Outline...[/cyan]")
        plot_gen_prompt = f"""
Based on the novel's subject ("{self.subject}"), genre ("{self.genre}"), character profiles, world details, and identified themes, create a compelling plot outline.
Key Character Roles: {[(p.get('name'), p.get('role')) for p in character_profiles_list]}
Primary Theme: {themes_motifs_dict.get('primary_theme', 'N/A')}
Setting Summary: {world_details_dict.get('setting_description', 'N/A')}
{resume_text_for_prompt}

Provide the following, clearly labeled:
- Logline: [1-2 sentence summary of the core plot]
- Inciting Incident: [The event that kicks off the main story]
- Rising Action Beat 1: [Description]
- Rising Action Beat 2: [Description]
- Rising Action Beat 3: [Description]
- Rising Action Beat 4: [Optional: Description]
- Rising Action Beat 5: [Optional: Description]
- Climax: [The major turning point/confrontation]
- Falling Action Beat 1: [Description]
- Falling Action Beat 2: [Description]
- Falling Action Beat 3: [Optional: Description]
- Resolution: [How main conflicts are resolved; new normal]
- Potential Subplot 1: [Optional: Idea]
- Potential Subplot 2: [Optional: Idea]

Use clear labels for each section. Ensure Rising Action Beats and Falling Action Beats are presented as a list under their respective labels.
"""
        system_plot_gen_prompt = "You are a master plotter. Generate a plot outline as clearly formatted text."
        raw_plot_text = await self._ollama_generate_text(plot_gen_prompt, system_message=system_plot_gen_prompt, temperature=0.65)
        if raw_plot_text.startswith("Error:"):
            raise RuntimeError(f"Failed to generate raw plot text: {raw_plot_text}")
        self._log_to_file("foundational_raw_plot_text.txt", raw_plot_text)

        plot_outline_dict = await self._extract_data_with_llm(
            raw_text_from_llm=raw_plot_text,
            extraction_target_description="plot outline",
            desired_format_description="A Python dictionary with keys: 'logline', 'inciting_incident', 'rising_action_beats' (list of strings), 'climax', 'falling_action_beats' (list of strings), 'resolution', 'potential_subplots' (list of strings, can be empty).",
            example_json_output='{"logline": "...", "inciting_incident": "...", "rising_action_beats": ["Beat 1...", "Beat 2..."], "climax": "...", "falling_action_beats": ["...", "..."], "resolution": "...", "potential_subplots": ["Subplot A..."]}'
        )
        if isinstance(plot_outline_dict, dict) and 'error' in plot_outline_dict:
            raise RuntimeError(f"Failed to extract plot outline: {plot_outline_dict.get('error')}")
        if not isinstance(plot_outline_dict, dict):
             raise RuntimeError(f"Plot outline extraction did not yield a dictionary. Got: {type(plot_outline_dict)}")
        
        for key in ['rising_action_beats', 'falling_action_beats', 'potential_subplots']:
            if key in plot_outline_dict and isinstance(plot_outline_dict[key], str):
                 plot_outline_dict[key] = [s.strip() for s in plot_outline_dict[key].split('\n') if s.strip()]

        self.console.print(Panel(json.dumps(plot_outline_dict, indent=2), title="Plot Outline", border_style="green", expand=False))
        
        self.foundational_elements = {
            "novel_title": Prompt.ask(Text("Confirm or Edit Novel Title", style="yellow"), default=character_profiles_list[0].get("name", "My_Novel") + "_Story"),
            "point_of_view": Prompt.ask(Text("Confirm or Edit Point of View (e.g., Third Person Limited - Elara)", style="yellow"), default="Third Person Limited - " + character_profiles_list[0].get("name", "Protagonist")),
            "narrative_tense": Prompt.ask(Text("Confirm or Edit Narrative Tense", style="yellow"), default="Past Tense", choices=["Past Tense", "Present Tense"]),
            "target_audience": Prompt.ask(Text("Confirm or Edit Target Audience", style="yellow"), default="Adult"),
            "character_profiles": character_profiles_list,
            "world_details": world_details_dict,
            "themes_and_motifs": themes_motifs_dict,
            "plot_outline": plot_outline_dict
        }
        self._log_to_file("foundational_elements_extracted.json", json.dumps(self.foundational_elements, indent=2))
        self.console.print("[bold green]Foundational Elements generation complete and saved.[/bold green]")

    async def generate_detailed_chapter_plans(self):
        if not self.foundational_elements:
            self.console.print("[bold red]Error: Foundational elements not generated. Cannot plan chapters.[/bold red]")
            return

        self.console.print(Rule("[bold_blue]Generating Detailed Chapter Plans[/bold_blue]", style="blue"))
        num_chapters_str = Prompt.ask(
            Text("How many chapters for the novel (typically 20-30)?", style="yellow"),
            default="20",
        )
        try:
            num_chapters = int(num_chapters_str)
            if not (5 <= num_chapters <= 50): 
                raise ValueError("Chapter count out of sensible range (5-50).")
        except ValueError:
            self.console.print("[bold red]Invalid chapter count. Defaulting to 20.[/bold red]")
            num_chapters = 20

        context_summary = f"""
Novel Title: {self.foundational_elements.get('novel_title')}
POV: {self.foundational_elements.get('point_of_view')}, Tense: {self.foundational_elements.get('narrative_tense')}
Primary Theme: {self.foundational_elements.get('themes_and_motifs', {}).get('primary_theme')}
Logline: {self.foundational_elements.get('plot_outline', {}).get('logline')}
Key Characters: {[(p.get('name'), p.get('role'), p.get('primary_goal')) for p in self.foundational_elements.get('character_profiles', [])]}
Overall Plot Beats:
  Inciting Incident: {self.foundational_elements.get('plot_outline', {}).get('inciting_incident')}
  Rising Action Summary: {' -> '.join(self.foundational_elements.get('plot_outline', {}).get('rising_action_beats', []))}
  Climax Summary: {self.foundational_elements.get('plot_outline', {}).get('climax')}
  Falling Action Summary: {' -> '.join(self.foundational_elements.get('plot_outline', {}).get('falling_action_beats', []))}
  Resolution Summary: {self.foundational_elements.get('plot_outline', {}).get('resolution')}
World Setting: {self.foundational_elements.get('world_details', {}).get('setting_description')}
{f"Resume Snippet (for inspiration, if relevant):\n```\n{self.resume_text}\n```" if self.resume_text else ""}
"""
        self._log_to_file("chapter_planning_context.txt", context_summary)

        for i in range(1, num_chapters + 1):
            self.console.print(f"\n[cyan]Planning Chapter {i}/{num_chapters}...[/cyan]")
            
            previous_chapter_summary_text = ""
            if self.detailed_chapter_plans:
                last_plan = self.detailed_chapter_plans[-1]
                previous_chapter_summary_text = f"Summary of Chapter {last_plan.get('chapter_number')} ('{last_plan.get('title')}'): {last_plan.get('goal_event')}. It ended with: {last_plan.get('closing_hook_or_cliffhanger_idea', 'N/A')}."

            chapter_plan_gen_prompt = f"""
You are planning Chapter {i} of a {num_chapters}-chapter novel.
Novel Context:
{context_summary}

Previous Chapter Summary (if applicable):
{previous_chapter_summary_text or "This is the first chapter."}

Instructions for Chapter {i}:
Provide the following details for Chapter {i}, clearly labeled using Markdown-style headers (e.g., "### Title", "### Goal/Event"):
- Chapter Number: {i} (This MUST be {i})
- Title: [Evocative chapter title]
- Goal/Event: [The primary purpose or key event of this chapter. Must connect to previous and next chapters, and the overall plot outline.]
- Plot Advancements: [List specific ways this chapter moves the main plot or significant subplots forward. Use bullet points if multiple.]
- Character Focus and Development: [Describe which character(s) are central. What specific development or challenge related to their arc occurs? How are their motivations/internal states shown? Example: Elara: Confronts her fear of heights to retrieve the McGuffin. Jax: Reveals a piece of his hidden past.]
- Setting, Mood, Atmosphere: [Key setting details and the dominant mood/atmosphere to establish.]
- Key Conflict or Tension: [The main source of conflict or tension that drives this chapter.]
- Estimated Word Count: [Approximate target, e.g., 2000-3500 words]
- Opening Hook Idea: [Optional: Brief idea for an engaging chapter opening]
- Closing Hook or Cliffhanger Idea: [Optional: Brief idea for a hook or cliffhanger at the chapter's end]
"""
            system_chapter_gen_prompt = "You are an expert chapter planner. Generate the chapter plan as clearly formatted text with labels/Markdown headers."
            raw_chapter_plan_text = await self._ollama_generate_text(chapter_plan_gen_prompt, system_message=system_chapter_gen_prompt, temperature=0.6)
            if raw_chapter_plan_text.startswith("Error:"):
                raise RuntimeError(f"Failed to generate raw text for chapter {i} plan: {raw_chapter_plan_text}")
            self._log_to_file(f"chapter_{i}_raw_plan_text.txt", raw_chapter_plan_text)

            chapter_plan_dict = await self._extract_data_with_llm(
                raw_text_from_llm=raw_chapter_plan_text,
                extraction_target_description=f"plan for Chapter {i}",
                desired_format_description="A Python dictionary with keys: 'chapter_number' (int), 'title' (str), 'goal_event' (str), 'plot_advancements' (list of str or single str), 'character_focus_and_dev' (str or list of dicts like [{'character_name': 'Name', 'development': 'Description'}]), 'setting_mood_atmosphere' (str), 'key_conflict_or_tension' (str), 'estimated_word_count' (int or str like '2500 words'), 'opening_hook_idea' (optional str), 'closing_hook_or_cliffhanger_idea' (optional str). Ensure 'plot_advancements' is a list. 'character_focus_and_dev' can be a string initially, or a list of dictionaries.",
                example_json_output='{"chapter_number": 1, "title": "The Discovery", "goal_event": "...", "plot_advancements": ["...", "..."], "character_focus_and_dev": [{"character_name": "Elara", "development": "Shows bravery"}], "estimated_word_count": 2500, ...}'
            )

            if isinstance(chapter_plan_dict, dict) and 'error' in chapter_plan_dict:
                raise RuntimeError(f"Failed to extract plan for chapter {i}: {chapter_plan_dict.get('error')}")
            if not isinstance(chapter_plan_dict, dict):
                raise RuntimeError(f"Chapter {i} plan extraction did not yield a dictionary. Got: {type(chapter_plan_dict)}")

            chapter_plan_dict['chapter_number'] = i 
            if isinstance(chapter_plan_dict.get('plot_advancements'), str):
                chapter_plan_dict['plot_advancements'] = [s.strip() for s in chapter_plan_dict['plot_advancements'].split('\n') if s.strip() and not s.strip().startswith('- ')] 
            if isinstance(chapter_plan_dict.get('estimated_word_count'), str):
                match = re.search(r'\d+', chapter_plan_dict['estimated_word_count'])
                chapter_plan_dict['estimated_word_count'] = int(match.group(0)) if match else 2500
            
            self.detailed_chapter_plans.append(chapter_plan_dict)
            self.console.print(Panel(json.dumps(chapter_plan_dict, indent=2), title=f"Plan for Chapter {i}: {chapter_plan_dict.get('title', 'Untitled')}", border_style="cyan", expand=False))
            self._log_to_file(f"chapter_{i}_extracted_plan.json", json.dumps(chapter_plan_dict, indent=2))
        
        self.console.print("[bold green]Detailed Chapter Plans generation complete.[/bold green]")

    async def _summarize_prose_with_llm(self, prose: str, chapter_num: int, max_words: int = 200) -> str:
        self.console.print(f"[dim]Generating LLM-based summary for prose of Chapter {chapter_num}...[/dim]")
        summarizer_llm = self._get_ollama_llm(temperature=0.3, num_predict=max_words + 100) # + buffer for LLM verbosity
        
        summary_prompt = f"""
Concisely summarize the key events, critical plot advancements, significant character developments (emotions, decisions, new knowledge), and any cliffhangers or unresolved threads from the following chapter prose.
The summary should be around {max_words} words and serve as a "Previously On..." style recap for the writer to begin the next chapter. Focus on information crucial for continuity.

Chapter {chapter_num} Prose to Summarize (first ~3000 chars):
---
{prose[:3000]}{'...' if len(prose) > 3000 else ''}
---

Provide ONLY the summary text.
"""
        system_summary_prompt = "You are an expert at summarizing novel chapters for continuity purposes."
        
        summary_text = await self._ollama_generate_text(summary_prompt, system_message=system_summary_prompt)
        if summary_text.startswith("Error:") or not summary_text.strip():
            self.console.print(f"[orange_red1]LLM Summarization failed for Chapter {chapter_num}. Using truncation.[/orange_red1]")
            self._log_to_file(f"chapter_{chapter_num}_summary_error.txt", f"LLM summarization failed. Fallback to truncation. Original error: {summary_text}")
            return self._truncate_prose(prose, max_words) 
        
        self._log_to_file(f"chapter_{chapter_num}_summary_llm.txt", summary_text)
        return summary_text

    def _truncate_prose(self, prose: str, max_words: int = 150) -> str:
        words = prose.split()
        if len(words) > max_words:
            return " ".join(words[:max_words]) + "..."
        return prose

    async def _update_character_states_from_prose(self, chapter_num: int, chapter_prose: str, current_states: Dict[str, Any]) -> Dict[str, Any]:
        if not self.foundational_elements:
            return current_states

        self.console.print(f"[dim]Updating character states after Chapter {chapter_num} using LLM analysis...[/dim]")
        char_profiles = self.foundational_elements.get('character_profiles', [])
        char_names = [p.get('name', f'UnknownCharacter{idx}') for idx, p in enumerate(char_profiles)]
        if not char_names:
            self.console.print("[yellow]No character profiles found to update states for.[/yellow]")
            return current_states
        
        update_prompt = f"""
Analyze the following chapter prose (Chapter {chapter_num}) and update the states of the key characters.
Focus on characters: {', '.join(char_names)}.

Current Character States (before this chapter):
```json
{json.dumps(current_states, indent=2)}
```

Chapter {chapter_num} Prose (first ~2000 chars for context):
---
{chapter_prose[:2000]}{'...' if len(chapter_prose) > 2000 else ''}
---
(You are analyzing the FULL chapter prose that was provided to you implicitly)

Based on the FULL chapter prose, for each key character ({', '.join(char_names)}), describe any significant changes to their:
1.  `status_or_condition`: (e.g., "Injured", "Captured", "Safe", "Emotionally drained")
2.  `current_location`: (If changed or noteworthy)
3.  `prevailing_emotion`: (Their dominant emotion at the end of the chapter)
4.  `newly_acquired_knowledge_or_realizations`: (Key things they learned or understood)
5.  `key_relationships_update`: (How their relationships with other key characters changed, e.g., "Strengthened bond with Jax", "Increased suspicion of Thorne")
6.  `immediate_goal_or_motivation_shift`: (Any new immediate goals or changes in their driving motivations)

Output your analysis as a valid JSON object where keys are character names. Each value should be an object with the fields above.
If a character's state is largely unchanged in this chapter, you can note that or provide minimal updates.
Your response must be ONLY the JSON object.
Begin JSON object output now:
"""
        system_update_prompt = "You are an expert in character tracking and narrative analysis. Output ONLY the JSON object detailing character state updates."
        
        raw_state_update_text = await self._ollama_generate_text(update_prompt, system_message=system_update_prompt, temperature=0.3, json_mode=True) # Request JSON
        
        if raw_state_update_text.startswith("Error:"):
            self.console.print(f"[bold red]LLM failed to generate text for character state update after Ch {chapter_num}.[/bold red]")
            self._log_to_file("character_state_errors.log", f"Chapter {chapter_num} state update LLM error: {raw_state_update_text}")
            return current_states

        # Since json_mode=True was used, _ollama_generate_text should return a string that is valid JSON.
        # We still use _extract_data_with_llm as a robust parser that can handle minor LLM deviations from pure JSON.
        updated_states_dict = await self._extract_data_with_llm(
            raw_text_from_llm=raw_state_update_text, # This should already be a JSON string
            extraction_target_description="character state updates after chapter " + str(chapter_num),
            desired_format_description="A Python dictionary where keys are character names and values are dictionaries with keys like 'status_or_condition', 'current_location', etc.",
            example_json_output='{"Elara": {"status_or_condition": "Exhausted", "prevailing_emotion": "Determined"}, "Thorne": {"prevailing_emotion": "Frustrated"}}'
        )

        if isinstance(updated_states_dict, dict) and 'error' in updated_states_dict:
            self.console.print(f"[bold red]Error extracting character states after Chapter {chapter_num}: {updated_states_dict['error']}[/bold red]")
            self._log_to_file("character_state_errors.log", f"Chapter {chapter_num} state extraction error: {updated_states_dict}")
            return current_states 

        if not isinstance(updated_states_dict, dict):
            self.console.print(f"[bold red]Character state update extraction did not return a dictionary for Ch {chapter_num}. Got: {type(updated_states_dict)}[/bold red]")
            self._log_to_file("character_state_errors.log", f"Chapter {chapter_num} state extraction not a dict. Got: {updated_states_dict}")
            return current_states

        final_states = json.loads(json.dumps(current_states)) 
        for char_name_from_update, updates in updated_states_dict.items():
            canonical_char_name = next((cn for cn in char_names if cn.lower() == char_name_from_update.lower()), None)
            if not canonical_char_name: 
                if char_name_from_update not in ["error", "raw_output"]: 
                    self.console.print(f"[yellow]Warning: LLM provided state update for unknown character '{char_name_from_update}'. Skipping.[/yellow]")
                continue

            if canonical_char_name not in final_states: 
                 final_states[canonical_char_name] = {}
            
            if "history" not in final_states[canonical_char_name]:
                final_states[canonical_char_name]["history"] = []
            
            update_summary = {k:v for k,v in updates.items()}
            final_states[canonical_char_name]["history"].append({f"chapter_{chapter_num}_update": update_summary})
            final_states[canonical_char_name].update(updates) 

        self._log_to_file(f"chapter_{chapter_num}_character_states.json", json.dumps(final_states, indent=2))
        return final_states

    def _build_context_string_for_agents(self, chapter_num: int, current_character_states: Dict[str, Any], previous_chapter_llm_summary: str) -> str:
        """Builds the continuity context string using LLM-generated summary."""
        if not self.foundational_elements:
            return "Error: Foundational elements missing."
        fe = self.foundational_elements
        context = f"Novel Title: {fe.get('novel_title', 'N/A')}\n"
        context += f"Overall Theme: {fe.get('themes_and_motifs', {}).get('primary_theme', 'N/A')}\n"
        context += f"POV: {fe.get('point_of_view', 'N/A')}, Tense: {fe.get('narrative_tense', 'N/A')}\n"
        context += f"Author Style to Emulate: {self.author_style}, Genre: {self.genre}\n"
        
        context += f"\n--- PREVIOUS CHAPTER SUMMARY (LLM Generated) ---\n{previous_chapter_llm_summary or 'This is the first chapter.'}\n"

        context += "\n--- CURRENT CHARACTER STATES (ENTERING THIS CHAPTER) ---\n"
        for char_name, state_data in current_character_states.items():
            context += f"- Character: {char_name}\n"
            if isinstance(state_data, dict):
                char_profile = next((p for p in fe.get('character_profiles', []) if p.get('name') == char_name), {})
                context += f"  - Role: {state_data.get('role', char_profile.get('role', 'N/A'))}\n"
                context += f"  - Current Status/Condition: {state_data.get('status_or_condition', 'Normal')}\n"
                context += f"  - Current Location: {state_data.get('current_location', 'Unknown')}\n"
                context += f"  - Prevailing Emotion: {state_data.get('prevailing_emotion', 'Neutral')}\n"
                context += f"  - Recent Knowledge/Realizations: {state_data.get('newly_acquired_knowledge_or_realizations', 'None specific')}\n"
                context += f"  - Immediate Goal/Motivation: {state_data.get('immediate_goal_or_motivation_shift', state_data.get('primary_goal', char_profile.get('primary_goal','Not specified')))}\n"
            else:
                context += f"  - State Info: {str(state_data)[:200]}...\n" 
        return context

    async def run_prose_editor_agent(self, chapter_prose: str, chapter_plan: Dict[str, Any], chapter_num: int, states_before_chapter: Dict[str, Any], prev_chapter_llm_summary: str) -> Optional[Dict[str, Any]]:
        self.console.print(f"[dim]Running Prose Editor Agent for Chapter {chapter_num}...[/dim]")
        if not self.foundational_elements: return None

        editor_prompt = f"""
You are an expert novel editor. Critically evaluate the following chapter prose against the provided plan and writing principles.

--- NOVEL WRITING PRINCIPLES ---
{NOVEL_WRITING_PRINCIPLES}

--- OVERALL NOVEL CONTEXT ---
Novel Title: {self.foundational_elements.get('novel_title')}
Genre: {self.genre}, Author Style: {self.author_style}
POV: {self.foundational_elements.get('point_of_view')}, Tense: {self.foundational_elements.get('narrative_tense')}
Primary Theme: {self.foundational_elements.get('themes_and_motifs', {}).get('primary_theme')}

--- PREVIOUS CHAPTER SUMMARY (LLM Generated) ---
{prev_chapter_llm_summary or "This is the first chapter."}

--- CHARACTER STATES (ENTERING THIS CHAPTER) ---
{json.dumps(states_before_chapter, indent=2)}

--- CURRENT CHAPTER ({chapter_num}: "{chapter_plan.get('title')}") PLAN ---
Goal/Event: {chapter_plan.get('goal_event')}
Plot Advancements: {chapter_plan.get('plot_advancements')}
Character Focus & Development: {chapter_plan.get('character_focus_and_dev')}
Setting/Mood/Atmosphere: {chapter_plan.get('setting_mood_atmosphere')}
Key Conflict/Tension: {chapter_plan.get('key_conflict_or_tension')}

--- CHAPTER {chapter_num} PROSE TO REVIEW (first ~5000 chars) ---
```text
{chapter_prose[:5000]}{'...' if len(chapter_prose) > 5000 else ''}
```
(You are reviewing the FULL chapter prose that was provided to you implicitly)

--- EDITORIAL TASK ---
Provide specific, actionable feedback. Identify up to 5-7 key areas for improvement. For each point, explain the issue and suggest how it might be addressed.
Consider:
1.  Adherence to Chapter Plan: Does it meet goals, advance plot, develop specified characters?
2.  Continuity: Does it flow logically from previous summary and character states?
3.  Characterization: Are portrayals consistent, believable? Is development shown effectively?
4.  Pacing & Tension: Is pacing appropriate? Is conflict/tension well-managed?
5.  Style & Voice: Is it consistent with the novel's style, POV, and tense?
6.  Dialogue: Is it natural, purposeful, character-specific?
7.  Show, Don't Tell: Are emotions, motivations, and descriptions vivid and shown through action/sensory details?
8.  Clarity & Engagement: Is the writing clear, immersive, and free of awkward phrasing?

Output your feedback as a JSON object with three keys:
"overall_assessment": "A brief (1-2 sentence) overall assessment of the chapter's quality and readiness."
"feedback_points": A list of strings, each a specific, actionable feedback point. (e.g., ["The pacing in the market scene feels too slow; consider shortening descriptions and increasing dialogue frequency.", "Elara's motivation for confronting Thorne seems underdeveloped in this draft; show more of her internal reasoning leading up to it."])
"revision_needed": boolean (true if significant revisions are recommended, false if only minor tweaks or acceptable as is).

Your response MUST be ONLY the JSON object.
Begin JSON object output now:
"""
        system_editor_prompt = "You are a meticulous novel editor. Provide constructive, specific feedback in the requested JSON format."
        
        # Generate raw text for editor feedback, requesting JSON format from Ollama
        raw_editor_feedback_text = await self._ollama_generate_text(editor_prompt, system_message=system_editor_prompt, temperature=0.4, json_mode=True)
        
        if raw_editor_feedback_text.startswith("Error:"):
            self.console.print(f"[bold red]Editor agent LLM call failed for Chapter {chapter_num}.[/bold red]")
            self._log_to_file(f"chapter_{chapter_num}_editor_error.txt", f"LLM call error: {raw_editor_feedback_text}")
            return {"overall_assessment": "Editor LLM call failed.", "feedback_points": [], "revision_needed": False, "error": True}

        # Extract structured feedback from the LLM's (hopefully) JSON output
        editor_feedback = await self._extract_data_with_llm(
            raw_text_from_llm=raw_editor_feedback_text, # This should be a JSON string
            extraction_target_description=f"editorial feedback for Chapter {chapter_num}",
            desired_format_description="A JSON object with keys: 'overall_assessment' (string), 'feedback_points' (list of strings), 'revision_needed' (boolean).",
            example_json_output='{"overall_assessment": "Good start, but needs work on pacing.", "feedback_points": ["Point 1...", "Point 2..."], "revision_needed": true}'
        )
        if isinstance(editor_feedback, dict) and 'error' in editor_feedback:
            self.console.print(f"[bold red]Failed to extract editor feedback for Chapter {chapter_num}.[/bold red]")
            self._log_to_file(f"chapter_{chapter_num}_editor_extraction_error.txt", f"Extraction error: {editor_feedback}\nRaw text: {raw_editor_feedback_text}")
            return {"overall_assessment": "Failed to extract feedback.", "feedback_points": [], "revision_needed": False, "error": True}
        
        self._log_to_file(f"chapter_{chapter_num}_editor_feedback.json", json.dumps(editor_feedback, indent=2))
        return editor_feedback


    async def run_prose_revision_agent(self, original_prose: str, editor_feedback: List[str], chapter_plan: Dict[str, Any], chapter_num: int, states_before_chapter: Dict[str, Any], prev_chapter_llm_summary: str) -> str:
        self.console.print(f"[dim]Running Prose Revision Agent for Chapter {chapter_num}...[/dim]")
        if not self.foundational_elements: return original_prose

        revision_prompt = f"""
You are a master story rewriter. Your task is to revise the provided chapter prose based on the editor's feedback.

--- NOVEL WRITING PRINCIPLES ---
{NOVEL_WRITING_PRINCIPLES}

--- OVERALL NOVEL CONTEXT ---
Novel Title: {self.foundational_elements.get('novel_title')}
Genre: {self.genre}, Author Style: {self.author_style}
POV: {self.foundational_elements.get('point_of_view')}, Tense: {self.foundational_elements.get('narrative_tense')}

--- PREVIOUS CHAPTER SUMMARY (LLM Generated) ---
{prev_chapter_llm_summary or "This is the first chapter."}

--- CHARACTER STATES (ENTERING THIS CHAPTER) ---
{json.dumps(states_before_chapter, indent=2)}

--- CURRENT CHAPTER ({chapter_num}: "{chapter_plan.get('title')}") PLAN (REMAIN UNCHANGED) ---
Goal/Event: {chapter_plan.get('goal_event')}
Plot Advancements: {chapter_plan.get('plot_advancements')}
Character Focus & Development: {chapter_plan.get('character_focus_and_dev')}
Setting/Mood/Atmosphere: {chapter_plan.get('setting_mood_atmosphere')}
Key Conflict/Tension: {chapter_plan.get('key_conflict_or_tension')}
Estimated Word Count: {chapter_plan.get('estimated_word_count', 2500)} words.

--- EDITOR'S FEEDBACK TO ADDRESS ---
{chr(10).join(f"- {fb}" for fb in editor_feedback)}

--- ORIGINAL CHAPTER {chapter_num} PROSE ---
```text
{original_prose}
```

--- REVISION INSTRUCTIONS ---
1.  Carefully consider each feedback point from the editor.
2.  Rewrite the chapter prose to meticulously address all feedback while preserving the original chapter's strengths.
3.  Ensure the revised chapter still adheres to the chapter plan, overall novel context (POV, tense, style), and writing principles.
4.  The revised prose should demonstrate clear improvement and be ready for publication.
5.  Your output should be ONLY the revised prose for this chapter. Do not include titles, chapter numbers, or any other meta-text.

Begin revised Chapter {chapter_num} prose now:
"""
        system_revision_prompt = "You are a skilled novelist revising a chapter based on editorial feedback. Output ONLY the revised chapter prose."
        
        reviser_llm = self._get_ollama_llm(temperature=0.7, num_predict=-1) 
        revised_prose = await self._ollama_generate_text(revision_prompt, system_message=system_revision_prompt)

        if revised_prose.startswith("Error:"):
            self.console.print(f"[bold red]Prose Revision Agent LLM call failed for Chapter {chapter_num}. Returning original prose.[/bold red]")
            self._log_to_file(f"chapter_{chapter_num}_revision_error.txt", f"LLM call error: {revised_prose}")
            return original_prose
        
        self._log_to_file(f"chapter_{chapter_num}_revision_feedback_applied.txt", "\n".join(editor_feedback))
        self._log_to_file(f"chapter_{chapter_num}_prose_REVISED.txt", revised_prose)
        return revised_prose


    async def generate_novel_prose(self): 
        if not self.foundational_elements or not self.detailed_chapter_plans:
            self.console.print("[bold red]Error: Foundational elements or chapter plans not generated. Cannot write prose.[/bold red]")
            return

        self.console.print(Rule("[bold_blue]Generating & Refining Novel Prose[/bold_blue]", style="blue"))
        
        current_character_states = {}
        if self.foundational_elements.get('character_profiles'):
            for profile in self.foundational_elements['character_profiles']:
                char_name = profile.get('name')
                if char_name:
                    current_character_states[char_name] = {
                        "role": profile.get('role'), "summary_backstory": profile.get('backstory_summary'),
                        "core_motivation": profile.get('core_motivation'), "primary_goal": profile.get('primary_goal'),
                        "status_or_condition": "As per start of novel", "current_location": "Initial setting",
                        "prevailing_emotion": "Neutral/As per backstory",
                        "newly_acquired_knowledge_or_realizations": "None yet",
                        "key_relationships_update": profile.get('initial_relationship_to_protagonist', "To be established"),
                        "immediate_goal_or_motivation_shift": profile.get('primary_goal'),
                        "history": [{"initial_state": profile}]
                    }
        self.character_states_after_chapter[0] = current_character_states

        progress_columns = [
            SpinnerColumn(spinner_name="dots12"),
            "[progress.description]{task.description}", BarColumn(bar_width=None), "{task.percentage:>3.0f}%",
            "Elapsed:", TimeElapsedColumn(), "ETA:", TimeRemainingColumn()
        ]
        with Progress(*progress_columns, console=self.console, transient=False) as progress_bar:
            prose_task = progress_bar.add_task("[#FFBF00]Processing Chapters...", total=len(self.detailed_chapter_plans))

            for chapter_plan in self.detailed_chapter_plans:
                chapter_num = chapter_plan.get('chapter_number', 0)
                chapter_title = chapter_plan.get('title', f'Chapter {chapter_num}')
                
                progress_bar.update(prose_task, description=f"[#FFBF00]Ch. {chapter_num} ('{chapter_title}'): Drafting...[/]")
                self.console.print(Rule(f"[bold #87CEEB]Drafting Chapter {chapter_num}: {chapter_title}[/bold #87CEEB]", style="#87CEEB"))

                states_before_this_chapter = self.character_states_after_chapter.get(chapter_num - 1, current_character_states)
                
                prev_chapter_llm_summary = ""
                if chapter_num > 1 and (chapter_num - 1) in self.generated_chapters_prose:
                    prev_prose = self.generated_chapters_prose[chapter_num - 1]
                    prev_chapter_llm_summary = await self._summarize_prose_with_llm(prev_prose, chapter_num -1, max_words=250) # Increased summary length
                    if prev_chapter_llm_summary.startswith("Error:"):
                        self.console.print(f"[orange_red1]Using truncated summary for Ch {chapter_num-1} due to LLM summarization error.[/orange_red1]")
                        prev_chapter_llm_summary = self._truncate_prose(prev_prose, max_words=250)
                
                continuity_context_for_drafting = self._build_context_string_for_agents(
                    chapter_num, states_before_this_chapter, prev_chapter_llm_summary
                )

                prose_prompt = f"""
You are a master novelist, channeling the style of "{self.author_style}" within the "{self.genre}" genre.
Write Chapter {chapter_num} titled "{chapter_title}".

--- NOVEL WRITING PRINCIPLES TO FOLLOW ---
{NOVEL_WRITING_PRINCIPLES}

--- CONTINUITY CONTEXT (PREVIOUS EVENTS & CHARACTER STATES) ---
{continuity_context_for_drafting}

--- CURRENT CHAPTER PLAN ---
Goal/Event: {chapter_plan.get('goal_event')}
Plot Advancements: {', '.join(chapter_plan.get('plot_advancements', []))}
Character Focus & Development: {json.dumps(chapter_plan.get('character_focus_and_dev', []), indent=2)}
Setting/Mood/Atmosphere: {chapter_plan.get('setting_mood_atmosphere')}
Key Conflict/Tension: {chapter_plan.get('key_conflict_or_tension')}
Opening Hook Idea: {chapter_plan.get('opening_hook_idea', 'Start engagingly.')}
Closing Hook/Cliffhanger Idea: {chapter_plan.get('closing_hook_or_cliffhanger_idea', 'End with impact or anticipation.')}
Estimated Word Count: Aim for approximately {chapter_plan.get('estimated_word_count', 2500)} words.

--- WRITING INSTRUCTIONS ---
1.  Adhere strictly to the novel's POV ({self.foundational_elements.get('point_of_view')}) and Tense ({self.foundational_elements.get('narrative_tense')}).
2.  Fulfill all aspects of the chapter plan. Show, don't tell.
3.  Weave in the setting, mood, and atmosphere naturally.
4.  Ensure character actions and dialogue are consistent with their established states and planned development.
5.  Maintain high literary quality, with vivid descriptions, engaging pacing, and purposeful dialogue.
6.  Your output should be ONLY the prose for this chapter. Do not include titles, chapter numbers, or any other meta-text.
Begin Chapter {chapter_num} prose now:
"""
                system_prose_prompt = f"You are writing a chapter for a novel. Embody the specified author style and genre. Focus on narrative flow, character depth, and fulfilling the chapter plan."
                
                prose_llm = self._get_ollama_llm(temperature=0.75, num_predict=-1) 
                draft_prose = await self._ollama_generate_text(prose_prompt, system_message=system_prose_prompt)

                if draft_prose.startswith("Error:") or not draft_prose.strip():
                    self.console.print(f"[bold red]Error generating DRAFT prose for Chapter {chapter_num}: {draft_prose}.[/bold red]")
                    self.generated_chapters_prose[chapter_num] = f"Error: Could not generate DRAFT prose. Details: {draft_prose}"
                    current_character_states = await self._update_character_states_from_prose(chapter_num, "PROSE GENERATION FAILED", states_before_this_chapter)
                    self.character_states_after_chapter[chapter_num] = current_character_states
                    progress_bar.advance(prose_task)
                    continue 

                self.console.print(f"[green]Draft prose for Chapter {chapter_num} generated.[/green]")
                self._log_to_file(f"chapter_{chapter_num}_prose_DRAFT.txt", draft_prose)
                
                current_prose_iteration = draft_prose

                for rev_attempt in range(MAX_PROSE_REVISION_ATTEMPTS):
                    progress_bar.update(prose_task, description=f"[#FFBF00]Ch. {chapter_num}: Editing (Pass {rev_attempt+1})...[/]")
                    editor_feedback_data = await self.run_prose_editor_agent(
                        current_prose_iteration, chapter_plan, chapter_num, states_before_this_chapter, prev_chapter_llm_summary
                    )
                    
                    if editor_feedback_data is None or editor_feedback_data.get("error"):
                        self.console.print(f"[orange_red1]Prose Editor Agent failed or errored for Chapter {chapter_num}. Using current prose iteration.[/orange_red1]")
                        break 
                    
                    self.console.print(f"[dim]Editor Assessment (Ch. {chapter_num}, Pass {rev_attempt+1}): {editor_feedback_data.get('overall_assessment')}[/dim]")
                    for fb_point in editor_feedback_data.get("feedback_points", []):
                        self.console.print(f"[italic #B0C4DE]  - {fb_point}[/italic #B0C4DE]")

                    if not editor_feedback_data.get("revision_needed", True): 
                        self.console.print(f"[green]Chapter {chapter_num} prose accepted by editor (Pass {rev_attempt+1}).[/green]")
                        break
                    
                    progress_bar.update(prose_task, description=f"[#FFBF00]Ch. {chapter_num}: Revising (Pass {rev_attempt+1})...[/]")
                    current_prose_iteration = await self.run_prose_revision_agent(
                        current_prose_iteration, editor_feedback_data.get("feedback_points", []),
                        chapter_plan, chapter_num, states_before_this_chapter, prev_chapter_llm_summary
                    )
                    self.console.print(f"[green]Chapter {chapter_num} prose revised (Pass {rev_attempt+1}).[/green]")
                    self._log_to_file(f"chapter_{chapter_num}_prose_REVISED_Pass{rev_attempt+1}.txt", current_prose_iteration)
                
                self.generated_chapters_prose[chapter_num] = current_prose_iteration 
                
                progress_bar.update(prose_task, description=f"[#FFBF00]Ch. {chapter_num}: Updating Character States...[/]")
                states_after_this_chapter = await self._update_character_states_from_prose(
                    chapter_num, self.generated_chapters_prose[chapter_num], states_before_this_chapter
                )
                self.character_states_after_chapter[chapter_num] = states_after_this_chapter
                current_character_states = states_after_this_chapter 

                progress_bar.advance(prose_task)
        self.console.print("[bold green]Novel Prose generation and refinement complete.[/bold green]")


    def compile_and_save_novel(self) -> Optional[str]:
        if not self.foundational_elements or not self.detailed_chapter_plans or not self.generated_chapters_prose:
            self.console.print("[bold red]Cannot compile novel: Missing foundational elements, chapter plans, or prose.[/bold red]")
            return None

        fe = self.foundational_elements
        novel_title_for_file = self._sanitize_filename(fe.get('novel_title', 'UntitledNovel'))
        model_name_sanitized = self._sanitize_filename(self.ollama_model_name)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        md_filename = os.path.join(self.output_dir, f"{novel_title_for_file}_{model_name_sanitized}_{timestamp}.md")
        
        md_content = [f"# {fe.get('novel_title', 'Untitled Novel')}\n"]
        md_content.append(f"_By AI ({self.ollama_model_name}) with guidance_\n")
        md_content.append(f"**Genre:** {self.genre}, **Author Style Influence:** {self.author_style}\n")
        md_content.append(f"**POV:** {fe.get('point_of_view')}, **Tense:** {fe.get('narrative_tense')}\n")
        md_content.append(f"**Primary Theme:** {fe.get('themes_and_motifs', {}).get('primary_theme')}\n")
        md_content.append(f"**Logline:** {fe.get('plot_outline', {}).get('logline')}\n\n---\n")

        for i in range(1, len(self.detailed_chapter_plans) + 1):
            plan = next((p for p in self.detailed_chapter_plans if p.get('chapter_number') == i), None)
            prose = self.generated_chapters_prose.get(i)
            if plan and prose:
                md_content.append(f"## Chapter {i}: {plan.get('title', f'Chapter {i}')}\n\n{prose}\n\n---\n")
            elif plan:
                 md_content.append(f"## Chapter {i}: {plan.get('title', f'Chapter {i}')}\n\n[PROSE GENERATION FAILED OR SKIPPED FOR THIS CHAPTER]\n\n---\n")

        try:
            with open(md_filename, "w", encoding="utf-8") as f:
                f.write("\n".join(md_content))
            self.console.print(Panel(f"Novel saved to Markdown: [bold cyan]{md_filename}[/bold cyan]", title="File Saved", border_style="cyan"))
            return md_filename
        except Exception as e:
            self.console.print(f"[bold red]Error saving Markdown novel: {e}[/bold red]")
            return None
            
    def _save_full_metadata(self):
        if not self.foundational_elements: return

        metadata_payload = {
            "generation_timestamp": datetime.datetime.now().isoformat(),
            "novel_subject": self.subject,
            "author_style_influence": self.author_style,
            "genre": self.genre,
            "ollama_model_used": self.ollama_model_name,
            "resume_snippet_provided": bool(self.resume_text),
            "foundational_elements": self.foundational_elements, 
            "detailed_chapter_plans": self.detailed_chapter_plans, 
            "character_states_chronology": {f"after_chapter_{k}": v for k,v in self.character_states_after_chapter.items()}
        }
        novel_title_for_file = self._sanitize_filename(self.foundational_elements.get('novel_title', 'UntitledNovel'))
        model_name_sanitized = self._sanitize_filename(self.ollama_model_name)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata_filename = os.path.join(self.output_dir, f"{novel_title_for_file}_{model_name_sanitized}_{timestamp}_METADATA.json")
        try:
            with open(metadata_filename, "w", encoding="utf-8") as f:
                json.dump(metadata_payload, f, indent=2)
            self.console.print(f"[dim]Full generation metadata saved to: {metadata_filename}[/dim]")
        except Exception as e:
            self.console.print(f"[bold red]Error saving metadata: {e}[/bold red]")

    async def run_pipeline(self):
        try:
            self.console.print(Rule(f"[bold #FFD700]--- Starting Advanced Novel Generation Pipeline (Ollama: {self.ollama_model_name}) ---[/bold #FFD700]", style="#FFD700"))
            await self.generate_foundational_elements()
            await self.generate_detailed_chapter_plans()
            await self.generate_novel_prose() 
            md_file = self.compile_and_save_novel()
            self._save_full_metadata()
            
            if md_file and Confirm.ask("\n[bold yellow3]Print first chapter to terminal?[/bold yellow3]", default=True):
                 if self.generated_chapters_prose.get(1) and self.detailed_chapter_plans:
                    console.print(Rule("[bold medium_purple1]First Chapter[/bold medium_purple1]", style="medium_purple1"))
                    console.print(f"\n[bold medium_purple1]## Chapter 1: {self.detailed_chapter_plans[0].get('title', 'Chapter 1')}[/bold medium_purple1]\n")
                    console.print(Text(self.generated_chapters_prose[1], style="default"))

            self.console.print(Rule("[bold spring_green2]--- Novel Generation Pipeline Complete ---[/bold spring_green2]", style="spring_green2"))
        except RuntimeError as e: 
            self.console.print(Panel(f"[bold red]Pipeline Halting due to error: {e}[/bold red]", title="Pipeline Error", border_style="red"))
            self._log_to_file("generation_pipeline.log", f"HALT: {e}")
        except Exception as e: 
            self.console.print(Panel(f"[bold red]An unexpected critical error occurred in the pipeline: {e}[/bold red]", title="Critical Pipeline Error", border_style="red"))
            import traceback
            tb_str = traceback.format_exc()
            self.console.print(tb_str)
            self._log_to_file("generation_pipeline.log", f"CRITICAL UNEXPECTED ERROR: {e}\n{tb_str}")


def get_multiline_input(prompt_message: str) -> str:
    lines = []
    console.print(f"[bold sky_blue1]{prompt_message}[/bold sky_blue1] (Type '[bold green]ENDINPUT[/bold green]' on a new line when done, or just press Enter if input is short):")
    while True:
        line = input()
        if line.strip().upper() == "ENDINPUT":
            break
        if not lines and not line.strip():
            break
        lines.append(line)
    return "\n".join(lines)

def extract_text_from_pdf(pdf_path: str) -> Optional[str]:
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        doc.close()
        return text
    except ImportError:
        console.print("[bold orange_red1]PyMuPDF (fitz) is not installed. Cannot extract text from PDF. Run: `pip install PyMuPDF`[/bold orange_red1]")
        return None
    except Exception as e:
        console.print(f"[bold red]Error extracting text from PDF {pdf_path}: {e}[/bold red]")
        return None

async def main_terminal_advanced():
    console.rule("[bold #FF8C00]Welcome to the Novel Generator (Advanced Prose Refinement)![/bold #FF8C00]", style="#FF8C00")
    console.print("-------------------------------------------------------------", style="dim #FF8C00")
    console.print("Ensure your Ollama server is running and models are pulled.", style="italic #FFA07A")
    console.print("This version includes LLM-powered summarization and prose review/revision.", style="italic #FFA07A")
    console.print("-------------------------------------------------------------", style="dim #FF8C00")

    global SELECTED_MODEL_NAME
    ollama_model_input = Prompt.ask(
        Text("Enter Ollama Model Name", style="bold sky_blue1"),
        default=SELECTED_MODEL_NAME
    )
    SELECTED_MODEL_NAME = ollama_model_input
    console.print(f"Using Ollama Model: [bold cyan]{SELECTED_MODEL_NAME}[/bold cyan]")

    resume_text_content = None
    resume_path = Prompt.ask(Text("Enter path to resume file (text/PDF) or press Enter to skip", style="sky_blue1"), default="")
    if resume_path.strip():
        if resume_path.lower().endswith(".pdf"):
            resume_text_content = extract_text_from_pdf(resume_path)
            if resume_text_content:
                 console.print(f"[green]Successfully extracted text from PDF: {resume_path}[/green]")
            else:
                console.print(f"[orange_red1]Could not extract text from PDF: {resume_path}.[/orange_red1]")
        elif resume_path.lower().endswith(".txt"):
            try:
                with open(resume_path, "r", encoding="utf-8") as f:
                    resume_text_content = f.read()
                console.print(f"[green]Successfully read text from: {resume_path}[/green]")
            except Exception as e:
                console.print(f"[orange_red1]Error reading text file {resume_path}: {e}.[/orange_red1]")
        else:
            console.print(f"[orange_red1]Unsupported resume file type: {resume_path}. Only .txt and .pdf.[/orange_red1]")

    subject = get_multiline_input("Enter the novel's subject/premise")
    author_style = Prompt.ask(Text("Enter desired author style (e.g., 'Stephen King', 'Jane Austen')", style="sky_blue1"), default="Stephen King")
    genre = Prompt.ask(Text("Enter genre(s) (e.g., 'Sci-Fi/Thriller', 'Historical Romance')", style="sky_blue1"), default="Sci-Fi")

    generator = NovelGenerator(
        subject=subject, author_style=author_style, genre=genre,
        ollama_model_name=SELECTED_MODEL_NAME, ollama_base_url=OLLAMA_BASE_URL,
        resume_text=resume_text_content
    )
    
    console.print(Panel(
        Text.assemble(
            ("Subject: ", "bold"), generator.subject[:200] + ("..." if len(generator.subject) > 200 else ""), "\n",
            ("Author Style: ", "bold"), generator.author_style, "\n",
            ("Genre: ", "bold"), generator.genre, "\n",
            ("Resume Provided: ", "bold"), "Yes" if generator.resume_text else "No"
        ),
        title="Confirm Inputs", border_style="steel_blue1"
    ))

    if Confirm.ask("\n[bold gold3]Start novel generation with these inputs?[/bold gold3]", default=True):
        await generator.run_pipeline()
    else:
        console.print("[italic]Novel generation cancelled.[/italic]")

if __name__ == "__main__":
    console.print(f"[info]Script starting... Default Ollama Model: [bold cyan]{SELECTED_MODEL_NAME}[/bold cyan] at [underline]{OLLAMA_BASE_URL}[/underline][/info]")
    console.print(f"[italic yellow]Make sure your Ollama server is running and the model '{SELECTED_MODEL_NAME}' is available.[/italic]")
    console.print(f"[italic yellow]You can typically pull a model using: ollama pull {SELECTED_MODEL_NAME}[/italic yellow]")
    
    import nest_asyncio
    nest_asyncio.apply()
    
    asyncio.run(main_terminal_advanced())
