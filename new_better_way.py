import requests
import json
import time
import os
import re


class BookGenerator:
    def __init__(self):
        self.base_url = "http://localhost:11434/api/generate"
        self.model = "gemma3:12b" # Consider using a model suited for creative writing if available
        self.story_premise = ""
        self.genre = "" # Added genre attribute
        self.num_chapters = 0
        self.story_outline = ""
        self.chapters = []
        self.characters = {}  # Changed from list to dictionary for easier lookup
        self.chapter_summaries = {}  # Store detailed summaries of each chapter
        self.settings = {}  # Track settings/locations
        self.plot_events = []  # Track major plot events
        self.world_name = ""  # Consistent world name
        self.chapter_plan = ""  # Store the detailed chapter plan
        self.timeline = {}  # Track time progression between chapters
        self.emotional_arc = {}  # Track emotional tone in chapters
        self.transitions = {}  # Store generated transitions between chapters
        self.recurring_motifs = []  # Track recurring motifs or symbols for continuity

    def get_user_input(self):
        """Get the story premise, genre, and number of chapters from the user"""
        print("""
CAPABILITIES: Creates novels or fanfiction with consistency checks
GENERATION TIME: Up to an hour depending on computational resources
STORY INPUT: One paragraph up to 830 characters with your plot idea
"""
        )
        self.story_premise = input("Please provide a paragraph about what your story is about: ")
        # Get genre input
        self.genre = input("Please specify the genre for your story (e.g., Sci-Fi, Fantasy, Mystery): ").strip()
        if not self.genre: # Basic fallback if user enters nothing
            self.genre = "General Fiction"
            print("No genre specified, defaulting to General Fiction.")

        while True:
            try:
                self.num_chapters = int(input("How many chapters would you like (minimum 3): "))
                if self.num_chapters >= 3:
                    break
                else:
                    print("Please enter at least 3 chapters.")
            except ValueError:
                print("Please enter a valid number.")

    def generate_text(self, prompt, system_prompt="You are a creative fiction writer."):
        """Make API call to Ollama with the given prompt"""
        data = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
        }

        try:
            response = requests.post(self.base_url, json=data)
            response.raise_for_status()
            # Basic check for empty or error response from the model itself
            response_data = response.json()
            if "response" not in response_data or not response_data["response"]:
                 print(f"Warning: Received empty response from model for prompt:\n---\n{prompt[:200]}...\n---")
                 return None # Return None for empty response
            return response_data["response"]
        except requests.exceptions.RequestException as e:
            print(f"Error making request to Ollama: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response from Ollama: {e}")
            print(f"Response text: {response.text}")
            return None


    def extract_characters(self, text):
        """Extract character information from text and create structured data"""
        if not text: # Handle case where text is None or empty
            return {}
        characters = {}
        # Look for patterns like "CHARACTER NAME: description"
        # Made pattern more robust to handle variations in spacing and line breaks
        pattern = r"([A-Z][A-Za-z\s'-]+):\s+([\s\S]*?)(?=\n[A-Z][A-Za-z\s'-]+:|\Z)"
        matches = re.findall(pattern, text)

        for match in matches:
            name = match[0].strip()
            description = match[1].strip()
            if name and description: # Ensure name and description are not empty
                characters[name] = {
                    "name": name,
                    "description": description,
                    "first_appearance": 0, # Will be updated when character first appears
                    "status": "unknown", # Initial status
                    "development": [],
                    "relationships": {},
                    "location": "unknown",
                    "emotional_state": "unknown"
                }
            else:
                print(f"Warning: Skipped potential character entry due to missing name or description: {match}")


        # Fallback if the primary pattern fails - look for capitalized names at start of lines
        if not characters:
             print("Primary character extraction pattern failed. Trying fallback.")
             # Simple fallback: look for capitalized words followed by a colon at the start of a line
             fallback_pattern = r"^\s*([A-Z][A-Za-z\s'-]+):\s*(.*)"
             lines = text.split('\n')
             for line in lines:
                 match = re.match(fallback_pattern, line)
                 if match:
                     name = match.group(1).strip()
                     description = match.group(2).strip()
                     if name and description and name not in characters: # Avoid duplicates
                         characters[name] = {
                             "name": name,
                             "description": description,
                             "first_appearance": 0,
                             "status": "unknown",
                             "development": [],
                             "relationships": {},
                             "location": "unknown",
                             "emotional_state": "unknown"
                         }

        if not characters:
            print("Warning: No characters extracted. Check the format of the character text.")

        return characters

    def extract_world_name(self, outline):
        """Extract consistent world name from the story outline"""
        if not outline: return ""
        # Generic pattern to find world names without hardcoding specific ones
        # Added more patterns and made them case-insensitive where appropriate
        patterns = [
            r"Neo-[A-Za-z]+",
            r"[A-Z][a-z]+land", r"(?i)\bthe\s+Kingdom\s+of\s+([A-Z][a-z]+)\b",
            r"(?i)\bthe\s+([A-Z][a-z]+)\s+Kingdom\b",
            r"(?i)\bthe\s+([A-Z][a-z]+)\s+Empire\b",
            r"(?i)\bthe\s+Realm\s+of\s+([A-Z][a-z]+)\b",
            r"(?i)\bthe\s+([A-Z][a-z]+)\s+Realm\b",
            r"(?i)\b([A-Z][a-z]+)\s+World\b",
            r"(?i)\b([A-Z][a-z]+)\s+City\b",
            r"(?i)\bCity\s+of\s+([A-Z][a-z]+)\b",
            r"(?i)\bPlanet\s+([A-Z][a-z]+)\b", # Added Planet
            r"(?i)\bthe\s+([A-Z][a-z]+)\s+Federation\b", # Added Federation
            r"(?i)\bthe\s+([A-Z][a-z]+)\s+Republic\b", # Added Republic
        ]

        found_names = []
        for pattern in patterns:
            # Find all matches for the pattern
            matches = re.findall(pattern, outline)
            # Add non-empty matches to the list
            found_names.extend([m.strip() for m in matches if isinstance(m, str) and m.strip()])
            # Handle patterns that return tuples (like those with capture groups)
            if matches and isinstance(matches[0], tuple):
                 found_names.extend([item.strip() for tpl in matches for item in tpl if isinstance(item, str) and item.strip()])


        if found_names:
            # Count occurrences of each name
            counts = {}
            for name in found_names:
                counts[name] = counts.get(name, 0) + 1
            # Return the most frequent name
            return max(counts, key=counts.get)

        # If no specific world name is found, let the LLM generate one
        print("No specific world name pattern matched in the outline.")
        return ""

    def create_story_outline(self):
        """Generate a high-level outline for the entire story with improved structure"""
        # Updated system prompt to include genre
        system_prompt = f"""You are a professional novelist and editor specializing in the {self.genre} genre.
You excel at creating compelling story outlines that maintain coherence and proper narrative structure.
Pay special attention to character development, consistent world-building, logical plot progression, and genre conventions for {self.genre}."""

        # Updated main prompt to include genre
        prompt = f"""Based on the following story premise, create a detailed story outline for a {self.num_chapters}-chapter {self.genre} book.

Story premise (Genre: {self.genre}): {self.story_premise}

For each chapter, provide:
1. A chapter title (evocative and relevant to the {self.genre} genre)
2. Key plot events (3-4 bullet points with specific details)
3. Character development points (which characters appear and how they develop)
4. Setting/location details (be specific, consistent, and fitting for {self.genre})

Also create a section titled "WORLD BUILDING" with:
1. The name of the main city/setting (create a unique, memorable name fitting the {self.genre} genre)
2. Key locations that will appear multiple times
3. Important technology, magic systems, or cultural elements relevant to {self.genre}

Then create a section titled "CHARACTERS" with:
1. Main protagonist (name, detailed description, motivation, arc)
2. Main antagonist (name, detailed description, motivation)
3. Supporting characters (name, role, connection to protagonist)

Also create a section titled "RECURRING MOTIFS" with 3-5 symbols, objects, or phrases that will recur throughout the story to provide thematic continuity, relevant to the {self.genre} genre.

The outline should have a clear beginning (chapters 1-2), middle (chapters 3 to {self.num_chapters-2}), and end (final {min(2, self.num_chapters-1)} chapters).
Ensure proper rising action, climax, and resolution structure appropriate for the {self.genre} genre.

For the climax chapters (chapters {self.num_chapters-2} and {self.num_chapters-1}), provide extra detail about:
1. How the final confrontation unfolds
2. What's at stake for each character
3. Specific steps in the resolution

Make sure all character names, settings, and plot elements remain 100% consistent throughout.
"""
        print("Generating detailed story outline...")
        self.story_outline = self.generate_text(prompt, system_prompt)
        if not self.story_outline:
             print("Error: Failed to generate story outline. Cannot proceed.")
             return # Stop if outline generation fails

        # Extract character information
        # Updated system prompt for character extraction
        char_system_prompt = f"You are a literary analyst specializing in character tracking within the {self.genre} genre."
        char_prompt = f"""Based on this story outline for a {self.genre} novel, create a detailed character guide:

{self.story_outline}

For EACH character, format as:
CHARACTER NAME: Brief description, role in story, key personality traits, motivation, background (ensure consistency with the {self.genre} genre)

Include EVERY character mentioned in the outline, even minor ones. Ensure names are spelled consistently.
"""
        print("Creating detailed character profiles...")
        character_text = self.generate_text(char_prompt, char_system_prompt)
        self.characters = self.extract_characters(character_text)
        if not self.characters:
            print("Warning: Character extraction yielded no results. Proceeding without pre-defined characters.")


        # Extract world name for consistency
        self.world_name = self.extract_world_name(self.story_outline)

        # If no world name was found, ask the LLM to create one
        if not self.world_name:
            print("Attempting to generate world name...")
            # Updated system prompt for world name generation
            world_system_prompt = f"You are a creative world-builder specializing in the {self.genre} genre."
            world_prompt = f"""Based on this story outline for a {self.genre} novel, create a unique and memorable name for the main world/city/setting:

{self.story_outline}

The name should be a single term (or a short phrase like 'City of X'), creative, and fitting the tone and genre ({self.genre}) of the story.
Reply with ONLY the world name, nothing else.
"""
            generated_world_name = self.generate_text(world_prompt, world_system_prompt)
            if generated_world_name:
                 # Clean up potential extra text from the response
                 self.world_name = generated_world_name.strip().split('\n')[0]
                 print(f"Generated world name: {self.world_name}")
            else:
                 print("Warning: Could not generate world name. Will proceed without one.")
                 self.world_name = "The Setting" # Fallback

        # Extract recurring motifs
        # Updated system prompt for motif extraction
        motif_system_prompt = f"You are a literary analyst identifying thematic elements in {self.genre} fiction."
        motif_prompt = f"""Based on this story outline for a {self.genre} novel, identify 3-5 recurring motifs, symbols, or objects that appear throughout the story:

{self.story_outline}

Format as a simple list of items, one per line.
These should be concrete objects, symbols, or phrases relevant to the {self.genre} genre that can recur throughout chapters.
"""
        print("Identifying recurring motifs...")
        motifs_text = self.generate_text(motif_prompt, motif_system_prompt)
        if motifs_text:
            # Filter empty lines and clean up potential bullet points/numbering
            self.recurring_motifs = [re.sub(r"^\s*[-\*\d]+\.?\s*", "", motif).strip()
                                     for motif in motifs_text.strip().split('\n')
                                     if motif.strip()]
        if self.recurring_motifs:
            print(f"Identified motifs: {', '.join(self.recurring_motifs)}")
        else:
            print("Warning: No recurring motifs identified.")


        # Create detailed chapter-by-chapter plan
        # Updated system prompt for chapter plan
        plan_system_prompt = f"You are a meticulous story planner specializing in {self.genre} narratives."
        chapter_plan_prompt = f"""Based on the story outline for this {self.genre} novel, create a VERY detailed chapter-by-chapter plan.

STORY OUTLINE: {self.story_outline}

GENRE: {self.genre}

For EACH chapter (1 through {self.num_chapters}), provide:
1. Chapter title (consistent with outline and genre)
2. Chapter summary (250-300 words, detailing key events and character interactions)
3. Scene breakdown (list each scene with location, characters present, and key action/dialogue points)
4. Character development in this chapter (specific changes or revelations for characters)
5. Plot advancement in this chapter (how the main plot moves forward)
6. Timeline indicators (time of day, date, or how much time has passed since previous chapter)
7. Emotional tone and tension level (1-10) at the end of the chapter
8. How this chapter connects to the next (create a narrative bridge or hook)

Be extremely specific and detailed, adhering to the {self.genre} conventions. This plan will be used to ensure narrative consistency.
"""
        print("Creating detailed chapter plan...")
        self.chapter_plan = self.generate_text(chapter_plan_prompt, plan_system_prompt)
        if not self.chapter_plan:
             print("Error: Failed to generate chapter plan. Book generation quality may be affected.")


    def create_chapter_summary(self, chapter_num, chapter_content):
        """Create a detailed summary of a chapter after it's written"""
        if not chapter_content: return None # Handle empty content
        # Updated system prompt
        system_prompt = f"""You are a literary analyst specializing in narrative structure and continuity for the {self.genre} genre.
Create comprehensive, detailed summaries that capture all key elements."""
        prompt = f"""Create a detailed summary of the following chapter content.
This is Chapter {chapter_num} of a {self.num_chapters}-chapter {self.genre} book.

Include in your summary:
1. All key plot developments relevant to the main story and subplots
2. Character appearances, actions, dialogue highlights, and development
3. Setting details and atmosphere established
4. Important dialogue or revelations
5. How this chapter connects to previous chapters and sets up future events
6. Emotional tone at the beginning and end of the chapter
7. Any introduction of new significant elements (characters, items, locations)

CHAPTER CONTENT:
{chapter_content}

Your summary should be comprehensive enough that another writer could use it to maintain perfect continuity for this {self.genre} story.
"""
        summary = self.generate_text(prompt, system_prompt)
        if summary:
            self.chapter_summaries[chapter_num] = summary
        else:
            print(f"Warning: Failed to generate summary for Chapter {chapter_num}.")
            self.chapter_summaries[chapter_num] = "[Summary generation failed]"
        return summary

    def update_character_tracking(self, chapter_num, chapter_content):
        """Update character tracking data based on a chapter's content"""
        if not chapter_content or not self.characters: return # Need content and characters
        # Updated system prompt
        system_prompt = f"""You are a narrative continuity expert specializing in tracking character development in {self.genre} stories.
Extract precise information about characters from text."""
        character_names = list(self.characters.keys())
        characters_str = ", ".join(character_names)
        prompt = f"""Based on the following chapter content from a {self.genre} novel, track the development of all characters mentioned.

CHAPTER CONTENT:
{chapter_content}

CHARACTERS TO TRACK: {characters_str}

For each character that APPEARS or is SIGNIFICANTLY MENTIONED in this chapter, provide updates on:
1. Current status (e.g., alive, injured, captured, location if changed significantly)
2. Key actions or decisions made in this chapter
3. Notable development or changes in motivation/personality
4. Significant interactions or changes in relationships with other characters
5. Emotional state at the end of the chapter within this context

Format as:
CHARACTER NAME: status|actions/decisions|development|relationships|emotional_state

Only include characters who actually appear or are directly impacted in this chapter. Be concise but informative.
If a character appears but has no significant changes in these areas, note that briefly (e.g., "CHARACTER NAME: present|no major changes|...").
"""
        character_updates = self.generate_text(prompt, system_prompt)

        if not character_updates:
            print(f"Warning: No character updates received or generated for Chapter {chapter_num}.")
            return

        # Parse and update character data
        # Improved pattern to handle potential missing fields and variations
        update_pattern = r"([A-Z][A-Za-z\s'-]+):\s*([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|([^\n]*)"
        matches = re.findall(update_pattern, character_updates)

        updated_chars_in_chapter = set()

        for match in matches:
            name = match[0].strip()
            if name in self.characters:
                updated_chars_in_chapter.add(name)
                # Update fields only if they provide new information
                status = match[1].strip()
                actions = match[2].strip()
                development = match[3].strip()
                relationships = match[4].strip()
                emotion = match[5].strip()

                if status and "no major changes" not in status.lower():
                    self.characters[name]["status"] = status
                if development and "no major changes" not in development.lower():
                    self.characters[name]["development"].append({
                        "chapter": chapter_num,
                        "development": development,
                        "actions": actions # Store actions too
                    })
                if relationships:
                    # Basic update - just note the interaction mentioned
                    self.characters[name]["relationships"][f"interaction_ch_{chapter_num}"] = relationships

                # Update location and emotional state if provided
                if status and "/" in status: # Check if location is embedded in status
                    parts = status.split('/')
                    self.characters[name]["status"] = parts[0].strip()
                    self.characters[name]["location"] = parts[1].strip()
                elif "location" in status.lower(): # Check if location is explicitly mentioned
                     self.characters[name]["location"] = status # Crude, needs refinement based on LLM output format

                if emotion:
                    self.characters[name]["emotional_state"] = emotion

                # Record first appearance if not already set
                if self.characters[name]["first_appearance"] == 0:
                    self.characters[name]["first_appearance"] = chapter_num
            else:
                 print(f"Warning: Received update for unknown character '{name}' in Chapter {chapter_num}.")

        # Note characters present but not explicitly updated (if any were extracted)
        if updated_chars_in_chapter:
             present_chars = set(re.findall(r'\b(' + '|'.join(re.escape(c) for c in character_names) + r')\b', chapter_content))
             for char_name in present_chars:
                 if char_name in self.characters and char_name not in updated_chars_in_chapter:
                     if self.characters[char_name]["first_appearance"] == 0:
                         self.characters[char_name]["first_appearance"] = chapter_num
                     # Optionally add a note that they were present but not updated
                     # self.characters[char_name]["development"].append({"chapter": chapter_num, "development": "Present, no specific update extracted"})


    def update_timeline(self, chapter_num, chapter_content):
        """Extract and update timeline information for chapter"""
        if not chapter_content: return None
        system_prompt = f"""You are a literary analyst specializing in temporal structure in {self.genre} narratives."""

        prompt = f"""Based on the following chapter content from a {self.genre} novel, determine:
        1. How much time seems to have passed DURING this chapter (e.g., a few hours, one day, indeterminate)
        2. What is the approximate time of day or date at the END of the chapter (e.g., evening, night, Tuesday, Day 5)
        3. Any specific time markers mentioned within the chapter (e.g., "at dawn", "three hours later", "midnight")

        CHAPTER CONTENT:
        {chapter_content}

        Reply only with the time information in this format, keeping it concise:
        TIME_ELAPSED: [estimated time passed during chapter]
        END_TIME: [estimated time/day at chapter end]
        TIME_MARKERS: [list any specific markers mentioned, separated by commas]
        """

        time_info = self.generate_text(prompt, system_prompt)
        if time_info:
            self.timeline[chapter_num] = time_info.strip()
        else:
            print(f"Warning: Failed to generate timeline info for Chapter {chapter_num}.")
            self.timeline[chapter_num] = "TIME_ELAPSED: Unknown\nEND_TIME: Unknown\nTIME_MARKERS: None"
        return self.timeline[chapter_num]


    def track_emotional_arc(self, chapter_num, chapter_content):
        """Track emotional tone and tension at the end of the chapter"""
        if not chapter_content: return None
        system_prompt = f"""You are a literary analyst specializing in emotional arcs in {self.genre} storytelling."""

        prompt = f"""Analyze the emotional tone at the end of this chapter from a {self.genre} novel:

        CHAPTER CONTENT (focus on the ending):
        {chapter_content[-1500:]} # Analyze last part for ending tone

        Determine:
        1. The primary emotion felt by the reader/protagonist at the chapter's end (e.g., tension, relief, hope, despair, mystery, anticipation)
        2. The level of narrative tension (1-10 scale, where 1 is low and 10 is high cliffhanger)
        3. The primary unresolved question or conflict driving interest into the next chapter

        Reply in this format:
        EMOTION: [primary emotion]
        TENSION: [level 1-10]
        UNRESOLVED: [main unresolved question or conflict]
        """

        emotional_status = self.generate_text(prompt, system_prompt)
        if emotional_status:
            self.emotional_arc[chapter_num] = emotional_status.strip()
        else:
            print(f"Warning: Failed to generate emotional arc info for Chapter {chapter_num}.")
            self.emotional_arc[chapter_num] = "EMOTION: Unknown\nTENSION: Unknown\nUNRESOLVED: Unknown"
        return self.emotional_arc[chapter_num]

    def create_chapter_transition(self, chapter_num, chapter_content):
        """Create a transition from current chapter to the next"""
        if chapter_num >= self.num_chapters or not chapter_content:
            return ""  # No transition needed for the last chapter or if content is missing

        # Updated system prompt
        system_prompt = f"""You are a master storyteller specializing in creating suspenseful
        chapter endings and smooth transitions between chapters, particularly for the {self.genre} genre."""

        # Extract relevant part of chapter plan for the next chapter
        plan_extract_prompt = f"""From this detailed chapter plan for a {self.genre} novel, extract ONLY the plan
        for Chapter {chapter_num + 1}:

        {self.chapter_plan}

        Include ONLY Chapter {chapter_num + 1}'s detailed plan (title, summary, scenes, etc.).
        """
        next_chapter_plan = self.generate_text(plan_extract_prompt)
        if not next_chapter_plan:
             print(f"Warning: Could not extract plan for Chapter {chapter_num + 1} for transition generation.")
             next_chapter_plan = "[Next chapter plan not available]"


        # Get emotional status
        emotional_status = self.emotional_arc.get(chapter_num, "EMOTION: Unknown\nTENSION: Unknown\nUNRESOLVED: Unknown")

        # Get timeline info
        timeline_info = self.timeline.get(chapter_num, "END_TIME: Unknown")

        # Choose a recurring motif to include
        chosen_motif = ""
        if self.recurring_motifs:
            # Cycle through motifs based on chapter number
            motif_index = (chapter_num -1) % len(self.recurring_motifs) # Use chapter_num - 1 for 0-based index
            chosen_motif = self.recurring_motifs[motif_index]


        prompt = f"""Create a compelling transition paragraph (or two short ones) to serve as the ending of Chapter {chapter_num}
        of this {self.genre} novel and create anticipation for Chapter {chapter_num + 1}.

        CURRENT CHAPTER CONTENT (last ~1000 characters):
        {chapter_content[-1000:]}

        NEXT CHAPTER ({chapter_num + 1}) PLAN:
        {next_chapter_plan}

        EMOTIONAL STATE AT CURRENT CHAPTER END:
        {emotional_status}

        TIMELINE INFORMATION AT CURRENT CHAPTER END:
        {timeline_info}

        RECURRING MOTIF TO INCLUDE IF POSSIBLE:
        {chosen_motif if chosen_motif else "N/A"}

        The transition should:
        1. Conclude the current chapter with a hook, an unanswered question, or lingering tension appropriate for {self.genre}.
        2. Create anticipation for what comes next based on the next chapter's plan.
        3. Avoid completely resolving all tension from the current chapter.
        4. Link to the theme or events of the upcoming chapter without being too explicit.
        5. Maintain the emotional tone but hint at potential shifts in the next chapter.
        6. Include a subtle, natural reference to the recurring motif ({chosen_motif}) if it fits the scene.

        Generate ONLY the 1-2 transition paragraphs. These will be the FINAL paragraphs of Chapter {chapter_num}.
        Do not add any extra commentary.
        """

        transition = self.generate_text(prompt, system_prompt)
        if transition:
            self.transitions[chapter_num] = transition.strip()
            return self.transitions[chapter_num]
        else:
            print(f"Warning: Failed to generate transition for Chapter {chapter_num}.")
            # Provide a generic fallback transition
            fallback_transition = f"The events of the day left much unresolved. As {self.world_name} settled into uncertainty, the stage was set for what was to come."
            self.transitions[chapter_num] = fallback_transition
            return fallback_transition


    def create_next_chapter_opener(self, chapter_num):
        """Create a strong opening for the next chapter that connects to the previous one"""
        if chapter_num <= 1:
            return ""  # First chapter doesn't need a special opener

        # Updated system prompt
        system_prompt = f"""You are a master storyteller specializing in creating
        engaging chapter openings that connect smoothly to previous events in {self.genre} narratives."""

        # Get information about the previous chapter
        prev_chapter_summary = self.chapter_summaries.get(chapter_num - 1, "[Previous summary unavailable]")
        # Use the generated transition as the end of the previous chapter
        prev_transition = self.transitions.get(chapter_num - 1, "[Previous transition unavailable]")
        prev_emotional_status = self.emotional_arc.get(chapter_num - 1, "EMOTION: Unknown\nTENSION: Unknown\nUNRESOLVED: Unknown")
        prev_timeline = self.timeline.get(chapter_num - 1, "END_TIME: Unknown")

        # Extract timeline info to know how much time has passed
        prev_end_time = "Unknown"
        if prev_timeline:
             # Try to extract END_TIME more reliably
             time_match = re.search(r"END_TIME:\s*([^\n]+)", prev_timeline)
             if time_match:
                 prev_end_time = time_match.group(1).strip()


        # Extract relevant part of chapter plan for this chapter
        plan_extract_prompt = f"""From this detailed chapter plan for a {self.genre} novel, extract ONLY the plan
        for Chapter {chapter_num}:

        {self.chapter_plan}

        Include ONLY Chapter {chapter_num}'s detailed plan (title, summary, scenes, etc.).
        """
        this_chapter_plan = self.generate_text(plan_extract_prompt)
        if not this_chapter_plan:
             print(f"Warning: Could not extract plan for Chapter {chapter_num} for opener generation.")
             this_chapter_plan = "[This chapter plan not available]"


        prompt = f"""Create a compelling opening paragraph (3-5 sentences) for Chapter {chapter_num} of this {self.genre} novel,
        connecting seamlessly with the end of Chapter {chapter_num - 1}.

        PREVIOUS CHAPTER ({chapter_num - 1}) SUMMARY (for context):
        {prev_chapter_summary}

        END OF PREVIOUS CHAPTER ({chapter_num - 1}) (The actual transition text):
        {prev_transition}

        EMOTIONAL STATE AT END OF PREVIOUS CHAPTER:
        {prev_emotional_status}

        TIME AT END OF PREVIOUS CHAPTER:
        {prev_end_time}

        THIS CHAPTER'S ({chapter_num}) PLAN:
        {this_chapter_plan}

        The opening paragraph should:
        1. Establish a clear time connection to the end of the previous chapter based on the timeline ({prev_end_time}) and this chapter's plan (e.g., "The next morning...", "Three days later...", "Moments after the confrontation...").
        2. Provide a smooth transition that builds on the tension, questions, or mood from the previous chapter's ending ({prev_transition}).
        3. Immediately orient readers in the new scene (location, characters present, initial action/observation) as per this chapter's plan.
        4. Avoid summarizing or repeating information from the previous chapter.
        5. Set the emotional tone for this chapter, aligning with the {self.genre} genre.

        Generate ONLY the single strong opening paragraph (3-5 sentences).
        Do not add any extra commentary or the chapter title.
        """

        opener = self.generate_text(prompt, system_prompt)
        if opener:
            return opener.strip()
        else:
            print(f"Warning: Failed to generate opener for Chapter {chapter_num}.")
            # Generic fallback opener
            return f"Time passed in {self.world_name}. Following the recent events, the atmosphere was charged with anticipation."


    def validate_chapter_consistency(self, chapter_num, chapter_content):
        """Check chapter for consistency issues"""
        if not chapter_content: return "CONSISTENT" # Cannot validate empty content
        # Updated system prompt
        system_prompt = f"""You are a meticulous literary editor specializing in narrative consistency for the {self.genre} genre.
Your job is to identify and flag any inconsistencies in a narrative based ONLY on the provided context."""

        # Create context for consistency check
        # Limit context to prevent excessive prompt length, focus on recent chapters
        context_chapter_limit = 5
        start_chapter = max(1, chapter_num - context_chapter_limit)
        previous_summaries = ""
        for i in range(start_chapter, chapter_num):
            if i in self.chapter_summaries:
                # Add chapter number for clarity
                previous_summaries += f"Chapter {i} Summary:\n{self.chapter_summaries[i]}\n\n"

        # Character status based on tracking up to the *previous* chapter
        character_status_list = []
        for name, data in self.characters.items():
            # Include characters who have appeared or are defined in the initial outline
            if data.get("first_appearance", 0) < chapter_num or data.get("first_appearance", 0) == 0 : # Include if appeared before this chapter or defined initially
                status = data.get("status", "unknown")
                desc = data.get("description", "N/A")
                # Get last known location and emotion if available
                location = data.get("location", "unknown")
                emotion = data.get("emotional_state", "unknown")
                # Summarize recent development
                dev_summary = "None recorded"
                if data.get("development"):
                     # Get the last recorded development entry
                     last_dev = data["development"][-1].get("development", "N/A") if data["development"] else "N/A"
                     dev_summary = f"Last noted: {last_dev} (Ch {data['development'][-1]['chapter']})" if data["development"] else "None recorded"

                character_status_list.append(f"- {name}: Status({status}), Loc({location}), Emotion({emotion}). {dev_summary}. Desc: {desc[:100]}...") # Truncate desc

        character_status_str = "\n".join(character_status_list) if character_status_list else "No character status available."


        # Add timeline information from recent chapters
        timeline_info = ""
        for i in range(start_chapter, chapter_num):
            if i in self.timeline:
                timeline_info += f"Chapter {i} End Timeline: {self.timeline[i]}\n"
        timeline_info = timeline_info if timeline_info else "No recent timeline info available."

        # Add relevant part of the chapter plan for context
        plan_extract_prompt = f"""From this detailed chapter plan for a {self.genre} novel, extract ONLY the plan
        for Chapter {chapter_num}:

        {self.chapter_plan}

        Include ONLY Chapter {chapter_num}'s detailed plan (title, summary, scenes, etc.).
        """
        this_chapter_plan = self.generate_text(plan_extract_prompt)
        if not this_chapter_plan:
             this_chapter_plan = "[Chapter plan not available for validation]"


        prompt = f"""Analyze this chapter for consistency issues compared ONLY to the provided context (previous summaries, character status, timeline, and chapter plan).

STORY PREMISE (Genre: {self.genre}): {self.story_premise}

WORLD NAME: {self.world_name}

RECENT PREVIOUS CHAPTER SUMMARIES (Chapters {start_chapter}-{chapter_num-1}):
{previous_summaries if previous_summaries else "No previous summaries provided."}

CHARACTER STATUS BEFORE THIS CHAPTER:
{character_status_str}

RECENT TIMELINE INFORMATION:
{timeline_info}

THIS CHAPTER'S ({chapter_num}) INTENDED PLAN:
{this_chapter_plan}

CURRENT CHAPTER ({chapter_num}) CONTENT TO VALIDATE:
{chapter_content}

Identify ANY specific inconsistencies between the CURRENT CHAPTER CONTENT and the PROVIDED CONTEXT related to:
1. Character details (names, status, location, established personality/abilities, relationships)
2. Setting/location names or descriptions conflicting with {self.world_name} or previous descriptions.
3. Timeline contradictions (sequence of events, time elapsed, time of day).
4. Plot contradictions (events contradicting previous summaries or character status).
5. Contradictions with the intended chapter plan provided above.
6. Incorrect use or appearance of recurring motifs based on context.
7. Genre inconsistencies (elements that strongly clash with {self.genre} conventions).

List ONLY specific inconsistencies found. Be precise.
If no inconsistencies are found based *solely* on the provided context, respond ONLY with the word "CONSISTENT".
"""
        consistency_check = self.generate_text(prompt, system_prompt)

        # Post-process the check result
        if not consistency_check:
             print(f"Warning: Consistency check for Chapter {chapter_num} failed to generate a response.")
             return "CHECK_FAILED" # Indicate failure
        elif "CONSISTENT" in consistency_check.upper() and len(consistency_check) < 20: # Check if it's likely just the word "CONSISTENT"
             return "CONSISTENT"
        else:
             # Return the list of issues
             return consistency_check


    def fix_chapter_inconsistencies(self, chapter_num, chapter_content, issues):
        """Fix identified consistency issues in a chapter"""
        if not chapter_content: return None # Cannot fix empty content
        # Updated system prompt
        system_prompt = f"""You are a professional novelist and meticulous editor specializing in the {self.genre} genre.
You excel at seamlessly fixing narrative inconsistencies while preserving the core narrative, tone, and character voices."""

        # Create context for the fix - similar context as validation
        context_chapter_limit = 5
        start_chapter = max(1, chapter_num - context_chapter_limit)
        previous_summaries = ""
        for i in range(start_chapter, chapter_num):
            if i in self.chapter_summaries:
                previous_summaries += f"Chapter {i} Summary:\n{self.chapter_summaries[i]}\n\n"

        character_status_list = []
        for name, data in self.characters.items():
             if data.get("first_appearance", 0) < chapter_num or data.get("first_appearance", 0) == 0 :
                status = data.get("status", "unknown")
                desc = data.get("description", "N/A")
                location = data.get("location", "unknown")
                emotion = data.get("emotional_state", "unknown")
                dev_summary = "None recorded"
                if data.get("development"):
                     last_dev = data["development"][-1].get("development", "N/A") if data["development"] else "N/A"
                     dev_summary = f"Last noted: {last_dev} (Ch {data['development'][-1]['chapter']})" if data["development"] else "None recorded"
                character_status_list.append(f"- {name}: Status({status}), Loc({location}), Emotion({emotion}). {dev_summary}. Desc: {desc[:100]}...")

        character_status_str = "\n".join(character_status_list) if character_status_list else "No character status available."

        timeline_info = ""
        for i in range(start_chapter, chapter_num):
            if i in self.timeline:
                timeline_info += f"Chapter {i} End Timeline: {self.timeline[i]}\n"
        timeline_info = timeline_info if timeline_info else "No recent timeline info available."

        plan_extract_prompt = f"""From this detailed chapter plan for a {self.genre} novel, extract ONLY the plan
        for Chapter {chapter_num}:

        {self.chapter_plan}

        Include ONLY Chapter {chapter_num}'s detailed plan (title, summary, scenes, etc.).
        """
        this_chapter_plan = self.generate_text(plan_extract_prompt)
        if not this_chapter_plan:
             this_chapter_plan = "[Chapter plan not available for fix context]"


        prompt = f"""Rewrite Chapter {chapter_num} of this {self.genre} novel to fix ALL the identified consistency issues, using the provided context. Maintain the original chapter's core plot, character interactions, and tone as much as possible.

STORY PREMISE (Genre: {self.genre}): {self.story_premise}

WORLD NAME: {self.world_name} (use this name consistently)

RECENT PREVIOUS CHAPTER SUMMARIES (Chapters {start_chapter}-{chapter_num-1}):
{previous_summaries if previous_summaries else "No previous summaries provided."}

CHARACTER STATUS BEFORE THIS CHAPTER:
{character_status_str}

RECENT TIMELINE INFORMATION:
{timeline_info}

THIS CHAPTER'S ({chapter_num}) INTENDED PLAN:
{this_chapter_plan}

ORIGINAL CHAPTER ({chapter_num}) CONTENT (with inconsistencies):
{chapter_content}

CONSISTENCY ISSUES TO FIX:
{issues}

Guidelines for fixing:
1. Directly address and correct every issue listed in "CONSISTENCY ISSUES TO FIX".
2. Maintain the essential plot progression and key events of the original chapter.
3. Ensure all character names, statuses, locations, abilities, and personalities align with the "CHARACTER STATUS BEFORE THIS CHAPTER" and previous summaries.
4. Use the established world name ({self.world_name}) and setting details consistently.
5. Ensure the timeline (sequence, elapsed time, time of day) flows logically from the "RECENT TIMELINE INFORMATION".
6. Align the chapter's events more closely with "THIS CHAPTER'S INTENDED PLAN" where inconsistencies were noted.
7. Preserve the original writing style and {self.genre} tone unless the inconsistency demands a change.
8. Ensure smooth transitions between scenes within the rewritten chapter.

Rewrite the COMPLETE chapter, starting with the chapter title (e.g., "## Chapter {chapter_num}: [Original or Revised Title]"). Do not include any commentary before or after the rewritten chapter content.
"""
        print(f"Attempting to fix Chapter {chapter_num}...")
        fixed_chapter = self.generate_text(prompt, system_prompt)
        # Basic validation of the fix
        if fixed_chapter and f"Chapter {chapter_num}" in fixed_chapter[:100]: # Check if it seems like a chapter
             # Optional: Run validation again on the fixed chapter? Could be slow.
             # print(f"Re-validating fixed Chapter {chapter_num}...")
             # re_check = self.validate_chapter_consistency(chapter_num, fixed_chapter) # Need to update context before re-validating
             # if re_check == "CONSISTENT":
             #     print("Fix successful based on re-validation.")
             # else:
             #     print(f"Warning: Fixed chapter still shows issues on re-validation: {re_check}")
             return fixed_chapter.strip()
        else:
             print(f"Error: Fixing Chapter {chapter_num} failed or produced invalid output.")
             if fixed_chapter: print(f"Output received:\n{fixed_chapter[:200]}...") # Log snippet if available
             return None # Indicate failure


    def generate_chapter(self, chapter_num):
        """Generate a single chapter with enhanced context awareness and consistency checks"""
        # Updated system prompt
        system_prompt = f"""You are a celebrated novelist known for writing engaging, coherent chapters
in the {self.genre} genre. Your chapters have clear narrative structure, natural flow, strong character development,
and maintain perfect consistency with previously established elements."""

        # Build context from previous chapters (limit context size)
        context_chapter_limit = 5
        start_chapter = max(1, chapter_num - context_chapter_limit)
        context_summaries = []
        for i in range(start_chapter, chapter_num):
            if i in self.chapter_summaries:
                # Add chapter number for clarity
                context_summaries.append(f"Chapter {i} Summary:\n{self.chapter_summaries[i]}")
        context = "\n\n".join(context_summaries) if context_summaries else "This is the first chapter."


        # Create character context based on tracking up to the *previous* chapter
        character_context_list = []
        for name, data in self.characters.items():
             if data.get("first_appearance", 0) < chapter_num or data.get("first_appearance", 0) == 0 :
                status = data.get("status", "unknown")
                desc = data.get("description", "N/A")
                location = data.get("location", "unknown")
                emotion = data.get("emotional_state", "unknown")
                dev_summary = "None recorded"
                if data.get("development"):
                     last_dev = data["development"][-1].get("development", "N/A") if data["development"] else "N/A"
                     dev_summary = f"Last noted: {last_dev} (Ch {data['development'][-1]['chapter']})" if data["development"] else "None recorded"
                # Provide concise context for the LLM
                character_context_list.append(f"- {name}: Status({status}), Loc({location}), Emotion({emotion}). {dev_summary}. Desc: {desc[:100]}...")

        characters_in_chapter_str = "\n".join(character_context_list) if character_context_list else "No character context available (treat as first appearance if needed)."


        # Add timeline information from the *previous* chapter's end
        timeline_context = "No previous timeline info available."
        if chapter_num > 1 and (chapter_num - 1) in self.timeline:
            timeline_context = f"End of previous chapter ({chapter_num-1}) timeline: {self.timeline[chapter_num - 1]}"

        # Add emotional arc information from the *previous* chapter's end
        emotional_context = "No previous emotional context available."
        if chapter_num > 1 and (chapter_num - 1) in self.emotional_arc:
            emotional_context = f"End of previous chapter ({chapter_num-1}) emotional state: {self.emotional_arc[chapter_num - 1]}"

        # Create chapter opener for chapters after the first
        chapter_opener = ""
        if chapter_num > 1:
            print(f"Creating opener for Chapter {chapter_num}...")
            chapter_opener = self.create_next_chapter_opener(chapter_num)
            if not chapter_opener:
                 print(f"Warning: Failed to generate specific opener for Chapter {chapter_num}.")
                 chapter_opener = "[Start chapter directly]" # Fallback instruction


        # Extract relevant part of chapter plan for this chapter
        plan_extract_prompt = f"""From this detailed chapter plan for a {self.genre} novel, extract ONLY the plan
        for Chapter {chapter_num}:

        {self.chapter_plan}

        Include ONLY Chapter {chapter_num}'s detailed plan (title, summary, scenes, character dev, plot adv, etc.).
        """
        this_chapter_plan = self.generate_text(plan_extract_prompt)
        if not this_chapter_plan:
             print(f"Critical Warning: Could not extract plan for Chapter {chapter_num}. Generation quality will be severely impacted.")
             this_chapter_plan = f"[PLAN MISSING FOR CHAPTER {chapter_num}] - Improvise based on outline and previous summaries."


        # Choose a recurring motif to include
        motif_instruction = ""
        if self.recurring_motifs:
            motif_index = (chapter_num - 1) % len(self.recurring_motifs)
            chosen_motif = self.recurring_motifs[motif_index]
            motif_instruction = f"Include the recurring motif '{chosen_motif}' somewhere in this chapter in a natural and meaningful way, relevant to the {self.genre} context."


        # Updated main prompt to include genre
        prompt = f"""Write Chapter {chapter_num} of a {self.genre} novel based on the following guidelines and context:

STORY PREMISE (Genre: {self.genre}): {self.story_premise}

WORLD NAME: {self.world_name} (use this name consistently throughout)

OVERALL STORY OUTLINE (Reference):
{self.story_outline[:1500]}...

THIS CHAPTER'S ({chapter_num}) DETAILED PLAN:
{this_chapter_plan}

CHARACTER CONTEXT (Status before this chapter):
{characters_in_chapter_str}

RECENT PREVIOUS CHAPTER SUMMARIES (Chapters {start_chapter}-{chapter_num-1}):
{context}

TIMELINE CONTEXT:
{timeline_context}

EMOTIONAL CONTEXT FROM PREVIOUS CHAPTER:
{emotional_context}

SUGGESTED CHAPTER OPENING (Use this to start the chapter unless it's '[Start chapter directly]'):
{chapter_opener}

{motif_instruction}

GUIDELINES FOR WRITING CHAPTER {chapter_num}:
- Write a complete, engaging chapter following THIS CHAPTER'S DETAILED PLAN.
- Adhere strictly to the {self.genre} genre conventions in tone, style, and content.
- Target chapter length: approximately 2500-3500 words (adjust as needed for plot).
- Include vivid descriptions, realistic dialogue, internal thoughts, and meaningful character interactions.
- Maintain 100% consistency with CHARACTER CONTEXT, WORLD NAME ({self.world_name}), TIMELINE CONTEXT, and PREVIOUS CHAPTER SUMMARIES.
- This is chapter {chapter_num} of {self.num_chapters}. Ensure it advances the plot according to the plan and overall outline.
- Do NOT re-explain established world-building elements unless necessary for context.
- Ensure the chapter has its own mini-arc (setup, rising action, climax/turning point, resolution/hook).
- Start with the chapter title using the format: "## Chapter {chapter_num}: [Chapter Title from Plan]"
- ONLY introduce new characters if specified in the plan or absolutely essential, providing proper context.
- If using the SUGGESTED CHAPTER OPENING, integrate it seamlessly as the first paragraph(s).
- End the chapter with a compelling hook, lingering question, or heightened tension suitable for {self.genre}, leading towards the next chapter's plan.

Format the chapter with clear paragraph breaks and standard dialogue formatting (e.g., using quotation marks).
Begin DIRECTLY with the chapter title (## Chapter X: Title). Do not add any introductory text.
"""
        print(f"Generating Chapter {chapter_num}...")
        chapter_content = self.generate_text(prompt, system_prompt)

        if not chapter_content or not re.match(rf"##\s*Chapter\s*{chapter_num}", chapter_content.strip()):
            print(f"Error: Failed to generate valid content for Chapter {chapter_num}.")
            # Try one regeneration attempt with a simpler prompt? (Optional)
            return f"## Chapter {chapter_num}: Generation Failed\n\n[Content generation failed for this chapter]"


        # --- Post-Generation Processing ---
        max_fix_attempts = 1 # Limit fix attempts to prevent loops
        for attempt in range(max_fix_attempts + 1):
             # Check for consistency issues
             print(f"Validating Chapter {chapter_num} (Attempt {attempt+1})...")
             consistency_check = self.validate_chapter_consistency(chapter_num, chapter_content)

             if consistency_check == "CONSISTENT":
                 print(f"Chapter {chapter_num} is consistent.")
                 break # Exit loop if consistent
             elif consistency_check == "CHECK_FAILED":
                 print(f"Consistency check failed for Chapter {chapter_num}. Proceeding with current content.")
                 break # Exit loop if check fails
             elif attempt < max_fix_attempts:
                 print(f"Consistency issues found in Chapter {chapter_num}. Issues:\n{consistency_check}\nAttempting to fix (Attempt {attempt+1}/{max_fix_attempts})...")
                 fixed_content = self.fix_chapter_inconsistencies(chapter_num, chapter_content, consistency_check)
                 if fixed_content:
                     chapter_content = fixed_content
                     print(f"Chapter {chapter_num} fixed attempt {attempt+1} successful.")
                     # Continue loop to re-validate the fixed content
                 else:
                     print(f"Error: Failed to fix consistency issues for Chapter {chapter_num} on attempt {attempt+1}. Using potentially inconsistent content.")
                     break # Exit loop if fix fails
             else:
                 print(f"Warning: Chapter {chapter_num} still has consistency issues after {max_fix_attempts} fix attempts. Issues:\n{consistency_check}\nProceeding with potentially inconsistent content.")
                 # Keep the last generated (potentially inconsistent) content


        # Create summary and update tracking *after* potential fixes
        print(f"Creating detailed summary for Chapter {chapter_num}...")
        self.create_chapter_summary(chapter_num, chapter_content) # Use potentially fixed content

        print(f"Updating character tracking for Chapter {chapter_num}...")
        self.update_character_tracking(chapter_num, chapter_content) # Use potentially fixed content

        print(f"Updating timeline information for Chapter {chapter_num}...")
        self.update_timeline(chapter_num, chapter_content) # Use potentially fixed content

        print(f"Analyzing emotional arc for Chapter {chapter_num}...")
        self.track_emotional_arc(chapter_num, chapter_content) # Use potentially fixed content

        # Add transition if not the last chapter
        if chapter_num < self.num_chapters:
            print(f"Creating transitional ending for Chapter {chapter_num}...")
            transition = self.create_chapter_transition(chapter_num, chapter_content) # Use potentially fixed content
            if transition:
                # Append the transition to the chapter content
                # Ensure there's appropriate spacing
                chapter_content = chapter_content.strip() + '\n\n' + transition
            else:
                print(f"Warning: Failed to generate or append transition for Chapter {chapter_num}.")


        return chapter_content.strip() # Return final content

    def check_chapter_transitions(self):
        """Check and improve transitions between all chapters after generation"""
        if len(self.chapters) < 2:
             print("Not enough chapters to check transitions.")
             return # Need at least two chapters

        print("Performing final check on chapter transitions...")
        improved_chapters = [self.chapters[0]] # Start with the first chapter

        for i in range(1, len(self.chapters)):
            current_chapter_text = self.chapters[i]
            prev_chapter_text = improved_chapters[i-1] # Use potentially improved previous chapter

            # Extract end of previous and start of current for context
            prev_ending_snippet = ' '.join(prev_chapter_text.strip().split()[-250:]) # More context
            current_opening_snippet = ' '.join(current_chapter_text.strip().split()[:250]) # More context

            # Updated system prompt
            system_prompt = f"""You are a professional editor specializing in narrative flow and chapter transitions for the {self.genre} genre."""

            prompt = f"""Analyze the transition between the end of Chapter {i} and the beginning of Chapter {i+1} in this {self.genre} novel.

            END OF PREVIOUS CHAPTER ({i}) (Snippet):
            ...{prev_ending_snippet}

            BEGINNING OF CURRENT CHAPTER ({i+1}) (Snippet):
            {current_opening_snippet}...

            Focus on the flow, time jump (if any), tone shift, and clarity of connection.

            If the transition is already smooth, natural, and effective for the {self.genre} genre, respond ONLY with the word "SMOOTH".

            If the transition feels abrupt, confusing, repetitive, or could be improved:
            Rewrite the opening paragraph(s) of the CURRENT chapter ({i+1}) (approx. 1-2 paragraphs, 100-200 words) to create a better bridge.
            The revised opening should:
            1. Connect more seamlessly to the previous chapter's ending mood and events.
            2. Clearly establish the time and setting for the current chapter.
            3. Avoid jarring tone shifts unless intentional and well-handled.
            4. Maintain character and plot consistency.
            5. Engage the reader immediately in the current chapter's action or theme.

            Start your response with "REVISED:" followed ONLY by the rewritten opening paragraph(s) for Chapter {i+1}.
            Do NOT include the chapter title or any other commentary.
            """

            transition_check_response = self.generate_text(prompt, system_prompt)

            if transition_check_response and transition_check_response.strip().upper() == "SMOOTH":
                 print(f"Transition to Chapter {i+1} is smooth.")
                 improved_chapters.append(current_chapter_text)
            elif transition_check_response and transition_check_response.startswith("REVISED:"):
                try:
                    revised_beginning_text = transition_check_response.split("REVISED:", 1)[1].strip()
                    if revised_beginning_text:
                        # Find the chapter title (e.g., "## Chapter X: Title")
                        title_match = re.match(r"(##\s*Chapter\s*\d+[:\s\w\W]*?\n+)", current_chapter_text) # More robust title match
                        if title_match:
                            chapter_title = title_match.group(1)
                            # Get content after the title
                            content_after_title = current_chapter_text[len(chapter_title):].strip()
                            # Split the content after title into paragraphs
                            paragraphs_after_title = re.split(r'\n\s*\n', content_after_title) # Split on blank lines

                            # Estimate how many paragraphs the revision replaces (simple heuristic)
                            num_paras_in_revision = len(re.split(r'\n\s*\n', revised_beginning_text))

                            if len(paragraphs_after_title) >= num_paras_in_revision:
                                # Join the remaining paragraphs back together
                                remaining_content = '\n\n'.join(paragraphs_after_title[num_paras_in_revision:])
                                # Construct the improved chapter
                                improved_chapter_text = chapter_title.strip() + '\n\n' + revised_beginning_text + '\n\n' + remaining_content
                                improved_chapters.append(improved_chapter_text.strip())
                                print(f"Transition improved for Chapter {i+1}.")
                            else:
                                # Fallback: Not enough paragraphs to replace, just prepend revised opening after title
                                print(f"Warning: Replacing paragraphs for Chapter {i+1} transition failed. Prepending.")
                                improved_chapters.append(chapter_title.strip() + '\n\n' + revised_beginning_text + '\n\n' + content_after_title)
                        else:
                            # Fallback: No title found, prepend revised beginning (less ideal)
                            print(f"Warning: Chapter title not found in standard format for Chapter {i+1}. Prepending revised opening to existing text.")
                            improved_chapters.append(revised_beginning_text + '\n\n' + current_chapter_text.strip())
                    else:
                        print(f"Transition for Chapter {i+1} marked for revision, but no revised text provided. Using original.")
                        improved_chapters.append(current_chapter_text)
                except Exception as e:
                    print(f"Error parsing or applying revised transition for Chapter {i+1}: {e}. Using original.")
                    improved_chapters.append(current_chapter_text)
            else:
                print(f"Transition check/revision failed for Chapter {i+1}. Using original. Response: {transition_check_response[:100] if transition_check_response else 'None'}")
                improved_chapters.append(current_chapter_text)

        self.chapters = improved_chapters
        print("Chapter transitions check complete.")


    def generate_book(self):
        """Generate the complete book with enhanced consistency checks"""
        self.get_user_input()
        self.create_story_outline()

        if not self.story_outline or not self.chapter_plan:
            print("Critical error: Story outline or chapter plan generation failed. Aborting book generation.")
            return "Book generation failed due to missing outline or plan."

        start_time = time.time()
        for i in range(1, self.num_chapters + 1):
            chapter_start_time = time.time()
            print(f"\n--- Generating Chapter {i} of {self.num_chapters} ---")
            chapter = self.generate_chapter(i)
            if chapter and f"Chapter {i}" in chapter[:100]: # Basic check for valid chapter
                self.chapters.append(chapter)
                chapter_end_time = time.time()
                print(f"--- Chapter {i} generated in {chapter_end_time - chapter_start_time:.2f} seconds ---")
            else:
                print(f"Critical error: Generation of Chapter {i} failed or produced invalid output. Aborting book generation.")
                # Add a placeholder to indicate failure but stop generation
                self.chapters.append(f"## Chapter {i}: Content Generation Failed\n\n[Generation aborted after failure in this chapter.]")
                return self.compile_book() # Compile what was generated so far


            # Optional: Add a small delay between chapters if needed
            if i < self.num_chapters:
                 # time.sleep(1) # Short delay
                 pass # Currently no delay needed unless API rate limits are hit

        # Perform final check on transitions between chapters
        if len(self.chapters) > 1: # Only check transitions if there's more than one chapter
            self.check_chapter_transitions()

        end_time = time.time()
        print(f"\n--- Book generation complete in {end_time - start_time:.2f} seconds ---")
        return self.compile_book()

    def compile_book(self):
        """Compile all chapters into a complete book, containing only title and chapters."""
        # Updated title prompt to include genre
        title_prompt = f"""Create a compelling and marketable title for a {self.genre} book with the following premise and outline snippet:

Genre: {self.genre}
Premise: {self.story_premise}

Story Outline Snippet (first few chapter ideas/themes):
{self.story_outline[:1000] if self.story_outline else "N/A"}

Respond with ONLY the book title, nothing else. Make it appropriate for the {self.genre} genre.
"""
        # Use a more robust system prompt for title generation
        title_system_prompt = f"You are an expert book marketer specializing in catchy titles for the {self.genre} genre."
        book_title = self.generate_text(title_prompt, title_system_prompt)

        # Clean up the generated title
        if not book_title:
            book_title = f"Untitled {self.genre} Story" # Fallback title
        else:
            book_title = book_title.strip().split('\n')[0] # Take first line
            # Remove potential labels like "Title:", "Book Title:", etc.
            book_title = re.sub(r"^(Title|Book Title):?\s*", "", book_title, flags=re.IGNORECASE).strip()
            # Remove surrounding quotes if present
            book_title = book_title.strip('"\'')
            if not book_title: # If cleaning leaves it empty
                book_title = f"Untitled {self.genre} Story"


        # Add Author Name (Placeholder - could be made dynamic)
        author_name = "Generated by AI Narrator" # Placeholder

        # Compile book content
        # Use Markdown formatting: Title H1, Author H3, Chapters H2
        book = f"# {book_title}\n\n"
        book += f"### By {author_name}\n\n"
        book += "---\n\n" # Add a horizontal rule after title/author

        # Iterate through the generated chapters and append them
        for i, chapter_content in enumerate(self.chapters):
            # Ensure each chapter starts with H2 title and has proper spacing
            # The generate_chapter function should already format the title correctly
            if chapter_content.strip().startswith(f"## Chapter {i+1}"):
                 book += f"{chapter_content.strip()}\n\n"
            elif chapter_content.strip().startswith(f"Chapter {i+1}"): # Handle missing '##'
                 # Add the H2 markdown if missing
                 book += f"## {chapter_content.strip()}\n\n"
            elif f"Chapter {i+1}: Generation Failed" in chapter_content:
                 # Keep failure message as is
                 book += f"{chapter_content.strip()}\n\n"
            else:
                 # Fallback: Add a generic title if missing entirely
                 print(f"Warning: Chapter {i+1} content missing standard title format. Adding generic title.")
                 book += f"## Chapter {i+1}: [Untitled Chapter]\n\n{chapter_content.strip()}\n\n"

            # Add a separator between chapters for clarity in the raw file
            if i < len(self.chapters) - 1:
                book += "---\n\n"


        return book.strip() # Remove any trailing whitespace/newlines


    def save_book(self, book_content, filename_prefix="generated_book"):
        """Save the generated book and metadata to files"""
        if not book_content or "Book generation failed" in book_content or not self.chapters:
            print(f"Book content is missing or incomplete. Not saving files.")
            return

        # Sanitize title for filename
        safe_title = re.sub(r'[\\/*?:"<>|]', "", book_title).strip()
        safe_title = re.sub(r'\s+', '_', safe_title) # Replace spaces with underscores
        filename_md = f"{filename_prefix}_{safe_title}_{self.genre}.md"
        filename_json = f"{filename_prefix}_{safe_title}_{self.genre}_metadata.json"


        try:
            with open(filename_md, "w", encoding="utf-8") as f:
                f.write(book_content)
            print(f"Book saved as {filename_md}")
        except IOError as e:
            print(f"Error saving book to {filename_md}: {e}")
            return # Stop if book saving fails


        # Prepare metadata for saving
        metadata = {
            "title_generated": book_title,
            "genre": self.genre,
            "story_premise": self.story_premise,
            "num_chapters_requested": self.num_chapters,
            "num_chapters_generated": len(self.chapters),
            "world_name": self.world_name,
            "characters": self.characters, # Save the final state
            "chapter_summaries": self.chapter_summaries,
            "recurring_motifs": self.recurring_motifs,
            "timeline": self.timeline,
            "emotional_arc": self.emotional_arc,
            "story_outline_snippet": (self.story_outline[:2000] + "..." if len(self.story_outline) > 2000 else self.story_outline) if self.story_outline else "N/A",
            "chapter_plan_snippet": (self.chapter_plan[:2000] + "..." if len(self.chapter_plan) > 2000 else self.chapter_plan) if self.chapter_plan else "N/A",
            "generation_model": self.model,
            "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z")
        }

        try:
            with open(filename_json, "w", encoding="utf-8") as f:
                # Use default=str to handle potential non-serializable data gracefully
                json.dump(metadata, f, indent=4, default=str)
            print(f"Book metadata saved as {filename_json}")
        except IOError as e:
            print(f"Error saving metadata to {filename_json}: {e}")
        except TypeError as e:
            print(f"Error serializing metadata to JSON: {e}. Some data might not be fully saved.")


if __name__ == "__main__":
    # Example usage:
    generator = BookGenerator()
    # Set base URL if your Ollama instance is running elsewhere
    # generator.base_url = "http://your_ollama_ip:11434/api/generate"
    book_text_content = generator.generate_book()

    # Check if book generation produced content before saving
    if book_text_content and "Generation Failed" not in book_text_content and generator.chapters:
        # Extract the generated title from the compiled book content for the filename
        title_match = re.match(r"#\s*(.*)", book_text_content)
        book_title = title_match.group(1).strip() if title_match else "Untitled_Book"
        generator.save_book(book_text_content, filename_prefix=f"book_{book_title[:20]}") # Use part of title in prefix
    else:
        print("\nBook generation was unsuccessful or incomplete. No file saved.")
