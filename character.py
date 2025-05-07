import ollama # Still used for the initial model check
import os
import re
import time
import random
from fpdf import FPDF

# --- LangChain Imports ---
from langchain_ollama import OllamaLLM  # Updated import
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Configuration ---
OLLAMA_MODEL = 'gemma3:4b' # Model specified by the user
NUM_CHARACTERS = 100
OUTPUT_DIR = './characters'

# Define character archetypes to introduce variety
CHARACTER_ARCHETYPES = [
    "a heroic adventurer", "a cunning rogue", "a wise mentor", "a tragic villain",
    "a determined survivor", "an eccentric inventor", "a mysterious stranger", "a reluctant leader",
    "a skilled artisan", "a spiritual guide", "a charming diplomat", "a hardened warrior",
    "a brilliant scholar", "a charismatic performer", "a dedicated healer", "a troubled artist",
    "a retired soldier", "an ambitious merchant", "a gifted prodigy", "a wandering nomad",
    "a disgraced noble", "a reformed criminal", "a visionary leader", "a cynical detective",
    "a naive dreamer", "a weary traveler", "a rebellious youth", "a foreign diplomat",
    "a reclusive genius", "a celebrated athlete", "a haunted medium", "a hopeful refugee",
    "a curious explorer", "a traditional craftsperson", "a disillusioned idealist", "a passionate revolutionary",
    "a stern guardian", "a flamboyant entertainer", "a patient teacher", "a meticulous planner"
]

# Define possible time periods/settings
CHARACTER_SETTINGS = [
    "a modern city", "a medieval fantasy world", "a cyberpunk future", "a post-apocalyptic wasteland",
    "a historical setting during the Renaissance", "an ancient civilization", "a space colony",
    "a small rural town", "a steampunk alternate history", "a magical academy", "a floating sky city",
    "a desert caravan society", "an underwater dome habitat", "a tropical island nation",
    "a nomadic tribe", "a frozen northern outpost", "a Victorian-era metropolis",
    "a Wild West frontier town", "a hidden valley civilization", "a busy spaceport",
    "a remote mountain monastery", "a sprawling underground network", "a bustling trading port",
    "a mythological realm", "a prehistoric tribal society", "a world recovering from magical cataclysm",
    "a feudal Japanese-inspired land", "a Nordic-inspired coastal settlement", "a Mesoamerican-style empire",
    "a Silk Road trading post", "a Mediterranean island republic", "a utopian community experiment",
    "a time-displaced pocket dimension", "an interdimensional nexus", "a planet with perpetual night",
    "a crystal-based civilization", "a settlement built on giant beasts", "a world where technology is forbidden",
    "a realm where dreams manifest physically", "a society evolved from a generation ship"
]

# Define the core prompt content here
CHARACTER_PROMPT_CONTENT = """
Create a unique character who is {archetype} living in {setting}. 

**1. Basic Information:**
   - Name: [Create a completely unique full name that fits the character and setting - DO NOT use the name Silas or any name you've used before]
   - Age: [Number]
   - Appearance: [Detailed description - height, build, hair, eyes, distinguishing features, typical clothing style]
   - Occupation: [Current job or primary role]

**2. Personal History and Background:**
   - Birthplace: [City, country, region, or world]
   - Family: [Parents, siblings, significant family dynamics, upbringing]
   - Key Childhood Events: [Formative experiences, traumas, joys]
   - Education/Training: [Formal or informal learning experiences]
   - Significant Past Events: [Major life changes, turning points before the defining moment]

**3. Personality:**
   - Core Traits: [List 5-7 key adjectives, e.g., cynical, compassionate, ambitious, cautious]
   - Quirks: [Unusual habits, odd preferences, unique behaviors]
   - Mannerisms: [Distinctive ways of speaking, moving, gesturing]
   - Strengths: [Positive attributes, things they excel at]
   - Flaws/Weaknesses: [Negative attributes, areas of struggle, vulnerabilities]

**4. Motivations, Desires, and Fears:**
   - Primary Motivation: [What drives them most?]
   - Core Desire(s): [What do they deeply want to achieve or obtain?]
   - Greatest Fear(s): [What terrifies them or what outcome do they desperately want to avoid?]
   - Goals (Short-term / Long-term): [Specific objectives they are working towards]

**5. Relationships and Social Connections:**
   - Closest Allies/Friends: [Names and nature of relationship]
   - Rivals/Enemies: [Names and nature of conflict]
   - Romantic Interests (if any): [Past or present]
   - General Social Attitude: [Introverted/Extroverted, Trusting/Suspicious, etc.]
   - Relationship with Authority: [Respectful, rebellious, indifferent?]

**6. Special Abilities or Skills (if any):**
   - [List any extraordinary abilities, magical powers, unique talents, or highly developed skills. If none, state "None."]

**7. Internal Conflicts and Personal Challenges:**
   - [Describe their main internal struggles, moral dilemmas, or psychological battles]

**8. Cultural Background:**
   - [Describe the culture they grew up in or identify with]

**9. Defining Moment:**
   - [Describe a specific event that fundamentally changed their life path, personality, or worldview]

**10. Typical Day and Living Environment:**
    - Routine: [Describe a typical day from waking to sleeping]
    - Living Space: [Describe their home - location, type, condition, notable features]
    - Environment: [Describe the broader setting they live in]
"""
# --- End Configuration ---

# --- Helper Functions ---

def extract_first_name(text):
    """Attempts to extract the first name from the 'Name:' line."""
    # Check if the name already has "Silas" and generate a random alternative if it does
    if re.search(r"Name:.*\bSilas\b", text, re.IGNORECASE | re.MULTILINE):
        alternative_names = ["Rowan", "Jasper", "Oliver", "Elias", "Theo", "Miles", "Atlas", "Felix", "Finn", "Hugo"]
        random_name = random.choice(alternative_names)
        print(f"   [Alert] Found 'Silas' again! Replacing with '{random_name}'")
        # Replace Silas with the random name
        text = re.sub(r"(Name:.*\b)Silas(\b)", f"\\1{random_name}\\2", text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Regex to find "Name:" followed by optional spaces and capture the first word
    match = re.search(r"Name:\s*([\w'-]+)", text, re.IGNORECASE | re.MULTILINE)
    if match:
        # Return the captured group (first name), remove problematic chars for filename
        first_name = re.sub(r"[^\w-]", '', match.group(1).strip())
        
        # If we somehow still got Silas, replace it
        if first_name.lower() == "silas":
            alternative_names = ["Rowan", "Jasper", "Oliver", "Elias", "Theo", "Miles", "Atlas", "Felix", "Finn", "Hugo"]
            first_name = random.choice(alternative_names)
            print(f"   [Alert] Found 'Silas' as first name! Replaced with '{first_name}'")
            
        return first_name
    else:
        # Fallback: Try finding the line and splitting
        lines = text.split('\n')
        for idx, line in enumerate(lines):
            line_lower = line.strip().lower()
            # Look for lines indicating the start of the basic info section
            if line_lower.startswith("**1. basic information") or line_lower.startswith("1. basic information") or line_lower.startswith("basic information:"):
                 # Search subsequent lines for Name:
                 search_limit = min(idx + 4, len(lines)) # Check next few lines
                 for sub_line in lines[idx+1:search_limit]:
                     sub_line_lower = sub_line.strip().lower()
                     if sub_line_lower.startswith("- name:") or sub_line_lower.startswith("name:"):
                          parts = sub_line.split(':', 1) # Split only on the first colon
                          if len(parts) > 1:
                              name_part = parts[1].strip()
                              # Take the first word
                              first_name = name_part.split(' ')[0]
                              if first_name:
                                   # Sanitize
                                   first_name = re.sub(r"[^\w-]", '', first_name.strip())
                                   # Check if we got Silas again
                                   if first_name.lower() == "silas":
                                       alternative_names = ["Rowan", "Jasper", "Oliver", "Elias", "Theo", "Miles", "Atlas", "Felix", "Finn", "Hugo"]
                                       first_name = random.choice(alternative_names)
                                       print(f"   [Alert] Found 'Silas' as first name in fallback! Replaced with '{first_name}'")
                                   return first_name
        
        # If all else fails, generate a random name
        alternative_names = ["Rowan", "Jasper", "Oliver", "Elias", "Theo", "Miles", "Atlas", "Felix", "Finn", "Hugo"]
        random_name = random.choice(alternative_names) + "_" + str(random.randint(100, 999))
        print(f"   [Warning] Could not extract a valid name. Generated random name: {random_name}")
        return random_name

def save_character(character_text, base_filename):
    """Saves the character description to both text and PDF formats when possible."""
    # Always save as text file first (most reliable)
    txt_filename = f"{base_filename}.txt"
    try:
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write(character_text)
        print(f"   Character saved to text file: {txt_filename}")
        
        # Try to create PDF as well, but don't rely on its success
        pdf_filename = f"{base_filename}.pdf"
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)
            
            # Use default font with ASCII encoding
            pdf.set_font('Arial', '', 11)
            
            # Convert Unicode to ASCII for PDF compatibility
            safe_text = character_text.encode('ascii', 'replace').decode('ascii')
            
            # Write text to PDF
            pdf.multi_cell(0, 5, safe_text)
            pdf.output(pdf_filename)
            print(f"   Character also saved as PDF: {pdf_filename}")
            return True
        except Exception as e:
            print(f"   [Note] PDF creation skipped: {e}")
            # Not counting as a failure since text file was saved
            return True
    except Exception as e:
        print(f"   [Error] Failed to save character: {e}")
        return False

# --- Main Script Logic ---

if __name__ == "__main__":
    print(f"Starting character generation using LangChain with Ollama model: {OLLAMA_MODEL}")

    # 1. Initial Model Check
    try:
        print("Checking if model is available locally via Ollama...")
        ollama.show(OLLAMA_MODEL) # Throws error if model doesn't exist
        print(f"Model '{OLLAMA_MODEL}' found.")
    except ollama.ResponseError as e:
         if e.status_code == 404:
              print(f"[Error] Model '{OLLAMA_MODEL}' not found locally.")
              print(f"Please pull it first using: ollama pull {OLLAMA_MODEL}")
         else:
              print(f"[Error] Failed to check for model '{OLLAMA_MODEL}' via Ollama: {e}")
         exit()
    except Exception as e:
         print(f"[Error] An unexpected error occurred checking for the model via Ollama: {e}")
         exit()

    # 2. Setup LangChain Components
    try:
        print("Initializing LangChain components...")
        # System prompt remains constant
        system_prompt = SystemMessagePromptTemplate.from_template(
            "Generate a complete fictional character profile filling in all sections with creative, detailed content. Make this character COMPLETELY UNIQUE with a DIFFERENT NAME from any previous characters. NEVER use the name Silas or any name you've already used. Always create totally new characters with distinctive names, personalities, appearances, and backgrounds. Do not include explanations, help text, or any content outside the character profile itself."
        )
        
        # The human prompt template now includes placeholders for variation
        human_prompt = HumanMessagePromptTemplate.from_template(CHARACTER_PROMPT_CONTENT)
        
        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
        
        # Updated to use OllamaLLM from langchain_ollama
        llm = OllamaLLM(model=OLLAMA_MODEL)
        output_parser = StrOutputParser()
        chain = chat_prompt | llm | output_parser
        print("LangChain chain created successfully.")

    except Exception as e:
        print(f"[Error] Failed to initialize LangChain components: {e}")
        exit()

    # 3. Generation Loop
    print(f"\nGenerating {NUM_CHARACTERS} characters...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")

    generated_count = 0
    failed_count = 0
    
    # Track used combinations to avoid repetition
    used_combinations = set()

    for i in range(1, NUM_CHARACTERS + 1):
        unique_id = f"{i:03d}"
        print(f"\nGenerating character {i}/{NUM_CHARACTERS}...")

        try:
            # Select random archetype and setting for this character
            # Keep trying until we get a new combination
            for _ in range(10):  # Limit attempts to avoid infinite loop
                archetype = random.choice(CHARACTER_ARCHETYPES)
                setting = random.choice(CHARACTER_SETTINGS)
                combo = (archetype, setting)
                
                # If we've already used this exact combo and we haven't used all combos
                if combo in used_combinations and len(used_combinations) < len(CHARACTER_ARCHETYPES) * len(CHARACTER_SETTINGS):
                    continue
                
                used_combinations.add(combo)
                break
            
            print(f"   Creating {archetype} in {setting}...")
            
            # Generate a list of suggested first names to avoid repetition
            first_names = [
                "Aria", "Zephyr", "Marcus", "Lyra", "Thorne", "Ember", "Jasper", "Elara", "Knox", "Nova",
                "Orion", "Seraphina", "Cyrus", "Amara", "Rowan", "Freya", "Darius", "Kira", "Rhys", "Luna",
                "Felix", "Vega", "Maddox", "Bianca", "Finn", "Astrid", "Lucian", "Octavia", "Griffin", "Ivy",
                "Soren", "Athena", "Caspian", "Juniper", "Tobias", "Celine", "Axel", "Dahlia", "Benedict", "Isolde",
                "Malcolm", "Raven", "Dorian", "Thalia", "Evander", "Willow", "Roman", "Scarlett", "Vincent", "Hope"
            ]
            
            # Generate a random name suggestion
            random_names = ", ".join(random.sample(first_names, 3))
            
            # Generate a unique seed value
            seed_value = random.randint(1, 10000)
            
            start_time = time.time()
            # Create a temporary file with previously generated names to avoid repetition
            previously_used_names_file = os.path.join(OUTPUT_DIR, "used_names.txt")
            
            # Check for existing names
            used_names = []
            if os.path.exists(previously_used_names_file):
                try:
                    with open(previously_used_names_file, 'r') as f:
                        used_names = [line.strip() for line in f.readlines() if line.strip()]
                except Exception as e:
                    print(f"   [Warning] Could not read used names file: {e}")
            
            used_names_str = ", ".join(used_names) if used_names else "None yet"
                
            character_description = chain.invoke({
                "archetype": archetype,
                "setting": setting,
                "seed": f"Seed value: {seed_value}. Suggested unique names (but feel free to create your own): {random_names}. IMPORTANT: Create a completely original character with a unique name. Previously used names to AVOID: {used_names_str}"
            })
            end_time = time.time()
            print(f"   ... LangChain response received ({end_time - start_time:.2f}s).")

            if not character_description or not isinstance(character_description, str):
                 print("   [Error] Received invalid response from LangChain chain.")
                 failed_count += 1
                 continue

            first_name = extract_first_name(character_description)
            if not first_name:
                print("   [Warning] Could not extract a valid first name. Using 'Character'.")
                base_name = "Character"
            else:
                 base_name = first_name
                 print(f"   Extracted first name: {base_name}")

            base_filename = os.path.join(OUTPUT_DIR, f"{base_name}_{unique_id}")
            
            print(f"   Saving character: {base_name}_{unique_id}")
            if save_character(character_description, base_filename):
                generated_count += 1
                
                # Record the name for future uniqueness checks
                if first_name:
                    try:
                        with open(previously_used_names_file, 'a+') as f:
                            f.write(f"{first_name}\n")
                    except Exception as e:
                        print(f"   [Warning] Could not update used names file: {e}")
            else:
                failed_count += 1

        except KeyboardInterrupt:
            print("\n\n[User Interrupt] Generation stopped by user.")
            break
        except Exception as e:
            print(f"   [Error] Failed to generate or process character {i}: {e}")
            failed_count += 1
            time.sleep(2)

        # Add a pause between generations to not overload the Ollama server
        time.sleep(1.5)

    # 4. Final Summary
    print("\n--- Generation Complete ---")
    print(f"Successfully generated characters: {generated_count}")
    print(f"Failed generations/creations: {failed_count}")
    print(f"Files saved in: {os.path.abspath(OUTPUT_DIR)}")