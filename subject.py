# subject.py

import os
import traceback
from dotenv import load_dotenv

# --- Langchain Imports ---
from langchain_ollama import OllamaLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
# --- End Imports ---

# Load environment variables (optional, but good practice)
load_dotenv()

# --- Constants ---
# Use the same model as your main app for consistency
DEFAULT_MODEL = "gemma3:4b"
# DEFAULT_MODEL = "llama3:latest" # Or your preferred model

# --- LLM Initialization Function (Adapted from app.py) ---
def create_llm(temperature=0.6): # Slightly lower temp for more focused analysis
    """Create and return an OllamaLLM instance with specified temperature."""
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    print(f"--- Connecting to Ollama at: {ollama_base_url} with Model: {DEFAULT_MODEL} ---")
    try:
        llm = OllamaLLM(
            model=DEFAULT_MODEL,
            temperature=temperature,
            base_url=ollama_base_url,
            # request_timeout=120.0 # Optional: Consider adding timeout
        )
        # Optional: Test connection
        # llm.invoke("Test connection.")
        print("--- LLM Instance Created ---")
        return llm
    except Exception as e:
        print(f"--- FATAL ERROR: Could not create OllamaLLM instance ---")
        print(f"Error: {e}")
        print("Ensure Ollama service is running and accessible at the specified base URL.")
        print(f"Attempted Base URL: {ollama_base_url}")
        print(f"Attempted Model: {DEFAULT_MODEL}")
        traceback.print_exc()
        raise

# --- Subject Generation Logic ---

class SubjectGenerator:
    # Prompt designed to extract themes and generate the specific subject format
    PROMPT_TEMPLATE = """
    Analyze the following text, which could be a piece of flash fiction, a journal entry, or a scene concept.
    Identify the dominant mood(s), central themes (e.g., introspection, insomnia, struggle with thoughts, connection to environment, time perception), potential character perspective/feelings, and any key setting details or motifs (like moonlight, silence, transition from night to day).

    Based on this analysis, craft a detailed, narrative 'subject' line (like a premise or logline, typically 2-4 sentences long) that captures the essence of the text. This subject should be suitable for seeding a creative writing process, focusing on character experience and thematic depth.
    Aim for a style similar to this example:
    "A slice-of-life story following [Character Description, e.g., an introspective individual] navigating [Key Challenge, e.g., restless nights and overwhelming thoughts] in [Setting Context, e.g., the liminal hours before dawn]. The story explores themes of [Theme 1, e.g., the nature of consciousness], [Theme 2, e.g., the search for control amidst mental chaos], and [Theme 3, e.g., the relationship between inner turmoil and the external environment], perhaps focusing on [Specific Motif, e.g., the symbolism of the 'silver hush']."

    Do NOT just summarize the plot points. Generate the narrative subject line itself.

    Input Text:
    ---------------------
    {text}
    ---------------------

    Generated Subject:
    """

    def __init__(self):
        self.llm = create_llm() # Use the shared LLM creation function
        self.chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate.from_template(self.PROMPT_TEMPLATE),
            verbose=True # Set to False for cleaner output once working
        )

    def generate(self, input_text):
        """
        Generates the subject line based on the input text.

        Args:
            input_text (str): The chunk of text to analyze.

        Returns:
            str: The generated subject line, or an error message.
        """
        if not input_text or not input_text.strip():
            return "Error: Input text cannot be empty."

        try:
            print("\n--- Generating Subject from Text ---")
            result = self.chain.invoke({"text": input_text})
            subject = result.get('text', "").strip()

            if not subject or len(subject) < 20: # Basic validation
                 print(f"Warning: Generated subject seems too short or invalid: '{subject}'")
                 return f"Error: Failed to generate a meaningful subject. LLM Output: {subject}"

            print("--- Subject Generation Complete ---")
            return subject

        except Exception as e:
            print(f"An error occurred during subject generation: {e}")
            traceback.print_exc()
            return f"Error generating subject: {e}"

# --- Main Execution ---

if __name__ == "__main__":
    print("====================================")
    print("=== SUBJECT GENERATION SCRIPT ===")
    print("====================================")

    # The input text provided by the user
    input_text_chunk = """Through the darkness, my windows every
night gleamed with the moonlight. It was
the hour, the darkness just turned a bit
brighter. Slightly, mildly, by a single shade
as if black turned silver. Chirping of birds
would start in an hour from now. It would
go unnoticed by many as they would be
soundly sleeping. I witnessed the
moonlight gleaming today as well. I wasn't
sleeping again. I stayed up late. I seem to
have noticed the darkness turning lighter
almost instantly and the very second too.
The m o m e n t t h e d a r k n e s s t u r n s a little
less quiet. A little less peaceful. Almost
pricking the eyes. Telling the world to be
awake a s if announcing the sunrise. I knew
it was too late for me to try sleeping now. I
wouldnt be able to. Still, I tried. I closed my
eyes. The moonlight made me squint my
eyes. And I yielded to the silver hush. I
could see my room now. Make out the
corners of the wall, the lining of the bed
and the other near things around me.
Staring into the night that bled silver
through the black, the barricade that held
the uproar of my thoughts broke free. In
my moonlit room, an unfiltered version of
my mind let loose. Thoughts,
unaccounted for, those that I put aside
throughout the day were laid bare. Spoke
to me like another person, not me. Singing
echoes of doubts and questions lost in
day's breath. As I conversed to this new
person solely by the thoughts that were
woven through the silence. I wanderered
through the thoughts tracing events only
my own mind can decipher. Diving deep
into those ripples. Almost mechanical
movements, i couldnt stop doing might be
the mind's quiet rebellion. When i realised,
it took all my inner strength and will to
break the loop. Until minutes passed, the
loop never ended but paused in day's
breath and silver's hush. I held the reigns
to the mechanical rhythms. I could hide on
purpose. I could willfully stop. My mind
wasn't insane enough. I was somewhere
close to the middle grounds. Yet, only
when my mind chose to be voluntary and
use all it's self control to not do it again
would my mind have the reigns. Or, it
would be stuck in an infinite loop forever.
Minutes at max, till i realise i have done it
enough and let myself voluntarily
intervene. Not the overthinking part, if I
really could reign it. I would be too
powerful. And there I lie stuck in the
middle of normalness and insanity. Finding
my middle grounds during the hours of
silver hush. One overpowered the other at
different times often. One single thread of
thought let loose and created havoc. When
it w a s over another broke loose. This time
it was events I consciously went thought.
Many faces, the past that I let go, broken
threads and thoughts far from today's
silver hush. It was morning already by the
t i m e t h e u n e a s i n e s s o v e r w h e l m e d m e into
writing it down. 7:30 a m the golden glow
was hitting my windows. The restlessness
didnt f a d e with the silver h a z e but b e c a m e
another story for another day.
-Divi
May 3, 2025
"""

    try:
        # Initialize the generator
        generator = SubjectGenerator()

        # Generate the subject
        generated_subject = generator.generate(input_text_chunk)

        # Print the result
        print("\n====================================")
        print("Generated Subject Line:")
        print("------------------------------------")
        print(generated_subject)
        print("====================================")

    except Exception as e:
        # Catch potential errors during LLM initialization
        print("\n--- SCRIPT EXECUTION FAILED ---")
        # Error details should have been printed by create_llm or SubjectGenerator
        print("Ensure Ollama is running and the model is available.")