# app.py - Modified with Plot Promise Approach
from dotenv import load_dotenv
import os
import docx
import traceback
import random
import math
from datetime import datetime

# --- Langchain Imports ---
from langchain_ollama import OllamaLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()

# --- Constants ---
DEFAULT_MODEL = "gemma3:4b"
OUTPUT_FOLDER = './docs'

# --- LLM Initialization Function ---
def create_llm():
    """Create and return an OllamaLLM instance"""
    return OllamaLLM(
        model=DEFAULT_MODEL,
        temperature=0.7,
    )

# --- Plot Promise System ---
class PlotPromise:
    """Represents a single plot promise in the narrative."""
    
    def __init__(self, description, importance=5, status="active", 
                 progression=0, dependencies=None, sequence=None):
        self.id = self._generate_id()
        self.description = description
        self.importance = importance  # 1-10 scale, higher = more important
        self.status = status  # "active", "completed", "backsliding", or "paused"
        self.progression = progression  # 0 to 100, where 100 = payoff reached
        self.last_progressed = 0  # Scene number when last progressed
        self.dependencies = dependencies or []  # List of promise IDs that must complete first
        self.progression_history = []  # List of progression points [(scene_num, progression_value)]
        self.progression_sequence = sequence or []  # Predefined sequence of progression events
    
    def _generate_id(self):
        """Generate a unique ID for this promise."""
        timestamp = int(datetime.now().timestamp())
        random_suffix = random.randint(1000, 9999)
        return f"p{timestamp}{random_suffix}"
    
    def is_progressable(self, all_promises):
        """Check if this promise can be progressed."""
        if self.status == "completed":
            return False
        
        if self.status == "paused":
            return False
            
        # Check dependencies
        for dep_id in self.dependencies:
            dep_promise = next((p for p in all_promises if p.id == dep_id), None)
            if dep_promise and dep_promise.status != "completed":
                return False
                
        return True
    
    def progress(self, amount, scene_num):
        """Progress this promise by the specified amount."""
        self.progression = min(100, self.progression + amount)
        self.last_progressed = scene_num
        self.progression_history.append((scene_num, self.progression))
        
        # Check if we've reached payoff
        if self.progression >= 100:
            self.status = "completed"
            
        return self.status
        
    def backslide(self, amount, scene_num):
        """Cause this promise to backslide (regression in progress)."""
        self.progression = max(0, self.progression - amount)
        self.status = "backsliding"
        self.last_progressed = scene_num
        self.progression_history.append((scene_num, self.progression))
        
        return self.status
        
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "description": self.description,
            "importance": self.importance,
            "status": self.status,
            "progression": self.progression,
            "last_progressed": self.last_progressed,
            "dependencies": self.dependencies,
            "progression_history": self.progression_history,
            "progression_sequence": self.progression_sequence
        }
        
    @classmethod
    def from_dict(cls, data):
        """Create a PlotPromise from a dictionary."""
        promise = cls(
            description=data["description"],
            importance=data["importance"],
            status=data["status"],
            progression=data["progression"],
            dependencies=data["dependencies"],
            sequence=data.get("progression_sequence", [])
        )
        promise.id = data["id"]
        promise.last_progressed = data["last_progressed"]
        promise.progression_history = data["progression_history"]
        return promise


class PromiseManager:
    """Manages all plot promises for a story."""
    
    def __init__(self, llm):
        self.promises = []
        self.scene_count = 0
        self.llm = llm
        self.promise_chain = self._create_promise_chain()
        self.progression_chain = self._create_progression_chain()
        self.choose_promise_chain = self._create_choose_promise_chain()
        
    def _create_promise_chain(self):
        """Create a chain for generating new plot promises."""
        prompt = """
        Generate {num_promises} new plot promises for a novel based on the provided information.
        A plot promise is a promise of something that will happen later in the story.
        It sets expectations early, then builds tension through obstacles, twists, and turning points—culminating in a powerful, satisfying climax.
        
        For each promise, provide:
        1. A clear description of what will eventually happen or be revealed
        2. An importance rating from 1-10 (higher = more central to the plot)
        3. Any dependencies (which existing promises must be fulfilled first)
        
        Current Story Context:
        - Title: {title}
        - Genre: {genre}
        - Main Character: {profile}
        - Plot Summary: {plot}
        
        Existing Plot Promises:
        {existing_promises}
        
        New Plot Promises (return in exactly this JSON-like format):
        [
          {{
            "description": "Character discovers hidden ability",
            "importance": 8,
            "dependencies": []
          }},
          ...
        ]
        """
        return LLMChain(
            llm=self.llm,
            prompt=PromptTemplate.from_template(prompt),
            verbose=True
        )
        
    def _create_progression_chain(self):
        """Create a chain for determining how a promise should progress."""
        prompt = """
        You are responsible for determining how a specific plot promise should progress in the next scene.
        A plot promise is a narrative element that is introduced early and developed throughout the story.
        
        Current Promise:
        {promise_description}
        Current Progress: {current_progress}%
        
        Previous Scene Context:
        {previous_scene}
        
        Overall Story Context:
        - Title: {title}
        - Genre: {genre}
        - Main Character: {profile}
        
        For this promise, determine:
        1. Should it progress forward, backslide, or remain static in the next scene?
        2. If progressing, by how much (10-30%)? If backsliding, by how much (5-15%)?
        3. What specific event or development should occur in this scene related to this promise?
        
        Return your answer in this exact format:
        Direction: [PROGRESS/BACKSLIDE/STATIC]
        Amount: [number between 5-30]
        Event: [detailed description of what happens with this promise in this scene]
        """
        return LLMChain(
            llm=self.llm,
            prompt=PromptTemplate.from_template(prompt),
            verbose=True
        )
        
    def _create_choose_promise_chain(self):
        """Create a chain for choosing which promise to progress next."""
        prompt = """
        You need to intelligently choose which plot promise to progress in the next scene of the story.
        A good story weaves multiple plot threads together, focusing on different promises at different times based on:
        1. What happened in the previous scene (immediate context)
        2. How long it's been since a promise was last addressed (pacing)
        3. The importance of each promise to the overall story
        4. Whether dependencies have been fulfilled
        
        Previous Scene:
        {previous_scene}
        
        Available Promises (sorted by algorithm suggestion):
        {available_promises}
        
        Based on narrative flow, pacing, and the previous scene's context, which ONE promise should be the focus of the next scene?
        Provide your reasoning and then your final choice.
        
        Note: Don't automatically choose the top-ranked promise. Consider what makes the most natural progression from the previous scene.
        
        Reasoning: [your step-by-step thought process]
        Chosen Promise ID: [promise_id]
        """
        return LLMChain(
            llm=self.llm,
            prompt=PromptTemplate.from_template(prompt),
            verbose=True
        )
        
    def generate_initial_promises(self, title, genre, profile, plot, num_promises=5):
        """Generate the initial set of plot promises for the story."""
        # Format existing promises (empty for initial generation)
        existing_promises_text = "None yet"
        
        # Generate promises
        try:
            result = self.promise_chain.invoke({
                "num_promises": num_promises,
                "title": title,
                "genre": genre,
                "profile": profile,
                "plot": plot,
                "existing_promises": existing_promises_text
            })
            
            raw_text = result.get('text', '[]')
            
            # Clean up the text to get just the JSON part
            json_start = raw_text.find('[')
            json_end = raw_text.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_text = raw_text[json_start:json_end]
                # Parse safely
                import json
                try:
                    promises_data = json.loads(json_text)
                    for data in promises_data:
                        promise = PlotPromise(
                            description=data.get("description", "Unknown promise"),
                            importance=data.get("importance", 5),
                            dependencies=data.get("dependencies", [])
                        )
                        self.promises.append(promise)
                    
                    print(f"Successfully generated {len(self.promises)} initial plot promises")
                except json.JSONDecodeError as e:
                    print(f"Error parsing promise JSON: {e}")
                    # Fallback: create some basic promises
                    self._create_fallback_promises(title, genre, profile, plot)
            else:
                print("Could not find valid JSON in the response")
                self._create_fallback_promises(title, genre, profile, plot)
                
        except Exception as e:
            print(f"Error generating initial promises: {e}")
            traceback.print_exc()
            self._create_fallback_promises(title, genre, profile, plot)
            
    def _create_fallback_promises(self, title, genre, profile, plot, num=3):
        """Create basic fallback promises if generation fails."""
        print("Creating fallback plot promises...")
        
        # Basic story structure promises
        basic_promises = [
            {"description": f"Main character in '{title}' overcomes a significant personal challenge", "importance": 9},
            {"description": f"A relationship conflict emerges and is eventually resolved", "importance": 7},
            {"description": f"An external threat or opportunity forces character growth", "importance": 8}
        ]
        
        for data in basic_promises[:num]:
            self.promises.append(PlotPromise(
                description=data["description"],
                importance=data["importance"]
            ))
            
        print(f"Created {len(self.promises)} fallback promises")
        
    def add_promise(self, description, importance=5, dependencies=None):
        """Add a new plot promise to the story."""
        promise = PlotPromise(description, importance, dependencies=dependencies or [])
        self.promises.append(promise)
        return promise.id
        
    def complete_promise(self, promise_id):
        """Mark a promise as completed."""
        for promise in self.promises:
            if promise.id == promise_id:
                promise.status = "completed"
                promise.progression = 100
                return True
        return False
        
    def _calculate_promise_scores(self):
        """Calculate a score for each promise to determine progression priority."""
        scores = {}
        
        for promise in self.promises:
            if not promise.is_progressable(self.promises):
                continue
                
            # Base score is the importance (1-10)
            score = promise.importance * 10
            
            # Time factor: higher score if it's been a while since this was progressed
            if promise.last_progressed > 0:
                time_since_last = self.scene_count - promise.last_progressed
                time_factor = min(50, time_since_last * 5)  # Cap at +50
                score += time_factor
            else:
                # Bonus for promises that haven't been started yet
                score += 30
                
            # Progress factor: lower score as we get closer to completion
            progress_factor = 100 - promise.progression
            score += progress_factor * 0.5
            
            scores[promise.id] = score
            
        return scores
        
    def choose_next_promise(self, previous_scene):
        """Intelligently choose which promise to progress next."""
        # First, calculate algorithmic scores
        scores = self._calculate_promise_scores()
        
        # Filter to progressable promises
        progressable_promises = [p for p in self.promises if p.is_progressable(self.promises)]
        
        if not progressable_promises:
            print("No progressable promises available. Creating a new one...")
            # Create a new promise if none are available
            new_promise = PlotPromise("New narrative development", importance=7)
            self.promises.append(new_promise)
            return new_promise
            
        # Sort by score
        sorted_promises = sorted(
            progressable_promises, 
            key=lambda p: scores.get(p.id, 0),
            reverse=True
        )
        
        # Format the available promises for the LLM
        available_promises_text = "\n".join([
            f"ID: {p.id} - {p.description} (Importance: {p.importance}, "
            f"Progress: {p.progression}%, Last Progressed: Scene {p.last_progressed or 'Never'})"
            for p in sorted_promises[:5]  # Top 5 candidates
        ])
        
        # Use LLM to make the final choice based on narrative context
        try:
            result = self.choose_promise_chain.invoke({
                "previous_scene": previous_scene,
                "available_promises": available_promises_text
            })
            
            response_text = result.get('text', '')
            
            # Parse the response to extract the chosen promise ID
            chosen_id = None
            for line in response_text.split('\n'):
                if line.startswith("Chosen Promise ID:"):
                    chosen_id = line.split(":", 1)[1].strip()
                    break
                    
            if chosen_id:
                chosen_promise = next((p for p in self.promises if p.id == chosen_id), None)
                if chosen_promise:
                    return chosen_promise
                    
            # Fallback to the highest scored promise if parsing failed
            print("Could not parse chosen promise ID, using highest scored promise")
            return sorted_promises[0]
            
        except Exception as e:
            print(f"Error choosing next promise: {e}")
            traceback.print_exc()
            # Fallback to algorithm choice
            return sorted_promises[0]
            
    def progress_promise(self, promise, previous_scene, title, genre, profile):
        """Determine how a promise should progress and update its state."""
        self.scene_count += 1
        
        try:
            # Get progression guidance from LLM
            result = self.progression_chain.invoke({
                "promise_description": promise.description,
                "current_progress": promise.progression,
                "previous_scene": previous_scene,
                "title": title,
                "genre": genre,
                "profile": profile
            })
            
            response_text = result.get('text', '')
            
            # Parse the response
            direction = "STATIC"
            amount = 0
            event = "No specific event occurs"
            
            for line in response_text.split('\n'):
                if line.startswith("Direction:"):
                    direction = line.split(":", 1)[1].strip().upper()
                elif line.startswith("Amount:"):
                    try:
                        amount = int(line.split(":", 1)[1].strip())
                    except ValueError:
                        amount = 15  # Default if parsing fails
                elif line.startswith("Event:"):
                    event = line.split(":", 1)[1].strip()
            
            # Update the promise based on the direction
            if direction == "PROGRESS":
                promise.progress(amount, self.scene_count)
            elif direction == "BACKSLIDE":
                promise.backslide(amount, self.scene_count)
            # STATIC requires no update
            
            # Return the event description
            return event
            
        except Exception as e:
            print(f"Error progressing promise: {e}")
            traceback.print_exc()
            # Default progression
            promise.progress(15, self.scene_count)
            return f"The story advances with respect to {promise.description.lower()}"
        
    def get_active_promises(self):
        """Get all active promises."""
        return [p for p in self.promises if p.status != "completed"]
        
    def get_completed_promises(self):
        """Get all completed promises."""
        return [p for p in self.promises if p.status == "completed"]
        
    def get_promise_status_text(self):
        """Get a textual summary of all promises and their status."""
        active = self.get_active_promises()
        completed = self.get_completed_promises()
        
        text = "ACTIVE PLOT PROMISES:\n"
        for p in active:
            text += f"- {p.description} (Progress: {p.progression}%, Importance: {p.importance})\n"
            
        text += "\nCOMPLETED PLOT PROMISES:\n"
        for p in completed:
            text += f"- {p.description}\n"
            
        return text


# --- Chain Classes ---

class MainCharacterChain:
    PROMPT = """
    You are provided with the resume text of a person.
    Describe the person's profile concisely in 2-3 sentences. Include the person's name if explicitly mentioned in the text.

    Resume Text:
    {text}

    Profile Description:"""

    def __init__(self):
        self.llm = create_llm()
        self.chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate.from_template(self.PROMPT),
            verbose=True
        )

    def load_resume(self, file_name):
        """Loads text content from a PDF file."""
        file_path = os.path.join(OUTPUT_FOLDER, file_name)
        if not os.path.exists(file_path):
             raise FileNotFoundError(f"Resume file not found at: {file_path}")
        try:
            print(f"Loading PDF from: {file_path}")
            loader = PyPDFLoader(file_path)
            # Use load() which returns a list of Document objects, one per page
            docs = loader.load()
            if not docs:
                print(f"Warning: PyPDFLoader didn't extract any documents/pages from {file_name}.")
                return None # Indicate no content found
            print(f"Successfully loaded {len(docs)} page(s) from PDF.")
            return docs
        except Exception as e:
             print(f"Error loading PDF {file_path}: {e}")
             traceback.print_exc() # Print full traceback for PDF errors
             raise # Reraise the exception

    def run(self, file_name):
        """Loads resume and generates the profile description."""
        try:
            docs = self.load_resume(file_name)
            if not docs:
                 print("Could not load resume content.")
                 return "Error: Could not generate profile due to missing resume content."

            # Combine text from all pages
            resume_text = '\n\n'.join([doc.page_content for doc in docs if doc.page_content])

            if not resume_text.strip():
                 print("Warning: Resume content appears empty after extraction.")
                 return "Error: Could not generate profile because the resume content is empty."

            # LangChain components now often prefer invoke over run
            # Use invoke which returns a dictionary
            result = self.chain.invoke({"text": resume_text})
            # Extract the actual response text (key might be 'text' or specific to the chain)
            # Check the verbose output or debug to confirm the key
            return result.get('text', "Error: Profile generation failed.").strip()

        except Exception as e:
            print(f"An error occurred in MainCharacterChain.run: {e}")
            traceback.print_exc()
            return f"Error generating profile: {e}"


class TitleChain:
    PROMPT = """
    Generate a compelling novel title based on the provided details.
    Return ONLY the title itself, without any quotation marks, labels, or extra text.
    The title must be consistent with the specified genre and author's style.

    Subject: {subject}
    Genre: {genre}
    Author's Style: {author}
    Main Character Profile: {profile}

    Novel Title:"""

    def __init__(self):
        self.llm = create_llm()
        self.chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate.from_template(self.PROMPT),
            verbose=True
        )

    def run(self, subject, genre, author, profile):
        """Generates the novel title."""
        try:
            result = self.chain.invoke({
                "subject": subject,
                "genre": genre,
                "author": author,
                "profile": profile
            })
            # Clean the output string thoroughly
            title = result.get('text', "Untitled").strip()
            # Remove potential surrounding quotes
            if title.startswith('"') and title.endswith('"'):
                title = title[1:-1]
            if title.startswith("'") and title.endswith("'"):
                title = title[1:-1]
            return title
        except Exception as e:
            print(f"An error occurred in TitleChain.run: {e}")
            traceback.print_exc()
            return "Error Generating Title"


class PlotChain:
    PROMPT = """
    Generate a detailed plot outline for a novel. Return ONLY the plot description.
    Create a compelling narrative, introducing necessary supporting characters or conflicts.
    The main character must be central to the plot.
    Ensure the plot aligns with the novel's genre, author's style, title, and subject.
    Incorporate some of the following exciting story attributes: {features}.

    Subject: {subject}
    Genre: {genre}
    Author's Style: {author}
    Title: "{title}"
    Main Character Profile: {profile}

    Plot Outline:"""

    HELPER_PROMPT = """
    Generate a comma-separated list of 5-7 key attributes that make a story plot exciting and engaging.
    Examples: High stakes, Moral ambiguity, Unpredictable twists, Compelling antagonist, Emotional depth, Unique world-building, Fast pacing.

    List of attributes:"""

    def __init__(self):
        self.llm = create_llm()
        # Main chain for plot generation
        self.chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate.from_template(self.PROMPT),
            verbose=True
        )
        # Helper chain to generate dynamic features
        self.helper_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate.from_template(self.HELPER_PROMPT),
            verbose=True # Make helper verbose to see feature generation
        )

    def run(self, subject, genre, author, profile, title):
        """Generates the novel plot."""
        try:
            # Generate exciting features using the helper chain
            # Note: Helper prompt doesn't have input variables, so pass dummy dict
            features_result = self.helper_chain.invoke({})
            features = features_result.get('text', "High stakes, Twists, Emotional depth").strip()
            print(f"Generated features for plot: {features}")

            # Generate the main plot
            plot_result = self.chain.invoke({
                "features": features,
                "subject": subject,
                "genre": genre,
                "author": author,
                "profile": profile,
                "title": title
            })
            return plot_result.get('text', "Error: Plot generation failed.").strip()
        except Exception as e:
            print(f"An error occurred in PlotChain.run: {e}")
            traceback.print_exc()
            return f"Error generating plot: {e}"


class SceneWriterChain:
    PROMPT = """
    You are a skilled novelist writing in the distinctive style of {author}.
    The novel, titled "{title}", is a {genre} story.
    Main Character: {profile}
    Novel's Plot Summary: {plot}

    You are writing a scene that progresses this specific plot promise:
    <PROMISE>
    {promise_description}
    </PROMISE>

    The specific event that should happen in this scene:
    <EVENT>
    {event_description}
    </EVENT>

    Previous scene context:
    <PREVIOUS_SCENE>
    {previous_scene}
    </PREVIOUS_SCENE>

    Active plot promises to keep in mind (these don't need to be directly addressed):
    <ACTIVE_PROMISES>
    {active_promises}
    </ACTIVE_PROMISES>

    Write an engaging scene (500-800 words) that naturally progresses the story.
    Focus on the specified event while maintaining the flow from the previous scene.
    Use vivid descriptions, natural dialogue, and character development appropriate to {author}'s style.
    Don't explicitly mention "plot promises" - just weave them naturally into the narrative.
    
    Scene:"""

    def __init__(self):
        self.llm = create_llm()
        self.chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate.from_template(self.PROMPT),
            verbose=True
        )

    def run(self, genre, author, title, profile, plot, promise_description, 
            event_description, previous_scene="", active_promises=""):
        """Generates a scene that progresses a specific plot promise."""
        try:
            result = self.chain.invoke({
                "genre": genre,
                "author": author,
                "title": title,
                "profile": profile,
                "plot": plot,
                "promise_description": promise_description,
                "event_description": event_description,
                "previous_scene": previous_scene,
                "active_promises": active_promises
            })
            return result.get('text', "Error writing scene").strip()
        except Exception as e:
            print(f"Error writing scene: {e}")
            traceback.print_exc()
            return f"[Error occurred while writing scene: {e}]"


# --- Novel Writing Orchestration ---

def write_novel_with_promises(genre, author, title, profile, plot, num_scenes=12):
    """Write a novel using the plot promise system."""
    print("\n--- Starting Novel Writing with Plot Promise System ---")
    
    # Setup the promise manager
    llm = create_llm()
    promise_manager = PromiseManager(llm)
    
    # Generate initial promises
    print("Generating initial plot promises...")
    promise_manager.generate_initial_promises(title, genre, profile, plot)
    
    # Setup scene writer
    scene_writer = SceneWriterChain()
    
    # Prepare for writing
    scenes = []
    previous_scene_summary = f"The story begins with {profile}"
    
    print(f"\n--- Writing {num_scenes} Scenes ---")
    
    for scene_num in range(1, num_scenes + 1):
        print(f"\n--- Writing Scene {scene_num}/{num_scenes} ---")
        
        # Choose which promise to progress in this scene
        chosen_promise = promise_manager.choose_next_promise(previous_scene_summary)
        print(f"Chosen promise: {chosen_promise.description} (Progress: {chosen_promise.progression}%)")
        
        # Determine how the promise progresses and what event happens
        event = promise_manager.progress_promise(
            chosen_promise, 
            previous_scene_summary,
            title, 
            genre, 
            profile
        )
        print(f"Event: {event}")
        
        # Get active promises summary (for context)
        active_promises = "\n".join([
            f"- {p.description} (Progress: {p.progression}%)" 
            for p in promise_manager.get_active_promises()[:5]
        ])
        
        # Write the scene
        scene_content = scene_writer.run(
            genre=genre,
            author=author,
            title=title,
            profile=profile,
            plot=plot,
            promise_description=chosen_promise.description,
            event_description=event,
            previous_scene=previous_scene_summary,
            active_promises=active_promises
        )
        
        # Save the scene
        scene_title = f"Scene {scene_num}: {event[:60]}..."
        scenes.append({
            "title": scene_title,
            "content": scene_content,
            "promise": chosen_promise.description,
            "event": event
        })
        
        # Update previous scene summary for context
        previous_scene_summary = scene_content[:500] + "..."  # Limit size for context
        
        # Every 3 scenes, consider adding a new promise to keep the story dynamic
        if scene_num % 3 == 0:
            print("Considering adding a new plot promise...")
            # Use the LLM to decide if and what new promise to add
            new_promise_chain = LLMChain(
                llm=llm,
                prompt=PromptTemplate.from_template("""
                Based on the current state of the story, consider whether a new plot promise should be introduced.
                
                Title: {title}
                Genre: {genre}
                Current scene: {current_scene}
                
                Current active promises:
                {active_promises}
                
                Completed promises:
                {completed_promises}
                
                Should a new plot promise be introduced? If yes, describe it in detail.
                If no, explain why the existing promises are sufficient.
                
                Decision: [YES/NO]
                Reasoning: [your explanation]
                New Promise (if YES): [description]
                Importance (if YES): [1-10]
                """),
                verbose=True
            )
            
            try:
                # Get active and completed promise summaries
                active_promises_text = "\n".join([
                    f"- {p.description} (Progress: {p.progression}%)" 
                    for p in promise_manager.get_active_promises()
                ])
                completed_promises_text = "\n".join([
                    f"- {p.description}" 
                    for p in promise_manager.get_completed_promises()
                ])
                
                # If there are no completed promises yet
                if not completed_promises_text:
                    completed_promises_text = "None yet"
                
                result = new_promise_chain.invoke({
                    "title": title,
                    "genre": genre,
                    "current_scene": scene_content[:300] + "...",
                    "active_promises": active_promises_text,
                    "completed_promises": completed_promises_text
                })
                
                response = result.get('text', '')
                
                # Parse response to see if a new promise shoul
